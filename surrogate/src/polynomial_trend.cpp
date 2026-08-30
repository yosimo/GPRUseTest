// SPDX-License-Identifier: MIT

#include <bayesian_optimization/surrogate/input_transform.hpp>
#include <bayesian_optimization/surrogate/polynomial_trend.hpp>

#include <Eigen/QR>

#include <cmath>
#include <memory>
#include <stdexcept>
#include <utility>

namespace bayesian_optimization::surrogate
{
namespace
{

Eigen::Index featureCount(
    Eigen::Index dimension,
    const PolynomialTrendConfig& config)
{
    Eigen::Index count = 1 + dimension;
    if (config.degree == PolynomialDegree::QUADRATIC)
    {
        count += dimension;
        if (config.include_interactions)
        {
            count += dimension * (dimension - 1) / 2;
        }
    }
    return count;
}

Eigen::MatrixXd buildFeatures(
    Eigen::Ref<const Eigen::MatrixXd> transformed_inputs,
    const PolynomialTrendConfig& config)
{
    Eigen::MatrixXd result(
        transformed_inputs.rows(),
        featureCount(transformed_inputs.cols(), config));
    result.col(0).setOnes();

    Eigen::Index feature = 1;
    for (Eigen::Index dimension = 0;
         dimension < transformed_inputs.cols();
         ++dimension)
    {
        result.col(feature++) = transformed_inputs.col(dimension);
    }

    if (config.degree == PolynomialDegree::QUADRATIC)
    {
        for (Eigen::Index dimension = 0;
             dimension < transformed_inputs.cols();
             ++dimension)
        {
            result.col(feature++) =
                transformed_inputs.col(dimension).array().square().matrix();
        }
        if (config.include_interactions)
        {
            for (Eigen::Index left = 0;
                 left < transformed_inputs.cols();
                 ++left)
            {
                for (Eigen::Index right = left + 1;
                     right < transformed_inputs.cols();
                     ++right)
                {
                    result.col(feature++) =
                        (transformed_inputs.col(left).array() *
                         transformed_inputs.col(right).array())
                            .matrix();
                }
            }
        }
    }

    return result;
}

void validateConfig(const PolynomialTrendConfig& config)
{
    if (config.degree != PolynomialDegree::LINEAR &&
        config.degree != PolynomialDegree::QUADRATIC)
    {
        throw std::invalid_argument("polynomial degree must be linear or quadratic");
    }
    if (!std::isfinite(config.ridge_lambda) ||
        config.ridge_lambda < 0.0)
    {
        throw std::invalid_argument(
            "polynomial ridge_lambda must be finite and non-negative");
    }
}

}  // namespace

class PolynomialTrendModel::Impl
{
public:
    struct State
    {
        FittedInputTransform input_transform;
        Eigen::VectorXd coefficients;
        Eigen::Index feature_rank{0};
    };

    explicit Impl(PolynomialTrendConfig value)
        : config(std::move(value))
    {
    }

    const State& requireState() const
    {
        if (!state)
        {
            throw std::logic_error("PolynomialTrendModel is not fitted");
        }
        return *state;
    }

    PolynomialTrendConfig config;
    std::unique_ptr<State> state;
};

PolynomialTrendModel::PolynomialTrendModel(PolynomialTrendConfig config)
    : impl_(std::make_unique<Impl>(std::move(config)))
{
}

PolynomialTrendModel::~PolynomialTrendModel() = default;
PolynomialTrendModel::PolynomialTrendModel(
    PolynomialTrendModel&&) noexcept = default;
PolynomialTrendModel& PolynomialTrendModel::operator=(
    PolynomialTrendModel&&) noexcept = default;

void PolynomialTrendModel::fit(const RegressionDataset& dataset)
{
    validateConfig(impl_->config);
    FittedInputTransform input_transform = FittedInputTransform::fit(
        dataset.inputs(),
        impl_->config.input_transform,
        impl_->config.input_lower_bounds,
        impl_->config.input_upper_bounds);
    const Eigen::MatrixXd transformed_inputs =
        input_transform.apply(dataset.inputs());
    const Eigen::MatrixXd features =
        buildFeatures(transformed_inputs, impl_->config);

    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> feature_decomposition(features);
    const Eigen::Index rank = feature_decomposition.rank();
    if (impl_->config.ridge_lambda == 0.0 &&
        rank < features.cols())
    {
        throw std::invalid_argument(
            "polynomial feature matrix is rank deficient; use ridge regularization");
    }

    Eigen::MatrixXd system;
    Eigen::VectorXd right_hand_side;
    if (impl_->config.ridge_lambda > 0.0)
    {
        const Eigen::Index penalty_count = features.cols() - 1;
        system = Eigen::MatrixXd::Zero(
            features.rows() + penalty_count,
            features.cols());
        system.topRows(features.rows()) = features;
        const double penalty = std::sqrt(impl_->config.ridge_lambda);
        for (Eigen::Index feature = 1;
             feature < features.cols();
             ++feature)
        {
            system(features.rows() + feature - 1, feature) = penalty;
        }

        right_hand_side = Eigen::VectorXd::Zero(system.rows());
        right_hand_side.head(features.rows()) = dataset.targets();
    }
    else
    {
        system = features;
        right_hand_side = dataset.targets();
    }

    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> decomposition(system);
    Eigen::VectorXd coefficients = decomposition.solve(right_hand_side);
    if (coefficients.size() != features.cols() ||
        !coefficients.allFinite())
    {
        throw std::runtime_error(
            "polynomial regression produced invalid coefficients");
    }

    auto next = std::make_unique<Impl::State>();
    next->input_transform = std::move(input_transform);
    next->coefficients = std::move(coefficients);
    next->feature_rank = rank;
    impl_->state = std::move(next);
}

bool PolynomialTrendModel::isFitted() const noexcept
{
    return impl_ && impl_->state != nullptr;
}

Eigen::Index PolynomialTrendModel::inputDimension() const noexcept
{
    return isFitted()
               ? impl_->state->input_transform.inputDimension()
               : 0;
}

Eigen::VectorXd PolynomialTrendModel::predict(
    Eigen::Ref<const Eigen::MatrixXd> inputs) const
{
    const Impl::State& state = impl_->requireState();
    const Eigen::MatrixXd transformed =
        state.input_transform.apply(inputs);
    return buildFeatures(transformed, impl_->config) *
           state.coefficients;
}

Eigen::MatrixXd PolynomialTrendModel::predictGradients(
    Eigen::Ref<const Eigen::MatrixXd> inputs) const
{
    const Impl::State& state = impl_->requireState();
    const Eigen::MatrixXd transformed =
        state.input_transform.apply(inputs);
    Eigen::MatrixXd result =
        Eigen::MatrixXd::Zero(inputs.rows(), inputs.cols());

    Eigen::Index feature = 1;
    for (Eigen::Index dimension = 0;
         dimension < inputs.cols();
         ++dimension)
    {
        result.col(dimension).array() +=
            state.coefficients(feature++) /
            state.input_transform.scale()(dimension);
    }

    if (impl_->config.degree == PolynomialDegree::QUADRATIC)
    {
        for (Eigen::Index dimension = 0;
             dimension < inputs.cols();
             ++dimension)
        {
            result.col(dimension).array() +=
                2.0 * state.coefficients(feature++) *
                transformed.col(dimension).array() /
                state.input_transform.scale()(dimension);
        }

        if (impl_->config.include_interactions)
        {
            for (Eigen::Index left = 0;
                 left < inputs.cols();
                 ++left)
            {
                for (Eigen::Index right = left + 1;
                     right < inputs.cols();
                     ++right)
                {
                    const double coefficient =
                        state.coefficients(feature++);
                    result.col(left).array() +=
                        coefficient * transformed.col(right).array() /
                        state.input_transform.scale()(left);
                    result.col(right).array() +=
                        coefficient * transformed.col(left).array() /
                        state.input_transform.scale()(right);
                }
            }
        }
    }

    return result;
}

std::unique_ptr<DeterministicTrendModel>
PolynomialTrendModel::clone() const
{
    auto result =
        std::make_unique<PolynomialTrendModel>(impl_->config);
    if (impl_->state)
    {
        result->impl_->state =
            std::make_unique<Impl::State>(*impl_->state);
    }
    return result;
}

const PolynomialTrendConfig&
PolynomialTrendModel::config() const noexcept
{
    return impl_->config;
}

const Eigen::VectorXd&
PolynomialTrendModel::coefficients() const
{
    return impl_->requireState().coefficients;
}

Eigen::Index PolynomialTrendModel::featureCount() const noexcept
{
    return isFitted()
               ? bayesian_optimization::surrogate::featureCount(
                     inputDimension(),
                     impl_->config)
               : 0;
}

Eigen::Index PolynomialTrendModel::featureMatrixRank() const
{
    return impl_->requireState().feature_rank;
}

}  // namespace bayesian_optimization::surrogate
