// SPDX-License-Identifier: MIT

#include <bayesian_optimization/surrogate/error.hpp>
#include <bayesian_optimization/surrogate/gaussian_process.hpp>
#include <bayesian_optimization/surrogate/input_transform.hpp>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <LBFGSB.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace bayesian_optimization::surrogate
{
namespace
{

constexpr double PI = 3.141592653589793238462643383279502884;

struct TransformState
{
    Eigen::VectorXd offset;
    Eigen::VectorXd scale;
};

struct Hyperparameters
{
    Eigen::VectorXd length_scales;
    double signal_variance{1.0};
    double noise_variance{0.0};
};

struct Factorization
{
    Eigen::MatrixXd lower;
    Eigen::VectorXd alpha;
    double jitter{0.0};
    double log_marginal_likelihood{0.0};
};

bool isEffectivelyZero(double scale, double reference)
{
    return scale <= std::sqrt(std::numeric_limits<double>::epsilon()) *
                        std::max(1.0, std::abs(reference));
}

TransformState fitOutputTransform(
    const Eigen::VectorXd& targets,
    OutputTransformType type)
{
    TransformState result;
    result.offset = Eigen::VectorXd::Zero(1);
    result.scale = Eigen::VectorXd::Ones(1);
    if (type == OutputTransformType::STANDARDIZE)
    {
        result.offset(0) = targets.mean();
        const double variance =
            (targets.array() - result.offset(0)).square().mean();
        const double scale = std::sqrt(std::max(0.0, variance));
        result.scale(0) =
            isEffectivelyZero(scale, result.offset(0)) ? 1.0 : scale;
    }
    return result;
}

Eigen::VectorXd applyOutputTransform(
    const Eigen::VectorXd& values,
    const TransformState& transform)
{
    return (values.array() - transform.offset(0)) / transform.scale(0);
}

double lengthScaleAt(
    const Eigen::VectorXd& length_scales,
    Eigen::Index dimension)
{
    return length_scales.size() == 1 ? length_scales(0)
                                     : length_scales(dimension);
}

double kernelValue(
    Eigen::Ref<const Eigen::VectorXd> left,
    Eigen::Ref<const Eigen::VectorXd> right,
    const Hyperparameters& parameters,
    KernelType kernel)
{
    double squared_distance = 0.0;
    for (Eigen::Index dimension = 0; dimension < left.size(); ++dimension)
    {
        const double delta =
            (left(dimension) - right(dimension)) /
            lengthScaleAt(parameters.length_scales, dimension);
        squared_distance += delta * delta;
    }

    if (kernel == KernelType::RBF)
    {
        return parameters.signal_variance *
               std::exp(-0.5 * squared_distance);
    }

    const double distance = std::sqrt(squared_distance);
    const double root_five_distance = std::sqrt(5.0) * distance;
    return parameters.signal_variance *
           (1.0 + root_five_distance +
            5.0 * squared_distance / 3.0) *
           std::exp(-root_five_distance);
}

Eigen::MatrixXd kernelMatrix(
    const Eigen::MatrixXd& inputs,
    const Hyperparameters& parameters,
    KernelType kernel)
{
    Eigen::MatrixXd result(inputs.rows(), inputs.rows());
    for (Eigen::Index row = 0; row < inputs.rows(); ++row)
    {
        result(row, row) = parameters.signal_variance;
        for (Eigen::Index column = 0; column < row; ++column)
        {
            const double value = kernelValue(
                inputs.row(row).transpose(),
                inputs.row(column).transpose(),
                parameters,
                kernel);
            result(row, column) = value;
            result(column, row) = value;
        }
    }
    return result;
}

Factorization factorize(
    const Eigen::MatrixXd& inputs,
    const Eigen::VectorXd& targets,
    const Hyperparameters& parameters,
    KernelType kernel,
    const JitterPolicy& jitter_policy)
{
    Eigen::MatrixXd covariance =
        kernelMatrix(inputs, parameters, kernel);
    const double initial_jitter =
        jitter_policy.initial_relative_value *
        std::max(parameters.signal_variance, 1.0);
    double jitter = initial_jitter;

    for (std::size_t attempt = 0; attempt < jitter_policy.max_attempts;
         ++attempt)
    {
        Eigen::MatrixXd stabilized = covariance;
        stabilized.diagonal().array() +=
            parameters.noise_variance + jitter;
        Eigen::LLT<Eigen::MatrixXd> decomposition(stabilized);
        if (decomposition.info() == Eigen::Success)
        {
            Factorization result;
            result.lower = decomposition.matrixL();
            result.alpha = decomposition.solve(targets);
            if (decomposition.info() != Eigen::Success ||
                !result.alpha.allFinite())
            {
                throw NumericalError(
                    "Cholesky solve produced non-finite values");
            }
            result.jitter = jitter;
            result.log_marginal_likelihood =
                -0.5 * targets.dot(result.alpha) -
                result.lower.diagonal().array().log().sum() -
                0.5 * static_cast<double>(inputs.rows()) *
                    std::log(2.0 * PI);
            return result;
        }
        jitter *= jitter_policy.multiplier;
    }

    std::ostringstream message;
    message << "Cholesky factorization failed for " << inputs.rows()
            << " observations after " << jitter_policy.max_attempts
            << " jitter attempts; last jitter=" << jitter;
    throw NumericalError(message.str());
}

Hyperparameters initialHyperparameters(
    const GaussianProcessConfig& config,
    Eigen::Index dimension)
{
    Hyperparameters result;
    const Eigen::Index count = config.use_ard ? dimension : 1;
    if (config.length_scales.values)
    {
        if (config.length_scales.values->size() == 1)
        {
            result.length_scales =
                Eigen::VectorXd::Constant(count, (*config.length_scales.values)(0));
        }
        else if (config.use_ard)
        {
            result.length_scales = *config.length_scales.values;
        }
        else
        {
            throw std::invalid_argument(
                "non-ARD GaussianProcess requires one length scale");
        }
    }
    else
    {
        result.length_scales = Eigen::VectorXd::Ones(count);
    }
    result.signal_variance = config.signal_variance.value;
    result.noise_variance = config.noise_variance.value;
    return result;
}

class LikelihoodObjective
{
public:
    LikelihoodObjective(
        const Eigen::MatrixXd& inputs,
        const Eigen::VectorXd& targets,
        const GaussianProcessConfig& config,
        Hyperparameters base)
        : inputs_(inputs),
          targets_(targets),
          config_(config),
          base_(std::move(base))
    {
    }

    Eigen::Index parameterCount() const
    {
        Eigen::Index result = 0;
        if (config_.length_scales.mode == HyperparameterMode::OPTIMIZE)
        {
            result += base_.length_scales.size();
        }
        if (config_.signal_variance.mode == HyperparameterMode::OPTIMIZE)
        {
            ++result;
        }
        if (config_.noise_variance.mode == HyperparameterMode::OPTIMIZE)
        {
            ++result;
        }
        return result;
    }

    Eigen::VectorXd encode(const Hyperparameters& parameters) const
    {
        Eigen::VectorXd result(parameterCount());
        Eigen::Index index = 0;
        if (config_.length_scales.mode == HyperparameterMode::OPTIMIZE)
        {
            for (Eigen::Index i = 0; i < parameters.length_scales.size(); ++i)
            {
                result(index++) = std::log(parameters.length_scales(i));
            }
        }
        if (config_.signal_variance.mode == HyperparameterMode::OPTIMIZE)
        {
            result(index++) = std::log(parameters.signal_variance);
        }
        if (config_.noise_variance.mode == HyperparameterMode::OPTIMIZE)
        {
            result(index) = std::log(parameters.noise_variance);
        }
        return result;
    }

    Hyperparameters decode(const Eigen::VectorXd& encoded) const
    {
        Hyperparameters result = base_;
        Eigen::Index index = 0;
        if (config_.length_scales.mode == HyperparameterMode::OPTIMIZE)
        {
            for (Eigen::Index i = 0; i < result.length_scales.size(); ++i)
            {
                result.length_scales(i) = std::exp(encoded(index++));
            }
        }
        if (config_.signal_variance.mode == HyperparameterMode::OPTIMIZE)
        {
            result.signal_variance = std::exp(encoded(index++));
        }
        if (config_.noise_variance.mode == HyperparameterMode::OPTIMIZE)
        {
            result.noise_variance = std::exp(encoded(index));
        }
        return result;
    }

    void bounds(Eigen::VectorXd& lower, Eigen::VectorXd& upper) const
    {
        lower.resize(parameterCount());
        upper.resize(parameterCount());
        Eigen::Index index = 0;
        if (config_.length_scales.mode == HyperparameterMode::OPTIMIZE)
        {
            for (Eigen::Index i = 0; i < base_.length_scales.size(); ++i)
            {
                lower(index) = std::log(config_.length_scales.lower_bound);
                upper(index++) = std::log(config_.length_scales.upper_bound);
            }
        }
        if (config_.signal_variance.mode == HyperparameterMode::OPTIMIZE)
        {
            lower(index) = std::log(config_.signal_variance.lower_bound);
            upper(index++) = std::log(config_.signal_variance.upper_bound);
        }
        if (config_.noise_variance.mode == HyperparameterMode::OPTIMIZE)
        {
            lower(index) = std::log(config_.noise_variance.lower_bound);
            upper(index) = std::log(config_.noise_variance.upper_bound);
        }
    }

    double operator()(const Eigen::VectorXd& encoded, Eigen::VectorXd& gradient)
    {
        const Hyperparameters parameters = decode(encoded);
        const Eigen::MatrixXd kernel =
            kernelMatrix(inputs_, parameters, config_.kernel);
        const Factorization factorization = factorize(
            inputs_,
            targets_,
            parameters,
            config_.kernel,
            config_.jitter);

        const Eigen::MatrixXd identity =
            Eigen::MatrixXd::Identity(inputs_.rows(), inputs_.rows());
        const Eigen::MatrixXd inverse =
            factorization.lower.transpose()
                .triangularView<Eigen::Upper>()
                .solve(
                    factorization.lower.triangularView<Eigen::Lower>().solve(
                        identity));
        const Eigen::MatrixXd common =
            factorization.alpha * factorization.alpha.transpose() - inverse;

        gradient.resize(parameterCount());
        Eigen::Index output_index = 0;
        if (config_.length_scales.mode == HyperparameterMode::OPTIMIZE)
        {
            for (Eigen::Index parameter_index = 0;
                 parameter_index < parameters.length_scales.size();
                 ++parameter_index)
            {
                Eigen::MatrixXd derivative =
                    Eigen::MatrixXd::Zero(inputs_.rows(), inputs_.rows());
                for (Eigen::Index row = 0; row < inputs_.rows(); ++row)
                {
                    for (Eigen::Index column = 0; column < row; ++column)
                    {
                        double squared_distance = 0.0;
                        double selected_squared_delta = 0.0;
                        for (Eigen::Index dimension = 0;
                             dimension < inputs_.cols();
                             ++dimension)
                        {
                            const double length_scale = lengthScaleAt(
                                parameters.length_scales,
                                dimension);
                            const double delta =
                                (inputs_(row, dimension) -
                                 inputs_(column, dimension)) /
                                length_scale;
                            const double squared_delta = delta * delta;
                            squared_distance += squared_delta;
                            if (parameters.length_scales.size() == 1 ||
                                dimension == parameter_index)
                            {
                                selected_squared_delta += squared_delta;
                            }
                        }

                        double value = 0.0;
                        if (config_.kernel == KernelType::RBF)
                        {
                            value = kernel(row, column) *
                                    selected_squared_delta;
                        }
                        else
                        {
                            const double distance =
                                std::sqrt(squared_distance);
                            value =
                                parameters.signal_variance * (5.0 / 3.0) *
                                std::exp(-std::sqrt(5.0) * distance) *
                                (1.0 + std::sqrt(5.0) * distance) *
                                selected_squared_delta;
                        }
                        derivative(row, column) = value;
                        derivative(column, row) = value;
                    }
                }
                gradient(output_index++) =
                    -0.5 * (common.cwiseProduct(derivative)).sum();
            }
        }
        if (config_.signal_variance.mode == HyperparameterMode::OPTIMIZE)
        {
            gradient(output_index++) =
                -0.5 * (common.cwiseProduct(kernel)).sum();
        }
        if (config_.noise_variance.mode == HyperparameterMode::OPTIMIZE)
        {
            gradient(output_index) =
                -0.5 * parameters.noise_variance * common.trace();
        }
        return -factorization.log_marginal_likelihood;
    }

private:
    const Eigen::MatrixXd& inputs_;
    const Eigen::VectorXd& targets_;
    const GaussianProcessConfig& config_;
    Hyperparameters base_;
};

Hyperparameters optimizeHyperparameters(
    const Eigen::MatrixXd& inputs,
    const Eigen::VectorXd& targets,
    const GaussianProcessConfig& config,
    Hyperparameters initial)
{
    LikelihoodObjective objective(inputs, targets, config, initial);
    if (objective.parameterCount() == 0 || inputs.rows() == 1)
    {
        return initial;
    }

    Eigen::VectorXd lower;
    Eigen::VectorXd upper;
    objective.bounds(lower, upper);

    LBFGSpp::LBFGSBParam<double> parameters;
    parameters.epsilon = config.hyperparameter_optimization.gradient_tolerance;
    parameters.max_iterations =
        static_cast<int>(config.hyperparameter_optimization.max_iterations);

    std::mt19937_64 random_engine(
        config.hyperparameter_optimization.random_seed);
    double best_value = std::numeric_limits<double>::infinity();
    Eigen::VectorXd best_encoded;

    for (std::size_t restart = 0;
         restart < config.hyperparameter_optimization.restart_count;
         ++restart)
    {
        Eigen::VectorXd encoded = objective.encode(initial);
        if (restart > 0)
        {
            for (Eigen::Index i = 0; i < encoded.size(); ++i)
            {
                std::uniform_real_distribution<double> distribution(
                    lower(i),
                    upper(i));
                encoded(i) = distribution(random_engine);
            }
        }

        try
        {
            LBFGSpp::LBFGSBSolver<double> solver(parameters);
            double value = 0.0;
            solver.minimize(objective, encoded, value, lower, upper);
            if (std::isfinite(value) && encoded.allFinite() &&
                value < best_value)
            {
                best_value = value;
                best_encoded = encoded;
            }
        }
        catch (const std::exception&)
        {
            // Other starts remain valid candidates.
        }
    }

    if (best_encoded.size() == 0)
    {
        throw NumericalError("all hyperparameter optimization starts failed");
    }
    return objective.decode(best_encoded);
}

Eigen::MatrixXd kernelToTraining(
    const Eigen::MatrixXd& training_inputs,
    Eigen::Ref<const Eigen::MatrixXd> test_inputs,
    const Hyperparameters& parameters,
    KernelType kernel)
{
    Eigen::MatrixXd result(training_inputs.rows(), test_inputs.rows());
    for (Eigen::Index training = 0; training < training_inputs.rows(); ++training)
    {
        for (Eigen::Index test = 0; test < test_inputs.rows(); ++test)
        {
            result(training, test) = kernelValue(
                training_inputs.row(training).transpose(),
                test_inputs.row(test).transpose(),
                parameters,
                kernel);
        }
    }
    return result;
}

Eigen::MatrixXd kernelGradientToTraining(
    const Eigen::MatrixXd& training_inputs,
    Eigen::Ref<const Eigen::VectorXd> test_input,
    const Hyperparameters& parameters,
    KernelType kernel)
{
    Eigen::MatrixXd result(training_inputs.rows(), training_inputs.cols());
    for (Eigen::Index row = 0; row < training_inputs.rows(); ++row)
    {
        const Eigen::VectorXd training = training_inputs.row(row).transpose();
        double squared_distance = 0.0;
        for (Eigen::Index dimension = 0;
             dimension < training_inputs.cols();
             ++dimension)
        {
            const double delta =
                (test_input(dimension) - training(dimension)) /
                lengthScaleAt(parameters.length_scales, dimension);
            squared_distance += delta * delta;
        }
        const double value =
            kernelValue(training, test_input, parameters, kernel);
        const double distance = std::sqrt(squared_distance);
        for (Eigen::Index dimension = 0;
             dimension < training_inputs.cols();
             ++dimension)
        {
            const double length_scale =
                lengthScaleAt(parameters.length_scales, dimension);
            const double base =
                (training(dimension) - test_input(dimension)) /
                (length_scale * length_scale);
            if (kernel == KernelType::RBF)
            {
                result(row, dimension) = value * base;
            }
            else
            {
                result(row, dimension) =
                    parameters.signal_variance * (5.0 / 3.0) *
                    std::exp(-std::sqrt(5.0) * distance) *
                    (1.0 + std::sqrt(5.0) * distance) * base;
            }
        }
    }
    return result;
}

}  // namespace

class GaussianProcess::Impl
{
public:
    struct State
    {
        State(
            RegressionDataset source_data,
            FittedInputTransform input,
            TransformState output,
            Eigen::MatrixXd transformed_inputs_value,
            Eigen::VectorXd transformed_targets_value,
            Hyperparameters hyperparameters_value,
            Factorization factorization_value)
            : data(std::move(source_data)),
              input_transform(std::move(input)),
              output_transform(std::move(output)),
              transformed_inputs(std::move(transformed_inputs_value)),
              transformed_targets(std::move(transformed_targets_value)),
              hyperparameters(std::move(hyperparameters_value)),
              factorization(std::move(factorization_value))
        {
        }

        RegressionDataset data;
        FittedInputTransform input_transform;
        TransformState output_transform;
        Eigen::MatrixXd transformed_inputs;
        Eigen::VectorXd transformed_targets;
        Hyperparameters hyperparameters;
        Factorization factorization;
    };

    explicit Impl(GaussianProcessConfig value)
        : config(std::move(value))
    {
    }

    std::unique_ptr<State> buildState(
        const RegressionDataset& dataset,
        SurrogateFitPolicy policy) const
    {
        config.validateForDimension(dataset.inputDimension());
        const FittedInputTransform input = FittedInputTransform::fit(
            dataset.inputs(),
            config.preprocessing.input_transform,
            config.preprocessing.input_lower_bounds,
            config.preprocessing.input_upper_bounds);
        const TransformState output = fitOutputTransform(
            dataset.targets(),
            config.preprocessing.output_transform);
        Eigen::MatrixXd transformed_inputs =
            input.apply(dataset.inputs());
        Eigen::VectorXd transformed_targets =
            applyOutputTransform(dataset.targets(), output);

        Hyperparameters hyperparameters =
            state ? state->hyperparameters
                  : initialHyperparameters(config, dataset.inputDimension());
        if (policy == SurrogateFitPolicy::CONFIGURED)
        {
            hyperparameters = optimizeHyperparameters(
                transformed_inputs,
                transformed_targets,
                config,
                std::move(hyperparameters));
        }

        Factorization factorization = factorize(
            transformed_inputs,
            transformed_targets,
            hyperparameters,
            config.kernel,
            config.jitter);
        return std::make_unique<State>(
            dataset,
            std::move(input),
            std::move(output),
            std::move(transformed_inputs),
            std::move(transformed_targets),
            std::move(hyperparameters),
            std::move(factorization));
    }

    const State& requireState() const
    {
        if (!state)
        {
            throw std::logic_error("GaussianProcess is not fitted");
        }
        return *state;
    }

    GaussianProcessConfig config;
    std::unique_ptr<State> state;
};

GaussianProcess::GaussianProcess(GaussianProcessConfig config)
    : impl_(std::make_unique<Impl>(std::move(config)))
{
}

GaussianProcess::~GaussianProcess() = default;
GaussianProcess::GaussianProcess(GaussianProcess&&) noexcept = default;
GaussianProcess& GaussianProcess::operator=(GaussianProcess&&) noexcept =
    default;

void GaussianProcess::fit(
    const RegressionDataset& dataset,
    SurrogateFitPolicy policy)
{
    if (impl_->state &&
        impl_->state->data.inputDimension() != dataset.inputDimension())
    {
        throw std::invalid_argument(
            "refitting GaussianProcess with a different dimension is not allowed");
    }
    std::unique_ptr<Impl::State> next = impl_->buildState(dataset, policy);
    impl_->state = std::move(next);
}

bool GaussianProcess::isFitted() const noexcept
{
    return impl_ && impl_->state != nullptr;
}

Eigen::Index GaussianProcess::inputDimension() const noexcept
{
    return isFitted() ? impl_->state->data.inputDimension() : 0;
}

Prediction GaussianProcess::predict(
    Eigen::Ref<const Eigen::MatrixXd> test_inputs) const
{
    return predictWithGradients(test_inputs).prediction;
}

PredictionWithGradients GaussianProcess::predictWithGradients(
    Eigen::Ref<const Eigen::MatrixXd> test_inputs) const
{
    const Impl::State& state = impl_->requireState();
    if (test_inputs.cols() != state.data.inputDimension())
    {
        throw std::invalid_argument(
            "prediction input dimension does not match the fitted model");
    }
    if (!test_inputs.allFinite())
    {
        throw std::invalid_argument("prediction inputs must be finite");
    }

    const Eigen::MatrixXd transformed_test_inputs =
        state.input_transform.apply(test_inputs);
    const Eigen::MatrixXd cross_covariance = kernelToTraining(
        state.transformed_inputs,
        transformed_test_inputs,
        state.hyperparameters,
        impl_->config.kernel);
    const Eigen::MatrixXd solved =
        state.factorization.lower.triangularView<Eigen::Lower>().solve(
            cross_covariance);

    PredictionWithGradients result;
    const Eigen::Index count = test_inputs.rows();
    const Eigen::Index dimension = test_inputs.cols();
    result.prediction.mean =
        cross_covariance.transpose() * state.factorization.alpha;
    result.prediction.latent_variance.resize(count);
    result.prediction.observation_variance.resize(count);
    result.mean_gradient.resize(count, dimension);
    result.latent_variance_gradient.resize(count, dimension);

    for (Eigen::Index test = 0; test < count; ++test)
    {
        double latent_variance =
            state.hyperparameters.signal_variance -
            solved.col(test).squaredNorm();
        const double tolerance =
            1.0e-10 *
            std::max(1.0, state.hyperparameters.signal_variance);
        if (!std::isfinite(latent_variance) ||
            latent_variance < -tolerance)
        {
            throw NumericalError("prediction produced an invalid variance");
        }
        latent_variance = std::max(0.0, latent_variance);
        result.prediction.latent_variance(test) = latent_variance;
        result.prediction.observation_variance(test) =
            latent_variance + state.hyperparameters.noise_variance;

        const Eigen::MatrixXd kernel_gradient = kernelGradientToTraining(
            state.transformed_inputs,
            transformed_test_inputs.row(test).transpose(),
            state.hyperparameters,
            impl_->config.kernel);
        const Eigen::VectorXd inverse_times_kernel =
            state.factorization.lower.transpose()
                .triangularView<Eigen::Upper>()
                .solve(solved.col(test));
        result.mean_gradient.row(test) =
            (kernel_gradient.transpose() * state.factorization.alpha)
                .transpose();
        result.latent_variance_gradient.row(test) =
            (-2.0 * kernel_gradient.transpose() * inverse_times_kernel)
                .transpose();
    }

    const double output_offset = state.output_transform.offset(0);
    const double output_scale = state.output_transform.scale(0);
    result.prediction.mean =
        (result.prediction.mean.array() * output_scale + output_offset)
            .matrix();
    result.prediction.latent_variance *= output_scale * output_scale;
    result.prediction.observation_variance *= output_scale * output_scale;
    for (Eigen::Index dimension_index = 0;
         dimension_index < dimension;
         ++dimension_index)
    {
        result.mean_gradient.col(dimension_index) *=
            output_scale /
            state.input_transform.scale()(dimension_index);
        result.latent_variance_gradient.col(dimension_index) *=
            (output_scale * output_scale) /
            state.input_transform.scale()(dimension_index);
    }
    return result;
}

OutputTransform GaussianProcess::fittedOutputTransform() const
{
    const Impl::State& state = impl_->requireState();
    return {state.output_transform.offset(0), state.output_transform.scale(0)};
}

const GaussianProcessConfig& GaussianProcess::config() const noexcept
{
    return impl_->config;
}

const RegressionDataset& GaussianProcess::trainingData() const
{
    return impl_->requireState().data;
}

FittedHyperparameters GaussianProcess::fittedHyperparameters() const
{
    const Impl::State& state = impl_->requireState();
    return {
        state.hyperparameters.length_scales,
        state.hyperparameters.signal_variance,
        state.hyperparameters.noise_variance,
        state.factorization.jitter,
        state.factorization.log_marginal_likelihood};
}

}  // namespace bayesian_optimization::surrogate

