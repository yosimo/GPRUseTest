// SPDX-License-Identifier: MIT

#include <bayesian_optimization/surrogate/hybrid_regression.hpp>

#include <cmath>
#include <stdexcept>
#include <utility>

namespace bayesian_optimization::surrogate
{

HybridRegressionModel::HybridRegressionModel(
    std::unique_ptr<DeterministicTrendModel> trend_model,
    GaussianProcessConfig residual_gp_config)
    : trend_model_(std::move(trend_model)),
      residual_model_(std::move(residual_gp_config))
{
    if (!trend_model_)
    {
        throw std::invalid_argument(
            "HybridRegressionModel requires a trend model");
    }
}

HybridRegressionModel::~HybridRegressionModel() = default;
HybridRegressionModel::HybridRegressionModel(
    HybridRegressionModel&&) noexcept = default;
HybridRegressionModel& HybridRegressionModel::operator=(
    HybridRegressionModel&&) noexcept = default;

void HybridRegressionModel::fit(
    const RegressionDataset& dataset,
    SurrogateFitPolicy policy)
{
    if (isFitted() &&
        inputDimension() != dataset.inputDimension())
    {
        throw std::invalid_argument(
            "refitting HybridRegressionModel with a different dimension is not allowed");
    }

    std::unique_ptr<DeterministicTrendModel> next_trend =
        trend_model_->clone();
    next_trend->fit(dataset);
    if (!next_trend->isFitted() ||
        next_trend->inputDimension() != dataset.inputDimension())
    {
        throw std::runtime_error(
            "trend model did not produce a compatible fitted state");
    }

    const Eigen::VectorXd trend_prediction =
        next_trend->predict(dataset.inputs());
    if (trend_prediction.size() != dataset.observationCount() ||
        !trend_prediction.allFinite())
    {
        throw std::runtime_error(
            "trend model produced invalid training predictions");
    }

    Eigen::VectorXd residuals =
        dataset.targets() - trend_prediction;
    const double next_bias = residuals.mean();
    if (!std::isfinite(next_bias))
    {
        throw std::runtime_error(
            "hybrid residual bias is not finite");
    }
    residuals.array() -= next_bias;

    auto next_training_data =
        std::make_unique<RegressionDataset>(dataset);
    residual_model_.fit(
        RegressionDataset(dataset.inputs(), std::move(residuals)),
        policy);

    trend_model_ = std::move(next_trend);
    training_data_ = std::move(next_training_data);
    residual_bias_ = next_bias;
}

bool HybridRegressionModel::isFitted() const noexcept
{
    return training_data_ != nullptr &&
           trend_model_ &&
           trend_model_->isFitted() &&
           residual_model_.isFitted();
}

Eigen::Index HybridRegressionModel::inputDimension() const noexcept
{
    return isFitted() ? training_data_->inputDimension() : 0;
}

Prediction HybridRegressionModel::predict(
    Eigen::Ref<const Eigen::MatrixXd> test_inputs) const
{
    if (!isFitted())
    {
        throw std::logic_error(
            "HybridRegressionModel is not fitted");
    }

    Prediction result = residual_model_.predict(test_inputs);
    const Eigen::VectorXd trend =
        trend_model_->predict(test_inputs);
    if (trend.size() != test_inputs.rows() ||
        !trend.allFinite())
    {
        throw std::runtime_error(
            "trend model produced invalid predictions");
    }
    result.mean.array() += trend.array() + residual_bias_;
    return result;
}

PredictionWithGradients
HybridRegressionModel::predictWithGradients(
    Eigen::Ref<const Eigen::MatrixXd> test_inputs) const
{
    if (!isFitted())
    {
        throw std::logic_error(
            "HybridRegressionModel is not fitted");
    }

    PredictionWithGradients result =
        residual_model_.predictWithGradients(test_inputs);
    const Eigen::VectorXd trend =
        trend_model_->predict(test_inputs);
    const Eigen::MatrixXd trend_gradient =
        trend_model_->predictGradients(test_inputs);
    if (trend.size() != test_inputs.rows() ||
        trend_gradient.rows() != test_inputs.rows() ||
        trend_gradient.cols() != test_inputs.cols() ||
        !trend.allFinite() ||
        !trend_gradient.allFinite())
    {
        throw std::runtime_error(
            "trend model produced invalid predictions or gradients");
    }

    result.prediction.mean.array() +=
        trend.array() + residual_bias_;
    result.mean_gradient += trend_gradient;
    return result;
}

OutputTransform
HybridRegressionModel::fittedOutputTransform() const
{
    if (!isFitted())
    {
        throw std::logic_error(
            "HybridRegressionModel is not fitted");
    }
    return {0.0, 1.0};
}

const DeterministicTrendModel&
HybridRegressionModel::trendModel() const
{
    if (!isFitted())
    {
        throw std::logic_error(
            "HybridRegressionModel is not fitted");
    }
    return *trend_model_;
}

const GaussianProcess&
HybridRegressionModel::residualModel() const noexcept
{
    return residual_model_;
}

double HybridRegressionModel::residualBias() const
{
    if (!isFitted())
    {
        throw std::logic_error(
            "HybridRegressionModel is not fitted");
    }
    return residual_bias_;
}

const RegressionDataset&
HybridRegressionModel::trainingData() const
{
    if (!isFitted())
    {
        throw std::logic_error(
            "HybridRegressionModel is not fitted");
    }
    return *training_data_;
}

}  // namespace bayesian_optimization::surrogate
