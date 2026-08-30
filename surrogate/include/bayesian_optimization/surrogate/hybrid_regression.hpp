// SPDX-License-Identifier: MIT
#pragma once

#include <bayesian_optimization/surrogate/deterministic_trend_model.hpp>
#include <bayesian_optimization/surrogate/gaussian_process.hpp>

#include <memory>

namespace bayesian_optimization::surrogate
{

/**
 * @brief Deterministic global trend plus a Gaussian-process residual.
 *
 * The predictive variance represents only the residual Gaussian process.
 */
class HybridRegressionModel final : public DifferentiableSurrogateModel
{
public:
    HybridRegressionModel(
        std::unique_ptr<DeterministicTrendModel> trend_model,
        GaussianProcessConfig residual_gp_config = {});
    ~HybridRegressionModel();

    HybridRegressionModel(const HybridRegressionModel&) = delete;
    HybridRegressionModel& operator=(const HybridRegressionModel&) = delete;
    HybridRegressionModel(HybridRegressionModel&&) noexcept;
    HybridRegressionModel& operator=(HybridRegressionModel&&) noexcept;

    void fit(
        const RegressionDataset& dataset,
        SurrogateFitPolicy policy = SurrogateFitPolicy::CONFIGURED) override;
    [[nodiscard]] bool isFitted() const noexcept override;
    [[nodiscard]] Eigen::Index inputDimension() const noexcept override;
    [[nodiscard]] Prediction predict(
        Eigen::Ref<const Eigen::MatrixXd> test_inputs) const override;
    [[nodiscard]] PredictionWithGradients predictWithGradients(
        Eigen::Ref<const Eigen::MatrixXd> test_inputs) const override;
    [[nodiscard]] OutputTransform fittedOutputTransform() const override;

    [[nodiscard]] const DeterministicTrendModel& trendModel() const;
    [[nodiscard]] const GaussianProcess& residualModel() const noexcept;
    [[nodiscard]] double residualBias() const;
    [[nodiscard]] const RegressionDataset& trainingData() const;

private:
    std::unique_ptr<DeterministicTrendModel> trend_model_;
    GaussianProcess residual_model_;
    std::unique_ptr<RegressionDataset> training_data_;
    double residual_bias_{0.0};
};

}  // namespace bayesian_optimization::surrogate
