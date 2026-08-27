// SPDX-License-Identifier: MIT
#pragma once

#include <bayesian_optimization/surrogate/config.hpp>
#include <bayesian_optimization/surrogate/prediction.hpp>
#include <bayesian_optimization/surrogate/regression_dataset.hpp>

#include <Eigen/Core>

namespace bayesian_optimization::surrogate
{

/** @brief Minimal trainable probabilistic regression interface. */
class SurrogateModel
{
public:
    virtual ~SurrogateModel() = default;

    SurrogateModel(const SurrogateModel&) = delete;
    SurrogateModel& operator=(const SurrogateModel&) = delete;
    SurrogateModel(SurrogateModel&&) noexcept = default;
    SurrogateModel& operator=(SurrogateModel&&) noexcept = default;

    virtual void fit(
        const RegressionDataset& dataset,
        SurrogateFitPolicy policy = SurrogateFitPolicy::CONFIGURED) = 0;
    [[nodiscard]] virtual bool isFitted() const noexcept = 0;
    [[nodiscard]] virtual Eigen::Index inputDimension() const noexcept = 0;
    [[nodiscard]] virtual Prediction predict(
        Eigen::Ref<const Eigen::MatrixXd> test_inputs) const = 0;
    [[nodiscard]] virtual OutputTransform fittedOutputTransform() const = 0;

protected:
    SurrogateModel() = default;
};

/** @brief Optional prediction-gradient capability. */
class DifferentiableSurrogateModel : public SurrogateModel
{
public:
    [[nodiscard]] virtual PredictionWithGradients predictWithGradients(
        Eigen::Ref<const Eigen::MatrixXd> test_inputs) const = 0;

protected:
    DifferentiableSurrogateModel() = default;
};

}  // namespace bayesian_optimization::surrogate

