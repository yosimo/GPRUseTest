// SPDX-License-Identifier: MIT
#pragma once

#include <bayesian_optimization/surrogate/regression_dataset.hpp>

#include <Eigen/Core>

#include <memory>

namespace bayesian_optimization::surrogate
{

/** @brief Trainable deterministic trend used as a surrogate mean model. */
class DeterministicTrendModel
{
public:
    virtual ~DeterministicTrendModel() = default;

    DeterministicTrendModel(const DeterministicTrendModel&) = delete;
    DeterministicTrendModel& operator=(const DeterministicTrendModel&) = delete;
    DeterministicTrendModel(DeterministicTrendModel&&) noexcept = default;
    DeterministicTrendModel& operator=(
        DeterministicTrendModel&&) noexcept = default;

    virtual void fit(const RegressionDataset& dataset) = 0;
    [[nodiscard]] virtual bool isFitted() const noexcept = 0;
    [[nodiscard]] virtual Eigen::Index inputDimension() const noexcept = 0;
    [[nodiscard]] virtual Eigen::VectorXd predict(
        Eigen::Ref<const Eigen::MatrixXd> inputs) const = 0;
    [[nodiscard]] virtual Eigen::MatrixXd predictGradients(
        Eigen::Ref<const Eigen::MatrixXd> inputs) const = 0;
    [[nodiscard]] virtual std::unique_ptr<DeterministicTrendModel> clone()
        const = 0;

protected:
    DeterministicTrendModel() = default;
};

}  // namespace bayesian_optimization::surrogate
