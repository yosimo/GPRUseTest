// SPDX-License-Identifier: MIT
#pragma once

#include <Eigen/Core>

namespace bayesian_optimization::surrogate
{

/** @brief Owning, validated regression inputs and targets. */
class RegressionDataset final
{
public:
    RegressionDataset(Eigen::MatrixXd inputs, Eigen::VectorXd targets);

    [[nodiscard]] Eigen::Index observationCount() const noexcept;
    [[nodiscard]] Eigen::Index inputDimension() const noexcept;
    [[nodiscard]] const Eigen::MatrixXd& inputs() const noexcept;
    [[nodiscard]] const Eigen::VectorXd& targets() const noexcept;

private:
    Eigen::MatrixXd inputs_;
    Eigen::VectorXd targets_;
};

}  // namespace bayesian_optimization::surrogate

