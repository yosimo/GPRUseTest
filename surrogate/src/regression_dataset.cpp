// SPDX-License-Identifier: MIT

#include <bayesian_optimization/surrogate/regression_dataset.hpp>

#include <stdexcept>
#include <utility>

namespace bayesian_optimization::surrogate
{

RegressionDataset::RegressionDataset(
    Eigen::MatrixXd inputs,
    Eigen::VectorXd targets)
    : inputs_(std::move(inputs)),
      targets_(std::move(targets))
{
    if (inputs_.rows() <= 0 || inputs_.cols() <= 0)
    {
        throw std::invalid_argument(
            "RegressionDataset requires at least one row and one column");
    }
    if (inputs_.rows() != targets_.size())
    {
        throw std::invalid_argument(
            "RegressionDataset input and target sizes do not match");
    }
    if (!inputs_.allFinite() || !targets_.allFinite())
    {
        throw std::invalid_argument(
            "RegressionDataset values must be finite");
    }
}

Eigen::Index RegressionDataset::observationCount() const noexcept
{
    return inputs_.rows();
}

Eigen::Index RegressionDataset::inputDimension() const noexcept
{
    return inputs_.cols();
}

const Eigen::MatrixXd& RegressionDataset::inputs() const noexcept
{
    return inputs_;
}

const Eigen::VectorXd& RegressionDataset::targets() const noexcept
{
    return targets_;
}

}  // namespace bayesian_optimization::surrogate

