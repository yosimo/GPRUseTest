// SPDX-License-Identifier: MIT

#include <bayesian_optimization/surrogate/input_transform.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace bayesian_optimization::surrogate
{
namespace
{

bool isEffectivelyZero(double scale, double reference)
{
    return scale <= std::sqrt(std::numeric_limits<double>::epsilon()) *
                        std::max(1.0, std::abs(reference));
}

}  // namespace

FittedInputTransform FittedInputTransform::fit(
    Eigen::Ref<const Eigen::MatrixXd> inputs,
    InputTransformType type,
    const std::optional<Eigen::VectorXd>& lower_bounds,
    const std::optional<Eigen::VectorXd>& upper_bounds)
{
    if (inputs.rows() <= 0 || inputs.cols() <= 0)
    {
        throw std::invalid_argument(
            "input transform requires at least one row and one column");
    }
    if (!inputs.allFinite())
    {
        throw std::invalid_argument("input transform values must be finite");
    }

    FittedInputTransform result;
    const Eigen::Index dimension = inputs.cols();
    result.offset_ = Eigen::VectorXd::Zero(dimension);
    result.scale_ = Eigen::VectorXd::Ones(dimension);

    if (type == InputTransformType::STANDARDIZE)
    {
        result.offset_ = inputs.colwise().mean();
        for (Eigen::Index column = 0; column < dimension; ++column)
        {
            const double variance =
                (inputs.col(column).array() - result.offset_(column))
                    .square()
                    .mean();
            const double candidate = std::sqrt(std::max(0.0, variance));
            result.scale_(column) =
                isEffectivelyZero(candidate, result.offset_(column))
                    ? 1.0
                    : candidate;
        }
    }
    else if (type == InputTransformType::MIN_MAX)
    {
        if (!lower_bounds || !upper_bounds ||
            lower_bounds->size() != dimension ||
            upper_bounds->size() != dimension ||
            !lower_bounds->allFinite() ||
            !upper_bounds->allFinite() ||
            (lower_bounds->array() >= upper_bounds->array()).any())
        {
            throw std::invalid_argument(
                "MIN_MAX requires finite increasing bounds for every dimension");
        }
        result.offset_ = *lower_bounds;
        result.scale_ = *upper_bounds - *lower_bounds;
    }

    return result;
}

Eigen::MatrixXd FittedInputTransform::apply(
    Eigen::Ref<const Eigen::MatrixXd> inputs) const
{
    if (offset_.size() <= 0 || scale_.size() != offset_.size())
    {
        throw std::logic_error("input transform is not fitted");
    }
    if (inputs.cols() != offset_.size())
    {
        throw std::invalid_argument(
            "input transform dimension does not match");
    }
    if (!inputs.allFinite())
    {
        throw std::invalid_argument("input transform values must be finite");
    }

    Eigen::MatrixXd result = inputs;
    result.rowwise() -= offset_.transpose();
    result.array().rowwise() /= scale_.transpose().array();
    return result;
}

Eigen::Index FittedInputTransform::inputDimension() const noexcept
{
    return offset_.size();
}

const Eigen::VectorXd& FittedInputTransform::offset() const noexcept
{
    return offset_;
}

const Eigen::VectorXd& FittedInputTransform::scale() const noexcept
{
    return scale_;
}

}  // namespace bayesian_optimization::surrogate
