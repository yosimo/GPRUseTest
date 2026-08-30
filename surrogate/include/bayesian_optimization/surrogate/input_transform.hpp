// SPDX-License-Identifier: MIT
#pragma once

#include <bayesian_optimization/surrogate/config.hpp>

#include <Eigen/Core>

#include <optional>

namespace bayesian_optimization::surrogate
{

/** @brief Fitted affine transform for regression inputs. */
class FittedInputTransform final
{
public:
    FittedInputTransform() = default;

    [[nodiscard]] static FittedInputTransform fit(
        Eigen::Ref<const Eigen::MatrixXd> inputs,
        InputTransformType type,
        const std::optional<Eigen::VectorXd>& lower_bounds = std::nullopt,
        const std::optional<Eigen::VectorXd>& upper_bounds = std::nullopt);

    [[nodiscard]] Eigen::MatrixXd apply(
        Eigen::Ref<const Eigen::MatrixXd> inputs) const;

    [[nodiscard]] Eigen::Index inputDimension() const noexcept;
    [[nodiscard]] const Eigen::VectorXd& offset() const noexcept;
    [[nodiscard]] const Eigen::VectorXd& scale() const noexcept;

private:
    Eigen::VectorXd offset_;
    Eigen::VectorXd scale_;
};

}  // namespace bayesian_optimization::surrogate
