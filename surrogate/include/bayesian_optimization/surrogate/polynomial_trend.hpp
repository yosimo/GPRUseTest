// SPDX-License-Identifier: MIT
#pragma once

#include <bayesian_optimization/surrogate/config.hpp>
#include <bayesian_optimization/surrogate/deterministic_trend_model.hpp>

#include <Eigen/Core>

#include <memory>
#include <optional>

namespace bayesian_optimization::surrogate
{

enum class PolynomialDegree
{
    LINEAR = 1,
    QUADRATIC = 2
};

struct PolynomialTrendConfig
{
    PolynomialDegree degree{PolynomialDegree::LINEAR};
    bool include_interactions{true};
    double ridge_lambda{1.0e-8};
    InputTransformType input_transform{InputTransformType::STANDARDIZE};
    std::optional<Eigen::VectorXd> input_lower_bounds;
    std::optional<Eigen::VectorXd> input_upper_bounds;
};

/** @brief Linear or quadratic ridge-regression trend model. */
class PolynomialTrendModel final : public DeterministicTrendModel
{
public:
    explicit PolynomialTrendModel(PolynomialTrendConfig config = {});
    ~PolynomialTrendModel();

    PolynomialTrendModel(const PolynomialTrendModel&) = delete;
    PolynomialTrendModel& operator=(const PolynomialTrendModel&) = delete;
    PolynomialTrendModel(PolynomialTrendModel&&) noexcept;
    PolynomialTrendModel& operator=(PolynomialTrendModel&&) noexcept;

    void fit(const RegressionDataset& dataset) override;
    [[nodiscard]] bool isFitted() const noexcept override;
    [[nodiscard]] Eigen::Index inputDimension() const noexcept override;
    [[nodiscard]] Eigen::VectorXd predict(
        Eigen::Ref<const Eigen::MatrixXd> inputs) const override;
    [[nodiscard]] Eigen::MatrixXd predictGradients(
        Eigen::Ref<const Eigen::MatrixXd> inputs) const override;
    [[nodiscard]] std::unique_ptr<DeterministicTrendModel> clone()
        const override;

    [[nodiscard]] const PolynomialTrendConfig& config() const noexcept;
    [[nodiscard]] const Eigen::VectorXd& coefficients() const;
    [[nodiscard]] Eigen::Index featureCount() const noexcept;
    [[nodiscard]] Eigen::Index featureMatrixRank() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace bayesian_optimization::surrogate
