// SPDX-License-Identifier: MIT
#pragma once

#include <Eigen/Core>

#include <cstddef>
#include <cstdint>
#include <optional>

namespace bayesian_optimization::surrogate
{

enum class KernelType
{
    RBF,
    MATERN_5_2
};

enum class HyperparameterMode
{
    FIXED,
    OPTIMIZE
};

enum class InputTransformType
{
    IDENTITY,
    STANDARDIZE,
    MIN_MAX
};

enum class OutputTransformType
{
    IDENTITY,
    STANDARDIZE
};

enum class SurrogateFitPolicy
{
    CONFIGURED,
    REUSE_MODEL_PARAMETERS
};

struct ScalarHyperparameterConfig
{
    HyperparameterMode mode{HyperparameterMode::OPTIMIZE};
    double value{1.0};
    double lower_bound{1.0e-6};
    double upper_bound{1.0e3};

    [[nodiscard]] static ScalarHyperparameterConfig fixed(double value);
    [[nodiscard]] static ScalarHyperparameterConfig optimized(
        double initial_value,
        double lower_bound,
        double upper_bound);
};

struct LengthScaleConfig
{
    HyperparameterMode mode{HyperparameterMode::OPTIMIZE};
    std::optional<Eigen::VectorXd> values;
    double lower_bound{1.0e-3};
    double upper_bound{1.0e3};
};

struct PreprocessingConfig
{
    InputTransformType input_transform{InputTransformType::STANDARDIZE};
    OutputTransformType output_transform{OutputTransformType::STANDARDIZE};
    std::optional<Eigen::VectorXd> input_lower_bounds;
    std::optional<Eigen::VectorXd> input_upper_bounds;
};

struct HyperparameterOptimizationConfig
{
    std::size_t restart_count{5};
    std::size_t max_iterations{200};
    double gradient_tolerance{1.0e-6};
    std::uint64_t random_seed{0};
};

struct JitterPolicy
{
    double initial_relative_value{1.0e-10};
    double multiplier{10.0};
    std::size_t max_attempts{8};
};

struct GaussianProcessConfig
{
    KernelType kernel{KernelType::MATERN_5_2};
    bool use_ard{true};
    LengthScaleConfig length_scales;
    ScalarHyperparameterConfig signal_variance{
        ScalarHyperparameterConfig::optimized(1.0, 1.0e-6, 1.0e3)};
    ScalarHyperparameterConfig noise_variance{
        ScalarHyperparameterConfig::optimized(1.0e-6, 1.0e-10, 1.0e1)};
    PreprocessingConfig preprocessing;
    HyperparameterOptimizationConfig hyperparameter_optimization;
    JitterPolicy jitter;

    void validateForDimension(Eigen::Index input_dimension) const;
};

}  // namespace bayesian_optimization::surrogate

