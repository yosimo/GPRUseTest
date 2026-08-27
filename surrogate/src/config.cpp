// SPDX-License-Identifier: MIT

#include <bayesian_optimization/surrogate/config.hpp>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace bayesian_optimization::surrogate
{
namespace
{

void validatePositiveFinite(double value, const char* name)
{
    if (!std::isfinite(value) || value <= 0.0)
    {
        throw std::invalid_argument(std::string(name) + " must be finite and positive");
    }
}

void validateScalar(
    const ScalarHyperparameterConfig& parameter,
    bool allow_fixed_zero,
    const char* name)
{
    if (!std::isfinite(parameter.value) ||
        parameter.value < (allow_fixed_zero &&
                                   parameter.mode == HyperparameterMode::FIXED
                               ? 0.0
                               : std::numeric_limits<double>::min()))
    {
        throw std::invalid_argument(std::string(name) + " has an invalid value");
    }
    if (parameter.mode == HyperparameterMode::OPTIMIZE)
    {
        validatePositiveFinite(parameter.lower_bound, "lower_bound");
        validatePositiveFinite(parameter.upper_bound, "upper_bound");
        if (parameter.lower_bound >= parameter.upper_bound ||
            parameter.value < parameter.lower_bound ||
            parameter.value > parameter.upper_bound)
        {
            throw std::invalid_argument(
                std::string(name) + " value must be inside valid bounds");
        }
    }
}

}  // namespace

ScalarHyperparameterConfig ScalarHyperparameterConfig::fixed(double value)
{
    ScalarHyperparameterConfig result;
    result.mode = HyperparameterMode::FIXED;
    result.value = value;
    return result;
}

ScalarHyperparameterConfig ScalarHyperparameterConfig::optimized(
    double initial_value,
    double lower_bound,
    double upper_bound)
{
    ScalarHyperparameterConfig result;
    result.mode = HyperparameterMode::OPTIMIZE;
    result.value = initial_value;
    result.lower_bound = lower_bound;
    result.upper_bound = upper_bound;
    return result;
}

void GaussianProcessConfig::validateForDimension(
    Eigen::Index input_dimension) const
{
    if (input_dimension <= 0)
    {
        throw std::invalid_argument("input_dimension must be positive");
    }

    validateScalar(signal_variance, false, "signal_variance");
    validateScalar(noise_variance, true, "noise_variance");
    validatePositiveFinite(length_scales.lower_bound, "length_scale lower_bound");
    validatePositiveFinite(length_scales.upper_bound, "length_scale upper_bound");
    if (length_scales.lower_bound >= length_scales.upper_bound)
    {
        throw std::invalid_argument("length scale bounds must be increasing");
    }

    if (length_scales.values)
    {
        const auto size = length_scales.values->size();
        if (size != 1 && size != input_dimension)
        {
            throw std::invalid_argument(
                "length scales must contain one value or input_dimension values");
        }
        if (!length_scales.values->allFinite() ||
            (length_scales.values->array() <= 0.0).any())
        {
            throw std::invalid_argument(
                "length scales must be finite and positive");
        }
    }
    else if (length_scales.mode == HyperparameterMode::FIXED)
    {
        throw std::invalid_argument(
            "fixed length scales require explicit values");
    }

    if (preprocessing.input_transform == InputTransformType::MIN_MAX)
    {
        if (!preprocessing.input_lower_bounds ||
            !preprocessing.input_upper_bounds ||
            preprocessing.input_lower_bounds->size() != input_dimension ||
            preprocessing.input_upper_bounds->size() != input_dimension ||
            !preprocessing.input_lower_bounds->allFinite() ||
            !preprocessing.input_upper_bounds->allFinite() ||
            (preprocessing.input_lower_bounds->array() >=
             preprocessing.input_upper_bounds->array())
                .any())
        {
            throw std::invalid_argument(
                "MIN_MAX requires finite increasing bounds for every dimension");
        }
    }

    if (hyperparameter_optimization.restart_count == 0 ||
        hyperparameter_optimization.max_iterations == 0)
    {
        throw std::invalid_argument(
            "hyperparameter optimization counts must be positive");
    }
    validatePositiveFinite(
        hyperparameter_optimization.gradient_tolerance,
        "gradient_tolerance");
    validatePositiveFinite(jitter.initial_relative_value, "initial jitter");
    if (!std::isfinite(jitter.multiplier) || jitter.multiplier <= 1.0 ||
        jitter.max_attempts == 0)
    {
        throw std::invalid_argument("jitter policy is invalid");
    }
}

}  // namespace bayesian_optimization::surrogate
