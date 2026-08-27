// SPDX-License-Identifier: MIT
#pragma once

#include <stdexcept>

namespace bayesian_optimization::surrogate
{

class NumericalError : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

class ModelFormatError : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

}  // namespace bayesian_optimization::surrogate

