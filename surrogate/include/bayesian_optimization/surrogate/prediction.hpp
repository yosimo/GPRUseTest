// SPDX-License-Identifier: MIT
#pragma once

#include <Eigen/Core>

namespace bayesian_optimization::surrogate
{

struct Prediction
{
    Eigen::VectorXd mean;
    Eigen::VectorXd latent_variance;
    Eigen::VectorXd observation_variance;
};

struct PredictionWithGradients
{
    Prediction prediction;
    Eigen::MatrixXd mean_gradient;
    Eigen::MatrixXd latent_variance_gradient;
};

struct OutputTransform
{
    double offset{0.0};
    double scale{1.0};
};

struct FittedHyperparameters
{
    Eigen::VectorXd length_scales;
    double signal_variance{0.0};
    double noise_variance{0.0};
    double effective_jitter{0.0};
    double log_marginal_likelihood{0.0};
};

}  // namespace bayesian_optimization::surrogate

