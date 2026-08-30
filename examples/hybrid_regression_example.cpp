#include <bayesian_optimization/surrogate/surrogate.hpp>

#include <Eigen/Core>

#include <cmath>
#include <iostream>
#include <memory>
#include <random>

namespace surrogate = bayesian_optimization::surrogate;

int main()
{
    constexpr Eigen::Index observation_count = 40;
    constexpr Eigen::Index input_dimension = 4;

    std::mt19937_64 engine(0);
    std::uniform_real_distribution<double> distribution(-1.0, 1.0);
    Eigen::MatrixXd training_inputs(
        observation_count,
        input_dimension);
    for (Eigen::Index row = 0; row < training_inputs.rows(); ++row)
    {
        for (Eigen::Index column = 0;
             column < training_inputs.cols();
             ++column)
        {
            training_inputs(row, column) = distribution(engine);
        }
    }

    Eigen::VectorXd training_targets =
        Eigen::VectorXd::Constant(observation_count, 2.0);
    training_targets.array() +=
        1.5 * training_inputs.col(0).array();
    training_targets.array() -=
        0.8 * training_inputs.col(1).array();
    training_targets.array() +=
        0.3 * training_inputs.col(2).array();
    training_targets.array() +=
        0.5 * training_inputs.col(3).array();
    training_targets.array() +=
        0.2 *
        (3.0 * training_inputs.col(0).array() *
         training_inputs.col(1).array())
            .sin();

    surrogate::PolynomialTrendConfig trend_config;
    trend_config.degree = surrogate::PolynomialDegree::LINEAR;
    trend_config.ridge_lambda = 1.0e-8;

    surrogate::GaussianProcessConfig residual_config;
    residual_config.kernel = surrogate::KernelType::RBF;
    residual_config.hyperparameter_optimization.restart_count = 3;
    residual_config.hyperparameter_optimization.max_iterations = 100;

    surrogate::HybridRegressionModel model(
        std::make_unique<surrogate::PolynomialTrendModel>(
            trend_config),
        residual_config);
    model.fit(surrogate::RegressionDataset(
        training_inputs,
        training_targets));

    Eigen::MatrixXd test_inputs(2, input_dimension);
    test_inputs <<
        0.2, -0.3, 0.1, 0.4,
        2.0, -1.5, 1.2, 1.8;

    const surrogate::Prediction prediction =
        model.predict(test_inputs);
    for (Eigen::Index row = 0; row < test_inputs.rows(); ++row)
    {
        std::cout
            << "point " << row
            << ": mean=" << prediction.mean(row)
            << ", latent_variance="
            << prediction.latent_variance(row)
            << '\n';
    }

    return 0;
}
