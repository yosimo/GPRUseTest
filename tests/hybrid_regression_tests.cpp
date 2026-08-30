#include <bayesian_optimization/surrogate/surrogate.hpp>

#include <Eigen/Core>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>

namespace surrogate = bayesian_optimization::surrogate;

namespace
{

void require(bool condition, const std::string& message)
{
    if (!condition)
    {
        throw std::runtime_error(message);
    }
}

void requireNear(
    double actual,
    double expected,
    double tolerance,
    const std::string& message)
{
    if (!std::isfinite(actual) ||
        std::abs(actual - expected) > tolerance)
    {
        throw std::runtime_error(
            message + ": actual=" + std::to_string(actual) +
            ", expected=" + std::to_string(expected));
    }
}

Eigen::MatrixXd randomInputs(
    Eigen::Index rows,
    Eigen::Index columns,
    std::uint64_t seed)
{
    std::mt19937_64 engine(seed);
    std::uniform_real_distribution<double> distribution(-1.0, 1.0);
    Eigen::MatrixXd result(rows, columns);
    for (Eigen::Index row = 0; row < rows; ++row)
    {
        for (Eigen::Index column = 0; column < columns; ++column)
        {
            result(row, column) = distribution(engine);
        }
    }
    return result;
}

Eigen::VectorXd linearTargets(const Eigen::MatrixXd& inputs)
{
    Eigen::VectorXd result =
        Eigen::VectorXd::Constant(inputs.rows(), 2.0);
    result.array() += 1.5 * inputs.col(0).array();
    result.array() -= 0.8 * inputs.col(1).array();
    result.array() += 0.3 * inputs.col(2).array();
    result.array() += 0.5 * inputs.col(3).array();
    return result;
}

Eigen::VectorXd quadraticTargets(const Eigen::MatrixXd& inputs)
{
    Eigen::VectorXd result = linearTargets(inputs);
    result.array() += 0.4 * inputs.col(0).array().square();
    result.array() -= 0.2 * inputs.col(1).array().square();
    result.array() += 0.1 * inputs.col(2).array().square();
    result.array() += 0.3 * inputs.col(3).array().square();
    result.array() +=
        0.25 * inputs.col(0).array() * inputs.col(1).array();
    result.array() -=
        0.15 * inputs.col(2).array() * inputs.col(3).array();
    return result;
}

void testLinearTrend()
{
    const Eigen::MatrixXd inputs = randomInputs(24, 4, 1);
    const Eigen::VectorXd targets = linearTargets(inputs);

    surrogate::PolynomialTrendConfig config;
    config.degree = surrogate::PolynomialDegree::LINEAR;
    config.ridge_lambda = 0.0;
    surrogate::PolynomialTrendModel model(config);
    model.fit(surrogate::RegressionDataset(inputs, targets));

    require(model.featureCount() == 5, "linear feature count");
    require(model.featureMatrixRank() == 5, "linear feature rank");

    const Eigen::MatrixXd test_inputs = randomInputs(7, 4, 2);
    const Eigen::VectorXd prediction = model.predict(test_inputs);
    const Eigen::VectorXd expected = linearTargets(test_inputs);
    require(
        (prediction - expected).cwiseAbs().maxCoeff() < 1.0e-10,
        "linear trend prediction");

    const Eigen::MatrixXd gradient =
        model.predictGradients(test_inputs);
    const Eigen::RowVector4d expected_gradient(1.5, -0.8, 0.3, 0.5);
    for (Eigen::Index row = 0; row < gradient.rows(); ++row)
    {
        require(
            (gradient.row(row) - expected_gradient)
                    .cwiseAbs()
                    .maxCoeff() < 1.0e-10,
            "linear trend gradient");
    }
}

void testQuadraticTrendAndGradient()
{
    const Eigen::MatrixXd inputs = randomInputs(48, 4, 3);
    const Eigen::VectorXd targets = quadraticTargets(inputs);

    surrogate::PolynomialTrendConfig config;
    config.degree = surrogate::PolynomialDegree::QUADRATIC;
    config.include_interactions = true;
    config.ridge_lambda = 0.0;
    surrogate::PolynomialTrendModel model(config);
    model.fit(surrogate::RegressionDataset(inputs, targets));

    require(model.featureCount() == 15, "quadratic feature count");
    require(model.featureMatrixRank() == 15, "quadratic feature rank");

    Eigen::MatrixXd point(1, 4);
    point << 0.2, -0.4, 0.1, 0.6;
    requireNear(
        model.predict(point)(0),
        quadraticTargets(point)(0),
        1.0e-10,
        "quadratic prediction");

    const Eigen::MatrixXd analytic =
        model.predictGradients(point);
    const double step = 1.0e-6;
    for (Eigen::Index dimension = 0; dimension < point.cols(); ++dimension)
    {
        Eigen::MatrixXd plus = point;
        Eigen::MatrixXd minus = point;
        plus(0, dimension) += step;
        minus(0, dimension) -= step;
        const double finite_difference =
            (model.predict(plus)(0) - model.predict(minus)(0)) /
            (2.0 * step);
        requireNear(
            analytic(0, dimension),
            finite_difference,
            1.0e-6,
            "quadratic gradient");
    }
}

surrogate::GaussianProcessConfig fixedResidualConfig()
{
    surrogate::GaussianProcessConfig config;
    config.kernel = surrogate::KernelType::RBF;
    config.use_ard = true;
    config.length_scales.mode =
        surrogate::HyperparameterMode::FIXED;
    config.length_scales.values = Eigen::VectorXd::Ones(4);
    config.signal_variance =
        surrogate::ScalarHyperparameterConfig::fixed(1.0);
    config.noise_variance =
        surrogate::ScalarHyperparameterConfig::fixed(1.0e-8);
    return config;
}

void testHybridComposition()
{
    const Eigen::MatrixXd inputs = randomInputs(32, 4, 4);
    Eigen::VectorXd targets = linearTargets(inputs);
    targets.array() +=
        0.2 *
        (3.0 * inputs.col(0).array() * inputs.col(1).array()).sin();

    surrogate::PolynomialTrendConfig trend_config;
    trend_config.degree = surrogate::PolynomialDegree::LINEAR;
    trend_config.ridge_lambda = 1.0e-8;

    auto trend =
        std::make_unique<surrogate::PolynomialTrendModel>(trend_config);
    surrogate::HybridRegressionModel model(
        std::move(trend),
        fixedResidualConfig());
    model.fit(surrogate::RegressionDataset(inputs, targets));

    require(model.isFitted(), "hybrid fitted state");
    require(model.inputDimension() == 4, "hybrid input dimension");

    const Eigen::MatrixXd test_inputs = randomInputs(6, 4, 5);
    const surrogate::Prediction prediction =
        model.predict(test_inputs);
    const Eigen::VectorXd expected_mean =
        model.trendModel().predict(test_inputs).array() +
        model.residualBias() +
        model.residualModel().predict(test_inputs).mean.array();
    require(
        (prediction.mean - expected_mean).cwiseAbs().maxCoeff() <
            1.0e-12,
        "hybrid mean composition");

    const surrogate::PredictionWithGradients with_gradients =
        model.predictWithGradients(test_inputs);
    const Eigen::MatrixXd expected_gradient =
        model.trendModel().predictGradients(test_inputs) +
        model.residualModel()
            .predictWithGradients(test_inputs)
            .mean_gradient;
    require(
        (with_gradients.mean_gradient - expected_gradient)
                .cwiseAbs()
                .maxCoeff() < 1.0e-12,
        "hybrid gradient composition");

    const surrogate::Prediction residual_prediction =
        model.residualModel().predict(test_inputs);
    require(
        (prediction.latent_variance -
         residual_prediction.latent_variance)
                .cwiseAbs()
                .maxCoeff() < 1.0e-12,
        "hybrid latent variance");
    require(
        (prediction.observation_variance -
         residual_prediction.observation_variance)
                .cwiseAbs()
                .maxCoeff() < 1.0e-12,
        "hybrid observation variance");

    Eigen::MatrixXd far_input = Eigen::MatrixXd::Constant(1, 4, 50.0);
    const double far_hybrid = model.predict(far_input).mean(0);
    const double far_trend =
        model.trendModel().predict(far_input)(0) +
        model.residualBias();
    requireNear(
        far_hybrid,
        far_trend,
        1.0e-8,
        "far-field prediction returns to trend");
}

}  // namespace

int main()
{
    try
    {
        testLinearTrend();
        testQuadraticTrendAndGradient();
        testHybridComposition();
        std::cout << "hybrid regression tests: OK\n";
        return EXIT_SUCCESS;
    }
    catch (const std::exception& error)
    {
        std::cerr << "hybrid regression tests failed: "
                  << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
