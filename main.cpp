#include <bayesian_optimization/surrogate/surrogate.hpp>

#include <Eigen/Core>

#include <cmath>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <string>

int main()
{
    using bayesian_optimization::surrogate::GaussianProcessConfig;
    using bayesian_optimization::surrogate::HybridRegressionModel;
    using bayesian_optimization::surrogate::HyperparameterMode;
    using bayesian_optimization::surrogate::InputTransformType;
    using bayesian_optimization::surrogate::KernelType;
    using bayesian_optimization::surrogate::OutputTransformType;
    using bayesian_optimization::surrogate::PolynomialDegree;
    using bayesian_optimization::surrogate::PolynomialTrendConfig;
    using bayesian_optimization::surrogate::PolynomialTrendModel;
    using bayesian_optimization::surrogate::RegressionDataset;
    using bayesian_optimization::surrogate::ScalarHyperparameterConfig;

    constexpr Eigen::Index observation_count = 40;
    constexpr Eigen::Index input_dimension = 4;

    // 4次元の線形トレンドに局所的な非線形成分を加えた学習データを生成する。
    std::mt19937_64 random_engine(0);
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
            training_inputs(row, column) =
                distribution(random_engine);
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

    // 大域的な傾向を1次のRidge回帰で学習する。
    PolynomialTrendConfig trend_config;
    trend_config.degree = PolynomialDegree::LINEAR;
    trend_config.include_interactions = false;
    trend_config.ridge_lambda = 1.0e-8;
    trend_config.input_transform =
        InputTransformType::STANDARDIZE;

    // Ridge回帰で説明できない残差をGPRで学習する。
    GaussianProcessConfig residual_config;
    residual_config.kernel = KernelType::MATERN_5_2;
    residual_config.use_ard = true;

    residual_config.length_scales.mode =
        HyperparameterMode::OPTIMIZE;
    residual_config.length_scales.values =
        Eigen::VectorXd::Ones(input_dimension);
    residual_config.length_scales.lower_bound = 1.0e-3;
    residual_config.length_scales.upper_bound = 1.0e3;
    residual_config.signal_variance =
        ScalarHyperparameterConfig::optimized(
            1.0,
            1.0e-6,
            1.0e3);
    residual_config.noise_variance =
        ScalarHyperparameterConfig::optimized(
            1.0e-6,
            1.0e-10,
            1.0e1);

    residual_config.preprocessing.input_transform =
        InputTransformType::STANDARDIZE;
    residual_config.preprocessing.output_transform =
        OutputTransformType::STANDARDIZE;

    residual_config.hyperparameter_optimization.restart_count = 5;
    residual_config.hyperparameter_optimization.max_iterations = 200;
    residual_config.hyperparameter_optimization.gradient_tolerance =
        1.0e-6;
    residual_config.hyperparameter_optimization.random_seed = 0;

    residual_config.jitter.initial_relative_value = 1.0e-10;
    residual_config.jitter.multiplier = 10.0;
    residual_config.jitter.max_attempts = 8;

    HybridRegressionModel model(
        std::make_unique<PolynomialTrendModel>(trend_config),
        residual_config);
    model.fit(RegressionDataset(
        training_inputs,
        training_targets));

    std::cout
        << "4次元ハイブリッドモデルの学習が完了しました。\n"
        << "x1 x2 x3 x4 の4値を空白区切りで入力してください"
        << "（q で終了）。\n";

    std::string line;
    while (true)
    {
        std::cout << "> ";
        if (!std::getline(std::cin, line) ||
            line == "q" || line == "quit" || line == "exit")
        {
            break;
        }

        std::istringstream input(line);
        Eigen::RowVector4d point;
        char extra = '\0';
        if (!(input >> point(0) >> point(1) >>
              point(2) >> point(3)) ||
            (input >> extra))
        {
            std::cout
                << "4個の数値、または終了キー q を入力してください。\n";
            continue;
        }

        Eigen::MatrixXd test_inputs(1, input_dimension);
        test_inputs.row(0) = point;

        const auto prediction = model.predict(test_inputs);
        const double trend_prediction =
            model.trendModel().predict(test_inputs)(0) +
            model.residualBias();
        const double residual_prediction =
            model.residualModel().predict(test_inputs).mean(0);

        std::cout
            << "trend mean = " << trend_prediction << '\n'
            << "GPR residual mean = " << residual_prediction << '\n'
            << "hybrid predicted mean = "
            << prediction.mean(0) << '\n'
            << "latent variance = "
            << prediction.latent_variance(0) << '\n'
            << "observation variance = "
            << prediction.observation_variance(0) << '\n';
    }

    std::cout << "終了します。\n";
    return 0;
}
