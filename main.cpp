#include <bayesian_optimization/surrogate/surrogate.hpp>

#include <Eigen/Core>

#include <iostream>
#include <sstream>
#include <string>

int main()
{
    using bayesian_optimization::surrogate::GaussianProcess;
    using bayesian_optimization::surrogate::GaussianProcessConfig;
    using bayesian_optimization::surrogate::HyperparameterMode;
    using bayesian_optimization::surrogate::InputTransformType;
    using bayesian_optimization::surrogate::KernelType;
    using bayesian_optimization::surrogate::OutputTransformType;
    using bayesian_optimization::surrogate::RegressionDataset;
    using bayesian_optimization::surrogate::ScalarHyperparameterConfig;

    // y = x^2 の5点を学習データとして与える。
    Eigen::MatrixXd training_inputs(5, 1);
    training_inputs << -2.0, -1.0, 0.0, 1.0, 2.0;

    Eigen::VectorXd training_targets(5);
    training_targets << 4.0, 1.0, 0.0, 1.0, 4.0;

    GaussianProcessConfig config;

    // カーネルとARDを設定する。
    config.kernel = KernelType::MATERN_5_2;
    config.use_ard = true;

    // ハイパーパラメータの初期値と探索範囲を設定する。
    config.length_scales.mode = HyperparameterMode::OPTIMIZE;
    config.length_scales.values =
        Eigen::VectorXd::Ones(training_inputs.cols());
    config.length_scales.lower_bound = 1.0e-3;
    config.length_scales.upper_bound = 1.0e3;
    config.signal_variance =
        ScalarHyperparameterConfig::optimized(1.0, 1.0e-6, 1.0e3);
    config.noise_variance =
        ScalarHyperparameterConfig::optimized(1.0e-6, 1.0e-10, 1.0e1);

    // 入出力を標準化する。
    config.preprocessing.input_transform = InputTransformType::STANDARDIZE;
    config.preprocessing.output_transform = OutputTransformType::STANDARDIZE;

    // ハイパーパラメータ最適化を設定する。
    config.hyperparameter_optimization.restart_count = 5;
    config.hyperparameter_optimization.max_iterations = 200;
    config.hyperparameter_optimization.gradient_tolerance = 1.0e-6;
    config.hyperparameter_optimization.random_seed = 0;

    // Cholesky分解を安定化するjitterを設定する。
    config.jitter.initial_relative_value = 1.0e-10;
    config.jitter.multiplier = 10.0;
    config.jitter.max_attempts = 8;

    GaussianProcess model(config);
    model.fit(RegressionDataset(training_inputs, training_targets));

    std::cout << "モデルの学習が完了しました。\n"
              << "予測したい x の値を入力してください（q で終了）。\n";

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
        double value = 0.0;
        char extra = '\0';
        if (!(input >> value) || (input >> extra))
        {
            std::cout << "数値、または終了キー q を入力してください。\n";
            continue;
        }

        Eigen::MatrixXd test_inputs(1, 1);
        test_inputs << value;
        const auto prediction = model.predict(test_inputs);

        std::cout << "x = " << value << '\n'
                  << "predicted mean = " << prediction.mean(0) << '\n'
                  << "latent variance = " << prediction.latent_variance(0)
                  << '\n';
    }

    std::cout << "終了します。\n";

    return 0;
}
