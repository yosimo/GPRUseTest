#include <bayesian_optimization/surrogate/surrogate.hpp>

#include <Eigen/Core>

#include <iostream>
#include <sstream>
#include <string>

int main()
{
    using bayesian_optimization::surrogate::GaussianProcess;
    using bayesian_optimization::surrogate::RegressionDataset;

    // y = x^2 の5点を学習データとして与える。
    Eigen::MatrixXd training_inputs(5, 1);
    training_inputs << -2.0, -1.0, 0.0, 1.0, 2.0;

    Eigen::VectorXd training_targets(5);
    training_targets << 4.0, 1.0, 0.0, 1.0, 4.0;

    GaussianProcess model;
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
