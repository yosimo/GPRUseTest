#include <bayesian_optimization/surrogate/surrogate.hpp>

#include <Eigen/Core>
#include <csv.h>

#include <array>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace surrogate = bayesian_optimization::surrogate;

namespace
{

constexpr Eigen::Index INPUT_DIMENSION = 4;
constexpr const char* DEFAULT_CSV_PATH =
    "data/hybrid_training_4d.csv";

// ヘッダ有無の判定を数値行の解析から分離し、CSV形式の違いを
// この関数内だけで吸収する。戻るとinputは最初のデータ行を指す。
bool skipOptionalHeader(std::ifstream& input)
{
    std::string first_line;
    if (!std::getline(input, first_line))
    {
        throw std::invalid_argument("CSVファイルが空です");
    }

    std::string compact;
    for (char character : first_line)
    {
        if (character != ' ' &&
            character != '\t' &&
            character != '\r')
        {
            compact.push_back(character);
        }
    }

    const bool has_header =
        compact == "x1,x2,x3,x4,y";
    if (!has_header)
    {
        input.clear();
        input.seekg(0);
        if (!input)
        {
            throw std::runtime_error(
                "CSVファイルの先頭へ戻れませんでした");
        }
    }
    return has_header;
}

// ファイルI/O、CSV解析、値検証、Eigen形式への変換をモデル構築から
// 分離する。以降の処理は検証済みRegressionDatasetだけを扱える。
surrogate::RegressionDataset loadTrainingData(
    const std::string& csv_path)
{
    std::ifstream input(csv_path, std::ios::binary);
    if (!input)
    {
        throw std::runtime_error(
            "CSVファイルを開けません: " + csv_path);
    }

    const bool has_header = skipOptionalHeader(input);
    io::CSVReader<5> reader(csv_path, input);
    reader.set_header("x1", "x2", "x3", "x4", "y");
    if (has_header)
    {
        reader.set_file_line(1);
    }

    std::vector<std::array<double, 5>> rows;
    std::array<double, 5> row{};
    while (reader.read_row(
        row[0],
        row[1],
        row[2],
        row[3],
        row[4]))
    {
        for (double value : row)
        {
            if (!std::isfinite(value))
            {
                throw std::invalid_argument(
                    "CSVには有限の数値だけを指定してください");
            }
        }
        rows.push_back(row);
    }

    if (rows.empty())
    {
        throw std::invalid_argument(
            "CSVに学習データがありません");
    }

    Eigen::MatrixXd inputs(
        static_cast<Eigen::Index>(rows.size()),
        INPUT_DIMENSION);
    Eigen::VectorXd targets(
        static_cast<Eigen::Index>(rows.size()));
    for (std::size_t index = 0; index < rows.size(); ++index)
    {
        const Eigen::Index output_index =
            static_cast<Eigen::Index>(index);
        for (Eigen::Index dimension = 0;
             dimension < INPUT_DIMENSION;
             ++dimension)
        {
            inputs(output_index, dimension) =
                rows[index][static_cast<std::size_t>(dimension)];
        }
        targets(output_index) = rows[index][4];
    }

    return surrogate::RegressionDataset(
        std::move(inputs),
        std::move(targets));
}

// 大域的な外挿傾向を決める設定を1か所にまとめる。
// 次数やRidge強度を比較するときは、この関数だけを変更すればよい。
surrogate::PolynomialTrendConfig makeTrendConfig()
{
    surrogate::PolynomialTrendConfig config;
    config.degree = surrogate::PolynomialDegree::LINEAR;
    config.include_interactions = false;
    config.ridge_lambda = 1.0e-8;
    config.input_transform =
        surrogate::InputTransformType::STANDARDIZE;
    return config;
}

// 局所的な残差補正を担当するGPR設定をトレンド設定から分離する。
// カーネル、ARD、最適化条件、数値安定化条件をここで一括管理する。
surrogate::GaussianProcessConfig makeResidualConfig()
{
    surrogate::GaussianProcessConfig config;
    config.kernel = surrogate::KernelType::MATERN_5_2;
    config.use_ard = true;

    config.length_scales.mode =
        surrogate::HyperparameterMode::OPTIMIZE;
    config.length_scales.values =
        Eigen::VectorXd::Ones(INPUT_DIMENSION);
    config.length_scales.lower_bound = 1.0e-3;
    config.length_scales.upper_bound = 1.0e3;
    config.signal_variance =
        surrogate::ScalarHyperparameterConfig::optimized(
            1.0,
            1.0e-6,
            1.0e3);
    config.noise_variance =
        surrogate::ScalarHyperparameterConfig::optimized(
            1.0e-6,
            1.0e-10,
            1.0e1);

    config.preprocessing.input_transform =
        surrogate::InputTransformType::STANDARDIZE;
    config.preprocessing.output_transform =
        surrogate::OutputTransformType::STANDARDIZE;

    config.hyperparameter_optimization.restart_count = 5;
    config.hyperparameter_optimization.max_iterations = 200;
    config.hyperparameter_optimization.gradient_tolerance =
        1.0e-6;
    config.hyperparameter_optimization.random_seed = 0;

    config.jitter.initial_relative_value = 1.0e-10;
    config.jitter.multiplier = 10.0;
    config.jitter.max_attempts = 8;
    return config;
}

// 「読込→モデル構築→学習→対話予測」というサンプル本体をまとめる。
// mainは引数処理と例外報告だけにし、この処理を別の入口からも再利用しやすくする。
int run(const std::string& csv_path)
{
    // 1. CSVをモデルが受け取れる検証済みデータセットへ変換する。
    const surrogate::RegressionDataset training_data =
        loadTrainingData(csv_path);

    // 2. 大域トレンドと残差GPRを組み合わせてモデルを構築する。
    surrogate::HybridRegressionModel model(
        std::make_unique<surrogate::PolynomialTrendModel>(
            makeTrendConfig()),
        makeResidualConfig());
    // 3. fit内部でトレンドを先に学習し、その残差をGPRで学習する。
    model.fit(training_data);

    // 4. 学習済みモデルに対して任意の4次元点を繰り返し予測する。
    std::cout
        << training_data.observationCount()
        << "件の学習データを読み込みました: "
        << csv_path << '\n'
        << "4次元ハイブリッドモデルの学習が完了しました。\n"
        << "x1 x2 x3 x4 の4値を空白区切りで入力してください"
        << "（q で終了）。\n";

    std::string line;
    while (true)
    {
        std::cout << "> ";
        if (!std::getline(std::cin, line) ||
            line == "q" ||
            line == "quit" ||
            line == "exit")
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

        Eigen::MatrixXd test_inputs(1, INPUT_DIMENSION);
        test_inputs.row(0) = point;

        // 合成結果だけでなく各成分も表示し、トレンドとGPRの
        // 役割分担をサンプル実行時に確認できるようにする。
        const surrogate::Prediction prediction =
            model.predict(test_inputs);
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
    return EXIT_SUCCESS;
}

}  // namespace

// コマンドライン境界をrunから分離し、使用方法と致命的エラーを
// 1か所で処理する。
int main(int argc, char* argv[])
{
    if (argc > 2)
    {
        std::cerr
            << "使用方法: " << argv[0]
            << " [training_data.csv]\n";
        return EXIT_FAILURE;
    }

    const std::string csv_path =
        argc == 2 ? argv[1] : DEFAULT_CSV_PATH;
    try
    {
        return run(csv_path);
    }
    catch (const std::exception& error)
    {
        std::cerr
            << "学習データの読み込みまたはモデル学習に失敗しました: "
            << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
