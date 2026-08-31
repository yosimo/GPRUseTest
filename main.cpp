#include <bayesian_optimization/surrogate/surrogate.hpp>

#include <Eigen/Core>
#include <csv.h>

#include <array>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
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
constexpr const char* DEFAULT_TRAINING_CSV_PATH =
    "data/hybrid_training_4d.csv";

// ヘッダ有無の判定を数値行の解析から分離し、CSV形式の違いを
// この関数内だけで吸収する。戻るとinputは最初のデータ行を指す。
bool skipOptionalHeader(
    std::ifstream& input,
    const std::string& expected_header)
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
        compact == expected_header;
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

void validateFinite(
    const double* values,
    std::size_t count,
    const char* csv_kind)
{
    for (std::size_t index = 0; index < count; ++index)
    {
        if (!std::isfinite(values[index]))
        {
            throw std::invalid_argument(
                std::string(csv_kind) +
                "CSVには有限の数値だけを指定してください");
        }
    }
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
            "学習CSVファイルを開けません: " + csv_path);
    }

    const bool has_header =
        skipOptionalHeader(input, "x1,x2,x3,x4,y");
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
        validateFinite(
            row.data(),
            row.size(),
            "学習");
        rows.push_back(row);
    }

    if (rows.empty())
    {
        throw std::invalid_argument(
            "学習CSVにデータがありません");
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

// テストCSVの読込を対話入力から分離し、複数点を一括予測できる
// Eigen行列へ変換する。学習CSVと異なり目的値yは持たない。
Eigen::MatrixXd loadTestInputs(
    const std::string& csv_path)
{
    std::ifstream input(csv_path, std::ios::binary);
    if (!input)
    {
        throw std::runtime_error(
            "テストCSVファイルを開けません: " + csv_path);
    }

    const bool has_header =
        skipOptionalHeader(input, "x1,x2,x3,x4");
    io::CSVReader<4> reader(csv_path, input);
    reader.set_header("x1", "x2", "x3", "x4");
    if (has_header)
    {
        reader.set_file_line(1);
    }

    std::vector<std::array<double, 4>> rows;
    std::array<double, 4> row{};
    while (reader.read_row(
        row[0],
        row[1],
        row[2],
        row[3]))
    {
        validateFinite(
            row.data(),
            row.size(),
            "テスト");
        rows.push_back(row);
    }

    if (rows.empty())
    {
        throw std::invalid_argument(
            "テストCSVにデータがありません");
    }

    Eigen::MatrixXd inputs(
        static_cast<Eigen::Index>(rows.size()),
        INPUT_DIMENSION);
    for (std::size_t index = 0; index < rows.size(); ++index)
    {
        for (Eigen::Index dimension = 0;
             dimension < INPUT_DIMENSION;
             ++dimension)
        {
            inputs(
                static_cast<Eigen::Index>(index),
                dimension) =
                rows[index][static_cast<std::size_t>(dimension)];
        }
    }
    return inputs;
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

// GPR単体とハイブリッドの残差GPRに同じ設定を使用する。
// これにより両者の差は学習対象がy全体か残差か、という点に限定される。
surrogate::GaussianProcessConfig makeGaussianProcessConfig()
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

// 3モデルを常に同じ学習データでfitするため、生成と学習をまとめる。
// trendとGPRは比較用、hybridは両者を組み合わせた本命モデルである。
class ComparisonModels
{
public:
    ComparisonModels()
        : trend_(makeTrendConfig()),
          gpr_(makeGaussianProcessConfig()),
          hybrid_(
              std::make_unique<
                  surrogate::PolynomialTrendModel>(
                  makeTrendConfig()),
              makeGaussianProcessConfig())
    {
    }

    void fit(const surrogate::RegressionDataset& data)
    {
        trend_.fit(data);
        gpr_.fit(data);
        hybrid_.fit(data);
    }

    [[nodiscard]]
    const surrogate::PolynomialTrendModel& trend() const noexcept
    {
        return trend_;
    }

    [[nodiscard]]
    const surrogate::GaussianProcess& gpr() const noexcept
    {
        return gpr_;
    }

    [[nodiscard]]
    const surrogate::HybridRegressionModel& hybrid() const noexcept
    {
        return hybrid_;
    }

private:
    surrogate::PolynomialTrendModel trend_;
    surrogate::GaussianProcess gpr_;
    surrogate::HybridRegressionModel hybrid_;
};

struct ComparisonPrediction
{
    Eigen::VectorXd trend_mean;
    surrogate::Prediction gpr;
    surrogate::Prediction hybrid;
};

// 3モデルの予測呼出しを1か所に集約し、対話入力とCSV入力が
// 同一の比較ロジックを利用するようにする。
ComparisonPrediction predictAll(
    const ComparisonModels& models,
    Eigen::Ref<const Eigen::MatrixXd> inputs)
{
    return {
        models.trend().predict(inputs),
        models.gpr().predict(inputs),
        models.hybrid().predict(inputs)};
}

void printSinglePrediction(
    const ComparisonPrediction& prediction)
{
    std::cout
        << "trend-only mean = "
        << prediction.trend_mean(0) << '\n'
        << "GPR-only mean = "
        << prediction.gpr.mean(0) << '\n'
        << "GPR-only latent variance = "
        << prediction.gpr.latent_variance(0) << '\n'
        << "GPR-only observation variance = "
        << prediction.gpr.observation_variance(0) << '\n'
        << "hybrid mean = "
        << prediction.hybrid.mean(0) << '\n'
        << "hybrid latent variance = "
        << prediction.hybrid.latent_variance(0) << '\n'
        << "hybrid observation variance = "
        << prediction.hybrid.observation_variance(0) << '\n';
}

// CSVテスト入力では、入力値と3モデルの予測を1行へ並べる。
// 出力を再利用しやすいよう、ヘッダ付きCSV形式で標準出力へ出す。
void printBatchPredictions(
    Eigen::Ref<const Eigen::MatrixXd> inputs,
    const ComparisonPrediction& prediction)
{
    std::cout
        << "x1,x2,x3,x4,"
        << "trend_mean,"
        << "gpr_mean,gpr_latent_variance,"
        << "gpr_observation_variance,"
        << "hybrid_mean,hybrid_latent_variance,"
        << "hybrid_observation_variance\n"
        << std::setprecision(10);

    for (Eigen::Index row = 0; row < inputs.rows(); ++row)
    {
        std::cout
            << inputs(row, 0) << ','
            << inputs(row, 1) << ','
            << inputs(row, 2) << ','
            << inputs(row, 3) << ','
            << prediction.trend_mean(row) << ','
            << prediction.gpr.mean(row) << ','
            << prediction.gpr.latent_variance(row) << ','
            << prediction.gpr.observation_variance(row) << ','
            << prediction.hybrid.mean(row) << ','
            << prediction.hybrid.latent_variance(row) << ','
            << prediction.hybrid.observation_variance(row)
            << '\n';
    }
}

void runInteractive(const ComparisonModels& models)
{
    std::cout
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
        printSinglePrediction(
            predictAll(models, test_inputs));
    }

    std::cout << "終了します。\n";
}

// 「読込→3モデル学習→対話またはCSV一括予測」をまとめる。
// mainは引数処理と例外報告だけにし、処理本体の順序を明確に保つ。
int run(
    const std::string& training_csv_path,
    const std::string* test_csv_path)
{
    // 1. 同一の検証済みデータセットを3モデルへ渡す。
    const surrogate::RegressionDataset training_data =
        loadTrainingData(training_csv_path);

    // 2. トレンド単体、GPR単体、ハイブリッドを独立に学習する。
    ComparisonModels models;
    models.fit(training_data);

    // バッチモードでは標準出力を予測CSVだけにするため、
    // 進捗メッセージを標準エラーへ分離する。
    std::ostream& status_output =
        test_csv_path ? std::cerr : std::cout;
    status_output
        << training_data.observationCount()
        << "件の学習データを読み込みました: "
        << training_csv_path << '\n'
        << "トレンド単体、GPR単体、ハイブリッドの学習が完了しました。\n";

    if (test_csv_path)
    {
        // 3a. テストCSVがあれば全行を一括予測して終了する。
        const Eigen::MatrixXd test_inputs =
            loadTestInputs(*test_csv_path);
        status_output
            << test_inputs.rows()
            << "件のテストデータを読み込みました: "
            << *test_csv_path << '\n';
        printBatchPredictions(
            test_inputs,
            predictAll(models, test_inputs));
    }
    else
    {
        // 3b. テストCSVがなければ従来どおり対話入力を受け付ける。
        runInteractive(models);
    }
    return EXIT_SUCCESS;
}

}  // namespace

// コマンドライン境界をrunから分離し、使用方法と致命的エラーを
// 1か所で処理する。
int main(int argc, char* argv[])
{
    if (argc > 3)
    {
        std::cerr
            << "使用方法: " << argv[0]
            << " [training_data.csv [test_data.csv]]\n";
        return EXIT_FAILURE;
    }

    const std::string training_csv_path =
        argc >= 2
            ? argv[1]
            : DEFAULT_TRAINING_CSV_PATH;
    const std::string test_csv_value =
        argc == 3 ? argv[2] : "";
    const std::string* test_csv_path =
        argc == 3 ? &test_csv_value : nullptr;

    try
    {
        return run(
            training_csv_path,
            test_csv_path);
    }
    catch (const std::exception& error)
    {
        std::cerr
            << "データ読込、モデル学習、または予測に失敗しました: "
            << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
