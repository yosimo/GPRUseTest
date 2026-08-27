// SPDX-License-Identifier: MIT

#include <bayesian_optimization/surrogate/error.hpp>
#include <bayesian_optimization/surrogate/serialization.hpp>

#include <cereal/archives/portable_binary.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/vector.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <ios>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

namespace bayesian_optimization::surrogate
{
namespace
{

constexpr std::array<char, 8> MODEL_MAGIC{
    {'B', 'O', 'M', 'O', 'D', 'E', 'L', '\0'}};
constexpr std::uint32_t MODEL_SCHEMA_VERSION = 1;

struct MatrixRecord
{
    std::uint64_t rows{0};
    std::uint64_t columns{0};
    std::vector<double> values;

    template <class Archive>
    void serialize(Archive& archive)
    {
        archive(rows, columns, values);
    }
};

struct ModelRecord
{
    std::string producer_version{"0.1.0"};
    std::uint64_t input_dimension{0};

    std::int32_t kernel_type{0};
    bool use_ard{true};
    std::vector<double> length_scales;
    double signal_variance{0.0};
    double noise_variance{0.0};
    double effective_jitter{0.0};

    std::int32_t length_scale_mode{0};
    double length_scale_lower_bound{0.0};
    double length_scale_upper_bound{0.0};
    std::int32_t signal_variance_mode{0};
    double signal_variance_lower_bound{0.0};
    double signal_variance_upper_bound{0.0};
    std::int32_t noise_variance_mode{0};
    double noise_variance_lower_bound{0.0};
    double noise_variance_upper_bound{0.0};

    std::uint64_t restart_count{0};
    std::uint64_t max_iterations{0};
    double gradient_tolerance{0.0};
    std::uint64_t random_seed{0};
    double jitter_relative_initial{0.0};
    double jitter_multiplier{0.0};
    std::uint64_t jitter_max_attempts{0};

    std::int32_t input_transform_type{0};
    std::vector<double> input_offset;
    std::vector<double> input_scale;
    std::int32_t output_transform_type{0};
    double output_offset{0.0};
    double output_scale{1.0};

    MatrixRecord training_inputs;
    std::vector<double> training_targets;

    template <class Archive>
    void serialize(Archive& archive)
    {
        archive(
            producer_version,
            input_dimension,
            kernel_type,
            use_ard,
            length_scales,
            signal_variance,
            noise_variance,
            effective_jitter,
            length_scale_mode,
            length_scale_lower_bound,
            length_scale_upper_bound,
            signal_variance_mode,
            signal_variance_lower_bound,
            signal_variance_upper_bound,
            noise_variance_mode,
            noise_variance_lower_bound,
            noise_variance_upper_bound,
            restart_count,
            max_iterations,
            gradient_tolerance,
            random_seed,
            jitter_relative_initial,
            jitter_multiplier,
            jitter_max_attempts,
            input_transform_type,
            input_offset,
            input_scale,
            output_transform_type,
            output_offset,
            output_scale,
            training_inputs,
            training_targets);
    }
};

std::vector<double> toVector(const Eigen::VectorXd& values)
{
    return {values.data(), values.data() + values.size()};
}

Eigen::VectorXd toEigenVector(
    const std::vector<double>& values,
    const char* name)
{
    if (values.empty())
    {
        throw ModelFormatError(std::string(name) + " must not be empty");
    }
    Eigen::VectorXd result(static_cast<Eigen::Index>(values.size()));
    for (std::size_t index = 0; index < values.size(); ++index)
    {
        result(static_cast<Eigen::Index>(index)) = values[index];
    }
    if (!result.allFinite())
    {
        throw ModelFormatError(std::string(name) + " must be finite");
    }
    return result;
}

MatrixRecord toMatrixRecord(const Eigen::MatrixXd& values)
{
    MatrixRecord result;
    result.rows = static_cast<std::uint64_t>(values.rows());
    result.columns = static_cast<std::uint64_t>(values.cols());
    result.values.reserve(
        static_cast<std::size_t>(values.rows() * values.cols()));
    for (Eigen::Index row = 0; row < values.rows(); ++row)
    {
        for (Eigen::Index column = 0; column < values.cols(); ++column)
        {
            result.values.push_back(values(row, column));
        }
    }
    return result;
}

Eigen::MatrixXd toEigenMatrix(
    const MatrixRecord& record,
    const char* name)
{
    if (record.rows == 0 || record.columns == 0 ||
        record.rows >
            static_cast<std::uint64_t>(
                std::numeric_limits<Eigen::Index>::max()) ||
        record.columns >
            static_cast<std::uint64_t>(
                std::numeric_limits<Eigen::Index>::max()) ||
        record.values.size() != record.rows * record.columns)
    {
        throw ModelFormatError(std::string(name) + " has an invalid shape");
    }

    Eigen::MatrixXd result(
        static_cast<Eigen::Index>(record.rows),
        static_cast<Eigen::Index>(record.columns));
    std::size_t index = 0;
    for (Eigen::Index row = 0; row < result.rows(); ++row)
    {
        for (Eigen::Index column = 0; column < result.cols(); ++column)
        {
            result(row, column) = record.values[index++];
        }
    }
    if (!result.allFinite())
    {
        throw ModelFormatError(std::string(name) + " must be finite");
    }
    return result;
}

template <typename Enum>
std::int32_t enumValue(Enum value)
{
    return static_cast<std::int32_t>(value);
}

template <typename Enum>
Enum checkedEnum(
    std::int32_t value,
    std::int32_t maximum,
    const char* name)
{
    if (value < 0 || value > maximum)
    {
        throw ModelFormatError(std::string(name) + " is invalid");
    }
    return static_cast<Enum>(value);
}

void fittedInputTransform(
    const RegressionDataset& data,
    const PreprocessingConfig& config,
    Eigen::VectorXd& offset,
    Eigen::VectorXd& scale)
{
    offset = Eigen::VectorXd::Zero(data.inputDimension());
    scale = Eigen::VectorXd::Ones(data.inputDimension());
    if (config.input_transform == InputTransformType::MIN_MAX)
    {
        offset = *config.input_lower_bounds;
        scale = *config.input_upper_bounds - *config.input_lower_bounds;
    }
    else if (config.input_transform == InputTransformType::STANDARDIZE)
    {
        offset = data.inputs().colwise().mean();
        for (Eigen::Index column = 0; column < data.inputDimension(); ++column)
        {
            const double variance =
                (data.inputs().col(column).array() - offset(column))
                    .square()
                    .mean();
            const double candidate = std::sqrt(std::max(0.0, variance));
            scale(column) =
                candidate <= std::sqrt(std::numeric_limits<double>::epsilon()) *
                                     std::max(1.0, std::abs(offset(column)))
                    ? 1.0
                    : candidate;
        }
    }
}

ModelRecord makeRecord(const GaussianProcess& model)
{
    if (!model.isFitted())
    {
        throw std::logic_error("an unfitted GaussianProcess cannot be saved");
    }

    const GaussianProcessConfig& config = model.config();
    const RegressionDataset& data = model.trainingData();
    const FittedHyperparameters parameters = model.fittedHyperparameters();
    const OutputTransform output = model.fittedOutputTransform();
    Eigen::VectorXd input_offset;
    Eigen::VectorXd input_scale;
    fittedInputTransform(data, config.preprocessing, input_offset, input_scale);

    ModelRecord record;
    record.input_dimension =
        static_cast<std::uint64_t>(data.inputDimension());
    record.kernel_type = enumValue(config.kernel);
    record.use_ard = config.use_ard;
    record.length_scales = toVector(parameters.length_scales);
    record.signal_variance = parameters.signal_variance;
    record.noise_variance = parameters.noise_variance;
    record.effective_jitter = parameters.effective_jitter;

    record.length_scale_mode = enumValue(config.length_scales.mode);
    record.length_scale_lower_bound = config.length_scales.lower_bound;
    record.length_scale_upper_bound = config.length_scales.upper_bound;
    record.signal_variance_mode = enumValue(config.signal_variance.mode);
    record.signal_variance_lower_bound = config.signal_variance.lower_bound;
    record.signal_variance_upper_bound = config.signal_variance.upper_bound;
    record.noise_variance_mode = enumValue(config.noise_variance.mode);
    record.noise_variance_lower_bound = config.noise_variance.lower_bound;
    record.noise_variance_upper_bound = config.noise_variance.upper_bound;

    record.restart_count =
        config.hyperparameter_optimization.restart_count;
    record.max_iterations =
        config.hyperparameter_optimization.max_iterations;
    record.gradient_tolerance =
        config.hyperparameter_optimization.gradient_tolerance;
    record.random_seed =
        config.hyperparameter_optimization.random_seed;
    record.jitter_relative_initial = config.jitter.initial_relative_value;
    record.jitter_multiplier = config.jitter.multiplier;
    record.jitter_max_attempts = config.jitter.max_attempts;

    record.input_transform_type =
        enumValue(config.preprocessing.input_transform);
    record.input_offset = toVector(input_offset);
    record.input_scale = toVector(input_scale);
    record.output_transform_type =
        enumValue(config.preprocessing.output_transform);
    record.output_offset = output.offset;
    record.output_scale = output.scale;
    record.training_inputs = toMatrixRecord(data.inputs());
    record.training_targets = toVector(data.targets());
    return record;
}

GaussianProcess makeModel(const ModelRecord& record)
{
    if (record.input_dimension == 0 ||
        record.input_dimension >
            static_cast<std::uint64_t>(
                std::numeric_limits<Eigen::Index>::max()))
    {
        throw ModelFormatError("input dimension is invalid");
    }
    const Eigen::Index dimension =
        static_cast<Eigen::Index>(record.input_dimension);
    const Eigen::MatrixXd inputs =
        toEigenMatrix(record.training_inputs, "training inputs");
    const Eigen::VectorXd targets =
        toEigenVector(record.training_targets, "training targets");
    if (inputs.cols() != dimension || inputs.rows() != targets.size())
    {
        throw ModelFormatError("training data dimensions are inconsistent");
    }

    GaussianProcessConfig config;
    config.kernel =
        checkedEnum<KernelType>(record.kernel_type, 1, "kernel type");
    config.use_ard = record.use_ard;
    config.length_scales.values =
        toEigenVector(record.length_scales, "length scales");
    config.signal_variance.value = record.signal_variance;
    config.noise_variance.value = record.noise_variance;

    config.length_scales.mode = checkedEnum<HyperparameterMode>(
        record.length_scale_mode,
        1,
        "length scale mode");
    config.length_scales.lower_bound =
        record.length_scale_lower_bound;
    config.length_scales.upper_bound =
        record.length_scale_upper_bound;
    config.signal_variance.mode = checkedEnum<HyperparameterMode>(
        record.signal_variance_mode,
        1,
        "signal variance mode");
    config.signal_variance.lower_bound =
        record.signal_variance_lower_bound;
    config.signal_variance.upper_bound =
        record.signal_variance_upper_bound;
    config.noise_variance.mode = checkedEnum<HyperparameterMode>(
        record.noise_variance_mode,
        1,
        "noise variance mode");
    config.noise_variance.lower_bound =
        record.noise_variance_lower_bound;
    config.noise_variance.upper_bound =
        record.noise_variance_upper_bound;

    config.hyperparameter_optimization.restart_count =
        static_cast<std::size_t>(record.restart_count);
    config.hyperparameter_optimization.max_iterations =
        static_cast<std::size_t>(record.max_iterations);
    config.hyperparameter_optimization.gradient_tolerance =
        record.gradient_tolerance;
    config.hyperparameter_optimization.random_seed =
        record.random_seed;
    config.jitter.initial_relative_value =
        record.jitter_relative_initial;
    config.jitter.multiplier = record.jitter_multiplier;
    config.jitter.max_attempts =
        static_cast<std::size_t>(record.jitter_max_attempts);

    config.preprocessing.input_transform =
        checkedEnum<InputTransformType>(
            record.input_transform_type,
            2,
            "input transform type");
    config.preprocessing.output_transform =
        checkedEnum<OutputTransformType>(
            record.output_transform_type,
            1,
            "output transform type");
    const Eigen::VectorXd stored_input_offset =
        toEigenVector(record.input_offset, "input offset");
    const Eigen::VectorXd stored_input_scale =
        toEigenVector(record.input_scale, "input scale");
    if (stored_input_offset.size() != dimension ||
        stored_input_scale.size() != dimension ||
        (stored_input_scale.array() <= 0.0).any())
    {
        throw ModelFormatError("input transform state is invalid");
    }
    if (config.preprocessing.input_transform == InputTransformType::MIN_MAX)
    {
        config.preprocessing.input_lower_bounds = stored_input_offset;
        config.preprocessing.input_upper_bounds =
            stored_input_offset + stored_input_scale;
    }

    config.validateForDimension(dimension);
    RegressionDataset data(inputs, targets);
    Eigen::VectorXd expected_offset;
    Eigen::VectorXd expected_scale;
    fittedInputTransform(
        data,
        config.preprocessing,
        expected_offset,
        expected_scale);
    if (!expected_offset.isApprox(stored_input_offset, 1.0e-12) ||
        !expected_scale.isApprox(stored_input_scale, 1.0e-12))
    {
        throw ModelFormatError(
            "stored input preprocessing state is inconsistent");
    }

    GaussianProcess result(config);
    result.fit(data, SurrogateFitPolicy::REUSE_MODEL_PARAMETERS);
    const OutputTransform actual_output = result.fittedOutputTransform();
    if (!std::isfinite(record.output_offset) ||
        !std::isfinite(record.output_scale) ||
        record.output_scale <= 0.0 ||
        std::abs(actual_output.offset - record.output_offset) > 1.0e-12 ||
        std::abs(actual_output.scale - record.output_scale) > 1.0e-12)
    {
        throw ModelFormatError(
            "stored output preprocessing state is inconsistent");
    }
    if (!std::isfinite(record.effective_jitter) ||
        record.effective_jitter <= 0.0)
    {
        throw ModelFormatError("stored jitter is invalid");
    }
    const double actual_jitter =
        result.fittedHyperparameters().effective_jitter;
    if (std::abs(actual_jitter - record.effective_jitter) >
        1.0e-12 * std::max(1.0, record.effective_jitter))
    {
        throw ModelFormatError("stored jitter is inconsistent");
    }
    return result;
}

}  // namespace

void saveGaussianProcess(const GaussianProcess& model, std::ostream& output)
{
    output.write(MODEL_MAGIC.data(), MODEL_MAGIC.size());
    if (!output)
    {
        throw std::ios_base::failure("failed to write model magic");
    }
    try
    {
        cereal::PortableBinaryOutputArchive archive(output);
        const std::uint32_t schema_version = MODEL_SCHEMA_VERSION;
        const ModelRecord record = makeRecord(model);
        archive(schema_version, record);
    }
    catch (const cereal::Exception& error)
    {
        throw std::ios_base::failure(
            std::string("failed to serialize GaussianProcess: ") +
            error.what());
    }
    if (!output)
    {
        throw std::ios_base::failure("failed to write GaussianProcess model");
    }
}

void saveGaussianProcess(
    const GaussianProcess& model,
    const std::filesystem::path& path)
{
    const std::filesystem::path temporary = path.string() + ".tmp";
    try
    {
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        if (!output)
        {
            throw std::ios_base::failure("failed to open temporary model file");
        }
        saveGaussianProcess(model, output);
        output.close();
        if (!output)
        {
            throw std::ios_base::failure("failed to close temporary model file");
        }
        std::error_code error;
        std::filesystem::rename(temporary, path, error);
        if (error)
        {
            throw std::ios_base::failure(
                "failed to replace model file: " + error.message());
        }
    }
    catch (...)
    {
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
        throw;
    }
}

GaussianProcess loadGaussianProcess(std::istream& input)
{
    try
    {
        std::array<char, MODEL_MAGIC.size()> magic{};
        input.read(magic.data(), magic.size());
        if (!input || magic != MODEL_MAGIC)
        {
            throw ModelFormatError("invalid GaussianProcess model magic");
        }

        cereal::PortableBinaryInputArchive archive(input);
        std::uint32_t schema_version = 0;
        archive(schema_version);
        if (schema_version != MODEL_SCHEMA_VERSION)
        {
            throw ModelFormatError(
                "unsupported GaussianProcess model schema version");
        }
        ModelRecord record;
        archive(record);
        return makeModel(record);
    }
    catch (const ModelFormatError&)
    {
        throw;
    }
    catch (const cereal::Exception& error)
    {
        throw ModelFormatError(
            std::string("invalid GaussianProcess binary model: ") +
            error.what());
    }
    catch (const std::invalid_argument& error)
    {
        throw ModelFormatError(
            std::string("invalid GaussianProcess model: ") + error.what());
    }
}

GaussianProcess loadGaussianProcess(const std::filesystem::path& path)
{
    std::ifstream input(path, std::ios::binary);
    if (!input)
    {
        throw std::ios_base::failure("failed to open GaussianProcess model");
    }
    return loadGaussianProcess(input);
}

}  // namespace bayesian_optimization::surrogate
