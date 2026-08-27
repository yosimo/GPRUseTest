// SPDX-License-Identifier: MIT
#pragma once

#include <bayesian_optimization/surrogate/config.hpp>
#include <bayesian_optimization/surrogate/surrogate_model.hpp>

#include <filesystem>
#include <iosfwd>
#include <memory>

namespace bayesian_optimization::surrogate
{

class GaussianProcess;

void saveGaussianProcess(
    const GaussianProcess& model,
    const std::filesystem::path& path);
void saveGaussianProcess(const GaussianProcess& model, std::ostream& output);
[[nodiscard]] GaussianProcess loadGaussianProcess(
    const std::filesystem::path& path);
[[nodiscard]] GaussianProcess loadGaussianProcess(std::istream& input);

/** @brief Exact Gaussian-process regression with homoscedastic noise. */
class GaussianProcess final : public DifferentiableSurrogateModel
{
public:
    explicit GaussianProcess(GaussianProcessConfig config = {});
    ~GaussianProcess();

    GaussianProcess(const GaussianProcess&) = delete;
    GaussianProcess& operator=(const GaussianProcess&) = delete;
    GaussianProcess(GaussianProcess&&) noexcept;
    GaussianProcess& operator=(GaussianProcess&&) noexcept;

    void fit(
        const RegressionDataset& dataset,
        SurrogateFitPolicy policy = SurrogateFitPolicy::CONFIGURED) override;
    [[nodiscard]] bool isFitted() const noexcept override;
    [[nodiscard]] Eigen::Index inputDimension() const noexcept override;
    [[nodiscard]] Prediction predict(
        Eigen::Ref<const Eigen::MatrixXd> test_inputs) const override;
    [[nodiscard]] PredictionWithGradients predictWithGradients(
        Eigen::Ref<const Eigen::MatrixXd> test_inputs) const override;
    [[nodiscard]] OutputTransform fittedOutputTransform() const override;
    [[nodiscard]] const GaussianProcessConfig& config() const noexcept;
    [[nodiscard]] const RegressionDataset& trainingData() const;
    [[nodiscard]] FittedHyperparameters fittedHyperparameters() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;

    friend void saveGaussianProcess(
        const GaussianProcess&,
        const std::filesystem::path&);
    friend void saveGaussianProcess(const GaussianProcess&, std::ostream&);
    friend GaussianProcess loadGaussianProcess(const std::filesystem::path&);
    friend GaussianProcess loadGaussianProcess(std::istream&);
};

}  // namespace bayesian_optimization::surrogate

