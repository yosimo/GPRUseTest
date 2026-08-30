# ハイブリッド回帰モデル利用ガイド

## 概要

`HybridRegressionModel`は、入力全体の大域的な傾向を表す決定論的トレンドモデルと、そのトレンドで説明できない残差を学習するGaussian Process Regression（GPR）を組み合わせます。

$$
y(\boldsymbol{x})
=
f_{\mathrm{trend}}(\boldsymbol{x})
+b
+f_{\mathrm{GPR}}(\boldsymbol{x})
$$

現在利用できるトレンドモデルは`PolynomialTrendModel`です。設定により、1次重回帰または2次多項式回帰を選択できます。係数推定にはRidge回帰を使用できます。

## 全体の設定・実行フロー

ハイブリッド回帰を使用するときは、次の順序でデータと2種類のモデル設定を準備します。

| 順序 | 設定・処理 | 目的 |
|---:|---|---|
| 1 | 入力次元とCSV列を決める | 入力`x1...xD`と目的値`y`の対応を固定する |
| 2 | 学習データを`RegressionDataset`へ変換する | 行数、次元、有限値を検証する |
| 3 | `PolynomialTrendConfig`を設定する | 大域的傾向と外挿時の基本形を決める |
| 4 | `GaussianProcessConfig`を設定する | トレンドで説明できない局所残差の学習方法を決める |
| 5 | `HybridRegressionModel`を構築する | トレンドモデルと残差GPRを合成する |
| 6 | `fit()`を呼ぶ | トレンド、残差バイアス、残差GPRの順で学習する |
| 7 | `predict()`または`predictWithGradients()`を呼ぶ | 合成平均、分散、必要なら勾配を得る |
| 8 | 内挿・外挿を別々に評価する | トレンド次数やGPR設定を選択する |

コード上の大きな流れは次のとおりです。

```cpp
// 1、2: 入力と目的値を用意し、検証済みデータセットにする。
RegressionDataset dataset(training_inputs, training_targets);

// 3: 先に大域トレンドを設定する。
PolynomialTrendConfig trend_config;
trend_config.degree = PolynomialDegree::LINEAR;
trend_config.ridge_lambda = 1.0e-8;

// 4: 次に局所残差を学習するGPRを設定する。
GaussianProcessConfig residual_config;
residual_config.kernel = KernelType::MATERN_5_2;
residual_config.length_scales.values =
    Eigen::VectorXd::Ones(dataset.inputDimension());

// 5: 2つのモデルを合成する。
HybridRegressionModel model(
    std::make_unique<PolynomialTrendModel>(trend_config),
    residual_config);

// 6: 学習する。残差の計算はHybridRegressionModelが内部で行う。
model.fit(dataset);

// 7: 元の入力スケールで予測する。
const Prediction prediction = model.predict(test_inputs);
```

設定時は、まず1次トレンドを基準にし、その後で2次項、交互作用項、Ridge強度を検討します。残差GPRのカーネルやlength scaleはその後に調整します。先にGPRを細かく調整すると、本来トレンドが担当すべき大域変化までGPRが吸収し、役割分担を判断しにくくなるためです。

ARDを有効にしてlength scaleの初期値をベクトルで指定する場合、その要素数は入力次元`D`と一致させます。トレンドとGPRはそれぞれ入力変換状態を内部に保持するため、呼び出し側は常に元の入力値を渡します。

## fit内部の学習順序

`HybridRegressionModel::fit()`は内部で次の処理を行います。

1. `PolynomialTrendModel`を目的値で学習する
2. 学習点におけるトレンド予測を計算する
3. 目的値とトレンド予測の差を残差として計算する
4. 残差平均を`residualBias()`として分離する
5. 平均を除いた残差を`GaussianProcess`で学習する

残差を平均ゼロに調整するため、学習点から十分離れてGPRの補正が小さくなると、予測平均はトレンドモデルと残差バイアスへ近づきます。

## CSVを使用するmainサンプル

`main.cpp`は、CSVファイルから4次元の学習データを読み込んでハイブリッドモデルを学習します。CSVの各行は次の5列です。

```text
x1,x2,x3,x4,y
```

先頭行には同じ内容のヘッダを付けることができます。ヘッダを付けない場合は、1行目から数値データとして読み込みます。列の並びは固定で、各値は有限の数値である必要があります。

テスト用データは次のファイルです。

```text
data/hybrid_training_4d.csv
```

リポジトリのルートから引数なしで実行すると、このファイルを読み込みます。

```bash
./build/gpr_test
```

別のCSVを使用する場合は、ファイルパスを第1引数へ指定します。

```bash
./build/gpr_test path/to/training_data.csv
```

CMake構成時にテスト用CSVは`build/data/`にもコピーされるため、buildディレクトリから実行することもできます。

CSVの読み込みには`fast-cpp-csv-parser`を使用します。既定では次の場所から`csv.h`を検索します。

```text
$HOME/lib/fast-cpp-csv-parser
```

別の場所へ配置した場合は、CMake構成時に指定してください。

```bash
cmake -S . -B build \
  -DFAST_CPP_CSV_PARSER_ROOT=/path/to/fast-cpp-csv-parser
```

## 基本的な使い方

公開APIは次のヘッダから利用できます。

```cpp
#include <bayesian_optimization/surrogate/surrogate.hpp>
```

### 1. 学習データを用意する

入力は`N × D`の行列、目的値は長さ`N`のベクトルです。4次元入力の場合は次の形になります。

```cpp
Eigen::MatrixXd training_inputs(observation_count, 4);
Eigen::VectorXd training_targets(observation_count);

// training_inputsとtraining_targetsへ値を設定する。

RegressionDataset dataset(
    training_inputs,
    training_targets);
```

入力と目的値はすべて有限値である必要があります。

### 2. 多項式トレンドを設定する

1次重回帰を使用する例です。

```cpp
PolynomialTrendConfig trend_config;
trend_config.degree = PolynomialDegree::LINEAR;
trend_config.include_interactions = false;
trend_config.ridge_lambda = 1.0e-8;
trend_config.input_transform =
    InputTransformType::STANDARDIZE;
```

2次回帰へ変更する場合は次のように設定します。

```cpp
trend_config.degree = PolynomialDegree::QUADRATIC;
trend_config.include_interactions = true;
```

4次元で使用される特徴数は次のとおりです。

| 設定 | 特徴数 |
|---|---:|
| 1次 | 5 |
| 2次、交互作用なし | 9 |
| 2次、交互作用あり | 15 |

特徴数には切片を含みます。`ridge_lambda`は切片以外の係数へ適用され、切片は正則化されません。`ridge_lambda = 0`の場合、特徴行列がrank不足だと学習時に例外が発生します。

### 3. 残差GPRを設定する

```cpp
GaussianProcessConfig residual_config;
residual_config.kernel = KernelType::MATERN_5_2;
residual_config.use_ard = true;

residual_config.length_scales.mode =
    HyperparameterMode::OPTIMIZE;
residual_config.length_scales.values =
    Eigen::VectorXd::Ones(4);

residual_config.preprocessing.input_transform =
    InputTransformType::STANDARDIZE;
residual_config.preprocessing.output_transform =
    OutputTransformType::STANDARDIZE;
```

カーネルは`MATERN_5_2`または`RBF`を選択できます。その他のハイパーパラメータ、探索範囲、restart回数、jitterについては`GaussianProcessConfig`で設定します。

### 4. ハイブリッドモデルを構築して学習する

```cpp
HybridRegressionModel model(
    std::make_unique<PolynomialTrendModel>(trend_config),
    residual_config);

model.fit(dataset);
```

再学習時にはトレンドモデルを毎回学習し直します。`SurrogateFitPolicy::REUSE_MODEL_PARAMETERS`を指定した場合、その指定は残差GPRのハイパーパラメータへ適用されます。

## 予測

複数の入力を一括して予測できます。

```cpp
Eigen::MatrixXd test_inputs(test_count, 4);
// test_inputsへ予測点を設定する。

const Prediction prediction = model.predict(test_inputs);
```

返される値は次のとおりです。

- `mean`: トレンド、残差バイアス、GPR残差を合成した予測平均
- `latent_variance`: GPRが表す潜在残差関数の分散
- `observation_variance`: 潜在分散とGPR観測ノイズ分散の和

予測分散に含まれるのは残差GPRの不確かさだけです。多項式係数の推定誤差は含まれません。

各成分を個別に確認することもできます。

```cpp
const double trend =
    model.trendModel().predict(test_inputs)(0) +
    model.residualBias();

const double residual =
    model.residualModel().predict(test_inputs).mean(0);

const double hybrid = model.predict(test_inputs).mean(0);
```

## 勾配付き予測

```cpp
const PredictionWithGradients result =
    model.predictWithGradients(test_inputs);
```

`mean_gradient`は多項式トレンドとGPR残差の勾配の和です。

$$
\nabla \hat{y}
=
\nabla f_{\mathrm{trend}}
+
\nabla f_{\mathrm{GPR}}
$$

`latent_variance_gradient`は残差GPRの潜在分散勾配です。すべての勾配は元の入力スケールに対して返されます。

## 学習済み情報の取得

```cpp
const auto& trend = model.trendModel();
const auto& residual_gp = model.residualModel();
const double bias = model.residualBias();
const auto& data = model.trainingData();
```

トレンドモデルが`PolynomialTrendModel`であることが分かっている場合は、構築前に保持した参照、または適切な型確認を行ったうえで、次の情報を利用できます。

- `coefficients()`: 学習済み係数
- `featureCount()`: 特徴数
- `featureMatrixRank()`: 学習時の特徴行列rank
- `config()`: 多項式設定

## 外挿時の注意

定常カーネルを使用するGPR残差は、学習点から離れるほど補正の影響が小さくなり、予測平均は多項式トレンドへ近づきます。ただし、多項式トレンド自体の外挿が正しいことを保証するものではありません。

特に2次モデルは入力範囲外で急速に増減する場合があります。次のモデルを境界領域を除外した検証データで比較してください。

- 1次回帰のみ
- 2次回帰のみ
- GPRのみ
- 1次回帰と残差GPR
- 2次回帰と残差GPR

通常のランダム分割は主に内挿性能を評価します。外挿性能には、特定入力の上端または下端を学習データから除外する境界ホールドアウトが適しています。

## 現在の制約

- トレンドモデルは確定的として扱い、その係数推定誤差を予測分散へ含めません
- ハイブリッドモデルの保存と読み込みは未実装です
- 多出力回帰には対応していません
- 現在提供している具体的なトレンド実装は`PolynomialTrendModel`です

4次元の実行可能なサンプルは`main.cpp`および`examples/hybrid_regression_example.cpp`を参照してください。
