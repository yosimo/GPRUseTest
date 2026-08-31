# サロゲートモジュール

`surrogate/` はベイズ最適化ループから独立して利用できる GPR および
ハイブリッド回帰の実装です。

コピー利用時は`include/`と`src/`をコピーし、C++17、Eigen 3.3以降、
LBFGS++、cereal を利用可能にしてください。

公開APIは
`#include <bayesian_optimization/surrogate/surrogate.hpp>` から利用できます。
具体的なコピー方法、外部プロジェクト用CMake、学習・予測・永続化の例は
[サロゲートのコピー利用ガイド](../docs/surrogate_copy_usage.md)を参照して
ください。

## ハイブリッド回帰

ハイブリッド回帰では、多項式回帰で入力全体の大域的な傾向を学習し、
その残差をGPRで学習します。

```text
予測平均 = 多項式トレンド + 残差バイアス + GPR残差
```

`PolynomialTrendModel`は1次・2次のRidge回帰に対応しています。
`HybridRegressionModel`へトレンドモデルとGPR設定を渡して使用します。

詳しい設定、学習、予測、勾配、外挿時の注意点については、
[ハイブリッド回帰モデル利用ガイド](../docs/hybrid_regression_usage.md)を
参照してください。

4次元の実行例は
[main.cpp](../main.cpp)および
[hybrid_regression_example.cpp](../examples/hybrid_regression_example.cpp)
にあります。`main.cpp`は
[学習用CSV](../data/hybrid_training_4d.csv)を読み込み、大域トレンド単体、
GPR単体、ハイブリッドを比較します。
[テスト入力CSV](../data/hybrid_test_4d.csv)を第2引数に指定すると、
全テスト点の3モデル予測を一括出力できます。

ハイブリッドモデルの保存・読み込みは現在未実装です。
