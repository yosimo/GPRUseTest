# サロゲートモジュール

`surrogate/` はベイズ最適化ループから独立して利用できる GPR 実装です。

コピー利用時は`include/`と`src/`をコピーし、C++17、Eigen 3.3以降、
LBFGS++、cereal を利用可能にしてください。

公開APIは
`#include <bayesian_optimization/surrogate/surrogate.hpp>` から利用できます。
具体的なコピー方法、外部プロジェクト用CMake、学習・予測・永続化の例は
[サロゲートのコピー利用ガイド](../docs/surrogate_copy_usage.md)を参照して
ください。
