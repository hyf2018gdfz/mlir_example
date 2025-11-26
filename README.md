参考 [https://github.com/billmuch/stephendiehl-mlir-introduction-examples]()，有所修改。

现通过两种不同的方式实现了 GEMM（通用矩阵乘法）：

1. `gemm.mlir`：使用 `memref` 实现
2. `tensor_gemm.mlir`：使用 `tensor` 实现

测试前需要 `make all`，然后运行 `python run_gemm.py` 进行测试。有关 MLIR 代码的详细编译参数可在 MLIR 中查看。

MLIR 的 commit-id：`87f0227cb60147a26a1eeb4fb06e3b505e9c7261`，版本为 20.1.8。
