参考 [https://github.com/billmuch/stephendiehl-mlir-introduction-examples]()，有所修改。

现通过两种不同的方式实现了 GEMM（通用矩阵乘法）：

1. `gemm.mlir`：使用 `memref` 实现
2. `tensor_gemm.mlir`：使用 `tensor` 实现

可以在 `run_gemm.py` 中测试。