# Makefile for MLIR GEMM example

# Default target
all: libgemm.so libtensor_gemm.so

# Optimize MLIR
gemm_lowered.mlir: gemm.mlir
	mlir-opt gemm.mlir \
	  --affine-loop-normalize \
	  --affine-parallelize \
	  --lower-affine \
	  --convert-scf-to-cf \
	  --convert-cf-to-llvm \
	  --convert-math-to-llvm \
	  --convert-arith-to-llvm \
	  --finalize-memref-to-llvm \
	  --convert-func-to-llvm \
	  --reconcile-unrealized-casts \
	  -o gemm_lowered.mlir

# Translate to LLVM IR
gemm.ll: gemm_lowered.mlir
	mlir-translate gemm_lowered.mlir -mlir-to-llvmir -o gemm.ll

# Compile to object file
gemm.o: gemm.ll
	llc -filetype=obj --relocation-model=pic gemm.ll -o gemm.o

# Compile to shared library (Linux)
libgemm.so: gemm.o
	clang -shared -L/usr/lib/llvm-18/lib -fopenmp -o libgemm.so gemm.o

# New build pipeline for tensor_gemm.mlir
tensor_gemm_lowered.mlir: tensor_gemm.mlir
	mlir-opt tensor_gemm.mlir \
	  --one-shot-bufferize="bufferize-function-boundaries" \
	  --convert-linalg-to-parallel-loops \
	  --convert-scf-to-openmp="num-threads=12" \
	  --convert-openmp-to-llvm \
	  --convert-scf-to-cf \
	  --convert-cf-to-llvm \
	  --convert-math-to-llvm \
	  --convert-arith-to-llvm \
	  --finalize-memref-to-llvm \
	  --convert-func-to-llvm \
	  --reconcile-unrealized-casts \
	  -o tensor_gemm_lowered.mlir

# Translate to LLVM IR
tensor_gemm.ll: tensor_gemm_lowered.mlir
	mlir-translate tensor_gemm_lowered.mlir --mlir-to-llvmir -o tensor_gemm.ll

# Compile to object file
tensor_gemm.o: tensor_gemm.ll
	llc -filetype=obj --relocation-model=pic tensor_gemm.ll -o tensor_gemm.o

# Compile to shared library (Linux)
libtensor_gemm.so: tensor_gemm.o
	clang -shared -L/usr/lib/llvm-18/lib -fopenmp -o libtensor_gemm.so tensor_gemm.o

# Clean up generated files
clean:
	rm -f gemm_lowered.mlir gemm.ll gemm.o libgemm.so tensor_gemm_lowered.mlir tensor_gemm.ll tensor_gemm.o libtensor_gemm.so

.PHONY: all clean