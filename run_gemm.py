import ctypes
import numpy as np
import time
from ctypes import (
    c_void_p,
    c_longlong,
    Structure,
    POINTER,
    cast,
    c_float,
)


class MemRef2DDescriptor(Structure):
    """Structure matching MLIR's 2D MemRef descriptor"""

    _fields_ = [
        ("allocated", c_void_p),
        ("aligned", c_void_p),
        ("offset", c_longlong),
        ("shape", c_longlong * 2),
        ("stride", c_longlong * 2),
    ]


def numpy_to_memref2d(arr: np.ndarray) -> MemRef2DDescriptor:
    """Convert a 2D NumPy array to a MemRef descriptor"""
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)

    desc = MemRef2DDescriptor()
    desc.allocated = arr.ctypes.data_as(c_void_p)
    desc.aligned = desc.allocated
    desc.offset = 0
    desc.shape[0] = arr.shape[0]
    desc.shape[1] = arr.shape[1]
    desc.stride[0] = arr.strides[0] // arr.itemsize
    desc.stride[1] = arr.strides[1] // arr.itemsize
    return desc


def memref2d_to_numpy(desc: MemRef2DDescriptor, dtype=np.float32) -> np.ndarray:
    """Convert a MemRef descriptor back to a NumPy array (assumes contiguous layout)."""
    shape = (int(desc.shape[0]), int(desc.shape[1]))
    total = shape[0] * shape[1]
    # Assume float32 data for simplicity (matches code generation)
    buffer_ptr = cast(desc.aligned, POINTER(c_float))
    flat = np.ctypeslib.as_array(buffer_ptr, shape=(total,))
    return flat.reshape(shape).copy()


M, N, K = 128, 512, 256


def call_gemm(
    lib_path: str, func_name: str, A: np.ndarray, B: np.ndarray, C: np.ndarray
) -> np.ndarray:
    module = ctypes.CDLL(lib_path)
    func = getattr(module, func_name)
    A_m = numpy_to_memref2d(A)
    B_m = numpy_to_memref2d(B)
    C_m = numpy_to_memref2d(C)
    func.argtypes = [POINTER(MemRef2DDescriptor)] * 3
    t0 = time.perf_counter()
    func(ctypes.byref(A_m), ctypes.byref(B_m), ctypes.byref(C_m))
    t1 = time.perf_counter()
    elapsed = t1 - t0
    return C, elapsed


def verify_results(C_mlir: np.ndarray, C_numpy: np.ndarray, tol: float = 1e-4) -> bool:
    if C_mlir.shape != C_numpy.shape:
        print(f"Shape mismatch: MLIR {C_mlir.shape} vs NumPy {C_numpy.shape}")
        return False
    abs_diff = np.abs(C_mlir - C_numpy)
    max_diff = abs_diff.max()
    mean_diff = abs_diff.mean()
    print(f"max diff={max_diff:.3e}, mean diff={mean_diff:.3e}")
    if np.allclose(C_mlir, C_numpy, atol=tol):
        print("✅ Results match within tolerance")
        return True
    else:
        print("❌ Results differ")
        return False


def run_gemm():
    np.random.seed(42)
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)

    mlir_times = []
    numpy_times = []
    C_mlir = None
    C_numpy = None

    print("Running memref gemm (libgemm.so) 20 iterations")
    for _ in range(20):
        C = np.zeros((M, N), dtype=np.float32)
        C_mlir, t_mlir = call_gemm("./libgemm.so", "_mlir_ciface_gemm", A, B, C)
        mlir_times.append(t_mlir)
    time.sleep(0.5)

    for _ in range(20):
        t0 = time.perf_counter()
        C_numpy = A @ B
        t1 = time.perf_counter()
        numpy_times.append(t1 - t0)
    time.sleep(0.5)

    avg_mlir = sum(mlir_times) / len(mlir_times)
    avg_numpy = sum(numpy_times) / len(numpy_times)
    print(f"Avg MLIR memref gemm time over 20 runs: {avg_mlir*1000:.3f} ms")
    print(f"Avg NumPy gemm time over 20 runs: {avg_numpy*1000:.3f} ms")

    return A, B, C_mlir, C_numpy


def run_anogemm():
    np.random.seed(42)
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)

    mlir_times = []
    numpy_times = []
    C_mlir = None
    C_numpy = None

    print("Running tensor gemm (libtensor_gemm.so) 20 iterations")
    for _ in range(20):
        C = np.zeros((M, N), dtype=np.float32)
        C_mlir, t_mlir = call_gemm(
            "./libtensor_gemm.so", "_mlir_ciface_tensor_gemm", A, B, C
        )
        mlir_times.append(t_mlir)
    time.sleep(0.5)

    for _ in range(20):
        t0 = time.perf_counter()
        C_numpy = A @ B
        t1 = time.perf_counter()
        numpy_times.append(t1 - t0)
    time.sleep(0.5)

    avg_mlir = sum(mlir_times) / len(mlir_times)
    avg_numpy = sum(numpy_times) / len(numpy_times)
    print(f"Avg MLIR tensor gemm time over 20 runs: {avg_mlir*1000:.3f} ms")
    print(f"Avg NumPy gemm time over 20 runs: {avg_numpy*1000:.3f} ms")

    return A, B, C_mlir, C_numpy


if __name__ == "__main__":
    A, B, C_mlir, C_numpy = run_gemm()
    verify_results(C_mlir, C_numpy)
    A, B, C_mlir, C_numpy = run_anogemm()
    verify_results(C_mlir, C_numpy)
