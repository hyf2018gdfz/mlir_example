import ctypes
import numpy as np
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


M, N, K = 4, 8, 6


def call_inplace_gemm(
    lib_path: str, func_name: str, A: np.ndarray, B: np.ndarray, C: np.ndarray
) -> np.ndarray:
    """Call an MLIR-generated in-place GEMM that writes into C (3-arg function)."""
    module = ctypes.CDLL(lib_path)
    func = getattr(module, func_name)
    A_m = numpy_to_memref2d(A)
    B_m = numpy_to_memref2d(B)
    C_m = numpy_to_memref2d(C)
    func.argtypes = [POINTER(MemRef2DDescriptor)] * 3
    func(ctypes.byref(A_m), ctypes.byref(B_m), ctypes.byref(C_m))
    return C


def call_returning_gemm(
    lib_path: str, func_name: str, A: np.ndarray, B: np.ndarray
) -> np.ndarray:
    """Call an MLIR-generated GEMM that returns a MemRef descriptor (2-arg function -> ret desc)."""
    module = ctypes.CDLL(lib_path)
    func = getattr(module, func_name)
    func.argtypes = [POINTER(MemRef2DDescriptor), POINTER(MemRef2DDescriptor)]
    func.restype = MemRef2DDescriptor
    A_m = numpy_to_memref2d(A)
    B_m = numpy_to_memref2d(B)
    ret = func(ctypes.byref(A_m), ctypes.byref(B_m))
    return memref2d_to_numpy(ret)


def verify_results(C_mlir: np.ndarray, C_numpy: np.ndarray, tol: float = 1e-5) -> bool:
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
    C = np.zeros((M, N), dtype=np.float32)

    print("Running in-place gemm (libgemm.so)")
    C_mlir = call_inplace_gemm("./libgemm.so", "_mlir_ciface_gemm", A, B, C)
    C_numpy = A @ B
    return A, B, C_mlir, C_numpy


def run_anogemm():
    np.random.seed(45)
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)

    print("Running returning gemm (libtensor_gemm.so)")
    C_mlir = call_returning_gemm(
        "./libtensor_gemm.so", "_mlir_ciface_tensor_gemm", A, B
    )
    C_numpy = A @ B
    return A, B, C_mlir, C_numpy


if __name__ == "__main__":
    A, B, C_mlir, C_numpy = run_gemm()
    verify_results(C_mlir, C_numpy)
    A, B, C_mlir, C_numpy = run_anogemm()
    verify_results(C_mlir, C_numpy)
