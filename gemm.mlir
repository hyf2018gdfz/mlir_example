module {
  func.func @gemm(%A: memref<4x6xf32>, %B: memref<6x8xf32>, %C: memref<4x8xf32>) 
        attributes {llvm.emit_c_interface} {
    // Loop over the output matrix dimensions (4x8)
    affine.for %i = 0 to 4 {
      affine.for %j = 0 to 8 {
        // Initialize accumulator to zero
        %zero = arith.constant 0.0 : f32
        %acc = affine.for %k = 0 to 6 iter_args(%acc = %zero) -> (f32) {
          // Load A[i, k]
          %a_val = affine.load %A[%i, %k] : memref<4x6xf32>
          
          // Load B[k, j]  
          %b_val = affine.load %B[%k, %j] : memref<6x8xf32>
          
          // Multiply A[i,k] * B[k,j]
          %prod = arith.mulf %a_val, %b_val : f32
          
          // Add to accumulator
          %new_acc = arith.addf %acc, %prod : f32
          affine.yield %new_acc : f32
        }
        
        // Store the result in C[i, j]
        affine.store %acc, %C[%i, %j] : memref<4x8xf32>
      }
    }
    return
  }
}
