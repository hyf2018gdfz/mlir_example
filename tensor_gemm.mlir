module {
  func.func @tensor_gemm(%A: tensor<4x6xf32>,
                         %B: tensor<6x8xf32>) -> tensor<4x8xf32> 
        attributes {llvm.emit_c_interface} {
    %zero = arith.constant 0.0 : f32
    %init = tensor.empty() : tensor<4x8xf32>
    %filled = linalg.fill ins(%zero : f32) outs(%init : tensor<4x8xf32>) -> tensor<4x8xf32>
    
    %result = linalg.matmul ins(%A, %B : tensor<4x6xf32>, tensor<6x8xf32>)
                           outs(%filled : tensor<4x8xf32>) -> tensor<4x8xf32>
    return %result : tensor<4x8xf32>
  }
}