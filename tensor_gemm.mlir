module {
  func.func @tensor_gemm(%A: tensor<128x256xf32>,
                         %B: tensor<256x512xf32>,
                         %output: tensor<128x512xf32>) -> ()
        attributes {llvm.emit_c_interface} {    
    %result = linalg.matmul ins(%A, %B : tensor<128x256xf32>, tensor<256x512xf32>)
                           outs(%output : tensor<128x512xf32>) -> tensor<128x512xf32>
    return
  }
}