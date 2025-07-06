#ifndef MATMUL_H
#define MATMUL_H

#include "tensor.h"

extern "C" void matmul_4x4(float* a, float* b, int* ind, float* c, size_t K, size_t i, size_t N, size_t k, size_t a_idx, size_t b_idx);

void matmul_blocked(const Tensor& A, Tensor& B, float* C_data);

#endif // MATMUL_H