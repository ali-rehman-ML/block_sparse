#include "matmul.h"
#include <stdexcept>
#include <omp.h>
#include <fstream>
#include <iostream>

void matmul_blocked(const Tensor& A, Tensor& B, float* C_data) {


    int M = A.getRows();
    int N = A.getCols();
    int K = B.getCols();

    float* A_data = A.getData();
    float* B_data = B.getData();
    // float* C_data = C.getData();
    int* indices = B.getRelativeIndices();
    int* sizes = B.getColumnCounts();
    int* start_idx = B.getStartIndices();

    int num_threads = omp_get_max_threads();


    size_t loop_count = 0;

    #pragma omp parallel for num_threads(num_threads) schedule(guided) shared(A_data, B_data, C_data, M, N, K, indices, sizes, start_idx) private(loop_count)
    for (int i = 0; i < M; i += 8) {
        for (int j = 0; j < K; j += 4) {
            int c_idx = i * K + j;
            int a_idx = i * N;
            int b_idx = j * N;

            loop_count = static_cast<size_t>(sizes[j/4])-1;
            int st_idx = start_idx[j/4];




            matmul_4x4(A_data + a_idx, B_data + b_idx, indices + st_idx, 
                       C_data + c_idx, N * sizeof(float), K * sizeof(float), 
                       loop_count, 0, a_idx, b_idx);



        }
    }
}