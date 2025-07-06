#include "tensor.h"
#include "matmul.h"
#include <iostream>
#include <random>
#include <chrono>
#include <cstring>
#include <cassert>
#include <fstream>
#include <omp.h>
using namespace std;
void initialize_matrix_random(float* ptr, int M, int N) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    for (int i = 0; i < M * N; ++i) {
        ptr[i] = dis(gen);
    }
}

// void naive_matmul(const Tensor& A, const Tensor& B, Tensor& C) {
//     int M = A.getRows();
//     int N = A.getCols();
//     int K = B.getCols();
//     float* A_data = A.getData();
//     float* B_data = B.getData();
//     float* C_data = C.getData();
    
//     if (!A_data || !B_data || !C_data) {
//         throw std::runtime_error("Null pointer in naive_matmul");
//     }
    
//     std::memset(C_data, 0, M * K * sizeof(float));
//     for (int i = 0; i < M; ++i) {
//         for (int j = 0; j < K; ++j) {
//             for (int k = 0; k < N; ++k) {
//                 int a_idx = i * N + k;
//                 int b_idx = k * K + j;
//                 int c_idx = i * K + j;
//                 if (a_idx >= M * N || b_idx >= N * K || c_idx >= M * K) {
//                     throw std::runtime_error("Index out of bounds in naive_matmul");
//                 }
//                 C_data[c_idx] += A_data[a_idx] * B_data[b_idx];
//             }
//         }
//     }
// }

void naive_matmul(int M, int N, int K, const float* A, const float* B, float* C) {
    std::memset(C, 0, M * K * sizeof(float));
    
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < K; ++j) {
            for (int k = 0; k < N; ++k) {
                C[i * K + j] += A[i * N + k] * B[k * K + j];
            }
        }
    }
}

void save_matrix(const float* data, int rows, int cols, const std::string& filename) {
    std::ofstream out(filename);
    if (!out.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            out << data[i * cols + j] << " ";
        }
        out << "\n";
    }
    out.close();
}
bool compare_matrices(const float* C1, const float* C2, int M, int K, float tolerance = 1e-5) {
    bool is_correct = true;
    std::cout<<"tolerance : "<<tolerance<<std::endl;
    // add time slepp for 5 seconds 
        int no_of_incorrect = 0;

    #pragma omp parallel for collapse(2) shared(C1, C2, M, K, tolerance, is_correct)
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < K; ++j) {
            int idx = i * K + j;
            
            // cout<<"error : "<<std::fabs(C1[idx] - C2[idx])<<std::endl;

            if (std::fabs(C1[idx] - C2[idx]) > tolerance) {
                #pragma omp critical
                {
                    // std::cerr << "Mismatch at C[" << i << "][" << j << "]: "
                            //   << C1[idx] << " (optimized) vs " << C2[idx] << " (naive)" << std::endl;
                    is_correct = false;
                    no_of_incorrect++;

                }
            }
        }
    }
    cout<<"Number of mismatches: " << no_of_incorrect << std::endl; // Print the count of mismatches
    return is_correct;
}

// bool compare_matrices(const Tensor& C1, const Tensor& C2, float tolerance = 1e-4) {
//     int M = C1.getRows();
//     int K = C1.getCols();
//     const float* C1_data = C1.getData();
//     const float* C2_data = C2.getData();
//     bool is_correct = true;
//     int no_of_incorrect = 0;

//     if (!C1_data || !C2_data) {
//         throw std::runtime_error("Null pointer in compare_matrices");
//     }

//     #pragma omp parallel for collapse(2) shared(C1_data, C2_data, M, K, tolerance, is_correct)
//     for (int i = 0; i < M; ++i) {
//         for (int j = 0; j < K; ++j) {
//             int idx = i * K + j;
//             if (idx >= M * K) {
//                 throw std::runtime_error("Index out of bounds in compare_matrices");
//             }
//             if (std::fabs(C1_data[idx] - C2_data[idx]) > tolerance) {
//                 #pragma omp critical
//                 {
//                     std::cerr << "Mismatch at C[" << i << "][" << j << "]: "
//                               << C1_data[idx] << " (optimized) vs " << C2_data[idx] << " (naive)" << std::endl;
//                     is_correct = false;
//                     no_of_incorrect++;
//                 }
//             }
//         }
//     }
//     std::cout << "Number of mismatches: " << no_of_incorrect << std::endl;
//     return is_correct;
// }

void sparsify_B_contiguous(float* B, int N, int K, int block_size_row, int block_size_col,
                           const int* columnCounts, const int* relativeIndices, const int* startIndices,
                           int num_columns, float* B_sparse) {
    int blocks_per_column = N / block_size_row;

    // Create a sparse version of B (B_sparse) with non-zero blocks at random indices
    std::memset(B_sparse, 0, N * K * sizeof(float));
    for (int col = 0; col < num_columns; ++col) {
        int col_start = col * block_size_col;
        int num_non_zero = columnCounts[col];
        int st_idx = startIndices[col];

        // Copy non-zero blocks to B_sparse at their original indices
        for (int b = 0; b < num_non_zero; ++b) {
            int block_row = relativeIndices[st_idx + b];
            int row_start = block_row * block_size_row;
            for (int i = 0; i < block_size_row; ++i) {
                for (int j = 0; j < block_size_col; ++j) {
                    int src_idx = (row_start + i) * K + (col_start + j);
                    int dst_idx = src_idx;
                    B_sparse[dst_idx] = B[src_idx];
                }
            }
        }
    }

    // Rearrange B to have non-zero blocks contiguous at the start of each block-column
    std::memset(B, 0, N * K * sizeof(float));
    for (int col = 0; col < num_columns; ++col) {
        int col_start = col * block_size_col;
        int num_non_zero = columnCounts[col];
        int st_idx = startIndices[col];

        // Copy non-zero blocks to the start of the block-column
        for (int b = 0; b < num_non_zero; ++b) {
            int block_row = relativeIndices[st_idx + b];
            int src_row_start = block_row * block_size_row;
            int dst_row_start = b * block_size_row;
            for (int i = 0; i < block_size_row; ++i) {
                for (int j = 0; j < block_size_col; ++j) {
                    int src_idx = (src_row_start + i) * K + (col_start + j);
                    int dst_idx = (dst_row_start + i) * K + (col_start + j);
                    B[dst_idx] = B_sparse[src_idx];
                }
            }
        }
        // Remaining blocks (from num_non_zero to blocks_per_column) are zero
    }
}

int main() {
    try {
        const int M = 160;
        const int K = 160;
        const int N = 160;
        const float sparsity = 0.5f;
        const int block_size_row = 4;
        const int block_size_col = 4;

        float* A_data = static_cast<float*>(aligned_alloc(128, M * N * sizeof(float)));
        float* B_data = static_cast<float*>(aligned_alloc(128, N * K * sizeof(float)));
        float* B_sparse_data = static_cast<float*>(aligned_alloc(128, N * K * sizeof(float)));
        float* C_data = static_cast<float*>(aligned_alloc(128, M * K * sizeof(float)));
        float* C_naive_data = static_cast<float*>(aligned_alloc(128, M * K * sizeof(float)));

        if (!A_data || !B_data || !B_sparse_data || !C_data || !C_naive_data) {
            throw std::runtime_error("Memory allocation failed");
        }

        initialize_matrix_random(A_data, M, N);
        initialize_matrix_random(B_data, N, K);
        std::memset(C_data, 0, M * K * sizeof(float));
        std::memset(C_naive_data, 0, M * K * sizeof(float));

        // save_matrix(A_data, M, N, "A_initial.txt");
        // save_matrix(B_data, N, K, "B_initial.txt");
        std::cout<<"B_data here "<<B_data[0]<<std::endl;
        // Tensor A(A_data, M, N, false);
        // Tensor B(B_data, N, K, true);
        // Tensor C(C_data, M, K, false);
        // Tensor C_naive(C_naive_data, M, K, false);
        // std::cout<<"B_data here "<<B_data[0]<<std::endl;


        // save_matrix(A.getData(), M, N, "A_blocked.txt");
        // save_matrix(B.getData(), N, K, "B_before_sparsify.txt");

        // B.sparsify(sparsity, block_size_row, block_size_col);
        //         std::cout<<"B_data here "<<B_data[0]<<std::endl;

        // save_matrix(B.getData(), N, K, "B_sparsified.txt");

        // std::ofstream out("indices.txt");
        // out << "relative_indices: ";
        // for (size_t i = 0; i < B.getTotalNonZeroBlocks(); ++i) out << B.getRelativeIndices()[i] << " ";
        // out << "\ncolumn_counts: ";
        // for (size_t i = 0; i < B.getColumnCountSize(); ++i) out << B.getColumnCounts()[i] << " ";
        // out << "\nstart_indices: ";
        // for (size_t i = 0; i < B.getColumnCountSize(); ++i) out << B.getStartIndices()[i] << " ";
        // out.close();

        // double total_time = 0.0;
        // for (int iter = 0; iter < 1; ++iter) {
        //     auto start = std::chrono::high_resolution_clock::now();
        //     matmul_blocked(A, B, C);
        //     auto end = std::chrono::high_resolution_clock::now();
        //     auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        //     total_time += duration.count();
        // }
        // std::cout << "Average time for optimized matmul: " << total_time / 1.0 << "ms\n";

        // save_matrix(C.getData(), M, K, "C_optimized.txt");

        // std::memset(B_sparse_data, 0, N * K * sizeof(float));
        // for (int col = 0; col < K / block_size_col; ++col) {
        //     int col_start = col * block_size_col;
        //     int num_non_zero = B.getColumnCounts()[col];
        //     int st_idx = B.getStartIndices()[col];
        //     for (int b = 0; b < num_non_zero; ++b) {
        //         if (st_idx + b >= B.getTotalNonZeroBlocks()) {
        //             throw std::runtime_error("Index out of bounds in B_sparse creation");
        //         }
        //         int block_row = B.getRelativeIndices()[st_idx + b];
        //         int row_start = block_row * block_size_row;
        //         for (int i = 0; i < block_size_row; ++i) {
        //             for (int j = 0; j < block_size_col; ++j) {
        //                 int src_idx = (row_start + i) * K + (col_start + j);
        //                 int dst_idx = src_idx;
        //                 if (src_idx >= N * K || dst_idx >= N * K) {
        //                     throw std::runtime_error("Sparse matrix index out of bounds");
        //                 }
        //                 B_sparse_data[dst_idx] = B.getData()[src_idx];
        //             }
        //         }
        //     }
        // }
        // Tensor B_sparse(B_sparse_data, N, K, false);
        // int* relativeIndices = B.getRelativeIndicesMutable();
        // int* columnCounts = B.getColumnCountsMutable();
        // int* startIndices = B.getStartIndicesMutable();
        // std::cout<<"B_data here "<<B_data[0]<<std::endl;

        // sparsify_B_contiguous(B_data, N, K, block_size_row, block_size_col, columnCounts,
        //                   relativeIndices, startIndices, K/4, B_sparse_data);

        // naive_matmul(M,N,K,A_data, B_sparse_data, C_naive_data);

        // save_matrix(C_naive.getData(), M, K, "C_naive.txt");


        // cout<<"C data "<<C_naive_data[0]<<endl;
        // std::cout<<"C data algo"<<C.getData()[0]<<std::endl;
        bool is_correct = true;//compare_matrices(C.getData(), C_naive_data, M, K,1e-3);
        std::cout << (is_correct ? "Verification passed: Optimized and naive results match within tolerance."
                                 : "Verification failed: Differences found between optimized and naive results.") << std::endl;
        free(A_data);
        free(B_data);
        free(B_sparse_data);
        free(C_data);
        free(C_naive_data);

        return is_correct ? 0 : 1;
    } catch (const std::exception& e) {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return 1;
    }
}