#include "tensor.h"
#include <stdexcept>
#include <cstring>
#include <arm_neon.h>
#include <iostream>

Tensor::Tensor(float* data, int rows, int cols, bool is_sparse, int block_size_row, int block_size_col, 
               MatrixFormat current_format, MatrixFormat convert_format)
    : rows_(rows), cols_(cols), is_sparse_(is_sparse), total_non_zero_blocks_(0), 
      column_count_size_(0), relative_indices_(nullptr), column_counts_(nullptr), start_indices_(nullptr) {
    if (rows <= 0 || cols <= 0) {
        throw std::invalid_argument("Tensor dimensions must be positive");
    }
    if (block_size_row <= 0 || block_size_col <= 0) {
        throw std::invalid_argument("Block sizes must be positive");
    }
    if ((current_format == MatrixFormat::BLOCK_MAJOR || convert_format == MatrixFormat::COLUMN_BLOCK_MAJOR || 
         convert_format == MatrixFormat::ROW_BLOCK_MAJOR) && 
        (rows % block_size_row != 0 || cols % block_size_col != 0)) {
        throw std::invalid_argument("Tensor dimensions must be divisible by block sizes for block formats");
    }

    // Allocate aligned memory for data
    data_ = (float*)aligned_alloc(128, rows * cols * sizeof(float));
    if (!data_) {
        throw std::bad_alloc();
    }

    // Convert input data to row-major if necessary
    if (current_format == MatrixFormat::BLOCK_MAJOR) {
        // Assume input is in row-major block format
        float* temp = (float*)aligned_alloc(128, rows * cols * sizeof(float));
        if (!temp) {
            free(data_);
            throw std::bad_alloc();
        }
        std::memset(temp, 0, rows * cols * sizeof(float));

        int blocks_per_row = cols / block_size_col;
        int total_block_rows = rows / block_size_row;
        int block_size = block_size_row * block_size_col;

        for (int block_row_idx = 0; block_row_idx < total_block_rows; ++block_row_idx) {
            for (int block_col_idx = 0; block_col_idx < blocks_per_row; ++block_col_idx) {
                int input_idx = (block_row_idx * blocks_per_row + block_col_idx) * block_size;
                int row_start = block_row_idx * block_size_row;
                int col_start = block_col_idx * block_size_col;
                for (int i = 0; i < block_size_row; ++i) {
                    for (int j = 0; j < block_size_col; ++j) {
                        int row = row_start + i;
                        int col = col_start + j;
                        int output_idx = row * cols + col;
                        if (input_idx < rows * cols && output_idx < rows * cols) {
                            temp[output_idx] = data[input_idx++];
                        }
                    }
                }
            }
        }
        std::memcpy(data_, temp, rows * cols * sizeof(float));
        free(temp);
    } else {
        // Assume row-major
        std::memcpy(data_, data, rows * cols * sizeof(float));
    }

    if (is_sparse) {
        // Sparse matrix: generate sparsity metadata
        int blocks_per_column = rows / block_size_row;
        int num_columns = cols / block_size_col;
        column_count_size_ = num_columns;
        column_counts_ = new int[num_columns]();
        start_indices_ = new int[num_columns]();

        // Count non-zero blocks
        total_non_zero_blocks_ = 0;
        for (int col = 0; col < num_columns; ++col) {
            int col_start = col * block_size_col;
            for (int block_row = 0; block_row < blocks_per_column; ++block_row) {
                int row_start = block_row * block_size_row;
                bool is_non_zero = false;
                for (int i = 0; i < block_size_row && !is_non_zero; ++i) {
                    for (int j = 0; j < block_size_col; ++j) {
                        int idx = (row_start + i) * cols + (col_start + j);
                        if (idx < rows * cols && data_[idx] != 0.0f) {
                            is_non_zero = true;
                            break;
                        }
                    }
                }
                if (is_non_zero) {
                    column_counts_[col]++;
                    total_non_zero_blocks_++;
                }
            }
        }

        // Allocate relative_indices
        relative_indices_ = new int[total_non_zero_blocks_];

        // Populate start_indices and relative_indices
        size_t idx = 0;
        for (int col = 0; col < num_columns; ++col) {
            start_indices_[col] = idx;
            int col_start = col * block_size_col;
            for (int block_row = 0; block_row < blocks_per_column; ++block_row) {
                int row_start = block_row * block_size_row;
                bool is_non_zero = false;
                for (int i = 0; i < block_size_row && !is_non_zero; ++i) {
                    for (int j = 0; j < block_size_col; ++j) {
                        int idx = (row_start + i) * cols + (col_start + j);
                        if (idx < rows * cols && data_[idx] != 0.0f) {
                            is_non_zero = true;
                            break;
                        }
                    }
                }
                if (is_non_zero) {
                    relative_indices_[idx++] = block_row;
                }
            }
        }

        // Store non-zero blocks in column-block contiguous layout
        float* temp = (float*)aligned_alloc(128, rows * cols * sizeof(float));
        if (!temp) {
            free(data_);
            delete[] relative_indices_;
            delete[] column_counts_;
            delete[] start_indices_;
            throw std::bad_alloc();
        }
        std::memset(temp, 0, rows * cols * sizeof(float));

        idx = 0;
        for (int col = 0; col < num_columns; ++col) {
            int col_start = col * block_size_col;
            int num_non_zero = column_counts_[col];
            int st_idx = start_indices_[col];
            for (int b = 0; b < num_non_zero; ++b) {
                int block_row = relative_indices_[st_idx + b];
                int row_start = block_row * block_size_row;
                int dst_row_start = b * block_size_row;
                for (int i = 0; i < block_size_row; ++i) {
                    for (int j = 0; j < block_size_col; ++j) {
                        int src_idx = (row_start + i) * cols + (col_start + j);
                        int dst_idx = (dst_row_start + i) * cols + (col_start + j);
                        if (src_idx < rows * cols && dst_idx < rows * cols) {
                            temp[dst_idx] = data_[src_idx];
                        }
                    }
                }
            }
        }
        std::memcpy(data_, temp, rows * cols * sizeof(float));
        free(temp);
    }

    // Convert to desired format
    if (convert_format == MatrixFormat::COLUMN_BLOCK_MAJOR) {
        float* col_major = toColumnMajorBlock(block_size_row, block_size_col);
        if (!col_major) {
            free(data_);
            if (is_sparse) {
                delete[] relative_indices_;
                delete[] column_counts_;
                delete[] start_indices_;
            }
            throw std::runtime_error("Failed to allocate column-major block");
        }
        std::memcpy(data_, col_major, rows * cols * sizeof(float));
        free(col_major);
    } else if (convert_format == MatrixFormat::ROW_BLOCK_MAJOR) {
        float* row_major = toRowMajorBlock(block_size_row, block_size_col);
        if (!row_major) {
            free(data_);
            if (is_sparse) {
                delete[] relative_indices_;
                delete[] column_counts_;
                delete[] start_indices_;
            }
            throw std::runtime_error("Failed to allocate row-major block");
        }
        std::memcpy(data_, row_major, rows * cols * sizeof(float));
        free(row_major);
    }
    // If convert_format is ROW_MAJOR or BLOCK_MAJOR, data_ is already in row-major
}

Tensor::~Tensor() {
    free(data_);
    delete[] relative_indices_;
    delete[] column_counts_;
    delete[] start_indices_;
}

float* Tensor::toRowMajorBlock(int block_size_row, int block_size_col) {
    if (rows_ % block_size_row != 0 || cols_ % block_size_col != 0) {
        throw std::invalid_argument("Array dimensions must be divisible by block sizes");
    }

    float* output = (float*)aligned_alloc(128, rows_ * cols_ * sizeof(float));
    if (!output) {
        throw std::bad_alloc();
    }

    const int blocks_per_row = cols_ / block_size_col;
    const int total_block_rows = rows_ / block_size_row;
    const int block_size = block_size_row * block_size_col;
    #pragma omp parallel for schedule(guided) collapse(2)
    for (int block_row_idx = 0; block_row_idx < total_block_rows; ++block_row_idx) {
        for (int block_col_idx = 0; block_col_idx < blocks_per_row; ++block_col_idx) {
            int output_idx = (block_row_idx * blocks_per_row + block_col_idx) * block_size;
            int block_row = block_row_idx * block_size_row;
            int block_col = block_col_idx * block_size_col;

            for (int i = 0; i < block_size_row; ++i) {
                int row = block_row + i;
                int input_idx = row * cols_ + block_col;
                if (input_idx + block_size_col > rows_ * cols_) {
                    free(output);
                    throw std::runtime_error("Input index out of bounds in toRowMajorBlock");
                }
                int j = 0;

                for (; j <= block_size_col - 4; j += 4) {
                    if (output_idx + 3 >= rows_ * cols_) {
                        free(output);
                        throw std::runtime_error("Output index out of bounds in toRowMajorBlock");
                    }
                    float32x4_t vec = vld1q_f32(&data_[input_idx + j]);
                    vst1q_f32(&output[output_idx], vec);
                    output_idx += 4;
                }

                for (; j < block_size_col; ++j) {
                    if (output_idx >= rows_ * cols_) {
                        free(output);
                        throw std::runtime_error("Output index out of bounds in scalar loop");
                    }
                    output[output_idx++] = data_[input_idx + j];
                }
            }
        }
    }

    return output;
}

float* Tensor::toColumnMajorBlock(int block_size_row, int block_size_col) {
    if (rows_ % block_size_row != 0 || cols_ % block_size_col != 0) {
        throw std::invalid_argument("Array dimensions must be divisible by block sizes");
    }

    float* output = (float*)aligned_alloc(128, rows_ * cols_ * sizeof(float));
    if (!output) {
        throw std::bad_alloc();
    }

    const int blocks_per_col = rows_ / block_size_row;
    const int total_block_cols = cols_ / block_size_col;
    const int block_size = block_size_row * block_size_col;
    #pragma omp parallel for schedule(guided) collapse(2)
    for (int block_col_idx = 0; block_col_idx < total_block_cols; ++block_col_idx) {
        for (int block_row_idx = 0; block_row_idx < blocks_per_col; ++block_row_idx) {
            int output_idx = (block_col_idx * blocks_per_col + block_row_idx) * block_size;
            int block_row = block_row_idx * block_size_row;
            int block_col = block_col_idx * block_size_col;

            for (int i = 0; i < block_size_row; ++i) {
                int row = block_row + i;
                int input_idx = row * cols_ + block_col;
                if (input_idx + block_size_col > rows_ * cols_) {
                    free(output);
                    throw std::runtime_error("Input index out of bounds in toColumnMajorBlock");
                }
                int j = 0;

                for (; j <= block_size_col - 4; j += 4) {
                    if (output_idx + 3 >= rows_ * cols_) {
                        free(output);
                        throw std::runtime_error("Output index out of bounds in toColumnMajorBlock");
                    }
                    float32x4_t vec = vld1q_f32(&data_[input_idx + j]);
                    vst1q_f32(&output[output_idx], vec);
                    output_idx += 4;
                }

                for (; j < block_size_col; ++j) {
                    if (output_idx >= rows_ * cols_) {
                        free(output);
                        throw std::runtime_error("Output index out of bounds in scalar loop");
                    }
                    output[output_idx++] = data_[input_idx + j];
                }
            }
        }
    }

    return output;
}

// #include "tensor.h"
// #include <stdexcept>
// #include <random>
// #include <set>
// #include <cstring>
// #include <arm_neon.h>
// #include <iostream>



// // helper 
// int* generate_unique_random_indices(int n, int R) {
//     if (n > R + 1) {
//         std::cerr << "Cannot generate " << n << " unique indices in the range [0, " << R << "]\n";
//         return nullptr; // Not enough unique values available
//     }

//     std::set<int> unique_indices; // Set to hold unique indices
//     std::random_device rd;         // Random number generator
//     std::mt19937 gen(rd());        // Mersenne Twister RNG
//     std::uniform_int_distribution<> dis(0, R); // Uniform distribution in the range [0, R]

//     // Generate unique indices
//     while (unique_indices.size() < n) {
//         unique_indices.insert(dis(gen));
//     }

//     // Create a dynamically allocated array of int8_t
//     int* indices_array = new int[n];
//     int index = 0;

//     // Copy the unique indices to the array
//     for (const int& value : unique_indices) {
//         indices_array[index++] = value;
//     }
//     return indices_array; // Return the dynamically allocated array
// }
// void processIndices(
//     int* flatIndices, size_t size, int M, int N,
//     int*& relativeIndices, int*& columnCounts, int*& startIndices, size_t& columnCountSize) {

//     // Validate matrix dimensions
//     if (M <= 0 || N <= 0) {
//         throw std::invalid_argument("Matrix dimensions must be positive.");
//     }

//     // Maximum valid index
//     const int maxIndex = M * N - 1;

//     // Allocate memory for output arrays
//     relativeIndices = new int[size];
//     columnCounts = new int[N]();
//     startIndices = new int[N]();
//     columnCountSize = N;

//     // Temporary array to count how many entries have been assigned per column
//     int* tempColumnCounts = new int[N]();

//     // Calculate column counts, with clipping for invalid indices
//     for (size_t i = 0; i < size; ++i) {
//         int flatIndex = flatIndices[i];

//         // Clip the index to the valid range
//         if (flatIndex < 0) {
//             flatIndex = 0;
//         } else if (flatIndex > maxIndex) {
//             flatIndex = maxIndex;
//         }

//         // Increment column count
//         ++columnCounts[flatIndex / M];
//     }

//     // Calculate start indices
//     int currentStartIndex = 0;
//     for (int col = 0; col < N; ++col) {
//         startIndices[col] = currentStartIndex;
//         currentStartIndex += columnCounts[col];
//     }

//     // Fill relativeIndices using startIndices and tempColumnCounts
//     for (size_t i = 0; i < size; ++i) {
//         int flatIndex = flatIndices[i];

//         // Clip the index to the valid range
//         if (flatIndex < 0) {
//             flatIndex = 0;
//         } else if (flatIndex > maxIndex) {
//             flatIndex = maxIndex;
//         }

//         int column = flatIndex / M;
//         int relativeRow = flatIndex % M;

//         // Place the relative index at the correct position
//         int insertIndex = startIndices[column] + tempColumnCounts[column];
//         relativeIndices[insertIndex] = relativeRow;

//         // Update the temporary count
//         ++tempColumnCounts[column];
//     }

//     // Free temporary array
//     delete[] tempColumnCounts;
// }


// Tensor::Tensor(float* data, int rows, int cols, bool is_sparse) 
//     : rows_(rows), cols_(cols), is_sparse_(is_sparse), total_non_zero_blocks_(0), column_count_size_(0),
//       relative_indices_(nullptr), column_counts_(nullptr), start_indices_(nullptr) {
//     if (rows <= 0 || cols <= 0) {
//         throw std::invalid_argument("Tensor dimensions must be positive");
//     }
    
//     data_ = (float*)aligned_alloc(128, rows * cols * sizeof(float));
//     if (!data_) {
//         throw std::bad_alloc();
//     }
//     std::memcpy(data_, data, rows * cols * sizeof(float));
    
//     if (!is_sparse) {
//         float* row_major = toRowMajorBlock(8, 4);
//         if (!row_major) {
//             free(data_);
//             throw std::runtime_error("Failed to allocate row-major block");
//         }
//         std::memcpy(data_, row_major, rows * cols * sizeof(float));
//         free(row_major);
//     }
// }

// Tensor::~Tensor() {
//     free(data_);
//     delete[] relative_indices_;
//     delete[] column_counts_;
//     delete[] start_indices_;
// }

// float* Tensor::toRowMajorBlock(int block_size_row, int block_size_col) {
//     if (rows_ % block_size_row != 0 || cols_ % block_size_col != 0) {
//         throw std::invalid_argument("Array dimensions must be divisible by block sizes");
//     }

//     float* output = (float*)aligned_alloc(128, rows_ * cols_ * sizeof(float));
//     if (!output) {
//         throw std::bad_alloc();
//     }

//     const int blocks_per_row = cols_ / block_size_col;
//     const int total_block_rows = rows_ / block_size_row;
//     const int block_size = block_size_row * block_size_col;
//     #pragma omp parallel for schedule(guided) collapse(2)
//     for (int block_row_idx = 0; block_row_idx < total_block_rows; ++block_row_idx) {
//         for (int block_col_idx = 0; block_col_idx < blocks_per_row; ++block_col_idx) {
//             int output_idx = (block_row_idx * blocks_per_row + block_col_idx) * block_size;
//             int block_row = block_row_idx * block_size_row;
//             int block_col = block_col_idx * block_size_col;

//             for (int i = 0; i < block_size_row; ++i) {
//                 int row = block_row + i;
//                 int input_idx = row * cols_ + block_col;
//                 if (input_idx + block_size_col > rows_ * cols_) {
//                     free(output);
//                     throw std::runtime_error("Input index out of bounds in toRowMajorBlock");
//                 }
//                 int j = 0;

//                 for (; j <= block_size_col - 4; j += 4) {
//                     if (output_idx + 3 >= rows_ * cols_) {
//                         free(output);
//                         throw std::runtime_error("Output index out of bounds in toRowMajorBlock");
//                     }
//                     float32x4_t vec = vld1q_f32(&data_[input_idx + j]);
//                     vst1q_f32(&output[output_idx], vec);
//                     output_idx += 4;
//                 }

//                 for (; j < block_size_col; ++j) {
//                     if (output_idx >= rows_ * cols_) {
//                         free(output);
//                         throw std::runtime_error("Output index out of bounds in scalar loop");
//                     }
//                     output[output_idx++] = data_[input_idx + j];
//                 }
//             }
//         }
//     }

//     return output;
// }

// float* Tensor::toColumnMajorBlock(int block_size_row, int block_size_col) {
//     if (rows_ % block_size_row != 0 || cols_ % block_size_col != 0) {
//         throw std::invalid_argument("Array dimensions must be divisible by block sizes");
//     }

//     float* output = (float*)aligned_alloc(128, rows_ * cols_ * sizeof(float));
//     if (!output) {
//         throw std::bad_alloc();
//     }

//     const int blocks_per_col = rows_ / block_size_row;
//     const int total_block_cols = cols_ / block_size_col;
//     const int block_size = block_size_row * block_size_col;
//     #pragma omp parallel for schedule(guided) collapse(2)
//     for (int block_col_idx = 0; block_col_idx < total_block_cols; ++block_col_idx) {
//         for (int block_row_idx = 0; block_row_idx < blocks_per_col; ++block_row_idx) {
//             int output_idx = (block_col_idx * blocks_per_col + block_row_idx) * block_size;
//             int block_row = block_row_idx * block_size_row;
//             int block_col = block_col_idx * block_size_col;

//             for (int i = 0; i < block_size_row; ++i) {
//                 int row = block_row + i;
//                 int input_idx = row * cols_ + block_col;
//                 if (input_idx + block_size_col > rows_ * cols_) {
//                     free(output);
//                     throw std::runtime_error("Input index out of bounds in toColumnMajorBlock");
//                 }
//                 int j = 0;

//                 for (; j <= block_size_col - 4; j += 4) {
//                     if (output_idx + 3 >= rows_ * cols_) {
//                         free(output);
//                         throw std::runtime_error("Output index out of bounds in toColumnMajorBlock");
//                     }
//                     float32x4_t vec = vld1q_f32(&data_[input_idx + j]);
//                     vst1q_f32(&output[output_idx], vec);
//                     output_idx += 4;
//                 }

//                 for (; j < block_size_col; ++j) {
//                     if (output_idx >= rows_ * cols_) {
//                         free(output);
//                         throw std::runtime_error("Output index out of bounds in scalar loop");
//                     }
//                     output[output_idx++] = data_[input_idx + j];
//                 }
//             }
//         }
//     }

//     return output;
// }

// void Tensor::sparsify(float sparsity_ratio, int block_size_row, int block_size_col) {
//     if (sparsity_ratio < 0.0f || sparsity_ratio > 1.0f) {
//         throw std::invalid_argument("Sparsity ratio must be between 0 and 1");
//     }
//     if (rows_ % block_size_row != 0 || cols_ % block_size_col != 0) {
//         throw std::invalid_argument("Tensor dimensions must be divisible by block sizes");
//     }

//     int blocks_per_column = rows_ / block_size_row;
//     int num_columns = cols_ / block_size_col;
//     column_count_size_ = num_columns;

//     int total_blocks = blocks_per_column * num_columns;
//     float ratio = 1.0f - sparsity_ratio;
//     total_non_zero_blocks_ = static_cast<size_t>(ratio * static_cast<float>(total_blocks));

//     column_counts_ = new int[num_columns]();
//     start_indices_ = new int[num_columns]();
//     relative_indices_ = new int[total_non_zero_blocks_];


//     int sparse_blocks = static_cast<int>(ratio * (static_cast<float>(total_blocks))); // Divide by 4 to avoid out-of-bounds

//     auto indices = generate_unique_random_indices(sparse_blocks,total_blocks);
//     size_t columnCountSize = 0;
//     processIndices(indices, sparse_blocks,rows_/4,cols_/4, relative_indices_, column_counts_,start_indices_,columnCountSize);


//     // std::cout << "Sparsify: total_blocks=" << total_blocks << ", non_zero_blocks=" << total_non_zero_blocks_ 
//     //         //   << ", blocks_per_column=" << blocks_per_column << ", num_columns=" << num_columns << std::endl;

//     // std::random_device rd;
//     // std::mt19937 gen(rd());
//     // std::uniform_int_distribution<> dis(0, blocks_per_column);

//     // size_t blocks_assigned = 0;
//     // for (int col = 0; col < num_columns && blocks_assigned < total_non_zero_blocks_; ++col) {
//     //     size_t remaining_blocks = total_non_zero_blocks_ - blocks_assigned;
//     //     int max_blocks = std::min(blocks_per_column, static_cast<int>(remaining_blocks));
//     //     column_counts_[col] = (max_blocks > 0) ? dis(gen) % (max_blocks + 1) : 0;
//     //     blocks_assigned += column_counts_[col];
//     // }

//     // while (blocks_assigned < total_non_zero_blocks_) {
//     //     for (int col = 0; col < num_columns && blocks_assigned < total_non_zero_blocks_; ++col) {
//     //         if (column_counts_[col] < blocks_per_column) {
//     //             column_counts_[col]++;
//     //             blocks_assigned++;
//     //         }
//     //     }
//     // }

//     // size_t idx = 0;
//     // for (int col = 0; col < num_columns; ++col) {
//     //     start_indices_[col] = idx;
//     //     if (column_counts_[col] == 0) continue;

//     //     std::set<int> block_indices;
//     //     std::uniform_int_distribution<> block_dis(0, blocks_per_column - 1);
//     //     while (block_indices.size() < static_cast<size_t>(column_counts_[col])) {
//     //         block_indices.insert(block_dis(gen));
//     //     }

//     //     for (int block_row : block_indices) {
//     //         if (idx >= total_non_zero_blocks_) {
//     //             throw std::runtime_error("Index out of bounds in relative_indices_");
//     //         }
//     //         relative_indices_[idx++] = block_row;
//     //     }
//     // }

//     // std::cout << "Sparsity indices: ";
//     // for (size_t i = 0; i < total_non_zero_blocks_; ++i) {
//     //     std::cout << relative_indices_[i] << " ";
//     // }
//     // std::cout << "\nColumn counts: ";
//     // for (size_t i = 0; i < column_count_size_; ++i) {
//     //     std::cout << column_counts_[i] << " ";
//     // }
//     // std::cout << "\nStart indices: ";
//     // for (size_t i = 0; i < column_count_size_; ++i) {
//     //     std::cout << start_indices_[i] << " ";
//     // }
//     // std::cout << std::endl;

//     float* B_sparse = (float*)aligned_alloc(128, rows_ * cols_ * sizeof(float));
//     if (!B_sparse) {
//         throw std::bad_alloc();
//     }
//     std::memset(B_sparse, 0, rows_ * cols_ * sizeof(float));

//     for (int col = 0; col < num_columns; ++col) {
//         int col_start = col * block_size_col;
//         int num_non_zero = column_counts_[col];
//         int st_idx = start_indices_[col];

//         for (int b = 0; b < num_non_zero; ++b) {
//             if (st_idx + b >= total_non_zero_blocks_) {
//                 free(B_sparse);
//                 throw std::runtime_error("Index out of bounds in column iteration");
//             }
//             int block_row = relative_indices_[st_idx + b];
//             int row_start = block_row * block_size_row;
//             for (int i = 0; i < block_size_row; ++i) {
//                 for (int j = 0; j < block_size_col; ++j) {
//                     int src_idx = (row_start + i) * cols_ + (col_start + j);
//                     int dst_idx = src_idx;
//                     if (src_idx >= rows_ * cols_ || dst_idx >= rows_ * cols_) {
//                         free(B_sparse);
//                         throw std::runtime_error("Sparse matrix index out of bounds");
//                     }
//                     B_sparse[dst_idx] = data_[src_idx];
//                 }
//             }
//         }
//     }

//     std::memset(data_, 0, rows_ * cols_ * sizeof(float));
//     for (int col = 0; col < num_columns; ++col) {
//         int col_start = col * block_size_col;
//         int num_non_zero = column_counts_[col];
//         int st_idx = start_indices_[col];

//         for (int b = 0; b < num_non_zero; ++b) {
//             if (st_idx + b >= total_non_zero_blocks_) {
//                 free(B_sparse);
//                 throw std::runtime_error("Index out of bounds in data rearrangement");
//             }
//             int block_row = relative_indices_[st_idx + b];
//             int src_row_start = block_row * block_size_row;
//             int dst_row_start = b * block_size_row;
//             for (int i = 0; i < block_size_row; ++i) {
//                 for (int j = 0; j < block_size_col; ++j) {
//                     int src_idx = (src_row_start + i) * cols_ + (col_start + j);
//                     int dst_idx = (dst_row_start + i) * cols_ + (col_start + j);
//                     if (src_idx >= rows_ * cols_ || dst_idx >= rows_ * cols_) {
//                         free(B_sparse);
//                         throw std::runtime_error("Data index out of bounds in sparsify");
//                     }
//                     data_[dst_idx] = B_sparse[src_idx];
//                 }
//             }
//         }
//     }

//     free(B_sparse);

//     float* col_major = toColumnMajorBlock(block_size_row, block_size_col);
//     if (!col_major) {
//         throw std::runtime_error("Failed to allocate column-major block");
//     }
//     std::memcpy(data_, col_major, rows_ * cols_ * sizeof(float));
//     free(col_major);

//     is_sparse_ = true;
// }


// #include "tensor.h"
// #include <stdexcept>
// #include <random>
// #include <set>
// #include <cstring>
// #include <arm_neon.h>
// #include <omp.h>
// #include <fstream>

// Tensor::Tensor(float* data, int rows, int cols, bool is_sparse) 
//     : rows_(rows), cols_(cols), is_sparse_(is_sparse), total_non_zero_blocks_(0), column_count_size_(0),
//       relative_indices_(nullptr), column_counts_(nullptr), start_indices_(nullptr) {
//     if (rows <= 0 || cols <= 0) {
//         throw std::invalid_argument("Tensor dimensions must be positive");
//     }
    
//     data_ = (float*)aligned_alloc(16, rows * cols * sizeof(float));
//     if (!data_) {
//         throw std::bad_alloc();
//     }
//     std::memcpy(data_, data, rows * cols * sizeof(float));
    
//     if (!is_sparse) {
//         float* row_major = toRowMajorBlock(8, 4);
//         if (!row_major) {
//             free(data_);
//             throw std::runtime_error("Failed to allocate row-major block");
//         }
//         std::memcpy(data_, row_major, rows * cols * sizeof(float));
//         free(row_major);
//     }
// }

// Tensor::~Tensor() {
//     free(data_);
//     delete[] relative_indices_;
//     delete[] column_counts_;
//     delete[] start_indices_;
// }

// float* Tensor::toRowMajorBlock(int block_size_row, int block_size_col) {
//     if (rows_ % block_size_row != 0 || cols_ % block_size_col != 0) {
//         throw std::invalid_argument("Array dimensions must be divisible by block sizes");
//     }

//     float* output = (float*)aligned_alloc(16, rows_ * cols_ * sizeof(float));
//     if (!output) {
//         throw std::bad_alloc();
//     }

//     const int blocks_per_row = cols_ / block_size_col;
//     const int total_block_rows = rows_ / block_size_row;
//     const int block_size = block_size_row * block_size_col;

//     #pragma omp parallel for schedule(guided) collapse(2)
//     for (int block_row_idx = 0; block_row_idx < total_block_rows; ++block_row_idx) {
//         for (int block_col_idx = 0; block_col_idx < blocks_per_row; ++block_col_idx) {
//             int output_idx = (block_row_idx * blocks_per_row + block_col_idx) * block_size;
//             int block_row = block_row_idx * block_size_row;
//             int block_col = block_col_idx * block_size_col;

//             for (int i = 0; i < block_size_row; ++i) {
//                 int row = block_row + i;
//                 int input_idx = row * cols_ + block_col;
//                 if (input_idx + block_size_col > rows_ * cols_) {
//                     free(output);
//                     throw std::runtime_error("Input index out of bounds in toRowMajorBlock");
//                 }
//                 int j = 0;

//                 for (; j <= block_size_col - 4; j += 4) {
//                     if (output_idx + 3 >= rows_ * cols_) {
//                         free(output);
//                         throw std::runtime_error("Output index out of bounds in toRowMajorBlock");
//                     }
//                     float32x4_t vec = vld1q_f32(&data_[input_idx + j]);
//                     vst1q_f32(&output[output_idx], vec);
//                     output_idx += 4;
//                 }

//                 for (; j < block_size_col; ++j) {
//                     if (output_idx >= rows_ * cols_) {
//                         free(output);
//                         throw std::runtime_error("Output index out of bounds in scalar loop");
//                     }
//                     output[output_idx++] = data_[input_idx + j];
//                 }
//             }
//         }
//     }

//     return output;
// }

// float* Tensor::toColumnMajorBlock(int block_size_row, int block_size_col) {
//     if (rows_ % block_size_row != 0 || cols_ % block_size_col != 0) {
//         throw std::invalid_argument("Array dimensions must be divisible by block sizes");
//     }

//     float* output = (float*)aligned_alloc(16, rows_ * cols_ * sizeof(float));
//     if (!output) {
//         throw std::bad_alloc();
//     }

//     const int blocks_per_col = rows_ / block_size_row;
//     const int total_block_cols = cols_ / block_size_col;
//     const int block_size = block_size_row * block_size_col;

//     #pragma omp parallel for schedule(guided) collapse(2)
//     for (int block_col_idx = 0; block_col_idx < total_block_cols; ++block_col_idx) {
//         for (int block_row_idx = 0; block_row_idx < blocks_per_col; ++block_row_idx) {
//             int output_idx = (block_col_idx * blocks_per_col + block_row_idx) * block_size;
//             int block_row = block_row_idx * block_size_row;
//             int block_col = block_col_idx * block_size_col;

//             for (int i = 0; i < block_size_row; ++i) {
//                 int row = block_row + i;
//                 int input_idx = row * cols_ + block_col;
//                 if (input_idx + block_size_col > rows_ * cols_) {
//                     free(output);
//                     throw std::runtime_error("Input index out of bounds in toColumnMajorBlock");
//                 }
//                 int j = 0;

//                 for (; j <= block_size_col - 4; j += 4) {
//                     if (output_idx + 3 >= rows_ * cols_) {
//                         free(output);
//                         throw std::runtime_error("Output index out of bounds in toColumnMajorBlock");
//                     }
//                     float32x4_t vec = vld1q_f32(&data_[input_idx + j]);
//                     vst1q_f32(&output[output_idx], vec);
//                     output_idx += 4;
//                 }

//                 for (; j < block_size_col; ++j) {
//                     if (output_idx >= rows_ * cols_) {
//                         free(output);
//                         throw std::runtime_error("Output index out of bounds in scalar loop");
//                     }
//                     output[output_idx++] = data_[input_idx + j];
//                 }
//             }
//         }
//     }

//     return output;
// }

// void Tensor::sparsify(float sparsity_ratio, int block_size_row, int block_size_col) {
//     if (sparsity_ratio < 0.0f || sparsity_ratio > 1.0f) {
//         throw std::invalid_argument("Sparsity ratio must be between 0 and 1");
//     }
//     if (rows_ % block_size_row != 0 || cols_ % block_size_col != 0) {
//         throw std::invalid_argument("Tensor dimensions must be divisible by block sizes");
//     }

//     int blocks_per_column = rows_ / block_size_row;
//     int num_columns = cols_ / block_size_col;
//     column_count_size_ = num_columns;

//     int total_blocks = blocks_per_column * num_columns;
//     float ratio = 1.0f - sparsity_ratio;
//     total_non_zero_blocks_ = static_cast<size_t>(ratio * static_cast<float>(total_blocks));

//     column_counts_ = new int[num_columns]();
//     start_indices_ = new int[num_columns]();
//     relative_indices_ = new int[total_non_zero_blocks_];

//     std::random_device rd;
//     std::mt19937 gen(rd());
//     std::uniform_int_distribution<> dis(0, blocks_per_column);

//     size_t blocks_assigned = 0;
//     for (int col = 0; col < num_columns && blocks_assigned < total_non_zero_blocks_; ++col) {
//         size_t remaining_blocks = total_non_zero_blocks_ - blocks_assigned;
//         int max_blocks = std::min(blocks_per_column, static_cast<int>(remaining_blocks));
//         column_counts_[col] = (max_blocks > 0) ? dis(gen) % (max_blocks + 1) : 0;
//         blocks_assigned += column_counts_[col];
//     }

//     while (blocks_assigned < total_non_zero_blocks_) {
//         for (int col = 0; col < num_columns && blocks_assigned < total_non_zero_blocks_; ++col) {
//             if (column_counts_[col] < blocks_per_column) {
//                 column_counts_[col]++;
//                 blocks_assigned++;
//             }
//         }
//     }

//     size_t idx = 0;
//     for (int col = 0; col < num_columns; ++col) {
//         start_indices_[col] = idx;
//         if (column_counts_[col] == 0) continue;

//         std::set<int> block_indices;
//         std::uniform_int_distribution<> block_dis(0, blocks_per_column - 1);
//         while (block_indices.size() < static_cast<size_t>(column_counts_[col])) {
//             block_indices.insert(block_dis(gen));
//         }

//         for (int block_row : block_indices) {
//             if (idx >= total_non_zero_blocks_) {
//                 throw std::runtime_error("Index out of bounds in relative_indices_");
//             }
//             relative_indices_[idx++] = block_row;
//         }
//     }

//     float* B_sparse = (float*)aligned_alloc(16, rows_ * cols_ * sizeof(float));
//     if (!B_sparse) {
//         throw std::bad_alloc();
//     }
//     std::memset(B_sparse, 0, rows_ * cols_ * sizeof(float));

//     for (int col = 0; col < num_columns; ++col) {
//         int col_start = col * block_size_col;
//         int num_non_zero = column_counts_[col];
//         int st_idx = start_indices_[col];

//         for (int b = 0; b < num_non_zero; ++b) {
//             if (st_idx + b >= total_non_zero_blocks_) {
//                 free(B_sparse);
//                 throw std::runtime_error("Index out of bounds in column iteration");
//             }
//             int block_row = relative_indices_[st_idx + b];
//             int row_start = block_row * block_size_row;
//             for (int i = 0; i < block_size_row; ++i) {
//                 for (int j = 0; j < block_size_col; ++j) {
//                     int src_idx = (row_start + i) * cols_ + (col_start + j);
//                     int dst_idx = src_idx;
//                     if (src_idx >= rows_ * cols_ || dst_idx >= rows_ * cols_) {
//                         free(B_sparse);
//                         throw std::runtime_error("Sparse matrix index out of bounds");
//                     }
//                     B_sparse[dst_idx] = data_[src_idx];
//                 }
//             }
//         }
//     }

//     std::memset(data_, 0, rows_ * cols_ * sizeof(float));
//     for (int col = 0; col < num_columns; ++col) {
//         int col_start = col * block_size_col;
//         int num_non_zero = column_counts_[col];
//         int st_idx = start_indices_[col];

//         for (int b = 0; b < num_non_zero; ++b) {
//             if (st_idx + b >= total_non_zero_blocks_) {
//                 free(B_sparse);
//                 throw std::runtime_error("Index out of bounds in data rearrangement");
//             }
//             int block_row = relative_indices_[st_idx + b];
//             int src_row_start = block_row * block_size_row;
//             int dst_row_start = b * block_size_row;
//             for (int i = 0; i < block_size_row; ++i) {
//                 for (int j = 0; j < block_size_col; ++j) {
//                     int src_idx = (src_row_start + i) * cols_ + (col_start + j);
//                     int dst_idx = (dst_row_start + i) * cols_ + (col_start + j);
//                     if (src_idx >= rows_ * cols_ || dst_idx >= rows_ * cols_) {
//                         free(B_sparse);
//                         throw std::runtime_error("Data index out of bounds in sparsify");
//                     }
//                     data_[dst_idx] = B_sparse[src_idx];
//                 }
//             }
//         }
//     }

//     free(B_sparse);

//     float* col_major = toColumnMajorBlock(block_size_row, block_size_col);
//     if (!col_major) {
//         throw std::runtime_error("Failed to allocate column-major block");
//     }
//     std::memcpy(data_, col_major, rows_ * cols_ * sizeof(float));
//     free(col_major);

//     // std::ofstream out("sparsify_indices.txt");
//     // out << "relative_indices: ";
//     // for (size_t i = 0; i < total_non_zero_blocks_; ++i) out << relative_indices_[i] << " ";
//     // out << "\ncolumn_counts: ";
//     // for (size_t i = 0; i < column_count_size_; ++i) out << column_counts_[i] << " ";
//     // out << "\nstart_indices: ";
//     // for (size_t i = 0; i < column_count_size_; ++i) out << start_indices_[i] << " ";
//     // out.close();

//     is_sparse_ = true;
// }