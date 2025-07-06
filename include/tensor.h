#ifndef TENSOR_H
#define TENSOR_H

#pragma once
#include <cstddef>
#include <arm_neon.h>

enum class MatrixFormat {
    ROW_MAJOR,
    BLOCK_MAJOR,
    COLUMN_BLOCK_MAJOR,
    ROW_BLOCK_MAJOR
};

class Tensor {
private:
    float* data_;
    int rows_;
    int cols_;
    bool is_sparse_;
    size_t total_non_zero_blocks_;
    size_t column_count_size_;
    int* relative_indices_;
    int* column_counts_;
    int* start_indices_;

public:
    Tensor(float* data, int rows, int cols, bool is_sparse, int block_size_row, int block_size_col, 
           MatrixFormat current_format, MatrixFormat convert_format);
    ~Tensor();

    float* getData() const { return data_; }
    int getRows() const { return rows_; }
    int getCols() const { return cols_; }
    bool isSparse() const { return is_sparse_; }
    size_t getTotalNonZeroBlocks() const { return total_non_zero_blocks_; }
    size_t getColumnCountSize() const { return column_count_size_; }
    int* getRelativeIndices() const { return relative_indices_; }
    int* getColumnCounts() const { return column_counts_; }
    int* getStartIndices() const { return start_indices_; }

private:
    float* toRowMajorBlock(int block_size_row, int block_size_col);
    float* toColumnMajorBlock(int block_size_row, int block_size_col);
};

#endif // TENSOR_H
// #ifndef TENSOR_H
// #define TENSOR_H

// #include <cstdint>
// #include <vector>
// #include <cstddef>
// class Tensor {
// public:
//     // Constructor
//     Tensor(float* data, int rows, int cols, bool is_sparse = false);
    
//     // Destructor
//     ~Tensor();

//     // Conversion functions
//     float* toRowMajorBlock(int block_size_row, int block_size_col);
//     float* toColumnMajorBlock(int block_size_row, int block_size_col);
    
//     // Sparsify function
//     void sparsify(float sparsity_ratio, int block_size_row, int block_size_col);

//     // Const getters
//     float* getData() const { return data_; }
//     int getRows() const { return rows_; }
//     int getCols() const { return cols_; }
//     bool isSparse() const { return is_sparse_; }
//     const int* getRelativeIndices() const { return relative_indices_; }
//     const int* getColumnCounts() const { return column_counts_; }
//     const int* getStartIndices() const { return start_indices_; }
//     size_t getTotalNonZeroBlocks() const { return total_non_zero_blocks_; }
//     size_t getColumnCountSize() const { return column_count_size_; }

//     // Non-const getters for modifiable access
//     int* getRelativeIndicesMutable() { return relative_indices_; }
//     int* getColumnCountsMutable() { return column_counts_; }
//     int* getStartIndicesMutable() { return start_indices_; }

// private:
//     float* data_;                      // Matrix data
//     int rows_;                         // Number of rows
//     int cols_;                         // Number of columns
//     bool is_sparse_;                   // Sparse flag
//     int* relative_indices_;            // Indices for sparse blocks
//     int* column_counts_;               // Non-zero block counts per column
//     int* start_indices_;               // Start indices for each column
//     size_t total_non_zero_blocks_;     // Total number of non-zero blocks
//     size_t column_count_size_;         // Size of column_counts_ and start_indices_
// };

// #endif // TENSOR_H