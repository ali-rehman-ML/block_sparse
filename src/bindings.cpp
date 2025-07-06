#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "matmul.h"
#include "tensor.h"
#include <stdexcept>
#include <cstring>
#include <iostream>
#include <random>
#include <memory>
#include <chrono>

namespace py = pybind11;

// Helper function to initialize matrix with random values
void initialize_matrix_random(float* ptr, int M, int N) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    for (int i = 0; i < M * N; ++i) {
        ptr[i] = dis(gen);
    }
}

// Create a Tensor object from a NumPy array
std::shared_ptr<Tensor> create_tensor(py::array_t<float> np_array, bool is_sparse,  
                                    int block_size_row = 4, int block_size_col = 4,
                                    MatrixFormat current_format = MatrixFormat::ROW_MAJOR,
                                    MatrixFormat convert_format = MatrixFormat::ROW_MAJOR) {
    py::gil_scoped_release release_gil;

    if (np_array.ndim() != 2) {
        throw std::invalid_argument("Input array must be 2-dimensional");
    }

    int rows = np_array.shape(0);
    int cols = np_array.shape(1);

    if (rows <= 0 || cols <= 0) {
        throw std::invalid_argument("Tensor dimensions must be positive");
    }

    if (block_size_row <= 0 || block_size_col <= 0) {
        throw std::invalid_argument("Block sizes must be positive");
    }

    if (is_sparse && (rows % block_size_row != 0 || cols % block_size_col != 0)) {
        throw std::invalid_argument("Tensor dimensions must be divisible by block sizes for sparse tensor");
    }

    py::array_t<float> contig_array = py::array_t<float>(py::array::ensure(np_array, py::array::c_style | py::array::forcecast));

    float* data = static_cast<float*>(aligned_alloc(128, rows * cols * sizeof(float)));
    if (!data) {
        throw std::runtime_error("Memory allocation failed for Tensor data");
    }

    if (reinterpret_cast<uintptr_t>(data) % 128 != 0) {
        free(data);
        throw std::runtime_error("Allocated memory not 128-byte aligned");
    }

    if (np_array.data() != nullptr) {
        std::memcpy(data, contig_array.data(), rows * cols * sizeof(float));
        for (int i = 0; i < rows * cols; ++i) {
            if (!std::isfinite(data[i])) {
                free(data);
                throw std::runtime_error("Invalid data in input array at index " + std::to_string(i));
            }
        }
    } else {
        std::cout << "Initializing Tensor data with random values: size=" << rows * cols * sizeof(float) << " bytes" << std::endl;
        initialize_matrix_random(data, rows, cols);
    }

    std::shared_ptr<Tensor> tensor;
    try {
        tensor = std::make_shared<Tensor>(data, rows, cols, is_sparse, block_size_row, block_size_col, 
                                        current_format, convert_format);
    } catch (...) {
        free(data);
        throw;
    }

    return tensor;
}

// Perform matrix multiplication with Tensor objects and return result as NumPy array
py::array_t<float> matmul_blocked_python(std::shared_ptr<Tensor> A, std::shared_ptr<Tensor> B) {
    py::gil_scoped_release release_gil;

    if (!A || !B) {
        throw std::invalid_argument("Tensor objects A and B cannot be null");
    }

    int M = A->getRows();
    int N = A->getCols();
    int K = B->getCols();

    if (B->getRows() != N) {
        throw std::invalid_argument("Incompatible Tensor dimensions: A(" + 
                                    std::to_string(M) + "," + std::to_string(N) + "), B(" + 
                                    std::to_string(B->getRows()) + "," + std::to_string(K) + ")");
    }

    if (!B->isSparse()) {
        throw std::invalid_argument("Tensor B must be sparse for matmul_blocked");
    }

    // Create Tensor C with zeros
    float* C_data = static_cast<float*>(aligned_alloc(128, M * K * sizeof(float)));
    if (!C_data) {
        throw std::runtime_error("Memory allocation failed for Tensor C");
    }
    if (reinterpret_cast<uintptr_t>(C_data) % 128 != 0) {
        free(C_data);
        throw std::runtime_error("Allocated memory for C not 128-byte aligned");
    }
    std::memset(C_data, 0, M * K * sizeof(float));

    // std::shared_ptr<Tensor> C;
    // try {
    //     C = std::make_shared<Tensor>(C_data, M, K, false, 4, 4, MatrixFormat::ROW_MAJOR, MatrixFormat::ROW_MAJOR);
    // } catch (...) {
    //     free(C_data);
    //     throw;
    // }

    auto start = std::chrono::high_resolution_clock::now();
    matmul_blocked(*A, *B, C_data);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    py::gil_scoped_acquire acquire_gil;
    py::array_t<float> C_np({M, K});
    py::buffer_info C_buf = C_np.request(true);
    std::memcpy(C_buf.ptr, C_data, M * K * sizeof(float));

    return C_np;
}

PYBIND11_MODULE(block_sparse, m) {
    m.doc() = "Block sparse matrix multiplication module for ARM CPUs";

    py::enum_<MatrixFormat>(m, "MatrixFormat")
        .value("ROW_MAJOR", MatrixFormat::ROW_MAJOR)
        .value("BLOCK_MAJOR", MatrixFormat::BLOCK_MAJOR)
        .value("COLUMN_BLOCK_MAJOR", MatrixFormat::COLUMN_BLOCK_MAJOR)
        .value("ROW_BLOCK_MAJOR", MatrixFormat::ROW_BLOCK_MAJOR)
        .export_values();

    py::class_<Tensor, std::shared_ptr<Tensor>>(m, "Tensor")
        .def(py::init<float*, int, int, bool, int, int, MatrixFormat, MatrixFormat>())
        .def("get_rows", &Tensor::getRows)
        .def("get_cols", &Tensor::getCols)
        .def("is_sparse", &Tensor::isSparse)
        .def("get_data", [](Tensor &t) {
            return py::array_t<float>({t.getRows(), t.getCols()}, t.getData());
        }, py::return_value_policy::reference_internal)
        .def("get_relative_indices", [](Tensor &t) {
            if (!t.isSparse()) throw std::runtime_error("Tensor is not sparse");
            return py::array_t<int>(t.getTotalNonZeroBlocks(), t.getRelativeIndices());
        }, py::return_value_policy::reference_internal)
        .def("get_column_counts", [](Tensor &t) {
            if (!t.isSparse()) throw std::runtime_error("Tensor is not sparse");
            return py::array_t<int>(t.getColumnCountSize(), t.getColumnCounts());
        }, py::return_value_policy::reference_internal)
        .def("get_start_indices", [](Tensor &t) {
            if (!t.isSparse()) throw std::runtime_error("Tensor is not sparse");
            return py::array_t<int>(t.getColumnCountSize(), t.getStartIndices());
        }, py::return_value_policy::reference_internal);

    m.def("create_tensor", &create_tensor, "Create a Tensor from a NumPy array",
          py::arg("np_array"), 
          py::arg("is_sparse") = false, 
          py::arg("block_size_row") = 4, 
          py::arg("block_size_col") = 4,
          py::arg("current_format") = MatrixFormat::ROW_MAJOR,
          py::arg("convert_format") = MatrixFormat::ROW_MAJOR);

    m.def("matmul_blocked", &matmul_blocked_python, "Perform blocked sparse matrix multiplication with Tensors and return result as NumPy array",
          py::arg("A"), py::arg("B"));
}
