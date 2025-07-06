import numpy as np
import block_sparse
from utils import make_block_sparse_matrix , benchmark_matmul
# Create random matrices
M, N, K = 1024, 1024, 1024
A = np.random.uniform(0, 1, (M, N)).astype(np.float32)
B = np.random.uniform(0, 1, (N, K)).astype(np.float32)

# Apply block sparsity to B
sparsity = 0.99
B = make_block_sparse_matrix(B, 4, 4, sparsity_ratio=sparsity)
print("Sparsity : ", sparsity)

# Create block_sparse tensors
A_tensor = block_sparse.create_tensor(
    np_array=A,
    is_sparse=False,
    block_size_row=8,
    block_size_col=4,
    current_format=block_sparse.MatrixFormat.ROW_MAJOR,
    convert_format=block_sparse.MatrixFormat.ROW_BLOCK_MAJOR
)
B_tensor = block_sparse.create_tensor(
    np_array=B,
    is_sparse=True,
    block_size_row=4,
    block_size_col=4,
    current_format=block_sparse.MatrixFormat.ROW_MAJOR,
    convert_format=block_sparse.MatrixFormat.COLUMN_BLOCK_MAJOR
)

# Run benchmark
runs = 10
sparse_avg, numpy_avg = benchmark_matmul(A, B, A_tensor, B_tensor, runs)

# Print results
print(f"Benchmark Results for {runs} runs:")
print(f"Average time per block_sparse run: {sparse_avg:.6f} ms")
print(f"Average time per NumPy run: {numpy_avg:.6f} ms")

#result matrices
C = block_sparse.matmul_blocked(A_tensor, B_tensor)
C_np = np.matmul(A, B)
