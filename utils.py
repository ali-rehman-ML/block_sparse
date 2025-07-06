import numpy as np
import time
import block_sparse
def make_block_sparse_matrix(matrix, block_size_row, block_size_col, sparsity_ratio):
    """
    Creates a sparse matrix by zeroing out blocks uniformly based on sparsity ratio.
    
    Parameters:
    matrix (np.ndarray): Input NumPy array
    block_size_row (int): Number of rows in each block
    block_size_col (int): Number of columns in each block
    sparsity_ratio (float): Ratio of blocks to zero out (0 to 1)
    
    Returns:
    np.ndarray: Sparse matrix with specified blocks zeroed out
    """
    if not 0 <= sparsity_ratio <= 1:
        raise ValueError("Sparsity ratio must be between 0 and 1")
    
    if block_size_row <= 0 or block_size_col <= 0:
        raise ValueError("Block sizes must be positive integers")
    
    # Get matrix dimensions
    rows, cols = matrix.shape
    
    # Calculate number of blocks
    num_blocks_row = (rows + block_size_row - 1) // block_size_row
    num_blocks_col = (cols + block_size_col - 1) // block_size_col
    total_blocks = num_blocks_row * num_blocks_col
    
    # Calculate number of blocks to zero out
    num_zero_blocks = int(total_blocks * sparsity_ratio)
    
    # Create copy of input matrix
    sparse_matrix = matrix.copy()
    
    # Generate random indices for blocks to zero out
    block_indices = np.random.choice(total_blocks, 
                                   size=num_zero_blocks, 
                                   replace=False)
    
    # Convert flat indices to row,col block coordinates
    for idx in block_indices:
        block_row = idx // num_blocks_col
        block_col = idx % num_blocks_col
        
        # Calculate block boundaries
        row_start = block_row * block_size_row
        row_end = min(row_start + block_size_row, rows)
        col_start = block_col * block_size_col
        col_end = min(col_start + block_size_col, cols)
        
        # Zero out the block
        sparse_matrix[row_start:row_end, col_start:col_end] = 0
    
    return sparse_matrix

def benchmark_matmul(A, B, A_tensor, B_tensor, runs=15):
    """
    Benchmarks block_sparse and NumPy matrix multiplication with 1 warmup run.
    
    Parameters:
    A (np.ndarray): First input matrix
    B (np.ndarray): Second input matrix (sparse)
    A_tensor: Sparse_matmul tensor for A
    B_tensor: Sparse_matmul tensor for B
    runs (int): Number of benchmark runs
    
    Returns:
    tuple: (sparse_avg_time, numpy_avg_time, norm_diff)
    """
    # Warmup run for block_sparse
    C = block_sparse.matmul_blocked(A_tensor, B_tensor)
    
    # Benchmark block_sparse
    start_time = time.time()
    for _ in range(runs):
        C = block_sparse.matmul_blocked(A_tensor, B_tensor)
    end_time = time.time()
    sparse_avg_time = ((end_time - start_time) / runs)*1000
    
    # Warmup run for NumPy
    C_np = np.matmul(A, B)
    
    # Benchmark NumPy
    start_time_np = time.time()
    for _ in range(runs):
        C_np = np.matmul(A, B)
    end_time_np = time.time()
    numpy_avg_time = ((end_time_np - start_time_np) / runs)*1000

    
    return sparse_avg_time, numpy_avg_time
