#include "kernel/kernel_hamilton.cuh"

CUDA_DEVICE void PC3::Hamilton::scalar( complex_number& regular, complex_number* __restrict__ vector, int index, const int row, const int col, const int N ) {
    const auto upper = upper_neighbour( vector, index, row, col, 1, N );
    const auto lower = lower_neighbour( vector, index, row, col, 1, N );
    const auto left = left_neighbour( vector, index, row, col, 1, N );
    const auto right = right_neighbour( vector, index, row, col, 1, N );
    regular = -4.0 * vector[index] + upper + lower + left + right;
}

CUDA_DEVICE void PC3::Hamilton::tetm_plus( complex_number& regular, complex_number& cross, complex_number* __restrict__ vector, int index, const int row, const int col, const int N ) {
    const auto upper = upper_neighbour( vector, index, row, col, 1, N );
    const auto lower = lower_neighbour( vector, index, row, col, 1, N );
    const auto left = left_neighbour( vector, index, row, col, 1, N );
    const auto right = right_neighbour( vector, index, row, col, 1, N );
    regular = -4.0 * vector[index] + upper + lower + left + right;
    cross = upper + lower - left - right + complex_number(0.0,-0.5) * ( right_neighbour( vector, index - N, row - 1, col, 1, N ) - left_neighbour( vector, index - N, row - 1, col, 1, N ) - right_neighbour( vector, index + N, row + 1, col, 1, N ) + left_neighbour( vector, index + N, row + 1, col, 1, N ) );
}

CUDA_DEVICE void PC3::Hamilton::tetm_minus( complex_number& regular, complex_number& cross, complex_number* __restrict__ vector, int index, const int row, const int col, const int N ) {
    const auto upper = upper_neighbour( vector, index, row, col, 1, N );
    const auto lower = lower_neighbour( vector, index, row, col, 1, N );
    const auto left = left_neighbour( vector, index, row, col, 1, N );
    const auto right = right_neighbour( vector, index, row, col, 1, N );
    regular = -4. * vector[index] + upper + lower + left + right;
    cross = upper + lower - left - right + complex_number(0.0,0.5) * ( right_neighbour( vector, index - N, row - 1, col, 1, N ) - left_neighbour( vector, index - N, row - 1, col, 1, N ) - right_neighbour( vector, index + N, row + 1, col, 1, N ) + left_neighbour( vector, index + N, row + 1, col, 1, N ) );
}