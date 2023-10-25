#pragma once
#include "cuda_complex_math.cuh"

CUDA_DEVICE static __inline__ bool is_valid_index( const int row, const int col, const int N ) {
    return row >= 0 && row < N && col >= 0 && col < N;
}

CUDA_DEVICE static __inline__ complex_number upper_neighbour( complex_number* vector, int index, const int row, const int col, const int distance, const int N ) {
    // We are alread in the most upper row
    if ( !is_valid_index( row - distance, col, N ) )
        return { 0.0, 0.0 };
    // Valid upper neighbour
    return vector[index - N * distance];
}
CUDA_DEVICE static __inline__ complex_number lower_neighbour( complex_number* vector, int index, const int row, const int col, const int distance, const int N ) {
    // We are already in the most lower row
    if ( !is_valid_index( row + distance, col, N ) )
        return { 0.0, 0.0 };
    // Valid lower neighbour
    return vector[index + N * distance];
}
CUDA_DEVICE static __inline__ complex_number left_neighbour( complex_number* vector, int index, const int row, const int col, const int distance, const int N ) {
    // Desired Index is in the previous row
    if ( !is_valid_index( row, col - distance, N ) )
        return { 0.0, 0.0 };
    // Valid left neighbour
    return vector[index - distance];
}
CUDA_DEVICE static __inline__ complex_number right_neighbour( complex_number* vector, int index, const int row, const int col, const int distance, const int N ) {
    // Desired Index is in the next row
    if ( !is_valid_index( row, col + distance, N ) )
        return { 0.0, 0.0 };
    // Valid right neighbour
    return vector[index + distance];
}

CUDA_DEVICE static __inline__ void hamilton( complex_number& regular, complex_number* __restrict__ vector, int index, const int row, const int col, const int N ) {
    const auto upper = upper_neighbour( vector, index, row, col, 1, N );
    const auto lower = lower_neighbour( vector, index, row, col, 1, N );
    const auto left = left_neighbour( vector, index, row, col, 1, N );
    const auto right = right_neighbour( vector, index, row, col, 1, N );
    regular = -4.0 * vector[index] + upper + lower + left + right;
}

CUDA_DEVICE static __inline__ void hamilton_1( complex_number& regular, complex_number& cross, complex_number* __restrict__ vector, int index, const int row, const int col, const int N ) {
    const auto upper = upper_neighbour( vector, index, row, col, 1, N );
    const auto lower = lower_neighbour( vector, index, row, col, 1, N );
    const auto left = left_neighbour( vector, index, row, col, 1, N );
    const auto right = right_neighbour( vector, index, row, col, 1, N );
    regular = -4.0 * vector[index] + upper + lower + left + right;
    cross = upper + lower - left - right - dev_half_i * ( right_neighbour( vector, index - N, row - 1, col, 1, N ) - left_neighbour( vector, index - N, row - 1, col, 1, N ) ) + dev_half_i * ( right_neighbour( vector, index + N, row + 1, col, 1, N ) - left_neighbour( vector, index + N, row + 1, col, 1, N ) );
}

CUDA_DEVICE static __inline__ void hamilton_2( complex_number& regular, complex_number& cross, complex_number* __restrict__ vector, int index, const int row, const int col, const int N ) {
    const auto upper = upper_neighbour( vector, index, row, col, 1, N );
    const auto lower = lower_neighbour( vector, index, row, col, 1, N );
    const auto left = left_neighbour( vector, index, row, col, 1, N );
    const auto right = right_neighbour( vector, index, row, col, 1, N );
    regular = -4. * vector[index] + upper + lower + left + right;
    cross = upper + lower - left - right + dev_half_i * ( right_neighbour( vector, index - N, row - 1, col, 1, N ) - left_neighbour( vector, index - N, row - 1, col, 1, N ) ) - dev_half_i * ( right_neighbour( vector, index + N, row + 1, col, 1, N ) - left_neighbour( vector, index + N, row + 1, col, 1, N ) );
}