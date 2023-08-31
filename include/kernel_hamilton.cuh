#pragma once
#include "cuda_complex_math.cuh"

__device__ static __inline__ bool is_valid_index( const int row, const int col, const int N ) {
    return row >= 0 && row < N && col >= 0 && col < N;
}

__device__ static __inline__ complex_number upper_neighbour( complex_number* vector, int index, const int row, const int col, const int distance, const int N ) {
    // We are alread in the most upper row
    if ( !is_valid_index( row - distance, col, N ) )
        return { 0.0, 0.0 };
    // Valid upper neighbour
    return vector[index - N * distance];
}
__device__ static __inline__ complex_number lower_neighbour( complex_number* vector, int index, const int row, const int col, const int distance, const int N ) {
    // We are already in the most lower row
    if ( !is_valid_index( row + distance, col, N ) )
        return { 0.0, 0.0 };
    // Valid lower neighbour
    return vector[index + N * distance];
}
__device__ static __inline__ complex_number left_neighbour( complex_number* vector, int index, const int row, const int col, const int distance, const int N ) {
    // Desired Index is in the previous row
    if ( !is_valid_index( row, col - distance, N ) )
        return { 0.0, 0.0 };
    // Valid left neighbour
    return vector[index - distance];
}
__device__ static __inline__ complex_number right_neighbour( complex_number* vector, int index, const int row, const int col, const int distance, const int N ) {
    // Desired Index is in the next row
    if ( !is_valid_index( row, col + distance, N ) )
        return { 0.0, 0.0 };
    // Valid right neighbour
    return vector[index + distance];
}

__device__ static __inline__ void hamilton( complex_number& DT1, complex_number* __restrict__ vector, int index, const int row, const int col, const int N ) {
    const auto upper = upper_neighbour( vector, index, row, col, 1, N );
    const auto lower = lower_neighbour( vector, index, row, col, 1, N );
    const auto left = left_neighbour( vector, index, row, col, 1, N );
    const auto right = right_neighbour( vector, index, row, col, 1, N );
    DT1 = -4.0 * vector[index] + upper + lower + left + right;
}

__device__ static __inline__ void hamilton_1( complex_number& DT1, complex_number& DT4, complex_number* __restrict__ vector, int index, const int row, const int col, const int N ) {
    const auto upper = upper_neighbour( vector, index, row, col, 1, N );
    const auto lower = lower_neighbour( vector, index, row, col, 1, N );
    const auto left = left_neighbour( vector, index, row, col, 1, N );
    const auto right = right_neighbour( vector, index, row, col, 1, N );
    DT1 = -4.0 * vector[index] + upper + lower + left + right;
    DT4 = upper + lower - left - right - dev_half_i * ( right_neighbour( vector, index - N, row - 1, col, 1, N ) - left_neighbour( vector, index - N, row - 1, col, 1, N ) ) + dev_half_i * ( right_neighbour( vector, index + N, row + 1, col, 1, N ) - left_neighbour( vector, index + N, row + 1, col, 1, N ) );
}

__device__ static __inline__ void hamilton_2( complex_number& DT2, complex_number& DT3, complex_number* __restrict__ vector, int index, const int row, const int col, const int N ) {
    const auto upper = upper_neighbour( vector, index, row, col, 1, N );
    const auto lower = lower_neighbour( vector, index, row, col, 1, N );
    const auto left = left_neighbour( vector, index, row, col, 1, N );
    const auto right = right_neighbour( vector, index, row, col, 1, N );
    DT3 = -4. * vector[index] + upper + lower + left + right;
    DT2 = upper + lower - left - right + dev_half_i * ( right_neighbour( vector, index - N, row - 1, col, 1, N ) - left_neighbour( vector, index - N, row - 1, col, 1, N ) ) - dev_half_i * ( right_neighbour( vector, index + N, row + 1, col, 1, N ) - left_neighbour( vector, index + N, row + 1, col, 1, N ) );
}