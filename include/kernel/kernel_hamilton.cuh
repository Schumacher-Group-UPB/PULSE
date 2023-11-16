#pragma once
#include "cuda/cuda_complex.cuh"

namespace PC3::Hamilton {

CUDA_DEVICE inline bool is_valid_index( const int row, const int col, const int N ) {
    return row >= 0 && row < N && col >= 0 && col < N;
}

CUDA_DEVICE inline complex_number upper_neighbour( complex_number* vector, int index, const int row, const int col, const int distance, const int N ) {
    // We are alread in the most upper row
    if ( !is_valid_index( row - distance, col, N ) )
        return { 0.0, 0.0 };
    // Valid upper neighbour
    return vector[index - N * distance];
}
CUDA_DEVICE inline complex_number lower_neighbour( complex_number* vector, int index, const int row, const int col, const int distance, const int N ) {
    // We are already in the most lower row
    if ( !is_valid_index( row + distance, col, N ) )
        return { 0.0, 0.0 };
    // Valid lower neighbour
    return vector[index + N * distance];
}
CUDA_DEVICE inline complex_number left_neighbour( complex_number* vector, int index, const int row, const int col, const int distance, const int N ) {
    // Desired Index is in the previous row
    if ( !is_valid_index( row, col - distance, N ) )
        return { 0.0, 0.0 };
    // Valid left neighbour
    return vector[index - distance];
}
CUDA_DEVICE inline complex_number right_neighbour( complex_number* vector, int index, const int row, const int col, const int distance, const int N ) {
    // Desired Index is in the next row
    if ( !is_valid_index( row, col + distance, N ) )
        return { 0.0, 0.0 };
    // Valid right neighbour
    return vector[index + distance];
}

CUDA_DEVICE void scalar( complex_number& regular, complex_number* __restrict__ vector, int index, const int row, const int col, const int N );

CUDA_DEVICE void tetm_plus( complex_number& regular, complex_number& cross, complex_number* __restrict__ vector, int index, const int row, const int col, const int N );

CUDA_DEVICE void tetm_minus( complex_number& regular, complex_number& cross, complex_number* __restrict__ vector, int index, const int row, const int col, const int N );

} // namespace PC3::Hamilton