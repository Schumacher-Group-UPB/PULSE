#pragma once
#include "cuda/cuda_complex.cuh"

namespace PC3::Hamilton {

CUDA_DEVICE CUDA_INLINE bool is_valid_index( const int row, const int col, const int N ) {
    return row >= 0 && row < N && col >= 0 && col < N;
}

/**
 * Dirichlet boundary conditions
 * For Dirichlet boundary conditions, the derivative is zero at the boundary.
 * Hence, when we leave the main grid, we simply return zero.
*/

CUDA_DEVICE CUDA_INLINE complex_number upper_neighbour( complex_number* vector, int index, const int row, const int col, const int distance, const int N ) {
    // We are alread in the most upper row
    if ( !is_valid_index( row - distance, col, N ) )
        return { 0.0, 0.0 };
    // Valid upper neighbour
    return vector[index - N * distance];
}
CUDA_DEVICE CUDA_INLINE complex_number lower_neighbour( complex_number* vector, int index, const int row, const int col, const int distance, const int N ) {
    // We are already in the most lower row
    if ( !is_valid_index( row + distance, col, N ) )
        return { 0.0, 0.0 };
    // Valid lower neighbour
    return vector[index + N * distance];
}
CUDA_DEVICE CUDA_INLINE complex_number left_neighbour( complex_number* vector, int index, const int row, const int col, const int distance, const int N ) {
    // Desired Index is in the previous row
    if ( !is_valid_index( row, col - distance, N ) )
        return { 0.0, 0.0 };
    // Valid left neighbour
    return vector[index - distance];
}
CUDA_DEVICE CUDA_INLINE complex_number right_neighbour( complex_number* vector, int index, const int row, const int col, const int distance, const int N ) {
    // Desired Index is in the next row
    if ( !is_valid_index( row, col + distance, N ) )
        return { 0.0, 0.0 };
    // Valid right neighbour
    return vector[index + distance];
}

/**
 * Von-Neumann boundary conditions.
 * For Von-Neumann boundary conditions, the derivative is non-zero at the boundary.
 * In this case, we implement periodic boundary conditions.
 * Hence, when we leave the main grid, we return the value of the opposite side.
*/

CUDA_DEVICE CUDA_INLINE complex_number upper_neighbour_periodic( complex_number* vector, int index, const int row, const int col, const int distance, const int N ) {
    // We are alread in the most upper row
    if ( !is_valid_index( row - distance, col, N ) )
        return vector[index + N * ( N - distance )];
    // Valid upper neighbour
    return vector[index - N * distance];
}
CUDA_DEVICE CUDA_INLINE complex_number lower_neighbour_periodic( complex_number* vector, int index, const int row, const int col, const int distance, const int N ) {
    // We are already in the most lower row
    if ( !is_valid_index( row + distance, col, N ) )
        return vector[index - N * ( N - distance )];
    // Valid lower neighbour
    return vector[index + N * distance];
}
CUDA_DEVICE CUDA_INLINE complex_number left_neighbour_periodic( complex_number* vector, int index, const int row, const int col, const int distance, const int N ) {
    // Desired Index is in the previous row
    if ( !is_valid_index( row, col - distance, N ) )
        return vector[index + ( N - distance )];
    // Valid left neighbour
    return vector[index - distance];
}
CUDA_DEVICE CUDA_INLINE complex_number right_neighbour_periodic( complex_number* vector, int index, const int row, const int col, const int distance, const int N ) {
    // Desired Index is in the next row
    if ( !is_valid_index( row, col + distance, N ) )
        return vector[index - ( N - distance )];
    // Valid right neighbour
    return vector[index + distance];
}


CUDA_DEVICE void scalar( complex_number& regular, complex_number* __restrict__ vector, int index, const int row, const int col, const int N, const bool periodic = false );

CUDA_DEVICE void tetm_plus( complex_number& regular, complex_number& cross, complex_number* __restrict__ vector, int index, const int row, const int col, const int N, const bool periodic = false );

CUDA_DEVICE void tetm_minus( complex_number& regular, complex_number& cross, complex_number* __restrict__ vector, int index, const int row, const int col, const int N, const bool periodic = false );

} // namespace PC3::Hamilton