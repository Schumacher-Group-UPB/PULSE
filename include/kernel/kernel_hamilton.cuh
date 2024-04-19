#pragma once
#include "cuda/cuda_complex.cuh"

namespace PC3::Hamilton {

CUDA_DEVICE CUDA_INLINE bool is_valid_index( const int row, const int col, const int N_x, const int N_y ) {
    return row >= 0 && row < N_y && col >= 0 && col < N_x;
}

/**
 * Dirichlet boundary conditions
 * For Dirichlet boundary conditions, the derivative is zero at the boundary.
 * Hence, when we leave the main grid, we simply return zero.
*/

CUDA_DEVICE CUDA_INLINE complex_number upper_neighbour( complex_number* vector, int index, const int row, const int col, const int distance, const int N_x, const int N_y ) {
    // We are alread in the most upper row
    if ( !is_valid_index( row - distance, col, N_x, N_y ) )
        return { 0.0, 0.0 };
    // Valid upper neighbour
    return vector[index - N_x * distance];
}
CUDA_DEVICE CUDA_INLINE complex_number lower_neighbour( complex_number* vector, int index, const int row, const int col, const int distance, const int N_x, const int N_y ) {
    // We are already in the most lower row
    if ( !is_valid_index( row + distance, col, N_x, N_y ) )
        return { 0.0, 0.0 };
    // Valid lower neighbour
    return vector[index + N_x * distance];
}
CUDA_DEVICE CUDA_INLINE complex_number left_neighbour( complex_number* vector, int index, const int row, const int col, const int distance, const int N_x, const int N_y ) {
    // Desired Index is in the previous row
    if ( !is_valid_index( row, col - distance, N_x, N_y ) )
        return { 0.0, 0.0 };
    // Valid left neighbour
    return vector[index - distance];
}
CUDA_DEVICE CUDA_INLINE complex_number right_neighbour( complex_number* vector, int index, const int row, const int col, const int distance, const int N_x, const int N_y ) {
    // Desired Index is in the next row
    if ( !is_valid_index( row, col + distance, N_x, N_y ) )
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

CUDA_DEVICE CUDA_INLINE complex_number upper_neighbour_periodic( complex_number* vector, int index, const int row, const int col, const int distance, const int N_x, const int N_y ) {
    // We are alread in the most upper row
    if ( !is_valid_index( row - distance, col, N_x, N_y ) )
        return vector[index - N_x * distance + N_x*N_y];
    // Valid upper neighbour
    return vector[index - N_x * distance];
}
CUDA_DEVICE CUDA_INLINE complex_number lower_neighbour_periodic( complex_number* vector, int index, const int row, const int col, const int distance, const int N_x, const int N_y ) {
    // We are already in the most lower row
    if ( !is_valid_index( row + distance, col, N_x, N_y ) )
        return vector[index + N_x * distance - N_x*N_y];
    // Valid lower neighbour
    return vector[index + N_x * distance];
}
CUDA_DEVICE CUDA_INLINE complex_number left_neighbour_periodic( complex_number* vector, int index, const int row, const int col, const int distance, const int N_x, const int N_y ) {
    // Desired Index is in the previous row
    if ( !is_valid_index( row, col - distance, N_x, N_y ) )
        return vector[index - distance + N_x ];
    // Valid left neighbour
    return vector[index - distance];
}
CUDA_DEVICE CUDA_INLINE complex_number right_neighbour_periodic( complex_number* vector, int index, const int row, const int col, const int distance, const int N_x, const int N_y ) {
    // Desired Index is in the next row
    if ( !is_valid_index( row, col + distance, N_x, N_y ) )
        return vector[index - N_x + distance ];
    // Valid right neighbour
    return vector[index + distance];
}

// Special top-right, top-left, bottom-right and bottom-left periodic boundary conditions that honour the periodicity of the grid
CUDA_DEVICE CUDA_INLINE complex_number upper_right_neighbour_periodic( complex_number* vector, int index, const int row, const int col, const int distance, const int N_x, const int N_y ) {
    if ( is_valid_index( row - distance, col + distance, N_x, N_y ) )
        // Valid upper neighbour
        return vector[index - N_x * distance + distance];
    // Check if we need to wrap around the rows
    if ( is_valid_index( row - distance, col, N_x, N_y ) )
        index += N_x*N_y - N_x * distance;
    // Check if we need to wrap around the columns
    if ( is_valid_index( row, col + distance, N_x, N_y ) )
        index += distance - N_x;
    return vector[index];
}
CUDA_DEVICE CUDA_INLINE complex_number upper_left_neighbour_periodic( complex_number* vector, int index, const int row, const int col, const int distance, const int N_x, const int N_y ) {
    if ( is_valid_index( row - distance, col - distance, N_x, N_y ) )
        // Valid upper neighbour
        return vector[index - N_x * distance - distance];
    // Check if we need to wrap around the rows
    if ( is_valid_index( row - distance, col, N_x, N_y ) )
        index += N_x*N_y - N_x * distance;
    // Check if we need to wrap around the columns
    if ( is_valid_index( row, col - distance, N_x, N_y ) )
        index -= distance + N_x;
    return vector[index];
}
CUDA_DEVICE CUDA_INLINE complex_number lower_right_neighbour_periodic( complex_number* vector, int index, const int row, const int col, const int distance, const int N_x, const int N_y ) {
    if ( is_valid_index( row + distance, col + distance, N_x, N_y ) )
        // Valid upper neighbour
        return vector[index + N_x * distance + distance];
    // Check if we need to wrap around the rows
    if ( is_valid_index( row + distance, col, N_x, N_y ) )
        index += N_x * distance - N_x*N_y;
    // Check if we need to wrap around the columns
    if ( is_valid_index( row, col + distance, N_x, N_y ) )
        index += distance + N_x;
    return vector[index];
}
CUDA_DEVICE CUDA_INLINE complex_number lower_left_neighbour_periodic( complex_number* vector, int index, const int row, const int col, const int distance, const int N_x, const int N_y ) {
    if ( is_valid_index( row + distance, col - distance, N_x, N_y ) )
        // Valid upper neighbour
        return vector[index + N_x * distance - distance];
    // Check if we need to wrap around the rows
    if ( is_valid_index( row + distance, col, N_x, N_y ) )
        index += N_x * distance - N_x*N_y;
    // Check if we need to wrap around the columns
    if ( is_valid_index( row, col - distance, N_x, N_y ) )
        index -= distance + N_x;
    return vector[index];
}

CUDA_DEVICE void scalar(complex_number& regular, complex_number* __restrict__ vector, int index, const int row, const int col, const int N_x, const int N_y, const real_number dx, const real_number dy, const bool periodic_x = false, const bool periodic_y = false);

CUDA_DEVICE void tetm_plus( complex_number& regular, complex_number& cross, complex_number* __restrict__ vector, int index, const int row, const int col, const int N_x, const int N_y, const real_number dx, const real_number dy, const bool periodic_x = false, const bool periodic_y = false );

CUDA_DEVICE void tetm_minus( complex_number& regular, complex_number& cross, complex_number* __restrict__ vector, int index, const int row, const int col, const int N_x, const int N_y, const real_number dx, const real_number dy, const bool periodic_x = false, const bool periodic_y = false );

} // namespace PC3::Hamilton