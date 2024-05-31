#pragma once
#include "cuda/typedef.cuh"

namespace PC3::Kernel::Hamilton {

PULSE_DEVICE PULSE_INLINE bool is_valid_index( const int row, const int col, const int N_x, const int N_y ) {
    return row >= 0 && row < N_y && col >= 0 && col < N_x;
}

/**
 * Dirichlet boundary conditions
 * For Dirichlet boundary conditions, the derivative is zero at the boundary.
 * Hence, when we leave the main grid, we simply return zero.
*/

PULSE_DEVICE PULSE_INLINE Type::complex upper_neighbour( Type::complex* vector, int index, const int row, const int col, const int distance, const int N_x, const int N_y ) {
    // We are alread in the most upper row
    if ( !is_valid_index( row - distance, col, N_x, N_y ) )
        return { 0.0, 0.0 };
    // Valid upper neighbour
    return vector[index - N_x * distance];
}
PULSE_DEVICE PULSE_INLINE Type::complex lower_neighbour( Type::complex* vector, int index, const int row, const int col, const int distance, const int N_x, const int N_y ) {
    // We are already in the most lower row
    if ( !is_valid_index( row + distance, col, N_x, N_y ) )
        return { 0.0, 0.0 };
    // Valid lower neighbour
    return vector[index + N_x * distance];
}
PULSE_DEVICE PULSE_INLINE Type::complex left_neighbour( Type::complex* vector, int index, const int row, const int col, const int distance, const int N_x, const int N_y ) {
    // Desired Index is in the previous row
    if ( !is_valid_index( row, col - distance, N_x, N_y ) )
        return { 0.0, 0.0 };
    // Valid left neighbour
    return vector[index - distance];
}
PULSE_DEVICE PULSE_INLINE Type::complex right_neighbour( Type::complex* vector, int index, const int row, const int col, const int distance, const int N_x, const int N_y ) {
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

PULSE_DEVICE PULSE_INLINE Type::complex upper_neighbour_periodic( Type::complex* vector, int index, const int row, const int col, const int distance, const int N_x, const int N_y ) {
    // We are alread in the most upper row
    if ( !is_valid_index( row - distance, col, N_x, N_y ) )
        return vector[index - N_x * distance + N_x*N_y];
    // Valid upper neighbour
    return vector[index - N_x * distance];
}
PULSE_DEVICE PULSE_INLINE Type::complex lower_neighbour_periodic( Type::complex* vector, int index, const int row, const int col, const int distance, const int N_x, const int N_y ) {
    // We are already in the most lower row
    if ( !is_valid_index( row + distance, col, N_x, N_y ) )
        return vector[index + N_x * distance - N_x*N_y];
    // Valid lower neighbour
    return vector[index + N_x * distance];
}
PULSE_DEVICE PULSE_INLINE Type::complex left_neighbour_periodic( Type::complex* vector, int index, const int row, const int col, const int distance, const int N_x, const int N_y ) {
    // Desired Index is in the previous row
    if ( !is_valid_index( row, col - distance, N_x, N_y ) )
        return vector[index - distance + N_x ];
    // Valid left neighbour
    return vector[index - distance];
}
PULSE_DEVICE PULSE_INLINE Type::complex right_neighbour_periodic( Type::complex* vector, int index, const int row, const int col, const int distance, const int N_x, const int N_y ) {
    // Desired Index is in the next row
    if ( !is_valid_index( row, col + distance, N_x, N_y ) )
        return vector[index - N_x + distance ];
    // Valid right neighbour
    return vector[index + distance];
}

// Special top-right, top-left, bottom-right and bottom-left periodic boundary conditions that honour the periodicity of the grid
PULSE_DEVICE PULSE_INLINE Type::complex upper_right_neighbour_periodic( Type::complex* vector, int index, const int row, const int col, const int distance, const int N_x, const int N_y ) {
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
PULSE_DEVICE PULSE_INLINE Type::complex upper_left_neighbour_periodic( Type::complex* vector, int index, const int row, const int col, const int distance, const int N_x, const int N_y ) {
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
PULSE_DEVICE PULSE_INLINE Type::complex lower_right_neighbour_periodic( Type::complex* vector, int index, const int row, const int col, const int distance, const int N_x, const int N_y ) {
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
PULSE_DEVICE PULSE_INLINE Type::complex lower_left_neighbour_periodic( Type::complex* vector, int index, const int row, const int col, const int distance, const int N_x, const int N_y ) {
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

PULSE_DEVICE void scalar(Type::complex& regular, Type::complex* __restrict__ vector, int index, const int row, const int col, const int N_x, const int N_y, const Type::real dx, const Type::real dy, const bool periodic_x = false, const bool periodic_y = false);

PULSE_DEVICE void tetm_plus( Type::complex& regular, Type::complex& cross, Type::complex* __restrict__ vector, int index, const int row, const int col, const int N_x, const int N_y, const Type::real dx, const Type::real dy, const bool periodic_x = false, const bool periodic_y = false );

PULSE_DEVICE void tetm_minus( Type::complex& regular, Type::complex& cross, Type::complex* __restrict__ vector, int index, const int row, const int col, const int N_x, const int N_y, const Type::real dx, const Type::real dy, const bool periodic_x = false, const bool periodic_y = false );

} // namespace PC3::Kernel::Hamilton