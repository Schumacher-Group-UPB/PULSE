
#include "kernel/kernel_hamilton.cuh"

CUDA_DEVICE void PC3::Hamilton::scalar( complex_number& regular, complex_number* __restrict__ vector, int index, const int row, const int col, const int N_x, const int N_y, const real_number dx, const real_number dy, const bool periodic_x, const bool periodic_y ) {
    complex_number upper, lower, left, right;
    if (periodic_x) {
        left = left_neighbour_periodic( vector, index, row, col, 1, N_x, N_y );
        right = right_neighbour_periodic( vector, index, row, col, 1, N_x, N_y );
    } else {
        left = left_neighbour( vector, index, row, col, 1, N_x, N_y );
        right = right_neighbour( vector, index, row, col, 1, N_x, N_y );
    }
    if (periodic_y) {
        upper = upper_neighbour_periodic( vector, index, row, col, 1, N_x, N_y );
        lower = lower_neighbour_periodic( vector, index, row, col, 1, N_x, N_y );
    } else {
        upper = upper_neighbour( vector, index, row, col, 1, N_x, N_y );
        lower = lower_neighbour( vector, index, row, col, 1, N_x, N_y );
    }
    const complex_number vec = vector[index];
    regular = (upper + lower - 2.0 * vec)/dy/dy + (left + right  - 2.0 * vec)/dx/dx;
}

CUDA_DEVICE void PC3::Hamilton::tetm_plus( complex_number& regular, complex_number& cross, complex_number* __restrict__ vector, int index, const int row, const int col, const int N_x, const int N_y, const real_number dx, const real_number dy, const bool periodic_x, const bool periodic_y ) {
    complex_number upper, lower, left, right;
    if (periodic_y) {
        upper = upper_neighbour_periodic( vector, index, row, col, 1, N_x, N_y );
        lower = lower_neighbour_periodic( vector, index, row, col, 1, N_x, N_y );
    } else {
        upper = upper_neighbour( vector, index, row, col, 1, N_x, N_y );
        lower = lower_neighbour( vector, index, row, col, 1, N_x, N_y );
    }
    if (periodic_x) {
        left = left_neighbour_periodic( vector, index, row, col, 1, N_x, N_y );
        right = right_neighbour_periodic( vector, index, row, col, 1, N_x, N_y );
        cross = (left + right)/dx/dx - (upper + lower)/dy/dy + complex_number(0.0,-0.5)/dx/dy * ( -lower_right_neighbour_periodic( vector, index, row, col, 1, N_x, N_y ) + lower_left_neighbour_periodic( vector, index, row, col, 1, N_x, N_y ) + upper_right_neighbour_periodic( vector, index, row, col, 1, N_x, N_y ) - upper_left_neighbour_periodic( vector, index, row, col, 1, N_x, N_y ) );
    } else {
        left = left_neighbour( vector, index, row, col, 1, N_x, N_y );
        right = right_neighbour( vector, index, row, col, 1, N_x, N_y );
        cross = (left + right)/dx/dx - (upper + lower)/dy/dy + complex_number(0.0,-0.5)/dx/dy * ( -right_neighbour( vector, index - N_x, row - 1, col, 1, N_x, N_y ) + left_neighbour( vector, index - N_x, row - 1, col, 1, N_x, N_y ) + right_neighbour( vector, index + N_x, row + 1, col, 1, N_x, N_y ) - left_neighbour( vector, index + N_x, row + 1, col, 1, N_x, N_y ) );
    }
    const complex_number vec = vector[index];
    regular = (upper + lower - 2.0 * vec)/dy/dy + (left + right  - 2.0 * vec)/dx/dx;
}

CUDA_DEVICE void PC3::Hamilton::tetm_minus( complex_number& regular, complex_number& cross, complex_number* __restrict__ vector, int index, const int row, const int col, const int N_x, const int N_y, const real_number dx, const real_number dy, const bool periodic_x, const bool periodic_y ) {
    complex_number upper, lower, left, right;
    if (periodic_y) {
        upper = upper_neighbour_periodic( vector, index, row, col, 1, N_x, N_y );
        lower = lower_neighbour_periodic( vector, index, row, col, 1, N_x, N_y );
    } else {
        upper = upper_neighbour( vector, index, row, col, 1, N_x, N_y );
        lower = lower_neighbour( vector, index, row, col, 1, N_x, N_y );
    }
    if (periodic_x) {
        left = left_neighbour_periodic( vector, index, row, col, 1, N_x, N_y );
        right = right_neighbour_periodic( vector, index, row, col, 1, N_x, N_y );
        cross = (left + right)/dx/dx - (upper + lower)/dy/dy + complex_number(0.0,0.5)/dx/dy * ( -lower_right_neighbour_periodic( vector, index, row, col, 1, N_x, N_y ) + lower_left_neighbour_periodic( vector, index, row, col, 1, N_x, N_y ) + upper_right_neighbour_periodic( vector, index, row, col, 1, N_x, N_y ) - upper_left_neighbour_periodic( vector, index, row, col, 1, N_x, N_y ) );
    } else {
        left = left_neighbour( vector, index, row, col, 1, N_x, N_y );
        right = right_neighbour( vector, index, row, col, 1, N_x, N_y );
        cross = (left + right)/dx/dx - (upper + lower)/dy/dy + complex_number(0.0,0.5)/dx/dy * ( -right_neighbour( vector, index - N_x, row - 1, col, 1, N_x, N_y ) + left_neighbour( vector, index - N_x, row - 1, col, 1, N_x, N_y ) + right_neighbour( vector, index + N_x, row + 1, col, 1, N_x, N_y ) - left_neighbour( vector, index + N_x, row + 1, col, 1, N_x, N_y ) );
    }
    const complex_number vec = vector[index];
    regular = (upper + lower - 2.0 * vec)/dy/dy + (left + right  - 2.0 * vec)/dx/dx;
}