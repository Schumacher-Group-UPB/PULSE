
#include "kernel/kernel_hamilton.cuh"

PULSE_DEVICE void PC3::Kernel::Hamilton::scalar( Type::complex& regular, Type::complex* __restrict__ vector, int index, const int row, const int col, const int N_x, const int N_y, const Type::real dx, const Type::real dy, const bool periodic_x, const bool periodic_y ) {
    Type::complex upper, lower, left, right;
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
    const Type::complex vec = vector[index];
    regular = (upper + lower - Type::real(2.0) * vec)/dy/dy + (left + right  - Type::real(2.0) * vec)/dx/dx;
}

PULSE_DEVICE void PC3::Kernel::Hamilton::tetm_plus( Type::complex& regular, Type::complex& cross, Type::complex* __restrict__ vector, int index, const int row, const int col, const int N_x, const int N_y, const Type::real dx, const Type::real dy, const bool periodic_x, const bool periodic_y ) {
    Type::complex upper, lower, left, right;
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
        cross = (left + right)/dx/dx - (upper + lower)/dy/dy + Type::complex(0.0,-0.5)/dx/dy * ( -lower_right_neighbour_periodic( vector, index, row, col, 1, N_x, N_y ) + lower_left_neighbour_periodic( vector, index, row, col, 1, N_x, N_y ) + upper_right_neighbour_periodic( vector, index, row, col, 1, N_x, N_y ) - upper_left_neighbour_periodic( vector, index, row, col, 1, N_x, N_y ) );
    } else {
        left = left_neighbour( vector, index, row, col, 1, N_x, N_y );
        right = right_neighbour( vector, index, row, col, 1, N_x, N_y );
        cross = (left + right)/dx/dx - (upper + lower)/dy/dy + Type::complex(0.0,-0.5)/dx/dy * ( -right_neighbour( vector, index - N_x, row - 1, col, 1, N_x, N_y ) + left_neighbour( vector, index - N_x, row - 1, col, 1, N_x, N_y ) + right_neighbour( vector, index + N_x, row + 1, col, 1, N_x, N_y ) - left_neighbour( vector, index + N_x, row + 1, col, 1, N_x, N_y ) );
    }
    const Type::complex vec = vector[index];
    regular = (upper + lower - Type::real(2.0) * vec)/dy/dy + (left + right  - Type::real(2.0) * vec)/dx/dx;
}

PULSE_DEVICE void PC3::Kernel::Hamilton::tetm_minus( Type::complex& regular, Type::complex& cross, Type::complex* __restrict__ vector, int index, const int row, const int col, const int N_x, const int N_y, const Type::real dx, const Type::real dy, const bool periodic_x, const bool periodic_y ) {
    Type::complex upper, lower, left, right;
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
        cross = (left + right)/dx/dx - (upper + lower)/dy/dy + Type::complex(0.0,0.5)/dx/dy * ( -lower_right_neighbour_periodic( vector, index, row, col, 1, N_x, N_y ) + lower_left_neighbour_periodic( vector, index, row, col, 1, N_x, N_y ) + upper_right_neighbour_periodic( vector, index, row, col, 1, N_x, N_y ) - upper_left_neighbour_periodic( vector, index, row, col, 1, N_x, N_y ) );
    } else {
        left = left_neighbour( vector, index, row, col, 1, N_x, N_y );
        right = right_neighbour( vector, index, row, col, 1, N_x, N_y );
        cross = (left + right)/dx/dx - (upper + lower)/dy/dy + Type::complex(0.0,0.5)/dx/dy * ( -right_neighbour( vector, index - N_x, row - 1, col, 1, N_x, N_y ) + left_neighbour( vector, index - N_x, row - 1, col, 1, N_x, N_y ) + right_neighbour( vector, index + N_x, row + 1, col, 1, N_x, N_y ) - left_neighbour( vector, index + N_x, row + 1, col, 1, N_x, N_y ) );
    }
    const Type::complex vec = vector[index];
    regular = (upper + lower - Type::real(2.0) * vec)/dy/dy + (left + right  - Type::real(2.0) * vec)/dx/dx;
}