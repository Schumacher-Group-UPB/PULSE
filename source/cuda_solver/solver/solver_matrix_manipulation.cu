#include <fstream>
#include <iostream>
#include <vector>
#include <complex>
#include <omp.h>
#include <memory>
#include <algorithm>
#include <random>
#include <ranges>
#include "cuda/typedef.cuh"
#include "misc/helperfunctions.hpp"

/**
 * @brief Calculates the abs2 of a buffer of (complex) numbers
 * @param z The buffer to calculate the abs2 of.
 * @param buffer The buffer to save the result to.
 * @param size The size of the buffer.
 */
void PC3::CUDA::cwiseAbs2( Type::complex* z, Type::real* buffer, int size ) {
#pragma omp parallel for
    for ( int i = 0; i < size; i++ )
        buffer[i] = real( z[i] ) * real( z[i] ) + imag( z[i] ) * imag( z[i] );
}

void PC3::CUDA::cwiseAbs2( Type::real* z, Type::real* buffer, int size ) {
#pragma omp parallel for
    for ( int i = 0; i < size; i++ )
        buffer[i] = z[i] * z[i];
}

void PC3::CUDA::normalize( Type::real* buffer, int size, Type::real min, Type::real max, bool device_pointer ) {
    if ( min == max )
        auto [min, max] = minmax( buffer, size, device_pointer );
#pragma omp parallel for
    for ( int i = 0; i < size; i++ )
        buffer[i] = ( buffer[i] - min ) / ( max - min );
}

void PC3::CUDA::angle( Type::complex* z, Type::real* buffer, int size ) {
#pragma omp parallel for
    for ( int i = 0; i < size; i++ )
        buffer[i] = CUDA::arg(z[i]);//CUDA::atan2( PC3::CUDA::imag( z[i] ), PC3::CUDA::real( z[i] ) );
}
