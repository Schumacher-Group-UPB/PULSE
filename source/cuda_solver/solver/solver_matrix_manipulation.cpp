#include <fstream>
#include <iostream>
#include <vector>
#include <complex>
#include <omp.h>
#include <memory>
#include <algorithm>
#include <random>
#include <ranges>
#include "cuda/cuda_complex.cuh"
#include "misc/helperfunctions.hpp"

// To cast the "pointers-to-device-memory" to actual device pointers for thrust
#ifndef USECPU
#    include <thrust/device_ptr.h>
#endif

void PC3::CUDA::normalize( real_number* buffer, int size, real_number min, real_number max, bool device_pointer ) {
    if ( min == max )
        auto [min, max] = minmax( buffer, size, device_pointer );
#pragma omp parallel for
    for ( int i = 0; i < size; i++ )
        buffer[i] = ( buffer[i] - min ) / ( max - min );
}

void PC3::CUDA::angle( complex_number* z, real_number* buffer, int size ) {
#pragma omp parallel for
    for ( int i = 0; i < size; i++ )
        buffer[i] = std::atan2( PC3::CUDA::imag( z[i] ), PC3::CUDA::real( z[i] ) );
}
