#include "cuda/cuda_macro.cuh"
#include "cuda/cuda_complex.cuh"
#include "kernel/kernel_index_overwrite.cuh"
#include "kernel/kernel_fft.cuh"
#ifndef USECPU
#include <cuda.h>
#endif

CUDA_GLOBAL void kernel_make_fft_visible( int i, complex_number* input, complex_number* output, const unsigned int N ) {
    GET_THREAD_INDEX( i, N );
    
    const auto val = input[i];
    output[i] = { PC3::CUDA::log( PC3::CUDA::real(val) * PC3::CUDA::real(val) + PC3::CUDA::imag(val) * PC3::CUDA::imag(val) ), 0 };
}

CUDA_GLOBAL void fft_shift_2D( int i, complex_number* data, const unsigned int N_x, const unsigned int N_y ) {
    GET_THREAD_INDEX( i, N_x*N_y );

    // Current indices of upper left quadrant
    const int k = i / N_x ;
    if ( k >= N_y/2 )
        return;
    const int l = i % N_x;
    if ( l >= N_x/2 )
        return;

    // Swap upper left quadrant with lower right quadrant
    swap_symbol( data[k * N_x + l], data[( k + N_y/2 ) * N_x + l + N_x/2] );
    
    // Swap lower left quadrant with upper right quadrant
    swap_symbol( data[k * N_x + l + N_x/2], data[( k + N_y/2 ) * N_x + l] );
}

CUDA_GLOBAL void kernel_mask_fft( int i, complex_number* data, real_number* mask, const unsigned int N ) {
    GET_THREAD_INDEX( i, N );

    data[i] = data[i] / real_number(N) * mask[i];
}