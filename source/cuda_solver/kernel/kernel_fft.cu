#include "cuda/cuda_macro.cuh"
#include "kernel/kernel_fft.cuh"
#ifndef USECPU
#include <cuda.h>
#endif

// These functions are not rquired for the CPU Benchmark Version
#ifndef USECPU

CUDA_GLOBAL void kernel_make_fft_visible( complex_number* input, complex_number* output, const unsigned int N ) {
    int blockId = ( gridDim.x * blockIdx.y ) + blockIdx.x;
    int index = ( blockId * ( blockDim.x * blockDim.y ) ) + ( threadIdx.y * blockDim.x ) + threadIdx.x;
    if ( index >= N * N )
        return;
    const auto val = input[index];
    output[index] = { log( val.x * val.x + val.y * val.y ), 0 };
}

CUDA_GLOBAL void fft_shift_2D( complex_number* data, const unsigned int N ) {
    int blockId = ( gridDim.x * blockIdx.y ) + blockIdx.x;
    int index = ( blockId * ( blockDim.x * blockDim.y ) ) + ( threadIdx.y * blockDim.x ) + threadIdx.x;
    if ( index >= N * N )
        return;
    const auto N_half = N / 2;
    // Current indices of upper left quadrant
    const int i = device_floor( index / N );
    if ( i >= N_half )
        return;
    const int j = index % N;
    if ( j >= N_half )
        return;
    // Swap upper left quadrant with lower right quadrant
    swap_symbol( data[i * N + j], data[( i + N_half ) * N + j + N_half] );
    // Swap lower left quadrant with upper right quadrant
    swap_symbol( data[i * N + j + N_half], data[( i + N_half ) * N + j] );
}

CUDA_GLOBAL void kernel_mask_fft( complex_number* data, real_number* mask, const unsigned int N ) {
    int blockId = ( gridDim.x * blockIdx.y ) + blockIdx.x;
    int index = ( blockId * ( blockDim.x * blockDim.y ) ) + ( threadIdx.y * blockDim.x ) + threadIdx.x;
    if ( index >= N * N )
        return;
    data[index] = data[index] / N / N * mask[index];
}

#else

CUDA_GLOBAL void kernel_make_fft_visible( complex_number* input, complex_number* output, const unsigned int N ) {}
CUDA_GLOBAL void fft_shift_2D( complex_number* data, const unsigned int N ) {}
CUDA_GLOBAL void kernel_mask_fft( complex_number* data, real_number* mask, const unsigned int N ) {}

#endif