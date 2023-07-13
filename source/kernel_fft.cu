#include "cuda_macro.cuh"
#include <cuda.h>
#include "cuda_complex_math.cuh"
#include "kernel_fft.cuh"

__global__ void kernel_makeFFTVisible( cuDoubleComplex* input, cuDoubleComplex* output ) {
    int blockId = ( gridDim.x * blockIdx.y ) + blockIdx.x;
    int index = ( blockId * ( blockDim.x * blockDim.y ) ) + ( threadIdx.y * blockDim.x ) + threadIdx.x;
    if ( index < dev_s_N * dev_s_N ) {
        const auto val = input[index];
        output[index] = make_cuDoubleComplex( log( val.x * val.x + val.y * val.y ), 0 );
    }
}

__global__ void fftshift_2D( cuDoubleComplex* data_plus, cuDoubleComplex* data_minus, int N_half ) {
    int blockId = ( gridDim.x * blockIdx.y ) + blockIdx.x;
    int index = ( blockId * ( blockDim.x * blockDim.y ) ) + ( threadIdx.y * blockDim.x ) + threadIdx.x;
    if ( index >= dev_s_N * dev_s_N )
        return;
    // Current indices of upper left quadrant
    const int i = device_floor( index / dev_s_N );
    if ( i >= N_half )
        return;
    const int j = index % dev_s_N;
    if ( j >= N_half )
        return;
    // Swap upper left quadrant with lower right quadrant
    swap_symbol( data_plus[i * dev_s_N + j], data_plus[( i + N_half ) * dev_s_N + j + N_half] );
    swap_symbol( data_minus[i * dev_s_N + j], data_minus[( i + N_half ) * dev_s_N + j + N_half] );
    // Swap lower left quadrant with upper right quadrant
    swap_symbol( data_plus[i * dev_s_N + j + N_half], data_plus[( i + N_half ) * dev_s_N + j] );
    swap_symbol( data_minus[i * dev_s_N + j + N_half], data_minus[( i + N_half ) * dev_s_N + j] );
}

__global__ void kernel_maskFFT( cuDoubleComplex* data_plus, cuDoubleComplex* data_minus, const double s, const double w, bool out_mask ) {
    int blockId = ( gridDim.x * blockIdx.y ) + blockIdx.x;
    int index = ( blockId * ( blockDim.x * blockDim.y ) ) + ( threadIdx.y * blockDim.x ) + threadIdx.x;
    if ( index < dev_s_N * dev_s_N ) {
        const int i = device_floor( index / dev_s_N );
        const int j = index % dev_s_N;
        double ky = 2. * i / dev_s_N - 1.;
        double kx = 2. * j / dev_s_N - 1.;
        double mask = exp( -1.0 * pow( ( kx * kx + ky * ky ) / w / w, s ) );
        data_plus[index] = out_mask ? make_cuDoubleComplex( sqrt( mask ), 0 ) : data_plus[index] / dev_s_N / dev_s_N * mask;
        data_minus[index] = out_mask ? make_cuDoubleComplex( sqrt( mask ), 0 ) : data_minus[index] / dev_s_N / dev_s_N * mask;
    }
}