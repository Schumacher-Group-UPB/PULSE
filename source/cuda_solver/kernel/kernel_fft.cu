#include "cuda/typedef.cuh"
#include "kernel/kernel_index_overwrite.cuh"
#include "kernel/kernel_fft.cuh"

PULSE_GLOBAL void PC3::Kernel::kernel_make_fft_visible( int i, Type::complex* input, Type::complex* output, const unsigned int N ) {
    GET_THREAD_INDEX( i, N );
    
    const auto val = input[i];
    output[i] = Type::complex( std::log( CUDA::real(val) * CUDA::real(val) + CUDA::imag(val) * PC3::CUDA::imag(val) ), 0 );
}

PULSE_GLOBAL void PC3::Kernel::fft_shift_2D( int i, Type::complex* data, const unsigned int N_x, const unsigned int N_y ) {
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

PULSE_GLOBAL void PC3::Kernel::kernel_mask_fft( int i, Type::complex* data, Type::real* mask, const unsigned int N ) {
    GET_THREAD_INDEX( i, N );

    data[i] = data[i] / Type::real(N) * mask[i];
}