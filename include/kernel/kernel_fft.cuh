#pragma once
#include "cuda/typedef.cuh"
#include "cuda/cuda_macro.cuh"
#include "kernel/kernel_index_overwrite.cuh"

namespace PC3::Kernel {
    
    template<typename T>
    PULSE_GLOBAL void kernel_make_fft_visible( int i, T* input, T* output, const Type::uint32 N ) {
        GET_THREAD_INDEX( i, N );
        
        const auto val = input[i];
        output[i] = Type::complex( std::log( CUDA::real(val) * CUDA::real(val) + CUDA::imag(val) * PC3::CUDA::imag(val) ), 0 );
    }
    
    template<typename T>
    PULSE_GLOBAL void fft_shift_2D( int i, T* data, const Type::uint32 N_x, const Type::uint32 N_y ) {
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
    
    template<typename T, typename U>
    PULSE_GLOBAL void kernel_mask_fft( int i, T* data, U* mask, const Type::uint32 N ) {
        GET_THREAD_INDEX( i, N );
    
        data[i] = data[i] / Type::real(N) * mask[i];
    }

}