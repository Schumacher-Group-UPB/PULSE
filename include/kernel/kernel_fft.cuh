#pragma once
#include "cuda/typedef.cuh"
#include "cuda/cuda_macro.cuh"

namespace PC3::Kernel {

PULSE_GLOBAL void kernel_make_fft_visible( int i, Type::complex* input, Type::complex* output, const Type::uint N );

PULSE_GLOBAL void fft_shift_2D( int i, Type::complex* data, const Type::uint N_x, const Type::uint N_y );

PULSE_GLOBAL void kernel_mask_fft( int i, Type::complex* data, Type::real* mask, const Type::uint N );

}