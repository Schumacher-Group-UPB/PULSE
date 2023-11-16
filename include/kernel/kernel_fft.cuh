#pragma once
#include "cuda/cuda_complex.cuh"
#include "cuda/cuda_macro.cuh"

CUDA_GLOBAL void kernel_make_fft_visible( complex_number* input, complex_number* output, const unsigned int N );

CUDA_GLOBAL void fft_shift_2D( complex_number* data, const unsigned int N );

CUDA_GLOBAL void kernel_mask_fft( complex_number* data, real_number* mask, const unsigned int N );