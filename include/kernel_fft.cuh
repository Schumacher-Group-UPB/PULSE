#pragma once
#include "cuda_complex_math.cuh"
#include "cuda_device_variables.cuh"
#include "cuda_macro.cuh"

__global__ void kernel_makeFFTVisible( complex_number* input, complex_number* output );

__global__ void fftshift_2D( complex_number* data_plus, complex_number* data_minus, int N_half );

__global__ void kernel_maskFFT( complex_number* data_plus, complex_number* data_minus, const real_number s, const real_number w, bool out_mask );