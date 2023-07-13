#pragma once
#include "cuda_complex_math.cuh"
#include "cuda_device_variables.cuh"
#include "cuda_macro.cuh"

__global__ void kernel_makeFFTVisible( cuDoubleComplex* input, cuDoubleComplex* output );

__global__ void fftshift_2D( cuDoubleComplex* data_plus, cuDoubleComplex* data_minus, int N_half );

__global__ void kernel_maskFFT( cuDoubleComplex* data_plus, cuDoubleComplex* data_minus, const double s, const double w, bool out_mask );