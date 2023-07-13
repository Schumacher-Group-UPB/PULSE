#pragma once
#include "cuda_complex_math.cuh"

__host__ __device__ void kernel_generateRingPhase( int s_N, double amp, int n, double w1, double w2, double xPos, double yPos, double p_xmax, double s_dx, bool normalize, cuDoubleComplex* buffer, bool reset = true );

__host__ __device__ void kernel_generateRingState( int s_N, double amp, double w1, double w2, double xPos, double yPos, double p_xmax, double s_dx, bool normalize, cuDoubleComplex* buffer, bool reset = true );