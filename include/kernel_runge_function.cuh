#pragma once
#include "cuda_complex_math.cuh"
#include "cuda_device_variables.cuh"
#include "kernel_hamilton.cuh"

__global__ void rungeFuncKernel( double t, cuDoubleComplex* __restrict__ in_Psi_Plus, cuDoubleComplex* __restrict__ in_Psi_Minus, cuDoubleComplex* __restrict__ in_n_Plus, cuDoubleComplex* __restrict__ in_n_Minus, cuDoubleComplex* __restrict__ k_Psi_Plus, cuDoubleComplex* __restrict__ k_Psi_Minus, cuDoubleComplex* __restrict__ k_n_Plus, cuDoubleComplex* __restrict__ k_n_Minus,
                                 /* Pump Parameters */ double* __restrict__ dev_pump_amp, double* __restrict__ dev_pump_width, double* __restrict__ dev_pump_X, double* __restrict__ dev_pump_Y, int* __restrict__ dev_pump_pol,
                                 /* Pulse Parameters */ double* __restrict__ dev_pulse_t0, double* __restrict__ dev_pulse_amp, double* __restrict__ dev_pulse_freq, double* __restrict__ dev_pulse_sigma, int* __restrict__ dev_pulse_m, int* __restrict__ dev_pulse_pol, double* __restrict__ dev_pulse_width, double* __restrict__ dev_pulse_X, double* __restrict__ dev_pulse_Y, bool evaluate_pulse );