#pragma once
#include <cuda.h>
#include <iostream>
#include "cuda_macro.cuh"
#include <cufft.h>

// Use -rdc=true when compiling with nvcc to allow for the "extern" keyword to work
extern __device__ double dev_s_dx;
extern __device__ double dev_p_g_r;
extern __device__ int dev_s_N;
extern __device__ double dev_p_m_eff;
extern __device__ double dev_p_gamma_c;
extern __device__ double dev_p_g_c;
extern __device__ double dev_p_g_pm;
extern __device__ double dev_p_gamma_r;
extern __device__ double dev_p_R;
extern __device__ double dev_p_delta_LT;
extern __device__ double dev_s_dt;
extern __device__ double dev_p_xmax;
extern __device__ double dev_one_over_h_bar_s;

extern __device__ int dev_n_pump;
extern __device__ int dev_n_pulse;

// Cached Device Variables
extern __device__ double dev_p_m_eff_scaled;
extern __device__ double dev_p_delta_LT_scaled;
extern __device__ cuDoubleComplex dev_pgr_plus_pR;

// Pump and Pulse device arrays
extern double* dev_pump_amp;
extern double* dev_pump_width;
extern double* dev_pump_X;
extern double* dev_pump_Y;
extern int* dev_pump_pol;
extern double* dev_pulse_t0;
extern double* dev_pulse_amp;
extern double* dev_pulse_freq;
extern double* dev_pulse_sigma;
extern int* dev_pulse_m;
extern int* dev_pulse_pol;
extern double* dev_pulse_width;
extern double* dev_pulse_X;
extern double* dev_pulse_Y;

// Device Pointers to input and output arrays
extern cuDoubleComplex* dev_current_Psi_Plus;
extern cuDoubleComplex* dev_current_Psi_Minus;
extern cuDoubleComplex* dev_current_n_Plus;
extern cuDoubleComplex* dev_current_n_Minus;
extern cuDoubleComplex* dev_next_Psi_Plus;
extern cuDoubleComplex* dev_next_Psi_Minus;
extern cuDoubleComplex* dev_next_n_Plus;
extern cuDoubleComplex* dev_next_n_Minus;

// Device Pointers to k1, k2, k3, k4, (k5, k6, k7) arrays
extern cuDoubleComplex* dev_k1_Psi_Plus;
extern cuDoubleComplex* dev_k1_Psi_Minus;
extern cuDoubleComplex* dev_k1_n_Plus;
extern cuDoubleComplex* dev_k1_n_Minus;
extern cuDoubleComplex* dev_k2_Psi_Plus;
extern cuDoubleComplex* dev_k2_Psi_Minus;
extern cuDoubleComplex* dev_k2_n_Plus;
extern cuDoubleComplex* dev_k2_n_Minus;
extern cuDoubleComplex* dev_k3_Psi_Plus;
extern cuDoubleComplex* dev_k3_Psi_Minus;
extern cuDoubleComplex* dev_k3_n_Plus;
extern cuDoubleComplex* dev_k3_n_Minus;
extern cuDoubleComplex* dev_k4_Psi_Plus;
extern cuDoubleComplex* dev_k4_Psi_Minus;
extern cuDoubleComplex* dev_k4_n_Plus;
extern cuDoubleComplex* dev_k4_n_Minus;
extern cuDoubleComplex* dev_k5_Psi_Plus;
extern cuDoubleComplex* dev_k5_Psi_Minus;
extern cuDoubleComplex* dev_k5_n_Plus;
extern cuDoubleComplex* dev_k5_n_Minus;
extern cuDoubleComplex* dev_k6_Psi_Plus;
extern cuDoubleComplex* dev_k6_Psi_Minus;
extern cuDoubleComplex* dev_k6_n_Plus;
extern cuDoubleComplex* dev_k6_n_Minus;
extern cuDoubleComplex* dev_k7_Psi_Plus;
extern cuDoubleComplex* dev_k7_Psi_Minus;
extern cuDoubleComplex* dev_k7_n_Plus;
extern cuDoubleComplex* dev_k7_n_Minus;

extern double* dev_rk_error;

// Device Pointers to pulse and pulse2
extern cuDoubleComplex* dev_fft_plus;
extern cuDoubleComplex* dev_fft_minus;

// CUDA FFT Plan
extern cufftHandle plan;

/**
 * Initialize device variables from host variables
 */

void initializeDeviceVariables( const double s_dx, const double s_dt, const double p_g_r, const int s_N, const double p_m_eff, const double p_gamma_c, const double p_g_c, const double p_g_pm, const double p_gamma_r, const double p_R, const double p_delta_LT, const double p_xmax, const double h_bar_s );

void initializePumpVariables( double* pump_amp, double* pump_width, double* pump_X, double* pump_Y, int* pump_pol, const int size );

void initializePulseVariables( double* pulse_t0, double* pulse_amp, double* pulse_freq, double* pulse_sigma, int* pulse_m, int* pulse_pol, double* pulse_width, double* pulse_X, double* pulse_Y, const int size );

/**
 * Initialize device arrays to zero
 */
void initializeDeviceArrays( const int s_N );

void setDeviceArrays( Scalar* psi_plus, Scalar* psi_minus, Scalar* n_plus, Scalar* n_minus, const int s_N );

void getDeviceArrays( Scalar* psi_plus, Scalar* psi_minus, Scalar* n_plus, Scalar* n_minus, Scalar* fft_plus, Scalar* fft_minus, const int s_N );

void getDeviceArraySlice(Scalar* buffer_in, Scalar* buffer_out, const int start, const int length);

void freeDeviceArrays();