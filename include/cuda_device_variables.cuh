#pragma once
#include <cuda.h>
#include <iostream>
#include "cuda_macro.cuh"
#include <cufft.h>
#include "cuda_complex.cuh"

// Use -rdc=true when compiling with nvcc to allow for the "extern" keyword to work
extern CUDA_DEVICE real_number dev_s_dx;
extern CUDA_DEVICE real_number dev_p_g_r;
extern CUDA_DEVICE int dev_s_N;
extern CUDA_DEVICE real_number dev_p_m_eff;
extern CUDA_DEVICE real_number dev_p_gamma_c;
extern CUDA_DEVICE real_number dev_p_g_c;
extern CUDA_DEVICE real_number dev_p_g_pm;
extern CUDA_DEVICE real_number dev_p_gamma_r;
extern CUDA_DEVICE real_number dev_p_R;
extern CUDA_DEVICE real_number dev_p_delta_LT;
extern CUDA_DEVICE real_number dev_s_dt;
extern CUDA_DEVICE real_number dev_p_xmax;
extern CUDA_DEVICE real_number dev_one_over_h_bar_s;

extern CUDA_DEVICE int dev_n_pump;
extern CUDA_DEVICE int dev_n_pulse;

// Cached Device Variables
extern CUDA_DEVICE real_number dev_p_m_eff_scaled;
extern CUDA_DEVICE real_number dev_p_delta_LT_scaled;
extern CUDA_DEVICE complex_number dev_pgr_plus_pR;

// Pump and Pulse device arrays
extern real_number* dev_pump_amp;
extern real_number* dev_pump_width;
extern real_number* dev_pump_X;
extern real_number* dev_pump_Y;
extern int* dev_pump_pol;
extern real_number* dev_pulse_t0;
extern real_number* dev_pulse_amp;
extern real_number* dev_pulse_freq;
extern real_number* dev_pulse_sigma;
extern int* dev_pulse_m;
extern int* dev_pulse_pol;
extern real_number* dev_pulse_width;
extern real_number* dev_pulse_X;
extern real_number* dev_pulse_Y;

// Device Pointers to input and output arrays
extern complex_number* dev_current_Psi_Plus;
extern complex_number* dev_current_Psi_Minus;
extern complex_number* dev_current_n_Plus;
extern complex_number* dev_current_n_Minus;
extern complex_number* dev_next_Psi_Plus;
extern complex_number* dev_next_Psi_Minus;
extern complex_number* dev_next_n_Plus;
extern complex_number* dev_next_n_Minus;

// Device Pointers to k1, k2, k3, k4, (k5, k6, k7) arrays
extern complex_number* dev_k1_Psi_Plus;
extern complex_number* dev_k1_Psi_Minus;
extern complex_number* dev_k1_n_Plus;
extern complex_number* dev_k1_n_Minus;
extern complex_number* dev_k2_Psi_Plus;
extern complex_number* dev_k2_Psi_Minus;
extern complex_number* dev_k2_n_Plus;
extern complex_number* dev_k2_n_Minus;
extern complex_number* dev_k3_Psi_Plus;
extern complex_number* dev_k3_Psi_Minus;
extern complex_number* dev_k3_n_Plus;
extern complex_number* dev_k3_n_Minus;
extern complex_number* dev_k4_Psi_Plus;
extern complex_number* dev_k4_Psi_Minus;
extern complex_number* dev_k4_n_Plus;
extern complex_number* dev_k4_n_Minus;
extern complex_number* dev_k5_Psi_Plus;
extern complex_number* dev_k5_Psi_Minus;
extern complex_number* dev_k5_n_Plus;
extern complex_number* dev_k5_n_Minus;
extern complex_number* dev_k6_Psi_Plus;
extern complex_number* dev_k6_Psi_Minus;
extern complex_number* dev_k6_n_Plus;
extern complex_number* dev_k6_n_Minus;
extern complex_number* dev_k7_Psi_Plus;
extern complex_number* dev_k7_Psi_Minus;
extern complex_number* dev_k7_n_Plus;
extern complex_number* dev_k7_n_Minus;

extern real_number* dev_rk_error;

// Device Pointers to pulse and pulse2
extern complex_number* dev_fft_plus;
extern complex_number* dev_fft_minus;

// CUDA FFT Plan
extern cufftHandle plan;

/**
 * Initialize device variables from host variables
 */

void initializeDeviceVariables( const real_number s_dx, const real_number s_dt, const real_number p_g_r, const int s_N, const real_number p_m_eff, const real_number p_gamma_c, const real_number p_g_c, const real_number p_g_pm, const real_number p_gamma_r, const real_number p_R, const real_number p_delta_LT, const real_number p_xmax, const real_number h_bar_s );

void initializePumpVariables( real_number* pump_amp, real_number* pump_width, real_number* pump_X, real_number* pump_Y, int* pump_pol, const int size );

void initializePulseVariables( real_number* pulse_t0, real_number* pulse_amp, real_number* pulse_freq, real_number* pulse_sigma, int* pulse_m, int* pulse_pol, real_number* pulse_width, real_number* pulse_X, real_number* pulse_Y, const int size );

/**
 * Initialize device arrays to zero
 */
void initializeDeviceArrays( const int s_N );

void setDeviceArrays( complex_number* psi_plus, complex_number* psi_minus, complex_number* n_plus, complex_number* n_minus, const int s_N );

void getDeviceArrays( complex_number* psi_plus, complex_number* psi_minus, complex_number* n_plus, complex_number* n_minus, complex_number* fft_plus, complex_number* fft_minus, const int s_N );

void getDeviceArraySlice(complex_number* buffer_in, complex_number* buffer_out, const int start, const int length);

void freeDeviceArrays();