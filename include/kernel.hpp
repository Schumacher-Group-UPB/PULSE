#pragma once
#include "cuda_complex.cuh"
#include "system.hpp"
#include "cuda_complex_math.cuh"
#include "kernel_ringstate.cuh"

void initializeDeviceVariables( const real_number s_dx, const real_number s_dt, const real_number p_g_r, const int s_N, const real_number p_m_eff, const real_number p_gamma_c, const real_number p_g_c, const real_number p_g_pm, const real_number p_gamma_r, const real_number p_R, const real_number p_delta_LT, const real_number p_xmax, const real_number h_bar_s );

void initializeDeviceArrays(const int s_N);

void setDeviceArrays(complex_number* psi_plus, complex_number* psi_minus, complex_number* n_plus, complex_number* n_minus, const int s_N);

void getDeviceArrays(complex_number* psi_plus, complex_number* psi_minus, complex_number* n_plus, complex_number* n_minus, complex_number* fft_plus, complex_number* fft_minus, const int s_N);

void freeDeviceArrays();

void rungeFunctionIterate( System& s, bool evaluate_pulse );

void generateRingPhase( int s_N, real_number amp, int n, real_number w1, real_number w2, real_number xPos, real_number yPos, real_number p_xmax, real_number s_dx, bool normalize, complex_number* buffer, bool reset = true );
void generateRingState( int s_N, real_number amp, real_number w1, real_number w2, real_number xPos, real_number yPos, real_number p_xmax, real_number s_dx, bool normalize, complex_number* buffer, bool reset = true );

void initializePumpVariables( real_number* pump_plus, real_number* pump_minus, const int size );
void initializePulseVariables( real_number* pulse_t0, real_number* pulse_amp, real_number* pulse_freq, real_number* pulse_sigma, int* pulse_m, int* pulse_pol, real_number* pulse_width, real_number* pulse_X, real_number* pulse_Y, const int size );

void generateRingPhase( int s_N, real_number amp, int n, real_number w1, real_number w2, real_number xPos, real_number yPos, real_number p_xmax, real_number s_dx, bool normalize, complex_number* buffer, bool reset );
void generateRingState( int s_N, real_number amp, real_number w1, real_number w2, real_number xPos, real_number yPos, real_number p_xmax, real_number s_dx, bool normalize, complex_number* buffer, bool reset );