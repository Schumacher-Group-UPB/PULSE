#pragma once
#include <complex>
using Scalar = std::complex<double>;

#include "system.hpp"
#include "cuda_complex_math.cuh"
#include "kernel_ringstate.cuh"

void initializeDeviceVariables( const double s_dx, const double s_dt, const double p_g_r, const int s_N, const double p_m_eff, const double p_gamma_c, const double p_g_c, const double p_g_pm, const double p_gamma_r, const double p_R, const double p_delta_LT, const double p_xmax, const double h_bar_s );

void initializeDeviceArrays(const int s_N);

void setDeviceArrays(Scalar* psi_plus, Scalar* psi_minus, Scalar* n_plus, Scalar* n_minus, const int s_N);

void getDeviceArrays(Scalar* psi_plus, Scalar* psi_minus, Scalar* n_plus, Scalar* n_minus, Scalar* fft_plus, Scalar* fft_minus, const int s_N);

void freeDeviceArrays();

void setPulse(Scalar* pulse, Scalar* pulse2, const int s_N);

void rungeFunctionIterate( System& s, bool evaluate_pulse );

void generateRingPhase( int s_N, double amp, int n, double w1, double w2, double xPos, double yPos, double p_xmax, double s_dx, bool normalize, Scalar* buffer, bool reset = true );
void generateRingState( int s_N, double amp, double w1, double w2, double xPos, double yPos, double p_xmax, double s_dx, bool normalize, Scalar* buffer, bool reset = true );

void initializePumpVariables( double* pump_amp, double* pump_width, double* pump_X, double* pump_Y, int* pump_pol, const int size );
void initializePulseVariables( double* pulse_t0, double* pulse_amp, double* pulse_freq, double* pulse_sigma, int* pulse_m, int* pulse_pol, double* pulse_width, double* pulse_X, double* pulse_Y, const int size );

void generateRingPhase( int s_N, double amp, int n, double w1, double w2, double xPos, double yPos, double p_xmax, double s_dx, bool normalize, Scalar* buffer, bool reset );
void generateRingState( int s_N, double amp, double w1, double w2, double xPos, double yPos, double p_xmax, double s_dx, bool normalize, Scalar* buffer, bool reset );