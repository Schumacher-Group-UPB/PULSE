#include "kernel_runge_function.cuh"
#include "omp.h"

#ifdef TETMSPLITTING

/**
 * Model with TE/TM Splitting
 * The differential equation for this model reads
 * ...
 */
CUDA_GLOBAL void rungeFuncKernel( int i, real_number t, complex_number* __restrict__ in_Psi_Plus, complex_number* __restrict__ in_Psi_Minus, complex_number* __restrict__ in_n_Plus, complex_number* __restrict__ in_n_Minus, complex_number* __restrict__ k_Psi_Plus, complex_number* __restrict__ k_Psi_Minus, complex_number* __restrict__ k_n_Plus, complex_number* __restrict__ k_n_Minus,
                                 /* Pump Parameters */ real_number* __restrict__ dev_pump_cache_Plus, real_number* __restrict__ dev_pump_cache_Minus,
                                 /* Pulse Parameters */ real_number* __restrict__ dev_pulse_t0, real_number* __restrict__ dev_pulse_amp, real_number* __restrict__ dev_pulse_freq, real_number* __restrict__ dev_pulse_sigma, int* __restrict__ dev_pulse_m, int* __restrict__ dev_pulse_pol, real_number* __restrict__ dev_pulse_width, real_number* __restrict__ dev_pulse_X, real_number* __restrict__ dev_pulse_Y, bool evaluate_pulse ) {
    
    // If the GPU is used, overwrite the current index with the gpu thread index.
    #ifndef USECPU
    int blockId = ( gridDim.x * blockIdx.y ) + blockIdx.x;
    i = ( blockId * ( blockDim.x * blockDim.y ) ) + ( threadIdx.y * blockDim.x ) + threadIdx.x;
    #endif
    if ( i >= dev_s_N * dev_s_N )
        return;

    const int row = device_floor( i / dev_s_N );
    const int col = i % dev_s_N;
    complex_number hamilton_regular_plus, hamilton_regular_minus, hamilton_cross_plus, hamilton_minus_cross;
    hamilton_1( hamilton_regular_plus, hamilton_minus_cross, in_Psi_Plus, i, row, col, dev_s_N );
    hamilton_2( hamilton_regular_minus, hamilton_cross_plus, in_Psi_Minus, i, row, col, dev_s_N );
    real_number in_psi_plus_norm = abs2( in_Psi_Plus[i] );
    real_number in_psi_minus_norm = abs2( in_Psi_Minus[i] );
    
    k_Psi_Plus[i] = -1.0*dev_minus_i_over_h_bar_s * ( dev_p_m_eff_scaled * hamilton_regular_plus + (dev_p_g_c * in_psi_plus_norm + dev_p_g_r * in_n_Plus[i] + 0.5*dev_i_h_bar_s*(dev_p_R*in_n_Plus[i] - dev_p_gamma_c) + dev_p_g_pm * in_psi_minus_norm) * in_Psi_Plus[i] + dev_p_delta_LT_scaled*hamilton_cross_plus );
    k_Psi_Minus[i] = -1.0*dev_minus_i_over_h_bar_s * ( dev_p_m_eff_scaled * hamilton_regular_minus + (dev_p_g_c * in_psi_minus_norm + dev_p_g_r * in_n_Minus[i] + 0.5*dev_i_h_bar_s*(dev_p_R*in_n_Minus[i] - dev_p_gamma_c) + dev_p_g_pm * in_psi_plus_norm) * in_Psi_Minus[i] + dev_p_delta_LT_scaled*hamilton_minus_cross );
    k_n_Plus[i] = dev_pump_cache_Plus[i] - (dev_p_gamma_r + dev_p_R * in_psi_plus_norm) * in_n_Plus[i];
    k_n_Minus[i] = dev_pump_cache_Minus[i] - (dev_p_gamma_r + dev_p_R * in_psi_minus_norm) * in_n_Minus[i];
    // Add Pulse
    if ( evaluate_pulse ) {
        auto x = -dev_p_xmax / 2.0 + dev_s_dx * col;
        auto y = -dev_p_xmax / 2.0 + dev_s_dx * row;
        for ( int c = 0; c < dev_n_pulse; c++ ) {
            const auto xpos = dev_pulse_X[c];
            const auto ypos = dev_pulse_Y[c];
            real_number r = sqrt( abs2( x - xpos ) + abs2( y - ypos ) );
            const auto w = dev_pulse_width[c];
            const auto exp_factor = r * r / w / w;
            complex_number space_shape = dev_pulse_amp[c] * r / w / w * exp( -exp_factor ) * pow( ( x - xpos + 1.0 * sign( dev_pulse_m[c] ) * complex_number{ 0, 1.0 } * ( y - ypos ) ), abs( dev_pulse_m[c] ) );
            const auto t0 = dev_pulse_t0[c];
            complex_number temp_shape = dev_one_over_h_bar_s * exp( -( t - t0 ) * ( t - t0 ) / dev_pulse_sigma[c] / dev_pulse_sigma[c] - complex_number{ 0, 1.0 } * dev_pulse_freq[c] * ( t - t0 ) );
            if ( dev_pulse_pol[c] >= 0 )
                k_Psi_Plus[i] += space_shape * temp_shape;
            if ( dev_pulse_pol[c] <= 0 )
                k_Psi_Minus[i] += space_shape * temp_shape;
        }
    }
}

#else

/**
 * Mode without TE/TM Splitting
 * The differential equation for this model reduces to
 * ...
 */
CUDA_GLOBAL void rungeFuncKernel( int i, real_number t, complex_number* __restrict__ in_Psi_Plus, complex_number* __restrict__ in_Psi_Minus, complex_number* __restrict__ in_n_Plus, complex_number* __restrict__ in_n_Minus, complex_number* __restrict__ k_Psi_Plus, complex_number* __restrict__ k_Psi_Minus, complex_number* __restrict__ k_n_Plus, complex_number* __restrict__ k_n_Minus,
                                 /* Pump Parameters */ real_number* __restrict__ dev_pump_cache_Plus, real_number* __restrict__ dev_pump_cache_Minus,
                                 /* Pulse Parameters */ real_number* __restrict__ dev_pulse_t0, real_number* __restrict__ dev_pulse_amp, real_number* __restrict__ dev_pulse_freq, real_number* __restrict__ dev_pulse_sigma, int* __restrict__ dev_pulse_m, int* __restrict__ dev_pulse_pol, real_number* __restrict__ dev_pulse_width, real_number* __restrict__ dev_pulse_X, real_number* __restrict__ dev_pulse_Y, bool evaluate_pulse ) {
    
    // If the GPU is used, overwrite the current index with the gpu thread index.
    #ifndef USECPU
    int blockId = ( gridDim.x * blockIdx.y ) + blockIdx.x;
    i = ( blockId * ( blockDim.x * blockDim.y ) ) + ( threadIdx.y * blockDim.x ) + threadIdx.x;
    #endif
    if ( i >= dev_s_N * dev_s_N )
        return;

    const int row = device_floor( i / dev_s_N );
    const int col = i % dev_s_N;
    complex_number hamilton_regular_plus;
    hamilton( hamilton_regular_plus, in_Psi_Plus, i, row, col, dev_s_N );
    const real_number in_psi_plus_norm = abs2( in_Psi_Plus[i] );
    k_Psi_Plus[i] = -1.0*dev_minus_i_over_h_bar_s * ( dev_p_m_eff_scaled * hamilton_regular_plus + (dev_p_g_c * in_psi_plus_norm + dev_p_g_r * in_n_Plus[i] + 0.5*dev_i_h_bar_s*(dev_p_R*in_n_Plus[i] - dev_p_gamma_c)) * in_Psi_Plus[i] );
    k_n_Plus[i] = dev_pump_cache_Plus[i] - (dev_p_gamma_r + dev_p_R * in_psi_plus_norm) * in_n_Plus[i];
    // Add Pulse
    if ( evaluate_pulse ) {
        const real_number x = -dev_p_xmax / 2.0 + dev_s_dx * col;
        const real_number y = -dev_p_xmax / 2.0 + dev_s_dx * row;
        for ( int c = 0; c < dev_n_pulse; c++ ) {
            const real_number xpos = dev_pulse_X[c];
            const real_number ypos = dev_pulse_Y[c];
            const real_number r = sqrt( abs2( x - xpos ) + abs2( y - ypos ) );
            const real_number w = dev_pulse_width[c];
            const real_number exp_factor = r * r / w / w;
            const complex_number space_shape = dev_pulse_amp[c] * r / w / w * exp( -exp_factor ) * pow( ( x - xpos + 1.0 * sign( dev_pulse_m[c] ) * complex_number{ 0, 1.0 } * ( y - ypos ) ), abs( dev_pulse_m[c] ) );
            const real_number t0 = dev_pulse_t0[c];
            const complex_number temp_shape = dev_one_over_h_bar_s*exp( -( t - t0 ) * ( t - t0 ) / dev_pulse_sigma[c] / dev_pulse_sigma[c] - complex_number{ 0, 1.0 } * dev_pulse_freq[c] * ( t - t0 ) );
            k_Psi_Plus[i] += space_shape * temp_shape;
        }
    }
}

#endif