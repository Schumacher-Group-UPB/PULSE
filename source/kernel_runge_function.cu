#include "kernel_runge_function.cuh"

__global__ void rungeFuncKernel( double t, cuDoubleComplex* __restrict__ in_Psi_Plus, cuDoubleComplex* __restrict__ in_Psi_Minus, cuDoubleComplex* __restrict__ in_n_Plus, cuDoubleComplex* __restrict__ in_n_Minus, cuDoubleComplex* __restrict__ k_Psi_Plus, cuDoubleComplex* __restrict__ k_Psi_Minus, cuDoubleComplex* __restrict__ k_n_Plus, cuDoubleComplex* __restrict__ k_n_Minus,
                                 /* Pump Parameters */ double* __restrict__ dev_pump_amp, double* __restrict__ dev_pump_width, double* __restrict__ dev_pump_X, double* __restrict__ dev_pump_Y, int* __restrict__ dev_pump_pol,
                                 /* Pulse Parameters */ double* __restrict__ dev_pulse_t0, double* __restrict__ dev_pulse_amp, double* __restrict__ dev_pulse_freq, double* __restrict__ dev_pulse_sigma, int* __restrict__ dev_pulse_m, int* __restrict__ dev_pulse_pol, double* __restrict__ dev_pulse_width, double* __restrict__ dev_pulse_X, double* __restrict__ dev_pulse_Y, bool evaluate_pulse ) {
    int blockId = ( gridDim.x * blockIdx.y ) + blockIdx.x;
    int i = ( blockId * ( blockDim.x * blockDim.y ) ) + ( threadIdx.y * blockDim.x ) + threadIdx.x;
    if ( i >= dev_s_N * dev_s_N )
        return;

    const int row = device_floor( i / dev_s_N );
    const int col = i % dev_s_N;
    cuDoubleComplex DT1, DT2, DT3, DT4;
    hamilton_1( DT1, DT4, in_Psi_Plus, i, row, col, dev_s_N );
    hamilton_2( DT2, DT3, in_Psi_Minus, i, row, col, dev_s_N );
    double in_psi_plus_norm = abs2( in_Psi_Plus[i] );
    double in_psi_minus_norm = abs2( in_Psi_Minus[i] );
    k_Psi_Plus[i] = dev_minus_i * ( dev_p_m_eff_scaled * DT1 - dev_half_i * dev_p_gamma_c * in_Psi_Plus[i] + dev_p_g_c * in_psi_plus_norm * in_Psi_Plus[i] + dev_pgr_plus_pR * in_n_Plus[i] * in_Psi_Plus[i] + dev_p_g_pm * in_psi_minus_norm * in_Psi_Plus[i] + dev_p_delta_LT_scaled * DT2 );
    k_Psi_Minus[i] = dev_minus_i * ( dev_p_m_eff_scaled * DT3 - dev_half_i * dev_p_gamma_c * in_Psi_Minus[i] + dev_p_g_c * in_psi_minus_norm * in_Psi_Minus[i] + dev_pgr_plus_pR * in_n_Minus[i] * in_Psi_Minus[i] + dev_p_g_pm * in_psi_plus_norm * in_Psi_Minus[i] + dev_p_delta_LT_scaled * DT4 );
    k_n_Plus[i] = -dev_p_gamma_r * in_n_Plus[i] - dev_p_R * in_psi_plus_norm * in_n_Plus[i];
    k_n_Minus[i] = -dev_p_gamma_r * in_n_Minus[i] - dev_p_R * in_psi_minus_norm * in_n_Minus[i];
    // Add Pumps
    if ( dev_n_pump ) {
        auto x = -dev_p_xmax / 2.0 + dev_s_dx * col;
        auto y = -dev_p_xmax / 2.0 + dev_s_dx * row;
        for ( int c = 0; c < dev_n_pump; c++ ) {
            const double r_squared = abs2( x - dev_pump_X[c] ) + abs2( y - dev_pump_Y[c] );
            const auto w = dev_pump_width[c];
            const auto exp_factor = r_squared / w / w;
            if ( dev_pump_pol[c] >= 0 )
                k_n_Plus[i] += dev_pump_amp[c] * exp_factor * exp( -exp_factor );
            if ( dev_pump_pol[c] <= 0 )
                k_n_Minus[i] += dev_pump_amp[c] * exp_factor * exp( -exp_factor );
        }
    }
    // Add Pulse
    if ( evaluate_pulse ) {
        auto x = -dev_p_xmax / 2.0 + dev_s_dx * col;
        auto y = -dev_p_xmax / 2.0 + dev_s_dx * row;
        for ( int c = 0; c < dev_n_pulse; c++ ) {
            const auto xpos = dev_pulse_X[c];
            const auto ypos = dev_pulse_Y[c];
            double r = sqrt( abs2( x - xpos ) + abs2( y - ypos ) );
            const auto w = dev_pulse_width[c];
            const auto exp_factor = r * r / w / w;
            cuDoubleComplex space_shape = dev_pulse_amp[c] * r / w / w * exp( -exp_factor ) * pow( ( x - xpos + 1.0 * sign( dev_pulse_m[c] ) * make_cuDoubleComplex( 0, 1.0 ) * ( y - ypos ) ), abs( dev_pulse_m[c] ) );
            const auto t0 = dev_pulse_t0[c];
            cuDoubleComplex temp_shape = dev_one_over_h_bar_s * exp( -( t - t0 ) * ( t - t0 ) / dev_pulse_sigma[c] / dev_pulse_sigma[c] - make_cuDoubleComplex( 0, 1.0 ) * dev_pulse_freq[c] * ( t - t0 ) );
            if ( dev_pulse_pol[c] >= 0 )
                k_Psi_Plus[i] += space_shape * temp_shape;
            if ( dev_pulse_pol[c] <= 0 )
                k_Psi_Minus[i] += space_shape * temp_shape;
        }
    }
}