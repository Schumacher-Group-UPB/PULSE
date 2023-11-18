#include "kernel/kernel_runge_function.cuh"
#include "kernel/kernel_hamilton.cuh"
#include "kernel/kernel_index_overwrite.cuh"
#include "omp.h"

CUDA_GLOBAL void PC3::Kernel::runge_func_kernel_tetm( int i, real_number t, Device::Pointers dev_ptrs, System::Parameters p,
                                                      Solver::PulseParameters pulse, bool evaluate_pulse,
                                                      InputOutput io ) {
    OVERWRITE_INDEX_GPU(i);
    if ( i >= p.N2 )
        return;

    const int row = device_floor( i / p.N );
    const int col = i % p.N;
    complex_number hamilton_regular_plus, hamilton_regular_minus, hamilton_cross_plus, hamilton_minus_cross;
    PC3::Hamilton::tetm_plus( hamilton_regular_plus, hamilton_minus_cross, io.in_wf_plus, i, row, col, p.N, p.periodic );
    PC3::Hamilton::tetm_minus( hamilton_regular_minus, hamilton_cross_plus, io.in_wf_minus, i, row, col, p.N, p.periodic );
    real_number in_psi_plus_norm = abs2( io.in_wf_plus[i] );
    real_number in_psi_minus_norm = abs2( io.in_wf_minus[i] );
    
    io.out_wf_plus[i] = -1.0 * p.minus_i_over_h_bar_s * ( p.m_eff_scaled * hamilton_regular_plus + (p.g_c * in_psi_plus_norm + p.g_r * io.in_rv_plus[i] + 0.5*p.i_h_bar_s*(p.R*io.in_rv_plus[i] - p.gamma_c) + p.g_pm * in_psi_minus_norm) * io.in_wf_plus[i] + p.delta_LT_scaled*hamilton_cross_plus );
    io.out_wf_minus[i] = -1.0 * p.minus_i_over_h_bar_s * ( p.m_eff_scaled * hamilton_regular_minus + (p.g_c * in_psi_minus_norm + p.g_r * io.in_rv_minus[i] + 0.5*p.i_h_bar_s*(p.R*io.in_rv_minus[i] - p.gamma_c) + p.g_pm * in_psi_plus_norm) * io.in_wf_minus[i] + p.delta_LT_scaled*hamilton_minus_cross );
    io.out_rv_plus[i] = dev_ptrs.pump_plus[i] - (p.gamma_r + p.R * in_psi_plus_norm) * io.in_rv_plus[i];
    io.out_rv_minus[i] = dev_ptrs.pump_minus[i] - (p.gamma_r + p.R * in_psi_minus_norm) * io.in_rv_minus[i];
    // Add Pulse
    if ( evaluate_pulse ) {
        auto x = -p.xmax + p.dx * col;
        auto y = -p.xmax + p.dx * row;
        for ( int c = 0; c < pulse.n; c++ ) {
            const auto xpos = pulse.x[c];
            const auto ypos = pulse.y[c];
            real_number r = sqrt( abs2( x - xpos ) + abs2( y - ypos ) );
            const auto w = pulse.width[c];
            const auto exp_factor = r * r / w / w;
            complex_number space_shape = pulse.amp[c] * r / w / w * exp( -exp_factor ) * pow( ( x - xpos + 1.0 * sign( pulse.m[c] ) * p.i * ( y - ypos ) ), abs( pulse.m[c] ) );
            const auto t0 = pulse.t0[c];
            complex_number temp_shape = p.one_over_h_bar_s * exp( -( t - t0 ) * ( t - t0 ) / pulse.sigma[c] / pulse.sigma[c] - p.i * pulse.freq[c] * ( t - t0 ) );
            if ( pulse.pol[c] >= 0 )
                io.out_wf_plus[i] += space_shape * temp_shape;
            if ( pulse.pol[c] <= 0 )
                io.out_wf_minus[i] += space_shape * temp_shape;
        }
    }
}

/**
 * Mode without TE/TM Splitting
 * The differential equation for this model reduces to
 * ...
 */
CUDA_GLOBAL void PC3::Kernel::runge_func_kernel_scalar( int i, real_number t, Device::Pointers dev_ptrs, System::Parameters p,
                                                        Solver::PulseParameters pulse, bool evaluate_pulse,
                                                        InputOutput io ) {
    OVERWRITE_INDEX_GPU(i);
    if ( i >= p.N2 )
        return;

    const int row = device_floor( i / p.N );
    const int col = i % p.N;
    complex_number hamilton_regular_plus;
    PC3::Hamilton::scalar( hamilton_regular_plus, io.in_wf_plus, i, row, col, p.N, p.periodic );
    const real_number in_psi_plus_norm = abs2( io.in_wf_plus[i] );
    io.out_wf_plus[i] = -1.0 * p.minus_i_over_h_bar_s * ( p.m_eff_scaled * hamilton_regular_plus + (p.g_c * in_psi_plus_norm + p.g_r * io.in_rv_plus[i] + 0.5*p.i_h_bar_s*(p.R*io.in_rv_plus[i] - p.gamma_c)) * io.in_wf_plus[i] );
    io.out_rv_plus[i] = dev_ptrs.pump_plus[i] - (p.gamma_r + p.R * in_psi_plus_norm) * io.in_rv_plus[i];
    // Add Pulse
    if ( evaluate_pulse ) {
        const real_number x = -p.xmax + p.dx * col;
        const real_number y = -p.xmax + p.dx * row;
        for ( int c = 0; c < pulse.n; c++ ) {
            const real_number xpos = pulse.x[c];
            const real_number ypos = pulse.y[c];
            const real_number r = sqrt( abs2( x - xpos ) + abs2( y - ypos ) );
            const real_number w = pulse.width[c];
            const real_number exp_factor = r * r / w / w;
            const complex_number space_shape = pulse.amp[c] * r / w / w * exp( -exp_factor ) * pow( ( x - xpos + 1.0 * sign( pulse.m[c] ) * p.i * ( y - ypos ) ), abs( pulse.m[c] ) );
            const real_number t0 = pulse.t0[c];
            const complex_number temp_shape = p.one_over_h_bar_s*exp( -( t - t0 ) * ( t - t0 ) / pulse.sigma[c] / pulse.sigma[c] - p.i * pulse.freq[c] * ( t - t0 ) );
            io.out_wf_plus[i] += space_shape * temp_shape;
        }
    }
}

// Here we can implement additional kernels for the runge kutta method.