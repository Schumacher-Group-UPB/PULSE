#include "kernel/kernel_runge_function.cuh"
#include "kernel/kernel_hamilton.cuh"
#include "kernel/kernel_index_overwrite.cuh"
#include "omp.h"

CUDA_GLOBAL void PC3::Kernel::runge_func_kernel_tetm( int i, real_number t, Device::Pointers dev_ptrs, System::Parameters p,
                                                      Solver::PulseParameters::Pointers pulse, bool evaluate_pulse,
                                                      InputOutput io ) {
    OVERWRITE_INDEX_GPU(i);
    if ( i >= p.N2 )
        return;

    const int row = CUDA::floor( i / p.N );
    const int col = i % p.N;
    complex_number hamilton_regular_plus, hamilton_regular_minus, hamilton_cross_plus, hamilton_minus_cross;
    PC3::Hamilton::tetm_plus( hamilton_regular_plus, hamilton_minus_cross, io.in_wf_plus, i, row, col, p.N, p.periodic );
    PC3::Hamilton::tetm_minus( hamilton_regular_minus, hamilton_cross_plus, io.in_wf_minus, i, row, col, p.N, p.periodic );
    real_number in_psi_plus_norm = CUDA::abs2( io.in_wf_plus[i] );
    real_number in_psi_minus_norm = CUDA::abs2( io.in_wf_minus[i] );
    
    io.out_wf_plus[i] = -1.0 * p.minus_i_over_h_bar_s * ( p.m_eff_scaled * hamilton_regular_plus + (p.g_c * in_psi_plus_norm + p.g_r * io.in_rv_plus[i] + 0.5*p.i_h_bar_s*(p.R*io.in_rv_plus[i] - p.gamma_c) + p.g_pm * in_psi_minus_norm) * io.in_wf_plus[i] + p.delta_LT_scaled*hamilton_cross_plus );
    io.out_wf_minus[i] = -1.0 * p.minus_i_over_h_bar_s * ( p.m_eff_scaled * hamilton_regular_minus + (p.g_c * in_psi_minus_norm + p.g_r * io.in_rv_minus[i] + 0.5*p.i_h_bar_s*(p.R*io.in_rv_minus[i] - p.gamma_c) + p.g_pm * in_psi_plus_norm) * io.in_wf_minus[i] + p.delta_LT_scaled*hamilton_minus_cross );
    io.out_rv_plus[i] = dev_ptrs.pump_plus[i] - (p.gamma_r + p.R * in_psi_plus_norm) * io.in_rv_plus[i];
    io.out_rv_minus[i] = dev_ptrs.pump_minus[i] - (p.gamma_r + p.R * in_psi_minus_norm) * io.in_rv_minus[i];
    // Add Pulse
    if ( evaluate_pulse ) {
        io.out_wf_plus[i] += kernel_inline_calculate_pulse(row, col, PC3::Envelope::Polarization::Plus, t, p, pulse);
        io.out_wf_minus[i] += kernel_inline_calculate_pulse(row, col, PC3::Envelope::Polarization::Minus, t, p, pulse);
    }
}

/**
 * Mode without TE/TM Splitting
 * The differential equation for this model reduces to
 * ...
 */
CUDA_GLOBAL void PC3::Kernel::runge_func_kernel_scalar( int i, real_number t, Device::Pointers dev_ptrs, System::Parameters p,
                                                        Solver::PulseParameters::Pointers pulse, bool evaluate_pulse,
                                                        InputOutput io ) {
    OVERWRITE_INDEX_GPU(i);
    if ( i >= p.N2 )
        return;

    const int row = CUDA::floor( i / p.N );
    const int col = i % p.N;
    complex_number hamilton_regular_plus;
    PC3::Hamilton::scalar( hamilton_regular_plus, io.in_wf_plus, i, row, col, p.N, p.periodic );
    const real_number in_psi_plus_norm = CUDA::abs2( io.in_wf_plus[i] );
    io.out_wf_plus[i] = -1.0 * p.minus_i_over_h_bar_s * ( p.m_eff_scaled * hamilton_regular_plus + (p.g_c * in_psi_plus_norm + p.g_r * io.in_rv_plus[i] + 0.5*p.i_h_bar_s*(p.R*io.in_rv_plus[i] - p.gamma_c)) * io.in_wf_plus[i] );
    io.out_rv_plus[i] = dev_ptrs.pump_plus[i] - (p.gamma_r + p.R * in_psi_plus_norm) * io.in_rv_plus[i];
    // Add Pulse
    if ( evaluate_pulse ) {
        io.out_wf_plus[i] += kernel_inline_calculate_pulse(row, col, PC3::Envelope::Polarization::Plus, t, p, pulse);
    }
}

// Here we can implement additional kernels for the runge kutta method.