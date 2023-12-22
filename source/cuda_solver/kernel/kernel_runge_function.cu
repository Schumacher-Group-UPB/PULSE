#include "kernel/kernel_runge_function.cuh"
#include "kernel/kernel_hamilton.cuh"
#include "kernel/kernel_index_overwrite.cuh"
#include "omp.h"

CUDA_GLOBAL void PC3::Kernel::runge_func_kernel_tetm( int i, real_number t, Device::Pointers dev_ptrs, System::Parameters p,
                                                      InputOutput io ) {
    OVERWRITE_THREAD_INDEX( i );

    const int row = i / p.N_x;
    const int col = i % p.N_x;

    complex_number hamilton_regular_plus, hamilton_regular_minus, hamilton_cross_plus, hamilton_cross_minus;
    PC3::Hamilton::tetm_plus( hamilton_regular_plus, hamilton_cross_minus, io.in_wf_plus, i, row, col, p.N_x, p.N_y, p.periodic_boundary_x, p.periodic_boundary_y );
    PC3::Hamilton::tetm_minus( hamilton_regular_minus, hamilton_cross_plus, io.in_wf_minus, i, row, col, p.N_x, p.N_y, p.periodic_boundary_x, p.periodic_boundary_y );

    const auto in_wf_plus = io.in_wf_plus[i];
    const auto in_rv_plus = io.in_rv_plus[i];
    const auto potential_plus = dev_ptrs.potential_plus[i];
    const auto pump_plus = dev_ptrs.pump_plus[i];
    const auto in_wf_minus = io.in_wf_minus[i];
    const auto in_rv_minus = io.in_rv_minus[i];
    const auto potential_minus = dev_ptrs.potential_minus[i];
    const auto pump_minus = dev_ptrs.pump_minus[i];
    const real_number in_psi_plus_norm = CUDA::abs2( in_wf_plus );
    const real_number in_psi_minus_norm = CUDA::abs2( in_wf_minus );

    complex_number result_wf = p.minus_i_over_h_bar_s * ( p.m_eff_scaled * hamilton_regular_plus );
    result_wf += p.minus_i_over_h_bar_s * (potential_plus + p.g_c * in_psi_plus_norm + p.g_r * in_rv_plus) * in_wf_plus;
    result_wf += -0.5 * (p.R * in_rv_plus - p.gamma_c) * in_wf_plus;
    result_wf += (p.g_pm * in_psi_minus_norm + p.delta_LT_scaled * hamilton_cross_plus) * in_wf_plus;
    io.out_wf_plus[i] = result_wf;
    io.out_rv_plus[i] = pump_plus - ( p.gamma_r + p.R * in_psi_plus_norm ) * in_rv_plus;

    result_wf = p.minus_i_over_h_bar_s * ( p.m_eff_scaled * hamilton_regular_minus );
    result_wf += p.minus_i_over_h_bar_s * (potential_minus + p.g_c * in_psi_minus_norm + p.g_r * in_rv_minus) * in_wf_minus;
    result_wf += -0.5 * (p.R * in_rv_minus - p.gamma_c) * in_wf_minus;
    result_wf += (p.g_pm * in_psi_plus_norm + p.delta_LT_scaled * hamilton_cross_minus) * in_wf_minus;
    io.out_wf_minus[i] = result_wf;
    io.out_rv_minus[i] = pump_minus - ( p.gamma_r + p.R * in_psi_minus_norm ) * in_rv_minus;
}

/**
 * Mode without TE/TM Splitting
 * The differential equation for this model reduces to
 * ...
 */
CUDA_GLOBAL void PC3::Kernel::runge_func_kernel_scalar( int i, real_number t, Device::Pointers dev_ptrs, System::Parameters p,
                                                        InputOutput io ) {
    OVERWRITE_THREAD_INDEX( i );

    const int row = i / p.N_x;
    const int col = i % p.N_x;

    complex_number hamilton;
    PC3::Hamilton::scalar( hamilton, io.in_wf_plus, i, row, col, p.N_x, p.N_y, p.periodic_boundary_x, p.periodic_boundary_y );
    
    const complex_number in_wf = io.in_wf_plus[i];
    const complex_number in_rv = io.in_rv_plus[i];
    const complex_number potential = dev_ptrs.potential_plus[i];
    const complex_number pump = dev_ptrs.pump_plus[i];
    const real_number in_psi_norm = CUDA::abs2( in_wf );
    
    complex_number result = p.minus_i_over_h_bar_s * ( p.m_eff_scaled * hamilton );
    result += p.minus_i_over_h_bar_s * potential * in_wf;
    result += p.minus_i_over_h_bar_s * p.g_c * in_psi_norm * in_wf;
    result += p.minus_i_over_h_bar_s * p.g_r * in_rv * in_wf;
    result += 0.5 * p.R * in_rv * in_wf;
    result -= 0.5 * p.gamma_c * in_wf;
    
    //result += p.minus_i_over_h_bar_s * (potential + p.g_c * in_psi_norm + p.g_r * in_rv) * in_wf;
    //result += -0.5 * (p.R * in_rv - p.gamma_c) * in_wf;
    io.out_wf_plus[i] = result;
    result = pump;
    result -= p.gamma_r * in_rv;
    result -= p.R * in_psi_norm * in_rv;
    io.out_rv_plus[i] = result;
}