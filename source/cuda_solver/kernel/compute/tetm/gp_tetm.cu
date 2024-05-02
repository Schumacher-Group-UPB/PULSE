#include "kernel/kernel_runge_function.cuh"
#include "kernel/kernel_hamilton.cuh"
#include "kernel/kernel_index_overwrite.cuh"

CUDA_GLOBAL void PC3::Kernel::Compute::gp_tetm( int i, real_number t, MatrixContainer::Pointers dev_ptrs, System::Parameters p, Solver::Oscillation::Pointers oscillation, InputOutput io ) {
    OVERWRITE_THREAD_INDEX( i );

    const int row = i / p.N_x;
    const int col = i % p.N_x;

    complex_number hamilton_regular_plus, hamilton_regular_minus, hamilton_cross_plus, hamilton_cross_minus;
    PC3::Hamilton::tetm_plus( hamilton_regular_plus, hamilton_cross_minus, io.in_wf_plus, i, row, col, p.N_x, p.N_y, p.dx, p.dy, p.periodic_boundary_x, p.periodic_boundary_y );
    PC3::Hamilton::tetm_minus( hamilton_regular_minus, hamilton_cross_plus, io.in_wf_minus, i, row, col, p.N_x, p.N_y, p.dx, p.dy, p.periodic_boundary_x, p.periodic_boundary_y );

    const auto in_wf_plus = io.in_wf_plus[i];
    const auto in_rv_plus = io.in_rv_plus[i];
    const auto in_wf_minus = io.in_wf_minus[i];
    const auto in_rv_minus = io.in_rv_minus[i];
    const real_number in_psi_plus_norm = CUDA::abs2( in_wf_plus );
    const real_number in_psi_minus_norm = CUDA::abs2( in_wf_minus );

    complex_number result_wf = p.minus_i_over_h_bar_s * p.m_eff_scaled * hamilton_regular_plus;
    
    for (int k = 0; k < oscillation.n; k++) {
        const size_t offset = k * p.N_x * p.N_y;
        const complex_number potential = dev_ptrs.potential_plus[i+offset] * PC3::CUDA::gaussian_oscillator(t, oscillation.t0[k], oscillation.sigma[k], oscillation.freq[k]);
        result_wf += p.minus_i_over_h_bar_s * potential * in_wf_plus;
    }

    result_wf += p.minus_i_over_h_bar_s * p.g_c * in_psi_plus_norm * in_wf_plus;
    result_wf += p.minus_i_over_h_bar_s * p.g_r * in_rv_plus * in_wf_plus;
    result_wf += real_number(0.5) * p.R * in_rv_plus * in_wf_plus;
    result_wf -= real_number(0.5)* p.gamma_c * in_wf_plus;

    result_wf += p.minus_i_over_h_bar_s * p.g_pm * in_psi_minus_norm * in_wf_plus;
    result_wf += p.minus_i_over_h_bar_s * p.delta_LT * hamilton_cross_plus;
    io.out_wf_plus[i] = result_wf;


    result_wf = p.minus_i_over_h_bar_s * p.m_eff_scaled * hamilton_regular_minus;
    
    for (int k = 0; k < oscillation.n; k++) {
        const size_t offset = k * p.N_x * p.N_y;
        const complex_number potential = dev_ptrs.potential_minus[i+offset] * PC3::CUDA::gaussian_oscillator(t, oscillation.t0[k], oscillation.sigma[k], oscillation.freq[k]);
        result_wf += p.minus_i_over_h_bar_s * potential * in_wf_minus;
    }

    result_wf += p.minus_i_over_h_bar_s * p.g_c * in_psi_minus_norm * in_wf_minus;
    result_wf += p.minus_i_over_h_bar_s * p.g_r * in_rv_minus * in_wf_minus;
    result_wf += real_number(0.5) * p.R * in_rv_minus * in_wf_minus;
    result_wf -= real_number(0.5) * p.gamma_c * in_wf_minus;
 
    result_wf += p.minus_i_over_h_bar_s * p.g_pm * in_psi_plus_norm * in_wf_minus;
    result_wf += p.minus_i_over_h_bar_s * p.delta_LT * hamilton_cross_minus;
    io.out_wf_minus[i] = result_wf;
}