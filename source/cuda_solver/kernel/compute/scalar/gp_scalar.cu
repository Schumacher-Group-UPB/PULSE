#include "kernel/kernel_runge_function.cuh"
#include "kernel/kernel_hamilton.cuh"
#include "kernel/kernel_index_overwrite.cuh"

/**
 * Mode without TE/TM Splitting
 * The differential equation for this model reduces to
 * ...
 */
CUDA_GLOBAL void PC3::Kernel::Compute::gp_scalar( int i, real_number t, Device::Pointers dev_ptrs, System::Parameters p, Solver::Oscillation::Pointers oscillation, InputOutput io ) {
    
    OVERWRITE_THREAD_INDEX( i );

    const int row = i / p.N_x;
    const int col = i % p.N_x;

    complex_number hamilton;
    PC3::Hamilton::scalar( hamilton, io.in_wf_plus, i, row, col, p.N_x, p.N_y, p.dx, p.dy, p.periodic_boundary_x, p.periodic_boundary_y );
    
    const complex_number in_wf = io.in_wf_plus[i];
    const complex_number in_rv = io.in_rv_plus[i];
    const real_number in_psi_norm = CUDA::abs2( in_wf );
    
    complex_number result = p.minus_i_over_h_bar_s * ( p.m_eff_scaled * hamilton );

    for (int k = 0; k < oscillation.n; k++) {
        const size_t offset = k * p.N_x * p.N_y;
        const complex_number potential = dev_ptrs.potential_plus[i+offset] * PC3::CUDA::gaussian_oscillator(t, oscillation.t0[k], oscillation.sigma[k], oscillation.freq[k]);
        result += p.minus_i_over_h_bar_s * potential * in_wf;
    }

    result += p.minus_i_over_h_bar_s * p.g_c * in_psi_norm * in_wf;
    result += p.minus_i_over_h_bar_s * p.g_r * in_rv * in_wf;
    result += 0.5 * p.R * in_rv * in_wf;
    result -= 0.5 * p.gamma_c * in_wf;
    
    io.out_wf_plus[i] = result;
}