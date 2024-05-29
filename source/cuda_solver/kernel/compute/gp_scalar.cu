#include "kernel/kernel_runge_function.cuh"
#include "kernel/kernel_hamilton.cuh"
#include "kernel/kernel_index_overwrite.cuh"

/**
 * Mode without TE/TM Splitting
 * The differential equation for this model reduces to
 * ...
 */
CUDA_GLOBAL void PC3::Kernel::Compute::gp_scalar( int i, real_number t, MatrixContainer::Pointers dev_ptrs, System::Parameters p, Solver::Oscillation::Pointers oscillation_pulse, Solver::Oscillation::Pointers oscillation_pump, Solver::Oscillation::Pointers oscillation_potential, InputOutput io ) {
    
    OVERWRITE_THREAD_INDEX( i );

    complex_number hamilton;
    PC3::Hamilton::scalar( hamilton, io.in_wf_plus, i, i / p.N_x /*Row*/, i % p.N_x /*Col*/, p.N_x, p.N_y, p.dx, p.dy, p.periodic_boundary_x, p.periodic_boundary_y );
    
    const complex_number in_wf = io.in_wf_plus[i];
    const complex_number in_rv = io.in_rv_plus[i];
    const real_number in_psi_norm = CUDA::abs2( in_wf );
    
    // MARK: Wavefunction
    complex_number result = p.minus_i_over_h_bar_s * ( p.m_eff_scaled * hamilton );

    for (int k = 0; k < oscillation_potential.n; k++) {
        const size_t offset = k * p.N_x * p.N_y;
        const complex_number potential = dev_ptrs.potential_plus[i+offset] * PC3::CUDA::gaussian_oscillator(t, oscillation_potential.t0[k], oscillation_potential.sigma[k], oscillation_potential.freq[k]);
        result += p.minus_i_over_h_bar_s * potential * in_wf;
    }

    result += p.minus_i_over_h_bar_s * p.g_c * in_psi_norm * in_wf;
    result += p.minus_i_over_h_bar_s * p.g_r * in_rv * in_wf;
    result += real_number(0.5) * p.R * in_rv * in_wf;
    result -= real_number(0.5) * p.gamma_c * in_wf;

    // MARK: Pulse
    for (int k = 0; k < oscillation_pulse.n; k++) {
        const size_t offset = k * p.N_x * p.N_y;
        const complex_number pulse = dev_ptrs.pulse_plus[i+offset];
        result += p.one_over_h_bar_s * pulse * PC3::CUDA::gaussian_complex_oscillator(t, oscillation_pulse.t0[k], oscillation_pulse.sigma[k], oscillation_pulse.freq[k]);
    }
    
    // MARK: Stochastic
    if (p.stochastic_amplitude > 0.0) {
        const complex_number dw = dev_ptrs.random_number[i] * PC3::CUDA::sqrt( ( p.R * in_rv + p.gamma_c ) / (real_number(4.0) * p.dV) );
        result -= p.minus_i_over_h_bar_s * p.g_c * in_wf / p.dV - dw / p.dt;
    }
    
    io.out_wf_plus[i] = result;
    
    // MARK: Reservoir
    result = -p.gamma_r * in_rv;
    result -= p.R * in_psi_norm * in_rv;
    for (int k = 0; k < oscillation_pump.n; k++) {
        const int offset = k * p.N_x * p.N_y;
        result += dev_ptrs.pump_plus[i+offset] * PC3::CUDA::gaussian_oscillator(t, oscillation_pump.t0[k], oscillation_pump.sigma[k], oscillation_pump.freq[k]);
    }
    result += dev_ptrs.pump_plus[i];

    // MARK: Stochastic-2
    if (p.stochastic_amplitude > 0.0)
        result += p.R * in_rv / p.dV;
    io.out_rv_plus[i] = result;

}

/**
 * Linear, Nonlinear and Independet parts of the upper Kernel
 * These isolated implementations serve for the Split Step
 * Fourier Method (SSFM)
*/

CUDA_GLOBAL void PC3::Kernel::Compute::gp_scalar_linear_fourier( int i, real_number t, MatrixContainer::Pointers dev_ptrs, System::Parameters p, Solver::Oscillation::Pointers oscillation_pulse, Solver::Oscillation::Pointers oscillation_pump, Solver::Oscillation::Pointers oscillation_potential, InputOutput io ) {
    
    OVERWRITE_THREAD_INDEX( i );
    size_t row = i / p.N_x;
    size_t col = i % p.N_x;
    
    real_number k_x = 3.1415926535 * real_number(col <= p.N_x/2 ? col : -p.N_x + col)/p.L_x;
    real_number k_y = 3.1415926535 * real_number(row <= p.N_y/2 ? row : -p.N_y + row)/p.L_y;

    real_number linear = p.h_bar_s/2.0/p.m_eff * (k_x*k_x + k_y*k_y);
    io.out_wf_plus[i] = io.in_wf_plus[i] / real_number(p.N2) * CUDA::exp( p.minus_i * linear * p.dt / real_number(2.0) );
}

CUDA_GLOBAL void PC3::Kernel::Compute::gp_scalar_nonlinear( int i, real_number t, MatrixContainer::Pointers dev_ptrs, System::Parameters p, Solver::Oscillation::Pointers oscillation_pulse, Solver::Oscillation::Pointers oscillation_pump, Solver::Oscillation::Pointers oscillation_potential, InputOutput io ) {
    
    OVERWRITE_THREAD_INDEX( i );
    
    const complex_number in_wf = io.in_wf_plus[i];
    const real_number in_psi_norm = CUDA::abs2( in_wf );
    
    // MARK: Wavefunction
    complex_number result = {p.g_c * in_psi_norm,0.0};
    result += p.minus_i*p.h_bar_s * real_number(0.5) * p.gamma_c;

    for (int k = 0; k < oscillation_potential.n; k++) {
        const size_t offset = k * p.N_x * p.N_y;
        const complex_number potential = dev_ptrs.potential_plus[i+offset] * PC3::CUDA::gaussian_oscillator(t, oscillation_potential.t0[k], oscillation_potential.sigma[k], oscillation_potential.freq[k]);
        result += potential;
    }

    io.out_wf_plus[i] = in_wf * CUDA::exp(p.minus_i_over_h_bar_s * result * p.dt);;
}

CUDA_GLOBAL void PC3::Kernel::Compute::gp_scalar_independent( int i, real_number t, MatrixContainer::Pointers dev_ptrs, System::Parameters p, Solver::Oscillation::Pointers oscillation_pulse, Solver::Oscillation::Pointers oscillation_pump, Solver::Oscillation::Pointers oscillation_potential, InputOutput io ) {
    
    OVERWRITE_THREAD_INDEX( i );
    complex_number result = {0.0,0.0};

    // MARK: Pulse
    for (int k = 0; k < oscillation_pulse.n; k++) {
        const size_t offset = k * p.N_x * p.N_y;
        const complex_number pulse = dev_ptrs.pulse_plus[i+offset];
        result += p.minus_i_over_h_bar_s * p.dt * pulse * PC3::CUDA::gaussian_complex_oscillator(t, oscillation_pulse.t0[k], oscillation_pulse.sigma[k], oscillation_pulse.freq[k]);
    }

    io.out_wf_plus[i] = io.in_wf_plus[i] + result;
}