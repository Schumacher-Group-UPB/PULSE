#include "kernel/kernel_compute.cuh"
#include "kernel/kernel_hamilton.cuh"
#include "kernel/kernel_index_overwrite.cuh"

/**
 * Mode without TE/TM Splitting
 * The differential equation for this model reduces to
 * ...
 */
PULSE_GLOBAL void PC3::Kernel::Compute::gp_scalar( int i, Type::uint current_halo, Solver::VKernelArguments time, Solver::KernelArguments args, Solver::InputOutput io ) {
    GENERATE_SUBGRID_INDEX(i, current_halo);

    const Type::complex in_wf = io.in_wf_plus[i];
    const Type::complex in_rv = io.in_rv_plus[i];
    Type::complex hamilton = args.p.m2_over_dx2_p_dy2 * in_wf;
    hamilton += (io.in_wf_plus[i + args.p.N_x + 2*args.p.halo_size] + io.in_wf_plus[i - args.p.N_x - 2*args.p.halo_size])*args.p.one_over_dy2 + (io.in_wf_plus[i + 1] + io.in_wf_plus[i - 1])*args.p.one_over_dx2;
    //hamilton += PC3::Kernel::Hamilton::scalar_neighbours( io.in_wf_plus, i, i / args.p.N_x /*Row*/, i % args.p.N_x /*Col*/, args.p.N_x, args.p.N_y, args.p.one_over_dx2, args.p.one_over_dy2, args.p.periodic_boundary_x, args.p.periodic_boundary_y );

    const Type::real in_psi_norm = CUDA::abs2( in_wf );
    
    // MARK: Wavefunction
    Type::complex result = args.p.minus_i_over_h_bar_s * ( args.p.m_eff_scaled * hamilton );

    for (int k = 0; k < args.potential_pointers.n; k++) {
        const Type::complex potential = args.dev_ptrs.potential_plus[k][i] * args.potential_pointers.amp[k];
        result += args.p.minus_i_over_h_bar_s * potential * in_wf;
    }

    result += args.p.minus_i_over_h_bar_s * args.p.g_c * in_psi_norm * in_wf;
    result += args.p.minus_i_over_h_bar_s * args.p.g_r * in_rv * in_wf;
    result += Type::real(0.5) * args.p.R * in_rv * in_wf;
    result -= Type::real(0.5) * args.p.gamma_c * in_wf;

    // MARK: Pulse
    for (int k = 0; k < args.pulse_pointers.n; k++) {
        const Type::complex pulse = args.dev_ptrs.pulse_plus[k][i];
        result += args.p.one_over_h_bar_s * pulse * args.pulse_pointers.amp[k];
    }
    
    // MARK: Stochastic
    if (args.p.stochastic_amplitude > 0.0)
        result -= args.p.minus_i_over_h_bar_s * args.p.g_c * in_wf / args.p.dV;
    
    io.out_wf_plus[i] = result;
    
    // MARK: Reservoir
    result = -args.p.gamma_r * in_rv;
    result -= args.p.R * in_psi_norm * in_rv;
    for (int k = 0; k < args.pump_pointers.n; k++) {
            result += args.dev_ptrs.pump_plus[k][i] * args.pump_pointers.amp[k];
    }

    // MARK: Stochastic-2
    if (args.p.stochastic_amplitude > 0.0)
        result += args.p.R * in_rv / args.p.dV;
    
    io.out_rv_plus[i] = result;

}

/**
 * Linear, Nonlinear and Independet parts of the upper Kernel
 * These isolated implementations serve for the Split Step
 * Fourier Method (SSFM)
*/

PULSE_GLOBAL void PC3::Kernel::Compute::gp_scalar_linear_fourier( int i, Solver::VKernelArguments time, Solver::KernelArguments args, Solver::InputOutput io ) {
    
    GET_THREAD_INDEX( i, args.p.N2 );

    // We do some weird looking casting to avoid intermediate casts to Type::uint
    Type::real row = Type::real(Type::uint(i / args.p.N_x));
    Type::real col = Type::real(Type::uint(i % args.p.N_x));
    
    const Type::real k_x = 2.0*3.1415926535 * Type::real(col <= args.p.N_x/2 ? col : -Type::real(args.p.N_x) + col)/args.p.L_x;
    const Type::real k_y = 2.0*3.1415926535 * Type::real(row <= args.p.N_y/2 ? row : -Type::real(args.p.N_y) + row)/args.p.L_y;

    Type::real linear = args.p.h_bar_s/2.0/args.p.m_eff * (k_x*k_x + k_y*k_y);
    io.out_wf_plus[i] = io.in_wf_plus[i] / Type::real(args.p.N2) * CUDA::exp( args.p.minus_i * linear * time.dt / Type::real(2.0) );
}

PULSE_GLOBAL void PC3::Kernel::Compute::gp_scalar_nonlinear( int i, Solver::VKernelArguments time, Solver::KernelArguments args, Solver::InputOutput io ) {
    
    GET_THREAD_INDEX( i, args.p.N2 );
    
    const Type::complex in_wf = io.in_wf_plus[i];
    const Type::complex in_rv = io.in_rv_plus[i];
    const Type::real in_psi_norm = CUDA::abs2( in_wf );
    
    // MARK: Wavefunction
    Type::complex result = {args.p.g_c * in_psi_norm, -args.p.h_bar_s * Type::real(0.5) * args.p.gamma_c};

    for (int k = 0; k < args.potential_pointers.n; k++) {
        const Type::complex potential = args.dev_ptrs.potential_plus[k][i] * args.potential_pointers.amp[k]; //CUDA::gaussian_oscillator(t, args.potential_pointers.t0[k], args.potential_pointers.sigma[k], args.potential_pointers.freq[k]);
        result += potential;
    }

    result += args.p.g_r * in_rv;
    result += args.p.i * args.p.h_bar_s * Type::real(0.5) * args.p.R * in_rv;

    // MARK: Stochastic
    if (args.p.stochastic_amplitude > 0.0) {
        const Type::complex dw = args.dev_ptrs.random_number[i] * CUDA::sqrt( ( args.p.R * in_rv + args.p.gamma_c ) / (Type::real(4.0) * args.p.dV) );
        result -= args.p.g_c / args.p.dV;
    }

    io.out_wf_plus[i] = in_wf * CUDA::exp(args.p.minus_i_over_h_bar_s * result * time.dt);

    // MARK: Reservoir
    result = -args.p.gamma_r * in_rv;
    result -= args.p.R * in_psi_norm * in_rv;
    for (int k = 0; k < args.pump_pointers.n; k++) {
        result += args.dev_ptrs.pump_plus[k][i] * args.pump_pointers.amp[k]; //CUDA::gaussian_oscillator(t, args.pump_pointerst0[k], args.pump_pointerssigma[k], args.pump_pointersfreq[k]);
    }
    // MARK: Stochastic-2
    if (args.p.stochastic_amplitude > 0.0)
        result += args.p.R * in_rv / args.p.dV;
    io.out_rv_plus[i] = in_rv + result * time.dt;
}

PULSE_GLOBAL void PC3::Kernel::Compute::gp_scalar_independent( int i, Solver::VKernelArguments time, Solver::KernelArguments args, Solver::InputOutput io ) {
    
    GET_THREAD_INDEX( i, args.p.N2 );
    Type::complex result = {0.0,0.0};

    // MARK: Pulse
    for (int k = 0; k < args.pulse_pointers.n; k++) {
        const Type::complex pulse = args.dev_ptrs.pulse_plus[k][i];
        result += args.p.minus_i_over_h_bar_s * time.dt * pulse * args.pulse_pointers.amp[k]; //CUDA::gaussian_complex_oscillator(t, args.pulse_pointers.t0[k], args.pulse_pointers.sigma[k], args.pulse_pointers.freq[k]);
    }
    
    // MARK: Stochastic
    if (args.p.stochastic_amplitude > 0.0) {
        const Type::complex in_rv = io.in_rv_plus[i];
        const Type::complex dw = args.dev_ptrs.random_number[i] * CUDA::sqrt( ( args.p.R * in_rv + args.p.gamma_c ) / (Type::real(4.0) * args.p.dV) );
        result += dw;
    }
    io.out_wf_plus[i] = io.in_wf_plus[i] + result;
}