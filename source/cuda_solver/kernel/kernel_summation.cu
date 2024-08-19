#include "cuda/typedef.cuh"
#include "kernel/kernel_compute.cuh"
#include "kernel/kernel_index_overwrite.cuh"

// Summs one K. TODO: remove, redundant
PULSE_GLOBAL void PC3::Kernel::RK::runge_sum_to_input_ki( int i, Solver::KernelArguments args, Solver::InputOutput io ) {
    OVERWRITE_THREAD_INDEX(i);

    io.out_wf_plus[i] = args.dev_ptrs.wavefunction_plus[i] + args.dt * io.in_wf_plus[i];
    io.out_rv_plus[i] = args.dev_ptrs.reservoir_plus[i] + args.dt * io.in_rv_plus[i];
    if ( not args.p.use_twin_mode ) 
        return;
    io.out_wf_minus[i] = args.dev_ptrs.wavefunction_minus[i] + args.dt * io.in_wf_minus[i];
    io.out_rv_minus[i] = args.dev_ptrs.reservoir_minus[i] + args.dt * io.in_rv_minus[i];
}

// Sums all Ks with weights. Oh yes, this looks terrible. For this to be pretty, we would need to create 
// yet another struct that holds all the buffers in an array. OR: we do the smart thing and restructure
// the original dev_ptrs struct to hold all the buffers in an array. This would make the code much more
// readable and maintainable. TODO
PULSE_GLOBAL void PC3::Kernel::RK::runge_sum_to_input_kw( int i, Solver::KernelArguments args, Solver::InputOutput io, RK::Weights weights ) {
    OVERWRITE_THREAD_INDEX(i);
    Type::complex wf = 0.0;
    Type::complex rv = 0.0;
    Type::complex dw = 0.0;
    if (args.p.stochastic_amplitude > 0.0) {
        dw = args.dev_ptrs.random_number[i] * CUDA::sqrt( ( args.p.R * args.dev_ptrs. + args.p.gamma_c ) / (Type::real(4.0) * args.p.dV) );
    }
    for (int n = weights.start; n < weights.n; n++) {
        const auto w = weights.weights[n];
        switch (n) { 
            case 0: wf += w * args.dev_ptrs.k1_wavefunction_plus[i]; rv += w * args.dev_ptrs.k1_reservoir_plus[i]; break;
            case 1: wf += w * args.dev_ptrs.k2_wavefunction_plus[i]; rv += w * args.dev_ptrs.k2_reservoir_plus[i]; break;
            case 2: wf += w * args.dev_ptrs.k3_wavefunction_plus[i]; rv += w * args.dev_ptrs.k3_reservoir_plus[i]; break;
            case 3: wf += w * args.dev_ptrs.k4_wavefunction_plus[i]; rv += w * args.dev_ptrs.k4_reservoir_plus[i]; break;
            case 4: wf += w * args.dev_ptrs.k5_wavefunction_plus[i]; rv += w * args.dev_ptrs.k5_reservoir_plus[i]; break;
            case 5: wf += w * args.dev_ptrs.k6_wavefunction_plus[i]; rv += w * args.dev_ptrs.k6_reservoir_plus[i]; break;
            case 6: wf += w * args.dev_ptrs.k7_wavefunction_plus[i]; rv += w * args.dev_ptrs.k7_reservoir_plus[i]; break;
            case 7: wf += w * args.dev_ptrs.k8_wavefunction_plus[i]; rv += w * args.dev_ptrs.k8_reservoir_plus[i]; break;
            case 8: wf += w * args.dev_ptrs.k9_wavefunction_plus[i]; rv += w * args.dev_ptrs.k9_reservoir_plus[i]; break;
            case 9: wf += w * args.dev_ptrs.k10_wavefunction_plus[i]; rv += w * args.dev_ptrs.k10_reservoir_plus[i]; break;
        }
        wf += w*dw;
    }
    
    io.out_wf_plus[i] = io.in_wf_plus[i] + args.dt * wf;
    io.out_rv_plus[i] = io.in_rv_plus[i] + args.dt * rv;
    
    if ( not args.p.use_twin_mode ) 
        return;
    
    wf = 0.0;
    rv = 0.0;
    if (args.p.stochastic_amplitude > 0.0) {
        dw = args.dev_ptrs.random_number[i] * CUDA::sqrt( ( args.p.R * args.dev_ptrs. + args.p.gamma_c ) / (Type::real(4.0) * args.p.dV) );
    }
    for (int n = weights.start; n < weights.n; n++) {
        const auto w = weights.weights[n];
        switch (n) {
            case 0: wf += w * args.dev_ptrs.k1_wavefunction_minus[i]; rv += w * args.dev_ptrs.k1_reservoir_minus[i]; break;
            case 1: wf += w * args.dev_ptrs.k2_wavefunction_minus[i]; rv += w * args.dev_ptrs.k2_reservoir_minus[i]; break;
            case 2: wf += w * args.dev_ptrs.k3_wavefunction_minus[i]; rv += w * args.dev_ptrs.k3_reservoir_minus[i]; break;
            case 3: wf += w * args.dev_ptrs.k4_wavefunction_minus[i]; rv += w * args.dev_ptrs.k4_reservoir_minus[i]; break;
            case 4: wf += w * args.dev_ptrs.k5_wavefunction_minus[i]; rv += w * args.dev_ptrs.k5_reservoir_minus[i]; break;
            case 5: wf += w * args.dev_ptrs.k6_wavefunction_minus[i]; rv += w * args.dev_ptrs.k6_reservoir_minus[i]; break;
            case 6: wf += w * args.dev_ptrs.k7_wavefunction_minus[i]; rv += w * args.dev_ptrs.k7_reservoir_minus[i]; break;
            case 7: wf += w * args.dev_ptrs.k8_wavefunction_minus[i]; rv += w * args.dev_ptrs.k8_reservoir_minus[i]; break;
            case 8: wf += w * args.dev_ptrs.k9_wavefunction_minus[i]; rv += w * args.dev_ptrs.k9_reservoir_minus[i]; break;
            case 9: wf += w * args.dev_ptrs.k10_wavefunction_minus[i]; rv += w * args.dev_ptrs.k10_reservoir_minus[i]; break;
        }
        wf += w*dw;
    }
    
    io.out_wf_minus[i] = io.in_wf_minus[i] + args.dt * wf;
    io.out_rv_minus[i] = io.in_rv_minus[i] + args.dt * rv;
}


PULSE_GLOBAL void PC3::Kernel::RK::runge_sum_to_error( int i, Solver::KernelArguments args, Solver::InputOutput io, RK::Weights weights ) {
    OVERWRITE_THREAD_INDEX(i);
    // The first weigth is for the input wavefunction, the rest are for the Ks
    Type::complex wf = weights.weights[0] * args.dev_ptrs.buffer_wavefunction_plus[i];
    for (int n = 1; n < weights.n; n++) {
        const auto w = weights.weights[n];
        switch (n) { 
            case 1: wf += w * args.dev_ptrs.k1_wavefunction_plus[i]; break;
            case 2: wf += w * args.dev_ptrs.k2_wavefunction_plus[i]; break;
            case 3: wf += w * args.dev_ptrs.k3_wavefunction_plus[i]; break;
            case 4: wf += w * args.dev_ptrs.k4_wavefunction_plus[i]; break;
            case 5: wf += w * args.dev_ptrs.k5_wavefunction_plus[i]; break;
            case 6: wf += w * args.dev_ptrs.k6_wavefunction_plus[i]; break;
            case 7: wf += w * args.dev_ptrs.k7_wavefunction_plus[i]; break;
            case 8: wf += w * args.dev_ptrs.k8_wavefunction_plus[i]; break;
            case 9: wf += w * args.dev_ptrs.k9_wavefunction_plus[i]; break;
            case 10: wf += w * args.dev_ptrs.k10_wavefunction_plus[i]; break;
        }
    }
    
    args.dev_ptrs.rk_error[i] = CUDA::abs2(args.dt * wf);
    if ( not args.p.use_twin_mode ) 
        return;
    
    wf = weights.weights[0] * args.dev_ptrs.buffer_wavefunction_minus[i];
    for (int n = 1; n < weights.n; n++) {
        const auto w = weights.weights[n];
        switch (n) {
            case 1: wf += w * args.dev_ptrs.k1_wavefunction_minus[i]; break;
            case 2: wf += w * args.dev_ptrs.k2_wavefunction_minus[i]; break;
            case 3: wf += w * args.dev_ptrs.k3_wavefunction_minus[i]; break;
            case 4: wf += w * args.dev_ptrs.k4_wavefunction_minus[i]; break;
            case 5: wf += w * args.dev_ptrs.k5_wavefunction_minus[i]; break;
            case 6: wf += w * args.dev_ptrs.k6_wavefunction_minus[i]; break;
            case 7: wf += w * args.dev_ptrs.k7_wavefunction_minus[i]; break;
            case 8: wf += w * args.dev_ptrs.k8_wavefunction_minus[i]; break;
            case 9: wf += w * args.dev_ptrs.k9_wavefunction_minus[i]; break;
            case 10: wf += w * args.dev_ptrs.k10_wavefunction_minus[i]; break;
        }
    }
    
    args.dev_ptrs.rk_error[i] += args.p.i*CUDA::abs2(args.dt * wf);
}