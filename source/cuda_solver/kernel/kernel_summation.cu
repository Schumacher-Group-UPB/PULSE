#include "cuda/typedef.cuh"
#include "kernel/kernel_compute.cuh"
#include "kernel/kernel_index_overwrite.cuh"
#include "solver/gpu_solver.hpp"

// TODO: change to templated kernel with weights as constepxr template.

PULSE_GLOBAL void PC3::Kernel::RK::runge_sum_to_input_kw( int i, Type::uint current_halo, Solver::VKernelArguments time, Solver::KernelArguments args, Solver::InputOutput io, RK::Weights weights ) {
    GENERATE_SUBGRID_INDEX(i, current_halo);
    Type::complex wf = 0.0;
    Type::complex rv = 0.0;
    Type::complex dw = 0.0;
    if (args.p.stochastic_amplitude > 0.0) {
        dw = args.dev_ptrs.random_number[i] * CUDA::sqrt( ( args.p.R * args.dev_ptrs.reservoir_plus[i] + args.p.gamma_c ) / (Type::real(4.0) * args.p.dV) );
    }

    //#pragma unroll MAX_K_VECTOR_SIZE
    //for (int n = 0; n < weights.n; n++) {
    //    const Type::real w = weights.weights[n];
    //    if (w == 0.0) 
    //        continue;
    //    wf += w * args.dev_ptrs.k_wavefunction_plus[n][i]; 
    //    wf += w*dw / time.dt;
    //    rv += w * args.dev_ptrs.k_reservoir_plus[n][i];
    //}
    for (int n = weights.start; n < weights.n; n++) {
        const Type::real w = weights.weights[n];
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
        wf += w*dw / time.dt;
    }
    
    io.out_wf_plus[i] = io.in_wf_plus[i] + time.dt * wf;
    io.out_rv_plus[i] = io.in_rv_plus[i] + time.dt * rv;
    
    if ( not args.p.use_twin_mode ) 
        return;
    
    wf = 0.0;
    rv = 0.0;
    if (args.p.stochastic_amplitude > 0.0) {
        dw = args.dev_ptrs.random_number[i] * CUDA::sqrt( ( args.p.R * args.dev_ptrs.reservoir_minus[i] + args.p.gamma_c ) / (Type::real(4.0) * args.p.dV) );
    }

    //#pragma unroll MAX_K_VECTOR_SIZE
    //for (int n = 0; n < weights.n; n++) {
    //    const Type::real w = weights.weights[n];
    //    if (w == 0.0) 
    //        continue;
    //    wf += w * args.dev_ptrs.k_wavefunction_minus[n][i]; 
    //    wf += w*dw / time.dt;
    //    rv += w * args.dev_ptrs.k_reservoir_minus[n][i];
    //}

    for (int n = weights.start; n < weights.n; n++) {
        const Type::real w = weights.weights[n];
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
        wf += w*dw / time.dt;
    }
    
    io.out_wf_minus[i] = io.in_wf_minus[i] + time.dt * wf;
    io.out_rv_minus[i] = io.in_rv_minus[i] + time.dt * rv;
}


PULSE_GLOBAL void PC3::Kernel::RK::runge_sum_to_error( int i, Type::uint current_halo, Solver::VKernelArguments time, Solver::KernelArguments args, Solver::InputOutput io, RK::Weights weights ) {
    GENERATE_SUBGRID_INDEX(i, current_halo);
    
    // The first weigth is for the input wavefunction, the rest are for the Ks
    Type::complex wf = weights.weights[0] * args.dev_ptrs.buffer_wavefunction_plus[i];

    //#pragma unroll MAX_K_VECTOR_SIZE
    //for (int n = 1; n < weights.n; n++) {
    //    const auto w = weights.weights[n];
    //    wf += w * args.dev_ptrs.k_wavefunction_plus[n][i];
    //}
    
    args.dev_ptrs.rk_error[i] = CUDA::abs2(time.dt * wf);
    if ( not args.p.use_twin_mode ) 
        return;
    
    wf = weights.weights[0] * args.dev_ptrs.buffer_wavefunction_minus[i];

    //#pragma unroll MAX_K_VECTOR_SIZE
    //for (int n = 1; n < weights.n; n++) {
    //    const auto w = weights.weights[n];
    //    wf += w * args.dev_ptrs.k_wavefunction_minus[n][i];
    //}
    
    args.dev_ptrs.rk_error[i] += args.p.i*CUDA::abs2(time.dt * wf);
}