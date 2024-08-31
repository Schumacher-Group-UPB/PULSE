#include "cuda/typedef.cuh"
#include "kernel/kernel_compute.cuh"
#include "kernel/kernel_index_overwrite.cuh"
#include "solver/gpu_solver.hpp"

// Sums all Ks with weights. Oh yes, this looks terrible. For this to be pretty, we would need to create 
// yet another struct that holds all the buffers in an array. OR: we do the smart thing and restructure
// the original dev_ptrs struct to hold all the buffers in an array. This would make the code much more
// readable and maintainable. TODO
PULSE_GLOBAL void PC3::Kernel::RK::runge_sum_to_input_kw( int i, Type::uint current_halo, Solver::VKernelArguments time, Solver::KernelArguments args, Solver::InputOutput io, RK::Weights weights ) {
    GENERATE_SUBGRID_INDEX(i, current_halo);
    Type::complex wf = 0.0;
    Type::complex rv = 0.0;
    Type::complex dw = 0.0;
    if (args.p.stochastic_amplitude > 0.0) {
        dw = args.dev_ptrs.random_number[i] * CUDA::sqrt( ( args.p.R * args.dev_ptrs.reservoir_plus[i] + args.p.gamma_c ) / (Type::real(4.0) * args.p.dV) );
    }

    #pragma unroll MAX_K_VECTOR_SIZE
    for (int n = 0; n < weights.n; n++) {
        const Type::real w = weights.weights[n];
        if (w == 0.0) 
            continue;
        wf += w * args.dev_ptrs.k_wavefunction_plus[n][i]; 
        wf += w*dw / time.dt;
        rv += w * args.dev_ptrs.k_reservoir_plus[n][i];
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

    #pragma unroll MAX_K_VECTOR_SIZE
    for (int n = 0; n < weights.n; n++) {
        const Type::real w = weights.weights[n];
        if (w == 0.0) 
            continue;
        wf += w * args.dev_ptrs.k_wavefunction_minus[n][i]; 
        wf += w*dw / time.dt;
        rv += w * args.dev_ptrs.k_reservoir_minus[n][i];
    }
    
    io.out_wf_minus[i] = io.in_wf_minus[i] + time.dt * wf;
    io.out_rv_minus[i] = io.in_rv_minus[i] + time.dt * rv;
}


PULSE_GLOBAL void PC3::Kernel::RK::runge_sum_to_error( int i, Type::uint current_halo, Solver::VKernelArguments time, Solver::KernelArguments args, Solver::InputOutput io, RK::Weights weights ) {
    GENERATE_SUBGRID_INDEX(i, current_halo);
    
    // The first weigth is for the input wavefunction, the rest are for the Ks
    Type::complex wf = weights.weights[0] * args.dev_ptrs.buffer_wavefunction_plus[i];

    #pragma unroll MAX_K_VECTOR_SIZE
    for (int n = 1; n < weights.n; n++) {
        const auto w = weights.weights[n];
        wf += w * args.dev_ptrs.k_wavefunction_plus[n][i];
    }
    
    args.dev_ptrs.rk_error[i] = CUDA::abs2(time.dt * wf);
    if ( not args.p.use_twin_mode ) 
        return;
    
    wf = weights.weights[0] * args.dev_ptrs.buffer_wavefunction_minus[i];

    #pragma unroll MAX_K_VECTOR_SIZE
    for (int n = 1; n < weights.n; n++) {
        const auto w = weights.weights[n];
        wf += w * args.dev_ptrs.k_wavefunction_minus[n][i];
    }
    
    args.dev_ptrs.rk_error[i] += args.p.i*CUDA::abs2(time.dt * wf);
}