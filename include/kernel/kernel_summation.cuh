#pragma once
#include "cuda/typedef.cuh"
#include "solver/gpu_solver.hpp"

namespace PC3::Kernel::Summation {

template <Type::uint32 NMax, Type::uint32 N, float w, float... W>
PULSE_DEVICE PULSE_INLINE void sum_single_k( Type::uint32 i, Type::complex& dw, Type::complex& wf, Type::complex& rv, Type::complex* k_wavefunction, Type::complex* k_reservoir,
                                             Type::uint32 offset ) {
    if constexpr ( w != 0.0 ) {
        wf += w * ( k_wavefunction[i + offset * ( NMax - N )] + dw );
        rv += w * k_reservoir[i + offset * ( NMax - N )];
    }
    if constexpr ( sizeof...( W ) > 0 ) {
        sum_single_k<NMax, N - 1, W...>( i, dw, wf, rv, k_wavefunction, k_reservoir, offset );
    }
}

// Specifically use Type::uint32 N instead of sizeof(Weights) to force MSVC to NOT inline this function for different solvers (RK3,RK4) which cases the respective RK solver to call the wrong template function.
template <Type::uint32 N, float... Weights>
PULSE_GLOBAL void runge_sum_to_input_kw( Type::uint32 i, Type::uint32 current_halo, Solver::VKernelArguments time, Solver::KernelArguments args, Solver::InputOutput io ) {
    GENERATE_SUBGRID_INDEX( i, current_halo );
    Type::complex wf = 0.0;
    Type::complex rv = 0.0;
    Type::complex dw = 0.0;
    if ( args.p.stochastic_amplitude > 0.0 ) {
        dw = args.dev_ptrs.random_number[i] * CUDA::sqrt( ( args.p.R * args.dev_ptrs.reservoir_plus[i] + args.p.gamma_c ) / ( Type::real( 4.0 ) * args.p.dV ) ) / time.dt;
    }

    sum_single_k<N, N, Weights...>( i, dw, wf, rv, args.dev_ptrs.k_wavefunction_plus, args.dev_ptrs.k_reservoir_plus,
                                                                          args.p.subgrid_N2_with_halo );

    io.out_wf_plus[i] = io.in_wf_plus[i] + time.dt * wf;
    io.out_rv_plus[i] = io.in_rv_plus[i] + time.dt * rv;

    if ( not args.p.use_twin_mode )
        return;

    wf = 0.0;
    rv = 0.0;
    if ( args.p.stochastic_amplitude > 0.0 ) {
        dw = args.dev_ptrs.random_number[i] * CUDA::sqrt( ( args.p.R * args.dev_ptrs.reservoir_minus[i] + args.p.gamma_c ) / ( Type::real( 4.0 ) * args.p.dV ) );
    }

    sum_single_k<N, N, Weights...>( i, dw, wf, rv, args.dev_ptrs.k_wavefunction_minus, args.dev_ptrs.k_reservoir_minus,
                                                                          args.p.subgrid_N2_with_halo );

    io.out_wf_minus[i] = io.in_wf_minus[i] + time.dt * wf;
    io.out_rv_minus[i] = io.in_rv_minus[i] + time.dt * rv;
}

template <int NMax, int N, float w, float... W>
PULSE_DEVICE void sum_single_error_k( int i, Type::complex& error, Type::complex* k_wavefunction, Type::uint32 offset ) {
    if constexpr ( w != 0.0 ) {
        error += w * k_wavefunction[i + offset * ( NMax - N )];
    }
    if constexpr ( sizeof...( W ) > 0 ) {
        sum_single_error_k<NMax, N - 1, W...>( i, error, k_wavefunction, offset );
    }
}

template <float... Weights>
PULSE_GLOBAL void runge_sum_to_error( int i, Type::uint32 current_halo, Solver::VKernelArguments time, Solver::KernelArguments args ) {
    GENERATE_SUBGRID_INDEX( i, current_halo );

    Type::complex error = 0.0;

    sum_single_error_k<sizeof...( Weights ), sizeof...( Weights ), Weights...>( i, error, args.dev_ptrs.k_wavefunction_plus, args.p.subgrid_N2_with_halo );

    args.dev_ptrs.rk_error[i] = CUDA::abs2( time.dt * error );
    if ( not args.p.use_twin_mode )
        return;

    error = 0.0;

    sum_single_error_k<sizeof...( Weights ), sizeof...( Weights ), Weights...>( i, error, args.dev_ptrs.k_wavefunction_minus, args.p.subgrid_N2_with_halo );

    args.dev_ptrs.rk_error[i] += args.p.i * CUDA::abs2( time.dt * error );
}
} // namespace PC3::Kernel::Summation