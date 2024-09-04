
#include "cuda/typedef.cuh"
#ifdef USE_CUDA
    #include <thrust/reduce.h>
    #include <thrust/transform_reduce.h>
    #include <thrust/execution_policy.h>
#else
    #include <numeric>
#endif
#include <iostream>
#include "solver/gpu_solver.hpp"
#include "kernel/kernel_compute.cuh"

// TODO: implement with subgridding
// For now, we can simply call matrix.toFull().dbegin(), but this will be slow, because it will call the subgrid->fullgrid kernel every time
// TODO: implement transform_reduce with subgrids and then call matrix.transform_reduce() and matrix.reduce() 

void PC3::Solver::normalizeImaginaryTimePropagation( ) {
    /*
    // Calculate min and max values
    #ifdef USE_CPU
        Type::real sum_psi_plus = std::transform_reduce( matrix.wavefunction_plus.dbegin(), matrix.wavefunction_plus.dend(), Type::real(0.0), std::plus<Type::real>(), PC3::SquareReduction() );
        Type::real sum_res_plus = std::transform_reduce( matrix.reservoir_plus.dbegin(), matrix.reservoir_plus.dend(), Type::real(0.0), std::plus<Type::real>(), PC3::SquareReduction() );
    #else
        Type::real sum_psi_plus = thrust::transform_reduce( matrix.wavefunction_plus.dbegin(),matrix.wavefunction_plus.dend(), PC3::SquareReduction(), Type::real(0.0), thrust::plus<Type::real>() );
        Type::real sum_res_plus = thrust::transform_reduce( matrix.reservoir_plus.dbegin(),matrix.reservoir_plus.dend(), PC3::SquareReduction(), Type::real(0.0), thrust::plus<Type::real>() );
    #endif

    if ( sum_psi_plus < 1e-10 )
        sum_psi_plus = 1.0;
    if ( sum_res_plus < 1e-10 )
        sum_res_plus = 1.0;

    sum_psi_plus = std::sqrt(system.imag_time_amplitude/(sum_psi_plus*system.p.dV));
    sum_res_plus = std::sqrt(system.imag_time_amplitude/(sum_res_plus*system.p.dV));

    #ifdef USE_CPU
        std::ranges::transform(matrix.wavefunction_plus.dbegin(), matrix.wavefunction_plus.dend(), matrix.wavefunction_plus.dbegin(), [sum_psi_plus](Type::complex val) { return val * sum_psi_plus; });
        std::ranges::transform(matrix.reservoir_plus.dbegin(), matrix.reservoir_plus.dend(), matrix.reservoir_plus.dbegin(), [sum_res_plus](Type::complex val) { return val * sum_res_plus; });
    #else
        thrust::transform(matrix.wavefunction_plus.dbegin(), matrix.wavefunction_plus.dend(), matrix.wavefunction_plus.dbegin(), thrust::placeholders::_1 * sum_psi_plus);
        thrust::transform(matrix.reservoir_plus.dbegin(), matrix.reservoir_plus.dend(), matrix.reservoir_plus.dbegin(), thrust::placeholders::_1 * sum_res_plus);
    #endif

    if (not system.p.use_twin_mode)
        return;

    // Calculate min and max values
    #ifdef USE_CPU
        Type::real sum_psi_minus = std::transform_reduce( matrix.wavefunction_minus.dbegin(), matrix.wavefunction_minus.dend(), Type::real(0.0), std::plus<Type::real>(), PC3::SquareReduction() );
        Type::real sum_res_minus = std::transform_reduce( matrix.reservoir_minus.dbegin(), matrix.reservoir_minus.dend(), Type::real(0.0), std::plus<Type::real>(), PC3::SquareReduction() );
    #else
        Type::real sum_psi_minus = thrust::transform_reduce( matrix.wavefunction_minus.dbegin(),matrix.wavefunction_minus.dend(), PC3::SquareReduction(), Type::real(0.0), thrust::plus<Type::real>() );
        Type::real sum_res_minus = thrust::transform_reduce( matrix.reservoir_minus.dbegin(),matrix.reservoir_minus.dend(), PC3::SquareReduction(), Type::real(0.0), thrust::plus<Type::real>() );
    #endif

    if ( sum_psi_minus < 1e-10 )
        sum_psi_minus = 1.0;
    if ( sum_res_minus < 1e-10 )
        sum_res_minus = 1.0;

    sum_psi_minus = std::sqrt(system.imag_time_amplitude/(sum_psi_minus*system.p.dV));
    sum_res_minus = std::sqrt(system.imag_time_amplitude/(sum_res_minus*system.p.dV));

    #ifdef USE_CPU
        std::ranges::transform(matrix.wavefunction_minus.dbegin(), matrix.wavefunction_minus.dend(), matrix.wavefunction_minus.dbegin(), [sum_psi_minus](Type::complex val) { return val * sum_psi_minus; });
        std::ranges::transform(matrix.reservoir_minus.dbegin(), matrix.reservoir_minus.dend(), matrix.reservoir_minus.dbegin(), [sum_res_minus](Type::complex val) { return val * sum_res_minus; });
    #else
        thrust::transform(matrix.wavefunction_plus.dbegin(), matrix.wavefunction_plus.dend(), matrix.wavefunction_plus.dbegin(), thrust::placeholders::_1 * sum_psi_minus);
        thrust::transform(matrix.reservoir_plus.dbegin(), matrix.reservoir_plus.dend(), matrix.reservoir_plus.dbegin(), thrust::placeholders::_1 * sum_res_minus);
    #endif
    */
}