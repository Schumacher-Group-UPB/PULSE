
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
#include "misc/helperfunctions.hpp"

void PC3::Solver::normalizeImaginaryTimePropagation( dim3 block_size, dim3 grid_size ) {
    
    // Calculate min and max values
    #ifdef USE_CPU
        Type::complex sum_psi_plus = std::transform_reduce( matrix.wavefunction_plus.dbegin(), matrix.wavefunction_plus.dend(), PC3::SquareReduction(), Type::real(0.0), std::plus<Type::real>() );
        Type::complex sum_res_plus = std::transform_reduce( matrix.reservoir_plus.dbegin(), matrix.reservoir_plus.dend(), PC3::SquareReduction(), Type::real(0.0), std::plus<Type::real>() );
    #else
        Type::complex sum_psi_plus = thrust::transform_reduce( matrix.wavefunction_plus.dbegin(),matrix.wavefunction_plus.dend(), PC3::SquareReduction(), Type::real(0.0), thrust::plus<Type::real>() );
        Type::complex sum_res_plus = thrust::transform_reduce( matrix.reservoir_plus.dbegin(),matrix.reservoir_plus.dend(), PC3::SquareReduction(), Type::real(0.0), thrust::plus<Type::real>() );
    #endif

    if ( PC3::CUDA::abs(sum_psi_plus) < 1e-10 )
        sum_psi_plus = 1.0;
    if ( PC3::CUDA::abs(sum_res_plus) < 1e-10 )
        sum_res_plus = 1.0;

    sum_psi_plus *= system.p.dV;
    sum_res_plus *= system.p.dV;

    #ifdef USE_CPU
        std::ranges::transform(matrix.wavefunction_plus.dbegin(), matrix.wavefunction_plus.dend(), matrix.wavefunction_plus.dbegin(), [sum_psi_plus](Type::complex val) { return val / sum_psi_plus; });
        std::ranges::transform(matrix.reservoir_plus.dbegin(), matrix.reservoir_plus.dend(), matrix.reservoir_plus.dbegin(), [sum_res_plus](Type::complex val) { return val / sum_res_plus; });
    #else
        thrust::transform(matrix.wavefunction_plus.dbegin(), matrix.wavefunction_plus.dend(), matrix.wavefunction_plus.dbegin(), thrust::placeholders::_1 / sum_psi_plus);
        thrust::transform(matrix.reservoir_plus.dbegin(), matrix.reservoir_plus.dend(), matrix.reservoir_plus.dbegin(), thrust::placeholders::_1 / sum_res_plus);
    #endif

    if (not system.p.use_twin_mode)
        return;

    // Calculate min and max values
    #ifdef USE_CPU
        Type::complex sum_psi_minus = std::transform_reduce( matrix.wavefunction_minus.dbegin(), matrix.wavefunction_minus.dend(), PC3::SquareReduction(), Type::real(0.0), std::plus<Type::real>() );
        Type::complex sum_res_minus = std::transform_reduce( matrix.reservoir_minus.dbegin(), matrix.reservoir_minus.dend(), PC3::SquareReduction(), Type::real(0.0), std::plus<Type::real>() );
    #else
        Type::complex sum_psi_minus = thrust::transform_reduce( matrix.wavefunction_minus.dbegin(),matrix.wavefunction_minus.dend(), PC3::SquareReduction(), Type::real(0.0), thrust::plus<Type::real>() );
        Type::complex sum_res_minus = thrust::transform_reduce( matrix.reservoir_minus.dbegin(),matrix.reservoir_minus.dend(), PC3::SquareReduction(), Type::real(0.0), thrust::plus<Type::real>() );
    #endif

    if ( PC3::CUDA::abs(sum_psi_minus) < 1e-10 )
        sum_psi_minus = 1.0;
    if ( PC3::CUDA::abs(sum_res_minus) < 1e-10 )
        sum_res_minus = 1.0;

    sum_psi_minus *= system.p.dV;
    sum_res_minus *= system.p.dV;

    #ifdef USE_CPU
        std::ranges::transform(matrix.wavefunction_minus.dbegin(), matrix.wavefunction_minus.dend(), matrix.wavefunction_minus.dbegin(), [sum_psi_minus](Type::complex val) { return val / sum_psi_minus; });
        std::ranges::transform(matrix.reservoir_minus.dbegin(), matrix.reservoir_minus.dend(), matrix.reservoir_minus.dbegin(), [sum_res_minus](Type::complex val) { return val / sum_res_minus; });
    #else
        thrust::transform(matrix.wavefunction_plus.dbegin(), matrix.wavefunction_plus.dend(), matrix.wavefunction_plus.dbegin(), thrust::placeholders::_1 / sum_psi_minus);
        thrust::transform(matrix.reservoir_plus.dbegin(), matrix.reservoir_plus.dend(), matrix.reservoir_plus.dbegin(), thrust::placeholders::_1 / sum_res_minus);
    #endif

}

/*
void PC3::Solver::normalizeImaginaryTimePropagation( dim3 block_size, dim3 grid_size ) {
    
    // Calculate min and max values
    auto [minimum_plus, maximum_plus] = CUDA::minmax( matrix.wavefunction_plus.getDevicePtr(), system.p.N2, true );
    auto [minimum_plus_r, maximum_plus_r] = CUDA::minmax( matrix.reservoir_plus.getDevicePtr(), system.p.N2, true );
    
    maximum_plus = std::abs(maximum_plus);
    maximum_plus_r = std::abs(maximum_plus_r);
    maximum_plus = maximum_plus > 1e-10 ? maximum_plus : 1.0;
    maximum_plus_r = maximum_plus_r > 1e-10 ? maximum_plus_r : 1.0;

    #ifdef USE_CPU
        std::ranges::transform(matrix.wavefunction_plus.dbegin(), matrix.wavefunction_plus.dend(), matrix.wavefunction_plus.dbegin(), [maximum_plus](Type::complex val) { return val / maximum_plus; });
        std::ranges::transform(matrix.reservoir_plus.dbegin(), matrix.reservoir_plus.dend(), matrix.reservoir_plus.dbegin(), [maximum_plus_r](Type::complex val) { return val / maximum_plus_r; });
    #else
        //CUDA_DEVICE auto transform_function_plus = [maximum_plus,maximum_plus_r](Type::complex val, Type::real c) { return val/c; };
        //thrust::transform(matrix.wavefunction_plus.dbegin(), matrix.wavefunction_plus.dend(), matrix.wavefunction_plus.dbegin(), thrust::placeholders::_1 / maximum_plus);
        //thrust::transform(matrix.reservoir_plus.dbegin(), matrix.reservoir_plus.dend(), matrix.reservoir_plus.dbegin(), thrust::placeholders::_1 / maximum_plus_r);
        Type::complex sum_psi_plus = thrust::reduce( matrix.wavefunction_plus.dbegin(),matrix.wavefunction_plus.dend(), Type::complex(0.0), thrust::plus<Type::complex>() );
        Type::complex sum_res_plus = thrust::reduce( matrix.reservoir_plus.dbegin(),matrix.reservoir_plus.dend(), Type::complex(0.0), thrust::plus<Type::complex>() );
    #endif

    if (not system.p.use_twin_mode)
        return;

    // Calculate min and max values
    auto [minimum_minus, maximum_minus] = CUDA::minmax( matrix.wavefunction_minus.getDevicePtr(), system.p.N2, true );
    auto [minimum_minus_r, maximum_minus_r] = CUDA::minmax( matrix.reservoir_minus.getDevicePtr(), system.p.N2, true );
    
    maximum_minus = std::abs(maximum_minus);
    maximum_minus_r = std::abs(maximum_minus_r);
    maximum_minus = maximum_minus > 1e-10 ? maximum_minus : 1.0;
    maximum_minus_r = maximum_minus_r > 1e-10 ? maximum_minus_r : 1.0;

    #ifdef USE_CPU
        std::ranges::transform(matrix.wavefunction_minus.dbegin(), matrix.wavefunction_minus.dend(), matrix.wavefunction_minus.dbegin(), [maximum_minus](Type::complex val) { return val / maximum_minus; });
        std::ranges::transform(matrix.reservoir_minus.dbegin(), matrix.reservoir_minus.dend(), matrix.reservoir_minus.dbegin(), [maximum_minus_r](Type::complex val) { return val / maximum_minus_r; });
    #else
        //CUDA_DEVICE auto transform_function_minus = [maximum_minus,maximum_minus_r](Type::complex val, Type::real c) { return val/c; };
        thrust::transform(matrix.wavefunction_minus.dbegin(), matrix.wavefunction_minus.dend(), matrix.wavefunction_minus.dbegin(), thrust::placeholders::_1 / maximum_minus);
        thrust::transform(matrix.reservoir_minus.dbegin(), matrix.reservoir_minus.dend(), matrix.reservoir_minus.dbegin(), thrust::placeholders::_1 / maximum_minus_r);
    #endif

}
*/