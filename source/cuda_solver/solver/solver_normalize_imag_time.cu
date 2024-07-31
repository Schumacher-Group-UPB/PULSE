#include <iostream>
#include "cuda/typedef.cuh"
#include "solver/gpu_solver.hpp"
#include "kernel/kernel_compute.cuh"
#include "misc/helperfunctions.hpp"

void PC3::Solver::normalizeImaginaryTimePropagation( dim3 block_size, dim3 grid_size ) {
    
    // Calculate min and max values
    auto [minimum_plus, maximum_plus] = CUDA::minmax( matrix.wavefunction_plus.getDevicePtr(), system.p.N2, true /*is a device pointer*/ );
    auto [minimum_plus_r, maximum_plus_r] = CUDA::minmax( matrix.reservoir_plus.getDevicePtr(), system.p.N2, true /*is a device pointer*/ );
    
    maximum_plus = std::abs(maximum_plus);
    maximum_plus_r = std::abs(maximum_plus_r);
    maximum_plus = maximum_plus > 1e-10 ? maximum_plus : 1.0;
    maximum_plus_r = maximum_plus_r > 1e-10 ? maximum_plus_r : 1.0;

    #ifdef USE_CPU
        std::ranges::transform(matrix.wavefunction_plus.dbegin(), matrix.wavefunction_plus.dend(), matrix.wavefunction_plus.dbegin(), [maximum_plus](Type::complex val) { return val / maximum_plus; });
        std::ranges::transform(matrix.reservoir_plus.dbegin(), matrix.reservoir_plus.dend(), matrix.reservoir_plus.dbegin(), [maximum_plus_r](Type::complex val) { return val / maximum_plus_r; });
    #else
        //CUDA_DEVICE auto transform_function_plus = [maximum_plus,maximum_plus_r](Type::complex val, Type::real c) { return val/c; };
        thrust::transform(matrix.wavefunction_plus.dbegin(), matrix.wavefunction_plus.dend(), matrix.wavefunction_plus.dbegin(), thrust::placeholders::_1 / maximum_plus);
        thrust::transform(matrix.reservoir_plus.dbegin(), matrix.reservoir_plus.dend(), matrix.reservoir_plus.dbegin(), thrust::placeholders::_1 / maximum_plus_r);
    #endif

    if (not system.p.use_twin_mode)
        return;

    // Calculate min and max values
    auto [minimum_minus, maximum_minus] = CUDA::minmax( matrix.wavefunction_minus.getDevicePtr(), system.p.N2, true /*is a device pointer*/ );
    auto [minimum_minus_r, maximum_minus_r] = CUDA::minmax( matrix.reservoir_minus.getDevicePtr(), system.p.N2, true /*is a device pointer*/ );
    
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