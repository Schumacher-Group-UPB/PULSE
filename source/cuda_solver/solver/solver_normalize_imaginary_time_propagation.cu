#include "cuda/cuda_complex.cuh"
#include "cuda/cuda_macro.cuh"
#include "solver/gpu_solver.hpp"
#include "kernel/kernel_normalize_imaginary_time_propagation.cuh"
#include "misc/helperfunctions.hpp"
#include <iostream>
void PC3::Solver::normalizeImaginaryTimePropagation( MatrixContainer::Pointers device_pointers, System::Parameters p, dim3 block_size, dim3 grid_size ) {
    if (not system.imaginary_time)
        return;
        
    // Calculate min and max values
    auto [minimum_plus, maximum_plus] = CUDA::minmax( device_pointers.buffer_wavefunction_plus, system.p.N2, true /*is a device pointer*/ );
    auto [minimum_plus_r, maximum_plus_r] = CUDA::minmax( device_pointers.buffer_reservoir_plus, system.p.N2, true /*is a device pointer*/ );
    real_number maximum_minus, maximum_minus_r;
    if (system.p.use_twin_mode) {
        real_number dummy;
        std::tie(dummy, maximum_minus) = CUDA::minmax( device_pointers.buffer_wavefunction_minus, system.p.N2, true /*is a device pointer*/ );
        std::tie(dummy, maximum_minus_r) = CUDA::minmax( device_pointers.buffer_reservoir_minus, system.p.N2, true /*is a device pointer*/ );
    }
    
    complex_number wf =  {CUDA::abs2(maximum_plus), CUDA::abs2(maximum_minus)};
    complex_number rv =  {CUDA::abs2(maximum_plus_r), CUDA::abs2(maximum_minus_r)};

    CALL_KERNEL(
        Kernel::normalize_imaginary_time_propagation, "Imag Time", grid_size, block_size,
        device_pointers, p, wf, rv
    );
    
}