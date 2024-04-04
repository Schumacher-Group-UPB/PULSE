#include "kernel/kernel_runge_function.cuh"
#include "kernel/kernel_hamilton.cuh"
#include "kernel/kernel_index_overwrite.cuh"

CUDA_GLOBAL void PC3::Kernel::Compute::scalar_reservoir( int i, real_number t, Device::Pointers dev_ptrs, System::Parameters p, InputOutput io ) {

    OVERWRITE_THREAD_INDEX( i );

    const complex_number in_wf = io.in_wf_plus[i];
    const complex_number in_rv = io.in_rv_plus[i];
    const complex_number pump = dev_ptrs.pump_plus[i];
    const real_number in_psi_norm = CUDA::abs2( in_wf );
    
    complex_number result = pump;
    result -= p.gamma_r * in_rv;
    result -= p.R * in_psi_norm * in_rv;
    io.out_rv_plus[i] = result;
}