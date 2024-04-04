#include "kernel/kernel_runge_function.cuh"
#include "kernel/kernel_hamilton.cuh"
#include "kernel/kernel_index_overwrite.cuh"

CUDA_GLOBAL void PC3::Kernel::Compute::tetm_reservoir( int i, real_number t, Device::Pointers dev_ptrs, System::Parameters p, InputOutput io ) {

    OVERWRITE_THREAD_INDEX( i );

    const auto in_wf_plus = io.in_wf_plus[i];
    const auto in_rv_plus = io.in_rv_plus[i];
    const auto pump_plus = dev_ptrs.pump_plus[i];
    const auto in_wf_minus = io.in_wf_minus[i];
    const auto in_rv_minus = io.in_rv_minus[i];
    const auto pump_minus = dev_ptrs.pump_minus[i];
    const real_number in_psi_plus_norm = CUDA::abs2( in_wf_plus );
    const real_number in_psi_minus_norm = CUDA::abs2( in_wf_minus );

    io.out_rv_plus[i] = pump_plus - ( p.gamma_r + p.R * in_psi_plus_norm ) * in_rv_plus;
    io.out_rv_minus[i] = pump_minus - ( p.gamma_r + p.R * in_psi_minus_norm ) * in_rv_minus;
}