#ifndef USECPU
#include <curand_kernel.h>
#endif
#include "kernel/kernel_runge_function.cuh"
#include "kernel/kernel_hamilton.cuh"
#include "kernel/kernel_index_overwrite.cuh"

CUDA_GLOBAL void PC3::Kernel::Compute::scalar_stochastic( int i, real_number t, Device::Pointers dev_ptrs, System::Parameters p, InputOutput io ) {
    OVERWRITE_THREAD_INDEX( i );

    const complex_number in_wf = io.in_wf_plus[i];
    const complex_number in_rv = io.in_rv_plus[i];

    const complex_number dw = dev_ptrs.random_number[i] * PC3::CUDA::sqrt( ( p.R * in_rv + p.gamma_c ) / (4.0 * p.dV) );
    io.out_wf_plus[i] -= p.minus_i_over_h_bar_s * p.g_c * in_wf / p.dV - dw / p.dt;
    io.out_rv_plus[i] += p.R * in_rv / p.dV;
}