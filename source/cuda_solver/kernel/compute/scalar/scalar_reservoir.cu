#include "kernel/kernel_runge_function.cuh"
#include "kernel/kernel_hamilton.cuh"
#include "kernel/kernel_index_overwrite.cuh"

CUDA_GLOBAL void PC3::Kernel::Compute::scalar_reservoir( int i, real_number t, MatrixContainer::Pointers dev_ptrs, System::Parameters p, Solver::Oscillation::Pointers oscillation, InputOutput io ) {

    OVERWRITE_THREAD_INDEX( i );

    const complex_number in_wf = io.in_wf_plus[i];
    const complex_number in_rv = io.in_rv_plus[i];
    const real_number in_psi_norm = CUDA::abs2( in_wf );
    
    complex_number result = -p.gamma_r * in_rv;
    result -= p.R * in_psi_norm * in_rv;
    for (int k = 0; k < oscillation.n; k++) {
        const int offset = k * p.N_x * p.N_y;
        result += dev_ptrs.pump_plus[i+offset] * PC3::CUDA::gaussian_oscillator(t, oscillation.t0[k], oscillation.sigma[k], oscillation.freq[k]);
    }
    io.out_rv_plus[i] = result;
}