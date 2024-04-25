#include "kernel/kernel_runge_function.cuh"
#include "kernel/kernel_hamilton.cuh"
#include "kernel/kernel_index_overwrite.cuh"

CUDA_GLOBAL void PC3::Kernel::Compute::scalar_pulse( int i, real_number t, Device::Pointers dev_ptrs, System::Parameters p, Solver::Oscillation::Pointers oscillation, InputOutput io ) {
    
    OVERWRITE_THREAD_INDEX( i );

    complex_number osc = {0,0};
    for (int k = 0; k < oscillation.n; k++) {
        const size_t offset = k * p.N_x * p.N_y;
        const complex_number pulse = dev_ptrs.pulse_plus[i+offset];
        osc += pulse * PC3::CUDA::gaussian_complex_oscillator(t, oscillation.t0[k], oscillation.sigma[k], oscillation.freq[k]);
    }

    io.out_wf_plus[i] += p.minus_i_over_h_bar_s * osc;
}