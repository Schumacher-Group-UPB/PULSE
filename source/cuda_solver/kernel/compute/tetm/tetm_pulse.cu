#include "kernel/kernel_runge_function.cuh"
#include "kernel/kernel_hamilton.cuh"
#include "kernel/kernel_index_overwrite.cuh"

CUDA_GLOBAL void PC3::Kernel::Compute::tetm_pulse( int i, real_number t, Device::Pointers dev_ptrs, System::Parameters p, Solver::Oscillation::Pointers oscillation, InputOutput io ) {

    OVERWRITE_THREAD_INDEX( i );

    complex_number pulse = dev_ptrs.pulse_plus[i];
    complex_number osc = {0,0};
    for (int k = 0; k < oscillation.n; k++) {
        if (oscillation.pol[k] == 1 or oscillation.pol[k] == 3)
            osc += CUDA::exp(- (t - oscillation.t0[k])*(t-oscillation.t0[k]) / (2.0*oscillation.sigma[k]*oscillation.sigma[k]) - p.i*oscillation.freq[k]*(t-oscillation.t0[k]));
    }
    io.out_wf_plus[i] += pulse * osc;
    
    pulse = dev_ptrs.pulse_minus[i];
    osc = {0,0};
    for (int k = 0; k < oscillation.n; k++) {
        if (oscillation.pol[k] == 2 or oscillation.pol[k] == 3)
            osc += CUDA::exp(- (t - oscillation.t0[k])*(t-oscillation.t0[k]) / (2.0*oscillation.sigma[k]*oscillation.sigma[k]) - p.i*oscillation.freq[k]*(t-oscillation.t0[k]));
    }
    io.out_wf_minus[i] += pulse * osc; 
}