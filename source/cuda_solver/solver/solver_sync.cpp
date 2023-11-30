#include "solver/gpu_solver.cuh"

// TODO: selector to choose which matrices to sync
void PC3::Solver::syncDeviceArrays() {
    device.wavefunction_plus.toHost( host.wavefunction_plus );
    device.reservoir_plus.toHost( host.reservoir_plus );
    device.fft_plus.toHost( host.fft_plus );
    
    // TE/TM Guard
    if (not system.use_te_tm_splitting)
        return;
    
    device.wavefunction_minus.toHost( host.wavefunction_minus );
    device.reservoir_minus.toHost( host.reservoir_minus );
    device.fft_minus.toHost( host.fft_minus );
}