#include <memory>
#include <algorithm>
#include <random>
#include <vector>
#include "misc/helperfunctions.hpp"
#include "solver/gpu_solver.cuh"
#include "cuda/cuda_macro.cuh"

void PC3::Solver::initializeDeviceMatricesFromHost() {
    // Construct all Device Matrices
    device.constructAll( system.s_N, use_te_tm_splitting, not system.fixed_time_step /* Use RK45 */ );

    // Initialize the Pulse Parameters. This is subject to change,
    // as we want to also cache the pulse parameters. This is also why
    // this part looks pretty ugly right now.
    const auto n = system.pulse.amp.size();
    dev_pulse_parameters.amp.construct(n, "dev.pulse_amp").setTo( system.pulse.amp.data() );
    dev_pulse_parameters.freq.construct(n, "dev.pulse_freq").setTo( system.pulse.freq.data() );
    dev_pulse_parameters.sigma.construct(n, "dev.pulse_sigma").setTo( system.pulse.sigma.data() );
    dev_pulse_parameters.m.construct(n, "dev.pulse_m").setTo( system.pulse.m.data() );
    dev_pulse_parameters.t0.construct(n, "dev.pulse_t0").setTo( system.pulse.t0.data() );
    dev_pulse_parameters.width.construct(n, "dev.pulse_width").setTo( system.pulse.width.data() );
    dev_pulse_parameters.x.construct(n, "dev.pulse_x").setTo( system.pulse.x.data() );
    dev_pulse_parameters.y.construct(n, "dev.pulse_y").setTo( system.pulse.y.data() );
    std::vector<int> pol( system.pulse.pol.size() );
    for (auto& p : system.pulse.pol)
        pol.emplace_back( p == PC3::Envelope::Polarization::Minus ? -1 : ( p == PC3::Envelope::Polarization::Plus ? 1 : 0) );
    dev_pulse_parameters.pol.construct(n, "dev.pulse_pol").setTo( pol.data() );
    dev_pulse_parameters.n = n;

    // Copy Buffer matrices to device equivalents
    device.wavefunction_plus.fromHost( host.initial_state_plus );
    device.reservoir_plus.fromHost( host.reservoir_plus );
    device.pump_plus.fromHost( host.pump_plus );
    // Set FFT Masks
    device.fft_mask_plus.fromHost( host.fft_mask_plus );
    
    // TE/TM Guard
    if (not use_te_tm_splitting) 
        return;

    device.wavefunction_minus.fromHost( host.initial_state_minus );
    device.reservoir_minus.fromHost( host.reservoir_minus );
    device.pump_minus.fromHost( host.pump_minus );
    device.fft_mask_minus.fromHost( host.fft_mask_minus );
}



// TODO: selector to choose which matrices to sync
void PC3::Solver::syncDeviceArrays() {
    device.wavefunction_plus.toHost( host.wavefunction_plus );
    device.reservoir_plus.toHost( host.reservoir_plus );
    device.fft_plus.toHost( host.fft_plus );
    
    // TE/TM Guard
    if (not use_te_tm_splitting)
        return;
    
    device.wavefunction_minus.toHost( host.wavefunction_minus );
    device.reservoir_minus.toHost( host.reservoir_minus );
    device.fft_minus.toHost( host.fft_minus );
}

void PC3::Solver::finalize() {
    std::cout << "Finalizing Solver" << std::endl;
    // Sync all device arrays
    syncDeviceArrays();
    // Output Matrices
    std::cout << "Outputting Matrices" << std::endl;
    outputMatrices();
    // Cache to files
    std::cout << "Caching to Files" << std::endl;
    cacheToFiles();
    // Calculate Soll Values
    std::cout << "Calculating Soll Values" << std::endl;
    calculateSollValues();
}