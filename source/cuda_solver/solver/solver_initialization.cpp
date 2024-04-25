#include <memory>
#include <algorithm>
#include <random>
#include <ranges>
#include <vector>
#include "misc/helperfunctions.hpp"
#include "solver/gpu_solver.hpp"
#include "cuda/cuda_macro.cuh"
#include "misc/escape_sequences.hpp"

void PC3::Solver::initializeHostMatricesFromSystem( ) {
    std::cout << EscapeSequence::BOLD << "--------------------------- Initializing Host Matrices ----------------------------" << EscapeSequence::RESET << std::endl;
    // First, construct all required host matrices
    host.constructAll( system.s_N_x, system.s_N_y, system.use_twin_mode, not system.fixed_time_step  /* Use RK45 */, system.pulse.groupSize(), system.pump.groupSize(), system.potential.groupSize() );

    // ==================================================
    // =................ Initial States ................=
    // ==================================================
    std::cout << "Initializing Host Matrices..." << std::endl;

    // First, check whether we should adjust the starting states to match a mask. This will initialize the buffer.
    system.calculateEnvelope( host.initial_state_plus.get(), system.initial_state, PC3::Envelope::AllGroups, PC3::Envelope::Polarization::Plus);
    std::ranges::for_each( host.initial_state_plus.get(), host.initial_state_plus.get() + system.s_N_x * system.s_N_y, [&,i=0] ( complex_number& z ) mutable { z = z + host.initial_state_plus[i]; i++; } );
    if ( system.use_twin_mode ) {
        system.calculateEnvelope( host.initial_state_minus.get(), system.initial_state, PC3::Envelope::AllGroups, PC3::Envelope::Polarization::Minus);
        std::ranges::for_each( host.initial_state_minus.get(), host.initial_state_minus.get() + system.s_N_x * system.s_N_y, [&,i=0] ( complex_number& z ) mutable { z = z + host.initial_state_minus[i]; i++; } );
    }
    // Then, check whether we should initialize the system randomly. Add that random value to the initial state.
    if (system.randomly_initialize_system) {
        // Fill the buffer with random values
        std::mt19937 gen{system.random_seed};
        std::uniform_real_distribution<real_number> dist{-system.random_system_amplitude, system.random_system_amplitude};
        std::ranges::for_each(host.initial_state_plus.get(), host.initial_state_plus.get() + system.s_N_x * system.s_N_y, [&dist,&gen](complex_number& z) { z += complex_number{dist(gen),dist(gen)}; });
        // Also fill minus component if use_twin_mode is true
        if ( system.use_twin_mode )
            std::ranges::for_each(host.initial_state_minus.get(), host.initial_state_minus.get() + system.s_N_x * system.s_N_y, [&dist,&gen](complex_number& z) { z += complex_number{dist(gen),dist(gen)}; });
    }
    
    //TODO: Hier: Übergebene pumps (system.pump, .pulse, .potential) nach osc parametern sortieren und gruppieren! 
    // Matrix host.pump_plus -> std::vector<Matrix>
    // für jeden gruppe envelope ausrechnen und in host.pump_plus pushen

    // ==================================================
    // =................ Pump Envelopes ................=
    // ==================================================
    std::cout << "Initializing Pump Envelopes..." << std::endl;
    for (int i = 0; i < system.pump.groupSize(); i++) {
        system.calculateEnvelope( host.pump_plus[i].get(), system.pump, i, PC3::Envelope::Polarization::Plus );
        if ( system.use_twin_mode ) {
            system.calculateEnvelope( host.pump_minus[i].get(), system.pump, i, PC3::Envelope::Polarization::Minus );
        }
    }
    std::cout << EscapeSequence::GRAY << "Pump Groups: " << system.pump.groupSize() << EscapeSequence::RESET << std::endl;

    // ==================================================
    // =............. Potential Envelopes ..............=
    // ==================================================
    std::cout << "Initializing Potential Envelopes..." << std::endl;
    for (int i = 0; i < system.potential.groupSize(); i++) {
        system.calculateEnvelope( host.potential_plus[i].get(), system.potential, i, PC3::Envelope::Polarization::Plus );
        if ( system.use_twin_mode ) {
            system.calculateEnvelope( host.potential_minus[i].get(), system.potential, i, PC3::Envelope::Polarization::Minus );
        }
    }
    std::cout << EscapeSequence::GRAY << "Potential Groups: " << system.potential.groupSize() << EscapeSequence::RESET << std::endl;
    
    // ==================================================
    // =............... Pulse Envelopes ................=
    // ==================================================
    std::cout << "Initializing Pulse Envelopes..." << std::endl;
    for (int i = 0; i < system.pulse.groupSize(); i++) {
        system.calculateEnvelope( host.pulse_plus[i].get(), system.pulse, i, PC3::Envelope::Polarization::Plus );
        if ( system.use_twin_mode ) {
            system.calculateEnvelope( host.pulse_minus[i].get(), system.pulse, i, PC3::Envelope::Polarization::Minus );
        }
    }
    std::cout << EscapeSequence::GRAY << "Pulse Groups: " << system.pulse.groupSize() << EscapeSequence::RESET << std::endl;

    // ==================================================
    // =................. FFT Envelopes ................=
    // ==================================================
    std::cout << "Initializing FFT Envelopes..." << std::endl;
    if (system.fft_mask.size() == 0) {
        std::cout << "No fft mask provided." << std::endl;
    } else {
        system.calculateEnvelope( host.fft_mask_plus.get(), system.fft_mask, PC3::Envelope::AllGroups, PC3::Envelope::Polarization::Plus, 1.0 /* Default if no mask is applied */ );
        if (system.use_twin_mode ) {
            system.calculateEnvelope( host.fft_mask_minus.get(), system.fft_mask, PC3::Envelope::AllGroups, PC3::Envelope::Polarization::Minus, 1.0 /* Default if no mask is applied */ );
        }
    }
}

void PC3::Solver::initializeDeviceMatricesFromHost() {

    std::cout << "Initializing Device Matrices..." << std::endl;
    // Construct all Device Matrices
    device.constructAll( system.s_N_x, system.s_N_y, system.use_twin_mode, not system.fixed_time_step /* Use RK45 */, system.pulse.groupSize(), system.pump.groupSize(), system.potential.groupSize() );

    // Initialize the Oscillation Parameters
    dev_pulse_oscillation.construct( system.pulse );
    dev_pump_oscillation.construct( system.pump );
    dev_potential_oscillation.construct( system.potential );

    // Copy Buffer matrices to device equivalents
    device.wavefunction_plus.fromHost( host.initial_state_plus );
    device.reservoir_plus.fromHost( host.reservoir_plus );
    for (int i = 0; i < system.pump.groupSize(); i++) {
        device.pump_plus.fromHost( host.pump_plus[i], system.s_N_x*system.s_N_y*i, system.s_N_x*system.s_N_y );
    }
    for (int i = 0; i < system.pulse.groupSize(); i++) {
        device.pulse_plus.fromHost( host.pulse_plus[i], system.s_N_x*system.s_N_y*i, system.s_N_x*system.s_N_y );
    }
    for (int i = 0; i < system.potential.groupSize(); i++) {
        device.potential_plus.fromHost( host.potential_plus[i], system.s_N_x*system.s_N_y*i, system.s_N_x*system.s_N_y );
    }
    // Set FFT Masks
    device.fft_mask_plus.fromHost( host.fft_mask_plus );
    // Check once of the reservoir is zero.
    // If yes, then the reservoir may not be avaluated.
    #pragma omp parallel for
    for (int i = 0; i < system.s_N_x * system.s_N_y; i++) {
        if (CUDA::abs2(host.reservoir_plus[i]) != 0.0) {
            system.evaluate_reservoir_kernel = true;
        }
        for (int g = 0; g < system.pump.groupSize(); g++) {
            if (CUDA::abs2(host.pump_plus[g][i]) != 0.0) {
                system.evaluate_reservoir_kernel = true;
            }
        }
        for (int g = 0; g < system.pulse.groupSize(); g++) {
            if (CUDA::abs2(host.pulse_plus[g][i]) != 0.0) {
                system.evaluate_pulse_kernel = true;
            }
        }
    }

    // TE/TM Guard
    if (not system.use_twin_mode) 
        return;

    device.wavefunction_minus.fromHost( host.initial_state_minus );
    device.reservoir_minus.fromHost( host.reservoir_minus );
    for (int i = 0; i < system.pump.groupSize(); i++) {
        device.pump_minus.fromHost( host.pump_minus[i], system.s_N_x*system.s_N_y*i, system.s_N_x*system.s_N_y );
    }
    for (int i = 0; i < system.pulse.groupSize(); i++) {
        device.pulse_minus.fromHost( host.pulse_minus[i], system.s_N_x*system.s_N_y*i, system.s_N_x*system.s_N_y );
    }
    for (int i = 0; i < system.potential.groupSize(); i++) {
        device.potential_minus.fromHost( host.potential_minus[i], system.s_N_x*system.s_N_y*i, system.s_N_x*system.s_N_y );
    }
    device.fft_mask_minus.fromHost( host.fft_mask_minus );
    // Check once of the reservoir is zero.
    // If yes, then the reservoir may not be avaluated.
    #pragma omp parallel for
    for (int i = 0; i < system.s_N_x * system.s_N_y; i++) {
        if (CUDA::abs2(host.reservoir_minus[i]) != 0.0) {
            system.evaluate_reservoir_kernel = true;
        }
        for (int g = 0; g < system.pump.groupSize(); g++) {
            if (CUDA::abs2(host.pump_minus[g][i]) != 0.0) {
                system.evaluate_reservoir_kernel = true;
            }
        }
        for (int g = 0; g < system.pulse.groupSize(); g++) {
            if (CUDA::abs2(host.pulse_minus[g][i]) != 0.0) {
                system.evaluate_pulse_kernel = true;
            }
        }
    }
}