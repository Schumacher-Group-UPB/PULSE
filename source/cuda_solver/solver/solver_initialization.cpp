#include <memory>
#include <algorithm>
#include <random>
#include <ranges>
#include <vector>
#include "misc/helperfunctions.hpp"
#include "solver/gpu_solver.hpp"
#include "cuda/cuda_macro.cuh"
#include "misc/escape_sequences.hpp"

void PC3::Solver::outputInitialMatrices() {
    std::cout << "--------------------------- Outputting Initial Matrices ---------------------------" << std::endl;
    auto header_information = PC3::FileHandler::Header(system.s_L_x, system.s_L_y, system.dx, system.dy, system.t);
    // Output Matrices to file
    if ( system.doOutput( "all", "mat", "initial_plus", "initial" ) )
        system.filehandler.outputMatrixToFile( host.initial_state_plus.get(), system.s_N_x, system.s_N_y, header_information, "initial_condition_plus" );
    if ( system.doOutput( "all", "mat", "pump_plus", "pump" ) )
        system.filehandler.outputMatrixToFile( host.pump_plus.get(), system.s_N_x, system.s_N_y, header_information, "pump_plus" );
    if ( system.doOutput( "all", "mat", "pulse_plus", "pulse" ) )
        system.filehandler.outputMatrixToFile( host.pulse_plus.get(), system.s_N_x, system.s_N_y, header_information, "pulse_plus" );
    if ( system.doOutput( "all", "mat", "potential_plus", "potential" ) )
        system.filehandler.outputMatrixToFile( host.potential_plus.get(), system.s_N_x, system.s_N_y, header_information, "potential_plus" );
    if ( system.doOutput( "all", "mat", "fftplus", "fft" ) )
        system.filehandler.outputMatrixToFile( host.fft_mask_plus.get(), system.s_N_x, system.s_N_y, header_information, "fft_mask_plus" );
    
    if (not system.use_twin_mode )
        return;

    if ( system.doOutput( "all", "mat", "initial_minus", "initial" ) )
        system.filehandler.outputMatrixToFile( host.initial_state_minus.get(), system.s_N_x, system.s_N_y, header_information, "initial_condition_minus" );
    if ( system.doOutput( "all", "mat", "pump_minus", "pump" ) )
        system.filehandler.outputMatrixToFile( host.pump_minus.get(), system.s_N_x, system.s_N_y, header_information, "pump_minus" );
    if ( system.doOutput( "all", "mat", "pulse_minus", "pulse" ) )
        system.filehandler.outputMatrixToFile( host.pulse_minus.get(), system.s_N_x, system.s_N_y, header_information, "pulse_minus" );
    if ( system.doOutput( "all", "mat", "potential_minus", "potential" ) )
        system.filehandler.outputMatrixToFile( host.potential_minus.get(), system.s_N_x, system.s_N_y, header_information, "potential_minus" );
    if ( system.doOutput( "all", "mat", "fftminus", "fft" ) )
        system.filehandler.outputMatrixToFile( host.fft_mask_minus.get(), system.s_N_x, system.s_N_y, header_information, "fft_mask_minus" );
}

void PC3::Solver::initializeHostMatricesFromSystem( ) {
    std::cout << EscapeSequence::BOLD << "--------------------------- Initializing Host Matrices ----------------------------" << EscapeSequence::RESET << std::endl;
    // First, construct all required host matrices
    host.constructAll( system.s_N_x, system.s_N_y, system.use_twin_mode, not system.fixed_time_step  /* Use RK45 */ );

    // ==================================================
    // =................ Initial States ................=
    // ==================================================
    std::cout << "Initializing Host Matrices..." << std::endl;

    // First, check whether we should adjust the starting states to match a mask. This will initialize the buffer.
    system.calculateEnvelope( host.initial_state_plus.get(), system.initial_state, PC3::Envelope::Polarization::Plus);
    std::ranges::for_each( host.initial_state_plus.get(), host.initial_state_plus.get() + system.s_N_x * system.s_N_y, [&,i=0] ( complex_number& z ) mutable { z = z + host.initial_state_plus[i]; i++; } );
    if ( system.use_twin_mode ) {
        system.calculateEnvelope( host.initial_state_minus.get(), system.initial_state, PC3::Envelope::Polarization::Minus);
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
    
    // ==================================================
    // =................ Pump Envelopes ................=
    // ==================================================
    std::cout << "Initializing Pump Envelopes..." << std::endl;
    system.calculateEnvelope( host.pump_plus.get(), system.pump, PC3::Envelope::Polarization::Plus );
    if ( system.use_twin_mode ) {
        system.calculateEnvelope( host.pump_minus.get(), system.pump, PC3::Envelope::Polarization::Minus );
    }
    
    // ==================================================
    // =............. Potential Envelopes ..............=
    // ==================================================
    std::cout << "Initializing Potential Envelopes..." << std::endl;
    system.calculateEnvelope( host.potential_plus.get(), system.potential, PC3::Envelope::Polarization::Plus );
    if ( system.use_twin_mode ) {
        system.calculateEnvelope( host.potential_minus.get(), system.potential, PC3::Envelope::Polarization::Minus );
    }

    // ==================================================
    // =............... Pulse Envelopes ................=
    // ==================================================
    std::cout << "Initializing Pulse Envelopes..." << std::endl;
    system.calculateEnvelope( host.pulse_plus.get(), system.pulse, PC3::Envelope::Polarization::Plus );
    if ( system.use_twin_mode ) {
        system.calculateEnvelope( host.pulse_minus.get(), system.pulse, PC3::Envelope::Polarization::Minus );
    }

    // ==================================================
    // =................. FFT Envelopes ................=
    // ==================================================
    std::cout << "Initializing FFT Envelopes..." << std::endl;
    if (system.fft_mask.size() == 0) {
        std::cout << "No fft mask provided. No fft will be calculated." << std::endl;
    } else {
        system.calculateEnvelope( host.fft_mask_plus.get(), system.fft_mask, PC3::Envelope::Polarization::Plus, 1.0 /* Default if no mask is applied */ );
        if (system.use_twin_mode ) {
            system.calculateEnvelope( host.fft_mask_minus.get(), system.fft_mask, PC3::Envelope::Polarization::Minus, 1.0 /* Default if no mask is applied */ );
        }
    }
}

#include <iostream>
void PC3::Solver::initializeDeviceMatricesFromHost() {
    std::cout << "Initializing Device Matrices..." << std::endl;
    // Construct all Device Matrices
    device.constructAll( system.s_N_x, system.s_N_y, system.use_twin_mode, not system.fixed_time_step /* Use RK45 */ );

    // Initialize the Oscillation Parameters
    const auto n = system.pulse.t0.size();
    dev_pulse_oscillation.t0.construct(n, "dev.pulse_t0").setTo( system.pulse.t0.data() );
    dev_pulse_oscillation.freq.construct(n, "dev.pulse_freq").setTo( system.pulse.freq.data() );
    dev_pulse_oscillation.sigma.construct(n, "dev.pulse_sigma").setTo( system.pulse.sigma.data() );
    std::vector<unsigned int> time_pol;
    for (int i = 0; i < n; i++) {
        unsigned int pol;
        if (system.pulse.pol[i] == PC3::Envelope::Polarization::Plus) {
            pol = 1; // Plus
        } else if (system.pulse.pol[i] == PC3::Envelope::Polarization::Minus) {
            pol = 2; // Minus
        } else {
            pol = 3; // Both
        }
        time_pol.push_back( pol );
    }
    dev_pulse_oscillation.pol.construct(n, "dev.pulse_pol").setTo( time_pol.data() );
    dev_pulse_oscillation.n = n;

    // Copy Buffer matrices to device equivalents
    device.wavefunction_plus.fromHost( host.initial_state_plus );
    device.reservoir_plus.fromHost( host.reservoir_plus );
    device.pump_plus.fromHost( host.pump_plus );
    device.pulse_plus.fromHost( host.pulse_plus );
    device.potential_plus.fromHost( host.potential_plus );
    // Set FFT Masks
    device.fft_mask_plus.fromHost( host.fft_mask_plus );
    // Check once of the reservoir is zero.
    // If yes, then the reservoir may not be avaluated.
    #pragma omp parallel for
    for (int i = 0; i < system.s_N_x * system.s_N_y; i++) {
        if (CUDA::abs2(host.reservoir_plus[i]) != 0.0 or CUDA::abs2(host.pump_plus[i]) != 0.0) {
            system.evaluate_reservoir_kernel = true;
        }
        if (CUDA::abs2(host.pulse_plus[i]) != 0.0) {
            system.evaluate_pulse_kernel = true;
        }
    }

    // TE/TM Guard
    if (not system.use_twin_mode) 
        return;

    device.wavefunction_minus.fromHost( host.initial_state_minus );
    device.reservoir_minus.fromHost( host.reservoir_minus );
    device.pump_minus.fromHost( host.pump_minus );
    device.pulse_minus.fromHost( host.pulse_minus );
    device.potential_minus.fromHost( host.potential_minus );
    device.fft_mask_minus.fromHost( host.fft_mask_minus );
    // Check once of the reservoir is zero.
    // If yes, then the reservoir may not be avaluated.
    #pragma omp parallel for
    for (int i = 0; i < system.s_N_x * system.s_N_y; i++) {
        if (CUDA::abs2(host.reservoir_minus[i]) != 0.0 or CUDA::abs2(host.pump_minus[i]) != 0.0){
            system.evaluate_reservoir_kernel = true;
        }
        if (CUDA::abs2(host.pulse_minus[i]) != 0.0) {
            system.evaluate_pulse_kernel = true;
        }
    }
}