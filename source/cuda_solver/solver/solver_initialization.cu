#include <memory>
#include <algorithm>
#include <random>
#include <ranges>
#include <vector>
#include "cuda/typedef.cuh"
#include "misc/helperfunctions.hpp"
#include "solver/gpu_solver.hpp"
#include "misc/escape_sequences.hpp"

void PC3::Solver::initializeHostMatricesFromSystem() {
    std::cout << EscapeSequence::BOLD << "--------------------------- Initializing Host Matrices ----------------------------" << EscapeSequence::RESET << std::endl;
    // First, construct all required host matrices
    matrix.constructAll( system.p.N_x, system.p.N_y, system.p.use_twin_mode, not system.fixed_time_step /* Use RK45 */, system.pulse.groupSize(), system.pump.groupSize(), system.potential.groupSize() );

    // ==================================================
    // =................ Initial States ................=
    // ==================================================
    std::cout << "Initializing Host Matrices..." << std::endl;

    Envelope::Dimensions dim{ system.p.N_x, system.p.N_y, system.p.L_x, system.p.L_y, system.p.dx, system.p.dy };

    // First, check whether we should adjust the starting states to match a mask. This will initialize the buffer.
    system.initial_state.calculate( system.filehandler, matrix.initial_state_plus.getHostPtr(), PC3::Envelope::AllGroups, PC3::Envelope::Polarization::Plus, dim );
    std::ranges::for_each( matrix.initial_state_plus.getHostPtr(), matrix.initial_state_plus.getHostPtr() + system.p.N_x * system.p.N_y, [&, i = 0]( Type::complex& z ) mutable { z = z + matrix.initial_state_plus[i]; i++; } );
    if ( system.p.use_twin_mode ) {
        system.initial_state.calculate( system.filehandler, matrix.initial_state_minus.getHostPtr(), PC3::Envelope::AllGroups, PC3::Envelope::Polarization::Minus, dim );
        std::ranges::for_each( matrix.initial_state_minus.getHostPtr(), matrix.initial_state_minus.getHostPtr() + system.p.N_x * system.p.N_y, [&, i = 0]( Type::complex& z ) mutable { z = z + matrix.initial_state_minus[i]; i++; } );
    }
    // Then, check whether we should initialize the system randomly. Add that random value to the initial state.
    if ( system.randomly_initialize_system ) {
        // Fill the buffer with random values
        std::mt19937 gen{ system.random_seed };
        std::uniform_real_distribution<Type::real> dist{ -system.random_system_amplitude, system.random_system_amplitude };
        std::ranges::for_each( matrix.initial_state_plus.getHostPtr(), matrix.initial_state_plus.getHostPtr() + system.p.N_x * system.p.N_y, [&dist, &gen]( Type::complex& z ) { z += Type::complex{ dist( gen ), dist( gen ) }; } );
        // Also fill minus component if use_twin_mode is true
        if ( system.p.use_twin_mode )
            std::ranges::for_each( matrix.initial_state_minus.getHostPtr(), matrix.initial_state_minus.getHostPtr() + system.p.N_x * system.p.N_y, [&dist, &gen]( Type::complex& z ) { z += Type::complex{ dist( gen ), dist( gen ) }; } );
    }

    // TODO: Hier: Übergebene pumps (system.pump, .pulse, .potential) nach osc parametern sortieren und gruppieren!
    //  Matrix matrix.pump_plus ->HostPtr std::vector<Matrix>
    //  für jeden gruppe envelope ausrechnen und in matrix.pump_plus pushenHostPtr

    // ==================================================
    // =................ Pump Envelopes ................=
    // ==================================================
    std::cout << "Initializing Pump Envelopes..." << std::endl;
    for ( int i = 0; i < system.pump.groupSize(); i++ ) {
        system.pump.calculate( system.filehandler, matrix.pump_plus.getHostPtr() + i * system.p.N2, i, PC3::Envelope::Polarization::Plus, dim );
        if ( system.p.use_twin_mode ) {
            system.pump.calculate( system.filehandler, matrix.pump_minus.getHostPtr() + i * system.p.N2, i, PC3::Envelope::Polarization::Minus, dim );
        }
    }
    std::cout << EscapeSequence::GRAY << "Pump Groups: " << system.pump.groupSize() << EscapeSequence::RESET << std::endl;

    // ==================================================
    // =............. Potential Envelopes ..............=
    // ==================================================
    std::cout << "Initializing Potential Envelopes..." << std::endl;
    for ( int i = 0; i < system.potential.groupSize(); i++ ) {
        system.potential.calculate( system.filehandler, matrix.potential_plus.getHostPtr() + i * system.p.N2, i, PC3::Envelope::Polarization::Plus, dim );
        if ( system.p.use_twin_mode ) {
            system.potential.calculate( system.filehandler, matrix.potential_minus.getHostPtr() + i * system.p.N2, i, PC3::Envelope::Polarization::Minus, dim );
        }
    }
    std::cout << EscapeSequence::GRAY << "Potential Groups: " << system.potential.groupSize() << EscapeSequence::RESET << std::endl;

    // ==================================================
    // =............... Pulse Envelopes ................=
    // ==================================================
    std::cout << "Initializing Pulse Envelopes..." << std::endl;
    for ( int i = 0; i < system.pulse.groupSize(); i++ ) {
        system.pulse.calculate( system.filehandler, matrix.pulse_plus.getHostPtr() + i * system.p.N2, i, PC3::Envelope::Polarization::Plus, dim );
        if ( system.p.use_twin_mode ) {
            system.pulse.calculate( system.filehandler, matrix.pulse_minus.getHostPtr() + i * system.p.N2, i, PC3::Envelope::Polarization::Minus, dim );
        }
    }
    std::cout << EscapeSequence::GRAY << "Pulse Groups: " << system.pulse.groupSize() << EscapeSequence::RESET << std::endl;

    // ==================================================
    // =................. FFT Envelopes ................=
    // ==================================================
    std::cout << "Initializing FFT Envelopes..." << std::endl;
    if ( system.fft_mask.size() == 0 ) {
        std::cout << "No fft mask provided." << std::endl;
    } else {
        system.fft_mask.calculate( system.filehandler, matrix.fft_mask_plus.getHostPtr(), PC3::Envelope::AllGroups, PC3::Envelope::Polarization::Plus, dim, 1.0 /* Default if no mask is applied */ );
        if ( system.p.use_twin_mode ) {
            system.fft_mask.calculate( system.filehandler, matrix.fft_mask_minus.getHostPtr(), PC3::Envelope::AllGroups, PC3::Envelope::Polarization::Minus, dim, 1.0 /* Default if no mask is applied */ );
        }
    }

    //////////////////////////////////////////////////
    // Custom Envelope Initializations go here      //
    // Just copy the one above and change the names //
    //////////////////////////////////////////////////
}

void PC3::Solver::initializeDeviceMatricesFromHost() {
    std::cout << "Initializing Device Matrices..." << std::endl;

    // Initialize the Oscillation Parameters
    dev_pulse_oscillation.construct( system.pulse );
    dev_pump_oscillation.construct( system.pump );
    dev_potential_oscillation.construct( system.potential );

    // Copy Initial State to wavefunction
    matrix.wavefunction_plus.setTo( matrix.initial_state_plus );

// Check once of the reservoir is zero.
// If yes, then the reservoir may not be avaluated.
#pragma omp parallel for
    for ( int i = 0; i < system.p.N_x * system.p.N_y; i++ ) {
        if ( CUDA::abs2( matrix.reservoir_plus[i] ) != 0.0 ) {
            system.evaluate_reservoir_kernel = true;
        }
        for ( int g = 0; g < system.pump.groupSize(); g++ ) {
            if ( CUDA::abs2( matrix.pump_plus[i + g * system.p.N2] ) != 0.0 ) {
                system.evaluate_reservoir_kernel = true;
            }
        }
        for ( int g = 0; g < system.pulse.groupSize(); g++ ) {
            if ( CUDA::abs2( matrix.pulse_plus[i + g * system.p.N2] ) != 0.0 ) {
                system.evaluate_pulse_kernel = true;
            }
        }
        for ( int g = 0; g < system.potential.groupSize(); g++ ) {
            if ( CUDA::abs2( matrix.potential_plus[i + g * system.p.N2] ) != 0.0 ) {
                system.evaluate_potential_kernel = true;
            }
        }
    }

    // TE/TM Guard
    if ( not system.p.use_twin_mode )
        return;

    // Copy Initial State to wavefunction
    matrix.wavefunction_minus.setTo( matrix.initial_state_minus );

// Check once of the reservoir is zero.
// If yes, then the reservoir may not be avaluated.
#pragma omp parallel for
    for ( int i = 0; i < system.p.N_x * system.p.N_y; i++ ) {
        if ( CUDA::abs2( matrix.reservoir_minus[i] ) != 0.0 ) {
            system.evaluate_reservoir_kernel = true;
        }
        for ( int g = 0; g < system.pump.groupSize(); g++ ) {
            if ( CUDA::abs2( matrix.pump_minus[i + g * system.p.N2] ) != 0.0 ) {
                system.evaluate_reservoir_kernel = true;
            }
        }
        for ( int g = 0; g < system.pulse.groupSize(); g++ ) {
            if ( CUDA::abs2( matrix.pulse_minus[i + g * system.p.N2] ) != 0.0 ) {
                system.evaluate_pulse_kernel = true;
            }
        }
        for ( int g = 0; g < system.potential.groupSize(); g++ ) {
            if ( CUDA::abs2( matrix.potential_minus[i + g * system.p.N2] ) != 0.0 ) {
                system.evaluate_potential_kernel = true;
            }
        }
    }
}