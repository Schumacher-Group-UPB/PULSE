#include <memory>
#include <algorithm>
#include <random>
#include <ranges>
#include <vector>
#include "cuda/typedef.cuh"
#include "misc/helperfunctions.hpp"
#include "solver/gpu_solver.hpp"
#include "misc/escape_sequences.hpp"
#include "misc/commandline_io.hpp"

void PC3::Solver::initializeHostMatricesFromSystem() {
    std::cout << EscapeSequence::BOLD << "--------------------------- Initializing Host Matrices ----------------------------" << EscapeSequence::RESET << std::endl;
    
    // Check wether or not the selected solver is available. If not, fallback to RK4
    if ( not iterator.count( system.iterator ) ) {
        std::cout << PC3::CLIO::prettyPrint( "Selected iterator not available. Falling back to RK4.", PC3::CLIO::Control::Secondary | PC3::CLIO::Control::Warning ) << std::endl;
        system.iterator = "rk4";
    }

    // First, construct all required host matrices
    matrix.constructAll( system.p.N_x, system.p.N_y, system.p.use_twin_mode, iterator[system.iterator].k_max , system.pulse.groupSize(), system.pump.groupSize(), system.potential.groupSize() );

    // ==================================================
    // =................ Initial States ................=
    // ==================================================
    std::cout << PC3::CLIO::prettyPrint( "Initializing Host Matrices...", PC3::CLIO::Control::Info ) << std::endl;

    Envelope::Dimensions dim{ system.p.N_x, system.p.N_y, system.p.L_x, system.p.L_y, system.p.dx, system.p.dy };

    // First, check whether we should adjust the starting states to match a mask. This will initialize the buffer.
    system.initial_state.calculate( system.filehandler, matrix.initial_state_plus.getHostPtr(), PC3::Envelope::AllGroups, PC3::Envelope::Polarization::Plus, dim );
    system.initial_reservoir.calculate( system.filehandler, matrix.initial_reservoir_plus.getHostPtr(), PC3::Envelope::AllGroups, PC3::Envelope::Polarization::Plus, dim );
    //std::ranges::for_each( matrix.initial_state_plus.getHostPtr(), matrix.initial_state_plus.getHostPtr() + system.p.N_x * system.p.N_y, [&, i = 0]( Type::complex& z ) mutable { z = z + matrix.initial_state_plus[i]; i++; } );
    if ( system.p.use_twin_mode ) {
        system.initial_state.calculate( system.filehandler, matrix.initial_state_minus.getHostPtr(), PC3::Envelope::AllGroups, PC3::Envelope::Polarization::Minus, dim );
        system.initial_reservoir.calculate( system.filehandler, matrix.initial_reservoir_minus.getHostPtr(), PC3::Envelope::AllGroups, PC3::Envelope::Polarization::Minus, dim );
        //std::ranges::for_each( matrix.initial_state_minus.getHostPtr(), matrix.initial_state_minus.getHostPtr() + system.p.N_x * system.p.N_y, [&, i = 0]( Type::complex& z ) mutable { z = z + matrix.initial_state_minus[i]; i++; } );
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

    // ==================================================
    // =................ Pump Envelopes ................=
    // ==================================================
    std::cout << PC3::CLIO::prettyPrint( "Initializing Pump Envelopes...", PC3::CLIO::Control::Info ) << std::endl;
    for ( int i = 0; i < system.pump.groupSize(); i++ ) {
        system.pump.calculate( system.filehandler, matrix.pump_plus.getHostPtr() + i * system.p.N2, i, PC3::Envelope::Polarization::Plus, dim );
        if ( system.p.use_twin_mode ) {
            system.pump.calculate( system.filehandler, matrix.pump_minus.getHostPtr() + i * system.p.N2, i, PC3::Envelope::Polarization::Minus, dim );
        }
    }
    std::cout << PC3::CLIO::prettyPrint( "Succesfull, designated number of pump groups: " + std::to_string(system.pump.groupSize()), PC3::CLIO::Control::Secondary | PC3::CLIO::Control::Success ) << std::endl;

    // ==================================================
    // =............. Potential Envelopes ..............=
    // ==================================================
    std::cout << PC3::CLIO::prettyPrint( "Initializing Potential Envelopes...", PC3::CLIO::Control::Info ) << std::endl;
    for ( int i = 0; i < system.potential.groupSize(); i++ ) {
        system.potential.calculate( system.filehandler, matrix.potential_plus.getHostPtr() + i * system.p.N2, i, PC3::Envelope::Polarization::Plus, dim );
        if ( system.p.use_twin_mode ) {
            system.potential.calculate( system.filehandler, matrix.potential_minus.getHostPtr() + i * system.p.N2, i, PC3::Envelope::Polarization::Minus, dim );
        }
    }
    std::cout << PC3::CLIO::prettyPrint( "Succesfull, designated number of potential groups: " + std::to_string(system.potential.groupSize()), PC3::CLIO::Control::Secondary | PC3::CLIO::Control::Success ) << std::endl;

    // ==================================================
    // =............... Pulse Envelopes ................=
    // ==================================================
    std::cout << PC3::CLIO::prettyPrint( "Initializing Pulse Envelopes...", PC3::CLIO::Control::Info ) << std::endl;
    for ( int i = 0; i < system.pulse.groupSize(); i++ ) {
        system.pulse.calculate( system.filehandler, matrix.pulse_plus.getHostPtr() + i * system.p.N2, i, PC3::Envelope::Polarization::Plus, dim );
        if ( system.p.use_twin_mode ) {
            system.pulse.calculate( system.filehandler, matrix.pulse_minus.getHostPtr() + i * system.p.N2, i, PC3::Envelope::Polarization::Minus, dim );
        }
    }
    std::cout << PC3::CLIO::prettyPrint( "Succesfull, designated number of pulse groups: " + std::to_string(system.pulse.groupSize()), PC3::CLIO::Control::Secondary | PC3::CLIO::Control::Success ) << std::endl;

    // ==================================================
    // =................. FFT Envelopes ................=
    // ==================================================
    std::cout << PC3::CLIO::prettyPrint( "Initializing FFT Envelopes...", PC3::CLIO::Control::Info ) << std::endl;
    if ( system.fft_mask.size() == 0 ) {
        std::cout << PC3::CLIO::prettyPrint( "No fft mask provided.", PC3::CLIO::Control::Secondary | PC3::CLIO::Control::Warning ) << std::endl;
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
    std::cout << PC3::CLIO::prettyPrint( "Initializing Device Matrices...", PC3::CLIO::Control::Info ) << std::endl;

    // Initialize the solver's temporal envelope vectors
    dev_pulse_oscillation.amp.construct( system.pulse.groupSize(), 1, "Temporal Envelope Pulse" );
    dev_potential_oscillation.amp.construct( system.potential.groupSize(), 1, "Temporal Envelope Potential" );
    dev_pump_oscillation.amp.construct( system.pump.groupSize(), 1, "Temporal Envelope Pump" );

    // Copy Initial State to wavefunction
    matrix.wavefunction_plus.setTo( matrix.initial_state_plus );
    matrix.reservoir_plus.setTo( matrix.initial_reservoir_plus );

    // TE/TM Guard
    if ( not system.p.use_twin_mode )
        return;

    // Copy Initial State to wavefunction
    matrix.wavefunction_minus.setTo( matrix.initial_state_minus );
    matrix.reservoir_minus.setTo( matrix.initial_reservoir_minus );
}