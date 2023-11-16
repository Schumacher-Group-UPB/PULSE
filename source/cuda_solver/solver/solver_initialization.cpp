#include <memory>
#include <algorithm>
#include <random>
#include <ranges>
#include <vector>
#include "misc/helperfunctions.hpp"
#include "solver/gpu_solver.cuh"
#include "cuda/cuda_macro.cuh"


void PC3::Solver::initializeHostMatricesFromSystem( ) {
    // First, construct all required host matrices
    host.constructAll( system.s_N, use_te_tm_splitting, not system.fixed_time_step  /* Use RK45 */ );

    // ==================================================
    // =................ Initial States ................=
    // ==================================================
    std::cout << "Initializing Host Matrices..." << std::endl;

    // First, check whether we should adjust the starting states to match a mask. This will initialize the buffer.
    system.calculateEnvelope( host.initial_state_plus.get(), system.initial_state, PC3::Envelope::Polarization::Plus);
    std::ranges::for_each( host.initial_state_plus.get(), host.initial_state_plus.get() + system.s_N * system.s_N, [&,i=0] ( complex_number& z ) mutable { z = z + host.initial_state_plus[i]; i++; } );
    if ( use_te_tm_splitting ) {
        system.calculateEnvelope( host.initial_state_minus.get(), system.initial_state, PC3::Envelope::Polarization::Minus);
        std::ranges::for_each( host.initial_state_minus.get(), host.initial_state_minus.get() + system.s_N * system.s_N, [&,i=0] ( complex_number& z ) mutable { z = z + host.initial_state_minus[i]; i++; } );
    }
    // Then, check whether we should initialize the system randomly. Add that random value to the initial state.
    if (system.randomly_initialize_system) {
        // Fill the buffer with random values
        std::mt19937 gen{system.random_seed};
        std::uniform_real_distribution<real_number> dist{-system.random_system_amplitude, system.random_system_amplitude};
        std::ranges::for_each(host.initial_state_plus.get(), host.initial_state_plus.get() + system.s_N * system.s_N, [&dist,&gen](complex_number& z) { z += complex_number{dist(gen),dist(gen)}; });
        // Also fill minus component if use_te_tm_splitting is true
        if ( use_te_tm_splitting )
            std::ranges::for_each(host.initial_state_minus.get(), host.initial_state_minus.get() + system.s_N * system.s_N, [&dist,&gen](complex_number& z) { z += complex_number{dist(gen),dist(gen)}; });
    }
    // Output Matrices to file
    if ( system.doOutput( "mat", "initial_plus", "initial" ) )
        system.filehandler.outputMatrixToFile( host.initial_state_plus.get(), system.s_N, system.xmax, system.dx, "initial_condition_plus" );
    if ( use_te_tm_splitting and system.doOutput( "mat", "initial_minus", "initial" ) )
        system.filehandler.outputMatrixToFile( host.initial_state_minus.get(), system.s_N, system.xmax, system.dx, "initial_condition_minus" );

    
    // ==================================================
    // =................ Pump Envelopes ................=
    // ==================================================
    std::cout << "Initializing Pump Envelopes..." << std::endl;
    system.calculateEnvelope( host.pump_plus.get(), system.pump, PC3::Envelope::Polarization::Plus );
    if ( system.doOutput( "mat", "pump_plus", "pump" ) )
        system.filehandler.outputMatrixToFile( host.pump_plus.get(), system.s_N, system.xmax, system.dx, "pump_plus" );
    if ( use_te_tm_splitting ) {
        system.calculateEnvelope( host.pump_minus.get(), system.pump, PC3::Envelope::Polarization::Minus );
        if ( system.doOutput( "mat", "pump_minus", "pump" ) )
            system.filehandler.outputMatrixToFile( host.pump_minus.get(), system.s_N, system.xmax, system.dx, "pump_minus" );
    }


    // ==================================================
    // =................. FFT Envelopes ................=
    // ==================================================
    std::cout << "Initializing FFT Envelopes..." << std::endl;
    if (system.fft_mask.size() == 0) {
        std::cout << "No fft mask provided. No fft will be calculated." << std::endl;
        system.fft_every = 2*system.t_max;
    } else {
        system.calculateEnvelope( host.fft_mask_plus.get(), system.fft_mask, PC3::Envelope::Polarization::Plus, 1.0 /* Default if no mask is applied */ );
        // Output Matrices to file
        if ( system.doOutput( "mat", "fftplus", "fft" ) )
            system.filehandler.outputMatrixToFile( host.fft_mask_plus.get(), system.s_N, system.xmax, system.dx, "fft_mask_plus" );
        if (use_te_tm_splitting ) {
            system.calculateEnvelope( host.fft_mask_minus.get(), system.fft_mask, PC3::Envelope::Polarization::Minus, 1.0 /* Default if no mask is applied */ );
            // Output Matrices to file
            if ( system.doOutput( "mat", "fftminus", "fft" ) )
                system.filehandler.outputMatrixToFile( host.fft_mask_minus.get(), system.s_N, system.xmax, system.dx, "fft_mask_minus" );
        }
    }

    // ==================================================
    // =.............. Soll Mask Envelopes .............=
    // ==================================================
    std::cout << "Initializing Soll Mask Envelopes..." << std::endl;
    if ( system.mask.x.size() > 0 ) {
        system.calculateEnvelope( host.soll_plus.get(), system.mask, PC3::Envelope::Polarization::Plus );
        if ( system.doOutput( "mat", "mask_plus", "mask" ) )
            system.filehandler.outputMatrixToFile( host.soll_plus.get(), system.s_N, system.xmax, system.dx, "mask_plus" );
        if ( use_te_tm_splitting ) {
            system.calculateEnvelope( host.soll_minus.get(), system.mask, PC3::Envelope::Polarization::Minus );
            if ( system.doOutput( "mat", "mask_minus", "mask" ) )
                system.filehandler.outputMatrixToFile( host.soll_minus.get(), system.s_N, system.xmax, system.dx, "mask_minus" );
        }
    }
}