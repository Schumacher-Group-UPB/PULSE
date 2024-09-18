#include <memory>
#include <algorithm>
#include <random>
#include <ranges>
#include <vector>
#include "cuda/typedef.cuh"
#include "solver/gpu_solver.hpp"
#include "misc/escape_sequences.hpp"
#include "misc/commandline_io.hpp"

void PC3::Solver::initializeMatricesFromSystem() {
    std::cout << EscapeSequence::BOLD << "-------------------- Initializing Host and Device Matrices ------------------------" << EscapeSequence::RESET << std::endl;

    // First, construct all required host matrices
    bool use_fft = system.fft_every < system.t_max;
    bool use_stochastic = system.p.stochastic_amplitude > 0.0;
    // For now, both the plus and the minus components are the same. TODO: Change
    Type::uint32 pulse_size = system.pulse.groupSize();
    Type::uint32 pump_size = system.pump.groupSize();
    Type::uint32 potential_size = system.potential.groupSize();
    matrix.constructAll( system.p.N_c, system.p.N_r, system.use_twin_mode, use_fft, use_stochastic, iterator[system.iterator].k_max, pulse_size, pump_size, potential_size,
                         pulse_size, pump_size, potential_size, system.p.subgrids_columns, system.p.subgrids_rows, system.p.halo_size );

    // ==================================================
    // =................... Halo Map ...................=
    // ==================================================
    initializeHaloMap();

    // ==================================================
    // =................ Initial States ................=
    // ==================================================
    std::cout << PC3::CLIO::prettyPrint( "Initializing Host Matrices...", PC3::CLIO::Control::Info ) << std::endl;

    Envelope::Dimensions dim{ system.p.N_c, system.p.N_r, system.p.L_x, system.p.L_y, system.p.dx, system.p.dy };

    // First, check whether we should adjust the starting states to match a mask. This will initialize the buffer.
    system.initial_state.calculate( system.filehandler, matrix.initial_state_plus.data(), PC3::Envelope::AllGroups, PC3::Envelope::Polarization::Plus, dim );
    system.initial_reservoir.calculate( system.filehandler, matrix.initial_reservoir_plus.data(), PC3::Envelope::AllGroups, PC3::Envelope::Polarization::Plus, dim );
    if ( system.use_twin_mode ) {
        system.initial_state.calculate( system.filehandler, matrix.initial_state_minus.data(), PC3::Envelope::AllGroups, PC3::Envelope::Polarization::Minus, dim );
        system.initial_reservoir.calculate( system.filehandler, matrix.initial_reservoir_minus.data(), PC3::Envelope::AllGroups, PC3::Envelope::Polarization::Minus, dim );
    }

    // Then, check whether we should initialize the system randomly. Add that random value to the initial state.
    if ( system.randomly_initialize_system ) {
        // Fill the buffer with random values
        std::mt19937 gen{ system.random_seed };
        std::uniform_real_distribution<Type::real> dist{ -system.random_system_amplitude, system.random_system_amplitude };
        std::ranges::for_each( matrix.initial_state_plus.data(), matrix.initial_state_plus.data() + system.p.N_c * system.p.N_r,
                               [&dist, &gen]( Type::complex& z ) { z += Type::complex{ dist( gen ), dist( gen ) }; } );
        // Also fill minus component if use_twin_mode is true
        if ( system.use_twin_mode )
            std::ranges::for_each( matrix.initial_state_minus.data(), matrix.initial_state_minus.data() + system.p.N_c * system.p.N_r,
                                   [&dist, &gen]( Type::complex& z ) { z += Type::complex{ dist( gen ), dist( gen ) }; } );
    }

    // Copy the initial state to the device wavefunction, synchronize it to the device and synchronize the halos
    matrix.wavefunction_plus.setTo( matrix.initial_state_plus );
    matrix.wavefunction_plus.hostToDeviceSync();
    SYNCHRONIZE_HALOS( 0, matrix.wavefunction_plus.getSubgridDevicePtrs() );
    matrix.reservoir_plus.setTo( matrix.initial_reservoir_plus );
    matrix.reservoir_plus.hostToDeviceSync();
    SYNCHRONIZE_HALOS( 0, matrix.reservoir_plus.getSubgridDevicePtrs() );

    if ( system.use_twin_mode ) {
        matrix.wavefunction_minus.setTo( matrix.initial_state_minus );
        matrix.wavefunction_minus.hostToDeviceSync();
        SYNCHRONIZE_HALOS( 0, matrix.wavefunction_minus.getSubgridDevicePtrs() );
        matrix.reservoir_minus.setTo( matrix.initial_reservoir_minus );
        matrix.reservoir_minus.hostToDeviceSync();
        SYNCHRONIZE_HALOS( 0, matrix.reservoir_minus.getSubgridDevicePtrs() );
    }

    // ==================================================
    // =................ Pump Envelopes ................=
    // ==================================================
    std::cout << PC3::CLIO::prettyPrint( "Initializing Pump Envelopes...", PC3::CLIO::Control::Info ) << std::endl;
    for ( int pump = 0; pump < system.pump.groupSize(); pump++ ) {
        system.pump.calculate( system.filehandler, matrix.pump_plus.getHostPtr( pump ), pump, PC3::Envelope::Polarization::Plus, dim );
        matrix.pump_plus.hostToDeviceSync( pump );
        SYNCHRONIZE_HALOS( 0, matrix.pump_plus.getSubgridDevicePtrs( pump ) );
        if ( system.use_twin_mode ) {
            system.pump.calculate( system.filehandler, matrix.pump_minus.getHostPtr( pump ), pump, PC3::Envelope::Polarization::Minus, dim );
            matrix.pump_minus.hostToDeviceSync(pump);
            SYNCHRONIZE_HALOS( 0, matrix.pump_minus.getSubgridDevicePtrs( pump ) );
        }
    }
    std::cout << PC3::CLIO::prettyPrint( "Succesfull, designated number of pump groups: " + std::to_string( system.pump.groupSize() ),
                                         PC3::CLIO::Control::Secondary | PC3::CLIO::Control::Success )
              << std::endl;

    // ==================================================
    // =............. Potential Envelopes ..............=
    // ==================================================
    std::cout << PC3::CLIO::prettyPrint( "Initializing Potential Envelopes...", PC3::CLIO::Control::Info ) << std::endl;
    for ( int potential = 0; potential < system.potential.groupSize(); potential++ ) {
        system.potential.calculate( system.filehandler, matrix.potential_plus.getHostPtr( potential ), potential, PC3::Envelope::Polarization::Plus, dim );
        matrix.potential_plus.hostToDeviceSync( potential);
        SYNCHRONIZE_HALOS( 0, matrix.potential_plus.getSubgridDevicePtrs( potential ) );
        if ( system.use_twin_mode ) {
            system.potential.calculate( system.filehandler, matrix.potential_minus.getHostPtr( potential ), potential, PC3::Envelope::Polarization::Minus, dim );
            matrix.potential_minus.hostToDeviceSync(potential);
            SYNCHRONIZE_HALOS( 0, matrix.potential_minus.getSubgridDevicePtrs( potential ) );
        }
    }
    std::cout << PC3::CLIO::prettyPrint( "Succesfull, designated number of potential groups: " + std::to_string( system.potential.groupSize() ),
                                         PC3::CLIO::Control::Secondary | PC3::CLIO::Control::Success )
              << std::endl;

    // ==================================================
    // =............... Pulse Envelopes ................=
    // ==================================================
    std::cout << PC3::CLIO::prettyPrint( "Initializing Pulse Envelopes...", PC3::CLIO::Control::Info ) << std::endl;
    for ( int pulse = 0; pulse < system.pulse.groupSize(); pulse++ ) {
        system.pulse.calculate( system.filehandler, matrix.pulse_plus.getHostPtr( pulse ), pulse, PC3::Envelope::Polarization::Plus, dim );
        matrix.pulse_plus.hostToDeviceSync(pulse);
        SYNCHRONIZE_HALOS( 0, matrix.pulse_plus.getSubgridDevicePtrs( pulse ) );
        if ( system.use_twin_mode ) {
            system.pulse.calculate( system.filehandler, matrix.pulse_minus.getHostPtr( pulse ), pulse, PC3::Envelope::Polarization::Minus, dim );
            matrix.pulse_minus.hostToDeviceSync(pulse);
            SYNCHRONIZE_HALOS( 0, matrix.pulse_minus.getSubgridDevicePtrs( pulse ) );
        }
    }
    std::cout << PC3::CLIO::prettyPrint( "Succesfull, designated number of pulse groups: " + std::to_string( system.pulse.groupSize() ),
                                         PC3::CLIO::Control::Secondary | PC3::CLIO::Control::Success )
              << std::endl;

    // ==================================================
    // =................. FFT Envelopes ................=
    // ==================================================
    Type::host_vector<Type::real> buffer( system.p.N_c * system.p.N_r, 0.0 );
    std::cout << PC3::CLIO::prettyPrint( "Initializing FFT Envelopes...", PC3::CLIO::Control::Info ) << std::endl;
    if ( system.fft_mask.size() == 0 ) {
        std::cout << PC3::CLIO::prettyPrint( "No fft mask provided.", PC3::CLIO::Control::Secondary | PC3::CLIO::Control::Warning ) << std::endl;
    } else {
        system.fft_mask.calculate( system.filehandler, buffer.data(), PC3::Envelope::AllGroups, PC3::Envelope::Polarization::Plus, dim, 1.0 /* Default if no mask is applied */ );
        matrix.fft_mask_plus = buffer;
        // Shift the filter
        auto [block_size, grid_size] = getLaunchParameters( system.p.N_c, system.p.N_r );
        CALL_FULL_KERNEL( PC3::Kernel::fft_shift_2D<Type::real>, "FFT Shift Plus", grid_size, block_size, 0, GET_RAW_PTR( matrix.fft_mask_plus ), system.p.N_c, system.p.N_r );
        if ( system.use_twin_mode ) {
            system.fft_mask.calculate( system.filehandler, buffer.data(), PC3::Envelope::AllGroups, PC3::Envelope::Polarization::Minus, dim,
                                       1.0 /* Default if no mask is applied */ );
            matrix.fft_mask_minus = buffer;
            // Shift the filter
            CALL_FULL_KERNEL( PC3::Kernel::fft_shift_2D<Type::real>, "FFT Shift Minus", grid_size, block_size, 0, GET_RAW_PTR( matrix.fft_mask_minus ), system.p.N_c, system.p.N_r );
        }
    }

    //////////////////////////////////////////////////
    // Custom Envelope Initializations go here      //
    // Just copy the one above and change the names //
    //////////////////////////////////////////////////
}

template <typename T>
T delta( T a, T b ) {
    return a == b ? (T)1 : (T)0;
}

void PC3::Solver::initializeHaloMap() {
    std::cout << PC3::CLIO::prettyPrint( "Initializing Halo Map...", PC3::CLIO::Control::Info ) << std::endl;

    PC3::Type::host_vector<int> halo_map;

    // Create subgrid map
    for ( int dr = -1; dr <= 1; dr++ ) {
        for ( int dc = -1; dc <= 1; dc++ ) {
            if ( dc == 0 and dr == 0 )
                continue;

            const Type::uint32 fr0 = delta( -1, dr ) * system.p.subgrid_N_r + ( 1 - delta( -1, dr ) ) * system.p.halo_size;
            const Type::uint32 fr1 = ( delta( 0, dr ) + delta( -1, dr ) ) * system.p.subgrid_N_r + system.p.halo_size + delta( dr, 1 ) * system.p.halo_size;
            const Type::uint32 fc0 = delta( -1, dc ) * system.p.subgrid_N_c + ( 1 - delta( -1, dc ) ) * system.p.halo_size;
            const Type::uint32 fc1 = ( delta( 0, dc ) + delta( -1, dc ) ) * system.p.subgrid_N_c + system.p.halo_size + delta( dc, 1 ) * system.p.halo_size;

            const Type::uint32 tr0 = delta( 1, dr ) * system.p.subgrid_N_r + ( 1 - delta( -1, dr ) ) * system.p.halo_size;
            const Type::uint32 tr1 = ( 1 - delta( -1, dr ) ) * system.p.subgrid_N_r + system.p.halo_size + delta( 1, dr ) * system.p.halo_size;
            const Type::uint32 tc0 = delta( 1, dc ) * system.p.subgrid_N_c + ( 1 - delta( -1, dc ) ) * system.p.halo_size;
            const Type::uint32 tc1 = ( 1 - delta( -1, dc ) ) * system.p.subgrid_N_c + system.p.halo_size + delta( 1, dc ) * system.p.halo_size;

            for ( int i = 0; i < fr1 - fr0; i++ ) {
                for ( int j = 0; j < fc1 - fc0; j++ ) {
                    const int from_row = fr0 + i;
                    const int from_col = fc0 + j;
                    const int to_row = tr0 + i;
                    const int to_col = tc0 + j;
                    halo_map.push_back( dr );
                    halo_map.push_back( dc );
                    halo_map.push_back( from_row );
                    halo_map.push_back( from_col );
                    halo_map.push_back( to_row );
                    halo_map.push_back( to_col );
                }
            }
        }
    }
    std::cout << PC3::CLIO::prettyPrint( "Designated number of halo cells: " + std::to_string( halo_map.size() / 6 ), PC3::CLIO::Control::Secondary | PC3::CLIO::Control::Success )
              << std::endl;
    matrix.halo_map = halo_map;
}