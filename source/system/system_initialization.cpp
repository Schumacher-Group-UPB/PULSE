#include <memory>
#include <algorithm>
#include <ranges>
#include <random>
#include "system/system_parameters.hpp"
#include "system/filehandler.hpp"
#include "misc/commandline_io.hpp"
#include "misc/escape_sequences.hpp"
#include "system/envelope.hpp"
#include "omp.h"

void PC3::SystemParameters::init( int argc, char** argv ) {
    // Initialize system
    int index = 0;

    // Structure
    p.use_twin_mode = false;
    if ( ( index = PC3::CLIO::findInArgv( "-tetm", argc, argv ) ) != -1 ) {
        p.use_twin_mode = true;
    }

    if ( ( index = PC3::CLIO::findInArgv( "--gammaC", argc, argv ) ) != -1 )
        p.gamma_c = PC3::CLIO::getNextInput( argv, argc, "gamma_c", ++index );
    if ( ( index = PC3::CLIO::findInArgv( "--gammaR", argc, argv ) ) != -1 )
        p.gamma_r = PC3::CLIO::getNextInput( argv, argc, "gamma_r", ++index );
    if ( ( index = PC3::CLIO::findInArgv( "--gc", argc, argv ) ) != -1 )
        p.g_c = PC3::CLIO::getNextInput( argv, argc, "g_c", ++index );
    if ( ( index = PC3::CLIO::findInArgv( "--gr", argc, argv ) ) != -1 )
        p.g_r = PC3::CLIO::getNextInput( argv, argc, "g_r", ++index );
    if ( ( index = PC3::CLIO::findInArgv( "--R", argc, argv ) ) != -1 ) {
        p.R = PC3::CLIO::getNextInput( argv, argc, "R", ++index );
    }
    if ( ( index = PC3::CLIO::findInArgv( "--L", argc, argv ) ) != -1 ) {
        p.L_x = PC3::CLIO::getNextInput( argv, argc, "L", ++index );
        p.L_y = PC3::CLIO::getNextInput( argv, argc, "L", index );
    }
    if ( ( index = PC3::CLIO::findInArgv( "--g_pm", argc, argv ) ) != -1 ) {
        p.g_pm = PC3::CLIO::getNextInput( argv, argc, "gm", ++index );
    }
    if ( ( index = PC3::CLIO::findInArgv( "--deltaLT", argc, argv ) ) != -1 ) {
        p.delta_LT = PC3::CLIO::getNextInput( argv, argc, "deltaLT", ++index );
    }

    omp_max_threads = 4;
    if ( ( index = PC3::CLIO::findInArgv( "--threads", argc, argv ) ) != -1 )
        omp_max_threads = (int)PC3::CLIO::getNextInput( argv, argc, "threads", ++index );
    omp_set_num_threads( omp_max_threads );

    if ( ( index = PC3::CLIO::findInArgv( "--blocksize", argc, argv ) ) != -1 )
        block_size = (int)PC3::CLIO::getNextInput( argv, argc, "block_size", ++index );

    if ( ( index = PC3::CLIO::findInArgv( "--output", argc, argv ) ) != -1 ) {
        output_keys.clear();
        auto output_string = PC3::CLIO::getNextStringInput( argv, argc, "output", ++index );
        // Split output_string at ","
        for ( auto range : output_string | std::views::split( ',' ) ) {
            std::string split_str;
            for (auto ch : range) {
                split_str += ch;
            }
            output_keys.emplace_back(split_str);
            //output_keys.emplace_back( std::string{ std::ranges::begin( range ), std::ranges::end( range ) } );
        }
    }

    // Numerik
    if ( ( index = PC3::CLIO::findInArgv( "--N", argc, argv ) ) != -1 ) {
        p.N_x = (int)PC3::CLIO::getNextInput( argv, argc, "N_x", ++index );
        p.N_y = (int)PC3::CLIO::getNextInput( argv, argc, "N_y", index );
    }
    p.subgrids_x = 2;
    p.subgrids_y = 2;
    p.halo_size = 4;
    p.subgrid_N_x = p.N_x;
    p.subgrid_N_y = p.N_y;

    std::cout << "Subgrid Configuration: " << p.subgrids_x << "x" << p.subgrids_y << std::endl;
    std::cout << "Subgrid Size: " << p.subgrid_N_x << "x" << p.subgrid_N_y << std::endl;
    std::cout << "Halo Size: " << p.halo_size << std::endl;

    // We can also disable to SFML renderer by using the --nosfml flag.
    disableRender = true;
    #ifdef SFML_RENDER
    if ( PC3::CLIO::findInArgv( "-nosfml", argc, argv ) == -1 )
        disableRender = false;
    #endif

    if ( ( index = PC3::CLIO::findInArgv( "--tmax", argc, argv ) ) != -1 )
        t_max = PC3::CLIO::getNextInput( argv, argc, "s_t_max", ++index );
    if ( ( index = PC3::CLIO::findInArgv( "--tstep", argc, argv ) ) != -1 ) {
        p.dt = PC3::CLIO::getNextInput( argv, argc, "t_step", ++index );
        do_overwrite_dt = false;
        std::cout << PC3::CLIO::prettyPrint( "Overwritten (initial) dt to " + PC3::CLIO::to_str(p.dt), PC3::CLIO::Control::Warning) << std::endl;
    }
    if ( ( index = PC3::CLIO::findInArgv( "--tol", argc, argv ) ) != -1 ) {
        tolerance = PC3::CLIO::getNextInput( argv, argc, "tol", ++index );
    }
    if ( ( index = PC3::CLIO::findInArgv( "--rk45dt", argc, argv ) ) != -1 ) {
        dt_min = PC3::CLIO::getNextInput( argv, argc, "dt_min", ++index );
        dt_max = PC3::CLIO::getNextInput( argv, argc, "dt_max", index );
    }
    imag_time_amplitude = 0.0;
    if ( ( index = PC3::CLIO::findInArgv( "--imagTime", argc, argv ) ) != -1 ) {
        imag_time_amplitude = PC3::CLIO::getNextInput( argv, argc, "imag_time_amplitude", ++index );
    }

    if ( ( index = PC3::CLIO::findInArgv( "--fftEvery", argc, argv ) ) != -1 ) {
        fft_every = PC3::CLIO::getNextInput( argv, argc, "fft_every", ++index );
    }

    // Choose the iterator
    iterator = "rk4";
    if ( ( index = PC3::CLIO::findInArgv( "-rk45", argc, argv ) ) != -1 ) {
        //std::cout << PC3::CLIO::prettyPrint( "Currently not implemented. Using default iterator RK4.", PC3::CLIO::Control::FullWarning) << std::endl;
        iterator = "rk45";
    }
    if ( ( index = PC3::CLIO::findInArgv( "-ssfm", argc, argv ) ) != -1 ) {
        //std::cout << PC3::CLIO::prettyPrint( "Currently not implemented. Using default iterator RK4.", PC3::CLIO::Control::FullWarning) << std::endl;
        iterator = "ssfm";
    }
    if ( ( index = PC3::CLIO::findInArgv( "--iterator", argc, argv ) ) != -1 ) {
        std::string it = PC3::CLIO::getNextStringInput( argv, argc, "iterator", ++index );
        iterator = it;
    }

    if ( ( index = PC3::CLIO::findInArgv( "--initRandom", argc, argv ) ) != -1 ) {
        randomly_initialize_system = true;
        random_system_amplitude = PC3::CLIO::getNextInput( argv, argc, "random_system_amplitude", ++index );
        random_seed = std::random_device{}();
        auto str_seed = PC3::CLIO::getNextStringInput( argv, argc, "random_seed", index );
        if ( str_seed != "random" ) {
            random_seed = (Type::uint)std::stod( str_seed );
            std::cout << PC3::CLIO::prettyPrint( "Overwritten random seed to " + std::to_string(random_seed), PC3::CLIO::Control::Info ) << std::endl;
        }
    }

    if ( ( index = PC3::CLIO::findInArgv( "--dw", argc, argv ) ) != -1 ) {
        p.stochastic_amplitude = PC3::CLIO::getNextInput( argv, argc, "dw", ++index );
    }

    if ( ( index = PC3::CLIO::findInArgv( "--outEvery", argc, argv ) ) != -1 )
        output_every = PC3::CLIO::getNextInput( argv, argc, "output_every", ++index );

    history_matrix_output_increment = 1u;
    history_matrix_start_x = 0;
    history_matrix_start_y = 0;
    history_matrix_end_x = p.N_x;
    history_matrix_end_y = p.N_y;
    do_output_history_matrix = false;
    output_history_matrix_every = 1;
    output_history_start_time = 0.0;
    if ( ( index = PC3::CLIO::findInArgv( "--historyMatrix", argc, argv ) ) != -1 ) {
        history_matrix_start_x = (Type::uint)PC3::CLIO::getNextInput( argv, argc, "history_matrix_start_x", ++index );
        history_matrix_end_x = (Type::uint)PC3::CLIO::getNextInput( argv, argc, "history_matrix_end_x", index );
        history_matrix_start_y = (Type::uint)PC3::CLIO::getNextInput( argv, argc, "history_matrix_start_y", index );
        history_matrix_end_y = (Type::uint)PC3::CLIO::getNextInput( argv, argc, "history_matrix_end_y", index );
        history_matrix_output_increment = (Type::uint)PC3::CLIO::getNextInput( argv, argc, "history_matrix_output_increment", index );
        do_output_history_matrix = true;
    }
    if ( ( index = PC3::CLIO::findInArgv( "--historyTime", argc, argv ) ) != -1 ) {
        output_history_start_time = PC3::CLIO::getNextInput( argv, argc, "history_time", ++index );
        output_history_matrix_every = int(PC3::CLIO::getNextInput( argv, argc, "history_time_every", index ));
    }

    p.periodic_boundary_x = false;
    p.periodic_boundary_y = false;
    if ( ( index = PC3::CLIO::findInArgv( "--boundary", argc, argv ) ) != -1 ) {
        auto boundary_x = PC3::CLIO::getNextStringInput( argv, argc, "boundary_x", ++index );
        auto boundary_y = PC3::CLIO::getNextStringInput( argv, argc, "boundary_y", index );
        if ( boundary_x == "periodic" ) {
            p.periodic_boundary_x = true;
        }
        if ( boundary_y == "periodic" ) {
            p.periodic_boundary_y = true;
        }
    }

    // Initialize t_0 as 0.
    p.t = 0.0;
    if ( ( index = PC3::CLIO::findInArgv( "--t0", argc, argv ) ) != -1 ) {
        p.t = PC3::CLIO::getNextInput( argv, argc, "t0", ++index );
        std::cout << PC3::CLIO::prettyPrint( "Overwritten (initial) t to " + PC3::CLIO::to_str( p.t ), PC3::CLIO::Control::Warning ) << std::endl;
    }

    // Even though one probably shouldn't do this, here we read the electron charge, hbar and electron mass from the commandline
    if ( ( index = PC3::CLIO::findInArgv( "--hbar", argc, argv ) ) != -1 ) {
        p.h_bar = PC3::CLIO::getNextInput( argv, argc, "hbar", ++index );
    }
    if ( ( index = PC3::CLIO::findInArgv( "--e", argc, argv ) ) != -1 ) {
        p.e_e = PC3::CLIO::getNextInput( argv, argc, "e", ++index );
    }
    if ( ( index = PC3::CLIO::findInArgv( "--me", argc, argv ) ) != -1 ) {
        p.m_e = PC3::CLIO::getNextInput( argv, argc, "me", ++index );
    }
    // We can also directly set h_bar_scaled, which will result in hbar, e and me to be ignored
    if ( ( index = PC3::CLIO::findInArgv( "--hbarscaled", argc, argv ) ) != -1 ) {
        p.h_bar_s = PC3::CLIO::getNextInput( argv, argc, "hbars", ++index );
    }
    // Same goes for the scaled mass.
    if ( ( index = PC3::CLIO::findInArgv( "--meff", argc, argv ) ) != -1 )
        p.m_eff = PC3::CLIO::getNextInput( argv, argc, "m_eff", ++index );

    // Solver Halo and Subgrid configuration
    

    //////////////////////////////
    // Custom Read-Ins go here! //
    //////////////////////////////

    // Pumps
    pump = PC3::Envelope::fromCommandlineArguments( argc, argv, "pump", true /* Can have oscillation component */ );
    // Potential
    potential = PC3::Envelope::fromCommandlineArguments( argc, argv, "potential", true /* Can have oscillation component */ );
    // Pulses
    pulse = PC3::Envelope::fromCommandlineArguments( argc, argv, "pulse", true /* Can have oscillation component */ );
    // FFT Mask
    fft_mask = PC3::Envelope::fromCommandlineArguments( argc, argv, "fftMask", false );
    // Initial State and Reservoir
    initial_state = PC3::Envelope::fromCommandlineArguments( argc, argv, "initialState", false );
    initial_reservoir = PC3::Envelope::fromCommandlineArguments( argc, argv, "initialReservoir", false );

    ///////////////////////////////////////
    // Custom Envelope Read-Ins go here! //
    ///////////////////////////////////////
}