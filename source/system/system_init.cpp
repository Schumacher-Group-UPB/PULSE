#include <memory>
#include <algorithm>
#include <ranges>
#include <random>
#include "system/system.hpp"
#include "system/filehandler.hpp"
#include "misc/commandline_input.hpp"
#include "misc/escape_sequences.hpp"
#include "system/envelope.hpp"
#include "cuda/cuda_matrix.cuh"
#include "omp.h"

void PC3::System::init( int argc, char** argv ) {
    // Initialize system
    int index = 0;

    // Structure
    use_te_tm_splitting = false;
    if ( ( index = findInArgv( "-tetm", argc, argv ) ) != -1 ) {
        use_te_tm_splitting = true;
    }

    // Systemparameter
    if ( ( index = findInArgv( "--meff", argc, argv ) ) != -1 )
        m_eff = getNextInput( argv, "m_eff", ++index );

    if ( ( index = findInArgv( "--gammaC", argc, argv ) ) != -1 )
        gamma_c = getNextInput( argv, "gamma_c", ++index );
    if ( ( index = findInArgv( "--gammaR", argc, argv ) ) != -1 )
        gamma_r = getNextInput( argv, "gamma_r", ++index );
    if ( ( index = findInArgv( "--gc", argc, argv ) ) != -1 )
        g_c = getNextInput( argv, "g_c", ++index );
    if ( ( index = findInArgv( "--gr", argc, argv ) ) != -1 )
        g_r = getNextInput( argv, "g_r", ++index );
    if ( ( index = findInArgv( "--R", argc, argv ) ) != -1 ) {
        R = getNextInput( argv, "R", ++index );
    }
    if ( ( index = findInArgv( "--xmax", argc, argv ) ) != -1 )
        xmax = getNextInput( argv, "xmax", ++index );
    if ( ( index = findInArgv( "--g_pm", argc, argv ) ) != -1 ) {
        g_pm = getNextInput( argv, "gm", ++index );
    }
    if ( ( index = findInArgv( "--deltaLT", argc, argv ) ) != -1 ) {
        delta_LT = getNextInput( argv, "deltaLT", ++index );
    }
    if ( ( index = findInArgv( "--threads", argc, argv ) ) != -1 )
        omp_max_threads = (int)getNextInput( argv, "threads", ++index );
    omp_set_num_threads( omp_max_threads );
    if ( ( index = findInArgv( "--blocksize", argc, argv ) ) != -1 )
        block_size = (int)getNextInput( argv, "block_size", ++index );

    if ( ( index = findInArgv( "--output", argc, argv ) ) != -1 ) {
        output_keys.clear();
        auto output_string = getNextStringInput( argv, "output", ++index );
        // Split output_string at ","
        for ( auto range : output_string | std::views::split( ',' ) ) {
            output_keys.emplace_back( std::string{ std::ranges::begin( range ), std::ranges::end( range ) } );
        }
    }
    if ( ( index = findInArgv( "--input", argc, argv ) ) != -1 ) {
        input_keys.clear();
        auto output_string = getNextStringInput( argv, "input", ++index );
        // Split output_string at ","
        for ( auto range : output_string | std::views::split( ',' ) ) {
            input_keys.emplace_back( std::string{ std::ranges::begin( range ), std::ranges::end( range ) } );
        }
    }

    // Numerik
    if ( ( index = findInArgv( "--N", argc, argv ) ) != -1 ) {
        s_N = (int)getNextInput( argv, "s_N", ++index );
        s_sub_N = (int)getNextInput( argv, "s_sub_N", index );
    }        

    //std::cout << EscapeSequence::GREY << "Calculated dx = " << dx << "\nCalculated dt = " << dt << EscapeSequence::RESET << std::endl;
    if ( ( index = findInArgv( "--tmax", argc, argv ) ) != -1 )
        t_max = getNextInput( argv, "s_t_max", ++index );
    if ( ( index = findInArgv( "--tstep", argc, argv ) ) != -1 ) {
        dt = getNextInput( argv, "t_step", ++index );
        do_overwrite_dt = false;
        std::cout << EscapeSequence::YELLOW << "Overwritten (initial) dt to " << dt << EscapeSequence::RESET << std::endl;
    }
    if ( ( index = findInArgv( "--tol", argc, argv ) ) != -1 ) {
        tolerance = getNextInput( argv, "tol", ++index );
    }
    if ( ( index = findInArgv( "--fftEvery", argc, argv ) ) != -1 ) {
        fft_every = getNextInput( argv, "fft_every", ++index );
    }
    if ( ( index = findInArgv( "-rk45", argc, argv ) ) != -1 ) {
        fixed_time_step = false;
    }
    if ( ( index = findInArgv( "--initRandom", argc, argv ) ) != -1 ) {
        randomly_initialize_system = true;
        random_system_amplitude = getNextInput( argv, "random_system_amplitude", ++index );
        random_seed = std::random_device{}();
        auto str_seed = getNextStringInput( argv, "random_seed", index );
        if (str_seed != "random") {
            random_seed = (unsigned int)std::stod( str_seed );
            std::cout << EscapeSequence::YELLOW << "Overwritten random seed to " << random_seed << EscapeSequence::RESET << std::endl;
        }
    }

    history_output_n = 1000u;
    if ( ( index = findInArgv( "--history", argc, argv ) ) != -1 ) {
        history_output_n = (unsigned int)getNextInput( argv, "history_output_n", ++index );
    }
    history_matrix_output_increment = 1u;
    history_matrix_start = 0;
    history_matrix_end = s_N;
    do_output_history_matrix = false;
    if ( ( index = findInArgv( "--historyMatrix", argc, argv ) ) != -1 ) {
        history_matrix_start = (unsigned int)getNextInput( argv, "history_matrix_start", ++index );
        history_matrix_end = (unsigned int)getNextInput( argv, "history_matrix_end", index );
        history_matrix_output_increment = (unsigned int)getNextInput( argv, "history_matrix_output_increment", index );
        do_output_history_matrix = true;
    }

    // If -masknorm is passed to the program, the mask and psi is normalized before the error calculation
    if ( ( index = findInArgv( "-masknorm", argc, argv ) ) != -1 ) {
        normalize_before_masking = true;
    }

    periodic_boundary_conditions = false;
    if ( ( index = findInArgv( "-periodic", argc, argv ) ) != -1 ) {
        periodic_boundary_conditions = true;
    }

    // Initialize t_0 as 0.
    t = 0.0;
    if ( ( index = findInArgv( "--t0", argc, argv ) ) != -1 ) {
        t = getNextInput( argv, "t0", ++index );
        std::cout << EscapeSequence::YELLOW << "Overwritten (initial) t to " << t << EscapeSequence::RESET << std::endl;
    }

    // Pumps
    pump = PC3::Envelope::fromCommandlineArguments( argc, argv, "pump", false );
    // Potential
    potential = PC3::Envelope::fromCommandlineArguments( argc, argv, "potential", false );
    // Pulses
    pulse = PC3::Envelope::fromCommandlineArguments( argc, argv, "pulse", true );
    // Soll Mask.
    mask = PC3::Envelope::fromCommandlineArguments( argc, argv, "mask", false );
    // FFT Mask
    fft_mask = PC3::Envelope::fromCommandlineArguments( argc, argv, "fftMask", false );
    // Initial State
    initial_state = PC3::Envelope::fromCommandlineArguments( argc, argv, "initialState", false );
}