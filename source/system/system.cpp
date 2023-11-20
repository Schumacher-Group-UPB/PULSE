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

/**
 * @brief Default Constructor for the System Class
 * Defaul-initializes system parameters, which are 
 * overwritten by the user cmd input.
 * 
 */
PC3::System::System() {
     // SI Rescaling Units
    m_e = 9.10938356E-31;
    h_bar = 1.0545718E-34;
    e_e = 1.60217662E-19;
    h_bar_s = 6.582119514E-4;

    // System Variables
    m_eff;
    gamma_c = 0.15;             // ps^-1
    gamma_r = 1.5 * gamma_c;    // ps^-1
    g_c = 3.E-6;                // meV mum^2
    g_r = 2. * g_c;             // meV mum^2
    R = 0.01;                   // ps^-1 mum^2
    xmax = 100.;                // mum
    g_pm = -g_c / 5;            // meV mum^2
    delta_LT = 0.025E-3;        // meV

    // Numerics
    s_N = 400;
    dx;
    t_max = 1000;
    iteration = 0;
    // RK Solver Variables
    dt;
    t;
    dt_max = 3;
    dt_min = 0.0001; // also dt_delta
    tolerance = 1E-1;

    // FFT Mask Parameter moved to mask syntax
    fft_every = 1; // ps

    // Kernel Block Size
    block_size = 16;
    omp_max_threads = omp_get_max_threads();

    // If this is true, the solver will use a fixed timestep RK4 method instead of the variable timestep RK45 method
    fixed_time_step = true;

    // Output of Variables
    output_keys = {"mat","scalar"};

    normalize_before_masking = false;

    randomly_initialize_system = false;
    random_system_amplitude = 1.0;
}

PC3::System::System( int argc, char** argv ) : System() {
    // Initialize system
    int index = 0;

    // Systemparameter
    m_eff = 1E-4 * 5.6856;
    double dt_scaling_factor = m_eff;
    if ( ( index = findInArgv( "--meff", argc, argv ) ) != -1 )
        m_eff = getNextInput( argv, "m_eff", ++index );
    dt_scaling_factor /= m_eff;
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
#ifndef TETMSPLITTING
        std::cout << EscapeSequence::YELLOW << "Warning! Input parameter g_m = " << g_pm << " is obsolete without TE/TM splitting!" << EscapeSequence::RESET << std::endl;
#endif
    }
    if ( ( index = findInArgv( "--deltaLT", argc, argv ) ) != -1 ) {
        delta_LT = getNextInput( argv, "deltaLT", ++index );
#ifndef TETMSPLITTING
        std::cout << EscapeSequence::YELLOW << "Warning! Input parameter delta_LT = " << delta_LT << " is obsolete without TE/TM splitting!" << EscapeSequence::RESET << std::endl;
#endif
    }
    if ( ( index = findInArgv( "--threads", argc, argv ) ) != -1 )
        omp_max_threads = (int)getNextInput( argv, "threads", ++index );
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
    if ( ( index = findInArgv( "--N", argc, argv ) ) != -1 )
        s_N = (int)getNextInput( argv, "s_N", ++index );
    if ( s_N % 2 == 1 ) {
        std::cout << EscapeSequence::YELLOW << "Adjusted N from " << s_N << " to N = " << ( s_N + 1 ) << EscapeSequence::RESET << std::endl;
        s_N++;
    }
    dx = 2.0 * xmax / s_N ; // x-range ist -xmax/2 bis xmax/2
    dt = 0.5 * dx * dx / dt_scaling_factor;
    std::cout << EscapeSequence::GREY << "Calculated dx = " << dx << "\nCalculated dt = " << dt << EscapeSequence::RESET << std::endl;
    if ( ( index = findInArgv( "--tmax", argc, argv ) ) != -1 )
        t_max = getNextInput( argv, "s_t_max", ++index );
    if ( ( index = findInArgv( "--tstep", argc, argv ) ) != -1 ) {
        dt = getNextInput( argv, "t_step", ++index );
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
        random_seed = (unsigned int)getNextInput( argv, "random_seed", index );
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
    // Pulses
    pulse = PC3::Envelope::fromCommandlineArguments( argc, argv, "pulse", true );
    // Soll Mask.
    mask = PC3::Envelope::fromCommandlineArguments( argc, argv, "mask", false );
    // FFT Mask
    fft_mask = PC3::Envelope::fromCommandlineArguments( argc, argv, "fftMask", false );
    // Initial State
    initial_state = PC3::Envelope::fromCommandlineArguments( argc, argv, "initialState", false );

    // Check if help is requested
    if ( findInArgv( "--help", argc, argv ) != -1 || findInArgv( "-h", argc, argv ) != -1 ) {
        printHelp();
        exit( 0 );
    }

    filehandler.init(argc, argv);
}


bool PC3::System::evaluatePulse( ) {
    bool evaluate_pulse = false;
    for ( int c = 0; c < pulse.t0.size(); c++ ) {
        const auto t0 = pulse.t0[c];
        const auto sigma = pulse.sigma[c];
        if ( t0 - 5. * sigma < t && t < t0 + 5. * sigma ) {
            evaluate_pulse = true;
            break;
        }
    }
    return evaluate_pulse;
}