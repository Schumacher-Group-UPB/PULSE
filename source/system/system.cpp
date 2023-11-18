#include <memory>
#include <algorithm>
#include <ranges>
#include <random>
#include "system/system.hpp"
#include "system/filehandler.hpp"
#include "misc/commandline_input.hpp"
#include "system/envelope.hpp"
#include "misc/timeit.hpp"
#include "omp.h"
#include "cuda/cuda_matrix.cuh"

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
        std::cout << "Warning! Input parameter g_m = " << g_pm << " is obsolete without TE/TM splitting!" << std::endl;
#endif
    }
    if ( ( index = findInArgv( "--deltaLT", argc, argv ) ) != -1 ) {
        delta_LT = getNextInput( argv, "deltaLT", ++index );
#ifndef TETMSPLITTING
        std::cout << "Warning! Input parameter delta_LT = " << delta_LT << " is obsolete without TE/TM splitting!" << std::endl;
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
        std::cout << "Adjusted N from " << s_N << " to N = " << ( s_N + 1 ) << std::endl;
        s_N++;
    }
    dx = 2.0 * xmax / s_N ; // x-range ist -xmax/2 bis xmax/2
    dt = 0.5 * dx * dx / dt_scaling_factor;
    std::cout << "Calculated dx = " << dx << "\nCalculated dt = " << dt << std::endl;
    if ( ( index = findInArgv( "--tmax", argc, argv ) ) != -1 )
        t_max = getNextInput( argv, "s_t_max", ++index );
    if ( ( index = findInArgv( "--tstep", argc, argv ) ) != -1 ) {
        dt = getNextInput( argv, "t_step", ++index );
        std::cout << "Overwritten (initial) dt to " << dt << std::endl;
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
        std::cout << "Overwritten (initial) t to " << t << std::endl;
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

void PC3::System::calculateEnvelope( real_number* buffer, const PC3::Envelope& mask, PC3::Envelope::Polarization polarization, real_number default_value_if_no_mask ) {
    #pragma omp parallel for
    for ( int col = 0; col < s_N; col++ ) {
        for ( int row = 0; row < s_N; row++ ) {
            int i = col * s_N + row;
            buffer[i] = 0;
            bool has_been_set = false;
            for ( int c = 0; c < mask.amp.size(); c++ ) {
                // Calculate X,Y in the grid space
                auto x = -xmax + dx * col;
                auto y = -xmax + dx * row;
                // If type contains "local", use local coordinates instead
                if ( mask.type[c] & PC3::Envelope::Type::Local ) {
                    x = -1.0 + 2.0 * col/s_N;
                    y = -1.0 + 2.0 * row/s_N;
                }
                // Check if the polarization matches or if the input polarization is both. If not, the envelope is skipped.
                if ( mask.pol[c] != PC3::Envelope::Polarization::Both and mask.pol[c] != polarization and polarization != PC3::Envelope::Polarization::Both )
                    continue;
                has_been_set = true;
                // Calculate ethe "r^2" distance to the center of the pump.
                const real_number r_squared = abs2( x - mask.x[c] ) + abs2( y - mask.y[c] );
                // Calculate Content of Exponential function
                const auto exp_factor = 0.5 * r_squared / mask.width[c] / mask.width[c];
                // Calculate the exponential function
                auto exp_function = exp( -std::pow( exp_factor, mask.exponent[c] ) );
                // If the type is a gaussian outer, we calculate exp(...)^N instead of exp((...)^N)
                if ( mask.type[c] & PC3::Envelope::Type::OuterExponent )
                    exp_function = std::pow( exp( -exp_factor ), mask.exponent[c] );
                // If the shape is a ring, we multiply the exp function with the exp_factor again.
                auto pre_fractor = 1.0;
                if ( mask.type[c] & PC3::Envelope::Type::Ring )
                    pre_fractor = exp_factor;
                // Default amplitude is A/sqrt(2pi)/w
                real_number amplitude = mask.amp[c];
                if ( not (mask.type[c] & PC3::Envelope::Type::NoDivide) )
                    amplitude /= mask.width[c] * sqrt( 2 * 3.1415 );
                // If the behaviour is adaptive, the amplitude is set to the current value of the buffer instead.
                if ( mask.behavior[c] & PC3::Envelope::Behavior::Adaptive )
                    amplitude = mask.amp[c]*buffer[i];
                real_number contribution = amplitude * pre_fractor * exp_function;
                // Add, multiply or replace the contribution to the buffer.
                if ( mask.behavior[c] & PC3::Envelope::Behavior::Add )
                    buffer[i] += contribution;
                else if ( mask.behavior[c] == PC3::Envelope::Behavior::Multiply )
                    buffer[i] *= contribution;
                else if ( mask.behavior[c] == PC3::Envelope::Behavior::Replace )
                    buffer[i] = contribution;
            }
            // If no mask has been applied, set the value to the default value.
            // This ensures the mask is always initialized
            if (not has_been_set)
                buffer[i] = default_value_if_no_mask;
        }
    }
}

/**
 * Hacky way to calculate the envelope as complex numbers.
 * This is only done at the beginning of the program and on the CPU.
 * Temporarily copying the results is probably fine.
*/
void PC3::System::calculateEnvelope( complex_number* buffer, const PC3::Envelope& mask, PC3::Envelope::Polarization polarization, real_number default_value_if_no_mask ) {
    std::unique_ptr<real_number[]> tmp_buffer = std::make_unique<real_number[]>( s_N * s_N );
    calculateEnvelope( tmp_buffer.get(), mask, polarization, default_value_if_no_mask );
    // Transfer tmp_buffer to buffer as complex numbers
    #pragma omp parallel
    for ( int i = 0; i < s_N * s_N; i++ ) {
        buffer[i] = {tmp_buffer[i],0};
    }
}

void PC3::System::printHelp() {
#ifdef USEFP64
    std::cout << "This program is compiled with double precision numbers.\n";
#else
    std::cout << "This program is compiled with single precision numbers.\n";
#endif
#ifdef USECPU
    std::cout << "This program is compiled with CPU support.\n";
    std::cout << "Maximum number of CPU cores utilized " << omp_get_max_threads() << std::endl;
#endif
    std::cout
        << unifyLength( "General parameters:", "", "\n" )
        << unifyLength( "Flag", "Inputs", "Description\n" )
        << unifyLength( "--path", "[string]", "Workingfolder. Standard is '" + filehandler.outputPath + "'\n" )
        << unifyLength( "--name", "[string]", "File prefix. Standard is '" + filehandler.outputName + "'\n" )
        << unifyLength( "--loadFrom", "[string] [files,...]", "Loads list of matrices from path.\n" )
        << unifyLength( "--outEvery", "[int]", "Number of Runge-Kutta iterations for each plot. Standard is every " + std::to_string( filehandler.out_modulo ) + " iteration\n" )
        << unifyLength( "--output", "[list of str]", "Comma seperated list of things to output. Available: mat,scalar,fft,pump,mask,psi,n. Many can also be specified with _plus or _minus.\n" )
        << unifyLength( "--input", "[list of str]", "Comma seperated list of things to input. Available: mat,scalar,fft,pump,mask,psi,n. Many can also be specified with _plus or _minus.\n" )
        << unifyLength( "-nosfml", "no arguments", "If passed to the program, disables all live graphical output. \n" );
    std::cout << unifyLength( "Numerical parameters", "", "\n" ) << unifyLength( "Flag", "Inputs", "Description\n" )
              << unifyLength( "--N", "[int]", "Grid Dimensions (N x N). Standard is " + std::to_string( s_N ) + " x " + std::to_string( s_N ) + "\n" )
              << unifyLength( "--tstep", "[double]", "Timestep, standard is magic-timestep\n" )
              << unifyLength( "--tmax", "[double]", "Timelimit, standard is " + std::to_string( t_max ) + " ps\n" );
    std::cout << unifyLength( "PC3::System parameters", "", "\n" )
              << unifyLength( "Flag", "Inputs", "Description\n" )
              << unifyLength( "--gammaC", "[double]", "Standard is " + std::to_string( gamma_c ) + " ps^-1\n" )
              << unifyLength( "--gammaR", "[double]", "Standard is " + std::to_string( gamma_r / gamma_c ) + "*gammaC\n" )
              << unifyLength( "--gc", "[double]", "Standard is " + std::to_string( g_c ) + " meV mum^2\n" )
              << unifyLength( "--gr", "[double]", "Standard is " + std::to_string( g_r / g_c ) + "*gc\n" )
              << unifyLength( "--meff", "[double]", "Standard is " + std::to_string( m_eff ) + "\n" )
              << unifyLength( "--R", "[double]", "Standard is " + std::to_string( R ) + " ps^-1 mum^2\n" )
              << unifyLength( "--g_pm", "[double]", "Standard is " + std::to_string( g_pm / g_c ) + "*gc. Only effective in a system with TE/TM splitting.\n" )
              << unifyLength( "--deltaLT", "[double]", "Standard is " + std::to_string( delta_LT ) + " meV. Only effective in a system with TE/TM splitting.\n" )
              << unifyLength( "--xmax", "[double]", "Standard is " + std::to_string( xmax ) + " mum\n" ) << std::endl;
    std::cout << unifyLength( "Pulse, pump and mask.", "", "\n" )
              << unifyLength( "Flag", "Inputs", "Description\n", 30, 80 )
              << unifyLength( "--pump", "[double] [string] [double] [double] [double] [string] [double] [string]", "amplitude, behaviour (add,multiply,replace,adaptive), width, posX, posY, pol (plus,minus,both), exponent, type (gauss, ring)\n", 30, 80)
              << unifyLength( "--mask", "[Same as Pump]","\n", 30, 80 )
              << unifyLength( "--initialState", "[Same as Pump]","\n", 30, 80 )
              << unifyLength( "--initRandom", "[double]","Amplitude. Randomly initialize Psi\n", 30, 80 )
              << unifyLength( "--pulse", "[Same as Pump] [double] [double] [double] [int]", "t0, frequency, sigma, m\n", 30, 80 )
              << unifyLength( "--fftMask", "[Same as Pump]","\n", 30, 80 )
              << unifyLength( "--fftEvery", "[int]","Apply FFT Filter every x ps\n", 30, 80 )
              << unifyLength( "-masknorm", "no arguments", "If passed, both the mask and Psi will be normalized before calculating the error.\n", 30, 80 ) << std::endl;
#ifdef USECPU
    std::cout << unifyLength( "--threads", "[int]", "Standard is " + std::to_string( omp_max_threads ) + " Threads\n" ) << std::endl;
#endif
}

void PC3::System::printSummary( std::map<std::string, std::vector<double>> timeit_times, std::map<std::string, double> timeit_times_total ) {
    const int l = 15;
    std::cout << "===================================================================================" << std::endl;
    std::cout << "============================== PC^3 Runtime Statistics ============================" << std::endl;
    std::cout << "===================================================================================" << std::endl;
    std::cout << "--------------------------------- System Parameters -------------------------------" << std::endl;
    std::cout << unifyLength("N", std::to_string(s_N), "",l,l) << std::endl;
    std::cout << unifyLength("N^2", std::to_string(s_N * s_N), "",l,l) << std::endl;
    std::cout << unifyLength("dx", std::to_string(dx), "mum",l,l) << std::endl;
    std::cout << unifyLength("dt", std::to_string(dt), "ps",l,l) << std::endl;
    std::cout << unifyLength("gamma_c", std::to_string(gamma_c), "ps^-1",l,l) << std::endl;
    std::cout << unifyLength("gamma_r", std::to_string(gamma_r), "ps^-1",l,l) << std::endl;
    std::cout << unifyLength("g_c", std::to_string(g_c), "meV mum^2",l,l) << std::endl;
    std::cout << unifyLength("g_r", std::to_string(g_r), "meV mum^2",l,l) << std::endl;
    std::cout << unifyLength("g_pm", std::to_string(g_pm), "meV mum^2",l,l) << std::endl;
    std::cout << unifyLength("R", std::to_string(R), "ps^-1 mum^-2",l,l) << std::endl;
    std::cout << unifyLength("delta_LT", std::to_string(delta_LT), "meV",l,l) << std::endl;
    std::cout << unifyLength("m_eff", std::to_string(m_eff), "",l,l) << std::endl;
    std::cout << unifyLength("xmax", std::to_string(xmax), "mum",l,l) << std::endl;
    std::cout << "--------------------------------- Envelope Functions ------------------------------" << std::endl;
    //TODO: overwrite << operator of the Envelope Class
    for (int i = 0; i < pulse.amp.size(); i++) {
        std::cout << "Pulse at t0 = " << pulse.t0[i] << ", amp = " << pulse.amp[i] << ", freq = " << pulse.freq[i] << ", sigma = " << pulse.sigma[i] << "\n         m = " << pulse.m[i] << ", pol = " << pulse.s_pol[i] << ", width = " << pulse.width[i] << ", X = " << pulse.x[i] << ", Y = " << pulse.y[i] << std::endl;
    }
    for (int i = 0; i < pump.amp.size(); i++) {
        std::cout << "Pump at amp = " << pump.amp[i] << ", width = " << pump.width[i] << ", X = " << pump.x[i] << ", Y = " << pump.y[i] << ", pol = " << pump.s_pol[i] << ", type = " << pump.s_type[i] << std::endl;
    }
    for (int i = 0; i < mask.amp.size(); i++) {
        std::cout << "Soll Mask at amp = " << mask.amp[i] << ", width = " << mask.width[i] << ", X = " << mask.x[i] << ", Y = " << mask.y[i] << ", pol = " << mask.s_pol[i] << ", type = " << mask.s_type[i] << std::endl;
    }
    for (int i = 0; i < fft_mask.amp.size(); i++) {
        std::cout << "FFT Mask at amp = " << fft_mask.amp[i] << ", width = " << fft_mask.width[i] << ", X = " << fft_mask.x[i] << ", Y = " << fft_mask.y[i] << ", pol = " << fft_mask.s_pol[i] << ", type = " << fft_mask.s_type[i] << std::endl;
    }
    if (fft_mask.size() > 0)
        std::cout << "Applying FFT every " << fft_every << " ps" << std::endl;
    std::cout << "--------------------------------- Runtime Statistics ------------------------------" << std::endl;
    double total = PC3::TimeIt::totalRuntime();
    for ( const auto& [key, value] : timeit_times_total ) {
        std::cout << unifyLength( key + ":", std::to_string( value ) + "s", std::to_string( value / t_max * 1E3 ) + "ms/ps", l,l ) << std::endl;
    }
    std::cout << unifyLength( "Total Runtime:", std::to_string( total ) + "s", std::to_string( total / t_max * 1E3 ) + "ms/ps", l,l ) << " --> " << std::to_string( total / iteration ) << "s/it" << std::endl;
    std::cout << "---------------------------------------- Infos ------------------------------------" << std::endl;
    if (filehandler.loadPath.size() > 0)
        std::cout << "Loaded Initial Matrices from " << filehandler.loadPath << std::endl;
    if (fixed_time_step)
        std::cout << "Calculations done using the fixed timestep RK4 solver" << std::endl;
    else {
        std::cout << "Calculations done using the variable timestep RK45 solver" << std::endl;
        std::cout << " = Tolerance used: " << tolerance << std::endl;
        std::cout << " = dt_max used: " << dt_max << std::endl;
        std::cout << " = dt_min used: " << dt_min << std::endl;
    }
    std::cout << "Calculated until t = " << t << "ps" << std::endl;
    std::cout << "Output variables and plots every " << filehandler.out_modulo << " iterations" << std::endl;
    std::cout << "Total allocated space for Device Matrices: " << CUDAMatrix<real_number>::total_mb_max + CUDAMatrix<complex_number>::total_mb_max << " MB." << std::endl;
    std::cout << "Total allocated space for Host Matrices: " << HostMatrix<real_number>::total_mb_max + HostMatrix<complex_number>::total_mb_max << " MB." << std::endl;
    std::cout << "===================================================================================" << std::endl;
}