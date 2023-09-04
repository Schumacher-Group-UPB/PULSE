#include "helperfunctions.hpp"
#include <fstream>
#include <iostream>
#include <vector>
#include <complex>
#include <omp.h>
#include "cuda_complex.cuh"
#include "cuda_device_variables.cuh"

// To cast the "pointers-to-device-memory" to actual device pointers for thrust
#include <thrust/device_ptr.h>

std::vector<std::string> argv_to_vec( int argc, char** argv ) {
    std::vector<std::string> ret;
    ret.reserve( argc );
    for ( int i = 0; i < argc; i++ )
        ret.push_back( std::string( argv[i] ) );
    return ret;
}

int vec_find_str( std::string toFind, const std::vector<std::string>& input, int start ) {
    for ( int i = start; i < input.size(); i++ ) {
        if ( input.at( i ).compare( toFind ) == 0 )
            return i;
    }
    return -1;
}

real_number getNextInput( const std::vector<std::string>& arguments, const std::string name, int& index ) {
    std::cout << "Read input " << name << " as " << arguments.at( index ) << std::endl;
    return std::stof( arguments.at( index++ ) );
}

std::string unifyLength( std::string indicator, std::string unit, std::string description, int L1, int L2 ) {
    int l1 = L1 - indicator.size();
    int l2 = L2 - unit.size();
    std::string ret = indicator;
    for ( int i = 0; i < l1; i++ )
        ret += " ";
    ret += unit;
    for ( int i = 0; i < l2; i++ )
        ret += " ";
    ret += description;
    return ret;
}

static void printSystemHelp( System& s, FileHandler& h ) {
    std::cout << "Welcome to the 'Spriddis fasdzinirnde Schdruhdelsdheoriesrechnungs'. Todey i wil sho you, how to modifi dis progrem.\n\n";
#ifdef TETMSPLITTING
    std::cout << "This program is compiled with TETM-Splitting enabled.\n";
#else
    std::cout << "This program is compiled with TETM-Splitting disabled.\n";
#endif
#ifdef USEFP64
    std::cout << "This program is compiled with double precision numbers.\n";
#else
    std::cout << "This program is compiled with single precision numbers.\n";
#endif
        std::cout
              << unifyLength( "General parameters:", "", "\n" )
              << unifyLength( "Flag", "Inputs", "Description\n" )
              << unifyLength( "--path", "[string]", "Workingfolder. Standard is '" + h.outputPath + "'\n" )
              << unifyLength( "--name", "[string]", "File prefix. Standard is '" + h.outputName + "'\n" )
              << unifyLength( "--load", "[string]", "Loads matrices from path.\n" )
              //<< unifyLength( "--addOut, --addOutEvery", "[double]", "[NOT IMPLEMENTED] Adds imageoutput at X ps or every X ps.\n" )
              << unifyLength( "--outEvery", "[int]", "Number of Runge-Kutta iterations for each plot. Standard is every " + std::to_string( h.out_modulo ) + " iteration\n" )
              << unifyLength( "-nosfml", "no arguments", "If passed to the program, disables all live graphical output. \n" );
    std::cout << unifyLength( "Numerical parameters", "", "\n" ) << unifyLength( "Flag", "Inputs", "Description\n" )
              << unifyLength( "--N", "[int]", "Grid Dimensions (N x N). Standard is " + std::to_string( s.s_N ) + " x " + std::to_string( s.s_N ) + "\n" )
              << unifyLength( "--tstep", "[double]", "Timestep, standard is magic-timestep\n" )
              << unifyLength( "--tmax", "[double]", "Timelimit, standard is " + std::to_string( s.t_max ) + " ps\n" );
    //<< unifyLength( "-noNormPhase", "no arguments", "If passed to the program, disables normalization of ring phase states\n" )
    //<< unifyLength( "-normPulsePhase", "no arguments", "If passed to the program, enables normalization of pulse phase\n" ) << std::endl;
    std::cout << unifyLength( "System parameters", "", "\n" )
              << unifyLength( "Flag", "Inputs", "Description\n" )
              << unifyLength( "--gammaC", "[double]", "Standard is " + std::to_string( s.gamma_c ) + " ps^-1\n" )
              << unifyLength( "--gammaR", "[double]", "Standard is " + std::to_string( s.gamma_r / s.gamma_c ) + "*gammaC\n" )
              << unifyLength( "--gc", "[double]", "Standard is " + std::to_string( s.g_c ) + " meV mum^2\n" )
              << unifyLength( "--gr", "[double]", "Standard is " + std::to_string( s.g_r / s.g_c ) + "*gc\n" )
#ifdef TETMSPLITTING
              << unifyLength( "--g_pm", "[double]", "Standard is " + std::to_string( s.g_pm / s.g_c ) + "*gc\n" )
              << unifyLength( "--R", "[double]", "Standard is " + std::to_string( s.R ) + " ps^-1 mum^2\n" )
              << unifyLength( "--deltaLT", "[double]", "Standard is " + std::to_string( s.delta_LT ) + " meV\n" )
#endif
              << unifyLength( "--xmax", "[double]", "Standard is " + std::to_string( s.xmax ) + " mum\n" ) << std::endl;
    std::cout << unifyLength( "Pulse and pump. Warning: Adding a pump here overwrites all previous pump settings!", "", "\n" )
              << unifyLength( "Flag", "Inputs", "Description\n", 30, 80 )
              << unifyLength( "--pulse ", "[double] [double] [double] [double] [int] [int] [double] [double] [double]", "t0, amplitude, frequency, sigma, m, pol, width, posX, posY, standard is no pulse. pol = +/-1 or 0 for both\n", 30, 80 )
              << unifyLength( "--pump ", "[double] [double] [double] [double] [int] {int} {int}", "amplitude, width, posX, posY, pol, {mPlus, mMinus} standard is the pump given by previous parameters.\n", 30, 80 ) << unifyLength( " ", " ", "mPlus and mMinus are optional and take effect when --adjustStartingStates is provided\n", 30, 80 ) << unifyLength( "--adjustStartingStates, -ASS ", "", "Adjusts the polarization and amplitude of the starting Psi(+,-) and n(+,-) to match the pump given by --pump. Does nothing if no --pump is provided.\n", 30, 80 ) << std::endl;
}

void addPulse( System& s, real_number t0, real_number amp, real_number freq, real_number sigma, int m, int pol, real_number width, real_number x, real_number y ) {
    if ( amp == 0.0 || sigma == 0 || freq == 0 || width == 0 ) {
        std::cout << "Pulse with amp = 0 or sigma = 0 or freq = 0 or width = 0 not allowed!" << std::endl;
        return;
    }
    s.pulse_t0.push_back( t0 );
    s.pulse_amp.push_back( amp );
    s.pulse_freq.push_back( freq );
    s.pulse_sigma.push_back( sigma );
    s.pulse_m.push_back( m );
    s.pulse_pol.push_back( pol );
    s.pulse_width.push_back( width );
    s.pulse_X.push_back( x );
    s.pulse_Y.push_back( y );
    std::cout << "Added pulse with t0 = " << t0 << " amp = " << amp << " freq = " << freq << " sigma = " << sigma << " m = " << m << " pol = " << pol << " width = " << width << " x = " << x << " y = " << y << std::endl;
}

void addPump( System& s, real_number P0, real_number w, real_number x, real_number y, int pol ) {
    if ( P0 == 0.0 || w == 0 ) {
        std::cout << "Pump with P0 = 0 or w = 0 not allowed!" << std::endl;
        return;
    }
    s.pump_amp.push_back( P0 );
    s.pump_width.push_back( w );
    s.pump_X.push_back( x );
    s.pump_Y.push_back( y );
    s.pump_pol.push_back( pol );
    std::cout << "Added pump with P0 = " << P0 << " w = " << w << " x = " << x << " y = " << y << " pol = " << pol << std::endl;
}

std::tuple<System, FileHandler> initializeSystem( int argc, char** argv ) {
    std::vector<std::string> arguments = argv_to_vec( argc, argv );
    // Initialize system and FileHandler
    System s;
    FileHandler h;
    // Check if help is requested
    if ( vec_find_str( "--help", arguments ) != -1 || vec_find_str( "-h", arguments ) != -1 ) {
        printSystemHelp( s, h );
        exit( 0 );
    }
    // Initialize system
    int index = 0;

    if ( ( index = vec_find_str( "--path", arguments ) ) != -1 )
        h.outputPath = arguments.at( ++index );
    if ( h.outputPath.back() != '/' )
        h.outputPath += "/";

    // Creating output directory
    const int dir_err = std::system( ( "mkdir " + h.outputPath ).c_str() );
    if ( -1 == dir_err ) {
        std::cout << "Error creating directory " << h.outputPath << std::endl;
    } else {
        std::cout << "Succesfully created directory " << h.outputPath << std::endl;
    }

    if ( ( index = vec_find_str( "--name", arguments ) ) != -1 )
        h.outputName = arguments.at( ++index );

    if ( ( index = vec_find_str( "--outEvery", arguments ) ) != -1 )
        h.out_modulo = (int)getNextInput( arguments, "out_modulo", ++index );

    // Systemparameter
    s.m_eff = 1E-4 / s.h_bar_s * 5.6856; //      *m_e/h_bar;                 // m_e/hbar
    if ( ( index = vec_find_str( "--gammaC", arguments ) ) != -1 )
        s.gamma_c = getNextInput( arguments, "gamma_c", ++index );
    if ( ( index = vec_find_str( "--gammaR", arguments ) ) != -1 )
        s.gamma_r = getNextInput( arguments, "gamma_r", ++index );
    if ( ( index = vec_find_str( "--gc", arguments ) ) != -1 )
        s.g_c = getNextInput( arguments, "g_c", ++index ) / s.h_bar_s;
    if ( ( index = vec_find_str( "--gr", arguments ) ) != -1 )
        s.g_r = getNextInput( arguments, "g_r", ++index );
    if ( ( index = vec_find_str( "--R", arguments ) ) != -1 ) {
        #ifndef TETMSPLITTING
        std::cout << "Warning! Input parameter R = " << s.R << " is obsolete without TE/TM splitting!" << std::endl;
        #endif
        s.R = getNextInput( arguments, "R", ++index );
    }
    if ( ( index = vec_find_str( "--xmax", arguments ) ) != -1 )
        s.xmax = getNextInput( arguments, "xmax", ++index );
    if ( ( index = vec_find_str( "--g_pm", arguments ) ) != -1 ) {
        s.g_pm = getNextInput( arguments, "s.gm", ++index );
        #ifndef TETMSPLITTING
        std::cout << "Warning! Input parameter g_m = " << s.g_pm << " is obsolete without TE/TM splitting!" << std::endl;
        #endif
    }
    if ( ( index = vec_find_str( "--deltaLT", arguments ) ) != -1 ) {
        s.delta_LT = getNextInput( arguments, "deltaLT", ++index );
        #ifndef TETMSPLITTING
        std::cout << "Warning! Input parameter delta_LT = " << s.delta_LT << " is obsolete without TE/TM splitting!" << std::endl;
        #endif
    }
    if ( ( index = vec_find_str( "--mPlus", arguments ) ) != -1 )
        s.m_plus = getNextInput( arguments, "m_plus", ++index );
    if ( ( index = vec_find_str( "--mMinus", arguments ) ) != -1 )
        s.m_minus = getNextInput( arguments, "m_plus", ++index );
    if ( ( index = vec_find_str( "--fft", arguments ) ) != -1 )
        s.fft_every = getNextInput( arguments, "fft_every", ++index );

    // Numerik
    // 351x -> 39.7s, 401x -> 65.3s, 451x -> 104.4s, 501x -> 158.3s, 551x -> 231.9s, 751x -> 837s/250ps, 1501 -> 13796.1
    if ( ( index = vec_find_str( "--N", arguments ) ) != -1 )
        s.s_N = (int)getNextInput( arguments, "s_N", ++index );
    if ( s.s_N % 2 == 1 ) {
        std::cout << "Adjusted N from " << s.s_N << " to N = " << ( s.s_N + 1 ) << std::endl;
        s.s_N++;
    }
    s.dx = s.xmax / ( s.s_N - 1 ); // x-range ist -xmax/2 bis xmax/2
    s.dt = 0.5 * s.dx * s.dx;
    std::cout << "Calculated dx = " << s.dx << "\nCalculated dt = " << s.dt << std::endl;
    if ( ( index = vec_find_str( "--tmax", arguments ) ) != -1 )
        s.t_max = getNextInput( arguments, "s_t_max", ++index );
    if ( ( index = vec_find_str( "--tstep", arguments ) ) != -1 ) {
        s.dt = getNextInput( arguments, "t_step", ++index );
        std::cout << "Overwritten (initial) dt to " << s.dt << std::endl;
    }
    if ( ( index = vec_find_str( "--tol", arguments ) ) != -1 ) {
        s.tolerance = getNextInput( arguments, "tol", ++index );
    }
    if ( ( index = vec_find_str( "--fft", arguments ) ) != -1 ) {
        s.fft_every = getNextInput( arguments, "fft_every", ++index );
        s.fft_power = getNextInput( arguments, "fft_pow", index );
        s.fft_mask_area = getNextInput( arguments, "fft_area", index );
    }
    if ( ( index = vec_find_str( "-rk45", arguments ) ) != -1 ) {
        s.fixed_time_step = false;
    }

    // Initialize t_0 as 0.
    s.t = 0.0;
    if ( ( index = vec_find_str( "--t0", arguments ) ) != -1 ) {
        s.t = getNextInput( arguments, "t0", ++index );
        std::cout << "Overwritten (initial) t to " << s.t << std::endl;
    }

    // Pumps
    index = 0;
    while ( ( index = vec_find_str( "--pump", arguments, index ) ) != -1 ) {
        real_number amp = getNextInput( arguments, "pump_amp", ++index );
        real_number w = getNextInput( arguments, "pump_width", index );
        real_number posX = getNextInput( arguments, "pump_X", index );
        real_number posY = getNextInput( arguments, "pump_Y", index );
        int pol = (int)getNextInput( arguments, "pump_pol", index );
        addPump( s, amp, w, posX, posY, pol );
    }
    // Pulses
    index = 0;
    while ( ( index = vec_find_str( "--pulse", arguments, index ) ) != -1 ) {
        real_number t0 = getNextInput( arguments, "pulse_t0", ++index );
        real_number amp = getNextInput( arguments, "pulse_amp", index );
        real_number freq = getNextInput( arguments, "pulse_freq", index );
        real_number sigma = getNextInput( arguments, "pulse_sigma", index );
        int m = getNextInput( arguments, "pulse_m", index );
        int pol = (int)getNextInput( arguments, "pulse_pol", index );
        real_number w = getNextInput( arguments, "pulse_width", index );
        real_number posX = getNextInput( arguments, "pulse_X", index );
        real_number posY = getNextInput( arguments, "pulse_Y", index );
        addPulse( s, t0, amp, freq, sigma, m, pol, w, posX, posY );
    }

    // Save Load Path if passed
    if ( ( index = vec_find_str( "--load", arguments ) ) != -1 )
        h.loadPath = arguments.at( ++index );

    // Colormap
    if ( ( index = vec_find_str( "--cmap", arguments ) ) != -1 )
        h.colorPalette = arguments.at( ++index );

    // We can also disable to SFML renderer by using the --nosfml flag.
    if ( vec_find_str( "-nosfml", arguments ) != -1 )
        h.disableRender = true;

    return std::make_tuple( s, h );
}

void initializePumpVariables( System& s ) {
    initializePumpVariables( s.pump_amp.data(), s.pump_width.data(), s.pump_X.data(), s.pump_Y.data(), s.pump_pol.data(), s.pump_amp.size() );
};
void initializePulseVariables( System& s ) {
    initializePulseVariables( s.pulse_t0.data(), s.pulse_amp.data(), s.pulse_freq.data(), s.pulse_sigma.data(), s.pulse_m.data(), s.pulse_pol.data(), s.pulse_width.data(), s.pulse_X.data(), s.pulse_Y.data(), s.pulse_t0.size() );
};

void normalize( real_number* buffer, int size, real_number min, real_number max, bool device_pointer ) {
    if ( min == max )
        auto [min, max] = minmax( buffer, size, device_pointer );
#pragma omp parallel for
    for ( int i = 0; i < size; i++ )
        buffer[i] = ( buffer[i] - min ) / ( max - min );
}

void angle( complex_number* z, real_number* buffer, int size ) {
#pragma omp parallel for
    for ( int i = 0; i < size; i++ )
        buffer[i] = std::atan2(z[i].y, z[i].x); //std::arg( z[i] );
}

bool doEvaluatePulse( const System& system ) {
    bool evaluate_pulse = false;
    for ( int c = 0; c < system.pulse_t0.size(); c++ ) {
        const auto t0 = system.pulse_t0[c];
        const auto sigma = system.pulse_sigma[c];
        if ( t0 - 5. * sigma < system.t && system.t < t0 + 5. * sigma ) {
            evaluate_pulse = true;
            break;
        }
    }
    return evaluate_pulse;
}

// DEPRECATED
std::vector<complex_number> cacheVector( const System& s, const complex_number* buffer ) {
    const complex_number* start = buffer + s.s_N * s.s_N / 2;
    const complex_number* end = start + s.s_N;

    std::vector<complex_number> ret( s.s_N );
    std::copy( start, end, ret.begin() );
    return ret;
}

void cacheValues( const System& system, Buffer& buffer ) {
    // Min and Max
    const auto [min_plus, max_plus] = minmax( dev_current_Psi_Plus, system.s_N * system.s_N, true /*Device Pointer*/ );
    buffer.cache_Psi_Plus_max.emplace_back( max_plus );
    #ifdef TETMSPLITTING
    const auto [min_minus, max_minus] = minmax( dev_current_Psi_Minus, system.s_N * system.s_N, true /*Device Pointer*/ );
    buffer.cache_Psi_Minus_max.emplace_back( max_minus );
    #endif
    // Cut at Y = 0
    std::unique_ptr<complex_number[]> buffer_cut = std::make_unique<complex_number[]>( system.s_N );
    getDeviceArraySlice( reinterpret_cast<complex_number*>( dev_current_Psi_Plus ), buffer_cut.get(), system.s_N * system.s_N / 2, system.s_N );
    std::vector<complex_number> temp( system.s_N * system.s_N / 2 );
    std::copy( buffer_cut.get(), buffer_cut.get() + system.s_N, temp.begin() );
    buffer.cache_Psi_Plus_history.emplace_back( temp );
    #ifdef TETMSPLITTING
    getDeviceArraySlice( reinterpret_cast<complex_number*>( dev_current_Psi_Minus ), buffer_cut.get(), system.s_N * system.s_N / 2, system.s_N );
    std::copy( buffer_cut.get(), buffer_cut.get() + system.s_N, temp.begin() );
    buffer.cache_Psi_Minus_history.emplace_back( temp );
    #endif
}