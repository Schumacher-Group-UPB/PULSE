#include "helperfunctions.hpp"
#include <fstream>
#include <iostream>
#include <vector>
#include <complex>
#include <omp.h>
#include "cuda_complex.cuh"
#include "cuda_device_variables.cuh"
#include <memory>
#include <algorithm>
#include <ranges>

// To cast the "pointers-to-device-memory" to actual device pointers for thrust
#ifndef USECPU
#    include <thrust/device_ptr.h>
#endif

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
    return std::stod( arguments.at( index++ ) );
}
std::string getNextStringInput( const std::vector<std::string>& arguments, const std::string name, int& index ) {
    std::cout << "Read input " << name << " as " << arguments.at( index ) << std::endl;
    return arguments.at( index++ );
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
#ifdef USECPU
    std::cout << "This program is compiled with CPU support.\n";
    std::cout << "Maximum number of CPU cores utilized " << omp_get_max_threads() << std::endl;
#endif
    std::cout
        << unifyLength( "General parameters:", "", "\n" )
        << unifyLength( "Flag", "Inputs", "Description\n" )
        << unifyLength( "--path", "[string]", "Workingfolder. Standard is '" + h.outputPath + "'\n" )
        << unifyLength( "--name", "[string]", "File prefix. Standard is '" + h.outputName + "'\n" )
        << unifyLength( "--load", "[string]", "Loads matrices from path.\n" )
        << unifyLength( "--outEvery", "[int]", "Number of Runge-Kutta iterations for each plot. Standard is every " + std::to_string( h.out_modulo ) + " iteration\n" )
        << unifyLength( "--output", "[list of str]", "Comma seperated list of things to output. Available: mat,scalar,fft,pump,mask,psi,n. Many can also be specified with _plus or _minus.\n" )
        << unifyLength( "-nosfml", "no arguments", "If passed to the program, disables all live graphical output. \n" );
    std::cout << unifyLength( "Numerical parameters", "", "\n" ) << unifyLength( "Flag", "Inputs", "Description\n" )
              << unifyLength( "--N", "[int]", "Grid Dimensions (N x N). Standard is " + std::to_string( s.s_N ) + " x " + std::to_string( s.s_N ) + "\n" )
              << unifyLength( "--tstep", "[double]", "Timestep, standard is magic-timestep\n" )
              << unifyLength( "--tmax", "[double]", "Timelimit, standard is " + std::to_string( s.t_max ) + " ps\n" );
    std::cout << unifyLength( "System parameters", "", "\n" )
              << unifyLength( "Flag", "Inputs", "Description\n" )
              << unifyLength( "--gammaC", "[double]", "Standard is " + std::to_string( s.gamma_c ) + " ps^-1\n" )
              << unifyLength( "--gammaR", "[double]", "Standard is " + std::to_string( s.gamma_r / s.gamma_c ) + "*gammaC\n" )
              << unifyLength( "--gc", "[double]", "Standard is " + std::to_string( s.g_c ) + " meV mum^2\n" )
              << unifyLength( "--gr", "[double]", "Standard is " + std::to_string( s.g_r / s.g_c ) + "*gc\n" )
              << unifyLength( "--meff", "[double]", "Standard is " + std::to_string( s.m_eff ) + "\n" )
              << unifyLength( "--R", "[double]", "Standard is " + std::to_string( s.R ) + " ps^-1 mum^2\n" )
#ifdef TETMSPLITTING
              << unifyLength( "--g_pm", "[double]", "Standard is " + std::to_string( s.g_pm / s.g_c ) + "*gc\n" )
              << unifyLength( "--deltaLT", "[double]", "Standard is " + std::to_string( s.delta_LT ) + " meV\n" )
#endif
              << unifyLength( "--xmax", "[double]", "Standard is " + std::to_string( s.xmax ) + " mum\n" ) << std::endl;
    std::cout << unifyLength( "Pulse, pump and mask.", "", "\n" )
              << unifyLength( "Flag", "Inputs", "Description\n", 30, 80 )
              << unifyLength( "--pulse", "[double] [double] [double] [double] [int] [int] [double] [double] [double]", "t0, amplitude, frequency, sigma, m, pol, width, posX, posY, standard is no pulse. pol = +/-1 or 0 for both\n", 30, 80 )
              << unifyLength( "--pump", "[double] [double] [double] [double] [int] [double] [string]", "amplitude or 'adaptive', width, posX, posY, pol, exponent, type. standard is no pump. type can be either gauss or ring.\n", 30, 80 ) << unifyLength( " ", " ", "mPlus and mMinus are optional and take effect when --adjustStartingStates is provided\n", 30, 80 ) << unifyLength( "--adjustStartingStates, -ASS ", "", "Adjusts the polarization and amplitude of the starting Psi(+,-) and n(+,-) to match the pump given by --pump. Does nothing if no --pump is provided.\n", 30, 80 )
              << unifyLength( "--mask", "[double] [double] [double] [double] [int] [double]", "amplitude or 'adaptive', width, posX, posY, pol, exponent. standard is no mask.\n", 30, 80 )
              << unifyLength( "-masknorm", "no arguments", "If passed, both the mask and Psi will be normalized before calculating the error.\n", 30, 80 ) << std::endl;
#ifdef USECPU
    std::cout << unifyLength( "--threads", "[int]", "Standard is " + std::to_string( s.omp_max_threads ) + " Threads\n" ) << std::endl;
#endif
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

void addPump( System& s, real_number P0, real_number w, real_number x, real_number y, int pol, real_number exponent, int type ) {
    if ( P0 == 0.0 || w == 0 ) {
        std::cout << "Pump with P0 = 0 or w = 0 not allowed!" << std::endl;
        return;
    }
    s.pump_amp.push_back( P0 );
    s.pump_width.push_back( w );
    s.pump_X.push_back( x );
    s.pump_Y.push_back( y );
    s.pump_pol.push_back( pol );
    s.pump_type.push_back( type );
    s.pump_exponent.push_back( exponent );
    std::cout << "Added pump with P0 = " << P0 << " w = " << w << " x = " << x << " y = " << y << " pol = " << pol << " exponent = " << exponent << " type = " << ( type == 0 ? "ring shaped" : "gauss shaped" ) << std::endl;
}
void addMask( System& s, real_number amp, real_number w, real_number posX, real_number posY, int pol, real_number exponent ) {
    s.mask.amp.push_back( amp );
    s.mask.width.push_back( w );
    s.mask.x.push_back( posX );
    s.mask.y.push_back( posY );
    s.mask.pol.push_back( pol );
    s.mask.exponent.push_back( exponent );
    std::cout << "Added mask with amp = " << amp << " w = " << w << " posX = " << posX << " posY = " << posY << " pol = " << pol << " exponent = " << exponent << std::endl;
}

std::tuple<System, FileHandler> initializeSystem( int argc, char** argv ) {
    std::vector<std::string> arguments = argv_to_vec( argc, argv );
    // Initialize system and FileHandler
    System s;
    FileHandler h;

    // Initialize system
    int index = 0;

    if ( ( index = vec_find_str( "--path", arguments ) ) != -1 )
        h.outputPath = arguments.at( ++index );
    if ( h.outputPath.back() != '/' )
        h.outputPath += "/";

    if ( ( index = vec_find_str( "--name", arguments ) ) != -1 )
        h.outputName = arguments.at( ++index );

    if ( ( index = vec_find_str( "--outEvery", arguments ) ) != -1 )
        h.out_modulo = (int)getNextInput( arguments, "out_modulo", ++index );

    // Systemparameter
    s.m_eff = 1E-4 / s.h_bar_s * 5.6856; //      *m_e/h_bar;                 // m_e/hbar
    if ( ( index = vec_find_str( "--meff", arguments ) ) != -1 )
        s.m_eff = getNextInput( arguments, "m_eff", ++index );
    if ( ( index = vec_find_str( "--gammaC", arguments ) ) != -1 )
        s.gamma_c = getNextInput( arguments, "gamma_c", ++index );
    if ( ( index = vec_find_str( "--gammaR", arguments ) ) != -1 )
        s.gamma_r = getNextInput( arguments, "gamma_r", ++index );
    if ( ( index = vec_find_str( "--gc", arguments ) ) != -1 )
        s.g_c = getNextInput( arguments, "g_c", ++index );
    if ( ( index = vec_find_str( "--gr", arguments ) ) != -1 )
        s.g_r = getNextInput( arguments, "g_r", ++index );
    if ( ( index = vec_find_str( "--R", arguments ) ) != -1 ) {
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
    if ( ( index = vec_find_str( "--threads", arguments ) ) != -1 )
        s.omp_max_threads = (int)getNextInput( arguments, "threads", ++index );
    if ( ( index = vec_find_str( "--output", arguments ) ) != -1 ) {
        s.output_keys.clear();
        auto output_string = getNextStringInput( arguments, "output", ++index );
        // Split output_string at ","
        for (auto range : output_string | std::views::split(',')) {
            s.output_keys.emplace_back(std::string{std::ranges::begin(range), std::ranges::end(range)});
        }
    }

    // Numerik
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

    // If -masknorm is passed to the program, the mask and psi is normalized before the error calculations.
    if ( ( index = vec_find_str( "-masknorm", arguments ) ) != -1 ) {
        s.normalize_before_masking = true;
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
        auto samp = getNextStringInput( arguments, "pump_amp", ++index );
        real_number amp = samp == "adaptive" ? -666. : std::stod( samp );
        real_number w = getNextInput( arguments, "pump_width", index );
        real_number posX = getNextInput( arguments, "pump_X", index );
        real_number posY = getNextInput( arguments, "pump_Y", index );
        int pol = (int)getNextInput( arguments, "pump_pol", index );
        real_number exponent = getNextInput( arguments, "pump_exponent", index );
        auto stype = getNextStringInput( arguments, "pump_type", index );
        int type = 0;
        if ( stype == "ring" )
            type = 0;
        else if ( stype == "gauss" )
            type = 1;
        addPump( s, amp, w, posX, posY, pol, exponent, type );
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
    // Soll Mask.
    index = 0;
    while ( ( index = vec_find_str( "--mask", arguments, index ) ) != -1 ) {
        auto samp = getNextStringInput( arguments, "mask_amp", ++index );
        real_number amp = samp == "adaptive" ? -666. : std::stod( samp );
        real_number w = getNextInput( arguments, "mask_width", index );
        real_number posX = getNextInput( arguments, "mask_X", index );
        real_number posY = getNextInput( arguments, "mask_Y", index );
        int pol = (int)getNextInput( arguments, "mask_pol", index );
        real_number exponent = getNextInput( arguments, "mask_exponent", index );
        addMask( s, amp, w, posX, posY, pol, exponent );
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

    // Check if help is requested
    if ( vec_find_str( "--help", arguments ) != -1 || vec_find_str( "-h", arguments ) != -1 ) {
        printSystemHelp( s, h );
        exit( 0 );
    }

    // Creating output directory
    const int dir_err = std::system( ( "mkdir " + h.outputPath ).c_str() );
    if ( -1 == dir_err ) {
        std::cout << "Error creating directory " << h.outputPath << std::endl;
    } else {
        std::cout << "Succesfully created directory " << h.outputPath << std::endl;
    }

    return std::make_tuple( s, h );
}

void initializePumpVariables( System& s, FileHandler& filehandler ) {
    // Allocate memory for pump cache on cpu
    std::unique_ptr<real_number[]> host_pump_plus, host_pump_minus;
    host_pump_plus = std::make_unique<real_number[]>( s.s_N * s.s_N );
    host_pump_minus = std::make_unique<real_number[]>( s.s_N * s.s_N );
    for ( int col = 0; col < s.s_N; col++ ) {
        for ( int row = 0; row < s.s_N; row++ ) {
            auto x = -s.xmax / 2.0 + s.dx * col;
            auto y = -s.xmax / 2.0 + s.dx * row;
            int i = col * s.s_N + row;
            host_pump_plus[i] = 0;
            host_pump_minus[i] = 0;
            for ( int c = 0; c < s.pump_amp.size(); c++ ) {
                const real_number r_squared = abs2( x - s.pump_X[c] ) + abs2( y - s.pump_Y[c] );
                const auto w = s.pump_width[c];
                const auto exp_factor = std::pow( r_squared / w / w, s.pump_exponent[c] );
                auto pre_fractor = s.pump_type[c] == 0 ? exp_factor : 1.0;
                if ( s.pump_pol[c] >= 0 ) {
                    auto pump_amp = s.pump_amp[c] == -666 ? -1. * host_pump_plus[i] : s.pump_amp[c] / sqrt(2*3.1415) / w;
                    host_pump_plus[i] += pump_amp * pre_fractor * exp( -exp_factor );
                }
#ifdef TETMSPLITTING
                if ( s.pump_pol[c] <= 0 ) {
                    auto pump_amp = s.pump_amp[c] == 666 ? -1. * host_pump_minus[i] : s.pump_amp[c] / sqrt(2*3.1415) / w;
                    host_pump_minus[i] += pump_amp * pre_fractor * exp( -exp_factor );
                }
#endif
            }
        }
    }

    // Output Matrices to file
    if (s.doOutput("mat","pump_plus", "pump"))
    filehandler.outputMatrixToFile( host_pump_plus.get(), s, "pump_plus" );
#ifdef TETMSPLITTING
    if (s.doOutput("mat","pump_minus", "pump"))
    filehandler.outputMatrixToFile( host_pump_minus.get(), s, "pump_minus" );
#endif

    initializePumpVariables( host_pump_plus.get(), host_pump_minus.get(), s.s_N * s.s_N );
}

void initializePulseVariables( System& s ) {
    initializePulseVariables( s.pulse_t0.data(), s.pulse_amp.data(), s.pulse_freq.data(), s.pulse_sigma.data(), s.pulse_m.data(), s.pulse_pol.data(), s.pulse_width.data(), s.pulse_X.data(), s.pulse_Y.data(), s.pulse_t0.size() );
}

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
        buffer[i] = std::atan2( imag( z[i] ), real( z[i] ) );
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

void calculateSollValue( System& s, Buffer& buffer, FileHandler& filehandler ) {
    if ( s.mask.x.size() == 0 ) {
        std::cout << "No mask provided, skipping soll value calculation!" << std::endl;
        return;
    }

    // First, calculate soll mask from s.mask
    std::unique_ptr<real_number[]> host_mask_plus, host_mask_minus;
    host_mask_plus = std::make_unique<real_number[]>( s.s_N * s.s_N );
    host_mask_minus = std::make_unique<real_number[]>( s.s_N * s.s_N );
    for ( int col = 0; col < s.s_N; col++ ) {
        for ( int row = 0; row < s.s_N; row++ ) {
            auto x = -s.xmax / 2.0 + s.dx * col;
            auto y = -s.xmax / 2.0 + s.dx * row;
            int i = col * s.s_N + row;
            host_mask_plus[i] = 0;
            host_mask_minus[i] = 0;
            for ( int c = 0; c < s.mask.amp.size(); c++ ) {
                const real_number r_squared = abs2( x - s.mask.x[c] ) + abs2( y - s.mask.y[c] );
                const auto w = s.mask.width[c];
                const auto exp_factor = std::pow( r_squared / w / w, s.mask.exponent[c] );
                if ( s.mask.pol[c] >= 0 ) {
                    auto amp = s.mask.amp[c] == -666 ? -1. * host_mask_plus[i] : s.mask.amp[c];
                    host_mask_plus[i] += amp * exp( -exp_factor );
                }
#ifdef TETMSPLITTING
                if ( s.mask.pol[c] <= 0 ) {
                    auto amp = s.mask.amp[c] == 666 ? -1. * host_mask_minus[i] : s.mask.amp[c];
                    host_mask_minus[i] += amp * exp( -exp_factor );
                }
#endif
            }
        }
    }

    // Calculate min and max to normalize both Psi and the mask. We dont really care about efficiency here, since its
    // only done once.
    real_number max_mask_plus = 1.;
    real_number max_psi_plus = 1.;
    real_number max_mask_minus = 1.;
    real_number max_psi_minus = 1.;
    real_number min_mask_plus = 0.;
    real_number min_psi_plus = 0.;
    real_number min_mask_minus = 0.;
    real_number min_psi_minus = 0.;

    if ( s.normalize_before_masking ) {
        std::tie( min_mask_plus, max_mask_plus ) = minmax( host_mask_plus.get(), s.s_N * s.s_N, false /*Device Pointer*/ );
        std::tie( min_psi_plus, max_psi_plus ) = minmax( buffer.Psi_Plus.get(), s.s_N * s.s_N, false /*Device Pointer*/ );
#ifdef TETMSPLITTING
        std::tie( min_mask_minus, max_mask_minus ) = minmax( host_mask_minus.get(), s.s_N * s.s_N, false /*Device Pointer*/ );
        std::tie( min_psi_minus, max_psi_minus ) = minmax( buffer.Psi_Minus.get(), s.s_N * s.s_N, false /*Device Pointer*/ );
#endif
        std::cout << "The mask calculation will use normalizing constants:" << std::endl;
        std::cout << "min_mask_plus = " << min_mask_plus << " max_mask_plus = " << max_mask_plus << std::endl;
        std::cout << "min_psi_plus = " << min_psi_plus << " max_psi_plus = " << max_psi_plus << std::endl;
#ifdef TETMSPLITTING
        std::cout << "min_mask_minus = " << min_mask_minus << " max_mask_minus = " << max_mask_minus << std::endl;
        std::cout << "min_psi_minus = " << min_psi_minus << " max_psi_minus = " << max_psi_minus << std::endl;
#endif
    // Devide error by N^2
    max_mask_plus *= s.s_N * s.s_N;
    max_psi_plus *= s.s_N * s.s_N;
    max_mask_minus *= s.s_N * s.s_N;
    max_psi_minus *= s.s_N * s.s_N;
    
    }

    // Output Mask
    if (s.doOutput("mat","mask_plus", "mask"))
    filehandler.outputMatrixToFile( host_mask_plus.get(), s, "mask_plus" );
#ifdef TETMSPLITTING
    if (s.doOutput("mat","mask_minus", "mask"))
    filehandler.outputMatrixToFile( host_mask_minus.get(), s, "mask_minus" );
#endif
// Then, check matching
#pragma omp parallel for
    for ( int i = 0; i < s.s_N * s.s_N; i++ ) {
        host_mask_plus[i] = abs( abs( buffer.Psi_Plus.get()[i] ) / max_psi_plus - host_mask_plus[i] / max_mask_plus );
#ifdef TETMSPLITTING
        host_mask_minus[i] = abs( abs( buffer.Psi_Minus.get()[i] ) / max_psi_minus - host_mask_minus[i] / max_mask_minus );
#endif
    }
    // Thirdly, calculate sum of all elements
    real_number sum_plus = 0;
    std::ranges::for_each( host_mask_plus.get(), host_mask_plus.get() + s.s_N * s.s_N, [&sum_plus]( real_number n ) { sum_plus += n; } );
    std::cout << "Error in Psi_Plus: " << sum_plus << std::endl;
#ifdef TETMSPLITTING
    real_number sum_minus = 0;
    std::ranges::for_each( host_mask_plus.get(), host_mask_plus.get() + s.s_N * s.s_N, [&sum_minus]( real_number n ) { sum_minus += n; } );
    std::cout << "Error in Psi_Minus: " << sum_minus << std::endl;
#endif
}