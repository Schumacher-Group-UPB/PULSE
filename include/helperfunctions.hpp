#pragma once
#include <fstream>
#include <iostream>
#include <vector>
#include <complex>
using namespace std::complex_literals;
#include "system.hpp"

std::vector<std::string> argv_to_vec( int argc, char** argv ) {
    std::vector<std::string> ret;
    ret.reserve( argc );
    for ( int i = 0; i < argc; i++ )
        ret.push_back( std::string( argv[i] ) );
    return ret;
}

int vec_find_str( std::string toFind, const std::vector<std::string>& input, int start = 0 ) {
    for ( int i = start; i < input.size(); i++ ) {
        if ( input.at( i ).compare( toFind ) == 0 )
            return i;
    }
    return -1;
}

double getNextInput( const std::vector<std::string>& arguments, const std::string name, int& index ) {
    std::cout << "Read input " << name << " as " << arguments.at( index ) << std::endl;
    return std::stof( arguments.at( index++ ) );
}

std::string unifyLength( std::string indicator, std::string unit, std::string description, int L1 = 30, int L2 = 30 ) {
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

static void printSystemHelp( System& s, MatrixHandler& h ) {
    std::cout << "Welcome to the 'Spriddis fasdzinirnde Schdruhdelsdheoriesrechnungs'. Todey i wil sho you, how to modifi dis progrem.\n\n";
    std::cout << unifyLength( "General parameters:", "", "\n" ) 
    << unifyLength( "Flag", "Inputs", "Description\n" ) 
    << unifyLength( "--path", "[string]", "Workingfolder. Standard is '" + h.outputPath + "'\n" ) 
    << unifyLength( "--name", "[string]", "File prefix. Standard is '" + h.outputName + "'\n" ) 
    << unifyLength( "--load", "[string]", "Loads matrices from path.\n" ) 
    << unifyLength( "--addOut, --addOutEvery", "[double]", "[NOT IMPLEMENTED] Adds imageoutput at X ps or every X ps.\n" ) 
    << unifyLength( "--plotEvery", "[int]", "Number of Runge-Kutta iterations for each plot. Standard is every " + std::to_string( h.plotmodulo ) + " iteration\n" ) 
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
    << unifyLength( "--g_pm", "[double]", "Standard is " + std::to_string( s.g_pm / s.g_c ) + "*gc\n" ) 
    << unifyLength( "--R", "[double]", "Standard is " + std::to_string( s.R ) + " ps^-1 mum^2\n" ) 
    << unifyLength( "--deltaLT", "[double]", "Standard is " + std::to_string( s.delta_LT ) + " meV\n" ) 
    << unifyLength( "--xmax", "[double]", "Standard is " + std::to_string( s.xmax ) + " mum\n" ) << std::endl;
    std::cout << unifyLength( "Pulse and pump. Warning: Adding a pump here overwrites all previous pump settings!", "", "\n" ) 
    << unifyLength( "Flag", "Inputs", "Description\n", 30, 80 ) 
    << unifyLength( "--pulse ", "[double] [double] [double] [double] [int] [int] [double] [double] [double]", "t0, amplitude, frequency, sigma, m, pol, width, posX, posY, standard is no pulse. pol = +/-1 or 0 for both\n", 30, 80 ) 
    << unifyLength( "--pump ", "[double] [double] [double] [double] [int] {int} {int}", "amplitude, width, posX, posY, pol, {mPlus, mMinus} standard is the pump given by previous parameters.\n", 30, 80 ) << unifyLength( " ", " ", "mPlus and mMinus are optional and take effect when --adjustStartingStates is provided\n", 30, 80 ) << unifyLength( "--adjustStartingStates, -ASS ", "", "Adjusts the polarization and amplitude of the starting Psi(+,-) and n(+,-) to match the pump given by --pump. Does nothing if no --pump is provided.\n", 30, 80 ) << std::endl;
}

void addPulse( System& s, double t0, double amp, double freq, double sigma, int m, int pol, double width, double x, double y ) {
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

void addPump( System& s, double P0, double w, double x, double y, int pol ) {
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

std::tuple<System, MatrixHandler> initializeSystem( int argc, char** argv ) {
    std::vector<std::string> arguments = argv_to_vec( argc, argv );
    // Initialize system and matrixhandler
    System s;
    MatrixHandler h;
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

    if ( ( index = vec_find_str( "--plotEvery", arguments ) ) != -1 )
        h.plotmodulo = (int)getNextInput( arguments, "plotmodulo", ++index );
    // if ( ( index = vec_find_str( "--threads", arguments ) ) != -1 )
    //     oms.set_num_threads( (int)getNextInput( arguments, "max_threads", ++index ) );

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
    if ( ( index = vec_find_str( "--R", arguments ) ) != -1 )
        s.R = getNextInput( arguments, "R", ++index );
     if ( ( index = vec_find_str( "--xmax", arguments ) ) != -1 )
         s.xmax = getNextInput( arguments, "xmax", ++index );
    if ( ( index = vec_find_str( "--g_pm", arguments ) ) != -1 )
        s.g_pm = getNextInput( arguments, "s.gm", ++index );
    if ( ( index = vec_find_str( "--deltaLT", arguments ) ) != -1 )
        s.delta_LT = getNextInput( arguments, "deltaLT", ++index );
    if ( ( index = vec_find_str( "--mPlus", arguments ) ) != -1 )
        s.m_plus = getNextInput( arguments, "m_plus", ++index );
    if ( ( index = vec_find_str( "--mMinus", arguments ) ) != -1 )
        s.m_minus = getNextInput( arguments, "m_plus", ++index );
    if ( ( index = vec_find_str( "--fft", arguments ) ) != -1 )
        s.fft_every = getNextInput( arguments, "fft_every", ++index );
    // TODO: das hier in --pump integrieren und dann hier weg
    // if ( ( index = vec_find_str( "--m", arguments ) ) != -1 ) {
    //    s.m_plus = getNextInput( arguments, "m_plus", ++index );
    //    s.m_minus = getNextInput( arguments, "m_minus", index );
    //}
    // TODO: das hier auch
    if ( ( index = vec_find_str( "-noNormPhase", arguments ) ) != -1 )
        s.normalize_phase_states = false;
    if ( ( index = vec_find_str( "-normPulsePhase", arguments ) ) != -1 )
        s.normalizePhasePulse = true;

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
    if ( ( index = vec_find_str( "--tstep", arguments ) ) != -1 )
        s.dt = getNextInput( arguments, "t_step", ++index );
    // Initialize t_0 as 0.
    s.t = 0.0;
    // Pumps
    index = 0;
    while ( ( index = vec_find_str( "--pump", arguments, index ) ) != -1 ) {
        double amp = getNextInput( arguments, "pump_amp", ++index );
        double w = getNextInput( arguments, "pump_width", index );
        double posX = getNextInput( arguments, "pump_X", index );
        double posY = getNextInput( arguments, "pump_Y", index );
        int pol = (int)getNextInput( arguments, "pump_pol", index );
        addPump( s, amp, w, posX, posY, pol );
    }
    // Pulses
    index = 0;
    while ( ( index = vec_find_str( "--pulse", arguments, index ) ) != -1 ) {
        double t0 = getNextInput( arguments, "pulse_t0", ++index );
        double amp = getNextInput( arguments, "pulse_amp", index );
        double freq = getNextInput( arguments, "pulse_freq", index );
        double sigma = getNextInput( arguments, "pulse_sigma", index );
        int m = getNextInput( arguments, "pulse_m", index );
        int pol = (int)getNextInput( arguments, "pulse_pol", index );
        double w = getNextInput( arguments, "pulse_width", index );
        double posX = getNextInput( arguments, "pulse_X", index );
        double posY = getNextInput( arguments, "pulse_Y", index );
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

template <typename T>
inline double cwiseAbs2( T z ) {
    return std::real( z ) * std::real( z ) + std::imag( z ) * std::imag( z );
}
template <typename T>
inline void cwiseAbs2( T* z, double* buffer, int size ) {
    #pragma omp parallel for
    for ( int i = 0; i < size; i++ )
        buffer[i] = std::real( z[i] ) * std::real( z[i] ) + std::imag( z[i] ) * std::imag( z[i] );
}
inline std::tuple<double, double> normalize( double* buffer, int size ) {
    double max = 0;
    double min = 0;
    for ( int i = 0; i < size; i++ ) {
        max = std::max( max, buffer[i] );
        min = std::min( min, buffer[i] );
    }
    #pragma omp parallel for
    for ( int i = 0; i < size; i++ )
        buffer[i] = ( buffer[i] - min ) / ( max - min );
    return std::make_tuple( min, max );
}

inline void angle( Scalar* z, double* buffer, int size ) {
    #pragma omp parallel for
    for ( int i = 0; i < size; i++ )
        buffer[i] = std::arg( z[i] );
}

inline double sign( double x ) {
    return ( x > 0 ) - ( x < 0 );
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