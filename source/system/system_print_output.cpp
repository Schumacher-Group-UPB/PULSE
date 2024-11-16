#include <ctime>
#include <iomanip> // std::setprecision, std::setw, std::setfill
#include "system/system_parameters.hpp"
#include "cuda/cuda_matrix_base.hpp"
#include "misc/commandline_io.hpp"
#include "misc/escape_sequences.hpp"
#include "misc/timeit.hpp"
#include "omp.h"

// File-Local Configuration
static size_t console_width = 100;
static char seperator = '-';

/*
Prints PHOENIX. Font is CyberLarge
 _____    _     _    _____    _______   __   _   _____   _     _  
|_____]   |_____|   |     |   |______   | \  |     |      \___/   
|       . |     | . |_____| . |______ . |  \_| . __|__ . _/   \_ .

*/
void print_name() {
    std::cout << PHOENIX::CLIO::fillLine( console_width, seperator ) << "\n\n"; // Horizontal Seperator
    std::cout << EscapeSequence::ORANGE << EscapeSequence::BOLD;                // Make Text Bold
    // Print Phoenix LOGO
    std::cout << PHOENIX::CLIO::centerString( " _____    _     _    _____    _______   __   _   _____   _     _  ", console_width ) << "\n";
    std::cout << PHOENIX::CLIO::centerString( "|_____]   |_____|   |     |   |______   | \\  |     |      \\___/   ", console_width ) << "\n";
    std::cout << PHOENIX::CLIO::centerString( "|       . |     | . |_____| . |______ . |  \\_| . __|__ . _/   \\_ .", console_width ) << "\n\n";
    std::stringstream ss;
    // Paderborn Highly Optimized and Energy efficient solver for two-dimensional Nonlinear Schrödinger equations with Integrated Xtensions
    ss << EscapeSequence::RESET << EscapeSequence::UNDERLINE << EscapeSequence::BOLD << EscapeSequence::BLUE << "P" << EscapeSequence::GRAY << "aderborn " << EscapeSequence::BLUE << "H" << EscapeSequence::GRAY << "ighly " << EscapeSequence::BLUE << "O" << EscapeSequence::GRAY << "ptimized and " << EscapeSequence::BLUE << "E" << EscapeSequence::GRAY << "nergy efficient solver for two-dimensional ";
    std::cout << PHOENIX::CLIO::centerStringRaw( ss.str(), console_width, "Paderborn Highly Optimized and Energy efficient solver for two-dimensional" ) << std::endl;
    ss.str( "" );
    ss.clear();
    ss << EscapeSequence::BLUE << "N" << EscapeSequence::GRAY << "onlinear Schroedinger equations with " << EscapeSequence::BLUE << "I" << EscapeSequence::GRAY << "ntegrated e" << EscapeSequence::BLUE << "X" << EscapeSequence::GRAY << "tensions" << EscapeSequence::RESET;
    std::cout << PHOENIX::CLIO::centerStringRaw( ss.str(), console_width, "Nonlinear Schroedinger equations with Integrated Xtensions" ) << std::endl;
    ss.str( "" );
    ss.clear();
    ss << "Version: " << EscapeSequence::BOLD << EscapeSequence::BLUE << "0.1.0" << EscapeSequence::RESET;
    std::cout << PHOENIX::CLIO::centerStringRaw( ss.str(), console_width, "Version: 0.1.0" ) << std::endl;
    std::cout << PHOENIX::CLIO::centerString( "https://github.com/Schumacher-Group-UPB/PHOENIX_", console_width ) << std::endl;
    std::cout << PHOENIX::CLIO::fillLine( console_width, seperator ) << "\n"; // Horizontal Seperator
}

void PHOENIX::SystemParameters::printHelp() {
    print_name();
#ifndef USE_32_BIT_PRECISION
    std::cout << "This program is compiled with " << EscapeSequence::UNDERLINE << EscapeSequence::YELLOW << "double precision" << EscapeSequence::RESET << " numbers.\n";
#else
    std::cout << "This program is compiled with " << EscapeSequence::UNDERLINE << EscapeSequence::YELLOW << "single precision" << EscapeSequence::RESET << " numbers.\n";
#endif
#ifdef USE_CPU
    std::cout << "This program is compiled as a " << EscapeSequence::UNDERLINE << EscapeSequence::YELLOW << "CPU Version" << EscapeSequence::RESET << ".\n";
    std::cout << "Maximum number of CPU cores utilized " << omp_get_max_threads() << std::endl;
#endif
    std::cout << EscapeSequence::BOLD << PHOENIX::CLIO::fillLine( console_width, seperator ) << EscapeSequence::RESET << std::endl
              << PHOENIX::CLIO::unifyLength( "Flag", "Inputs", "Description\n" ) << std::endl
              << PHOENIX::CLIO::unifyLength( "--path", "<string>", "Workingfolder. Standard is '" + filehandler.outputPath + "'" ) << std::endl
              << PHOENIX::CLIO::unifyLength( "--name", "<string>", "File prefix. Standard is '" + filehandler.outputName + "'" )
              << std::endl
              //<< PHOENIX::CLIO::unifyLength( "--loadFrom", "<string> <string...>", "Loads list of matrices from path." ) << std::endl
              << PHOENIX::CLIO::unifyLength( "--config", "<string>", "Loads configuration from file." ) << std::endl
              << PHOENIX::CLIO::unifyLength( "--outEvery", "<int>", "Number of Runge-Kutta iterations for each plot. Standard is every " + std::to_string( output_every ) + " ps" ) << std::endl
              << PHOENIX::CLIO::unifyLength( "--output", "<string...>", "Comma seperated list of things to output. Available: mat,scalar,fft,pump,mask,psi,n. Many can also be specified with _plus or _minus." )
              << std::endl
              //<< PHOENIX::CLIO::unifyLength( "--history", "<Y> <points>", "Outputs a maximum number of x-slices at Y for history. y-slices are not supported." ) << std::endl
              << PHOENIX::CLIO::unifyLength( "--historyMatrix", "<int> <int> <int> <int> <int>", "Outputs the matrices specified in --output with specified startx,endx,starty,endy index and increment." ) << PHOENIX::CLIO::unifyLength( "--historyTime", "<float> <int>", "Outputs the matrices specified in --output after starting time is reached and only every multiple*outEvery times with <start> <multiple>." )
              << std::endl
              //<< PHOENIX::CLIO::unifyLength( "--input", "<string...>", "Comma seperated list of things to input. Available: mat,scalar,fft,pump,mask,psi,n. Many can also be specified with _plus or _minus." ) << std::endl
              << PHOENIX::CLIO::unifyLength( "-nosfml", "no arguments", "If passed to the program, disables all live graphical output. " ) << std::endl;
    std::cout << PHOENIX::CLIO::fillLine( console_width, seperator ) << std::endl;
    std::cout << PHOENIX::CLIO::unifyLength( "Numerical parameters", "", "" ) << std::endl
              << PHOENIX::CLIO::unifyLength( "Flag", "Inputs", "Description" ) << std::endl
              << PHOENIX::CLIO::unifyLength( "--N", "<int> <int>", "Grid Dimensions (N x N). Standard is " + std::to_string( p.N_c ) + " x " + std::to_string( p.N_r ) ) << std::endl
              << PHOENIX::CLIO::unifyLength( "--subgrids", "<int> <int>", "Subgrid Dimensions (N x N). Standard is " + std::to_string( p.subgrids_columns ) + " x " + std::to_string( p.subgrids_rows ) ) << std::endl
              << PHOENIX::CLIO::unifyLength( "--tstep", "<double>", "Timestep, standard is magic-timestep = " + PHOENIX::CLIO::to_str( magic_timestep ) + "ps" ) << std::endl
              << PHOENIX::CLIO::unifyLength( "--tmax", "<double>", "Timelimit, standard is " + PHOENIX::CLIO::to_str( t_max ) + " ps" ) << std::endl
              << PHOENIX::CLIO::unifyLength( "--iterator", "<string>", "RK4, RK45 or SSFM" ) << std::endl
              << PHOENIX::CLIO::unifyLength( "-rk45", "no arguments", "Shortcut to use RK45" ) << std::endl
              << PHOENIX::CLIO::unifyLength( "--rk45dt", "<double> <double>", "dt_min and dt_max for the RK45 method" ) << std::endl
              << PHOENIX::CLIO::unifyLength( "--tol", "<double>", "RK45 Tolerance, standard is " + PHOENIX::CLIO::to_str( tolerance ) + " ps" ) << std::endl
              << PHOENIX::CLIO::unifyLength( "-ssfm", "no arguments", "Shortcut to use SSFM" ) << std::endl
              << PHOENIX::CLIO::unifyLength( "--imagTime", "<double>", "Use imaginary time propagation with a given norm. Currently only works in conjunction with -ssfm/--iterator ssfm" ) << std::endl
              << PHOENIX::CLIO::unifyLength( "--boundary", "<string> <string>", "Boundary conditions for x and y. Is either 'periodic' or 'zero'." ) << std::endl;
    std::cout << PHOENIX::CLIO::fillLine( console_width, seperator ) << std::endl;
    std::cout << PHOENIX::CLIO::unifyLength( "System Parameters", "", "" ) << std::endl
              << PHOENIX::CLIO::unifyLength( "Flag", "Inputs", "Description" ) << std::endl
              << PHOENIX::CLIO::unifyLength( "--gammaC", "<double>", "Standard is " + PHOENIX::CLIO::to_str( p.gamma_c ) + " ps^-1" ) << std::endl
              << PHOENIX::CLIO::unifyLength( "--gammaR", "<double>", "Standard is " + PHOENIX::CLIO::to_str( p.gamma_r / p.gamma_c ) + "*gammaC" ) << std::endl
              << PHOENIX::CLIO::unifyLength( "--gc", "<double>", "Standard is " + PHOENIX::CLIO::to_str( p.g_c ) + " eV mum^2" ) << std::endl
              << PHOENIX::CLIO::unifyLength( "--gr", "<double>", "Standard is " + PHOENIX::CLIO::to_str( p.g_r / p.g_c ) + "*gc" ) << std::endl
              << PHOENIX::CLIO::unifyLength( "--meff", "<double>", "Standard is " + PHOENIX::CLIO::to_str( p.m_eff ) ) << std::endl
              << PHOENIX::CLIO::unifyLength( "--R", "<double>", "Standard is " + PHOENIX::CLIO::to_str( p.R ) + " ps^-1 mum^2" ) << std::endl
              << PHOENIX::CLIO::unifyLength( "--g_pm", "<double>", "Standard is " + PHOENIX::CLIO::to_str( p.g_pm / p.g_c ) + "*gc. Only effective in a system with TE/TM splitting." ) << std::endl
              << PHOENIX::CLIO::unifyLength( "--deltaLT", "<double>", "Standard is " + PHOENIX::CLIO::to_str( p.delta_LT ) + " eV. Only effective in a system with TE/TM splitting." ) << std::endl
              << PHOENIX::CLIO::unifyLength( "--L", "<double> <double>", "Standard is " + PHOENIX::CLIO::to_str( p.L_x ) + ", " + PHOENIX::CLIO::to_str( p.L_y ) + " mum" ) << std::endl;
    std::cout << PHOENIX::CLIO::fillLine( console_width, seperator ) << std::endl;
    std::cout << PHOENIX::CLIO::unifyLength( "Envelopes.", "", "" ) << std::endl
              << PHOENIX::CLIO::unifyLength( "Envelopes are passed using either their spatial and temporal characteristics, or by loading an external file. Syntax:", "", "" ) << std::endl
              << PHOENIX::CLIO::unifyLength( "--envelope", "<double> <string> <double> <double> <double> <double> <string> <double> <double> <string> time <double> <double> <double>",
                                             "amplitude, behaviour (add,multiply,replace,adaptive,complex), widthX, widthY, posX, posY, pol (plus,minus,both), exponent, charge, type "
                                             "(gauss, ring), [t0, frequency, sigma]. 'time' signals the temporal envelope, which can be omitted for constant envelope. If 'time' is "
                                             "given: kind (gauss: ~cos(wt), cos: ~cos(wt), iexp: ~exp(iwt),) t0, frequency, sigma" )
              << std::endl
              << PHOENIX::CLIO::unifyLength( "--envelope", "<double> <string> <double> <double> <double> <double> <string> <double> <double> <string> time load <string>",
                                             "amplitude, behaviour (add,multiply,replace,adaptive,complex), widthX, widthY, posX, posY, pol (plus,minus,both), exponent, charge, type (gauss, ring), [t0, "
                                             "frequency, sigma]. 'time' signals the temporal envelope, which can be omitted for constant envelope. If 'time' is given: path" )
              << std::endl
              << PHOENIX::CLIO::unifyLength( "--envelope", "load <string> <double> <string> <string> time <string> <double> <double> <double>", "path, amplitude, behaviour (add,multiply,replace,adaptive,complex), pol (plus,minus,both)." ) << std::endl
              << PHOENIX::CLIO::unifyLength( "--envelope", "load <string> <double> <string> <string> time load <string> ", "path, amplitude, behaviour (add,multiply,replace,adaptive,complex), pol (plus,minus,both). For time: path" ) << std::endl
              << "Possible Envelopes include:" << std::endl
              << PHOENIX::CLIO::unifyLength( "--pump", "Spatial and Temporal ~cos(wt)", "" ) << std::endl
              << PHOENIX::CLIO::unifyLength( "--potential", "Spatial and Temporal ~cos(wt)", "" ) << std::endl
              << PHOENIX::CLIO::unifyLength( "--initialState", "Spatial", "" ) << std::endl
              << PHOENIX::CLIO::unifyLength( "--initialReservoir", "Spatial", "" ) << std::endl
              << PHOENIX::CLIO::unifyLength( "--pulse", "Spatial and Temporal ~exp(iwt)", "" ) << std::endl
              << PHOENIX::CLIO::unifyLength( "--fftMask", "Spatial", "" ) << std::endl
              << "Additional Parameters:" << std::endl
              << PHOENIX::CLIO::unifyLength( "--fftEvery", "<int>", "Apply FFT Filter every x ps" ) << std::endl
              << PHOENIX::CLIO::unifyLength( "--initRandom", "<double>", "Amplitude. Randomly initialize Psi" ) << std::endl;
    std::cout << PHOENIX::CLIO::fillLine( console_width, seperator ) << std::endl;
    std::cout << PHOENIX::CLIO::unifyLength( "SI Scalings", "", "" ) << std::endl
              << PHOENIX::CLIO::unifyLength( "Flag", "Inputs", "Description" ) << std::endl
              << PHOENIX::CLIO::unifyLength( "--hbar", "<double>", "Standard is " + PHOENIX::CLIO::to_str( p.h_bar ) ) << std::endl
              << PHOENIX::CLIO::unifyLength( "--e", "<double>", "Standard is " + PHOENIX::CLIO::to_str( p.e_e ) ) << std::endl
              << PHOENIX::CLIO::unifyLength( "--me", "<double>", "Standard is " + PHOENIX::CLIO::to_str( p.m_e ) ) << std::endl
              << PHOENIX::CLIO::unifyLength( "--hbarscaled", "<double>", "Standard is " + PHOENIX::CLIO::to_str( p.h_bar_s ) ) << std::endl
              << PHOENIX::CLIO::unifyLength( "--meff", "<double>", "Standard is " + PHOENIX::CLIO::to_str( p.m_eff ) ) << std::endl;
#ifdef USE_CPU
    std::cout << PHOENIX::CLIO::unifyLength( "--threads", "<int>", "Standard is " + std::to_string( omp_max_threads ) + " Threads\n" ) << std::endl;
#endif
}

void PHOENIX::SystemParameters::printSummary( std::map<std::string, std::vector<double>> timeit_times, std::map<std::string, double> timeit_times_total ) {
    print_name();
    const int l = 35;
    std::cout << EscapeSequence::BOLD << PHOENIX::CLIO::fillLine( console_width, seperator ) << EscapeSequence::RESET << std::endl;
    std::cout << EscapeSequence::BOLD << PHOENIX::CLIO::centerString( " Parameters ", console_width, '-' ) << EscapeSequence::RESET << std::endl;
    std::cout << PHOENIX::CLIO::unifyLength( "Grid Configuration", "---", "---", l, l, l, " " ) << std::endl;
    std::cout << PHOENIX::CLIO::unifyLength( "N", std::to_string( p.N_c ) + ", " + std::to_string( p.N_r ), "", l, l, l, " " ) << std::endl;
    std::cout << PHOENIX::CLIO::unifyLength( "N^2", std::to_string( p.N_c * p.N_r ), "", l, l, l, " " ) << std::endl;
    std::cout << PHOENIX::CLIO::unifyLength( "Subgrids", std::to_string( p.subgrids_columns ) + ", " + std::to_string( p.subgrids_rows ), "", l, l, l, " " ) << std::endl;
    std::cout << PHOENIX::CLIO::unifyLength( "Total Subgrids", std::to_string( p.subgrids_columns * p.subgrids_rows ), "", l, l, l, " " ) << std::endl;
    const double subgrid_overhead = ( ( p.subgrid_N_r + 2.0 * p.halo_size ) * ( p.subgrid_N_c + 2 * p.halo_size ) * ( p.subgrids_columns * p.subgrids_rows ) / ( p.N_r * p.N_c ) - 1.0 ) * 100.0;
    std::cout << PHOENIX::CLIO::unifyLength( "Subgrid Overhead", std::to_string( subgrid_overhead ), "%", l, l, l, " " ) << std::endl;
    std::cout << PHOENIX::CLIO::unifyLength( "Lx", PHOENIX::CLIO::to_str( p.L_x ), "mum", l, l, l, " " ) << std::endl;
    std::cout << PHOENIX::CLIO::unifyLength( "Ly", PHOENIX::CLIO::to_str( p.L_y ), "mum", l, l, l, " " ) << std::endl;
    std::cout << PHOENIX::CLIO::unifyLength( "dx", PHOENIX::CLIO::to_str( p.dx ), "mum", l, l, l, " " ) << std::endl;
    std::cout << PHOENIX::CLIO::unifyLength( "dy", PHOENIX::CLIO::to_str( p.dx ), "mum", l, l, l, " " ) << std::endl;
    std::cout << PHOENIX::CLIO::unifyLength( "System Configuration", "---", "---", l, l, l, " " ) << std::endl;
    std::cout << PHOENIX::CLIO::unifyLength( "tmax", PHOENIX::CLIO::to_str( t_max ), "ps", l, l, l, " " ) << std::endl;
    std::cout << PHOENIX::CLIO::unifyLength( "dt", PHOENIX::CLIO::to_str( p.dt ), "ps", l, l, l, " " ) << std::endl;
    std::cout << PHOENIX::CLIO::unifyLength( "gamma_c", PHOENIX::CLIO::to_str( p.gamma_c ), "ps^-1", l, l, l, " " ) << std::endl;
    std::cout << PHOENIX::CLIO::unifyLength( "gamma_r", PHOENIX::CLIO::to_str( p.gamma_r ), "ps^-1", l, l, l, " " ) << std::endl;
    std::cout << PHOENIX::CLIO::unifyLength( "g_c", PHOENIX::CLIO::to_str( p.g_c ), "eV mum^2", l, l, l, " " ) << std::endl;
    std::cout << PHOENIX::CLIO::unifyLength( "g_r", PHOENIX::CLIO::to_str( p.g_r ), "eV mum^2", l, l, l, " " ) << std::endl;
    std::cout << PHOENIX::CLIO::unifyLength( "g_pm", PHOENIX::CLIO::to_str( p.g_pm ), "eV mum^2", l, l, l, " " ) << std::endl;
    std::cout << PHOENIX::CLIO::unifyLength( "R", PHOENIX::CLIO::to_str( p.R ), "ps^-1 mum^-2", l, l, l, " " ) << std::endl;
    std::cout << PHOENIX::CLIO::unifyLength( "delta_LT", PHOENIX::CLIO::to_str( p.delta_LT ), "eV", l, l, l, " " ) << std::endl;
    std::cout << PHOENIX::CLIO::unifyLength( "m_eff", PHOENIX::CLIO::to_str( p.m_eff ), "", l, l, l, " " ) << std::endl;
    std::cout << PHOENIX::CLIO::unifyLength( "h_bar_s", PHOENIX::CLIO::to_str( p.h_bar_s ), "", l, l, l, " " ) << std::endl;
    std::cout << "Boundary Condition: " << ( p.periodic_boundary_x ? "Periodic" : "Zero" ) << "(x):" << ( p.periodic_boundary_y ? "Periodic" : "Zero" ) << "(y)" << std::endl;
    std::cout << PHOENIX::CLIO::centerString( " Envelope Functions ", console_width, '-' ) << std::endl;
    // TODO: overwrite << operator of the Envelope Class
    if ( pulse.size() > 0 )
        std::cout << "Pulse Envelopes:\n" << pulse.toString();
    if ( pump.size() > 0 )
        std::cout << "Pump Envelopes:\n" << pump.toString();
    if ( potential.size() > 0 )
        std::cout << "Potential Envelopes:\n" << potential.toString();
    if ( fft_mask.size() > 0 )
        std::cout << "FFT Mask Envelopes:\n" << fft_mask.toString();
    if ( initial_state.size() > 0 )
        std::cout << "Initial State Envelopes:\n" << initial_state.toString();
    std::cout << EscapeSequence::BOLD << PHOENIX::CLIO::centerString( " Runtime Statistics ", console_width, '-' ) << EscapeSequence::RESET << std::endl;
    double total = PHOENIX::TimeIt::totalRuntime();
#ifdef BENCH
    std::cout << "Total Runtime: " << total << "s --> " << ( total / p.t * 1E3 ) << "ms/ps --> " << ( p.t / total ) << "ps/s --> " << ( total / iteration ) * 1e6 << " mus/it" << std::endl;
#else
    std::cout << "Total Runtime: " << total << "s --> " << ( total / p.t * 1E3 ) << "ms/ps --> " << ( p.t / total ) << "ps/s --> " << ( total / iteration ) << "s/it" << std::endl;
#endif
    std::cout << EscapeSequence::BOLD << PHOENIX::CLIO::centerString( " Infos ", console_width, '-' ) << EscapeSequence::RESET << std::endl;

    std::cout << "Calculations done using the '" << iterator << "' solver" << std::endl;
    if ( iterator == "rk45" ) {
        std::cout << " = Tolerance used: " << tolerance << std::endl;
        std::cout << " = dt_max used: " << dt_max << std::endl;
        std::cout << " = dt_min used: " << dt_min << std::endl;
    }

    std::cout << "Calculated until t = " << p.t << "ps" << std::endl;
    if ( fft_mask.size() > 0 )
        std::cout << "Applying FFT every " << fft_every << " ps" << std::endl;
    std::cout << "Output variables and plots every " << output_every << " ps" << std::endl;
    std::cout << "Total allocated space for Device Matrices: " << CUDAMatrixBase::global_total_device_mb_max << " MB." << std::endl;
    std::cout << "Total allocated space for Host Matrices: " << CUDAMatrixBase::global_total_host_mb_max << " MB." << std::endl;
    std::cout << "Random Seed was: " << random_seed << std::endl;
#ifdef USE_32_BIT_PRECISION
    std::cout << "This program is compiled using " << EscapeSequence::UNDERLINE << EscapeSequence::BLUE << "single precision" << EscapeSequence::RESET << " numbers.\n";
#else
    std::cout << "This program is compiled using " << EscapeSequence::UNDERLINE << EscapeSequence::BLUE << "double precision" << EscapeSequence::RESET << " numbers.\n";
#endif
#ifdef USE_CPU
    std::cout << "Device Used: " << EscapeSequence::BOLD << EscapeSequence::YELLOW << "CPU" << EscapeSequence::RESET << std::endl;
    std::cout << EscapeSequence::GRAY << "  CPU cores utilized: " << omp_max_threads << EscapeSequence::RESET << std::endl;
#else
    // The Headers required for this come from system_parameters.hpp->typedef.cuh
    int nDevices;
    cudaGetDeviceCount( &nDevices );
    int device;
    cudaGetDevice( &device );
    cudaDeviceProp prop;
    cudaGetDeviceProperties( &prop, device );
    std::cout << "Device Used: " << EscapeSequence::GREEN << EscapeSequence::BOLD << prop.name << EscapeSequence::RESET << std::endl;
    //std::cout << EscapeSequence::GRAY << "  Memory Clock Rate (GHz): " << prop.memoryClockRate / 1024.0 / 1024.0 << std::endl;
    std::cout << "  Peak Memory Bandwidth (GB/s): " << 2.0 * prop.memoryClockRate * ( prop.memoryBusWidth / 8 ) / 1.0e6 << std::endl;
    std::cout << "  Total Global Memory (GB): " << (float)( prop.totalGlobalMem ) / 1024.0 / 1024.0 / 1024.0 << std::endl;
    std::cout << "  Total L2 Memory (MB): " << (float)( prop.l2CacheSize ) / 1024.0 / 1024.0 << std::endl;
    //std::cout << "  Warp-size: " << prop.warpSize << std::endl;
    //std::cout << "  CUDA Cores: " << prop.multiProcessorCount * _ConvertSMVer2Cores( prop.major, prop.minor ) << std::endl;
    //std::cout << "  GPU Clock Rate (GHz): " << prop.clockRate / 1024.0 / 1024.0 << EscapeSequence::RESET << std::endl;
#endif
    std::cout << EscapeSequence::BOLD << PHOENIX::CLIO::fillLine( console_width, '=' ) << EscapeSequence::RESET << std::endl;
}

double _PHOENIX_last_output_time = 0.;

void PHOENIX::SystemParameters::printCMD( double complete_duration, double complete_iterations ) {
    // TODO: move this into function and hide the ugly thing where noone can find it.
    if ( std::time( nullptr ) - _PHOENIX_last_output_time < 0.25 )
        return;
    // Print Runtime
    std::cout << EscapeSequence::HIDE_CURSOR;
    std::cout << PHOENIX::CLIO::fillLine( console_width, seperator ) << std::endl;
    std::cout << "    T = " << int( p.t ) << "ps - dt = " << std::setprecision( 2 ) << p.dt << "ps    \n";
    // Progressbar for p.t/t_max
    std::cout << "    Progress:  " << PHOENIX::CLIO::createProgressBar( p.t, t_max, console_width - 30 ) << "    \n";
    bool evaluate_stochastic = evaluateStochastic();
    std::cout << "    Current System: " << ( use_twin_mode ? "TE/TM" : "Scalar" ) << " - " << ( evaluate_stochastic ? "With Stochastic" : "No Stochastic" ) << "    \n";
    std::cout << "    Runtime: " << int( complete_duration ) << "s, remaining: " << int( complete_duration * ( t_max - p.t ) / p.t ) << "s    \n";
    std::cout << "    Time per ps: " << complete_duration / p.t << "s/ps  -  " << std::setprecision( 3 ) << p.t / complete_duration << "ps/s  -  " << complete_iterations / complete_duration << "it/s    \n";
    std::cout << PHOENIX::CLIO::fillLine( console_width, seperator ) << std::endl;
    std::cout << EscapeSequence::LINE_UP << EscapeSequence::LINE_UP << EscapeSequence::LINE_UP << EscapeSequence::LINE_UP << EscapeSequence::LINE_UP << EscapeSequence::LINE_UP << EscapeSequence::LINE_UP;
    _PHOENIX_last_output_time = std::time( nullptr );
}

void PHOENIX::SystemParameters::finishCMD() {
    std::cout << "\n\n\n\n\n\n\n" << EscapeSequence::SHOW_CURSOR;
}
