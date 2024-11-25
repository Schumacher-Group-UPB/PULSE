#include <ctime>
#include <iomanip> // std::setprecision, std::setw, std::setfill
#include <algorithm>
#include "system/system_parameters.hpp"
#include "cuda/cuda_matrix_base.hpp"
#include "misc/commandline_io.hpp"
#include "misc/escape_sequences.hpp"
#include "misc/timeit.hpp"
#include "omp.h"

// Automatically determine console width depending on windows or linux
#ifdef _WIN32
    #include <windows.h>
static size_t getConsoleWidth() {
    CONSOLE_SCREEN_BUFFER_INFO csbi;
    GetConsoleScreenBufferInfo( GetStdHandle( STD_OUTPUT_HANDLE ), &csbi );
    return csbi.srWindow.Right - csbi.srWindow.Left + 1;
}
#else
    #include <sys/ioctl.h>
static size_t getConsoleWidth() {
    struct winsize w;
    ioctl( 0, TIOCGWINSZ, &w );
    return w.ws_col;
}
#endif
static size_t console_width = std::max<size_t>( getConsoleWidth(), 100 );

// File-Local Configuration
static char major_seperator = '=';
static char minor_seperator = '-';
static char seperator = '.';

/*
Prints PHOENIX. Font is CyberLarge
 _____    _     _    _____    _______   __   _   _____   _     _  
|_____]   |_____|   |     |   |______   | \  |     |      \___/   
|       . |     | . |_____| . |______ . |  \_| . __|__ . _/   \_ .

*/
void print_name() {
    std::cout << PHOENIX::CLIO::fillLine( console_width, major_seperator ) << "\n\n"; // Horizontal Separator
    std::cout << EscapeSequence::ORANGE << EscapeSequence::BOLD;                      // Make Text Bold

    // Print Phoenix LOGO
    std::cout << PHOENIX::CLIO::centerString( " _____    _     _    _____    _______   __   _   _____   _     _  ", console_width ) << "\n";
    std::cout << PHOENIX::CLIO::centerString( "|_____]   |_____|   |     |   |______   | \\  |     |      \\___/   ", console_width ) << "\n";
    std::cout << PHOENIX::CLIO::centerString( "|       . |     | . |_____| . |______ . |  \\_| . __|__ . _/   \\_ .", console_width ) << "\n\n";

    std::stringstream ss;

    // Program Description
    ss << EscapeSequence::RESET << EscapeSequence::UNDERLINE << EscapeSequence::BOLD << EscapeSequence::BLUE << "P" << EscapeSequence::GRAY << "aderborn " << EscapeSequence::BLUE << "H" << EscapeSequence::GRAY << "ighly " << EscapeSequence::BLUE << "O" << EscapeSequence::GRAY << "ptimized and " << EscapeSequence::BLUE << "E" << EscapeSequence::GRAY << "nergy efficient solver for two-dimensional" << EscapeSequence::RESET;
    std::cout << PHOENIX::CLIO::centerStringRaw( ss.str(), console_width, "Paderborn Highly Optimized and Energy efficient solver for two-dimensional" ) << std::endl;
    ss.str( "" );
    ss.clear();

    ss << EscapeSequence::RESET << EscapeSequence::UNDERLINE << EscapeSequence::BLUE << "N" << EscapeSequence::GRAY << "onlinear Schroedinger equations with " << EscapeSequence::BLUE << "I" << EscapeSequence::GRAY << "ntegrated e" << EscapeSequence::BLUE << "X" << EscapeSequence::GRAY << "tensions" << EscapeSequence::RESET;
    std::cout << PHOENIX::CLIO::centerStringRaw( ss.str(), console_width, "Nonlinear Schroedinger equations with Integrated Xtensions" ) << std::endl;
    ss.str( "" );
    ss.clear();

    // Version Information
    ss << "Version: " << EscapeSequence::BOLD << EscapeSequence::BLUE << "0.1.0" << EscapeSequence::RESET;
    std::cout << PHOENIX::CLIO::centerStringRaw( ss.str(), console_width, "Version: 0.1.0" ) << std::endl;
    //std::cout << PHOENIX::CLIO::centerString( "https://github.com/Schumacher-Group-UPB/PHOENIX", console_width ) << std::endl;

    // Citation Information
    std::cout << "\n" << PHOENIX::CLIO::fillLine( console_width, minor_seperator ) << "\n"; // Horizontal Separator
    ss.str( "" );
    ss.clear();

    // Citation Message
    ss << "If you use this program, please cite the repository using: " << EscapeSequence::RESET;
    std::cout << EscapeSequence::BOLD << EscapeSequence::GRAY << PHOENIX::CLIO::centerStringRaw( ss.str(), console_width ) << std::endl;
    ss.str( "" );
    ss.clear();

    ss << "Bauch, D., Schade, R., Wingenbach, J., and Schumacher, S.";
    std::cout << EscapeSequence::ORANGE << PHOENIX::CLIO::centerStringRaw( ss.str(), console_width ) << EscapeSequence::RESET << std::endl;
    ss.str( "" );
    ss.clear();
    ss << "PHOENIX: A High-Performance Solver for the Gross-Pitaevskii Equation [Computer software].";
    std::cout << EscapeSequence::ORANGE << PHOENIX::CLIO::centerStringRaw( ss.str(), console_width ) << EscapeSequence::RESET << std::endl;
    ss.str( "" );
    ss.clear();
    ss << "https://github.com/Schumacher-Group-UPB/PHOENIX";
    std::cout << EscapeSequence::ORANGE << PHOENIX::CLIO::centerStringRaw( ss.str(), console_width ) << EscapeSequence::RESET << std::endl;
}

void PHOENIX::SystemParameters::printHelp() {
    print_name();

    std::cout << PHOENIX::CLIO::fillLine( console_width, major_seperator ) << "\n"; // Horizontal Separator

#ifndef USE_32_BIT_PRECISION
    std::cout << "This program is compiled with " << EscapeSequence::UNDERLINE << EscapeSequence::YELLOW << "double precision" << EscapeSequence::RESET << " numbers.\n";
#else
    std::cout << "This program is compiled with " << EscapeSequence::UNDERLINE << EscapeSequence::YELLOW << "single precision" << EscapeSequence::RESET << " numbers.\n";
#endif

#ifdef USE_CPU
    std::cout << "This program is compiled as a " << EscapeSequence::UNDERLINE << EscapeSequence::YELLOW << "CPU Version" << EscapeSequence::RESET << ".\n";
    std::cout << "Maximum number of CPU cores utilized: " << omp_get_max_threads() << std::endl;
#endif

    std::cout << EscapeSequence::BOLD << PHOENIX::CLIO::fillLine( console_width, major_seperator ) << EscapeSequence::RESET << std::endl;
    std::cout << PHOENIX::CLIO::unifyLength( "Option", "Inputs", "Description\n" ) << std::endl;
    std::cout << EscapeSequence::BOLD << PHOENIX::CLIO::fillLine( console_width, minor_seperator ) << EscapeSequence::RESET << std::endl;

    // Program Options
    std::cout << PHOENIX::CLIO::unifyLength( "--path", "<string>", "Working folder. Default is '" + filehandler.outputPath + "'" ) << std::endl;
    std::cout << PHOENIX::CLIO::unifyLength( "--name", "<string>", "File prefix. Default is '" + filehandler.outputName + "'" ) << std::endl;
    std::cout << PHOENIX::CLIO::unifyLength( "--config", "<string>", "Loads configuration from file. Multiple configurations can be superimposed." ) << std::endl;
    std::cout << PHOENIX::CLIO::unifyLength( "...", "...", "Example: --config path/to/config.txt loads commandline arguments from config.txt" ) << std::endl;
    std::cout << PHOENIX::CLIO::unifyLength( "--outEvery", "<float>", "Output every x ps. Default is every " + std::to_string( output_every ) + " ps" ) << std::endl;
    std::cout << PHOENIX::CLIO::unifyLength( "...", "...", "Example: --outEvery 10.0 outputs every 10ps." ) << std::endl;
    std::cout << PHOENIX::CLIO::fillLine( console_width, minor_seperator ) << std::endl;
    std::cout << PHOENIX::CLIO::unifyLength( "--output", "<string...>", "Comma-separated list of things to output. Available: mat, scalar, fft, pump, mask, psi, n. Options with _plus or _minus are also supported." ) << std::endl;
    std::cout << PHOENIX::CLIO::unifyLength( "...", "...", "Examples: --output all, --output wavefunction, --output fft,scalar." ) << std::endl;
    std::cout << PHOENIX::CLIO::unifyLength( "--historyMatrix", "<int> <int> <int> <int> <int>", "Outputs matrices specified in --output with startx, endx, starty, endy index, and increment." ) << std::endl;
    std::cout << PHOENIX::CLIO::unifyLength( "...", "...", "Example: --historyMatrix 0 100 0 100 1 outputs matrices from 0 to 100 in x and y." ) << std::endl;
    std::cout << PHOENIX::CLIO::unifyLength( "...", "...", "Example: --historyMatrix 0 100 0 100 5 outputs matrices from 0 to 100 in x and y with a step of 5." ) << std::endl;
    std::cout << PHOENIX::CLIO::unifyLength( "--historyTime", "<float> <int>", "Outputs matrices specified in --output after starting time, then every multiple*outEvery times." ) << std::endl;
    std::cout << PHOENIX::CLIO::unifyLength( "...", "...", "Example: --historyTime 1000.0 2 outputs matrices after t = 1000ps every 2*outEvery ps." ) << std::endl;
    std::cout << PHOENIX::CLIO::fillLine( console_width, minor_seperator ) << std::endl;
    std::cout << PHOENIX::CLIO::unifyLength( "-norender", "no arguments", "Disables all live graphical output if passed." ) << std::endl;

    std::cout << PHOENIX::CLIO::fillLine( console_width, minor_seperator ) << std::endl;
    std::cout << PHOENIX::CLIO::unifyLength( "Numerical parameters", "", "" ) << std::endl;
    std::cout << PHOENIX::CLIO::unifyLength( "--N", "<int> <int>", "Grid Dimensions (N x N). Default is " + std::to_string( p.N_c ) + " x " + std::to_string( p.N_r ) ) << std::endl;
    std::cout << PHOENIX::CLIO::unifyLength( "...", "...", "Example: --N 100 100 sets the grid to 100x100. --N 500 1000 sets the grid to 500x1000." ) << std::endl;
    std::cout << PHOENIX::CLIO::unifyLength( "--subgrids", "<int> <int>", "Subgrid Dimensions (N x N). Need to integer devide Nx,Ny. Default is " + std::to_string( p.subgrids_columns ) + " x " + std::to_string( p.subgrids_rows ) ) << std::endl;
    std::cout << PHOENIX::CLIO::unifyLength( "...", "...", "Example: --subgrids 2 2 results in 2*2 = 4 subgrids. --subgrids 1 5 results in 1*5 = 5 subgrids." ) << std::endl;
    std::cout << PHOENIX::CLIO::unifyLength( "--tstep", "<double>", "Timestep. Default is " + PHOENIX::CLIO::to_str( magic_timestep ) + " ps. It's advised to leave this parameter at its default value." ) << std::endl;
    std::cout << PHOENIX::CLIO::unifyLength( "...", "...", "Example: --tstep 0.1 sets the timestep to 0.1ps." ) << std::endl;
    std::cout << PHOENIX::CLIO::unifyLength( "--tmax", "<double>", "Timelimit. Default is " + PHOENIX::CLIO::to_str( t_max ) + " ps" ) << std::endl;
    std::cout << PHOENIX::CLIO::unifyLength( "...", "...", "Example: --tmax 1000 sets the simulation time to 1000ps." ) << std::endl;
    std::cout << PHOENIX::CLIO::unifyLength( "--iterator", "<string>", "RK4 or SSFM" ) << std::endl;
    std::cout << PHOENIX::CLIO::unifyLength( "...", "...", "Example: --iterator rk4 sets the iterator to RK4. --iterator ssfm sets the iterator to SSFM." ) << std::endl;
    //std::cout << PHOENIX::CLIO::unifyLength( "-rk45", "no arguments", "Shortcut to use RK45" ) << std::endl;
    //std::cout << PHOENIX::CLIO::unifyLength( "--rk45dt", "<double> <double>", "dt_min and dt_max for RK45 method" ) << std::endl;
    //std::cout << PHOENIX::CLIO::unifyLength( "--tol", "<double>", "RK45 Tolerance. Default is " + PHOENIX::CLIO::to_str( tolerance ) + " ps" ) << std::endl;
    //std::cout << PHOENIX::CLIO::unifyLength( "-ssfm", "no arguments", "Shortcut to use SSFM" ) << std::endl;
    std::cout << PHOENIX::CLIO::unifyLength( "--imagTime", "<double>", "Use imaginary time propagation with normalization constant. Default is " + PHOENIX::CLIO::to_str( imag_time_amplitude ) ) << std::endl;
    std::cout << PHOENIX::CLIO::unifyLength( "...", "...", "Example: --imagTime 1 sets the imaginary time amplitude to 1, --imagTime 10 sets the normalization constant to 10." ) << std::endl;
    std::cout << PHOENIX::CLIO::unifyLength( "--boundary", "<string> <string>", "Boundary conditions for x and y: 'periodic' or 'zero'" ) << std::endl;
    std::cout << PHOENIX::CLIO::unifyLength( "...", "...", "Example: --boundary periodic zero sets periodic boundary conditions in x and zero boundary conditions in y." ) << std::endl;

    std::cout << PHOENIX::CLIO::fillLine( console_width, minor_seperator ) << std::endl;
    std::cout << PHOENIX::CLIO::unifyLength( "System Parameters", "", "" ) << std::endl;
    std::cout << PHOENIX::CLIO::unifyLength( "--gammaC", "<double>", "Default is " + PHOENIX::CLIO::to_str( p.gamma_c ) + " ps^-1" ) << std::endl;
    std::cout << PHOENIX::CLIO::unifyLength( "--gammaR", "<double>", "Default is " + PHOENIX::CLIO::to_str( p.gamma_r / p.gamma_c ) + "*gammaC" ) << std::endl;
    std::cout << PHOENIX::CLIO::unifyLength( "--gc", "<double>", "Default is " + PHOENIX::CLIO::to_str( p.g_c ) + " eV mum^2" ) << std::endl;
    std::cout << PHOENIX::CLIO::unifyLength( "--gr", "<double>", "Default is " + PHOENIX::CLIO::to_str( p.g_r / p.g_c ) + "*gc" ) << std::endl;
    std::cout << PHOENIX::CLIO::unifyLength( "--meff", "<double>", "Default is " + PHOENIX::CLIO::to_str( p.m_eff ) ) << std::endl;
    std::cout << PHOENIX::CLIO::unifyLength( "--R", "<double>", "Default is " + PHOENIX::CLIO::to_str( p.R ) + " ps^-1 mum^2" ) << std::endl;
    std::cout << PHOENIX::CLIO::unifyLength( "--g_pm", "<double>", "Default is " + PHOENIX::CLIO::to_str( p.g_pm / p.g_c ) + "*gc. Effective in a system with TE/TM splitting." ) << std::endl;
    std::cout << PHOENIX::CLIO::unifyLength( "--deltaLT", "<double>", "Default is " + PHOENIX::CLIO::to_str( p.delta_LT ) + " eV. Effective in a system with TE/TM splitting." ) << std::endl;
    std::cout << PHOENIX::CLIO::unifyLength( "--L", "<double> <double>", "Default is " + PHOENIX::CLIO::to_str( p.L_x ) + ", " + PHOENIX::CLIO::to_str( p.L_y ) + " mum" ) << std::endl;

    std::cout << PHOENIX::CLIO::fillLine( console_width, minor_seperator ) << std::endl;
    std::cout << PHOENIX::CLIO::unifyLength( "Envelopes", "", "" ) << std::endl;
    std::cout << PHOENIX::CLIO::unifyLength( "Syntax for spatial and temporal envelopes or loading external files:", "", "" ) << std::endl;

    // Envelope Syntax
    std::cout << PHOENIX::CLIO::unifyLength( "--envelope", "<double> <string> <double> <double> <double> <double> <string> <double> <double> <string> time <double> <double> <double>",
                                             "amplitude, behaviour (add, multiply, replace, adaptive, complex), widthX, widthY, posX, posY, pol (plus, minus, both), exponent, charge, type (gauss, ring), "
                                             "time: kind (gauss, cos, iexp), t0, frequency, sigma" )
              << std::endl;

    std::cout << PHOENIX::CLIO::unifyLength( "--envelope", "load <string> <double> <string> <string> time <string> <double> <double> <double>", "path, amplitude, behaviour, pol (plus, minus, both). For time: path" ) << std::endl;

    std::cout << "Possible Envelopes include:" << std::endl;
    std::cout << PHOENIX::CLIO::unifyLength( "--pump", "Spatial and Temporal ~cos(wt)", "" ) << std::endl;
    std::cout << PHOENIX::CLIO::unifyLength( "--potential", "Spatial and Temporal ~cos(wt)", "" ) << std::endl;
    std::cout << PHOENIX::CLIO::unifyLength( "--initialState", "Spatial", "" ) << std::endl;
    std::cout << PHOENIX::CLIO::unifyLength( "--initialReservoir", "Spatial", "" ) << std::endl;
    std::cout << PHOENIX::CLIO::unifyLength( "--pulse", "Spatial and Temporal ~exp(iwt)", "" ) << std::endl;
    std::cout << PHOENIX::CLIO::unifyLength( "--fftMask", "Spatial", "" ) << std::endl;

    // Additional Parameters
    std::cout << "Additional Parameters:" << std::endl;
    std::cout << PHOENIX::CLIO::unifyLength( "--fftEvery", "<int>", "Apply FFT Filter every x ps" ) << std::endl;
    std::cout << PHOENIX::CLIO::unifyLength( "--initRandom", "<double>", "Amplitude. Randomly initialize Psi" ) << std::endl;

    std::cout << PHOENIX::CLIO::fillLine( console_width, major_seperator ) << std::endl;
    std::cout << PHOENIX::CLIO::unifyLength( "SI Scalings", "", "" ) << std::endl;
    std::cout << PHOENIX::CLIO::unifyLength( "Flag", "Inputs", "Description" ) << std::endl;
    std::cout << PHOENIX::CLIO::unifyLength( "--hbar", "<double>", "Default is " + PHOENIX::CLIO::to_str( p.h_bar ) ) << std::endl;
    std::cout << PHOENIX::CLIO::unifyLength( "--e", "<double>", "Default is " + PHOENIX::CLIO::to_str( p.e_e ) ) << std::endl;
    std::cout << PHOENIX::CLIO::unifyLength( "--me", "<double>", "Default is " + PHOENIX::CLIO::to_str( p.m_e ) ) << std::endl;
    std::cout << PHOENIX::CLIO::unifyLength( "--hbarscaled", "<double>", "Default is " + PHOENIX::CLIO::to_str( p.h_bar_s ) ) << std::endl;
    std::cout << PHOENIX::CLIO::unifyLength( "--meff", "<double>", "Default is " + PHOENIX::CLIO::to_str( p.m_eff ) ) << std::endl;

#ifdef USE_CPU
    std::cout << PHOENIX::CLIO::unifyLength( "--threads", "<int>", "Default is " + std::to_string( omp_max_threads ) + " Threads" ) << std::endl;
#endif

    std::cout << PHOENIX::CLIO::fillLine( console_width, major_seperator ) << std::endl;
}

void PHOENIX::SystemParameters::printSummary( std::map<std::string, std::vector<double>> timeit_times, std::map<std::string, double> timeit_times_total ) {
    print_name();
    const int l = 35;

    // Print Header
    std::cout << EscapeSequence::BOLD << PHOENIX::CLIO::fillLine( console_width, major_seperator ) << EscapeSequence::RESET << std::endl;
    std::cout << EscapeSequence::BOLD << PHOENIX::CLIO::centerString( " Parameters ", console_width, '-' ) << EscapeSequence::RESET << std::endl;

    // Grid Configuration
    std::cout << PHOENIX::CLIO::unifyLength( "Grid Configuration", "---", "---", l, l, l, " " ) << std::endl;
    std::cout << PHOENIX::CLIO::unifyLength( "N", std::to_string( p.N_c ) + ", " + std::to_string( p.N_r ), "", l, l, l, " " ) << std::endl;
    std::cout << PHOENIX::CLIO::unifyLength( "N^2", std::to_string( p.N_c * p.N_r ), "", l, l, l, " " ) << std::endl;
    std::cout << PHOENIX::CLIO::unifyLength( "Subgrids", std::to_string( p.subgrids_columns ) + ", " + std::to_string( p.subgrids_rows ), "", l, l, l, " " ) << std::endl;
    std::cout << PHOENIX::CLIO::unifyLength( "Total Subgrids", std::to_string( p.subgrids_columns * p.subgrids_rows ), "", l, l, l, " " ) << std::endl;

    // Subgrid Overhead
    const double subgrid_overhead = ( ( p.subgrid_N_r + 2.0 * p.halo_size ) * ( p.subgrid_N_c + 2 * p.halo_size ) * ( p.subgrids_columns * p.subgrids_rows ) / ( p.N_r * p.N_c ) - 1.0 ) * 100.0;
    std::cout << PHOENIX::CLIO::unifyLength( "Subgrid Overhead", std::to_string( subgrid_overhead ), "%", l, l, l, " " ) << std::endl;

    // Grid Dimensions
    std::cout << PHOENIX::CLIO::unifyLength( "Lx", PHOENIX::CLIO::to_str( p.L_x ), "mum", l, l, l, " " ) << std::endl;
    std::cout << PHOENIX::CLIO::unifyLength( "Ly", PHOENIX::CLIO::to_str( p.L_y ), "mum", l, l, l, " " ) << std::endl;
    std::cout << PHOENIX::CLIO::unifyLength( "dx", PHOENIX::CLIO::to_str( p.dx ), "mum", l, l, l, " " ) << std::endl;
    std::cout << PHOENIX::CLIO::unifyLength( "dy", PHOENIX::CLIO::to_str( p.dx ), "mum", l, l, l, " " ) << std::endl;

    // System Configuration
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

    // Boundary Conditions
    std::cout << "Boundary Condition: " << ( p.periodic_boundary_x ? "Periodic" : "Zero" ) << "(x):" << ( p.periodic_boundary_y ? "Periodic" : "Zero" ) << "(y)" << std::endl;

    // Envelopes
    std::cout << PHOENIX::CLIO::centerString( " Envelope Functions ", console_width, '-' ) << std::endl;
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

    // Runtime Statistics
    std::cout << EscapeSequence::BOLD << PHOENIX::CLIO::centerString( " Runtime Statistics ", console_width, '-' ) << EscapeSequence::RESET << std::endl;
    double total = PHOENIX::TimeIt::totalRuntime();
    std::cout << "Total Runtime: " << total << " s --> " << ( total / p.t * 1E3 ) << " ms/ps --> " << ( p.t / total ) << " ps/s --> " << ( total / iteration ) * 1e6 << " mus/it" << std::endl;

    // Additional Information
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
    std::cout << "Random Seed: " << random_seed << std::endl;

    // Precision and Device Info
#ifdef USE_32_BIT_PRECISION
    std::cout << "This program is compiled using " << EscapeSequence::UNDERLINE << EscapeSequence::BLUE << "single precision" << EscapeSequence::RESET << " numbers.\n";
#else
    std::cout << "This program is compiled using " << EscapeSequence::UNDERLINE << EscapeSequence::BLUE << "double precision" << EscapeSequence::RESET << " numbers.\n";
#endif

#ifdef USE_CPU
    std::cout << "Device Used: " << EscapeSequence::BOLD << EscapeSequence::YELLOW << "CPU" << EscapeSequence::RESET << std::endl;
    std::cout << EscapeSequence::GRAY << "  CPU cores utilized: " << omp_max_threads << EscapeSequence::RESET << std::endl;
#else
    int nDevices;
    cudaGetDeviceCount( &nDevices );
    int device;
    cudaGetDevice( &device );
    cudaDeviceProp prop;
    cudaGetDeviceProperties( &prop, device );

    std::cout << "Device Used: " << EscapeSequence::GREEN << EscapeSequence::BOLD << prop.name << EscapeSequence::RESET << std::endl;
    std::cout << "  Peak Memory Bandwidth (GB/s): " << 2.0 * prop.memoryClockRate * ( prop.memoryBusWidth / 8 ) / 1.0e6 << std::endl;
    std::cout << "  Total Global Memory (GB): " << (float)( prop.totalGlobalMem ) / 1024.0 / 1024.0 / 1024.0 << std::endl;
    std::cout << "  Total L2 Memory (MB): " << (float)( prop.l2CacheSize ) / 1024.0 / 1024.0 << std::endl;
#endif

    // Footer
    std::cout << EscapeSequence::BOLD << PHOENIX::CLIO::fillLine( console_width, '=' ) << EscapeSequence::RESET << std::endl;
}

double _PHOENIX_last_output_time = 0.;

void PHOENIX::SystemParameters::printCMD( double complete_duration, double complete_iterations ) {
    // TODO: non-cmd mode where progress is output in an easily parseable format

    // Limit output frequency to once every 0.25 seconds
    if ( std::time( nullptr ) - _PHOENIX_last_output_time < 0.25 ) {
        return;
    }

    // Hide the cursor during output
    std::cout << EscapeSequence::HIDE_CURSOR;
    std::cout << PHOENIX::CLIO::fillLine( console_width, major_seperator ) << std::endl;

    // Print current simulation time and timestep
    std::cout << "    T = " << int( p.t ) << "ps - dt = " << std::setprecision( 2 ) << p.dt << "ps\n";

    // Display progress bar for p.t/t_max
    std::cout << "    Progress: " << PHOENIX::CLIO::createProgressBar( p.t, t_max, console_width - 30 ) << "\n";

    // Determine if the system uses stochastic evaluation
    bool evaluate_stochastic = evaluateStochastic();
    std::cout << "    Current System: " << ( use_twin_mode ? "TE/TM" : "Scalar" ) << " - " << ( evaluate_stochastic ? "With Stochastic" : "No Stochastic" ) << "\n";

    // Display runtime and estimated time remaining
    std::cout << "    Runtime: " << int( complete_duration ) << "s, remaining: " << int( complete_duration * ( t_max - p.t ) / p.t ) << "s\n";

    // Display time metrics
    std::cout << "    Time per ps: " << complete_duration / p.t << "s/ps-  " << std::setprecision( 3 ) << p.t / complete_duration << "ps/s-  " << complete_iterations / complete_duration << "it/s\n";

    // Print bottom separator line
    std::cout << PHOENIX::CLIO::fillLine( console_width, major_seperator ) << std::endl;

    // Move cursor up to overwrite the previous output
    std::cout << EscapeSequence::LINE_UP << EscapeSequence::LINE_UP << EscapeSequence::LINE_UP << EscapeSequence::LINE_UP << EscapeSequence::LINE_UP << EscapeSequence::LINE_UP << EscapeSequence::LINE_UP;

    // Update the last output time
    _PHOENIX_last_output_time = std::time( nullptr );
}

void PHOENIX::SystemParameters::finishCMD() {
    std::cout << "\n\n\n\n\n\n\n" << EscapeSequence::SHOW_CURSOR;
}
