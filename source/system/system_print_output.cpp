#include "system/system.hpp"
#include "misc/commandline_input.hpp"
#include "misc/escape_sequences.hpp"
#include "misc/timeit.hpp"
#include "omp.h"

// TODO: Create a few wrapper functions for formatting, make help screen and final screen fixed width

void print_name() {
    std::cout << "-----------------------------------------------------------------------------------\n\n";
    std::cout << EscapeSequence::BLUE << EscapeSequence::BOLD;
    std::cout << "                  _____    _     _            _______   _______\n";
    std::cout << "                 |_____]   |     |   |        |______   |______\n";
    std::cout << "                 |       . |_____| . |_____ . ______| . |______ .\n\n";
    std::cout << "        " << EscapeSequence::RESET << EscapeSequence::UNDERLINE << EscapeSequence::BOLD;
    std::cout << EscapeSequence::BLUE << "P" << EscapeSequence::GRAY << "aderborn " << EscapeSequence::BLUE;
    std::cout << "U" << EscapeSequence::GRAY << "ltrafast So" << EscapeSequence::BLUE << "L" << EscapeSequence::GRAY;
    std::cout << "ver for the nonlinear " << EscapeSequence::BLUE << "S" << EscapeSequence::GRAY << "chroedinger ";
    std::cout << EscapeSequence::BLUE << "E" << EscapeSequence::GRAY << "quation";
    std::cout << EscapeSequence::RESET << "                         \n" << std::endl;
    std::cout << "                                Version: " << EscapeSequence::BOLD << EscapeSequence::BLUE << "0.1.0" << EscapeSequence::RESET << std::endl;
    std::cout << "                      https://github.com/davidbauch/PC3" << std::endl;
    std::cout << "-----------------------------------------------------------------------------------" << std::endl;
}

template <typename T> 
std::string to_str(T t) {
    auto numeric_precision = std::numeric_limits<T>::max_digits10;
    std::stringstream ss;
    if (t < std::pow(10, -numeric_precision) or t > std::pow(10, numeric_precision) and t != 0.0)
        ss << std::scientific;
    else
        ss << std::fixed;
    ss << t;
    return ss.str();
}

void PC3::System::printHelp() {
    print_name();
#ifdef USEFP64
    std::cout << "This program is compiled with " << EscapeSequence::UNDERLINE << EscapeSequence::YELLOW << "double precision" << EscapeSequence::RESET << " numbers.\n";
#else
    std::cout << "This program is compiled with " << EscapeSequence::UNDERLINE << EscapeSequence::YELLOW << "single precision" << EscapeSequence::RESET << " numbers.\n";
#endif
#ifdef USECPU
    std::cout << "This program is compiled as a " << EscapeSequence::UNDERLINE << EscapeSequence::YELLOW << "CPU Version" << EscapeSequence::RESET << ".\n";
    std::cout << "Maximum number of CPU cores utilized " << omp_get_max_threads() << std::endl;
#endif
    std::cout
        << unifyLength( "General parameters:", "", "\n" )
        << unifyLength( "Flag", "Inputs", "Description\n" )
        << unifyLength( "--path", "<string>", "Workingfolder. Standard is '" + filehandler.outputPath + "'\n" )
        << unifyLength( "--name", "<string>", "File prefix. Standard is '" + filehandler.outputName + "'\n" )
        << unifyLength( "--loadFrom", "<string> <string...>", "Loads list of matrices from path.\n" )
        << unifyLength( "--outEvery", "<int>", "Number of Runge-Kutta iterations for each plot. Standard is every " + std::to_string( output_every ) + " ps\n" )
        << unifyLength( "--output", "<string...>", "Comma seperated list of things to output. Available: mat,scalar,fft,pump,mask,psi,n. Many can also be specified with _plus or _minus.\n" )
        << unifyLength( "--history", "<Y> <points>", "Outputs a maximum number of x-slices at Y for history. y-slices are not supported.\n" )
        << unifyLength( "--historyMatrix", "<int> <int> <int> <int> <int>", "Outputs the matrices specified in --output with specified startx,endx,starty,endy index and increment.\n" )
        << unifyLength( "--input", "<string...>", "Comma seperated list of things to input. Available: mat,scalar,fft,pump,mask,psi,n. Many can also be specified with _plus or _minus.\n" )
        << unifyLength( "-nosfml", "no arguments", "If passed to the program, disables all live graphical output. \n" );
    std::cout << "-----------------------------------------------------------------------------------\n";
    std::cout << unifyLength( "Numerical parameters", "", "\n" ) << unifyLength( "Flag", "Inputs", "Description\n" )
              << unifyLength( "--N", "<int> <int>", "Grid Dimensions (N x N). Standard is " + std::to_string( p.N_x ) + " x " + std::to_string( p.N_y ) + "\n" )
              << unifyLength( "--tstep", "<double>", "Timestep, standard is magic-timestep = "+to_str(magic_timestep)+"ps\n" )
              << unifyLength( "-rk45", "no arguments", "Use RK45\n" )
              << unifyLength( "--tol", "<double>", "RK45 Tolerance, standard is " + to_str( tolerance ) + " ps\n" )
              << unifyLength( "--tmax", "<double>", "Timelimit, standard is " + to_str( t_max ) + " ps\n" )
              << unifyLength( "--boundary", "<string> <string>", "Boundary conditions for x and y. Is either 'periodic' or 'zero'.\n" );
    std::cout << "-----------------------------------------------------------------------------------\n";
    std::cout << unifyLength( "System Parameters", "", "\n" )
              << unifyLength( "Flag", "Inputs", "Description\n" )
              << unifyLength( "--gammaC", "<double>", "Standard is " + to_str( p.gamma_c ) + " ps^-1\n" )
              << unifyLength( "--gammaR", "<double>", "Standard is " + to_str( p.gamma_r / p.gamma_c ) + "*gammaC\n" )
              << unifyLength( "--gc", "<double>", "Standard is " + to_str( p.g_c ) + " eV mum^2\n" )
              << unifyLength( "--gr", "<double>", "Standard is " + to_str( p.g_r / p.g_c ) + "*gc\n" )
              << unifyLength( "--meff", "<double>", "Standard is " + to_str( p.m_eff ) + "\n" )
              << unifyLength( "--R", "<double>", "Standard is " + to_str( p.R ) + " ps^-1 mum^2\n" )
              << unifyLength( "--g_pm", "<double>", "Standard is " + to_str( p.g_pm / p.g_c ) + "*gc. Only effective in a system with TE/TM splitting.\n" )
              << unifyLength( "--deltaLT", "<double>", "Standard is " + to_str( p.delta_LT ) + " eV. Only effective in a system with TE/TM splitting.\n" )
              << unifyLength( "--L", "<double> <double>", "Standard is " + to_str( p.L_x ) + ", " + to_str( p.L_y ) + " mum\n" ) << std::endl;
    std::cout << "-----------------------------------------------------------------------------------\n";
    std::cout << unifyLength( "Envelopes.", "", "\n" )
              << "Envelopes are passed using either their spatial and temporal characteristics, or by loading an external file. Syntax:\n"
              << unifyLength( "--envelope", "<double> <string> <double> <double> <double> <double> <string> <double> <double> <string> [osc <double> <double> <double>]", "amplitude, behaviour (add,multiply,replace,adaptive,complex), widthX, widthY, posX, posY, pol (plus,minus,both), exponent, charge, type (gauss, ring) [t0, frequency, sigma]\n", 30, 80 )
              << unifyLength( "--envelope", "<string> <double> <string> <string> [osc <double> <double> <double>]", "path, amplitude, behaviour (add,multiply,replace,adaptive,complex), pol (plus,minus,both) [t0, frequency, sigma]\n", 30, 80 )
              << "Possible Envelopes include:"
              << unifyLength( "--pump", "Spatial and Temporal ~cos(wt)", "\n", 30, 80 )
              << unifyLength( "--potential", "Spatial and Temporal ~cos(wt)", "\n", 30, 80 )
              << unifyLength( "--initialState", "Spatial", "\n", 30, 80 )
              << unifyLength( "--pulse", "Spatial and Temporal ~exp(iwt)", "\n", 30, 80 )
              << unifyLength( "--fftMask", "Spatial", "\n", 30, 80 )
              << "Additional Parameters:\n"
              << unifyLength( "--fftEvery", "<int>", "Apply FFT Filter every x ps\n", 30, 80 )
              << unifyLength( "--initRandom", "<double>", "Amplitude. Randomly initialize Psi\n", 30, 80 );
    std::cout << "-----------------------------------------------------------------------------------\n";
    std::cout << unifyLength("SI Scalings", "", "\n")
              << unifyLength( "Flag", "Inputs", "Description\n" )
              << unifyLength("--hbar", "<double>", "Standard is " + to_str(p.h_bar) + "\n")
              << unifyLength("--e", "<double>", "Standard is " + to_str(p.e_e) + "\n")
              << unifyLength("--me", "<double>", "Standard is " + to_str(p.m_e) + "\n")
              << unifyLength("--hbarscaled", "<double>", "Standard is " + to_str(p.h_bar_s) + "\n")
              << unifyLength("--meff", "<double>", "Standard is " + to_str(p.m_eff) + "\n") << std::endl;
#ifdef USECPU
    std::cout << unifyLength( "--threads", "<int>", "Standard is " + std::to_string( omp_max_threads ) + " Threads\n" ) << std::endl;
#endif
}

void PC3::System::printSummary( std::map<std::string, std::vector<double>> timeit_times, std::map<std::string, double> timeit_times_total ) {
    print_name();
    const int l = 15;
    std::cout << EscapeSequence::BOLD << "-----------------------------------------------------------------------------------" << EscapeSequence::RESET << std::endl;
    std::cout << EscapeSequence::BOLD << "------------------------------------ Parameters -----------------------------------" << EscapeSequence::RESET << std::endl;
    std::cout << unifyLength( "N", std::to_string( p.N_x ) + ", " + std::to_string( p.N_y ), "", l, l ) << std::endl;
    std::cout << unifyLength( "N^2", std::to_string( p.N_x * p.N_y ), "", l, l ) << std::endl;
    std::cout << unifyLength( "Lx", to_str( p.L_x ), "mum", l, l ) << std::endl;
    std::cout << unifyLength( "Ly", to_str( p.L_y ), "mum", l, l ) << std::endl;
    std::cout << unifyLength( "dx", to_str( p.dx ), "mum", l, l ) << std::endl;
    std::cout << unifyLength( "dy", to_str( p.dx ), "mum", l, l ) << std::endl;
    std::cout << unifyLength( "tmax", to_str( t_max ), "ps", l, l ) << std::endl;
    std::cout << unifyLength( "dt", to_str( p.dt ), "ps", l, l ) << std::endl;
    std::cout << unifyLength( "gamma_c", to_str( p.gamma_c ), "ps^-1", l, l ) << std::endl;
    std::cout << unifyLength( "gamma_r", to_str( p.gamma_r ), "ps^-1", l, l ) << std::endl;
    std::cout << unifyLength( "g_c", to_str( p.g_c ), "eV mum^2", l, l ) << std::endl;
    std::cout << unifyLength( "g_r", to_str( p.g_r ), "eV mum^2", l, l ) << std::endl;
    std::cout << unifyLength( "g_pm", to_str( p.g_pm ), "eV mum^2", l, l ) << std::endl;
    std::cout << unifyLength( "R", to_str( p.R ), "ps^-1 mum^-2", l, l ) << std::endl;
    std::cout << unifyLength( "delta_LT", to_str( p.delta_LT ), "eV", l, l ) << std::endl;
    std::cout << unifyLength( "m_eff", to_str( p.m_eff ), "", l, l ) << std::endl;
    std::cout << unifyLength( "h_bar_s", to_str( p.h_bar_s ), "", l, l ) << std::endl;
    std::cout << "Boundary Condition: " << (p.periodic_boundary_x ? "Periodic" : "Zero" ) << "(x):" << (p.periodic_boundary_y ? "Periodic" : "Zero" ) << "(y)" << std::endl;
    std::cout << "--------------------------------- Envelope Functions ------------------------------" << std::endl;
    // TODO: overwrite << operator of the Envelope Class
    std::cout << "Pulse Envelopes:\n" << pulse.toString();
    std::cout << "Pump Envelopes:\n" << pump.toString();
    std::cout << "Potential Envelopes:\n" << potential.toString();
    std::cout << "FFT Mask Envelopes:\n" << fft_mask.toString();
    std::cout << "Initial State Envelopes:\n" << initial_state.toString();
    std::cout << EscapeSequence::BOLD << "--------------------------------- Runtime Statistics ------------------------------" << EscapeSequence::RESET << std::endl;
    double total = PC3::TimeIt::totalRuntime();
    std::cout << unifyLength( "Total Runtime:", std::to_string( total ) + "s", std::to_string( total / p.t * 1E3 ) + "ms/ps", l, l ) << " --> " << std::to_string( p.t/total ) << "ps/s" << " --> " << std::to_string( total / iteration ) << "s/it" << std::endl;
    std::cout << EscapeSequence::BOLD << "---------------------------------------- Infos ------------------------------------" << EscapeSequence::RESET << std::endl;
    if ( fixed_time_step )
        std::cout << "Calculations done using the fixed timestep RK4 solver" << std::endl;
    else {
        std::cout << "Calculations done using the variable timestep RK45 solver" << std::endl;
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
#ifdef USEFP32
    std::cout << "This program is compiled using " << EscapeSequence::UNDERLINE << EscapeSequence::BLUE << "single precision" << EscapeSequence::RESET << " numbers.\n";
#else
    std::cout << "This program is compiled using " << EscapeSequence::UNDERLINE << EscapeSequence::BLUE << "double precision" << EscapeSequence::RESET << " numbers.\n";
#endif
#ifdef USECPU
    std::cout << "Device Used: " << EscapeSequence::BOLD << EscapeSequence::YELLOW << "CPU" << EscapeSequence::RESET << std::endl;
    std::cout << EscapeSequence::GRAY << "  Cores utilized " << omp_max_threads << " of " << omp_get_max_threads() << " total cores." << EscapeSequence::RESET << std::endl;
#else
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
std::cout << "Device Used: " << EscapeSequence::GREEN << EscapeSequence::BOLD << prop.name << EscapeSequence::RESET << std::endl;
    std::cout << EscapeSequence::GRAY << "  Memory Clock Rate (MHz): " << prop.memoryClockRate/1024 << std::endl;
    std::cout << "  Peak Memory Bandwidth (GB/s): " << 2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6 << std::endl;
    std::cout << "  Total global memory (Gbytes): " <<(float)(prop.totalGlobalMem)/1024.0/1024.0/1024.0 << std::endl;
    std::cout << "  Warp-size: " << prop.warpSize << EscapeSequence::RESET << std::endl;
#endif
    std::cout << EscapeSequence::BOLD << "===================================================================================" << EscapeSequence::RESET << std::endl;
}