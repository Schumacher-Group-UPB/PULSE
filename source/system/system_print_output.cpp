#include "system/system.hpp"
#include "misc/commandline_input.hpp"
#include "misc/escape_sequences.hpp"
#include "misc/timeit.hpp"
#include "omp.h"

void PC3::System::printHelp() {
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
        << unifyLength( "--path", "[string]", "Workingfolder. Standard is '" + filehandler.outputPath + "'\n" )
        << unifyLength( "--name", "[string]", "File prefix. Standard is '" + filehandler.outputName + "'\n" )
        << unifyLength( "--loadFrom", "[string] [files,...]", "Loads list of matrices from path.\n" )
        << unifyLength( "--outEvery", "[int]", "Number of Runge-Kutta iterations for each plot. Standard is every " + std::to_string( filehandler.output_every ) + " iteration\n" )
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
              << unifyLength( "--pump", "[double] [string] [double] [double] [double] [string] [double] [string]", "amplitude, behaviour (add,multiply,replace,adaptive,complex), width, posX, posY, pol (plus,minus,both), exponent, type (gauss, ring)\n", 30, 80)
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
    std::cout << EscapeSequence::BOLD << "===================================================================================" << EscapeSequence::RESET << std::endl;
    std::cout << EscapeSequence::BOLD << "============================== PC^3 Runtime Statistics ============================" << EscapeSequence::RESET << std::endl;
    std::cout << EscapeSequence::BOLD << "===================================================================================" << EscapeSequence::RESET << std::endl;
    std::cout << EscapeSequence::BOLD << "--------------------------------- System Parameters -------------------------------" << EscapeSequence::RESET << std::endl;
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
    std::cout << EscapeSequence::BOLD << "--------------------------------- Runtime Statistics ------------------------------" << EscapeSequence::RESET << std::endl;
    double total = PC3::TimeIt::totalRuntime();
    for ( const auto& [key, value] : timeit_times_total ) {
        std::cout << unifyLength( key + ":", std::to_string( value ) + "s", std::to_string( value / t_max * 1E3 ) + "ms/ps", l,l ) << std::endl;
    }
    std::cout << unifyLength( "Total Runtime:", std::to_string( total ) + "s", std::to_string( total / t_max * 1E3 ) + "ms/ps", l,l ) << " --> " << std::to_string( total / iteration ) << "s/it" << std::endl;
    std::cout << EscapeSequence::BOLD << "---------------------------------------- Infos ------------------------------------" << EscapeSequence::RESET << std::endl;
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
    std::cout << "Output variables and plots every " << filehandler.output_every << " iterations" << std::endl;
    std::cout << "Total allocated space for Device Matrices: " << CUDAMatrix<real_number>::total_mb_max + CUDAMatrix<complex_number>::total_mb_max << " MB." << std::endl;
    std::cout << "Total allocated space for Host Matrices: " << HostMatrix<real_number>::total_mb_max + HostMatrix<complex_number>::total_mb_max << " MB." << std::endl;
    std::cout << EscapeSequence::BOLD << "===================================================================================" << EscapeSequence::RESET << std::endl;
}