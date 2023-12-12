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
    m_eff = 1E-4 * 5.6856;
    dt_scaling_factor = m_eff;
    gamma_c = 0.15;          // ps^-1
    gamma_r = 1.5 * gamma_c; // ps^-1
    g_c = 3.E-6;             // meV mum^2
    g_r = 2. * g_c;          // meV mum^2
    R = 0.01;                // ps^-1 mum^2
    xmax = 100.;             // mum
    g_pm = -g_c / 5;         // meV mum^2
    delta_LT = 0.025E-3;     // meV

    // Numerics
    s_N = 400;
    t_max = 1000;
    iteration = 0;
    // RK Solver Variables
    t = 0;
    dt_max = 3;
    dt_min = 0.0001; // also dt_delta
    tolerance = 1E-1;
    do_overwrite_dt = true;

    // FFT Mask every x ps
    fft_every = 1; // ps

    // Kernel Block Size
    block_size = 16;
    omp_max_threads = omp_get_max_threads();

    // If this is true, the solver will use a fixed timestep RK4 method instead of the variable timestep RK45 method
    fixed_time_step = true;

    // Output of Variables
    output_keys = { "mat", "scalar" };

    normalize_before_masking = false;

    randomly_initialize_system = false;
    random_system_amplitude = 1.0;
}

void PC3::System::calculateAuto() {
    // dt_scaling_factor is again divided by m_eff, meaning smaller input m_eff result in smaller dt
    dt_scaling_factor /= m_eff;
    // Calculate dx and dt
    dx = 2.0 * xmax / ( s_N - 1 ); 
    magic_timestep = 0.5 * dx * dx / dt_scaling_factor;
    if ( do_overwrite_dt ) {
        dt = magic_timestep;
    }
}

PC3::System::System( int argc, char** argv ) : System() {

    // Check if help is requested
    if ( findInArgv( "--help", argc, argv ) != -1 || findInArgv( "-h", argc, argv ) != -1 ) {
        calculateAuto();
        printHelp();
        exit( 0 );
    }

    std::cout << EscapeSequence::BOLD << "---------------------------- Inputting System Parameters --------------------------" << EscapeSequence::RESET << std::endl;

    // Read-In commandline arguments
    init( argc, argv );

    // Calculate or scale variables
    calculateAuto();

    // Validate them
    validateInputs();

    filehandler.init( argc, argv );
}

bool PC3::System::evaluatePulse() {
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