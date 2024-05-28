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
    p.m_e = 9.10938356E-31;
    p.h_bar = 1.0545718E-34;
    p.e_e = 1.60217662E-19;
    // h_bar_scaled will be calculated in the calculateAuto() function
    p.h_bar_s = -1; // = hbar*e*1E12 == 6.582119514E-4;

    // System Variables
    p.m_eff = -1; // = 1E-4 * m_e / e * 1E12  = 0.00056856;
    p.m_eff_scaled = 0;
    p.gamma_c = 0.15;          // ps^-1
    p.gamma_r = 1.5 * p.gamma_c; // ps^-1
    p.g_c = 3.E-6;             // meV mum^2
    p.g_r = 2. * p.g_c;          // meV mum^2
    p.R = 0.01;                // ps^-1 mum^2
    p.L_x = 100.;             // mum
    p.L_y = 100.;             // mum
    p.g_pm = -p.g_c / 5;         // meV mum^2
    p.delta_LT = 0.025E-3;     // meV

    // Numerics
    p.N_x = 400;
    p.N_y = 400;
    t_max = 1000;
    iteration = 0;
    // RK Solver Variables
    p.t = 1000;
    output_every = 1;
    dt_max = 3;
    dt_min = 0.0001; // also dt_delta
    tolerance = 1E-1;
    do_overwrite_dt = true;

    // FFT Mask every x ps
    fft_every = 1; // ps

    // Kernel Block Size
    block_size = 256;
    omp_max_threads = omp_get_max_threads();

    // If this is true, the solver will use a fixed timestep RK4 method instead of the variable timestep RK45 method
    fixed_time_step = true;

    // Output of Variables
    output_keys = { "mat", "scalar" };

    // By Default, no stochstic noise is added
    p.stochastic_amplitude = 0.0;

    randomly_initialize_system = false;
    random_system_amplitude = 1.0;
}

void PC3::System::calculateAuto() {
    // If hbar_s is < 0, calculate it
    if ( p.h_bar_s < 0 ) {
        p.h_bar_s = p.h_bar / p.e_e * 1E12;
    }
    // If m_eff is < 0, calculate it
    if ( p.m_eff < 0 ) {
        p.m_eff = p.m_e / p.e_e * 1E8;
    }

    // SI scaling factor for the magic timestep. If the user choses a custom
    // effective mass, the magic timestep will be scaled accordingly.
    const auto dt_scaling_factor = p.m_e / p.e_e * 1E8 / p.m_eff;
    // Calculate dx and dt
    p.dx = 2.0 * p.L_x / ( p.N_x - 1 ); 
    p.dy = 2.0 * p.L_y / ( p.N_y - 1 ); 
    p.m_eff_scaled = -0.5 * p.h_bar_s * p.h_bar_s / p.m_eff;
    magic_timestep = 0.5 * p.dx * p.dy / dt_scaling_factor;
    if ( do_overwrite_dt ) {
        p.dt = magic_timestep;
    }

    // Calculate scaled imaginary values
    p.one_over_h_bar_s = 1.0 / p.h_bar_s;
    p.minus_i_over_h_bar_s = { 0.0, -real_number(1.0) / p.h_bar_s };
    p.i_h_bar_s = { 0.0, p.h_bar_s };
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
    if (not evaluate_pulse_kernel)
        return false;
    bool evaluate_pulse = false;
    for ( int c = 0; c < pulse.t0.size(); c++ ) {
        const auto t0 = pulse.t0[c];
        const auto sigma = pulse.sigma[c];
        if ( t0 - 5. * sigma < p.t && p.t < t0 + 5. * sigma ) {
            evaluate_pulse = true;
            break;
        }
    }
    return evaluate_pulse;
}

bool PC3::System::evaluateReservoir() {
    return evaluate_reservoir_kernel;
}

bool PC3::System::evaluateStochastic() {
    return p.stochastic_amplitude != 0.0;
}