#pragma once

#include <fstream>
#include <iostream>
#include <map>
#include <vector>
#include <algorithm>
#include <memory>
#include <bit>
#include "cuda/cuda_complex.cuh"
#include "cuda/cuda_matrix.cuh"
#include "system/filehandler.hpp"
#include "system/envelope.hpp"

namespace PC3 {

/**
 * @brief Lightweight System Class containing all of the required system variables.
 * The System class reads the inputs from the commandline and saves them. The system class
 * also contains envelope wrappers for the different system variables and provides a few
 * helper functions called by the main loop cpu functions.
 */
class System {
   public:

    // System Parameters. These are also passed to the Kernels
    struct Parameters {

        // Size Variables
        unsigned int N_x, N_y, N2;
        // Time variables
        real_number t, dt;
        
        // SI Rescaling Units
        real_number m_e, h_bar, e_e, h_bar_s, m_eff, m_eff_scaled;
        
        // System Variables
        real_number L_x, L_y, dx, dy, dV;
        real_number gamma_c, gamma_r, g_c, g_r, R, g_pm, delta_LT;
        
        // Complex Scaled Values
        real_number one_over_h_bar_s;
        complex_number minus_i_over_h_bar_s, i_h_bar_s;
        complex_number i = {0.0,1.0};
        complex_number half_i = {0.0,0.5};
        complex_number minus_half_i = {0.0,-0.5};
        complex_number minus_i = {0.0,-1.0};

        // Boundary Conditions
        bool periodic_boundary_x, periodic_boundary_y;

        ////////////////////////////////
        // Custom Parameters go here! //
        ////////////////////////////////
        // TODO: maybe (at compile time) do a "include custom_parameters.hpp" here. bad practice, but it works

    } kernel_parameters;
    Parameters& p = kernel_parameters;

    // Numerics
    unsigned int iteration;

    // RK Solver Variables
    real_number t_max, dt_max, dt_min, tolerance, fft_every, random_system_amplitude, magic_timestep;

    // Kernel Block Size
    unsigned int block_size, omp_max_threads;

    bool fixed_time_step, normalize_before_masking, randomly_initialize_system, periodic_boundary_x, periodic_boundary_y;
    unsigned int random_seed;
    
    // History Output
    unsigned int history_output_n, history_y, history_matrix_start_x, history_matrix_start_y, history_matrix_end_x, history_matrix_end_y, history_matrix_output_increment;
    bool do_output_history_matrix;
    real_number output_every;

    bool do_overwrite_dt;
    bool imaginary_time;

    // Output of Variables
    std::vector<std::string> output_keys, input_keys;

    // Envelope ReadIns
    PC3::Envelope pulse, pump, mask, initial_state, fft_mask, potential;
    bool evaluate_reservoir_kernel = false;
    bool evaluate_pulse_kernel = false;

    real_number stochastic_amplitude;

    FileHandler filehandler;

    // Default Constructor; Initializes all variables to their default values
    System();
    // CMD Argument Constructor; Initializes passed variables according to the CMD arguments
    System( int argc, char** argv );

    template <typename... Args>
    bool doOutput( const Args&... args ) {
        return ( ( std::find( output_keys.begin(), output_keys.end(), args ) != output_keys.end() ) || ... );
    }

    template <typename... Args>
    bool doInput( const Args&... args ) {
        return ( ( std::find( input_keys.begin(), input_keys.end(), args ) != input_keys.end() ) || ... );
    }

    // Save structure as bool use_twin_mode
    bool use_twin_mode;

    bool evaluatePulse();
    bool evaluateReservoir();
    bool evaluateStochastic();

    void init( int argc, char** argv );
    void calculateAuto();
    void validateInputs();

    void printHelp();
    void printSummary( std::map<std::string, std::vector<double>> timeit_times, std::map<std::string, double> timeit_times_total );
};

} // namespace PC3