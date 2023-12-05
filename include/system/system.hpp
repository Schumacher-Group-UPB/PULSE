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
    // SI Rescaling Units
    real_number m_e, h_bar, e_e;
    real_number h_bar_s;

    // System Variables
    real_number m_eff, gamma_c, gamma_r, g_c, g_r, R, g_pm, delta_LT;

    // Numerics
    unsigned int s_N, iteration;

    // RK Solver Variables
    real_number xmax, dx, t_max, dt, t, dt_max, dt_min, tolerance, fft_every, random_system_amplitude;

    // Kernel Block Size
    unsigned int block_size, omp_max_threads;

    bool fixed_time_step, normalize_before_masking, randomly_initialize_system, periodic_boundary_conditions;
    unsigned int random_seed;
    
    // History Output
    unsigned int history_output_n, history_matrix_start, history_matrix_end, history_matrix_output_increment;
    bool do_output_history_matrix;

    // Output of Variables
    std::vector<std::string> output_keys, input_keys;

    // Envelope ReadIns
    PC3::Envelope pulse, pump, mask, initial_state, fft_mask, potential;

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

    // Different System structures implemented
    //enum class Structure : unsigned int {
    //    Scalar = 1,
    //    TETM = 1 << 1,
    //    // TODO:
    //    // Resonant = 1 << 2, // via +
    //} structure;
    // For now, save structure as bool use_te_tm_splitting
    bool use_te_tm_splitting;

    /**
     * @brief Calculates a mask for the given system and saves it to the mask buffer.
     * @param buffer The buffer to save the mask to.
     * @param mask The envelope of the mask.
     * @param polarization The polarization of the mask. If set to 0, the mask will
     * always be applied no matter the polarization. If set to 1, the mask will only
     * be applied to the plus polarization. If set to -1, the mask will only be
     * applied to the minus polarization.
     */
    void calculateEnvelope( real_number* buffer, const PC3::Envelope& mask, PC3::Envelope::Polarization polarization, real_number default_value_if_no_mask = 0.0 );
    void calculateEnvelope( complex_number* buffer, const PC3::Envelope& mask, PC3::Envelope::Polarization polarization, real_number default_value_if_no_mask = 0.0 );

    bool evaluatePulse();

    void printHelp();
    void printSummary( std::map<std::string, std::vector<double>> timeit_times, std::map<std::string, double> timeit_times_total );

    // System Parameters to be passed to a Kernel
    struct Parameters {
        unsigned int N, N2;
        real_number t, dt, dt_half, xmax, dx, m_e, h_bar_s, m_eff, gamma_c, gamma_r, g_c, g_r, R, g_pm, delta_LT;
        real_number one_over_h_bar_s;
        complex_number minus_i_over_h_bar_s, i_h_bar_s, half_i, i, minus_half_i, minus_i;
        real_number m_eff_scaled, delta_LT_scaled;
        bool periodic;
        Parameters( unsigned int N, real_number t, real_number dt, real_number xmax, real_number dx,
                    real_number m_e, real_number h_bar_s, real_number m_eff, real_number gamma_c, real_number gamma_r, real_number g_c, real_number g_r, real_number R, real_number g_pm, real_number delta_LT, bool periodic ) : N( N ), N2( N * N ), t( t ), dt( dt ), dt_half( dt / real_number( 2.0 ) ), xmax( xmax ), dx( dx ), m_e( m_e ), m_eff( m_eff ), gamma_c( gamma_c ), gamma_r( gamma_r ), g_c( g_c ), g_r( g_r ), R( R ), g_pm( g_pm ), delta_LT( delta_LT ), h_bar_s( h_bar_s ), one_over_h_bar_s( real_number( 1.0 ) / h_bar_s ), minus_i_over_h_bar_s( complex_number( 0.0, real_number( -1.0 ) / h_bar_s ) ), i_h_bar_s( complex_number( 0.0, h_bar_s ) ), half_i( complex_number( 0.0, 0.5 ) ), i( complex_number( 0.0, 1.0 ) ), minus_half_i( complex_number( 0.0, real_number( -0.5 ) ) ), minus_i( complex_number( 0.0, real_number( -1.0 ) ) ), periodic( periodic ) {
            m_eff_scaled = real_number( -0.5 ) * h_bar_s * h_bar_s / ( m_eff * dx * dx );
            delta_LT_scaled = delta_LT / dx / dx;
        }
    };

    Parameters snapshotParameters() {
        return Parameters( s_N, t, dt, xmax, dx, m_e, h_bar_s, m_eff, gamma_c, gamma_r, g_c, g_r, R, g_pm, delta_LT, periodic_boundary_conditions );
    }

};

} // namespace PC3