#pragma once
#include <vector>
#include "cuda/cuda_complex.cuh"
#include "cuda/cuda_matrix.cuh"
#include "cuda/cuda_macro.cuh"

namespace PC3 {

/**
 * The Host Struct contains all the data that is needed for outputting and 
 * plotting. At the beginning of the program, we generate the initial matrices
 * on the CPU, which are saved in this struct. Then, the Solver copies all
 * the data to the GPU, where it is processed. After the simulation, the data
 * is copied back to the CPU, where it is saved in the Host struct.
 * Hence, the Host Struct mostly mirrors the host Struct.
*/
struct Host {
    PC3::HostMatrix<complex_number> initial_state_plus;
    PC3::HostMatrix<complex_number> initial_state_minus;

    PC3::HostMatrix<complex_number> wavefunction_plus;
    PC3::HostMatrix<complex_number> wavefunction_minus;
    PC3::HostMatrix<complex_number> reservoir_plus;
    PC3::HostMatrix<complex_number> reservoir_minus;
    std::vector<PC3::HostMatrix<complex_number>> pump_plus;
    std::vector<PC3::HostMatrix<complex_number>> pump_minus;
    std::vector<PC3::HostMatrix<complex_number>> pulse_plus;
    std::vector<PC3::HostMatrix<complex_number>> pulse_minus;
    std::vector<PC3::HostMatrix<complex_number>> potential_plus;
    std::vector<PC3::HostMatrix<complex_number>> potential_minus;

    // Snapshot Matrices (GUI only)
    PC3::HostMatrix<complex_number> snapshot_wavefunction_plus;
    PC3::HostMatrix<complex_number> snapshot_wavefunction_minus;
    PC3::HostMatrix<complex_number> snapshot_reservoir_plus;
    PC3::HostMatrix<complex_number> snapshot_reservoir_minus;
    
    // FFT Mask Matrices
    PC3::HostMatrix<real_number> fft_mask_plus;
    PC3::HostMatrix<real_number> fft_mask_minus;
    PC3::HostMatrix<complex_number> fft_plus;
    PC3::HostMatrix<complex_number> fft_minus;

    // "History" vectors; TODO: move to map
    std::vector<std::vector<complex_number>> wavefunction_plus_history, wavefunction_minus_history;
    std::vector<real_number> wavefunction_max_plus, wavefunction_max_minus;
    std::vector<real_number> times;

    // Empty Constructor
    Host() = default;

    // Construction Chain
    void constructAll( const int N_x, const int N_y, bool use_twin_mode, bool use_rk_45, const int n_pulses, const int n_pumps, const int n_potentials ) {
        pump_plus = std::vector<PC3::HostMatrix<complex_number>>( n_pumps );
        pulse_plus = std::vector<PC3::HostMatrix<complex_number>>( n_pulses );
        potential_plus = std::vector<PC3::HostMatrix<complex_number>>( n_potentials );
        pump_minus = std::vector<PC3::HostMatrix<complex_number>>( n_pumps );
        pulse_minus = std::vector<PC3::HostMatrix<complex_number>>( n_pulses );
        potential_minus = std::vector<PC3::HostMatrix<complex_number>>( n_potentials );

        // Wavefunction, Reservoir, Pump and FFT Matrices
        initial_state_plus.construct( N_x, N_y, "host.initial_state_plus" );
        wavefunction_plus.construct( N_x, N_y, "host.wavefunction_plus" );
        reservoir_plus.construct( N_x, N_y, "host.reservoir_plus" );
        for (auto i = 0; i < n_pumps; i++) {
            pump_plus[i].construct( N_x, N_y, "host.pump_plus" );
        }
        for (auto i = 0; i < n_pulses; i++) {
            pulse_plus[i].construct( N_x, N_y, "host.pulse_plus" );
        }
        for (auto i = 0; i < n_potentials; i++) {
            potential_plus[i].construct( N_x, N_y, "host.potential_plus" );
        }
        fft_mask_plus.construct( N_x, N_y, "host.fft_mask_plus" );
        fft_plus.construct( N_x, N_y, "host.fft_plus" );

        snapshot_wavefunction_plus.construct( N_x, N_y, "host.snapshot_wavefunction_plus" );
        snapshot_reservoir_plus.construct( N_x, N_y, "host.snapshot_reservoir_plus" );

        if ( not use_twin_mode )
            return;
            
        initial_state_minus.construct( N_x, N_y, "host.initial_state_minus" );
        wavefunction_minus.construct( N_x, N_y, "host.wavefunction_minus" );
        reservoir_minus.construct( N_x, N_y, "host.reservoir_minus" );
        for (auto i = 0; i < n_pumps; i++) {
            pump_minus[i].construct( N_x, N_y, "host.pump_minus" );
        }
        for (auto i = 0; i < n_pulses; i++) {
            pulse_minus[i].construct( N_x, N_y, "host.pulse_minus" );
        }
        for (auto i = 0; i < n_potentials; i++) {
            potential_minus[i].construct( N_x, N_y, "host.potential_minus" );
        }
        fft_mask_minus.construct( N_x, N_y, "host.fft_mask_minus" );
        fft_minus.construct( N_x, N_y, "host.fft_minus" );

        snapshot_wavefunction_minus.construct( N_x, N_y, "host.snapshot_wavefunction_minus" );
        snapshot_reservoir_minus.construct( N_x, N_y, "host.snapshot_reservoir_minus" );
    }
};

}