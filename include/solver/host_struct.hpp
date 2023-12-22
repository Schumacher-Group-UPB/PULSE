#pragma once
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
    PC3::HostMatrix<complex_number> pump_plus;
    PC3::HostMatrix<complex_number> pump_minus;
    PC3::HostMatrix<complex_number> potential_plus;
    PC3::HostMatrix<complex_number> potential_minus;
    
    // Alias References to the plus components for easy access in a scalar child classes
    PC3::HostMatrix<complex_number>& wavefunction = wavefunction_plus;
    PC3::HostMatrix<complex_number>& reservoir = reservoir_plus;
    PC3::HostMatrix<complex_number>& pump = pump_plus;
    PC3::HostMatrix<complex_number>& potential = potential_plus;
    
    // FFT Mask Matrices
    PC3::HostMatrix<real_number> fft_mask_plus;
    PC3::HostMatrix<real_number> fft_mask_minus;
    PC3::HostMatrix<complex_number> fft_plus;
    PC3::HostMatrix<complex_number> fft_minus;

    // Alias References to the plus components for easy access in a scalar child classes
    PC3::HostMatrix<real_number>& fft_mask = fft_mask_plus;
    PC3::HostMatrix<complex_number>& fft = fft_plus;

    // Soll Matrices
    PC3::HostMatrix<real_number> soll_plus;
    PC3::HostMatrix<real_number> soll_minus;

    // "History" vectors
    std::vector<std::vector<complex_number>> wavefunction_plus_history, wavefunction_minus_history;
    std::vector<real_number> wavefunction_max_plus, wavefunction_max_minus;

    // Empty Constructor
    Host() = default;

    // Construction Chain
    void constructAll( const int N_x, const int N_y, bool use_te_tm_splitting, bool use_rk_45 ) {
        // Wavefunction, Reservoir, Pump and FFT Matrices
        initial_state_plus.construct( N_x, N_y, "host.initial_state_plus" );
        wavefunction_plus.construct( N_x, N_y, "host.wavefunction_plus" );
        reservoir_plus.construct( N_x, N_y, "host.reservoir_plus" );
        pump_plus.construct( N_x, N_y, "host.pump_plus" );
        potential_plus.construct( N_x, N_y, "host.potential_plus" );
        fft_mask_plus.construct( N_x, N_y, "host.fft_mask_plus" );
        fft_plus.construct( N_x, N_y, "host.fft_plus" );
        soll_plus.construct( N_x, N_y, "host.soll_plus" );
        if ( use_te_tm_splitting ) {
            initial_state_minus.construct( N_x, N_y, "host.initial_state_minus" );
            wavefunction_minus.construct( N_x, N_y, "host.wavefunction_minus" );
            reservoir_minus.construct( N_x, N_y, "host.reservoir_minus" );
            pump_minus.construct( N_x, N_y, "host.pump_minus" );
            potential_minus.construct( N_x, N_y, "host.potential_minus" );
            fft_mask_minus.construct( N_x, N_y, "host.fft_mask_minus" );
            fft_minus.construct( N_x, N_y, "host.fft_minus" );
            soll_minus.construct( N_x, N_y, "host.soll_minus" );
        }
    }
};

}