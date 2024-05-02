#pragma once
#include "cuda/cuda_complex.cuh"
#include "cuda/cuda_matrix.cuh"
#include "cuda/cuda_macro.cuh"

namespace PC3 {

struct MatrixContainer {
    // WaveFunction Matrices and Reservoir Matrices
    PC3::CUDAMatrix<complex_number> initial_state_plus;
    PC3::CUDAMatrix<complex_number> initial_state_minus;
    PC3::CUDAMatrix<complex_number> wavefunction_plus;
    PC3::CUDAMatrix<complex_number> wavefunction_minus;
    PC3::CUDAMatrix<complex_number> reservoir_plus;
    PC3::CUDAMatrix<complex_number> reservoir_minus;
    PC3::CUDAMatrix<complex_number> pump_plus;
    PC3::CUDAMatrix<complex_number> pump_minus;
    PC3::CUDAMatrix<complex_number> pulse_plus;
    PC3::CUDAMatrix<complex_number> pulse_minus;
    PC3::CUDAMatrix<complex_number> potential_plus;
    PC3::CUDAMatrix<complex_number> potential_minus;
    // "Next" versions of the above matrices. These are used to store the next iteration of the wavefunction.
    PC3::CUDAMatrix<complex_number> buffer_wavefunction_plus;
    PC3::CUDAMatrix<complex_number> buffer_wavefunction_minus;
    PC3::CUDAMatrix<complex_number> buffer_reservoir_plus;
    PC3::CUDAMatrix<complex_number> buffer_reservoir_minus;

    // FFT Mask Matrices
    PC3::CUDAMatrix<real_number> fft_mask_plus;
    PC3::CUDAMatrix<real_number> fft_mask_minus;
    PC3::CUDAMatrix<complex_number> fft_plus;
    PC3::CUDAMatrix<complex_number> fft_minus;

    // K Matrices. We need 7 K matrices for RK45 and 4 K matrices for RK4.
    // We define all of them here, and allocate/construct only the ones we need.
    PC3::CUDAMatrix<complex_number> k1_wavefunction_plus;
    PC3::CUDAMatrix<complex_number> k1_wavefunction_minus;
    PC3::CUDAMatrix<complex_number> k1_reservoir_plus;
    PC3::CUDAMatrix<complex_number> k1_reservoir_minus;

    PC3::CUDAMatrix<complex_number> k2_wavefunction_plus;
    PC3::CUDAMatrix<complex_number> k2_wavefunction_minus;
    PC3::CUDAMatrix<complex_number> k2_reservoir_plus;
    PC3::CUDAMatrix<complex_number> k2_reservoir_minus;

    PC3::CUDAMatrix<complex_number> k3_wavefunction_plus;
    PC3::CUDAMatrix<complex_number> k3_wavefunction_minus;
    PC3::CUDAMatrix<complex_number> k3_reservoir_plus;
    PC3::CUDAMatrix<complex_number> k3_reservoir_minus;

    PC3::CUDAMatrix<complex_number> k4_wavefunction_plus;
    PC3::CUDAMatrix<complex_number> k4_wavefunction_minus;
    PC3::CUDAMatrix<complex_number> k4_reservoir_plus;
    PC3::CUDAMatrix<complex_number> k4_reservoir_minus;

    PC3::CUDAMatrix<complex_number> k5_wavefunction_plus;
    PC3::CUDAMatrix<complex_number> k5_wavefunction_minus;
    PC3::CUDAMatrix<complex_number> k5_reservoir_plus;
    PC3::CUDAMatrix<complex_number> k5_reservoir_minus;

    PC3::CUDAMatrix<complex_number> k6_wavefunction_plus;
    PC3::CUDAMatrix<complex_number> k6_wavefunction_minus;
    PC3::CUDAMatrix<complex_number> k6_reservoir_plus;
    PC3::CUDAMatrix<complex_number> k6_reservoir_minus;

    PC3::CUDAMatrix<complex_number> k7_wavefunction_plus;
    PC3::CUDAMatrix<complex_number> k7_wavefunction_minus;
    PC3::CUDAMatrix<complex_number> k7_reservoir_plus;
    PC3::CUDAMatrix<complex_number> k7_reservoir_minus;

    // RK45 Error Matrix
    PC3::CUDAMatrix<real_number> rk_error;

    // Random Number Cache
    PC3::CUDAMatrix<complex_number> random_number;
    PC3::CUDAMatrix<cuda_random_state> random_state;

    std::vector<complex_number*> pump_plus_array;
    std::vector<complex_number*> pulse_plus_array;
    std::vector<complex_number*> potential_plus_array;
    std::vector<complex_number*> pump_minus_array;
    std::vector<complex_number*> pulse_minus_array;
    std::vector<complex_number*> potential_minus_array;

    // Snapshot Matrices (GUI only)
    PC3::CUDAMatrix<complex_number> snapshot_wavefunction_plus;
    PC3::CUDAMatrix<complex_number> snapshot_wavefunction_minus;
    PC3::CUDAMatrix<complex_number> snapshot_reservoir_plus;
    PC3::CUDAMatrix<complex_number> snapshot_reservoir_minus;

    // "History" vectors; TODO: move to map
    std::vector<std::vector<complex_number>> wavefunction_plus_history, wavefunction_minus_history;
    std::vector<real_number> wavefunction_max_plus, wavefunction_max_minus;
    std::vector<real_number> times;

    //////////////////////////////
    // Custom Matrices go here! //
    // Don't forget to add them //
    // to the construction and  //
    // pointers too!            //
    //////////////////////////////

    // Empty Constructor
    MatrixContainer() = default;

    // Construction Chain
    void constructAll( const int N_x, const int N_y, bool use_twin_mode, bool use_rk_45, const int n_pulses, const int n_pumps, const int n_potentials ) {
        // Construct Random Number Cache
        random_number.constructDevice( N_x, N_y, "random_number" );
        random_state.constructDevice( N_x, N_y, "random_state" );

        // Wavefunction, Reservoir, Pump and FFT Matrices
        initial_state_plus.constructHost( N_x, N_y, "initial_state_plus" );
        wavefunction_plus.construct( N_x, N_y, "wavefunction_plus" );
        reservoir_plus.construct( N_x, N_y, "reservoir_plus" );
        pump_plus.construct( N_x, N_y * n_pumps, "pump_plus" );
        pulse_plus.construct( N_x, N_y * n_pulses, "pulse_plus" );
        potential_plus.construct( N_x, N_y * n_potentials, "potential_plus" );
        buffer_wavefunction_plus.construct( N_x, N_y, "buffer_wavefunction_plus" );
        buffer_reservoir_plus.construct( N_x, N_y, "buffer_reservoir_plus" );
        fft_mask_plus.construct( N_x, N_y, "fft_mask_plus" );
        fft_plus.construct( N_x, N_y, "fft_plus" );

        // RK4(5) Matrices
        k1_wavefunction_plus.constructDevice( N_x, N_y, "k1_wavefunction_plus" );
        k1_reservoir_plus.constructDevice( N_x, N_y, "k1_reservoir_plus" );
        k2_wavefunction_plus.constructDevice( N_x, N_y, "k2_wavefunction_plus" );
        k2_reservoir_plus.constructDevice( N_x, N_y, "k2_reservoir_plus" );
        k3_wavefunction_plus.constructDevice( N_x, N_y, "k3_wavefunction_plus" );
        k3_reservoir_plus.constructDevice( N_x, N_y, "k3_reservoir_plus" );
        k4_wavefunction_plus.constructDevice( N_x, N_y, "k4_wavefunction_plus" );
        k4_reservoir_plus.constructDevice( N_x, N_y, "k4_reservoir_plus" );

        rk_error.constructDevice( N_x, N_y, "rk_error" );

        if ( use_rk_45 ) {
            k5_wavefunction_plus.constructDevice( N_x, N_y, "k5_wavefunction_plus" );
            k5_reservoir_plus.constructDevice( N_x, N_y, "k5_reservoir_plus" );
            k6_wavefunction_plus.constructDevice( N_x, N_y, "k6_wavefunction_plus" );
            k6_reservoir_plus.constructDevice( N_x, N_y, "k6_reservoir_plus" );
            k7_wavefunction_plus.constructDevice( N_x, N_y, "k7_wavefunction_plus" );
            k7_reservoir_plus.constructDevice( N_x, N_y, "k7_reservoir_plus" );
        }

        snapshot_wavefunction_plus.construct( N_x, N_y, "snapshot_wavefunction_plus" );
        snapshot_reservoir_plus.construct( N_x, N_y, "snapshot_reservoir_plus" );

        // TE/TM Guard
        if ( not use_twin_mode )
            return;

        initial_state_minus.constructHost( N_x, N_y, "initial_state_minus" );
        wavefunction_minus.construct( N_x, N_y, "wavefunction_minus" );
        reservoir_minus.construct( N_x, N_y, "reservoir_minus" );
        pump_minus.construct( N_x, N_y * n_pumps, "pump_minus" );
        pulse_minus.construct( N_x, N_y * n_pulses, "pulse_minus" );
        potential_minus.construct( N_x, N_y * n_potentials, "potential_minus" );
        buffer_wavefunction_minus.construct( N_x, N_y, "buffer_wavefunction_minus" );
        buffer_reservoir_minus.construct( N_x, N_y, "buffer_reservoir_minus" );
        fft_mask_minus.construct( N_x, N_y, "fft_mask_minus" );
        fft_minus.construct( N_x, N_y, "fft_minus" );

        k1_wavefunction_minus.constructDevice( N_x, N_y, "k1_wavefunction_minus" );
        k1_reservoir_minus.constructDevice( N_x, N_y, "k1_reservoir_minus" );
        k2_wavefunction_minus.constructDevice( N_x, N_y, "k2_wavefunction_minus" );
        k2_reservoir_minus.constructDevice( N_x, N_y, "k2_reservoir_minus" );
        k3_wavefunction_minus.constructDevice( N_x, N_y, "k3_wavefunction_minus" );
        k3_reservoir_minus.constructDevice( N_x, N_y, "k3_reservoir_minus" );
        k4_wavefunction_minus.constructDevice( N_x, N_y, "k4_wavefunction_minus" );
        k4_reservoir_minus.constructDevice( N_x, N_y, "k4_reservoir_minus" );
        if ( use_rk_45 ) {
            k5_wavefunction_minus.constructDevice( N_x, N_y, "k5_wavefunction_minus" );
            k5_reservoir_minus.constructDevice( N_x, N_y, "k5_reservoir_minus" );
            k6_wavefunction_minus.constructDevice( N_x, N_y, "k6_wavefunction_minus" );
            k6_reservoir_minus.constructDevice( N_x, N_y, "k6_reservoir_minus" );
            k7_wavefunction_minus.constructDevice( N_x, N_y, "k7_wavefunction_minus" );
            k7_reservoir_minus.constructDevice( N_x, N_y, "k7_reservoir_minus" );
        }

        snapshot_wavefunction_minus.construct( N_x, N_y, "snapshot_wavefunction_minus" );
        snapshot_reservoir_minus.construct( N_x, N_y, "snapshot_reservoir_minus" );
    }

    // TODO: Try and just pass the Device struct to the kernels; see if performance
    // is affected.
    struct Pointers {
        complex_number* wavefunction_plus;
        complex_number* reservoir_plus;
        complex_number* pump_plus;
        complex_number* pulse_plus;
        complex_number* potential_plus;
        complex_number* buffer_wavefunction_plus;
        complex_number* buffer_reservoir_plus;
        complex_number* k1_wavefunction_plus;
        complex_number* k1_reservoir_plus;
        complex_number* k2_wavefunction_plus;
        complex_number* k2_reservoir_plus;
        complex_number* k3_wavefunction_plus;
        complex_number* k3_reservoir_plus;
        complex_number* k4_wavefunction_plus;
        complex_number* k4_reservoir_plus;
        complex_number* k5_wavefunction_plus;
        complex_number* k5_reservoir_plus;
        complex_number* k6_wavefunction_plus;
        complex_number* k6_reservoir_plus;
        complex_number* k7_wavefunction_plus;
        complex_number* k7_reservoir_plus;

        real_number* rk_error;

        complex_number* wavefunction_minus;
        complex_number* reservoir_minus;
        complex_number* pump_minus;
        complex_number* pulse_minus;
        complex_number* potential_minus;
        complex_number* buffer_wavefunction_minus;
        complex_number* buffer_reservoir_minus;
        complex_number* k1_wavefunction_minus;
        complex_number* k1_reservoir_minus;
        complex_number* k2_wavefunction_minus;
        complex_number* k2_reservoir_minus;
        complex_number* k3_wavefunction_minus;
        complex_number* k3_reservoir_minus;
        complex_number* k4_wavefunction_minus;
        complex_number* k4_reservoir_minus;
        complex_number* k5_wavefunction_minus;
        complex_number* k5_reservoir_minus;
        complex_number* k6_wavefunction_minus;
        complex_number* k6_reservoir_minus;
        complex_number* k7_wavefunction_minus;
        complex_number* k7_reservoir_minus;

        complex_number* random_number;
        cuda_random_state* random_state;
    };

    Pointers pointers() {
        return Pointers{
            wavefunction_plus.getDevicePtr(),
            reservoir_plus.getDevicePtr(),
            pump_plus.getDevicePtr(),
            pulse_plus.getDevicePtr(),
            potential_plus.getDevicePtr(),
            buffer_wavefunction_plus.getDevicePtr(),
            buffer_reservoir_plus.getDevicePtr(),
            k1_wavefunction_plus.getDevicePtr(),
            k1_reservoir_plus.getDevicePtr(),
            k2_wavefunction_plus.getDevicePtr(),
            k2_reservoir_plus.getDevicePtr(),
            k3_wavefunction_plus.getDevicePtr(),
            k3_reservoir_plus.getDevicePtr(),
            k4_wavefunction_plus.getDevicePtr(),
            k4_reservoir_plus.getDevicePtr(),
            k5_wavefunction_plus.getDevicePtr(),
            k5_reservoir_plus.getDevicePtr(),
            k6_wavefunction_plus.getDevicePtr(),
            k6_reservoir_plus.getDevicePtr(),
            k7_wavefunction_plus.getDevicePtr(),
            k7_reservoir_plus.getDevicePtr(),

            rk_error.getDevicePtr(),

            wavefunction_minus.getDevicePtr(),
            reservoir_minus.getDevicePtr(),
            pump_minus.getDevicePtr(),
            pulse_minus.getDevicePtr(),
            potential_minus.getDevicePtr(),
            buffer_wavefunction_minus.getDevicePtr(),
            buffer_reservoir_minus.getDevicePtr(),
            k1_wavefunction_minus.getDevicePtr(),
            k1_reservoir_minus.getDevicePtr(),
            k2_wavefunction_minus.getDevicePtr(),
            k2_reservoir_minus.getDevicePtr(),
            k3_wavefunction_minus.getDevicePtr(),
            k3_reservoir_minus.getDevicePtr(),
            k4_wavefunction_minus.getDevicePtr(),
            k4_reservoir_minus.getDevicePtr(),
            k5_wavefunction_minus.getDevicePtr(),
            k5_reservoir_minus.getDevicePtr(),
            k6_wavefunction_minus.getDevicePtr(),
            k6_reservoir_minus.getDevicePtr(),
            k7_wavefunction_minus.getDevicePtr(),
            k7_reservoir_minus.getDevicePtr(),

            random_number.getDevicePtr(),
            random_state.getDevicePtr() };
    }
};

} // namespace PC3