#pragma once
#include "cuda/cuda_complex.cuh"
#include "cuda/cuda_matrix.cuh"
#include "cuda/cuda_macro.cuh"

namespace PC3 {

/**
 * The Device struct contains all device pointer variables.
 * It also contains a Pointers struct, which is used to pass the device pointers
 * to the kernels.
 */
struct Device {
    // WaveFunction Matrices and Reservoir Matrices
    PC3::CUDAMatrix<complex_number> wavefunction_plus;
    PC3::CUDAMatrix<complex_number> wavefunction_minus;
    PC3::CUDAMatrix<complex_number> reservoir_plus;
    PC3::CUDAMatrix<complex_number> reservoir_minus;
    PC3::CUDAMatrix<complex_number> pump_plus;
    PC3::CUDAMatrix<complex_number> pump_minus;
    PC3::CUDAMatrix<complex_number> potential_plus;
    PC3::CUDAMatrix<complex_number> potential_minus;
    // "Next" versions of the above matrices. These are used to store the next iteration of the wavefunction.
    PC3::CUDAMatrix<complex_number> buffer_wavefunction_plus;
    PC3::CUDAMatrix<complex_number> buffer_wavefunction_minus;
    PC3::CUDAMatrix<complex_number> buffer_reservoir_plus;
    PC3::CUDAMatrix<complex_number> buffer_reservoir_minus;

    // Alias References to the plus components for easy access in a scalar child classes
    PC3::CUDAMatrix<complex_number>& wavefunction = wavefunction_plus;
    PC3::CUDAMatrix<complex_number>& reservoir = reservoir_plus;
    PC3::CUDAMatrix<complex_number>& pump = pump_plus;
    PC3::CUDAMatrix<complex_number>& potential = potential_plus;
    // "Next" versions
    PC3::CUDAMatrix<complex_number>& buffer_wavefunction = buffer_wavefunction_plus;
    PC3::CUDAMatrix<complex_number>& buffer_reservoir = buffer_reservoir_plus;

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

    // Empty Constructor
    Device() = default;

    // Construction Chain
    void constructAll( const int N_x, const int N_y, bool use_te_tm_splitting, bool use_rk_45 ) {
        // Wavefunction, Reservoir, Pump and FFT Matrices
        wavefunction_plus.construct( N_x, N_y, "device.wavefunction_plus" );
        reservoir_plus.construct( N_x, N_y, "device.reservoir_plus" );
        pump_plus.construct( N_x, N_y, "device.pump_plus" );
        potential_plus.construct( N_x, N_y, "device.potential_plus" );
        buffer_wavefunction_plus.construct( N_x, N_y, "device.buffer_wavefunction_plus" );
        buffer_reservoir_plus.construct( N_x, N_y, "device.buffer_reservoir_plus" );
        fft_mask_plus.construct( N_x, N_y, "device.fft_mask_plus" );
        fft_plus.construct( N_x, N_y, "device.fft_plus" );

        // RK4(5) Matrices
        k1_wavefunction_plus.construct( N_x, N_y, "device.k1_wavefunction_plus" );
        k1_reservoir_plus.construct( N_x, N_y, "device.k1_reservoir_plus" );
        k2_wavefunction_plus.construct( N_x, N_y, "device.k2_wavefunction_plus" );
        k2_reservoir_plus.construct( N_x, N_y, "device.k2_reservoir_plus" );
        k3_wavefunction_plus.construct( N_x, N_y, "device.k3_wavefunction_plus" );
        k3_reservoir_plus.construct( N_x, N_y, "device.k3_reservoir_plus" );
        k4_wavefunction_plus.construct( N_x, N_y, "device.k4_wavefunction_plus" );
        k4_reservoir_plus.construct( N_x, N_y, "device.k4_reservoir_plus" );

        rk_error.construct( N_x, N_y, "device.rk_error" );

        if ( use_rk_45 ) {
            k5_wavefunction_plus.construct( N_x, N_y, "device.k5_wavefunction_plus" );
            k5_reservoir_plus.construct( N_x, N_y, "device.k5_reservoir_plus" );
            k6_wavefunction_plus.construct( N_x, N_y, "device.k6_wavefunction_plus" );
            k6_reservoir_plus.construct( N_x, N_y, "device.k6_reservoir_plus" );
            k7_wavefunction_plus.construct( N_x, N_y, "device.k7_wavefunction_plus" );
            k7_reservoir_plus.construct( N_x, N_y, "device.k7_reservoir_plus" );
        }

        // TE/TM Guard
        if ( not use_te_tm_splitting )
            return;

        wavefunction_minus.construct( N_x, N_y, "device.wavefunction_minus" );
        reservoir_minus.construct( N_x, N_y, "device.reservoir_minus" );
        pump_minus.construct( N_x, N_y, "device.pump_minus" );
        potential_minus.construct( N_x, N_y, "device.potential_minus" );
        buffer_wavefunction_minus.construct( N_x, N_y, "device.buffer_wavefunction_minus" );
        buffer_reservoir_minus.construct( N_x, N_y, "device.buffer_reservoir_minus" );
        fft_mask_minus.construct( N_x, N_y, "device.fft_mask_minus" );
        fft_minus.construct( N_x, N_y, "device.fft_minus" );


        k1_wavefunction_minus.construct( N_x, N_y, "device.k1_wavefunction_minus" );
        k1_reservoir_minus.construct( N_x, N_y, "device.k1_reservoir_minus" );
        k2_wavefunction_minus.construct( N_x, N_y, "device.k2_wavefunction_minus" );
        k2_reservoir_minus.construct( N_x, N_y, "device.k2_reservoir_minus" );
        k3_wavefunction_minus.construct( N_x, N_y, "device.k3_wavefunction_minus" );
        k3_reservoir_minus.construct( N_x, N_y, "device.k3_reservoir_minus" );
        k4_wavefunction_minus.construct( N_x, N_y, "device.k4_wavefunction_minus" );
        k4_reservoir_minus.construct( N_x, N_y, "device.k4_reservoir_minus" );
        if ( use_rk_45 ) {
            k5_wavefunction_minus.construct( N_x, N_y, "device.k5_wavefunction_minus" );
            k5_reservoir_minus.construct( N_x, N_y, "device.k5_reservoir_minus" );
            k6_wavefunction_minus.construct( N_x, N_y, "device.k6_wavefunction_minus" );
            k6_reservoir_minus.construct( N_x, N_y, "device.k6_reservoir_minus" );
            k7_wavefunction_minus.construct( N_x, N_y, "device.k7_wavefunction_minus" );
            k7_reservoir_minus.construct( N_x, N_y, "device.k7_reservoir_minus" );
        }
    }

    // TODO: Try and just pass the Device struct to the kernels; see if performance
    // is affected.
    struct Pointers {
        complex_number* wavefunction_plus;
        complex_number* reservoir_plus;
        complex_number* pump_plus;
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
    };
    Pointers pointers(){
        return Pointers{
            wavefunction_plus.get(),
            reservoir_plus.get(),
            pump_plus.get(),
            potential_plus.get(),
            buffer_wavefunction_plus.get(),
            buffer_reservoir_plus.get(),
            k1_wavefunction_plus.get(),
            k1_reservoir_plus.get(),
            k2_wavefunction_plus.get(),
            k2_reservoir_plus.get(),
            k3_wavefunction_plus.get(),
            k3_reservoir_plus.get(),
            k4_wavefunction_plus.get(),
            k4_reservoir_plus.get(),
            k5_wavefunction_plus.get(),
            k5_reservoir_plus.get(),
            k6_wavefunction_plus.get(),
            k6_reservoir_plus.get(),
            k7_wavefunction_plus.get(),
            k7_reservoir_plus.get(),

            rk_error.get(),

            wavefunction_minus.get(),
            reservoir_minus.get(),
            pump_minus.get(),
            potential_minus.get(),
            buffer_wavefunction_minus.get(),
            buffer_reservoir_minus.get(),
            k1_wavefunction_minus.get(),
            k1_reservoir_minus.get(),
            k2_wavefunction_minus.get(),
            k2_reservoir_minus.get(),
            k3_wavefunction_minus.get(),
            k3_reservoir_minus.get(),
            k4_wavefunction_minus.get(),
            k4_reservoir_minus.get(),
            k5_wavefunction_minus.get(),
            k5_reservoir_minus.get(),
            k6_wavefunction_minus.get(),
            k6_reservoir_minus.get(),
            k7_wavefunction_minus.get(),
            k7_reservoir_minus.get(),
        };
    } 
};

} // namespace PC3