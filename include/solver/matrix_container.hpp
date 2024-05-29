#pragma once
#include "cuda/cuda_complex.cuh"
#include "cuda/cuda_matrix.cuh"
#include "cuda/cuda_macro.cuh"

namespace PC3 {

/**
* DEFINE_MATRIX(type, name, size_scaling)
* type: The type of the matrix (real_number, complex_number, etc.)
* name: The name of the matrix
* size_scaling: The scaling factor for the size of the matrix
* 
* The matrices will always be constructed with the size N_x x N_y * size_scaling 
* on the Host only. When the device pointer is used for the first time, the
* Matrix class will handle the initialization of the device memory and the
* copying of the data from the host to the device.
*/

#define MATRIX_LIST \
    DEFINE_MATRIX(complex_number, initial_state_plus, 1) \
    DEFINE_MATRIX(complex_number, initial_state_minus, 1) \
    DEFINE_MATRIX(complex_number, wavefunction_plus, 1) \
    DEFINE_MATRIX(complex_number, wavefunction_minus, 1) \
    DEFINE_MATRIX(complex_number, reservoir_plus, 1) \
    DEFINE_MATRIX(complex_number, reservoir_minus, 1) \
    DEFINE_MATRIX(complex_number, pump_plus, n_pumps) \
    DEFINE_MATRIX(complex_number, pump_minus, n_pumps) \
    DEFINE_MATRIX(complex_number, pulse_plus, n_pulses) \
    DEFINE_MATRIX(complex_number, pulse_minus, n_pulses) \
    DEFINE_MATRIX(complex_number, potential_plus, n_potentials) \
    DEFINE_MATRIX(complex_number, potential_minus, n_potentials) \
    DEFINE_MATRIX(complex_number, buffer_wavefunction_plus, 1) \
    DEFINE_MATRIX(complex_number, buffer_wavefunction_minus, 1) \
    DEFINE_MATRIX(complex_number, buffer_reservoir_plus, 1) \
    DEFINE_MATRIX(complex_number, buffer_reservoir_minus, 1) \
    DEFINE_MATRIX(real_number, fft_mask_plus, 1) \
    DEFINE_MATRIX(real_number, fft_mask_minus, 1) \
    DEFINE_MATRIX(complex_number, fft_plus, 1) \
    DEFINE_MATRIX(complex_number, fft_minus, 1) \
    DEFINE_MATRIX(complex_number, k1_wavefunction_plus, 1) \
    DEFINE_MATRIX(complex_number, k1_wavefunction_minus, 1) \
    DEFINE_MATRIX(complex_number, k1_reservoir_plus, 1) \
    DEFINE_MATRIX(complex_number, k1_reservoir_minus, 1) \
    DEFINE_MATRIX(complex_number, k2_wavefunction_plus, 1) \
    DEFINE_MATRIX(complex_number, k2_wavefunction_minus, 1) \
    DEFINE_MATRIX(complex_number, k2_reservoir_plus, 1) \
    DEFINE_MATRIX(complex_number, k2_reservoir_minus, 1) \
    DEFINE_MATRIX(complex_number, k3_wavefunction_plus, 1) \
    DEFINE_MATRIX(complex_number, k3_wavefunction_minus, 1) \
    DEFINE_MATRIX(complex_number, k3_reservoir_plus, 1) \
    DEFINE_MATRIX(complex_number, k3_reservoir_minus, 1) \
    DEFINE_MATRIX(complex_number, k4_wavefunction_plus, 1) \
    DEFINE_MATRIX(complex_number, k4_wavefunction_minus, 1) \
    DEFINE_MATRIX(complex_number, k4_reservoir_plus, 1) \
    DEFINE_MATRIX(complex_number, k4_reservoir_minus, 1) \
    DEFINE_MATRIX(complex_number, k5_wavefunction_plus, 1) \
    DEFINE_MATRIX(complex_number, k5_wavefunction_minus, 1) \
    DEFINE_MATRIX(complex_number, k5_reservoir_plus, 1) \
    DEFINE_MATRIX(complex_number, k5_reservoir_minus, 1) \
    DEFINE_MATRIX(complex_number, k6_wavefunction_plus, 1) \
    DEFINE_MATRIX(complex_number, k6_wavefunction_minus, 1) \
    DEFINE_MATRIX(complex_number, k6_reservoir_plus, 1) \
    DEFINE_MATRIX(complex_number, k6_reservoir_minus, 1) \
    DEFINE_MATRIX(complex_number, k7_wavefunction_plus, 1) \
    DEFINE_MATRIX(complex_number, k7_wavefunction_minus, 1) \
    DEFINE_MATRIX(complex_number, k7_reservoir_plus, 1) \
    DEFINE_MATRIX(complex_number, k7_reservoir_minus, 1) \
    DEFINE_MATRIX(real_number, rk_error, 1) \
    DEFINE_MATRIX(complex_number, random_number, 1) \
    DEFINE_MATRIX(cuda_random_state, random_state, 1) \
    DEFINE_MATRIX(complex_number, snapshot_wavefunction_plus, 1) \
    DEFINE_MATRIX(complex_number, snapshot_wavefunction_minus, 1) \
    DEFINE_MATRIX(complex_number, snapshot_reservoir_plus, 1) \
    DEFINE_MATRIX(complex_number, snapshot_reservoir_minus, 1) // \ // <-- Don't forget the backslash!
    /////////////////////////////
    // Add your matrices here. //
    // Make sure to end each   //
    // but the last line with  //
    // a backslash!            //
    /////////////////////////////

struct MatrixContainer {
    // Declare all matrices using a macro
    #define DEFINE_MATRIX(type, name, size_scaling) PC3::CUDAMatrix<type> name;
    MATRIX_LIST
    #undef X

    // Empty Constructor
    MatrixContainer() = default;

    // Construction Chain
    void constructAll( const int N_x, const int N_y, bool use_twin_mode, bool use_rk_45, const int n_pulses, const int n_pumps, const int n_potentials ) {
        #define DEFINE_MATRIX(type, name, size_scaling) \
            name.construct( N_x, N_y * size_scaling, #name);
            MATRIX_LIST
        #undef X
     }

    // TODO: Now, only one line defines the matrix
    // But all matrices are synchronzized to the devie
    // Add macor preprocessing if statement that
    // Only creates the pointer if the matrix is used
    // On the other hand, there were only a few matrices that were
    // explicitely host-only: initial_state_plus and initial_state_minus

    struct Pointers {
        #define DEFINE_MATRIX(type, name, size_scaling) \
            type* name;
        MATRIX_LIST
        #undef X
    };

    Pointers pointers() {
        Pointers ptrs;
        #define DEFINE_MATRIX(type, name, size_scaling) \
                ptrs.name = name.getDevicePtr();
        MATRIX_LIST
        #undef X
        return ptrs;
    }
};

} // namespace PC3