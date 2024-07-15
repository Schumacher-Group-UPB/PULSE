#pragma once
#include "cuda/typedef.cuh"
#include "cuda/cuda_matrix.cuh"

namespace PC3 {

/**
* DEFINE_MATRIX(type, name, size_scaling, condition_for_construction)
* type: The type of the matrix (Type::real, Type::complex, etc.)
* name: The name of the matrix
* size_scaling: The scaling factor for the size of the matrix
* condition_for_construction: if false, neither the host nor the device matrices are constructed
* 
* We use this not-so-pretty macro to define matrices to shorten this file and to 
* make it easier for the user to add new matrices.
*/

#define MATRIX_LIST \
    DEFINE_MATRIX(Type::complex, initial_state_plus, 1, false) \
    DEFINE_MATRIX(Type::complex, initial_state_minus, 1, false) \
    DEFINE_MATRIX(Type::complex, initial_reservoir_plus, 1, false) \
    DEFINE_MATRIX(Type::complex, initial_reservoir_minus, 1, false) \
    DEFINE_MATRIX(Type::complex, wavefunction_plus, 1, true) \
    DEFINE_MATRIX(Type::complex, wavefunction_minus, 1, use_twin_mode) \
    DEFINE_MATRIX(Type::complex, reservoir_plus, 1, true) \
    DEFINE_MATRIX(Type::complex, reservoir_minus, 1, use_twin_mode) \
    DEFINE_MATRIX(Type::complex, pump_plus, n_pumps, true) \
    DEFINE_MATRIX(Type::complex, pump_minus, n_pumps, use_twin_mode) \
    DEFINE_MATRIX(Type::complex, pulse_plus, n_pulses, true) \
    DEFINE_MATRIX(Type::complex, pulse_minus, n_pulses, use_twin_mode) \
    DEFINE_MATRIX(Type::complex, potential_plus, n_potentials, true) \
    DEFINE_MATRIX(Type::complex, potential_minus, n_potentials, use_twin_mode) \
    DEFINE_MATRIX(Type::complex, buffer_wavefunction_plus, 1, true) \
    DEFINE_MATRIX(Type::complex, buffer_wavefunction_minus, 1, use_twin_mode) \
    DEFINE_MATRIX(Type::complex, buffer_reservoir_plus, 1, true) \
    DEFINE_MATRIX(Type::complex, buffer_reservoir_minus, 1, use_twin_mode) \
    DEFINE_MATRIX(Type::real, fft_mask_plus, 1, true) \
    DEFINE_MATRIX(Type::real, fft_mask_minus, 1, use_twin_mode) \
    DEFINE_MATRIX(Type::complex, fft_plus, 1, true) \
    DEFINE_MATRIX(Type::complex, fft_minus, 1, use_twin_mode) \
    DEFINE_MATRIX(Type::complex, k1_wavefunction_plus, 1, true) \
    DEFINE_MATRIX(Type::complex, k1_wavefunction_minus, 1, use_twin_mode) \
    DEFINE_MATRIX(Type::complex, k1_reservoir_plus, 1, true) \
    DEFINE_MATRIX(Type::complex, k1_reservoir_minus, 1, use_twin_mode) \
    DEFINE_MATRIX(Type::complex, k2_wavefunction_plus, 1, true) \
    DEFINE_MATRIX(Type::complex, k2_wavefunction_minus, 1, use_twin_mode) \
    DEFINE_MATRIX(Type::complex, k2_reservoir_plus, 1, true) \
    DEFINE_MATRIX(Type::complex, k2_reservoir_minus, 1, use_twin_mode) \
    DEFINE_MATRIX(Type::complex, k3_wavefunction_plus, 1, true) \
    DEFINE_MATRIX(Type::complex, k3_wavefunction_minus, 1, use_twin_mode) \
    DEFINE_MATRIX(Type::complex, k3_reservoir_plus, 1, true) \
    DEFINE_MATRIX(Type::complex, k3_reservoir_minus, 1, use_twin_mode) \
    DEFINE_MATRIX(Type::complex, k4_wavefunction_plus, 1, true) \
    DEFINE_MATRIX(Type::complex, k4_wavefunction_minus, 1, use_twin_mode) \
    DEFINE_MATRIX(Type::complex, k4_reservoir_plus, 1, true) \
    DEFINE_MATRIX(Type::complex, k4_reservoir_minus, 1, use_twin_mode) \
    DEFINE_MATRIX(Type::complex, k5_wavefunction_plus, 1, true) \
    DEFINE_MATRIX(Type::complex, k5_wavefunction_minus, 1, use_twin_mode and true) \
    DEFINE_MATRIX(Type::complex, k5_reservoir_plus, 1, true) \
    DEFINE_MATRIX(Type::complex, k5_reservoir_minus, 1, use_twin_mode and true) \
    DEFINE_MATRIX(Type::complex, k6_wavefunction_plus, 1, true) \
    DEFINE_MATRIX(Type::complex, k6_wavefunction_minus, 1, use_twin_mode and true) \
    DEFINE_MATRIX(Type::complex, k6_reservoir_plus, 1, true) \
    DEFINE_MATRIX(Type::complex, k6_reservoir_minus, 1, use_twin_mode and true) \
    DEFINE_MATRIX(Type::real, rk_error, 1, true) \
    DEFINE_MATRIX(Type::complex, random_number, 1, true) \
    DEFINE_MATRIX(Type::cuda_random_state, random_state, 1, true) \
    DEFINE_MATRIX(Type::complex, snapshot_wavefunction_plus, 1, false) \
    DEFINE_MATRIX(Type::complex, snapshot_wavefunction_minus, 1, false) \
    DEFINE_MATRIX(Type::complex, snapshot_reservoir_plus, 1, false) \
    DEFINE_MATRIX(Type::complex, snapshot_reservoir_minus, 1, false) // \ // <-- Don't forget the backslash!
    /////////////////////////////
    // Add your matrices here. //
    // Make sure to end each   //
    // but the last line with  //
    // a backslash!            //
    /////////////////////////////

struct MatrixContainer {

    // Cache triggers
    bool use_twin_mode, use_rk_45;

    // Declare all matrices using a macro
    #define DEFINE_MATRIX(type, name, size_scaling, condition_for_construction) PC3::CUDAMatrix<type> name;
    MATRIX_LIST
    #undef X

    // Empty Constructor
    MatrixContainer() = default;

    // TODO: if reservoir... system.evaluateReservoir() !

    // Construction Chain. The Host Matrix is always constructed (who carese about RAM right?) and the device matrix is constructed if the condition is met.
    void constructAll( const int N_x, const int N_y, bool use_twin_mode, bool use_rk_45, const int n_pulses, const int n_pumps, const int n_potentials ) {
        this->use_twin_mode = use_twin_mode;
        this->use_rk_45 = use_rk_45;
        #define DEFINE_MATRIX(type, name, size_scaling, condition_for_construction) \
            name.constructHost( N_x, N_y * size_scaling, #name); \
            if (condition_for_construction) \
                name.constructDevice( N_x, N_y * size_scaling, #name); 
            //if (not std::is_same<type, Type::cuda_random_state>::value) \
            //    name.fill(type(0.0));
        MATRIX_LIST
        #undef X
     }

    struct Pointers {
        #define DEFINE_MATRIX(type, name, size_scaling, condition_for_construction) \
            type* name;
        MATRIX_LIST
        #undef X
    };

    Pointers pointers() {
        Pointers ptrs;
        #define DEFINE_MATRIX(type, name, size_scaling, condition_for_construction) \
            if (condition_for_construction) \
                ptrs.name = name.getDevicePtr(); \
            else \
                ptrs.name = nullptr;
        MATRIX_LIST
        #undef X
        return ptrs;
    }
};

} // namespace PC3