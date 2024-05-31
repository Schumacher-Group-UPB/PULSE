#pragma once
#include "cuda/typedef.cuh"
#include "kernel_coefficients_dormand_prince.cuh"
#include "solver/matrix_container.hpp"
#include "system/system.hpp"

namespace PC3::Kernel {
    
    namespace RK45 {
        PULSE_GLOBAL void runge_sum_to_input_of_k2( int i, Type::complex dt, MatrixContainer::Pointers dev_ptrs, System::Parameters p );
        PULSE_GLOBAL void runge_sum_to_input_of_k3( int i, Type::complex dt, MatrixContainer::Pointers dev_ptrs, System::Parameters p );
        PULSE_GLOBAL void runge_sum_to_input_of_k4( int i, Type::complex dt, MatrixContainer::Pointers dev_ptrs, System::Parameters p );
        PULSE_GLOBAL void runge_sum_to_input_of_k5( int i, Type::complex dt, MatrixContainer::Pointers dev_ptrs, System::Parameters p );
        PULSE_GLOBAL void runge_sum_to_input_of_k6( int i, Type::complex dt, MatrixContainer::Pointers dev_ptrs, System::Parameters p );
        PULSE_GLOBAL void runge_sum_to_final( int i, Type::complex dt, MatrixContainer::Pointers dev_ptrs, System::Parameters p );
        PULSE_GLOBAL void runge_sum_final_error( int i, Type::complex dt, MatrixContainer::Pointers dev_ptrs, System::Parameters p );
    } // namespace RK45

    namespace RK4 {
        PULSE_GLOBAL void runge_sum_to_input_k2( int i, Type::complex dt, MatrixContainer::Pointers dev_ptrs, System::Parameters p );
        PULSE_GLOBAL void runge_sum_to_input_k3( int i, Type::complex dt, MatrixContainer::Pointers dev_ptrs, System::Parameters p );
        PULSE_GLOBAL void runge_sum_to_input_k4( int i, Type::complex dt, MatrixContainer::Pointers dev_ptrs, System::Parameters p );
        PULSE_GLOBAL void runge_sum_to_final( int i, Type::complex dt, MatrixContainer::Pointers dev_ptrs, System::Parameters p );
    } // namespace RK4

} // namespace PC3::Kernel