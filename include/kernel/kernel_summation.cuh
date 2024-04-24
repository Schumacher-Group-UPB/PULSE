#pragma once
#include "cuda/cuda_complex.cuh"
#include "kernel_coefficients_dormand_prince.cuh"
#include "solver/device_struct.hpp"
#include "system/system.hpp"

namespace PC3::Kernel {
    
    using cr = complex_number* __restrict__;

    namespace RK45 {
        CUDA_GLOBAL void runge_sum_to_input_of_k2( int i, complex_number dt, Device::Pointers dev_ptrs, System::Parameters p, bool use_twin_mode );
        CUDA_GLOBAL void runge_sum_to_input_of_k3( int i, complex_number dt, Device::Pointers dev_ptrs, System::Parameters p, bool use_twin_mode );
        CUDA_GLOBAL void runge_sum_to_input_of_k4( int i, complex_number dt, Device::Pointers dev_ptrs, System::Parameters p, bool use_twin_mode );
        CUDA_GLOBAL void runge_sum_to_input_of_k5( int i, complex_number dt, Device::Pointers dev_ptrs, System::Parameters p, bool use_twin_mode );
        CUDA_GLOBAL void runge_sum_to_input_of_k6( int i, complex_number dt, Device::Pointers dev_ptrs, System::Parameters p, bool use_twin_mode );
        CUDA_GLOBAL void runge_sum_to_final( int i, complex_number dt, Device::Pointers dev_ptrs, System::Parameters p, bool use_twin_mode );
        CUDA_GLOBAL void runge_sum_final_error( int i, complex_number dt, Device::Pointers dev_ptrs, System::Parameters p, bool use_twin_mode );
    } // namespace RK45

    namespace RK4 {
        CUDA_GLOBAL void runge_sum_to_input_k2( int i, complex_number dt, Device::Pointers dev_ptrs, System::Parameters p, bool use_twin_mode );
        CUDA_GLOBAL void runge_sum_to_input_k3( int i, complex_number dt, Device::Pointers dev_ptrs, System::Parameters p, bool use_twin_mode );
        CUDA_GLOBAL void runge_sum_to_input_k4( int i, complex_number dt, Device::Pointers dev_ptrs, System::Parameters p, bool use_twin_mode );
        CUDA_GLOBAL void runge_sum_to_final( int i, complex_number dt, Device::Pointers dev_ptrs, System::Parameters p, bool use_twin_mode );
    } // namespace RK4

} // namespace PC3::Kernel