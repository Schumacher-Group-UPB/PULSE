#pragma once
#include "cuda/cuda_complex.cuh"
#include "kernel_coefficients_dormand_prince.cuh"
#include "solver/device_struct.hpp"
#include "system/system.hpp"

namespace PC3::Kernel {
    
    using cr = complex_number* __restrict__;
    
    /**
     * The summation kernels are used to reduce a given expression of the Runge Kutta
     * Methods to a single value or matrix. Consider this example:
     * 
     * The K3 of the Runge Kutta Dormand Prince method is calculated as 
     * K3 = dt*f( t + 3/10dt, y + 3/40dt*K1 + 9/40dt*K2 )
     * The matrices on the device are in a 1xN^2 array, where N is the matrix
     * dimension. To now calculate f(...), we first have to evaluate
     * y + 3/40dt*K1 + 9/40dt*K2. This is done by the summation kernel for
     * all of the possible Ks. The result for a given Kx is then stored in a 
     * temporary array and used by the Runge Kutta Kernel to calculate Kx.
    */

    namespace RK45 {
        CUDA_GLOBAL void runge_sum_to_input_of_k2( int i, Device::Pointers dev_ptrs, System::Parameters p, bool use_te_tm_splitting );
        CUDA_GLOBAL void runge_sum_to_input_of_k3( int i, Device::Pointers dev_ptrs, System::Parameters p, bool use_te_tm_splitting );
        CUDA_GLOBAL void runge_sum_to_input_of_k4( int i, Device::Pointers dev_ptrs, System::Parameters p, bool use_te_tm_splitting );
        CUDA_GLOBAL void runge_sum_to_input_of_k5( int i, Device::Pointers dev_ptrs, System::Parameters p, bool use_te_tm_splitting );
        CUDA_GLOBAL void runge_sum_to_input_of_k6( int i, Device::Pointers dev_ptrs, System::Parameters p, bool use_te_tm_splitting );
        CUDA_GLOBAL void runge_sum_to_final( int i, Device::Pointers dev_ptrs, System::Parameters p, bool use_te_tm_splitting );
        CUDA_GLOBAL void runge_sum_final_error( int i, Device::Pointers dev_ptrs, System::Parameters p, bool use_te_tm_splitting );
    } // namespace RK45

    namespace RK4 {
        CUDA_GLOBAL void runge_sum_to_input_k2( int i, Device::Pointers dev_ptrs, System::Parameters p, bool use_te_tm_splitting );
        CUDA_GLOBAL void runge_sum_to_input_k3( int i, Device::Pointers dev_ptrs, System::Parameters p, bool use_te_tm_splitting );
        CUDA_GLOBAL void runge_sum_to_input_k4( int i, Device::Pointers dev_ptrs, System::Parameters p, bool use_te_tm_splitting );
        CUDA_GLOBAL void runge_sum_to_final( int i, Device::Pointers dev_ptrs, System::Parameters p, bool use_te_tm_splitting );
    } // namespace RK4

} // namespace PC3::Kernel