#pragma once
#include "cuda/cuda_complex.cuh"
#include "solver/device_struct.hpp"
#include "system/envelope.hpp"
#include "system/system.hpp"
#include "solver/gpu_solver.cuh" // For PulseParameters. TODO: Change

namespace PC3::Kernel {

    // Helper Struct to pass input-target data pointers to the kernel
    struct InputOutput {
        complex_number* in_wf_plus;
        complex_number* in_wf_minus;
        complex_number* in_rv_plus;
        complex_number* in_rv_minus;
        complex_number* out_wf_plus;
        complex_number* out_wf_minus;
        complex_number* out_rv_plus;
        complex_number* out_rv_minus;
    };

    CUDA_GLOBAL void runge_func_kernel_tetm( int i, real_number t, Device::Pointers dev_ptrs, System::Parameters p,
                                        Solver::PulseParameters::Pointers pulse, bool evaluate_pulse,
                                        InputOutput inout  );
    
    CUDA_GLOBAL void runge_func_kernel_scalar( int i, real_number t, Device::Pointers dev_ptrs, System::Parameters p,
                                        Solver::PulseParameters::Pointers pulse, bool evaluate_pulse,
                                        InputOutput inout );

    CUDA_DEVICE complex_number kernel_inline_calculate_pulse( const int row, const int col, PC3::Envelope::Polarization polarization, real_number t, const System::Parameters& p, Solver::PulseParameters::Pointers& pulse );

} // namespace PC3::Kernel