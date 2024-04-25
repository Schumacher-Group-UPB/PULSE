#pragma once
#include "cuda/cuda_complex.cuh"
#include "solver/device_struct.hpp"
#include "system/envelope.hpp"
#include "system/system.hpp"
#include "solver/gpu_solver.hpp" // For PulseParameters. TODO: Change

namespace PC3::Kernel {

// Helper Struct to pass input-target data pointers to the kernel
struct InputOutput {
    complex_number* CUDA_RESTRICT in_wf_plus;
    complex_number* CUDA_RESTRICT in_wf_minus;
    complex_number* CUDA_RESTRICT in_rv_plus;
    complex_number* CUDA_RESTRICT in_rv_minus;
    complex_number* CUDA_RESTRICT out_wf_plus;
    complex_number* CUDA_RESTRICT out_wf_minus;
    complex_number* CUDA_RESTRICT out_rv_plus;
    complex_number* CUDA_RESTRICT out_rv_minus;
};

namespace Compute {

CUDA_GLOBAL void gp_tetm( int i, real_number t, Device::Pointers dev_ptrs, System::Parameters p, Solver::Oscillation::Pointers oscillation, InputOutput io );
CUDA_GLOBAL void gp_scalar( int i, real_number t, Device::Pointers dev_ptrs, System::Parameters p, Solver::Oscillation::Pointers oscillation, InputOutput io );

CUDA_GLOBAL void scalar_pulse( int i, real_number t, Device::Pointers dev_ptrs, System::Parameters p, Solver::Oscillation::Pointers oscillation, InputOutput io );
CUDA_GLOBAL void scalar_reservoir( int i, real_number t, Device::Pointers dev_ptrs, System::Parameters p, Solver::Oscillation::Pointers oscillation, InputOutput io );
CUDA_GLOBAL void scalar_stochastic( int i, real_number t, Device::Pointers dev_ptrs, System::Parameters p, InputOutput io );
CUDA_GLOBAL void tetm_pulse( int i, real_number t, Device::Pointers dev_ptrs, System::Parameters p, Solver::Oscillation::Pointers oscillation, InputOutput io );
CUDA_GLOBAL void tetm_reservoir( int i, real_number t, Device::Pointers dev_ptrs, System::Parameters p, Solver::Oscillation::Pointers oscillation, InputOutput io );
CUDA_GLOBAL void tetm_stochastic( int i, real_number t, Device::Pointers dev_ptrs, System::Parameters p, InputOutput io );

} // namespace Compute

} // namespace PC3::Kernel