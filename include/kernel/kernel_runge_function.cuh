#pragma once
#include "cuda/typedef.cuh"
#include "solver/matrix_container.hpp"
#include "system/envelope.hpp"
#include "system/system_parameters.hpp"
#include "solver/gpu_solver.hpp" // For PulseParameters. TODO: Change

namespace PC3::Kernel {

// Helper Struct to pass input-target data pointers to the kernel
struct InputOutput {
    Type::complex* PULSE_RESTRICT in_wf_plus;
    Type::complex* PULSE_RESTRICT in_wf_minus;
    Type::complex* PULSE_RESTRICT in_rv_plus;
    Type::complex* PULSE_RESTRICT in_rv_minus;
    Type::complex* PULSE_RESTRICT out_wf_plus;
    Type::complex* PULSE_RESTRICT out_wf_minus;
    Type::complex* PULSE_RESTRICT out_rv_plus;
    Type::complex* PULSE_RESTRICT out_rv_minus;
};

namespace Compute {

PULSE_GLOBAL void gp_tetm( int i, Type::real t, MatrixContainer::Pointers dev_ptrs, SystemParameters::KernelParameters p, Solver::Oscillation::Pointers oscillation_pulse, Solver::Oscillation::Pointers oscillation_pump, Solver::Oscillation::Pointers oscillation_potential, InputOutput io );
PULSE_GLOBAL void gp_scalar( int i, Type::real t, MatrixContainer::Pointers dev_ptrs, SystemParameters::KernelParameters p, Solver::Oscillation::Pointers oscillation_pulse, Solver::Oscillation::Pointers oscillation_pump, Solver::Oscillation::Pointers oscillation_potential, InputOutput io );

PULSE_GLOBAL void gp_scalar_linear_fourier( int i, Type::real t, MatrixContainer::Pointers dev_ptrs, SystemParameters::KernelParameters p, Solver::Oscillation::Pointers oscillation_pulse, Solver::Oscillation::Pointers oscillation_pump, Solver::Oscillation::Pointers oscillation_potential, InputOutput io );
PULSE_GLOBAL void gp_scalar_nonlinear( int i, Type::real t, MatrixContainer::Pointers dev_ptrs, SystemParameters::KernelParameters p, Solver::Oscillation::Pointers oscillation_pulse, Solver::Oscillation::Pointers oscillation_pump, Solver::Oscillation::Pointers oscillation_potential, InputOutput io );
PULSE_GLOBAL void gp_scalar_independent( int i, Type::real t, MatrixContainer::Pointers dev_ptrs, SystemParameters::KernelParameters p, Solver::Oscillation::Pointers oscillation_pulse, Solver::Oscillation::Pointers oscillation_pump, Solver::Oscillation::Pointers oscillation_potential, InputOutput io );

} // namespace Compute

} // namespace PC3::Kernel