#pragma once
#include "cuda/typedef.cuh"
#include "kernel_coefficients_dormand_prince.cuh"
#include "solver/matrix_container.hpp"
#include "solver/gpu_solver.hpp"
#include "system/system_parameters.hpp"

namespace PC3::Kernel {
    
    namespace RK45 {

        PULSE_GLOBAL void runge_sum_to_input_of_k2( int i, Type::real dt, MatrixContainer::Pointers dev_ptrs, SystemParameters::KernelParameters p );
        PULSE_GLOBAL void runge_sum_to_input_of_k3( int i, Type::real dt, MatrixContainer::Pointers dev_ptrs, SystemParameters::KernelParameters p );
        PULSE_GLOBAL void runge_sum_to_input_of_k4( int i, Type::real dt, MatrixContainer::Pointers dev_ptrs, SystemParameters::KernelParameters p );
        PULSE_GLOBAL void runge_sum_to_input_of_k5( int i, Type::real dt, MatrixContainer::Pointers dev_ptrs, SystemParameters::KernelParameters p );
        PULSE_GLOBAL void runge_sum_to_input_of_k6( int i, Type::real dt, MatrixContainer::Pointers dev_ptrs, SystemParameters::KernelParameters p );
        PULSE_GLOBAL void runge_sum_to_final( int i, Type::real dt, MatrixContainer::Pointers dev_ptrs, SystemParameters::KernelParameters p );
        PULSE_GLOBAL void runge_sum_final_error( int i, Type::real dt, MatrixContainer::Pointers dev_ptrs, SystemParameters::KernelParameters p );

    } // namespace RK45

    namespace RK4 {

        PULSE_GLOBAL void runge_sum_to_input_k2( int i, Type::real dt, MatrixContainer::Pointers dev_ptrs, SystemParameters::KernelParameters p );
        PULSE_GLOBAL void runge_sum_to_input_k3( int i, Type::real dt, MatrixContainer::Pointers dev_ptrs, SystemParameters::KernelParameters p );
        PULSE_GLOBAL void runge_sum_to_input_k4( int i, Type::real dt, MatrixContainer::Pointers dev_ptrs, SystemParameters::KernelParameters p );
        PULSE_GLOBAL void runge_sum_to_final( int i, Type::real dt, MatrixContainer::Pointers dev_ptrs, SystemParameters::KernelParameters p );

    } // namespace RK4

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

    //PULSE_GLOBAL void normalize_imaginary_time_propagation(int i, MatrixContainer::Pointers dev_ptrs, SystemParameters::KernelParameters p, Type::complex normalization_wavefunction, Type::complex normalization_reservoir);

    PULSE_GLOBAL void initialize_random_number_generator(int i, unsigned int seed, Type::cuda_random_state* state, const unsigned int N);
    PULSE_GLOBAL void generate_random_numbers(int i, Type::cuda_random_state* state, Type::complex* buffer, const unsigned int N, const Type::real real_amp, const Type::real imag_amp);

} // namespace PC3::Kernel