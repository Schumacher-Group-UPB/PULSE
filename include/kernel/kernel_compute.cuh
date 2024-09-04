#pragma once
#include "cuda/typedef.cuh"
#include "kernel_coefficients_dormand_prince.cuh"
#include "solver/gpu_solver.hpp"

/**
 * Contains the Kernels required for the Runge Kutta Solver
 * These Kernels rely on Solver::KernelArguments and Solver::InputOutput
 * to access the necessary data.
 */

namespace PC3::Kernel {

    namespace Compute {

        PULSE_GLOBAL void gp_tetm( int i, Type::uint32 current_halo, Solver::VKernelArguments time, Solver::KernelArguments args, Solver::InputOutput io );
        PULSE_GLOBAL void gp_scalar( int i, Type::uint32 current_halo, Solver::VKernelArguments time, Solver::KernelArguments args, Solver::InputOutput io );

        // These kernels operate on a full grid and do not require subgrid information.
        // The matrices are initialized with subgridsize 1 and no halo.
        PULSE_GLOBAL void gp_scalar_linear_fourier( int i, Solver::VKernelArguments time, Solver::KernelArguments args, Solver::InputOutput io );
        PULSE_GLOBAL void gp_scalar_nonlinear( int i, Solver::VKernelArguments time, Solver::KernelArguments args, Solver::InputOutput io );
        PULSE_GLOBAL void gp_scalar_independent( int i, Solver::VKernelArguments time, Solver::KernelArguments args, Solver::InputOutput io );
        PULSE_GLOBAL void gp_tetm_linear_fourier( int i, Solver::VKernelArguments time, Solver::KernelArguments args, Solver::InputOutput io );
        PULSE_GLOBAL void gp_tetm_nonlinear( int i, Solver::VKernelArguments time, Solver::KernelArguments args, Solver::InputOutput io );
        PULSE_GLOBAL void gp_tetm_independent( int i, Solver::VKernelArguments time, Solver::KernelArguments args, Solver::InputOutput io );

    } // namespace Compute

    PULSE_GLOBAL void initialize_random_number_generator(int i, Type::uint32 seed, Type::cuda_random_state* state, const Type::uint32 N);
    PULSE_GLOBAL void generate_random_numbers(int i, Type::cuda_random_state* state, Type::complex* buffer, const Type::uint32 N, const Type::real real_amp, const Type::real imag_amp);

} // namespace PC3::Kernel