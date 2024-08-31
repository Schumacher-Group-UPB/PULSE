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

    namespace RK {
        // Weights for up to 4 ks
        struct Weights { 
            int n;
            int start;
            Type::real weights[MAX_K_VECTOR_SIZE];
            template <typename ...Args>
            Weights( Args... _weights ) : n(sizeof...(_weights)), start(-1) {
                double _w[] = { _weights... };
                for (int i = 0; i < n; i++) {
                    // If the weight zero, we skip this weight, until we find a valid one
                    if (_w[i] != 0.0 and start < 0 ) 
                        start = i;
                    // Always assign the weight, even if it is invalid
                    weights[i] = Type::real(_w[i]);
                }
            }
        };
        PULSE_GLOBAL void runge_sum_to_input_kw( int i, Type::uint current_halo, Solver::VKernelArguments time, Solver::KernelArguments args, Solver::InputOutput io, RK::Weights weights );
        PULSE_GLOBAL void runge_sum_to_error( int i, Type::uint current_halo, Solver::VKernelArguments time, Solver::KernelArguments args, Solver::InputOutput io, RK::Weights weights );
    } // namespace RK

    namespace Compute {

        PULSE_GLOBAL void gp_tetm( int i, Type::uint current_halo, Solver::VKernelArguments time, Solver::KernelArguments args, Solver::InputOutput io );
        PULSE_GLOBAL void gp_scalar( int i, Type::uint current_halo, Solver::VKernelArguments time, Solver::KernelArguments args, Solver::InputOutput io );

        // These kernels operate on a full grid and do not require subgrid information.
        // The matrices are initialized with subgridsize 1 and no halo.
        PULSE_GLOBAL void gp_scalar_linear_fourier( int i, Solver::VKernelArguments time, Solver::KernelArguments args, Solver::InputOutput io );
        PULSE_GLOBAL void gp_scalar_nonlinear( int i, Solver::VKernelArguments time, Solver::KernelArguments args, Solver::InputOutput io );
        PULSE_GLOBAL void gp_scalar_independent( int i, Solver::VKernelArguments time, Solver::KernelArguments args, Solver::InputOutput io );
        PULSE_GLOBAL void gp_tetm_linear_fourier( int i, Solver::VKernelArguments time, Solver::KernelArguments args, Solver::InputOutput io );
        PULSE_GLOBAL void gp_tetm_nonlinear( int i, Solver::VKernelArguments time, Solver::KernelArguments args, Solver::InputOutput io );
        PULSE_GLOBAL void gp_tetm_independent( int i, Solver::VKernelArguments time, Solver::KernelArguments args, Solver::InputOutput io );

    } // namespace Compute

    PULSE_GLOBAL void initialize_random_number_generator(int i, Type::uint seed, Type::cuda_random_state* state, const Type::uint N);
    PULSE_GLOBAL void generate_random_numbers(int i, Type::cuda_random_state* state, Type::complex* buffer, const Type::uint N, const Type::real real_amp, const Type::real imag_amp);

} // namespace PC3::Kernel