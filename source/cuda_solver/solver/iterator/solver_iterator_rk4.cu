#include <omp.h>

// Include Cuda Kernel headers
#include "cuda/typedef.cuh"
#include "kernel/kernel_compute.cuh"
#include "kernel/kernel_summation.cuh"
#include "kernel/kernel_halo.cuh"
#include "system/system_parameters.hpp"
#include "cuda/cuda_matrix.cuh"
#include "solver/gpu_solver.hpp"
#include "misc/commandline_io.hpp"

/*
 * This function iterates the Runge Kutta Kernel using a fixed time step.
 * A 4th order Runge-Kutta method is used. This function calls a single
 * rungeFuncSum function with varying delta-t. Calculation of the inputs
 * for the next rungeFuncKernel call is done in the rungeFuncSum function.
 * The general implementation of the RK4 method goes as follows:
 * ------------------------------------------------------------------------------
 * k1 = f(t, y) = rungeFuncKernel(current)
 * input_for_k2 = current + 0.5 * dt * k1
 * k2 = f(t + 0.5 * dt, input_for_k2) = rungeFuncKernel(input_for_k2)
 * input_for_k3 = current + 0.5 * dt * k2
 * k3 = f(t + 0.5 * dt, input_for_k3) = rungeFuncKernel(input_for_k3)
 * input_for_k4 = current + dt * k3
 * k4 = f(t + dt, input_for_k4) = rungeFuncKernel(input_for_k4) 
 * next = current + dt * (1/6 * k1 + 1/3 * k2 + 1/3 * k3 + 1/6 * k4)
 * ------------------------------------------------------------------------------ 
 * The Runge method iterates psi,k1-k4 to psi_next using a wave-like approach.
 * We calculate 4 rows of k1, 3 rows of k2, 2 rows of k3 and 1 row of k4 before the first iteration.
 * Then, we iterate all of the remaining rows after each other, incrementing the buffer for the next iteration.
 */

//#define __I_LIKE_DUCKS

#include <chrono> 

void PC3::Solver::iterateFixedTimestepRungeKutta4() {  

#ifndef __I_LIKE_DUCKS 

    SOLVER_SEQUENCE( true /*Capture CUDA Graph*/,  
  
                     CALCULATE_K( 1, wavefunction, reservoir );  

                     INTERMEDIATE_SUM_K( 1, 0.5f );

                     CALCULATE_K( 2, buffer_wavefunction, buffer_reservoir );

                     INTERMEDIATE_SUM_K( 2, 0.5f );

                     CALCULATE_K( 3, buffer_wavefunction, buffer_reservoir );

                     INTERMEDIATE_SUM_K( 3, 1.0f );

                     CALCULATE_K( 4, buffer_wavefunction, buffer_reservoir );  

                     FINAL_SUM_K( 4, 1.01f / 6.0f, 1.0f / 3.0f, 1.0f / 3.0f, 1.0f / 6.0f );

    );

#else

    PC3::Type::stream_t stream;
    static bool first_time = false;
    static std::vector<Solver::KernelArguments> v_kernel_arguments;
    if ( not first_time ) {
        for ( Type::uint32 subgrid = 0; subgrid < system.p.subgrids_columns * system.p.subgrids_rows; subgrid++ ) {
            v_kernel_arguments.push_back( generateKernelArguments( subgrid ) );
        }
        first_time = true;
    }

    constexpr Type::uint32 index = 1;
    //#pragma omp parallel for schedule( static )
    for ( Type::uint32 subgrid = 0; subgrid < system.p.subgrids_columns * system.p.subgrids_rows; subgrid++ ) {
        PULSE_NUMA_INSERT;
        auto &kernel_arguments = v_kernel_arguments[subgrid];

        Type::uint32 current_halo = system.p.halo_size - index;
        auto [current_block, current_grid] = getLaunchParameters( system.p.subgrid_N_c + 2 * current_halo, system.p.subgrid_N_r + 2 * current_halo );
        Solver::InputOutput io{ matrix.wavefunction_plus.getDevicePtr( subgrid ),        matrix.wavefunction_minus.getDevicePtr( subgrid ),
                                matrix.reservoir_plus.getDevicePtr( subgrid ),           matrix.reservoir_minus.getDevicePtr( subgrid ),
                                matrix.buffer_wavefunction_plus.getDevicePtr( subgrid ), matrix.buffer_wavefunction_minus.getDevicePtr( subgrid ),
                                matrix.buffer_reservoir_plus.getDevicePtr( subgrid ),    matrix.buffer_reservoir_minus.getDevicePtr( subgrid ) };
        Type::complex *k_vec_wf_plus = matrix.k_wavefunction_plus.getDevicePtr( subgrid );
        Type::uint32 execution_range = current_block.x * current_grid.x;
        const Type::real dt = CUDA::real(kernel_arguments.time[1])*0.5f;
        const Type::complex dtc = kernel_arguments.time[1]*0.5f;
            // Test 1: Vectorized loop with real dt
            auto start = std::chrono::high_resolution_clock::now();

            for (int m = 0; m < 10000; m++) {
        #pragma omp simd
        for ( Type::uint32 i = 0; i < execution_range; ++i ) {
            //io.out_wf_plus[i] = io.in_wf_plus[i] + dt*0.1f * k_vec_wf_plus[i] + dt*0.25f * k_vec_wf_plus[i + kernel_arguments.p.subgrid_N2_with_halo] + dt*0.01f * k_vec_wf_plus[i + 2 * kernel_arguments.p.subgrid_N2_with_halo] + dt*0.1337f * k_vec_wf_plus[i + 3 * kernel_arguments.p.subgrid_N2_with_halo];
            PC3::Kernel::Compute::gp_scalar< false, false, false, false, false, false >(i, current_halo, kernel_arguments, io );
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;
    std::cout << "Elapsed time: " << elapsed_seconds.count() << "s\n";
    CALL_SUBGRID_KERNEL( PC3::Kernel::Compute::gp_scalar<GCC_EXPAND_VA_ARGS( false, false, false, false, false, false )>, "K", current_grid, current_block, 0, current_halo, kernel_arguments, io );                                                               
    }

#endif
}