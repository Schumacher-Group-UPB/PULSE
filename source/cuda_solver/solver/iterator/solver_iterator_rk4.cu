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

void PC3::Solver::iterateFixedTimestepRungeKutta4() {
#ifndef __I_LIKE_DUCKS

    SOLVER_SEQUENCE( true /*Capture CUDA Graph*/,

                     CALCULATE_K( 1, wavefunction, reservoir );

                     INTERMEDIATE_SUM_K( 1, 0.5f );

                     CALCULATE_K( 2, buffer_wavefunction, buffer_reservoir );

                     INTERMEDIATE_SUM_K( 2, 0.0f, 0.5f );

                     CALCULATE_K( 3, buffer_wavefunction, buffer_reservoir );

                     INTERMEDIATE_SUM_K( 3, 0.0f, 0.0f, 1.0f );

                     CALCULATE_K( 4, buffer_wavefunction, buffer_reservoir );

                     FINAL_SUM_K( 4, 1.01f / 6.0f, 1.0f / 3.0f, 1.0f / 3.0f, 1.0f / 6.0f );

    );

#else

    // CPU Specific implementation using SIMD thingy

    // Precalculate the kernel arguments for each subgrid
    static bool first_time = false;
    static std::vector<Solver::KernelArguments> v_kernel_arguments;
    if ( not first_time ) {
        for ( Type::uint32 subgrid = 0; subgrid < system.p.subgrids_x * system.p.subgrids_y; subgrid++ ) {
            v_kernel_arguments.push_back( generateKernelArguments( subgrid ) );
        }
        first_time = true;
    }

    // Synchronize halos
    // WHAT I TESTED:
    // - Move the kernel to here in raw code; does not change the runtime, at least on my machine.
    {
        auto halo_map_size = matrix.halo_map.size() / 6;
        auto [current_block, current_grid] = getLaunchParameters( halo_map_size * system.p.subgrids_x * system.p.subgrids_y );
        {
    #pragma omp parallel for schedule( static )
            for ( Type::uint32 i = 0; i < current_block.x; ++i ) {
                for ( Type::uint32 j = 0; j < current_grid.x; ++j ) {
                    const Type::uint32 index = i * current_grid.x + j;
                    Kernel::Halo::synchronize_halos( index, system.p.subgrids_x, system.p.subgrids_y, system.p.subgrid_N_x, system.p.subgrid_N_y, system.p.halo_size, halo_map_size,
                                                     system.p.periodic_boundary_x, system.p.periodic_boundary_y, matrix.halo_map.data(),
                                                     matrix.wavefunction_plus.getSubgridDevicePtrs() );
                }
            }
        }
        {
    #pragma omp parallel for schedule( static )
            for ( Type::uint32 i = 0; i < current_block.x; ++i ) {
                for ( Type::uint32 j = 0; j < current_grid.x; ++j ) {
                    const Type::uint32 index = i * current_grid.x + j;
                    Kernel::Halo::synchronize_halos( index, system.p.subgrids_x, system.p.subgrids_y, system.p.subgrid_N_x, system.p.subgrid_N_y, system.p.halo_size, halo_map_size,
                                                     system.p.periodic_boundary_x, system.p.periodic_boundary_y, matrix.halo_map.data(),
                                                     matrix.reservoir_plus.getSubgridDevicePtrs() );
                }
            }
        }
    }

    void ( *rf )( int, Type::uint32, Solver::VKernelArguments, Solver::KernelArguments, Solver::InputOutput ) =
        ( system.p.use_twin_mode ? PC3::Kernel::Compute::gp_tetm : PC3::Kernel::Compute::gp_scalar );

    // Main Loop
    const Type::uint32 subgrid_threads = system.p.subgrids_x * system.p.subgrids_y > 1 ? system.omp_max_threads : 1;
    #pragma omp parallel for schedule( static ) num_threads( subgrid_threads )
    for ( Type::uint32 subgrid = 0; subgrid < system.p.subgrids_x * system.p.subgrids_y; subgrid++ ) {
        auto& kernel_arguments = v_kernel_arguments[subgrid];
        {
            const Type::uint32 current_halo = system.p.halo_size - 1;
            auto [current_block, current_grid] = getLaunchParameters( system.p.subgrid_N_x + 2 * current_halo, system.p.subgrid_N_y + 2 * current_halo );
            // K1
            Solver::InputOutput io{ matrix.wavefunction_plus.getDevicePtr( subgrid ),
                                    matrix.wavefunction_minus.getDevicePtr( subgrid ),
                                    matrix.reservoir_plus.getDevicePtr( subgrid ),
                                    matrix.reservoir_minus.getDevicePtr( subgrid ),
                                    matrix.k_wavefunction_plus.getDevicePtr( subgrid, 1 - 1 ),
                                    matrix.k_wavefunction_minus.getDevicePtr( subgrid, 1 - 1 ),
                                    matrix.k_reservoir_plus.getDevicePtr( subgrid, 1 - 1 ),
                                    matrix.k_reservoir_minus.getDevicePtr( subgrid, 1 - 1 ) };
            auto t = getCurrentTime();
            auto execution_range = ( system.p.subgrid_N_x + 2 * current_halo ) * ( system.p.subgrid_N_y + 2 * current_halo );
            for ( Type::uint32 i = 0; i < execution_range; ++i ) 
                    rf( i, current_halo, t, kernel_arguments, io );
            
            // Argument for K2
            Solver::InputOutput io_sum{ matrix.wavefunction_plus.getDevicePtr( subgrid ),        matrix.wavefunction_minus.getDevicePtr( subgrid ),
                                        matrix.reservoir_plus.getDevicePtr( subgrid ),           matrix.reservoir_minus.getDevicePtr( subgrid ),
                                        matrix.buffer_wavefunction_plus.getDevicePtr( subgrid ), matrix.buffer_wavefunction_minus.getDevicePtr( subgrid ),
                                        matrix.buffer_reservoir_plus.getDevicePtr( subgrid ),    matrix.buffer_reservoir_minus.getDevicePtr( subgrid ) };
            Type::complex* PULSE_RESTRICT in_wf_plus = io_sum.in_wf_plus;
            Type::complex* PULSE_RESTRICT in_rv_plus = io_sum.in_rv_plus;
            Type::complex* PULSE_RESTRICT out_wf_plus = io_sum.out_wf_plus;
            Type::complex* PULSE_RESTRICT out_rv_plus = io_sum.out_rv_plus;
            Type::complex* PULSE_RESTRICT k_wavefunction_plus = kernel_arguments.dev_ptrs.k_wavefunction_plus;
            Type::complex* PULSE_RESTRICT k_reservoir_plus = kernel_arguments.dev_ptrs.k_reservoir_plus;
                                        #pragma omp simd
            for ( Type::uint32 i = 0; i < execution_range; ++i ) 
                out_wf_plus[i] = in_wf_plus[i] + 0.5f * system.p.dt * k_wavefunction_plus[i];
            #pragma omp simd
                for ( Type::uint32 i = 0; i < execution_range; ++i ) 
                out_rv_plus[i] = in_rv_plus[i] + 0.5f * system.p.dt * k_reservoir_plus[i];
            
        };
        {
            const Type::uint32 current_halo = system.p.halo_size - 2;
            auto [current_block, current_grid] = getLaunchParameters( system.p.subgrid_N_x + 2 * current_halo, system.p.subgrid_N_y + 2 * current_halo );
            // K2
            Solver::InputOutput io{ matrix.buffer_wavefunction_plus.getDevicePtr( subgrid ),   matrix.buffer_wavefunction_minus.getDevicePtr( subgrid ),
                                    matrix.buffer_reservoir_plus.getDevicePtr( subgrid ),      matrix.buffer_reservoir_minus.getDevicePtr( subgrid ),
                                    matrix.k_wavefunction_plus.getDevicePtr( subgrid, 2 - 1 ), matrix.k_wavefunction_minus.getDevicePtr( subgrid, 2 - 1 ),
                                    matrix.k_reservoir_plus.getDevicePtr( subgrid, 2 - 1 ),    matrix.k_reservoir_minus.getDevicePtr( subgrid, 2 - 1 ) };
            auto t = getCurrentTime();
            auto execution_range = ( system.p.subgrid_N_x + 2 * current_halo ) * ( system.p.subgrid_N_y + 2 * current_halo );
            for ( Type::uint32 i = 0; i < execution_range; ++i ) 
                    rf( i, current_halo, t, kernel_arguments, io );
            
            // Argument for K3
            Solver::InputOutput io_sum{ matrix.wavefunction_plus.getDevicePtr( subgrid ),        matrix.wavefunction_minus.getDevicePtr( subgrid ),
                                        matrix.reservoir_plus.getDevicePtr( subgrid ),           matrix.reservoir_minus.getDevicePtr( subgrid ),
                                        matrix.buffer_wavefunction_plus.getDevicePtr( subgrid ), matrix.buffer_wavefunction_minus.getDevicePtr( subgrid ),
                                        matrix.buffer_reservoir_plus.getDevicePtr( subgrid ),    matrix.buffer_reservoir_minus.getDevicePtr( subgrid ) };
            auto offset = system.p.subgrid_N2_with_halo;
            Type::complex* PULSE_RESTRICT in_wf_plus = io_sum.in_wf_plus;
            Type::complex* PULSE_RESTRICT in_rv_plus = io_sum.in_rv_plus;
            Type::complex* PULSE_RESTRICT out_wf_plus = io_sum.out_wf_plus;
            Type::complex* PULSE_RESTRICT out_rv_plus = io_sum.out_rv_plus;
            Type::complex* PULSE_RESTRICT k_wavefunction_plus = kernel_arguments.dev_ptrs.k_wavefunction_plus;
            Type::complex* PULSE_RESTRICT k_reservoir_plus = kernel_arguments.dev_ptrs.k_reservoir_plus;
            #pragma omp simd
            for ( Type::uint32 i = 0; i < execution_range; ++i ) 
            out_wf_plus[i] = in_wf_plus[i] + 0.5f * system.p.dt * k_wavefunction_plus[i + offset];
            #pragma omp simd
            for ( Type::uint32 i = 0; i < execution_range; ++i ) 
                out_rv_plus[i] = in_rv_plus[i] + 0.5f * system.p.dt * k_reservoir_plus[i + offset];
            
        };
        {
            const Type::uint32 current_halo = system.p.halo_size - 3;
            auto [current_block, current_grid] = getLaunchParameters( system.p.subgrid_N_x + 2 * current_halo, system.p.subgrid_N_y + 2 * current_halo );
            Solver::InputOutput io{ matrix.buffer_wavefunction_plus.getDevicePtr( subgrid ),   matrix.buffer_wavefunction_minus.getDevicePtr( subgrid ),
                                    matrix.buffer_reservoir_plus.getDevicePtr( subgrid ),      matrix.buffer_reservoir_minus.getDevicePtr( subgrid ),
                                    matrix.k_wavefunction_plus.getDevicePtr( subgrid, 3 - 1 ), matrix.k_wavefunction_minus.getDevicePtr( subgrid, 3 - 1 ),
                                    matrix.k_reservoir_plus.getDevicePtr( subgrid, 3 - 1 ),    matrix.k_reservoir_minus.getDevicePtr( subgrid, 3 - 1 ) };
            auto t = getCurrentTime();
            auto execution_range = ( system.p.subgrid_N_x + 2 * current_halo ) * ( system.p.subgrid_N_y + 2 * current_halo );
            for ( Type::uint32 i = 0; i < execution_range; ++i ) 
                    rf( i, current_halo, t, kernel_arguments, io );
            
            Solver::InputOutput io_sum{ matrix.wavefunction_plus.getDevicePtr( subgrid ),        matrix.wavefunction_minus.getDevicePtr( subgrid ),
                                        matrix.reservoir_plus.getDevicePtr( subgrid ),           matrix.reservoir_minus.getDevicePtr( subgrid ),
                                        matrix.buffer_wavefunction_plus.getDevicePtr( subgrid ), matrix.buffer_wavefunction_minus.getDevicePtr( subgrid ),
                                        matrix.buffer_reservoir_plus.getDevicePtr( subgrid ),    matrix.buffer_reservoir_minus.getDevicePtr( subgrid ) };
            auto offset = system.p.subgrid_N2_with_halo * 2;
            Type::complex* PULSE_RESTRICT in_wf_plus = io_sum.in_wf_plus;
            Type::complex* PULSE_RESTRICT in_rv_plus = io_sum.in_rv_plus;
            Type::complex* PULSE_RESTRICT out_wf_plus = io_sum.out_wf_plus;
            Type::complex* PULSE_RESTRICT out_rv_plus = io_sum.out_rv_plus;
            Type::complex* PULSE_RESTRICT k_wavefunction_plus = kernel_arguments.dev_ptrs.k_wavefunction_plus;
            Type::complex* PULSE_RESTRICT k_reservoir_plus = kernel_arguments.dev_ptrs.k_reservoir_plus;
            #pragma omp simd
            for ( Type::uint32 i = 0; i < execution_range; ++i ) 
            out_wf_plus[i] = in_wf_plus[i] + 0.5f * system.p.dt * k_wavefunction_plus[i + offset];
            #pragma omp simd
            for ( Type::uint32 i = 0; i < execution_range; ++i ) 
                out_rv_plus[i] = in_rv_plus[i] + 0.5f * system.p.dt * k_reservoir_plus[i + offset];
            
        };
        {
            const Type::uint32 current_halo = system.p.halo_size - 4;
            auto [current_block, current_grid] = getLaunchParameters( system.p.subgrid_N_x + 2 * current_halo, system.p.subgrid_N_y + 2 * current_halo );
            Solver::InputOutput io{ matrix.buffer_wavefunction_plus.getDevicePtr( subgrid ),   matrix.buffer_wavefunction_minus.getDevicePtr( subgrid ),
                                    matrix.buffer_reservoir_plus.getDevicePtr( subgrid ),      matrix.buffer_reservoir_minus.getDevicePtr( subgrid ),
                                    matrix.k_wavefunction_plus.getDevicePtr( subgrid, 4 - 1 ), matrix.k_wavefunction_minus.getDevicePtr( subgrid, 4 - 1 ),
                                    matrix.k_reservoir_plus.getDevicePtr( subgrid, 4 - 1 ),    matrix.k_reservoir_minus.getDevicePtr( subgrid, 4 - 1 ) };
            auto t = getCurrentTime();
            auto execution_range = ( system.p.subgrid_N_x + 2 * current_halo ) * ( system.p.subgrid_N_y + 2 * current_halo );
            for ( Type::uint32 i = 0; i < execution_range; ++i ) 
                    rf( i, current_halo, t, kernel_arguments, io );
            
        };
        {
            Solver::InputOutput io{ matrix.wavefunction_plus.getDevicePtr( subgrid ), matrix.wavefunction_minus.getDevicePtr( subgrid ),
                                    matrix.reservoir_plus.getDevicePtr( subgrid ),    matrix.reservoir_minus.getDevicePtr( subgrid ),
                                    matrix.wavefunction_plus.getDevicePtr( subgrid ), matrix.wavefunction_minus.getDevicePtr( subgrid ),
                                    matrix.reservoir_plus.getDevicePtr( subgrid ),    matrix.reservoir_minus.getDevicePtr( subgrid ) };
            auto t = getCurrentTime();
            auto [current_block, current_grid] = getLaunchParameters( system.p.subgrid_N_x, system.p.subgrid_N_y );
            auto offset_k2 = system.p.subgrid_N2_with_halo;
            auto offset_k3 = system.p.subgrid_N2_with_halo * 2;
            auto offset_k4 = system.p.subgrid_N2_with_halo * 3;
            auto execution_range = system.p.subgrid_N_x * system.p.subgrid_N_y;
            Type::complex* PULSE_RESTRICT k_wavefunction_plus = kernel_arguments.dev_ptrs.k_wavefunction_plus;
            Type::complex* PULSE_RESTRICT k_reservoir_plus = kernel_arguments.dev_ptrs.k_reservoir_plus;
            #pragma omp simd
            for ( Type::uint32 i = 0; i < execution_range; ++i ) {
                io.out_wf_plus[i] = io.in_wf_plus[i] + system.p.dt * ( 1.01f / 6.0f * k_wavefunction_plus[i] +
                                                                       1.0f / 3.0f * k_wavefunction_plus[i + offset_k2] +
                                                                       1.0f / 3.0f * k_wavefunction_plus[i + offset_k3] +
                                                                       1.0f / 6.0f * k_wavefunction_plus[i + offset_k4] );
            }
            #pragma omp simd
            for ( Type::uint32 i = 0; i < execution_range; ++i ) {
                io.out_rv_plus[i] = io.in_rv_plus[i] + system.p.dt * ( 1.01f / 6.0f * k_reservoir_plus[i] +
                                                                       1.0f / 3.0f * k_reservoir_plus[i + offset_k2] +
                                                                       1.0f / 3.0f * k_reservoir_plus[i + offset_k3] +
                                                                       1.0f / 6.0f * k_reservoir_plus[i + offset_k4] );
            }
        };
    }

#endif

    return;
}