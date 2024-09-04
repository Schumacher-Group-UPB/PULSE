#include <omp.h>

// Include Cuda Kernel headers
#include "cuda/typedef.cuh"
#include "kernel/kernel_compute.cuh"
#include "kernel/kernel_summation.cuh"
#include "system/system_parameters.hpp"
#include "cuda/cuda_matrix.cuh"
#include "solver/gpu_solver.hpp"
#include "misc/commandline_io.hpp"

/**
 * Simpson's Rule for RK3
 */

void PC3::Solver::iterateFixedTimestepRungeKutta3() {
    SOLVER_SEQUENCE( true /*Capture CUDA Graph*/,

                     CALCULATE_K( 1, wavefunction, reservoir );

                     INTERMEDIATE_SUM_K( 1, 0.5f );

                     CALCULATE_K( 2, buffer_wavefunction, buffer_reservoir );

                     INTERMEDIATE_SUM_K( 1, -1.0f, 2.0f );

                     CALCULATE_K( 3, buffer_wavefunction, buffer_reservoir );

                     FINAL_SUM_K( 1.0f / 6.0f, 4.0f / 6.0f, 1.0f / 6.0f ); )

    return;
}