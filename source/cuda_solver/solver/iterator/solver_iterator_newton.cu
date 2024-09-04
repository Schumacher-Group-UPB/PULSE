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
 * Newton Integration
 * Psi_next = Psi_current + dt * f(Psi_current)
 */

void PC3::Solver::iterateNewton() {
    SOLVER_SEQUENCE( true /*Capture CUDA Graph*/,

                     CALCULATE_K( 1, wavefunction, reservoir );

                     FINAL_SUM_K( 1.0f ); )

    return;
}