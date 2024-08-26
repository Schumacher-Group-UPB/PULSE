#include <omp.h>

// Include Cuda Kernel headers
#include "cuda/typedef.cuh"
#include "kernel/kernel_compute.cuh"
#include "system/system_parameters.hpp"
#include "misc/helperfunctions.hpp"
#include "cuda/cuda_matrix.cuh"
#include "solver/gpu_solver.hpp"
#include "misc/commandline_io.hpp"

/**
 * Simpson's Rule for RK3
 */

void PC3::Solver::iterateFixedTimestepRungeKutta3( dim3 block_size, dim3 grid_size ) {
    /*
    Type::complex dt = system.imag_time_amplitude != 0.0 ? Type::complex(0.0, -system.p.dt) : Type::complex(system.p.dt, 0.0);

    updateKernelArguments( system.p.t, dt );

    MERGED_CALL(

        CALCULATE_K( 1, wavefunction, reservoir );
    
        INTERMEDIATE_SUM_K( 1, 0.5 );

        CALCULATE_K( 2, buffer_wavefunction, buffer_reservoir );

        INTERMEDIATE_SUM_K( 1, -1.0, 2.0 );

        CALCULATE_K( 3, buffer_wavefunction, buffer_reservoir);
    
        FINAL_SUM_K( 1.0/6.0, 4.0/6.0, 1.0/6.0 );
    )
    */
    // Swap the next and current wavefunction buffers. This only swaps the pointers, not the data.
    //swapBuffers();
    
    return;
}