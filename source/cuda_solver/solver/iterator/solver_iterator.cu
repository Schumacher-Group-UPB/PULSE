#include <omp.h>

// Include Cuda Kernel headers
#include "cuda/typedef.cuh"
#include "kernel/kernel_compute.cuh"
#include "system/system_parameters.hpp"
#include "misc/helperfunctions.hpp"
#include "cuda/cuda_matrix.cuh"
#include "solver/gpu_solver.hpp"
#include "misc/commandline_io.hpp"

/*
* Helper variable for caching the current time for FFT evaluations.
* We dont need this variable anywhere else, so we just create it
* locally to this file here.
*/
PC3::Type::real cached_t = 0.0;
bool first_time = true;

/**
 * Iterates the Runge-Kutta-Method on the GPU
 * Note, that all device arrays and variables have to be initialized at this point
 * @param t Current time, will be updated to t + dt
 * @param dt Time step, will be updated to the next time step
 * @param N_x Number of grid points in one dimension
 * @param N_y Number of grid points in the other dimension
 */
bool PC3::Solver::iterate( ) {

    // First, check if the maximum time has been reached
    if ( system.p.t >= system.t_max )
        return false;

    dim3 block_size( system.block_size, 1 );
    dim3 grid_size( ( system.p.N_x*system.p.N_y + block_size.x ) / block_size.x, 1 );
    
    if (first_time and system.evaluateStochastic()) {
        first_time = false;
        auto device_pointers = matrix.pointers();
        CALL_KERNEL(
                PC3::Kernel::initialize_random_number_generator, "random_number_init", grid_size, block_size,
                system.random_seed, device_pointers.random_state, system.p.N_x*system.p.N_y
            );
        std::cout << PC3::CLIO::prettyPrint( "Initialized Random Number Generator", PC3::CLIO::Control::Info ) << std::endl;
    }
    
    if ( system.iterator == SystemParameters::Iterator::RK4 )
        iterateFixedTimestepRungeKutta( block_size, grid_size );
    else if ( system.iterator == SystemParameters::Iterator::RK45 )
        iterateVariableTimestepRungeKutta( block_size, grid_size );
    else 
        iterateSplitStepFourier( block_size, grid_size );

    // Increase t.
    system.p.t = system.p.t + system.p.dt;

    // For statistical purposes, increase the iteration counter
    system.iteration++;

    // FFT Guard 
    if ( system.p.t - cached_t < system.fft_every )
        return true;

    // Calculate the FFT
    cached_t = system.p.t; 
    applyFFTFilter( block_size, grid_size, system.fft_mask.size() > 0 );

    return true;
}