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
PC3::Type::real fft_cached_t = 0.0;
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

    /*    
    // If required, calculate new set of random numbers.
    if (system.evaluateStochastic()) {
        auto device_pointers = matrix.pointers();
        if (first_time) {
            first_time = false;
            CALL_KERNEL(
                    PC3::Kernel::initialize_random_number_generator, "random_number_init", grid_size, block_size, 0,
                    system.random_seed, device_pointers.random_state, system.p.N_x*system.p.N_y
                );
            std::cout << PC3::CLIO::prettyPrint( "Initialized Random Number Generator", PC3::CLIO::Control::Info ) << std::endl;
        }
        CALL_KERNEL(
            PC3::Kernel::generate_random_numbers, "random_number_gen", grid_size, block_size, 0,
            device_pointers.random_state, device_pointers.random_number, system.p.N_x*system.p.N_y, system.p.stochastic_amplitude*std::sqrt(system.p.dt), system.p.stochastic_amplitude*std::sqrt(system.p.dt)
        );
    }
    */

    // TODO: Merhe these device arrays with the kernelParameters struct.
    // should be easily possible because the sizes of the arrays are known at launch
    // which means we can allocate the memory in the kernelParameters struct
    // Update the temporal envelopes
    system.pulse.updateTemporal( system.p.t );
    system.potential.updateTemporal( system.p.t );
    system.pump.updateTemporal( system.p.t );
    // And update the solver struct accordingly
    dev_pulse_oscillation.amp = system.pulse.temporal_envelope;
    dev_potential_oscillation.amp = system.potential.temporal_envelope;
    dev_pump_oscillation.amp = system.pump.temporal_envelope;
    
    // Iterate RK4(45)/ssfm/itp
    iterator[system.iterator].iterate( block_size, grid_size );

    // Call the normalization for imaginary time propagation if required
    if (system.imag_time_amplitude != 0.0) 
        normalizeImaginaryTimePropagation( block_size, grid_size );

    // Increase t. 
    system.p.t = system.p.t + system.p.dt;
    
    // For statistical purposes, increase the iteration counter
    system.iteration++;

    // FFT Guard 
    if ( system.p.t - fft_cached_t < system.fft_every )
        return true;

    // Calculate the FFT
    fft_cached_t = system.p.t; 
    applyFFTFilter( block_size, grid_size, system.fft_mask.size() > 0 );

    return true;
}