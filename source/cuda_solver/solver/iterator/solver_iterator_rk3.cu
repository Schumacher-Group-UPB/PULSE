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
    // This variable contains all the system parameters the kernel could need
    auto p = system.kernel_parameters;
    
    // This variable contains all the device pointers the kernel could need
    auto device_pointers = matrix.pointers();
    // Same IO every time
    Kernel::InputOutput io = { 
        device_pointers.wavefunction_plus, device_pointers.wavefunction_minus, 
        device_pointers.reservoir_plus, device_pointers.reservoir_minus, 
        device_pointers.buffer_wavefunction_plus, device_pointers.buffer_wavefunction_minus, 
        device_pointers.buffer_reservoir_plus, device_pointers.buffer_reservoir_minus 
    };

    // The CPU should briefly evaluate wether the stochastic kernel is used
    bool evaluate_stochastic = system.evaluateStochastic();

    // Pointers to Oscillation Parameters
    auto pulse_pointers = dev_pulse_oscillation.pointers();
    auto pump_pointers = dev_pump_oscillation.pointers();
    auto potential_pointers = dev_potential_oscillation.pointers();

    // The delta time is either real or imaginary, depending on the system configuration
    //Type::complex delta_time = system.imaginary_time ? Type::complex(0.0, -p.dt) : Type::complex(p.dt, 0.0);

    // If required, calculate new set of random numbers.
    if (evaluate_stochastic)
    CALL_KERNEL(
        PC3::Kernel::generate_random_numbers, "random_number_gen", grid_size, block_size,
        device_pointers.random_state, device_pointers.random_number, p.N_x*p.N_y, system.p.stochastic_amplitude*std::sqrt(p.dt), system.p.stochastic_amplitude*std::sqrt(p.dt)
    );

    CALCULATE_K( 1, p.t, wavefunction, reservoir );
 
    CALL_KERNEL(
        Kernel::RK::runge_sum_to_input_kw, "Sum for K4", grid_size, block_size,
        p.dt, device_pointers, p, io,
        { 0.5 } // 0.5*dt*K1
    );

    CALCULATE_K( 2, p.t + 0.5 * p.dt, buffer_wavefunction, buffer_reservoir );

    CALL_KERNEL(
        Kernel::RK::runge_sum_to_input_kw, "Sum for K3", grid_size, block_size,
        p.dt, device_pointers, p, io,
        { -1.0, 2.0 } // -dt*K1 + 2*dt*K2
    );

    CALCULATE_K( 3, p.t + 0.5 * p.dt, buffer_wavefunction, buffer_reservoir);
 
    CALL_KERNEL(
        Kernel::RK::runge_sum_to_input_kw, "Final Sum", grid_size, block_size,
        p.dt, device_pointers, p, io,
        { 1.0/6.0, 4.0/6.0, 1.0/6.0 } // 1/6*dt*K1 + 4/6*dt*K2 + 1/6*dt*K3
    );

    // Swap the next and current wavefunction buffers. This only swaps the pointers, not the data.
    swapBuffers();
    
    return;
}