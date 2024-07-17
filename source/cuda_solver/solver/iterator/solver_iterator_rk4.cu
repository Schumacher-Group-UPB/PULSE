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

void PC3::Solver::iterateFixedTimestepRungeKutta4( dim3 block_size, dim3 grid_size ) {
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
        { 0.0, 0.5 } // 0.5*dt*K2
    );

    CALCULATE_K( 3, p.t + 0.5 * p.dt, buffer_wavefunction, buffer_reservoir);

    CALL_KERNEL(
        Kernel::RK::runge_sum_to_input_kw, "Sum for K4", grid_size, block_size,
        p.dt, device_pointers, p, io,
        { 0.0, 0.0, 1.0 } // dt*K3
    );

    CALCULATE_K( 4, p.t + p.dt, buffer_wavefunction, buffer_reservoir);

    CALL_KERNEL(
        Kernel::RK::runge_sum_to_input_kw, "Final Sum", grid_size, block_size,
        p.dt, device_pointers, p, io,
        { 1.0/6.0, 1.0/3.0, 1.0/3.0, 1.0/6.0 } // RK Final Weights
    );

    // Swap the next and current wavefunction buffers. This only swaps the pointers, not the data.
    swapBuffers();
    
    return;
}