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
 * Split Step Fourier Method
 */
void PC3::Solver::iterateSplitStepFourier( dim3 block_size, dim3 grid_size ) {
    
    auto p = system.kernel_parameters;
    
    // This variable contains all the device pointers the kernel could need
    auto device_pointers = matrix.pointers();

    // Pointers to Oscillation Parameters
    auto pulse_pointers = dev_pulse_oscillation.pointers();
    auto pump_pointers = dev_pump_oscillation.pointers();
    auto potential_pointers = dev_potential_oscillation.pointers();

    // Liner Half Step
    // Calculate the FFT of Psi
    calculateFFT( device_pointers.wavefunction_plus, device_pointers.k1_wavefunction_plus, FFT::forward );
    CALL_KERNEL(
        Kernel::Compute::gp_scalar_linear_fourier, "linear_half_step", grid_size, block_size, 
        p.t, device_pointers, p, pulse_pointers, pump_pointers, potential_pointers,
        { 
            device_pointers.k1_wavefunction_plus, device_pointers.k1_wavefunction_minus, device_pointers.discard, device_pointers.discard,
            device_pointers.k2_wavefunction_plus, device_pointers.k2_wavefunction_minus, device_pointers.discard, device_pointers.discard
        }
    );
    // Transform back. K1 now holds the half-stepped wavefunction.
    calculateFFT( device_pointers.k2_wavefunction_plus, device_pointers.k1_wavefunction_plus, FFT::inverse );

    // Nonlinear Full Step
    CALL_KERNEL(
        Kernel::Compute::gp_scalar_nonlinear, "nonlinear_full_step", grid_size, block_size, 
        p.t, device_pointers, p, pulse_pointers, pump_pointers, potential_pointers,
        { 
            device_pointers.k1_wavefunction_plus, device_pointers.k1_wavefunction_minus, device_pointers.reservoir_plus, device_pointers.reservoir_minus,
            device_pointers.k2_wavefunction_plus, device_pointers.k2_wavefunction_minus, device_pointers.buffer_reservoir_plus, device_pointers.buffer_reservoir_minus
        }
    );
    // K2 now holds the nonlinearly evolved wavefunction.

    // Liner Half Step 
    // Calculate the FFT of Psi
    calculateFFT( device_pointers.k2_wavefunction_plus, device_pointers.k1_wavefunction_plus, FFT::forward );
    CALL_KERNEL(
        Kernel::Compute::gp_scalar_linear_fourier, "linear_half_step", grid_size, block_size, 
        p.t, device_pointers, p, pulse_pointers, pump_pointers, potential_pointers,
        { 
            device_pointers.k1_wavefunction_plus, device_pointers.k1_wavefunction_minus, device_pointers.discard, device_pointers.discard,
            device_pointers.k2_wavefunction_plus, device_pointers.k2_wavefunction_minus, device_pointers.discard, device_pointers.discard
        }
    );
    // Transform back. K3 now holds the half-stepped wavefunction.
    calculateFFT( device_pointers.k2_wavefunction_plus,  device_pointers.k1_wavefunction_plus, FFT::inverse );

    CALL_KERNEL(
        Kernel::Compute::gp_scalar_independent, "independent", grid_size, block_size, 
        p.t, device_pointers, p, pulse_pointers, pump_pointers, potential_pointers,
        { 
            device_pointers.k1_wavefunction_plus, device_pointers.k1_wavefunction_minus, device_pointers.reservoir_plus, device_pointers.reservoir_minus,
            device_pointers.buffer_wavefunction_plus, device_pointers.buffer_wavefunction_minus, device_pointers.buffer_reservoir_plus, device_pointers.buffer_reservoir_minus
        }
    );
    // Buffer now holds the new result

    // Swap the next and current wavefunction buffers. This only swaps the pointers, not the data.
    swapBuffers();

}