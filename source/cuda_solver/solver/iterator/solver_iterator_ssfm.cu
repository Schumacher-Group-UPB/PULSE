#include <omp.h>

// Include Cuda Kernel headers
#include "cuda/typedef.cuh"
#include "kernel/kernel_compute.cuh"
#include "system/system_parameters.hpp"
#include "cuda/cuda_matrix.cuh"
#include "solver/gpu_solver.hpp"
#include "misc/commandline_io.hpp"

/**
 * Split Step Fourier Method
 */
void PC3::Solver::iterateSplitStepFourier(  ) {
    
    auto kernel_arguments = generateKernelArguments( );
    Solver::VKernelArguments v_time = getCurrentTime(); 
    auto [block_size, grid_size] = getLaunchParameters( system.p.N_x, system.p.N_y );

    // Liner Half Step
    // Calculate the FFT of Psi
    calculateFFT( kernel_arguments.dev_ptrs.wavefunction_plus, kernel_arguments.dev_ptrs.fft_plus, FFT::forward );
    if (system.p.use_twin_mode)
        calculateFFT( kernel_arguments.dev_ptrs.wavefunction_minus, kernel_arguments.dev_ptrs.fft_minus, FFT::forward );
    CALL_KERNEL(
        RUNGE_FUNCTION_GP_LINEAR, "linear_half_step", grid_size, block_size, 0,
        v_time, kernel_arguments, 
        { 
            kernel_arguments.dev_ptrs.fft_plus, kernel_arguments.dev_ptrs.fft_minus, kernel_arguments.dev_ptrs.discard, kernel_arguments.dev_ptrs.discard,
            kernel_arguments.dev_ptrs.wavefunction_plus, kernel_arguments.dev_ptrs.wavefunction_minus, kernel_arguments.dev_ptrs.discard, kernel_arguments.dev_ptrs.discard
        }
    );
    // Transform back. WF now holds the half-stepped wavefunction.
    calculateFFT( kernel_arguments.dev_ptrs.wavefunction_plus, kernel_arguments.dev_ptrs.fft_plus, FFT::inverse );
    if (system.p.use_twin_mode)
        calculateFFT( kernel_arguments.dev_ptrs.wavefunction_minus, kernel_arguments.dev_ptrs.fft_minus, FFT::inverse );

    // Nonlinear Full Step
    CALL_KERNEL(
        RUNGE_FUNCTION_GP_NONLINEAR, "nonlinear_full_step", grid_size, block_size, 0,
        v_time, kernel_arguments, 
        { 
            kernel_arguments.dev_ptrs.fft_plus, kernel_arguments.dev_ptrs.fft_minus, kernel_arguments.dev_ptrs.reservoir_plus, kernel_arguments.dev_ptrs.reservoir_minus,
            kernel_arguments.dev_ptrs.wavefunction_plus, kernel_arguments.dev_ptrs.wavefunction_minus, kernel_arguments.dev_ptrs.buffer_reservoir_plus, kernel_arguments.dev_ptrs.buffer_reservoir_minus
        }
    );
    // WF now holds the nonlinearly evolved wavefunction.

    // Liner Half Step 
    // Calculate the FFT of Psi
    calculateFFT( kernel_arguments.dev_ptrs.wavefunction_plus, kernel_arguments.dev_ptrs.fft_plus, FFT::forward );
    if (system.p.use_twin_mode)
        calculateFFT( kernel_arguments.dev_ptrs.wavefunction_minus, kernel_arguments.dev_ptrs.fft_minus, FFT::forward );
    CALL_KERNEL(
        RUNGE_FUNCTION_GP_LINEAR, "linear_half_step", grid_size, block_size, 0,
        v_time, kernel_arguments, 
        { 
            kernel_arguments.dev_ptrs.fft_plus, kernel_arguments.dev_ptrs.fft_minus, kernel_arguments.dev_ptrs.discard, kernel_arguments.dev_ptrs.discard,
            kernel_arguments.dev_ptrs.wavefunction_plus, kernel_arguments.dev_ptrs.wavefunction_minus, kernel_arguments.dev_ptrs.discard, kernel_arguments.dev_ptrs.discard
        }
    );
    // Transform back. WF now holds the half-stepped wavefunction.
    calculateFFT( kernel_arguments.dev_ptrs.wavefunction_plus, kernel_arguments.dev_ptrs.fft_plus, FFT::inverse );
    if (system.p.use_twin_mode)
        calculateFFT( kernel_arguments.dev_ptrs.wavefunction_minus, kernel_arguments.dev_ptrs.fft_minus, FFT::inverse );

    CALL_KERNEL(
        RUNGE_FUNCTION_GP_INDEPENDENT, "independent", grid_size, block_size, 0,
        v_time, kernel_arguments, 
        {  
            kernel_arguments.dev_ptrs.fft_plus, kernel_arguments.dev_ptrs.fft_minus, kernel_arguments.dev_ptrs.buffer_reservoir_plus, kernel_arguments.dev_ptrs.buffer_reservoir_minus,
            kernel_arguments.dev_ptrs.wavefunction_plus, kernel_arguments.dev_ptrs.wavefunction_minus, kernel_arguments.dev_ptrs.reservoir_plus, kernel_arguments.dev_ptrs.reservoir_minus
        }
    );
    // WF now holds the new result
    
}