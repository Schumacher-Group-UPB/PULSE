#ifndef USECPU
#include <cuComplex.h>
#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cuda.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>
#include <thrust/execution_policy.h>
#else
#include <numeric>
#endif

#include <complex>
#include <omp.h>

// Include Cuda Kernel headers
#include "cuda/cuda_complex.cuh"
#include "cuda/cuda_macro.cuh"
#include "kernel/kernel_runge_function.cuh"
#include "kernel/kernel_summation.cuh"
#include "kernel/kernel_fft.cuh"
#include "system/system.hpp"
#include "misc/helperfunctions.hpp"
#include "cuda/cuda_matrix.cuh"
#include "solver/gpu_solver.cuh"

/*
 * Helper variable for caching the current time for FFT evaluations.
 * We dont need this variable anywhere else, so we just create it
 * locally to this file here.
 */
real_number cached_t = 0.0;

// Helper macro to choose the correct runge function
#define RUNGE_FUNCTION (system.use_te_tm_splitting ? PC3::Kernel::runge_func_kernel_tetm : PC3::Kernel::runge_func_kernel_scalar)

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
 * Note, that this function uses k7-k3 instead of k4-k1 as temporary variables
 * due to the size increases of the temporary arrays.
 * @param system The system to iterate
 * @param evaluate_pulse If true, the pulse is evaluated at the current time step
 */
void PC3::Solver::iterateFixedTimestepRungeKutta( bool evaluate_pulse, dim3 block_size, dim3 grid_size ) {
    // This variable contains all the system parameters the kernel could need
    auto current_system_parameters = system.snapshotParameters();

    // This variable contains all the device pointers the kernel could need
    auto device_pointers = device.pointers();
    // Pointers to Pulse Variables. This is subject to change
    auto pulse_pointers = dev_pulse_parameters.pointers();

    CALL_KERNEL(
        RUNGE_FUNCTION, "K1", grid_size, block_size, 
        system.t, device_pointers, current_system_parameters, pulse_pointers, evaluate_pulse,
        { 
            device_pointers.wavefunction_plus, device_pointers.wavefunction_minus, device_pointers.reservoir_plus, device_pointers.reservoir_minus,
            device_pointers.k1_wavefunction_plus, device_pointers.k1_wavefunction_minus, device_pointers.k1_reservoir_plus, device_pointers.k1_reservoir_minus
        }
    );

    CALL_KERNEL(
        Kernel::RK4::runge_sum_to_input_k2, "Sum for K2", grid_size, block_size, 
        device_pointers, current_system_parameters, system.use_te_tm_splitting
    );

    CALL_KERNEL(
        RUNGE_FUNCTION, "K2", grid_size, block_size, 
        system.t + 0.5*system.dt, device_pointers, current_system_parameters, pulse_pointers, evaluate_pulse,
        { 
            device_pointers.buffer_wavefunction_plus, device_pointers.buffer_wavefunction_minus, device_pointers.buffer_reservoir_plus, device_pointers.buffer_reservoir_minus,
            device_pointers.k2_wavefunction_plus, device_pointers.k2_wavefunction_minus, device_pointers.k2_reservoir_plus, device_pointers.k2_reservoir_minus
        }
    );

    CALL_KERNEL(
        Kernel::RK4::runge_sum_to_input_k3, "Sum for K3", grid_size, block_size, 
        device_pointers, current_system_parameters, system.use_te_tm_splitting
    );

    CALL_KERNEL(
        RUNGE_FUNCTION, "K3", grid_size, block_size, 
        system.t + 0.5*system.dt, device_pointers, current_system_parameters, pulse_pointers, evaluate_pulse,
        { 
            device_pointers.buffer_wavefunction_plus, device_pointers.buffer_wavefunction_minus, device_pointers.buffer_reservoir_plus, device_pointers.buffer_reservoir_minus,
            device_pointers.k3_wavefunction_plus, device_pointers.k3_wavefunction_minus, device_pointers.k3_reservoir_plus, device_pointers.k3_reservoir_minus
        }
    );

    CALL_KERNEL(
        Kernel::RK4::runge_sum_to_input_k4, "Sum for K4", grid_size, block_size, 
        device_pointers, current_system_parameters, system.use_te_tm_splitting
    );

    CALL_KERNEL(
        RUNGE_FUNCTION, "K4", grid_size, block_size, 
        system.t + system.dt, device_pointers, current_system_parameters, pulse_pointers, evaluate_pulse,
        { 
            device_pointers.buffer_wavefunction_plus, device_pointers.buffer_wavefunction_minus, device_pointers.buffer_reservoir_plus, device_pointers.buffer_reservoir_minus,
            device_pointers.k4_wavefunction_plus, device_pointers.k4_wavefunction_minus, device_pointers.k4_reservoir_plus, device_pointers.k4_reservoir_minus
        }
    );

    // Final result is in the wavefunction_ arrays, NOT the buffers.
    CALL_KERNEL(
        Kernel::RK4::runge_sum_to_final, "Final Sum", grid_size, block_size, 
        device_pointers, current_system_parameters, system.use_te_tm_splitting
    );

    return;
}

void PC3::Solver::dummy( bool evaluate_pulse, dim3 block_size, dim3 grid_size ) {
    // This variable contains all the system parameters the kernel could need
    auto current_system_parameters = system.snapshotParameters();
    // This variable contains all the device pointers the kernel could need
    auto device_pointers = device.pointers();
    // Pointers to Pulse Variables. This is subject to change
    auto pulse_pointers = dev_pulse_parameters.pointers();
    
    CALL_KERNEL(
        RUNGE_FUNCTION, "K1", grid_size, block_size, 
        system.t, device_pointers, current_system_parameters, pulse_pointers, evaluate_pulse,
        { 
            device_pointers.wavefunction_plus, device_pointers.wavefunction_minus, device_pointers.reservoir_plus, device_pointers.reservoir_minus,
            device_pointers.k1_wavefunction_plus, device_pointers.k1_wavefunction_minus, device_pointers.k1_reservoir_plus, device_pointers.k1_reservoir_minus
        }
    );

    CALL_KERNEL(
        Kernel::RK4::runge_sum_to_input_k2, "Sum for K2", grid_size, block_size, 
        device_pointers, current_system_parameters, system.use_te_tm_splitting
    );

    CALL_KERNEL(
        RUNGE_FUNCTION, "K2", grid_size, block_size, 
        system.t + 0.5*system.dt, device_pointers, current_system_parameters, pulse_pointers, evaluate_pulse,
        { 
            device_pointers.buffer_wavefunction_plus, device_pointers.buffer_wavefunction_minus, device_pointers.buffer_reservoir_plus, device_pointers.buffer_reservoir_minus,
            device_pointers.k2_wavefunction_plus, device_pointers.k2_wavefunction_minus, device_pointers.k2_reservoir_plus, device_pointers.k2_reservoir_minus
        }
    );

    CALL_KERNEL(
        Kernel::RK4::runge_sum_to_input_k3, "Sum for K3", grid_size, block_size, 
        device_pointers, current_system_parameters, system.use_te_tm_splitting
    );

    CALL_KERNEL(
        RUNGE_FUNCTION, "K3", grid_size, block_size, 
        system.t + 0.5*system.dt, device_pointers, current_system_parameters, pulse_pointers, evaluate_pulse,
        { 
            device_pointers.buffer_wavefunction_plus, device_pointers.buffer_wavefunction_minus, device_pointers.buffer_reservoir_plus, device_pointers.buffer_reservoir_minus,
            device_pointers.k3_wavefunction_plus, device_pointers.k3_wavefunction_minus, device_pointers.k3_reservoir_plus, device_pointers.k3_reservoir_minus
        }
    );

    CALL_KERNEL(
        Kernel::RK4::runge_sum_to_input_k4, "Sum for K4", grid_size, block_size, 
        device_pointers, current_system_parameters, system.use_te_tm_splitting
    );

    CALL_KERNEL(
        RUNGE_FUNCTION, "K4", grid_size, block_size, 
        system.t + system.dt, device_pointers, current_system_parameters, pulse_pointers, evaluate_pulse,
        { 
            device_pointers.buffer_wavefunction_plus, device_pointers.buffer_wavefunction_minus, device_pointers.buffer_reservoir_plus, device_pointers.buffer_reservoir_minus,
            device_pointers.k4_wavefunction_plus, device_pointers.k4_wavefunction_minus, device_pointers.k4_reservoir_plus, device_pointers.k4_reservoir_minus
        }
    );

    // Final somethign else that does summation
    CALL_KERNEL(
        Kernel::RK4::runge_sum_to_input_k3, "Sum for K3", grid_size, block_size, 
        device_pointers, current_system_parameters, system.use_te_tm_splitting
    );

    return;
}



struct square_reduction
{
    CUDA_HOST_DEVICE real_number operator()(const complex_number& x) const { 
        const real_number res = PC3::CUDA::abs2(x);
        return res; 
    }
};


/*
* This function iterates the Runge Kutta Kernel using a variable time step.
* A 4th order Runge-Kutta method is used to calculate
* the next y iteration; a 5th order solution is
* used to calculate the iteration error.
* This function calls multiple different rungeFuncSum functions with varying
* delta-t and coefficients. Calculation of the inputs for the next
* rungeFuncKernel call is done in the rungeFuncSum function.
* The general implementation of the RK45 method goes as follows:
* ------------------------------------------------------------------------------
* k1 = f(t, y) = rungeFuncKernel(current)
* input_for_k2 = current + b11 * dt * k1
* k2 = f(t + a2 * dt, input_for_k2) = rungeFuncKernel(input_for_k2)
* input_for_k3 = current + b21 * dt * k1 + b22 * dt * k2
* k3 = f(t + a3 * dt, input_for_k3) = rungeFuncKernel(input_for_k3)
* input_for_k4 = current + b31 * dt * k1 + b32 * dt * k2 + b33 * dt * k3
* k4 = f(t + a4 * dt, input_for_k4) = rungeFuncKernel(input_for_k4)
* input_for_k5 = current + b41 * dt * k1 + b42 * dt * k2 + b43 * dt * k3
                 + b44 * dt * k4
* k5 = f(t + a5 * dt, input_for_k5) = rungeFuncKernel(input_for_k5)
* input_for_k6 = current + b51 * dt * k1 + b52 * dt * k2 + b53 * dt * k3
                 + b54 * dt * k4 + b55 * dt * k5
* k6 = f(t + a6 * dt, input_for_k6) = rungeFuncKernel(input_for_k6)
* next = current + dt * (b61 * k1 + b63 * k3 + b64 * k4 + b65 * k5 + b66 * k6)
* k7 = f(t + a7 * dt, next) = rungeFuncKernel(next)
* error = dt * (e1 * k1 + e3 * k3 + e4 * k4 + e5 * k5 + e6 * k6 + e7 * k7)
* ------------------------------------------------------------------------------
* The error is then used to update the timestep; If the error is below threshold,
* the iteration is accepted and the total time is increased by dt. If the error
* is above threshold, the iteration is rejected and the timestep is decreased.
* The timestep is always bounded by dt_min and dt_max and will only increase
* using whole multiples of dt_min.
* ------------------------------------------------------------------------------
* @param system The system to iterate
* @param evaluate_pulse If true, the pulse is evaluated at the current time step
*/
void PC3::Solver::iterateVariableTimestepRungeKutta( bool evaluate_pulse, dim3 block_size, dim3 grid_size ) {
    // Accept current step?
    bool accept = false;
    // This variable contains all the device pointers the kernel could need
    auto device_pointers = device.pointers();
    // This variable contains all the system parameters the kernel could need
    auto current_system_parameters = system.snapshotParameters();
    // Pointers to Pulse Variables. This is subject to change
    auto pulse_pointers = dev_pulse_parameters.pointers();
    
    do {

        CALL_KERNEL(
            RUNGE_FUNCTION, "K1", grid_size, block_size, 
            system.t, device_pointers, current_system_parameters, pulse_pointers, evaluate_pulse,
            { 
                device_pointers.wavefunction_plus, device_pointers.wavefunction_minus, device_pointers.reservoir_plus, device_pointers.reservoir_minus,
                device_pointers.k1_wavefunction_plus, device_pointers.k1_wavefunction_minus, device_pointers.k1_reservoir_plus, device_pointers.k1_reservoir_minus
            }
        );

        CALL_KERNEL(
            PC3::Kernel::RK45::runge_sum_to_input_of_k2, "Sum for K2", grid_size, block_size, 
            device_pointers, current_system_parameters, system.use_te_tm_splitting
        );

        CALL_KERNEL(
            RUNGE_FUNCTION, "K2", grid_size, block_size, 
            system.t + RKCoefficients::a2 * system.dt, device_pointers, current_system_parameters, pulse_pointers, evaluate_pulse,
            { 
                device_pointers.buffer_wavefunction_plus, device_pointers.buffer_wavefunction_minus, device_pointers.buffer_reservoir_plus, device_pointers.buffer_reservoir_minus,
                device_pointers.k2_wavefunction_plus, device_pointers.k2_wavefunction_minus, device_pointers.k2_reservoir_plus, device_pointers.k2_reservoir_minus
            }
        );

        CALL_KERNEL(
            PC3::Kernel::RK45::runge_sum_to_input_of_k3, "Sum for K3", grid_size, block_size, 
            device_pointers, current_system_parameters, system.use_te_tm_splitting
        );


        CALL_KERNEL(
            RUNGE_FUNCTION, "K3", grid_size, block_size, 
            system.t + RKCoefficients::a3 * system.dt, device_pointers, current_system_parameters, pulse_pointers, evaluate_pulse,
            { 
                device_pointers.buffer_wavefunction_plus, device_pointers.buffer_wavefunction_minus, device_pointers.buffer_reservoir_plus, device_pointers.buffer_reservoir_minus,
                device_pointers.k3_wavefunction_plus, device_pointers.k3_wavefunction_minus, device_pointers.k3_reservoir_plus, device_pointers.k3_reservoir_minus
            }
        );

        CALL_KERNEL(
            PC3::Kernel::RK45::runge_sum_to_input_of_k4, "Sum for K4", grid_size, block_size, 
            device_pointers, current_system_parameters, system.use_te_tm_splitting
        );

        CALL_KERNEL(
            RUNGE_FUNCTION, "K4", grid_size, block_size, 
            system.t + RKCoefficients::a4 * system.dt, device_pointers, current_system_parameters, pulse_pointers, evaluate_pulse,
            { 
                device_pointers.buffer_wavefunction_plus, device_pointers.buffer_wavefunction_minus, device_pointers.buffer_reservoir_plus, device_pointers.buffer_reservoir_minus,
                device_pointers.k4_wavefunction_plus, device_pointers.k4_wavefunction_minus, device_pointers.k4_reservoir_plus, device_pointers.k4_reservoir_minus
            }
        );

        CALL_KERNEL(
            PC3::Kernel::RK45::runge_sum_to_input_of_k5, "Sum for K5", grid_size, block_size, 
            device_pointers, current_system_parameters, system.use_te_tm_splitting
        );

        CALL_KERNEL(
            RUNGE_FUNCTION, "K5", grid_size, block_size, 
            system.t + RKCoefficients::a5 * system.dt, device_pointers, current_system_parameters, pulse_pointers, evaluate_pulse,
            { 
                device_pointers.buffer_wavefunction_plus, device_pointers.buffer_wavefunction_minus, device_pointers.buffer_reservoir_plus, device_pointers.buffer_reservoir_minus,
                device_pointers.k5_wavefunction_plus, device_pointers.k5_wavefunction_minus, device_pointers.k5_reservoir_plus, device_pointers.k5_reservoir_minus
            }
        );

        CALL_KERNEL(
            PC3::Kernel::RK45::runge_sum_to_input_of_k6, "Sum for K6", grid_size, block_size, 
            device_pointers, current_system_parameters, system.use_te_tm_splitting
        );

        CALL_KERNEL(
            RUNGE_FUNCTION, "K6", grid_size, block_size, 
            system.t + RKCoefficients::a6 * system.dt, device_pointers, current_system_parameters, pulse_pointers, evaluate_pulse,
            { 
                device_pointers.buffer_wavefunction_plus, device_pointers.buffer_wavefunction_minus, device_pointers.buffer_reservoir_plus, device_pointers.buffer_reservoir_minus,
                device_pointers.k6_wavefunction_plus, device_pointers.k6_wavefunction_minus, device_pointers.k6_reservoir_plus, device_pointers.k6_reservoir_minus
            }
        );

        // Final Result is in the buffer_ arrays
        CALL_KERNEL(
            PC3::Kernel::RK45::runge_sum_to_final, "Final Sum", grid_size, block_size, 
            device_pointers, current_system_parameters, system.use_te_tm_splitting
        );

        CALL_KERNEL(
            RUNGE_FUNCTION, "K7", grid_size, block_size, 
            system.t + RKCoefficients::a7 * system.dt, device_pointers, current_system_parameters, pulse_pointers, evaluate_pulse,
            { 
                device_pointers.buffer_wavefunction_plus, device_pointers.buffer_wavefunction_minus, device_pointers.buffer_reservoir_plus, device_pointers.buffer_reservoir_minus,
                device_pointers.k7_wavefunction_plus, device_pointers.k7_wavefunction_minus, device_pointers.k7_reservoir_plus, device_pointers.k7_reservoir_minus
            }
        );

        CALL_KERNEL(
            PC3::Kernel::RK45::runge_sum_final_error, "Final Sum Error", grid_size, block_size, 
            device_pointers, current_system_parameters, system.use_te_tm_splitting
        );

        #ifndef USECPU
        real_number final_error = thrust::reduce( THRUST_DEVICE, device.rk_error.get(), device.rk_error.get() + system.s_N * system.s_N, 0.0, thrust::plus<real_number>() );
        real_number sum_abs2 = thrust::transform_reduce( THRUST_DEVICE, device.wavefunction_plus.get(), device.wavefunction_plus.get() + system.s_N * system.s_N, square_reduction(), 0.0, thrust::plus<real_number>() );
        #else
        real_number final_error = std::reduce( device.rk_error.get(), device.rk_error.get() + system.s_N * system.s_N, 0.0, std::plus<real_number>() );
        real_number sum_abs2 = std::transform_reduce( device.wavefunction_plus.get(), device.wavefunction_plus.get() + system.s_N * system.s_N, 0.0, std::plus<real_number>(), square_reduction() );
        #endif

        // TODO: maybe go back to using max since thats faster
        //auto plus_max = std::get<1>( minmax( device.wavefunction_plus.get(), system.s_N * system.s_N, true /*Device Pointer*/ ) );
        final_error = final_error / sum_abs2;

        // Calculate dh
        real_number dh = std::pow( system.tolerance / 2. / CUDA::max( final_error, 1E-15 ), 0.25 );
        // Check if dh is nan
        if ( std::isnan( dh ) ) {
            dh = 1.0;
        }
        if ( std::isnan( final_error ) )
            dh = 0.5;
        
        //  Set new timestep
        // system.dt = min(system.dt * dh, system.dt_max);
        if ( dh < 1.0 )
            system.dt = CUDA::max( system.dt - system.dt_min * CUDA::floor( 1.0 / dh ), system.dt_min );
            //system.dt -= system.dt_min;
        else
            system.dt = CUDA::min( system.dt + system.dt_min * CUDA::floor( dh ), system.dt_max );
            //system.dt += system.dt_min;

        // Make sure to also update dt from current_system_parameters
        current_system_parameters.dt = system.dt;

        // Accept step if error is below tolerance
        if ( final_error < system.tolerance ) {
            accept = true;
            // Since the "next" Y is in the buffer_ arrays, we swap current_wavefunction and buffer_wf
            // This is fast, because we just swap pointers instead of copying data.
            device.wavefunction_plus.swap( device.buffer_wavefunction_plus );
            device.reservoir_plus.swap( device.buffer_reservoir_plus );
            if ( system.use_te_tm_splitting ) {
                device.wavefunction_minus.swap( device.buffer_wavefunction_minus );
                device.reservoir_minus.swap( device.buffer_reservoir_minus );
            }
        }
    } while ( !accept );
}

/*
 * This function calculates the Fast Fourier Transformation of Psi+ and Psi-
 * and saves the result in dev_fft_plus and dev_fft_minus. These values can
 * then be grabbed using the getDeviceArrays() function. The FFT is then
 * shifted such that k = 0 is in the center of the FFT matrix. Then, the
 * FFT Filter is applied to the FFT, and the FFT is shifted back. Finally,
 * the inverse FFT is calculated and the result is saved in dev_current_Psi_Plus
 * and dev_current_Psi_Minus. The FFT Arrays are shifted once again for
 * visualization purposes.
 * NOTE/OPTIMIZATION: The Shift->Filter->Shift function will be changed later
 * to a cached filter mask, which itself will be shifted.
 */
void PC3::Solver::applyFFTFilter( dim3 block_size, dim3 grid_size, bool apply_mask ) {
    #ifndef USECPU
    // Calculate the actual FFTs
    CHECK_CUDA_ERROR( FFTSOLVER( plan, (fft_complex_number*)device.wavefunction_plus.get(), (fft_complex_number*)device.fft_plus.get(), CUFFT_FORWARD ), "FFT Exec" );


    // For now, we shift, transform, shift the results. TODO: Move this into one function without shifting
    // Shift FFT to center k = 0
    fft_shift_2D<<<grid_size, block_size>>>( device.fft_plus.get(), system.s_N );
    CHECK_CUDA_ERROR( {}, "FFT Shift Plus" );

    // Do the FFT and the shifting here already for visualization only
    if ( system.use_te_tm_splitting ) {
        CHECK_CUDA_ERROR( FFTSOLVER( plan, (fft_complex_number*)device.wavefunction_minus.get(), (fft_complex_number*)device.fft_minus.get(), CUFFT_FORWARD ), "FFT Exec" );
        fft_shift_2D<<<grid_size, block_size>>>( device.fft_minus.get(), system.s_N );
        CHECK_CUDA_ERROR( {}, "FFT Shift Minus" );
    }
    
    if (not apply_mask)
        return;
    
    // Apply the FFT Mask Filter
    kernel_mask_fft<<<grid_size, block_size>>>( device.fft_plus.get(), device.fft_mask_plus.get(), system.s_N );
    CHECK_CUDA_ERROR( {}, "FFT Filter" )
    
    // Undo the shift
    fft_shift_2D<<<grid_size, block_size>>>( device.fft_plus.get(), system.s_N );
    CHECK_CUDA_ERROR( {}, "FFT Shift" )

    // Transform back.
    CHECK_CUDA_ERROR( FFTSOLVER( plan, device.fft_plus.get(), device.wavefunction_plus.get(), CUFFT_INVERSE ), "iFFT Exec" );
    
    // Shift FFT Once again for visualization
    fft_shift_2D<<<grid_size, block_size>>>( device.fft_plus.get(), system.s_N );
    CHECK_CUDA_ERROR( {}, "FFT Shift" );
    
    // Do the same for the minus component
    if (not system.use_te_tm_splitting)
        return;
    kernel_mask_fft<<<grid_size, block_size>>>( device.fft_minus.get(), device.fft_mask_minus.get(), system.s_N );
    CHECK_CUDA_ERROR( {}, "FFT Filter" )
    fft_shift_2D<<<grid_size, block_size>>>( device.fft_minus.get(), system.s_N );
    CHECK_CUDA_ERROR( {}, "FFT Shift" )
    CHECK_CUDA_ERROR( FFTSOLVER( plan, device.fft_minus.get(), device.wavefunction_minus.get(), CUFFT_INVERSE ), "iFFT Exec" );
    fft_shift_2D<<<grid_size, block_size>>>( device.fft_minus.get(), system.s_N );
    CHECK_CUDA_ERROR( {}, "FFT Shift" );
    #endif
}

/**
 * Iterates the Runge-Kutta-Method on the GPU
 * Note, that all device arrays and variables have to be initialized at this point
 * @param t Current time, will be updated to t + dt
 * @param dt Time step, will be updated to the next time step
 * @param s_N Number of grid points in one dimension
 */
void PC3::Solver::iterateRungeKutta( ) {

    // First, check if the maximum time has been reached
    if ( system.t >= system.t_max )
        return;

    dim3 block_size( system.block_size, system.block_size );
    int gs = ceil( system.s_N/block_size.x );
    dim3 grid_size( ( system.s_N + block_size.x - 1 ) / block_size.x, ( system.s_N + block_size.y - 1 ) / block_size.y );

    // The CPU should briefly evaluate wether the pulses have to be evaluated
    bool evaluate_pulse = system.evaluatePulse();

    if ( system.fixed_time_step ) {
        iterateFixedTimestepRungeKutta( evaluate_pulse, block_size, grid_size );
        //for (int i = 0; i < 4; i++)
        //dummy( evaluate_pulse, block_size, grid_size );
        }
    else
        iterateVariableTimestepRungeKutta( evaluate_pulse, block_size, grid_size );

    // Syncronize
    //CHECK_CUDA_ERROR( cudaDeviceSynchronize(), "Sync" );
    
    // Increase t.
    system.t = system.t + system.dt;

    // For statistical purposes, increase the iteration counter
    system.iteration++;

    // FFT Guard 
    if ( system.t - cached_t < system.fft_every )
        return;  
    // Calculate the FFT
    cached_t = system.t; 
    applyFFTFilter( block_size, grid_size, system.fft_mask.size() > 0 );

    // Syncronize
    //CHECK_CUDA_ERROR( cudaDeviceSynchronize(), "Sync" );
}