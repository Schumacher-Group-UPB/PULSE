#include "cuda/typedef.cuh"

#ifdef USE_CUDA
    #include <thrust/reduce.h>
    #include <thrust/transform_reduce.h>
    #include <thrust/execution_policy.h>
#else
    #include <numeric>
#endif
#include <omp.h>

// Include Cuda Kernel headers
#include "kernel/kernel_compute.cuh"
#include "system/system_parameters.hpp"
#include "misc/helperfunctions.hpp"
#include "cuda/cuda_matrix.cuh"
#include "solver/gpu_solver.hpp"
#include "misc/commandline_io.hpp"

struct square_reduction
{
    PULSE_HOST_DEVICE PC3::Type::real operator()(const PC3::Type::complex& x) const { 
        const PC3::Type::real res = PC3::CUDA::abs2(x);
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
void PC3::Solver::iterateVariableTimestepRungeKutta( dim3 block_size, dim3 grid_size ) {
    // Accept current step?
    bool accept = false;

    // This variable contains all the device pointers the kernel could need
    auto device_pointers = matrix.pointers();
    // Same IO every time except for error kernel
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

    auto cf = RKCoefficients::DP();

    // The delta time is either real or imaginary, depending on the system configuration
    //Type::complex delta_time = system.imaginary_time ? Type::complex(0.0, -system.p.dt) : Type::complex(system.p.dt, 0.0);

    // If required, calculate new set of random numbers.
    if (evaluate_stochastic)
    CALL_KERNEL(
        PC3::Kernel::generate_random_numbers, "random_number_gen", grid_size, block_size,
        device_pointers.random_state, device_pointers.random_number, system.p.N_x*system.p.N_y, system.p.stochastic_amplitude*std::sqrt(system.p.dt), system.p.stochastic_amplitude*std::sqrt(system.p.dt)
    );

    do {
        // We snapshot here to make sure that the dt is updated
        auto p = system.kernel_parameters;

        CALCULATE_K( 1, p.t, wavefunction, reservoir );
        
        CALL_KERNEL(
            Kernel::RK::runge_sum_to_input_kw, "Sum for K1", grid_size, block_size,
            p.dt, device_pointers, p, io,
            { cf.b11 }
        );
        
        CALCULATE_K( 2, p.t + cf.a2 * p.dt, buffer_wavefunction, buffer_reservoir );
        
        CALL_KERNEL(
            Kernel::RK::runge_sum_to_input_kw, "Sum for K2", grid_size, block_size,
            p.dt, device_pointers, p, io,
            { cf.b21, cf.b22 }
        );

        CALCULATE_K( 3, p.t + cf.a3 * p.dt, buffer_wavefunction, buffer_reservoir );

        CALL_KERNEL(
            Kernel::RK::runge_sum_to_input_kw, "Sum for K3", grid_size, block_size,
            p.dt, device_pointers, p, io,
            { cf.b31, cf.b32, cf.b33 }
        );


        CALCULATE_K( 4, p.t + cf.a4 * p.dt, buffer_wavefunction, buffer_reservoir );

        CALL_KERNEL(
            Kernel::RK::runge_sum_to_input_kw, "Sum for K4", grid_size, block_size,
            p.dt, device_pointers, p, io,
            { cf.b41, cf.b42, cf.b43, cf.b44 }
        );

        CALCULATE_K( 5, p.t + cf.a5 * p.dt, buffer_wavefunction, buffer_reservoir );

        CALL_KERNEL(
            Kernel::RK::runge_sum_to_input_kw, "Sum for K5", grid_size, block_size,
            p.dt, device_pointers, p, io,
            { cf.b51, cf.b52, cf.b53, cf.b54, cf.b55 }
        );

        CALCULATE_K( 6, p.t + cf.a6 * p.dt, buffer_wavefunction, buffer_reservoir );

        // Final Result is in the buffer_ arrays.
        CALL_KERNEL(
            Kernel::RK::runge_sum_to_input_kw, "Sum for K6", grid_size, block_size,
            p.dt, device_pointers, p, io,
            { cf.b61, cf.b62, cf.b63, cf.b64, cf.b65, cf.b66 }
        );

        CALL_KERNEL(
            Kernel::RK::runge_sum_to_error, "Error", grid_size, block_size,
            p.dt, device_pointers, p,
            { cf.e7 /* WF */, cf.e1, cf.e2, cf.e3, cf.e4, cf.e5, cf.e6 }
        );

        #ifdef USE_CUDA
            Type::complex error = thrust::reduce( matrix.rk_error.dbegin(), matrix.rk_error.dend(), Type::complex(0.0), thrust::plus<Type::complex>() );
            Type::real sum_abs2 = thrust::transform_reduce( matrix.wavefunction_plus.dbegin(), matrix.wavefunction_plus.dend(), square_reduction(), Type::real(0.0), thrust::plus<Type::real>() );
        #else
            Type::complex error = std::reduce( matrix.rk_error.dbegin(), matrix.rk_error.dend(), Type::complex(0.0), std::plus<Type::complex>() );
            Type::real sum_abs2 = std::transform_reduce( matrix.wavefunction_plus.dbegin(), matrix.wavefunction_plus.dend(), Type::real(0.0), std::plus<Type::real>(), square_reduction() );
        #endif
        // TODO: maybe go back to using max since thats faster
        //auto plus_max = std::get<1>( minmax( matrix.wavefunction_plus.getDevicePtr(), p.N_x * p.N_y, true /*Device Pointer*/ ) );
        Type::real final_error = CUDA::abs(error) / sum_abs2;

        
        // Calculate dh
        Type::real dh = std::pow<Type::real>( system.tolerance / 2. / std::max<Type::real>( final_error, 1E-15 ), 0.25 );
        // Check if dh is nan
        if ( std::isnan( dh ) ) {
            dh = 1.0;
        }
        if ( std::isnan( final_error ) )
            dh = 0.5;
        
        //  Set new timestep
        system.p.dt = std::min<Type::real>(p.dt * dh, system.dt_max);
        if ( dh < 1.0 )
           //system.p.dt = std::max<Type::real>( p.dt - system.dt_min * std::floor( 1.0 / dh ), system.dt_min );
           p.dt -= system.dt_min;
        else
           //system.p.dt = std::min<Type::real>( p.dt + system.dt_min * std::floor( dh ), system.dt_max );
           p.dt += system.dt_min;

        // Make sure to also update dt from p
        system.kernel_parameters.dt = p.dt;
        

        // Accept step if error is below tolerance
        if ( final_error < system.tolerance ) {
            accept = true;
            // Swap the next and current wavefunction buffers. This only swaps the pointers, not the data.
            swapBuffers();
        }
    } while ( !accept );
}