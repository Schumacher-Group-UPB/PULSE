#pragma once

#include <iostream>
#include <map>
#include <functional>
#include "cuda/typedef.cuh"
#include "cuda/cuda_matrix.cuh"
#include "cuda/cuda_macro.cuh"
#include "kernel/kernel_fft.cuh"
#include "system/system_parameters.hpp"
#include "system/filehandler.hpp"
#include "solver/matrix_container.cuh"
#include "misc/escape_sequences.hpp"

namespace PC3 {

/**
 * @brief GPU Solver class providing the interface for the GPU solver.
 * Implements RK4, RK45, FFT calculations.
 *
 */
class Solver {
   public:
    // References to system and filehandler so we dont need to pass them around all the time
    PC3::SystemParameters& system;
    PC3::FileHandler& filehandler;

    // TODO: amp zu Type::device_vector. cudamatrix not needed
    struct TemporalEvelope {
        Type::device_vector<Type::complex> amp;

        struct Pointers {
            Type::complex* amp;
            Type::uint n;
        };

        Pointers pointers() {
            return Pointers{ GET_RAW_PTR(amp), Type::uint(amp.size()) };
        }
    } dev_pulse_oscillation, dev_pump_oscillation, dev_potential_oscillation;

    // Host/Device Matrices
    MatrixContainer matrix;

    struct InputOutput {
        Type::complex* PULSE_RESTRICT in_wf_plus = nullptr;
        Type::complex* PULSE_RESTRICT in_wf_minus = nullptr;
        Type::complex* PULSE_RESTRICT in_rv_plus = nullptr;
        Type::complex* PULSE_RESTRICT in_rv_minus = nullptr;
        Type::complex* PULSE_RESTRICT out_wf_plus = nullptr;
        Type::complex* PULSE_RESTRICT out_wf_minus = nullptr;
        Type::complex* PULSE_RESTRICT out_rv_plus = nullptr;
        Type::complex* PULSE_RESTRICT out_rv_minus = nullptr;
    };

    struct KernelArguments {
        TemporalEvelope::Pointers pulse_pointers; // The pointers to the envelopes. These are obtained by calling the .pointers() method on the envelopes.
        TemporalEvelope::Pointers pump_pointers; // The pointers to the envelopes. These are obtained by calling the .pointers() method on the envelopes.
        TemporalEvelope::Pointers potential_pointers; // The pointers to the envelopes. These are obtained by calling the .pointers() method on the envelopes.
        MatrixContainer::Pointers dev_ptrs; // All the pointers to the matrices. These are obtained by calling the .pointers() method on the matrices.
        SystemParameters::KernelParameters p; // The kernel parameters. These are obtained by copying the kernel_parameters object of the system.
    };

    // Fixed Kernel Arguments. Every Compute Kernel will take one of these.
    KernelArguments generateKernelArguments(const Type::uint subgrid) {
        auto kernel_arguments = KernelArguments();
        kernel_arguments.pulse_pointers = dev_pulse_oscillation.pointers();
        kernel_arguments.pump_pointers = dev_pump_oscillation.pointers();
        kernel_arguments.potential_pointers = dev_potential_oscillation.pointers();
        kernel_arguments.dev_ptrs = matrix.pointers(subgrid);
        kernel_arguments.p = system.kernel_parameters;
        return kernel_arguments;
    }

    struct VKernelArguments {
        Type::complex t, dt;
    };

    // Cache Maps
    std::map<std::string, std::vector<Type::real>> cache_map_scalar;

    Solver( PC3::SystemParameters& system ) : system( system ), filehandler( system.filehandler ) {
        std::cout << PC3::CLIO::prettyPrint( "Creating Solver...", PC3::CLIO::Control::Info ) << std::endl;

        // Initialize all host matrices
        initializeHostMatricesFromSystem();
        // Then output all matrices to file. If --output was not passed in argv, this method outputs everything.
        outputInitialMatrices();
        // Copy remaining stuff to Device.
        initializeDeviceMatricesFromHost();
    }

    void initializeHostMatricesFromSystem();               // Evaluates the envelopes and initializes the host matrices
    void initializeDeviceMatricesFromHost();               // Transfers the host matrices to their device equivalents

    // Output (Final) Host Matrices to files
    void outputMatrices( const Type::uint start_x, const Type::uint end_x, const Type::uint start_y, const Type::uint end_y, const Type::uint increment, const std::string& suffix = "", const std::string& prefix = "" );
    // Output Initial Host Matrices to files
    void outputInitialMatrices();

    // Output the history and max caches to files. should be called from finalize()
    void cacheToFiles();

    void finalize();

    void iterateNewton( dim3 block_size, dim3 grid_size );
    void iterateFixedTimestepRungeKutta3( dim3 block_size, dim3 grid_size );
    void iterateFixedTimestepRungeKutta4( dim3 block_size, dim3 grid_size );
    void iterateVariableTimestepRungeKutta( dim3 block_size, dim3 grid_size );
    void iterateSplitStepFourier( dim3 block_size, dim3 grid_size );
    void normalizeImaginaryTimePropagation( dim3 block_size, dim3 grid_size );

    struct iteratorFunction {
        int k_max;
        std::function<void( dim3, dim3 )> iterate;
    };
    std::map<std::string, iteratorFunction> iterator = {
        { "newton", { 1, std::bind( &Solver::iterateNewton, this, std::placeholders::_1, std::placeholders::_2 ) } },
        { "rk3", { 3, std::bind( &Solver::iterateFixedTimestepRungeKutta3, this, std::placeholders::_1, std::placeholders::_2 ) } },
        { "rk4", { 4, std::bind( &Solver::iterateFixedTimestepRungeKutta4, this, std::placeholders::_1, std::placeholders::_2 ) } },
        { "rk45", { 6, std::bind( &Solver::iterateVariableTimestepRungeKutta, this, std::placeholders::_1, std::placeholders::_2 ) } },
        { "ssfm", { 2, std::bind( &Solver::iterateSplitStepFourier, this, std::placeholders::_1, std::placeholders::_2 ) } }
    };

    bool iterate();

    void applyFFTFilter( dim3 block_size, dim3 grid_size, bool apply_mask = true );

    enum class FFT {
        inverse,
        forward
    };
    void calculateFFT( Type::complex* device_ptr_in, Type::complex* device_ptr_out, FFT dir );

    void swapBuffers();

    void cacheValues();
    void cacheMatrices();

    // The block size is specified by the user in the system.block_size variable.
    // This solver function the calculates the appropriate grid size for the given execution range.
    std::pair<dim3,dim3> getLaunchParameters( const Type::uint N_x, const Type::uint N_y = 1 ) {
        dim3 block_size = { system.block_size, 1, 1 };
        dim3 grid_size = { ( N_x * N_y + block_size.x ) / block_size.x, 1, 1 };
        return { block_size, grid_size };
    }

};

// Helper macro to choose the correct runge function
#define RUNGE_FUNCTION_GP (system.p.use_twin_mode ? PC3::Kernel::Compute::gp_tetm : PC3::Kernel::Compute::gp_scalar)
#define RUNGE_FUNCTION_GP_LINEAR (system.p.use_twin_mode ? PC3::Kernel::Compute::gp_tetm_linear_fourier : PC3::Kernel::Compute::gp_scalar_linear_fourier)
#define RUNGE_FUNCTION_GP_NONLINEAR (system.p.use_twin_mode ? PC3::Kernel::Compute::gp_tetm_nonlinear : PC3::Kernel::Compute::gp_scalar_nonlinear)
#define RUNGE_FUNCTION_GP_INDEPENDENT (system.p.use_twin_mode ? PC3::Kernel::Compute::gp_tetm_independent : PC3::Kernel::Compute::gp_scalar_independent)

struct SquareReduction
{
    PULSE_HOST_DEVICE PC3::Type::real operator()(const PC3::Type::complex& x) const { 
        const PC3::Type::real res = PC3::CUDA::abs2(x);
        return res; 
    }
};

} // namespace PC3