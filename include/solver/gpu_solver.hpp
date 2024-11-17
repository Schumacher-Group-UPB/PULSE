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

namespace PHOENIX {

/** 
 * @brief GPU Solver class providing the interface for the GPU solver.
 * Implements RK4, RK45, FFT calculations.
 *
 */
class Solver {
   public:
    // References to system and filehandler so we dont need to pass them around all the time
    PHOENIX::SystemParameters& system;
    PHOENIX::FileHandler& filehandler;

    // TODO: amp zu Type::device_vector. cudamatrix not needed
    struct TemporalEvelope {
        Type::device_vector<Type::complex> amp;

        struct Pointers {
            Type::complex* amp;
            Type::uint32 n;
        };

        Pointers pointers() {
            return Pointers{ GET_RAW_PTR( amp ), Type::uint32( amp.size() ) };
        }
    } dev_pulse_oscillation, dev_pump_oscillation, dev_potential_oscillation;

    // Host/Device Matrices
    MatrixContainer matrix;

    struct InputOutput {
        Type::complex* PHOENIX_RESTRICT in_wf_plus = nullptr;
        Type::complex* PHOENIX_RESTRICT in_wf_minus = nullptr;
#ifdef BENCH
        Type::complex* PHOENIX_RESTRICT in_wf_plus_i = nullptr;
        Type::complex* PHOENIX_RESTRICT in_wf_minus_i = nullptr;
#endif
        Type::complex* PHOENIX_RESTRICT in_rv_plus = nullptr;
        Type::complex* PHOENIX_RESTRICT in_rv_minus = nullptr;
        Type::complex* PHOENIX_RESTRICT out_wf_plus = nullptr;
        Type::complex* PHOENIX_RESTRICT out_wf_minus = nullptr;
        Type::complex* PHOENIX_RESTRICT out_rv_plus = nullptr;
        Type::complex* PHOENIX_RESTRICT out_rv_minus = nullptr;
    };

    Type::device_vector<Type::real> time; // [0] is t, [1] is dt

    struct KernelArguments {
        TemporalEvelope::Pointers pulse_pointers;     // The pointers to the envelopes. These are obtained by calling the .pointers() method on the envelopes.
        TemporalEvelope::Pointers pump_pointers;      // The pointers to the envelopes. These are obtained by calling the .pointers() method on the envelopes.
        TemporalEvelope::Pointers potential_pointers; // The pointers to the envelopes. These are obtained by calling the .pointers() method on the envelopes.
        Type::real* time;                             // Pointer to Device Memory of the time array. [0] is t, [1] is dt
        MatrixContainer::Pointers dev_ptrs;           // All the pointers to the matrices. These are obtained by calling the .pointers() method on the matrices.
        SystemParameters::KernelParameters p;         // The kernel parameters. These are obtained by copying the kernel_parameters object of the system.
    };

    // Fixed Kernel Arguments. Every Compute Kernel will take one of these.
    KernelArguments generateKernelArguments( const Type::uint32 subgrid = 0 ) {
        auto kernel_arguments = KernelArguments();
        kernel_arguments.pulse_pointers = dev_pulse_oscillation.pointers();
        kernel_arguments.pump_pointers = dev_pump_oscillation.pointers();
        kernel_arguments.potential_pointers = dev_potential_oscillation.pointers();
        kernel_arguments.dev_ptrs = matrix.pointers( subgrid );
        kernel_arguments.p = system.kernel_parameters;
        kernel_arguments.time = GET_RAW_PTR( time );
        return kernel_arguments;
    }

    // Cache Maps
    std::map<std::string, std::vector<Type::real>> cache_map_scalar;

    Solver( PHOENIX::SystemParameters& system ) : system( system ), filehandler( system.filehandler ) {
        std::cout << PHOENIX::CLIO::prettyPrint( "Creating Solver...", PHOENIX::CLIO::Control::Info ) << std::endl;
        // Initialize all matrices
        initializeMatricesFromSystem();
        // Then output all matrices to file. If --output was not passed in argv, this method outputs everything.
#ifndef BENCH
        outputInitialMatrices();
#endif
    }

    void initializeMatricesFromSystem(); // Evaluates the envelopes and initializes the matrices
    void initializeHaloMap();            // Initializes the halo map

    // Output (Final) Host Matrices to files
    void outputMatrices( const Type::uint32 start_x, const Type::uint32 end_x, const Type::uint32 start_y, const Type::uint32 end_y, const Type::uint32 increment, const std::string& suffix = "", const std::string& prefix = "" );
    // Output Initial Host Matrices to files
    void outputInitialMatrices();

    // Output the history and max caches to files. should be called from finalize()
    void cacheToFiles();

    void finalize();

    void iterateNewton();
    void iterateFixedTimestepRungeKutta3();
    void iterateFixedTimestepRungeKutta4();
    void iterateVariableTimestepRungeKutta();
    void iterateSplitStepFourier();
    void normalizeImaginaryTimePropagation();

    struct iteratorFunction {
        int k_max;
        std::function<void()> iterate;
    };
    std::map<std::string, iteratorFunction> iterator = { { "newton", { 1, std::bind( &Solver::iterateNewton, this ) } }, { "rk4", { 4, std::bind( &Solver::iterateFixedTimestepRungeKutta4, this ) } }, { "ssfm", { 0, std::bind( &Solver::iterateSplitStepFourier, this ) } } };

    // Main System function. Either gp_scalar or gp_tetm.
    // Both functions have signature void(int i, Type::uint32 current_halo, Solver::VKernelArguments time, Solver::KernelArguments args, Solver::InputOutput io)
    std::function<void( int, Type::uint32, KernelArguments, InputOutput )> runge_function;

    bool iterate();

    void applyFFTFilter( bool apply_mask = true );

    enum class FFT { inverse, forward };
    void calculateFFT( Type::complex* device_ptr_in, Type::complex* device_ptr_out, FFT dir );

    void cacheValues();
    void cacheMatrices();

    // The block size is specified by the user in the system.block_size variable.
    // This solver function the calculates the appropriate grid size for the given execution range.
    std::pair<dim3, dim3> getLaunchParameters( const Type::uint32 N_c, const Type::uint32 N_r = 1 ) {
#ifdef USE_CPU
        dim3 block_size = { N_r, 1, 1 };
        dim3 grid_size = { N_c, 1, 1 };
#else
        dim3 block_size = { system.block_size, 1, 1 };
        dim3 grid_size = { ( N_c * N_r + block_size.x ) / block_size.x, 1, 1 };
#endif
        return { block_size, grid_size };
    }
};

} // namespace PHOENIX
