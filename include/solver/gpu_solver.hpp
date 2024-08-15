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
#include "solver/matrix_container.hpp"
#include "misc/escape_sequences.hpp"

namespace PC3 {

#define INSTANCIATE_K( index, twin )                                                                           \
    matrix.k##index##_wavefunction_plus.constructDevice( system.N_x, system.N_y, "matrix.k" #index "_wavefunction_plus" );       \
    matrix.k##index##_reservoir_plus.constructDevice( system.N_x, system.N_y, "matrix.k" #index "_reservoir_plus" );             \
    if ( twin ) {                                                                                              \
        matrix.k##index##_wavefunction_minus.constructDevice( system.N_x, system.N_y, "matrix.k" #index "_wavefunction_minus" ); \
        matrix.k##index##_reservoir_minus.constructDevice( system.N_x, system.N_y, "matrix.k" #index "_reservoir_minus" );       \
    }

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

    struct TemporalEvelope {
        PC3::CUDAMatrix<Type::complex> amp;

        struct Pointers {
            Type::complex* amp;
            unsigned int n;
        };

        Pointers pointers() {
            return Pointers{ amp.getDevicePtr(), amp.getTotalSize() };
        }
    } dev_pulse_oscillation, dev_pump_oscillation, dev_potential_oscillation;

    // Host/Device Matrices
    MatrixContainer matrix;

    // Since the Solver knows about the System class, the IO class, the Kernels and the Matrices, it can combine all of them into a single struct that is then allocated on the device.
    // When modifying the System, IO, Kernels or Matrices, make sure to update this struct as well.
    // Helper Struct to pass input-target data pointers to the kernel
    struct InputOutput {
        Type::complex* PULSE_RESTRICT in_wf_plus;
        Type::complex* PULSE_RESTRICT in_wf_minus;
        Type::complex* PULSE_RESTRICT in_rv_plus;
        Type::complex* PULSE_RESTRICT in_rv_minus;
        Type::complex* PULSE_RESTRICT out_wf_plus;
        Type::complex* PULSE_RESTRICT out_wf_minus;
        Type::complex* PULSE_RESTRICT out_rv_plus;
        Type::complex* PULSE_RESTRICT out_rv_minus;
    };

    struct KernelArguments {
        TemporalEvelope::Pointers pulse_pointers; // The pointers to the envelopes. These are obtained by calling the .pointers() method on the envelopes.
        TemporalEvelope::Pointers pump_pointers; // The pointers to the envelopes. These are obtained by calling the .pointers() method on the envelopes.
        TemporalEvelope::Pointers potential_pointers; // The pointers to the envelopes. These are obtained by calling the .pointers() method on the envelopes.
        InputOutput io; // All the pointers to the input/output matrices. These are manually configured.
        MatrixContainer::Pointers dev_ptrs; // All the pointers to the matrices. These are obtained by calling the .pointers() method on the matrices.
        SystemParameters::KernelParameters p; // The kernel parameters. These are obtained by copying the kernel_parameters object of the system.
        Type::complex dt; // dt is set explicitely
        Type::complex t; // t is set explicitely
    } kernel_arguments;

    void updateKernelArguments( Type::complex t, Type::complex dt ) {
        kernel_arguments.t = t;
        kernel_arguments.dt = dt;
    }

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

        kernel_arguments.pulse_pointers = dev_pulse_oscillation.pointers();
        kernel_arguments.pump_pointers = dev_pump_oscillation.pointers();
        kernel_arguments.potential_pointers = dev_potential_oscillation.pointers();
        kernel_arguments.dev_ptrs = matrix.pointers();
        kernel_arguments.p = system.kernel_parameters;
    }

    void initializeHostMatricesFromSystem();               // Evaluates the envelopes and initializes the host matrices
    void initializeDeviceMatricesFromHost();               // Transfers the host matrices to their device equivalents

    // Output (Final) Host Matrices to files
    void outputMatrices( const unsigned int start_x, const unsigned int end_x, const unsigned int start_y, const unsigned int end_y, const unsigned int increment, const std::string& suffix = "", const std::string& prefix = "" );
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
};

// Helper macro to choose the correct runge function
#define RUNGE_FUNCTION_GP (system.p.use_twin_mode ? PC3::Kernel::Compute::gp_tetm : PC3::Kernel::Compute::gp_scalar)
#define RUNGE_FUNCTION_GP_LINEAR (system.p.use_twin_mode ? PC3::Kernel::Compute::gp_tetm_linear_fourier : PC3::Kernel::Compute::gp_scalar_linear_fourier)
#define RUNGE_FUNCTION_GP_NONLINEAR (system.p.use_twin_mode ? PC3::Kernel::Compute::gp_tetm_nonlinear : PC3::Kernel::Compute::gp_scalar_nonlinear)
#define RUNGE_FUNCTION_GP_INDEPENDENT (system.p.use_twin_mode ? PC3::Kernel::Compute::gp_tetm_independent : PC3::Kernel::Compute::gp_scalar_independent)

// Helper Macro to iterate a specific RK K
#define CALCULATE_K( index, input_wavefunction, input_reservoir ) \
CALL_KERNEL( \
    RUNGE_FUNCTION_GP, "K"#index, grid_size, block_size, stream,  \
    kernel_arguments, {      \
kernel_arguments.dev_ptrs.input_wavefunction##_plus, kernel_arguments.dev_ptrs.input_wavefunction##_minus, kernel_arguments.dev_ptrs.input_reservoir##_plus, kernel_arguments.dev_ptrs.input_reservoir##_minus, \
kernel_arguments.dev_ptrs.k##index##_wavefunction_plus, kernel_arguments.dev_ptrs.k##index##_wavefunction_minus, kernel_arguments.dev_ptrs.k##index##_reservoir_plus, kernel_arguments.dev_ptrs.k##index##_reservoir_minus \
} \
);

#define INTERMEDIATE_SUM_K( index, ... ) \
CALL_KERNEL( \
    Kernel::RK::runge_sum_to_input_kw, "Sum for K"#index, grid_size, block_size, stream, \
    kernel_arguments, {kernel_arguments.dev_ptrs.wavefunction_plus, kernel_arguments.dev_ptrs.wavefunction_minus, \
            kernel_arguments.dev_ptrs.reservoir_plus, kernel_arguments.dev_ptrs.reservoir_minus, \
            kernel_arguments.dev_ptrs.buffer_wavefunction_plus, kernel_arguments.dev_ptrs.buffer_wavefunction_minus, \
            kernel_arguments.dev_ptrs.buffer_reservoir_plus, kernel_arguments.dev_ptrs.buffer_reservoir_minus },\
    {__VA_ARGS__} \
);

#define FINAL_SUM_K( ... ) \
CALL_KERNEL( \
    Kernel::RK::runge_sum_to_input_kw, "Sum for Psi", grid_size, block_size, stream, \
    kernel_arguments, {kernel_arguments.dev_ptrs.wavefunction_plus, kernel_arguments.dev_ptrs.wavefunction_minus, \
            kernel_arguments.dev_ptrs.reservoir_plus, kernel_arguments.dev_ptrs.reservoir_minus, \
            kernel_arguments.dev_ptrs.wavefunction_plus, kernel_arguments.dev_ptrs.wavefunction_minus, \
            kernel_arguments.dev_ptrs.reservoir_plus, kernel_arguments.dev_ptrs.reservoir_minus },\
    {__VA_ARGS__} \
);

struct SquareReduction
{
    PULSE_HOST_DEVICE PC3::Type::real operator()(const PC3::Type::complex& x) const { 
        const PC3::Type::real res = PC3::CUDA::abs2(x);
        return res; 
    }
};

} // namespace PC3