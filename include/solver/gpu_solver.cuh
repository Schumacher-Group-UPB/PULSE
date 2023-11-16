#pragma once

#include <iostream>
#include "cuda/cuda_complex.cuh"
#include "cuda/cuda_matrix.cuh"
#include "cuda/cuda_macro.cuh"
#include "kernel/kernel_fft.cuh"
#include "system/system.hpp"
#include "system/filehandler.hpp"
#include "solver/device_struct.hpp"
#include "solver/host_struct.hpp"

namespace PC3 {



#define INSTANCIATE_K( index, twin) \
    device.k##index##_wavefunction_plus.construct(system.s_N, "device.k" #index "_wavefunction_plus"); \
    device.k##index##_reservoir_plus.construct(system.s_N, "device.k" #index "_reservoir_plus"); \
    if (twin) { \
        device.k##index##_wavefunction_minus.construct(system.s_N, "device.k" #index "_wavefunction_minus"); \
        device.k##index##_reservoir_minus.construct(system.s_N, "device.k" #index "_reservoir_minus"); \
    }

/**
 * @brief GPU Solver class providing the interface for the GPU solver.
 * Implements RK4, RK45, FFT calculations.
 * 
 */
class Solver {

    public:

    // References to system and filehandler so we dont need to pass them around all the time
    PC3::System& system;
    PC3::FileHandler& filehandler;

    // Pulse Parameters. For now, we just cache the pulse parameters on the device.
    // Later, we want to also cache the pulse shapes. This is harder to do than
    // with the pump, since the user can provide arbitrary amounts of different 
    // pulses. My current idea on how to handle this is to just limit the number
    // of pulses to e.g. 10, and then construct and cache the pulse shapes only when
    // needed. This is not a problem for the pump, since we only have one pump.
    struct PulseParameters {
        PC3::CUDAMatrix<real_number> amp;
        PC3::CUDAMatrix<real_number> sigma;
        PC3::CUDAMatrix<real_number> x;
        PC3::CUDAMatrix<real_number> y;
        PC3::CUDAMatrix<int> m;
        PC3::CUDAMatrix<int> pol;
        PC3::CUDAMatrix<real_number> t0;
        PC3::CUDAMatrix<real_number> freq;
        PC3::CUDAMatrix<real_number> width;
        unsigned int n;
    } dev_pulse_parameters;

    // Device Variables
    Device device;
    // Host Variables
    Host host;

    // FFT Plan
    cuda_fft_plan plan;

    // If true, we have a twin system. If false, we have a scalar system.
    bool use_te_tm_splitting;

    enum class Symmetry { Scalar, TETM };

    Solver( PC3::System& system, Symmetry scalar_or_twin ) : system(system), filehandler(system.filehandler) {
        use_te_tm_splitting = (scalar_or_twin == Symmetry::TETM);

        std::cout << "Creating Solver with TE/TM Splitting: " << use_te_tm_splitting << std::endl;

        // Instanciate all Psi Matrices. Only Instanciate the Minus components if we have a twin system.
        device.wavefunction_plus.construct(system.s_N, "device.wavefunction_plus");
        device.reservoir_plus.construct(system.s_N, "device.reservoir_plus");
        device.buffer_wavefunction_plus.construct(system.s_N, "device.buffer_wavefunction_plus");
        // Construct Pump and Pulse cache (TODO: Pulse cache)
        device.pump_plus.construct(system.s_N, "device.pump_plus");
        
        // Construct the _minus components only when we have a TE/TM splitting system
        if (use_te_tm_splitting) {
            device.wavefunction_minus.construct(system.s_N, "device.wavefunction_minus");
            device.reservoir_minus.construct(system.s_N, "device.reservoir_minus");
            device.buffer_wavefunction_minus.construct(system.s_N, "device.buffer_wavefunction_minus");
            device.pump_minus.construct(system.s_N, "device.pump_minus");
        }

        
        // Construct K1 to K4 Matrices for RK4
        INSTANCIATE_K( 1, use_te_tm_splitting );
        INSTANCIATE_K( 2, use_te_tm_splitting );
        INSTANCIATE_K( 3, use_te_tm_splitting );
        INSTANCIATE_K( 4, use_te_tm_splitting );
        // Construct K5-K7 and the RK Error Matrix for RK45
        if (not system.fixed_time_step) {
            INSTANCIATE_K( 5, use_te_tm_splitting );
            INSTANCIATE_K( 6, use_te_tm_splitting );
            INSTANCIATE_K( 7, use_te_tm_splitting );
            // Single Error Matrix for RK45
            device.rk_error.construct(system.s_N, "device.rk_error");
        }

        // Finally, initialize the FFT Plan
        CUDA_FFT_CREATE(&plan, system.s_N );

        // Initialize all host matrices
        initializeHostMatricesFromSystem();
        // Overwrite them by loading Matrices from File. If --input was not passed in argv, this method does nothing.
        loadMatrices();
        // Then output all matrices to file. If --output was not passed in argv, this method outputs everything.
        outputInitialMatrices();
        // Copy to Device. 
        initializeDeviceMatricesFromHost();
    }

    ~Solver() {
        CUDA_FFT_DESTROY( plan );
    }

    // Functions the global kernel can do

    /**
     * Initializes the FFT Mask device cache matrices. 
     * If twin_system is used, initializes the _plus and _minus components of 
     * the mask. If not, only initializes the _plus component.
     * Excepts the system host components to be initialized.
    */
    void initializeHostMatricesFromSystem(); // Evaluates the envelopes and initializes the host matrices
    void initializeDeviceParametersFromSystemParameters(); // Transfers the host parameters to their device equivalents
    void initializeDeviceMatricesFromHost( ); // Transfers the host matrices to their device equivalents
    
    // Output (Final) Host Matrices to files
    void outputMatrices();
    // Load Host Matrices from files
    void loadMatrices();
    // Output Initial Host Matrices to files
    void outputInitialMatrices();

    // Output the history and max caches to files. should be called from finalize()
    void cacheToFiles();

    void calculateSollValues();

    // "Syncs" or copies the device arrays to host.
    void syncDeviceArrays();

    void finalize();

    void iterateFixedTimestepRungeKutta( bool evaluate_pulse, dim3 block_size, dim3 grid_size );
    void iterateVariableTimestepRungeKutta( bool evaluate_pulse, dim3 block_size, dim3 grid_size );
    void iterateRungeKutta( bool evaluate_pulse );

    void applyFFTFilter( dim3 block_size, dim3 grid_size );

    void cacheValues();

};

} // namespace PC3