#pragma once

#include <iostream>
#include <map>
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

    // TODO: Move these into one single float buffer.
    struct Oscillation {
        PC3::CUDAMatrix<Type::real> t0;
        PC3::CUDAMatrix<Type::real> freq;
        PC3::CUDAMatrix<Type::real> sigma;
        std::vector<bool> active;
        unsigned int n;

        struct Pointers {
            Type::real* t0;
            Type::real* freq;
            Type::real* sigma;
            unsigned int n;
        };

        void construct( Envelope& envelope) {
            const auto n = envelope.groupSize();
            t0.construct( n, 1, "oscillation_t0" ).setTo( envelope.t0 );
            freq.construct( n, 1, "oscillation_freq" ).setTo( envelope.freq );
            sigma.construct( n, 1, "oscillation_sigma" ).setTo( envelope.sigma );
            this->n = n;
            for (int i = 0; i < n; i++) {
                bool is_active = true;
                if ( envelope.t0.data()[i] == 0.0 && envelope.freq.data()[i] == 0.0 && envelope.sigma.data()[i] > 1E11 ) {
                    is_active = false;
                }
                active.push_back( is_active );
            } 
        }

        Pointers pointers() {
            return Pointers{ t0.getDevicePtr(), freq.getDevicePtr(), sigma.getDevicePtr(), n };
        }
    } dev_pulse_oscillation, dev_pump_oscillation, dev_potential_oscillation;

    // Device Variables
    MatrixContainer matrix;

    // Cache Maps
    std::map<std::string, std::vector<Type::real>> cache_map_scalar;
    //std::vector<std::vector<Type::complex>> wavefunction_plus_history, wavefunction_minus_history;
    //std::vector<Type::real> wavefunction_max_plus, wavefunction_max_minus;
    //std::vector<Type::real> times;

    Solver( PC3::SystemParameters& system ) : system( system ), filehandler( system.filehandler ) {
        std::cout << PC3::CLIO::prettyPrint( "Creating Solver...", PC3::CLIO::Control::Info ) << std::endl;

        // Initialize all host matrices
        initializeHostMatricesFromSystem();
        // Then output all matrices to file. If --output was not passed in argv, this method outputs everything.
        outputInitialMatrices();
        // Copy remaining stuff to Device.
        initializeDeviceMatricesFromHost();

        // Set Pump, Potential and Pulse n to zero if they are not evaluated
        if (not system.evaluate_reservoir_kernel) {
            std::cout << PC3::CLIO::prettyPrint( "Reservoir is not evaluated, setting n to zero.", PC3::CLIO::Control::Info ) << std::endl;
            dev_pump_oscillation.n = 0;
        }
        if (not system.evaluate_potential_kernel) {
            std::cout << PC3::CLIO::prettyPrint( "Potential is not evaluated, setting n to zero.", PC3::CLIO::Control::Info ) << std::endl;
            dev_potential_oscillation.n = 0;
        } 
        if (not system.evaluate_pulse_kernel) {
            std::cout << PC3::CLIO::prettyPrint( "Pulse is not evaluated, setting n to zero.", PC3::CLIO::Control::Info ) << std::endl;
            dev_pulse_oscillation.n = 0;
        }
    }
    
    // Functions the global kernel can do

    /**
     * Initializes the FFT Mask device cache matrices.
     * If twin_system is used, initializes the _plus and _minus components of
     * the mask. If not, only initializes the _plus component.
     * Excepts the system host components to be initialized.
     */
    void initializeHostMatricesFromSystem();               // Evaluates the envelopes and initializes the host matrices
    void initializeDeviceMatricesFromHost();               // Transfers the host matrices to their device equivalents

    // Output (Final) Host Matrices to files
    void outputMatrices( const unsigned int start_x, const unsigned int end_x, const unsigned int start_y, const unsigned int end_y, const unsigned int increment, const std::string& suffix = "", const std::string& prefix = "" );
    // Output Initial Host Matrices to files
    void outputInitialMatrices();

    // Output the history and max caches to files. should be called from finalize()
    void cacheToFiles();

    void finalize();

    // TODO: grid and block size can be system (or solver) variables
    void iterateFixedTimestepRungeKutta( dim3 block_size, dim3 grid_size );
    void iterateVariableTimestepRungeKutta( dim3 block_size, dim3 grid_size );
    void iterateSplitStepFourier( dim3 block_size, dim3 grid_size );
    void iterateImaginaryTimePropagation( dim3 block_size, dim3 grid_size );
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

// TODO: move these to a Header that makes sense
namespace CUDA {

    PULSE_HOST_DEVICE static PULSE_INLINE Type::complex gaussian_complex_oscillator( Type::real t, Type::real t0, Type::real sigma, Type::real freq ) {
        return CUDA::exp( -Type::complex(( t - t0 )*( t - t0 ) / ( Type::real(2.0)*sigma*sigma ), freq * ( t - t0 )) );
    }
    PULSE_HOST_DEVICE static PULSE_INLINE Type::real gaussian_oscillator( Type::real t, Type::real t0, Type::real sigma, Type::real freq ) {
        const auto p = ( t - t0 )/sigma;
        return std::exp( -0.5*p*p ) * (1.0 + std::cos( freq * ( t - t0 ) ))/2.0;
    }

} // namespace CUDA

// Helper macro to choose the correct runge function
#define RUNGE_FUNCTION_GP (p.use_twin_mode ? PC3::Kernel::Compute::gp_tetm : PC3::Kernel::Compute::gp_scalar)

// Helper Macro to iterate a specific RK K
#define CALCULATE_K( index, time, input_wavefunction, input_reservoir ) \
CALL_KERNEL( \
    RUNGE_FUNCTION_GP, "K"#index, grid_size, block_size,  \
    time, device_pointers, p, pulse_pointers, pump_pointers, potential_pointers, \
    {  \
        device_pointers.input_wavefunction##_plus, device_pointers.input_wavefunction##_minus, device_pointers.input_reservoir##_plus, device_pointers.input_reservoir##_minus, \
        device_pointers.k##index##_wavefunction_plus, device_pointers.k##index##_wavefunction_minus, device_pointers.k##index##_reservoir_plus, device_pointers.k##index##_reservoir_minus \
    } \
);


} // namespace PC3