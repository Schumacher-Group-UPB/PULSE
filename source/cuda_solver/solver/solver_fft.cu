#include "cuda/typedef.cuh"
#include "kernel/kernel_fft.cuh"
#include "system/system_parameters.hpp"
#include "solver/gpu_solver.hpp"
#include "misc/commandline_io.hpp"

#ifdef USE_CPU

#    include <fftw3.h>

#else

#    include <cufft.h>
#    include <curand_kernel.h>
#    define cuda_fft_plan cufftHandle
#    ifdef USE_HALF_PRECISION
#        define FFTSOLVER cufftExecC2C
#        define FFTPLAN CUFFT_C2C
using fft_type = cufftComplex;
#    else
#        define FFTSOLVER cufftExecZ2Z
#        define FFTPLAN CUFFT_Z2Z
using fft_type = cufftDoubleComplex;
#    endif

#endif

#ifdef USE_CUDA
/**
 * Static Helper Function to get the cuFFT Plan. The static variables ensure the
 * fft plan is only created once. We don't destroy the plan and hope the operating
 * system will forgive us. We could also implement a small wrapper class that
 * holds the plan and calls the destruct method when the class instance is destroyed.
 */
static cufftHandle& getFFTPlan( PC3::Type::uint32 N_x, PC3::Type::uint32 N_y ) {
    static cufftHandle plan = 0;
    static bool isInitialized = false;

    if ( not isInitialized ) {
        if ( cufftPlan2d( &plan, N_x, N_y, FFTPLAN ) != CUFFT_SUCCESS ) {
            std::cout << PC3::CLIO::prettyPrint( "Error Creating CUDA FFT Plan!", PC3::CLIO::Control::FullError ) << std::endl;
            return plan;
        }
        isInitialized = true;
    }

    return plan;
}

#endif

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

#define fft_template_type PC3::Type::complex,PC3::Type::real

void PC3::Solver::applyFFTFilter( bool apply_mask ) {

    auto [block_size, grid_size] = getLaunchParameters( system.p.N_x, system.p.N_y );

    matrix.wavefunction_plus.toFull( matrix.fft_plus );
    // Calculate the actual FFTs
    calculateFFT( GET_RAW_PTR( matrix.fft_plus ), GET_RAW_PTR( matrix.fft_plus ), FFT::forward );

    // Do the FFT and the shifting here already for visualization only
    if ( system.p.use_twin_mode ) {
        matrix.wavefunction_minus.toFull( matrix.fft_minus );
        calculateFFT( GET_RAW_PTR( matrix.fft_minus ), GET_RAW_PTR( matrix.fft_minus ), FFT::forward );
    }

    if ( not apply_mask )
        return;

    // Apply the FFT Mask Filter
    CALL_FULL_KERNEL( PC3::Kernel::kernel_mask_fft<fft_template_type>, "FFT Mask Plus", grid_size, block_size, 0, // 0 = default stream
                 GET_RAW_PTR( matrix.fft_plus ), GET_RAW_PTR( matrix.fft_mask_plus ), system.p.N_x * system.p.N_y );

    if ( system.p.use_twin_mode ) {
        CALL_FULL_KERNEL( PC3::Kernel::kernel_mask_fft<fft_template_type>, "FFT Mask Minus", grid_size, block_size, 0, // 0 = default stream
                     GET_RAW_PTR( matrix.fft_minus ), GET_RAW_PTR( matrix.fft_mask_minus ), system.p.N_x * system.p.N_y );
    }

    // Transform back.
    calculateFFT( GET_RAW_PTR( matrix.fft_plus ), matrix.wavefunction_plus.fullMatrixPointer(), FFT::inverse );
    matrix.wavefunction_plus.toSubgrids();

    if ( system.p.use_twin_mode ) {
        calculateFFT( GET_RAW_PTR( matrix.fft_minus ), matrix.wavefunction_minus.fullMatrixPointer(), FFT::inverse );
        matrix.wavefunction_minus.toSubgrids();
    }
}

void PC3::Solver::calculateFFT( Type::complex* device_ptr_in, Type::complex* device_ptr_out, FFT dir ) {
#ifdef USE_CUDA
    // Do FFT using CUDAs FFT functions
    auto plan = getFFTPlan( system.p.N_x, system.p.N_y );
    CHECK_CUDA_ERROR( FFTSOLVER( plan, reinterpret_cast<fft_type*>( device_ptr_in ), reinterpret_cast<fft_type*>( device_ptr_out ), dir == FFT::inverse ? CUFFT_INVERSE : CUFFT_FORWARD ), "FFT Exec" );
#else
// auto [plan_forward, plan_inverse] = getFFTPlan(system.p.N_x, system.p.N_y, device_ptr_in, device_ptr_out);
#    ifdef USE_HALF_PRECISION
    auto plan = fftwf_plan_dft_2d( system.p.N_x, system.p.N_y,
                                   reinterpret_cast<fftwf_complex*>( device_ptr_in ),
                                   reinterpret_cast<fftwf_complex*>( device_ptr_out ),
                                   dir == FFT::inverse ? FFTW_BACKWARD : FFTW_FORWARD, FFTW_ESTIMATE );
    fftwf_execute( plan );
    fftwf_destroy_plan( plan );
#    else
    auto plan = fftw_plan_dft_2d( system.p.N_x, system.p.N_y,
                                  reinterpret_cast<fftw_complex*>( device_ptr_in ),
                                  reinterpret_cast<fftw_complex*>( device_ptr_out ),
                                  dir == FFT::inverse ? FFTW_BACKWARD : FFTW_FORWARD, FFTW_ESTIMATE );
    fftw_execute( plan );
    fftw_destroy_plan( plan );
#    endif
#endif
}