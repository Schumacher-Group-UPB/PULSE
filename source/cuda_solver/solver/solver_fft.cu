#include "cuda/typedef.cuh"
#include "kernel/kernel_fft.cuh"
#include "system/system_parameters.hpp"
#include "solver/gpu_solver.hpp"
#include "misc/commandline_io.hpp"

#ifdef USE_CPU

    #include <fftw3.h>

#else

    #include <cufft.h>
    #include <curand_kernel.h>
    #define cuda_fft_plan cufftHandle
    #ifdef USE_HALF_PRECISION
        #define FFTSOLVER cufftExecC2C
        #define FFTPLAN CUFFT_C2C
        using fft_type = cufftComplex;
    #else
        #define FFTSOLVER cufftExecZ2Z
        #define FFTPLAN CUFFT_Z2Z
        using fft_type = cufftDoubleComplex;
    #endif

#endif


#ifdef USE_CUDA
    /**
     * Static Helper Function to get the cuFFT Plan. The static variables ensure the
     * fft plan is only created once. We don't destroy the plan and hope the operating
     * system will forgive us. We could also implement a small wrapper class that
     * holds the plan and calls the destruct method when the class instance is destroyed.
    */
    static cufftHandle& getFFTPlan( size_t N_x, size_t N_y ) {
        static cufftHandle plan = 0;
        static bool isInitialized = false;

        if (not isInitialized) {
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
void PC3::Solver::applyFFTFilter( dim3 block_size, dim3 grid_size, bool apply_mask ) {
    
    // Calculate the actual FFTs
    calculateFFT( matrix.wavefunction_plus.getDevicePtr(), matrix.fft_plus.getDevicePtr(), FFT::forward );

    // For now, we shift, transform, shift the results. TODO: Move this into one function without shifting
    // Shift FFT to center k = 0
    CALL_KERNEL( PC3::Kernel::fft_shift_2D, "FFT Shift Plus", grid_size, block_size, 0, // 0 = default stream 
        matrix.fft_plus.getDevicePtr(), system.p.N_x, system.p.N_y 
    );

    // Do the FFT and the shifting here already for visualization only
    if ( system.p.use_twin_mode ) {
        calculateFFT( matrix.wavefunction_minus.getDevicePtr(), matrix.fft_minus.getDevicePtr(), FFT::forward );
        
        CALL_KERNEL( PC3::Kernel::fft_shift_2D, "FFT Shift Minus", grid_size, block_size, 0, // 0 = default stream 
            matrix.fft_minus.getDevicePtr(), system.p.N_x, system.p.N_y 
        );
    }
    
    if (not apply_mask)
        return;
    
    // Apply the FFT Mask Filter
    CALL_KERNEL(PC3::Kernel::kernel_mask_fft, "FFT Mask Plus", grid_size, block_size, 0, // 0 = default stream 
        matrix.fft_plus.getDevicePtr(), matrix.fft_mask_plus.getDevicePtr(), system.p.N_x*system.p.N_y
    );
    
    // Undo the shift
    CALL_KERNEL( PC3::Kernel::fft_shift_2D, "FFT Shift Plus", grid_size, block_size, 0, // 0 = default stream 
         matrix.fft_plus.getDevicePtr(), system.p.N_x, system.p.N_y 
    );

    // Transform back.
    calculateFFT(  matrix.fft_plus.getDevicePtr(), matrix.wavefunction_plus.getDevicePtr(), FFT::inverse );
    
    // Shift FFT Once again for visualization
    CALL_KERNEL( PC3::Kernel::fft_shift_2D, "FFT Shift Plus", grid_size, block_size, 0, // 0 = default stream 
        matrix.fft_plus.getDevicePtr(), system.p.N_x, system.p.N_y 
    );
    
    // Do the same for the minus component
    if (not system.p.use_twin_mode)
        return;

    CALL_KERNEL(PC3::Kernel::kernel_mask_fft, "FFT Mask Plus", grid_size, block_size, 0, // 0 = default stream 
        matrix.fft_minus.getDevicePtr(), matrix.fft_mask_minus.getDevicePtr(), system.p.N_x*system.p.N_y 
    );

    CALL_KERNEL( PC3::Kernel::fft_shift_2D, "FFT Minus Plus", grid_size, block_size, 0, // 0 = default stream 
        matrix.fft_minus.getDevicePtr(), system.p.N_x,system.p.N_y 
    );
    
    calculateFFT( matrix.fft_minus.getDevicePtr(), matrix.wavefunction_minus.getDevicePtr(), FFT::inverse );

    CALL_KERNEL( PC3::Kernel::fft_shift_2D, "FFT Minus Plus", grid_size, block_size, 0, // 0 = default stream 
        matrix.fft_minus.getDevicePtr(), system.p.N_x,system.p.N_y 
    );

}

void PC3::Solver::calculateFFT( Type::complex* device_ptr_in, Type::complex* device_ptr_out, FFT dir ) {
    #ifdef USE_CUDA
        // Do FFT using CUDAs FFT functions
        auto plan = getFFTPlan( system.p.N_x, system.p.N_y );
        CHECK_CUDA_ERROR( FFTSOLVER( plan, reinterpret_cast<fft_type*>(device_ptr_in), reinterpret_cast<fft_type*>(device_ptr_out), dir == FFT::inverse ? CUFFT_INVERSE : CUFFT_FORWARD ), "FFT Exec" );
    #else   
        //auto [plan_forward, plan_inverse] = getFFTPlan(system.p.N_x, system.p.N_y, device_ptr_in, device_ptr_out);
        #ifdef USE_HALF_PRECISION
        auto plan = fftwf_plan_dft_2d(system.p.N_x, system.p.N_y,
                                      reinterpret_cast<fftwf_complex*>(device_ptr_in),
                                      reinterpret_cast<fftwf_complex*>(device_ptr_out),
                                      dir == FFT::inverse ? FFTW_BACKWARD : FFTW_FORWARD, FFTW_ESTIMATE);        
        fftwf_execute(plan);
        fftwf_destroy_plan(plan);
        #else
        auto plan = fftw_plan_dft_2d(system.p.N_x, system.p.N_y,
                                      reinterpret_cast<fftw_complex*>(device_ptr_in),
                                      reinterpret_cast<fftw_complex*>(device_ptr_out),
                                      dir == FFT::inverse ? FFTW_BACKWARD : FFTW_FORWARD, FFTW_ESTIMATE);        
        fftw_execute(plan);
        fftw_destroy_plan(plan);
        #endif
    #endif
}