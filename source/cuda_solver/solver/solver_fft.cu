#include "cuda/cuda_complex.cuh"
#include "cuda/cuda_macro.cuh"
#include "kernel/kernel_fft.cuh"
#include "system/system.hpp"
#include "solver/gpu_solver.hpp"

#ifdef USECPU
#ifndef PC3_DISABLE_FFT
    #include <fftw3.h>
#endif
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
    CALL_KERNEL( fft_shift_2D, "FFT Shift Plus", grid_size, block_size, 
        matrix.fft_plus.getDevicePtr(), system.p.N_x, system.p.N_y 
    );

    // Do the FFT and the shifting here already for visualization only
    if ( system.p.use_twin_mode ) {
        calculateFFT( matrix.wavefunction_minus.getDevicePtr(), matrix.fft_minus.getDevicePtr(), FFT::forward );
        
        CALL_KERNEL( fft_shift_2D, "FFT Shift Minus", grid_size, block_size, 
            matrix.fft_minus.getDevicePtr(), system.p.N_x, system.p.N_y 
        );
    }
    
    if (not apply_mask)
        return;
    
    // Apply the FFT Mask Filter
    CALL_KERNEL(kernel_mask_fft, "FFT Mask Plus", grid_size, block_size, 
        matrix.fft_plus.getDevicePtr(), matrix.fft_mask_plus.getDevicePtr(), system.p.N_x*system.p.N_y
    );
    
    // Undo the shift
    CALL_KERNEL( fft_shift_2D, "FFT Shift Plus", grid_size, block_size, 
         matrix.fft_plus.getDevicePtr(), system.p.N_x, system.p.N_y 
    );

    // Transform back.
    calculateFFT(  matrix.fft_plus.getDevicePtr(), matrix.wavefunction_plus.getDevicePtr(), FFT::inverse );
    
    // Shift FFT Once again for visualization
    CALL_KERNEL( fft_shift_2D, "FFT Shift Plus", grid_size, block_size, 
        matrix.fft_plus.getDevicePtr(), system.p.N_x, system.p.N_y 
    );
    
    // Do the same for the minus component
    if (not system.p.use_twin_mode)
        return;

    CALL_KERNEL(kernel_mask_fft, "FFT Mask Plus", grid_size, block_size, 
        matrix.fft_minus.getDevicePtr(), matrix.fft_mask_minus.getDevicePtr(), system.p.N_x*system.p.N_y 
    );

    CALL_KERNEL( fft_shift_2D, "FFT Minus Plus", grid_size, block_size, 
        matrix.fft_minus.getDevicePtr(), system.p.N_x,system.p.N_y 
    );
    
    calculateFFT( matrix.fft_minus.getDevicePtr(), matrix.wavefunction_minus.getDevicePtr(), FFT::inverse );

    CALL_KERNEL( fft_shift_2D, "FFT Minus Plus", grid_size, block_size, 
        matrix.fft_minus.getDevicePtr(), system.p.N_x,system.p.N_y 
    );

}

void PC3::Solver::calculateFFT( complex_number* device_ptr_in, complex_number* device_ptr_out, FFT dir ) {
    #ifndef USECPU
        // Do FFT using CUDAs FFT functions
        CHECK_CUDA_ERROR( FFTSOLVER( plan, device_ptr_in, device_ptr_out, dir == FFT::inverse ? CUFFT_INVERSE : CUFFT_FORWARD ), "FFT Exec" );
    #else   
        #ifndef PC3_DISABLE_FFT 
        // Do FFT on CPU using external Library.
        fftw_plan plan = fftw_plan_dft_2d(system.p.N_x, system.p.N_y,
                                      reinterpret_cast<fftw_complex*>(device_ptr_in),
                                      reinterpret_cast<fftw_complex*>(device_ptr_out),
                                      dir == FFT::inverse ? FFTW_BACKWARD : FFTW_FORWARD, FFTW_ESTIMATE);
        fftw_execute(plan);
        fftw_destroy_plan(plan);
        fftw_cleanup();
        #else
        #warning FFTW is disabled!
        #endif
    #endif
}