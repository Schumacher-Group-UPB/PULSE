#include "kernel/kernel_index_overwrite.cuh"
#include "kernel/kernel_normalize_imaginary_time_propagation.cuh"

/*
    sumf=sum(sum(sum(abs(u).^2)))*(max(diff(x)).^2);  % Fixation of the Norm for the u-component
    u=u.*sqrt(sum0./sumf);
    
    sumfpsi=sum(sum(sum(abs(psi).^2)))*(max(diff(x)).^2);  % Fixation of the Norm for the psi-component
    psi=psi.*sqrt(sum0psi./sumfpsi);
    
    maxn(n)= max(max(max(abs(u))));  
*/

CUDA_GLOBAL void PC3::Kernel::normalize_imaginary_time_propagation(int i, MatrixContainer::Pointers dev_ptrs, System::Parameters p, complex_number normalization_wavefunction, complex_number normalization_reservoir) {
    OVERWRITE_THREAD_INDEX( i );

    // MARK: Normalize Wavefunction and Reservoir Plus
    dev_ptrs.buffer_wavefunction_plus[i] = dev_ptrs.buffer_wavefunction_plus[i] / CUDA::sqrt( CUDA::real(normalization_wavefunction) );//* CUDA::abs2( dev_ptrs.buffer_wavefunction_plus[(i+1)%p.N2] - dev_ptrs.buffer_wavefunction_plus[i] ) );
    dev_ptrs.buffer_reservoir_plus[i] = dev_ptrs.buffer_reservoir_plus[i] / CUDA::sqrt( CUDA::real(normalization_reservoir) );//* CUDA::abs2( dev_ptrs.buffer_reservoir_plus[(i+1)%p.N2] - dev_ptrs.buffer_reservoir_plus[i] ) );
    
    if ( not p.use_twin_mode ) 
        return;
    // MARK: Normalize Wavefunction and Reservoir Minus
    dev_ptrs.buffer_wavefunction_minus[i] = dev_ptrs.buffer_wavefunction_minus[i] / CUDA::sqrt( CUDA::imag(normalization_wavefunction) );//* CUDA::abs2( dev_ptrs.buffer_wavefunction_minus[(i+1)%p.N2] - dev_ptrs.buffer_wavefunction_minus[i] ) );
    dev_ptrs.buffer_reservoir_minus[i] = dev_ptrs.buffer_reservoir_minus[i] / CUDA::sqrt( CUDA::imag(normalization_reservoir) );//* CUDA::abs2( dev_ptrs.buffer_reservoir_minus[(i+1)%p.N2] - dev_ptrs.buffer_reservoir_minus[i] ) );
}