#include "kernel/kernel_summation.cuh"
#include "kernel/kernel_index_overwrite.cuh"

/**
 * The Summation Kernels take an additional dt argument although dt is already included in the System::Parameters struct.
 * This is because we want to be able to set the dt to a complex value to enable imaginary time propagation of the system.
*/

CUDA_GLOBAL void PC3::Kernel::RK45::runge_sum_to_input_of_k2( int i, complex_number dt, MatrixContainer::Pointers dev_ptrs, System::Parameters p, bool use_twin_mode ) {
    OVERWRITE_THREAD_INDEX(i);

    dev_ptrs.buffer_wavefunction_plus[i] = dev_ptrs.wavefunction_plus[i] + RKCoefficients::b11 * dt * dev_ptrs.k1_wavefunction_plus[i];
    dev_ptrs.buffer_reservoir_plus[i] = dev_ptrs.reservoir_plus[i] + RKCoefficients::b11 * dt * dev_ptrs.k1_reservoir_plus[i];
    if ( not use_twin_mode ) 
        return;
    dev_ptrs.buffer_wavefunction_minus[i] = dev_ptrs.wavefunction_minus[i] + RKCoefficients::b11 * dt * dev_ptrs.k1_wavefunction_minus[i];
    dev_ptrs.buffer_reservoir_minus[i] = dev_ptrs.reservoir_minus[i] + RKCoefficients::b11 * dt * dev_ptrs.k1_reservoir_minus[i];
}

CUDA_GLOBAL void PC3::Kernel::RK45::runge_sum_to_input_of_k3( int i, complex_number dt, MatrixContainer::Pointers dev_ptrs, System::Parameters p, bool use_twin_mode ) {
    OVERWRITE_THREAD_INDEX(i);

    dev_ptrs.buffer_wavefunction_plus[i] = dev_ptrs.wavefunction_plus[i] + dt*(RKCoefficients::b21 * dev_ptrs.k1_wavefunction_plus[i] + RKCoefficients::b22 * dev_ptrs.k2_wavefunction_plus[i]);
    dev_ptrs.buffer_reservoir_plus[i] = dev_ptrs.reservoir_plus[i] + dt*(RKCoefficients::b21 * dev_ptrs.k1_reservoir_plus[i] + RKCoefficients::b22 * dev_ptrs.k2_reservoir_plus[i]);
    if ( not use_twin_mode ) 
        return;
    dev_ptrs.buffer_wavefunction_minus[i] = dev_ptrs.wavefunction_minus[i] + dt*(RKCoefficients::b21 * dev_ptrs.k1_wavefunction_minus[i] + RKCoefficients::b22 * dev_ptrs.k2_wavefunction_minus[i]);
    dev_ptrs.buffer_reservoir_minus[i] = dev_ptrs.reservoir_minus[i] + dt*(RKCoefficients::b21 * dev_ptrs.k1_reservoir_minus[i] + RKCoefficients::b22 * dev_ptrs.k2_reservoir_minus[i]);
}

CUDA_GLOBAL void PC3::Kernel::RK45::runge_sum_to_input_of_k4( int i, complex_number dt, MatrixContainer::Pointers dev_ptrs, System::Parameters p, bool use_twin_mode ) {
    OVERWRITE_THREAD_INDEX(i);

    dev_ptrs.buffer_wavefunction_plus[i] = dev_ptrs.wavefunction_plus[i] + dt*(RKCoefficients::b31 * dev_ptrs.k1_wavefunction_plus[i] + RKCoefficients::b32 * dev_ptrs.k2_wavefunction_plus[i] + RKCoefficients::b33 * dev_ptrs.k3_wavefunction_plus[i]);
    dev_ptrs.buffer_reservoir_plus[i] = dev_ptrs.reservoir_plus[i] + dt*(RKCoefficients::b31 * dev_ptrs.k1_reservoir_plus[i] + RKCoefficients::b32 * dev_ptrs.k2_reservoir_plus[i] + RKCoefficients::b33 * dev_ptrs.k3_reservoir_plus[i]);
    if ( not use_twin_mode ) 
        return;
    dev_ptrs.buffer_wavefunction_minus[i] = dev_ptrs.wavefunction_minus[i] + dt*(RKCoefficients::b31 * dev_ptrs.k1_wavefunction_minus[i] + RKCoefficients::b32 * dev_ptrs.k2_wavefunction_minus[i] + RKCoefficients::b33 * dev_ptrs.k3_wavefunction_minus[i]);
    dev_ptrs.buffer_reservoir_minus[i] = dev_ptrs.reservoir_minus[i] + dt*(RKCoefficients::b31 * dev_ptrs.k1_reservoir_minus[i] + RKCoefficients::b32 * dev_ptrs.k2_reservoir_minus[i] + RKCoefficients::b33 * dev_ptrs.k3_reservoir_minus[i]);
}

CUDA_GLOBAL void PC3::Kernel::RK45::runge_sum_to_input_of_k5( int i, complex_number dt, MatrixContainer::Pointers dev_ptrs, System::Parameters p, bool use_twin_mode ) {
    OVERWRITE_THREAD_INDEX(i);

    dev_ptrs.buffer_wavefunction_plus[i] = dev_ptrs.wavefunction_plus[i] + dt*(RKCoefficients::b41 * dev_ptrs.k1_wavefunction_plus[i] + RKCoefficients::b42 * dev_ptrs.k2_wavefunction_plus[i] + RKCoefficients::b43 * dev_ptrs.k3_wavefunction_plus[i] + RKCoefficients::b44 * dev_ptrs.k4_wavefunction_plus[i]);
    dev_ptrs.buffer_reservoir_plus[i] = dev_ptrs.reservoir_plus[i] + dt*(RKCoefficients::b41 * dev_ptrs.k1_reservoir_plus[i] + RKCoefficients::b42 * dev_ptrs.k2_reservoir_plus[i] + RKCoefficients::b43 * dev_ptrs.k3_reservoir_plus[i] + RKCoefficients::b44 * dev_ptrs.k4_reservoir_plus[i]);
    if ( not use_twin_mode ) 
        return;
    dev_ptrs.buffer_wavefunction_minus[i] = dev_ptrs.wavefunction_minus[i] + dt*(RKCoefficients::b41 * dev_ptrs.k1_wavefunction_minus[i] + RKCoefficients::b42 * dev_ptrs.k2_wavefunction_minus[i] + RKCoefficients::b43 * dev_ptrs.k3_wavefunction_minus[i] + RKCoefficients::b44 * dev_ptrs.k4_wavefunction_minus[i]);
    dev_ptrs.buffer_reservoir_minus[i] = dev_ptrs.reservoir_minus[i] + dt*(RKCoefficients::b41 * dev_ptrs.k1_reservoir_minus[i] + RKCoefficients::b42 * dev_ptrs.k2_reservoir_minus[i] + RKCoefficients::b43 * dev_ptrs.k3_reservoir_minus[i] + RKCoefficients::b44 * dev_ptrs.k4_reservoir_minus[i]);
}

CUDA_GLOBAL void PC3::Kernel::RK45::runge_sum_to_input_of_k6( int i, complex_number dt, MatrixContainer::Pointers dev_ptrs, System::Parameters p, bool use_twin_mode ) {
    OVERWRITE_THREAD_INDEX(i);

    dev_ptrs.buffer_wavefunction_plus[i] = dev_ptrs.wavefunction_plus[i] + dt*(RKCoefficients::b51 * dev_ptrs.k1_wavefunction_plus[i] + RKCoefficients::b52 * dev_ptrs.k2_wavefunction_plus[i] + RKCoefficients::b53 * dev_ptrs.k3_wavefunction_plus[i] + RKCoefficients::b54 * dev_ptrs.k4_wavefunction_plus[i] + RKCoefficients::b55 * dev_ptrs.k5_wavefunction_plus[i]);
    dev_ptrs.buffer_reservoir_plus[i] = dev_ptrs.reservoir_plus[i] + dt*(RKCoefficients::b51 * dev_ptrs.k1_reservoir_plus[i] + RKCoefficients::b52 * dev_ptrs.k2_reservoir_plus[i] + RKCoefficients::b53 * dev_ptrs.k3_reservoir_plus[i] + RKCoefficients::b54 * dev_ptrs.k4_reservoir_plus[i] + RKCoefficients::b55 * dev_ptrs.k5_reservoir_plus[i]);
    if ( not use_twin_mode ) 
        return;
    dev_ptrs.buffer_wavefunction_minus[i] = dev_ptrs.wavefunction_minus[i] + dt*(RKCoefficients::b51 * dev_ptrs.k1_wavefunction_minus[i] + RKCoefficients::b52 * dev_ptrs.k2_wavefunction_minus[i] + RKCoefficients::b53 * dev_ptrs.k3_wavefunction_minus[i] + RKCoefficients::b54 * dev_ptrs.k4_wavefunction_minus[i] + RKCoefficients::b55 * dev_ptrs.k5_wavefunction_minus[i]);
    dev_ptrs.buffer_reservoir_minus[i] = dev_ptrs.reservoir_minus[i] + dt*(RKCoefficients::b51 * dev_ptrs.k1_reservoir_minus[i] + RKCoefficients::b52 * dev_ptrs.k2_reservoir_minus[i] + RKCoefficients::b53 * dev_ptrs.k3_reservoir_minus[i] + RKCoefficients::b54 * dev_ptrs.k4_reservoir_minus[i] + RKCoefficients::b55 * dev_ptrs.k5_reservoir_minus[i]);
}

CUDA_GLOBAL void PC3::Kernel::RK45::runge_sum_to_final( int i, complex_number dt, MatrixContainer::Pointers dev_ptrs, System::Parameters p, bool use_twin_mode ) {
    OVERWRITE_THREAD_INDEX(i);

    dev_ptrs.buffer_wavefunction_plus[i] = dev_ptrs.wavefunction_plus[i] + dt*(RKCoefficients::b61 * dev_ptrs.k1_wavefunction_plus[i] + RKCoefficients::b63 * dev_ptrs.k3_wavefunction_plus[i] + RKCoefficients::b64 * dev_ptrs.k4_wavefunction_plus[i] + RKCoefficients::b65 * dev_ptrs.k5_wavefunction_plus[i] + RKCoefficients::b66 * dev_ptrs.k6_wavefunction_plus[i]);
    dev_ptrs.buffer_reservoir_plus[i] = dev_ptrs.reservoir_plus[i] + dt*(RKCoefficients::b61 * dev_ptrs.k1_reservoir_plus[i] + RKCoefficients::b63 * dev_ptrs.k3_reservoir_plus[i] + RKCoefficients::b64 * dev_ptrs.k4_reservoir_plus[i] + RKCoefficients::b65 * dev_ptrs.k5_reservoir_plus[i] + RKCoefficients::b66 * dev_ptrs.k6_reservoir_plus[i]);
    if ( not use_twin_mode ) 
        return;
    dev_ptrs.buffer_wavefunction_minus[i] = dev_ptrs.wavefunction_minus[i] + dt*(RKCoefficients::b61 * dev_ptrs.k1_wavefunction_minus[i] + RKCoefficients::b63 * dev_ptrs.k3_wavefunction_minus[i] + RKCoefficients::b64 * dev_ptrs.k4_wavefunction_minus[i] + RKCoefficients::b65 * dev_ptrs.k5_wavefunction_minus[i] + RKCoefficients::b66 * dev_ptrs.k6_wavefunction_minus[i]);
    dev_ptrs.buffer_reservoir_minus[i] = dev_ptrs.reservoir_minus[i] + dt*(RKCoefficients::b61 * dev_ptrs.k1_reservoir_minus[i] + RKCoefficients::b63 * dev_ptrs.k3_reservoir_minus[i] + RKCoefficients::b64 * dev_ptrs.k4_reservoir_minus[i] + RKCoefficients::b65 * dev_ptrs.k5_reservoir_minus[i] + RKCoefficients::b66 * dev_ptrs.k6_reservoir_minus[i]);
}

CUDA_GLOBAL void PC3::Kernel::RK45::runge_sum_final_error( int i, complex_number dt, MatrixContainer::Pointers dev_ptrs, System::Parameters p, bool use_twin_mode ) {
    OVERWRITE_THREAD_INDEX(i);

    dev_ptrs.rk_error[i] = CUDA::abs2( dt * ( RKCoefficients::e1 * dev_ptrs.k1_wavefunction_plus[i] + RKCoefficients::e3 * dev_ptrs.k3_wavefunction_plus[i] + RKCoefficients::e4 * dev_ptrs.k4_wavefunction_plus[i] + RKCoefficients::e5 * dev_ptrs.k5_wavefunction_plus[i] + RKCoefficients::e6 * dev_ptrs.k6_wavefunction_plus[i] + RKCoefficients::e7 * dev_ptrs.k7_wavefunction_plus[i] ) );
    //dev_ptrs.rk_error[i] += CUDA::abs2( dt * ( RKCoefficients::e1 * dev_ptrs.k1_reservoir_plus[i] + RKCoefficients::e3 * dev_ptrs.k3_reservoir_plus[i] + RKCoefficients::e4 * dev_ptrs.k4_reservoir_plus[i] + RKCoefficients::e5 * dev_ptrs.k5_reservoir_plus[i] + RKCoefficients::e6 * dev_ptrs.k6_reservoir_plus[i] + RKCoefficients::e7 * dev_ptrs.k7_reservoir_plus[i] ) );
    if ( not use_twin_mode ) 
        return;
    dev_ptrs.rk_error[i] += CUDA::abs2( dt * ( RKCoefficients::e1 * dev_ptrs.k1_wavefunction_minus[i] + RKCoefficients::e3 * dev_ptrs.k3_wavefunction_minus[i] + RKCoefficients::e4 * dev_ptrs.k4_wavefunction_minus[i] + RKCoefficients::e5 * dev_ptrs.k5_wavefunction_minus[i] + RKCoefficients::e6 * dev_ptrs.k6_wavefunction_minus[i] + RKCoefficients::e7 * dev_ptrs.k7_wavefunction_minus[i] ) );
    //dev_ptrs.rk_error[i] += CUDA::abs2( dt * ( RKCoefficients::e1 * dev_ptrs.k1_reservoir_minus[i] + RKCoefficients::e3 * dev_ptrs.k3_reservoir_minus[i] + RKCoefficients::e4 * dev_ptrs.k4_reservoir_minus[i] + RKCoefficients::e5 * dev_ptrs.k5_reservoir_minus[i] + RKCoefficients::e6 * dev_ptrs.k6_reservoir_minus[i] + RKCoefficients::e7 * dev_ptrs.k7_reservoir_minus[i] ) );
}

CUDA_GLOBAL void PC3::Kernel::RK4::runge_sum_to_input_k2( int i, complex_number dt, MatrixContainer::Pointers dev_ptrs, System::Parameters p, bool use_twin_mode ) {
    OVERWRITE_THREAD_INDEX(i);

    dev_ptrs.buffer_wavefunction_plus[i] = dev_ptrs.wavefunction_plus[i] + real_number(0.5)*dt * dev_ptrs.k1_wavefunction_plus[i];
    dev_ptrs.buffer_reservoir_plus[i] = dev_ptrs.reservoir_plus[i] + real_number(0.5)*dt * dev_ptrs.k1_reservoir_plus[i];
    if ( not use_twin_mode ) 
        return;
    dev_ptrs.buffer_wavefunction_minus[i] = dev_ptrs.wavefunction_minus[i] + real_number(0.5)*dt * dev_ptrs.k1_wavefunction_minus[i];
    dev_ptrs.buffer_reservoir_minus[i] = dev_ptrs.reservoir_minus[i] + real_number(0.5)*dt * dev_ptrs.k1_reservoir_minus[i];
}

CUDA_GLOBAL void PC3::Kernel::RK4::runge_sum_to_input_k3( int i, complex_number dt, MatrixContainer::Pointers dev_ptrs, System::Parameters p, bool use_twin_mode ) {
    OVERWRITE_THREAD_INDEX(i);

    dev_ptrs.buffer_wavefunction_plus[i] = dev_ptrs.wavefunction_plus[i] + real_number(0.5)*dt * dev_ptrs.k2_wavefunction_plus[i];
    dev_ptrs.buffer_reservoir_plus[i] = dev_ptrs.reservoir_plus[i] + real_number(0.5)*dt * dev_ptrs.k2_reservoir_plus[i];
    if ( not use_twin_mode ) 
        return;
    dev_ptrs.buffer_wavefunction_minus[i] = dev_ptrs.wavefunction_minus[i] + real_number(0.5)*dt * dev_ptrs.k2_wavefunction_minus[i];
    dev_ptrs.buffer_reservoir_minus[i] = dev_ptrs.reservoir_minus[i] + real_number(0.5)*dt * dev_ptrs.k2_reservoir_minus[i];
}

CUDA_GLOBAL void PC3::Kernel::RK4::runge_sum_to_input_k4( int i, complex_number dt, MatrixContainer::Pointers dev_ptrs, System::Parameters p, bool use_twin_mode ) {
    OVERWRITE_THREAD_INDEX(i);

    dev_ptrs.buffer_wavefunction_plus[i] = dev_ptrs.wavefunction_plus[i] + dt * dev_ptrs.k3_wavefunction_plus[i];
    dev_ptrs.buffer_reservoir_plus[i] = dev_ptrs.reservoir_plus[i] + dt * dev_ptrs.k3_reservoir_plus[i];
    if ( not use_twin_mode ) 
        return;
    dev_ptrs.buffer_wavefunction_minus[i] = dev_ptrs.wavefunction_minus[i] + dt * dev_ptrs.k3_wavefunction_minus[i];
    dev_ptrs.buffer_reservoir_minus[i] = dev_ptrs.reservoir_minus[i] + dt * dev_ptrs.k3_reservoir_minus[i];
}

CUDA_GLOBAL void PC3::Kernel::RK4::runge_sum_to_final( int i, complex_number dt, MatrixContainer::Pointers dev_ptrs, System::Parameters p, bool use_twin_mode ) {
    OVERWRITE_THREAD_INDEX(i);

    dev_ptrs.buffer_wavefunction_plus[i] = dev_ptrs.wavefunction_plus[i] + dt / real_number(6.0) * ( dev_ptrs.k1_wavefunction_plus[i] + real_number(2.0) * dev_ptrs.k2_wavefunction_plus[i] + real_number(2.0) * dev_ptrs.k3_wavefunction_plus[i] + dev_ptrs.k4_wavefunction_plus[i] );
    dev_ptrs.buffer_reservoir_plus[i] = dev_ptrs.reservoir_plus[i] + dt / real_number(6.0) * ( dev_ptrs.k1_reservoir_plus[i] + real_number(2.0) * dev_ptrs.k2_reservoir_plus[i] + real_number(2.0) * dev_ptrs.k3_reservoir_plus[i] + dev_ptrs.k4_reservoir_plus[i] );
    if ( not use_twin_mode ) 
        return;
    dev_ptrs.buffer_wavefunction_minus[i] = dev_ptrs.wavefunction_minus[i] + dt / real_number(6.0) * ( dev_ptrs.k1_wavefunction_minus[i] + real_number(2.0) * dev_ptrs.k2_wavefunction_minus[i] + real_number(2.0) * dev_ptrs.k3_wavefunction_minus[i] + dev_ptrs.k4_wavefunction_minus[i] );
    dev_ptrs.buffer_reservoir_minus[i] = dev_ptrs.reservoir_minus[i] + dt / real_number(6.0) * ( dev_ptrs.k1_reservoir_minus[i] + real_number(2.0) * dev_ptrs.k2_reservoir_minus[i] + real_number(2.0) * dev_ptrs.k3_reservoir_minus[i] + dev_ptrs.k4_reservoir_minus[i] );
}