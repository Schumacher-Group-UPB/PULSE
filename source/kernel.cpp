#include "cuda_complex.cuh"
#include "kernel.hpp"

void generateRingPhase( int s_N, real_number amp, int n, real_number w1, real_number w2, real_number xPos, real_number yPos, real_number p_xmax, real_number s_dx, bool normalize, complex_number* buffer, bool reset ) {
    // Cache cuComplexDouble result
    complex_number* buffer_cuDoubleComplex = reinterpret_cast<complex_number*>( buffer );
    kernel_generateRingPhase( s_N, amp, n, w1, w2, xPos, yPos, p_xmax, s_dx, normalize, buffer_cuDoubleComplex, reset );
}
void generateRingState( int s_N, real_number amp, real_number w1, real_number w2, real_number xPos, real_number yPos, real_number p_xmax, real_number s_dx, bool normalize, complex_number* buffer, bool reset ) {
    complex_number* buffer_cuDoubleComplex = reinterpret_cast<complex_number*>( buffer );
    kernel_generateRingState( s_N, amp, w1, w2, xPos, yPos, p_xmax, s_dx, normalize, buffer_cuDoubleComplex, reset );
}