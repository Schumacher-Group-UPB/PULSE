#include "kernel.hpp"

void generateRingPhase( int s_N, double amp, int n, double w1, double w2, double xPos, double yPos, double p_xmax, double s_dx, bool normalize, Scalar* buffer, bool reset ) {
    // Cache cuComplexDouble result
    cuDoubleComplex* buffer_cuDoubleComplex = reinterpret_cast<cuDoubleComplex*>( buffer );
    kernel_generateRingPhase( s_N, amp, n, w1, w2, xPos, yPos, p_xmax, s_dx, normalize, buffer_cuDoubleComplex, reset );
}
void generateRingState( int s_N, double amp, double w1, double w2, double xPos, double yPos, double p_xmax, double s_dx, bool normalize, Scalar* buffer, bool reset ) {
    cuDoubleComplex* buffer_cuDoubleComplex = reinterpret_cast<cuDoubleComplex*>( buffer );
    kernel_generateRingState( s_N, amp, w1, w2, xPos, yPos, p_xmax, s_dx, normalize, buffer_cuDoubleComplex, reset );
}