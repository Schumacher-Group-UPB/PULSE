#include "kernel_ringstate.cuh"

__host__ __device__ void kernel_generateRingPhase( int s_N, double amp, int n, double w1, double w2, double xPos, double yPos, double p_xmax, double s_dx, bool normalize, cuDoubleComplex* buffer, bool reset ) {
    double largest_r = 0.0;
    for ( int i = 0; i < s_N; i++ )
        for ( int j = 0; j < s_N; j++ ) {
            auto index = i * s_N + j;
            if ( reset )
                buffer[index] = make_cuDoubleComplex( 0.0, 0.0 );
            auto x = -p_xmax / 2.0 + s_dx * i;
            auto y = -p_xmax / 2.0 + s_dx * j;
            double r = sqrt( abs2( x - xPos ) + abs2( y - yPos ) );
            buffer[index] += amp * r / w1 / w1 * exp( -r * r / w2 / w2 ) * pow( ( x - xPos + 1.0 * sign( n ) * make_cuDoubleComplex( 0, 1.0 ) * ( y - yPos ) ), abs( n ) );

            largest_r = max( largest_r, abs2( buffer[index] ) );
        }
    if ( !normalize )
        return;
    for ( int i = 0; i < s_N; i++ )
        for ( int j = 0; j < s_N; j++ ) {
            auto index = i * s_N + j;
            buffer[index] = buffer[index] / sqrt( largest_r );
        }
}

__host__ __device__ void kernel_generateRingState( int s_N, double amp, double w1, double w2, double xPos, double yPos, double p_xmax, double s_dx, bool normalize, cuDoubleComplex* buffer, bool reset ) {
    double max_buffer = 0.0;
    for ( int i = 0; i < s_N; i++ )
        for ( int j = 0; j < s_N; j++ ) {
            auto index = i * s_N + j;
            if ( reset )
                buffer[index] = make_cuDoubleComplex( 0.0, 0.0 );
            auto x = -p_xmax / 2.0 + s_dx * i;
            auto y = -p_xmax / 2.0 + s_dx * j;
            double r = sqrt( abs2( x - xPos ) + abs2( y - yPos ) );
            buffer[index] += amp * r * r / w1 / w1 * exp( -r * r / w2 / w2 );
            max_buffer = max( max_buffer, abs2( buffer[index] ) );
        }
    if ( !normalize )
        return;
    for ( int i = 0; i < s_N; i++ )
        for ( int j = 0; j < s_N; j++ ) {
            auto index = i * s_N + j;
            buffer[index] = buffer[index] / sqrt( max_buffer );
        }
}