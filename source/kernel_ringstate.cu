#include "cuda_complex.cuh"
#include "kernel_ringstate.cuh"

CUDA_HOST_DEVICE void kernel_generateRingPhase( int s_N, real_number amp, int n, real_number w1, real_number w2, real_number xPos, real_number yPos, real_number p_xmax, real_number s_dx, bool normalize, complex_number* buffer, bool reset ) {
    real_number largest_r = 0.0;
    for ( int i = 0; i < s_N; i++ )
        for ( int j = 0; j < s_N; j++ ) {
            auto index = i * s_N + j;
            if ( reset )
                buffer[index] = { 0.0, 0.0 };
            real_number x = -p_xmax / 2.0 + s_dx * i;
            real_number y = -p_xmax / 2.0 + s_dx * j;
            real_number r = sqrt( abs2( x - xPos ) + abs2( y - yPos ) );
            buffer[index] += amp * r / w1 / w1 * exp( -r * r / w2 / w2 ) * pow( ( x - xPos + 1.0 * sign( n ) * complex_number{ 0, 1.0 } * ( y - yPos ) ), abs( n ) );

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

CUDA_HOST_DEVICE void kernel_generateRingState( int s_N, real_number amp, real_number w1, real_number w2, real_number xPos, real_number yPos, real_number p_xmax, real_number s_dx, bool normalize, complex_number* buffer, bool reset ) {
    real_number max_buffer = 0.0;
    for ( int i = 0; i < s_N; i++ )
        for ( int j = 0; j < s_N; j++ ) {
            auto index = i * s_N + j;
            if ( reset )
                buffer[index] = { 0.0, 0.0 };
            real_number x = -p_xmax / 2.0 + s_dx * i;
            real_number y = -p_xmax / 2.0 + s_dx * j;
            real_number r = sqrt( abs2( x - xPos ) + abs2( y - yPos ) );
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