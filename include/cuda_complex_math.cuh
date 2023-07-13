#pragma once
#include "cuda_complex.cuh"

__device__ static __inline__ double device_floor( const double x ) {
    return floor( x );
}

__host__ __device__ static __inline__ double sign( double x ) {
    return x > 0 ? 1 : -1;
}

__host__ __device__ static __inline__ double abs2( const cuDoubleComplex& x ) {
    return cuCreal( x ) * cuCreal( x ) + cuCimag( x ) * cuCimag( x );
}
__host__ __device__ static __inline__ double abs2( const double& x ) {
    return x * x;
}

__host__ __device__ static __inline__ cuDoubleComplex pow( const cuDoubleComplex& a, const int N ) {
    cuDoubleComplex res = make_cuDoubleComplex( 1.0, 0 );
    for ( int i = 0; i < abs( N ); i++ )
        res = res * a;
    return N > 0 ? res : 1. / res;
}

__host__ __device__ static __inline__ cuDoubleComplex exp( cuDoubleComplex z ) {
    return make_cuDoubleComplex( exp( z.x ) * cos( z.y ), exp( z.x ) * sin( z.y ) );
}
__host__ __device__ static __inline__ cuDoubleComplex cuCsqrt( cuDoubleComplex x ) {
    double radius = cuCabs( x );
    double cosA = x.x / radius;
    cuDoubleComplex out;
    out.x = sqrt( radius * ( cosA + 1.0 ) / 2.0 );
    out.y = sqrt( radius * ( 1.0 - cosA ) / 2.0 );
    // signbit should be false if x.y is negative
    if ( signbit( x.y ) )
        out.y *= -1.0;

    return out;
}