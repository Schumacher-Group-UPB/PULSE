#pragma once
#include <cuComplex.h>
#include <cuda.h>
#include <complex>
using Scalar = std::complex<double>;

__host__ __device__ static __inline__ cuDoubleComplex operator+( const cuDoubleComplex& a, const cuDoubleComplex& b ) {
    return cuCadd( a, b );
}
__host__ __device__ static __inline__ cuDoubleComplex operator-( const cuDoubleComplex& a, const cuDoubleComplex& b ) {
    return cuCsub( a, b );
}
__host__ __device__ static __inline__ cuDoubleComplex operator+( const cuDoubleComplex& a, const double& b ) {
    return make_cuDoubleComplex( b, 0.0 ) + a;
}
__host__ __device__ static __inline__ cuDoubleComplex operator-( const cuDoubleComplex& a, const double& b ) {
    return make_cuDoubleComplex( b, 0.0 ) - a;
}
__host__ __device__ static __inline__ cuDoubleComplex operator+( const double& a, const cuDoubleComplex& b ) {
    return make_cuDoubleComplex( a, 0.0 ) + b;
}
__host__ __device__ static __inline__ cuDoubleComplex operator-( const double& a, const cuDoubleComplex& b ) {
    return make_cuDoubleComplex( a, 0.0 ) - b;
}

__host__ __device__ static __inline__ cuDoubleComplex operator*( const cuDoubleComplex& a, const cuDoubleComplex& b ) {
    return cuCmul( a, b );
}
__host__ __device__ static __inline__ cuDoubleComplex operator*( const cuDoubleComplex& a, const double& b ) {
    return make_cuDoubleComplex( b, 0.0 ) * a;
}
__host__ __device__ static __inline__ cuDoubleComplex operator/( const cuDoubleComplex& a, const cuDoubleComplex& b ) {
    return cuCdiv( a, b );
}
__host__ __device__ static __inline__ cuDoubleComplex operator/( const cuDoubleComplex& a, const double& b ) {
    return make_cuDoubleComplex( a.x / b, a.y / b );
}
__host__ __device__ static __inline__ cuDoubleComplex operator*( const double& a, const cuDoubleComplex& b ) {
    return make_cuDoubleComplex( a, 0.0 ) * b;
}
__host__ __device__ static __inline__ cuDoubleComplex operator/( const double& a, const cuDoubleComplex& b ) {
    return make_cuDoubleComplex( a, 0 ) / b;
}

__host__ __device__ static __inline__ cuDoubleComplex square( const cuDoubleComplex& a ) {
    return make_cuDoubleComplex( a.x * a.x, a.y * a.y );
}
__host__ __device__ static __inline__ void operator+=( cuDoubleComplex& a, const cuDoubleComplex& b ) {
    a = a + b;
}
__host__ __device__ static __inline__ void operator+=( cuDoubleComplex& a, const double& b ) {
    a.x = a.x + b;
}


// Use -rdc=true when compiling with nvcc to allow for the "extern" keyword to work
extern __device__ cuDoubleComplex dev_half_i;
extern __device__ cuDoubleComplex dev_i;
extern __device__ cuDoubleComplex dev_minus_half_i;
extern __device__ cuDoubleComplex dev_minus_i;