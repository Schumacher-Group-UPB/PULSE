#pragma once
#include <cuComplex.h>
#include <cuda.h>
#include <complex>

//#define USEFP32

#ifdef USEFP32
#define real_number float
#define complex_number cuComplex
#define fft_complex_number cufftComplex
#define FFTSOLVER cufftExecC2C
#define FFTPLAN CUFFT_C2C
#else
#define real_number double
#define complex_number cuDoubleComplex
#define fft_complex_number cufftDoubleComplex
#define FFTSOLVER cufftExecZ2Z
#define FFTPLAN CUFFT_Z2Z
#endif

__host__ __device__ static __inline__ cuDoubleComplex operator+( const cuDoubleComplex& a, const cuDoubleComplex& b ) {
    return cuCadd( a, b );
}
__host__ __device__ static __inline__ cuDoubleComplex operator-( const cuDoubleComplex& a, const cuDoubleComplex& b ) {
    return cuCsub( a, b );
}
__host__ __device__ static __inline__ cuDoubleComplex operator+( const cuDoubleComplex& a, const real_number& b ) {
    return make_cuDoubleComplex( b, 0.0 ) + a;
}
__host__ __device__ static __inline__ cuDoubleComplex operator-( const cuDoubleComplex& a, const real_number& b ) {
    return make_cuDoubleComplex( b, 0.0 ) - a;
}
__host__ __device__ static __inline__ cuDoubleComplex operator+( const real_number& a, const cuDoubleComplex& b ) {
    return make_cuDoubleComplex( a, 0.0 ) + b;
}
__host__ __device__ static __inline__ cuDoubleComplex operator-( const real_number& a, const cuDoubleComplex& b ) {
    return make_cuDoubleComplex( a, 0.0 ) - b;
}

__host__ __device__ static __inline__ cuDoubleComplex operator*( const cuDoubleComplex& a, const cuDoubleComplex& b ) {
    return cuCmul( a, b );
}
__host__ __device__ static __inline__ cuDoubleComplex operator*( const cuDoubleComplex& a, const real_number& b ) {
    return make_cuDoubleComplex( b, 0.0 ) * a;
}
__host__ __device__ static __inline__ cuDoubleComplex operator/( const cuDoubleComplex& a, const cuDoubleComplex& b ) {
    return cuCdiv( a, b );
}
__host__ __device__ static __inline__ cuDoubleComplex operator/( const cuDoubleComplex& a, const real_number& b ) {
    return make_cuDoubleComplex( a.x / b, a.y / b );
}
__host__ __device__ static __inline__ cuDoubleComplex operator*( const real_number& a, const cuDoubleComplex& b ) {
    return make_cuDoubleComplex( a, 0.0 ) * b;
}
__host__ __device__ static __inline__ cuDoubleComplex operator/( const real_number& a, const cuDoubleComplex& b ) {
    return make_cuDoubleComplex( a, 0 ) / b;
}

__host__ __device__ static __inline__ cuDoubleComplex square( const cuDoubleComplex& a ) {
    return make_cuDoubleComplex( a.x * a.x, a.y * a.y );
}
__host__ __device__ static __inline__ void operator+=( cuDoubleComplex& a, const cuDoubleComplex& b ) {
    a = a + b;
}
__host__ __device__ static __inline__ void operator+=( cuDoubleComplex& a, const real_number& b ) {
    a.x = a.x + b;
}

// Same for floating point precision

__host__ __device__ static __inline__ cuComplex operator+( const cuComplex& a, const cuComplex& b ) {
    return cuCaddf( a, b );
}
__host__ __device__ static __inline__ cuComplex operator-( const cuComplex& a, const cuComplex& b ) {
    return cuCsubf( a, b );
}
__host__ __device__ static __inline__ cuComplex operator+( const cuComplex& a, const real_number& b ) {
    return make_cuComplex( b, 0.0 ) + a;
}
__host__ __device__ static __inline__ cuComplex operator-( const cuComplex& a, const real_number& b ) {
    return make_cuComplex( b, 0.0 ) - a;
}
__host__ __device__ static __inline__ cuComplex operator+( const real_number& a, const cuComplex& b ) {
    return make_cuComplex( a, 0.0 ) + b;
}
__host__ __device__ static __inline__ cuComplex operator-( const real_number& a, const cuComplex& b ) {
    return make_cuComplex( a, 0.0 ) - b;
}

__host__ __device__ static __inline__ cuComplex operator*( const cuComplex& a, const cuComplex& b ) {
    return cuCmulf( a, b );
}
__host__ __device__ static __inline__ cuComplex operator*( const cuComplex& a, const real_number& b ) {
    return make_cuComplex( b, 0.0 ) * a;
}
__host__ __device__ static __inline__ cuComplex operator/( const cuComplex& a, const cuComplex& b ) {
    return cuCdivf( a, b );
}
__host__ __device__ static __inline__ cuComplex operator/( const cuComplex& a, const real_number& b ) {
    return make_cuComplex( a.x / b, a.y / b );
}
__host__ __device__ static __inline__ cuComplex operator*( const real_number& a, const cuComplex& b ) {
    return make_cuComplex( a, 0.0 ) * b;
}
__host__ __device__ static __inline__ cuComplex operator/( const real_number& a, const cuComplex& b ) {
    return make_cuComplex( a, 0 ) / b;
}

__host__ __device__ static __inline__ cuComplex square( const cuComplex& a ) {
    return make_cuComplex( a.x * a.x, a.y * a.y );
}
__host__ __device__ static __inline__ void operator+=( cuComplex& a, const cuComplex& b ) {
    a = a + b;
}
__host__ __device__ static __inline__ void operator+=( cuComplex& a, const real_number& b ) {
    a.x = a.x + b;
}

/**
 * @brief Calculates the abs2 of a (complex) number, as long as
 * @param z The number to calculate the abs2 of.
 * @return real_number The abs2 of the number.
 */
template <typename T>
__host__ __device__ static __inline__ real_number cwiseAbs2( T z ) {
    return z.x*z.x + z.y*z.y;
}

/**
 * @brief Calculates the abs2 of a buffer of (complex) numbers
 * @param z The buffer to calculate the abs2 of.
 * @param buffer The buffer to save the result to.
 * @param size The size of the buffer.
 */
__host__ __device__ static __inline__ void cwiseAbs2( complex_number* z, real_number* buffer, int size ) {
#pragma omp parallel for
    for ( int i = 0; i < size; i++ )
        buffer[i] = z[i].x * z[i].x + z[i].y * z[i].y;
}
template <typename T>
__host__ __device__ static __inline__ void cwiseAbs2( T* z, real_number* buffer, int size ) {
#pragma omp parallel for
    for ( int i = 0; i < size; i++ )
        buffer[i] = z[i] * z[i];
}

__device__ static __inline__ real_number device_floor( const real_number x ) {
    return floor( x );
}

__host__ __device__ static __inline__ real_number sign( real_number x ) {
    return x > 0 ? 1 : -1;
}

__host__ __device__ static __inline__ real_number abs2( const complex_number& x ) {
    return x.x*x.x + x.y*x.y;
}

__host__ __device__ static __inline__ real_number abs2( const real_number& x ) {
    return x * x;
}

__host__ __device__ static __inline__ complex_number pow( const complex_number& a, const int N ) {
    complex_number res = {1.0, 0 };
    for ( int i = 0; i < abs( N ); i++ )
        res = res * a;
    return N > 0 ? res : 1. / res;
}

__host__ __device__ static __inline__ complex_number exp( complex_number z ) {
    return { exp( z.x ) * cos( z.y ), exp( z.x ) * sin( z.y ) };
}
__host__ __device__ static __inline__ cuDoubleComplex cuCsqrt( cuDoubleComplex x ) {
    real_number radius = cuCabs( x );
    real_number cosA = x.x / radius;
    cuDoubleComplex out;
    out.x = sqrt( radius * ( cosA + 1.0 ) / 2.0 );
    out.y = sqrt( radius * ( 1.0 - cosA ) / 2.0 );
    // signbit should be false if x.y is negative
    if ( signbit( x.y ) )
        out.y *= -1.0;

    return out;
}
__host__ __device__ static __inline__ cuComplex cuCsqrt( cuComplex x ) {
    real_number radius = cuCabsf( x );
    real_number cosA = x.x / radius;
    cuComplex out;
    out.x = sqrt( radius * ( cosA + 1.0 ) / 2.0 );
    out.y = sqrt( radius * ( 1.0 - cosA ) / 2.0 );
    // signbit should be false if x.y is negative
    if ( signbit( x.y ) )
        out.y *= -1.0;

    return out;
}

/**
 * Overwrite abs for complex numbers
 */
__host__ __device__ static __inline__ real_number abs( cuDoubleComplex z ) {
    return sqrt( z.x * z.x + z.y * z.y );
}
__host__ __device__ static __inline__ real_number abs( cuComplex z ) {
    return sqrtf( z.x * z.x + z.y * z.y );
}

// Use -rdc=true when compiling with nvcc to allow for the "extern" keyword to work
extern __device__ complex_number dev_half_i;
extern __device__ complex_number dev_i;
extern __device__ complex_number dev_minus_half_i;
extern __device__ complex_number dev_minus_i;