#pragma once
#include "cuda_macro.cuh"
#ifndef USECPU
#    include <cuComplex.h>
#    include <cuda.h>
#endif
#include <complex>

/**
 * CUDA Macros; We hide CPU and GPU as well as FP32 and FP64 specific code behind these macros.
 */
// GPU Version
#ifndef USECPU
// Define CUDA_INLINE as nvcc's __inline__ when using the GPU
#    define CUDA_INLINE __inline__
// Floating Point Precision Macros
#    ifdef USEFP32
#        define real_number float
#        define complex_number cuComplex
#        define fft_complex_number cufftComplex
#        define FFTSOLVER cufftExecC2C
#        define FFTPLAN CUFFT_C2C
#    else
#        define real_number double
#        define complex_number cuDoubleComplex
#        define fft_complex_number cufftDoubleComplex
#        define FFTSOLVER cufftExecZ2Z
#        define FFTPLAN CUFFT_Z2Z
#    endif
// CPU Version
#else
// Define CUDA_INLINE as inline when using the CPU
#    define CUDA_INLINE inline
// Floating Point Precision Macros
#    ifdef USEFP32
#        define real_number float
#        define complex_number std::complex<float>
#    else
#        define real_number double
#        define complex_number std::complex<double>
#    endif
// The FFT variables are empty because we don't do the FFT on the CPU
#    define fft_complex_number
#    define FFTSOLVER
#    define FFTPLAN

#endif

#include <cmath>

/**
 * Basic Operators for cuComplex and cuDoubleComplex GPU types
*/
#ifndef USECPU
CUDA_HOST_DEVICE static CUDA_INLINE cuDoubleComplex operator+( const cuDoubleComplex& a, const cuDoubleComplex& b ) {
    return cuCadd( a, b );
}
CUDA_HOST_DEVICE static CUDA_INLINE cuDoubleComplex operator-( const cuDoubleComplex& a, const cuDoubleComplex& b ) {
    return cuCsub( a, b );
}
CUDA_HOST_DEVICE static CUDA_INLINE cuDoubleComplex operator+( const cuDoubleComplex& a, const real_number& b ) {
    return make_cuDoubleComplex( b, 0.0 ) + a;
}
CUDA_HOST_DEVICE static CUDA_INLINE cuDoubleComplex operator-( const cuDoubleComplex& a, const real_number& b ) {
    return a - make_cuDoubleComplex( b, 0.0 );
}
CUDA_HOST_DEVICE static CUDA_INLINE cuDoubleComplex operator+( const real_number& a, const cuDoubleComplex& b ) {
    return make_cuDoubleComplex( a, 0.0 ) + b;
}
CUDA_HOST_DEVICE static CUDA_INLINE cuDoubleComplex operator-( const real_number& a, const cuDoubleComplex& b ) {
    return make_cuDoubleComplex( a, 0.0 ) - b;
}

CUDA_HOST_DEVICE static CUDA_INLINE cuDoubleComplex operator*( const cuDoubleComplex& a, const cuDoubleComplex& b ) {
    return cuCmul( a, b );
}
CUDA_HOST_DEVICE static CUDA_INLINE cuDoubleComplex operator*( const cuDoubleComplex& a, const real_number& b ) {
    return a * make_cuDoubleComplex( b, 0.0 );
}
CUDA_HOST_DEVICE static CUDA_INLINE cuDoubleComplex operator/( const cuDoubleComplex& a, const cuDoubleComplex& b ) {
    return cuCdiv( a, b );
}
CUDA_HOST_DEVICE static CUDA_INLINE cuDoubleComplex operator/( const cuDoubleComplex& a, const real_number& b ) {
    return make_cuDoubleComplex( a.x / b, a.y / b );
}
CUDA_HOST_DEVICE static CUDA_INLINE cuDoubleComplex operator*( const real_number& a, const cuDoubleComplex& b ) {
    return make_cuDoubleComplex( a, 0.0 ) * b;
}
CUDA_HOST_DEVICE static CUDA_INLINE cuDoubleComplex operator/( const real_number& a, const cuDoubleComplex& b ) {
    return make_cuDoubleComplex( a, 0.0 ) / b;
}

CUDA_HOST_DEVICE static CUDA_INLINE cuDoubleComplex square( const cuDoubleComplex& a ) {
    return make_cuDoubleComplex( a.x * a.x - a.y * a.y, 2.0 * a.x * a.y );
}
CUDA_HOST_DEVICE static CUDA_INLINE void operator+=( cuDoubleComplex& a, const cuDoubleComplex& b ) {
    a = a + b;
}
CUDA_HOST_DEVICE static CUDA_INLINE void operator+=( cuDoubleComplex& a, const real_number& b ) {
    a.x = a.x + b;
}
CUDA_HOST_DEVICE static CUDA_INLINE void operator-=( cuDoubleComplex& a, const cuDoubleComplex& b ) {
    a = a - b;
}
CUDA_HOST_DEVICE static CUDA_INLINE void operator-=( cuDoubleComplex& a, const real_number& b ) {
    a.x = a.x - b;
}
CUDA_HOST_DEVICE static CUDA_INLINE cuDoubleComplex operator-( const cuDoubleComplex& a ) {
    return make_cuDoubleComplex( -a.x, -a.y );
}

// Same for floating point precision

CUDA_HOST_DEVICE static CUDA_INLINE cuComplex operator+( const cuComplex& a, const cuComplex& b ) {
    return cuCaddf( a, b );
}
CUDA_HOST_DEVICE static CUDA_INLINE cuComplex operator-( const cuComplex& a, const cuComplex& b ) {
    return cuCsubf( a, b );
}
CUDA_HOST_DEVICE static CUDA_INLINE cuComplex operator+( const cuComplex& a, const real_number& b ) {
    return make_cuComplex( b, 0.0 ) + a;
}
CUDA_HOST_DEVICE static CUDA_INLINE cuComplex operator-( const cuComplex& a, const real_number& b ) {
    return a - make_cuComplex( b, 0.0 );
}
CUDA_HOST_DEVICE static CUDA_INLINE cuComplex operator+( const real_number& a, const cuComplex& b ) {
    return make_cuComplex( a, 0.0 ) + b;
}
CUDA_HOST_DEVICE static CUDA_INLINE cuComplex operator-( const real_number& a, const cuComplex& b ) {
    return make_cuComplex( a, 0.0 ) - b;
}

CUDA_HOST_DEVICE static CUDA_INLINE cuComplex operator*( const cuComplex& a, const cuComplex& b ) {
    return cuCmulf( a, b );
}
CUDA_HOST_DEVICE static CUDA_INLINE cuComplex operator*( const cuComplex& a, const real_number& b ) {
    return a * make_cuComplex( b, 0.0 );
}
CUDA_HOST_DEVICE static CUDA_INLINE cuComplex operator/( const cuComplex& a, const cuComplex& b ) {
    return cuCdivf( a, b );
}
CUDA_HOST_DEVICE static CUDA_INLINE cuComplex operator/( const cuComplex& a, const real_number& b ) {
    return make_cuComplex( a.x / b, a.y / b );
}
CUDA_HOST_DEVICE static CUDA_INLINE cuComplex operator*( const real_number& a, const cuComplex& b ) {
    return make_cuComplex( a, 0.0 ) * b;
}
CUDA_HOST_DEVICE static CUDA_INLINE cuComplex operator/( const real_number& a, const cuComplex& b ) {
    return make_cuComplex( a, 0 ) / b;
}

CUDA_HOST_DEVICE static CUDA_INLINE cuComplex square( const cuComplex& a ) {
    return make_cuComplex( a.x * a.x - a.y * a.y, 2.0 * a.x * a.y );
}
CUDA_HOST_DEVICE static CUDA_INLINE void operator+=( cuComplex& a, const cuComplex& b ) {
    a = a + b;
}
CUDA_HOST_DEVICE static CUDA_INLINE void operator+=( cuComplex& a, const real_number& b ) {
    a.x = a.x + b;
}
CUDA_HOST_DEVICE static CUDA_INLINE void operator-=( cuComplex& a, const cuComplex& b ) {
    a = a - b;
}
CUDA_HOST_DEVICE static CUDA_INLINE void operator-=( cuComplex& a, const real_number& b ) {
    a.x = a.x - b;
}
CUDA_HOST_DEVICE static CUDA_INLINE cuComplex operator-( const cuComplex& a ) {
    return make_cuComplex( -a.x, -a.y );
}
#endif

namespace PC3::CUDA {
    
/**
 * real() and imag() functions for complex numbers. Defining them for the CPU and GPU
 * allows us to use the same functions for both CPU and GPU code.
*/
#ifdef USECPU
// Define real() and imag() for the CPU
static CUDA_INLINE real_number real( const complex_number& x ) {
    return x.real();
}
static CUDA_INLINE real_number imag( const complex_number& x ) {
    return x.imag();
}
#else
// Define real() and imag() for the GPU
CUDA_HOST_DEVICE static CUDA_INLINE real_number real( const complex_number& x ) {
    return x.x;
}
CUDA_HOST_DEVICE static CUDA_INLINE real_number imag( const complex_number& x ) {
    return x.y;
}
#endif

// For real numbers, the operators can be the same
CUDA_HOST_DEVICE static CUDA_INLINE real_number real( const real_number& x ) {
    return x;
}
CUDA_HOST_DEVICE static CUDA_INLINE real_number imag( const real_number& x ) {
    return real_number(0.0);
}

/**
 * Host and Device Functions for abs, sqrt, floor, etc.
 * We seperate FP32 and FP64 code here.
*/


CUDA_HOST_DEVICE static CUDA_INLINE real_number floor( const real_number x ) {
    return std::floor( x );
}

CUDA_HOST_DEVICE static CUDA_INLINE real_number sqrt( const real_number x ) {
    return std::sqrt( x );
}

template <typename T>
CUDA_HOST_DEVICE static CUDA_INLINE T sign( const T x ) {
    return x > 0 ? 1 : -1;
}

CUDA_HOST_DEVICE static CUDA_INLINE complex_number sqrt( const complex_number& x ) {
    const auto x_real = real( x );
    const auto x_imag = imag( x );
    const auto r = sqrt( x_real * x_real + x_imag * x_imag );
    return { sqrt( ( r + x_real ) / 2.0 ), sign( x_imag ) * sqrt( ( r - x_real ) / 2.0 ) };
}

CUDA_HOST_DEVICE static CUDA_INLINE real_number exp( const real_number x ) {
    return std::exp( x );
}

CUDA_HOST_DEVICE static CUDA_INLINE real_number cos( const real_number x ) {
    return std::cos( x );
}

CUDA_HOST_DEVICE static CUDA_INLINE real_number sin( const real_number x ) {
    return std::sin( x );
}

CUDA_HOST_DEVICE static CUDA_INLINE real_number sign( const real_number x ) {
    return x > 0.0 ? 1.0 : -1.0;
}

CUDA_HOST_DEVICE static CUDA_INLINE real_number abs( const complex_number& x ) {
    const auto x_real = real( x );
    const auto x_imag = imag( x );
    return sqrt(x_real * x_real + x_imag * x_imag);
}

CUDA_HOST_DEVICE static CUDA_INLINE real_number abs( const real_number x ) {
    return sqrt(x * x);
}

CUDA_HOST_DEVICE static CUDA_INLINE real_number abs2( const complex_number& x ) {
    const auto x_real = real( x );
    const auto x_imag = imag( x );
    return x_real * x_real + x_imag * x_imag;
}

CUDA_HOST_DEVICE static CUDA_INLINE real_number abs2( const real_number x ) {
    return x * x;
}

CUDA_HOST_DEVICE static CUDA_INLINE real_number log( const real_number x ) {
    return std::log( x );
}

// Naive Integer Power
CUDA_HOST_DEVICE static CUDA_INLINE complex_number pow( const complex_number& a, const int N ) {
    complex_number res = { 1.0, 0 };
    for ( int i = 0; i < abs( N ); i++ )
        res = res * a;
    return N > 0 ? res : 1. / res;    
}
CUDA_HOST_DEVICE static CUDA_INLINE real_number pow( const real_number& x, const real_number N ) {
    return std::pow(x, N);
}


/**
 * @brief Calculates the abs2 of a buffer of (complex) numbers
 * @param z The buffer to calculate the abs2 of.
 * @param buffer The buffer to save the result to.
 * @param size The size of the buffer.
 */
CUDA_HOST_DEVICE static CUDA_INLINE void cwiseAbs2( complex_number* z, real_number* buffer, int size ) {
#pragma omp parallel for
    for ( int i = 0; i < size; i++ )
        buffer[i] = real( z[i] ) * real( z[i] ) + imag( z[i] ) * imag( z[i] );
}

template <typename T>
CUDA_HOST_DEVICE static CUDA_INLINE void cwiseAbs2( T* z, real_number* buffer, int size ) {
#pragma omp parallel for
    for ( int i = 0; i < size; i++ )
        buffer[i] = real( z[i] ) * real( z[i] );
}

CUDA_HOST_DEVICE static CUDA_INLINE complex_number exp( complex_number z ) {
    return { exp( real(z) ) * cos( imag(z) ), exp( real(z) ) * sin( imag(z) ) };
}

static CUDA_INLINE real_number max( real_number x, real_number y ) {
    return x > y ? x : y;
}
static CUDA_INLINE real_number min( real_number x, real_number y ) {
    return x < y ? x : y;
}

} // namespace PC3::CUDA