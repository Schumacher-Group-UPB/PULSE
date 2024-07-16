#pragma once

/**
 * This file includes the USE_CPU and USE_CUDA preprocessor macros and should
 * ALWAYS be included as the first file in this project.
*/

#ifdef USE_CPU

    // Complex Host Numbers
    #include <complex>
    #include <vector>
    #include <random>
#else
    // Include the required CUDA headers
    #include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
    #include <cuda.h>
    #include <device_launch_parameters.h>
    #include <cuda_runtime.h>
    #include "cuda/helper_cuda.h"

    // We use Thrust for most of the type definitions on the GPU
    // Complex Host and Device Numbers
    #include <thrust/complex.h>
    #include <thrust/functional.h>
    #include <thrust/host_vector.h>
    #include <thrust/device_vector.h>
    #include <thrust/copy.h>
    #include <thrust/device_ptr.h>
    #include <curand_kernel.h>

    // Define Helper Macro so we dont always have to use "ifneq USE_CPU"
    #define USE_CUDA
#endif

#include <cmath>
    
#include "cuda/cuda_macro.cuh"

namespace PC3::Type {

    // Real Numbers
    #ifdef USE_HALF_PRECISION
        using real = float;
    #else
        using real = double;
    #endif

    #ifdef USE_CPU
        // std::complex numbers are host only
        using complex = std::complex<real>;
    #else
        // Thurst complex numbers are host and device
        using complex = thrust::complex<real>;
    #endif

    // Host and Device Matrix. This is a std::vector when using GCC and a thrust::vector when using nvcc
    #ifdef USE_CPU
        template <typename T>
        using host_vector = std::vector<T>;
        template <typename T>
        using device_vector = std::vector<T>;
    #else
        template <typename T>
        using host_vector = thrust::host_vector<T>;
        template <typename T>
        using device_vector = thrust::device_vector<T>;
    #endif

    // Random Generator States
    #ifdef USE_CPU
        using cuda_random_state = std::mt19937;
    #else
        using cuda_random_state = curandState;
    #endif
}

#ifdef USE_CPU
    // Define PULSE_INLINE as inline when using the CPU
    #define PULSE_INLINE inline
    #define PULSE_HOST_DEVICE
    #define PULSE_DEVICE
    #define PULSE_HOST
    #define PULSE_GLOBAL
    #define PULSE_RESTRICT
#else
    // Define PULSE_INLINE as nvcc's __inline__ when using the GPU
    #define PULSE_INLINE __inline__
    #define PULSE_HOST_DEVICE __host__ __device__
    #define PULSE_DEVICE __device__
    #define PULSE_HOST __host__
    #define PULSE_GLOBAL __global__
    #define PULSE_RESTRICT __restrict__
#endif

// If nvcc is not used, redefine dim3
#ifdef USE_CPU
    class dim3 {
        public:
         int x, y;
    };
#endif

/**
 * Basic Operators for cuComplex and cuDoubleComplex GPU types
*/

namespace PC3::CUDA {
    
    // Define real() and imag() for the GPU
    PULSE_HOST_DEVICE static PULSE_INLINE Type::real real( const Type::complex& x ) {
        return x.real();
    }
    PULSE_HOST_DEVICE static PULSE_INLINE Type::real imag( const Type::complex& x ) {
        return x.imag();
    }

    // For real numbers, the operators can be the same
    PULSE_HOST_DEVICE static PULSE_INLINE Type::real real( const Type::real& x ) {
        return x;
    }
    PULSE_HOST_DEVICE static PULSE_INLINE Type::real imag( const Type::real& x ) {
        return Type::real(0.0);
    }

    PULSE_HOST_DEVICE static PULSE_INLINE Type::real abs2( const Type::complex& x ) {
        const auto x_real = real( x );
        const auto x_imag = imag( x );
        return x_real * x_real + x_imag * x_imag;
    }

    PULSE_HOST_DEVICE static PULSE_INLINE Type::real abs2( const Type::real x ) {
        return x * x;
    }

    PULSE_HOST_DEVICE static PULSE_INLINE Type::real arg( const Type::complex z ) {
        #ifdef USE_CPU
            return std::arg(z);
        #else
            return thrust::arg(z);
        #endif
    }

    // Some Functions require thrust::__ to be called instead of std::__
    // We define those functions here. Simply "using x" in our own namespace
    // will map the function to CUDA::x. Very interesting mechanic.
    #ifdef USE_CPU 
        using std::exp;
        using std::sqrt;
        using std::log;
        using std::atan2;
        using std::max;
        using std::min;
        using std::tan;
        using std::pow;
        using std::sin;
        using std::cos;
        using std::abs;
    #else
        using thrust::exp;
        using thrust::sqrt;
        using thrust::log;
        using std::atan2;
        using thrust::max;
        using thrust::min;
        using thrust::tan;
        using thrust::pow;
        using thrust::sin;
        using thrust::cos;
        using thrust::abs;
    #endif

} // namespace PC3::CUDA
