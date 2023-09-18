#pragma once

// #define USECPU

#ifndef USECPU
#    define CHECK_CUDA_ERROR( func, msg )                             \
        {                                                             \
            func;                                                     \
            cudaError_t err = cudaGetLastError();                     \
            if ( err != cudaSuccess ) {                               \
                printf( "%s: %s\n", msg, cudaGetErrorString( err ) ); \
            }                                                         \
        }
#    define DEVICE_ALLOC( ptr, size, name )                             \
        {                                                               \
            CHECK_CUDA_ERROR( cudaMalloc( (void**)&ptr, size ), name ); \
        }
#else
#    define CHECK_CUDA_ERROR( func, msg ) \
        {                                 \
            func;                         \
        }
#    define DEVICE_ALLOC( ptr, size, name ) \
        {                                   \
            ptr = std::make_unique<decltype(ptr)[]>(size); \
        }
#endif

#define swap_symbol( a, b ) \
    {                       \
        auto tmp = a;       \
        a = b;              \
        b = tmp;            \
    }

#ifdef USECPU
#    define CUDA_HOST_DEVICE
#    define CUDA_DEVICE
#    define CUDA_HOST
#    define CUDA_GLOBAL
#else
#    define CUDA_HOST_DEVICE __host__ __device__
#    define CUDA_DEVICE __device__
#    define CUDA_HOST __host__
#    define CUDA_GLOBAL __global__
#endif

// TODO: macros für CALL_CUDA_KERNEL, dass dann die dims übergibt, oder wenn cpu
// gewählt ist die funktion mittels loop und system.s_N über openmp aufruft.