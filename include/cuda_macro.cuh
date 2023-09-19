#pragma once

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
#    define MEMCOPY_TO_DEVICE( dst, src, size, name )                                        \
        {                                                                                   \
            CHECK_CUDA_ERROR( cudaMemcpy( dst, src, size, cudaMemcpyHostToDevice ), name ); \
        }
#    define MEMCOPY_FROM_DEVICE( dst, src, size, name )                                        \
        {                                                                                   \
            CHECK_CUDA_ERROR( cudaMemcpy( dst, src, size, cudaMemcpyDeviceToHost ), name ); \
        }
#    define SYMBOL_TO_DEVICE( dest, source, size, name )                             \
        {                                                               \
            CHECK_CUDA_ERROR( cudaMemcpyToSymbol( dest, source, size ), name ); \
        }
#    define SYMBOL_TO_HOST( dest, source, size, name )                             \
        {                                                               \
            CHECK_CUDA_ERROR( cudaMemcpyFromSymbol( dest, source, size ), name ); \
        }
#    define DEVICE_FREE( ptr, name ) \
        {                             \
            CHECK_CUDA_ERROR( cudaFree( ptr ), name ); \
        }
#    define CUDA_FFT_DESTROY( plan ) \
        {                                      \
            CHECK_CUDA_ERROR( cufftDestroy( plan ), "FFT Plan Destroy" ); \
        }
#    define CUDA_FFT_CREATE( plan, N ) \
        {                                      \
            CHECK_CUDA_ERROR( cufftPlan2d( plan, N, N, FFTPLAN ), "FFT Plan" ); \
        }
#    define CALL_KERNEL( func, name, grid, block, ... ) \
        {                                                         \
            func<<<grid, block>>>( 0, __VA_ARGS__ ); \
            CHECK_CUDA_ERROR( {}, name );\
        }
        
#else
#    define CHECK_CUDA_ERROR( func, msg )
#define DEVICE_ALLOC( ptr, size, name ) \
        {                                   \
            ptr = (decltype(ptr))malloc(size); \
        }
#define MEMCOPY_TO_DEVICE( dst, src, size, name ) \
        {                                         \
            memcpy( dst, src, size );             \
        }
#define MEMCOPY_FROM_DEVICE( dst, src, size, name ) \
        {                                         \
            memcpy( dst, src, size );             \
        }
#define SYMBOL_TO_DEVICE( dest, source, size, name ) \
        {                                         \
            dest = *(source);\
        }
#define SYMBOL_TO_HOST( dest, source, size, name ) \
        {                                         \
            dest = *(source);\
        }
#define DEVICE_FREE( ptr, name ) \
        {                           \
            free( ptr );            \
        }
#define CUDA_FFT_DESTROY( plan )
#define CUDA_FFT_CREATE( plan, N )
#define CALL_KERNEL( func, name, grid, block, ... ) \
        {                                                         \
        _Pragma( "omp parallel for schedule(dynamic) num_threads(system.omp_max_threads)" ) \
            for ( int i = 0; i < system.s_N; ++i ) { \
            for ( int j = 0; j < system.s_N; ++j ) { \
                const auto index = i * system.s_N + j; \
                func( index, __VA_ARGS__ ); \
            } \
            } \
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
#    define THRUST_DEVICE thrust::host
#else
#    define CUDA_HOST_DEVICE __host__ __device__
#    define CUDA_DEVICE __device__
#    define CUDA_HOST __host__
#    define CUDA_GLOBAL __global__
#    define THRUST_DEVICE thrust::device
#endif

// TODO: macros für CALL_CUDA_KERNEL, dass dann die dims übergibt, oder wenn cpu
// gewählt ist die funktion mittels loop und system.s_N über openmp aufruft.