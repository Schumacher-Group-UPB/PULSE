#pragma once

#ifdef USE_CUDA
    // Execudes a CUDA Command, checks for the latest error and prints it
    // This is technically not a requirement, but usually good practice
    #define CHECK_CUDA_ERROR( func, msg )                             \
        {                                                             \
            func;                                                     \
            cudaError_t err = cudaGetLastError();                     \
            if ( err != cudaSuccess ) {                               \
                printf( "%s: %s\n", msg, cudaGetErrorString( err ) ); \
            }                                                         \
        }

    // Calls a Kernel and also checks for errors.
    // The Kernel call requires a name and a grid and block size that 
    // are not further passed to the actual compute Kernel. Instead, they
    // are used as launch parameters and for debugging.
    #define CALL_KERNEL( func, name, grid, block, stream, ... ) \
        {                                               \
            func<<<grid, block, 0, stream>>>( 0, __VA_ARGS__ );    \
        }
    
    // Partially Calls a Kernel with less threads, thus only executing the 
    // Kernel on a subset of indices. A CUDA stream can be passed, theoretically
    // enabling the parallel execution of Kernels. The range of the subset executed
    // is determinded by the grid and block sizes.
    #define CALL_PARTIAL_KERNEL( func, name, grid, block, start, stream, ... ) \
        {                                                                      \
            func<<<grid, block, 0, stream>>>( start, __VA_ARGS__ );            \
        }
    // Wraps the successive calls to the CUDA Kernels into a single CUDA Graph.
    #define MERGED_CALL(content) \
        {                           \
            static bool cuda_graph_created = false; \
            static cudaGraph_t graph; \
            static cudaGraphExec_t instance; \
            static cudaStream_t stream; \
            if ( !cuda_graph_created ) { \
                cudaStreamCreate( &stream ); \
                cudaStreamBeginCapture( stream, cudaStreamCaptureModeGlobal ); \
                content; \
                cudaStreamEndCapture( stream, &graph ); \
                cudaGraphInstantiate( &instance, graph, NULL, NULL, 0 ); \
                cuda_graph_created = true; \
            } \
            else { \
                CHECK_CUDA_ERROR( cudaGraphLaunch( instance, stream ), "graph launch" ); \
            } \
        }

#else
    // On the CPU, the check for CUDA errors does nothing
    #define CHECK_CUDA_ERROR( func, msg )
    // On the CPU, the Kernel call does not execute a parallel GPU Kernel. Instead,
    // it launches a group of threads using a #pragma omp instruction. This executes
    // the Kernel in parallel on the CPU.
    #include <functional> 
    #define CALL_KERNEL( func, name, grid, block, stream, ... )                                                                                                 \
        {                                                                                                                                                \
            std::function exec_func = func;                                                                                                                                            \
            _Pragma( "omp parallel for schedule(static) num_threads(system.omp_max_threads)" ) for ( size_t i = 0; i < system.p.N_y; ++i ) {           \
                for ( size_t j = 0; j < system.p.N_x; ++j ) {                                                                                           \
                    const size_t index = i * system.p.N_x + j;                                                                                          \
                    exec_func( index, __VA_ARGS__ );                                                                                                         \
                }                                                                                                                                       \
            }                                                                                                                                           \
        }
    // Partially calls a Kernel with less threads. Stream does nothing here.
    // The range of the subset executed is also determiend by the grid and block sizes.
    #define CALL_PARTIAL_KERNEL( func, name, grid, block, start, stream, ... )                                                                  \
        {                                                                                                                                       \
            std::function exec_func = func;                                                                                                                                            \
            size_t total_threads = gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;                                    \
            _Pragma( "omp parallel for schedule(dynamic) num_threads(system.omp_max_threads)" ) for ( int i = start; i < total_threads; ++i ) { \
                    exec_func( i, __VA_ARGS__ );                                                                                                     \
            }                                                                                                                                   \
        }
    // Merges the Kernel calls into a single function call. This is not required on the CPU.
    #define MERGED_CALL(content) content
#endif

// Swaps symbols a and b
#define swap_symbol( a, b ) \
    {                       \
        auto tmp = a;       \
        a = b;              \
        b = tmp;            \
    }

// CUDA Specific Alloc and Free
#ifndef USECPU
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
#else
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
#endif