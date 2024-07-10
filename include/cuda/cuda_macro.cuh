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
    #define CALL_KERNEL( func, name, grid, block, ... ) \
        {                                               \
            func<<<grid, block>>>( 0, __VA_ARGS__ );    \
            CHECK_CUDA_ERROR( {}, name );               \
        }
    
    // Partially Calls a Kernel with less threads, thus only executing the 
    // Kernel on a subset of indices. A CUDA stream can be passed, theoretically
    // enabling the parallel execution of Kernels. The range of the subset executed
    // is determinded by the grid and block sizes.
    #define CALL_PARTIAL_KERNEL( func, name, grid, block, start, stream, ... ) \
        {                                                                      \
            func<<<grid, block, 0, stream>>>( start, __VA_ARGS__ );            \
            CHECK_CUDA_ERROR( {}, name );                                      \
        }

#else
    // On the CPU, the check for CUDA errors does nothing
    #define CHECK_CUDA_ERROR( func, msg )
    // On the CPU, the Kernel call does not execute a parallel GPU Kernel. Instead,
    // it launches a group of threads using a #pragma omp instruction. This executes
    // the Kernel in parallel on the CPU. 
    #define CALL_KERNEL( func, name, grid, block, ... )                                                                                                 \
        {                                                                                                                                               \
            _Pragma( "omp parallel for schedule(dynamic) num_threads(system.omp_max_threads)" ) for ( size_t i = 0; i < system.p.N_y; ++i ) {           \
                for ( size_t j = 0; j < system.p.N_x; ++j ) {                                                                                           \
                    const size_t index = i * system.p.N_x + j;                                                                                          \
                    func( index, __VA_ARGS__ );                                                                                                         \
                }                                                                                                                                       \
            }                                                                                                                                           \
        }
    // Partially calls a Kernel with less threads. Stream does nothing here.
    // The range of the subset executed is also determiend by the grid and block sizes.
    #define CALL_PARTIAL_KERNEL( func, name, grid, block, start, stream, ... )                                                                  \
        {                                                                                                                                       \
            size_t total_threads = gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;                                    \
            _Pragma( "omp parallel for schedule(dynamic) num_threads(system.omp_max_threads)" ) for ( int i = start; i < total_threads; ++i ) { \
                    func( i, __VA_ARGS__ );                                                                                                     \
            }                                                                                                                                   \
        }
#endif

// Swaps symbols a and b
#define swap_symbol( a, b ) \
    {                       \
        auto tmp = a;       \
        a = b;              \
        b = tmp;            \
    }
