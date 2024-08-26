#pragma once

// Helper Macro to iterate a specific RK K
#define CALCULATE_K( index, input_wavefunction, input_reservoir ) \
{ \
const size_t current_halo = system.p.halo_size - index; \
CALL_KERNEL( \
    RUNGE_FUNCTION_GP, "K"#index, grid_size, block_size, stream,  \
    current_halo, kernel_arguments, \
    {      \
        kernel_arguments.dev_ptrs.input_wavefunction##_plus, kernel_arguments.dev_ptrs.input_wavefunction##_minus, kernel_arguments.dev_ptrs.input_reservoir##_plus, kernel_arguments.dev_ptrs.input_reservoir##_minus, \
        kernel_arguments.dev_ptrs.k##index##_wavefunction_plus, kernel_arguments.dev_ptrs.k##index##_wavefunction_minus, kernel_arguments.dev_ptrs.k##index##_reservoir_plus, kernel_arguments.dev_ptrs.k##index##_reservoir_minus \
    } \
); \
};

#define INTERMEDIATE_SUM_K( index, ... ) \
{ \
const size_t current_halo = system.p.halo_size - index; \
CALL_KERNEL( \
    Kernel::RK::runge_sum_to_input_kw, "Sum for K"#index, grid_size, block_size, stream, \
    current_halo, kernel_arguments, { \
        kernel_arguments.dev_ptrs.wavefunction_plus, kernel_arguments.dev_ptrs.wavefunction_minus, \
        kernel_arguments.dev_ptrs.reservoir_plus, kernel_arguments.dev_ptrs.reservoir_minus, \
        kernel_arguments.dev_ptrs.buffer_wavefunction_plus, kernel_arguments.dev_ptrs.buffer_wavefunction_minus, \
        kernel_arguments.dev_ptrs.buffer_reservoir_plus, kernel_arguments.dev_ptrs.buffer_reservoir_minus \
    },\
    {__VA_ARGS__} \
); \
};

#define FINAL_SUM_K( ... ) \
{ \
CALL_KERNEL( \
    Kernel::RK::runge_sum_to_input_kw, "Sum for Psi", grid_size, block_size, stream, \
    0, kernel_arguments, { \
        kernel_arguments.dev_ptrs.wavefunction_plus, kernel_arguments.dev_ptrs.wavefunction_minus, \
        kernel_arguments.dev_ptrs.reservoir_plus, kernel_arguments.dev_ptrs.reservoir_minus, \
        kernel_arguments.dev_ptrs.wavefunction_plus, kernel_arguments.dev_ptrs.wavefunction_minus, \
        kernel_arguments.dev_ptrs.reservoir_plus, kernel_arguments.dev_ptrs.reservoir_minus \
    },\
    {__VA_ARGS__} \
); \
};

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
    // Wraps the successive calls to the CUDA Kernels into a single CUDA Graph.
    #define MERGED_CALL(content) \
        {                           \
            static bool cuda_graph_created = false; \
            static cudaGraph_t graph; \
            static cudaGraphExec_t instance; \
            static cudaStream_t stream; \
            dim3 block( system.block_size, 1 ); \
            dim3 grid( ( system.p.N_x*system.p.N_y + block_size.x ) / block_size.x, 1 ); \
            if ( !cuda_graph_created ) { \
                cudaStreamCreate( &stream ); \
                cudaStreamBeginCapture( stream, cudaStreamCaptureModeGlobal ); \
                for (size_t subgrid = 0; subgrid < system.p.subgrids_x * system.p.subgrids_y; subgrid++) { \
                    updateKernelArguments(subgrid); \
                    content; \
                } \
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
            auto exec_func = func;                                                                                                                                            \
            _Pragma( "omp parallel for schedule(static) num_threads(system.omp_max_threads)" ) \
            for (size_t subgrid = 0; subgrid < system.p.subgrids_x * system.p.subgrids_y; subgrid++)  \
                for ( size_t i = 0; i < system.p.subgrid_N_y; ++i ) {           \
                    for ( size_t j = 0; j < system.p.subgrid_N_x; ++j ) {                                                                                            \
                        const size_t index = i * system.p.subgrid_N_x + j;                                                                                          \
                        exec_func( index, subgrid, __VA_ARGS__ );                                                                                                         \
                    }                                                                                                                                   \
                }                                                                                                                                           \
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

// Helper to retrieve the raw device pointer. When using nvcc and thrust, we need a raw pointer cast.
#ifdef USECPU
    #define GET_RAW_PTR( vec ) vec.data()
#else
    #define GET_RAW_PTR( vec ) thrust::raw_pointer_cast( vec.data() )
#endif