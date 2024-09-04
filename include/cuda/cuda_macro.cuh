#pragma once

#include "cuda/typedef.cuh"

// Helper Macro to iterate a specific RK K. // Only Callable from within the solver
#define CALCULATE_K( index, input_wavefunction, input_reservoir )                                                                                     \
    {                                                                                                                                                 \
        const Type::uint32 current_halo = system.p.halo_size - index;                                                                                 \
        auto [current_block, current_grid] = getLaunchParameters( system.p.subgrid_N_x + 2 * current_halo, system.p.subgrid_N_y + 2 * current_halo ); \
        CALL_KERNEL( RUNGE_FUNCTION_GP, "K" #index, current_grid, current_block, stream, current_halo, getCurrentTime(), kernel_arguments,            \
                     { matrix.input_wavefunction##_plus.getDevicePtr( subgrid ), matrix.input_wavefunction##_minus.getDevicePtr( subgrid ),           \
                       matrix.input_reservoir##_plus.getDevicePtr( subgrid ), matrix.input_reservoir##_minus.getDevicePtr( subgrid ),                 \
                       matrix.k_wavefunction_plus.getDevicePtr( subgrid, index - 1 ), matrix.k_wavefunction_minus.getDevicePtr( subgrid, index - 1 ), \
                       matrix.k_reservoir_plus.getDevicePtr( subgrid, index - 1 ), matrix.k_reservoir_minus.getDevicePtr( subgrid, index - 1 ) } );   \
    };

// For some reason, GCC needs this to correctly unpack the variadic arguments into a templated function
#define GCC_EXPAND_VA_ARGS( ... ) __VA_ARGS__
// Only Callable from within the solver
#define INTERMEDIATE_SUM_K( index, ... )                                                                                                                                 \
    {                                                                                                                                                                    \
        Type::uint32 current_halo = system.p.halo_size - index;                                                                                                          \
        auto [current_block, current_grid] = getLaunchParameters( system.p.subgrid_N_x + 2 * current_halo, system.p.subgrid_N_y + 2 * current_halo );                    \
        CALL_KERNEL( Kernel::Summation::runge_sum_to_input_kw<GCC_EXPAND_VA_ARGS( __VA_ARGS__ )>, "Sum for K" #index, current_grid, current_block, stream, current_halo, \
                     getCurrentTime(), kernel_arguments,                                                                                                                 \
                     Solver::InputOutput{ matrix.wavefunction_plus.getDevicePtr( subgrid ), matrix.wavefunction_minus.getDevicePtr( subgrid ),                           \
                                          matrix.reservoir_plus.getDevicePtr( subgrid ), matrix.reservoir_minus.getDevicePtr( subgrid ),                                 \
                                          matrix.buffer_wavefunction_plus.getDevicePtr( subgrid ), matrix.buffer_wavefunction_minus.getDevicePtr( subgrid ),             \
                                          matrix.buffer_reservoir_plus.getDevicePtr( subgrid ), matrix.buffer_reservoir_minus.getDevicePtr( subgrid ) } );               \
    };

// Only Callable from within the solver
#define FINAL_SUM_K( ... )                                                                                                                                                 \
    {                                                                                                                                                                      \
        auto [current_block, current_grid] = getLaunchParameters( system.p.subgrid_N_x, system.p.subgrid_N_y );                                                            \
        CALL_KERNEL( Kernel::Summation::runge_sum_to_input_kw<GCC_EXPAND_VA_ARGS( __VA_ARGS__ )>, "Sum for Psi", current_grid, current_block, stream, 0, getCurrentTime(), \
                     kernel_arguments,                                                                                                                                     \
                     Solver::InputOutput{ matrix.wavefunction_plus.getDevicePtr( subgrid ), matrix.wavefunction_minus.getDevicePtr( subgrid ),                             \
                                          matrix.reservoir_plus.getDevicePtr( subgrid ), matrix.reservoir_minus.getDevicePtr( subgrid ),                                   \
                                          matrix.wavefunction_plus.getDevicePtr( subgrid ), matrix.wavefunction_minus.getDevicePtr( subgrid ),                             \
                                          matrix.reservoir_plus.getDevicePtr( subgrid ), matrix.reservoir_minus.getDevicePtr( subgrid ) } );                               \
    };

#define ERROR_K( ... )                                                                                                                                               \
    {                                                                                                                                                                \
        auto [current_block, current_grid] = getLaunchParameters( system.p.subgrid_N_x, system.p.subgrid_N_y );                                                      \
        CALL_KERNEL( Kernel::Summation::runge_sum_to_error<GCC_EXPAND_VA_ARGS( __VA_ARGS__ )>, "Error", current_grid, current_block, stream, 0, getCurrentTime(), \
                     kernel_arguments );                                                                                                                             \
    };

// Only Callable from within the solver
// For now, use this macro to synchronize the halos. This is a bit of a mess, but it works. TODO: move halo_map to static CUDAMatrix vector and call synchronize_halos from there.
#define SYNCHRONIZE_HALOS( _stream, subgrids )                                                                                                                                 \
    {                                                                                                                                                                          \
        auto [current_block, current_grid] = getLaunchParameters( matrix.halo_map.size() / 6 * system.p.subgrids_x * system.p.subgrids_y );                                    \
        CALL_KERNEL( Kernel::Halo::synchronize_halos, "Synchronization", current_grid, current_block, _stream, system.p.subgrids_x, system.p.subgrids_y, system.p.subgrid_N_x, \
                     system.p.subgrid_N_y, system.p.halo_size, matrix.halo_map.size() / 6, system.p.periodic_boundary_x, system.p.periodic_boundary_y,                         \
                     GET_RAW_PTR( matrix.halo_map ), subgrids )                                                                                                                \
    }

// Helper to retrieve the raw device pointer. When using nvcc and thrust, we need a raw pointer cast.
#ifdef USE_CPU
    #define GET_RAW_PTR( vec ) vec.data()
#else
    #define GET_RAW_PTR( vec ) thrust::raw_pointer_cast( vec.data() )
#endif

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
        { func<<<grid, block, 0, stream>>>( 0, __VA_ARGS__ ); }
    // Wraps the successive calls to the CUDA Kernels into a single CUDA Graph.
    // Edit: Oh God what a mess.
    #define SOLVER_SEQUENCE( with_graph, content )                                                                                                        \
        {                                                                                                                                                 \
            static bool cuda_graph_created = false;                                                                                                       \
            static cudaGraph_t graph;                                                                                                                     \
            static cudaGraphExec_t instance;                                                                                                              \
            static cudaStream_t stream;                                                                                                                   \
            static cudaGraphNode_t* nodes;                                                                                                                \
            static size_t num_nodes;                                                                                                                      \
            if ( not cuda_graph_created or not with_graph ) {                                                                                             \
                std::vector<Solver::KernelArguments> v_kernel_arguments;                                                                                  \
                for ( Type::uint32 subgrid = 0; subgrid < system.p.subgrids_x * system.p.subgrids_y; subgrid++ ) {                                        \
                    v_kernel_arguments.push_back( generateKernelArguments( subgrid ) );                                                                   \
                }                                                                                                                                         \
                if ( with_graph ) {                                                                                                                       \
                    cudaStreamCreate( &stream );                                                                                                          \
                    cudaStreamBeginCapture( stream, cudaStreamCaptureModeGlobal );                                                                        \
                    std::cout << PC3::CLIO::prettyPrint( "Capturing CUDA Graph", PC3::CLIO::Control::Secondary | PC3::CLIO::Control::Info ) << std::endl; \
                }                                                                                                                                         \
                if ( system.p.use_twin_mode ) {                                                                                                           \
                    SYNCHRONIZE_HALOS( stream, matrix.wavefunction_plus.getSubgridDevicePtrs() );                                                         \
                    SYNCHRONIZE_HALOS( stream, matrix.reservoir_plus.getSubgridDevicePtrs() );                                                            \
                    SYNCHRONIZE_HALOS( stream, matrix.wavefunction_minus.getSubgridDevicePtrs() );                                                        \
                    SYNCHRONIZE_HALOS( stream, matrix.reservoir_minus.getSubgridDevicePtrs() );                                                           \
                } else {                                                                                                                                  \
                    SYNCHRONIZE_HALOS( stream, matrix.wavefunction_plus.getSubgridDevicePtrs() );                                                         \
                    SYNCHRONIZE_HALOS( stream, matrix.reservoir_plus.getSubgridDevicePtrs() );                                                            \
                }                                                                                                                                         \
                for ( Type::uint32 subgrid = 0; subgrid < system.p.subgrids_x * system.p.subgrids_y; subgrid++ ) {                                        \
                    auto& kernel_arguments = v_kernel_arguments[subgrid];                                                                                 \
                    content;                                                                                                                              \
                }                                                                                                                                         \
                if ( with_graph ) {                                                                                                                       \
                    cudaStreamEndCapture( stream, &graph );                                                                                               \
                    cudaGraphInstantiate( &instance, graph, NULL, NULL, 0 );                                                                              \
                    cuda_graph_created = true;                                                                                                            \
                    cudaGraphGetNodes( graph, nullptr, &num_nodes );                                                                                      \
                    nodes = new cudaGraphNode_t[num_nodes];                                                                                               \
                    cudaGraphGetNodes( graph, nodes, &num_nodes );                                                                                        \
                    std::cout << PC3::CLIO::prettyPrint( "CUDA Graph created with " + std::to_string( num_nodes ) + " nodes",                             \
                                                         PC3::CLIO::Control::Secondary | PC3::CLIO::Control::Info )                                       \
                              << std::endl;                                                                                                               \
                }                                                                                                                                         \
            } else {                                                                                                                                      \
                Solver::VKernelArguments v_time = getCurrentTime();                                                                                       \
                if ( with_graph ) {                                                                                                                       \
                    for ( Type::uint32 i = 0; i < num_nodes; i++ ) {                                                                                      \
                        cudaKernelNodeParams knp;                                                                                                         \
                        cudaGraphKernelNodeGetParams( nodes[i], &knp );                                                                                   \
                        if ( knp.func == (void*)( RUNGE_FUNCTION_GP ) ) {                                                                                 \
                            knp.kernelParams[2] = &v_time;                                                                                                \
                            cudaGraphExecKernelNodeSetParams( instance, nodes[i], &knp );                                                                 \
                        }                                                                                                                                 \
                    }                                                                                                                                     \
                }                                                                                                                                         \
                cudaGraphLaunch( instance, stream );                                                                                                      \
            }                                                                                                                                             \
        }

#else
    // On the CPU, the check for CUDA errors does nothing
    #define CHECK_CUDA_ERROR( func, msg )
    // On the CPU, the Kernel call does not execute a parallel GPU Kernel. Instead,
    // it launches a group of threads using a #pragma omp instruction.
    #include <functional>
    #define CALL_KERNEL( func, name, grid, block, stream, ... ) \
        {                                                       \
            for ( Type::uint32 i = 0; i < block.x; ++i ) {      \
                for ( Type::uint32 j = 0; j < grid.x; ++j ) {   \
                    const Type::uint32 index = i * grid.x + j;  \
                    func( index, __VA_ARGS__ );                 \
                }                                               \
            }                                                   \
        }
    // Merges the Kernel calls into a single function call. This is not required on the CPU.
    #define SOLVER_SEQUENCE( with_graph, content )                                                                                                            \
        {                                                                                                                                                     \
            PC3::Type::stream_t stream;                                                                                                                       \
            if ( system.p.use_twin_mode ) {                                                                                                                   \
                SYNCHRONIZE_HALOS( stream, matrix.wavefunction_plus.getSubgridDevicePtrs() )                                                                  \
                SYNCHRONIZE_HALOS( stream, matrix.reservoir_plus.getSubgridDevicePtrs() )                                                                     \
                SYNCHRONIZE_HALOS( stream, matrix.wavefunction_minus.getSubgridDevicePtrs() )                                                                 \
                SYNCHRONIZE_HALOS( stream, matrix.reservoir_minus.getSubgridDevicePtrs() )                                                                    \
            } else {                                                                                                                                          \
                SYNCHRONIZE_HALOS( stream, matrix.wavefunction_plus.getSubgridDevicePtrs() )                                                                  \
                SYNCHRONIZE_HALOS( stream, matrix.reservoir_plus.getSubgridDevicePtrs() )                                                                     \
            }                                                                                                                                                 \
            _Pragma( "omp parallel for schedule(static)" ) for ( Type::uint32 subgrid = 0; subgrid < system.p.subgrids_x * system.p.subgrids_y; subgrid++ ) { \
                auto kernel_arguments = generateKernelArguments( subgrid );                                                                                   \
                content;                                                                                                                                      \
            }                                                                                                                                                 \
        }
#endif

// Swaps symbols a and b
#define swap_symbol( a, b ) \
    {                       \
        auto tmp = a;       \
        a = b;              \
        b = tmp;            \
    }

// CUDA Specific Alloc and Free
#ifndef USE_CPU
    #define DEVICE_ALLOC( ptr, size, name ) \
        { CHECK_CUDA_ERROR( cudaMalloc( (void**)&ptr, size ), name ); }
    #define MEMCOPY_TO_DEVICE( dst, src, size, name ) \
        { CHECK_CUDA_ERROR( cudaMemcpy( dst, src, size, cudaMemcpyHostToDevice ), name ); }
    #define MEMCOPY_FROM_DEVICE( dst, src, size, name ) \
        { CHECK_CUDA_ERROR( cudaMemcpy( dst, src, size, cudaMemcpyDeviceToHost ), name ); }
    #define SYMBOL_TO_DEVICE( dest, source, size, name ) \
        { CHECK_CUDA_ERROR( cudaMemcpyToSymbol( dest, source, size ), name ); }
    #define SYMBOL_TO_HOST( dest, source, size, name ) \
        { CHECK_CUDA_ERROR( cudaMemcpyFromSymbol( dest, source, size ), name ); }
    #define DEVICE_FREE( ptr, name ) \
        { CHECK_CUDA_ERROR( cudaFree( ptr ), name ); }
#else
    #define DEVICE_ALLOC( ptr, size, name ) \
        { ptr = (decltype( ptr ))malloc( size ); }
    #define MEMCOPY_TO_DEVICE( dst, src, size, name ) \
        { memcpy( dst, src, size ); }
    #define MEMCOPY_FROM_DEVICE( dst, src, size, name ) \
        { memcpy( dst, src, size ); }
    #define SYMBOL_TO_DEVICE( dest, source, size, name ) \
        { dest = *( source ); }
    #define SYMBOL_TO_HOST( dest, source, size, name ) \
        { dest = *( source ); }
    #define DEVICE_FREE( ptr, name ) \
        { free( ptr ); }
#endif