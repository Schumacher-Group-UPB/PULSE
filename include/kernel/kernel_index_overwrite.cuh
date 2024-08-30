#pragma once
#include "cuda/typedef.cuh"

namespace PC3::Kernel {

#ifdef USE_CUDA
    // If the GPU is used, overwrite the current index with the gpu thread index.
    #define GENERATE_SUBGRID_INDEX(i, ch) \
        i = blockIdx.x * blockDim.x + threadIdx.x; \
        if (i >= (args.p.subgrid_N_x + 2*ch)*(args.p.subgrid_N_y + 2*ch)) return; \
        Type::uint r = i / (args.p.subgrid_N_x + 2*ch); \
        Type::uint c = i % (args.p.subgrid_N_x + 2*ch); \
        i = (args.p.subgrid_N_x + 2*args.p.halo_size)*(r+args.p.halo_size-ch) + args.p.halo_size - ch + c; 
    #define GENERATE_THREAD_INDEX( N ) \
        int i = blockIdx.x * blockDim.x + threadIdx.x; \
        if (i >= N) return;
    #define GET_THREAD_INDEX( i, N ) \
        i = blockIdx.x * blockDim.x + threadIdx.x; \
        if (i >= N) return;

    #define LOCAL_SHARE_STRUCT( T, in, out ) \
        __shared__ T out; \
        if (threadIdx.x == 0) { \
            out = in; \
        } \
        __syncthreads();
#else
    #define GENERATE_SUBGRID_INDEX(i, ch) \
        if (i >= (args.p.subgrid_N_x + 2*ch)*(args.p.subgrid_N_y + 2*ch)) return; \
        Type::uint r = i / (args.p.subgrid_N_x + 2*ch); \
        Type::uint c = i % (args.p.subgrid_N_x + 2*ch); \
        i = (args.p.subgrid_N_x + 2*args.p.halo_size)*(r+args.p.halo_size-ch) + args.p.halo_size - ch + c; 
    // Else the macro is empty.
    #define GENERATE_THREAD_INDEX( N ) \
        int i = 0;
    #define GET_THREAD_INDEX( i, N ) \
        if (i >= N) return;

    #define LOCAL_SHARE_STRUCT( T, in, out ) \
        T& out = in;

#endif

}

//int blockId = ( gridDim.x * blockIdx.y ) + blockIdx.x;
//i += ( blockId * ( blockDim.x * blockDim.y ) ) + ( threadIdx.y * blockDim.x ) + threadIdx.x;