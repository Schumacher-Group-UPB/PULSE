#pragma once
#include "cuda/typedef.cuh"

namespace PC3::Kernel {

#ifdef USE_CUDA
    // If the GPU is used, overwrite the current index with the gpu thread index.
    #define OVERWRITE_THREAD_INDEX( i ) \
        i += blockIdx.x * blockDim.x + threadIdx.x; \
        if (i >= args.p.N2) return;
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

    // Else the macro is empty.
    #define OVERWRITE_THREAD_INDEX( i )
    #define GENERATE_THREAD_INDEX( N ) \
        int i = 0;
    #define GET_THREAD_INDEX( i, N )

    #define LOCAL_SHARE_STRUCT( T, in, out ) \
        T& out = in;

#endif

}

//int blockId = ( gridDim.x * blockIdx.y ) + blockIdx.x;
//i += ( blockId * ( blockDim.x * blockDim.y ) ) + ( threadIdx.y * blockDim.x ) + threadIdx.x;