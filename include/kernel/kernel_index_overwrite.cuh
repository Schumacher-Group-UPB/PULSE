#pragma once

namespace PC3::Kernel {

#ifndef USECPU
// If the GPU is used, overwrite the current index with the gpu thread index.
#define OVERWRITE_THREAD_INDEX( i ) \
    i += blockIdx.x * blockDim.x + threadIdx.x; \
    if (i >= p.N2) return;
#define GET_THREAD_INDEX( N ) \
    int i = blockIdx.x * blockDim.x + threadIdx.x; \
    if (i >= N) return;
#else
// Else the macro is empty.
#define OVERWRITE_THREAD_INDEX( i )
#define GET_THREAD_INDEX( N )
#endif

}

//int blockId = ( gridDim.x * blockIdx.y ) + blockIdx.x; \
//i += ( blockId * ( blockDim.x * blockDim.y ) ) + ( threadIdx.y * blockDim.x ) + threadIdx.x;