#pragma once

namespace PC3::Kernel {

#ifndef USECPU
// If the GPU is used, overwrite the current index with the gpu thread index.
#define OVERWRITE_INDEX_GPU(i) \
    int blockId = ( gridDim.x * blockIdx.y ) + blockIdx.x; \
    i = ( blockId * ( blockDim.x * blockDim.y ) ) + ( threadIdx.y * blockDim.x ) + threadIdx.x;
#else
// Else the macro is empty.
#define OVERWRITE_INDEX_CPU(i)
#endif

}