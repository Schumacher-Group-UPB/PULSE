#pragma once

#define CHECK_CUDA_ERROR( func, msg )                             \
    {                                                             \
        func;                                                     \
        cudaError_t err = cudaGetLastError();                     \
        if ( err != cudaSuccess ) {                               \
            printf( "%s: %s\n", msg, cudaGetErrorString( err ) ); \
        }                                                         \
    }

#define swap_symbol( a, b )  \
    {                 \
        auto tmp = a; \
        a = b;        \
        b = tmp;      \
    }