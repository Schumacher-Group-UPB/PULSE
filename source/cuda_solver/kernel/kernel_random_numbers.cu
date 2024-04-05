#include "kernel/kernel_random_numbers.cuh"
#include "kernel/kernel_index_overwrite.cuh"

CUDA_GLOBAL void PC3::Kernel::initialize_random_number_generator( int i, unsigned int seed, cuda_random_state* state, const unsigned int N) {
    GET_THREAD_INDEX( i, N );
    #ifdef USECPU
    state[i] = cuda_random_state(seed + i);
    #else
    curand_init(seed, i, 0, &state[i]);
    #endif 
}

CUDA_GLOBAL void PC3::Kernel::generate_random_numbers( int i, cuda_random_state* state, complex_number* buffer, const unsigned int N, const real_number real_amp, const real_number imag_amp) {
    GET_THREAD_INDEX( i, N );
    #ifdef USECPU
    buffer[i] = {state[i]() / state[i].max() * real_amp, state[i]() / state[i].max() * imag_amp};
    #else
    // there is no curand_uniform2, which is why we do it this way.
    buffer[i] = {curand_uniform(&state[i]) * real_amp, curand_uniform(&state[i]) * imag_amp};
    #endif
}