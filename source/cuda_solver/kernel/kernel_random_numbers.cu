#include "cuda/typedef.cuh"
#include "kernel/kernel_compute.cuh"
#include "kernel/kernel_index_overwrite.cuh"

PULSE_GLOBAL void PC3::Kernel::initialize_random_number_generator( int i, unsigned int seed, Type::cuda_random_state* state, const unsigned int N) {
    GET_THREAD_INDEX( i, N );
    #ifdef USE_CPU
        state[i] = Type::cuda_random_state(seed + i);
    #else
        curand_init(seed, i, 0, &state[i]);
    #endif 
}

PULSE_GLOBAL void PC3::Kernel::generate_random_numbers( int i, Type::cuda_random_state* state, Type::complex* buffer, const unsigned int N, const Type::real real_amp, const Type::real imag_amp) {
    GET_THREAD_INDEX( i, N );
    #ifdef USE_CPU
        buffer[i] = {state[i]() / state[i].max() * real_amp, state[i]() / state[i].max() * imag_amp};
    #else
        // there is no curand_uniform2, which is why we do it this way.
        //buffer[i] = Type::complex( curand_normal(&state[i]) * real_amp, curand_normal(&state[i]) * imag_amp );
        float2 r = curand_normal2(&state[i]);
        buffer[i] = Type::complex( r.x * real_amp, r.y * imag_amp );
    #endif
}