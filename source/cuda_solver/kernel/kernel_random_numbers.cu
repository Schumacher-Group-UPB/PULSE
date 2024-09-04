#include "cuda/typedef.cuh"
#include "kernel/kernel_compute.cuh"
#include "kernel/kernel_index_overwrite.cuh"

PULSE_GLOBAL void PC3::Kernel::initialize_random_number_generator( int i, Type::uint32 seed, Type::cuda_random_state* state, const Type::uint32 N) {
    GET_THREAD_INDEX( i, N );
    #ifdef USE_CPU
        state[i] = Type::cuda_random_state(seed + i);
    #else
        curand_init(seed, i, 0, &state[i]);
    #endif 
}

#ifdef USE_CPU
      std::normal_distribution<PC3::Type::real> _local_normal_distribution(0.0, 1.0);
#endif

PULSE_GLOBAL void PC3::Kernel::generate_random_numbers( int i, Type::cuda_random_state* state, Type::complex* buffer, const Type::uint32 N, const Type::real real_amp, const Type::real imag_amp) {
    GET_THREAD_INDEX( i, N );
    #ifdef USE_CPU
        // TODO: normal distribution for cpu random numbers
        buffer[i] = {_local_normal_distribution(state[i])*real_amp, _local_normal_distribution(state[i])*imag_amp};
    #else
        float2 r = curand_normal2(&state[i]);
        buffer[i] = Type::complex( r.x * real_amp, r.y * imag_amp );
    #endif
}