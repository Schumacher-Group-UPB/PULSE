#pragma once
#include "cuda/typedef.cuh"
#include "cuda/cuda_macro.cuh"
#include "solver/matrix_container.hpp"
#include "system/system_parameters.hpp"

namespace PC3::Kernel {
    PULSE_GLOBAL void initialize_random_number_generator(int i, unsigned int seed, Type::cuda_random_state* state, const unsigned int N);
    PULSE_GLOBAL void generate_random_numbers(int i, Type::cuda_random_state* state, Type::complex* buffer, const unsigned int N, const Type::real real_amp, const Type::real imag_amp);

} // namespace PC3::Kernel