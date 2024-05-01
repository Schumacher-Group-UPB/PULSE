#pragma once
#include "cuda/cuda_complex.cuh"
#include "cuda/cuda_macro.cuh"
#include "solver/matrix_folder.hpp"
#include "system/system.hpp"

namespace PC3::Kernel {
    CUDA_GLOBAL void initialize_random_number_generator(int i, unsigned int seed, cuda_random_state* state, const unsigned int N);
    CUDA_GLOBAL void generate_random_numbers(int i, cuda_random_state* state, complex_number* buffer, const unsigned int N, const real_number real_amp, const real_number imag_amp);

} // namespace PC3::Kernel