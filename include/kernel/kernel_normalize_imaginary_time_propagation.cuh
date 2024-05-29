#pragma once
#include "cuda/cuda_complex.cuh"
#include "cuda/cuda_macro.cuh"
#include "solver/matrix_container.hpp"
#include "system/system.hpp"

namespace PC3::Kernel {
    CUDA_GLOBAL void normalize_imaginary_time_propagation(int i, MatrixContainer::Pointers dev_ptrs, System::Parameters p, complex_number normalization_wavefunction, complex_number normalization_reservoir);
} // namespace PC3::Kernel