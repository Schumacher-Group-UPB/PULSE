#pragma once
#include "cuda/typedef.cuh"
#include "cuda/cuda_macro.cuh"
#include "solver/matrix_container.hpp"
#include "system/system_parameters.hpp"

namespace PC3::Kernel {
    PULSE_GLOBAL void normalize_imaginary_time_propagation(int i, MatrixContainer::Pointers dev_ptrs, SystemParameters::KernelParameters p, Type::complex normalization_wavefunction, Type::complex normalization_reservoir);
} // namespace PC3::Kernel