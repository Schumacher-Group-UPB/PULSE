#pragma once
#include "cuda/typedef.cuh"
#include "cuda/cuda_macro.cuh"

struct compare_complex_abs2 {
    PULSE_HOST_DEVICE bool operator()( PC3::Type::complex lhs, PC3::Type::complex rhs ) {
        return PC3::CUDA::real(lhs) * PC3::CUDA::real(lhs) + PC3::CUDA::imag(lhs) * PC3::CUDA::imag(lhs) < PC3::CUDA::real(rhs) * PC3::CUDA::real(rhs) + PC3::CUDA::imag(rhs) * PC3::CUDA::imag(rhs);
    }
};

namespace PC3::CUDA {

/**
 * @brief Calculates the minimum and maximum of a buffer of (complex) numbers
 * @param z The buffer to calculate the minimum and maximum of.
 * @param size The size of the buffer.
 * @return std::tuple<PC3::Type::real, PC3::Type::real> A tuple containing the minimum and maximum
 */
std::tuple<Type::real, Type::real> minmax( Type::complex* buffer, int size, bool device_pointer = false );
std::tuple<Type::real, Type::real> minmax( Type::real* buffer, int size, bool device_pointer = false );

/**
 * @brief Normalizes a buffer of real numbers using the minimum and maximum
 * values passed to the function. If min == max == 0, the min and max are
 * recalculated using the minmax function.
 * @param buffer The buffer to normalize.
 * @param size The size of the buffer.
 * @param min The minimum value to normalize to.
 * @param max The maximum value to normalize to.
 */
void normalize( Type::real* buffer, int size, Type::real min = 0, Type::real max = 0, bool device_pointer = false );

/**
 * @brief Calculates the angle of a buffer of complex numbers, as long as
 * std::arg is defined for the type T.
 * @param z The complex number buffer to calculate the angle of.
 * @param buffer The buffer to save the result to.
 * @param size The size of the buffer.
 */
void angle( Type::complex* z, Type::real* buffer, int size );

void cwiseAbs2( Type::complex* z, Type::real* buffer, int size );
void cwiseAbs2( Type::real* z, Type::real* buffer, int size );

} // namespace PC3::Kernel