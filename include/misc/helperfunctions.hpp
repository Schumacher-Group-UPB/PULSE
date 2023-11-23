#pragma once
#include "cuda/cuda_complex.cuh"

struct compare_complex_abs2 {
    CUDA_HOST_DEVICE bool operator()( complex_number lhs, complex_number rhs ) {
        return PC3::CUDA::real(lhs) * PC3::CUDA::real(lhs) + PC3::CUDA::imag(lhs) * PC3::CUDA::imag(lhs) < PC3::CUDA::real(rhs) * PC3::CUDA::real(rhs) + PC3::CUDA::imag(rhs) * PC3::CUDA::imag(rhs);
    }
};

namespace PC3::CUDA {

/**
 * @brief Calculates the minimum and maximum of a buffer of (complex) numbers
 * @param z The buffer to calculate the minimum and maximum of.
 * @param size The size of the buffer.
 * @return std::tuple<real_number, real_number> A tuple containing the minimum and maximum
 */
std::tuple<real_number, real_number> minmax( complex_number* buffer, int size, bool device_pointer = false );
std::tuple<real_number, real_number> minmax( real_number* buffer, int size, bool device_pointer = false );

/**
 * @brief Normalizes a buffer of real numbers using the minimum and maximum
 * values passed to the function. If min == max == 0, the min and max are
 * recalculated using the minmax function.
 * @param buffer The buffer to normalize.
 * @param size The size of the buffer.
 * @param min The minimum value to normalize to.
 * @param max The maximum value to normalize to.
 */
void normalize( real_number* buffer, int size, real_number min = 0, real_number max = 0, bool device_pointer = false );

/**
 * @brief Calculates the angle of a buffer of complex numbers, as long as
 * std::arg is defined for the type T.
 * @param z The complex number buffer to calculate the angle of.
 * @param buffer The buffer to save the result to.
 * @param size The size of the buffer.
 */
void angle( complex_number* z, real_number* buffer, int size );

} // namespace PC3::Kernel