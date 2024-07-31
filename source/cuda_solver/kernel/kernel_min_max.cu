#include "cuda/typedef.cuh"
#ifdef USE_CUDA
    #include <thrust/extrema.h>
    #include <thrust/execution_policy.h>
    #include <thrust/pair.h>
    #include <thrust/device_ptr.h>
#else
    #include <ranges>
    #include <algorithm>
#endif

#include "misc/helperfunctions.hpp"

// BIG TODO: WTF is this? Rework this. The device_pointer flag is not required if we use thrust:: vectors anyways.

std::tuple<PC3::Type::real, PC3::Type::real> PC3::CUDA::minmax( PC3::Type::complex* buffer, int size, bool device_pointer ) {
    #ifdef USE_CUDA
    if ( device_pointer ) {
        thrust::device_ptr<PC3::Type::complex> dev_buffer = thrust::device_pointer_cast( buffer );
        auto mm = thrust::minmax_element( thrust::device, dev_buffer, dev_buffer + size, compare_complex_abs2() );
        PC3::Type::complex min = *mm.first;
        PC3::Type::complex max = *mm.second;
        return std::make_tuple( std::sqrt(CUDA::real(min) * CUDA::real(min) + CUDA::imag(min) * CUDA::imag(min)), std::sqrt(CUDA::real(max) * CUDA::real(max) + CUDA::imag(max) * CUDA::imag(max)) );
    }
    const auto [first, second] = thrust::minmax_element( buffer, buffer + size, compare_complex_abs2() );
    #else
    const auto [first, second] = std::ranges::minmax_element( buffer, buffer + size, compare_complex_abs2() );
    #endif
    return std::make_tuple( std::sqrt(CUDA::real( *first ) * CUDA::real( *first ) + CUDA::imag( *first ) * CUDA::imag( *first )), std::sqrt(CUDA::real( *second ) * CUDA::real( *second ) + CUDA::imag( *second ) * CUDA::imag( *second )) );
}
std::tuple<PC3::Type::real, PC3::Type::real> PC3::CUDA::minmax( PC3::Type::real* buffer, int size, bool device_pointer ) {
    #ifdef USE_CUDA
    if (device_pointer) {
        thrust::device_ptr<PC3::Type::real> dev_buffer = thrust::device_pointer_cast(buffer);
        auto mm = thrust::minmax_element( thrust::device, dev_buffer, dev_buffer + size, thrust::less<PC3::Type::real>() );
        PC3::Type::real min = *mm.first;
        PC3::Type::real max = *mm.second;
        return std::make_tuple( min, max );
    }
    const auto [first, second] = thrust::minmax_element( buffer, buffer + size, thrust::less<PC3::Type::real>() );
    #else
    const auto [first, second] = std::ranges::minmax_element( buffer, buffer + size, std::less<PC3::Type::real>() );
    #endif
    return std::make_tuple( *first, *second );
}