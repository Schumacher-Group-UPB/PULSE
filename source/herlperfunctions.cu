#include "helperfunctions.hpp"
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>
#include <thrust/pair.h>
#include <thrust/device_ptr.h>
#include "cuda_complex.cuh"

std::tuple<real_number, real_number> minmax( complex_number* buffer, int size, bool device_pointer ) {
    if ( device_pointer ) {
        thrust::device_ptr<complex_number> dev_buffer = thrust::device_pointer_cast( buffer );
        auto mm = thrust::minmax_element( thrust::device, dev_buffer, dev_buffer + size, compare_complex_abs2() );
        complex_number min = *mm.first;
        complex_number max = *mm.second;
        return std::make_tuple( min.x * min.x + min.y * min.y, max.x * max.x + max.y * max.y );
    }
    auto mm = thrust::minmax_element( buffer, buffer + size, compare_complex_abs2() );
    return std::make_tuple( ( *mm.first ).x * ( *mm.first ).x + ( *mm.first ).y * ( *mm.first ).y, ( *mm.second ).x * ( *mm.second ).x + ( *mm.second ).y * ( *mm.second ).y );

    // Old CPU side version. can also be implemented using thrust::host
    //     real_number max = 0;
    //     real_number min = 0;
    //     auto n_cpus = omp_get_max_threads();
    //     std::vector<real_number> maxs( n_cpus );
    //     std::vector<real_number> mins( n_cpus );

    // #pragma omp parallel for
    //     for ( int i = 0; i < size; i++ ) {
    //         int cpu = omp_get_thread_num();
    //         maxs[cpu] = std::max( maxs[cpu], cwiseAbs2( buffer[i] ) );
    //         mins[cpu] = std::min( mins[cpu], cwiseAbs2( buffer[i] ) );
    //     }

    //     for ( int i = 0; i < n_cpus; i++ ) {
    //         max = std::max( max, maxs[i] );
    //         min = std::min( min, mins[i] );
    //     }
    //     return std::make_tuple( std::sqrt( min ), std::sqrt( max ) );
}
std::tuple<real_number, real_number> minmax( real_number* buffer, int size, bool device_pointer ) {
    if (device_pointer) {
        thrust::device_ptr<real_number> dev_buffer = thrust::device_pointer_cast(buffer);
        auto mm = thrust::minmax_element( thrust::device, dev_buffer, dev_buffer + size, thrust::less<real_number>() );
        real_number min = *mm.first;
        real_number max = *mm.second;
        return std::make_tuple( min, max );
    }
    auto mm = thrust::minmax_element( buffer, buffer + size, thrust::less<real_number>() );
    return std::make_tuple( *mm.first, *mm.second );
}