#include "cuda/cuda_matrix.cuh"
#include "kernel/kernel_runge_function.cuh"
#include "system/envelope.hpp"

/**
 * Device-Version of the calculateEnvelope function. Uses the Solver::PulseParameters
 * struct to calculate the envelope each iteration. We do this, because PC3 should
 * be able to support arbitrary numbers of pulses. This is not possible with the
 * current implementation of the device buffers, as we would need a buffer for each
 * pulse. This would be a waste of memory, as most pulses are not used throughout most
 * of the simulation. Additionally, we dont want to dynamically allocate the pulses on the device.
 */

template <typename T>
CUDA_HOST_DEVICE CUDA_INLINE bool cmp_active( T a, T b ) {
    return static_cast<unsigned int>( a ) & static_cast<unsigned int>( b );
}

CUDA_DEVICE complex_number PC3::Kernel::kernel_inline_calculate_pulse( const int row, const int col, PC3::Envelope::Polarization polarization, real_number t, const System::Parameters& p, Solver::PulseParameters::Pointers& pulse ) {
    complex_number ret = { 0.0, 0.0 };
    for ( int c = 0; c < pulse.n; c++ ) {
        // Calculate X,Y in the grid space
        auto x = -p.xmax + p.dx * col;
        auto y = -p.xmax + p.dx * row;
        // If type contains "local", use local coordinates instead
        if ( cmp_active( pulse.type[c], PC3::Envelope::Type::Local ) ) {
            x = -1.0 + 2.0 * col / p.N;
            y = -1.0 + 2.0 * row / p.N;
        }
        // Check if the polarization matches or if the input polarization is both. If not, the envelope is skipped.
        if ( pulse.pol[c] != PC3::Envelope::Polarization::Both and pulse.pol[c] != polarization and polarization != PC3::Envelope::Polarization::Both )
            continue;
        // Calculate ethe "r^2" distance to the center of the pump.
        const real_number r_squared = CUDA::abs2( x - pulse.x[c] ) + CUDA::abs2( y - pulse.y[c] );
        // Calculate Content of Exponential function
        const auto exp_factor = 0.5 * r_squared / pulse.width[c] / pulse.width[c];
        // Calculate the exponential function
        auto exp_function = CUDA::exp( -CUDA::pow( exp_factor, pulse.exponent[c] ) );
        // If the type is a gaussian outer, we calculate CUDA::exp(...)^N instead of CUDA::exp((...)^N)
        if ( cmp_active( pulse.type[c], PC3::Envelope::Type::OuterExponent ) )
            exp_function = CUDA::pow( CUDA::exp( -exp_factor ), pulse.exponent[c] );
        // If the shape is a ring, we multiply the exp function with the exp_factor again.
        auto pre_fractor = 1.0;
        if ( cmp_active( pulse.type[c], PC3::Envelope::Type::Ring ) )
            pre_fractor = exp_factor;
        // Default amplitude is A/sqrt(2pi)/w
        complex_number amplitude = { pulse.amp[c], 0.0 };
        if ( not( cmp_active( pulse.type[c], PC3::Envelope::Type::NoDivide ) ) )
            amplitude = amplitude / pulse.width[c] * sqrt( 2 * 3.1415 );
        // If the behaviour is adaptive, the amplitude is set to the current value of the buffer instead.
        if ( cmp_active( pulse.behavior[c], PC3::Envelope::Behavior::Adaptive ) )
            amplitude = complex_number( pulse.amp[c] * CUDA::real(ret), 0.0 );
        if ( cmp_active( pulse.behavior[c], PC3::Envelope::Behavior::Complex ) )
            amplitude = complex_number( 0.0, CUDA::real( amplitude ) );
        complex_number contribution = amplitude * pre_fractor * exp_function;
        // Add Charge
        complex_number combined = contribution * CUDA::pow( ( ( x - pulse.x[c] ) / p.xmax + 1.0 * CUDA::sign( pulse.m[c] ) * p.i * ( y - pulse.y[c] ) / p.xmax ), CUDA::abs( pulse.m[c] ) );
        const auto t0 = pulse.t0[c];
        complex_number temp_shape = p.one_over_h_bar_s * CUDA::exp( -( t - t0 ) * ( t - t0 ) / pulse.sigma[c] / pulse.sigma[c] - p.i * pulse.freq[c] * ( t - t0 ) );
        if ( not( cmp_active( pulse.type[c], PC3::Envelope::Type::NoDivide ) ) )
            temp_shape = temp_shape / pulse.sigma[c] * sqrt( 2 * 3.1415 );
        // Combine Spacial shape with temporal shape
        combined = combined * temp_shape;

        // Add, multiply or replace the contribution to the buffer.
        if ( cmp_active( pulse.behavior[c], PC3::Envelope::Behavior::Add ) )
            ret += combined;
        else if ( pulse.behavior[c] == PC3::Envelope::Behavior::Multiply )
            ret = ret * combined;
        else if ( pulse.behavior[c] == PC3::Envelope::Behavior::Replace )
            ret = combined;
    }
    // Return complete pulse
    return ret;
}