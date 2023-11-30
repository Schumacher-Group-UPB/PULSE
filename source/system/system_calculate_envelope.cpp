#include "cuda/cuda_matrix.cuh"
#include "system/system.hpp"
#include "system/envelope.hpp"
#include "omp.h"

/**
 * Hacky way to calculate the envelope as CUDA::real numbers.
 * This is only done at the beginning of the program and on the CPU.
 * Temporarily copying the results is probably fine.
 */
void PC3::System::calculateEnvelope( real_number* buffer, const PC3::Envelope& mask, PC3::Envelope::Polarization polarization, real_number default_value_if_no_mask ) {
    std::unique_ptr<complex_number[]> tmp_buffer = std::make_unique<complex_number[]>( s_N * s_N );
    calculateEnvelope( tmp_buffer.get(), mask, polarization, default_value_if_no_mask );
// Transfer tmp_buffer to buffer as complex numbers
#pragma omp parallel for
    for ( int i = 0; i < s_N * s_N; i++ ) {
        buffer[i] = CUDA::real( tmp_buffer[i] );
    }
}

void PC3::System::calculateEnvelope( complex_number* buffer, const PC3::Envelope& mask, PC3::Envelope::Polarization polarization, real_number default_value_if_no_mask ) {
#pragma omp parallel for
    for ( int col = 0; col < s_N; col++ ) {
        for ( int row = 0; row < s_N; row++ ) {
            int i = col * s_N + row;
            buffer[i] = complex_number(0.0,0.0);
            bool has_been_set = false;
            for ( int c = 0; c < mask.amp.size(); c++ ) {
                // Calculate X,Y in the grid space
                auto x = -xmax + dx * col;
                auto y = -xmax + dx * row;
                // If type contains "local", use local coordinates instead
                if ( mask.type[c] & PC3::Envelope::Type::Local ) {
                    x = -1.0 + 2.0 * col / (s_N - 1);
                    y = -1.0 + 2.0 * row / (s_N - 1);
                }
                // Check if the polarization matches or if the input polarization is both. If not, the envelope is skipped.
                if ( mask.pol[c] != PC3::Envelope::Polarization::Both and mask.pol[c] != polarization and polarization != PC3::Envelope::Polarization::Both )
                    continue;
                has_been_set = true;
                // Calculate ethe "r^2" distance to the center of the pump.
                const real_number r_squared = CUDA::abs2( x - mask.x[c] ) + CUDA::abs2( y - mask.y[c] );
                // Calculate Content of Exponential function
                const real_number exp_factor = 0.5 * r_squared / mask.width[c] / mask.width[c];
                // Calculate the exponential function
                real_number exp_function = CUDA::exp( -CUDA::pow( exp_factor, mask.exponent[c] ) );
                // If the type is a gaussian outer, we calculate CUDA::exp(...)^N instead of CUDA::exp((...)^N)
                if ( mask.type[c] & PC3::Envelope::Type::OuterExponent )
                    exp_function = CUDA::pow( CUDA::exp( -exp_factor ), mask.exponent[c] );
                // If the shape is a ring, we multiply the exp function with the exp_factor again.
                real_number pre_fractor = 1.0;
                if ( mask.type[c] & PC3::Envelope::Type::Ring )
                    pre_fractor = exp_factor;
                // Default amplitude is A/sqrt(2pi)/w
                complex_number amplitude = { mask.amp[c], 0.0 };
                if ( not( mask.type[c] & PC3::Envelope::Type::NoDivide ) )
                    amplitude = amplitude / mask.width[c] / sqrt( 2 * 3.1415 );
                // If the behaviour is adaptive, the amplitude is set to the current value of the buffer instead.
                if ( mask.behavior[c] & PC3::Envelope::Behavior::Adaptive )
                    amplitude = mask.amp[c] * buffer[i];
                if ( mask.behavior[c] & PC3::Envelope::Behavior::Complex )
                    amplitude = complex_number( 0.0, CUDA::real( amplitude ) );
                complex_number charge = CUDA::pow( complex_number( ( x - mask.x[c] ) / xmax, 1.0 * CUDA::sign( mask.m[c] ) * ( y - mask.y[c] ) / xmax ), CUDA::abs( mask.m[c] ) );
                complex_number contribution = amplitude * pre_fractor * exp_function * charge;
                // Add, multiply or replace the contribution to the buffer.
                if ( mask.behavior[c] & PC3::Envelope::Behavior::Add )
                    buffer[i] = buffer[i] + contribution;
                else if ( mask.behavior[c] == PC3::Envelope::Behavior::Multiply )
                    buffer[i] = buffer[i] * contribution;
                else if ( mask.behavior[c] == PC3::Envelope::Behavior::Replace )
                    buffer[i] = contribution;
            }
            // If no mask has been applied, set the value to the default value.
            // This ensures the mask is always initialized
            if ( not has_been_set )
                buffer[i] = { default_value_if_no_mask, 0 };
        }
    }
}