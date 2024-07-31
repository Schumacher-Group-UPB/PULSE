#pragma once
#include "cuda/typedef.cuh"
#include "kernel_coefficients_dormand_prince.cuh"
#include "solver/matrix_container.hpp"
#include "solver/gpu_solver.hpp"
#include "system/system_parameters.hpp"

namespace PC3::Kernel {

    // Helper Struct to pass input-target data pointers to the kernel
    struct InputOutput {
        Type::complex* PULSE_RESTRICT in_wf_plus;
        Type::complex* PULSE_RESTRICT in_wf_minus;
        Type::complex* PULSE_RESTRICT in_rv_plus;
        Type::complex* PULSE_RESTRICT in_rv_minus;
        Type::complex* PULSE_RESTRICT out_wf_plus;
        Type::complex* PULSE_RESTRICT out_wf_minus;
        Type::complex* PULSE_RESTRICT out_rv_plus;
        Type::complex* PULSE_RESTRICT out_rv_minus;
    };

    // Generalized Ki summation. If no specialized kernel is required (like for RK45) this one can be used instead.
    namespace RK {
        // Weights for up to 10 ks
        struct Weights { 
            int n;
            int start;
            Type::real weights[10];
            template <typename ...Args>
            Weights( Args... _weights ) : n(sizeof...(_weights)), start(0) {
                double _w[] = { _weights... };
                for (int i = 0; i < n; i++) {
                    // If the weight is smaller than -1E10, we skip this weight, until we find a valid one
                    if (_w[i] != 0.0 and start == 0 ) 
                        start = i;
                    // Always assign the weight, even if it is invalid
                    weights[i] = Type::real(_w[i]);
                }
            }
        };
        PULSE_GLOBAL void runge_sum_to_input_ki( int i, Type::complex dt, MatrixContainer::Pointers dev_ptrs, SystemParameters::KernelParameters p, InputOutput io );
        PULSE_GLOBAL void runge_sum_to_input_kw( int i, Type::complex dt, MatrixContainer::Pointers dev_ptrs, SystemParameters::KernelParameters p, InputOutput io, RK::Weights weights );
        PULSE_GLOBAL void runge_sum_to_error( int i, Type::complex dt, MatrixContainer::Pointers dev_ptrs, SystemParameters::KernelParameters p, RK::Weights weights );
    } // namespace RK

    namespace Compute {

        PULSE_GLOBAL void gp_tetm( int i, Type::real t, MatrixContainer::Pointers dev_ptrs, SystemParameters::KernelParameters p, Solver::TemporalEvelope::Pointers oscillation_pulse, Solver::TemporalEvelope::Pointers oscillation_pump, Solver::TemporalEvelope::Pointers oscillation_potential, InputOutput io );
        PULSE_GLOBAL void gp_scalar( int i, Type::real t, MatrixContainer::Pointers dev_ptrs, SystemParameters::KernelParameters p, Solver::TemporalEvelope::Pointers oscillation_pulse, Solver::TemporalEvelope::Pointers oscillation_pump, Solver::TemporalEvelope::Pointers oscillation_potential, InputOutput io );

        PULSE_GLOBAL void gp_scalar_linear_fourier( int i, Type::real t, Type::complex dtc, MatrixContainer::Pointers dev_ptrs, SystemParameters::KernelParameters p, Solver::TemporalEvelope::Pointers oscillation_pulse, Solver::TemporalEvelope::Pointers oscillation_pump, Solver::TemporalEvelope::Pointers oscillation_potential, InputOutput io );
        PULSE_GLOBAL void gp_scalar_nonlinear( int i, Type::real t, Type::complex dtc, MatrixContainer::Pointers dev_ptrs, SystemParameters::KernelParameters p, Solver::TemporalEvelope::Pointers oscillation_pulse, Solver::TemporalEvelope::Pointers oscillation_pump, Solver::TemporalEvelope::Pointers oscillation_potential, InputOutput io );
        PULSE_GLOBAL void gp_scalar_independent( int i, Type::real t, MatrixContainer::Pointers dev_ptrs, SystemParameters::KernelParameters p, Solver::TemporalEvelope::Pointers oscillation_pulse, Solver::TemporalEvelope::Pointers oscillation_pump, Solver::TemporalEvelope::Pointers oscillation_potential, InputOutput io );

    } // namespace Compute

    PULSE_GLOBAL void initialize_random_number_generator(int i, unsigned int seed, Type::cuda_random_state* state, const unsigned int N);
    PULSE_GLOBAL void generate_random_numbers(int i, Type::cuda_random_state* state, Type::complex* buffer, const unsigned int N, const Type::real real_amp, const Type::real imag_amp);

} // namespace PC3::Kernel