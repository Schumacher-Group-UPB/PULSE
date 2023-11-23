#include <memory>
#include <algorithm>
#include <random>
#include <ranges>
#include <vector>
#include "misc/helperfunctions.hpp"
#include "solver/gpu_solver.cuh"
#include "cuda/cuda_macro.cuh"


void PC3::Solver::calculateSollValues() {
    if ( system.mask.x.size() == 0 ) {
        std::cout << "No mask provided, skipping soll value calculation!" << std::endl;
        return;
    }

    // Calculate min and max to normalize both Psi and the mask. We dont really care about efficiency here, since its
    // only done once.
    real_number max_mask_plus = 1.;
    real_number max_psi_plus = 1.;
    real_number max_mask_minus = 1.;
    real_number max_psi_minus = 1.;
    real_number min_mask_plus = 0.;
    real_number min_psi_plus = 0.;
    real_number min_mask_minus = 0.;
    real_number min_psi_minus = 0.;

    if ( system.normalize_before_masking ) {
        std::tie( min_mask_plus, max_mask_plus ) = CUDA::minmax( host.soll_plus.get(), system.s_N * system.s_N, false /*Device Pointer*/ );
        std::tie( min_psi_plus, max_psi_plus ) = CUDA::minmax( host.wavefunction_plus.get(), system.s_N * system.s_N, false /*Device Pointer*/ );
        std::cout << "The mask calculation will use normalizing constants:" << std::endl;
        std::cout << "min_mask_plus = " << min_mask_plus << " max_mask_plus = " << max_mask_plus << std::endl;
        std::cout << "min_psi_plus = " << min_psi_plus << " max_psi_plus = " << max_psi_plus << std::endl;

        if (use_te_tm_splitting) {
            std::tie( min_mask_minus, max_mask_minus ) = CUDA::minmax( host.soll_minus.get(), system.s_N * system.s_N, false /*Device Pointer*/ );
            std::tie( min_psi_minus, max_psi_minus ) = CUDA::minmax( host.wavefunction_minus.get(), system.s_N * system.s_N, false /*Device Pointer*/ );
            std::cout << "min_mask_minus = " << min_mask_minus << " max_mask_minus = " << max_mask_minus << std::endl;
            std::cout << "min_psi_minus = " << min_psi_minus << " max_psi_minus = " << max_psi_minus << std::endl;
        }
        // Devide error by N^2
        max_mask_plus *= system.s_N * system.s_N;
        max_psi_plus *= system.s_N * system.s_N;
        max_mask_minus *= system.s_N * system.s_N;
        max_psi_minus *= system.s_N * system.s_N;
    }

    // Output Mask
    if ( system.doOutput( "mat", "mask_plus", "mask" ) )
        system.filehandler.outputMatrixToFile( host.soll_plus.get(), system.s_N, system.xmax, system.dx, "mask_plus" );
    if ( use_te_tm_splitting and system.doOutput( "mat", "mask_minus", "mask" ) )
        system.filehandler.outputMatrixToFile( host.soll_minus.get(), system.s_N, system.xmax, system.dx, "mask_minus" );
    
    // Calculate sum of all elements and check matching
#pragma omp parallel for
    for ( int i = 0; i < system.s_N * system.s_N; i++ ) {
        host.soll_plus[i] = CUDA::abs( CUDA::abs( host.wavefunction_plus[i] ) / max_psi_plus - host.soll_plus[i] / max_mask_plus );
    }
    real_number sum_plus = 0;
    std::ranges::for_each( host.soll_plus.get(), host.soll_plus.get() + system.s_N * system.s_N, [&sum_plus]( real_number n ) { sum_plus += n; } );
    std::cout << "Error in Psi_Plus: " << sum_plus << std::endl;

    if (use_te_tm_splitting) {
    #pragma omp parallel for
        for ( int i = 0; i < system.s_N * system.s_N; i++ ) {
            host.soll_minus[i] = CUDA::abs( CUDA::abs( host.wavefunction_minus[i] ) / max_psi_minus - host.soll_minus[i] / max_mask_minus );
        }
        real_number sum_minus = 0;
        std::ranges::for_each( host.soll_minus.get(), host.soll_minus.get() + system.s_N * system.s_N, [&sum_minus]( real_number n ) { sum_minus += n; } );
        std::cout << "Error in Psi_Minus: " << sum_minus << std::endl;
    }
}