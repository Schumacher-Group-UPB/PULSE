#include <vector>
#include <string>
#include "misc/helperfunctions.hpp"
#include "solver/gpu_solver.cuh"
#include "cuda/cuda_macro.cuh"

void PC3::Solver::cacheValues() {
    // Min and Max
    const auto [min_plus, max_plus] = CUDA::minmax( device.wavefunction_plus.get(), system.s_N * system.s_N, true /*Device Pointer*/ );
    host.wavefunction_max_plus.emplace_back( max_plus );
    // Cut at Y = 0
    auto cut_p = device.wavefunction_plus.slice( system.s_N * system.s_N / 2, system.s_N );
    host.wavefunction_plus_history.emplace_back( cut_p );
    
    // TE/TM Guard
    if (not use_te_tm_splitting)
        return;

    // Same for _minus component if use_te_tm_splitting is true
    const auto [min_minus, max_minus] = CUDA::minmax( device.wavefunction_minus.get(), system.s_N * system.s_N, true /*Device Pointer*/ );
    host.wavefunction_max_minus.emplace_back( max_minus );
    auto cut_m = device.wavefunction_minus.slice( system.s_N * system.s_N / 2, system.s_N );
    host.wavefunction_minus_history.emplace_back( cut_m );
}

void PC3::Solver::cacheToFiles() {
    if ( system.doOutput( "max", "scalar" ) ) {
        auto& file_max = filehandler.getFile( "max" );
        file_max << "t Psi_Plus";
        if ( use_te_tm_splitting )
            file_max << " Psi_Minus";
        file_max << "\n";
        for ( int i = 0; i < host.wavefunction_max_plus.size(); i++ ) {
            if ( use_te_tm_splitting )
                file_max << i << " " << host.wavefunction_max_plus[i] << " " << host.wavefunction_max_minus[i] << "\n";
            else
                file_max << i << " " << host.wavefunction_max_plus[i] << "\n";
        }
        file_max.close();
    }

    // Guard when not outputting history
    if ( not system.doOutput( "mat", "history" ) )
        return;
        
    auto& file_history_plus = filehandler.getFile( "history_plus" );
    const auto interval_time = int( std::max( 1., host.wavefunction_plus_history.size() / 200. ) );
    const auto interval_x = int( std::max( 1., host.wavefunction_plus_history.front().size() / 200. ) );
    for ( int i = 0; i < host.wavefunction_plus_history.size(); i += interval_time ) {
        std::cout << "Writing history " << i << " of " << host.wavefunction_max_plus.size() << "\r";
        for ( int k = 0; k < host.wavefunction_plus_history.front().size(); k += interval_x ) {
            const auto current_plus = host.wavefunction_plus_history[i][k];
            file_history_plus << i << " " << k << " " << CUDA::real( current_plus ) << " " << CUDA::imag( current_plus ) << "\n";
        }
        file_history_plus << "\n";
        
    }
    file_history_plus.close();

    // TE/TM Guard
    if (not use_te_tm_splitting)
        return;

    auto& file_history_minus = filehandler.getFile( "history_minus" );
    for ( int i = 0; i < host.wavefunction_minus_history.size(); i += interval_time ) {
        std::cout << "Writing history " << i << " of " << host.wavefunction_max_minus.size() << "\r";
        for ( int k = 0; k < host.wavefunction_minus_history.front().size(); k += interval_x ) {
            const auto current_plus = host.wavefunction_minus_history[i][k];
            file_history_minus << i << " " << k << " " << CUDA::real( current_plus ) << " " << CUDA::imag( current_plus ) << "\n";
        }
        file_history_minus << "\n";
        
    }
    file_history_minus.close();
}