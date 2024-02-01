#include <vector>
#include <string>

#include "misc/helperfunctions.hpp"
#include "solver/gpu_solver.hpp"
#include "cuda/cuda_macro.cuh"

void PC3::Solver::cacheValues() {
    // System Time
    host.times.emplace_back( system.t );

    // Min and Max
    const auto [min_plus, max_plus] = CUDA::minmax( device.wavefunction_plus.get(), system.s_N_x * system.s_N_y, true /*Device Pointer*/ );
    host.wavefunction_max_plus.emplace_back( max_plus );
    // Cut at Y = 0
    auto cut_p = device.wavefunction_plus.slice( system.s_N_x * system.s_N_y / 2, system.s_N_x );
    host.wavefunction_plus_history.emplace_back( cut_p );

    // TE/TM Guard
    if ( not system.use_te_tm_splitting )
        return;

    // Same for _minus component if use_te_tm_splitting is true
    const auto [min_minus, max_minus] = CUDA::minmax( device.wavefunction_minus.get(), system.s_N_x * system.s_N_y, true /*Device Pointer*/ );
    host.wavefunction_max_minus.emplace_back( max_minus );
    auto cut_m = device.wavefunction_minus.slice( system.s_N_x * system.s_N_y / 2, system.s_N_x );
    host.wavefunction_minus_history.emplace_back( cut_m );
}

void PC3::Solver::cacheToFiles() {
    if ( system.doOutput( "max", "scalar" ) ) {
        auto& file_max = filehandler.getFile( "max" );
        file_max << "index time Psi_Plus";
        if ( system.use_te_tm_splitting )
            file_max << " Psi_Minus";
        file_max << "\n";
        for ( int i = 0; i < host.wavefunction_max_plus.size(); i++ ) {
            if ( system.use_te_tm_splitting )
                file_max << i << " " << host.times[i] << " " << host.wavefunction_max_plus[i] << " " << host.wavefunction_max_minus[i] << "\n";
            else
                file_max << i << " " << host.times[i] << " " << host.wavefunction_max_plus[i] << "\n";
        }
        file_max.close();
    }

    // Guard when not outputting history
    if ( not system.doOutput( "mat", "history" ) )
        return;

    auto& file_history_plus = filehandler.getFile( "history_plus" );
    const auto interval_time = std::max<unsigned int>( 1u, host.wavefunction_plus_history.size() / system.history_output_n );
    const auto interval_x = std::max<unsigned int>( 1u, host.wavefunction_plus_history.front().size() / system.history_output_n );
    for ( unsigned int i = 0; i < host.wavefunction_plus_history.size(); i += interval_time ) {
        std::cout << "Writing history " << i << " of " << host.wavefunction_max_plus.size() << "\r";
        for ( int k = 0; k < host.wavefunction_plus_history.front().size(); k += interval_x ) {
            const auto current_plus = host.wavefunction_plus_history[i][k];
            file_history_plus << i << " " << k << " " << CUDA::real( current_plus ) << " " << CUDA::imag( current_plus ) << "\n";
        }
        file_history_plus << "\n";
    }
    file_history_plus.close();

    // TE/TM Guard
    if ( not system.use_te_tm_splitting )
        return;

    auto& file_history_minus = filehandler.getFile( "history_minus" );
    for ( unsigned int i = 0; i < host.wavefunction_minus_history.size(); i += interval_time ) {
        std::cout << "Writing history " << i << " of " << host.wavefunction_max_minus.size() << "\r";
        for ( int k = 0; k < host.wavefunction_minus_history.front().size(); k += interval_x ) {
            const auto current_plus = host.wavefunction_minus_history[i][k];
            file_history_minus << i << " " << k << " " << CUDA::real( current_plus ) << " " << CUDA::imag( current_plus ) << "\n";
        }
        file_history_minus << "\n";
    }
    file_history_minus.close();
}

size_t _local_file_out_counter = 0;
void PC3::Solver::cacheMatrices() {
    if (not system.do_output_history_matrix)
        return;
    std::string suffix = "_"+std::to_string(_local_file_out_counter);
    _local_file_out_counter++;
    outputMatrices( system.history_matrix_start_x, system.history_matrix_end_x, system.history_matrix_start_y, system.history_matrix_end_y, system.history_matrix_output_increment, suffix, "timeoutput/" );
}