#include <vector>
#include <string>

#include "cuda/typedef.cuh"
#include "misc/helperfunctions.hpp"
#include "solver/gpu_solver.hpp"

void PC3::Solver::cacheValues() {
    // System Time
    cache_map_scalar["t"].emplace_back( system.p.t );
    //matrix.times.emplace_back( system.p.t );

    // Min and Max
    const auto [min_plus, max_plus] = CUDA::minmax( matrix.wavefunction_plus.getDevicePtr(), system.p.N_x * system.p.N_y, true /*Device Pointer*/ );
    cache_map_scalar["min_plus"].emplace_back( min_plus );
    cache_map_scalar["max_plus"].emplace_back( max_plus );
    //matrix.wavefunction_max_plus.emplace_back( max_plus );
    // DEPRECATED: Cuts. Use history instead.
    // Cut at Y = 0
    //auto cut_p = matrix.wavefunction_plus.sliceDevice( system.p.N_x * system.p.N_y / 2, system.p.N_x );
    //matrix.wavefunction_plus_history.emplace_back( cut_p );

    // Output Pulse, Pump and Potential Envelope functions to cache_map_scalar
    for (int g = 0; g < system.pulse.groupSize(); g++) {
        if ( system.pulse.temporal[g] & PC3::Envelope::Temporal::Constant )
            continue;
        Type::complex pulse = system.pulse.temporal_envelope[g];
        cache_map_scalar["pulse_"+std::to_string(g)+"_real"].push_back( PC3::CUDA::real(pulse) );
        cache_map_scalar["pulse_"+std::to_string(g)+"_imag"].push_back( PC3::CUDA::imag(pulse) );
    }
    for (int g = 0; g < system.pump.groupSize(); g++) {
        if ( system.pump.temporal[g] & PC3::Envelope::Temporal::Constant )
            continue;
        Type::real pump = PC3::CUDA::real(system.pump.temporal_envelope[g]);
        cache_map_scalar["pump_"+std::to_string(g)].push_back( PC3::CUDA::real(pump) );
    }
    for (int g = 0; g < system.potential.groupSize(); g++) {
        if ( system.potential.temporal[g] & PC3::Envelope::Temporal::Constant )
            continue;
        Type::real potential = PC3::CUDA::real(system.potential.temporal_envelope[g]);
        cache_map_scalar["potential_"+std::to_string(g)].push_back( PC3::CUDA::real(potential) );
    }

    // TE/TM Guard
    if ( not system.p.use_twin_mode )
        return;

    // Same for _minus component if use_twin_mode is true
    const auto [min_minus, max_minus] = CUDA::minmax( matrix.wavefunction_minus.getDevicePtr(), system.p.N_x * system.p.N_y, true /*Device Pointer*/ );
    cache_map_scalar["min_minus"].emplace_back( min_minus );
    cache_map_scalar["max_minus"].emplace_back( max_minus );
    //matrix.wavefunction_max_minus.emplace_back( max_minus );
    //auto cut_m = matrix.wavefunction_minus.sliceDevice( system.p.N_x * system.p.N_y / 2, system.p.N_x );
    //matrix.wavefunction_minus_history.emplace_back( cut_m );
}

void PC3::Solver::cacheToFiles() {
    if ( not system.doOutput( "all", "max", "scalar" ) )
        return;

    auto& file_max = filehandler.getFile( "scalar" );
    file_max << "index ";
    for ( const auto& [key, _] : cache_map_scalar )
        file_max << key << " ";
    file_max << "\n";
    for ( int i = 0; i < cache_map_scalar["t"].size(); i++ ) {
        file_max << i << " ";
        for ( const auto& [_, vec] : cache_map_scalar )
            file_max << vec[i] << " ";
        file_max << "\n";
    }
    file_max.close();

    // Guard when not outputting history
    // DEPRECATED: Cuts. Use history instead.
    //if ( not system.doOutput( "all", "mat", "history" ) )
    //    return;
    //auto& file_history_plus = filehandler.getFile( "history_plus" );
    //const auto interval_time = CUDA::max<unsigned int>( 1u, matrix.wavefunction_plus_history.size() / system.history_output_n );
    //const auto interval_x = CUDA::max<unsigned int>( 1u, matrix.wavefunction_plus_history.front().size() / system.history_output_n );
    //for ( unsigned int i = 0; i < matrix.wavefunction_plus_history.size(); i += interval_time ) {
    //    std::cout << "Writing history " << i << " of " << matrix.wavefunction_max_plus.size() << "\r";
    //    for ( int k = 0; k < matrix.wavefunction_plus_history.front().size(); k += interval_x ) {
    //        const auto current_plus = matrix.wavefunction_plus_history[i][k];
    //        file_history_plus << i << " " << k << " " << CUDA::real( current_plus ) << " " << CUDA::imag( current_plus ) << "\n";
    //    }
    //    file_history_plus << "\n";
    //}
    //file_history_plus.close();

    // TE/TM Guard
    //if ( not system.p.use_twin_mode )
    //    return;
    //auto& file_history_minus = filehandler.getFile( "history_minus" );
    //for ( unsigned int i = 0; i < matrix.wavefunction_minus_history.size(); i += interval_time ) {
    //    std::cout << "Writing history " << i << " of " << matrix.wavefunction_max_minus.size() << "\r";
    //    for ( int k = 0; k < matrix.wavefunction_minus_history.front().size(); k += interval_x ) {
    //        const auto current_plus = matrix.wavefunction_minus_history[i][k];
    //        file_history_minus << i << " " << k << " " << CUDA::real( current_plus ) << " " << CUDA::imag( current_plus ) << "\n";
    //    }
    //    file_history_minus << "\n";
    //}
    //file_history_minus.close();
}

// TODO: Support Multiple History Outputs, and also support piping them into a single file.
// something like "append" mode, that doesnt open a new file but instead appends to the existing one.
size_t _local_file_out_counter = 0;
void PC3::Solver::cacheMatrices() {
    if (not system.do_output_history_matrix)
        return;
    std::string suffix = "_"+std::to_string(_local_file_out_counter);
    _local_file_out_counter++;
    outputMatrices( system.history_matrix_start_x, system.history_matrix_end_x, system.history_matrix_start_y, system.history_matrix_end_y, system.history_matrix_output_increment, suffix, "timeoutput/" );
}

