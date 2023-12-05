#include <vector>
#include <string>
#include "solver/gpu_solver.cuh"
#include "cuda/cuda_macro.cuh"

void PC3::Solver::outputMatrices( const unsigned int start, const unsigned int end, real_number increment, const std::string& suffix, const std::string& prefix ) {
    std::vector<std::string> fileoutputkeys = { "wavefunction_plus", "wavefunction_minus", "reservoir_plus", "reservoir_minus", "fft_plus", "fft_minus" };
#pragma omp parallel for
    for ( int i = 0; i < fileoutputkeys.size(); i++ ) {
        auto key = fileoutputkeys[i];
        if ( key == "wavefunction_plus" and system.doOutput( "wavefunction", "psi", "wavefunction_plus", "psi_plus", "plus", "wf", "mat", "all", "initial", "initial_plus" ) )
            filehandler.outputMatrixToFile( host.wavefunction_plus.get(), start, end, start, end, system.s_N, increment, system.xmax, system.dx, prefix+key+suffix );
        if ( key == "reservoir_plus" and system.doOutput( "mat", "reservoir", "n", "reservoir_plus", "n_plus", "plus", "rv", "mat", "all" ) )
            filehandler.outputMatrixToFile( host.reservoir_plus.get(), start, end, start, end, system.s_N, increment, system.xmax, system.dx, prefix+key+suffix );
        if ( system.fft_every < system.t_max and key == "fft_plus" and system.doOutput( "fft_mask", "fft", "fft_plus", "plus", "mat", "all" ) )
            filehandler.outputMatrixToFile( host.fft_plus.get(), start, end, start, end, system.s_N, increment, -1.0, 2.0 / system.s_N, prefix+key+suffix );
        // Guard when not useing TE/TM splitting
        if ( not system.use_te_tm_splitting )
            continue;
        if ( key == "wavefunction_minus" and system.doOutput( "wavefunction", "psi", "wavefunction_minus", "psi_minus", "plus", "wf", "mat", "all", "initial", "initial_minus" ) )
            filehandler.outputMatrixToFile( host.wavefunction_minus.get(), start, end, start, end, system.s_N, increment, system.xmax, system.dx, prefix+key+suffix );
        if ( key == "reservoir_minus" and system.doOutput( "reservoir", "n", "reservoir_minus", "n_minus", "plus", "rv", "mat", "all" ) )
            filehandler.outputMatrixToFile( host.reservoir_minus.get(), start, end, start, end, system.s_N, increment, system.xmax, system.dx, prefix+key+suffix );
        if ( system.fft_every < system.t_max and key == "fft_minus" and system.doOutput( "fft_mask", "fft", "fft_minus", "plus", "mat", "all" ) )
            filehandler.outputMatrixToFile( host.fft_minus.get(), start, end, start, end, system.s_N, increment, -1.0, 2.0 / system.s_N, prefix+key+suffix );
    }
}