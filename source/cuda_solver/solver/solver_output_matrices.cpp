#include <vector>
#include <string>
#include "solver/gpu_solver.cuh"
#include "cuda/cuda_macro.cuh"

void PC3::Solver::outputMatrices() {
    std::cout << "---------------------------- Outputting Final Matrices ----------------------------" << std::endl;
    std::vector<std::string> fileoutputkeys = { "wavefunction_plus", "wavefunction_minus", "reservoir_plus", "reservoir_minus", "fft_plus", "fft_minus" };
#pragma omp parallel for
    for ( int i = 0; i < fileoutputkeys.size(); i++ ) {
        auto key = fileoutputkeys[i];
        if ( key == "wavefunction_plus" and system.doOutput( "mat", "wavefunction_plus", "plus", "wf" ) )
            filehandler.outputMatrixToFile( host.wavefunction_plus.get(), system.s_N, system.xmax, system.dx, key );
        if ( key == "reservoir_plus" and system.doOutput( "mat", "reservoir_plus", "plus", "n" ) )
            filehandler.outputMatrixToFile( host.reservoir_plus.get(), system.s_N, system.xmax, system.dx, key );
        if ( system.fft_every < system.t_max and key == "fft_plus" and system.doOutput( "mat", "fft_plus", "plus", "fft" ) )
            filehandler.outputMatrixToFile( host.fft_plus.get(), system.s_N, -1.0, 2.0/system.s_N, key );
        // Guard when not useing TE/TM splitting
        if (not use_te_tm_splitting)
            continue;
        if ( key == "wavefunction_minus" and system.doOutput( "mat", "wavefunction_minus", "minus", "wf" ) )
            filehandler.outputMatrixToFile( host.wavefunction_minus.get(), system.s_N, system.xmax, system.dx, key );
        if ( key == "reservoir_minus" and system.doOutput( "mat", "reservoir_minus", "minus", "n" ) )
            filehandler.outputMatrixToFile( host.reservoir_minus.get(), system.s_N, system.xmax, system.dx, key );
        if ( system.fft_every < system.t_max and key == "fft_minus" and system.doOutput( "mat", "fft_minus", "minus", "fft" ) )
            filehandler.outputMatrixToFile( host.fft_minus.get(), system.s_N, -1.0, 2.0/system.s_N, key );
    }
}