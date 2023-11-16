#include <vector>
#include <string>
#include "solver/gpu_solver.cuh"
#include "cuda/cuda_macro.cuh"

// TODO: Better specify what to load. Is in Features.
void PC3::Solver::loadMatrices() {
    if ( filehandler.loadPath.size() < 1 )
        return;
    std::cout << "Loading Matrices from " << filehandler.loadPath << std::endl;
    if ( filehandler.loadPath.back() != '/' )
        filehandler.loadPath += "/";
    std::vector<std::string> fileoutputkeys = { "wavefunction_plus", "wavefunction_minus", "reservoir_plus", "reservoir_minus" };
#pragma omp parallel for
    for ( auto i = 0; i < fileoutputkeys.size(); i++ ) {
        if ( fileoutputkeys[i] == "wavefunction_plus" )
            filehandler.loadMatrixFromFile( filehandler.loadPath + fileoutputkeys[i] + ".txt", host.wavefunction_plus.get() );
        else if ( fileoutputkeys[i] == "reservoir_plus" )
            filehandler.loadMatrixFromFile( filehandler.loadPath + fileoutputkeys[i] + ".txt", host.reservoir_plus.get() );
        // Guard when not useing TE/TM splitting
        if (not use_te_tm_splitting)
            continue;
        else if ( fileoutputkeys[i] == "wavefunction_minus" )
            filehandler.loadMatrixFromFile( filehandler.loadPath + fileoutputkeys[i] + ".txt", host.wavefunction_minus.get() );
        else if ( fileoutputkeys[i] == "reservoir_minus" )
            filehandler.loadMatrixFromFile( filehandler.loadPath + fileoutputkeys[i] + ".txt", host.reservoir_minus.get() );
    }
}