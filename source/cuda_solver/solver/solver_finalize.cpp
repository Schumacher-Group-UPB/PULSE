#include "solver/gpu_solver.cuh"
#include "misc/escape_sequences.hpp"

void PC3::Solver::finalize() {
    std::cout << EscapeSequence::CLEAR_LINE << "--------------------------------- Finalizing Solver -------------------------------" << std::endl;
    // Sync all device arrays
    syncDeviceArrays();
    // Output Matrices
    outputMatrices( 0 /*start*/, system.s_N /*end*/, 1.0 /*increment*/);
    // Cache to files
    std::cout << "Caching to Files... " << std::endl;
    cacheToFiles();
    // Calculate Soll Values
    std::cout << "Calculating Soll Values... " << std::endl;
    calculateSollValues();
}