#include "solver/gpu_solver.cuh"

void PC3::Solver::finalize() {
    std::cout << "Finalizing Solver" << std::endl;
    // Sync all device arrays
    syncDeviceArrays();
    // Output Matrices
    std::cout << "Outputting Matrices" << std::endl;
    outputMatrices();
    // Cache to files
    std::cout << "Caching to Files" << std::endl;
    cacheToFiles();
    // Calculate Soll Values
    std::cout << "Calculating Soll Values" << std::endl;
    calculateSollValues();
}