#include "solver/gpu_solver.hpp"
#include "misc/escape_sequences.hpp"

void PC3::Solver::finalize() {
    // Output Matrices
    outputMatrices( 0 /*start*/, system.p.N_x /*end*/, 0 /*start*/, system.p.N_y /*end*/, 1.0 /*increment*/);
    // Cache to files
    std::cout << "Caching to Files... " << std::endl;
    cacheToFiles();
}