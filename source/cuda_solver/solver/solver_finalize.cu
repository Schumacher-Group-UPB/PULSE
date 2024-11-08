#include "solver/gpu_solver.hpp"
#include "misc/escape_sequences.hpp"
#include "misc/commandline_io.hpp"

void PHOENIX::Solver::finalize() {
    // Output Matrices
    outputMatrices( 0 /*start*/, system.p.N_c /*end*/, 0 /*start*/, system.p.N_r /*end*/, 1.0 /*increment*/ );
    // Cache to files
    std::cout << PHOENIX::CLIO::prettyPrint( "Caching to Files... ", PHOENIX::CLIO::Control::Info ) << std::endl;
    cacheToFiles();
}