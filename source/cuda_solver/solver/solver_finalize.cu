#include "solver/gpu_solver.hpp"
#include "misc/escape_sequences.hpp"
#include "misc/commandline_io.hpp"

void PC3::Solver::finalize() {
    // Output Matrices
    outputMatrices( 0 /*start*/, system.p.N_c /*end*/, 0 /*start*/, system.p.N_r /*end*/, 1.0 /*increment*/ );
    // Cache to files
    std::cout << PC3::CLIO::prettyPrint( "Caching to Files... ", PC3::CLIO::Control::Info ) << std::endl;
    cacheToFiles();
}