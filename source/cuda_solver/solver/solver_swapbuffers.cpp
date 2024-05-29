#include "solver/gpu_solver.hpp"

void PC3::Solver::swapBuffers() {
    matrix.wavefunction_plus.swap( matrix.buffer_wavefunction_plus );
    matrix.reservoir_plus.swap( matrix.buffer_reservoir_plus );
    if ( system.p.use_twin_mode ) {
        matrix.wavefunction_minus.swap( matrix.buffer_wavefunction_minus );
        matrix.reservoir_minus.swap( matrix.buffer_reservoir_minus );
    }
}