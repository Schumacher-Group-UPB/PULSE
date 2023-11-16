#include <cmath>
#include <iostream>
#include <fstream>
#include <complex>
#include <vector>
#include <cstdlib>
#include <omp.h>
#include <chrono>

#include "cuda/cuda_complex.cuh"
#include "system/system.hpp"
#include "system/filehandler.hpp"
#include "misc/helperfunctions.hpp"
#include "misc/timeit.hpp"
#include "misc/sfml_helper.hpp"
#include "solver/gpu_solver.cuh"

int main( int argc, char* argv[] ) {
    // Convert input arguments to system and handler variables
    auto system = PC3::System(argc, argv);

    // Create Solver Class
    auto cuda_solver = PC3::Solver( system, PC3::Solver::Symmetry::TETM );

    // Create Main Plotwindow. Needs to be compiled with -DSFML_RENDER
    initSFMLWindow( cuda_solver );
 
    bool running = true;
    // Main Loop
    while ( system.t < system.t_max and running ) {
        TimeThis(
            // The CPU should briefly evaluate wether the pulses have to be evaluated
            bool evaluate_pulse = system.evaluatePulse();
            // Iterate #out_modulo times
            for ( int i = 0; i < cuda_solver.system.filehandler.out_modulo; i++ ) {
                cuda_solver.iterateRungeKutta( evaluate_pulse );
            }
            , "Main" );

        TimeThis(
            // Sync the current device arrays to their host array equivalents
            cuda_solver.syncDeviceArrays();
            // Cache the history and max values
            cuda_solver.cacheValues();
            // Plot
            running = plotSFMLWindow( cuda_solver );
            , "Plotting" );
        double duration = PC3::TimeIt::get( "Main" ) + PC3::TimeIt::get( "Plotting" );
        auto [min, max] = minmax( cuda_solver.device.wavefunction_plus.get(), system.s_N * system.s_N, true /*This is a device pointer*/ );
        std::cout << "T = " << int( system.t ) << ", Time per " << system.filehandler.out_modulo << " iterations: " << duration << "s -> " << 1. / (duration)*system.dt * system.filehandler.out_modulo << "ps/s, current dt = " << system.dt << "                \r";
    }

    // Fileoutput
    cuda_solver.finalize();

    // Print Time statistics and output to file
    system.printSummary( PC3::TimeIt::getTimes(), PC3::TimeIt::getTimesTotal() );
    PC3::TimeIt::toFile( system.filehandler.getFile( "times" ) );

    return 0;
}