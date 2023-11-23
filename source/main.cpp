/*
 * MIT License
 * Copyright (c) 2023 David Bauch
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS," WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

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
    auto solver = PC3::Solver( system, PC3::Solver::Symmetry::Scalar );

    // Create Main Plotwindow. Needs to be compiled with -DSFML_RENDER
    initSFMLWindow( solver );
 
    bool running = true;
    double duration = 0.;
    // Main Loop
    while ( system.t < system.t_max and running ) {
        TimeThis(
            // The CPU should briefly evaluate wether the pulses have to be evaluated
            bool evaluate_pulse = system.evaluatePulse();
            // Iterate #output_every times
            for ( int i = 0; i < system.filehandler.output_every; i++ ) {
                solver.iterateRungeKutta( evaluate_pulse );
            }
            , "Main" );

        TimeThis(
            // Sync the current device arrays to their host array equivalents
            solver.syncDeviceArrays();
            // Cache the history and max values
            solver.cacheValues();
            // Plot
            running = plotSFMLWindow( solver, 1. / (duration)*system.dt * system.filehandler.output_every );
            , "Plotting" );
        duration = PC3::TimeIt::get( "Main" ) + PC3::TimeIt::get( "Plotting" );
        auto [min, max] = PC3::CUDA::minmax( solver.device.wavefunction_plus.get(), system.s_N * system.s_N, true /*This is a device pointer*/ );
        std::cout << "T = " << int( system.t ) << ", Time per " << system.filehandler.output_every << " iterations: " << duration << "s -> " << 1. / (duration)*system.dt * system.filehandler.output_every << "ps/s, current dt = " << system.dt << "                \r" << std::flush;
    }

    // Fileoutput
    solver.finalize();

    // Print Time statistics and output to file
    system.printSummary( PC3::TimeIt::getTimes(), PC3::TimeIt::getTimesTotal() );
    PC3::TimeIt::toFile( system.filehandler.getFile( "times" ) );

    return 0;
}