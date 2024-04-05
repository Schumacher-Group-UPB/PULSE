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
#include "solver/gpu_solver.hpp"

int main( int argc, char* argv[] ) {
    // Try and read-in any config file
    auto config = PC3::readConfigFromFile( argc, argv );

    // Convert input arguments to system and handler variables
    auto system = PC3::System( config.size(), config.data() );

    // Create Solver Class
    auto solver = PC3::Solver( system );

    // Create Main Plotwindow. Needs to be compiled with -DSFML_RENDER
    initSFMLWindow( solver );

    // Some Helper Variables
    bool running = true;
    double complete_duration = 0.;
    size_t complete_iterations = 0;

    // Main Loop
    while ( system.t < system.t_max and running ) {
        TimeThis(
            // Iterate #output_every ps
            auto start = system.t;
            while ( system.t <= start + system.output_every and solver.iterateRungeKutta() ) {
                complete_iterations++;
            }

            // Sync the current device arrays to their host array equivalents
            solver.syncDeviceArrays();
            // Cache the history and max values
            solver.cacheValues();
            // Output Matrices if enabled
            solver.cacheMatrices();
            // Plot
            running = plotSFMLWindow( solver, system.t, complete_duration, complete_iterations );
            , "Main-Loop" );
        complete_duration = PC3::TimeIt::totalRuntime();

        // Print Runtime
        std::cout << EscapeSequence::HIDE_CURSOR;
        std::cout << "-----------------------------------------------------------------------------------\n";
        std::cout << "    T = " << int( system.t ) << "ps - dt = " << std::setprecision( 2 ) << system.dt << "ps    \n";
        // Progressbar for system.t/system.t_max
        std::cout << "    Progress:  [";
        for ( int i = 0; i < 50. * system.t / system.t_max; i++ ) {
            std::cout << EscapeSequence::BLUE << "#" << EscapeSequence::RESET; // █
        }
        for ( int i = 0; i < 50. * ( 1. - system.t / system.t_max ); i++ ) {
            std::cout << EscapeSequence::GREY << "#" << EscapeSequence::RESET;
        }
        std::cout << "]  " << int( 100. * system.t / system.t_max ) << "%  \n";
        bool evaluate_pulse = system.evaluatePulse();
        bool evaluate_reservoir = system.evaluateReservoir();
        bool evaluate_stochastic = system.evaluateStochastic();
        std::cout << "    Current System: " << ( system.use_twin_mode ? "TE/TM" : "Scalar" ) << " - " << ( evaluate_reservoir ? "With Reservoir" : "No Reservoir" ) << " - " << ( evaluate_pulse ? "With Pulse" : "No Pulse" ) << " - " << ( evaluate_stochastic ? "With Stochastic" : "No Stochastic" ) << "    \n";
        std::cout << "    Runtime: " << int( complete_duration ) << "s, remaining: " << int( complete_duration * ( system.t_max - system.t ) / system.t ) << "s    \n";
        std::cout << "    Time per ps: " << complete_duration / system.t << "s/ps  -  " << std::setprecision( 3 ) << system.t / complete_duration << "ps/s  -  " << complete_iterations/complete_duration << "it/s    \n";
        std::cout << "-----------------------------------------------------------------------------------" << std::endl;
        std::cout << EscapeSequence::LINE_UP << EscapeSequence::LINE_UP << EscapeSequence::LINE_UP << EscapeSequence::LINE_UP << EscapeSequence::LINE_UP << EscapeSequence::LINE_UP << EscapeSequence::LINE_UP;
    }
    std::cout << "\n\n\n\n\n\n\n"
              << EscapeSequence::SHOW_CURSOR;

    // Fileoutput
    solver.finalize();

    // Print Time statistics and output to file
    system.printSummary( PC3::TimeIt::getTimes(), PC3::TimeIt::getTimesTotal() );
    PC3::TimeIt::toFile( system.filehandler.getFile( "times" ) );

    return 0;
}