#include <cmath>
#include <iostream>
#include <fstream>
#include <complex>
#include <vector>
#include <cstdlib>
#include <omp.h>
#include <chrono>

#include "cuda_complex.cuh"
#include "system.hpp"
#include "kernel.hpp"
#include "helperfunctions.hpp"
#include "timeit.hpp"
#include "sfml_helper.hpp"

int main( int argc, char* argv[] ) {
    // Convert input arguments to system and handler variables
    auto [system, filehandler] = initializeSystem( argc, argv );
    auto buffer = Buffer( system.s_N /*Matrix Size*/ );

    // Initialize Buffer Arrays. Either randomly, or using a mask.
    initializeSystem(system, buffer);

    // Load Matrices from File. If --load was not passed in argv, this method does nothing.
    filehandler.loadMatrices( system, buffer );


    // Create Main Plotwindow. Needs to be compiled with -DSFML_RENDER
    initSFMLWindow( system, filehandler );

    // TODO: das hier in eine funktion und dann nur system übergeben!
    initializeDeviceVariables( system.dx, system.dt, system.g_r, system.s_N, system.m_eff, system.gamma_c, system.g_c, system.g_pm, system.gamma_r, system.R, system.delta_LT, system.xmax, system.h_bar_s );
    initializeDeviceArrays( system.s_N );
    
    // Copy pump to device
    initializePumpVariables( system, filehandler );
    initializePulseVariables( system );

    // Move Initial State to the GPU
    setDeviceArrays( buffer.Psi_Plus.get(), buffer.Psi_Minus.get(), buffer.n_Plus.get(), buffer.n_Minus.get(), system.s_N );

    bool running = true;
    // Main Loop
    while ( system.t < system.t_max and running ) {
        timeit(
            // The CPU should briefly evaluate wether the pulses have to be evaluated
            bool evaluate_pulse = doEvaluatePulse( system );
            // Iterate #out_modulo times
            for ( int i = 0; i < filehandler.out_modulo; i++ ) {
                rungeFunctionIterate( system, evaluate_pulse );
            },
            "Main" );

        timeit(
            cacheValues( system, buffer );
            running = plotSFMLWindow( system, filehandler, buffer );
            , "Plotting" );
        double duration = timeitGet( "Main" ) + timeitGet( "Plotting" );
        auto [min, max] = minmax( dev_current_Psi_Plus, system.s_N * system.s_N, true /*This is a device pointer*/ );
        std::cout << "T = " << int( system.t ) << ", Time per " << filehandler.out_modulo << " iterations: " << duration << "s -> " << 1. / (duration)*system.dt * filehandler.out_modulo << "ps/s, current dt = " << system.dt << "                \r";
    }

    // Get final state from GPU
    getDeviceArrays( buffer.Psi_Plus.get(), buffer.Psi_Minus.get(), buffer.n_Plus.get(), buffer.n_Minus.get(), buffer.fft_plus.get(), buffer.fft_minus.get(), system.s_N );
    
    calculateSollValue(system, buffer, filehandler);

    // Fileoutput
    filehandler.outputMatrices( system, buffer );
    filehandler.cacheToFiles( system, buffer );

    // Free Device Memory
    freeDeviceArrays();

    // Print Time statistics and output to file
    timeitStatisticsSummary( system, filehandler );
    timeitToFile( filehandler.getFile( "times" ) );

    return 0;
}