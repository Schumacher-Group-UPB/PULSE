#pragma once

#include <iostream>
#include <sstream>
#include <iomanip>
#include "solver/gpu_solver.cuh"
#include "misc/helperfunctions.hpp"

std::string toScientific( const real_number in ) {
    std::stringstream ss;
    ss << std::scientific << std::setprecision( 2 ) << in;
    return ss.str();
}

#ifdef SFML_RENDER
#    include <SFML/Graphics.hpp>
#    include <SFML/Window.hpp>
#    include "sfml_window.hpp"
#    include "colormap.hpp"


std::unique_ptr<real_number[]> plotarray;
template <typename T>
void plotMatrix( BasicWindow& window, T* buffer, int N, int posX, int posY, int skip, ColorPalette& cp, const std::string& title = "" ) {
    PC3::CUDA::cwiseAbs2( buffer, plotarray.get(), N * N );
    auto [min, max] = minmax( plotarray.get(), N * N );
    PC3::CUDA::normalize( plotarray.get(), N * N, min, max );
    window.blitMatrixPtr( plotarray.get(), cp, N, N, posX, posY, 1 /*border*/, skip );
    N = N / skip;
    auto text_height = N * 0.05;
    window.print( posX + 5, posY + N - text_height - 5, text_height, title + "Min: " + toScientific( min ) + " Max: " + toScientific( max ), sf::Color::White );
}

// Very bad practice, as this header can only be imported once without redefinitions
BasicWindow window;
ColorPalette colorpalette_phase;
ColorPalette colorpalette;

void initSFMLWindow( PC3::Solver& solver ) {
    if ( solver.system.filehandler.disableRender ) {
        std::cout << "Manually disabled SFML Renderer!" << std::endl;
        return;
    }
    if (solver.use_te_tm_splitting)
        window.construct( 1920, 1080, solver.system.s_N * 3, solver.system.s_N * 2, "PC3 - TE/TM" );
    else
        window.construct( 1920, 540, solver.system.s_N * 3, solver.system.s_N, "PC3 - Scalar" );

    // if .pal in colorpalette, read gnuplot colorpalette, else read as .txt
    if ( solver.system.filehandler.color_palette.find( ".pal" ) != std::string::npos ){
        colorpalette.readColorPaletteFromGnuplotDOTPAL( solver.system.filehandler.color_palette );
        colorpalette_phase.readColorPaletteFromGnuplotDOTPAL( solver.system.filehandler.color_palette_phase );
    }else{
        colorpalette.readColorPaletteFromTXT( solver.system.filehandler.color_palette );
        colorpalette_phase.readColorPaletteFromTXT( solver.system.filehandler.color_palette_phase );
    }
    colorpalette.initColors();
    colorpalette_phase.initColors();
    window.init();
    plotarray = std::make_unique<real_number[]>( solver.system.s_N * solver.system.s_N );
}

bool plotSFMLWindow( PC3::Solver& solver, double ps_per_second  ) {
    if ( solver.system.filehandler.disableRender )
        return true;
    bool running = window.run();
    
    // Get Device arrays
    solver.syncDeviceArrays();
    
    // Plot Plus
    plotMatrix( window, solver.host.wavefunction_plus.get(), solver.system.s_N /*size*/, solver.system.s_N, 0, 1, colorpalette, "Psi+ " );
    plotMatrix( window, solver.host.fft_plus.get(), solver.system.s_N /*size*/, solver.system.s_N, 0, 3, colorpalette, "FFT+ " );
    plotMatrix( window, solver.host.reservoir_plus.get(), solver.system.s_N /*size*/, 2 * solver.system.s_N, 0, 1, colorpalette, "n+ " );
    PC3::CUDA::angle( solver.host.wavefunction_plus.get(), plotarray.get(), solver.system.s_N * solver.system.s_N );
    plotMatrix( window, plotarray.get(), solver.system.s_N, 0, 0, 1, colorpalette_phase, "ang(Psi+) " );
    
    // Plot Minus
    if (solver.use_te_tm_splitting) {
    plotMatrix( window, solver.host.wavefunction_minus.get(), solver.system.s_N /*size*/, solver.system.s_N, solver.system.s_N, 1, colorpalette, "Psi- " );
    plotMatrix( window, solver.host.fft_minus.get(), solver.system.s_N /*size*/, solver.system.s_N, solver.system.s_N, 3, colorpalette, "FFT- " );
    plotMatrix( window, solver.host.reservoir_minus.get(), solver.system.s_N /*size*/, 2 * solver.system.s_N, solver.system.s_N, 1, colorpalette, "n- " );
    PC3::CUDA::angle( solver.host.wavefunction_minus.get(), plotarray.get(), solver.system.s_N * solver.system.s_N );
    plotMatrix( window, plotarray.get(), solver.system.s_N, 0, solver.system.s_N, 1, colorpalette_phase, "ang(Psi-) " );
    }

    // FPS and ps/s
    window.print( 5, 5, 0.05, "t = " + std::to_string(int(solver.system.t)) + ", FPS: " + std::to_string( int(window.fps) ) + ", ps/s: " + std::to_string( int(ps_per_second) ), sf::Color::White );

    // Blit
    window.flipscreen();
    return running;
}

#else
void initSFMLWindow( PC3::Solver& solver ) {};
bool plotSFMLWindow( PC3::Solver& solver, double ps_per_second  ) {
    return true;
};
#endif

