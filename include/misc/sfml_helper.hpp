#pragma once

#include <iostream>
#include <sstream>
#include <iomanip>
#include "solver/gpu_solver.hpp"
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

BasicWindow& getWindow() {
    static BasicWindow window;
    return window;
}

std::unique_ptr<real_number[]> plotarray;
template <typename T>
void plotMatrix( T* buffer, int NX, int NY, int posX, int posY, int skip, ColorPalette& cp, const std::string& title = "" ) {
    PC3::CUDA::cwiseAbs2( buffer, plotarray.get(), NX * NY );
    auto [min, max] = PC3::CUDA::minmax( plotarray.get(), NX * NY );
    PC3::CUDA::normalize( plotarray.get(), NX * NY, min, max );
    getWindow().blitMatrixPtr( plotarray.get(), cp, NX, NY, posX, posY, 1 /*border*/, skip );
    NX = NX / skip;
    NY = NY / skip;
    auto text_height = NY * 0.05;
    getWindow().print( posX + 5, posY + NY - text_height - 5, text_height, title + "Min: " + toScientific( sqrt(min) ) + " Max: " + toScientific( sqrt(max) ), sf::Color::White );
}

// Very bad practice, as this header can only be imported once without redefinitions
ColorPalette colorpalette_phase;
ColorPalette colorpalette;

void initSFMLWindow( PC3::Solver& solver ) {
    if ( solver.system.filehandler.disableRender ) {
        std::cout << "Manually disabled SFML Renderer!" << std::endl;
        return;
    }
    if ( solver.system.use_te_tm_splitting )
        getWindow().construct( 1920, 1080, solver.system.s_N_x * 3, solver.system.s_N_y * 2, "PULSE - TE/TM" );
    else
        getWindow().construct( 1920, 540, solver.system.s_N_x * 3, solver.system.s_N_y, "PULSE - Scalar" );

    // if .pal in colorpalette, read gnuplot colorpalette, else read as .txt
    if ( solver.system.filehandler.color_palette.find( ".pal" ) != std::string::npos ) {
        colorpalette.readColorPaletteFromGnuplotDOTPAL( solver.system.filehandler.color_palette );
        colorpalette_phase.readColorPaletteFromGnuplotDOTPAL( solver.system.filehandler.color_palette_phase );
    } else if ( solver.system.filehandler.color_palette.find( ".txt" ) != std::string::npos ) {
        colorpalette.readColorPaletteFromTXT( solver.system.filehandler.color_palette );
        colorpalette_phase.readColorPaletteFromTXT( solver.system.filehandler.color_palette_phase );
    } else {
        colorpalette.readColorPaletteFromTXT( "resources/"+solver.system.filehandler.color_palette+".txt" );
        colorpalette_phase.readColorPaletteFromTXT( "resources/"+solver.system.filehandler.color_palette_phase+".txt" );
    }
    colorpalette.initColors();
    colorpalette_phase.initColors();
    getWindow().init();
    plotarray = std::make_unique<real_number[]>( solver.system.s_N_x * solver.system.s_N_y );
}

bool plotSFMLWindow( PC3::Solver& solver, double ps_per_second ) {
    if ( solver.system.filehandler.disableRender )
        return true;
    bool running = getWindow().run();

    // Get Device arrays
    solver.syncDeviceArrays();

    // Plot Plus
    plotMatrix( solver.host.wavefunction_plus.get(), solver.system.s_N_x, solver.system.s_N_y /*size*/, solver.system.s_N_x, 0, 1, colorpalette, "Psi+ " );
    plotMatrix( solver.host.fft_plus.get(), solver.system.s_N_x, solver.system.s_N_y /*size*/, solver.system.s_N_x, 0, 3, colorpalette, "FFT+ " );
    plotMatrix( solver.host.reservoir_plus.get(), solver.system.s_N_x, solver.system.s_N_y /*size*/, 2 * solver.system.s_N_x, 0, 1, colorpalette, "n+ " );
    PC3::CUDA::angle( solver.host.wavefunction_plus.get(), plotarray.get(), solver.system.s_N_x * solver.system.s_N_y );
    plotMatrix( plotarray.get(), solver.system.s_N_x, solver.system.s_N_y, 0, 0, 1, colorpalette_phase, "ang(Psi+) " );

    // Plot Minus
    if ( solver.system.use_te_tm_splitting ) {
        plotMatrix( solver.host.wavefunction_minus.get(), solver.system.s_N_x, solver.system.s_N_y /*size*/, solver.system.s_N_x, solver.system.s_N_y, 1, colorpalette, "Psi- " );
        plotMatrix( solver.host.fft_minus.get(), solver.system.s_N_x, solver.system.s_N_y /*size*/, solver.system.s_N_x, solver.system.s_N_y, 3, colorpalette, "FFT- " );
        plotMatrix( solver.host.reservoir_minus.get(), solver.system.s_N_x, solver.system.s_N_y /*size*/, 2 * solver.system.s_N_x, solver.system.s_N_y, 1, colorpalette, "n- " );
        PC3::CUDA::angle( solver.host.wavefunction_minus.get(), plotarray.get(), solver.system.s_N_x * solver.system.s_N_y );
        plotMatrix( plotarray.get(), solver.system.s_N_x, solver.system.s_N_y, 0, solver.system.s_N_y, 1, colorpalette_phase, "ang(Psi-) " );
    }

    // FPS and ps/s
    getWindow().print( 5, 5, 0.05, "t = " + std::to_string( int( solver.system.t ) ) + ", FPS: " + std::to_string( int( getWindow().fps ) ) + ", ps/s: " + std::to_string( int( ps_per_second ) ), sf::Color::White );

    // Blit
    getWindow().flipscreen();
    return running;
}

#else
void initSFMLWindow( PC3::Solver& solver ){};
bool plotSFMLWindow( PC3::Solver& solver, double ps_per_second ) {
    return true;
};
#endif
