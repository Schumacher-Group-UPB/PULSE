#pragma once

// #define SFML_RENDER
#include "system.hpp"
#include "helperfunctions.hpp"
#include "kernel.hpp"
#include "cuda_device_variables.cuh"

#ifdef SFML_RENDER
#    include <SFML/Graphics.hpp>
#    include <SFML/Window.hpp>
#    include "sfml_window.hpp"
#    include "colormap.hpp"

#    include <iostream>
#    include <sstream>
#    include <iomanip>

std::string toScientific( const real_number in ) {
    std::stringstream ss;
    ss << std::scientific << std::setprecision( 2 ) << in;
    return ss.str();
}

std::unique_ptr<real_number[]> plotarray;
template <typename T>
void plotMatrix( BasicWindow& window, T* buffer, int N, int posX, int posY, int skip, ColorPalette& cp, const std::string& title = "" ) {
    cwiseAbs2( buffer, plotarray.get(), N * N );
    auto [min, max] = minmax( plotarray.get(), N * N );
    normalize( plotarray.get(), N * N, min, max );
    window.blitMatrixPtr( plotarray.get(), cp, N, N, posX, posY, 1 /*border*/, skip );
    N = N / skip;
    auto text_height = N * 0.05;
    window.print( posX + 5, posY + N - text_height - 5, text_height, title + "Min: " + toScientific( min ) + " Max: " + toScientific( max ), sf::Color::White );
}

// Very bad practice, as this header can only be imported once without redefinitions
BasicWindow window;
ColorPalette colorpalette_phase;
ColorPalette colorpalette;

#endif

void initSFMLWindow( System& system, FileHandler& filehandler ) {
#ifdef SFML_RENDER
    if ( filehandler.disableRender ) {
        std::cout << "Manually disabled SFML Renderer!" << std::endl;
        return;
    }
    #ifdef TETMSPLITTING
    window.construct( 1920, 1080, system.s_N * 3, system.s_N * 2, "PC3" );
    #else
    window.construct( 1920, 540, system.s_N * 3, system.s_N, "PC3" );
    #endif
    // if .pal in colorpalette, read gnuplot colorpalette, else read as .txt
    if ( filehandler.color_palette.find( ".pal" ) != std::string::npos ){
        colorpalette.readColorPaletteFromGnuplotDOTPAL( filehandler.color_palette );
        colorpalette_phase.readColorPaletteFromGnuplotDOTPAL( filehandler.color_palette_phase );
    }else{
        colorpalette.readColorPaletteFromTXT( filehandler.color_palette );
        colorpalette_phase.readColorPaletteFromTXT( filehandler.color_palette_phase );
    }
    colorpalette.initColors();
    colorpalette_phase.initColors();
    window.init();
    plotarray = std::make_unique<real_number[]>( system.s_N * system.s_N );
#else
    std::cout << "PC^3 Compiled without SFML Renderer!" << std::endl;
#endif
}

bool plotSFMLWindow( System& system, FileHandler& handler, Buffer& buffer ) {
#ifdef SFML_RENDER
    if ( handler.disableRender )
        return true;
    bool running = window.run();
    getDeviceArrays( buffer.Psi_Plus.get(), buffer.Psi_Minus.get(), buffer.n_Plus.get(), buffer.n_Minus.get(), buffer.fft_plus.get(), buffer.fft_minus.get(), system.s_N );
    plotMatrix( window, buffer.Psi_Plus.get(), system.s_N /*size*/, system.s_N, 0, 1, colorpalette, "Psi+ " );
    plotMatrix( window, buffer.fft_plus.get(), system.s_N /*size*/, system.s_N, 0, 3, colorpalette, "FFT+ " );
    plotMatrix( window, buffer.n_Plus.get(), system.s_N /*size*/, 2 * system.s_N, 0, 1, colorpalette, "n+ " );
    angle( buffer.Psi_Plus.get(), plotarray.get(), system.s_N * system.s_N );
    plotMatrix( window, plotarray.get(), system.s_N, 0, 0, 1, colorpalette_phase, "ang(Psi+) " );
    #ifdef TETMSPLITTING
    plotMatrix( window, buffer.Psi_Minus.get(), system.s_N /*size*/, system.s_N, system.s_N, 1, colorpalette, "Psi- " );
    plotMatrix( window, buffer.fft_minus.get(), system.s_N /*size*/, system.s_N, system.s_N, 3, colorpalette, "FFT- " );
    plotMatrix( window, buffer.n_Minus.get(), system.s_N /*size*/, 2 * system.s_N, system.s_N, 1, colorpalette, "n- " );
    angle( buffer.Psi_Minus.get(), plotarray.get(), system.s_N * system.s_N );
    plotMatrix( window, plotarray.get(), system.s_N, 0, system.s_N, 1, colorpalette_phase, "ang(Psi-) " );
    #endif
    window.flipscreen();
    return running;
#endif
    return true;
}