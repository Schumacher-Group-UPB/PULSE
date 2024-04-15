#pragma once

#include <iostream>
#include <sstream>
#include <iomanip>
#include "solver/gpu_solver.hpp"
#include "misc/helperfunctions.hpp"

#ifdef SFML_RENDER
#    include <SFML/Graphics.hpp>
#    include <SFML/Window.hpp>
#    include "sfml_window.hpp"
#    include "colormap.hpp"
#endif

/*
 * The solver can be rendered using the SFML library, which is a very simple and easy to use library,
 * which we use to implement different methods to plot the solver in real time.
 * 
 * The Plus and Minus Components are always plotted seperately in different rows.
 * Default: Plot Psi and Reservoir with inset
 * The inset can be switched by pressing 'i' in the SFML window.
 * The inset then cycles between the FFT, logscale FFT, Quantum Noise
 * 
 * This plot "helper" is an abomination by itself and will probably be replaced by a more sophisticated
 * plotting library in the future such as QT.
 * 
*/

namespace PC3 {

std::string toScientific( const real_number in ) {
    std::stringstream ss;
    ss << std::scientific << std::setprecision( 2 ) << in;
    return ss.str();
}

#ifdef SFML_RENDER

BasicWindow& getWindow() {
    static BasicWindow window;
    return window;
}

std::unique_ptr<real_number[]> __plotarray;
template <typename T>
void plotMatrix( T* buffer, int NX, int NY, int posX, int posY, int skip, ColorPalette& cp, const std::string& title = "", bool plot_min_max = true ) {
    if (buffer == nullptr)
        return;
    PC3::CUDA::cwiseAbs2( buffer, __plotarray.get(), NX * NY );
    auto [min, max] = PC3::CUDA::minmax( __plotarray.get(), NX * NY );
    PC3::CUDA::normalize( __plotarray.get(), NX * NY, min, max );
    getWindow().blitMatrixPtr( __plotarray.get(), cp, NX, NY, posX, posY, 1 /*border*/, skip );
    if (!plot_min_max)
        return;
    NX = NX / skip;
    NY = NY / skip;
    auto text_height = 0.05f / skip;
    getWindow().scaledPrint( posX + 5, posY + NY - text_height*NY - 5, text_height, title + "Min: " + toScientific( sqrt(min) ) + " Max: " + toScientific( sqrt(max) ), sf::Color::White );
}

// Very bad practice, as this header can only be imported once without redefinitions.
// TODO: make this into a class like a normal friggin programmer.
ColorPalette __local_colorpalette_phase;
ColorPalette __local_colorpalette;

CheckBox* cb;
CheckBox* min_and_max;
Button* b_add_outevery;
Button* b_sub_outevery;

void initSFMLWindow( PC3::Solver& solver ) {
    if ( solver.system.filehandler.disableRender ) {
        std::cout << "Manually disabled SFML Renderer!" << std::endl;
        return;
    }
    if ( solver.system.use_twin_mode )
        getWindow().construct( 1920, 1080, solver.system.s_N_x * 3, solver.system.s_N_y * 2, "PULSE - TE/TM" );
    else
        getWindow().construct( 1920, 540, solver.system.s_N_x * 3, solver.system.s_N_y, "PULSE - Scalar" );

    // if .pal in __local_colorpalette, read gnuplot __local_colorpalette, else read as .txt
    if ( solver.system.filehandler.color_palette.find( ".pal" ) != std::string::npos ) {
        __local_colorpalette.readColorPaletteFromGnuplotDOTPAL( solver.system.filehandler.color_palette );
        __local_colorpalette_phase.readColorPaletteFromGnuplotDOTPAL( solver.system.filehandler.color_palette_phase );
    } else if ( solver.system.filehandler.color_palette.find( ".txt" ) != std::string::npos ) {
        __local_colorpalette.readColorPaletteFromTXT( solver.system.filehandler.color_palette );
        __local_colorpalette_phase.readColorPaletteFromTXT( solver.system.filehandler.color_palette_phase );
    } else {
        __local_colorpalette.readColorPaletteFromTXT( "resources/"+solver.system.filehandler.color_palette+".txt" );
        __local_colorpalette_phase.readColorPaletteFromTXT( "resources/"+solver.system.filehandler.color_palette_phase+".txt" );
    }
    __local_colorpalette.initColors();
    __local_colorpalette_phase.initColors();
    getWindow().init();
    __plotarray = std::make_unique<real_number[]>( solver.system.s_N_x * solver.system.s_N_y );

    cb = new CheckBox( 10, 50, "Toggle FFT Plot", false );
    getWindow().addObject( cb );
    min_and_max = new CheckBox( 10, 80, "Toggle Min/Max", false );
    getWindow().addObject( min_and_max );
    b_add_outevery = new Button( 10, 150, "Increase" );
    getWindow().addObject( b_add_outevery );
    b_sub_outevery = new Button( 10, 180, "Decrease" );
    getWindow().addObject( b_sub_outevery );
}

int __local_inset = 0;

bool plotSFMLWindow( PC3::Solver& solver, double simulation_time, double elapsed_time, size_t iterations ) {
    if ( solver.system.filehandler.disableRender )
        return true;
    bool running = getWindow().run();

    // Get Device arrays
    solver.syncDeviceArrays();

    complex_number* inset_plot_array_plus = nullptr, *inset_plot_array_minus = nullptr;
    if ( __local_inset == 1 ) {
        inset_plot_array_plus = solver.host.fft_plus.get();
        if ( solver.system.use_twin_mode )
            inset_plot_array_minus = solver.host.fft_minus.get();
    }
    if ( getWindow().keyPressed( BasicWindow::KEY_i ) ) {
        __local_inset = ( __local_inset + 1 ) % 2;
    }

    if (cb->isChecked())
        __local_inset = 1;
    else
        __local_inset = 0;

    bool plot_min_max = min_and_max->isChecked();
    // Plot Plus
    plotMatrix( solver.host.wavefunction_plus.get(), solver.system.s_N_x, solver.system.s_N_y /*size*/, solver.system.s_N_x, 0, 1, __local_colorpalette, "Psi+ ", plot_min_max );
    plotMatrix( inset_plot_array_plus, solver.system.s_N_x, solver.system.s_N_y /*size*/, solver.system.s_N_x, 0, 3, __local_colorpalette, "FFT+ ", plot_min_max );
    plotMatrix( solver.host.reservoir_plus.get(), solver.system.s_N_x, solver.system.s_N_y /*size*/, 2 * solver.system.s_N_x, 0, 1, __local_colorpalette, "n+ ", plot_min_max );
    PC3::CUDA::angle( solver.host.wavefunction_plus.get(), __plotarray.get(), solver.system.s_N_x * solver.system.s_N_y );
    plotMatrix( __plotarray.get(), solver.system.s_N_x, solver.system.s_N_y, 0, 0, 1, __local_colorpalette_phase, "ang(Psi+) ", plot_min_max );

    // Plot Minus
    if ( solver.system.use_twin_mode ) {
        plotMatrix( solver.host.wavefunction_minus.get(), solver.system.s_N_x, solver.system.s_N_y /*size*/, solver.system.s_N_x, solver.system.s_N_y, 1, __local_colorpalette, "Psi- ", plot_min_max );
        plotMatrix( inset_plot_array_minus, solver.system.s_N_x, solver.system.s_N_y /*size*/, solver.system.s_N_x, solver.system.s_N_y, 3, __local_colorpalette, "FFT- ", plot_min_max );
        plotMatrix( solver.host.reservoir_minus.get(), solver.system.s_N_x, solver.system.s_N_y /*size*/, 2 * solver.system.s_N_x, solver.system.s_N_y, 1, __local_colorpalette, "n- ", plot_min_max );
        PC3::CUDA::angle( solver.host.wavefunction_minus.get(), __plotarray.get(), solver.system.s_N_x * solver.system.s_N_y );
        plotMatrix( __plotarray.get(), solver.system.s_N_x, solver.system.s_N_y, 0, solver.system.s_N_y, 1, __local_colorpalette_phase, "ang(Psi-) ", plot_min_max );
    }

    const auto ps_per_second = simulation_time / elapsed_time;
    const auto iterations_per_second = iterations / elapsed_time;

    // FPS and ps/s
    getWindow().print( 5, 5, 0.05, "t = " + std::to_string( int( solver.system.t ) ) + ", FPS: " + std::to_string( int( getWindow().fps ) ) + ", ps/s: " + std::to_string( int( ps_per_second ) ) + ", it/s: " + std::to_string( int( iterations_per_second ) ), sf::Color::White );

    // If the mouse position is less than 200 on the x axis, draw the gui. else, set all gui components invisible
    if (getWindow().MouseX() < 200) {
        getWindow().drawRect( 0, 200, 0, getWindow().height, sf::Color(0,0,0,160), true );
        for (auto& obj : getWindow().objects) {
            obj->visible = true;
        }
        getWindow().print( 10, 120, "Out Every: " + std::to_string(solver.system.output_every) + "ps");
    } else {
        for (auto& obj : getWindow().objects) {
            obj->visible = false;
        }
    }

    if (b_add_outevery->isToggled()) {
        if (solver.system.output_every == 0.0) 
            solver.system.output_every = solver.system.dt;
        solver.system.output_every *= 2;
    }
    if (b_sub_outevery->isToggled()) {
        solver.system.output_every /= 2;
    }

    getWindow().drawObjects();

    // Blit
    getWindow().flipscreen();
    return running;
}

#else
void initSFMLWindow( PC3::Solver& solver ){};
bool plotSFMLWindow( PC3::Solver& solver, double simulation_time, double elapsed_time, size_t iterationsd ) {
    return true;
};
#endif

} // namespace PC3