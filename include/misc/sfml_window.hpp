#pragma once

#include <SFML/Graphics.hpp>
#include <SFML/Window.hpp>
#include <omp.h>
#include <iostream>
#include "misc/colormap.hpp"
#include "cuda/cuda_complex.cuh"

/* Most basic colors */
#define COLOR_WHITE sf::Color( 255, 255, 255 )
#define COLOR_BLACK sf::Color( 0, 0, 0 )
#define COLOR_GREEN sf::Color( 0, 255, 0 )
#define COLOR_RED sf::Color( 255, 0, 0 )
#define COLOR_BLUE sf::Color( 0, 0, 255 )

/* Keys */
#define KEY_a sf::Keyboard::A
#define KEY_b sf::Keyboard::B
#define KEY_c sf::Keyboard::C
#define KEY_d sf::Keyboard::D
#define KEY_e sf::Keyboard::E
#define KEY_f sf::Keyboard::F
#define KEY_g sf::Keyboard::G
#define KEY_h sf::Keyboard::H
#define KEY_i sf::Keyboard::I
#define KEY_j sf::Keyboard::J
#define KEY_k sf::Keyboard::K
#define KEY_l sf::Keyboard::L
#define KEY_m sf::Keyboard::M
#define KEY_n sf::Keyboard::N
#define KEY_o sf::Keyboard::O
#define KEY_p sf::Keyboard::P
#define KEY_q sf::Keyboard::Q
#define KEY_r sf::Keyboard::R
#define KEY_s sf::Keyboard::S
#define KEY_t sf::Keyboard::T
#define KEY_u sf::Keyboard::U
#define KEY_v sf::Keyboard::V
#define KEY_w sf::Keyboard::W
#define KEY_x sf::Keyboard::X
#define KEY_y sf::Keyboard::Y
#define KEY_z sf::Keyboard::Z
#define KEY_UP sf::Keyboard::Up
#define KEY_DOWN sf::Keyboard::Down
#define KEY_LEFT sf::Keyboard::Left
#define KEY_RIGHT sf::Keyboard::Right
#define KEY_SPACE sf::Keyboard::Space
#define KEY_LSHIFT sf::Keyboard::LShift
#define KEY_RSHIFT sf::Keyboard::RShift
#define KEY_ESCAPE sf::Keyboard::Escape

class BasicWindow {
   protected:
    int mouseX, mouseY;
    bool mouseLB, mouseRB;
    int texture_w, texture_h;

   public:
    int width;
    int height;
    double frametime = 0;
    int fps = 0;
    std::string name;
    double time_ticks = 0;
    int maxFPS = 60;
    sf::RenderWindow window;
    sf::Font font;
    sf::RenderTexture mainTexture;
    std::vector<sf::Vertex> pixMat;
    sf::Text printtext;
    sf::Clock clock;
    int textheight;
    int keyDown = -1;
    bool maintexture_has_changed = true;

    BasicWindow( int w = 300, int h = 300, std::string n = "Unnamed" );

    void construct( int window_w, int window_h, int tx_w, int tx_h, std::string n );

    void init();

    bool run();

    void flipscreen();

    void blitMatrixPtr( const real_number* vector, ColorPalette& cp, int cols = 0, int rows = 0, int posX = 0, int posY = 0, int border = 0, int skip = 1 );

    sf::Color convertColorToSFML( int r, int g, int b );

    void print( int x, int y, std::string text, sf::Color textcolor = COLOR_WHITE, int background = 0, sf::Color backgroundcolor = COLOR_BLACK );

    void print( int x, int y, int h, std::string text, sf::Color textcolor = COLOR_WHITE, int background = 0, sf::Color backgroundcolor = COLOR_BLACK );

    bool keyPressed( int key );

    void updateMouseState();

    int MouseX();

    int MouseY();

    bool leftMouseDown();

    bool rightMouseDown();

    void horLine( int y0, int x0, int x1, sf::Color color = COLOR_WHITE );

    void verLine( int x0, int y0, int y1, sf::Color color = COLOR_WHITE );

    void drawRect( int x0, int x1, int y0, int y1, sf::Color color = COLOR_WHITE );

};