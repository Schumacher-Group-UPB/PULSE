#pragma once

#include <SFML/Graphics.hpp>
#include <SFML/Window.hpp>
#include <omp.h>
#include <iostream>
#include "misc/colormap.hpp"
#include "cuda/cuda_complex.cuh"

class BasicWindow {
   protected:
    int mouseX, mouseY;
    bool mouseLB, mouseRB;
    int texture_w, texture_h;

   public:

    // Most basic colors
    static inline auto COLOR_WHITE = sf::Color( 255, 255, 255 );
    static inline auto COLOR_BLACK = sf::Color( 0, 0, 0 );
    static inline auto COLOR_GREEN = sf::Color( 0, 255, 0 );
    static inline auto COLOR_RED = sf::Color( 255, 0, 0 );
    static inline auto COLOR_BLUE = sf::Color( 0, 0, 255 );

    // Keys
    static inline auto KEY_a = sf::Keyboard::A;
    static inline auto KEY_b = sf::Keyboard::B;
    static inline auto KEY_c = sf::Keyboard::C;
    static inline auto KEY_d = sf::Keyboard::D;
    static inline auto KEY_e = sf::Keyboard::E;
    static inline auto KEY_f = sf::Keyboard::F;
    static inline auto KEY_g = sf::Keyboard::G;
    static inline auto KEY_h = sf::Keyboard::H;
    static inline auto KEY_i = sf::Keyboard::I;
    static inline auto KEY_j = sf::Keyboard::J;
    static inline auto KEY_k = sf::Keyboard::K;
    static inline auto KEY_l = sf::Keyboard::L;
    static inline auto KEY_m = sf::Keyboard::M;
    static inline auto KEY_n = sf::Keyboard::N;
    static inline auto KEY_o = sf::Keyboard::O;
    static inline auto KEY_p = sf::Keyboard::P;
    static inline auto KEY_q = sf::Keyboard::Q;
    static inline auto KEY_r = sf::Keyboard::R;
    static inline auto KEY_s = sf::Keyboard::S;
    static inline auto KEY_t = sf::Keyboard::T;
    static inline auto KEY_u = sf::Keyboard::U;
    static inline auto KEY_v = sf::Keyboard::V;
    static inline auto KEY_w = sf::Keyboard::W;
    static inline auto KEY_x = sf::Keyboard::X;
    static inline auto KEY_y = sf::Keyboard::Y;
    static inline auto KEY_z = sf::Keyboard::Z;
    static inline auto KEY_UP = sf::Keyboard::Up;
    static inline auto KEY_DOWN = sf::Keyboard::Down;
    static inline auto KEY_LEFT = sf::Keyboard::Left;
    static inline auto KEY_RIGHT = sf::Keyboard::Right;
    static inline auto KEY_SPACE = sf::Keyboard::Space;
    static inline auto KEY_LSHIFT = sf::Keyboard::LShift;
    static inline auto KEY_RSHIFT = sf::Keyboard::RShift;
    static inline auto KEY_ESCAPE = sf::Keyboard::Escape;

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