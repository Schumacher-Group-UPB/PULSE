#pragma once

#include <SFML/Graphics.hpp>
#include <SFML/Window.hpp>
#include <omp.h>
#include <iostream>
#include "misc/colormap.hpp"
#include "cuda/cuda_complex.cuh"

class BasicWindow;

class WindowObject {
    public:
     int x, y;
     bool visible = true;
    
     WindowObject( int x, int y, bool visible = true ) : x( x ), y( y ), visible( visible ) {}
    
     virtual bool draw( BasicWindow* basicwindow ) = 0;
     virtual void update() = 0;
     virtual void onClick( int x, int y ) = 0;
     virtual void onHover( int x, int y, bool mouseDown ) = 0;
};

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
    static inline auto KEY_PLUS = sf::Keyboard::Add;
    static inline auto KEY_MINUS = sf::Keyboard::Subtract;

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
    float textheight;
    int keyDown = -1;
    bool maintexture_has_changed = true;
    bool mouseLB_old = false;
    bool mouseRB_old = false;

    std::vector<WindowObject*> objects;

    BasicWindow( int w = 300, int h = 300, std::string n = "Unnamed" );

    void construct( int window_w, int window_h, int tx_w, int tx_h, std::string n );

    void init();

    bool run();

    void flipscreen();

    void blitMatrixPtr( const real_number* vector, ColorPalette& cp, int cols = 0, int rows = 0, int posX = 0, int posY = 0, int border = 0, int skip = 1 );

    sf::Color convertColorToSFML( int r, int g, int b );

    void print( int x, int y, std::string text, sf::Color textcolor = COLOR_WHITE, int background = 0, sf::Color backgroundcolor = COLOR_BLACK );
    void print( int x, int y, float h, std::string text, sf::Color textcolor = COLOR_WHITE, int background = 0, sf::Color backgroundcolor = COLOR_BLACK );
    void scaledPrint( int x, int y, std::string text, sf::Color textcolor = COLOR_WHITE, int background = 0, sf::Color backgroundcolor = COLOR_BLACK );
    void scaledPrint( int x, int y, float h, std::string text, sf::Color textcolor = COLOR_WHITE, int background = 0, sf::Color backgroundcolor = COLOR_BLACK );

    bool keyPressed( int key );

    void updateMouseState();

    int MouseX();

    int MouseY();

    bool leftMouseDown();
    bool rightMouseDown();
    bool leftMouseClicked();
    bool rightMouseClicked();

    void horLine( int y0, int x0, int x1, sf::Color color = COLOR_WHITE );

    void verLine( int x0, int y0, int y1, sf::Color color = COLOR_WHITE );

    void drawRect( int x0, int x1, int y0, int y1, sf::Color color = COLOR_WHITE, bool filled = false );

    void addObject( WindowObject* object );
    void drawObjects();

};

class CheckBox : public WindowObject {
    public:
     bool checked = false;
     std::string text;
     int w,h;
     CheckBox( int x, int y, std::string text, bool checked = false ) : WindowObject( x, y ), text( text ), checked( checked ) {
        w = 20;
        h = 20;
     }
    
     bool isChecked() {
        return checked;
     }

     bool draw( BasicWindow* basicwindow ) override {
        if ( !visible )
            return false;
        sf::RectangleShape rect_outer( sf::Vector2f( w, h ) );
        sf::RectangleShape rect_inner( sf::Vector2f( w-4, h-4 ) );
        sf::RectangleShape rect( sf::Vector2f( w-8, h-8 ) );
        rect_outer.setPosition( x, y );
        rect_inner.setPosition( x+2, y+2 );
        rect.setPosition( x+4, y+4 );

        rect_outer.setFillColor( sf::Color::White );
        rect_inner.setFillColor( sf::Color::Black );
        rect.setFillColor( checked ? sf::Color( 50, 50, 50 ) : sf::Color::White );
        basicwindow->window.draw( rect_outer );
        basicwindow->window.draw( rect_inner );
        basicwindow->window.draw( rect );
        basicwindow->print( x + w+5, y, text );
        return true;
    }

    void update() override {}

    void onClick( int x, int y ) override {
        if ( x >= this->x && x <= this->x + w && y >= this->y && y <= this->y + h ) {
            checked = !checked;
        }
    }
    void onHover( int x, int y, bool mouseDown ) override {
        //if ( x >= this->x && x <= this->x + w && y >= this->y && y <= this->y + h )
    }
};  

class Button : public WindowObject {
    public:
     std::string text;
     bool toggled;
     int w,h;
     Button( int x, int y, std::string text, bool toggled = false ) : WindowObject( x, y ), text( text ), toggled( toggled ){
        w = 100;
        h = 25;
     }

     bool draw( BasicWindow* basicwindow ) override {
        if ( !visible )
            return false;
        sf::RectangleShape rect_outer( sf::Vector2f( w, h ) );
        sf::RectangleShape rect_inner( sf::Vector2f( w-4, h-4 ) );
        rect_outer.setPosition( x, y );
        rect_inner.setPosition( x+2, y+2 );

        rect_outer.setFillColor( sf::Color::Black );
        rect_inner.setFillColor( sf::Color( 50, 50, 50 ) );
        basicwindow->window.draw( rect_outer );
        basicwindow->window.draw( rect_inner );
        basicwindow->print( x +5, y-2, text );
        return true;
    }

    void update() override {}

    void onClick( int x, int y ) override {
        if ( x >= this->x && x <= this->x + w && y >= this->y && y <= this->y + h ) {
            toggled = true;        
        }
    }
    void onHover( int x, int y, bool mouseDown ) override {
        //if ( x >= this->x && x <= this->x + w && y >= this->y && y <= this->y + h )
    }
    bool isToggled() {
        if (toggled) {
            toggled = false;
            return true;
        }
        return false;
    }
};  