#pragma once

#include <SFML/Graphics.hpp>
#include <SFML/Window.hpp>
#include <omp.h>
#include <iostream>
#include "colormap.hpp"

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

    BasicWindow( int w = 300, int h = 300, std::string n = "Unnamed" ) {
        construct( w, h, n );
    }

    void construct( int w, int h, std::string n ) {
        width = w;
        height = h;
        name = n;
        mainTexture.create( w, h );
        pixMat.clear();
        pixMat.reserve( ( w + 1 ) * ( h + 1 ) );
        font.loadFromFile( "resources/font.ttf" );
        textheight = (int)( 0.015 * h );
        printtext.setFont( font );
        printtext.setCharacterSize( textheight );
        printtext.setOutlineColor( COLOR_BLACK );
        printtext.setOutlineThickness( 1.0f );
        for ( int i = 0; i < width; i++ ) {
            for ( int j = 0; j < height; j++ ) {
                pixMat.push_back( sf::Vertex( sf::Vector2f( i + .5f, j + .5f ), sf::Color( 0, 0, 0 ) ) );
            }
        }
        std::cout << "Constructed Basic window with " << width << "x" << height << " pixels, reserved " << pixMat.size() << " pixels." << std::endl;
    }

    void init() {
        window.create( sf::VideoMode( width, height, 32 ), name, sf::Style::Default, sf::ContextSettings( 0, 0, 8, 2, 0 ) );
        window.setVerticalSyncEnabled( true );
        mainTexture.setSmooth( true );
        // window.setFramerateLimit(2);
    }

    bool run() {
        keyDown = -1;
        window.clear();
        updateMouseState();
        // mainSprite.setScale(1.0f,-1.0f);
        window.draw( sf::Sprite( mainTexture.getTexture() ) );
        /* Time, FPS */
        frametime = clock.restart().asMilliseconds();
        fps = (int)( 1000.0 / frametime );

        sf::Event event;
        while ( window.pollEvent( event ) ) {
            if ( event.type == sf::Event::Closed )
                window.close();
            if ( event.type == sf::Event::KeyPressed )
                keyDown = event.key.code;
        }
        return window.isOpen();
    }

    void flipscreen() {
        window.display();
    }

    void blitMatrixPtr( const double* vector, ColorPalette& cp, int cols = 0, int rows = 0, int posX = 0, int posY = 0, int border = 0, int skip = 1 ) {
    const int cols_over_skip = cols / skip;
    const int rows_over_skip = rows / skip;
    #pragma omp parallel for schedule( dynamic )
        for ( int i = 0; i < cols_over_skip; i++ ) {
            for ( int j = 0; j < rows_over_skip; j++ ) {
                auto c = cp.getColor( vector[( i * skip ) * rows + j * skip] );
                //const auto index = ( i + posX ) * height + j - posY;
                const auto index = ( i + 1 + posX ) * height - 1 - ( j + posY );
                pixMat.at( index ).color.r = c.r;
                pixMat.at( index ).color.g = c.g;
                pixMat.at( index ).color.b = c.b;
                if ( border != 0 && ( i < border || i >= cols_over_skip - border || j < border || j >= rows_over_skip - border ) ) {
                    pixMat.at( index ).color.r = 0;
                    pixMat.at( index ).color.g = 0;
                    pixMat.at( index ).color.b = 0;
                }
            }
        }
        mainTexture.draw( pixMat.data(), height * width, sf::Points );
    }

    sf::Color convertColorToSFML( int r, int g, int b ) {
        return sf::Color( r, g, b );
    }

    void print( int x, int y, std::string text, sf::Color textcolor = COLOR_WHITE, int background = 0, sf::Color backgroundcolor = COLOR_BLACK ) {
        print( x, y, textheight, text, textcolor, background, backgroundcolor );
    }
    void print( int x, int y, int height, std::string text, sf::Color textcolor = COLOR_WHITE, int background = 0, sf::Color backgroundcolor = COLOR_BLACK ) {
        printtext.setFillColor( textcolor );
        printtext.setPosition( (float)x, (float)y );
        printtext.setString( text );
        if ( height > 0 )
            printtext.setCharacterSize( height );
        window.draw( printtext ); // std::cout << "Test " << text << std::endl;
    }

    bool keyPressed( int key ) {
        if ( keyDown == key )
            std::cout << "Key down? " << std::to_string( keyDown == key ) << " and key is " << std::to_string( key ) << std::endl;
        return keyDown == key;
    }
    void updateMouseState() {
        mouseLB = sf::Mouse::isButtonPressed( sf::Mouse::Left );
        mouseRB = sf::Mouse::isButtonPressed( sf::Mouse::Right );
        sf::Vector2i position = sf::Mouse::getPosition( window );
        mouseX = position.x;
        mouseY = position.y;
    }
    int MouseX() {
        return mouseX;
    }
    int MouseY() {
        return mouseY;
    }
    bool leftMouseDown() {
        return mouseLB;
    }
    bool rightMouseDown() {
        return mouseRB;
    }
    void horLine( int y0, int x0, int x1, sf::Color color = COLOR_WHITE ) {
        sf::RectangleShape line( sf::Vector2f( x1 - x0, 1.0f ) );
        line.setPosition( sf::Vector2f( x0, y0 ) );
        line.setFillColor( color );
        window.draw( line );
    }
    void verLine( int x0, int y0, int y1, sf::Color color = COLOR_WHITE ) {
        sf::RectangleShape line( sf::Vector2f( y1 - y0, 1.0f ) );
        line.setPosition( sf::Vector2f( x0, y0 ) );
        line.setFillColor( color );
        line.rotate( 90 );
        window.draw( line );
    }
    void drawRect( int x0, int x1, int y0, int y1, sf::Color color = COLOR_WHITE ) {
        if ( x0 <= 0 ) x0 = 0;
        if ( x1 >= width ) x1 = width;
        if ( y0 <= 0 ) y0 = 0;
        if ( y1 >= height ) y1 = height;
        horLine( y0, x0, x1, color );
        horLine( y1, x0, x1, color );
        verLine( x0, y0, y1, color );
        verLine( x1, y0, y1, color );
    }
};