#pragma once
#include "misc/sfml_window.hpp"

BasicWindow::BasicWindow( int w, int h, std::string n ) {
    construct( w, h, w, h, n );
}

void BasicWindow::construct( int window_w, int window_h, int tx_w, int tx_h, std::string n ) {
    // window_w = tx_w;
    // window_h = tx_h;
    width = window_w;
    height = window_h;
    texture_w = tx_w;
    texture_h = tx_h;
    name = n;
    mainTexture.create( tx_w, tx_h );
    pixMat.clear();
    pixMat.reserve( ( tx_w + 1 ) * ( tx_h + 1 ) );
    font.loadFromFile( "resources/font.ttf" );
    textheight = (int)( 0.015 * window_h );
    printtext.setFont( font );
    printtext.setCharacterSize( textheight );
    printtext.setOutlineColor( COLOR_BLACK );
    printtext.setOutlineThickness( 1.0f );
    for ( int i = 0; i < tx_w; i++ ) {
        for ( int j = 0; j < tx_h; j++ ) {
            pixMat.push_back( sf::Vertex( sf::Vector2f( i + .5f, j + .5f ), sf::Color( 0, 0, 0 ) ) );
        }
    }
    //std::cout << "Constructed Basic window with " << width << "x" << height << " pixels, reserved are " << texture_w << "x" << texture_h << " -> " << pixMat.size() << " pixels." << std::endl;
}

void BasicWindow::init() {
    window.create( sf::VideoMode( width, height, 32 ), name, sf::Style::Default, sf::ContextSettings( 0, 0, 1, 2, 0 ) );
    window.setVerticalSyncEnabled( true );
    mainTexture.setSmooth( true );
    // window.setFramerateLimit(2);
}

bool BasicWindow::run() {
    keyDown = -1;
    window.clear();
    updateMouseState();
    if ( maintexture_has_changed ) {
        mainTexture.draw( pixMat.data(), texture_w * texture_h, sf::Points );
        maintexture_has_changed = false;
    }
    sf::Sprite mainSprite( mainTexture.getTexture() );
    mainSprite.setScale( (float)width / texture_w, (float)height / texture_h );
    window.draw( mainSprite );

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

void BasicWindow::flipscreen() {
    window.display();
}

void BasicWindow::blitMatrixPtr( const real_number* vector, ColorPalette& cp, int cols, int rows, int posX, int posY, int border, int skip ) {
    const int cols_over_skip = cols / skip;
    const int rows_over_skip = rows / skip;
#pragma omp parallel for schedule( dynamic )
    for ( int i = 0; i < cols_over_skip; i++ ) {
        for ( int j = 0; j < rows_over_skip; j++ ) {
            auto c = cp.getColor( vector[( i * skip ) * rows + j * skip] );
            const auto index = ( i + 1 + posX ) * texture_h - 1 - ( j + posY );
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
    maintexture_has_changed = true;
}

sf::Color BasicWindow::convertColorToSFML( int r, int g, int b ) {
    return sf::Color( r, g, b );
}

void BasicWindow::print( int x, int y, std::string text, sf::Color textcolor, int background, sf::Color backgroundcolor ) {
    print( x, y, textheight, text, textcolor, background, backgroundcolor );
}

void BasicWindow::print( int x, int y, int h, std::string text, sf::Color textcolor, int background, sf::Color backgroundcolor ) {
    float x_scale = (float)width / texture_w;
    float y_scale = (float)height / texture_h;
    x = x * x_scale;
    y = y * y_scale;
    printtext.setFillColor( textcolor );
    printtext.setPosition( (float)x, (float)y );
    printtext.setString( text );
    if ( h > 0 )
        printtext.setCharacterSize( h * y_scale );
    window.draw( printtext ); // std::cout << "Test " << text << std::endl;
}

bool BasicWindow::keyPressed( int key ) {
    if ( keyDown == key )
        std::cout << "Key down? " << std::to_string( keyDown == key ) << " and key is " << std::to_string( key ) << std::endl;
    return keyDown == key;
}

void BasicWindow::updateMouseState() {
    mouseLB = sf::Mouse::isButtonPressed( sf::Mouse::Left );
    mouseRB = sf::Mouse::isButtonPressed( sf::Mouse::Right );
    sf::Vector2i position = sf::Mouse::getPosition( window );
    mouseX = position.x;
    mouseY = position.y;
}

int BasicWindow::MouseX() {
    return mouseX;
}

int BasicWindow::MouseY() {
    return mouseY;
}

bool BasicWindow::leftMouseDown() {
    return mouseLB;
}

bool BasicWindow::rightMouseDown() {
    return mouseRB;
}

void BasicWindow::horLine( int y0, int x0, int x1, sf::Color color ) {
    float x_scale = (float)width / texture_w;
    float y_scale = (float)height / texture_h;
    x0 = x0 * x_scale;
    y0 = y0 * y_scale;
    sf::RectangleShape line( sf::Vector2f( x1 - x0, 1.0f ) );
    line.setPosition( sf::Vector2f( x0, y0 ) );
    line.setFillColor( color );
    window.draw( line );
}

void BasicWindow::verLine( int x0, int y0, int y1, sf::Color color ) {
    float x_scale = (float)width / texture_w;
    float y_scale = (float)height / texture_h;
    x0 = x0 * x_scale;
    y0 = y0 * y_scale;
    sf::RectangleShape line( sf::Vector2f( y1 - y0, 1.0f ) );
    line.setPosition( sf::Vector2f( x0, y0 ) );
    line.setFillColor( color );
    line.rotate( 90 );
    window.draw( line );
}

void BasicWindow::drawRect( int x0, int x1, int y0, int y1, sf::Color color ) {
    if ( x0 <= 0 ) x0 = 0;
    if ( x1 >= width ) x1 = width;
    if ( y0 <= 0 ) y0 = 0;
    if ( y1 >= height ) y1 = height;
    horLine( y0, x0, x1, color );
    horLine( y1, x0, x1, color );
    verLine( x0, y0, y1, color );
    verLine( x1, y0, y1, color );
}