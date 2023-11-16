#include <iomanip>

#include "system/filehandler.hpp"
#include "misc/commandline_input.hpp"

PC3::FileHandler::FileHandler() : 
    outputPath( "data" ),
    loadPath( "" ),
    outputName( "" ),
    color_palette( "resources/vik.txt" ),
    color_palette_phase( "resources/viko.txt" ),
    out_modulo( 5 ),
    disableRender( false ) {};

PC3::FileHandler::FileHandler( int argc, char** argv) : FileHandler() {
    init( argc, argv );
}

void PC3::FileHandler::init( int argc, char** argv ) {
    int index = 0;

    if ( ( index = findInArgv( "--path", argc, argv ) ) != -1 )
        outputPath = getNextStringInput( argv, "path", ++index );
    if ( outputPath.back() != '/' )
        outputPath += "/";

    if ( ( index = findInArgv( "--name", argc, argv ) ) != -1 )
        outputName = getNextStringInput( argv, "name", ++index );

    if ( ( index = findInArgv( "--outEvery", argc, argv ) ) != -1 )
        out_modulo = (int)getNextInput( argv, "out_modulo", ++index );

    // Save Load Path if passed
    if ( ( index = findInArgv( "--load", argc, argv ) ) != -1 )
        loadPath = getNextStringInput( argv, "load", ++index );

    // Colormap
    if ( ( index = findInArgv( "--cmap", argc, argv ) ) != -1 )
        color_palette = getNextStringInput( argv, "cmap", ++index );

    // We can also disable to SFML renderer by using the --nosfml flag.
    if ( findInArgv( "-nosfml", argc, argv ) != -1 )
        disableRender = true;

    // Creating output directory
    const int dir_err = std::system( ( "mkdir " + outputPath ).c_str() );
    if ( -1 == dir_err ) {
        std::cout << "Error creating directory " << outputPath << std::endl;
    } else {
        std::cout << "Succesfully created directory " << outputPath << std::endl;
    }
}

std::string PC3::FileHandler::toPath( const std::string& name ) {
    return outputPath + ( outputPath.back() == '/' ? "" : "/" ) + outputName + ( outputName.empty() ? "" : "_" ) + name + ".txt";
}

std::ofstream& PC3::FileHandler::getFile( const std::string& name ) {
    if ( files.find( name ) == files.end() ) {
        files[name] = std::ofstream( toPath( name ) );
    }
    return files[name];
}

void PC3::FileHandler::loadMatrixFromFile( const std::string& filepath, complex_number* buffer ) {
    std::ifstream filein;
    filein.open( filepath, std::ios::in );
    std::istringstream inputstring;
    std::string line;
    int i = 0;
    real_number x, y, re, im;
    if ( filein.is_open() ) {
        while ( getline( filein, line ) ) {
            if ( line.size() > 2 ) {
                inputstring = std::istringstream( line );
                inputstring >> x >> y >> re >> im;
                buffer[i] = { re, im };
                i++;
            }
        }
        filein.close();
        std::cout << "Loaded " << i << " elements from " << filepath << std::endl;
    } else {
        std::cout << "Error: Couldn't load " << filepath << std::endl;
    }
}

void PC3::FileHandler::outputMatrixToFile( const complex_number* buffer, int row_start, int row_stop, int col_start, int col_stop, const unsigned int N, const real_number xmax, const real_number dx, std::ofstream& out, const std::string& name ) {
    if ( !out.is_open() )
        std::cout << "File " << name << " is not open!" << std::endl;
    for ( int i = row_start; i < row_stop; i++ ) {
        for ( int j = col_start; j < col_stop; j++ ) {
            auto index = j + i * N;
            auto x = -xmax + dx * i;
            auto y = -xmax + dx * j;
            out << x << " " << y << " " << real( buffer[index] ) << " " << imag( buffer[index] ) << "\n";
        }
        out << "\n";
    }
    std::cout << "Output " << ( row_stop - row_start ) * ( col_stop - col_start ) << " elements to " << toPath( name ) << "." << "\n";
}

void PC3::FileHandler::outputMatrixToFile( const complex_number* buffer, int row_start, int row_stop, int col_start, int col_stop, const unsigned int N, const real_number xmax, const real_number dx, const std::string& out ) {
    auto& file = getFile( out );
    outputMatrixToFile( buffer, row_start, row_stop, col_start, col_stop, N, xmax, dx, file, out );
}
void PC3::FileHandler::outputMatrixToFile( const complex_number* buffer, const unsigned int N, const real_number xmax, const real_number dx, const std::string& out ) {
    auto& file = getFile( out );
    outputMatrixToFile( buffer, 0, N, 0, N, N, xmax, dx, file, out );
}
void PC3::FileHandler::outputMatrixToFile( const complex_number* buffer, const unsigned int N, const real_number xmax, const real_number dx, std::ofstream& out, const std::string& name ) {
    outputMatrixToFile( buffer, 0, N, 0, N, N, xmax, dx, out, name );
}

void PC3::FileHandler::outputMatrixToFile( const real_number* buffer, int row_start, int row_stop, int col_start, int col_stop, const unsigned int N, const real_number xmax, const real_number dx, std::ofstream& out, const std::string& name ) {
    if ( !out.is_open() )
        std::cout << "File " << name << " is not open!" << std::endl;
    for ( int i = row_start; i < row_stop; i++ ) {
        for ( int j = col_start; j < col_stop; j++ ) {
            auto index = j + i * N;
            auto x = -xmax + dx * i;
            auto y = -xmax + dx * j;
            out << x << " " << y << " " << buffer[index] << "\n";
        }
        out << "\n";
    }
    std::cout << "Output " << ( row_stop - row_start ) * ( col_stop - col_start ) << " elements to " << toPath( name ) << "." << std::endl;
}
void PC3::FileHandler::outputMatrixToFile( const real_number* buffer, int row_start, int row_stop, int col_start, int col_stop, const unsigned int N, const real_number xmax, const real_number dx, const std::string& out ) {
    auto& file = getFile( out );
    outputMatrixToFile( buffer, row_start, row_stop, col_start, col_stop, N, xmax, dx, file, out );
}
void PC3::FileHandler::outputMatrixToFile( const real_number* buffer, const unsigned int N, const real_number xmax, const real_number dx, const std::string& out ) {
    auto& file = getFile( out );
    outputMatrixToFile( buffer, 0, N, 0, N, N, xmax, dx, file, out );
}
void PC3::FileHandler::outputMatrixToFile( const real_number* buffer, const unsigned int N, const real_number xmax, const real_number dx, std::ofstream& out, const std::string& name ) {
    outputMatrixToFile( buffer, 0, N, 0, N, N, xmax, dx, out, name );
}