#include <iomanip>

#include <filesystem>
#include "system/filehandler.hpp"
#include "misc/commandline_input.hpp"
#include "misc/escape_sequences.hpp"
#include "omp.h"

PC3::FileHandler::FileHandler() : outputPath( "data" ),
                                  loadPath( "data" ),
                                  outputName( "" ),
                                  color_palette( "vik" ),
                                  color_palette_phase( "viko" ),
                                  disableRender( false ){};

PC3::FileHandler::FileHandler( int argc, char** argv ) : FileHandler() {
    init( argc, argv );
}

void PC3::FileHandler::init( int argc, char** argv ) {
    int index = 0;
    if ( ( index = findInArgv( "--path", argc, argv ) ) != -1 )
        outputPath = getNextStringInput( argv, argc, "path", ++index );
    if ( outputPath.back() != '/' )
        outputPath += "/";

    if ( ( index = findInArgv( "--name", argc, argv ) ) != -1 )
        outputName = getNextStringInput( argv, argc, "name", ++index );

    // Save Load Path if passed, else use output path as laod path
    loadPath = outputPath;
    if ( ( index = findInArgv( "--loadFrom", argc, argv ) ) != -1 )
        loadPath = getNextStringInput( argv, argc, "loadFrom", ++index );

    // Colormap
    if ( ( index = findInArgv( "--cmap", argc, argv ) ) != -1 ) {
        color_palette = getNextStringInput( argv, argc, "cmap", ++index );
        color_palette_phase = getNextStringInput( argv, argc, "cmap", index );
    }

    // We can also disable to SFML renderer by using the --nosfml flag.
    if ( findInArgv( "-nosfml", argc, argv ) != -1 )
        disableRender = true;

    // Creating output directory.
    try {
        std::filesystem::create_directories( outputPath );
        std::cout << "Successfully created directory " << outputPath << std::endl;
    } catch ( std::filesystem::filesystem_error& e ) {
        std::cout << EscapeSequence::RED << "Error creating directory " << outputPath << ": " << e.what() << EscapeSequence::RESET << std::endl;
    }

    // Create timeoutput subdirectory if --historyMatrix is passed.
    if ( findInArgv( "--historyMatrix", argc, argv ) != -1 ) {
        try {
            std::filesystem::create_directories( outputPath + "timeoutput" );
            std::cout << "Successfully created sub directory " << outputPath + "timeoutput" << std::endl;
        } catch ( std::filesystem::filesystem_error& e ) {
            std::cout << EscapeSequence::RED << "Error creating directory " << outputPath + "timeoutput"
                      << ": " << e.what() << EscapeSequence::RESET << std::endl;
        }
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
    double re, im;
    if ( not filein.is_open() ) {
#pragma omp critical
        std::cout << EscapeSequence::YELLOW << "Warning: Unable to load '" << filepath << "'" << EscapeSequence::RESET << std::endl;
        return;
    }
    // Header
    getline( filein, line );
    inputstring = std::istringstream( line );
    // Read SIZE Nx Ny sLx sLy dx dy
    size_t n_x, n_y;
    real_number header;
    inputstring >> line >> line >> n_x >> n_y;
    size_t N = n_x * n_y;
    while ( getline( filein, line ) ) {
        inputstring = std::istringstream( line );
        if ( line.size() < 2 )
            continue;
        if (i < N)
            while ( inputstring >> re ) {
                buffer[i] = { real_number(re), 0 };
                i++;
            }
        else
            while ( inputstring >> im ) {
                buffer[i - N] = { CUDA::real( buffer[i - N] ), real_number(im) };
                i++;
            }
    }
    filein.close();
    std::cout << "Loaded " << i << " elements from '" << filepath << "'" << std::endl;
}

void PC3::FileHandler::loadMatrixFromFile( const std::string& filepath, real_number* buffer ) {
    std::ifstream filein;
    filein.open( filepath, std::ios::in );
    std::istringstream inputstring;
    std::string line;
    int i = 0;
    double val;
    if ( not filein.is_open() ) {
#pragma omp critical
        std::cout << EscapeSequence::YELLOW << "Warning: Unable to load '" << filepath << "'" << EscapeSequence::RESET << std::endl;
        return;
    }

    // Header
    getline( filein, line );
    inputstring = std::istringstream( line );
    // Read SIZE Nx Ny sLx sLy dx dy
    size_t n_x, n_y;
    real_number header;
    inputstring >> line >> line >> n_x >> n_y;
    size_t N = n_x * n_y;
    std::cout << "Loading " << N << " elements from '" << filepath << "'" << std::endl;
    while ( getline( filein, line ) ) {
        if ( line.size() < 2 )
            continue;
        while ( inputstring >> val ) {
            buffer[i] = real_number(val);
            i++;
        }
    }
    filein.close();
    std::cout << "Loaded " << i << " elements from " << filepath << std::endl;
}

void PC3::FileHandler::outputMatrixToFile( const complex_number* buffer,unsigned  int col_start, unsigned int col_stop, unsigned int row_start, unsigned int row_stop, const unsigned int N_x, const unsigned int N_y, unsigned int increment, const Header& header, std::ofstream& out, const std::string& name ) {
    if ( !out.is_open() ) {
        std::cout << "File '" << name << "' is not open!" << std::endl;
        return;
    }
    // Header
    out << "# SIZE " << col_stop-col_start << " " << row_stop-row_start << " " << header << " :: PULSE MATRIX\n";
    std::stringstream output_buffer;
    // Real
    for ( int i = row_start; i < row_stop; i += increment ) {
        for ( int j = col_start; j < col_stop; j += increment ) {
            auto index = j + i * N_x;
            output_buffer << CUDA::real( buffer[index] ) << " ";
        }
        output_buffer << "\n";
    }
    // Imag
    for ( int i = row_start; i < row_stop; i += increment ) {
        for ( int j = col_start; j < col_stop; j += increment ) {
            auto index = j + i * N_x;
            output_buffer << CUDA::imag( buffer[index] ) << " ";
        }
        output_buffer << "\n";
    }
    out << output_buffer.str();
    out.flush();
    out.close();
#pragma omp critical
    std::cout << "Output " << ( row_stop - row_start ) * ( col_stop - col_start ) / increment << " elements to " << toPath( name ) << "."
              << "\n";
}

void PC3::FileHandler::outputMatrixToFile( const complex_number* buffer,unsigned  int col_start, unsigned int col_stop, unsigned int row_start, unsigned int row_stop, const unsigned int N_x, const unsigned int N_y, unsigned int increment, const Header& header, const std::string& out ) {
    auto& file = getFile( out );
    outputMatrixToFile( buffer, col_start, col_stop, row_start, row_stop, N_x, N_y, increment, header, file, out );
}
void PC3::FileHandler::outputMatrixToFile( const complex_number* buffer, const unsigned int N_x, const unsigned int N_y, const Header& header, const std::string& out ) {
    auto& file = getFile( out );
    outputMatrixToFile( buffer, 0, N_x, 0, N_y, N_x, N_y, 1.0, header, file, out );
}
void PC3::FileHandler::outputMatrixToFile( const complex_number* buffer, const unsigned int N_x, const unsigned int N_y, const Header& header, std::ofstream& out, const std::string& name ) {
    outputMatrixToFile( buffer, 0, N_x, 0, N_y, N_x, N_y, 1.0, header, out, name );
}

void PC3::FileHandler::outputMatrixToFile( const real_number* buffer,unsigned  int col_start, unsigned int col_stop, unsigned int row_start, unsigned int row_stop, const unsigned int N_x, const unsigned int N_y, unsigned int increment, const Header& header, std::ofstream& out, const std::string& name ) {
    if ( !out.is_open() ) {
        std::cout << "File " << name << " is not open!" << std::endl;
        return;
    }
    // Header
    out << "# SIZE " << col_stop-col_start << " " << row_stop-row_start << " " << header << " :: PULSE MATRIX\n";
    std::stringstream output_buffer;
    // Real
    for ( int i = row_start; i < row_stop; i += increment ) {
        for ( int j = col_start; j < col_stop; j += increment ) {
            auto index = j + i * N_x;
            output_buffer << buffer[index] << " ";
        }
        output_buffer << "\n";
    }
    out << output_buffer.str();
    out.flush();
    out.close();
#pragma omp critical
    std::cout << "Output " << ( row_stop - row_start ) * ( col_stop - col_start ) / increment << " elements to '" << toPath( name ) << "'." << std::endl;
}
void PC3::FileHandler::outputMatrixToFile( const real_number* buffer,unsigned  int col_start, unsigned int col_stop, unsigned int row_start, unsigned int row_stop, const unsigned int N_x, const unsigned int N_y, unsigned int increment, const Header& header, const std::string& out ) {
    auto& file = getFile( out );
    outputMatrixToFile( buffer, col_start, col_stop, row_start, row_stop, N_x, N_y, increment, header, file, out );
}
void PC3::FileHandler::outputMatrixToFile( const real_number* buffer, const unsigned int N_x, const unsigned int N_y, const Header& header, const std::string& out ) {
    auto& file = getFile( out );
    outputMatrixToFile( buffer, 0, N_x, 0, N_y, N_x, N_y, 1.0, header, file, out );
}
void PC3::FileHandler::outputMatrixToFile( const real_number* buffer, const unsigned int N_x, const unsigned int N_y, const Header& header, std::ofstream& out, const std::string& name ) {
    outputMatrixToFile( buffer, 0, N_x, 0, N_y, N_x, N_y, 1.0, header, out, name );
}