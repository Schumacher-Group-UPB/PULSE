#include <filesystem>
#include "system/filehandler.hpp"
#include "misc/commandline_input.hpp"
#include "misc/escape_sequences.hpp"

std::vector<char*> PC3::readConfigFromFile(int argc, char** argv) {
    int index = 0;
    std::string config = "";
    
    std::vector<char*> buffer;
    for ( int i = 0; i < argc; i++ ) {
        buffer.push_back( argv[i] );
    }

    if ( ( index = findInArgv( "--config", argc, argv ) ) == -1 ) 
        return buffer;


    while ((config = getNextStringInput( argv, argc, "config", ++index ) ) != "") {
        std::cout << "Reading configs from file: '" << config << "'" << std::endl;
    
        // If config file exists, read all lines, split all lines and store them in the argv array
        if ( not std::filesystem::exists( config ) ) {
            std::cout << "Config file " << config << " does not exist." << std::endl;
            return buffer;
        }

        std::ifstream file( config );
        std::string line;
        while ( std::getline( file, line ) ) {
            std::istringstream iss( line );
            std::string word;
            while ( iss >> word ) {
                if ( word[0] == '#' ) break;
                buffer.push_back( strdup(word.c_str()) );
            }
        }

    }

    return buffer;
}