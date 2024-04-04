#include <iostream>
#include "misc/commandline_input.hpp"
#include "misc/escape_sequences.hpp"

static inline bool global_log_inputs = false;

int findInArgv( std::string toFind, int argc, char** argv, int start ) {
    for ( int i = start; i < argc; i++ ) {
        std::string current = std::string( argv[i] );
        if ( current.compare( toFind ) == 0 )
            return i;
    }
    return -1;
}

real_number getNextInput( char** argv, const int argc, const std::string name, int& index ) {
    if (index >= argc) 
        return 0.0;
    if ( global_log_inputs ) {
        std::cout << EscapeSequence::GREY << "Read input " << name << " as " << argv[ index ] << EscapeSequence::RESET << std::endl;
    }
    real_number result = 0.0;
    try {
        result = std::stod( argv[ index++ ] );
    } catch (const std::invalid_argument& e) {
        std::cerr << "Error: Invalid argument for " << name << std::endl;
        std::cout << EscapeSequence::RED << "Error: parsing envelope " << name << " as " << argv[ index ] << " cannot be converted to a numerical value! Exitting!" << EscapeSequence::RESET << std::endl;
        exit( EXIT_FAILURE );
    }
    return result;
}

std::string getNextStringInput( char** argv, const int argc, const std::string name, int& index ) {
    if (index >= argc) 
        return "";
    if ( global_log_inputs )
        std::cout << EscapeSequence::GREY << "Read input " << name << " as " << argv[ index ] << EscapeSequence::RESET << std::endl;
    try {
        return std::string( argv[ index++ ] );
    } catch (const std::invalid_argument& e) {
        std::cerr << "Error: Invalid argument for " << name << std::endl;
        std::cout << EscapeSequence::RED << "Error: parsing envelope. Maybe you missed some arguments?" << EscapeSequence::RESET << std::endl;
        exit( EXIT_FAILURE );
    }
}

std::string unifyLength( std::string indicator, std::string unit, std::string description, int L1, int L2 ) {
    int l1 = L1 - indicator.size();
    int l2 = L2 - unit.size();
    std::string ret = indicator;
    for ( int i = 0; i < l1; i++ )
        ret += " ";
    ret += unit;
    for ( int i = 0; i < l2; i++ )
        ret += " ";
    ret += description;
    return ret;
}