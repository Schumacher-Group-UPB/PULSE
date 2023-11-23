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

real_number getNextInput( char** argv, const std::string name, int& index ) {
    if ( global_log_inputs )
        std::cout << EscapeSequence::GREY << "Read input " << name << " as " << argv[ index ] << EscapeSequence::RESET << std::endl;
    return std::stod( argv[ index++ ] );
}

std::string getNextStringInput( char** argv, const std::string name, int& index ) {
    if ( global_log_inputs )
        std::cout << EscapeSequence::GREY << "Read input " << name << " as " << argv[ index ] << EscapeSequence::RESET << std::endl;
    return argv[ index++ ];
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