#include <sstream>
#include <string>
#include <vector>
#include <iostream>
#include <iomanip>
#include "misc/escape_sequences.hpp"
#include "misc/commandline_io.hpp"

// Helper Variable for logging inputs. Only for simple debugging purposes.
static inline bool global_log = false;

// #define PHOENIX_NO_EXTENDED_SYMBOLS

#ifdef PHOENIX_NO_EXTENDED_SYMBOLS

static std::string string_info_sym = "[i]";
static std::string string_warning_sym = "[w]";
static std::string string_error_sym = "[x]";
static std::string string_success_sym = "[v]";
static std::string string_debug_sym = "[o]";

    #ifdef PHOENIX_NO_ANSI_COLORS
static std::string string_progress_sym_front = "=";
static std::string string_progress_sym_back = "-";
    #else
static std::string string_progress_sym_back = "#";
static std::string string_progress_sym_front = "#";
    #endif

#else

// If PHOENIX_NO_EXTENDED_SYMBOLS is defined globally, colors are disabled.
static std::string string_info_sym = EscapeSequence::BLUE + std::string( reinterpret_cast<const char*>( u8"\u2139" ) ) + EscapeSequence::RESET;
static std::string string_warning_sym = EscapeSequence::YELLOW + std::string( reinterpret_cast<const char*>( u8"\u26A0" ) ) + EscapeSequence::RESET;
static std::string string_error_sym = EscapeSequence::RED + std::string( reinterpret_cast<const char*>( u8"\u2612" ) ) + EscapeSequence::RESET;
static std::string string_success_sym = EscapeSequence::GREEN + std::string( reinterpret_cast<const char*>( u8"\u2611" ) ) + EscapeSequence::RESET;
static std::string string_debug_sym = EscapeSequence::GRAY + std::string( reinterpret_cast<const char*>( u8"\u2699" ) ) + EscapeSequence::RESET;

    #ifdef PHOENIX_NO_ANSI_COLORS
static std::string string_progress_sym_front = std::string( reinterpret_cast<const char*>( u8"\u2587" ) );
static std::string string_progress_sym_back = " ";
    #else
static std::string string_progress_sym_front = std::string( reinterpret_cast<const char*>( u8"\u2587" ) );
static std::string string_progress_sym_back = std::string( reinterpret_cast<const char*>( u8"\u2587" ) );
    #endif

#endif

std::string PHOENIX::CLIO::prettyPrint( const std::string& message, Control control ) {
    std::string ret = "";

    // If the debug flag is set, but global debug is not, return an empty string.
    if ( (size_t)control & (size_t)Control::Debug && !global_log )
        return ret;

    // print Infobox.
    if ( (size_t)control & (size_t)Control::Info )
        ret += string_info_sym;
    else if ( (size_t)control & (size_t)Control::Warning )
        ret += string_warning_sym;
    else if ( (size_t)control & (size_t)Control::Error )
        ret += string_error_sym;
    else if ( (size_t)control & (size_t)Control::Success )
        ret += string_success_sym;
    else if ( (size_t)control & (size_t)Control::Debug )
        ret += string_debug_sym;
    ret += "  | ";

    // If the FullColor flag is set, print the message in color.
    if ( (size_t)control & (size_t)Control::FullColor ) {
        if ( (size_t)control & (size_t)Control::Info )
            ret += EscapeSequence::BLUE;
        else if ( (size_t)control & (size_t)Control::Warning )
            ret += EscapeSequence::YELLOW;
        else if ( (size_t)control & (size_t)Control::Error )
            ret += EscapeSequence::RED;
        else if ( (size_t)control & (size_t)Control::Success )
            ret += EscapeSequence::GREEN;
        else if ( (size_t)control & (size_t)Control::Debug )
            ret += EscapeSequence::GRAY;
    } else {
        if ( (size_t)control & (size_t)Control::Secondary )
            ret += EscapeSequence::GRAY;
    }

    // Set Underline and bold flags
    if ( (size_t)control & (size_t)Control::Underline )
        ret += EscapeSequence::UNDERLINE;
    if ( (size_t)control & (size_t)Control::Bold )
        ret += EscapeSequence::BOLD;

    // Print the message
    ret += message + EscapeSequence::RESET;

    return ret;
}
std::string PHOENIX::CLIO::createProgressBar( double current, double total, size_t width ) {
    std::string ret = "[" + EscapeSequence::BLUE;
    for ( int i = 0; i < width * current / total; i++ ) {
        ret += string_progress_sym_front;
    }
    ret += EscapeSequence::GRAY;
    for ( int i = 0; i < width * ( 1. - current / total ); i++ ) {
        ret += string_progress_sym_back;
    }
    ret += "]  " + EscapeSequence::RESET + std::to_string( int( 100. * current / total ) ) + "%   ";
    return ret;
}

int PHOENIX::CLIO::findInArgv( const std::string& toFind, int argc, char** argv, int start ) {
    for ( int i = start; i < argc; i++ ) {
        std::string current = std::string( argv[i] );
        if ( current.compare( toFind ) == 0 )
            return i;
    }
    return -1;
}

int PHOENIX::CLIO::findInArgv( const std::vector<std::string>& toFind, int argc, char** argv, int start, const std::string& prefix ) {
    for ( int i = start; i < argc; i++ ) {
        std::string current = std::string( argv[i] );
        for ( auto& tf : toFind )
            if ( current.compare( prefix+tf ) == 0 )
                return i;
    }
    return -1;
}

PHOENIX::Type::real PHOENIX::CLIO::getNextInput( char** argv, const int argc, const std::string name, int& index ) {
    if ( index >= argc )
        return 0.0;
    if ( global_log ) {
        std::cout << prettyPrint( "Read input '" + name + "' as '" + std::string( argv[index] ) + "'", Control::Secondary | Control::Info ) << std::endl;
    }
    PHOENIX::Type::real result = 0.0;
    try {
        result = std::stod( argv[index++] );
    } catch ( const std::invalid_argument& e ) {
        std::cout << prettyPrint( "Error: Invalid argument for '" + name + "'", Control::FullError ) << std::endl;
        std::cout << prettyPrint( "Error: parsing variable '" + name + "' as '" + std::string( argv[index] ) + "' cannot be converted to a numerical value! Exitting!",
                                  Control::FullError )
                  << std::endl;
        exit( EXIT_FAILURE );
    }
    return result;
}

std::string PHOENIX::CLIO::getNextStringInput( char** argv, const int argc, const std::string name, int& index ) {
    if ( index >= argc )
        return "";
    if ( global_log )
        std::cout << prettyPrint( "Read input " + name + " as " + std::string( argv[index] ), Control::Secondary | Control::Info ) << std::endl;
    try {
        return std::string( argv[index++] );
    } catch ( const std::invalid_argument& e ) {
        std::cout << prettyPrint( "Error: Invalid argument for " + name, Control::FullError ) << std::endl;
        std::cout << prettyPrint( "Error: parsing variable '" + name + "' as '" + std::string( argv[index] ) + "'. Maybe you missed some arguments? Exitting!", Control::FullError )
                  << std::endl;
        exit( EXIT_FAILURE );
    }
}

std::string padString( const std::string& str, int len ) {
    std::string result = str;
    if ( result.length() < len ) {
        result.append( len - result.length(), ' ' );
    }
    return result;
}

std::vector<std::string> splitIntoLines( const std::string& text, int maxLen ) {
    std::istringstream iss( text );
    std::vector<std::string> lines;
    std::string word;
    std::string line;

    while ( iss >> word ) {
        if ( line.length() + word.length() + 1 > maxLen ) {
            lines.push_back( padString( line, maxLen ) );
            line = word;
        } else {
            if ( !line.empty() ) {
                line += " ";
            }
            line += word;
        }
    }
    if ( !line.empty() ) {
        lines.push_back( padString( line, maxLen ) );
    }

    return lines;
}

std::string PHOENIX::CLIO::unifyLength( std::string w1, std::string w2, std::string w3, int L1, int L2, int L3, std::string seperator ) {
    auto lines1 = splitIntoLines( w1, L1 );
    auto lines2 = splitIntoLines( w2, L2 );
    auto lines3 = splitIntoLines( w3, L3 );

    std::string result;
    size_t maxLines = std::max( std::max( lines1.size(), lines2.size() ), lines3.size() );

    for ( size_t i = 0; i < maxLines; ++i ) {
        if ( i < lines1.size() )
            result += lines1[i];
        else
            result += std::string( L1, ' ' );
        result += seperator;
        if ( i < lines2.size() )
            result += lines2[i];
        else
            result += std::string( L2, ' ' );
        result += seperator;
        if ( i < lines3.size() )
            result += lines3[i];
        else
            result += std::string( L3, ' ' );
        if ( i < maxLines - 1 )
            result += "\n";
    }

    return result;
}

std::string PHOENIX::CLIO::centerString( const std::string& input, size_t size, const char fill, const std::string& raw ) {
    size_t raw_size = raw.size() > 0 ? raw.size() : input.size();
    int padding = std::floor( ( size - raw_size ) / 2 );
    std::stringstream ss;
    ss << std::setw( padding ) << std::setfill( fill ) << "" << input;
    ss << std::setw( size - padding - raw_size ) << std::setfill( fill ) << "";
    return ss.str();
}

std::string PHOENIX::CLIO::centerStringRaw( const std::string& input, size_t size, const std::string& raw, const char fill ) {
    return PHOENIX::CLIO::centerString( input, size, fill, raw );
}

std::string PHOENIX::CLIO::fillLine( size_t size, const char fill ) {
    std::stringstream ss;
    ss << std::setfill( fill ) << std::setw( size ) << "";
    return ss.str();
}