#pragma once
#include <string>
#include <iostream>
#include "cuda/typedef.cuh"

namespace PHOENIX::CLIO {

/**
 * Pretty Commandline Output. Extended Symbols can be disabled by defining PHOENIX_NO_EXTENDED_SYMBOLS
 * in which case no colors, symbols or other extended features will be used.
 */
enum class Control : size_t {
    Info = 1 << 0,                     // Print blue info sign
    Warning = 1 << 1,                  // Print yellow warning sign
    Error = 1 << 2,                    // Print red cross
    Success = 1 << 3,                  // Print green checkmark
    Debug = 1 << 4,                    // Only print when the internal debug flag is set
    Regular = 1 << 5,                  // Print in regular colors
    Primary = 1 << 6,                  // Print in solid colors
    Secondary = 1 << 7,                // Print in light colors, for secondary information
    FullColor = 1 << 8,                // Also Print the text in color
    FullInfo = Info | FullColor,       // Print the info sign in color
    FullWarning = Warning | FullColor, // Print the warning sign in color
    FullError = Error | FullColor,     // Print the error sign in color
    FullSuccess = Success | FullColor, // Print the success sign in color
    FullDebug = Debug | FullColor,     // Print the debug sign in color
    Underline = 1 << 9,                // Underline the text
    Bold = 1 << 10,                    // Bold the text
    Follow = 1 << 11,                  // -> text
};

// Overwrite | and & operators to allow for combining flags
inline Control operator|( Control a, Control b ) {
    return static_cast<Control>( static_cast<size_t>( a ) | static_cast<size_t>( b ) );
}
inline Control operator&( Control a, Control b ) {
    return static_cast<Control>( static_cast<size_t>( a ) & static_cast<size_t>( b ) );
}

std::string prettyPrint( const std::string& message, Control control = Control::Regular );

/**
 * @brief Finds a string in a the argv array and returns its index.
 * @param toFind The string to find.
 * @param argc The number of commandline arguments.
 * @param argv The commandline arguments.
 * @param start The index to start searching from.
 */
int findInArgv( std::string toFind, int argc, char** argv, int start = 0 );

/**
 * @brief Takes the input vector of commandline arguments and returns parameters
 * at the passed index. The index is then increased by one, allowing for chains
 * of this function to parse multiple parameters.
 * @param arguments The vector of commandline arguments.
 * @param name The name of the parameter to find. This field is just for
 * logging purposes.
 * @param index The index to start searching from. This field is updated to the
 * index of the next parameter.
 */
PHOENIX::Type::real getNextInput( char** arguments, const int argc, const std::string name, int& index );
std::string getNextStringInput( char** arguments, const int argc, const std::string name, int& index );

/**
 * @brief Helper function to print the help message for the program. The function
 * will print the indicator, pad empty spaces until the length L1 is reached, then
 * print the unit, pad empty spaces until the length L2 is reached, then print the
 * description.
 * @param indicator The indicator for the parameter.
 * @param unit The unit of the parameter.
 * @param description The description of the parameter.
 */
std::string unifyLength( std::string w1, std::string w2, std::string w3, int L1 = 40, int L2 = 40, int L3 = 100, std::string seperator = " | " );

/**
 * Centers a given input string using a fixed width
 * @param input The input string
 * @param size The desired width of the line
 * @param fill The fill character, usually a whitespace char ' '
 * @param raw If the input string contains formatting characters, repeat the string as
 * a raw string to correctly calculate the sizes
*/
std::string centerString( const std::string& input, size_t size, const char fill = ' ', const std::string& raw = "" );
std::string centerStringRaw( const std::string& input, size_t size, const std::string& raw = "", const char fill = ' ' );

/**
 * Fills a line with characters
 * @param size The width of the line
 * @param fill The character used to fill the line
*/
std::string fillLine( size_t size, const char fill = ' ' );

/**
 * Automatically calculates the correct representation of a number
 * If the decimal point surpasses the numerical precision, use
 * scientific notation. Else, use fixed point notation.
*/
template <typename T>
std::string to_str( T t ) {
    auto numeric_precision = std::numeric_limits<T>::max_digits10;
    std::stringstream ss;
    if ( t < std::pow( 10, -numeric_precision ) or t > std::pow( 10, numeric_precision ) and t != 0.0 )
        ss << std::scientific;
    else
        ss << std::fixed;
    ss << t;
    return ss.str();
}

std::string createProgressBar( double current, double total, size_t width = 50 );

// Table Printing; Straight from https://en.cppreference.com/w/cpp/io/ios_base/width
/*
class Table {
 public:

    using t_type = std::vector<std::vector<std::string>>;

    std::vector<size_t> widths;
    const std::string sep_left = 
    Table(const std::vector<size_t>& widths) : widths(widths) {};
 
    void print_row(const std::vector<std::string>& row) {
        std::cout << '|';
        for (size_t i = 0; i != row.size(); ++i)
        {
            std::cout << ' ';
            std::cout.width(widths[i]);
            std::cout << row[i] << " |";
        }
        std::cout << '\n';
    };
 
void print_break(const widths_t& widths)
{
    const std::size_t margin = 1;
    std::cout.put('+').fill('-');
    for (std::size_t w : widths)
    {
        std::cout.width(w + margin * 2);
        std::cout << '-' << '+';
    }
    std::cout.put('\n').fill(' ');
};

}

*/

} // namespace PHOENIX::CLIO