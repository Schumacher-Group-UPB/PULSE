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

PC3::Type::real getNextInput( char** argv, const int argc, const std::string name, int& index ) {
    if (index >= argc) 
        return 0.0;
    if ( global_log_inputs ) {
        std::cout << EscapeSequence::GRAY << "Read input " << name << " as " << argv[ index ] << EscapeSequence::RESET << std::endl;
    }
    PC3::Type::real result = 0.0;
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
        std::cout << EscapeSequence::GRAY << "Read input " << name << " as " << argv[ index ] << EscapeSequence::RESET << std::endl;
    try {
        return std::string( argv[ index++ ] );
    } catch (const std::invalid_argument& e) {
        std::cerr << "Error: Invalid argument for " << name << std::endl;
        std::cout << EscapeSequence::RED << "Error: parsing envelope. Maybe you missed some arguments?" << EscapeSequence::RESET << std::endl;
        exit( EXIT_FAILURE );
    }
}

#include <sstream>
#include <vector>

std::string padString(const std::string& str, int len) {
    std::string result = str;
    if (result.length() < len) {
        result.append(len - result.length(), ' ');
    }
    return result;
}

std::vector<std::string> splitIntoLines(const std::string& text, int maxLen) {
    std::istringstream iss(text);
    std::vector<std::string> lines;
    std::string word;
    std::string line;

    while (iss >> word) {
        if (line.length() + word.length() + 1 > maxLen) {
            lines.push_back( padString(line, maxLen) );
            line = word;
        } else {
            if (!line.empty()) {
                line += " ";
            }
            line += word;
        }
    }
    if (!line.empty()) {
        lines.push_back(padString(line, maxLen));
    }

    return lines;
}

std::string unifyLength(std::string w1, std::string w2, std::string w3, int L1, int L2, int L3, std::string seperator) {
    auto lines1 = splitIntoLines(w1, L1);
    auto lines2 = splitIntoLines(w2, L2);
    auto lines3 = splitIntoLines(w3, L3);

    std::string result;
    size_t maxLines = std::max(std::max(lines1.size(), lines2.size()), lines3.size());

    for (size_t i = 0; i < maxLines; ++i) {
        if (i < lines1.size()) result += lines1[i]; else result += std::string(L1, ' '); result += seperator;
        if (i < lines2.size()) result += lines2[i]; else result += std::string(L2, ' '); result += seperator;
        if (i < lines3.size()) result += lines3[i]; else result += std::string(L3, ' ');
        if (i < maxLines - 1)
            result += "\n";
    }

    return result;
}