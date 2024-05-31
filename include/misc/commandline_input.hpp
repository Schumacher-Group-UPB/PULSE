#pragma once
#include <string>
#include "cuda/typedef.cuh"

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
PC3::Type::real getNextInput( char** arguments, const int argc, const std::string name, int& index );
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
std::string unifyLength(std::string w1, std::string w2, std::string w3, int L1 = 40, int L2 = 40, int L3 = 100, std::string seperator = " | ");
