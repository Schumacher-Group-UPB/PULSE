#pragma once
#include <fstream>
#include <iostream>
#include <vector>
#include "cuda_complex.cuh"
#include "cuda_complex_math.cuh"
#include <omp.h>
#include "system.hpp"

/**
 * @brief Converts the command line arguments to a vector of strings.
 *
 * @param argc The number of command line arguments.
 * @param argv The command line arguments.
 * @return std::vector<std::string> A vector of strings containing the command line arguments.
 */
std::vector<std::string> argv_to_vec( int argc, char** argv );

/**
 * @brief Finds a string in a vector of strings and returns its index.
 * @param toFind The string to find.
 * @param input The vector of strings to search in.
 * @param start The index to start searching from.
 */
int vec_find_str( std::string toFind, const std::vector<std::string>& input, int start = 0 );

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
real_number getNextInput( const std::vector<std::string>& arguments, const std::string name, int& index );
std::string getNextStringInput( const std::vector<std::string>& arguments, const std::string name, int& index );

/**
 * @brief Helper function to print the help message for the program. The function
 * will print the indicator, pad empty spaces until the length L1 is reached, then
 * print the unit, pad empty spaces until the length L2 is reached, then print the
 * description.
 * @param indicator The indicator for the parameter.
 * @param unit The unit of the parameter.
 * @param description The description of the parameter.
 */
std::string unifyLength( std::string indicator, std::string unit, std::string description, int L1 = 30, int L2 = 30 );

/**
 * @brief Prints the help message for the program.
 */
static void printSystemHelp( System& s, FileHandler& h );

/**
 * @brief Takes the system and a given set of pulse parameters and adds it to the
 * system pulse cache array. This array is then later pushed to the GPU memory.
 */
void addPulse( System& s, real_number t0, real_number amp, real_number freq, real_number sigma, int m, int pol, real_number width, real_number x, real_number y );

/**
 * @brief Takes the system and a given set of pump parameters and adds it to the
 * system pump cache array. This array is then later pushed to the GPU memory.
 */
void addPump( System& s, real_number P0, real_number w, real_number x, real_number y, int pol, real_number exponent, int type );

/**
 * @brief Initializes the system and the file handler variables from the argc
 * and argv commandline arguments. This function also handles the help message.
 * @param argc The number of commandline arguments.
 * @param argv The commandline arguments.
 */
std::tuple<System, FileHandler> initializeSystem( int argc, char** argv );

/**
 * @brief Takes the system variable and transfers its pump array to the GPU.
 */
void initializePumpVariables( System& s, FileHandler& filehandler );

/**
 * @brief Takes the system variable and transfers its pulse array to the GPU.
 */
void initializePulseVariables( System& s );


struct compare_complex_abs2 {
    CUDA_HOST_DEVICE bool operator()( complex_number lhs, complex_number rhs ) {
        return real(lhs) * real(lhs) + imag(lhs) * imag(lhs) < real(rhs) * real(rhs) + imag(rhs) * imag(rhs);
    }
};

/**
 * @brief Calculates the minimum and maximum of a buffer of (complex) numbers
 * @param z The buffer to calculate the minimum and maximum of.
 * @param size The size of the buffer.
 * @return std::tuple<real_number, real_number> A tuple containing the minimum and maximum
 */
std::tuple<real_number, real_number> minmax( complex_number* buffer, int size, bool device_pointer = false );
std::tuple<real_number, real_number> minmax( real_number* buffer, int size, bool device_pointer = false );

/**
 * @brief Normalizes a buffer of real numbers using the minimum and maximum
 * values passed to the function. If min == max == 0, the min and max are
 * recalculated using the minmax function.
 * @param buffer The buffer to normalize.
 * @param size The size of the buffer.
 * @param min The minimum value to normalize to.
 * @param max The maximum value to normalize to.
 */
void normalize( real_number* buffer, int size, real_number min = 0, real_number max = 0, bool device_pointer = false );

/**
 * @brief Calculates the angle of a buffer of complex numbers, as long as
 * std::arg is defined for the type T.
 * @param z The complex number buffer to calculate the angle of.
 * @param buffer The buffer to save the result to.
 * @param size The size of the buffer.
 */
void angle( complex_number* z, real_number* buffer, int size );

/**
 * @brief Helper function to figure out wether or not to evaluate the pulse at the
 * current time. Returns true if the pulse should be evaluated.
 */
bool doEvaluatePulse( const System& system );

/**
 * @brief Helper function to grab the current wavefunction cut at Y = 0 and save
 * it to a vector.
 */
std::vector<complex_number> cacheVector( const System& s, const complex_number* buffer );

/**
 * @brief Calculates the current maximum (and minimum) of the wavefunction and
 * saves it to the buffer. Also saves the current wavefunction cut at Y = 0 to
 * the buffer.
 */
void cacheValues( const System& system, Buffer& buffer );