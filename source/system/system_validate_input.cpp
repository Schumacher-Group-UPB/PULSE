#include "system/system.hpp"
#include "misc/escape_sequences.hpp"

#define VALIDATE( condition, variable, message ) \
    if ( not ( condition ) ) { \
        std::cout << EscapeSequence::YELLOW << "Warning: " << variable << " = " << message << EscapeSequence::RESET << std::endl; \
    }

void PC3::System::validateInputs() {
    // TODO.
    // Also: make sure Envelopes are constructed using try:catch blocks. if pc3 tries to read an envelope but fails, it should display a bright yellow message
    // Maybe split envelope read-in and evaluation into two parts? read in in init, evaluate in validateInputs?   
    // Make sure everything that should be positive is positive
    // Give warnings if any variables appear to be too large or too small
    bool valid = true;

    if ( s_N <= 0 ) {
        std::cout << EscapeSequence::YELLOW << "N = " << s_N << " cannot be negative!" << EscapeSequence::RESET << std::endl;
        valid = false;
    }

    if ( s_N % 2 != 0 ) {
        std::cout << EscapeSequence::YELLOW << "Adjusted N from " << s_N << " to N = " << ( s_N + 1 ) << EscapeSequence::RESET << std::endl;
        valid = false;    
    }

    if (t_max < 0) {
        std::cout << EscapeSequence::YELLOW << "t_max = " << t_max << " cannot be negative!" << EscapeSequence::RESET << std::endl;
        valid = false;
    }
    if (dt <= 0) {
        std::cout << EscapeSequence::YELLOW << "dt = " << dt << " cannot be negative or zero!" << EscapeSequence::RESET << std::endl;
        valid = false;
    }
    if ( dt > t_max)  {
        std::cout << EscapeSequence::YELLOW << "dt = " << dt << " cannot be larger than t_max = " << t_max << "!" << EscapeSequence::RESET << std::endl;
        valid = false;
    }
    if (dt_max < 0) {
        std::cout << EscapeSequence::YELLOW << "dt_max = " << dt_max << " cannot be negative!" << EscapeSequence::RESET << std::endl;
        valid = false;
    }
    if (dt_min < 0) {
        std::cout << EscapeSequence::YELLOW << "dt_min = " << dt_min << " cannot be negative!" << EscapeSequence::RESET << std::endl;
        valid = false;
    }
    if (abs( dt > 2.0*magic_timestep )) {
        std::cout << EscapeSequence::YELLOW << "dt = " << dt << " is very large! Is this intended?" << EscapeSequence::RESET << std::endl;
    }

    if ( s_sub_N <= 0 ) {
        std::cout << EscapeSequence::YELLOW << "Subgrid N = " << s_sub_N << " cannot be negative or zero!" << EscapeSequence::RESET << std::endl;
        valid = false;
    }
    if ( s_N <= 0 ) {
        std::cout << EscapeSequence::YELLOW << "Grid N = " << s_sub_N << " cannot be negative or zero!" << EscapeSequence::RESET << std::endl;
        valid = false;
    }
    
    if ( s_N % s_sub_N != 0 ) {
        std::cout << EscapeSequence::YELLOW << "Subgrid N has to be an integer divisor of main grid N! N = " << s_N << " with subgrid N = " << s_sub_N << EscapeSequence::RESET << std::endl;
        valid = false;
    } 

    if (not valid) {
        std::cout << EscapeSequence::RED << "Invalid input! Exitting!" << EscapeSequence::RESET << std::endl;
        exit( 1 );
    }
}