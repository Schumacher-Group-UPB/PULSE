#include "system/system.hpp"
#include "misc/escape_sequences.hpp"

#define VALIDATE( condition, variable, message ) \
    if ( not ( condition ) ) { \
        std::cout << EscapeSequence::YELLOW << "Warning: " << variable << " = " << message << EscapeSequence::RESET << std::endl; \
    }

void PC3::System::validateInputs() {
    // Warnings
    if (output_every < 2*p.dt) {
        std::cout << EscapeSequence::YELLOW << "Output Interval = " << output_every << " is very small!\nThis may lead to slower runtimes due to extensive caching." << EscapeSequence::RESET << std::endl;
    }
    // TODO.
    // Also: make sure Envelopes are constructed using try:catch blocks. if pc3 tries to read an envelope but fails, it should display a bright yellow message
    // Maybe split envelope read-in and evaluation into two parts? read in in init, evaluate in validateInputs?   
    // Make sure everything that should be positive is positive
    // Give warnings if any variables appear to be too large or too small
    bool valid = true;

    if ( p.N_x <= 0 or p.N_y <= 0 ) {
        std::cout << EscapeSequence::YELLOW << "N = " << p.N_x << ", " << p.N_y << " cannot be negative!" << EscapeSequence::RESET << std::endl;
        valid = false;
    }

    if ( p.N_x % 2 != 0 or p.N_y % 2) {
        std::cout << EscapeSequence::YELLOW << "Input Dimensions have to be even!" << EscapeSequence::RESET << std::endl;
        valid = false;    
    }

    if (t_max < 0) {
        std::cout << EscapeSequence::YELLOW << "t_max = " << t_max << " cannot be negative!" << EscapeSequence::RESET << std::endl;
        valid = false;
    }
    if (p.dt <= 0) {
        std::cout << EscapeSequence::YELLOW << "dt = " << p.dt << " cannot be negative or zero!" << EscapeSequence::RESET << std::endl;
        valid = false;
    }
    if ( p.dt > t_max)  {
        std::cout << EscapeSequence::YELLOW << "dt = " << p.dt << " cannot be larger than t_max = " << t_max << "!" << EscapeSequence::RESET << std::endl;
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
    if (abs( p.dt > 1.1*magic_timestep )) {
        std::cout << EscapeSequence::YELLOW << "dt = " << p.dt << " is very large! Is this intended?" << EscapeSequence::RESET << std::endl;
    }

    if (not valid) {
        std::cout << EscapeSequence::RED << "Invalid input! Exitting!" << EscapeSequence::RESET << std::endl;
        exit( 1 );
    }
}