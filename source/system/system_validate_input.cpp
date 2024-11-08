#include "system/system_parameters.hpp"
#include "misc/escape_sequences.hpp"
#include "misc/commandline_io.hpp"

// TODO: make sure this does the right things for every possible input. 

void PHOENIX::SystemParameters::validateInputs() {
    // Warnings
    if ( output_every < 2 * p.dt ) {
        std::cout << PHOENIX::CLIO::prettyPrint(
                         "Output Interval = " + PHOENIX::CLIO::to_str( output_every ) + " is very small! This may lead to slower runtimes due to extensive caching.",
                         PHOENIX::CLIO::Control::Warning )
                  << std::endl;
    }
    // TODO.
    // Also: make sure Envelopes are constructed using try:catch blocks. if PHOENIX tries to read an envelope but fails, it should display a bright yellow message
    // Maybe split envelope read-in and evaluation into two parts? read in in init, evaluate in validateInputs?
    // Make sure everything that should be positive is positive
    // Give warnings if any variables appear to be too large or too small
    bool valid = true;

    if ( p.N_c <= 0 or p.N_r <= 0 ) {
        std::cout << PHOENIX::CLIO::prettyPrint( "N = " + PHOENIX::CLIO::to_str( p.N_c ) + ", " + PHOENIX::CLIO::to_str( p.N_r ) + " cannot be negative!", PHOENIX::CLIO::Control::Warning )
                  << std::endl;
        valid = false;
    }

    if ( p.N_c % 2 != 0 or p.N_r % 2 ) {
        std::cout << PHOENIX::CLIO::prettyPrint( "Input Dimensions have to be even!", PHOENIX::CLIO::Control::Warning ) << std::endl;
        valid = false;
    }

    if ( t_max < 0 ) {
        std::cout << PHOENIX::CLIO::prettyPrint( "t_max = " + PHOENIX::CLIO::to_str( t_max ) + " cannot be negative!", PHOENIX::CLIO::Control::Warning ) << std::endl;
        valid = false;
    }
    if ( p.dt <= 0 ) {
        std::cout << PHOENIX::CLIO::prettyPrint( "dt = " + PHOENIX::CLIO::to_str( p.dt ) + " cannot be negative or zero!", PHOENIX::CLIO::Control::Warning ) << std::endl;
        valid = false;
    }
    if ( p.dt > t_max ) {
        std::cout << PHOENIX::CLIO::prettyPrint( "dt = " + PHOENIX::CLIO::to_str( p.dt ) + " cannot be larger than t_max = " + PHOENIX::CLIO::to_str( t_max ) + "!",
                                             PHOENIX::CLIO::Control::Warning )
                  << std::endl;
        valid = false;
    }
    if ( dt_max < 0 ) {
        std::cout << PHOENIX::CLIO::prettyPrint( "dt_max = " + PHOENIX::CLIO::to_str( dt_max ) + " cannot be negative!", PHOENIX::CLIO::Control::Warning ) << std::endl;
        valid = false;
    }
    if ( dt_min < 0 ) {
        std::cout << PHOENIX::CLIO::prettyPrint( "dt_min = " + PHOENIX::CLIO::to_str( dt_min ) + " cannot be negative!", PHOENIX::CLIO::Control::Warning ) << std::endl;
        valid = false;
    }
    if ( abs( p.dt > 1.1 * magic_timestep ) ) {
        std::cout << PHOENIX::CLIO::prettyPrint( "dt = " + PHOENIX::CLIO::to_str( p.dt ) + " is very large! Is this intended?", PHOENIX::CLIO::Control::Warning ) << std::endl;
    }

    if ( not valid ) {
        std::cout << PHOENIX::CLIO::prettyPrint( "Invalid input! Exitting!", PHOENIX::CLIO::Control::FullError ) << std::endl;
        exit( 1 );
    }
}