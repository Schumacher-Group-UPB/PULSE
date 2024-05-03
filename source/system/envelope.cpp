#include "cuda/cuda_complex.cuh"
#include "misc/commandline_input.hpp"
#include "system/envelope.hpp"
#include "cuda/cuda_complex.cuh"
#include "system/filehandler.hpp"

/**
 * @brief Helperfunction to cast a token seperated list to an enum,
 *        as long as the enum is contained in eval. If not, this
 *        function will throw an exception due to the mat .at access.
 * @tparam eval_enum Enum Class
 * @param input 'split_at' token seperated list of strings
 * @param split_at Token to split the input at
 * @param eval Map that maps a string to an enum
 * @return eval_enum Combined Enum
 */
template <class eval_enum>
eval_enum _cast_string_list_to_enum( const std::string& input, std::string split_at, std::map<std::string, eval_enum> eval ) {
    eval_enum ret = static_cast<eval_enum>( 0 );
    std::string::size_type pos = 0;
    std::string::size_type prev = 0;
    while ( ( pos = input.find( split_at, prev ) ) != std::string::npos ) {
        const auto substr = input.substr( prev, pos - prev );
        const eval_enum eval_result = eval.at( substr );
        ret = ret | eval_result;
        prev = pos + 1;
    }
    const auto substr = input.substr( prev, pos - prev );
    const eval_enum eval_result = eval.at( substr );
    ret = ret | eval_result;
    return ret;
}

void PC3::Envelope::addSpacial( real_number amp, real_number width_x, real_number width_y, real_number x, real_number y,
                                real_number exponent, const std::string& s_type, const std::string& s_pol,
                                const std::string& s_behavior, const std::string& s_m ) {
    this->amp.push_back( amp );
    this->width_x.push_back( width_x );
    this->width_y.push_back( width_y );
    this->x.push_back( x );
    this->y.push_back( y );
    this->exponent.push_back( exponent );
    this->s_type.push_back( s_type );
    auto type = _cast_string_list_to_enum<Type>( s_type, "+", TypeFromString );
    this->type.push_back( type );
    this->s_pol.push_back( s_pol );
    auto pol = _cast_string_list_to_enum<Polarization>( s_pol, "+", PolarizationFromString );
    this->pol.push_back( pol );
    this->s_behavior.push_back( s_behavior );
    auto behavior = _cast_string_list_to_enum<Behavior>( s_behavior, "+", BehaviorFromString );
    this->behavior.push_back( behavior );
    if ( s_m == "None" or s_m == "none" )
        this->m.emplace_back( 0 );
    else
        this->m.emplace_back( std::stoi( s_m ) );
    // Add dummy to path
    this->load_path.push_back( "" );
}

void PC3::Envelope::addSpacial( const std::string& path, real_number amp, const std::string& s_behaviour, const std::string& s_pol ) {
    this->amp.push_back( amp );
    this->width_x.push_back( 0 );
    this->width_y.push_back( 0 );
    this->x.push_back( 0 );
    this->y.push_back( 0 );
    this->exponent.push_back( 0 );
    this->s_type.push_back( "" );
    this->type.push_back( Type::Gauss );
    this->s_pol.push_back( s_pol );
    auto pol = _cast_string_list_to_enum<Polarization>( s_pol, "+", PolarizationFromString );
    this->pol.push_back( pol );
    this->s_behavior.push_back( s_behaviour );
    auto behavior = _cast_string_list_to_enum<Behavior>( s_behaviour, "+", BehaviorFromString );
    this->behavior.push_back( behavior );
    this->m.emplace_back( 0 );
    this->load_path.push_back( path );
}

#include <iostream>
void PC3::Envelope::addTemporal( real_number t0, real_number sigma, real_number freq ) {
    // Create the temporal grouping identifier
    std::string group_identifier = std::to_string( t0 ) + std::to_string( sigma ) + std::to_string( freq );

    // If the group identifier is not present in the map, add it and the temporal components
    if ( not str_to_group_identifier.contains( group_identifier ) ) {
        str_to_group_identifier[group_identifier] = str_to_group_identifier.size();
        this->t0.push_back( t0 );
        this->sigma.push_back( sigma );
        this->freq.push_back( freq );
    }

    // Add the group identifier to the group_identifier vector
    this->group_identifier.push_back( str_to_group_identifier[group_identifier] );
}

int PC3::Envelope::size() const {
    return amp.size();
}

// Returns the group size, which is the number of unique temporal components
// that are present in the envelope. The return value is at least 1 to ensure at
// least one matrix is generated. Maybe this will change later.
int PC3::Envelope::groupSize() const {
    return std::max<int>( 1, str_to_group_identifier.size() );
}

// Returns the size of a specific group. This is the number of spacial components
// that are present in the envelope that belong to the group g.
int PC3::Envelope::sizeOfGroup(int g) const {
    int count = 0;
    for ( int i = 0; i < group_identifier.size(); i++ ) {
        if ( group_identifier[i] == g )
            count++;
    }
    return count;
}

PC3::Envelope PC3::Envelope::fromCommandlineArguments( int argc, char** argv, const std::string& key, const bool time ) {
    int index = 0;
    PC3::Envelope ret;

    while ( ( index = findInArgv( "--" + key, argc, argv, index ) ) != -1 ) {
        // Spacial Component

        // If first argument is "load", save the next argument as the path to the file to load!
        if ( getNextStringInput( argv, argc, key + "_load", ++index ) == "load" ) {
            auto path = getNextStringInput( argv, argc, key + "_path", index );
            std::cout << "Loading envelope from file: " << path << std::endl;
            // Ampltitude.
            real_number amp = getNextInput( argv, argc, key + "_amp", index );
            // Behaviour
            auto sbehavior = getNextStringInput( argv, argc, key + "_behaviour", index );
            // Polarization
            auto spol = getNextStringInput( argv, argc, key + "_pol", index );
            ret.addSpacial( path, amp, sbehavior, spol );
        } else {
            index--;
            // Ampltitude.
            real_number amp = getNextInput( argv, argc, key + "_amp", index );
            // Behaviour
            auto sbehavior = getNextStringInput( argv, argc, key + "_behaviour", index );
            // Width
            real_number width_x = getNextInput( argv, argc, key + "_width_x", index );
            real_number width_y = getNextInput( argv, argc, key + "_width_y", index );
            // X Position
            real_number pos_x = getNextInput( argv, argc, key + "_X", index );
            // Y Position
            real_number pos_y = getNextInput( argv, argc, key + "_Y", index );

            // Polarization
            auto spol = getNextStringInput( argv, argc, key + "_pol", index );

            // Exponent
            real_number exponent = getNextInput( argv, argc, key + "_exponent", index );

            // Charge
            auto sm = getNextStringInput( argv, argc, key + "_m", index );

            // Type
            auto stype = getNextStringInput( argv, argc, key + "_type", index );

            ret.addSpacial( amp, width_x, width_y, pos_x, pos_y, exponent, stype, spol, sbehavior, sm );
        }

        // If the next argument is "osc", then we read the temporal component if time is not false
        auto next = getNextStringInput( argv, argc, key + "_next", index );

        if ( not time or next != "osc" ) {
            ret.addTemporal( 0, 1E20, 0 );
            index--;
            continue;
        }

        // Temporal Component
        real_number t0 = getNextInput( argv, argc, key + "_t0", index );
        real_number freq = getNextInput( argv, argc, key + "_freq", index );
        real_number sigma = getNextInput( argv, argc, key + "_sigma", index );
        ret.addTemporal( t0, sigma, freq );
    }

    // If no envelope was passed, we add a default time envelope. This ensures the constructor of the solver's
    // oscillation struct does not fail and can copy at least one temporal component, even if its not used.
    if ( ret.size() == 0 ) {
        ret.addTemporal( 0, 1E20, 0 );
    }

    return ret;
}

/**
 * Hacky way to calculate the envelope as CUDA::real numbers.
 * This is only done at the beginning of the program and on the CPU.
 * Temporarily copying the results is probably fine.
 */
void PC3::Envelope::calculate( real_number* buffer, const int group, PC3::Envelope::Polarization polarization, Dimensions dim, real_number default_value_if_no_mask ) {
    std::unique_ptr<complex_number[]> tmp_buffer = std::make_unique<complex_number[]>( dim.N_x * dim.N_y );
    calculate( tmp_buffer.get(), group, polarization, dim, default_value_if_no_mask );
// Transfer tmp_buffer to buffer as complex numbers
#pragma omp parallel for
    for ( int i = 0; i < dim.N_x * dim.N_y; i++ ) {
        buffer[i] = CUDA::real( tmp_buffer[i] );
    }
}

void PC3::Envelope::calculate( complex_number* buffer, const int group, PC3::Envelope::Polarization polarization, Dimensions dim, real_number default_value_if_no_mask ) {
#pragma omp parallel for
    for ( int row = 0; row < dim.N_y; row++ ) {
        for ( int col = 0; col < dim.N_x; col++ ) {
            int i = row * dim.N_x + col;
            buffer[i] = complex_number( 0.0, 0.0 );
            bool has_been_set = false;
            for ( int c = 0; c < amp.size(); c++ ) {
                // If the group identifier does not match, skip the mask
                if ( group >= 0 and group_identifier[c] != group )
                    continue;
                // Check if the polarization matches or if the input polarization is both. If not, the envelope is skipped.
                if ( pol[c] != PC3::Envelope::Polarization::Both and pol[c] != polarization and polarization != PC3::Envelope::Polarization::Both )
                    continue;

                // Calculate X,Y in the grid space
                auto cx = -dim.L_x + dim.dx * col;
                auto cy = -dim.L_y + dim.dy * row;
                // If type contains "local", use local coordinates instead
                if ( type[c] & PC3::Envelope::Type::Local ) {
                    cx = -1.0 + 2.0 * col / ( dim.N_x - 1 );
                    cy = -1.0 + 2.0 * row / ( dim.N_y - 1 );
                }
                has_been_set = true;

                // Default Amplitude
                complex_number amplitude = { amp[c], 0.0 };

                real_number exp_factor = 1, pre_fractor = 1, exp_function = 1;
                complex_number charge = { 1.0, 0 };

                // If the matrix was loaded, use cached value
                if ( cache[c] != nullptr ) {
                    amplitude = cache[c][i] * amp[c];
                } else {
                    // Calculate Content of Exponential function
                    exp_factor = 0.5 * ( CUDA::abs2( cx - x[c] ) / width_x[c] / width_x[c] + CUDA::abs2( cy - y[c] ) / width_y[c] / width_y[c] );
                    // Calculate the exponential function
                    exp_function = CUDA::exp( -CUDA::pow( exp_factor, exponent[c] ) );
                    // If the type is a gaussian outer, we calculate CUDA::exp(...)^N instead of CUDA::exp((...)^N)
                    if ( type[c] & PC3::Envelope::Type::OuterExponent )
                        exp_function = CUDA::pow( CUDA::exp( -exp_factor ), exponent[c] );
                    // If the shape is a ring, we multiply the exp function with r^2/w^2 again.
                    pre_fractor = 1.0;
                    if ( type[c] & PC3::Envelope::Type::Ring )
                        pre_fractor = exp_factor;

                    // Charge is e^(i*m*phi) where phi is the angle or r = [x,y]
                    charge = CUDA::exp( complex_number( 0.0, m[c] * atan2( cx - x[c], cy - y[c] ) ) );

                    // Default amplitude is A/sqrt(2pi)/w
                    if ( not( type[c] & PC3::Envelope::Type::NoDivide ) )
                        amplitude = amplitude / CUDA::sqrt( 2 * 3.1415 * width_x[c] * width_y[c] );
                }

                // If the behaviour is adaptive, the amplitude is set to the current value of the buffer instead.
                if ( behavior[c] & PC3::Envelope::Behavior::Adaptive )
                    amplitude = amp[c] * buffer[i];
                if ( behavior[c] & PC3::Envelope::Behavior::Complex )
                    amplitude = complex_number( 0.0, CUDA::real( amplitude ) );

                complex_number contribution = amplitude * pre_fractor * exp_function * charge;
                // Add, multiply or replace the contribution to the buffer.
                if ( behavior[c] & PC3::Envelope::Behavior::Add )
                    buffer[i] = buffer[i] + contribution;
                else if ( behavior[c] == PC3::Envelope::Behavior::Multiply )
                    buffer[i] = buffer[i] * contribution;
                else if ( behavior[c] == PC3::Envelope::Behavior::Replace )
                    buffer[i] = contribution;
            }
            // If no mask has been applied, set the value to the default value.
            // This ensures the mask is always initialized
            if ( not has_been_set )
                buffer[i] = { default_value_if_no_mask, 0 };
        }
    }
    cache.clear();
}