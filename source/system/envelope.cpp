#include <iostream>
#include "cuda/typedef.cuh"
#include "misc/commandline_io.hpp"
#include "misc/escape_sequences.hpp"
#include "system/envelope.hpp"
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

void PC3::Envelope::addSpacial( PC3::Type::real amp, PC3::Type::real width_x, PC3::Type::real width_y, PC3::Type::real x, PC3::Type::real y,
                                PC3::Type::real exponent, const std::string& s_type, const std::string& s_pol,
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

void PC3::Envelope::addSpacial( const std::string& path, PC3::Type::real amp, const std::string& s_behaviour, const std::string& s_pol ) {
    this->amp.push_back( amp );
    width_x.push_back( 0 );
    width_y.push_back( 0 );
    x.push_back( 0 );
    y.push_back( 0 );
    exponent.push_back( 0 );
    s_type.push_back( "" );
    type.push_back( Type::Gauss );
    this->s_pol.push_back( s_pol );
    auto pol = _cast_string_list_to_enum<Polarization>( s_pol, "+", PolarizationFromString );
    this->pol.push_back( pol );
    this->s_behavior.push_back( s_behaviour );
    auto behavior = _cast_string_list_to_enum<Behavior>( s_behaviour, "+", BehaviorFromString );
    this->behavior.push_back( behavior );
    m.emplace_back( 0 );
    load_path.push_back( path );
}

void PC3::Envelope::addTemporal( PC3::Type::real t0, PC3::Type::real sigma, PC3::Type::real freq, const std::string& s_temp ) {
    // Create the temporal grouping identifier
    std::string group_identifier = std::to_string( t0 ) + std::to_string( sigma ) + std::to_string( freq ) + s_temp;

    // If the group identifier is not present in the map, add it and the temporal components
    if ( not str_to_group_identifier.count( group_identifier ) ) {
        str_to_group_identifier[group_identifier] = str_to_group_identifier.size();
        this->t0.push_back( t0 );
        this->sigma.push_back( sigma );
        this->freq.push_back( freq );
        this->s_temp.push_back( s_temp );
        auto temp = _cast_string_list_to_enum<Temporal>( s_temp, "+", TemporalFromString );
        temporal.push_back( temp );
        // Add Dummy to path
        load_path_temporal.push_back( "" );
    }

    // Add the group identifier to the group_identifier vector
    this->group_identifier.push_back( str_to_group_identifier[group_identifier] );
}

void PC3::Envelope::addTemporal( const std::string& path ) {
    // Create the temporal grouping identifier
    std::string group_identifier = path;

    // If the group identifier is not present in the map, add it and the temporal components
    if ( not str_to_group_identifier.count( group_identifier ) ) {
        str_to_group_identifier[group_identifier] = str_to_group_identifier.size();
        t0.push_back( 0 );
        sigma.push_back( 0 );
        freq.push_back( 0 );
        s_temp.push_back( "loaded" );
        temporal.push_back( Temporal::Loaded );
        load_path_temporal.push_back( path );
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
    return CUDA::max<int>( 0, str_to_group_identifier.size() );
}

// Returns the size of a specific group. This is the number of spacial components
// that are present in the envelope that belong to the group g.
int PC3::Envelope::sizeOfGroup( int g ) const {
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

    while ( ( index = PC3::CLIO::findInArgv( "--" + key, argc, argv, index ) ) != -1 ) {
        // Spacial Component
        std::cout << PC3::CLIO::prettyPrint( "Parsing envelope '" + key + "'", PC3::CLIO::Control::Info | PC3::CLIO::Control::Secondary) << std::endl;
        // If first argument is "load", save the next argument as the path to the file to load!
        if ( PC3::CLIO::getNextStringInput( argv, argc, key + "_load", ++index ) == "load" ) {
            auto path = PC3::CLIO::getNextStringInput( argv, argc, key + "_path", index );
            std::cout << PC3::CLIO::prettyPrint( "Queuing envelope '" + key + "' to be loaded from file: '" + path + "'", PC3::CLIO::Control::Info | PC3::CLIO::Control::Secondary) << std::endl;
            // Ampltitude.
            PC3::Type::real amp = PC3::CLIO::getNextInput( argv, argc, key + "_amp", index );
            // Behaviour
            auto sbehavior = PC3::CLIO::getNextStringInput( argv, argc, key + "_behaviour", index );
            // Polarization
            auto spol = PC3::CLIO::getNextStringInput( argv, argc, key + "_pol", index );
            ret.addSpacial( path, amp, sbehavior, spol );
        } else {
            index--;
            // Ampltitude.
            PC3::Type::real amp = PC3::CLIO::getNextInput( argv, argc, key + "_amp", index );
            // Behaviour
            auto sbehavior = PC3::CLIO::getNextStringInput( argv, argc, key + "_behaviour", index );
            // Width
            PC3::Type::real width_x = PC3::CLIO::getNextInput( argv, argc, key + "_width_x", index );
            PC3::Type::real width_y = PC3::CLIO::getNextInput( argv, argc, key + "_width_y", index );
            // X Position
            PC3::Type::real pos_x = PC3::CLIO::getNextInput( argv, argc, key + "_X", index );
            // Y Position
            PC3::Type::real pos_y = PC3::CLIO::getNextInput( argv, argc, key + "_Y", index );

            // Polarization
            auto spol = PC3::CLIO::getNextStringInput( argv, argc, key + "_pol", index );

            // Exponent
            PC3::Type::real exponent = PC3::CLIO::getNextInput( argv, argc, key + "_exponent", index );

            // Charge
            auto sm = PC3::CLIO::getNextStringInput( argv, argc, key + "_m", index );

            // Type
            auto stype = PC3::CLIO::getNextStringInput( argv, argc, key + "_type", index );

            ret.addSpacial( amp, width_x, width_y, pos_x, pos_y, exponent, stype, spol, sbehavior, sm );
        }

        std::cout << PC3::CLIO::prettyPrint( "Added Spacial Component to Envelope '" + key + "'", PC3::CLIO::Control::Success | PC3::CLIO::Control::Secondary ) << std::endl;

        // If the next argument is "osc", then we read the temporal component if time is not false
        auto next = PC3::CLIO::getNextStringInput( argv, argc, key + "_next", index );

        if ( not time or next != "time" ) {
            ret.addTemporal( 0, 0, 0, "constant" );
            index--;
            continue;
        }
        if ( PC3::CLIO::getNextStringInput( argv, argc, key + "_load", index ) == "load" ) {
            auto path = PC3::CLIO::getNextStringInput( argv, argc, key + "_path", index );
            std::cout << PC3::CLIO::prettyPrint( "Queuing temporal envelope '" + key + "' to be loaded from file: '" + path + "'", PC3::CLIO::Control::Info | PC3::CLIO::Control::Secondary) << std::endl;
            ret.addTemporal( path );
        } else {
            index--;
            // Temporal Component
            PC3::Type::real t0, sigma, freq;
            auto s_type = PC3::CLIO::getNextStringInput( argv, argc, key + "_type", index );
            t0 = PC3::CLIO::getNextInput( argv, argc, key + "_t0", index );
            sigma = PC3::CLIO::getNextInput( argv, argc, key + "_sigma", index );
            freq = PC3::CLIO::getNextInput( argv, argc, key + "_freq", index );
            ret.addTemporal( t0, sigma, freq, s_type );
            std::cout << PC3::CLIO::prettyPrint( "Added Temporal Component '" + s_type + "' to Envelope '" + key + "'", PC3::CLIO::Control::Success | PC3::CLIO::Control::Secondary ) << std::endl;
        }
    }

    // Finally, intiialize the temporal envelope vector using the group size.
    ret.temporal_envelope = std::vector<PC3::Type::complex>( ret.groupSize(), 1.0 );

    return ret;
}

/**
 * Hacky way to calculate the envelope as CUDA::real numbers.
 * This is only done at the beginning of the program and on the CPU.
 * Temporarily copying the results is probably fine.
 */
void PC3::Envelope::calculate( PC3::Type::real* buffer, const int group, PC3::Envelope::Polarization polarization, Dimensions dim, PC3::Type::real default_value_if_no_mask ) {
    std::unique_ptr<PC3::Type::complex[]> tmp_buffer = std::make_unique<PC3::Type::complex[]>( dim.N_x * dim.N_y );
    calculate( tmp_buffer.get(), group, polarization, dim, default_value_if_no_mask );
// Transfer tmp_buffer to buffer as complex numbers
#pragma omp parallel for
    for ( int i = 0; i < dim.N_x * dim.N_y; i++ ) {
        buffer[i] = CUDA::real( tmp_buffer[i] );
    }
}

void PC3::Envelope::calculate( PC3::Type::complex* buffer, const int group, PC3::Envelope::Polarization polarization, Dimensions dim, PC3::Type::real default_value_if_no_mask ) {
#pragma omp parallel for
    for ( int row = 0; row < dim.N_y; row++ ) {
        for ( int col = 0; col < dim.N_x; col++ ) {
            int i = row * dim.N_x + col;
            buffer[i] = PC3::Type::complex( 0.0, 0.0 );
            bool has_been_set = false;
            for ( int c = 0; c < amp.size(); c++ ) {
                // If the group identifier does not match, skip the mask
                if ( group >= 0 and group_identifier[c] != group )
                    continue;
                // Check if the polarization matches or if the input polarization is both. If not, the envelope is skipped.
                if ( pol[c] != PC3::Envelope::Polarization::Both and pol[c] != polarization and polarization != PC3::Envelope::Polarization::Both )
                    continue;

                // Calculate X,Y in the grid space
                auto cx = -dim.L_x/2.0 + dim.dx * col;
                auto cy = -dim.L_y/2.0 + dim.dy * row;
                // If type contains "local", use local coordinates instead
                if ( type[c] & PC3::Envelope::Type::Local ) {
                    cx = -1.0 + 2.0 * col / ( dim.N_x - 1 );
                    cy = -1.0 + 2.0 * row / ( dim.N_y - 1 );
                }
                has_been_set = true;

                // Default Amplitude
                PC3::Type::complex amplitude( amp[c], 0.0 );

                PC3::Type::real exp_factor = 1, pre_fractor = 1, exp_function = 1;
                PC3::Type::complex charge( 1.0, 0 );

                // If the matrix was loaded, use cached value
                if ( cache[c] != nullptr ) {
                    amplitude = cache[c][i] * amp[c];
                } else {
                    // Calculate Content of Exponential function
                    exp_factor = 0.5 * ( CUDA::abs2( ( cx - x[c] ) / width_x[c] ) + CUDA::abs2( ( cy - y[c] ) / width_y[c] ) );
                    // Calculate the exponential function
                    exp_function = std::exp( -std::pow( exp_factor, exponent[c] ) );
                    // If the type is a gaussian outer, we calculate std::exp(...)^N instead of std::exp((...)^N)
                    if ( type[c] & PC3::Envelope::Type::OuterExponent )
                        exp_function = std::pow( std::exp( -exp_factor ), exponent[c] );
                    // If the shape is a ring, we multiply the exp function with r^2/w^2 again.
                    pre_fractor = 1.0;
                    if ( type[c] & PC3::Envelope::Type::Ring )
                        pre_fractor = exp_factor;

                    // Charge is e^(i*m*phi) where phi is the angle or r = [x,y]
                    charge = CUDA::exp( PC3::Type::complex( 0.0, m[c] * std::atan2( cx - x[c], cy - y[c] ) ) );

                    // Default amplitude is A/sqrt(2pi)/w
                    if ( not( type[c] & PC3::Envelope::Type::NoDivide ) )
                        amplitude = amplitude / CUDA::sqrt<PC3::Type::real>( 2 * 3.1415 * width_x[c] * width_y[c] );
                }

                // If the behaviour is adaptive, the amplitude is set to the current value of the buffer instead.
                if ( behavior[c] & PC3::Envelope::Behavior::Adaptive )
                    amplitude = amp[c] * buffer[i];
                if ( behavior[c] & PC3::Envelope::Behavior::Complex )
                    amplitude = PC3::Type::complex( 0.0, CUDA::real( amplitude ) );

                PC3::Type::complex contribution = amplitude * pre_fractor * exp_function * charge;
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
                buffer[i] = PC3::Type::complex(default_value_if_no_mask, 0);
        }
    }
    cache.clear();
}

std::string PC3::Envelope::toString() const {
    auto os = std::ostringstream();
    auto gs = groupSize();
    std::string b = "";
    if ( gs > 1 ) {
        os << " Temporal Groups: " << gs << std::endl;
        b = "  ";
    }
    for ( int g = 0; g < groupSize(); g++ ) {
        if ( gs > 1 ) {
            os << "  Temporal Group: " << g << " - contains " << sizeOfGroup( g );
            if ( sizeOfGroup( g ) == 1 )
                os << " Spatial Envelope." << std::endl;
            else
                os << " Spatial Envelopes." << std::endl;
        }
        if ( temporal[g] & PC3::Envelope::Temporal::Constant )
            os << b << "  Constant Temporal Envelope" << std::endl;
        else if ( temporal[g] & PC3::Envelope::Temporal::Loaded )
            os << b << "  Loaded Temporal Envelope from '" << load_path_temporal[g] << "'" << std::endl;
        else if ( temporal[g] & PC3::Envelope::Temporal::IExp )
            os << b << "  Temporal Envelope: IExp with t0 = " << t0[g] << ", sigma = " << sigma[g] << ", freq = " << freq[g] << std::endl;
        else if ( temporal[g] & PC3::Envelope::Temporal::Cos )
            os << b << "  Temporal Envelope: Cos with t0 = " << t0[g] << ", sigma = " << sigma[g] << ", freq = " << freq[g] << std::endl;
        else if ( temporal[g] & PC3::Envelope::Temporal::Gauss )
            os << b << "  Temporal Envelope: Gauss with t0 = " << t0[g] << ", sigma = " << sigma[g] << ", power = " << freq[g] << std::endl;
        for ( int i = 0; i < size(); i++ ) {
            if ( group_identifier[i] != g )
                continue;
            if ( load_path[i] == "" ) {
                os << b << "  Spatial Envelope " << i << ":" << std::endl
                   << "    " << b << "Generated from Parameters:" << std::endl
                   << "    " << b << EscapeSequence::GRAY << PC3::CLIO::unifyLength( "Amplitude: ", std::to_string( amp[i] ), "", 25, 25, 25, " " ) << EscapeSequence::RESET << std::endl
                   << "    " << b << EscapeSequence::GRAY << PC3::CLIO::unifyLength( "Width X: ", std::to_string( width_x[i] ), "mum", 25, 25, 25, " " ) << EscapeSequence::RESET << std::endl
                   << "    " << b << EscapeSequence::GRAY << PC3::CLIO::unifyLength( "Width Y: ", std::to_string( width_y[i] ), "mum", 25, 25, 25, " " ) << EscapeSequence::RESET << std::endl
                   << "    " << b << EscapeSequence::GRAY << PC3::CLIO::unifyLength( "At X: ", std::to_string( x[i] ), "mum", 25, 25, 25, " " ) << EscapeSequence::RESET << std::endl
                   << "    " << b << EscapeSequence::GRAY << PC3::CLIO::unifyLength( "At Y: ", std::to_string( y[i] ), "mum", 25, 25, 25, " " ) << EscapeSequence::RESET << std::endl
                   << "    " << b << EscapeSequence::GRAY << PC3::CLIO::unifyLength( "Gauss Exponent: ", std::to_string( exponent[i] ), "", 25, 25, 25, " " ) << EscapeSequence::RESET << std::endl
                   << "    " << b << EscapeSequence::GRAY << PC3::CLIO::unifyLength( "Type: ", s_type[i], "", 25, 25, 25, " " ) << EscapeSequence::RESET << std::endl
                   << "    " << b << EscapeSequence::GRAY << PC3::CLIO::unifyLength( "Polarization: ", s_pol[i], "", 25, 25, 25, " " ) << EscapeSequence::RESET << std::endl
                   << "    " << b << EscapeSequence::GRAY << PC3::CLIO::unifyLength( "Behavior: ", s_behavior[i], "", 25, 25, 25, " " ) << EscapeSequence::RESET << std::endl;
            } else {
                os << b << "  Envelope " << i << ":" << std::endl
                   << b << EscapeSequence::GRAY << "     Loaded from: " << load_path[i] << EscapeSequence::RESET << std::endl
                   << b << EscapeSequence::GRAY << "     Scaling Amp: " << amp[i] << EscapeSequence::RESET << std::endl
                   << b << EscapeSequence::GRAY << "     Behavior: " << s_behavior[i] << EscapeSequence::RESET << std::endl
                   << b << EscapeSequence::GRAY << "     Polarization: " << s_pol[i] << EscapeSequence::RESET << std::endl;
            }
        }
    }
    return os.str();
}

void PC3::Envelope::updateTemporal( const PC3::Type::real t ) {
    // Iterate only over the group size, as the temporal envelope is the same for all envelopes in the group.
    for (int g = 0; g < groupSize(); g++) {
        temporal_envelope[g] = 1.0;
        // If the envelope is constant, skip
        if ( temporal[g] & PC3::Envelope::Temporal::Constant )
            continue;
        // If the envelope is loaded, interpolate between cached points
        if ( temporal[g] & PC3::Envelope::Temporal::Loaded ) {
            // If index is equal to the last element, we use the last element as the envelope
            if ( t > temporal_time_points[g][0].back() ) {
                temporal_envelope[g] = PC3::Type::complex( temporal_time_points[g][1].back(), temporal_time_points[g][2].back() );
                continue;
            }
            // Find the closest time points to t using std::lower_bound
            auto it = std::lower_bound( temporal_time_points[g][0].begin(), temporal_time_points[g][0].end(), t );
            // We use points[t]-points[t-1] to interpolate, hence we limit the index to 1:points.size()-1
            size_t index = std::min<size_t>( std::max<size_t>( 1, std::distance( temporal_time_points[g][0].begin(), it ) ),temporal_time_points[g][0].size()-1 );
            // Interpolate between the two closest points
            const auto t1 = temporal_time_points[g][0][index-1];
            const auto t2 = temporal_time_points[g][0][index];
            const auto v1 = PC3::Type::complex( temporal_time_points[g][1][index-1], temporal_time_points[g][2][index-1] );
            const auto v2 = PC3::Type::complex( temporal_time_points[g][1][index], temporal_time_points[g][2][index] );
            temporal_envelope[g] = v1 + ( v2 - v1 ) * ( t - t1 ) / ( t2 - t1 );
        }
        // Else, calculate iexp, cos or gaussian envelope
        if ( temporal[g] & PC3::Envelope::Temporal::IExp ) {
            temporal_envelope[g] = gaussian_complex_oscillator( t, t0[g], sigma[g], freq[g] );
        } else if ( temporal[g] & PC3::Envelope::Temporal::Cos ) {
            temporal_envelope[g] = gaussian_oscillator( t, t0[g], sigma[g], freq[g] );
        } else if ( temporal[g] & PC3::Envelope::Temporal::Gauss ) {
            temporal_envelope[g] = gaussian_envelope( t, t0[g], sigma[g], freq[g] );
        }
    }
}