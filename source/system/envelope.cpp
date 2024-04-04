#include "cuda/cuda_complex.cuh"
#include "misc/commandline_input.hpp"
#include "system/envelope.hpp"

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
}

void PC3::Envelope::addTemporal( real_number t0, real_number sigma, real_number freq ) {
    this->t0.push_back( t0 );
    this->sigma.push_back( sigma );
    this->freq.push_back( freq );
    // Cache the index of the spacial shape that belongs to this temporal shape
    time_to_index.push_back( amp.size() - 1);
}

int PC3::Envelope::size() {
    return amp.size();
}

PC3::Envelope PC3::Envelope::fromCommandlineArguments( int argc, char** argv, const std::string& key, const bool time ) {
    int index = 0;
    PC3::Envelope ret;

    while ( ( index = findInArgv( "--" + key, argc, argv, index ) ) != -1 ) {
        // Spacial Component

        // Ampltitude.
        real_number amp = getNextInput( argv, argc, key + "_amp", ++index );
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

        // If the next argument is "osc", then we read the temporal component if time is not false
        auto next = getNextStringInput( argv, argc, key + "_next", index );

        if ( not time or next != "osc")
            continue;

        // Temporal Component
        real_number t0 = getNextInput( argv, argc, key + "_t0", index );
        real_number freq = getNextInput( argv, argc, key + "_freq", index );
        real_number sigma = getNextInput( argv, argc, key + "_sigma", index );
        ret.addTemporal( t0, sigma, freq );
    }
    return ret;
}