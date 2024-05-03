#pragma once

#include <vector>
#include <string>
#include <map>
#include <bit>

#include "cuda/cuda_complex.cuh"

namespace PC3 {

class Envelope {
    public:
    // Parameters to Construct the Envelope from
    std::vector<real_number> amp, width_x, width_y, x, y, exponent;
    std::vector<int> m;
    std::vector<real_number> freq, sigma, t0;
    std::vector<std::string> s_type, s_pol, s_behavior;
    // Or path to load the matrix from
    std::vector<std::string> load_path;
    // Either way, all of these vectors should have the same length

    // Identifier for temporal grouping
    // Same length as amp, width, ... and maps the spatial envelope to the temporal envelopes
    std::vector<int> group_identifier;
    // Helper map to map the temporal group identifier to an index in group_identifier
    std::map<std::string, int> str_to_group_identifier;
    // Helper to load cache matrices from paths
    std::vector<std::unique_ptr<complex_number[]>> cache;

    enum class Type : unsigned int {
        Gauss = 1, // Gaussian Envelope
        OuterExponent = 1 << 1, // Exponent is applied to the total envelope and not just the function argument 
        Ring = 1 << 2, // Ring shape is enabled
        NoDivide = 1 << 3, // The Amplitude is NOT devided by sqrt(2*pi)*w 
        Local = 1 << 4 // The grid is treated from -1 to 1 instead of from -xmax to xmax
    };
    std::vector<Type> type;

    enum class Polarization : unsigned int {
        Plus = 1,
        Minus = 1 << 1,
        Both = 3 // Set to three explicitly such that Plus,Minus = Both
    };
    std::vector<Polarization> pol;

    enum class Behavior : unsigned int {
        Add = 1,
        Multiply = 1 << 1,
        Replace = 1 << 2,
        Adaptive = 1 << 3,
        Complex = 1 << 4,
    };
    std::vector<Behavior> behavior;

    static inline std::map<std::string, Behavior> BehaviorFromString = {
        { "add", Behavior::Add },
        { "multiply", Behavior::Multiply },
        { "replace", Behavior::Replace },
        { "adaptive", Behavior::Adaptive },
        { "complex", Behavior::Complex },
    };
    static inline std::map<std::string, Polarization> PolarizationFromString = {
        { "plus", Polarization::Plus },
        { "minus", Polarization::Minus },
        { "both", Polarization::Both },
    };
    static inline std::map<std::string, Type> TypeFromString = {
        { "gauss" , Type::Gauss },
        { "outerExponent", Type::OuterExponent },
        { "ring", Type::Ring },
        { "noDivide", Type::NoDivide },
        { "local", Type::Local },
    };

    static inline int AllGroups = -1;

    void addSpacial(real_number amp, real_number width_x, real_number width_y, real_number x, real_number y, real_number exponent, const std::string& s_type, const std::string& s_pol, const std::string& s_behavior, const std::string& s_m);
    void addSpacial(const std::string& path, real_number amp, const std::string& s_behaviour, const std::string& s_pol);
    void addTemporal(real_number t0, real_number sigma, real_number freq);
    int size() const;
    int groupSize() const;
    int sizeOfGroup(int g) const;

    struct Dimensions {
        size_t N_x, N_y;
        real_number L_x,L_y,dx,dy;
        Dimensions(size_t N_x, size_t N_y, real_number L_x, real_number L_y, real_number dx, real_number dy) : N_x(N_x), N_y(N_y), L_x(L_x), L_y(L_y), dx(dx), dy(dy) {}
    };

    void calculate( real_number* buffer, const int group, Polarization polarization, Dimensions dim, real_number default_value_if_no_mask = 0.0 );
    void calculate( complex_number* buffer, const int group, Polarization polarization, Dimensions dim, real_number default_value_if_no_mask = 0.0 );

    // We use template functions here to avoid circular dependencies
    template <class FH>
    void prepareCache(FH& filehandler, const Dimensions& dim) {
        if ( cache.size() > 0 )
            return;
        for ( int c = 0; c < load_path.size(); c++ ) {
            cache.push_back( nullptr );
            if ( load_path[c] == "" )
                continue;
            cache.back() = std::make_unique<complex_number[]>( dim.N_x * dim.N_y );
            filehandler.loadMatrixFromFile( load_path[c], cache.back().get() );
        }
    }
    template <class FH, typename T> 
    void calculate(FH& filehandler, T* buffer, const int group, Polarization polarization, Dimensions dim, real_number default_value_if_no_mask = 0.0) {
        prepareCache( filehandler, dim );
        calculate( buffer, group, polarization, dim, default_value_if_no_mask );
    }

    static Envelope fromCommandlineArguments( int argc, char** argv, const std::string& key, const bool time );

    std::string toString() const;
};

// Overload the bitwise OR (|) operator
template <typename T>
typename std::enable_if<std::is_enum<T>::value && (std::is_same<T, Envelope::Behavior>::value || std::is_same<T, Envelope::Polarization>::value || std::is_same<T, Envelope::Type>::value), T>::type operator|(T lhs, T rhs) {
    using underlying_type = typename std::underlying_type<T>::type;
    return static_cast<T>(static_cast<underlying_type>(lhs) | static_cast<underlying_type>(rhs));
}

// Overload the bitwise AND (&) operator. Return a boolean, because we dont need the '&' operator for enums
template <typename T>
typename std::enable_if<std::is_enum<T>::value && (std::is_same<T, Envelope::Behavior>::value || std::is_same<T, Envelope::Polarization>::value || std::is_same<T, Envelope::Type>::value), bool>::type operator&(T lhs, T rhs) {
    using underlying_type = typename std::underlying_type<T>::type;
    return std::has_single_bit<underlying_type>(static_cast<underlying_type>(lhs) & static_cast<underlying_type>(rhs));
}

} // namespace PC3