#pragma once

#include <vector>
#include <string>
#include <map>
#include <bit>

#include "cuda/cuda_complex.cuh"

namespace PC3 {

class Envelope {
    public:
    std::vector<real_number> amp, width_x, width_y, x, y, exponent;
    std::vector<int> m;
    std::vector<real_number> freq, sigma, t0;
    std::vector<int> time_to_index;
    std::vector<std::string> s_type, s_pol, s_behavior;

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

    void addSpacial(real_number amp, real_number width_x, real_number width_y, real_number x, real_number y, real_number exponent, const std::string& s_type, const std::string& s_pol, const std::string& s_behavior, const std::string& s_m);
    void addTemporal(real_number t0, real_number sigma, real_number freq);
    int size();

    static Envelope fromCommandlineArguments( int argc, char** argv, const std::string& key, const bool time );
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