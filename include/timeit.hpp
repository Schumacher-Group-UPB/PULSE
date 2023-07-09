#pragma once
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include "system.hpp"
#include "helperfunctions.hpp"

std::map<std::string, std::vector<double>> timeit_times;
std::map<std::string, double> timeit_times_total;

// build a macro that takes a function as an argument, calls the function and then returns the runtime of said function
#define timeit( func, name )                                                                               \
    {                                                                                                      \
        auto start = std::chrono::high_resolution_clock::now();                                            \
        func;                                                                                              \
        auto end = std::chrono::high_resolution_clock::now();                                              \
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>( end - start ).count() / 1E9; \
        timeit_times[name].emplace_back( duration );                                                       \
        timeit_times_total[name] += duration;                                                              \
    }

double timeitGet( std::string name ) {
    if ( timeit_times[name].size() == 0 )
        return 0;
    return timeit_times[name].back();
}

double timeitGetTotalRuntime() {
    double total = 0;
    for ( const auto& [key, value] : timeit_times_total ) {
        total += value;
    }
    return total;
}

void timeitStatisticsSummary( System& s, MatrixHandler& mh ) {
    const int l = 15;
    std::cout << "===================================================================================" << std::endl;
    std::cout << "============================== PC^3 Runtime Statistics ============================" << std::endl;
    std::cout << "===================================================================================" << std::endl;
    std::cout << "--------------------------------- System Parameters -------------------------------" << std::endl;
    std::cout << unifyLength("N", std::to_string(s.s_N), "",l,l) << std::endl;
    std::cout << unifyLength("N^2", std::to_string(s.s_N * s.s_N), "",l,l) << std::endl;
    std::cout << unifyLength("dx", std::to_string(s.dx), "mum",l,l) << std::endl;
    std::cout << unifyLength("dt", std::to_string(s.dt), "ps",l,l) << std::endl;
    std::cout << unifyLength("gamma_c", std::to_string(s.gamma_c), "ps^-1",l,l) << std::endl;
    std::cout << unifyLength("gamma_r", std::to_string(s.gamma_r), "ps^-1",l,l) << std::endl;
    std::cout << unifyLength("g_c", std::to_string(s.g_c), "meV mum^2",l,l) << std::endl;
    std::cout << unifyLength("g_r", std::to_string(s.g_r), "meV mum^2",l,l) << std::endl;
    std::cout << unifyLength("g_pm", std::to_string(s.g_pm), "meV mum^2",l,l) << std::endl;
    std::cout << unifyLength("R", std::to_string(s.R), "ps^-1 mum^-2",l,l) << std::endl;
    std::cout << unifyLength("delta_LT", std::to_string(s.delta_LT), "meV",l,l) << std::endl;
    std::cout << unifyLength("m_plus", std::to_string(s.m_plus), "",l,l) << std::endl;
    std::cout << unifyLength("m_minus", std::to_string(s.m_minus), "",l,l) << std::endl;
    std::cout << unifyLength("m_eff", std::to_string(s.m_eff), "",l,l) << std::endl;
    std::cout << unifyLength("xmax", std::to_string(s.xmax), "mum",l,l) << std::endl;
    std::cout << "---------------------------------- Pumps and Pulses -------------------------------" << std::endl;
    for (int i = 0; i < s.pulse_amp.size(); i++) {
        std::cout << "Pulse at t0 = " << s.pulse_t0[i] << ", amp = " << s.pulse_amp[i] << ", freq = " << s.pulse_freq[i] << ", sigma = " << s.pulse_sigma[i] << "\n         m = " << s.pulse_m[i] << ", pol = " << s.pulse_pol[i] << ", width = " << s.pulse_width[i] << ", X = " << s.pulse_X[i] << ", Y = " << s.pulse_Y[i] << std::endl;
    }
    for (int i = 0; i < s.pump_amp.size(); i++) {
        std::cout << "Pump at amp = " << s.pump_amp[i] << ", width = " << s.pump_width[i] << ", X = " << s.pump_X[i] << ", Y = " << s.pump_Y[i] << ", pol = " << s.pump_pol[i] << std::endl;
    }
    std::cout << "--------------------------------- Runtime Statistics ------------------------------" << std::endl;
    double total = timeitGetTotalRuntime();
    for ( const auto& [key, value] : timeit_times_total ) {
        std::cout << unifyLength( key + ":", std::to_string( value ) + "s", std::to_string( value / s.t_max * 1E3 ) + "ms/ps", l,l ) << std::endl;
    }
    std::cout << unifyLength( "Total Runtime:", std::to_string( total ) + "s", std::to_string( total / s.t_max * 1E3 ) + "ms/ps", l,l ) << " --> " << std::to_string( total / s.iteration ) << "s/it" << std::endl;
    std::cout << "---------------------------------------- Infos ------------------------------------" << std::endl;
    if (mh.loadPath.size() > 0)
        std::cout << "Loaded Initial Matrices from " << mh.loadPath << std::endl;
    std::cout << "===================================================================================" << std::endl;
}

void timeitToFile(std::ofstream& file) {
    file << "t ";
    for ( const auto& [key, total] : timeit_times ) {
        file << key << " ";
    }
    file << std::endl;
    for (auto index = 0; index < timeit_times.begin()->second.size(); index++) {
        for ( const auto& [key, value] : timeit_times ) {
            file << value[index] << " ";
        }
        file << std::endl;
    }
}