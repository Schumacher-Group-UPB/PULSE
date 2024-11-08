#pragma once
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <chrono>
#include <fstream>

namespace PHOENIX::TimeIt {

double get( std::string name );

double totalRuntime();

void toFile( std::ofstream& file );

void addTime( const std::string& name, double duration );

void clear();

std::map<std::string, std::vector<double>>& getTimes();

std::map<std::string, double>& getTimesTotal();

} // namespace PHOENIX::TimeIt

// build a macro that takes a function as an argument, calls the function and then returns the runtime of said function
#define TimeThis( func, name )                                                                                                           \
    {                                                                                                                                    \
        auto _timethis_start = std::chrono::high_resolution_clock::now();                                                                \
        func;                                                                                                                            \
        auto _timethis_end = std::chrono::high_resolution_clock::now();                                                                  \
        auto _timethis_duration = std::chrono::duration_cast<std::chrono::nanoseconds>( _timethis_end - _timethis_start ).count() / 1E9; \
        PHOENIX::TimeIt::addTime( name, _timethis_duration );                                                                                \
    }
