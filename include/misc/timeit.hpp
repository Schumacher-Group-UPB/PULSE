#pragma once
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <chrono>
#include <fstream>

namespace PC3::TimeIt {

double get( std::string name );

double totalRuntime();

void toFile( std::ofstream& file );

void addTime( const std::string& name, double duration );

void clear();

std::map<std::string, std::vector<double>>& getTimes();

std::map<std::string, double>& getTimesTotal();


}  // namespace PC3

// build a macro that takes a function as an argument, calls the function and then returns the runtime of said function
#define TimeThis( func, name )                                                                             \
    {                                                                                                      \
        auto start = std::chrono::high_resolution_clock::now();                                            \
        func;                                                                                              \
        auto end = std::chrono::high_resolution_clock::now();                                              \
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>( end - start ).count() / 1E9; \
        PC3::TimeIt::addTime(name, duration);                                                              \
    }
    