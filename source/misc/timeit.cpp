#include "misc/timeit.hpp"

std::map<std::string, std::vector<double>> times;
std::map<std::string, double> times_total;

double PHOENIX::TimeIt::get( std::string name ) {
    if ( times[name].size() == 0 )
        return 0;
    return times[name].back();
}

double PHOENIX::TimeIt::totalRuntime() {
    double total = 0;
    for ( const auto& [key, value] : times_total ) {
        total += value;
    }
    return total;
}

void PHOENIX::TimeIt::toFile( std::ofstream& file ) {
    file << "iteration ";
    for ( const auto& [key, total] : times ) {
        file << key << " ";
    }
    file << std::endl;
    for ( auto index = 0; index < times.begin()->second.size(); index++ ) {
        file << index << " ";
        for ( const auto& [key, value] : times ) {
            file << value[index] << " ";
        }
        file << std::endl;
    }
}

void PHOENIX::TimeIt::addTime( const std::string& name, double duration ) {
    if ( not times.count( name ) )
        times_total[name] = 0;
    times[name].emplace_back( duration );
    times_total[name] += duration;
}

void PHOENIX::TimeIt::clear() {
    times.clear();
    times_total.clear();
}

std::map<std::string, std::vector<double>>& PHOENIX::TimeIt::getTimes() {
    return times;
}

std::map<std::string, double>& PHOENIX::TimeIt::getTimesTotal() {
    return times_total;
}