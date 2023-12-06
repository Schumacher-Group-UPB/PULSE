#include "misc/timeit.hpp"

std::map<std::string, std::vector<double>> times;
std::map<std::string, double> times_total;

double PC3::TimeIt::get( std::string name ) {
    if ( times[name].size() == 0 )
        return 0;
    return times[name].back();
}

double PC3::TimeIt::totalRuntime() {
    double total = 0;
    for ( const auto& [key, value] : times_total ) {
        total += value;
    }
    return total;
}

void PC3::TimeIt::toFile( std::ofstream& file ) {
    file << "iteration ";
    for ( const auto& [key, total] : times ) {
        file << key << " ";
    }
    file << std::endl;
    for (auto index = 0; index < times.begin()->second.size(); index++) {
        file << index << " ";
        for ( const auto& [key, value] : times ) {
            file << value[index] << " ";
        }
        file << std::endl;
    }
}

void PC3::TimeIt::addTime(const std::string& name, double duration) {
    if ( not times.contains(name) )
        times_total[name] = 0;
    times[name].emplace_back( duration );
    times_total[name] += duration;
}

void PC3::TimeIt::clear() {
    times.clear();
    times_total.clear();
}

std::map<std::string, std::vector<double>>& PC3::TimeIt::getTimes() {
    return times;
}

std::map<std::string, double>& PC3::TimeIt::getTimesTotal() {
    return times_total;
}