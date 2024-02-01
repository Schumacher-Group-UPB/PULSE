#include <vector>
#include <string>
#include <future>
#include <mutex>
#include "solver/gpu_solver.hpp"
#include "cuda/cuda_macro.cuh"

// TODO: to ensure the async call outputs the correct matrices, we need to make sure the lambda takes a *copy* of the host arrays.
// This will be fine, because copying 100MB in memory is much faster than writing to disk.
// For now, use mutex, making the async call not really async if the matrices are too large.
std::mutex mtx;

void PC3::Solver::outputMatrices( const unsigned int start_x, const unsigned int end_x, const unsigned int start_y, const unsigned int end_y, const unsigned int increment, const std::string& suffix, const std::string& prefix ) {
    const static std::vector<std::string> fileoutputkeys = { "wavefunction_plus", "wavefunction_minus", "reservoir_plus", "reservoir_minus", "fft_plus", "fft_minus" };
    auto header_information = PC3::FileHandler::Header( system.s_L_x * (end_x-start_x)/system.s_N_x, system.s_L_y* (end_y-start_y)/system.s_N_y, system.dx, system.dy, system.t );
    auto fft_header_information = PC3::FileHandler::Header( -1.0* (end_x-start_x)/system.s_N_x, -1.0* (end_y-start_y)/system.s_N_y, 2.0 / system.s_N_x, 2.0 / system.s_N_y, system.t );
    auto res = std::async( std::launch::async, [&]() {
        std::lock_guard<std::mutex> lock( mtx );
#pragma omp parallel for
        for ( int i = 0; i < fileoutputkeys.size(); i++ ) {
            auto key = fileoutputkeys[i];
            if ( key == "wavefunction_plus" and system.doOutput( "wavefunction", "psi", "wavefunction_plus", "psi_plus", "plus", "wf", "mat", "all", "initial", "initial_plus" ) )
                filehandler.outputMatrixToFile( host.wavefunction_plus.get(), start_x, end_x, start_y, end_y, system.s_N_x, system.s_N_y, increment, header_information, prefix + key + suffix );
            if ( key == "reservoir_plus" and system.doOutput( "mat", "reservoir", "n", "reservoir_plus", "n_plus", "plus", "rv", "mat", "all" ) )
                filehandler.outputMatrixToFile( host.reservoir_plus.get(), start_x, end_x, start_y, end_y, system.s_N_x, system.s_N_y, increment, header_information, prefix + key + suffix );
            if ( system.fft_every < system.t_max and key == "fft_plus" and system.doOutput( "fft_mask", "fft", "fft_plus", "plus", "mat", "all" ) )
                filehandler.outputMatrixToFile( host.fft_plus.get(), start_x, end_x, start_y, end_y, system.s_N_x, system.s_N_y, increment, fft_header_information, prefix + key + suffix );
            // Guard when not useing TE/TM splitting
            if ( not system.use_te_tm_splitting )
                continue;
            if ( key == "wavefunction_minus" and system.doOutput( "wavefunction", "psi", "wavefunction_minus", "psi_minus", "plus", "wf", "mat", "all", "initial", "initial_minus" ) )
                filehandler.outputMatrixToFile( host.wavefunction_minus.get(), start_x, end_x, start_y, end_y, system.s_N_x, system.s_N_y, increment, header_information, prefix + key + suffix );
            if ( key == "reservoir_minus" and system.doOutput( "reservoir", "n", "reservoir_minus", "n_minus", "plus", "rv", "mat", "all" ) )
                filehandler.outputMatrixToFile( host.reservoir_minus.get(), start_x, end_x, start_y, end_y, system.s_N_x, system.s_N_y, increment, header_information, prefix + key + suffix );
            if ( system.fft_every < system.t_max and key == "fft_minus" and system.doOutput( "fft_mask", "fft", "fft_minus", "plus", "mat", "all" ) )
                filehandler.outputMatrixToFile( host.fft_minus.get(), start_x, end_x, start_y, end_y, system.s_N_x, system.s_N_y, increment, fft_header_information, prefix + key + suffix );
        }
    } );
}