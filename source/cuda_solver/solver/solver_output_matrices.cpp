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
    auto header_information = PC3::FileHandler::Header( system.p.L_x * (end_x-start_x)/system.p.N_x, system.p.L_y* (end_y-start_y)/system.p.N_y, system.p.dx, system.p.dy, system.p.t );
    auto fft_header_information = PC3::FileHandler::Header( -1.0* (end_x-start_x)/system.p.N_x, -1.0* (end_y-start_y)/system.p.N_y, 2.0 / system.p.N_x, 2.0 / system.p.N_y, system.p.t );
    auto res = std::async( std::launch::async, [&]() {
        std::lock_guard<std::mutex> lock( mtx );
#pragma omp parallel for
        for ( int i = 0; i < fileoutputkeys.size(); i++ ) {
            auto key = fileoutputkeys[i];
            if ( key == "wavefunction_plus" and system.doOutput( "wavefunction", "psi", "wavefunction_plus", "psi_plus", "plus", "wf", "mat", "all", "initial", "initial_plus" ) )
                filehandler.outputMatrixToFile( matrix.wavefunction_plus.getHostPtr(), start_x, end_x, start_y, end_y, system.p.N_x, system.p.N_y, increment, header_information, prefix + key + suffix );
            if ( key == "reservoir_plus" and system.doOutput( "mat", "reservoir", "n", "reservoir_plus", "n_plus", "plus", "rv", "mat", "all" ) )
                filehandler.outputMatrixToFile( matrix.reservoir_plus.getHostPtr(), start_x, end_x, start_y, end_y, system.p.N_x, system.p.N_y, increment, header_information, prefix + key + suffix );
            if ( system.fft_every < system.t_max and key == "fft_plus" and system.doOutput( "fft_mask", "fft", "fft_plus", "plus", "mat", "all" ) )
                filehandler.outputMatrixToFile( matrix.fft_plus.getHostPtr(), start_x, end_x, start_y, end_y, system.p.N_x, system.p.N_y, increment, fft_header_information, prefix + key + suffix );
            // Guard when not useing TE/TM splitting
            if ( not system.use_twin_mode )
                continue;
            if ( key == "wavefunction_minus" and system.doOutput( "wavefunction", "psi", "wavefunction_minus", "psi_minus", "plus", "wf", "mat", "all", "initial", "initial_minus" ) )
                filehandler.outputMatrixToFile( matrix.wavefunction_minus.getHostPtr(), start_x, end_x, start_y, end_y, system.p.N_x, system.p.N_y, increment, header_information, prefix + key + suffix );
            if ( key == "reservoir_minus" and system.doOutput( "reservoir", "n", "reservoir_minus", "n_minus", "plus", "rv", "mat", "all" ) )
                filehandler.outputMatrixToFile( matrix.reservoir_minus.getHostPtr(), start_x, end_x, start_y, end_y, system.p.N_x, system.p.N_y, increment, header_information, prefix + key + suffix );
            if ( system.fft_every < system.t_max and key == "fft_minus" and system.doOutput( "fft_mask", "fft", "fft_minus", "plus", "mat", "all" ) )
                filehandler.outputMatrixToFile( matrix.fft_minus.getHostPtr(), start_x, end_x, start_y, end_y, system.p.N_x, system.p.N_y, increment, fft_header_information, prefix + key + suffix );
        }
    } );
}

void PC3::Solver::outputInitialMatrices() {
    std::cout << "--------------------------- Outputting Initial Matrices ---------------------------" << std::endl;
    auto header_information = PC3::FileHandler::Header(system.p.L_x, system.p.L_y, system.p.dx, system.p.dy, system.p.t);
    // Output Matrices to file
    if ( system.doOutput( "all", "mat", "initial_plus", "initial" ) )
        system.filehandler.outputMatrixToFile( matrix.initial_state_plus.getHostPtr(), system.p.N_x, system.p.N_y, header_information, "initial_condition_plus" );
    if ( system.doOutput( "all", "mat", "pump_plus", "pump" ) )
        for (int i = 0; i < system.pump.groupSize(); i++) {
            auto osc_header_information = PC3::FileHandler::Header(system.p.L_x, system.p.L_y, system.p.dx, system.p.dy, system.p.t, system.pump.t0[i], system.pump.freq[i], system.pump.sigma[i]);
            std::string suffix = i > 0 ? "_" + std::to_string(i) : "";
            system.filehandler.outputMatrixToFile( matrix.pump_plus.getHostPtr()+i*system.p.N2, system.p.N_x, system.p.N_y, osc_header_information, "pump_plus" + suffix );
        }
    if ( system.doOutput( "all", "mat", "pulse_plus", "pulse" ) )
        for (int i = 0; i < system.pulse.groupSize(); i++) {
            auto osc_header_information = PC3::FileHandler::Header(system.p.L_x, system.p.L_y, system.p.dx, system.p.dy, system.p.t, system.pulse.t0[i], system.pulse.freq[i], system.pulse.sigma[i]);
            std::string suffix = i > 0 ? "_" + std::to_string(i) : "";
            system.filehandler.outputMatrixToFile( matrix.pulse_plus.getHostPtr()+i*system.p.N2, system.p.N_x, system.p.N_y, osc_header_information, "pulse_plus" + suffix );
        }
    if ( system.doOutput( "all", "mat", "potential_plus", "potential" ) )
        for (int i = 0; i < system.potential.groupSize(); i++) {
            auto osc_header_information = PC3::FileHandler::Header(system.p.L_x, system.p.L_y, system.p.dx, system.p.dy, system.p.t, system.potential.t0[i], system.potential.freq[i], system.potential.sigma[i]);
            std::string suffix = i > 0 ? "_" + std::to_string(i) : "";
            system.filehandler.outputMatrixToFile( matrix.potential_plus.getHostPtr()+i*system.p.N2, system.p.N_x, system.p.N_y, osc_header_information, "potential_plus" + suffix );
        }
    if ( system.doOutput( "all", "mat", "fftplus", "fft" ) )
        system.filehandler.outputMatrixToFile( matrix.fft_mask_plus.getHostPtr(), system.p.N_x, system.p.N_y, header_information, "fft_mask_plus" );
    
    if (not system.use_twin_mode )
        return;

    if ( system.doOutput( "all", "mat", "initial_minus", "initial" ) )
        system.filehandler.outputMatrixToFile( matrix.initial_state_minus.getHostPtr(), system.p.N_x, system.p.N_y, header_information, "initial_condition_minus" );
    if ( system.doOutput( "all", "mat", "pump_minus", "pump" ) )
        for (int i = 0; i < system.pump.groupSize(); i++) {
            auto osc_header_information = PC3::FileHandler::Header(system.p.L_x, system.p.L_y, system.p.dx, system.p.dy, system.p.t, system.pump.t0[i], system.pump.freq[i], system.pump.sigma[i]);
            std::string suffix = i > 0 ? "_" + std::to_string(i) : "";
            system.filehandler.outputMatrixToFile( matrix.pump_minus.getHostPtr()+i*system.p.N2, system.p.N_x, system.p.N_y, osc_header_information, "pump_minus" + suffix );
        }
    if ( system.doOutput( "all", "mat", "pulse_minus", "pulse" ) )
        for (int i = 0; i < system.pulse.groupSize(); i++) {
            auto osc_header_information = PC3::FileHandler::Header(system.p.L_x, system.p.L_y, system.p.dx, system.p.dy, system.p.t, system.pulse.t0[i], system.pulse.freq[i], system.pulse.sigma[i]);
            std::string suffix = i > 0 ? "_" + std::to_string(i) : "";
            system.filehandler.outputMatrixToFile( matrix.pulse_minus.getHostPtr()+i*system.p.N2, system.p.N_x, system.p.N_y, osc_header_information, "pulse_minus" + suffix );
        }
    if ( system.doOutput( "all", "mat", "potential_minus", "potential" ) )
        for (int i = 0; i < system.potential.groupSize(); i++) {
            auto osc_header_information = PC3::FileHandler::Header(system.p.L_x, system.p.L_y, system.p.dx, system.p.dy, system.p.t, system.potential.t0[i], system.potential.freq[i], system.potential.sigma[i]);
            std::string suffix = i > 0 ? "_" + std::to_string(i) : "";
            system.filehandler.outputMatrixToFile( matrix.potential_minus.getHostPtr()+i*system.p.N2, system.p.N_x, system.p.N_y, osc_header_information, "potential_minus" + suffix );
        }
    if ( system.doOutput( "all", "mat", "fftminus", "fft" ) )
        system.filehandler.outputMatrixToFile( matrix.fft_mask_minus.getHostPtr(), system.p.N_x, system.p.N_y, header_information, "fft_mask_minus" );
}