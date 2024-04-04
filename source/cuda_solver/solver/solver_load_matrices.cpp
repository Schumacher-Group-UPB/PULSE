#include <vector>
#include <string>
#include "solver/gpu_solver.hpp"
#include "cuda/cuda_macro.cuh"

std::vector<std::string> _split_string_at(const std::string& input, std::string split_at) {
    std::vector<std::string> ret;
    std::string::size_type pos = 0;
    std::string::size_type prev = 0;
    while ((pos = input.find(split_at, prev)) != std::string::npos) {
        const auto substr = input.substr(prev, pos - prev);
        ret.emplace_back(substr);
        prev = pos + 1;
    }
    const auto substr = input.substr(prev, pos - prev);
    ret.emplace_back(substr);
    return ret;
}

void PC3::Solver::loadMatrices() {
    if ( filehandler.loadPath.size() < 1 or system.input_keys.empty() )
        return;
    std::cout << "-------------------------------- Loading Matrices ---------------------------------" << std::endl;
    std::cout << "Load Path: " << filehandler.loadPath << std::endl;

    // Cleanup Path
    if ( filehandler.loadPath.back() != '/' )
        filehandler.loadPath += "/";

    // All possible files that could potentially be loaded
    std::vector<std::string> fileinputkeys = { 
        "wavefunction_plus", "wavefunction_minus", // Wavefunction matrices
        "reservoir_plus", "reservoir_minus", // Reservoir Matrices
        "fft_mask_plus", "fft_mask_minus", // FFT Masks
        "pump_plus", "pump_minus", // Pumps
        "pulse_plus", "pulse_minus", // Pulses
        "potential_plus", "potential_minus", // Potentials
        "initial_condition_plus", "initial_condition_minus" // Initial Conditions
    };

    // Load all matrices from fileinputkeys that overlap with system.input_keys
//#pragma omp parallel for
    for ( auto i = 0; i < fileinputkeys.size(); i++ ) {
        if ( fileinputkeys[i] == "wavefunction_plus" and system.doInput( "wavefunction", "psi", "wavefunction_plus", "psi_plus", "plus", "wf", "mat", "all", "initial", "initial_plus" ) ) {
            filehandler.loadMatrixFromFile( filehandler.loadPath + fileinputkeys[i] + ".txt", host.wavefunction_plus.get() );
            // The initial condition is the same as the wavefunction, so we can just copy it over
            filehandler.loadMatrixFromFile( filehandler.loadPath + fileinputkeys[i] + ".txt", host.initial_state_plus.get() ); 
        }
        else if ( fileinputkeys[i] == "reservoir_plus" and system.doInput( "reservoir", "n", "reservoir_plus", "n_plus", "plus", "rv", "mat", "all" ) )
            filehandler.loadMatrixFromFile( filehandler.loadPath + fileinputkeys[i] + ".txt", host.reservoir_plus.get() );
        else if ( fileinputkeys[i] == "fft_mask_plus" and system.doInput( "fft_mask", "fft", "fft_plus", "plus", "mat", "all" ) )
            filehandler.loadMatrixFromFile( filehandler.loadPath + fileinputkeys[i] + ".txt", host.fft_mask_plus.get() );
        else if ( fileinputkeys[i] == "pump_plus" and system.doInput( "pump", "pump_plus", "plus", "mat", "all" ) )
            filehandler.loadMatrixFromFile( filehandler.loadPath + fileinputkeys[i] + ".txt", host.pump_plus.get() );
        else if ( fileinputkeys[i] == "pulse_plus" and system.doInput( "pulse", "pulse_plus", "plus", "mat", "all" ) )
            filehandler.loadMatrixFromFile( filehandler.loadPath + fileinputkeys[i] + ".txt", host.pulse_plus.get() );
        else if ( fileinputkeys[i] == "potential_plus" and system.doInput( "potential", "potential_plus", "plus", "mat", "all" ) )
            filehandler.loadMatrixFromFile( filehandler.loadPath + fileinputkeys[i] + ".txt", host.potential_plus.get() );
        else if ( fileinputkeys[i] == "initial_condition_plus" and system.doInput( "init", "initial_condition", "init_plus", "initial_condition_plus", "plus", "mat", "all" ) ) 
            filehandler.loadMatrixFromFile( filehandler.loadPath + fileinputkeys[i] + ".txt", host.initial_state_plus.get() );
        
        // Guard when not useing TE/TM splitting
        if (not system.use_twin_mode)
            continue;

        if ( fileinputkeys[i] == "wavefunction_minus" and system.doInput( "wavefunction", "psi", "wavefunction_minus", "psi_minus", "plus", "wf", "mat", "all", "initial", "initial_minus" ) ){
            filehandler.loadMatrixFromFile( filehandler.loadPath + fileinputkeys[i] + ".txt", host.wavefunction_minus.get() );
            // The initial condition is the same as the wavefunction, so we can just copy it over
            filehandler.loadMatrixFromFile( filehandler.loadPath + fileinputkeys[i] + ".txt", host.initial_state_minus.get() ); 
        }
        else if ( fileinputkeys[i] == "reservoir_minus" and system.doInput( "reservoir", "n", "reservoir_minus", "n_minus", "plus", "rv", "mat", "all" ) )
            filehandler.loadMatrixFromFile( filehandler.loadPath + fileinputkeys[i] + ".txt", host.reservoir_minus.get() );
        else if ( fileinputkeys[i] == "fft_mask_minus" and system.doInput( "fft_mask", "fft", "fft_minus", "plus", "mat", "all" ) )
            filehandler.loadMatrixFromFile( filehandler.loadPath + fileinputkeys[i] + ".txt", host.fft_mask_minus.get() );
        else if ( fileinputkeys[i] == "pump_minus" and system.doInput( "pump", "pump_minus", "plus", "mat", "all" ) )
            filehandler.loadMatrixFromFile( filehandler.loadPath + fileinputkeys[i] + ".txt", host.pump_minus.get() );
        else if ( fileinputkeys[i] == "pulse_minus" and system.doInput( "pulse", "pulse_minus", "plus", "mat", "all" ) )
            filehandler.loadMatrixFromFile( filehandler.loadPath + fileinputkeys[i] + ".txt", host.pulse_minus.get() );
        else if ( fileinputkeys[i] == "potential_minus" and system.doInput( "potential", "potential_minus", "plus", "mat", "all" ) )
            filehandler.loadMatrixFromFile( filehandler.loadPath + fileinputkeys[i] + ".txt", host.potential_minus.get() );
        else if ( fileinputkeys[i] == "initial_condition_minus" and system.doInput( "init", "initial_condition", "init_minus", "initial_condition_minus", "plus", "mat", "all" ) )
            filehandler.loadMatrixFromFile( filehandler.loadPath + fileinputkeys[i] + ".txt", host.initial_state_minus.get() );
    }
}