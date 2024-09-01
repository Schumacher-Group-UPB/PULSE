#pragma once
#include "cuda/typedef.cuh"
#include "cuda/cuda_matrix.cuh"

namespace PC3 {

#define MATRIX_LIST // \  <--- Backslash at the end of every but the last line!
    // DEFINE_MATRIX(Type::complex, custom_matrix ) // \  <--- Backslash at the end of every but the last line!
    /////////////////////////////
    // Add your matrices here. //
    // Make sure to end each   //
    // but the last line with  //
    // a backslash!            //
    /////////////////////////////

struct MatrixContainer {

    // Cache triggers
    bool use_twin_mode, use_fft, use_stochastic;

    // Wavefunction and Reservoir Matrices.
    PC3::CUDAMatrix<Type::complex> wavefunction_plus, wavefunction_minus, reservoir_plus, reservoir_minus;
    // Corresponding Buffer Matrices
    PC3::CUDAMatrix<Type::complex> buffer_wavefunction_plus, buffer_wavefunction_minus, buffer_reservoir_plus, buffer_reservoir_minus;
    // Corresponding initial States. These are simple host vectors, not CUDAMatrices.
    PC3::Type::host_vector<Type::complex> initial_state_plus, initial_state_minus, initial_reservoir_plus, initial_reservoir_minus;

    // Pump, Pulse and Potential Matrices. These are vectors of CUDAMatrices.
    std::vector<PC3::CUDAMatrix<Type::complex>> pump_plus, pump_minus, pulse_plus, pulse_minus, potential_plus, potential_minus;
    // Device Vectors holding the respective device pointers for the matrices. This is a vector of device vectors, because we need multiple instances of the pointer struct.
    std::vector<Type::device_vector<Type::complex*>> pump_plus_ptrs, pump_minus_ptrs, pulse_plus_ptrs, pulse_minus_ptrs, potential_plus_ptrs, potential_minus_ptrs;

    // FFT Matrices. These are simple device vectors, not CUDAMatrices.
    PC3::Type::device_vector<Type::complex> fft_plus, fft_minus;
    PC3::Type::device_vector<Type::real> fft_mask_plus, fft_mask_minus;

    // Random Number generator and buffer. We only need a single random number matrix of size subgrid_x*subgrid_y
    // These can also be simple device vectors, as subgridding is not required here
    PC3::Type::device_vector<Type::complex> random_number;
    PC3::Type::device_vector<Type::cuda_random_state> random_state;
    
    // Snapshot matrices. These are simple host vectors, not CUDAMatrices.
    PC3::Type::host_vector<Type::complex> snapshot_wavefunction_plus, snapshot_wavefunction_minus, snapshot_reservoir_plus, snapshot_reservoir_minus;
    
    // RK45 Style Error Matrix. 
    PC3::CUDAMatrix<Type::complex> rk_error;

    // K Matrices. These are vectors of CUDAMatrices.
    std::vector<PC3::CUDAMatrix<Type::complex>> k_wavefunction_plus, k_wavefunction_minus, k_reservoir_plus, k_reservoir_minus;
    // Device Vectors holding the respective device pointers for the matrices
    std::vector<Type::device_vector<Type::complex*>> k_wavefunction_plus_ptrs, k_wavefunction_minus_ptrs, k_reservoir_plus_ptrs, k_reservoir_minus_ptrs;

    // Halo Map
    PC3::Type::device_vector<int> halo_map;

    // User Defined Matrices
    #define DEFINE_MATRIX(type, name) PC3::CUDAMatrix<type> name;
    MATRIX_LIST
    #undef X

    // Empty Constructor
    MatrixContainer() = default;

    // Construction Chain. The Host Matrix is always constructed (who carese about RAM right?) and the device matrix is constructed if the condition is met.
    void constructAll( const int N_x, const int N_y, bool use_twin_mode, bool use_fft, bool use_stochastic, int k_max, 
                       const int n_pulses_plus, const int n_pumps_plus, const int n_potentials_plus,
                       const int n_pulses_minus, const int n_pumps_minus, const int n_potentials_minus,
                       const int subgrids_x, const int subgrids_y, const int halo_size ) {
        
        // Cache triggers
        this->use_twin_mode = use_twin_mode;
        this->use_fft = use_fft;
        this->use_stochastic = use_stochastic;
        
        // MARK: Plus Components
        // ======================================================================================================== //
        // =------------------------- Construct Plus Components of the matrices ----------------------------------= //
        // ======================================================================================================== //

        // Wavefunction and Reservoir Matrices
        wavefunction_plus.construct( N_x, N_y, subgrids_x, subgrids_y, halo_size, "wavefunction_plus");
        reservoir_plus.construct( N_x, N_y, subgrids_x, subgrids_y, halo_size, "reservoir_plus");
        buffer_wavefunction_plus.construct( N_x, N_y, subgrids_x, subgrids_y, halo_size, "buffer_wavefunction_plus");
        buffer_reservoir_plus.construct( N_x, N_y, subgrids_x, subgrids_y, halo_size, "buffer_reservoir_plus");
        initial_state_plus = PC3::Type::device_vector<Type::complex>(N_x*N_y);
        initial_reservoir_plus = PC3::Type::device_vector<Type::complex>(N_x*N_y);

        // Pump, Pulse and Potential Matrices
        pump_plus.resize(n_pumps_plus);
        pump_plus_ptrs.resize(subgrids_x*subgrids_y);
        for (int i = 0; i < n_pumps_plus; i++) {
            pump_plus[i].construct( N_x, N_y, subgrids_x, subgrids_y, halo_size, "pump_plus_"+std::to_string(i));
        }
        pulse_plus.resize(n_pulses_plus);
        pulse_plus_ptrs.resize(subgrids_x*subgrids_y);
        for (int i = 0; i < n_pulses_plus; i++) {
            pulse_plus[i].construct( N_x, N_y, subgrids_x, subgrids_y, halo_size, "pulse_plus_"+std::to_string(i));
        }
        potential_plus.resize(n_potentials_plus);
        potential_plus_ptrs.resize(subgrids_x*subgrids_y);
        for (int i = 0; i < n_potentials_plus; i++) {
            potential_plus[i].construct( N_x, N_y, subgrids_x, subgrids_y, halo_size, "potential_plus_"+std::to_string(i));
        }

        // We then require a device vector of pointers to the respective subgrids of the matrix. We also require this vector for each of the subgrids, resulting in N_subgrids*N_envelope pointers.
        for (int subgrid = 0; subgrid < subgrids_x*subgrids_y; subgrid++) {
            pump_plus_ptrs[subgrid] = PC3::Type::device_vector<Type::complex*>(n_pumps_plus);
            pulse_plus_ptrs[subgrid] = PC3::Type::device_vector<Type::complex*>(n_pulses_plus);
            potential_plus_ptrs[subgrid] = PC3::Type::device_vector<Type::complex*>(n_potentials_plus);
            // We can also already fill the device pointers for the matrices, as we know the subgrid size and the number of envelopes.
            for (auto i = 0; i < pump_plus.size(); i++)
                pump_plus_ptrs[subgrid][i] = pump_plus[i].getDevicePtr(subgrid);
            for (auto i = 0; i < pulse_plus.size(); i++) 
                pulse_plus_ptrs[subgrid][i] = pulse_plus[i].getDevicePtr(subgrid);
            for (auto i = 0; i < potential_plus.size(); i++)
                potential_plus_ptrs[subgrid][i] = potential_plus[i].getDevicePtr(subgrid);
        }

        // K Matrices
        k_wavefunction_plus.resize(k_max);
        k_reservoir_plus.resize(k_max);
        k_wavefunction_plus_ptrs.resize(subgrids_x*subgrids_y);
        k_reservoir_plus_ptrs.resize(subgrids_x*subgrids_y);
        for (int i = 0; i < k_max; i++) {
            k_wavefunction_plus[i].construct( N_x, N_y, subgrids_x, subgrids_y, halo_size, "k_wavefunction_plus_"+std::to_string(i));
            k_reservoir_plus[i].construct( N_x, N_y, subgrids_x, subgrids_y, halo_size, "k_reservoir_plus_"+std::to_string(i));
        }
        // We also need to initialize the CUDAMatrices for the minus components, even if we don't use them. They will only get constructed if the twin mode is enabled.
        k_wavefunction_minus.resize(k_max);
        k_reservoir_minus.resize(k_max);
        k_wavefunction_minus_ptrs.resize(subgrids_x*subgrids_y);
        k_reservoir_minus_ptrs.resize(subgrids_x*subgrids_y);
        
        // Same for the k matrices. We require a device vector of pointers to the respective subgrids of the matrix. We also require this vector for each of the subgrids, resulting in N_subgrids*k_max pointers.
        for (int subgrid = 0; subgrid < subgrids_x*subgrids_y; subgrid++) {
            k_wavefunction_plus_ptrs[subgrid] = PC3::Type::device_vector<Type::complex*>(k_max);
            k_reservoir_plus_ptrs[subgrid] = PC3::Type::device_vector<Type::complex*>(k_max);
            // We can also already fill the device pointers for the matrices, as we know the subgrid size and the number of envelopes.
            for (auto i = 0; i < k_wavefunction_plus.size(); i++) {
                k_wavefunction_plus_ptrs[subgrid][i] = k_wavefunction_plus[i].getDevicePtr(subgrid);
                k_reservoir_plus_ptrs[subgrid][i] = k_reservoir_plus[i].getDevicePtr(subgrid);
            }
        }

        // FFT Matrices
        if (use_fft) {
            fft_plus = PC3::Type::device_vector<Type::complex>(N_x*N_y);
            fft_mask_plus = PC3::Type::device_vector<Type::real>(N_x*N_y);
        }

        // MARK: Independent Components
        // ======================================================================================================== //
        // =------------------------------------ Independent Components ------------------------------------------= //
        // ======================================================================================================== //

        // Random Number generator and buffer
        if (use_stochastic) {
            const Type::uint subgrid_N = N_x*N_y / subgrids_x / subgrids_y;
            random_number = PC3::Type::device_vector<Type::complex>(subgrid_N);
            random_state = PC3::Type::device_vector<Type::cuda_random_state>(subgrid_N);
        }

        // RK Error Matrix.
        // TODO: We could define some kinda of ".define()" method for the matrices. Then, when the hostPtr is called, the matrix is constructed if it is not already constructed.
        // Same with the device pointer. This way, we can avoid constructing matrices that are not used.
        rk_error.construct( N_x, N_y, subgrids_x, subgrids_y, halo_size, "rk_error");

        // Construct the halo map. 6*total halo points because we have 6 coordinates for each point
        const Type::uint total_halo_points = (N_x+2*halo_size)*(N_y+2*halo_size) - N_x*N_y;
        halo_map = PC3::Type::device_vector<int>(total_halo_points*6);
        
        // User defined matrices
        #define DEFINE_MATRIX(type, name) \
            name.construct( N_x, N_y, subgrids_x, subgrids_y, halo_size, #name);
        MATRIX_LIST
        #undef X

        if (not use_twin_mode)
            return;

        // MARK: Minus Components
        // ======================================================================================================== //
        // =------------------------- Construct Minus Components of the matrices ---------------------------------= //
        // ======================================================================================================== //
        
        // Wavefunction and Reservoir Matrices
        wavefunction_minus.construct( N_x, N_y, subgrids_x, subgrids_y, halo_size, "wavefunction_minus");
        reservoir_minus.construct( N_x, N_y, subgrids_x, subgrids_y, halo_size, "reservoir_minus");
        buffer_wavefunction_minus.construct( N_x, N_y, subgrids_x, subgrids_y, halo_size, "buffer_wavefunction_minus");
        buffer_reservoir_minus.construct( N_x, N_y, subgrids_x, subgrids_y, halo_size, "buffer_reservoir_minus");
        initial_state_minus = PC3::Type::device_vector<Type::complex>(N_x*N_y);
        initial_reservoir_minus = PC3::Type::device_vector<Type::complex>(N_x*N_y);

        // Pump, Pulse and Potential Matrices
        pump_minus.resize(n_pumps_minus);
        pump_minus_ptrs.resize(subgrids_x*subgrids_y);
        for (int i = 0; i < n_pumps_minus; i++) {
            pump_minus[i].construct( N_x, N_y, subgrids_x, subgrids_y, halo_size, "pump_minus_"+std::to_string(i));
        }
        pulse_minus.resize(n_pulses_minus);
        pulse_minus_ptrs.resize(subgrids_x*subgrids_y);
        for (int i = 0; i < n_pulses_minus; i++) {
            pulse_minus[i].construct( N_x, N_y, subgrids_x, subgrids_y, halo_size, "pulse_minus_"+std::to_string(i));
        }
        potential_minus.resize(n_potentials_minus);
        potential_minus_ptrs.resize(subgrids_x*subgrids_y);
        for (int i = 0; i < n_potentials_minus; i++) {
            potential_minus[i].construct( N_x, N_y, subgrids_x, subgrids_y, halo_size, "potential_minus_"+std::to_string(i));
        }

        // We then require a device vector of pointers to the respective subgrids of the matrix. We also require this vector for each of the subgrids, resulting in N_subgrids*N_envelope pointers.
        for (int subgrid = 0; subgrid < subgrids_x*subgrids_y; subgrid++) {
            pump_minus_ptrs[subgrid] = PC3::Type::device_vector<Type::complex*>(n_pumps_minus);
            pulse_minus_ptrs[subgrid] = PC3::Type::device_vector<Type::complex*>(n_pulses_minus);
            potential_minus_ptrs[subgrid] = PC3::Type::device_vector<Type::complex*>(n_potentials_minus);
            // We can also already fill the device pointers for the matrices, as we know the subgrid size and the number of envelopes.
            for (auto i = 0; i < pump_minus.size(); i++)
                pump_minus_ptrs[subgrid][i] = pump_minus[i].getDevicePtr(subgrid);
            for (auto i = 0; i < pulse_minus.size(); i++)
                pulse_minus_ptrs[subgrid][i] = pulse_minus[i].getDevicePtr(subgrid);
            for (auto i = 0; i < potential_minus.size(); i++)
                potential_minus_ptrs[subgrid][i] = potential_minus[i].getDevicePtr(subgrid);
        }

        // K Matrices
        for (int i = 0; i < k_max; i++) {
            k_wavefunction_minus[i].construct( N_x, N_y, subgrids_x, subgrids_y, halo_size, "k_wavefunction_minus_"+std::to_string(i));
            k_reservoir_minus[i].construct( N_x, N_y, subgrids_x, subgrids_y, halo_size, "k_reservoir_minus_"+std::to_string(i));
        }
        // Same for the k matrices. We require a device vector of pointers to the respective subgrids of the matrix. We also require this vector for each of the subgrids, resulting in N_subgrids*k_max pointers.
        for (int subgrid = 0; subgrid < subgrids_x*subgrids_y; subgrid++) {
            k_wavefunction_minus_ptrs[subgrid] = PC3::Type::device_vector<Type::complex*>(k_max);
            k_reservoir_minus_ptrs[subgrid] = PC3::Type::device_vector<Type::complex*>(k_max);
            // We can also already fill the device pointers for the matrices, as we know the subgrid size and the number of envelopes.
            for (auto i = 0; i < k_wavefunction_minus.size(); i++) {
                k_wavefunction_minus_ptrs[subgrid][i] = k_wavefunction_minus[i].getDevicePtr(subgrid);
                k_reservoir_minus_ptrs[subgrid][i] = k_reservoir_minus[i].getDevicePtr(subgrid);
            }
        }

        // FFT Matrices
        if (use_fft) {
            fft_minus = PC3::Type::device_vector<Type::complex>(N_x*N_y);
            fft_mask_minus = PC3::Type::device_vector<Type::real>(N_x*N_y);
        }
     }

    struct Pointers {

        // Wavefunction and Reservoir Matrices
        Type::complex* wavefunction_plus = nullptr;
        Type::complex* wavefunction_minus = nullptr;
        Type::complex* reservoir_plus = nullptr;
        Type::complex* reservoir_minus = nullptr;
        // Corresponding Buffer Matrices
        Type::complex* buffer_wavefunction_plus = nullptr;
        Type::complex* buffer_wavefunction_minus = nullptr;
        Type::complex* buffer_reservoir_plus = nullptr;
        Type::complex* buffer_reservoir_minus = nullptr;

        // Pump, Pulse and Potential Matrices
        Type::complex** pump_plus = nullptr;
        Type::complex** pump_minus = nullptr;
        Type::complex** pulse_plus = nullptr;
        Type::complex** pulse_minus = nullptr;
        Type::complex** potential_plus = nullptr;
        Type::complex** potential_minus = nullptr;

        // K Matrices
        Type::complex* k1_wavefunction_plus = nullptr;
        Type::complex* k1_wavefunction_minus = nullptr;
        Type::complex* k1_reservoir_plus = nullptr;
        Type::complex* k1_reservoir_minus = nullptr;
        Type::complex* k2_wavefunction_plus = nullptr;
        Type::complex* k2_wavefunction_minus = nullptr;
        Type::complex* k2_reservoir_plus = nullptr;
        Type::complex* k2_reservoir_minus = nullptr;
        Type::complex* k3_wavefunction_plus = nullptr;
        Type::complex* k3_wavefunction_minus = nullptr;
        Type::complex* k3_reservoir_plus = nullptr;
        Type::complex* k3_reservoir_minus = nullptr;
        Type::complex* k4_wavefunction_plus = nullptr;
        Type::complex* k4_wavefunction_minus = nullptr;
        Type::complex* k4_reservoir_plus = nullptr;
        Type::complex* k4_reservoir_minus = nullptr;
        Type::complex* k5_wavefunction_plus = nullptr;
        Type::complex* k5_wavefunction_minus = nullptr;
        Type::complex* k5_reservoir_plus = nullptr;
        Type::complex* k5_reservoir_minus = nullptr;
        Type::complex* k6_wavefunction_plus = nullptr;
        Type::complex* k6_wavefunction_minus = nullptr;
        Type::complex* k6_reservoir_plus = nullptr;
        Type::complex* k6_reservoir_minus = nullptr;
        Type::complex* k7_wavefunction_plus = nullptr;
        Type::complex* k7_wavefunction_minus = nullptr;
        Type::complex* k7_reservoir_plus = nullptr;
        Type::complex* k7_reservoir_minus = nullptr;
        Type::complex* k8_wavefunction_plus = nullptr;
        Type::complex* k8_wavefunction_minus = nullptr;
        Type::complex* k8_reservoir_plus = nullptr;
        Type::complex* k8_reservoir_minus = nullptr;
        Type::complex* k9_wavefunction_plus = nullptr;
        Type::complex* k9_wavefunction_minus = nullptr;
        Type::complex* k9_reservoir_plus = nullptr;
        Type::complex* k9_reservoir_minus = nullptr;
        Type::complex* k10_wavefunction_plus = nullptr;
        Type::complex* k10_wavefunction_minus = nullptr;
        Type::complex* k10_reservoir_plus = nullptr;
        Type::complex* k10_reservoir_minus = nullptr;


        // FFT Matrices
        Type::complex* fft_plus = nullptr;
        Type::complex* fft_minus = nullptr;
        Type::real* fft_mask_plus = nullptr;
        Type::real* fft_mask_minus = nullptr;

        // Random Number generator and buffer
        Type::complex* random_number = nullptr;
        Type::cuda_random_state* random_state = nullptr;

        // RK Error
        Type::complex* rk_error = nullptr;

        // Halo Map
        int* halo_map = nullptr;
        
        // Custom Components
        #define DEFINE_MATRIX(type, ptrstruct, name, size_scaling, condition_for_construction) \
                type* name = nullptr;
        MATRIX_LIST
        #undef X

        // Nullptr
        std::nullptr_t discard = nullptr;
    };

    Pointers pointers(const Type::uint subgrid) {
        Pointers ptrs;

        // MARK: Plus Component
        // Wavefunction and Reservoir Matrices. Only the Plus components are initialized here. If the twin mode is enabled, the minus components are initialized in the next step.
        // The kernels can then check for nullptr and use the minus components if they are not nullptr.
        ptrs.wavefunction_plus = wavefunction_plus.getDevicePtr(subgrid);
        ptrs.reservoir_plus = reservoir_plus.getDevicePtr(subgrid);
        ptrs.buffer_wavefunction_plus = buffer_wavefunction_plus.getDevicePtr(subgrid);
        ptrs.buffer_reservoir_plus = buffer_reservoir_plus.getDevicePtr(subgrid);

        // Pump, Pulse and Potential Matrices
        ptrs.pump_plus = GET_RAW_PTR( pump_plus_ptrs[subgrid] );
        ptrs.pulse_plus = GET_RAW_PTR( pulse_plus_ptrs[subgrid] );
        ptrs.potential_plus = GET_RAW_PTR( potential_plus_ptrs[subgrid] );

        // K Matrices
        //ptrs.k_wavefunction_plus = GET_RAW_PTR( k_wavefunction_plus_ptrs[subgrid] );
        //ptrs.k_reservoir_plus = GET_RAW_PTR( k_reservoir_plus_ptrs[subgrid] );
        // Hate the game not the player.
        int k_max = k_wavefunction_plus.size();
        ptrs.k1_wavefunction_plus = k_wavefunction_plus[0].getDevicePtr(subgrid);
        ptrs.k1_reservoir_plus = k_reservoir_plus[0].getDevicePtr(subgrid);
        if (k_max > 1) {
            ptrs.k2_wavefunction_plus = k_wavefunction_plus[1].getDevicePtr(subgrid);
            ptrs.k2_reservoir_plus = k_reservoir_plus[1].getDevicePtr(subgrid);
        }
        if (k_max > 2) {
            ptrs.k3_wavefunction_plus = k_wavefunction_plus[2].getDevicePtr(subgrid);
            ptrs.k3_reservoir_plus = k_reservoir_plus[2].getDevicePtr(subgrid);
        }
        if (k_max > 3) {
            ptrs.k4_wavefunction_plus = k_wavefunction_plus[3].getDevicePtr(subgrid);
            ptrs.k4_reservoir_plus = k_reservoir_plus[3].getDevicePtr(subgrid);
        }
        if (k_max > 4) {
            ptrs.k5_wavefunction_plus = k_wavefunction_plus[4].getDevicePtr(subgrid);
            ptrs.k5_reservoir_plus = k_reservoir_plus[4].getDevicePtr(subgrid);
        }
        if (k_max > 5) {
            ptrs.k6_wavefunction_plus = k_wavefunction_plus[5].getDevicePtr(subgrid);
            ptrs.k6_reservoir_plus = k_reservoir_plus[5].getDevicePtr(subgrid);
        }
        if (k_max > 6) {
            ptrs.k7_wavefunction_plus = k_wavefunction_plus[6].getDevicePtr(subgrid);
            ptrs.k7_reservoir_plus = k_reservoir_plus[6].getDevicePtr(subgrid);
        }
        if (k_max > 7) {
            ptrs.k8_wavefunction_plus = k_wavefunction_plus[7].getDevicePtr(subgrid);
            ptrs.k8_reservoir_plus = k_reservoir_plus[7].getDevicePtr(subgrid);
        }
        if (k_max > 8) {
            ptrs.k9_wavefunction_plus = k_wavefunction_plus[8].getDevicePtr(subgrid);
            ptrs.k9_reservoir_plus = k_reservoir_plus[8].getDevicePtr(subgrid);
        }
        if (k_max > 9) {
            ptrs.k10_wavefunction_plus = k_wavefunction_plus[9].getDevicePtr(subgrid);
            ptrs.k10_reservoir_plus = k_reservoir_plus[9].getDevicePtr(subgrid);
        }

        // FFT Matrices
        if (use_fft) {
            ptrs.fft_plus = GET_RAW_PTR( fft_plus );
            ptrs.fft_mask_plus = GET_RAW_PTR( fft_mask_plus );
        }

        // MARK: Independent Components

        if (use_stochastic) {
            ptrs.random_number = GET_RAW_PTR( random_number );
            ptrs.random_state = GET_RAW_PTR( random_state );
        }

        ptrs.rk_error = rk_error.getDevicePtr(subgrid);

        // Halo Map
        ptrs.halo_map = GET_RAW_PTR( halo_map );

        // User Defined Matrices
        #define DEFINE_MATRIX(type, ptrstruct, name, size_scaling, condition_for_construction) \
                ptrs.name = name.getDevicePtr(subgrid);
        MATRIX_LIST
        #undef X

        // MARK: Minus Component
        if (not use_twin_mode)
            return ptrs;

        // Wavefunction and Reservoir Matrices
        ptrs.wavefunction_minus = wavefunction_minus.getDevicePtr(subgrid);
        ptrs.reservoir_minus = reservoir_minus.getDevicePtr(subgrid);
        ptrs.buffer_wavefunction_minus = buffer_wavefunction_minus.getDevicePtr(subgrid);
        ptrs.buffer_reservoir_minus = buffer_reservoir_minus.getDevicePtr(subgrid);

        // Pump, Pulse and Potential Matrices
        ptrs.pump_minus = GET_RAW_PTR( pump_minus_ptrs[subgrid] );
        ptrs.pulse_minus = GET_RAW_PTR( pulse_minus_ptrs[subgrid] );
        ptrs.potential_minus = GET_RAW_PTR( potential_minus_ptrs[subgrid] );

        // K Matrices
        //ptrs.k_wavefunction_minus = GET_RAW_PTR( k_wavefunction_minus_ptrs[subgrid] );
        //ptrs.k_reservoir_minus = GET_RAW_PTR( k_reservoir_minus_ptrs[subgrid] );
        ptrs.k1_wavefunction_minus = k_wavefunction_minus[0].getDevicePtr(subgrid);
        ptrs.k1_reservoir_minus = k_reservoir_minus[0].getDevicePtr(subgrid);
        if (k_max > 1) {
            ptrs.k2_wavefunction_minus = k_wavefunction_minus[1].getDevicePtr(subgrid);
            ptrs.k2_reservoir_minus = k_reservoir_minus[1].getDevicePtr(subgrid);
        }
        if (k_max > 2) {
            ptrs.k3_wavefunction_minus = k_wavefunction_minus[2].getDevicePtr(subgrid);
            ptrs.k3_reservoir_minus = k_reservoir_minus[2].getDevicePtr(subgrid);
        }
        if (k_max > 3) {
            ptrs.k4_wavefunction_minus = k_wavefunction_minus[3].getDevicePtr(subgrid);
            ptrs.k4_reservoir_minus = k_reservoir_minus[3].getDevicePtr(subgrid);
        }
        if (k_max > 4) {
            ptrs.k5_wavefunction_minus = k_wavefunction_minus[4].getDevicePtr(subgrid);
            ptrs.k5_reservoir_minus = k_reservoir_minus[4].getDevicePtr(subgrid);
        }
        if (k_max > 5) {
            ptrs.k6_wavefunction_minus = k_wavefunction_minus[5].getDevicePtr(subgrid);
            ptrs.k6_reservoir_minus = k_reservoir_minus[5].getDevicePtr(subgrid);
        }
        if (k_max > 6) {
            ptrs.k7_wavefunction_minus = k_wavefunction_minus[6].getDevicePtr(subgrid);
            ptrs.k7_reservoir_minus = k_reservoir_minus[6].getDevicePtr(subgrid);
        }
        if (k_max > 7) {
            ptrs.k8_wavefunction_minus = k_wavefunction_minus[7].getDevicePtr(subgrid);
            ptrs.k8_reservoir_minus = k_reservoir_minus[7].getDevicePtr(subgrid);
        }
        if (k_max > 8) {
            ptrs.k9_wavefunction_minus = k_wavefunction_minus[8].getDevicePtr(subgrid);
            ptrs.k9_reservoir_minus = k_reservoir_minus[8].getDevicePtr(subgrid);
        }
        if (k_max > 9) {
            ptrs.k10_wavefunction_minus = k_wavefunction_minus[9].getDevicePtr(subgrid);
            ptrs.k10_reservoir_minus = k_reservoir_minus[9].getDevicePtr(subgrid);
        }

        // FFT Matrices
        if (use_fft) {
            ptrs.fft_minus = GET_RAW_PTR( fft_minus );
            ptrs.fft_mask_minus = GET_RAW_PTR( fft_mask_minus );
        }

        return ptrs;
    }
};

} // namespace PC3