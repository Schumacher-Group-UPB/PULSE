#include "cuda_complex.cuh"
#include "cuda_macro.cuh"
#include "cuda_device_variables.cuh"

__device__ real_number dev_s_dx = 0;
__device__ real_number dev_p_g_r = 0;
__device__ int dev_s_N = 0;
__device__ real_number dev_p_m_eff = 0;
__device__ real_number dev_p_gamma_c = 0;
__device__ real_number dev_p_g_c = 0;
__device__ real_number dev_p_g_pm = 0;
__device__ real_number dev_p_gamma_r = 0;
__device__ real_number dev_p_R = 0;
__device__ real_number dev_p_delta_LT = 0;
__device__ real_number dev_s_dt = 0;
__device__ real_number dev_p_xmax = 0;
__device__ real_number dev_one_over_h_bar_s = 0;

__device__ int dev_n_pump = 0;
__device__ int dev_n_pulse = 0;

// Cached Device Variables
__device__ real_number dev_p_m_eff_scaled = 0;
__device__ real_number dev_p_delta_LT_scaled = 0;
__device__ complex_number dev_pgr_plus_pR = {0,0};

// Pump and Pulse device arrays
real_number* dev_pump_amp = nullptr;
real_number* dev_pump_width = nullptr;
real_number* dev_pump_X = nullptr;
real_number* dev_pump_Y = nullptr;
int* dev_pump_pol = nullptr;
real_number* dev_pulse_t0 = nullptr;
real_number* dev_pulse_amp = nullptr;
real_number* dev_pulse_freq = nullptr;
real_number* dev_pulse_sigma = nullptr;
int* dev_pulse_m = nullptr;
int* dev_pulse_pol = nullptr;
real_number* dev_pulse_width = nullptr;
real_number* dev_pulse_X = nullptr;
real_number* dev_pulse_Y = nullptr;

// Device Pointers to input and output arrays
complex_number* dev_current_Psi_Plus = nullptr;
complex_number* dev_current_Psi_Minus = nullptr;
complex_number* dev_current_n_Plus = nullptr;
complex_number* dev_current_n_Minus = nullptr;
complex_number* dev_next_Psi_Plus = nullptr;
complex_number* dev_next_Psi_Minus = nullptr;
complex_number* dev_next_n_Plus = nullptr;
complex_number* dev_next_n_Minus = nullptr;

// Device Pointers to k1, k2, k3, k4, (k5, k6, k7) arrays
complex_number* dev_k1_Psi_Plus = nullptr;
complex_number* dev_k1_Psi_Minus = nullptr;
complex_number* dev_k1_n_Plus = nullptr;
complex_number* dev_k1_n_Minus = nullptr;
complex_number* dev_k2_Psi_Plus = nullptr;
complex_number* dev_k2_Psi_Minus = nullptr;
complex_number* dev_k2_n_Plus = nullptr;
complex_number* dev_k2_n_Minus = nullptr;
complex_number* dev_k3_Psi_Plus = nullptr;
complex_number* dev_k3_Psi_Minus = nullptr;
complex_number* dev_k3_n_Plus = nullptr;
complex_number* dev_k3_n_Minus = nullptr;
complex_number* dev_k4_Psi_Plus = nullptr;
complex_number* dev_k4_Psi_Minus = nullptr;
complex_number* dev_k4_n_Plus = nullptr;
complex_number* dev_k4_n_Minus = nullptr;
complex_number* dev_k5_Psi_Plus = nullptr;
complex_number* dev_k5_Psi_Minus = nullptr;
complex_number* dev_k5_n_Plus = nullptr;
complex_number* dev_k5_n_Minus = nullptr;
complex_number* dev_k6_Psi_Plus = nullptr;
complex_number* dev_k6_Psi_Minus = nullptr;
complex_number* dev_k6_n_Plus = nullptr;
complex_number* dev_k6_n_Minus = nullptr;
complex_number* dev_k7_Psi_Plus = nullptr;
complex_number* dev_k7_Psi_Minus = nullptr;
complex_number* dev_k7_n_Plus = nullptr;
complex_number* dev_k7_n_Minus = nullptr;

// Device Pointers to pulse and pulse2
complex_number* dev_fft_plus = nullptr;
complex_number* dev_fft_minus = nullptr;

real_number* dev_rk_error = nullptr;

// CUDA FFT Plan
cufftHandle plan;

__device__ complex_number dev_half_i = {0,0};
__device__ complex_number dev_i = {0,0};
__device__ complex_number dev_minus_half_i = {0,0};
__device__ complex_number dev_minus_i = {0,0};

void initializeDeviceVariables( const real_number s_dx, const real_number s_dt, const real_number p_g_r, const int s_N, const real_number p_m_eff, const real_number p_gamma_c, const real_number p_g_c, const real_number p_g_pm, const real_number p_gamma_r, const real_number p_R, const real_number p_delta_LT, const real_number p_xmax, const real_number h_bar_s ) {
    CHECK_CUDA_ERROR( cudaMemcpyToSymbol( dev_s_dx, &s_dx, sizeof( real_number ) ), "cudaMemcpyToSymbol dx" );
    CHECK_CUDA_ERROR( cudaMemcpyToSymbol( dev_s_dt, &s_dt, sizeof( real_number ) ), "cudaMemcpyToSymbol dt" );
    CHECK_CUDA_ERROR( cudaMemcpyToSymbol( dev_p_g_r, &p_g_r, sizeof( real_number ) ), "cudaMemcpyToSymbol g_r" );
    CHECK_CUDA_ERROR( cudaMemcpyToSymbol( dev_s_N, &s_N, sizeof( int ) ), "cudaMemcpyToSymbol N" );
    CHECK_CUDA_ERROR( cudaMemcpyToSymbol( dev_p_m_eff, &p_m_eff, sizeof( real_number ) ), "cudaMemcpyToSymbol m_eff" );
    CHECK_CUDA_ERROR( cudaMemcpyToSymbol( dev_p_gamma_c, &p_gamma_c, sizeof( real_number ) ), "cudaMemcpyToSymbol gamma_c" );
    CHECK_CUDA_ERROR( cudaMemcpyToSymbol( dev_p_g_c, &p_g_c, sizeof( real_number ) ), "cudaMemcpyToSymbol g_c" );
    CHECK_CUDA_ERROR( cudaMemcpyToSymbol( dev_p_g_pm, &p_g_pm, sizeof( real_number ) ), "cudaMemcpyToSymbol g_pm" );
    CHECK_CUDA_ERROR( cudaMemcpyToSymbol( dev_p_gamma_r, &p_gamma_r, sizeof( real_number ) ), "cudaMemcpyToSymbol gamma_r" );
    CHECK_CUDA_ERROR( cudaMemcpyToSymbol( dev_p_R, &p_R, sizeof( real_number ) ), "cudaMemcpyToSymbol R" );
    CHECK_CUDA_ERROR( cudaMemcpyToSymbol( dev_p_delta_LT, &p_delta_LT, sizeof( real_number ) ), "cudaMemcpyToSymbol delta_LT" );
    CHECK_CUDA_ERROR( cudaMemcpyToSymbol( dev_p_xmax, &p_xmax, sizeof( real_number ) ), "cudaMemcpyToSymbol dev_p_xmax" );
    const real_number one_over_h_bar_s = 1.0 / h_bar_s;
    CHECK_CUDA_ERROR( cudaMemcpyToSymbol( dev_one_over_h_bar_s, &one_over_h_bar_s, sizeof( real_number ) ), "cudaMemcpyToSymbol dev_one_over_h_bar_s" );
    // P/M 0.5i/1i
    complex_number half_i = { 0.0, 0.5 };
    complex_number i = { 0.0, 1.0 };
    complex_number minus_half_i = { 0.0, -0.5 };
    complex_number minus_i = { 0.0, -1.0 };
    CHECK_CUDA_ERROR( cudaMemcpyToSymbol( dev_half_i, &half_i, sizeof( complex_number ) ), "cudaMemcpyToSymbol dev_half_i" );
    CHECK_CUDA_ERROR( cudaMemcpyToSymbol( dev_i, &i, sizeof( complex_number ) ), "cudaMemcpyToSymbol dev_i" );
    CHECK_CUDA_ERROR( cudaMemcpyToSymbol( dev_minus_half_i, &minus_half_i, sizeof( complex_number ) ), "cudaMemcpyToSymbol dev_minus_half_i" );
    CHECK_CUDA_ERROR( cudaMemcpyToSymbol( dev_minus_i, &minus_i, sizeof( complex_number ) ), "cudaMemcpyToSymbol dev_minus_i" );
    // Constant variables
    const real_number p_m_eff_scaled = -0.5 / ( p_m_eff * s_dx * s_dx );
    const real_number p_delta_LT_scaled = p_delta_LT / s_dx / s_dx;
    const complex_number pgr_plus_pR = { p_g_r, 0.5 * p_R };
    CHECK_CUDA_ERROR( cudaMemcpyToSymbol( dev_p_m_eff_scaled, &p_m_eff_scaled, sizeof( real_number ) ), "cudaMemcpyToSymbol dev_p_m_eff_scaled" );
    CHECK_CUDA_ERROR( cudaMemcpyToSymbol( dev_p_delta_LT_scaled, &p_delta_LT_scaled, sizeof( real_number ) ), "cudaMemcpyToSymbol dev_p_delta_LT_scaled" );
    CHECK_CUDA_ERROR( cudaMemcpyToSymbol( dev_pgr_plus_pR, &pgr_plus_pR, sizeof( complex_number ) ), "cudaMemcpyToSymbol dev_pgr_plus_pR" );
}

void initializePumpVariables( real_number* pump_amp, real_number* pump_width, real_number* pump_X, real_number* pump_Y, int* pump_pol, const int size ) {
    CHECK_CUDA_ERROR( cudaMalloc( (void**)&dev_pump_amp, size * sizeof( real_number ) ), "malloc dev_pump_amp" );
    CHECK_CUDA_ERROR( cudaMemcpy( dev_pump_amp, pump_amp, size * sizeof( real_number ), cudaMemcpyHostToDevice ), "memcopy host to device dev_pump_amp" );
    CHECK_CUDA_ERROR( cudaMalloc( (void**)&dev_pump_width, size * sizeof( real_number ) ), "malloc dev_pump_width" );
    CHECK_CUDA_ERROR( cudaMemcpy( dev_pump_width, pump_width, size * sizeof( real_number ), cudaMemcpyHostToDevice ), "memcopy host to device dev_pump_width" );
    CHECK_CUDA_ERROR( cudaMalloc( (void**)&dev_pump_X, size * sizeof( real_number ) ), "malloc dev_pump_X" );
    CHECK_CUDA_ERROR( cudaMemcpy( dev_pump_X, pump_X, size * sizeof( real_number ), cudaMemcpyHostToDevice ), "memcopy host to device dev_pump_X" );
    CHECK_CUDA_ERROR( cudaMalloc( (void**)&dev_pump_Y, size * sizeof( real_number ) ), "malloc dev_pump_Y" );
    CHECK_CUDA_ERROR( cudaMemcpy( dev_pump_Y, pump_Y, size * sizeof( real_number ), cudaMemcpyHostToDevice ), "memcopy host to device dev_pump_Y" );
    CHECK_CUDA_ERROR( cudaMalloc( (void**)&dev_pump_pol, size * sizeof( int ) ), "malloc dev_pump_pol" );
    CHECK_CUDA_ERROR( cudaMemcpy( dev_pump_pol, pump_pol, size * sizeof( int ), cudaMemcpyHostToDevice ), "memcopy host to device dev_pump_pol" );

    CHECK_CUDA_ERROR( cudaMemcpyToSymbol( dev_n_pump, &size, sizeof( int ) ), "cudaMemcpyToSymbol dev_n_pump" );
}

void initializePulseVariables( real_number* pulse_t0, real_number* pulse_amp, real_number* pulse_freq, real_number* pulse_sigma, int* pulse_m, int* pulse_pol, real_number* pulse_width, real_number* pulse_X, real_number* pulse_Y, const int size ) {
    CHECK_CUDA_ERROR( cudaMalloc( (void**)&dev_pulse_t0, size * sizeof( real_number ) ), "malloc dev_pulse_t0" );
    CHECK_CUDA_ERROR( cudaMemcpy( dev_pulse_t0, pulse_t0, size * sizeof( real_number ), cudaMemcpyHostToDevice ), "memcopy host to device dev_pulse_t0" );
    CHECK_CUDA_ERROR( cudaMalloc( (void**)&dev_pulse_amp, size * sizeof( real_number ) ), "malloc dev_pulse_amp" );
    CHECK_CUDA_ERROR( cudaMemcpy( dev_pulse_amp, pulse_amp, size * sizeof( real_number ), cudaMemcpyHostToDevice ), "memcopy host to device dev_pulse_amp" );
    CHECK_CUDA_ERROR( cudaMalloc( (void**)&dev_pulse_freq, size * sizeof( real_number ) ), "malloc dev_pulse_freq" );
    CHECK_CUDA_ERROR( cudaMemcpy( dev_pulse_freq, pulse_freq, size * sizeof( real_number ), cudaMemcpyHostToDevice ), "memcopy host to device dev_pulse_freq" );
    CHECK_CUDA_ERROR( cudaMalloc( (void**)&dev_pulse_sigma, size * sizeof( real_number ) ), "malloc dev_pulse_sigma" );
    CHECK_CUDA_ERROR( cudaMemcpy( dev_pulse_sigma, pulse_sigma, size * sizeof( real_number ), cudaMemcpyHostToDevice ), "memcopy host to device dev_pulse_sigma" );
    CHECK_CUDA_ERROR( cudaMalloc( (void**)&dev_pulse_m, size * sizeof( int ) ), "malloc dev_pulse_m" );
    CHECK_CUDA_ERROR( cudaMemcpy( dev_pulse_m, pulse_m, size * sizeof( int ), cudaMemcpyHostToDevice ), "memcopy host to device dev_pulse_m" );
    CHECK_CUDA_ERROR( cudaMalloc( (void**)&dev_pulse_pol, size * sizeof( int ) ), "malloc dev_pulse_pol" );
    CHECK_CUDA_ERROR( cudaMemcpy( dev_pulse_pol, pulse_pol, size * sizeof( int ), cudaMemcpyHostToDevice ), "memcopy host to device dev_pulse_pol" );
    CHECK_CUDA_ERROR( cudaMalloc( (void**)&dev_pulse_width, size * sizeof( real_number ) ), "malloc dev_pulse_width" );
    CHECK_CUDA_ERROR( cudaMemcpy( dev_pulse_width, pulse_width, size * sizeof( real_number ), cudaMemcpyHostToDevice ), "memcopy host to device dev_pulse_width" );
    CHECK_CUDA_ERROR( cudaMalloc( (void**)&dev_pulse_X, size * sizeof( real_number ) ), "malloc dev_pulse_X" );
    CHECK_CUDA_ERROR( cudaMemcpy( dev_pulse_X, pulse_X, size * sizeof( real_number ), cudaMemcpyHostToDevice ), "memcopy host to device dev_pulse_X" );
    CHECK_CUDA_ERROR( cudaMalloc( (void**)&dev_pulse_Y, size * sizeof( real_number ) ), "malloc dev_pulse_Y" );
    CHECK_CUDA_ERROR( cudaMemcpy( dev_pulse_Y, pulse_Y, size * sizeof( real_number ), cudaMemcpyHostToDevice ), "memcopy host to device dev_pulse_Y" );

    CHECK_CUDA_ERROR( cudaMemcpyToSymbol( dev_n_pulse, &size, sizeof( int ) ), "cudaMemcpyToSymbol dev_n_pulse" );
}

/**
 * Initialize device arrays to zero. 
 * @param s_N The size of the array
 */
void initializeDeviceArrays( const int s_N ) {
    const auto size = s_N * s_N;
    // std::unique_ptr dummy = std::make_unique<complex_number[]>( size );
    CHECK_CUDA_ERROR( cudaMalloc( (void**)&dev_current_Psi_Plus, size * sizeof( complex_number ) ), "malloc dev_current_Psi_Plus" )
    CHECK_CUDA_ERROR( cudaMalloc( (void**)&dev_current_Psi_Minus, size * sizeof( complex_number ) ), "malloc dev_current_Psi_Minus" );
    CHECK_CUDA_ERROR( cudaMalloc( (void**)&dev_current_n_Plus, size * sizeof( complex_number ) ), "malloc dev_current_n_Plus" );
    CHECK_CUDA_ERROR( cudaMalloc( (void**)&dev_current_n_Minus, size * sizeof( complex_number ) ), "malloc dev_current_n_Minus" );
    CHECK_CUDA_ERROR( cudaMalloc( (void**)&dev_next_Psi_Plus, size * sizeof( complex_number ) ), "malloc dev_next_Psi_Plus" );
    CHECK_CUDA_ERROR( cudaMalloc( (void**)&dev_next_Psi_Minus, size * sizeof( complex_number ) ), "malloc dev_next_Psi_Minus" );
    CHECK_CUDA_ERROR( cudaMalloc( (void**)&dev_next_n_Plus, size * sizeof( complex_number ) ), "malloc dev_next_n_Plus" );
    CHECK_CUDA_ERROR( cudaMalloc( (void**)&dev_next_n_Minus, size * sizeof( complex_number ) ), "malloc dev_next_n_Minus" );
    CHECK_CUDA_ERROR( cudaMalloc( (void**)&dev_k1_Psi_Plus, size * sizeof( complex_number ) ), "malloc dev_k1_Psi_Plus" );
    CHECK_CUDA_ERROR( cudaMalloc( (void**)&dev_k1_Psi_Minus, size * sizeof( complex_number ) ), "malloc dev_k1_Psi_Minus" );
    CHECK_CUDA_ERROR( cudaMalloc( (void**)&dev_k1_n_Plus, size * sizeof( complex_number ) ), "malloc dev_k1_n_Plus" );
    CHECK_CUDA_ERROR( cudaMalloc( (void**)&dev_k1_n_Minus, size * sizeof( complex_number ) ), "malloc dev_k1_n_Minus" );
    CHECK_CUDA_ERROR( cudaMalloc( (void**)&dev_k2_Psi_Plus, size * sizeof( complex_number ) ), "malloc dev_k2_Psi_Plus" );
    CHECK_CUDA_ERROR( cudaMalloc( (void**)&dev_k2_Psi_Minus, size * sizeof( complex_number ) ), "malloc dev_k2_Psi_Minus" );
    CHECK_CUDA_ERROR( cudaMalloc( (void**)&dev_k2_n_Plus, size * sizeof( complex_number ) ), "malloc dev_k2_n_Plus" );
    CHECK_CUDA_ERROR( cudaMalloc( (void**)&dev_k2_n_Minus, size * sizeof( complex_number ) ), "malloc dev_k2_n_Minus" );
    CHECK_CUDA_ERROR( cudaMalloc( (void**)&dev_k3_Psi_Plus, size * sizeof( complex_number ) ), "malloc dev_k3_Psi_Plus" );
    CHECK_CUDA_ERROR( cudaMalloc( (void**)&dev_k3_Psi_Minus, size * sizeof( complex_number ) ), "malloc dev_k3_Psi_Minus" );
    CHECK_CUDA_ERROR( cudaMalloc( (void**)&dev_k3_n_Plus, size * sizeof( complex_number ) ), "malloc dev_k3_n_Plus" );
    CHECK_CUDA_ERROR( cudaMalloc( (void**)&dev_k3_n_Minus, size * sizeof( complex_number ) ), "malloc dev_k3_n_Minus" );
    CHECK_CUDA_ERROR( cudaMalloc( (void**)&dev_k4_Psi_Plus, size * sizeof( complex_number ) ), "malloc dev_k4_Psi_Plus" );
    CHECK_CUDA_ERROR( cudaMalloc( (void**)&dev_k4_Psi_Minus, size * sizeof( complex_number ) ), "malloc dev_k4_Psi_Minus" );
    CHECK_CUDA_ERROR( cudaMalloc( (void**)&dev_k4_n_Plus, size * sizeof( complex_number ) ), "malloc dev_k4_n_Plus" );
    CHECK_CUDA_ERROR( cudaMalloc( (void**)&dev_k4_n_Minus, size * sizeof( complex_number ) ), "malloc dev_k4_n_Minus" );
    CHECK_CUDA_ERROR( cudaMalloc( (void**)&dev_k5_Psi_Plus, size * sizeof( complex_number ) ), "malloc dev_k4_Psi_Plus" );
    CHECK_CUDA_ERROR( cudaMalloc( (void**)&dev_k5_Psi_Minus, size * sizeof( complex_number ) ), "malloc dev_k4_Psi_Minus" );
    CHECK_CUDA_ERROR( cudaMalloc( (void**)&dev_k5_n_Plus, size * sizeof( complex_number ) ), "malloc dev_k4_n_Plus" );
    CHECK_CUDA_ERROR( cudaMalloc( (void**)&dev_k5_n_Minus, size * sizeof( complex_number ) ), "malloc dev_k4_n_Minus" );
    CHECK_CUDA_ERROR( cudaMalloc( (void**)&dev_k6_Psi_Plus, size * sizeof( complex_number ) ), "malloc dev_k4_Psi_Plus" );
    CHECK_CUDA_ERROR( cudaMalloc( (void**)&dev_k6_Psi_Minus, size * sizeof( complex_number ) ), "malloc dev_k4_Psi_Minus" );
    CHECK_CUDA_ERROR( cudaMalloc( (void**)&dev_k6_n_Plus, size * sizeof( complex_number ) ), "malloc dev_k4_n_Plus" );
    CHECK_CUDA_ERROR( cudaMalloc( (void**)&dev_k6_n_Minus, size * sizeof( complex_number ) ), "malloc dev_k4_n_Minus" );
    CHECK_CUDA_ERROR( cudaMalloc( (void**)&dev_k7_Psi_Plus, size * sizeof( complex_number ) ), "malloc dev_k4_Psi_Plus" );
    CHECK_CUDA_ERROR( cudaMalloc( (void**)&dev_k7_Psi_Minus, size * sizeof( complex_number ) ), "malloc dev_k4_Psi_Minus" );
    CHECK_CUDA_ERROR( cudaMalloc( (void**)&dev_k7_n_Plus, size * sizeof( complex_number ) ), "malloc dev_k4_n_Plus" );
    CHECK_CUDA_ERROR( cudaMalloc( (void**)&dev_k7_n_Minus, size * sizeof( complex_number ) ), "malloc dev_k4_n_Minus" );
    CHECK_CUDA_ERROR( cudaMalloc( (void**)&dev_fft_plus, size * sizeof( complex_number ) ), "malloc dev_fft_plus" );
    CHECK_CUDA_ERROR( cudaMalloc( (void**)&dev_fft_minus, size * sizeof( complex_number ) ), "malloc dev_fft_minus" );
    CHECK_CUDA_ERROR( cudaMalloc( (void**)&dev_rk_error, size * sizeof( real_number ) ), "malloc dev_rk_error" );

    CHECK_CUDA_ERROR( cufftPlan2d( &plan, s_N, s_N, FFTPLAN ), "FFT Plan" );
}

void setDeviceArrays( complex_number* psi_plus, complex_number* psi_minus, complex_number* n_plus, complex_number* n_minus, const int s_N ) {
    const auto size = s_N * s_N;
    if ( psi_plus != nullptr )
        CHECK_CUDA_ERROR( cudaMemcpy( dev_current_Psi_Plus, psi_plus, size * sizeof( complex_number ), cudaMemcpyHostToDevice ), "memcopy host to device psi_plus" );
    if ( psi_minus != nullptr )
        CHECK_CUDA_ERROR( cudaMemcpy( dev_current_Psi_Minus, psi_minus, size * sizeof( complex_number ), cudaMemcpyHostToDevice ), "memcopy host to device psi_minus" );
    if ( n_plus != nullptr )
        CHECK_CUDA_ERROR( cudaMemcpy( dev_current_n_Plus, n_plus, size * sizeof( complex_number ), cudaMemcpyHostToDevice ), "memcopy host to device n_plus" );
    if ( n_minus != nullptr )
        CHECK_CUDA_ERROR( cudaMemcpy( dev_current_n_Minus, n_minus, size * sizeof( complex_number ), cudaMemcpyHostToDevice ), "memcopy host to device n_minus" );
}

void getDeviceArrays( complex_number* psi_plus, complex_number* psi_minus, complex_number* n_plus, complex_number* n_minus, complex_number* fft_plus, complex_number* fft_minus, const int s_N ) {
    const auto size = s_N * s_N;
    if ( psi_plus != nullptr )
        CHECK_CUDA_ERROR( cudaMemcpy( psi_plus, dev_current_Psi_Plus, size * sizeof( complex_number ), cudaMemcpyDeviceToHost ), "memcpy device to host psi_plus" );
    if ( psi_minus != nullptr )
        CHECK_CUDA_ERROR( cudaMemcpy( psi_minus, dev_current_Psi_Minus, size * sizeof( complex_number ), cudaMemcpyDeviceToHost ), "memcpy device to host psi_minus" );
    if ( n_plus != nullptr )
        CHECK_CUDA_ERROR( cudaMemcpy( n_plus, dev_current_n_Plus, size * sizeof( complex_number ), cudaMemcpyDeviceToHost ), "memcpy device to host n_plus" );
    if ( n_minus != nullptr )
        CHECK_CUDA_ERROR( cudaMemcpy( n_minus, dev_current_n_Minus, size * sizeof( complex_number ), cudaMemcpyDeviceToHost ), "memcpy device to host n_minus" );
    if ( fft_plus != nullptr )
        CHECK_CUDA_ERROR( cudaMemcpy( fft_plus, dev_fft_plus, size * sizeof( complex_number ), cudaMemcpyDeviceToHost ), "memcpy device to host fft_plus" );
    if ( fft_minus != nullptr )
        CHECK_CUDA_ERROR( cudaMemcpy( fft_minus, dev_fft_minus, size * sizeof( complex_number ), cudaMemcpyDeviceToHost ), "memcpy device to host fft_minus" );
}

void getDeviceArraySlice(complex_number* buffer_in, complex_number* buffer_out, const int start, const int length) {
    CHECK_CUDA_ERROR( cudaMemcpy( buffer_out, buffer_in + start, length * sizeof( complex_number ), cudaMemcpyDeviceToHost ), "memcpy device to host buffer" );
}

void freeDeviceArrays() {
    for ( const auto pointer : { dev_current_Psi_Plus, dev_current_Psi_Minus, dev_current_n_Plus, dev_current_n_Minus, dev_next_Psi_Plus, dev_next_Psi_Minus, dev_next_n_Plus, dev_next_n_Minus, dev_k1_Psi_Plus, dev_k1_Psi_Minus, dev_k1_n_Plus, dev_k1_n_Minus, dev_k2_Psi_Plus, dev_k2_Psi_Minus, dev_k2_n_Plus, dev_k2_n_Minus, dev_k3_Psi_Plus, dev_k3_Psi_Minus, dev_k3_n_Plus, dev_k3_n_Minus, dev_k4_Psi_Plus, dev_k4_Psi_Minus, dev_k4_n_Plus, dev_k4_n_Minus, dev_k5_Psi_Plus, dev_k5_Psi_Minus, dev_k5_n_Plus, dev_k5_n_Minus, dev_k6_Psi_Plus, dev_k6_Psi_Minus, dev_k6_n_Plus, dev_k6_n_Minus, dev_k7_Psi_Plus, dev_k7_Psi_Minus, dev_k7_n_Plus, dev_k7_n_Minus, dev_fft_plus, dev_fft_minus } ) {
        CHECK_CUDA_ERROR( cudaFree( pointer ), "free" );
    }
    for ( const auto pointer : { dev_pump_amp, dev_pump_width, dev_pump_X, dev_pump_Y, dev_pulse_t0, dev_pulse_amp, dev_pulse_freq, dev_pulse_sigma, dev_pulse_width, dev_pulse_X, dev_pulse_Y } ) {
        CHECK_CUDA_ERROR( cudaFree( pointer ), "free" );
    }
    for ( const auto pointer : { dev_pump_pol, dev_pulse_m, dev_pulse_pol } ) {
        CHECK_CUDA_ERROR( cudaFree( pointer ), "free" );
    }
    CHECK_CUDA_ERROR( cudaFree( dev_rk_error ), "free" );
    CHECK_CUDA_ERROR( cufftDestroy( plan ), "FFT Destroy" );
}