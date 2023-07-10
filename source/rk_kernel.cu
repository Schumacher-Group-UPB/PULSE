#include <cuComplex.h>
#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cuda.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <complex>

#include "rk_kernel.hpp"

#define CHECK_CUDA_ERROR( func, msg )                             \
    {                                                             \
        func;                                                     \
        cudaError_t err = cudaGetLastError();                     \
        if ( err != cudaSuccess ) {                               \
            printf( "%s: %s\n", msg, cudaGetErrorString( err ) ); \
        }                                                         \
    }

__device__ double dev_s_dx;
__device__ double dev_p_g_r;
__device__ int dev_s_N;
__device__ double dev_p_m_eff;
__device__ double dev_p_gamma_c;
__device__ double dev_p_g_c;
__device__ double dev_p_g_pm;
__device__ double dev_p_gamma_r;
__device__ double dev_p_R;
__device__ double dev_p_delta_LT;
__device__ double dev_s_dt;
__device__ double dev_p_xmax;
__device__ double dev_one_over_h_bar_s;

__device__ int dev_n_pump;
__device__ int dev_n_pulse;

// Double Complex 0.5i and 1i
__device__ cuDoubleComplex dev_half_i;
__device__ cuDoubleComplex dev_i;
__device__ cuDoubleComplex dev_minus_half_i;
__device__ cuDoubleComplex dev_minus_i;

// Cached Device Variables
__device__ double dev_p_m_eff_scaled;
__device__ double dev_p_delta_LT_scaled;
__device__ cuDoubleComplex dev_pgr_plus_pR;

// Pump and Pulse device arrays
double* dev_pump_amp = nullptr;
double* dev_pump_width = nullptr;
double* dev_pump_X = nullptr;
double* dev_pump_Y = nullptr;
int* dev_pump_pol = nullptr;
double* dev_pulse_t0 = nullptr;
double* dev_pulse_amp = nullptr;
double* dev_pulse_freq = nullptr;
double* dev_pulse_sigma = nullptr;
int* dev_pulse_m = nullptr;
int* dev_pulse_pol = nullptr;
double* dev_pulse_width = nullptr;
double* dev_pulse_X = nullptr;
double* dev_pulse_Y = nullptr;

// Device Pointers to input and output arrays
cuDoubleComplex* dev_current_Psi_Plus = nullptr;
cuDoubleComplex* dev_current_Psi_Minus = nullptr;
cuDoubleComplex* dev_current_n_Plus = nullptr;
cuDoubleComplex* dev_current_n_Minus = nullptr;
cuDoubleComplex* dev_next_Psi_Plus = nullptr;
cuDoubleComplex* dev_next_Psi_Minus = nullptr;
cuDoubleComplex* dev_next_n_Plus = nullptr;
cuDoubleComplex* dev_next_n_Minus = nullptr;

// Device Pointers to k1, k2, k3, k4 arrays
cuDoubleComplex* dev_k1_Psi_Plus = nullptr;
cuDoubleComplex* dev_k1_Psi_Minus = nullptr;
cuDoubleComplex* dev_k1_n_Plus = nullptr;
cuDoubleComplex* dev_k1_n_Minus = nullptr;
cuDoubleComplex* dev_k2_Psi_Plus = nullptr;
cuDoubleComplex* dev_k2_Psi_Minus = nullptr;
cuDoubleComplex* dev_k2_n_Plus = nullptr;
cuDoubleComplex* dev_k2_n_Minus = nullptr;
cuDoubleComplex* dev_k3_Psi_Plus = nullptr;
cuDoubleComplex* dev_k3_Psi_Minus = nullptr;
cuDoubleComplex* dev_k3_n_Plus = nullptr;
cuDoubleComplex* dev_k3_n_Minus = nullptr;
cuDoubleComplex* dev_k4_Psi_Plus = nullptr;
cuDoubleComplex* dev_k4_Psi_Minus = nullptr;
cuDoubleComplex* dev_k4_n_Plus = nullptr;
cuDoubleComplex* dev_k4_n_Minus = nullptr;

// Device Pointers to pulse and pulse2
cuDoubleComplex* dev_fft_plus = nullptr;
cuDoubleComplex* dev_fft_minus = nullptr;

// Overload Operator of cuDoubleComplex

__device__ static __inline__ double device_floor( const double x ) {
    return floor( x );
}

__host__ __device__ static __inline__ double sign( double x ) {
    return x > 0 ? 1 : -1;
}

__host__ __device__ static __inline__ double abs2( const cuDoubleComplex& x ) {
    return cuCreal( x ) * cuCreal( x ) + cuCimag( x ) * cuCimag( x );
}
__host__ __device__ static __inline__ double abs2( const double& x ) {
    return x * x;
}

__host__ __device__ cuDoubleComplex operator+( const cuDoubleComplex& a, const cuDoubleComplex& b ) {
    return cuCadd( a, b );
}
__host__ __device__ cuDoubleComplex operator-( const cuDoubleComplex& a, const cuDoubleComplex& b ) {
    return cuCsub( a, b );
}
__host__ __device__ cuDoubleComplex operator+( const cuDoubleComplex& a, const double& b ) {
    return make_cuDoubleComplex( b, 0.0 ) + a;
}
__host__ __device__ cuDoubleComplex operator-( const cuDoubleComplex& a, const double& b ) {
    return make_cuDoubleComplex( b, 0.0 ) - a;
}
__host__ __device__ cuDoubleComplex operator+( const double& a, const cuDoubleComplex& b ) {
    return make_cuDoubleComplex( a, 0.0 ) + b;
}
__host__ __device__ cuDoubleComplex operator-( const double& a, const cuDoubleComplex& b ) {
    return make_cuDoubleComplex( a, 0.0 ) - b;
}
// Overload multiplication Operator of cuDoubleComplex
__host__ __device__ cuDoubleComplex operator*( const cuDoubleComplex& a, const cuDoubleComplex& b ) {
    return cuCmul( a, b );
}
__host__ __device__ cuDoubleComplex operator*( const cuDoubleComplex& a, const double& b ) {
    return make_cuDoubleComplex( b, 0.0 ) * a;
}
__host__ __device__ cuDoubleComplex operator/( const cuDoubleComplex& a, const cuDoubleComplex& b ) {
    return cuCdiv( a, b );
}
__host__ __device__ cuDoubleComplex operator/( const cuDoubleComplex& a, const double& b ) {
    return make_cuDoubleComplex( a.x / b, a.y / b );
}
__host__ __device__ cuDoubleComplex operator*( const double& a, const cuDoubleComplex& b ) {
    return make_cuDoubleComplex( a, 0.0 ) * b;
}
__host__ __device__ cuDoubleComplex operator/( const double& a, const cuDoubleComplex& b ) {
    return make_cuDoubleComplex( a, 0 ) / b;
}

__host__ __device__ static __inline__ cuDoubleComplex square( const cuDoubleComplex& a ) {
    return make_cuDoubleComplex( a.x * a.x, a.y * a.y );
}
__host__ __device__ void operator+=( cuDoubleComplex& a, const cuDoubleComplex& b ) {
    a = a + b;
}
__host__ __device__ void operator+=( cuDoubleComplex& a, const double& b ) {
    a.x = a.x + b;
}

__host__ __device__ static __inline__ cuDoubleComplex pow( const cuDoubleComplex& a, const int N ) {
    cuDoubleComplex res = make_cuDoubleComplex( 1.0, 0 );
    for ( int i = 0; i < abs( N ); i++ )
        res = res * a;
    return N > 0 ? res : 1. / res;
}

__host__ __device__ static __inline__ cuDoubleComplex exp( cuDoubleComplex z ) {
    return make_cuDoubleComplex( exp( z.x ) * cos( z.y ), exp( z.x ) * sin( z.y ) );
}
__host__ __device__ static __inline__ cuDoubleComplex cuCsqrt( cuDoubleComplex x ) {
    double radius = cuCabs( x );
    double cosA = x.x / radius;
    cuDoubleComplex out;
    out.x = sqrt( radius * ( cosA + 1.0 ) / 2.0 );
    out.y = sqrt( radius * ( 1.0 - cosA ) / 2.0 );
    // signbit should be false if x.y is negative
    if ( signbit( x.y ) )
        out.y *= -1.0;

    return out;
}

__device__ static __inline__ bool is_valid_index( const int row, const int col, const int N ) {
    return row >= 0 && row < N && col >= 0 && col < N;
}

__device__ static __inline__ cuDoubleComplex upper_neighbour( cuDoubleComplex* vector, int index, const int row, const int col, const int distance, const int N ) {
    // We are alread in the most upper row
    if ( !is_valid_index( row - distance, col, N ) )
        return make_cuDoubleComplex( 0.0, 0.0 );
    // Valid upper neighbour
    return vector[index - N * distance];
}
__device__ static __inline__ cuDoubleComplex lower_neighbour( cuDoubleComplex* vector, int index, const int row, const int col, const int distance, const int N ) {
    // We are already in the most lower row
    if ( !is_valid_index( row + distance, col, N ) )
        return make_cuDoubleComplex( 0.0, 0.0 );
    // Valid lower neighbour
    return vector[index + N * distance];
}
__device__ static __inline__ cuDoubleComplex left_neighbour( cuDoubleComplex* vector, int index, const int row, const int col, const int distance, const int N ) {
    // Desired Index is in the previous row
    if ( !is_valid_index( row, col - distance, N ) )
        return make_cuDoubleComplex( 0.0, 0.0 );
    // Valid left neighbour
    return vector[index - distance];
}
__device__ static __inline__ cuDoubleComplex right_neighbour( cuDoubleComplex* vector, int index, const int row, const int col, const int distance, const int N ) {
    // Desired Index is in the next row
    if ( !is_valid_index( row, col + distance, N ) )
        return make_cuDoubleComplex( 0.0, 0.0 );
    // Valid right neighbour
    return vector[index + distance];
}

__device__ static __inline__ void hamilton_1( cuDoubleComplex& DT1, cuDoubleComplex& DT4, cuDoubleComplex* __restrict__ vector, int index, const int row, const int col, const int N ) {
    const auto upper = upper_neighbour( vector, index, row, col, 1, N );
    const auto lower = lower_neighbour( vector, index, row, col, 1, N );
    const auto left = left_neighbour( vector, index, row, col, 1, N );
    const auto right = right_neighbour( vector, index, row, col, 1, N );
    DT1 = -4.0* vector[index] + upper + lower + left + right;
    DT4 = upper + lower - left - right - dev_half_i * ( right_neighbour( vector, index - N, row - 1, col, 1, N ) - left_neighbour( vector, index - N, row - 1, col, 1, N ) ) + dev_half_i * ( right_neighbour( vector, index + N, row + 1, col, 1, N ) - left_neighbour( vector, index + N, row + 1, col, 1, N ) );
}

__device__ static __inline__ void hamilton_2( cuDoubleComplex& DT2, cuDoubleComplex& DT3, cuDoubleComplex* __restrict__ vector, int index, const int row, const int col, const int N ) {
    const auto upper = upper_neighbour( vector, index, row, col, 1, N );
    const auto lower = lower_neighbour( vector, index, row, col, 1, N );
    const auto left = left_neighbour( vector, index, row, col, 1, N );
    const auto right = right_neighbour( vector, index, row, col, 1, N );
    DT3 = -4. * vector[index] + upper + lower + left + right;
    DT2 = upper + lower - left - right + dev_half_i * ( right_neighbour( vector, index - N, row - 1, col, 1, N ) - left_neighbour( vector, index - N, row - 1, col, 1, N ) ) - dev_half_i * ( right_neighbour( vector, index + N, row + 1, col, 1, N ) - left_neighbour( vector, index + N, row + 1, col, 1, N ) );
}

__global__ void rungeFuncKernel( double t, cuDoubleComplex* __restrict__ in_Psi_Plus, cuDoubleComplex* __restrict__ in_Psi_Minus, cuDoubleComplex* __restrict__ in_n_Plus, cuDoubleComplex* __restrict__ in_n_Minus, cuDoubleComplex* __restrict__ fft_plus, cuDoubleComplex* __restrict__ fft_minus, cuDoubleComplex* __restrict__ k_Psi_Plus, cuDoubleComplex* __restrict__ k_Psi_Minus, cuDoubleComplex* __restrict__ k_n_Plus, cuDoubleComplex* __restrict__ k_n_Minus,
                                 /* Pump Parameters */ double* __restrict__ dev_pump_amp, double* __restrict__ dev_pump_width, double* __restrict__ dev_pump_X, double* __restrict__ dev_pump_Y, int* __restrict__ dev_pump_pol,
                                 /* Pulse Parameters */ double* __restrict__ dev_pulse_t0, double* __restrict__ dev_pulse_amp, double* __restrict__ dev_pulse_freq, double* __restrict__ dev_pulse_sigma, int* __restrict__ dev_pulse_m, int* __restrict__ dev_pulse_pol, double* __restrict__ dev_pulse_width, double* __restrict__ dev_pulse_X, double* __restrict__ dev_pulse_Y, bool evaluate_pulse ) {
    int blockId = ( gridDim.x * blockIdx.y ) + blockIdx.x;
    int i = ( blockId * ( blockDim.x * blockDim.y ) ) + ( threadIdx.y * blockDim.x ) + threadIdx.x;
    if ( i >= dev_s_N * dev_s_N )
        return;

    const int row = device_floor( i / dev_s_N );
    const int col = i % dev_s_N;
    cuDoubleComplex DT1, DT2, DT3, DT4;
    hamilton_1( DT1, DT4, in_Psi_Plus, i, row, col, dev_s_N );
    hamilton_2( DT2, DT3, in_Psi_Minus, i, row, col, dev_s_N );
    double in_psi_plus_norm = abs2( in_Psi_Plus[i] );
    double in_psi_minus_norm = abs2( in_Psi_Minus[i] );
    k_Psi_Plus[i] = dev_minus_i * ( dev_p_m_eff_scaled * DT1 - dev_half_i * dev_p_gamma_c * in_Psi_Plus[i] + dev_p_g_c * in_psi_plus_norm * in_Psi_Plus[i] + dev_pgr_plus_pR * in_n_Plus[i] * in_Psi_Plus[i] + dev_p_g_pm * in_psi_minus_norm * in_Psi_Plus[i] + dev_p_delta_LT_scaled * DT2 );
    k_Psi_Minus[i] = dev_minus_i * ( dev_p_m_eff_scaled * DT3 - dev_half_i * dev_p_gamma_c * in_Psi_Minus[i] + dev_p_g_c * in_psi_minus_norm * in_Psi_Minus[i] + dev_pgr_plus_pR * in_n_Minus[i] * in_Psi_Minus[i] + dev_p_g_pm * in_psi_plus_norm * in_Psi_Minus[i] + dev_p_delta_LT_scaled * DT4 );
    k_n_Plus[i] = -dev_p_gamma_r * in_n_Plus[i] - dev_p_R * in_psi_plus_norm * in_n_Plus[i];    
    k_n_Minus[i] = -dev_p_gamma_r * in_n_Minus[i] - dev_p_R * in_psi_minus_norm * in_n_Minus[i]; 
    // Add Pumps
    if ( dev_n_pump ) {
        auto x = -dev_p_xmax / 2.0 + dev_s_dx * col;
        auto y = -dev_p_xmax / 2.0 + dev_s_dx * row;
        for ( int c = 0; c < dev_n_pump; c++ ) {
            const double r_squared = abs2( x - dev_pump_X[c] ) + abs2( y - dev_pump_Y[c] );
            const auto w = dev_pump_width[c];
            const auto exp_factor = r_squared / w / w;
            if ( dev_pump_pol[c] >= 0 )
                k_n_Plus[i] += dev_pump_amp[c] * exp_factor * exp( -exp_factor );
            if ( dev_pump_pol[c] <= 0 )
                k_n_Minus[i] += dev_pump_amp[c] * exp_factor * exp( -exp_factor );
        }
    }
    // Add Pulse
    if ( evaluate_pulse ) {
        auto x = -dev_p_xmax / 2.0 + dev_s_dx * col;
        auto y = -dev_p_xmax / 2.0 + dev_s_dx * row;
        for ( int c = 0; c < dev_n_pulse; c++ ) {
            const auto xpos = dev_pulse_X[c];
            const auto ypos = dev_pulse_Y[c];
            double r = sqrt( abs2( x - xpos ) + abs2( y - ypos ) );
            const auto w = dev_pulse_width[c];
            const auto exp_factor = r * r / w / w;
            cuDoubleComplex space_shape = dev_pulse_amp[c] * r / w / w * exp( -exp_factor ) * pow( ( x - xpos + 1.0 * sign( dev_pulse_m[c] ) * make_cuDoubleComplex( 0, 1.0 ) * ( y - ypos ) ), abs( dev_pulse_m[c] ) );
            const auto t0 = dev_pulse_t0[c];
            cuDoubleComplex temp_shape = dev_one_over_h_bar_s * exp( -( t - t0 ) * ( t - t0 ) / dev_pulse_sigma[c] / dev_pulse_sigma[c] - make_cuDoubleComplex( 0, 1.0 ) * dev_pulse_freq[c] * ( t - t0 ) );
            if ( dev_pulse_pol[c] >= 0 )
                k_Psi_Plus[i] += space_shape * temp_shape;
            if ( dev_pulse_pol[c] <= 0 )
                k_Psi_Minus[i] += space_shape * temp_shape;
        }
    }
}

/**
 * @brief Summation of the k1, k2, k3, k4 arrays
 *
 */
__global__ void rungeFuncSum( double s_dt, cuDoubleComplex* __restrict__ out_Psi_Plus, cuDoubleComplex* __restrict__ out_Psi_Minus, cuDoubleComplex* __restrict__ out_n_Plus, cuDoubleComplex* __restrict__ out_n_Minus, cuDoubleComplex* __restrict__ in_Psi_Plus, cuDoubleComplex* __restrict__ in_Psi_Minus, cuDoubleComplex* __restrict__ in_n_Plus, cuDoubleComplex* __restrict__ in_n_Minus, cuDoubleComplex* __restrict__ k_Psi_Plus, cuDoubleComplex* __restrict__ k_Psi_Minus, cuDoubleComplex* __restrict__ k_n_Plus, cuDoubleComplex* __restrict__ k_n_Minus ) {
    int blockId = ( gridDim.x * blockIdx.y ) + blockIdx.x;
    int i = ( blockId * ( blockDim.x * blockDim.y ) ) + ( threadIdx.y * blockDim.x ) + threadIdx.x;
    if ( i < dev_s_N * dev_s_N ) {
        out_Psi_Plus[i] = in_Psi_Plus[i] + s_dt * dev_s_dt * k_Psi_Plus[i];
        out_Psi_Minus[i] = in_Psi_Minus[i] + s_dt * dev_s_dt * k_Psi_Minus[i];
        out_n_Plus[i] = in_n_Plus[i] + s_dt * dev_s_dt * k_n_Plus[i];
        out_n_Minus[i] = in_n_Minus[i] + s_dt * dev_s_dt * k_n_Minus[i];
    }
}

/**
 * @brief Final Sum for the next iteration
 */
__global__ void rungeFuncFinal( cuDoubleComplex* out_Psi_Plus, cuDoubleComplex* out_Psi_Minus, cuDoubleComplex* out_n_Plus, cuDoubleComplex* out_n_Minus, cuDoubleComplex* in_Psi_Plus, cuDoubleComplex* in_Psi_Minus, cuDoubleComplex* in_n_Plus, cuDoubleComplex* in_n_Minus, cuDoubleComplex* __restrict__ k1_Psi_Plus, cuDoubleComplex* __restrict__ k1_Psi_Minus, cuDoubleComplex* __restrict__ k1_n_Plus, cuDoubleComplex* __restrict__ k1_n_Minus, cuDoubleComplex* __restrict__ k2_Psi_Plus, cuDoubleComplex* __restrict__ k2_Psi_Minus, cuDoubleComplex* __restrict__ k2_n_Plus, cuDoubleComplex* __restrict__ k2_n_Minus, cuDoubleComplex* __restrict__ k3_Psi_Plus, cuDoubleComplex* __restrict__ k3_Psi_Minus, cuDoubleComplex* __restrict__ k3_n_Plus, cuDoubleComplex* __restrict__ k3_n_Minus, cuDoubleComplex* __restrict__ k4_Psi_Plus, cuDoubleComplex* __restrict__ k4_Psi_Minus, cuDoubleComplex* __restrict__ k4_n_Plus, cuDoubleComplex* __restrict__ k4_n_Minus ) {
    int blockId = ( gridDim.x * blockIdx.y ) + blockIdx.x;
    int i = ( blockId * ( blockDim.x * blockDim.y ) ) + ( threadIdx.y * blockDim.x ) + threadIdx.x;
    if ( i < dev_s_N * dev_s_N ) {
        out_Psi_Plus[i] = in_Psi_Plus[i] + dev_s_dt / 6.0 * ( k1_Psi_Plus[i] + 2.0 * k2_Psi_Plus[i] + 2.0 * k3_Psi_Plus[i] + k4_Psi_Plus[i] );
        out_Psi_Minus[i] = in_Psi_Minus[i] + dev_s_dt / 6.0 * ( k1_Psi_Minus[i] + 2.0 * k2_Psi_Minus[i] + 2.0 * k3_Psi_Minus[i] + k4_Psi_Minus[i] );
        out_n_Plus[i] = in_n_Plus[i] + dev_s_dt / 6.0 * ( k1_n_Plus[i] + 2.0 * k2_n_Plus[i] + 2.0 * k3_n_Plus[i] + k4_n_Plus[i] );
        out_n_Minus[i] = in_n_Minus[i] + dev_s_dt / 6.0 * ( k1_n_Minus[i] + 2.0 * k2_n_Minus[i] + 2.0 * k3_n_Minus[i] + k4_n_Minus[i] );
    }
}

__host__ __device__ void kernel_generateRingPhase( int s_N, double amp, int n, double w1, double w2, double xPos, double yPos, double p_xmax, double s_dx, bool normalize, cuDoubleComplex* buffer, bool reset = true ) {
    double largest_r = 0.0;
    for ( int i = 0; i < s_N; i++ )
        for ( int j = 0; j < s_N; j++ ) {
            auto index = i * s_N + j;
            if ( reset )
                buffer[index] = make_cuDoubleComplex( 0.0, 0.0 );
            auto x = -p_xmax / 2.0 + s_dx * i;
            auto y = -p_xmax / 2.0 + s_dx * j;
            double r = sqrt( abs2( x - xPos ) + abs2( y - yPos ) );
            buffer[index] += amp * r / w1 / w1 * exp( -r * r / w2 / w2 ) * pow( ( x - xPos + 1.0 * sign( n ) * make_cuDoubleComplex( 0, 1.0 ) * ( y - yPos ) ), abs( n ) );

            largest_r = max( largest_r, abs2( buffer[index] ) );
        }
    if ( !normalize )
        return;
    for ( int i = 0; i < s_N; i++ )
        for ( int j = 0; j < s_N; j++ ) {
            auto index = i * s_N + j;
            buffer[index] = buffer[index] / sqrt( largest_r );
        }
}

__host__ __device__ void kernel_generateRingState( int s_N, double amp, double w1, double w2, double xPos, double yPos, double p_xmax, double s_dx, bool normalize, cuDoubleComplex* buffer, bool reset = true ) {
    double max_buffer = 0.0;
    for ( int i = 0; i < s_N; i++ )
        for ( int j = 0; j < s_N; j++ ) {
            auto index = i * s_N + j;
            if ( reset )
                buffer[index] = make_cuDoubleComplex( 0.0, 0.0 );
            auto x = -p_xmax / 2.0 + s_dx * i;
            auto y = -p_xmax / 2.0 + s_dx * j;
            double r = sqrt( abs2( x - xPos ) + abs2( y - yPos ) );
            buffer[index] += amp * r * r / w1 / w1 * exp( -r * r / w2 / w2 );
            max_buffer = max( max_buffer, abs2( buffer[index] ) );
        }
    if ( !normalize )
        return;
    for ( int i = 0; i < s_N; i++ )
        for ( int j = 0; j < s_N; j++ ) {
            auto index = i * s_N + j;
            buffer[index] = buffer[index] / sqrt( max_buffer );
        }
}

__global__ void kernel_makeFFTVisible( cuDoubleComplex* input, cuDoubleComplex* output ) {
    int blockId = ( gridDim.x * blockIdx.y ) + blockIdx.x;
    int index = ( blockId * ( blockDim.x * blockDim.y ) ) + ( threadIdx.y * blockDim.x ) + threadIdx.x;
    if ( index < dev_s_N * dev_s_N ) {
        const auto val = input[index];
        output[index] = make_cuDoubleComplex( log( val.x * val.x + val.y * val.y ), 0 );
    }
}

#define swap( a, b )  \
    {                 \
        auto tmp = a; \
        a = b;        \
        b = tmp;      \
    }

__global__ void fftshift_2D( cuDoubleComplex* data_plus, cuDoubleComplex* data_minus, int N_half ) {
    int blockId = ( gridDim.x * blockIdx.y ) + blockIdx.x;
    int index = ( blockId * ( blockDim.x * blockDim.y ) ) + ( threadIdx.y * blockDim.x ) + threadIdx.x;
    if ( index >= dev_s_N * dev_s_N )
        return;
    // Current indices of upper left quadrant
    const int i = device_floor( index / dev_s_N );
    if ( i >= N_half )
        return;
    const int j = index % dev_s_N;
    if ( j >= N_half )
        return;
    // Swap upper left quadrant with lower right quadrant
    swap( data_plus[i * dev_s_N + j], data_plus[( i + N_half ) * dev_s_N + j + N_half] );
    swap( data_minus[i * dev_s_N + j], data_minus[( i + N_half ) * dev_s_N + j + N_half] );
    // Swap lower left quadrant with upper right quadrant
    swap( data_plus[i * dev_s_N + j + N_half], data_plus[( i + N_half ) * dev_s_N + j] );
    swap( data_minus[i * dev_s_N + j + N_half], data_minus[( i + N_half ) * dev_s_N + j] );
}

__global__ void kernel_maskFFT( cuDoubleComplex* data_plus, cuDoubleComplex* data_minus, const double s, const double w, bool out_mask ) {
    int blockId = ( gridDim.x * blockIdx.y ) + blockIdx.x;
    int index = ( blockId * ( blockDim.x * blockDim.y ) ) + ( threadIdx.y * blockDim.x ) + threadIdx.x;
    if ( index < dev_s_N * dev_s_N ) {
        const int i = device_floor( index / dev_s_N );
        const int j = index % dev_s_N;
        double ky = 2. * i / dev_s_N - 1.;
        double kx = 2. * j / dev_s_N - 1.;
        double mask = exp( -1.0 * pow( ( kx * kx + ky * ky ) / w / w, s ) );
        data_plus[index] = out_mask ? make_cuDoubleComplex( sqrt( mask ), 0 ) : data_plus[index] / dev_s_N / dev_s_N;// * mask;
        data_minus[index] = out_mask ? make_cuDoubleComplex( sqrt( mask ), 0 ) : data_minus[index] / dev_s_N / dev_s_N;// * mask;
    }
}

void generateRingPhase( int s_N, double amp, int n, double w1, double w2, double xPos, double yPos, double p_xmax, double s_dx, bool normalize, Scalar* buffer, bool reset ) {
    // Cache cuComplexDouble result
    cuDoubleComplex* buffer_cuDoubleComplex = reinterpret_cast<cuDoubleComplex*>( buffer );
    kernel_generateRingPhase( s_N, amp, n, w1, w2, xPos, yPos, p_xmax, s_dx, normalize, buffer_cuDoubleComplex, reset );
}
void generateRingState( int s_N, double amp, double w1, double w2, double xPos, double yPos, double p_xmax, double s_dx, bool normalize, Scalar* buffer, bool reset ) {
    cuDoubleComplex* buffer_cuDoubleComplex = reinterpret_cast<cuDoubleComplex*>( buffer );
    kernel_generateRingState( s_N, amp, w1, w2, xPos, yPos, p_xmax, s_dx, normalize, buffer_cuDoubleComplex, reset );
}

/**
 * @brief Initialize device variables from host variables
 */
#include <iostream>
void initializeDeviceVariables( const double s_dx, const double s_dt, const double p_g_r, const int s_N, const double p_m_eff, const double p_gamma_c, const double p_g_c, const double p_g_pm, const double p_gamma_r, const double p_R, const double p_delta_LT, const double p_xmax, const double h_bar_s ) {
    CHECK_CUDA_ERROR( cudaMemcpyToSymbol( dev_s_dx, &s_dx, sizeof( double ) ), "cudaMemcpyToSymbol dx" );
    CHECK_CUDA_ERROR( cudaMemcpyToSymbol( dev_s_dt, &s_dt, sizeof( double ) ), "cudaMemcpyToSymbol dt" );
    CHECK_CUDA_ERROR( cudaMemcpyToSymbol( dev_p_g_r, &p_g_r, sizeof( double ) ), "cudaMemcpyToSymbol g_r" );
    CHECK_CUDA_ERROR( cudaMemcpyToSymbol( dev_s_N, &s_N, sizeof( int ) ), "cudaMemcpyToSymbol N" );
    CHECK_CUDA_ERROR( cudaMemcpyToSymbol( dev_p_m_eff, &p_m_eff, sizeof( double ) ), "cudaMemcpyToSymbol m_eff" );
    CHECK_CUDA_ERROR( cudaMemcpyToSymbol( dev_p_gamma_c, &p_gamma_c, sizeof( double ) ), "cudaMemcpyToSymbol gamma_c" );
    CHECK_CUDA_ERROR( cudaMemcpyToSymbol( dev_p_g_c, &p_g_c, sizeof( double ) ), "cudaMemcpyToSymbol g_c" );
    CHECK_CUDA_ERROR( cudaMemcpyToSymbol( dev_p_g_pm, &p_g_pm, sizeof( double ) ), "cudaMemcpyToSymbol g_pm" );
    CHECK_CUDA_ERROR( cudaMemcpyToSymbol( dev_p_gamma_r, &p_gamma_r, sizeof( double ) ), "cudaMemcpyToSymbol gamma_r" );
    CHECK_CUDA_ERROR( cudaMemcpyToSymbol( dev_p_R, &p_R, sizeof( double ) ), "cudaMemcpyToSymbol R" );
    CHECK_CUDA_ERROR( cudaMemcpyToSymbol( dev_p_delta_LT, &p_delta_LT, sizeof( double ) ), "cudaMemcpyToSymbol delta_LT" );
    CHECK_CUDA_ERROR( cudaMemcpyToSymbol( dev_p_xmax, &p_xmax, sizeof( double ) ), "cudaMemcpyToSymbol dev_p_xmax" );
    const auto one_over_h_bar_s = 1.0 / h_bar_s;
    CHECK_CUDA_ERROR( cudaMemcpyToSymbol( dev_one_over_h_bar_s, &one_over_h_bar_s, sizeof( double ) ), "cudaMemcpyToSymbol dev_one_over_h_bar_s" );
    // P/M 0.5i/1i
    auto half_i = make_cuDoubleComplex( 0.0, 0.5 );
    auto i = make_cuDoubleComplex( 0.0, 1.0 );
    auto minus_half_i = make_cuDoubleComplex( 0.0, -0.5 );
    auto minus_i = make_cuDoubleComplex( 0.0, -1.0 );
    CHECK_CUDA_ERROR( cudaMemcpyToSymbol( dev_half_i, &half_i, sizeof( cuDoubleComplex ) ), "cudaMemcpyToSymbol dev_one_over_h_bar_s" );
    CHECK_CUDA_ERROR( cudaMemcpyToSymbol( dev_i, &i, sizeof( cuDoubleComplex ) ), "cudaMemcpyToSymbol dev_one_over_h_bar_s" );
    CHECK_CUDA_ERROR( cudaMemcpyToSymbol( dev_minus_half_i, &minus_half_i, sizeof( cuDoubleComplex ) ), "cudaMemcpyToSymbol dev_one_over_h_bar_s" );
    CHECK_CUDA_ERROR( cudaMemcpyToSymbol( dev_minus_i, &minus_i, sizeof( cuDoubleComplex ) ), "cudaMemcpyToSymbol dev_one_over_h_bar_s" );
    // Constant variables
    const auto p_m_eff_scaled = -0.5 / ( p_m_eff * s_dx * s_dx );
    const auto p_delta_LT_scaled = p_delta_LT / s_dx / s_dx;
    const auto pgr_plus_pR = make_cuDoubleComplex( p_g_r, 0.5 * p_R );
    CHECK_CUDA_ERROR( cudaMemcpyToSymbol( dev_p_m_eff_scaled, &p_m_eff_scaled, sizeof( double ) ), "cudaMemcpyToSymbol dev_p_m_eff_scaled" );
    CHECK_CUDA_ERROR( cudaMemcpyToSymbol( dev_p_delta_LT_scaled, &p_delta_LT_scaled, sizeof( double ) ), "cudaMemcpyToSymbol dev_p_delta_LT_scaled" );
    CHECK_CUDA_ERROR( cudaMemcpyToSymbol( dev_pgr_plus_pR, &pgr_plus_pR, sizeof( cuDoubleComplex ) ), "cudaMemcpyToSymbol dev_pgr_plus_pR" );
}

void initializePumpVariables( double* pump_amp, double* pump_width, double* pump_X, double* pump_Y, int* pump_pol, const int size ) {
    CHECK_CUDA_ERROR( cudaMalloc( (void**)&dev_pump_amp, size * sizeof( double ) ), "malloc dev_pump_amp" );
    CHECK_CUDA_ERROR( cudaMemcpy( dev_pump_amp, pump_amp, size * sizeof( double ), cudaMemcpyHostToDevice ), "memcopy host to device dev_pump_amp" );
    CHECK_CUDA_ERROR( cudaMalloc( (void**)&dev_pump_width, size * sizeof( double ) ), "malloc dev_pump_width" );
    CHECK_CUDA_ERROR( cudaMemcpy( dev_pump_width, pump_width, size * sizeof( double ), cudaMemcpyHostToDevice ), "memcopy host to device dev_pump_width" );
    CHECK_CUDA_ERROR( cudaMalloc( (void**)&dev_pump_X, size * sizeof( double ) ), "malloc dev_pump_X" );
    CHECK_CUDA_ERROR( cudaMemcpy( dev_pump_X, pump_X, size * sizeof( double ), cudaMemcpyHostToDevice ), "memcopy host to device dev_pump_X" );
    CHECK_CUDA_ERROR( cudaMalloc( (void**)&dev_pump_Y, size * sizeof( double ) ), "malloc dev_pump_Y" );
    CHECK_CUDA_ERROR( cudaMemcpy( dev_pump_Y, pump_Y, size * sizeof( double ), cudaMemcpyHostToDevice ), "memcopy host to device dev_pump_Y" );
    CHECK_CUDA_ERROR( cudaMalloc( (void**)&dev_pump_pol, size * sizeof( int ) ), "malloc dev_pump_pol" );
    CHECK_CUDA_ERROR( cudaMemcpy( dev_pump_pol, pump_pol, size * sizeof( int ), cudaMemcpyHostToDevice ), "memcopy host to device dev_pump_pol" );

    CHECK_CUDA_ERROR( cudaMemcpyToSymbol( dev_n_pump, &size, sizeof( int ) ), "cudaMemcpyToSymbol dev_n_pump" );
}

void initializePulseVariables( double* pulse_t0, double* pulse_amp, double* pulse_freq, double* pulse_sigma, int* pulse_m, int* pulse_pol, double* pulse_width, double* pulse_X, double* pulse_Y, const int size ) {
    CHECK_CUDA_ERROR( cudaMalloc( (void**)&dev_pulse_t0, size * sizeof( double ) ), "malloc dev_pulse_t0" );
    CHECK_CUDA_ERROR( cudaMemcpy( dev_pulse_t0, pulse_t0, size * sizeof( double ), cudaMemcpyHostToDevice ), "memcopy host to device dev_pulse_t0" );
    CHECK_CUDA_ERROR( cudaMalloc( (void**)&dev_pulse_amp, size * sizeof( double ) ), "malloc dev_pulse_amp" );
    CHECK_CUDA_ERROR( cudaMemcpy( dev_pulse_amp, pulse_amp, size * sizeof( double ), cudaMemcpyHostToDevice ), "memcopy host to device dev_pulse_amp" );
    CHECK_CUDA_ERROR( cudaMalloc( (void**)&dev_pulse_freq, size * sizeof( double ) ), "malloc dev_pulse_freq" );
    CHECK_CUDA_ERROR( cudaMemcpy( dev_pulse_freq, pulse_freq, size * sizeof( double ), cudaMemcpyHostToDevice ), "memcopy host to device dev_pulse_freq" );
    CHECK_CUDA_ERROR( cudaMalloc( (void**)&dev_pulse_sigma, size * sizeof( double ) ), "malloc dev_pulse_sigma" );
    CHECK_CUDA_ERROR( cudaMemcpy( dev_pulse_sigma, pulse_sigma, size * sizeof( double ), cudaMemcpyHostToDevice ), "memcopy host to device dev_pulse_sigma" );
    CHECK_CUDA_ERROR( cudaMalloc( (void**)&dev_pulse_m, size * sizeof( int ) ), "malloc dev_pulse_m" );
    CHECK_CUDA_ERROR( cudaMemcpy( dev_pulse_m, pulse_m, size * sizeof( int ), cudaMemcpyHostToDevice ), "memcopy host to device dev_pulse_m" );
    CHECK_CUDA_ERROR( cudaMalloc( (void**)&dev_pulse_pol, size * sizeof( int ) ), "malloc dev_pulse_pol" );
    CHECK_CUDA_ERROR( cudaMemcpy( dev_pulse_pol, pulse_pol, size * sizeof( int ), cudaMemcpyHostToDevice ), "memcopy host to device dev_pulse_pol" );
    CHECK_CUDA_ERROR( cudaMalloc( (void**)&dev_pulse_width, size * sizeof( double ) ), "malloc dev_pulse_width" );
    CHECK_CUDA_ERROR( cudaMemcpy( dev_pulse_width, pulse_width, size * sizeof( double ), cudaMemcpyHostToDevice ), "memcopy host to device dev_pulse_width" );
    CHECK_CUDA_ERROR( cudaMalloc( (void**)&dev_pulse_X, size * sizeof( double ) ), "malloc dev_pulse_X" );
    CHECK_CUDA_ERROR( cudaMemcpy( dev_pulse_X, pulse_X, size * sizeof( double ), cudaMemcpyHostToDevice ), "memcopy host to device dev_pulse_X" );
    CHECK_CUDA_ERROR( cudaMalloc( (void**)&dev_pulse_Y, size * sizeof( double ) ), "malloc dev_pulse_Y" );
    CHECK_CUDA_ERROR( cudaMemcpy( dev_pulse_Y, pulse_Y, size * sizeof( double ), cudaMemcpyHostToDevice ), "memcopy host to device dev_pulse_Y" );

    CHECK_CUDA_ERROR( cudaMemcpyToSymbol( dev_n_pulse, &size, sizeof( int ) ), "cudaMemcpyToSymbol dev_n_pulse" );
}

// CUDA FFT Plan
cufftHandle plan;

/**
 * @brief Initialize device arrays to zero
 */
void initializeDeviceArrays( const int s_N ) {
    const auto size = s_N * s_N;
    //std::unique_ptr dummy = std::make_unique<cuDoubleComplex[]>( size );
    CHECK_CUDA_ERROR( cudaMalloc( (void**)&dev_current_Psi_Plus, size * sizeof( cuDoubleComplex ) ), "malloc dev_current_Psi_Plus" )
    CHECK_CUDA_ERROR( cudaMalloc( (void**)&dev_current_Psi_Minus, size * sizeof( cuDoubleComplex ) ), "malloc dev_current_Psi_Minus" );
    CHECK_CUDA_ERROR( cudaMalloc( (void**)&dev_current_n_Plus, size * sizeof( cuDoubleComplex ) ), "malloc dev_current_n_Plus" );
    CHECK_CUDA_ERROR( cudaMalloc( (void**)&dev_current_n_Minus, size * sizeof( cuDoubleComplex ) ), "malloc dev_current_n_Minus" );
    CHECK_CUDA_ERROR( cudaMalloc( (void**)&dev_next_Psi_Plus, size * sizeof( cuDoubleComplex ) ), "malloc dev_next_Psi_Plus" );
    CHECK_CUDA_ERROR( cudaMalloc( (void**)&dev_next_Psi_Minus, size * sizeof( cuDoubleComplex ) ), "malloc dev_next_Psi_Minus" );
    CHECK_CUDA_ERROR( cudaMalloc( (void**)&dev_next_n_Plus, size * sizeof( cuDoubleComplex ) ), "malloc dev_next_n_Plus" );
    CHECK_CUDA_ERROR( cudaMalloc( (void**)&dev_next_n_Minus, size * sizeof( cuDoubleComplex ) ), "malloc dev_next_n_Minus" );
    CHECK_CUDA_ERROR( cudaMalloc( (void**)&dev_k1_Psi_Plus, size * sizeof( cuDoubleComplex ) ), "malloc dev_k1_Psi_Plus" );
    CHECK_CUDA_ERROR( cudaMalloc( (void**)&dev_k1_Psi_Minus, size * sizeof( cuDoubleComplex ) ), "malloc dev_k1_Psi_Minus" );
    CHECK_CUDA_ERROR( cudaMalloc( (void**)&dev_k1_n_Plus, size * sizeof( cuDoubleComplex ) ), "malloc dev_k1_n_Plus" );
    CHECK_CUDA_ERROR( cudaMalloc( (void**)&dev_k1_n_Minus, size * sizeof( cuDoubleComplex ) ), "malloc dev_k1_n_Minus" );
    CHECK_CUDA_ERROR( cudaMalloc( (void**)&dev_k2_Psi_Plus, size * sizeof( cuDoubleComplex ) ), "malloc dev_k2_Psi_Plus" );
    CHECK_CUDA_ERROR( cudaMalloc( (void**)&dev_k2_Psi_Minus, size * sizeof( cuDoubleComplex ) ), "malloc dev_k2_Psi_Minus" );
    CHECK_CUDA_ERROR( cudaMalloc( (void**)&dev_k2_n_Plus, size * sizeof( cuDoubleComplex ) ), "malloc dev_k2_n_Plus" );
    CHECK_CUDA_ERROR( cudaMalloc( (void**)&dev_k2_n_Minus, size * sizeof( cuDoubleComplex ) ), "malloc dev_k2_n_Minus" );
    CHECK_CUDA_ERROR( cudaMalloc( (void**)&dev_k3_Psi_Plus, size * sizeof( cuDoubleComplex ) ), "malloc dev_k3_Psi_Plus" );
    CHECK_CUDA_ERROR( cudaMalloc( (void**)&dev_k3_Psi_Minus, size * sizeof( cuDoubleComplex ) ), "malloc dev_k3_Psi_Minus" );
    CHECK_CUDA_ERROR( cudaMalloc( (void**)&dev_k3_n_Plus, size * sizeof( cuDoubleComplex ) ), "malloc dev_k3_n_Plus" );
    CHECK_CUDA_ERROR( cudaMalloc( (void**)&dev_k3_n_Minus, size * sizeof( cuDoubleComplex ) ), "malloc dev_k3_n_Minus" );
    CHECK_CUDA_ERROR( cudaMalloc( (void**)&dev_k4_Psi_Plus, size * sizeof( cuDoubleComplex ) ), "malloc dev_k4_Psi_Plus" );
    CHECK_CUDA_ERROR( cudaMalloc( (void**)&dev_k4_Psi_Minus, size * sizeof( cuDoubleComplex ) ), "malloc dev_k4_Psi_Minus" );
    CHECK_CUDA_ERROR( cudaMalloc( (void**)&dev_k4_n_Plus, size * sizeof( cuDoubleComplex ) ), "malloc dev_k4_n_Plus" );
    CHECK_CUDA_ERROR( cudaMalloc( (void**)&dev_k4_n_Minus, size * sizeof( cuDoubleComplex ) ), "malloc dev_k4_n_Minus" );
    CHECK_CUDA_ERROR( cudaMalloc( (void**)&dev_fft_plus, size * sizeof( cuDoubleComplex ) ), "malloc dev_fft_plus" );
    CHECK_CUDA_ERROR( cudaMalloc( (void**)&dev_fft_minus, size * sizeof( cuDoubleComplex ) ), "malloc dev_fft_minus" );
    
    CHECK_CUDA_ERROR( cufftPlan2d( &plan, s_N, s_N, CUFFT_Z2Z ), "FFT Plan" );
}

void setDeviceArrays( Scalar* psi_plus, Scalar* psi_minus, Scalar* n_plus, Scalar* n_minus, const int s_N ) {
    const auto size = s_N * s_N;
    if ( psi_plus != nullptr )
        CHECK_CUDA_ERROR( cudaMemcpy( dev_current_Psi_Plus, psi_plus, size * sizeof( cuDoubleComplex ), cudaMemcpyHostToDevice ), "memcopy host to device psi_plus" );
    if ( psi_minus != nullptr )
        CHECK_CUDA_ERROR( cudaMemcpy( dev_current_Psi_Minus, psi_minus, size * sizeof( cuDoubleComplex ), cudaMemcpyHostToDevice ), "memcopy host to device psi_minus" );
    if ( n_plus != nullptr )
        CHECK_CUDA_ERROR( cudaMemcpy( dev_current_n_Plus, n_plus, size * sizeof( cuDoubleComplex ), cudaMemcpyHostToDevice ), "memcopy host to device n_plus" );
    if ( n_minus != nullptr )
        CHECK_CUDA_ERROR( cudaMemcpy( dev_current_n_Minus, n_minus, size * sizeof( cuDoubleComplex ), cudaMemcpyHostToDevice ), "memcopy host to device n_minus" );
}

void getDeviceArrays( Scalar* psi_plus, Scalar* psi_minus, Scalar* n_plus, Scalar* n_minus, Scalar* fft_plus, Scalar* fft_minus, const int s_N ) {
    const auto size = s_N * s_N;
    if ( psi_plus != nullptr )
        CHECK_CUDA_ERROR( cudaMemcpy( psi_plus, dev_current_Psi_Plus, size * sizeof( cuDoubleComplex ), cudaMemcpyDeviceToHost ), "memcpy device to host psi_plus" );
    if ( psi_minus != nullptr )
        CHECK_CUDA_ERROR( cudaMemcpy( psi_minus, dev_current_Psi_Minus, size * sizeof( cuDoubleComplex ), cudaMemcpyDeviceToHost ), "memcpy device to host psi_minus" );
    if ( n_plus != nullptr )
        CHECK_CUDA_ERROR( cudaMemcpy( n_plus, dev_current_n_Plus, size * sizeof( cuDoubleComplex ), cudaMemcpyDeviceToHost ), "memcpy device to host n_plus" );
    if ( n_minus != nullptr )
        CHECK_CUDA_ERROR( cudaMemcpy( n_minus, dev_current_n_Minus, size * sizeof( cuDoubleComplex ), cudaMemcpyDeviceToHost ), "memcpy device to host n_minus" );
    if ( fft_plus != nullptr )
        CHECK_CUDA_ERROR( cudaMemcpy( fft_plus, dev_fft_plus, size * sizeof( cuDoubleComplex ), cudaMemcpyDeviceToHost ), "memcpy device to host fft_plus" );
    if ( fft_minus != nullptr )
        CHECK_CUDA_ERROR( cudaMemcpy( fft_minus, dev_fft_minus, size * sizeof( cuDoubleComplex ), cudaMemcpyDeviceToHost ), "memcpy device to host fft_minus" );
}

void freeDeviceArrays() {
    for ( const auto pointer : { dev_current_Psi_Plus, dev_current_Psi_Minus, dev_current_n_Plus, dev_current_n_Minus, dev_next_Psi_Plus, dev_next_Psi_Minus, dev_next_n_Plus, dev_next_n_Minus, dev_k1_Psi_Plus, dev_k1_Psi_Minus, dev_k1_n_Plus, dev_k1_n_Minus, dev_k2_Psi_Plus, dev_k2_Psi_Minus, dev_k2_n_Plus, dev_k2_n_Minus, dev_k3_Psi_Plus, dev_k3_Psi_Minus, dev_k3_n_Plus, dev_k3_n_Minus, dev_k4_Psi_Plus, dev_k4_Psi_Minus, dev_k4_n_Plus, dev_k4_n_Minus, dev_fft_plus, dev_fft_minus } ) {
        CHECK_CUDA_ERROR( cudaFree( pointer ), "free" );
    }
    for ( const auto pointer : { dev_pump_amp, dev_pump_width, dev_pump_X, dev_pump_Y, dev_pulse_t0, dev_pulse_amp, dev_pulse_freq, dev_pulse_sigma, dev_pulse_width, dev_pulse_X, dev_pulse_Y } ) {
        CHECK_CUDA_ERROR( cudaFree( pointer ), "free" );
    }
    for ( const auto pointer : { dev_pump_pol, dev_pulse_m, dev_pulse_pol } ) {
        CHECK_CUDA_ERROR( cudaFree( pointer ), "free" );
    }
    CHECK_CUDA_ERROR( cufftDestroy( plan ), "FFT Destroy" );
}

// Helper variable for caching the current time for FFT evaluations.
double cached_t = 0.0;

/**
 * @brief Iterates the Runge-Kutta-Method on the GPU
 * Note, that all device arrays and variables have to be initialized at this point
 * @param evaluate_pulse If true, the pulse is evaluated at the current time step
 * @param t Current time, will be updated to t + dt
 * @param dt Time step, will be updated to the next time step
 * @param s_N Number of grid points in one dimension
 */
void rungeFuncIterative( System& system, bool evaluate_pulse ) {
    dim3 block_size( 16, 16 );
    dim3 grid_size( ( system.s_N + block_size.x ) / block_size.x, ( system.s_N + block_size.y ) / block_size.y );
    dim3 grid_size_half( ( system.s_N / 2 + block_size.x ) / block_size.x, ( system.s_N / 2 + block_size.y ) / block_size.y );

    // Iterate the Runge Function on the current Psi and Calculate K1
    rungeFuncKernel<<<grid_size, block_size>>>( system.t, dev_current_Psi_Plus, dev_current_Psi_Minus, dev_current_n_Plus, dev_current_n_Minus, dev_fft_plus, dev_fft_minus, dev_k1_Psi_Plus, dev_k1_Psi_Minus, dev_k1_n_Plus, dev_k1_n_Minus, dev_pump_amp, dev_pump_width, dev_pump_X, dev_pump_Y, dev_pump_pol, dev_pulse_t0, dev_pulse_amp, dev_pulse_freq, dev_pulse_sigma, dev_pulse_m, dev_pulse_pol, dev_pulse_width, dev_pulse_X, dev_pulse_Y, evaluate_pulse );
    CHECK_CUDA_ERROR( {}, "K1" );
    CHECK_CUDA_ERROR( cudaDeviceSynchronize(), "Sync" );

    // Sum K1 to get next_Psi_Plus, next_Psi_Minus, next_n_Plus, next_n_Minus
    rungeFuncSum<<<grid_size, block_size>>>( 0.5, dev_next_Psi_Plus, dev_next_Psi_Minus, dev_next_n_Plus, dev_next_n_Minus, dev_current_Psi_Plus, dev_current_Psi_Minus, dev_current_n_Plus, dev_current_n_Minus, dev_k1_Psi_Plus, dev_k1_Psi_Minus, dev_k1_n_Plus, dev_k1_n_Minus );
    CHECK_CUDA_ERROR( {}, "Sum(K1)" );
    CHECK_CUDA_ERROR( cudaDeviceSynchronize(), "Sync" );

    // Iterate the Runge Function on next_Psi and Calculate K2
    rungeFuncKernel<<<grid_size, block_size>>>( system.t + 0.5*system.dt, dev_next_Psi_Plus, dev_next_Psi_Minus, dev_next_n_Plus, dev_next_n_Minus, dev_fft_plus, dev_fft_minus, dev_k2_Psi_Plus, dev_k2_Psi_Minus, dev_k2_n_Plus, dev_k2_n_Minus, dev_pump_amp, dev_pump_width, dev_pump_X, dev_pump_Y, dev_pump_pol, dev_pulse_t0, dev_pulse_amp, dev_pulse_freq, dev_pulse_sigma, dev_pulse_m, dev_pulse_pol, dev_pulse_width, dev_pulse_X, dev_pulse_Y, evaluate_pulse );
    CHECK_CUDA_ERROR( {}, "K2" );
    CHECK_CUDA_ERROR( cudaDeviceSynchronize(), "Sync" );

    // Sum K2 to get next_Psi_Plus, next_Psi_Minus, next_n_Plus, next_n_Minus
    rungeFuncSum<<<grid_size, block_size>>>( 0.5, dev_next_Psi_Plus, dev_next_Psi_Minus, dev_next_n_Plus, dev_next_n_Minus, dev_current_Psi_Plus, dev_current_Psi_Minus, dev_current_n_Plus, dev_current_n_Minus, dev_k2_Psi_Plus, dev_k2_Psi_Minus, dev_k2_n_Plus, dev_k2_n_Minus );
    CHECK_CUDA_ERROR( {}, "Sum(K2)" );
    CHECK_CUDA_ERROR( cudaDeviceSynchronize(), "Sync" );

    // Iterate the Runge Function on next_Psi and Calculate K3
    rungeFuncKernel<<<grid_size, block_size>>>( system.t + 0.5*system.dt, dev_next_Psi_Plus, dev_next_Psi_Minus, dev_next_n_Plus, dev_next_n_Minus, dev_fft_plus, dev_fft_minus, dev_k3_Psi_Plus, dev_k3_Psi_Minus, dev_k3_n_Plus, dev_k3_n_Minus, dev_pump_amp, dev_pump_width, dev_pump_X, dev_pump_Y, dev_pump_pol, dev_pulse_t0, dev_pulse_amp, dev_pulse_freq, dev_pulse_sigma, dev_pulse_m, dev_pulse_pol, dev_pulse_width, dev_pulse_X, dev_pulse_Y, evaluate_pulse );
    CHECK_CUDA_ERROR( {}, "K3" );
    CHECK_CUDA_ERROR( cudaDeviceSynchronize(), "Sync" );

    // Sum K3 to get next_Psi_Plus, next_Psi_Minus, next_n_Plus, next_n_Minus
    rungeFuncSum<<<grid_size, block_size>>>( 1.0, dev_next_Psi_Plus, dev_next_Psi_Minus, dev_next_n_Plus, dev_next_n_Minus, dev_current_Psi_Plus, dev_current_Psi_Minus, dev_current_n_Plus, dev_current_n_Minus, dev_k3_Psi_Plus, dev_k3_Psi_Minus, dev_k3_n_Plus, dev_k3_n_Minus );
    CHECK_CUDA_ERROR( {}, "Sum(K3)" );
    CHECK_CUDA_ERROR( cudaDeviceSynchronize(), "Sync" );

    // Iterate the Runge Function on next_Psi and Calculate K4
    rungeFuncKernel<<<grid_size, block_size>>>( system.t + system.dt, dev_next_Psi_Plus, dev_next_Psi_Minus, dev_next_n_Plus, dev_next_n_Minus, dev_fft_plus, dev_fft_minus, dev_k4_Psi_Plus, dev_k4_Psi_Minus, dev_k4_n_Plus, dev_k4_n_Minus, dev_pump_amp, dev_pump_width, dev_pump_X, dev_pump_Y, dev_pump_pol, dev_pulse_t0, dev_pulse_amp, dev_pulse_freq, dev_pulse_sigma, dev_pulse_m, dev_pulse_pol, dev_pulse_width, dev_pulse_X, dev_pulse_Y, evaluate_pulse );
    CHECK_CUDA_ERROR( {}, "K4" );
    CHECK_CUDA_ERROR( cudaDeviceSynchronize(), "Sync" );

    // Calculate the final Runge Kutta sum, saving the result in dev_in_Psi
    rungeFuncFinal<<<grid_size, block_size>>>( dev_current_Psi_Plus, dev_current_Psi_Minus, dev_current_n_Plus, dev_current_n_Minus, dev_current_Psi_Plus, dev_current_Psi_Minus, dev_current_n_Plus, dev_current_n_Minus, dev_k1_Psi_Plus, dev_k1_Psi_Minus, dev_k1_n_Plus, dev_k1_n_Minus, dev_k2_Psi_Plus, dev_k2_Psi_Minus, dev_k2_n_Plus, dev_k2_n_Minus, dev_k3_Psi_Plus, dev_k3_Psi_Minus, dev_k3_n_Plus, dev_k3_n_Minus, dev_k4_Psi_Plus, dev_k4_Psi_Minus, dev_k4_n_Plus, dev_k4_n_Minus );
    CHECK_CUDA_ERROR( {}, "Final Sum" );
    CHECK_CUDA_ERROR( cudaDeviceSynchronize(), "Sync" );

    // Increase t. In this RK method, dt remains the same
    system.t = system.t + system.dt;
    // For statistical purposes, increase the iteration counter
    system.iteration++;

    // Test: Calculate the FFT of dev_current_Psi_Plus using cufft
    if (system.t - cached_t < system.fft_every)
        return;
    cached_t = system.t;
    CHECK_CUDA_ERROR( cufftExecZ2Z( plan, (cufftDoubleComplex*)dev_current_Psi_Plus, (cufftDoubleComplex*)dev_fft_plus, CUFFT_FORWARD ), "FFT Exec" );
    CHECK_CUDA_ERROR( cufftExecZ2Z( plan, (cufftDoubleComplex*)dev_current_Psi_Minus, (cufftDoubleComplex*)dev_fft_minus, CUFFT_FORWARD ), "FFT Exec" );
    CHECK_CUDA_ERROR( cudaDeviceSynchronize(), "Sync" );
    fftshift_2D<<<grid_size, block_size>>>( dev_fft_plus, dev_fft_minus, system.s_N / 2 );
    CHECK_CUDA_ERROR( cudaDeviceSynchronize(), "Sync" );
    CHECK_CUDA_ERROR( {}, "FFT Shift" );
    kernel_maskFFT<<<grid_size, block_size>>>( dev_fft_plus, dev_fft_minus, 6., 0.7, false );
    CHECK_CUDA_ERROR( {}, "FFT Filter" );
    CHECK_CUDA_ERROR( cudaDeviceSynchronize(), "Sync" );
    fftshift_2D<<<grid_size, block_size>>>( dev_fft_plus, dev_fft_minus, system.s_N / 2 );
    CHECK_CUDA_ERROR( cudaDeviceSynchronize(), "Sync" );
    //// Transform back.
    CHECK_CUDA_ERROR( cufftExecZ2Z( plan, dev_fft_plus, dev_current_Psi_Plus, CUFFT_INVERSE ), "iFFT Exec" );
    CHECK_CUDA_ERROR( cufftExecZ2Z( plan, dev_fft_minus, dev_current_Psi_Minus, CUFFT_INVERSE ), "iFFT Exec" );
    CHECK_CUDA_ERROR( cudaDeviceSynchronize(), "Sync" );
    // Shift FFT Once again for visualization
    fftshift_2D<<<grid_size, block_size>>>( dev_fft_plus, dev_fft_minus, system.s_N / 2 ); 
    // kernel_makeFFTVisible<<<grid_size, block_size>>>( dev_fft_plus, dev_fft_plus );
    // kernel_makeFFTVisible<<<grid_size, block_size>>>( dev_fft_minus, dev_fft_minus );
    // CHECK_CUDA_ERROR( {}, "FFT Vis" );
}