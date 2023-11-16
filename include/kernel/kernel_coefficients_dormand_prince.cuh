#pragma once
#include "cuda/cuda_complex.cuh"

/*
* This file contains the coefficients for the Runge-Kutta Dormand-Prince method
* The coefficients are constexpr and known at compile time
* The coefficients are not external because this header file is only included once.
*/
namespace RKCoefficients {

// RK 45 coefficients (Dormand–Prince method)
CUDA_DEVICE constexpr real_number a2 = 1. / 5.;
CUDA_DEVICE constexpr real_number a3 = 3. / 10.;
CUDA_DEVICE constexpr real_number a4 = 4. / 5.;
CUDA_DEVICE constexpr real_number a5 = 8. / 9.;
CUDA_DEVICE constexpr real_number a6 = 1.0;
CUDA_DEVICE constexpr real_number a7 = 1.0;

CUDA_DEVICE constexpr real_number b11 = 1. / 5.;
CUDA_DEVICE constexpr real_number b21 = 3. / 40.;
CUDA_DEVICE constexpr real_number b31 = 44. / 45.;
CUDA_DEVICE constexpr real_number b41 = 19372. / 6561.;
CUDA_DEVICE constexpr real_number b51 = 9017. / 3168.;
CUDA_DEVICE constexpr real_number b61 = 35. / 384.;
CUDA_DEVICE constexpr real_number b22 = 9. / 40.;
CUDA_DEVICE constexpr real_number b32 = -56. / 15.;
CUDA_DEVICE constexpr real_number b42 = -25360. / 2187.;
CUDA_DEVICE constexpr real_number b52 = -355. / 33.;
CUDA_DEVICE constexpr real_number b33 = 32. / 9.;
CUDA_DEVICE constexpr real_number b43 = 64448. / 6561.;
CUDA_DEVICE constexpr real_number b53 = 46732. / 5247.;
CUDA_DEVICE constexpr real_number b63 = 500. / 1113.;
CUDA_DEVICE constexpr real_number b44 = -212. / 729.;
CUDA_DEVICE constexpr real_number b54 = 49. / 176.;
CUDA_DEVICE constexpr real_number b64 = 125. / 192.;
CUDA_DEVICE constexpr real_number b55 = -5103 / 18656.;
CUDA_DEVICE constexpr real_number b65 = -2187. / 6784.;
CUDA_DEVICE constexpr real_number b66 = 11. / 84.;



/**
 * The First Psi Delta Matrix is calculated using 
 * delta_psi_1 = b61 * k1 + b63 * k3 + b64 * k4 + b65 * k5 + b66 * k6;
 * The Second Psi Delta Matrix is calculated using
 * delta_psi_2 = e1*k1 + e3*k3 + e4*k4 + e5*k5 + e6*k6 + e7*k7
 * The RK Error is then calculated by using
 * error = sum( abs2( delta_psi_1 - delta_psi_2 ) )
 * Directly evaluating this expression at compile time yields
 * error = sum( abs2( (b61 - e1)*k1 + (b63 - e2)*k3 + (b64 - e4)*k4 + (b65 - e5)*k5 + (b66 - e6)*k6 ) + e7*k7 )
 * which can be evaluated directly by the RKError Kernel
*/
CUDA_DEVICE constexpr real_number e1 = b61 - 5179. / 57600.;
CUDA_DEVICE constexpr real_number e3 = b63 - 7571. / 16695.;   
CUDA_DEVICE constexpr real_number e4 = b64 - 393. / 640.;      
CUDA_DEVICE constexpr real_number e5 = b65 - -92097. / 339200.;
CUDA_DEVICE constexpr real_number e6 = b66 - 187. / 2100.;     
CUDA_DEVICE constexpr real_number e7 = 1. / 40.;         
} // namespace RKCoefficients