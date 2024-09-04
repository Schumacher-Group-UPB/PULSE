#pragma once
#include "cuda/typedef.cuh"

/*
* This file contains the coefficients for the Runge-Kutta Dormand-Prince method
*/
namespace RKCoefficients {

// RK 45 coefficients (Dormand–Prince method)
struct DP {

    static constexpr float a2 = 1.f / 5.f;
    static constexpr float a3 = 3.f / 10.f;
    static constexpr float a4 = 4.f / 5.f;
    static constexpr float a5 = 8.f / 9.f;
    static constexpr float a6 = 1.0f;
    static constexpr float a7 = 1.0f;

    static constexpr float b11 = 1.f / 5.f;
    static constexpr float b21 = 3.f / 40.f;
    static constexpr float b31 = 44.f / 45.f;
    static constexpr float b41 = 19372.f / 6561.f;
    static constexpr float b51 = 9017.f / 3168.f;
    static constexpr float b61 = 35.f / 384.f;
    static constexpr float b22 = 9.f / 40.f;
    static constexpr float b32 = -56.f / 15.f;
    static constexpr float b42 = -25360.f / 2187.f;
    static constexpr float b52 = -355.f / 33.f;
    static constexpr float b62 = 0.0f;
    static constexpr float b33 = 32.f / 9.f;
    static constexpr float b43 = 64448.f / 6561.f;
    static constexpr float b53 = 46732.f / 5247.f;
    static constexpr float b63 = 500.f / 1113.f;
    static constexpr float b44 = -212.f / 729.f;
    static constexpr float b54 = 49.f / 176.f;
    static constexpr float b64 = 125.f / 192.f;
    static constexpr float b55 = -5103.f / 18656.f;
    static constexpr float b65 = -2187.f / 6784.f;
    static constexpr float b66 = 11.f / 84.f;



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
    static constexpr float e1 = b61 - 5179.f / 57600.f;
    static constexpr float e2 = 0.0f;
    static constexpr float e3 = b63 - 7571.f / 16695.f;   
    static constexpr float e4 = b64 - 393.f / 640.f;      
    static constexpr float e5 = b65 - -92097.f / 339200.f;
    static constexpr float e6 = b66 - 187.f / 2100.f;     
    static constexpr float e7 = -1.f / 40.f;         

};
} // namespace RKCoefficients