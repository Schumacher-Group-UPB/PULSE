#pragma once
#include "cuda/typedef.cuh"

/*
* This file contains the coefficients for the Runge-Kutta Dormand-Prince method
*/
namespace RKCoefficients {

// RK 45 coefficients (Dormand–Prince method)
struct DP {

    const PC3::Type::real a2 = 1. / 5.;
    const PC3::Type::real a3 = 3. / 10.;
    const PC3::Type::real a4 = 4. / 5.;
    const PC3::Type::real a5 = 8. / 9.;
    const PC3::Type::real a6 = 1.0;
    const PC3::Type::real a7 = 1.0;

    const PC3::Type::real b11 = 1. / 5.;
    const PC3::Type::real b21 = 3. / 40.;
    const PC3::Type::real b31 = 44. / 45.;
    const PC3::Type::real b41 = 19372. / 6561.;
    const PC3::Type::real b51 = 9017. / 3168.;
    const PC3::Type::real b61 = 35. / 384.;
    const PC3::Type::real b22 = 9. / 40.;
    const PC3::Type::real b32 = -56. / 15.;
    const PC3::Type::real b42 = -25360. / 2187.;
    const PC3::Type::real b52 = -355. / 33.;
    const PC3::Type::real b62 = 0.0;
    const PC3::Type::real b33 = 32. / 9.;
    const PC3::Type::real b43 = 64448. / 6561.;
    const PC3::Type::real b53 = 46732. / 5247.;
    const PC3::Type::real b63 = 500. / 1113.;
    const PC3::Type::real b44 = -212. / 729.;
    const PC3::Type::real b54 = 49. / 176.;
    const PC3::Type::real b64 = 125. / 192.;
    const PC3::Type::real b55 = -5103 / 18656.;
    const PC3::Type::real b65 = -2187. / 6784.;
    const PC3::Type::real b66 = 11. / 84.;



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
    const PC3::Type::real e1 = b61 - 5179. / 57600.;
    const PC3::Type::real e2 = 0.0;
    const PC3::Type::real e3 = b63 - 7571. / 16695.;   
    const PC3::Type::real e4 = b64 - 393. / 640.;      
    const PC3::Type::real e5 = b65 - -92097. / 339200.;
    const PC3::Type::real e6 = b66 - 187. / 2100.;     
    const PC3::Type::real e7 = -1. / 40.;         

};
} // namespace RKCoefficients