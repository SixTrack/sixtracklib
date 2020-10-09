/////////////////////////////////////////////////////////////////////////////
//
// FILE NAME
//   ErrorFunctions.c
//
//   02/19/2015, 08/18/2015
//
// AUTHORS
//   Hannes Bartosik, Adrian Oeftiger
//
// DESCRIPTION
//   Error functions
//
/////////////////////////////////////////////////////////////////////////////

#ifndef SIXTACKLIB_COMMON_BE_BEAMFIELDS_FADDEEVA_CERN_HEADER_H__
#define SIXTACKLIB_COMMON_BE_BEAMFIELDS_FADDEEVA_CERN_HEADER_H__

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus ) && !defined( _GPUCODE )
extern "C" {
#endif /* !defined( __cplusplus ) && !defined( _GPUCODE ) */

SIXTRL_FN SIXTRL_STATIC void cerrf(
    SIXTRL_REAL_T in_real, SIXTRL_REAL_T in_imag,
    SIXTRL_ARGPTR_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT out_real,
    SIXTRL_ARGPTR_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT out_imag );

#if defined( __cplusplus ) && !defined( _GPUCODE )
}
#endif /* !defined( __cplusplus ) && !defined( _GPUCODE ) */

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
	#include <math.h>
#endif

#if defined( __cplusplus ) && !defined( _GPUCODE )
extern "C" {
#endif /* !defined( __cplusplus ) && !defined( _GPUCODE ) */

/* From: be_beamfields/faddeeva_cern.h */
SIXTRL_INLINE void cerrf( SIXTRL_REAL_T in_real, SIXTRL_REAL_T in_imag,
    SIXTRL_ARGPTR_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT out_real,
    SIXTRL_ARGPTR_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT out_imag )
{
    /* This function calculates the SIXTRL_REAL_T precision complex error fnct.
    based on the algorithm of the FORTRAN function written at CERN by K. Koelbig
    Program C335, 1970. See also M. Bassetti and G.A. Erskine, "Closed
    expression for the electric field of a two-dimensional Gaussian charge
    density", CERN-ISR-TH/80-06; */

    int n, nc, nu;
    SIXTRL_REAL_T a_constant = 1.12837916709551;
    SIXTRL_REAL_T xLim = 5.33;
    SIXTRL_REAL_T yLim = 4.29;
    SIXTRL_REAL_T h, q, Saux, Sx, Sy, Tn, Tx, Ty, Wx, Wy, xh, xl, x, yh, y;
    SIXTRL_REAL_T Rx [33];
    SIXTRL_REAL_T Ry [33];

    x = fabs(in_real);
    y = fabs(in_imag);

    if (y < yLim && x < xLim){
        q = (1.0 - y / yLim) * sqrt(1.0 - (x / xLim) * (x / xLim));
        h  = 1.0 / (3.2 * q);
        nc = 7 + (int) (23.0 * q);
        xl = pow(h, (SIXTRL_REAL_T) (1 - nc));
        xh = y + 0.5 / h;
        yh = x;
        nu = 10 + (int) (21.0 * q);
        Rx[nu] = 0.;
        Ry[nu] = 0.;
        for (n = nu; n > 0; n--){
            Tx = xh + n * Rx[n];
            Ty = yh - n * Ry[n];
            Tn = Tx*Tx + Ty*Ty;
            Rx[n-1] = 0.5 * Tx / Tn;
            Ry[n-1] = 0.5 * Ty / Tn;
            }
        /* .... */
        Sx = 0.;
        Sy = 0.;
        for (n = nc; n>0; n--){
            Saux = Sx + xl;
            Sx = Rx[n-1] * Saux - Ry[n-1] * Sy;
            Sy = Rx[n-1] * Sy + Ry[n-1] * Saux;
            xl = h * xl;
        };
        Wx = a_constant * Sx;
        Wy = a_constant * Sy;
    }
    else{
        xh = y;
        yh = x;
        Rx[0] = 0.;
        Ry[0] = 0.;
        for (n = 9; n>0; n--){
            Tx = xh + n * Rx[0];
            Ty = yh - n * Ry[0];
            Tn = Tx * Tx + Ty * Ty;
            Rx[0] = 0.5 * Tx / Tn;
            Ry[0] = 0.5 * Ty / Tn;
        };
        Wx = a_constant * Rx[0];
        Wy = a_constant * Ry[0];
    }
    if (y == 0.) {Wx = exp(-x * x);}
    if (in_imag < 0.){
        Wx =   2.0 * exp(y * y - x * x) * cos(2.0 * x * y) - Wx;
        Wy = - 2.0 * exp(y * y - x * x) * sin(2.0 * x * y) - Wy;
        if (in_real > 0.) {Wy = -Wy;}
    }
    else if (in_real < 0.) {Wy = -Wy;}

    *out_real = Wx;
    *out_imag = Wy;
}

#if defined( __cplusplus ) && !defined( _GPUCODE )
}
#endif /* !defined( __cplusplus ) && !defined( _GPUCODE ) */

#endif /* SIXTACKLIB_COMMON_BE_BEAMFIELDS_FADDEEVA_CERN_HEADER_H__ */

/* end: sixtracklib/common/be_beamfields/faddeeva_cern.h */
