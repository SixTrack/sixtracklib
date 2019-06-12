#ifndef SIXTRACKLIB_COMMON_CONSTANTS_H__
#define SIXTRACKLIB_COMMON_CONSTANTS_H__

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

/* ------------------------------------------------------------------------- */

#if !defined( SIXTRL_C_LIGHT )
    #define   SIXTRL_C_LIGHT ( 299792458.0 )
#endif /* !defined( C_LIGHT ) */

#if !defined( SIXTRL_EPSILON_0 )
    #define   SIXTRL_EPSILON_0 (8.854187817620e-12)
#endif /* !defined( SIXTRL_EPSILON_0 ) */

#if !defined( SIXTRL_PI )
    #define SIXTRL_PI (3.1415926535897932384626433832795028841971693993751)
#endif /* !defined( SIXTRL_PI ) */

#if !defined( SIXTRL_DEG2RAD )
    #define SIXTRL_DEG2RAD (0.0174532925199432957692369076848861271344287188854)
#endif /* !defiend( SIXTRL_DEG2RAD ) */

#if !defined( SIXTRL_RAD2DEG )
    #define SIXTRL_RAD2DEG (57.29577951308232087679815481410517033240547246656442)
#endif /* !defiend( SIXTRL_RAD2DEG ) */

#if !defined( SIXTRL_SQRT_PI )
    #define SIXTRL_SQRT_PI (1.7724538509055160272981674833411451827975494561224)
#endif /* !defined( SIXTRL_SQRT_PI ) */

#if !defined( SIXTRL_QELEM )
    #define SIXTRL_QELEM (1.60217662e-19)
#endif /* !defined( SIXTRL_QELEM ) */

/* ------------------------------------------------------------------------- */

#if !defined( _GPUCODE ) || defined( __CUDACC__ )

#if defined( __cplusplus )
extern "C" {
#endif /* defined( __cplusplus ) */

SIXTRL_STATIC_VAR SIXTRL_REAL_T const NS(C_LIGHT) = 
    ( SIXTRL_REAL_T )SIXTRL_C_LIGHT;
    
SIXTRL_STATIC_VAR SIXTRL_REAL_T const NS(EPSILON_0) = 
    ( SIXTRL_REAL_T )SIXTRL_EPSILON_0;
    
SIXTRL_STATIC_VAR SIXTRL_REAL_T const NS(PI) = 
    ( SIXTRL_REAL_T )SIXTRL_PI;

SIXTRL_STATIC_VAR SIXTRL_REAL_T const NS(DEG2RAD) =
    ( SIXTRL_REAL_T )SIXTRL_DEG2RAD;
    
SIXTRL_STATIC_VAR SIXTRL_REAL_T const NS(RAD2DEG) =
    ( SIXTRL_REAL_T )SIXTRL_RAD2DEG;

SIXTRL_STATIC_VAR SIXTRL_REAL_T const NS(SQRT_PI) = 
    ( SIXTRL_REAL_T )SIXTRL_SQRT_PI;
    
SIXTRL_STATIC_VAR SIXTRL_REAL_T const NS(QELEM) = 
    ( SIXTRL_REAL_T )SIXTRL_QELEM;

#if defined( __cplusplus )
}
#endif /* defined( __cplusplus ) */

#endif /* !defined( _GPUCODE ) || defined( __CUDACC__ ) */

#if defined( __cplusplus )

namespace SIXTRL_CXX_NAMESPACE
{
    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST
        SIXTRL_REAL_T C_LIGHT = SIXTRL_C_LIGHT;

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST
        SIXTRL_REAL_T EPSILON_0 = SIXTRL_EPSILON_0;

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST
        SIXTRL_REAL_T PI = SIXTRL_PI;
        
    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST 
        SIXTRL_REAL_T DEG2RAD = SIXTRL_DEG2RAD;
    
    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST 
        SIXTRL_REAL_T RAD2DEG = SIXTRL_RAD2DEG;

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST
        SIXTRL_REAL_T SQRT_PI = SIXTRL_SQRT_PI;

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST
        SIXTRL_REAL_T QELEM = SIXTRL_QELEM;
}

#endif /* defined( __cplusplus ) */

#endif /* SIXTRACKLIB_COMMON_CONSTANTS_H__ */

/* end: sixtracklib/common/constants.h */
