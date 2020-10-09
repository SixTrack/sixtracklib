#ifndef SIXTRACKLIB_COMMON_INTERNAL_MATH_CONSTANTS_H__
#define SIXTRACKLIB_COMMON_INTERNAL_MATH_CONSTANTS_H__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #if defined( __cplusplus )
        #include <cstddef>
        #include <cstdint>
        #include <cstdlib>
        #include <cmath>
        #include <type_traits>
    #else /* defined( __cplusplus ) */
        #include <stddef.h>
        #include <stdint.h>
        #include <stdlib.h>
        #include <math.h>
    #endif /* defined( __cplusplus ) */
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/particles/definitions.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( SIXTRL_MATH_CONST_PI )
    #define SIXTRL_MATH_CONST_PI \
        3.1415926535897932384626433832795028841971693993751L
#endif /* !defined( SIXTRL_MATH_CONST_PI ) */

#if !defined( SIXTRL_MATH_CONST_DEG2RAD )
    #define SIXTRL_MATH_CONST_DEG2RAD \
        0.0174532925199432957692369076848861271344287188854172546L
#endif /* !defined( SIXTRL_MATH_CONST_DEG2RAD ) */

#if !defined( SIXTRL_MATH_CONST_RAD2DEG )
    #define SIXTRL_MATH_CONST_RAD2DEG \
        57.29577951308232087679815481410517033240547246656432154916L
#endif /* !defined( SIXTRL_MATH_CONST_RAD2DEG ) */

#if !defined( SIXTRL_MATH_CONST_SQRT_PI )
    #define SIXTRL_MATH_CONST_SQRT_PI \
        1.77245385090551602729816748334114518279754945612238712821381L
#endif /* !defined( SIXTRL_MATH_CONST_SQRT_PI ) */

#if defined( __cplusplus )
#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/internal/type_store_traits.hpp"
    #include "sixtracklib/common/internal/type_comparison_helpers.hpp"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

namespace SIXTRL_CXX_NAMESPACE
{
    template< class R, typename SFINAE_Enabled = void >
    struct MathConstHelper
    {
        SIXTRL_STATIC SIXTRL_INLINE SIXTRL_FN R get_pi() SIXTRL_NOEXCEPT_COND(
            std::is_nothrow_copy_constructible< R >::value &&
            std::is_nothrow_move_constructible< R >::value )
        {
            return static_cast< R >( SIXTRL_MATH_CONST_PI );
        }

        SIXTRL_STATIC SIXTRL_INLINE SIXTRL_FN R get_deg2rad() SIXTRL_NOEXCEPT_COND(
            std::is_nothrow_copy_constructible< R >::value &&
            std::is_nothrow_move_constructible< R >::value )
        {
            return static_cast< R >( SIXTRL_MATH_CONST_DEG2RAD );
        }

        SIXTRL_STATIC SIXTRL_INLINE R get_rad2deg() SIXTRL_NOEXCEPT_COND(
            std::is_nothrow_copy_constructible< R >::value &&
            std::is_nothrow_move_constructible< R >::value )
        {
            return static_cast< R >( SIXTRL_MATH_CONST_RAD2DEG );
        }

        SIXTRL_STATIC SIXTRL_INLINE SIXTRL_FN R get_sqrt_pi() SIXTRL_NOEXCEPT_COND(
            std::is_nothrow_copy_constructible< R >::value &&
            std::is_nothrow_move_constructible< R >::value )
        {
            return static_cast< R >( SIXTRL_MATH_CONST_SQRT_PI );
        }
    };

    /* --------------------------------------------------------------------- */

    template< class R >
    SIXTRL_STATIC SIXTRL_INLINE SIXTRL_FN R MathConst_pi()
        SIXTRL_NOEXCEPT_COND(
            std::is_nothrow_copy_constructible< R >::value &&
            std::is_nothrow_move_constructible< R >::value )
    {
        return MathConstHelper< R >::get_pi();
    }

    template< class R >
    SIXTRL_STATIC SIXTRL_INLINE SIXTRL_FN R MathConst_deg2rad()
        SIXTRL_NOEXCEPT_COND(
            std::is_nothrow_copy_constructible< R >::value &&
            std::is_nothrow_move_constructible< R >::value )
    {
        return MathConstHelper< R >::get_deg2rad();
    }

    template< class R >
    SIXTRL_STATIC SIXTRL_INLINE SIXTRL_FN R MathConst_rad2deg()
        SIXTRL_NOEXCEPT_COND(
            std::is_nothrow_copy_constructible< R >::value &&
            std::is_nothrow_move_constructible< R >::value )
    {
        return MathConstHelper< R >::get_rad2deg();
    }

    template< class R >
    SIXTRL_STATIC SIXTRL_INLINE SIXTRL_FN R MathConst_sqrt_pi()
        SIXTRL_NOEXCEPT_COND(
            std::is_nothrow_copy_constructible< R >::value &&
            std::is_nothrow_move_constructible< R >::value )
    {
        return MathConstHelper< R >::get_sqrt_pi();
    }
}

template< class R >
SIXTRL_STATIC SIXTRL_INLINE SIXTRL_FN R NS(MathConst_pi)()
{
    return SIXTRL_CXX_NAMESPACE::MathConst_pi< R >();
}

template< class R >
SIXTRL_STATIC SIXTRL_INLINE SIXTRL_FN R NS(MathConst_deg2rad)()
{
    return SIXTRL_CXX_NAMESPACE::MathConst_deg2rad< R >();
}

template< class R >
SIXTRL_STATIC SIXTRL_INLINE SIXTRL_FN R NS(MathConst_rad2deg)()
{
    return SIXTRL_CXX_NAMESPACE::MathConst_rad2deg< R >();
}

template< class R >
SIXTRL_STATIC SIXTRL_INLINE SIXTRL_FN R NS(MathConst_sqrt_pi)()
{
    return SIXTRL_CXX_NAMESPACE::MathConst_sqrt_pi< R >();
}

#endif /* C++ */

#if defined( __cplusplus ) && !defined( _GPUCODE )
extern "C" {
#endif /* defined( __cplusplus ) && !defined( _GPUCODE ) */

SIXTRL_STATIC SIXTRL_FN SIXTRL_REAL_T NS(MathConst_pi)( void ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_REAL_T
NS(MathConst_deg2rad)( void ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_REAL_T
    NS(MathConst_rad2deg)( void ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_REAL_T
    NS(MathConst_sqrt_pi)( void ) SIXTRL_NOEXCEPT;

#if defined( __cplusplus ) && !defined( _GPUCODE )
}
#endif /* C++, Host */

/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */
/* !!!!!!!        Inline Methods and Functions Implementations       !!!!!!!! */
/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */

#if defined( __cplusplus ) && !defined( _GPUCODE )
extern "C" {
#endif /* defined( __cplusplus ) && !defined( _GPUCODE ) */

SIXTRL_INLINE SIXTRL_REAL_T NS(MathConst_pi)( void ) SIXTRL_NOEXCEPT
{
    return ( SIXTRL_REAL_T )SIXTRL_MATH_CONST_PI;
}

SIXTRL_INLINE SIXTRL_REAL_T NS(MathConst_deg2rad)( void ) SIXTRL_NOEXCEPT
{
    return ( SIXTRL_REAL_T )SIXTRL_MATH_CONST_DEG2RAD;
}

SIXTRL_INLINE SIXTRL_REAL_T NS(MathConst_rad2deg)( void ) SIXTRL_NOEXCEPT
{
    return ( SIXTRL_REAL_T )SIXTRL_MATH_CONST_RAD2DEG;
}

SIXTRL_INLINE SIXTRL_REAL_T NS(MathConst_sqrt_pi)( void ) SIXTRL_NOEXCEPT
{
    return ( SIXTRL_REAL_T )SIXTRL_MATH_CONST_SQRT_PI;
}

#if defined( __cplusplus ) && !defined( _GPUCODE )
}
#endif /* defined( __cplusplus ) && !defined( _GPUCODE ) */

#endif /* SIXTRACKLIB_COMMON_INTERNAL_MATH_CONSTANTS_H__ */
