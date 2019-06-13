#ifndef SIXTRACKLIB_COMMON_BE_LIMIT_DEFINTIONS_C99_H__
#define SIXTRACKLIB_COMMON_BE_LIMIT_DEFINTIONS_C99_H__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stddef.h>
    #include <stdint.h>
    #include <stdlib.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/generated/config.h"
    #include "sixtracklib/common/particles.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined(  _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

#if !defined( SIXTRL_LIMIT_DEFAULT_MAX_X )
    #define SIXTRL_LIMIT_DEFAULT_MAX_X SIXTRL_APERTURE_X_LIMIT
#endif /* !defined( SIXTRL_LIMIT_DEFAULT_MAX_X ) */

#if !defined( SIXTRL_LIMIT_DEFAULT_MIN_X )
    #define SIXTRL_LIMIT_DEFAULT_MIN_X -SIXTRL_APERTURE_X_LIMIT
#endif /* !defined( SIXTRL_LIMIT_DEFAULT_MIN_X ) */

#if !defined( SIXTRL_LIMIT_DEFAULT_MAX_Y )
    #define SIXTRL_LIMIT_DEFAULT_MAX_Y SIXTRL_APERTURE_Y_LIMIT
#endif /* !defined( SIXTRL_LIMIT_DEFAULT_MAX_Y ) */

#if !defined( SIXTRL_LIMIT_DEFAULT_MIN_Y )
    #define SIXTRL_LIMIT_DEFAULT_MIN_Y -SIXTRL_APERTURE_Y_LIMIT
#endif /* !defined( SIXTRL_LIMIT_DEFAULT_MIN_Y ) */
    
#if !defined( _GPUCODE ) 

SIXTRL_STATIC_VAR NS(particle_real_t) const NS(LIMIT_DEFAULT_MIN_X) = (
    NS(particle_real_t) )SIXTRL_LIMIT_DEFAULT_MIN_X;

SIXTRL_STATIC_VAR NS(particle_real_t) const NS(LIMIT_DEFAULT_MAX_X) = (
    NS(particle_real_t) )SIXTRL_LIMIT_DEFAULT_MAX_X;

SIXTRL_STATIC_VAR NS(particle_real_t) const NS(LIMIT_DEFAULT_MIN_Y) = (
    NS(particle_real_t) )SIXTRL_LIMIT_DEFAULT_MIN_Y;

SIXTRL_STATIC_VAR NS(particle_real_t) const NS(LIMIT_DEFAULT_MAX_Y) = (
    NS(particle_real_t) )SIXTRL_LIMIT_DEFAULT_MAX_Y;  
    
    
    
SIXTRL_STATIC_VAR NS(particle_real_t) const NS(LIMIT_DEFAULT_X_ORIGIN) =
    ( NS(particle_real_t) )( 0.5 * ( 
        SIXTRL_LIMIT_DEFAULT_MAX_X + SIXTRL_LIMIT_DEFAULT_MIN_X ) );

SIXTRL_STATIC_VAR NS(particle_real_t) const NS(LIMIT_DEFAULT_Y_ORIGIN) =
    ( NS(particle_real_t) )( 0.5 * ( 
        SIXTRL_LIMIT_DEFAULT_MAX_Y + SIXTRL_LIMIT_DEFAULT_MIN_Y ) );
    
SIXTRL_STATIC_VAR NS(particle_real_t) const NS(LIMIT_DEFAULT_X_HALF_AXIS) =
    ( NS(particle_real_t) )( 0.5 * ( 
        SIXTRL_LIMIT_DEFAULT_MAX_X - SIXTRL_LIMIT_DEFAULT_MIN_X ) );

SIXTRL_STATIC_VAR NS(particle_real_t) const NS(LIMIT_DEFAULT_Y_HALF_AXIS) =
    ( NS(particle_real_t) )( 0.5 * ( 
        SIXTRL_LIMIT_DEFAULT_MAX_Y - SIXTRL_LIMIT_DEFAULT_MIN_Y ) );
    
#endif /* !defined( _GPUCODE ) */
    
#if defined( __cplusplus ) && !defined( _GPUCODE )
    
namespace SIXTRL_CXX_NAMESPACE
{
    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST NS(particle_real_t)
        LIMIT_DEFAULT_MAX_X = static_cast< NS(particle_real_t) >(
            SIXTRL_LIMIT_DEFAULT_MAX_X );
    
    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST NS(particle_real_t)
        LIMIT_DEFAULT_MIN_X = static_cast< NS(particle_real_t) >(
            SIXTRL_LIMIT_DEFAULT_MIN_X );
        
    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST NS(particle_real_t)
        LIMIT_DEFAULT_MAX_Y = static_cast< NS(particle_real_t) >(
            SIXTRL_LIMIT_DEFAULT_MAX_Y );
    
    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST NS(particle_real_t)
        LIMIT_DEFAULT_MIN_Y = static_cast< NS(particle_real_t) >(
            SIXTRL_LIMIT_DEFAULT_MIN_Y );
        
        
    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST NS(particle_real_t) 
        LIMIT_DEFAULT_X_ORIGIN = static_cast< NS(particle_real_t) >( 0.5 * ( 
            SIXTRL_LIMIT_DEFAULT_MAX_X + SIXTRL_LIMIT_DEFAULT_MIN_X ) );

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST NS(particle_real_t) 
        LIMIT_DEFAULT_Y_ORIGIN = static_cast< NS(particle_real_t) >( 0.5 * 
            ( SIXTRL_LIMIT_DEFAULT_MAX_Y + SIXTRL_LIMIT_DEFAULT_MIN_Y ) );
        
    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST NS(particle_real_t) 
        LIMIT_DEFAULT_X_HALF_AXIS = static_cast< NS(particle_real_t) >( 0.5 *  
            ( SIXTRL_LIMIT_DEFAULT_MAX_X - SIXTRL_LIMIT_DEFAULT_MIN_X ) );

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST NS(particle_real_t) 
        LIMIT_DEFAULT_Y_HALF_AXIS = static_cast< NS(particle_real_t) >( 0.5 *  
            ( SIXTRL_LIMIT_DEFAULT_MAX_Y - SIXTRL_LIMIT_DEFAULT_MIN_Y ) );
}

#endif /* C++, host */

#if !defined(  _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

#endif /* SIXTRACKLIB_COMMON_BE_LIMIT_DEFINTIONS_C99_H__ */
/* end: sixtracklib/common/be_limit/definitions.h */
