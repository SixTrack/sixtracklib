#ifndef SIXTRACKLIB_COMMON_IMPL_BEAM_ELEMENT_SROTATION_H__
#define SIXTRACKLIB_COMMON_IMPL_BEAM_ELEMENT_SROTATION_H__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stdbool.h>
    #include <stdint.h>
    #include <stdlib.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/_impl/definitions.h"
    #include "sixtracklib/common/impl/buffer_type.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined(  _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

/* ========================================================================= */

typedef struct NS(SRotation)
{
    SIXTRL_REAL_T cos_z SIXTRL_ALIGN( 8 );
    SIXTRL_REAL_T sin_z SIXTRL_ALIGN( 8 );
}
NS(SRotation);

/* ------------------------------------------------------------------------- */

SIXTRL_FN SIXTRL_STATIC NS(buffer_size_t) NS(SRotation_get_num_dataptrs)(
    SIXTRL_ARGPTR_DEC const NS(SRotation) *const SIXTRL_RESTRICT srotation );

SIXTRL_FN SIXTRL_STATIC NS(SRotation)* NS(SRotation_preset)(
    SIXTRL_ARGPTR_DEC NS(SRotation)* SIXTRL_RESTRICT srotation );

SIXTRL_FN SIXTRL_STATIC SIXTRL_REAL_T NS(SRotation_get_angle_deg)(
    SIXTRL_ARGPTR_DEC const NS(SRotation) *const SIXTRL_RESTRICT srotation );

SIXTRL_FN SIXTRL_STATIC SIXTRL_REAL_T NS(SRotation_get_angle)(
    SIXTRL_ARGPTR_DEC const NS(SRotation) *const SIXTRL_RESTRICT srotation );

SIXTRL_FN SIXTRL_STATIC SIXTRL_REAL_T NS(SRotation_get_cos_angle)(
    SIXTRL_ARGPTR_DEC const NS(SRotation) *const SIXTRL_RESTRICT srotation );

SIXTRL_FN SIXTRL_STATIC SIXTRL_REAL_T NS(SRotation_get_sin_angle)(
    SIXTRL_ARGPTR_DEC const NS(SRotation) *const SIXTRL_RESTRICT srotation );

SIXTRL_FN SIXTRL_STATIC void NS(SRotation_set_angle)(
    SIXTRL_ARGPTR_DEC NS(SRotation)* SIXTRL_RESTRICT srotation,
    SIXTRL_REAL_T const angle );

SIXTRL_FN SIXTRL_STATIC void NS(SRotation_set_angle_deg)(
    SIXTRL_ARGPTR_DEC NS(SRotation)* SIXTRL_RESTRICT srotation,
    SIXTRL_REAL_T const angle_deg );

SIXTRL_FN SIXTRL_STATIC bool NS(SRotation_can_be_added)(
    SIXTRL_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_objects,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_slots,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_dataptrs );

SIXTRL_FN SIXTRL_STATIC SIXTRL_ARGPTR_DEC NS(SRotation)* NS(SRotation_new)(
    SIXTRL_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer );

SIXTRL_FN SIXTRL_STATIC SIXTRL_ARGPTR_DEC NS(SRotation)* NS(SRotation_add)(
    SIXTRL_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_REAL_T const angle );

SIXTRL_FN SIXTRL_STATIC SIXTRL_ARGPTR_DEC NS(SRotation)*
NS(SRotation_add_detailed)(
    SIXTRL_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_REAL_T const cos_z, SIXTRL_REAL_T const sin_z );

/* ========================================================================= */

#if !defined(  _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

/* ************************************************************************* */
/* *****          Implementation of C inline functions                   *** */
/* ************************************************************************* */

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <math.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/buffer.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined(  _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

SIXTRL_INLINE NS(buffer_size_t) NS(SRotation_get_num_dataptrs)(
    SIXTRL_ARGPTR_DEC const NS(SRotation) *const SIXTRL_RESTRICT srotation )
{
    return ( NS(buffer_size_t) )0u;
}

SIXTRL_INLINE NS(SRotation)* NS(SRotation_preset)(
    SIXTRL_ARGPTR_DEC NS(SRotation)* SIXTRL_RESTRICT srotation )
{
    if( srotation != SIXTRL_NULLPTR )
    {
        srotation->cos_z = ( SIXTRL_REAL_T )1;
        srotation->sin_z = ( SIXTRL_REAL_T )0;
    }

    return srotation;
}

SIXTRL_INLINE SIXTRL_REAL_T NS(SRotation_get_angle_deg)(
    SIXTRL_ARGPTR_DEC const NS(SRotation) *const SIXTRL_RESTRICT srotation )
{
    SIXTRL_REAL_T const RAD2DEG = ( SIXTRL_REAL_T )180.0 / M_PI;
    return RAD2DEG * NS(SRotation_get_angle)( srotation );
}

SIXTRL_INLINE SIXTRL_REAL_T NS(SRotation_get_angle)(
    SIXTRL_ARGPTR_DEC const NS(SRotation) *const SIXTRL_RESTRICT srotation )
{
    SIXTRL_STATIC_VAR SIXTRL_REAL_T const ZERO = ( SIXTRL_REAL_T )0.0;
    SIXTRL_STATIC_VAR SIXTRL_REAL_T const ONE  = ( SIXTRL_REAL_T )1.0;

    #if !defined( NDEBUG )
    SIXTRL_STATIC_VAR SIXTRL_REAL_T const EPS  = ( SIXTRL_REAL_T )1e-6;
    #endif /* !defined( NDEBUG ) */

    SIXTRL_REAL_T const sin_z = ( srotation != SIXTRL_NULLPTR )
        ? srotation->sin_z : ZERO;

    SIXTRL_REAL_T const cos_z = ( srotation != SIXTRL_NULLPTR )
        ? srotation->cos_z : ONE;

    SIXTRL_REAL_T const angle = ( sin_z >= ZERO )
        ? acos( cos_z ) : -acos( cos_z );

    #if !defined( NDEBUG )
    SIXTRL_REAL_T const temp_sin_z = sin( angle );
    SIXTRL_REAL_T const delta      = temp_sin_z - srotation->sin_z;
    SIXTRL_ASSERT( fabs( delta ) < EPS );
    #endif /* !defined( NDEBUG ) */

    return angle;
}

SIXTRL_INLINE SIXTRL_REAL_T NS(SRotation_get_cos_angle)(
    SIXTRL_ARGPTR_DEC const NS(SRotation) *const SIXTRL_RESTRICT srotation )
{
    return ( srotation != SIXTRL_NULLPTR )
        ? srotation->cos_z : ( SIXTRL_REAL_T )1;
}

SIXTRL_INLINE SIXTRL_REAL_T NS(SRotation_get_sin_angle)(
    SIXTRL_ARGPTR_DEC const NS(SRotation) *const SIXTRL_RESTRICT srotation )
{
    return ( srotation != SIXTRL_NULLPTR )
        ? srotation->sin_z : ( SIXTRL_REAL_T )0;
}

SIXTRL_INLINE void NS(SRotation_set_angle)(
    SIXTRL_ARGPTR_DEC NS(SRotation)* SIXTRL_RESTRICT srotation,
    SIXTRL_REAL_T const angle )
{
    if( srotation != SIXTRL_NULLPTR )
    {
        srotation->cos_z = cos( angle );
        srotation->sin_z = sin( angle );
    }

    return;
}

SIXTRL_INLINE void NS(SRotation_set_angle_deg)(
    SIXTRL_ARGPTR_DEC NS(SRotation)* SIXTRL_RESTRICT srotation,
    SIXTRL_REAL_T const angle_deg )
{
    SIXTRL_STATIC SIXTRL_REAL_T const DEG2RAD = M_PI / ( SIXTRL_REAL_T )180.0;
    NS(SRotation_set_angle)( srotation, DEG2RAD * angle_deg );

    return;
}

SIXTRL_INLINE bool NS(SRotation_can_be_added)(
    SIXTRL_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_objects,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_slots,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_dataptrs )
{
    return NS(Buffer_can_add_object)( buffer, sizeof( NS(SRotation) ), 0u,
        SIXTRL_NULLPTR, SIXTRL_NULLPTR, ptr_requ_objects, ptr_requ_slots,
            ptr_requ_dataptrs );
}

SIXTRL_INLINE SIXTRL_ARGPTR_DEC NS(SRotation)* NS(SRotation_new)(
    SIXTRL_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer )
{
    typedef NS(SRotation)                    elem_t;
    typedef SIXTRL_ARGPTR_DEC NS(SRotation)* ptr_to_elem_t;
    elem_t temp_obj;

    return ( ptr_to_elem_t )( uintptr_t )NS(Object_get_begin_addr)(
        NS(Buffer_add_object)( buffer, NS(SRotation_preset)( &temp_obj ),
            sizeof( elem_t ), NS(OBJECT_TYPE_SROTATION), 0u,
                SIXTRL_NULLPTR, SIXTRL_NULLPTR, SIXTRL_NULLPTR ) );
}

SIXTRL_INLINE SIXTRL_ARGPTR_DEC NS(SRotation)* NS(SRotation_add)(
    SIXTRL_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_REAL_T const angle )
{
    typedef NS(SRotation)                    elem_t;
    typedef SIXTRL_ARGPTR_DEC NS(SRotation)* ptr_to_elem_t;

    elem_t temp_obj;
    NS(SRotation_set_angle)( &temp_obj, angle );

    return ( ptr_to_elem_t )( uintptr_t )NS(Object_get_begin_addr)(
        NS(Buffer_add_object)( buffer, &temp_obj, sizeof( elem_t ),
            NS(OBJECT_TYPE_SROTATION), 0u, SIXTRL_NULLPTR, SIXTRL_NULLPTR,
                SIXTRL_NULLPTR ) );
}

SIXTRL_INLINE SIXTRL_ARGPTR_DEC NS(SRotation)* NS(SRotation_add_detailed)(
    SIXTRL_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_REAL_T const cos_z, SIXTRL_REAL_T const sin_z )
{
    typedef NS(buffer_size_t)                buf_size_t;
    typedef NS(SRotation)                    elem_t;
    typedef SIXTRL_ARGPTR_DEC NS(SRotation)* ptr_to_elem_t;

    SIXTRL_STATIC_VAR buf_size_t const num_dataptrs = ( buf_size_t )0u;

    #if !defined( NDEBUG )
    SIXTRL_REAL_T temp = ( SIXTRL_REAL_T )1 - ( cos_z * cos_z + sin_z * sin_z );
    if( temp < ( SIXTRL_REAL_T )0.0 ) temp = -temp;
    SIXTRL_ASSERT( temp < ( SIXTRL_REAL_T )1e-12 );
    #endif /* !defined( NDEBUG ) */

    elem_t temp_obj;
    temp_obj.cos_z = cos_z;
    temp_obj.sin_z = sin_z;

    return ( ptr_to_elem_t )( uintptr_t )NS(Object_get_begin_addr)(
        NS(Buffer_add_object)( buffer, &temp_obj, sizeof( elem_t ),
            NS(OBJECT_TYPE_SROTATION), num_dataptrs,
                SIXTRL_NULLPTR, SIXTRL_NULLPTR, SIXTRL_NULLPTR ) );
}

#if !defined(  _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

#endif /* SIXTRACKLIB_COMMON_IMPL_BEAM_ELEMENT_SROTATION_H__ */
/*end: sixtracklib/common/impl/be_srotation.h */
