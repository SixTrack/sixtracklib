#ifndef SIXTRACKLIB_COMMON_IMPL_BEAM_ELEMENT_CAVITY_H__
#define SIXTRACKLIB_COMMON_IMPL_BEAM_ELEMENT_CAVITY_H__

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

typedef struct NS(Cavity)
{
    SIXTRL_REAL_T   voltage     SIXTRL_ALIGN( 8 );
    SIXTRL_REAL_T   frequency   SIXTRL_ALIGN( 8 );
    SIXTRL_REAL_T   lag         SIXTRL_ALIGN( 8 );
}
NS(Cavity);

/* ------------------------------------------------------------------------- */

SIXTRL_FN SIXTRL_STATIC NS(buffer_size_t) NS(Cavity_get_num_dataptrs)(
    SIXTRL_ARGPTR_DEC const NS(Cavity) *const SIXTRL_RESTRICT cavity );

SIXTRL_FN SIXTRL_STATIC NS(Cavity)* NS(Cavity_preset)(
    SIXTRL_ARGPTR_DEC NS(Cavity)* SIXTRL_RESTRICT cavity );

SIXTRL_FN SIXTRL_STATIC SIXTRL_REAL_T NS(Cavity_get_voltage)(
    SIXTRL_ARGPTR_DEC const NS(Cavity) *const SIXTRL_RESTRICT cavity );

SIXTRL_FN SIXTRL_STATIC SIXTRL_REAL_T NS(Cavity_get_frequency)(
    SIXTRL_ARGPTR_DEC const NS(Cavity) *const SIXTRL_RESTRICT cavity );

SIXTRL_FN SIXTRL_STATIC SIXTRL_REAL_T NS(Cavity_get_lag)(
    SIXTRL_ARGPTR_DEC const NS(Cavity) *const SIXTRL_RESTRICT cavity );

SIXTRL_FN SIXTRL_STATIC void NS(Cavity_set_voltage)(
    SIXTRL_ARGPTR_DEC NS(Cavity)* SIXTRL_RESTRICT cavity,
    SIXTRL_REAL_T const voltage );

SIXTRL_FN SIXTRL_STATIC void NS(Cavity_set_frequency)(
    SIXTRL_ARGPTR_DEC NS(Cavity)* SIXTRL_RESTRICT cavity,
    SIXTRL_REAL_T const frequency );

SIXTRL_FN SIXTRL_STATIC void NS(Cavity_set_lag)(
    SIXTRL_ARGPTR_DEC NS(Cavity)* SIXTRL_RESTRICT cavity,
    SIXTRL_REAL_T const lag );

#if !defined( _GPUCODE )

SIXTRL_FN SIXTRL_STATIC bool NS(Cavity_can_be_added)(
    SIXTRL_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_objects,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_slots,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_dataptrs );

SIXTRL_FN SIXTRL_STATIC SIXTRL_ARGPTR_DEC NS(Cavity)* NS(Cavity_new)(
    SIXTRL_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer );

SIXTRL_FN SIXTRL_STATIC SIXTRL_ARGPTR_DEC NS(Cavity)* NS(Cavity_add)(
    SIXTRL_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_REAL_T  const voltage,
    SIXTRL_REAL_T  const frequency,
    SIXTRL_REAL_T  const lag );

#endif /* !defined( _GPUCODE ) */

#if !defined(  _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

/* ************************************************************************* */
/* *****          Implementation of C inline functions                   *** */
/* ************************************************************************* */

#if !defined( SIXTRL_NO_INCLUDES )
    #if !defined( _GPUCODE )
        #include "sixtracklib/common/buffer.h"
    #endif /* !defined( _GPUCODE ) */
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined(  _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */


SIXTRL_INLINE NS(buffer_size_t) NS(Cavity_get_num_dataptrs)(
    SIXTRL_ARGPTR_DEC const NS(Cavity) *const SIXTRL_RESTRICT cavity )
{
    return ( NS(buffer_size_t) )0u;
}

SIXTRL_INLINE NS(Cavity)* NS(Cavity_preset)(
    SIXTRL_ARGPTR_DEC NS(Cavity)* SIXTRL_RESTRICT cavity )
{
    if( cavity != SIXTRL_NULLPTR )
    {
        NS(Cavity_set_voltage)(   cavity, ( SIXTRL_REAL_T )0 );
        NS(Cavity_set_frequency)( cavity, ( SIXTRL_REAL_T )0 );
        NS(Cavity_set_lag)( cavity, ( SIXTRL_REAL_T )0 );
    }

    return cavity;
}

SIXTRL_INLINE SIXTRL_REAL_T NS(Cavity_get_voltage)(
    SIXTRL_ARGPTR_DEC const NS(Cavity) *const SIXTRL_RESTRICT cavity )
{
    return ( cavity != SIXTRL_NULLPTR ) ? cavity->voltage : ( SIXTRL_REAL_T )0;
}

SIXTRL_INLINE SIXTRL_REAL_T NS(Cavity_get_frequency)(
    SIXTRL_ARGPTR_DEC const NS(Cavity) *const SIXTRL_RESTRICT cavity )
{
    return ( cavity != SIXTRL_NULLPTR )
        ? cavity->frequency : ( SIXTRL_REAL_T )0;
}

SIXTRL_INLINE SIXTRL_REAL_T NS(Cavity_get_lag)(
    SIXTRL_ARGPTR_DEC const NS(Cavity) *const SIXTRL_RESTRICT cavity )
{
    return ( cavity != SIXTRL_NULLPTR ) ? cavity->lag : ( SIXTRL_REAL_T )0;
}

SIXTRL_INLINE void NS(Cavity_set_voltage)(
    SIXTRL_ARGPTR_DEC NS(Cavity)* SIXTRL_RESTRICT cavity,
    SIXTRL_REAL_T const voltage )
{
    SIXTRL_ASSERT( cavity != SIXTRL_NULLPTR );
    cavity->voltage = voltage;
    return;
}

SIXTRL_INLINE void NS(Cavity_set_frequency)(
    SIXTRL_ARGPTR_DEC NS(Cavity)* SIXTRL_RESTRICT cavity,
    SIXTRL_REAL_T const frequency )
{
    SIXTRL_ASSERT( cavity != SIXTRL_NULLPTR );
    cavity->frequency = frequency;
    return;
}

SIXTRL_INLINE void NS(Cavity_set_lag)(
    SIXTRL_ARGPTR_DEC NS(Cavity)* SIXTRL_RESTRICT cavity,
    SIXTRL_REAL_T const lag )
{
    SIXTRL_ASSERT( cavity != SIXTRL_NULLPTR );
    cavity->lag = lag;
    return;
}

#if !defined( _GPUCODE )

SIXTRL_INLINE bool NS(Cavity_can_be_added)(
    SIXTRL_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_objects,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_slots,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_dataptrs )
{
    return NS(Buffer_can_add_object)( buffer, sizeof( NS(Cavity) ), 0u,
        SIXTRL_NULLPTR, SIXTRL_NULLPTR, ptr_requ_objects, ptr_requ_slots,
            ptr_requ_dataptrs );
}

SIXTRL_INLINE SIXTRL_ARGPTR_DEC NS(Cavity)* NS(Cavity_new)(
    SIXTRL_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer )
{
    typedef NS(Cavity) elem_t;
    typedef SIXTRL_ARGPTR_DEC elem_t* ptr_elem_t;
    elem_t temp_obj;

    return ( ptr_elem_t )( uintptr_t )NS(Object_get_begin_addr)(
        NS(Buffer_add_object)( buffer, NS(Cavity_preset)( &temp_obj ),
            sizeof( elem_t ), NS(OBJECT_TYPE_CAVITY), 0u,
                SIXTRL_NULLPTR, SIXTRL_NULLPTR, SIXTRL_NULLPTR ) );
}

SIXTRL_INLINE SIXTRL_ARGPTR_DEC NS(Cavity)* NS(Cavity_add)(
    SIXTRL_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_REAL_T  const voltage,
    SIXTRL_REAL_T  const frequency,
    SIXTRL_REAL_T  const lag )
{
    typedef NS(Cavity) elem_t;
    typedef SIXTRL_ARGPTR_DEC elem_t* ptr_elem_t;
    elem_t temp_obj;

    NS(Cavity_set_voltage)( &temp_obj, voltage );
    NS(Cavity_set_frequency)( &temp_obj, frequency );
    NS(Cavity_set_lag)( &temp_obj, lag );

    return ( ptr_elem_t )( uintptr_t )NS(Object_get_begin_addr)(
        NS(Buffer_add_object)( buffer, &temp_obj, sizeof( elem_t ),
            NS(OBJECT_TYPE_CAVITY), 0u, SIXTRL_NULLPTR, SIXTRL_NULLPTR,
                SIXTRL_NULLPTR ) );
}

#endif /* !defined( _GPUCODE ) */

#if !defined(  _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

#endif /* SIXTRACKLIB_COMMON_IMPL_BEAM_ELEMENT_CAVITY_H__ */

/*end: sixtracklib/common/impl/be_cavity.h */
