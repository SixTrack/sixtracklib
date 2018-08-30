#ifndef SIXTRACKLIB_COMMON_IMPL_BEAM_ELEMENT_DRIFT_H__
#define SIXTRACKLIB_COMMON_IMPL_BEAM_ELEMENT_DRIFT_H__

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

typedef SIXTRL_REAL_T NS(drift_real_t);

typedef struct NS(Drift)
{
    NS(drift_real_t) length SIXTRL_ALIGN( 8 );
}
NS(Drift);

typedef struct NS(DriftExact)
{
    NS(drift_real_t) length SIXTRL_ALIGN( 8 );
}
NS(DriftExact);

/* ========================================================================= */

SIXTRL_FN SIXTRL_STATIC NS(Drift)* NS(Drift_preset)(
    NS(Drift)* SIXTRL_RESTRICT drift );

SIXTRL_FN SIXTRL_STATIC NS(drift_real_t) NS(Drift_get_length)(
    const NS(Drift) *const SIXTRL_RESTRICT drift );

SIXTRL_FN SIXTRL_STATIC void NS(Drift_set_length)(
    NS(Drift)* SIXTRL_RESTRICT drift, NS(drift_real_t) const length );

SIXTRL_FN SIXTRL_STATIC bool NS(Drift_can_be_added)(
    SIXTRL_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_objects,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_slots,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_dataptrs );

SIXTRL_FN SIXTRL_STATIC SIXTRL_ARGPTR_DEC NS(Drift)* NS(Drift_new)(
    SIXTRL_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer );

SIXTRL_FN SIXTRL_STATIC SIXTRL_ARGPTR_DEC NS(Drift)* NS(Drift_add)(
    SIXTRL_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(drift_real_t) const length );

/* ------------------------------------------------------------------------- */

SIXTRL_FN SIXTRL_STATIC SIXTRL_ARGPTR_DEC NS(DriftExact)*
NS(DriftExact_preset)(
    SIXTRL_ARGPTR_DEC NS(DriftExact)* SIXTRL_RESTRICT drift );

SIXTRL_FN SIXTRL_STATIC NS(drift_real_t) NS(DriftExact_get_length)(
    SIXTRL_ARGPTR_DEC const NS(DriftExact) *const SIXTRL_RESTRICT drift );

SIXTRL_FN SIXTRL_STATIC void NS(DriftExact_set_length)(
    SIXTRL_ARGPTR_DEC NS(DriftExact)* SIXTRL_RESTRICT drift,
    NS(drift_real_t) const length );

SIXTRL_FN SIXTRL_STATIC bool NS(DriftExact_can_be_added)(
    SIXTRL_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_objects,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_slots,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_dataptrs );

SIXTRL_FN SIXTRL_STATIC SIXTRL_ARGPTR_DEC NS(DriftExact)* NS(DriftExact_new)(
    SIXTRL_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer );

SIXTRL_FN SIXTRL_STATIC SIXTRL_ARGPTR_DEC NS(DriftExact)* NS(DriftExact_add)(
    SIXTRL_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(drift_real_t) const length );

/* ========================================================================= */

#if !defined(  _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

/* ************************************************************************* */
/* *****          Implementation of C inline functions                   *** */
/* ************************************************************************* */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/buffer.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined(  _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

SIXTRL_INLINE  NS(Drift)* NS(Drift_preset)(
    NS(Drift)* SIXTRL_RESTRICT drift )
{
    if( drift != SIXTRL_NULLPTR )
    {
        NS(Drift_set_length)( drift, ( NS(drift_real_t) )0 );
    }

    return drift;
}

SIXTRL_INLINE  NS(drift_real_t) NS(Drift_get_length)(
    const NS(Drift) *const SIXTRL_RESTRICT drift )
{
    return ( drift != SIXTRL_NULLPTR ) ? drift->length : ( NS(drift_real_t) )0;
}

SIXTRL_INLINE void NS(Drift_set_length)(
    NS(Drift)* SIXTRL_RESTRICT drift, NS(drift_real_t) const length )
{
    if( drift != SIXTRL_NULLPTR ) drift->length = length;
    return;
}

SIXTRL_INLINE  bool NS(Drift_can_be_added)(
    SIXTRL_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_objects,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_slots,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_dataptrs )
{
    return NS(Buffer_can_add_object)( buffer, sizeof( NS(Drift) ), 0u,
        SIXTRL_NULLPTR, SIXTRL_NULLPTR, ptr_requ_objects, ptr_requ_slots,
            ptr_requ_dataptrs );
}

SIXTRL_INLINE SIXTRL_ARGPTR_DEC NS(Drift)* NS(Drift_new)(
    SIXTRL_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer )
{
    typedef NS(Drift) elem_t;
    elem_t temp_obj;

    return ( elem_t* )( uintptr_t )NS(Object_get_begin_addr)(
        NS(Buffer_add_object)( buffer, NS(Drift_preset)( &temp_obj ),
            sizeof( elem_t ), NS(OBJECT_TYPE_DRIFT), 0u,
                SIXTRL_NULLPTR, SIXTRL_NULLPTR, SIXTRL_NULLPTR ) );
}

SIXTRL_INLINE SIXTRL_ARGPTR_DEC NS(Drift)* NS(Drift_add)(
    SIXTRL_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(drift_real_t) const length )
{
    typedef NS(Drift) elem_t;
    typedef SIXTRL_ARGPTR_DEC elem_t*   ptr_to_elem_t;

    elem_t temp_obj;

    NS(Drift_preset)( &temp_obj );
    NS(Drift_set_length)( &temp_obj, length );

    return ( ptr_to_elem_t )( uintptr_t )NS(Object_get_begin_addr)(
        NS(Buffer_add_object)( buffer, &temp_obj, sizeof( elem_t ),
            NS(OBJECT_TYPE_DRIFT), 0u, SIXTRL_NULLPTR, SIXTRL_NULLPTR,
                SIXTRL_NULLPTR ) );
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE SIXTRL_ARGPTR_DEC NS(DriftExact)* NS(DriftExact_preset)(
    SIXTRL_ARGPTR_DEC NS(DriftExact)* SIXTRL_RESTRICT drift )
{
    if( drift != SIXTRL_NULLPTR )
    {
        drift->length = ( NS(drift_real_t) )0;
    }

    return drift;
}

SIXTRL_INLINE  NS(drift_real_t) NS(DriftExact_get_length)(
    SIXTRL_ARGPTR_DEC const NS(DriftExact) *const SIXTRL_RESTRICT drift )
{
    return ( drift != SIXTRL_NULLPTR ) ? drift->length : ( NS(drift_real_t) )0;
}

SIXTRL_INLINE void NS(DriftExact_set_length)(
    SIXTRL_ARGPTR_DEC NS(DriftExact)* SIXTRL_RESTRICT drift,
    NS(drift_real_t) const length )
{
    SIXTRL_ASSERT( drift != SIXTRL_NULLPTR );
    drift->length = length;
    return;
}

SIXTRL_INLINE  bool NS(DriftExact_can_be_added)(
    SIXTRL_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_objects,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_slots,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_dataptrs )
{
    return NS(Buffer_can_add_object)( buffer, sizeof( NS(DriftExact) ), 0u,
        SIXTRL_NULLPTR, SIXTRL_NULLPTR, ptr_requ_objects, ptr_requ_slots,
            ptr_requ_dataptrs );
}

SIXTRL_INLINE  SIXTRL_ARGPTR_DEC NS(DriftExact)* NS(DriftExact_new)(
    SIXTRL_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer )
{
    typedef NS(DriftExact)    elem_t;
    typedef SIXTRL_ARGPTR_DEC elem_t*   ptr_to_elem_t;

    elem_t temp_obj;

    return ( ptr_to_elem_t )( uintptr_t )NS(Object_get_begin_addr)(
        NS(Buffer_add_object)( buffer, NS(DriftExact_preset)( &temp_obj ),
            sizeof( elem_t ), NS(OBJECT_TYPE_DRIFT_EXACT), 0u,
                SIXTRL_NULLPTR, SIXTRL_NULLPTR, SIXTRL_NULLPTR ) );
}

SIXTRL_INLINE  SIXTRL_ARGPTR_DEC NS(DriftExact)* NS(DriftExact_add)(
    SIXTRL_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(drift_real_t) const length )
{
    typedef NS(DriftExact)    elem_t;
    typedef SIXTRL_ARGPTR_DEC elem_t*   ptr_to_elem_t;

    elem_t temp_obj;

    NS(DriftExact_preset)( &temp_obj );
    NS(DriftExact_set_length)( &temp_obj, length );

    return ( ptr_to_elem_t )( uintptr_t )NS(Object_get_begin_addr)(
        NS(Buffer_add_object)( buffer, &temp_obj, sizeof( elem_t ),
            NS(OBJECT_TYPE_DRIFT_EXACT), 0u, SIXTRL_NULLPTR, SIXTRL_NULLPTR,
                SIXTRL_NULLPTR ) );
}

#if !defined(  _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

#endif /* SIXTRACKLIB_COMMON_IMPL_BEAM_ELEMENT_DRIFT_H__ */

/* end: sixtracklib/common/impl/be_drift.h */
