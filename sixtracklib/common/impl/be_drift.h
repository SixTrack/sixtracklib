#ifndef SIXTRACKLIB_COMMON_IMPL_BEAM_ELEMENT_DRIFT_H__
#define SIXTRACKLIB_COMMON_IMPL_BEAM_ELEMENT_DRIFT_H__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stdbool.h>
    #include <stdint.h>
    #include <stdlib.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/_impl/definitions.h"
    #include "sixtracklib/common/impl/buffer_defines.h"
    #include "sixtracklib/common/impl/beam_elements_defines.h"
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

/* ------------------------------------------------------------------------- */

SIXTRL_FN SIXTRL_STATIC NS(buffer_size_t) NS(Drift_get_num_dataptrs)(
    SIXTRL_BE_ARGPTR_DEC const NS(Drift) *const SIXTRL_RESTRICT drift );

SIXTRL_FN SIXTRL_STATIC SIXTRL_BE_ARGPTR_DEC NS(Drift)* NS(Drift_preset)(
    SIXTRL_BE_ARGPTR_DEC NS(Drift)* SIXTRL_RESTRICT drift );

SIXTRL_FN SIXTRL_STATIC NS(drift_real_t) NS(Drift_get_length)(
    SIXTRL_BE_ARGPTR_DEC const NS(Drift) *const SIXTRL_RESTRICT drift );

SIXTRL_FN SIXTRL_STATIC void NS(Drift_set_length)(
    SIXTRL_BE_ARGPTR_DEC NS(Drift)* SIXTRL_RESTRICT drift,
    NS(drift_real_t) const length );

SIXTRL_FN SIXTRL_STATIC void NS(Drift_copy)(
    SIXTRL_BE_ARGPTR_DEC NS(Drift)* SIXTRL_RESTRICT destination,
    SIXTRL_BE_ARGPTR_DEC const NS(Drift) *const SIXTRL_RESTRICT source );

SIXTRL_FN SIXTRL_STATIC int NS(Drift_compare)(
    SIXTRL_BE_ARGPTR_DEC const NS(Drift) *const SIXTRL_RESTRICT lhs,
    SIXTRL_BE_ARGPTR_DEC const NS(Drift) *const SIXTRL_RESTRICT rhs );

SIXTRL_FN SIXTRL_STATIC int NS(Drift_compare_with_treshold)(
    SIXTRL_BE_ARGPTR_DEC const NS(Drift) *const SIXTRL_RESTRICT lhs,
    SIXTRL_BE_ARGPTR_DEC const NS(Drift) *const SIXTRL_RESTRICT rhs,
    NS(drift_real_t) const treshold );

/* ------------------------------------------------------------------------- */

#if !defined( _GPUCODE )

SIXTRL_FN SIXTRL_STATIC bool NS(Drift_can_be_added)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_objects,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_slots,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_dataptrs );

SIXTRL_FN SIXTRL_STATIC SIXTRL_BUFFER_DATAPTR_DEC NS(Drift)* NS(Drift_new)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer );

SIXTRL_FN SIXTRL_STATIC SIXTRL_BUFFER_DATAPTR_DEC NS(Drift)* NS(Drift_add)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(drift_real_t) const length );

SIXTRL_FN SIXTRL_STATIC
SIXTRL_BUFFER_DATAPTR_DEC NS(Drift)* NS(Drift_add_copy)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_BE_ARGPTR_DEC const NS(Drift) *const SIXTRL_RESTRICT drift );

#endif /* !defined( _GPUCODE ) */

/* ========================================================================= */

typedef struct NS(DriftExact)
{
    NS(drift_real_t) length SIXTRL_ALIGN( 8 );
}
NS(DriftExact);

/* ------------------------------------------------------------------------- */

SIXTRL_FN SIXTRL_STATIC NS(buffer_size_t) NS(DriftExact_get_num_dataptrs)(
    SIXTRL_BE_ARGPTR_DEC const NS(DriftExact) *const SIXTRL_RESTRICT drift );

SIXTRL_FN SIXTRL_STATIC SIXTRL_BE_ARGPTR_DEC NS(DriftExact)*
NS(DriftExact_preset)( SIXTRL_BE_ARGPTR_DEC
    NS(DriftExact)* SIXTRL_RESTRICT drift );

SIXTRL_FN SIXTRL_STATIC NS(drift_real_t) NS(DriftExact_get_length)(
    SIXTRL_BE_ARGPTR_DEC const NS(DriftExact) *const SIXTRL_RESTRICT drift );

SIXTRL_FN SIXTRL_STATIC void NS(DriftExact_set_length)(
    SIXTRL_BE_ARGPTR_DEC NS(DriftExact)* SIXTRL_RESTRICT drift,
    NS(drift_real_t) const length );

SIXTRL_FN SIXTRL_STATIC void NS(DriftExact_copy)(
    SIXTRL_BE_ARGPTR_DEC NS(DriftExact)* SIXTRL_RESTRICT destination,
    SIXTRL_BE_ARGPTR_DEC const NS(DriftExact) *const SIXTRL_RESTRICT source );

SIXTRL_FN SIXTRL_STATIC int NS(DriftExact_compare)(
    SIXTRL_BE_ARGPTR_DEC const NS(DriftExact) *const SIXTRL_RESTRICT lhs,
    SIXTRL_BE_ARGPTR_DEC const NS(DriftExact) *const SIXTRL_RESTRICT rhs );

SIXTRL_FN SIXTRL_STATIC int NS(DriftExact_compare_with_treshold)(
    SIXTRL_BE_ARGPTR_DEC const NS(DriftExact) *const SIXTRL_RESTRICT lhs,
    SIXTRL_BE_ARGPTR_DEC const NS(DriftExact) *const SIXTRL_RESTRICT rhs,
    NS(drift_real_t) const treshold );

#if !defined( _GPUCODE )

SIXTRL_FN SIXTRL_STATIC bool NS(DriftExact_can_be_added)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_objects,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_slots,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_dataptrs );

SIXTRL_FN SIXTRL_STATIC SIXTRL_BUFFER_DATAPTR_DEC NS(DriftExact)*
NS(DriftExact_new)( SIXTRL_BUFFER_ARGPTR_DEC
    NS(Buffer)* SIXTRL_RESTRICT buffer );

SIXTRL_FN SIXTRL_STATIC SIXTRL_BUFFER_DATAPTR_DEC NS(DriftExact)*
NS(DriftExact_add)( SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)*
    SIXTRL_RESTRICT buffer, NS(drift_real_t) const length );

SIXTRL_FN SIXTRL_STATIC SIXTRL_BUFFER_DATAPTR_DEC NS(DriftExact)*
NS(DriftExact_add_copy)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_BE_ARGPTR_DEC const NS(DriftExact) *const SIXTRL_RESTRICT drift );


#endif /* !defined( _GPUCODE ) */

/* ========================================================================= */

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

SIXTRL_INLINE NS(buffer_size_t) NS(Drift_get_num_dataptrs)(
    SIXTRL_BE_ARGPTR_DEC const NS(Drift) *const SIXTRL_RESTRICT drift )
{
    ( void )drift;
    return ( NS(buffer_size_t) )0u;
}

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(Drift)* NS(Drift_preset)(
    SIXTRL_BE_ARGPTR_DEC NS(Drift)* SIXTRL_RESTRICT drift )
{
    if( drift != SIXTRL_NULLPTR )
    {
        NS(Drift_set_length)( drift, ( NS(drift_real_t) )0 );
    }

    return drift;
}

SIXTRL_INLINE  NS(drift_real_t) NS(Drift_get_length)(
    SIXTRL_BE_ARGPTR_DEC const NS(Drift) *const SIXTRL_RESTRICT drift )
{
    return ( drift != SIXTRL_NULLPTR ) ? drift->length : ( NS(drift_real_t) )0;
}

SIXTRL_INLINE void NS(Drift_set_length)(
    SIXTRL_BE_ARGPTR_DEC NS(Drift)* SIXTRL_RESTRICT drift,
    NS(drift_real_t) const length )
{
    if( drift != SIXTRL_NULLPTR ) drift->length = length;
    return;
}

SIXTRL_INLINE void NS(Drift_copy)(
    SIXTRL_BE_ARGPTR_DEC NS(Drift)* SIXTRL_RESTRICT destination,
    SIXTRL_BE_ARGPTR_DEC const NS(Drift) *const SIXTRL_RESTRICT source )
{
    NS(Drift_set_length)( destination, NS(Drift_get_length)( source ) );
    return;
}

SIXTRL_INLINE int NS(Drift_compare)(
    SIXTRL_BE_ARGPTR_DEC const NS(Drift) *const SIXTRL_RESTRICT lhs,
    SIXTRL_BE_ARGPTR_DEC const NS(Drift) *const SIXTRL_RESTRICT rhs )
{
    int compare_value = -1;

    if( ( lhs != SIXTRL_NULLPTR ) &&
        ( rhs != SIXTRL_NULLPTR ) )
    {
        if( NS(Drift_get_length)( lhs ) > NS(Drift_get_length)( rhs ) )
        {
            compare_value = +1;
        }
        else if( NS(Drift_get_length)( lhs ) < NS(Drift_get_length)( rhs ) )
        {
            compare_value = -1;
        }
        else
        {
            compare_value = 0;
        }
    }
    else if( rhs != SIXTRL_NULLPTR )
    {
        compare_value = +1;
    }

    return compare_value;
}

SIXTRL_INLINE int NS(Drift_compare_with_treshold)(
    SIXTRL_BE_ARGPTR_DEC const NS(Drift) *const SIXTRL_RESTRICT lhs,
    SIXTRL_BE_ARGPTR_DEC const NS(Drift) *const SIXTRL_RESTRICT rhs,
    NS(drift_real_t) const treshold )
{
    int compare_value = -1;

    if( ( lhs != SIXTRL_NULLPTR ) &&
        ( rhs != SIXTRL_NULLPTR ) )
    {
        SIXTRL_ASSERT( treshold >= ( NS(drift_real_t) )0.0 );

        NS(drift_real_t) const diff =
            NS(Drift_get_length)( lhs ) - NS(Drift_get_length)( rhs );

        NS(drift_real_t) const abs_diff =
            ( diff > ( NS(drift_real_t) )0.0 ) ? diff : -diff;

        if( abs_diff < treshold )
        {
            compare_value = 0;
        }
        else if( diff > ( NS(drift_real_t) )0.0 )
        {
            compare_value = +1;
        }
        else
        {
            compare_value = -1;
        }
    }
    else if( rhs != SIXTRL_NULLPTR )
    {
        compare_value = +1;
    }

    return compare_value;
}

#if !defined( _GPUCODE )

SIXTRL_INLINE  bool NS(Drift_can_be_added)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_objects,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_slots,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_dataptrs )
{
    typedef NS(Drift)           elem_t;
    typedef NS(buffer_size_t)   buf_size_t;

    buf_size_t const num_dataptrs =
        NS(DriftExact_get_num_dataptrs)( SIXTRL_NULLPTR );

    SIXTRL_BUFFER_ARGPTR_DEC buf_size_t const* sizes   = SIXTRL_NULLPTR;
    SIXTRL_BUFFER_ARGPTR_DEC buf_size_t const* counts  = SIXTRL_NULLPTR;

    SIXTRL_ASSERT( num_dataptrs == ( buf_size_t )0u );

    return NS(Buffer_can_add_object)( buffer, sizeof( elem_t ), num_dataptrs,
        sizes, counts, requ_objects, requ_slots, requ_dataptrs );
}

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(Drift)* NS(Drift_new)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer )
{
    typedef NS(Drift)                            elem_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC elem_t*    ptr_to_elem_t;
    typedef NS(buffer_size_t)                    buf_size_t;

    buf_size_t const num_dataptrs =
        NS(DriftExact_get_num_dataptrs)( SIXTRL_NULLPTR );

    SIXTRL_BUFFER_ARGPTR_DEC buf_size_t const* offsets = SIXTRL_NULLPTR;
    SIXTRL_BUFFER_ARGPTR_DEC buf_size_t const* sizes   = SIXTRL_NULLPTR;
    SIXTRL_BUFFER_ARGPTR_DEC buf_size_t const* counts  = SIXTRL_NULLPTR;

    elem_t temp_obj;
    temp_obj.length = ( NS(drift_real_t) )0.0;

    SIXTRL_ASSERT( num_dataptrs == ( buf_size_t )0u );

    return ( ptr_to_elem_t )( uintptr_t )NS(Object_get_begin_addr)(
        NS(Buffer_add_object)( buffer, &temp_obj, sizeof( elem_t ),
            NS(OBJECT_TYPE_DRIFT), num_dataptrs, offsets, sizes, counts ) );
}

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(Drift)* NS(Drift_add)(
    SIXTRL_BUFFER_ARGPTR_DEC  NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(drift_real_t) const length )
{
    typedef NS(Drift)                            elem_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC elem_t*    ptr_to_elem_t;
    typedef NS(buffer_size_t)                    buf_size_t;

    buf_size_t const num_dataptrs =
        NS(DriftExact_get_num_dataptrs)( SIXTRL_NULLPTR );

    SIXTRL_BUFFER_ARGPTR_DEC buf_size_t const* offsets = SIXTRL_NULLPTR;
    SIXTRL_BUFFER_ARGPTR_DEC buf_size_t const* sizes   = SIXTRL_NULLPTR;
    SIXTRL_BUFFER_ARGPTR_DEC buf_size_t const* counts  = SIXTRL_NULLPTR;

    elem_t temp_obj;
    temp_obj.length = length;

    SIXTRL_ASSERT( num_dataptrs == ( buf_size_t )0u );

    return ( ptr_to_elem_t )( uintptr_t )NS(Object_get_begin_addr)(
        NS(Buffer_add_object)( buffer, &temp_obj, sizeof( elem_t ),
            NS(OBJECT_TYPE_DRIFT), num_dataptrs, offsets, sizes, counts ) );
}

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(Drift)* NS(Drift_add_copy)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_BE_ARGPTR_DEC const NS(Drift) *const SIXTRL_RESTRICT drift )
{
    return NS(Drift_add)( buffer, NS(Drift_get_length)( drift ) );
}

#endif /* !defined( _GPUCODE ) */

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE NS(buffer_size_t) NS(DriftExact_get_num_dataptrs)(
    SIXTRL_BE_ARGPTR_DEC  const NS(DriftExact) *const SIXTRL_RESTRICT drift )
{
    ( void )drift;
    return ( NS(buffer_size_t) )0u;
}

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(DriftExact)* NS(DriftExact_preset)(
    SIXTRL_BE_ARGPTR_DEC NS(DriftExact)* SIXTRL_RESTRICT drift )
{
    if( drift != SIXTRL_NULLPTR )
    {
        drift->length = ( NS(drift_real_t) )0;
    }

    return drift;
}

SIXTRL_INLINE  NS(drift_real_t) NS(DriftExact_get_length)(
    SIXTRL_BE_ARGPTR_DEC const NS(DriftExact) *const SIXTRL_RESTRICT drift )
{
    return ( drift != SIXTRL_NULLPTR ) ? drift->length : ( NS(drift_real_t) )0;
}

SIXTRL_INLINE void NS(DriftExact_set_length)(
    SIXTRL_BE_ARGPTR_DEC NS(DriftExact)* SIXTRL_RESTRICT drift,
    NS(drift_real_t) const length )
{
    SIXTRL_ASSERT( drift != SIXTRL_NULLPTR );
    drift->length = length;
    return;
}


SIXTRL_INLINE void NS(DriftExact_copy)(
    SIXTRL_BE_ARGPTR_DEC NS(DriftExact)* SIXTRL_RESTRICT destination,
    SIXTRL_BE_ARGPTR_DEC const NS(DriftExact) *const SIXTRL_RESTRICT source )
{
    NS(DriftExact_set_length)(
        destination, NS(DriftExact_get_length)( source ) );

    return;
}

SIXTRL_INLINE int NS(DriftExact_compare)(
    SIXTRL_BE_ARGPTR_DEC const NS(DriftExact) *const SIXTRL_RESTRICT lhs,
    SIXTRL_BE_ARGPTR_DEC const NS(DriftExact) *const SIXTRL_RESTRICT rhs )
{
    int compare_value = -1;

    if( ( lhs != SIXTRL_NULLPTR ) &&
        ( rhs != SIXTRL_NULLPTR ) )
    {
        if( NS(DriftExact_get_length)( lhs ) >
            NS(DriftExact_get_length)( rhs ) )
        {
            compare_value = +1;
        }
        else if( NS(DriftExact_get_length)( lhs ) <
                 NS(DriftExact_get_length)( rhs ) )
        {
            compare_value = -1;
        }
        else
        {
            compare_value = 0;
        }
    }
    else if( rhs != SIXTRL_NULLPTR )
    {
        compare_value = +1;
    }

    return compare_value;
}

SIXTRL_INLINE int NS(DriftExact_compare_with_treshold)(
    SIXTRL_BE_ARGPTR_DEC const NS(DriftExact) *const SIXTRL_RESTRICT lhs,
    SIXTRL_BE_ARGPTR_DEC const NS(DriftExact) *const SIXTRL_RESTRICT rhs,
    NS(drift_real_t) const treshold )
{
    int compare_value = -1;

    typedef NS(drift_real_t) real_t;

    SIXTRL_STATIC_VAR real_t const ZERO = ( real_t )0.0;

    if( ( lhs != SIXTRL_NULLPTR ) && ( rhs != SIXTRL_NULLPTR ) &&
        ( treshold > ZERO ) )
    {
        SIXTRL_ASSERT( treshold >= ZERO );

        NS(drift_real_t) const diff =
            NS(DriftExact_get_length)( lhs ) - NS(DriftExact_get_length)( rhs );

        NS(drift_real_t) const abs_diff = ( diff > ZERO ) ? diff : -diff;

        if( abs_diff < treshold )
        {
            compare_value = 0;
        }
        else if( diff > ZERO )
        {
            compare_value = +1;
        }
        else
        {
            compare_value = -1;
        }
    }
    else if( ( rhs != SIXTRL_NULLPTR ) && ( treshold > ZERO ) )
    {
        compare_value = +1;
    }

    return compare_value;
}

#if !defined( _GPUCODE )

SIXTRL_INLINE  bool NS(DriftExact_can_be_added)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_objects,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_slots,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_dataptrs )
{
    typedef NS(buffer_size_t) buf_size_t;

    buf_size_t const num_dataptrs =
        NS(DriftExact_get_num_dataptrs)( SIXTRL_NULLPTR );

    SIXTRL_BUFFER_ARGPTR_DEC buf_size_t const* sizes  = SIXTRL_NULLPTR;
    SIXTRL_BUFFER_ARGPTR_DEC buf_size_t const* counts = SIXTRL_NULLPTR;

    SIXTRL_ASSERT( num_dataptrs == ( buf_size_t )0u );

    return NS(Buffer_can_add_object)( buffer, sizeof( NS(DriftExact) ),
        num_dataptrs, sizes, counts, requ_objects, requ_slots, requ_dataptrs );
}

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(DriftExact)*
NS(DriftExact_new)( SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)*
    SIXTRL_RESTRICT buffer )
{
    typedef NS(DriftExact)                      elem_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC elem_t*   ptr_to_elem_t;
    typedef NS(buffer_size_t)                   buf_size_t;

    buf_size_t const num_dataptrs =
        NS(DriftExact_get_num_dataptrs)( SIXTRL_NULLPTR );

    SIXTRL_BUFFER_ARGPTR_DEC buf_size_t const* offsets = SIXTRL_NULLPTR;
    SIXTRL_BUFFER_ARGPTR_DEC buf_size_t const* sizes   = SIXTRL_NULLPTR;
    SIXTRL_BUFFER_ARGPTR_DEC buf_size_t const* counts  = SIXTRL_NULLPTR;

    elem_t temp_obj;
    temp_obj.length = ( NS(drift_real_t) )0.0;

    SIXTRL_ASSERT( num_dataptrs == ( buf_size_t )0u );

    return ( ptr_to_elem_t )( uintptr_t )NS(Object_get_begin_addr)(
        NS(Buffer_add_object)( buffer, NS(DriftExact_preset)( &temp_obj ),
            sizeof( elem_t ), NS(OBJECT_TYPE_DRIFT_EXACT), num_dataptrs,
                offsets, sizes, counts ) );
}

SIXTRL_INLINE  SIXTRL_BUFFER_DATAPTR_DEC NS(DriftExact)*
NS(DriftExact_add)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(drift_real_t) const length )
{
    typedef NS(DriftExact)                      elem_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC elem_t*   ptr_to_elem_t;
    typedef NS(buffer_size_t)                   buf_size_t;

    buf_size_t const num_dataptrs =
        NS(DriftExact_get_num_dataptrs)( SIXTRL_NULLPTR );

    SIXTRL_BUFFER_ARGPTR_DEC buf_size_t const* offsets = SIXTRL_NULLPTR;
    SIXTRL_BUFFER_ARGPTR_DEC buf_size_t const* sizes   = SIXTRL_NULLPTR;
    SIXTRL_BUFFER_ARGPTR_DEC buf_size_t const* counts  = SIXTRL_NULLPTR;

    elem_t temp_obj;
    temp_obj.length = length;

    SIXTRL_ASSERT( num_dataptrs == ( buf_size_t )0u );

    return ( ptr_to_elem_t )( uintptr_t )NS(Object_get_begin_addr)(
        NS(Buffer_add_object)( buffer, &temp_obj, sizeof( elem_t ),
            NS(OBJECT_TYPE_DRIFT_EXACT), num_dataptrs, offsets, sizes, counts ) );
}

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(DriftExact)* NS(DriftExact_add_copy)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_BE_ARGPTR_DEC const NS(DriftExact) *const SIXTRL_RESTRICT drift )
{
    return NS(DriftExact_add)( buffer, NS(DriftExact_get_length)( drift ) );
}

#endif /* !defined( _GPUCODE ) */

#if !defined(  _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

#endif /* SIXTRACKLIB_COMMON_IMPL_BEAM_ELEMENT_DRIFT_H__ */

/* end: sixtracklib/common/impl/be_drift.h */
