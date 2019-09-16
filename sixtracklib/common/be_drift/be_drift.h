#ifndef SIXTRACKLIB_COMMON_BE_DRIFT_BEAM_ELEMENT_DRIFT_H__
#define SIXTRACKLIB_COMMON_BE_DRIFT_BEAM_ELEMENT_DRIFT_H__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stdbool.h>
    #include <stdint.h>
    #include <stdlib.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/internal/buffer_main_defines.h"
    #include "sixtracklib/common/internal/beam_elements_defines.h"
    #include "sixtracklib/common/internal/objects_type_id.h"
    #include "sixtracklib/common/buffer/buffer_type.h"
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

SIXTRL_FN SIXTRL_STATIC NS(buffer_size_t) NS(Drift_get_num_slots)(
    SIXTRL_BE_ARGPTR_DEC const NS(Drift) *const SIXTRL_RESTRICT drift,
    NS(buffer_size_t) const slot_size );

SIXTRL_FN SIXTRL_STATIC SIXTRL_BE_ARGPTR_DEC NS(Drift)* NS(Drift_preset)(
    SIXTRL_BE_ARGPTR_DEC NS(Drift)* SIXTRL_RESTRICT drift );

SIXTRL_FN SIXTRL_STATIC NS(drift_real_t) NS(Drift_get_length)(
    SIXTRL_BE_ARGPTR_DEC const NS(Drift) *const SIXTRL_RESTRICT drift );

SIXTRL_FN SIXTRL_STATIC void NS(Drift_set_length)(
    SIXTRL_BE_ARGPTR_DEC NS(Drift)* SIXTRL_RESTRICT drift,
    NS(drift_real_t) const length );

SIXTRL_FN SIXTRL_STATIC void NS(Drift_clear)(
    SIXTRL_BE_ARGPTR_DEC NS(Drift)* SIXTRL_RESTRICT drift );

SIXTRL_FN SIXTRL_STATIC int NS(Drift_copy)(
    SIXTRL_BE_ARGPTR_DEC NS(Drift)* SIXTRL_RESTRICT destination,
    SIXTRL_BE_ARGPTR_DEC const NS(Drift) *const SIXTRL_RESTRICT source );

SIXTRL_FN SIXTRL_STATIC int NS(Drift_compare_values)(
    SIXTRL_BE_ARGPTR_DEC const NS(Drift) *const SIXTRL_RESTRICT lhs,
    SIXTRL_BE_ARGPTR_DEC const NS(Drift) *const SIXTRL_RESTRICT rhs );

SIXTRL_FN SIXTRL_STATIC int NS(Drift_compare_values_with_treshold)(
    SIXTRL_BE_ARGPTR_DEC const NS(Drift) *const SIXTRL_RESTRICT lhs,
    SIXTRL_BE_ARGPTR_DEC const NS(Drift) *const SIXTRL_RESTRICT rhs,
    NS(drift_real_t) const treshold );

/* ------------------------------------------------------------------------- */

SIXTRL_FN SIXTRL_STATIC SIXTRL_BUFFER_DATAPTR_DEC NS(Drift) const*
NS(BufferIndex_get_const_drift)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC const NS(Object) *const index_obj );

SIXTRL_FN SIXTRL_STATIC SIXTRL_BUFFER_DATAPTR_DEC NS(Drift)*
NS(BufferIndex_get_drift)( SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)* index_obj );

SIXTRL_FN SIXTRL_STATIC SIXTRL_BUFFER_DATAPTR_DEC NS(Drift) const*
NS(BeamElements_managed_buffer_get_const_drift)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT pbuffer,
    NS(buffer_size_t) const be_index,
    NS(buffer_size_t) const slot_size );

SIXTRL_FN SIXTRL_STATIC SIXTRL_BUFFER_DATAPTR_DEC NS(Drift)*
NS(BeamElements_managed_buffer_get_drift)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT pbuffer,
    NS(buffer_size_t) const be_index, NS(buffer_size_t) const slot_size );

#if !defined( _GPUCODE )

SIXTRL_STATIC SIXTRL_HOST_FN SIXTRL_BUFFER_DATAPTR_DEC NS(Drift) const*
NS(BeamElements_buffer_get_const_drift)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const be_index );

SIXTRL_STATIC SIXTRL_HOST_FN SIXTRL_BUFFER_DATAPTR_DEC NS(Drift)*
NS(BeamElements_buffer_get_drift)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const be_index );

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

SIXTRL_FN SIXTRL_STATIC NS(buffer_size_t) NS(DriftExact_get_num_slots)(
    SIXTRL_BE_ARGPTR_DEC const NS(DriftExact) *const SIXTRL_RESTRICT drift,
    NS(buffer_size_t) const slot_size );

SIXTRL_FN SIXTRL_STATIC SIXTRL_BE_ARGPTR_DEC NS(DriftExact)*
NS(DriftExact_preset)( SIXTRL_BE_ARGPTR_DEC
    NS(DriftExact)* SIXTRL_RESTRICT drift );

SIXTRL_FN SIXTRL_STATIC NS(drift_real_t) NS(DriftExact_get_length)(
    SIXTRL_BE_ARGPTR_DEC const NS(DriftExact) *const SIXTRL_RESTRICT drift );

SIXTRL_FN SIXTRL_STATIC void NS(DriftExact_set_length)(
    SIXTRL_BE_ARGPTR_DEC NS(DriftExact)* SIXTRL_RESTRICT drift,
    NS(drift_real_t) const length );

SIXTRL_FN SIXTRL_STATIC void NS(DriftExact_clear)(
    SIXTRL_BE_ARGPTR_DEC NS(DriftExact)* SIXTRL_RESTRICT drift );

SIXTRL_FN SIXTRL_STATIC int NS(DriftExact_copy)(
    SIXTRL_BE_ARGPTR_DEC NS(DriftExact)* SIXTRL_RESTRICT destination,
    SIXTRL_BE_ARGPTR_DEC const NS(DriftExact) *const SIXTRL_RESTRICT source );

SIXTRL_FN SIXTRL_STATIC int NS(DriftExact_compare_values)(
    SIXTRL_BE_ARGPTR_DEC const NS(DriftExact) *const SIXTRL_RESTRICT lhs,
    SIXTRL_BE_ARGPTR_DEC const NS(DriftExact) *const SIXTRL_RESTRICT rhs );

SIXTRL_FN SIXTRL_STATIC int NS(DriftExact_compare_values_with_treshold)(
    SIXTRL_BE_ARGPTR_DEC const NS(DriftExact) *const SIXTRL_RESTRICT lhs,
    SIXTRL_BE_ARGPTR_DEC const NS(DriftExact) *const SIXTRL_RESTRICT rhs,
    NS(drift_real_t) const treshold );


SIXTRL_FN SIXTRL_STATIC SIXTRL_BUFFER_DATAPTR_DEC NS(DriftExact) const*
NS(BufferIndex_get_const_drift_exact)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC const NS(Object) *const index_obj );

SIXTRL_FN SIXTRL_STATIC SIXTRL_BUFFER_DATAPTR_DEC NS(DriftExact)*
NS(BufferIndex_get_drift_exact)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)* index_obj );

SIXTRL_FN SIXTRL_STATIC SIXTRL_BUFFER_DATAPTR_DEC NS(DriftExact) const*
NS(BeamElements_managed_buffer_get_const_drift_exact)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT pbuffer,
    NS(buffer_size_t) const be_index,
    NS(buffer_size_t) const slot_size );

SIXTRL_FN SIXTRL_STATIC SIXTRL_BUFFER_DATAPTR_DEC NS(DriftExact)*
NS(BeamElements_managed_buffer_get_drift_exact)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT pbuffer,
    NS(buffer_size_t) const be_index,
    NS(buffer_size_t) const slot_size );

#if !defined( _GPUCODE )

SIXTRL_STATIC SIXTRL_HOST_FN SIXTRL_BUFFER_DATAPTR_DEC NS(DriftExact) const*
NS(BeamElements_buffer_get_const_drift_exact)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const be_index );

SIXTRL_STATIC SIXTRL_HOST_FN SIXTRL_BUFFER_DATAPTR_DEC NS(DriftExact)*
NS(BeamElements_buffer_get_drift_exact)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const be_index );

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
    #include "sixtracklib/common/buffer/managed_buffer_minimal.h"
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

SIXTRL_INLINE NS(buffer_size_t) NS(Drift_get_num_slots)(
    SIXTRL_BE_ARGPTR_DEC const NS(Drift) *const SIXTRL_RESTRICT drift,
    NS(buffer_size_t) const slot_size )
{
    typedef NS(buffer_size_t) buf_size_t;
    typedef NS(Drift)         beam_element_t;

    SIXTRL_STATIC_VAR buf_size_t const ZERO = ( buf_size_t )0u;

    ( void )drift;

    buf_size_t extent = NS(ManagedBuffer_get_slot_based_length)(
        sizeof( beam_element_t ), slot_size );

    SIXTRL_ASSERT( ( slot_size == ZERO ) || ( ( extent % slot_size ) == ZERO ) );
    return ( slot_size > ZERO ) ? ( extent / slot_size ) : ( ZERO );
}

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(Drift)* NS(Drift_preset)(
    SIXTRL_BE_ARGPTR_DEC NS(Drift)* SIXTRL_RESTRICT drift )
{
    NS(Drift_clear)( drift );
    return drift;
}

SIXTRL_INLINE  NS(drift_real_t) NS(Drift_get_length)(
    SIXTRL_BE_ARGPTR_DEC const NS(Drift) *const SIXTRL_RESTRICT drift )
{
    SIXTRL_ASSERT( drift != SIXTRL_NULLPTR );
    return drift->length;
}

SIXTRL_INLINE void NS(Drift_set_length)(
    SIXTRL_BE_ARGPTR_DEC NS(Drift)* SIXTRL_RESTRICT drift,
    NS(drift_real_t) const length )
{
    if( drift != SIXTRL_NULLPTR ) drift->length = length;
    return;
}

SIXTRL_INLINE void NS(Drift_clear)(
    SIXTRL_BE_ARGPTR_DEC NS(Drift)* SIXTRL_RESTRICT drift )
{
    NS(Drift_set_length)( drift, 0.0 );
    return;
}

SIXTRL_INLINE int NS(Drift_copy)(
    SIXTRL_BE_ARGPTR_DEC NS(Drift)* SIXTRL_RESTRICT destination,
    SIXTRL_BE_ARGPTR_DEC const NS(Drift) *const SIXTRL_RESTRICT source )
{
    int success = -1;

    if( ( destination != SIXTRL_NULLPTR ) && ( source != SIXTRL_NULLPTR ) )
    {
        NS(Drift_set_length)( destination, NS(Drift_get_length)( source ) );
        success = 0;
    }

    return success;
}

SIXTRL_INLINE int NS(Drift_compare_values)(
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

SIXTRL_INLINE int NS(Drift_compare_values_with_treshold)(
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
/* ------------------------------------------------------------------------- */

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(Drift) const*
NS(BufferIndex_get_const_drift)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC const NS(Object) *const index_obj )
{
    typedef NS(Drift) beam_element_t;
    typedef SIXTRL_BUFFER_OBJ_DATAPTR_DEC beam_element_t const* ptr_to_be_t;
    ptr_to_be_t ptr_to_be = SIXTRL_NULLPTR;

    if( ( index_obj != SIXTRL_NULLPTR ) &&
        ( NS(Object_get_type_id)( index_obj ) == NS(OBJECT_TYPE_DRIFT) ) &&
        ( NS(Object_get_size)( index_obj ) >= sizeof( beam_element_t ) ) )
    {
        ptr_to_be = ( ptr_to_be_t )( uintptr_t
            )NS(Object_get_begin_addr)( index_obj );
    }

    return ptr_to_be;
}

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(Drift)*
NS(BufferIndex_get_drift)( SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)* index_obj )
{
    return ( SIXTRL_BUFFER_OBJ_DATAPTR_DEC NS(Drift)*
        )NS(BufferIndex_get_const_drift)( index_obj );
}

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(Drift) const*
NS(BeamElements_managed_buffer_get_const_drift)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT pbuffer,
    NS(buffer_size_t) const be_index, NS(buffer_size_t) const slot_size )
{
    return NS(BufferIndex_get_const_drift)(
        NS(ManagedBuffer_get_const_object)( pbuffer, be_index, slot_size ) );
}

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(Drift)*
NS(BeamElements_managed_buffer_get_drift)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT pbuffer,
    NS(buffer_size_t) const be_index,
    NS(buffer_size_t) const slot_size )
{
    return NS(BufferIndex_get_drift)( NS(ManagedBuffer_get_object)(
        pbuffer, be_index, slot_size ) );
}

#if !defined( _GPUCODE )

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(Drift) const*
NS(BeamElements_buffer_get_const_drift)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const be_index )
{
    typedef SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* ptr_raw_t;
    return NS(BeamElements_managed_buffer_get_const_drift)(
        ( ptr_raw_t )( uintptr_t )NS(Buffer_get_data_begin_addr)( buffer ),
        be_index, NS(Buffer_get_slot_size)( buffer ) );
}

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(Drift)*
NS(BeamElements_buffer_get_drift)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const be_index )
{
    typedef SIXTRL_BUFFER_DATAPTR_DEC unsigned char* ptr_raw_t;
    return NS(BeamElements_managed_buffer_get_drift)(
        ( ptr_raw_t )( uintptr_t )NS(Buffer_get_data_begin_addr)( buffer ),
        be_index, NS(Buffer_get_slot_size)( buffer ) );
}

SIXTRL_INLINE  bool NS(Drift_can_be_added)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_objects,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_slots,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_dataptrs )
{
    typedef NS(Drift)           elem_t;
    typedef NS(buffer_size_t)   buf_size_t;

    buf_size_t const num_dataptrs = NS(Drift_get_num_dataptrs)( SIXTRL_NULLPTR );

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

    buf_size_t const num_dataptrs = NS(Drift_get_num_dataptrs)( SIXTRL_NULLPTR );

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

    buf_size_t const num_dataptrs = NS(Drift_get_num_dataptrs)( SIXTRL_NULLPTR );

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

SIXTRL_INLINE NS(buffer_size_t) NS(DriftExact_get_num_slots)(
    SIXTRL_BE_ARGPTR_DEC const NS(DriftExact) *const SIXTRL_RESTRICT drift,
    NS(buffer_size_t) const slot_size )
{
    typedef NS(buffer_size_t) buf_size_t;
    typedef NS(DriftExact)    beam_element_t;

    SIXTRL_STATIC_VAR buf_size_t const ZERO = ( buf_size_t )0u;

    ( void )drift;

    buf_size_t extent = NS(ManagedBuffer_get_slot_based_length)(
        sizeof( beam_element_t ), slot_size );

    SIXTRL_ASSERT( ( slot_size == ZERO ) || ( ( extent % slot_size ) == ZERO ) );
    return ( slot_size > ZERO ) ? ( extent / slot_size ) : ( ZERO );
}

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(DriftExact)* NS(DriftExact_preset)(
    SIXTRL_BE_ARGPTR_DEC NS(DriftExact)* SIXTRL_RESTRICT drift )
{
    NS(DriftExact_clear)( drift );
    return drift;
}

SIXTRL_INLINE  NS(drift_real_t) NS(DriftExact_get_length)(
    SIXTRL_BE_ARGPTR_DEC const NS(DriftExact) *const SIXTRL_RESTRICT drift )
{
    SIXTRL_ASSERT( drift != SIXTRL_NULLPTR );
    return drift->length;
}

SIXTRL_INLINE void NS(DriftExact_set_length)(
    SIXTRL_BE_ARGPTR_DEC NS(DriftExact)* SIXTRL_RESTRICT drift,
    NS(drift_real_t) const length )
{
    SIXTRL_ASSERT( drift != SIXTRL_NULLPTR );
    drift->length = length;
    return;
}

SIXTRL_INLINE void NS(DriftExact_clear)(
    SIXTRL_BE_ARGPTR_DEC NS(DriftExact)* SIXTRL_RESTRICT drift )
{
    NS(DriftExact_set_length)( drift, 0.0 );
    return;
}


SIXTRL_INLINE int NS(DriftExact_copy)(
    SIXTRL_BE_ARGPTR_DEC NS(DriftExact)* SIXTRL_RESTRICT destination,
    SIXTRL_BE_ARGPTR_DEC const NS(DriftExact) *const SIXTRL_RESTRICT source )
{
    int success = -1;

    if( ( destination != SIXTRL_NULLPTR ) && ( source != SIXTRL_NULLPTR ) )
    {
        NS(DriftExact_set_length)( destination,
                                   NS(DriftExact_get_length)( source ) );
        success = 0;
    }

    return success;
}

SIXTRL_INLINE int NS(DriftExact_compare_values)(
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

SIXTRL_INLINE int NS(DriftExact_compare_values_with_treshold)(
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

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(DriftExact) const*
NS(BufferIndex_get_const_drift_exact)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC const NS(Object) *const index_obj )
{
    typedef NS(DriftExact) beam_element_t;
    typedef SIXTRL_BUFFER_OBJ_DATAPTR_DEC beam_element_t const* ptr_to_be_t;
    ptr_to_be_t ptr_to_be = SIXTRL_NULLPTR;

    if( ( index_obj != SIXTRL_NULLPTR ) &&
        ( NS(Object_get_type_id)( index_obj ) == NS(OBJECT_TYPE_DRIFT_EXACT) ) &&
        ( NS(Object_get_size)( index_obj ) >= sizeof( beam_element_t ) ) )
    {
        ptr_to_be = ( ptr_to_be_t )( uintptr_t
            )NS(Object_get_begin_addr)( index_obj );
    }

    return ptr_to_be;
}

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(DriftExact)*
NS(BufferIndex_get_drift_exact)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)* index_obj )
{
    return ( SIXTRL_BUFFER_OBJ_DATAPTR_DEC NS(DriftExact)*
        )NS(BufferIndex_get_const_drift_exact)( index_obj );
}

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(DriftExact) const*
NS(BeamElements_managed_buffer_get_const_drift_exact)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT pbuffer,
    NS(buffer_size_t) const be_index,
    NS(buffer_size_t) const slot_size )
{
    return NS(BufferIndex_get_const_drift_exact)(
        NS(ManagedBuffer_get_const_object)( pbuffer, be_index, slot_size ) );
}

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(DriftExact)*
NS(BeamElements_managed_buffer_get_drift_exact)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT pbuffer,
    NS(buffer_size_t) const be_index,
    NS(buffer_size_t) const slot_size )
{
    return NS(BufferIndex_get_drift_exact)( NS(ManagedBuffer_get_object)(
        pbuffer, be_index, slot_size ) );
}

#if !defined( _GPUCODE )

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(DriftExact) const*
NS(BeamElements_buffer_get_const_drift_exact)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const be_index )
{
    typedef SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* ptr_raw_t;
    return NS(BeamElements_managed_buffer_get_const_drift_exact)(
        ( ptr_raw_t )( uintptr_t )NS(Buffer_get_data_begin_addr)( buffer ),
        be_index, NS(Buffer_get_slot_size)( buffer ) );
}

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(DriftExact)*
NS(BeamElements_buffer_get_drift_exact)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const be_index )
{
    typedef SIXTRL_BUFFER_DATAPTR_DEC unsigned char* ptr_raw_t;
    return NS(BeamElements_managed_buffer_get_drift_exact)(
        ( ptr_raw_t )( uintptr_t )NS(Buffer_get_data_begin_addr)( buffer ),
        be_index, NS(Buffer_get_slot_size)( buffer ) );
}

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

#endif /* SIXTRACKLIB_COMMON_BE_DRIFT_BEAM_ELEMENT_DRIFT_H__ */

/* end: sixtracklib/common/be_drift/be_drift.h */
