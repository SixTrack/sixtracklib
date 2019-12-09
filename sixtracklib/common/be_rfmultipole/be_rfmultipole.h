#ifndef SIXTRACKL_COMMON_BE_RFMULTIPOLE_BE_RFMULTIPOLE_C99_H__
#define SIXTRACKL_COMMON_BE_RFMULTIPOLE_BE_RFMULTIPOLE_C99_H__

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/control/definitions.h"
    #include "sixtracklib/common/buffer/buffer_type.h"
    #include "sixtracklib/common/buffer/buffer_object.h"
    #include "sixtracklib/common/internal/objects_type_id.h"
    #if !defined( _GPUCODE )
        #include "sixtracklib/common/buffer.h"
    #endif /* !defined( _GPUCODE ) */
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stddef.h>
    #include <stdlib.h>
    #include <stdint.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if defined( __cplusplus ) && !defined( _GPUCODE )
extern "C" {
#endif /* defined( __cplusplus ) && !defined( _GPUCODE ) */

typedef SIXTRL_REAL_T  NS(rf_multipole_real_t);
typedef SIXTRL_INT64_T NS(rf_multipole_int_t);

typedef struct NS(RFMultiPole)
{
    NS(rf_multipole_int_t)    order               SIXTRL_ALIGN( 8 );
    NS(rf_multipole_real_t)   voltage             SIXTRL_ALIGN( 8 );
    NS(rf_multipole_real_t)   frequency           SIXTRL_ALIGN( 8 );
    NS(rf_multipole_real_t)   lag                 SIXTRL_ALIGN( 8 );
    NS(buffer_addr_t)         bal                 SIXTRL_ALIGN( 8 );
    NS(buffer_addr_t)         p                   SIXTRL_ALIGN( 8 );
}
NS(RFMultiPole);

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_ARGPTR_DEC NS(RFMultiPole)*
NS(RFMultiPole_preset)(
    SIXTRL_BE_ARGPTR_DEC NS(RFMultiPole)* SIXTRL_RESTRICT multipole );

SIXTRL_STATIC SIXTRL_FN void NS(RFMultiPole_clear)(
    SIXTRL_BE_ARGPTR_DEC NS(RFMultiPole)* SIXTRL_RESTRICT multipole );

/* ------------------------------------------------------------------- */

SIXTRL_STATIC SIXTRL_FN NS(object_type_id_t) NS(RFMultiPole_type_id)(
    SIXTRL_BE_ARGPTR_DEC const NS(RFMultiPole) *const SIXTRL_RESTRICT mpole );

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t) NS(RFMultiPole_num_dataptrs)(
    SIXTRL_BE_ARGPTR_DEC const NS(RFMultiPole) *const SIXTRL_RESTRICT mpole );

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t) NS(RFMultiPole_num_slots)(
    SIXTRL_BE_ARGPTR_DEC const NS(RFMultiPole) *const SIXTRL_RESTRICT mpole,
    NS(buffer_size_t) const slot_size );

/* ------------------------------------------------------------------- */

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_DATAPTR_DEC NS(rf_multipole_real_t) const*
NS(RFMultiPole_const_bal)( SIXTRL_BE_ARGPTR_DEC const NS(RFMultiPole)
    *const SIXTRL_RESTRICT mpole );

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_DATAPTR_DEC NS(rf_multipole_real_t)*
NS(RFMultiPole_bal)( SIXTRL_BE_ARGPTR_DEC NS(RFMultiPole)* SIXTRL_RESTRICT mpole );

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_DATAPTR_DEC NS(rf_multipole_real_t) const*
NS(RFMultiPole_const_p)( SIXTRL_BE_ARGPTR_DEC const NS(RFMultiPole)
    *const SIXTRL_RESTRICT mpole );

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_DATAPTR_DEC NS(rf_multipole_real_t)*
NS(RFMultiPole_p)( SIXTRL_BE_ARGPTR_DEC NS(RFMultiPole)* SIXTRL_RESTRICT mpole );

/* ------------------------------------------------------------------- */

SIXTRL_STATIC SIXTRL_FN void NS(RFMultiPole_set_bal_addr)(
    SIXTRL_BE_ARGPTR_DEC NS(RFMultiPole)* SIXTRL_RESTRICT mpole,
    NS(buffer_addr_t) const bal_addr );

SIXTRL_STATIC SIXTRL_FN void NS(RFMultiPole_set_p_addr)(
    SIXTRL_BE_ARGPTR_DEC NS(RFMultiPole)* SIXTRL_RESTRICT mpole,
    NS(buffer_addr_t) const p_addr );

/* ------------------------------------------------------------------- */

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_ARGPTR_DEC NS(RFMultiPole) const*
NS(RFMultiPole_const_from_obj_index)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC const NS(Object) *const obj );

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_ARGPTR_DEC NS(RFMultiPole)*
NS(RFMultiPole_from_obj_index)( SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)* obj );

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_ARGPTR_DEC NS(RFMultiPole) const*
NS(RFMultiPole_const_from_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT buffer_begin,
    NS(buffer_size_t) const index, NS(buffer_size_t) const slot_size );

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_ARGPTR_DEC NS(RFMultiPole)*
NS(RFMultiPole_from_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT buffer_begin,
    NS(buffer_size_t) const index, NS(buffer_size_t) const slot_size );

#if !defined( _GPUCODE )

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BE_ARGPTR_DEC NS(RFMultiPole) const*
NS(RFMultiPole_const_from_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const index );

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BE_ARGPTR_DEC NS(RFMultiPole)*
NS(RFMultiPole_from_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const index );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t) NS(RFMultiPole_attributes_offsets)(
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT offsets_begin,
    NS(buffer_size_t) const max_num_offsets,
    SIXTRL_BUFFER_DATAPTR_DEC const NS(RFMultiPole) *const SIXTRL_RESTRICT data,
    NS(buffer_size_t) const slot_size );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t) NS(RFMultiPole_attributes_sizes)(
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT sizes_begin,
    NS(buffer_size_t) const max_num_sizes,
    SIXTRL_BUFFER_DATAPTR_DEC const NS(RFMultiPole) *const SIXTRL_RESTRICT data,
    NS(buffer_size_t) const slot_size );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t) NS(RFMultiPole_attributes_counts)(
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT counts_begin,
    NS(buffer_size_t) const max_num_counts,
    SIXTRL_BUFFER_DATAPTR_DEC const NS(RFMultiPole) *const SIXTRL_RESTRICT data );

/* ------------------------------------------------------------------------- */

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(RFMultiPole_can_be_added)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(rf_multipole_int_t) const order,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_objects,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_slots,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_dataptrs );

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BE_ARGPTR_DEC NS(RFMultiPole)*
NS(RFMultiPole_new)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(rf_multipole_int_t) const order );

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BE_ARGPTR_DEC NS(RFMultiPole)*
NS(RFMultiPole_add)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(rf_multipole_int_t) const order,
    NS(rf_multipole_real_t) const voltage,
    NS(rf_multipole_real_t) const frequency,
    NS(rf_multipole_real_t) const lag,
    SIXTRL_ARGPTR_DEC NS(rf_multipole_real_t) const* SIXTRL_RESTRICT bal_values,
    SIXTRL_ARGPTR_DEC NS(rf_multipole_real_t) const* SIXTRL_RESTRICT p_values );

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BE_ARGPTR_DEC NS(RFMultiPole)*
NS(RFMultiPole_add_copy)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(RFMultiPole) *const SIXTRL_RESTRICT mpole );

#endif /* !defined( _GPUCODE ) */

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(RFMultiPole_copy)(
    SIXTRL_BE_ARGPTR_DEC NS(RFMultiPole)* SIXTRL_RESTRICT dest,
    SIXTRL_BUFFER_DATAPTR_DEC const NS(RFMultiPole) *const SIXTRL_RESTRICT src );

#if defined( __cplusplus ) && !defined( _GPUCODE )
}
#endif /* defined( __cplusplus ) && !defined( _GPUCODE ) */

/* ****************************************************************** */

#if defined( __cplusplus ) && !defined( _GPUCODE )
extern "C" {
#endif /* defined( __cplusplus ) && !defined( _GPUCODE ) */

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(RFMultiPole)*
NS(RFMultiPole_preset)(
    SIXTRL_BE_ARGPTR_DEC NS(RFMultiPole)* SIXTRL_RESTRICT multipole )
{
    ( void )multipole;
    return multipole;
}

SIXTRL_INLINE void NS(RFMultiPole_clear)(
    SIXTRL_BE_ARGPTR_DEC NS(RFMultiPole)* SIXTRL_RESTRICT multipole )
{
    ( void )multipole;
}

/* ----------------------------------------------------------------------- */

SIXTRL_INLINE NS(object_type_id_t) NS(RFMultiPole_type_id)(
    SIXTRL_BE_ARGPTR_DEC const NS(RFMultiPole) *const SIXTRL_RESTRICT mpole )
{
    ( void )mpole;
    return NS(OBJECT_TYPE_RF_MULTIPOLE);
}

SIXTRL_INLINE NS(buffer_size_t) NS(RFMultiPole_num_dataptrs)(
    SIXTRL_BE_ARGPTR_DEC const NS(RFMultiPole) *const SIXTRL_RESTRICT mpole )
{
    ( void )mpole;
    return ( NS(buffer_size_t) )2u;
}

SIXTRL_INLINE NS(buffer_size_t) NS(RFMultiPole_num_slots)(
    SIXTRL_BE_ARGPTR_DEC const NS(RFMultiPole) *const SIXTRL_RESTRICT mpole,
    NS(buffer_size_t) const slot_size )
{
    NS(buffer_size_t) num_slots = ( NS(buffer_size_t) )0u;
    NS(buffer_size_t) buffer_size = NS(ManagedBuffer_get_slot_based_length)(
        sizeof( NS(RFMultiPole) ), slot_size );


    ( void )mpole;
    /* TODO: calculate size based on order; */

    num_slots = buffer_size / slot_size;

    if( num_slots * slot_size < buffer_size ) ++num_slots;
    return num_slots;
}

/* ------------------------------------------------------------------- */

SIXTRL_INLINE SIXTRL_BE_DATAPTR_DEC NS(rf_multipole_real_t) const*
NS(RFMultiPole_const_bal)( SIXTRL_BE_ARGPTR_DEC const NS(RFMultiPole)
    *const SIXTRL_RESTRICT mpole )
{
    SIXTRL_ASSERT( mpole != SIXTRL_NULLPTR );
    return ( SIXTRL_BE_DATAPTR_DEC NS(rf_multipole_real_t) const* )(
        uintptr_t )mpole->bal;
}

SIXTRL_INLINE SIXTRL_BE_DATAPTR_DEC NS(rf_multipole_real_t)*
NS(RFMultiPole_bal)( SIXTRL_BE_ARGPTR_DEC NS(RFMultiPole)* SIXTRL_RESTRICT mpole )
{
    return ( SIXTRL_BE_DATAPTR_DEC NS(rf_multipole_real_t)*
        )NS(RFMultiPole_const_bal)( mpole );
}

SIXTRL_INLINE SIXTRL_BE_DATAPTR_DEC NS(rf_multipole_real_t) const*
NS(RFMultiPole_const_p)( SIXTRL_BE_ARGPTR_DEC const NS(RFMultiPole)
    *const SIXTRL_RESTRICT mpole )
{
    SIXTRL_ASSERT( mpole != SIXTRL_NULLPTR );
    return ( SIXTRL_BE_DATAPTR_DEC NS(rf_multipole_real_t) const* )(
        uintptr_t )mpole->p;
}

SIXTRL_INLINE SIXTRL_BE_DATAPTR_DEC NS(rf_multipole_real_t)*
NS(RFMultiPole_p)( SIXTRL_BE_ARGPTR_DEC NS(RFMultiPole)* SIXTRL_RESTRICT mpole )
{
    return ( SIXTRL_BE_DATAPTR_DEC NS(rf_multipole_real_t)*
        )NS(RFMultiPole_const_p)( mpole );
}

/* ------------------------------------------------------------------- */

SIXTRL_INLINE void NS(RFMultiPole_set_bal_addr)(
    SIXTRL_BE_ARGPTR_DEC NS(RFMultiPole)* SIXTRL_RESTRICT mpole,
    NS(buffer_addr_t) const bal_addr )
{
    if( mpole != SIXTRL_NULLPTR ) mpole->bal = bal_addr;
}

SIXTRL_INLINE void NS(RFMultiPole_set_p_addr)(
    SIXTRL_BE_ARGPTR_DEC NS(RFMultiPole)* SIXTRL_RESTRICT mpole,
    NS(buffer_addr_t) const p_addr )
{
    if( mpole != SIXTRL_NULLPTR ) mpole->p = p_addr;
}

/* ------------------------------------------------------------------- */

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(RFMultiPole) const*
NS(RFMultiPole_const_from_obj_index)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC const NS(Object) *const obj )
{
    return (
        ( NS(Object_get_type_id)( obj ) == NS(OBJECT_TYPE_RF_MULTIPOLE) ) &&
        ( NS(Object_get_size)( obj ) >= sizeof( NS(RFMultiPole) ) ) )
        ? ( SIXTRL_BE_ARGPTR_DEC NS(RFMultiPole) const* )( uintptr_t
            )NS(Object_get_begin_addr)( obj )
        : SIXTRL_NULLPTR;
}

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(RFMultiPole)*
NS(RFMultiPole_from_obj_index)( SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)* obj )
{
    return ( SIXTRL_BE_ARGPTR_DEC NS(RFMultiPole)*
        )NS(RFMultiPole_const_from_obj_index)( obj );
}

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(RFMultiPole) const*
NS(RFMultiPole_const_from_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT buffer_begin,
    NS(buffer_size_t) const index, NS(buffer_size_t) const slot_size )
{
    return NS(RFMultiPole_const_from_obj_index)(
        NS(ManagedBuffer_get_const_object)( buffer_begin, index, slot_size ) );
}

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(RFMultiPole)*
NS(RFMultiPole_from_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT buffer_begin,
    NS(buffer_size_t) const index, NS(buffer_size_t) const slot_size )
{
    return NS(RFMultiPole_from_obj_index)( NS(ManagedBuffer_get_object)(
        buffer_begin, index, slot_size ) );
}

SIXTRL_INLINE NS(arch_status_t) NS(RFMultiPole_copy)(
    SIXTRL_BE_ARGPTR_DEC NS(RFMultiPole)* SIXTRL_RESTRICT dest,
    SIXTRL_BUFFER_DATAPTR_DEC const NS(RFMultiPole) *const SIXTRL_RESTRICT src )
{
    ( void )src;
    ( void )dest;

    return SIXTRL_ARCH_STATUS_GENERAL_FAILURE;
}

#if defined( __cplusplus ) && !defined( _GPUCODE )
}
#endif /* defined( __cplusplus ) && !defined( _GPUCODE ) */

#endif /* SIXTRACKL_COMMON_BE_RFMULTIPOLE_BE_RFMULTIPOLE_C99_H__ */
