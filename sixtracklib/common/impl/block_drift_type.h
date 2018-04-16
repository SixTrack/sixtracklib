#ifndef SIXTRACKLIB_COMMON_IMPL_BLOCK_DRIFT_TYPE_H__
#define SIXTRACKLIB_COMMON_IMPL_BLOCK_DRIFT_TYPE_H__

#if !defined( _GPUCODE )

#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "sixtracklib/_impl/definitions.h"
#include "sixtracklib/common/impl/block_type.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#endif /* !defined( _GPUCODE ) */
    
typedef struct NS(Drift)
{
    SIXTRL_UINT64_T  type_id;
    SIXTRL_REAL_T*   length;
    SIXTRL_INT64_T*  element_id;
}
NS(Drift);

#if !defined( _GPUCODE )

typedef struct NS(DriftSingle)
{
    SIXTRL_UINT64_T type_id;
    SIXTRL_REAL_T   length;
    SIXTRL_INT64_T  element_id;
}
NS(DriftSingle);

SIXTRL_STATIC SIXTRL_UINT64_T NS(DriftSingle_get_type_id)( 
    const NS(DriftSingle) *const SIXTRL_RESTRICT drift );

SIXTRL_STATIC SIXTRL_REAL_T NS(DriftSingle_get_length)(
    const NS(DriftSingle) *const SIXTRL_RESTRICT drift );

SIXTRL_STATIC SIXTRL_REAL_T* NS(DriftSingle_get_length_ptr)(
    NS(DriftSingle)* SIXTRL_RESTRICT drift );

SIXTRL_STATIC SIXTRL_INT64_T NS(DriftSingle_get_element_id)(
    const NS(DriftSingle) *const SIXTRL_RESTRICT drift );

SIXTRL_STATIC SIXTRL_INT64_T* NS(DriftSingle_get_element_id_ptr)(
    NS(DriftSingle)* SIXTRL_RESTRICT drift );

SIXTRL_STATIC void NS(DriftSingle_set_type_id)(
    NS(DriftSingle)* SIXTRL_RESTRICT drift, SIXTRL_UINT64_T const type_id );

SIXTRL_STATIC void NS(DriftSingle_set_length)(
    NS(DriftSingle)* SIXTRL_RESTRICT drift, SIXTRL_REAL_T const length );

SIXTRL_STATIC void NS(DriftSingle_set_element_id)(
    NS(DriftSingle)* SIXTRL_RESTRICT drift, SIXTRL_INT64_T const element_id );

/* -------------------------------------------------------------------------- */

SIXTRL_STATIC void NS(Drift_map_from_single_drift)(
    NS(Drift)* SIXTRL_RESTRICT drift, NS(DriftSingle)* SIXTRL_RESTRICT single );

#endif /* !defined( _GPUCODE ) */

/* -------------------------------------------------------------------------- */

SIXTRL_STATIC SIXTRL_UINT64_T NS(Drift_get_type_id)(
    const NS(Drift) *const SIXTRL_RESTRICT drift );

SIXTRL_STATIC void NS(Drift_set_type_id)(
    NS(Drift)* SIXTRL_RESTRICT drift, SIXTRL_UINT64_T const type_id );

/* -------------------------------------------------------------------------- */

SIXTRL_STATIC SIXTRL_INT64_T NS(Drift_get_element_id)( 
    const NS(Drift) *const SIXTRL_RESTRICT drift );

SIXTRL_STATIC SIXTRL_INT64_T* NS(Drift_get_element_id_ptr)( 
    NS(Drift)* SIXTRL_RESTRICT drift );

SIXTRL_STATIC void NS(Drift_set_element_id)(
    const NS(Drift) *const SIXTRL_RESTRICT drift, SIXTRL_INT64_T const elem_id );

SIXTRL_STATIC void NS(Drift_assign_ptr_to_element_id)(
    NS(Drift)* SIXTRL_RESTRICT drift, SIXTRL_INT64_T* ptr_element_id );
    
/* -------------------------------------------------------------------------- */

SIXTRL_STATIC SIXTRL_REAL_T NS(Drift_get_length)( 
    const NS(Drift) *const SIXTRL_RESTRICT drift );

SIXTRL_STATIC SIXTRL_REAL_T* NS(Drift_get_length_ptr)(
    NS(Drift)* SIXTRL_RESTRICT drift );

SIXTRL_STATIC void NS(Drift_set_length)( 
    NS(Drift)* SIXTRL_RESTRICT drift, SIXTRL_REAL_T const length );

SIXTRL_STATIC void NS(Drift_assign_ptr_to_length)(
    NS(Drift)* SIXTRL_RESTRICT drift, SIXTRL_REAL_T* ptr_length );

/* -------------------------------------------------------------------------- */

SIXTRL_STATIC SIXTRL_SIZE_T NS(Drift_map_to_flat_memory)(
    NS(Drift)* SIXTRL_RESTRICT mapped, unsigned char* mem_begin,
    NS(BeamElementType) const type_id, SIXTRL_SIZE_T const alignment );

SIXTRL_STATIC unsigned char* NS(Drift_unpack_from_flat_memory)(
    NS(Drift)* SIXTRL_RESTRICT drift, unsigned char* mem );

/* ************************************************************************** */
/* ******          Implementation of inline functions                   ***** */
/* ************************************************************************** */

SIXTRL_INLINE SIXTRL_UINT64_T NS(Drift_get_type_id)(
    const NS(Drift) *const SIXTRL_RESTRICT drift )
{
    SIXTRL_ASSERT( ( drift != 0 ) && (
        ( drift->type_id == NS(ELEMENT_TYPE_DRIFT) ) ||
        ( drift->type_id == NS(ELEMENT_TYPE_DRIFT_EXACT) ) ) );
    
    return drift->type_id;
}

SIXTRL_INLINE void NS(Drift_set_type_id)(
    NS(Drift)* SIXTRL_RESTRICT drift, SIXTRL_UINT64_T const type_id )
{
    SIXTRL_ASSERT( ( drift != 0 ) && ( 
        ( type_id == NS(ELEMENT_TYPE_DRIFT) ) ||
        ( type_id == NS(ELEMENT_TYPE_DRIFT_EXACT) ) ) );
    
    drift->type_id = type_id;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE SIXTRL_INT64_T NS(Drift_get_element_id)( 
    const NS(Drift) *const SIXTRL_RESTRICT drift )
{
    SIXTRL_ASSERT( ( drift != 0 ) && ( drift->element_id != 0 ) );
    return *( drift->element_id );
}

SIXTRL_INLINE SIXTRL_INT64_T* NS(Drift_get_element_id_ptr)( 
    NS(Drift)* SIXTRL_RESTRICT drift )
{
    SIXTRL_ASSERT( drift != 0 );
    return drift->element_id;
}

SIXTRL_INLINE void NS(Drift_set_element_id)(
    const NS(Drift) *const SIXTRL_RESTRICT drift, SIXTRL_INT64_T const elem_id )
{
    SIXTRL_ASSERT( ( drift != 0 ) && ( drift->element_id != 0 ) );    
    *( drift->element_id ) = elem_id;
    return;
}

SIXTRL_INLINE void NS(Drift_assign_ptr_to_element_id)(
    NS(Drift)* SIXTRL_RESTRICT drift, SIXTRL_INT64_T* ptr_element_id )
{
    SIXTRL_ASSERT( drift != 0 );
    drift->element_id = ptr_element_id;
    return;
}
    
/* -------------------------------------------------------------------------- */

SIXTRL_INLINE SIXTRL_REAL_T NS(Drift_get_length)( 
    const NS(Drift) *const SIXTRL_RESTRICT drift )
{
    SIXTRL_ASSERT( ( drift != 0 ) && ( drift->length != 0 ) );
    return *( drift->length );
}

SIXTRL_INLINE SIXTRL_REAL_T* NS(Drift_get_length_ptr)( 
    NS(Drift)* SIXTRL_RESTRICT drift )
{
    SIXTRL_ASSERT( drift != 0 );
    return drift->length;
}

SIXTRL_INLINE void NS(Drift_set_length)( 
    NS(Drift)* SIXTRL_RESTRICT drift, SIXTRL_REAL_T const length )
{
    SIXTRL_ASSERT( ( drift != 0 ) && ( drift->length != 0 ) );
    
    *( drift->length ) = length;
    return;
}

SIXTRL_INLINE void NS(Drift_assign_ptr_to_length)(
    NS(Drift)* SIXTRL_RESTRICT drift, SIXTRL_REAL_T* ptr_length )
{
    SIXTRL_ASSERT( drift != 0 );
    drift->length = ptr_length;
    return;
}

/* -------------------------------------------------------------------------- */

SIXTRL_INLINE SIXTRL_SIZE_T NS(Drift_map_to_flat_memory)(
    NS(Drift)* SIXTRL_RESTRICT mapped, unsigned char* mem, 
    NS(BeamElementType) const type_id, SIXTRL_SIZE_T const alignment )
{
    SIXTRL_STATIC SIXTRL_SIZE_T const ZERO = ( SIXTRL_SIZE_T )0u;
        
    SIXTRL_STATIC SIXTRL_SIZE_T const U64_SIZE = sizeof( SIXTRL_UINT64_T );
    SIXTRL_STATIC SIXTRL_SIZE_T const TYPEID_ADDR_OFFSET = 
        sizeof( SIXTRL_UINT64_T );
        
    SIXTRL_STATIC SIXTRL_SIZE_T const NUM_DRIFTS_ADDR_OFFSET = 
        sizeof( SIXTRL_UINT64_T ) * 2;        
        
    SIXTRL_STATIC SIXTRL_SIZE_T const NUM_ATTRS_ADDR_OFFSET  = 
        sizeof( SIXTRL_UINT64_T ) * 3;
        
    SIXTRL_STATIC SIXTRL_SIZE_T const LENGTH_OFF_ADDR_OFFSET = 
        sizeof( SIXTRL_UINT64_T ) * 4;
        
    SIXTRL_STATIC SIXTRL_SIZE_T const ELEMID_OFF_ADDR_OFFSET = 
        sizeof( SIXTRL_UINT64_T ) * 5;
        
    SIXTRL_STATIC SIXTRL_UINT64_T const NUM_ATTRIBUTES = ( SIXTRL_UINT64_T )2u;
    SIXTRL_STATIC SIXTRL_UINT64_T const NUM_ELEMENTS   = ( SIXTRL_UINT64_T )1u;
    
    SIXTRL_UINT64_T DATA_ADDR_OFFSET = U64_SIZE * 6;
    SIXTRL_UINT64_T const DATA_ADDR_OFFSET_MOD = DATA_ADDR_OFFSET % alignment;
    
    SIXTRL_UINT64_T LENGTH_BLOCK_LEN = sizeof( SIXTRL_REAL_T  );
    SIXTRL_UINT64_T const LENGTH_BLOCK_LEN_MOD = LENGTH_BLOCK_LEN % alignment;
    
    SIXTRL_UINT64_T ELEMID_BLOCK_LEN = sizeof( SIXTRL_INT64_T );
    SIXTRL_UINT64_T const ELEMID_BLOCK_LEN_MOD = ELEMID_BLOCK_LEN % alignment;
    
    #if !defined( _GPUCODE )
    
    assert( ( mapped != 0 ) && ( alignment >= U64_SIZE ) &&
            ( ( alignment % U64_SIZE ) == ZERO ) &&
            ( ( alignment % sizeof( SIXTRL_REAL_T  ) ) == ZERO ) &&
            ( ( alignment % sizeof( SIXTRL_INT64_T ) ) == ZERO ) &&
            ( ( ( ( uintptr_t )mem ) % U64_SIZE ) == ZERO ) &&
            ( ( type_id == NS(ELEMENT_TYPE_DRIFT) ) ||
              ( type_id == NS(ELEMENT_TYPE_DRIFT_EXACT ) ) ) );
    
    #endif /* !defined( _GPUCODE ) */
    
    NS(Drift_set_type_id)( mapped, ( SIXTRL_UINT64_T )type_id );
    
    DATA_ADDR_OFFSET += ( DATA_ADDR_OFFSET_MOD == ZERO ) 
        ? ZERO : ( alignment - DATA_ADDR_OFFSET_MOD );
        
    LENGTH_BLOCK_LEN += ( LENGTH_BLOCK_LEN_MOD == ZERO )
        ? ZERO : ( alignment - LENGTH_BLOCK_LEN_MOD );
    
    ELEMID_BLOCK_LEN += ( ELEMID_BLOCK_LEN_MOD == ZERO )
        ? ZERO : ( alignment - ELEMID_BLOCK_LEN_MOD );
        
    *( ( SIXTRL_UINT64_T* )( mem + TYPEID_ADDR_OFFSET ) ) = ( SIXTRL_UINT64_T )type_id;
    *( ( SIXTRL_UINT64_T* )( mem + NUM_DRIFTS_ADDR_OFFSET ) ) = NUM_ELEMENTS;
    *( ( SIXTRL_UINT64_T* )( mem + NUM_ATTRS_ADDR_OFFSET  ) ) = NUM_ATTRIBUTES;
           
    *( ( SIXTRL_UINT64_T* )( mem + LENGTH_OFF_ADDR_OFFSET ) ) = DATA_ADDR_OFFSET;
    NS(Drift_assign_ptr_to_length)( 
        mapped, ( SIXTRL_REAL_T* )( mem + DATA_ADDR_OFFSET ) );    
    DATA_ADDR_OFFSET += LENGTH_BLOCK_LEN;
    
    *( ( SIXTRL_UINT64_T* )( mem + ELEMID_OFF_ADDR_OFFSET ) ) = DATA_ADDR_OFFSET;
    NS(Drift_assign_ptr_to_element_id)(
        mapped, ( SIXTRL_INT64_T* )( mem + DATA_ADDR_OFFSET ) );
    DATA_ADDR_OFFSET += ELEMID_BLOCK_LEN;
    
    *( ( SIXTRL_UINT64_T* )( mem ) ) = DATA_ADDR_OFFSET;
    
    return DATA_ADDR_OFFSET;
}


SIXTRL_INLINE unsigned char* NS(Drift_unpack_from_flat_memory)(
    NS(Drift)* SIXTRL_RESTRICT drift, unsigned char* mem )
{
    SIXTRL_UINT64_T lengths_offset;
    SIXTRL_UINT64_T element_ids_offset;
    
    SIXTRL_STATIC const SIXTRL_SIZE_T U64_SIZE = sizeof( SIXTRL_UINT64_T );
    
    SIXTRL_STATIC const SIXTRL_SIZE_T TYPEID_ADDR_OFFSET = 
        sizeof( SIXTRL_UINT64_T );
    
    SIXTRL_STATIC const SIXTRL_SIZE_T NDRIFT_ADDR_OFFSET = 
        sizeof( SIXTRL_UINT64_T ) * 2;        
        
    SIXTRL_STATIC const SIXTRL_SIZE_T LENGTH_ADDR_OFFSET = 
        sizeof( SIXTRL_UINT64_T ) * 4;
        
    SIXTRL_STATIC const SIXTRL_SIZE_T ELEMID_ADDR_OFFSET = 
        sizeof( SIXTRL_UINT64_T ) * 5;
    
    unsigned char* ptr_num_drifts = 0;
    unsigned char* ptr_type_id    = 0;
    unsigned char* ptr_lengths    = 0;
    unsigned char* ptr_elemids    = 0;
    
    #if !defined( _GPUCODE )
    
    SIXTRL_STATIC SIXTRL_SIZE_T const ZERO = ( SIXTRL_SIZE_T )0u;
    SIXTRL_STATIC SIXTRL_SIZE_T const REAL_SIZE = sizeof( SIXTRL_REAL_T );
    SIXTRL_STATIC SIXTRL_SIZE_T const I64_SIZE  = sizeof( SIXTRL_INT64_T );

    SIXTRL_STATIC SIXTRL_SIZE_T const MIN_HEADER_LENGTH = 
        sizeof( SIXTRL_UINT64_T ) * 6;
    
    SIXTRL_UINT64_T* serial_len_ptr = ( SIXTRL_UINT64_T* )( mem );
    
    SIXTRL_UINT64_T* pack_id_ptr = ( SIXTRL_UINT64_T* )( mem + U64_SIZE );
    
    SIXTRL_STATIC SIXTRL_UINT64_T const CMP_DRIFT_ID = 
        ( SIXTRL_UINT64_T )NS(ELEMENT_TYPE_DRIFT);
    
    SIXTRL_STATIC SIXTRL_UINT64_T const CMP_DRIFT_EXACT_ID =
        ( SIXTRL_UINT64_T )NS(ELEMENT_TYPE_DRIFT_EXACT);
        
    SIXTRL_STATIC SIXTRL_UINT64_T const CMP_NUM_ATTR = ( SIXTRL_UINT64_T )2u;
    SIXTRL_UINT64_T* num_attr_ptr = ( SIXTRL_UINT64_T* )( mem + U64_SIZE * 3 );
    
    assert( ( mem != 0 ) && ( drift != 0 ) &&
            ( ( ( ( uintptr_t)mem ) % U64_SIZE ) == ZERO ) &&
            ( *serial_len_ptr >= MIN_HEADER_LENGTH ) &&
            ( ( *pack_id_ptr == CMP_DRIFT_ID ) || 
              ( *pack_id_ptr == CMP_DRIFT_EXACT_ID ) ) &&
            ( *num_attr_ptr  == CMP_NUM_ATTR ) );
    
    #endif /* !defined( GPU_CODE ) */
    
    ptr_type_id = mem + TYPEID_ADDR_OFFSET;
    NS(Drift_set_type_id)( drift, *( ( SIXTRL_UINT64_T* )ptr_type_id ) );
    
    ptr_num_drifts = mem + NDRIFT_ADDR_OFFSET;
    lengths_offset = *( ( SIXTRL_UINT64_T* )( mem + LENGTH_ADDR_OFFSET ) );
    ptr_lengths =  mem + lengths_offset;
    
    element_ids_offset = *( ( SIXTRL_UINT64_T* )( mem + ELEMID_ADDR_OFFSET ) );
    ptr_elemids = mem + element_ids_offset;
    
    #if !defined( _GPUCODE )
    
    assert( ( *ptr_num_drifts == ( SIXTRL_UINT64_T )1u ) &&
            ( ( ( ( uintptr_t )ptr_lengths ) % REAL_SIZE ) == ZERO ) &&
            ( ( ( ( uintptr_t )ptr_elemids ) % I64_SIZE  ) == ZERO ) );
    
    #endif /* !defined( _GPUCODE ) */
    
    NS(Drift_assign_ptr_to_length)( drift, ( SIXTRL_REAL_T* )ptr_lengths );    
    NS(Drift_assign_ptr_to_element_id)( drift, ( SIXTRL_INT64_T* )ptr_elemids );
    
    return ( mem + *serial_len_ptr );
}


/* -------------------------------------------------------------------------- */

#if !defined( _GPUCODE )

SIXTRL_INLINE SIXTRL_UINT64_T NS(DriftSingle_get_type_id)( 
    const NS(DriftSingle) *const SIXTRL_RESTRICT drift )
{
    SIXTRL_ASSERT( drift != 0 );
    return drift->type_id;
}

SIXTRL_INLINE SIXTRL_REAL_T NS(DriftSingle_get_length)(
    const NS(DriftSingle) *const SIXTRL_RESTRICT drift )
{
    SIXTRL_ASSERT( drift != 0 );
    return drift->length;
}

SIXTRL_INLINE SIXTRL_REAL_T* NS(DriftSingle_get_length_ptr)(
    NS(DriftSingle)* SIXTRL_RESTRICT drift )
{
    SIXTRL_ASSERT( drift != 0 );
    return &drift->length;
}

SIXTRL_INLINE SIXTRL_INT64_T NS(DriftSingle_get_element_id)(
    const NS(DriftSingle) *const SIXTRL_RESTRICT drift )
{
    SIXTRL_ASSERT( drift != 0 );
    return drift->element_id;
}

SIXTRL_INLINE SIXTRL_INT64_T* NS(DriftSingle_get_element_id_ptr)(
    NS(DriftSingle)* SIXTRL_RESTRICT drift )
{
    SIXTRL_ASSERT( drift != 0 );
    return &drift->element_id;
}

SIXTRL_INLINE void NS(DriftSingle_set_type_id)(
    NS(DriftSingle)* SIXTRL_RESTRICT drift, SIXTRL_UINT64_T const type_id )
{
    SIXTRL_ASSERT( 
        ( drift != 0 ) && 
        ( ( type_id == ( SIXTRL_UINT64_T )NS(ELEMENT_TYPE_DRIFT ) ) ||
          ( type_id == ( SIXTRL_UINT64_T )NS(ELEMENT_TYPE_DRIFT_EXACT ) ) ) );
    
    drift->type_id = type_id;
    return;
}

SIXTRL_INLINE void NS(DriftSingle_set_length)(
    NS(DriftSingle)* SIXTRL_RESTRICT drift, SIXTRL_REAL_T const length )
{
    SIXTRL_ASSERT( ( drift != 0 ) && ( length >= ( SIXTRL_REAL_T )0 ) );
    drift->length = length;
    return;
}

SIXTRL_INLINE void NS(DriftSingle_set_element_id)(
    NS(DriftSingle)* SIXTRL_RESTRICT drift, SIXTRL_INT64_T const element_id )
{
    SIXTRL_ASSERT( drift != 0 );
    drift->element_id = element_id;
    return;
}

SIXTRL_INLINE void NS(Drift_map_from_single_drift)(
    NS(Drift)* SIXTRL_RESTRICT drift, NS(DriftSingle)* SIXTRL_RESTRICT single )
{
    if( ( drift != 0 ) && ( single != 0 ) )
    {
        NS(Drift_set_type_id)( drift, NS(DriftSingle_get_type_id)( single ) );        
        
        NS(Drift_assign_ptr_to_length)( 
            drift, NS(DriftSingle_get_length_ptr)( single ) );        
        
        NS(Drift_assign_ptr_to_element_id)( 
            drift, NS(DriftSingle_get_element_id_ptr)( single ) );
    }
    
    return;
}

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* !defined( _GPUCODE ) */

/* -------------------------------------------------------------------------- */

#endif /* SIXTRACKLIB_COMMON_IMPL_BLOCK_DRIFT_TYPE_H__ */

/* end: sixtracklib/common/impl/block_drift_type.h */
