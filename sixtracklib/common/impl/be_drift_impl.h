#ifndef SIXTRACKLIB_COMMON_IMPL_BEAM_ELEMENT_DRIFT_IMPL_H__
#define SIXTRACKLIB_COMMON_IMPL_BEAM_ELEMENT_DRIFT_IMPL_H__

#if !defined( _GPUCODE )

#include "sixtracklib/_impl/definitions.h"

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "sixtracklib/common/impl/alignment_impl.h"
#include "sixtracklib/common/impl/block_info_impl.h"
#include "sixtracklib/common/impl/particles_impl.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#endif /* !defined( _GPUCODE ) */
   
typedef struct NS(DriftSingle)
{
    NS(block_type_num_t) type_id_num;
    SIXTRL_REAL_T        length;
    NS(element_id_t)     element_id;
}
NS(DriftSingle);

SIXTRL_STATIC int NS(DriftSingle_is_valid)( 
    const NS(DriftSingle) *const SIXTRL_RESTRICT drift );


SIXTRL_STATIC NS(block_type_num_t) NS(DriftSingle_get_type_id_num)( 
    const NS(DriftSingle) *const SIXTRL_RESTRICT drift );

SIXTRL_STATIC NS(BlockType) NS(DriftSingle_get_type_id)(
    const NS(DriftSingle) *const SIXTRL_RESTRICT drift );

SIXTRL_STATIC void NS(DriftSingle_set_type_id_num)(
    NS(DriftSingle)* SIXTRL_RESTRICT drift, 
    NS(block_type_num_t) const type_id );

SIXTRL_STATIC void NS(DriftSingle_set_type_id)(
    NS(DriftSingle)* SIXTRL_RESTRICT drift, 
    NS(BlockType) const type_id );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC SIXTRL_REAL_T NS(DriftSingle_get_length)(
    const NS(DriftSingle) *const SIXTRL_RESTRICT drift );

SIXTRL_STATIC void NS(DriftSingle_set_length)(
    NS(DriftSingle)* SIXTRL_RESTRICT drift, 
    SIXTRL_REAL_T const length );

SIXTRL_STATIC SIXTRL_REAL_T* NS(DriftSingle_get_length_ptr)(
    NS(DriftSingle)* SIXTRL_RESTRICT drift );

SIXTRL_STATIC SIXTRL_REAL_T const* NS(DriftSingle_get_length_const_ptr)(
    const NS(DriftSingle) *const SIXTRL_RESTRICT drift );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC SIXTRL_INT64_T NS(DriftSingle_get_element_id)(
    const NS(DriftSingle) *const SIXTRL_RESTRICT drift );

SIXTRL_STATIC SIXTRL_INT64_T* NS(DriftSingle_get_element_id_ptr)(
    NS(DriftSingle)* SIXTRL_RESTRICT drift );

SIXTRL_STATIC SIXTRL_INT64_T const* NS(DriftSingle_get_element_id_const_ptr)(
        const NS(DriftSingle) *const SIXTRL_RESTRICT drift );

SIXTRL_STATIC void NS(DriftSingle_set_element_id)(
    NS(DriftSingle)* SIXTRL_RESTRICT drift, SIXTRL_INT64_T const element_id );

/* ========================================================================= */

typedef struct NS(Drift)
{
    NS(block_type_num_t)     type_id_num;
    NS(block_num_elements_t) num_elements ;
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T*    SIXTRL_RESTRICT length;
    SIXTRL_GLOBAL_DEC NS(element_id_t)* SIXTRL_RESTRICT element_id;
}
NS(Drift);


SIXTRL_STATIC NS(Drift)* NS(Drift_preset)( NS(Drift)* SIXTRL_RESTRICT drift );

SIXTRL_STATIC int NS(Drift_is_not_exact)(
    const NS(Drift) *const SIXTRL_RESTRICT drift );

SIXTRL_STATIC int NS(Drift_is_exact)( 
    const NS(Drift) *const SIXTRL_RESTRICT drift );

SIXTRL_STATIC NS(block_num_elements_t) NS(Drift_get_num_elements)(
    const NS(Drift) *const SIXTRL_RESTRICT drift );

SIXTRL_STATIC void NS(Drift_set_num_elements)(
    NS(Drift)* SIXTRL_RESTRICT drift, 
    NS(block_num_elements_t) const num_elements );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC NS(block_type_num_t) NS(Drift_get_type_id_num)(
    const NS(Drift) *const SIXTRL_RESTRICT drift );

SIXTRL_STATIC NS(BlockType) NS(Drift_get_type_id)(
    const NS(Drift) *const SIXTRL_RESTRICT drift );

SIXTRL_STATIC void NS(Drift_set_type_id_num)(
    NS(Drift)* SIXTRL_RESTRICT drift, 
    NS(block_type_num_t) const type_id_num );

SIXTRL_STATIC void NS(Drift_set_type_id)(
    NS(Drift)* SIXTRL_RESTRICT drift, NS(BlockType) const type_id );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC NS(element_id_t) NS(Drift_get_element_id_value)( 
    const NS(Drift) *const SIXTRL_RESTRICT drift, 
    NS(block_size_t) const elem_index );

SIXTRL_STATIC SIXTRL_GLOBAL_DEC NS(element_id_t)* NS(Drift_get_element_id)( 
    NS(Drift)* SIXTRL_RESTRICT drift );

SIXTRL_STATIC SIXTRL_GLOBAL_DEC NS(element_id_t) const* 
NS(Drift_get_const_element_id)( const NS(Drift) *const SIXTRL_RESTRICT drift );

SIXTRL_STATIC void NS(Drift_set_element_id_value)(
    NS(Drift)* SIXTRL_RESTRICT drift, 
    NS(block_num_elements_t) const elem_index,
    NS(element_id_t) const elem_id );

SIXTRL_STATIC void NS(Drift_set_element_id)(
    NS(Drift)* SIXTRL_RESTRICT drift, 
    NS(element_id_t) const* SIXTRL_RESTRICT elem_ids );

SIXTRL_STATIC void NS(Drift_assign_ptr_to_element_id)(
    NS(Drift)* SIXTRL_RESTRICT drift, 
    SIXTRL_GLOBAL_DEC NS(element_id_t)* elem_id );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC SIXTRL_REAL_T NS(Drift_get_length_value)( 
    const NS(Drift) *const SIXTRL_RESTRICT drift, 
    NS(block_num_elements_t) const elem_index );

SIXTRL_STATIC SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* NS(Drift_get_length)( 
    NS(Drift)* SIXTRL_RESTRICT drift );

SIXTRL_STATIC SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const* 
NS(Drift_get_const_length)( const NS(Drift) *const SIXTRL_RESTRICT drift );

SIXTRL_STATIC void NS(Drift_set_length_value)(
    NS(Drift)* SIXTRL_RESTRICT drift, 
    NS(block_num_elements_t) const elem_index,
    SIXTRL_REAL_T const elem_id );

SIXTRL_STATIC void NS(Drift_set_length)(
    NS(Drift)* SIXTRL_RESTRICT drift, 
    SIXTRL_REAL_T const* SIXTRL_RESTRICT ptr_lenghts );

SIXTRL_STATIC void NS(Drift_assign_ptr_to_length)(
    NS(Drift)* SIXTRL_RESTRICT drift, 
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* lengths );

/* -------------------------------------------------------------------------- */

SIXTRL_STATIC int NS(Drift_map_to_memory_for_writing_aligned)(
    NS(Drift)* SIXTRL_RESTRICT drift,
    NS(BlockInfo)* SIXTRL_RESTRICT block_info, 
    SIXTRL_GLOBAL_DEC unsigned char* SIXTRL_RESTRICT mem_begin, 
    NS(block_size_t) max_num_bytes_on_mem );

SIXTRL_STATIC int NS(Drift_map_from_memory_for_reading_aligned)(
    NS(Drift)* SIXTRL_RESTRICT drift, 
    SIXTRL_GLOBAL_DEC NS(BlockInfo)*  SIXTRL_RESTRICT block_info,
    SIXTRL_GLOBAL_DEC unsigned char*  SIXTRL_RESTRICT mem_begin, 
    NS(block_size_t) const max_num_bytes_on_mem );

/* -------------------------------------------------------------------------- */

SIXTRL_STATIC int NS(Drift_create_one_on_memory)(
    NS(Drift)* SIXTRL_RESTRICT drift, 
    NS(BlockInfo)* SIXTRL_RESTRICT block_info, 
    const NS(BlockMappingInfo) *const SIXTRL_RESTRICT mapping_info,
    SIXTRL_GLOBAL_DEC unsigned char* SIXTRL_RESTRICT mem_begin, 
    NS(block_size_t) const  max_num_bytes_on_mem, 
    SIXTRL_REAL_T    const lengths, 
    NS(element_id_t) const element_ids );

SIXTRL_STATIC int NS(Drift_create_on_memory)(
    NS(Drift)* SIXTRL_RESTRICT drift, 
    NS(BlockInfo)* SIXTRL_RESTRICT block_info, 
    const NS(BlockMappingInfo) *const SIXTRL_RESTRICT mapping_info,
    SIXTRL_GLOBAL_DEC unsigned char* SIXTRL_RESTRICT mem_begin, 
    NS(block_size_t) const  max_num_bytes_on_mem, 
    SIXTRL_REAL_T    const* SIXTRL_RESTRICT lengths, 
    NS(element_id_t) const* SIXTRL_RESTRICT element_ids );

/* -------------------------------------------------------------------------- */

SIXTRL_STATIC int NS(Drift_track_particle_over_single_elem)( 
    NS(Particles)* SIXTRL_RESTRICT particles, 
    NS(block_num_elements_t) const particle_index,
    const NS(Drift) *const SIXTRL_RESTRICT drift, 
    NS(block_num_elements_t) const element_index );

SIXTRL_STATIC int NS(DriftExact_track_particle_over_single_elem)(
    NS(Particles)* SIXTRL_RESTRICT particles, 
    NS(block_num_elements_t) const particle_index,
    const NS(Drift) *const SIXTRL_RESTRICT drift, 
    NS(block_num_elements_t) const element_index );

SIXTRL_STATIC int NS(Drift_track_particle)( 
    NS(Particles)* SIXTRL_RESTRICT particles, 
    NS(block_num_elements_t) const particle_index,
    const NS(Drift) *const SIXTRL_RESTRICT drift );

SIXTRL_STATIC int NS(DriftExact_track_particle)(
    NS(Particles)* SIXTRL_RESTRICT particles, 
    NS(block_num_elements_t) const particle_index,
    const NS(Drift) *const SIXTRL_RESTRICT drift );

/* ========================================================================= */
/* ======             Implementation of inline functions            ======== */
/* ========================================================================= */

SIXTRL_INLINE int NS(DriftSingle_is_valid)( 
    const NS(DriftSingle) *const SIXTRL_RESTRICT drift )
{
    NS(BlockType) const type_id = NS(DriftSingle_get_type_id )( drift );
    
    return ( ( drift != 0 ) && 
             ( ( type_id == NS(BLOCK_TYPE_DRIFT) ) ||
               ( type_id == NS(BLOCK_TYPE_DRIFT_EXACT) ) ) &&
             ( NS(DriftSingle_get_length)( drift ) >= 
                ( SIXTRL_REAL_T )0.0 ) &&
             ( NS(DriftSingle_get_element_id)( drift ) >= 
                ( NS(element_id_t) )0 ) );
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  */

SIXTRL_INLINE NS(block_type_num_t) NS(DriftSingle_get_type_id_num)( 
    const NS(DriftSingle) *const SIXTRL_RESTRICT single_drift )
{
    SIXTRL_ASSERT( single_drift != 0 );    
    return single_drift->type_id_num;
}

SIXTRL_INLINE NS(BlockType) NS(DriftSingle_get_type_id)(
    const NS(DriftSingle) *const SIXTRL_RESTRICT single_drift )
{
    return NS(BlockType_from_number)( 
        NS(DriftSingle_get_type_id_num)( single_drift ) );
}

SIXTRL_INLINE void NS(DriftSingle_set_type_id_num)(
    NS(DriftSingle)* SIXTRL_RESTRICT single_drift, 
    NS(block_type_num_t) const type_id_num )
{
    SIXTRL_ASSERT( single_drift != 0 );    
    single_drift->type_id_num = type_id_num;
    
    return;
}

SIXTRL_INLINE void NS(DriftSingle_set_type_id)(
    NS(DriftSingle)* SIXTRL_RESTRICT single_drift, 
    NS(BlockType) const type_id )
{
    SIXTRL_ASSERT( single_drift != 0 );
    single_drift->type_id_num = NS(BlockType_to_number)( type_id );
    return;
}

SIXTRL_INLINE SIXTRL_REAL_T NS(DriftSingle_get_length)(
    const NS(DriftSingle) *const SIXTRL_RESTRICT single_drift )
{
    SIXTRL_ASSERT( single_drift != 0 );
    return single_drift->length;
}

SIXTRL_INLINE void NS(DriftSingle_set_length)(
    NS(DriftSingle)* SIXTRL_RESTRICT single_drift, SIXTRL_REAL_T const length )
{
    SIXTRL_ASSERT( single_drift != 0 );
    single_drift->length = length;
    
    return;
}

SIXTRL_INLINE SIXTRL_REAL_T* NS(DriftSingle_get_length_ptr)(
    NS(DriftSingle)* SIXTRL_RESTRICT single_drift )
{
    return ( SIXTRL_REAL_T* )NS(DriftSingle_get_length_const_ptr)( 
        single_drift );
}

SIXTRL_INLINE SIXTRL_REAL_T const* NS(DriftSingle_get_length_const_ptr)(
    const NS(DriftSingle) *const SIXTRL_RESTRICT single_drift )
{
    SIXTRL_ASSERT( single_drift != 0 );
    return &single_drift->length;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE NS(element_id_t) NS(DriftSingle_get_element_id)(
    const NS(DriftSingle) *const SIXTRL_RESTRICT single_drift )
{
    SIXTRL_ASSERT( single_drift != 0 );
    return single_drift->element_id;
}

SIXTRL_INLINE NS(element_id_t)* NS(DriftSingle_get_element_id_ptr)(
    NS(DriftSingle)* SIXTRL_RESTRICT single_drift )
{
    return ( NS(element_id_t)* )NS(DriftSingle_get_element_id_const_ptr)( 
        single_drift );
}

SIXTRL_INLINE NS(element_id_t) const* NS(DriftSingle_get_element_id_const_ptr)(
    const NS(DriftSingle) *const SIXTRL_RESTRICT single_drift )
{
    SIXTRL_ASSERT( single_drift != 0 );
    return &single_drift->element_id;
}


SIXTRL_INLINE void NS(DriftSingle_set_element_id)(
    NS(DriftSingle)* SIXTRL_RESTRICT single_drift, 
    NS(element_id_t) const element_id )
{
    SIXTRL_ASSERT( single_drift );
    single_drift->element_id = element_id;
    
    return;
}

/* ========================================================================= */

SIXTRL_INLINE NS(Drift)* NS(Drift_preset)( NS(Drift)* SIXTRL_RESTRICT drift )
{
    if( drift != 0 )
    {
        NS(Drift_set_type_id)( drift, NS(BLOCK_TYPE_INVALID) );
        NS(Drift_set_num_elements)( drift, 0 );
        NS(Drift_assign_ptr_to_length)( drift, 0 );
        NS(Drift_assign_ptr_to_element_id)( drift, 0 );
    }
    
    return drift;
}


SIXTRL_INLINE int NS(Drift_is_not_exact)(
    const NS(Drift) *const SIXTRL_RESTRICT drift )
{
    return ( NS(Drift_get_type_id)( drift ) == NS(BLOCK_TYPE_DRIFT ) );
}

SIXTRL_INLINE int NS(Drift_is_exact)( 
    const NS(Drift) *const SIXTRL_RESTRICT drift )
{
    return ( NS(Drift_get_type_id)( drift ) == NS(BLOCK_TYPE_DRIFT_EXACT ) );
}

SIXTRL_INLINE NS(block_num_elements_t) NS(Drift_get_num_elements)(
    const NS(Drift) *const SIXTRL_RESTRICT drift )
{
    SIXTRL_ASSERT( drift != 0 );
    return drift->num_elements;
}

SIXTRL_INLINE void NS(Drift_set_num_elements)(
    NS(Drift)* SIXTRL_RESTRICT drift,
    NS(block_num_elements_t) const num_elements )
{
    SIXTRL_ASSERT( drift != 0 );
    drift->num_elements = num_elements;
    return;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE NS(block_type_num_t) NS(Drift_get_type_id_num)(
    const NS(Drift) *const SIXTRL_RESTRICT drift )
{
    SIXTRL_ASSERT( drift != 0 );
    return drift->type_id_num;
}

SIXTRL_INLINE NS(BlockType) NS(Drift_get_type_id)(
    const NS(Drift) *const SIXTRL_RESTRICT drift )
{
    SIXTRL_ASSERT( drift != 0 );
    return NS(BlockType_from_number)( drift->type_id_num );
}

SIXTRL_INLINE void NS(Drift_set_type_id_num)(
    NS(Drift)* SIXTRL_RESTRICT drift, 
    NS(block_type_num_t) const type_id_num )
{
    SIXTRL_ASSERT( drift != 0 );
    drift->type_id_num = type_id_num;
}

SIXTRL_INLINE  void NS(Drift_set_type_id)(
    NS(Drift)* SIXTRL_RESTRICT drift, NS(BlockType) const type_id )
{
    SIXTRL_ASSERT( drift != 0 );
    drift->type_id_num = NS(BlockType_to_number)( type_id );
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE  NS(element_id_t) NS(Drift_get_element_id_value)( 
    const NS(Drift) *const SIXTRL_RESTRICT drift, 
    NS(block_size_t) const elem_index )
{
    SIXTRL_ASSERT( ( drift != 0 ) && ( drift->element_id != 0 ) &&
                   ( elem_index < drift->num_elements ) );
    return drift->element_id[ elem_index ];
}

SIXTRL_INLINE  SIXTRL_GLOBAL_DEC NS(element_id_t)* NS(Drift_get_element_id)( 
    NS(Drift)* SIXTRL_RESTRICT drift )
{
    typedef SIXTRL_GLOBAL_DEC NS(element_id_t)* g_ptr_i64_t;    
    return ( g_ptr_i64_t )NS(Drift_get_const_element_id)( drift );
}

SIXTRL_INLINE  SIXTRL_GLOBAL_DEC NS(element_id_t) const* 
NS(Drift_get_const_element_id)( 
    const NS(Drift) *const SIXTRL_RESTRICT drift )
{
    SIXTRL_ASSERT( drift != 0 );
    return drift->element_id;
}

SIXTRL_INLINE  void NS(Drift_set_element_id_value)(
    NS(Drift)* SIXTRL_RESTRICT drift, 
    NS(block_num_elements_t) const elem_index,
    NS(element_id_t) const elem_id )
{
    SIXTRL_ASSERT( ( drift != 0 ) && ( drift->element_id != 0 ) && 
                   ( elem_index < drift->num_elements ) );
    drift->element_id[ elem_index ] = elem_id;
    
    return;
}

SIXTRL_INLINE  void NS(Drift_set_element_id)(
    NS(Drift)* SIXTRL_RESTRICT drift, 
    NS(element_id_t) const* SIXTRL_RESTRICT elem_ids )
{
    SIXTRL_ASSERT( ( drift != 0 ) && ( drift->element_id != 0 ) && 
                   ( elem_ids != 0 ) );
    
    SIXTRACKLIB_COPY_VALUES( 
        NS(element_id_t), drift->element_id, elem_ids, drift->num_elements );
    
    return;
}

SIXTRL_INLINE  void NS(Drift_assign_ptr_to_element_id)(
    NS(Drift)* SIXTRL_RESTRICT drift, 
    SIXTRL_GLOBAL_DEC NS(element_id_t)* ptr_elem_id )
{
    SIXTRL_ASSERT( drift != 0 );
    drift->element_id = ptr_elem_id;
    
    return;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE  SIXTRL_REAL_T NS(Drift_get_length_value)( 
    const NS(Drift) *const SIXTRL_RESTRICT drift, 
    NS(block_num_elements_t) const elem_index )
{
    SIXTRL_ASSERT( ( drift != 0 ) && ( drift->length != 0 ) && 
                   ( elem_index < drift->num_elements ) );
    
    return drift->length[ elem_index ];
}

SIXTRL_INLINE  SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* NS(Drift_get_length)( 
    NS(Drift)* SIXTRL_RESTRICT drift )
{
    typedef SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* g_ptr_real_t;
    return ( g_ptr_real_t )NS(Drift_get_const_length)( drift );
}

SIXTRL_INLINE  SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const* 
NS(Drift_get_const_length)( const NS(Drift) *const SIXTRL_RESTRICT drift )
{
    SIXTRL_ASSERT( drift != 0 );
    return drift->length;
}

SIXTRL_INLINE  void NS(Drift_set_length_value)(
    NS(Drift)* SIXTRL_RESTRICT drift, NS(block_num_elements_t) const elem_index,
    SIXTRL_REAL_T const length )
{
    SIXTRL_ASSERT( ( drift != 0 ) && ( drift->length != 0 ) &&
                   ( elem_index < drift->num_elements ) );
    drift->length[ elem_index ] = length;
    
    return;
}

SIXTRL_INLINE  void NS(Drift_set_length)(
    NS(Drift)* SIXTRL_RESTRICT drift, 
    SIXTRL_REAL_T const* SIXTRL_RESTRICT lengths )
{
    SIXTRL_ASSERT( ( drift != 0 ) && ( lengths != 0 ) );
    SIXTRACKLIB_COPY_VALUES( 
        SIXTRL_REAL_T, drift->length, lengths, drift->num_elements );
    
    return;
}

SIXTRL_INLINE  void NS(Drift_assign_ptr_to_length)(
    NS(Drift)* SIXTRL_RESTRICT drift, 
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* ptr_lengths )
{
    SIXTRL_ASSERT( drift != 0 );
    drift->length = ptr_lengths;
    return;
}

/* -------------------------------------------------------------------------- */

SIXTRL_INLINE int NS(Drift_map_to_memory_for_writing_aligned)(
    NS(Drift)* SIXTRL_RESTRICT drift,
    NS(BlockInfo)* SIXTRL_RESTRICT block_info, 
    SIXTRL_GLOBAL_DEC unsigned char* SIXTRL_RESTRICT mem_begin, 
    NS(block_size_t) max_num_bytes_on_mem )
{
    typedef SIXTRL_GLOBAL_DEC unsigned char* g_ptr_uchar_t;
    
    int success = -1;
    
    NS(block_size_t) const num_attributes = ( NS(block_size_t) )2u;
    
    g_ptr_uchar_t attributes_ptr[] = { 0, 0 };
    
    NS(block_size_t) num_bytes_for_attribute[] = 
        { sizeof( SIXTRL_REAL_T ), sizeof( NS(element_id_t) ) };
    
    if( 0 == NS(BlockInfo_generic_map_to_memory_for_writing_aligned)(
        block_info, &attributes_ptr[ 0 ], &num_bytes_for_attribute[ 0 ], 
        num_attributes, NS(Drift_get_num_elements)( drift ), 
        NS(Drift_get_type_id)( drift ), mem_begin, max_num_bytes_on_mem ) )
    {
        typedef SIXTRL_GLOBAL_DEC SIXTRL_REAL_T*    g_ptr_real_t;
        typedef SIXTRL_GLOBAL_DEC NS(element_id_t)* g_ptr_elemid_t;
        
        SIXTRL_ASSERT(
            ( NS(BlockInfo_has_common_alignment)( block_info ) ) &&
            ( NS(BlockInfo_get_common_alignment)( block_info ) != 0 ) &&            
            ( attributes_ptr[ 0 ] != 0 ) && ( attributes_ptr[ 1 ] != 0 ) &&
            ( ( ( ( uintptr_t )attributes_ptr[ 0 ] ) % 
                  NS(BlockInfo_get_common_alignment)( block_info ) ) == 0 ) &&
            ( ( ( ( uintptr_t )attributes_ptr[ 1 ] ) % 
                  NS(BlockInfo_get_common_alignment)( block_info ) ) == 0 ) &&
            ( num_bytes_for_attribute[ 0 ] >= sizeof( SIXTRL_REAL_T ) * 
                NS(Drift_get_num_elements)( drift ) ) &&
            ( num_bytes_for_attribute[ 1 ] >= sizeof( NS(element_id_t) ) *
                NS(Drift_get_num_elements)( drift ) ) &&
            ( NS(BlockInfo_get_type_id)( block_info ) == 
              NS(Drift_get_type_id)( drift ) ) &&
            ( NS(BlockInfo_get_num_elements)( block_info ) ==
              NS(Drift_get_num_elements)( drift ) ) );
        
        NS(Drift_assign_ptr_to_length)( 
            drift, ( g_ptr_real_t )attributes_ptr[ 0 ] );
        
        NS(Drift_assign_ptr_to_element_id)(
            drift, ( g_ptr_elemid_t )attributes_ptr[ 1 ] );
        
        success = 0;
    }
    
    return success;
}

SIXTRL_INLINE int NS(Drift_map_from_memory_for_reading_aligned)(
    NS(Drift)* SIXTRL_RESTRICT drift, 
    SIXTRL_GLOBAL_DEC NS(BlockInfo)*  SIXTRL_RESTRICT block_info,
    SIXTRL_GLOBAL_DEC unsigned char*  SIXTRL_RESTRICT mem_begin, 
    NS(block_size_t) const max_num_bytes_on_mem )
{
    typedef SIXTRL_GLOBAL_DEC unsigned char* g_ptr_uchar_t;
    
    int success = -1;
    
    NS(block_size_t) const num_attributes = ( NS(block_size_t) )2u;
    
    NS(BlockType) const type_id = 
        NS(BlockInfo_get_type_id)( block_info );
    
    NS(block_num_elements_t) const num_elements = 
        NS(BlockInfo_get_num_elements)( block_info );
    
    g_ptr_uchar_t attributes_ptr[] = { 0, 0 };
    
    NS(block_size_t) num_bytes_for_attribute[] = 
        { sizeof( SIXTRL_REAL_T ), sizeof( NS(element_id_t) ) };
        
    if( ( num_elements > ( NS(block_num_elements_t) )0u ) &&
        ( ( type_id == NS(BLOCK_TYPE_DRIFT) ) ||
          ( type_id == NS(BLOCK_TYPE_DRIFT_EXACT) ) ) &&        
        ( 0 == NS(BlockInfo_generic_map_from_memory_for_reading_aligned)(
            block_info, &attributes_ptr[ 0 ], &num_bytes_for_attribute[ 0 ],
            num_attributes, mem_begin, max_num_bytes_on_mem ) ) )
    {
        typedef SIXTRL_GLOBAL_DEC SIXTRL_REAL_T*    g_ptr_real_t;
        typedef SIXTRL_GLOBAL_DEC NS(element_id_t)* g_ptr_elemid_t;
        
        SIXTRL_ASSERT(
            ( NS(BlockInfo_get_num_elements)( block_info ) == num_elements ) &&
            ( NS(BlockInfo_get_type_id)( block_info ) == type_id ) &&
            ( NS(BlockInfo_has_common_alignment)( block_info ) ) &&
            ( NS(BlockInfo_get_common_alignment)( block_info ) != 0 ) &&            
            ( attributes_ptr[ 0 ] != 0 ) && ( attributes_ptr[ 1 ] != 0 ) &&
            ( ( ( ( uintptr_t )attributes_ptr[ 0 ] ) % 
                  NS(BlockInfo_get_common_alignment)( block_info ) ) == 0 ) &&
            ( ( ( ( uintptr_t )attributes_ptr[ 1 ] ) % 
                  NS(BlockInfo_get_common_alignment)( block_info ) ) == 0 ) &&
            ( num_bytes_for_attribute[ 0 ] >= sizeof( SIXTRL_REAL_T ) * 
                NS(BlockInfo_get_num_elements)( block_info ) ) &&
            ( num_bytes_for_attribute[ 1 ] >= sizeof( NS(element_id_t) ) *
                NS(BlockInfo_get_num_elements)( block_info ) ) );
        
        NS(Drift_set_type_id)( drift, type_id );        
        NS(Drift_set_num_elements)( drift, num_elements );
        
        NS(Drift_assign_ptr_to_length)( 
            drift, ( g_ptr_real_t )attributes_ptr[ 0 ] );
        
        NS(Drift_assign_ptr_to_element_id)(
            drift, ( g_ptr_elemid_t )attributes_ptr[ 1 ] );
        
        success = 0;
    }
    
    return success;
}

/* -------------------------------------------------------------------------- */

SIXTRL_INLINE int NS(Drift_create_one_on_memory)(
    NS(Drift)* SIXTRL_RESTRICT drift, 
    NS(BlockInfo)* SIXTRL_RESTRICT block_info, 
    const NS(BlockMappingInfo) *const SIXTRL_RESTRICT mapping_info,
    SIXTRL_GLOBAL_DEC unsigned char* SIXTRL_RESTRICT mem_begin, 
    NS(block_size_t) const  max_num_bytes_on_mem, 
    SIXTRL_REAL_T    const length, 
    NS(element_id_t) const element_id )
{
    int success = -1;
    
    NS(block_num_elements_t) const num_elements =
        NS(BlockInfo_get_num_elements)( block_info );
        
    if( ( num_elements == ( NS(block_num_elements_t) )1u ) &&
        ( 0 == NS(Drift_map_to_memory_for_writing_aligned)( drift, block_info, 
            mem_begin, max_num_bytes_on_mem ) ) )
    {
        SIXTRL_ASSERT( NS(Drift_get_length)( drift ) != 0 );
        SIXTRL_ASSERT( NS(Drift_get_element_id)( drift ) != 0 );
        SIXTRL_ASSERT( NS(Drift_get_num_elements)( drift ) == num_elements );
        
        NS(Drift_set_length_value)( drift, 0, length );
        NS(Drift_set_element_id_value)( drift, 0, element_id );
        
        success = 0;
    }
            
    return success;
}

SIXTRL_INLINE int NS(Drift_create_on_memory)(
    NS(Drift)* SIXTRL_RESTRICT drift, 
    NS(BlockInfo)* SIXTRL_RESTRICT block_info, 
    const NS(BlockMappingInfo) *const SIXTRL_RESTRICT mapping_info,
    SIXTRL_GLOBAL_DEC unsigned char* SIXTRL_RESTRICT mem_begin, 
    NS(block_size_t) const  max_num_bytes_on_mem, 
    SIXTRL_REAL_T    const* SIXTRL_RESTRICT lengths, 
    NS(element_id_t) const* SIXTRL_RESTRICT element_ids )
{
    int success = -1;
    
    NS(block_num_elements_t) const num_elements =
        NS(BlockInfo_get_num_elements)( block_info );
        
    if( ( num_elements > ( NS(block_num_elements_t) )0u ) &&
        ( 0 == NS(Drift_map_to_memory_for_writing_aligned)( drift, block_info, 
            mem_begin, max_num_bytes_on_mem ) ) )
    {
        SIXTRL_ASSERT( NS(Drift_get_length)( drift ) != 0 );
        SIXTRL_ASSERT( NS(Drift_get_element_id)( drift ) != 0 );
        SIXTRL_ASSERT( NS(Drift_get_num_elements)( drift ) == num_elements );
        
        SIXTRACKLIB_COPY_VALUES( SIXTRL_REAL_T, NS(Drift_get_length)( drift ), 
                                 lengths, num_elements );
        
        SIXTRACKLIB_COPY_VALUES( NS(element_id_t), 
                                 NS(Drift_get_element_id)( drift ), 
                                 element_ids, num_elements );
        
        success = 0;
    }
            
    return success;    
}

/* -------------------------------------------------------------------------- */

SIXTRL_INLINE int NS(Drift_track_particle_over_single_elem)( 
    NS(Particles)* SIXTRL_RESTRICT particles, NS(block_num_elements_t) const ii, 
    const NS(Drift) *const SIXTRL_RESTRICT drift, NS(block_num_elements_t) const jj )
{
    typedef SIXTRL_REAL_T real_t;
    
    real_t const length = NS(Drift_get_length_value)( drift, jj );
    
    real_t const _rpp = NS(Particles_get_rpp_value)( particles, ii );
    real_t const _px  = NS(Particles_get_px_value )( particles, ii ) * _rpp; 
    real_t const _py  = NS(Particles_get_py_value )( particles, ii ) * _rpp;    
    real_t const dsigma = ( 1.0 - NS(Particles_get_rvv_value)( particles, ii ) * 
                          ( 1.0 + ( _px * _px + _py * _py ) / 2.0 ) );
    
    real_t _sigma = NS(Particles_get_sigma_value)( particles, ii );
    real_t _s     = NS(Particles_get_s_value)( particles, ii );
    real_t _x     = NS(Particles_get_x_value)( particles, ii );
    real_t _y     = NS(Particles_get_y_value)( particles, ii );
    
    SIXTRL_ASSERT( 
        ( particles != 0 ) && ( drift != 0 ) && 
        ( NS(Drift_get_type_id)( drift ) == NS(BLOCK_TYPE_DRIFT) ) &&
        ( ii < NS(Particles_get_num_particles)( particles ) ) &&
        ( jj < NS(Drift_get_num_elements)( drift ) ) );
    
    _sigma += length * dsigma;
    _s     += length;
    _x     += length * _px;
    _y     += length * _py;
    
    NS(Particles_set_s_value)( particles, ii, _s );
    NS(Particles_set_x_value)( particles, ii, _x );
    NS(Particles_set_y_value)( particles, ii, _y );
    NS(Particles_set_sigma_value)( particles, ii, _sigma );
    
    return 0;
}

SIXTRL_INLINE int NS(DriftExact_track_particle_over_single_elem)(
    NS(Particles)* SIXTRL_RESTRICT particles, 
    NS(block_num_elements_t) const ii,
    const NS(Drift) *const SIXTRL_RESTRICT drift, 
    NS(block_num_elements_t) const jj )
{
    typedef SIXTRL_REAL_T real_t;
    
    SIXTRL_STATIC real_t const ONE = ( real_t )1u;
    
    real_t const _length = NS(Drift_get_length_value)( drift, jj );
    real_t const _delta  = NS(Particles_get_delta_value)( particles, ii );
    real_t const _beta0  = NS(Particles_get_beta0_value)( particles, ii );
    real_t const _px     = NS(Particles_get_px_value)(    particles, ii );
    real_t const _py     = NS(Particles_get_py_value)(    particles, ii );
    real_t _sigma        = NS(Particles_get_sigma_value)( particles, ii );
                        
    real_t const _opd    = _delta + ONE;
    real_t const _lpzi   = ( _length ) / sqrt( 
        _opd * _opd - _px * _px - _py * _py );
    
    real_t const _lbzi   = ( _beta0 * _beta0 * _sigma + ONE ) * _lpzi;
    
    real_t _x     = NS(Particles_get_x_value)( particles, ii );
    real_t _y     = NS(Particles_get_y_value)( particles, ii );
    real_t _s     = NS(Particles_get_s_value)( particles, ii );
    
    SIXTRL_ASSERT( 
        ( particles != 0 ) && ( drift != 0 ) && 
        ( NS(Drift_get_type_id)( drift ) == NS(BLOCK_TYPE_DRIFT) ) &&
        ( ii < NS(Particles_get_num_particles)( particles ) ) &&
        ( jj < NS(Drift_get_num_elements)( drift ) ) );
    
    _x     += _px * _lpzi;
    _y     += _py * _lpzi;
    _s     += _length;
    _sigma += _length - _lbzi;
    
    NS(Particles_set_x_value)(     particles, ii, _x     );
    NS(Particles_set_y_value)(     particles, ii, _y     );
    NS(Particles_set_s_value)(     particles, ii, _s     );
    NS(Particles_set_sigma_value)( particles, ii, _sigma );
    
    return 0;
}

SIXTRL_INLINE int NS(Drift_track_particle)( 
    NS(Particles)* SIXTRL_RESTRICT particles, 
    NS(block_num_elements_t) const particle_index,
    const NS(Drift) *const SIXTRL_RESTRICT drift )
{
    int success = 0;
    
    NS(block_num_elements_t) jj = 0;
    NS(block_num_elements_t) const nn = NS(Drift_get_num_elements)( drift );
    
    for( ; jj < nn ; ++jj )
    {
        success |= NS(Drift_track_particle_over_single_elem)( 
            particles, particle_index, drift, jj );
    }
    
    return success;
}

SIXTRL_INLINE int NS(DriftExact_track_particle)(
    NS(Particles)* SIXTRL_RESTRICT particles, 
    NS(block_num_elements_t) const particle_index,
    const NS(Drift) *const SIXTRL_RESTRICT drift )
{
    int success = 0;
    
    NS(block_num_elements_t) jj = 0;
    NS(block_num_elements_t) const nn = NS(Drift_get_num_elements)( drift );
    
    for( ; jj < nn ; ++jj )
    {
        success |= NS(DriftExact_track_particle_over_single_elem)( 
            particles, particle_index, drift, jj );
    }
    
    return success;
}


#if !defined( _GPUCODE )
    
#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* !defined( _GPUCODE ) */

#endif /* SIXTRACKLIB_COMMON_IMPL_BEAM_ELEMENT_DRIFT_IMPL_H__ */

/* end: sixtracklib/common/impl/be_drift_impl.h */
