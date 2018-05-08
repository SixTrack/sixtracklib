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
    NS(block_type_num_t)                type_id_num;
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T*    SIXTRL_RESTRICT length;
    SIXTRL_GLOBAL_DEC NS(element_id_t)* SIXTRL_RESTRICT element_id;
}
NS(Drift);


SIXTRL_STATIC NS(Drift)* NS(Drift_preset)( 
    NS(Drift)* SIXTRL_RESTRICT drift );

SIXTRL_STATIC int NS(Drift_is_not_exact)(
    const NS(Drift) *const SIXTRL_RESTRICT drift );

SIXTRL_STATIC int NS(Drift_is_exact)( 
    const NS(Drift) *const SIXTRL_RESTRICT drift );

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
    const NS(Drift) *const SIXTRL_RESTRICT drift );

SIXTRL_STATIC SIXTRL_GLOBAL_DEC NS(element_id_t)* NS(Drift_get_element_id)( 
    NS(Drift)* SIXTRL_RESTRICT drift );

SIXTRL_STATIC SIXTRL_GLOBAL_DEC NS(element_id_t) const* 
NS(Drift_get_const_element_id)( const NS(Drift) *const SIXTRL_RESTRICT drift );

SIXTRL_STATIC void NS(Drift_set_element_id_value)( 
    NS(Drift)* SIXTRL_RESTRICT drift, 
    NS(element_id_t) const elem_id );

SIXTRL_STATIC void NS(Drift_assign_ptr_to_element_id)(
    NS(Drift)* SIXTRL_RESTRICT drift, 
    SIXTRL_GLOBAL_DEC NS(element_id_t)* elem_id );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC SIXTRL_REAL_T NS(Drift_get_length_value)( 
    const NS(Drift) *const SIXTRL_RESTRICT drift );

SIXTRL_STATIC SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* NS(Drift_get_length)( 
    NS(Drift)* SIXTRL_RESTRICT drift );

SIXTRL_STATIC SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const* 
NS(Drift_get_const_length)( const NS(Drift) *const SIXTRL_RESTRICT drift );

SIXTRL_STATIC void NS(Drift_set_length_value)( NS(Drift)* SIXTRL_RESTRICT drift, 
    SIXTRL_REAL_T const length );

SIXTRL_STATIC void NS(Drift_assign_ptr_to_length)(
    NS(Drift)* SIXTRL_RESTRICT drift, 
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* lengths );

/* -------------------------------------------------------------------------- */

SIXTRL_STATIC int NS(Drift_create_on_memory)(
    NS(Drift)* SIXTRL_RESTRICT drift,
    SIXTRL_GLOBAL_DEC NS(BlockInfo)* SIXTRL_RESTRICT block_info, 
    SIXTRL_GLOBAL_DEC unsigned char* SIXTRL_RESTRICT mem_begin, 
    NS(block_size_t) max_num_bytes_on_mem );

SIXTRL_STATIC int NS(Drift_remap_from_memory)(
    NS(Drift)* SIXTRL_RESTRICT drift, 
    const SIXTRL_GLOBAL_DEC NS(BlockInfo) *const SIXTRL_RESTRICT block_info,
    SIXTRL_GLOBAL_DEC unsigned char* SIXTRL_RESTRICT mem_begin, 
    NS(block_size_t) const max_num_bytes_on_mem );

/* -------------------------------------------------------------------------- */

SIXTRL_STATIC int NS(Drift_add_to_memory)(
    NS(Drift)* SIXTRL_RESTRICT drift, 
    SIXTRL_GLOBAL_DEC NS(BlockInfo)* SIXTRL_RESTRICT block_info, 
    const NS(BlockMappingInfo) *const SIXTRL_RESTRICT mapping_info,
    SIXTRL_GLOBAL_DEC unsigned char* SIXTRL_RESTRICT mem_begin, 
    NS(block_size_t) const  max_num_bytes_on_mem, 
    SIXTRL_REAL_T    const lengths, 
    NS(element_id_t) const element_id );

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
//              ( NS(DriftSingle_get_length)( drift ) >= 
//                 ( SIXTRL_REAL_T )0.0 ) &&
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
    NS(DriftSingle)* SIXTRL_RESTRICT drift, SIXTRL_INT64_T const element_id )
{
    SIXTRL_ASSERT( drift != 0 );
    drift->element_id = element_id;
    return;
}

/* ========================================================================= */

SIXTRL_INLINE NS(Drift)* NS(Drift_preset)( NS(Drift)* SIXTRL_RESTRICT drift )
{
    if( drift != 0 )
    {
        NS(Drift_set_type_id)( drift, NS(BLOCK_TYPE_INVALID) );
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
    const NS(Drift) *const SIXTRL_RESTRICT drift )
{
    SIXTRL_ASSERT( ( drift != 0 ) && ( drift->element_id != 0 ) );
    return *( drift->element_id );
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
    NS(Drift)* SIXTRL_RESTRICT drift, NS(element_id_t) const elem_id )
{
    SIXTRL_ASSERT( ( drift != 0 ) && ( drift->element_id != 0 ) );
    *( drift->element_id ) = elem_id;
    
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
    const NS(Drift) *const SIXTRL_RESTRICT drift )
{
    SIXTRL_ASSERT( ( drift != 0 ) && ( drift->length != 0 ) );
    return *( drift->length );
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
    NS(Drift)* SIXTRL_RESTRICT drift, SIXTRL_REAL_T const length )
{
    SIXTRL_ASSERT( ( drift != 0 ) && ( drift->length != 0 ) );
    *drift->length = length;
    
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

SIXTRL_INLINE int NS(Drift_create_on_memory)(
    NS(Drift)* SIXTRL_RESTRICT drift,
    SIXTRL_GLOBAL_DEC NS(BlockInfo)* SIXTRL_RESTRICT block_info, 
    SIXTRL_GLOBAL_DEC unsigned char* SIXTRL_RESTRICT mem_begin, 
    NS(block_size_t) max_num_bytes_on_mem )
{
    typedef SIXTRL_GLOBAL_DEC unsigned char* g_ptr_uchar_t;
    
    int success = -1;
    
    NS(block_size_t) const num_attributes = ( NS(block_size_t) )2u;
    
    g_ptr_uchar_t attributes_ptr[] = { 0, 0 };
    
    NS(block_size_t) num_bytes_for_attribute[] = 
        { sizeof( SIXTRL_REAL_T ), sizeof( NS(element_id_t) ) };
    
    if( 0 == NS(BlockInfo_map_to_memory_aligned)(
        block_info, &attributes_ptr[ 0 ], &num_bytes_for_attribute[ 0 ], 
        num_attributes, ( NS(block_num_elements_t) )1u, 
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
            ( num_bytes_for_attribute[ 0 ] >= sizeof( SIXTRL_REAL_T ) ) &&
            ( num_bytes_for_attribute[ 1 ] >= sizeof( NS(element_id_t) ) ) &&
            ( NS(BlockInfo_get_type_id)( block_info ) == 
              NS(Drift_get_type_id)( drift ) ) &&
            ( NS(BlockInfo_get_num_elements)( block_info ) == 
              ( NS(block_num_elements_t) )1u ) );
        
        NS(Drift_assign_ptr_to_length)( 
            drift, ( g_ptr_real_t )attributes_ptr[ 0 ] );
        
        NS(Drift_assign_ptr_to_element_id)(
            drift, ( g_ptr_elemid_t )attributes_ptr[ 1 ] );
        
        success = 0;
    }
    
    return success;
}

/* -------------------------------------------------------------------------- */

SIXTRL_INLINE int NS(Drift_remap_from_memory)(
    NS(Drift)* SIXTRL_RESTRICT drift, 
    const SIXTRL_GLOBAL_DEC NS(BlockInfo) *const SIXTRL_RESTRICT block_info,
    SIXTRL_GLOBAL_DEC unsigned char* SIXTRL_RESTRICT mem_begin, 
    NS(block_size_t) const max_num_bytes_on_mem )
{
    typedef SIXTRL_GLOBAL_DEC unsigned char* g_ptr_uchar_t;
    
    int success = -1;
    
    NS(block_size_t) const num_attributes = ( NS(block_size_t) )2u;
    
    NS(BlockType) const type_id = 
        NS(BlockInfo_get_type_id)( block_info );
    
    g_ptr_uchar_t attributes_ptr[] = { 0, 0 };
    
    NS(block_size_t) num_bytes_for_attribute[] = 
        { sizeof( SIXTRL_REAL_T ), sizeof( NS(element_id_t) ) };
        
    if( ( NS(BlockInfo_get_num_elements)( block_info ) == 
            ( NS(block_num_elements_t) )1u ) &&
        ( ( type_id == NS(BLOCK_TYPE_DRIFT) ) ||
          ( type_id == NS(BLOCK_TYPE_DRIFT_EXACT) ) ) &&        
        ( 0 == NS(BlockInfo_remap_from_memory_aligned)(block_info, 
                &attributes_ptr[ 0 ], &num_bytes_for_attribute[ 0 ],
                num_attributes, mem_begin, max_num_bytes_on_mem ) ) )
    {
        typedef SIXTRL_GLOBAL_DEC SIXTRL_REAL_T*    g_ptr_real_t;
        typedef SIXTRL_GLOBAL_DEC NS(element_id_t)* g_ptr_elemid_t;
        
        SIXTRL_ASSERT(
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
        
        NS(Drift_assign_ptr_to_length)( 
            drift, ( g_ptr_real_t )attributes_ptr[ 0 ] );
        
        NS(Drift_assign_ptr_to_element_id)(
            drift, ( g_ptr_elemid_t )attributes_ptr[ 1 ] );
        
        success = 0;
    }
    
    return success;
}

/* -------------------------------------------------------------------------- */

SIXTRL_INLINE int NS(Drift_add_to_memory)(
    NS(Drift)* SIXTRL_RESTRICT drift, 
    SIXTRL_GLOBAL_DEC NS(BlockInfo)* SIXTRL_RESTRICT block_info, 
    const NS(BlockMappingInfo) *const SIXTRL_RESTRICT mapping_info,
    SIXTRL_GLOBAL_DEC unsigned char* SIXTRL_RESTRICT mem_begin, 
    NS(block_size_t) const  max_num_bytes_on_mem, 
    SIXTRL_REAL_T    const length, 
    NS(element_id_t) const element_id )
{
    int success = -1;
    
    if( ( NS(BlockInfo_get_num_elements)( block_info ) == 
            ( NS(block_num_elements_t) )1u ) &&
        ( 0 == NS(Drift_create_on_memory)( drift, block_info, 
            mem_begin, max_num_bytes_on_mem ) ) )
    {
        SIXTRL_ASSERT( NS(Drift_get_const_length)( drift ) != 0 );
        SIXTRL_ASSERT( NS(Drift_get_const_element_id)( drift ) != 0 );
        
        NS(Drift_set_length_value)( drift, length );
        NS(Drift_set_element_id_value)( drift, element_id );
        
        success = 0;
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
