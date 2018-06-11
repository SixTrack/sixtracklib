#ifndef SIXTRACKLIB_COMMON_IMPL_BLOCK_INFO_IMPL_H__
#define SIXTRACKLIB_COMMON_IMPL_BLOCK_INFO_IMPL_H__

#if !defined( _GPUCODE )

#include "sixtracklib/_impl/definitions.h"

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include "sixtracklib/common/impl/alignment_impl.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#endif /* !defined( _GPUCODE ) */

typedef SIXTRL_UINT64_T NS(block_size_t);
typedef SIXTRL_INT64_T  NS(block_num_elements_t);
typedef SIXTRL_UINT32_T NS(block_type_num_t);
typedef SIXTRL_INT32_T  NS(block_map_info_id_t);
typedef SIXTRL_INT32_T  NS(block_alignment_t);
typedef SIXTRL_INT32_T  NS(block_store_info_t);
    
SIXTRL_STATIC SIXTRL_GLOBAL_DEC unsigned char*
NS(Block_map_begin_to_memory_aligned)(
    NS(block_size_t)* SIXTRL_RESTRICT ptr_mem_offset,
    NS(block_alignment_t) const alignment, 
    SIXTRL_GLOBAL_DEC unsigned char* SIXTRL_RESTRICT mem_begin,
    NS(block_size_t) const max_num_of_bytes_in_buffer );

SIXTRL_STATIC SIXTRL_GLOBAL_DEC unsigned char* 
NS(Block_map_attribute_to_memory_aligned)(
    NS(block_size_t)* SIXTRL_RESTRICT ptr_num_bytes_in_block,
    NS(block_size_t) const mem_offset,
    NS(block_size_t) const num_bytes_for_attribute, 
    NS(block_alignment_t) const alignment,
    SIXTRL_GLOBAL_DEC unsigned char* SIXTRL_RESTRICT mem_begin,
    NS(block_size_t) const max_num_of_bytes_in_buffer );

SIXTRL_STATIC SIXTRL_GLOBAL_DEC unsigned char*
NS(Block_map_attribute_from_memory_aligned)(
    NS(block_size_t)* SIXTRL_RESTRICT ptr_num_bytes_in_block, 
    NS(block_size_t) const total_num_bytes_in_block,
    NS(block_size_t) const mem_offset, 
    NS(block_size_t) const num_bytes_for_attribute, 
    NS(block_alignment_t) const alignment,
    SIXTRL_GLOBAL_DEC unsigned char* SIXTRL_RESTRICT mem_begin,
    NS(block_size_t) const max_num_of_bytes_in_buffer );

SIXTRL_STATIC SIXTRL_GLOBAL_DEC unsigned char const* 
NS(Block_map_attribute_from_const_memory_aligned)(
    NS(block_size_t)* SIXTRL_RESTRICT ptr_num_bytes_in_block, 
    NS(block_size_t) const total_num_bytes_in_block,
    NS(block_size_t) const mem_offset, 
    NS(block_size_t) const num_bytes_for_attribute, 
    NS(block_alignment_t) const alignment,
    SIXTRL_GLOBAL_DEC unsigned char* SIXTRL_RESTRICT mem_begin,
    NS(block_size_t) const max_num_of_bytes_in_buffer );

/* ------------------------------------------------------------------------- */

typedef enum NS(BlockType)
{
    NS(BLOCK_TYPE_NONE)             = 0x00000000,
    NS(BLOCK_TYPE_EXT_MAP_INFO)     = 0x00000001,
    NS(BLOCK_TYPE_PARTICLE)         = 0x00000010,
    NS(BLOCK_TYPE_DRIFT)            = 0x00000100,
    NS(BLOCK_TYPE_DRIFT_EXACT)      = 0x00000200,
    NS(BLOCK_TYPE_MULTIPOLE)        = 0x00000300,
    NS(BLOCK_TYPE_CAVITY)           = 0x00000400,
    NS(BLOCK_TYPE_ALIGN)            = 0x00000500,
    NS(BLOCK_TYPE_USERDEFINED)      = 0x0000fffe,
    NS(BLOCK_TYPE_INVALID)          = 0x0000ffff
}
NS(BlockType);

SIXTRL_STATIC NS(block_type_num_t) NS(BlockType_to_number)(
    NS(BlockType) const type_id );
    
SIXTRL_STATIC NS(BlockType) NS(BlockType_from_number)(
    NS(block_type_num_t) const type_id_num );

SIXTRL_STATIC int NS(BlockType_is_valid_number)(
    NS(block_type_num_t) const type_id_num );
    
/* ------------------------------------------------------------------------- */

struct __attribute__ (( packed )) NS(BlockInfo)
{
    NS(block_size_t)         mem_offset;
    NS(block_size_t)         num_bytes;
    NS(block_num_elements_t) num_elements ; /* We loop over this in OpenMP -> int! */
    NS(block_type_num_t)     type_id_num; 
    NS(block_store_info_t)   store_info;
};

typedef struct NS(BlockInfo) 
        NS(BlockInfo) __attribute__ ( ( aligned ) ); 
        
/* TODO: Replace the placeholder below with a proper implementation, 
 *       once the design is completed */

typedef void NS(BlockExtHeader); 
typedef void NS(BlockMappingInfo);

SIXTRL_STATIC SIXTRL_GLOBAL_DEC NS(BlockInfo)* NS(BlockInfo_preset)( 
    SIXTRL_GLOBAL_DEC NS(BlockInfo)* SIXTRL_RESTRICT info );

SIXTRL_STATIC NS(block_size_t) 
NS(BlockInfo_get_total_num_of_elements_in_blocks)(
    const SIXTRL_GLOBAL_DEC NS(BlockInfo) *const SIXTRL_RESTRICT infos, 
    NS(block_size_t) const num_of_blocks );

SIXTRL_STATIC NS(block_size_t) NS(BlockInfo_get_total_storage_size)(
    const SIXTRL_GLOBAL_DEC NS(BlockInfo) *const SIXTRL_RESTRICT infos, 
    NS(block_size_t) const num_of_blocks );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC NS(block_size_t) NS(BlockInfo_get_mem_offset)(
    const SIXTRL_GLOBAL_DEC NS(BlockInfo) *const SIXTRL_RESTRICT info );

SIXTRL_STATIC NS(block_size_t) NS(BlockInfo_get_next_mem_offset)(
    const SIXTRL_GLOBAL_DEC NS(BlockInfo) *const SIXTRL_RESTRICT info );

SIXTRL_STATIC NS(block_size_t) NS(BlockInfo_get_num_of_bytes)(
    const SIXTRL_GLOBAL_DEC NS(BlockInfo) *const SIXTRL_RESTRICT info );

SIXTRL_STATIC SIXTRL_GLOBAL_DEC unsigned char* 
    NS(BlockInfo_get_ptr_to_data_begin)(
        const SIXTRL_GLOBAL_DEC NS(BlockInfo) *const SIXTRL_RESTRICT info, 
        SIXTRL_GLOBAL_DEC unsigned char* SIXTRL_RESTRICT memory_begin );

SIXTRL_STATIC SIXTRL_GLOBAL_DEC unsigned char const* 
    NS(BlockInfo_get_const_ptr_to_data_begin)(
        const SIXTRL_GLOBAL_DEC NS(BlockInfo) *const SIXTRL_RESTRICT info, 
        SIXTRL_GLOBAL_DEC unsigned char const* SIXTRL_RESTRICT memory_begin );

SIXTRL_STATIC SIXTRL_GLOBAL_DEC unsigned char* 
    NS(BlockInfo_get_ptr_to_next_data_begin)(
        const SIXTRL_GLOBAL_DEC NS(BlockInfo) *const SIXTRL_RESTRICT info, 
        SIXTRL_GLOBAL_DEC unsigned char* SIXTRL_RESTRICT memory_begin );

SIXTRL_STATIC SIXTRL_GLOBAL_DEC unsigned char const* 
    NS(BlockInfo_get_const_ptr_to_next_data_begin)(
        const SIXTRL_GLOBAL_DEC NS(BlockInfo) *const SIXTRL_RESTRICT info, 
        SIXTRL_GLOBAL_DEC unsigned char const* SIXTRL_RESTRICT memory_begin );
    
SIXTRL_STATIC void NS(BlockInfo_set_mem_offset)(
    SIXTRL_GLOBAL_DEC NS(BlockInfo)* SIXTRL_RESTRICT info, 
    NS(block_size_t) const offset );

SIXTRL_STATIC void NS(BlockInfo_set_num_of_bytes)(
    SIXTRL_GLOBAL_DEC NS(BlockInfo)* SIXTRL_RESTRICT info, 
    NS(block_size_t) const num_bytes );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC NS(BlockType) NS(BlockInfo_get_type_id)(
    const SIXTRL_GLOBAL_DEC NS(BlockInfo) *const SIXTRL_RESTRICT info );

SIXTRL_STATIC NS(block_type_num_t) NS(BlockInfo_get_type_id_num)(
    const SIXTRL_GLOBAL_DEC NS(BlockInfo) *const SIXTRL_RESTRICT info );

SIXTRL_STATIC void NS(BlockInfo_set_type_id)(
    SIXTRL_GLOBAL_DEC NS(BlockInfo)* SIXTRL_RESTRICT info, NS(BlockType) const type_id );

SIXTRL_STATIC void NS(BlockInfo_set_type_id_num)(
    SIXTRL_GLOBAL_DEC NS(BlockInfo)* SIXTRL_RESTRICT info, 
    NS(block_type_num_t) const type_id );

SIXTRL_STATIC int NS(BlockInfo_is_a_particles_block)(
    const SIXTRL_GLOBAL_DEC NS(BlockInfo) *const SIXTRL_RESTRICT info );

SIXTRL_STATIC int NS(BlockInfo_is_a_beam_element_block)(
    const SIXTRL_GLOBAL_DEC NS(BlockInfo) *const SIXTRL_RESTRICT info );

SIXTRL_STATIC int NS(BlockInfo_is_a_mapping_info_block)(
    const SIXTRL_GLOBAL_DEC NS(BlockInfo) *const SIXTRL_RESTRICT info );

SIXTRL_STATIC int NS(BlockInfo_is_a_userdefined_block)(
    const SIXTRL_GLOBAL_DEC NS(BlockInfo) *const SIXTRL_RESTRICT info );


/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC NS(block_num_elements_t) NS(BlockInfo_get_num_elements)(
    const SIXTRL_GLOBAL_DEC NS(BlockInfo) *const SIXTRL_RESTRICT info );

SIXTRL_STATIC void NS(BlockInfo_set_num_elements)(
    SIXTRL_GLOBAL_DEC NS(BlockInfo)* SIXTRL_RESTRICT info, 
    NS(block_num_elements_t) const num_elements );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC int NS(BlockInfo_has_common_alignment)(
    const SIXTRL_GLOBAL_DEC NS(BlockInfo) *const SIXTRL_RESTRICT info );

SIXTRL_STATIC int NS(BlockInfo_has_associated_mapping_info)(
    const SIXTRL_GLOBAL_DEC NS(BlockInfo) *const SIXTRL_RESTRICT info );

SIXTRL_STATIC NS(block_map_info_id_t) 
NS(BlockInfo_get_associated_map_info_id)(
    const SIXTRL_GLOBAL_DEC NS(BlockInfo) *const SIXTRL_RESTRICT info );

SIXTRL_STATIC NS(block_alignment_t) NS(BlockInfo_get_common_alignment)(
    const SIXTRL_GLOBAL_DEC NS(BlockInfo) *const SIXTRL_RESTRICT info );

SIXTRL_STATIC void NS(BlockInfo_set_common_alignment)(
    SIXTRL_GLOBAL_DEC NS(BlockInfo)* SIXTRL_RESTRICT info, 
    NS(block_alignment_t) const alignment );

SIXTRL_STATIC void NS(BlockInfo_set_associated_mapping_header_id)(
    SIXTRL_GLOBAL_DEC NS(BlockInfo)* SIXTRL_RESTRICT info,
    NS(block_map_info_id_t) const map_info_id );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC int NS(BlockInfo_map_to_memory_aligned)(
    SIXTRL_GLOBAL_DEC NS(BlockInfo)* SIXTRL_RESTRICT block_info, 
    SIXTRL_GLOBAL_DEC unsigned char** SIXTRL_RESTRICT attrs_ptr,
    NS(block_size_t)* SIXTRL_RESTRICT num_bytes_for_attrs,
    NS(block_size_t) const num_of_attributes,
    NS(block_num_elements_t) const num_elements, 
    NS(BlockType) const type_id,
    SIXTRL_GLOBAL_DEC unsigned char* SIXTRL_RESTRICT mem_begin, 
    NS(block_size_t) const max_num_of_bytes_in_buffer );    

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC int NS(BlockInfo_remap_from_memory_aligned)(
    const SIXTRL_GLOBAL_DEC NS(BlockInfo) *const SIXTRL_RESTRICT block_info,
    SIXTRL_GLOBAL_DEC unsigned char** SIXTRL_RESTRICT attrs_ptr, 
    NS(block_size_t)* SIXTRL_RESTRICT num_bytes_for_attrs, 
    NS(block_size_t) const num_of_attributes, 
    SIXTRL_GLOBAL_DEC unsigned char* SIXTRL_RESTRICT mem_begin,
    NS(block_size_t) const max_num_of_bytes_in_buffer );

/* ------------------------------------------------------------------------- */

typedef struct NS(BlocksContainer)
{
    SIXTRL_GLOBAL_DEC NS(BlockInfo)* info_begin;
    SIXTRL_GLOBAL_DEC unsigned char* data_begin;
        
    void*                            ptr_info_store;
    void*                            ptr_data_store;
                                     
    NS(block_size_t)                 num_blocks;
    NS(block_size_t)                 blocks_capacity;
                                     
    NS(block_size_t)                 data_raw_size;
    NS(block_size_t)                 data_raw_capacity;
                                     
    NS(block_alignment_t)            info_begin_alignment;
    NS(block_alignment_t)            info_alignment;
                                     
    NS(block_alignment_t)            data_begin_alignment;
    NS(block_alignment_t)            data_alignment;
}
NS(BlocksContainer);

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC NS(BlocksContainer)* NS(BlocksContainer_preset)( 
    NS(BlocksContainer)* SIXTRL_RESTRICT container );

SIXTRL_STATIC int NS(BlocksContainer_set_info_begin_alignment)(
    NS(BlocksContainer)* SIXTRL_RESTRICT container, 
    NS(block_alignment_t) const begin_alignment );

SIXTRL_STATIC int NS(BlocksContainer_set_data_begin_alignment)(
    NS(BlocksContainer)* SIXTRL_RESTRICT container, 
    NS(block_alignment_t) const begin_alignment );

SIXTRL_STATIC int NS(BlocksContainer_set_data_alignment)(
    NS(BlocksContainer)* SIXTRL_RESTRICT container, 
    NS(block_alignment_t) const alignment );

SIXTRL_STATIC int NS(BlocksContainer_set_info_alignment )(
    NS(BlocksContainer)* SIXTRL_RESTRICT container, 
    NS(block_alignment_t) const alignment );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC SIXTRL_GLOBAL_DEC unsigned char const* 
NS(BlocksContainer_get_const_ptr_data_begin)(
    const NS(BlocksContainer) *const SIXTRL_RESTRICT container );

SIXTRL_STATIC SIXTRL_GLOBAL_DEC unsigned char* 
NS(BlocksContainer_get_ptr_data_begin)(
    NS(BlocksContainer)* SIXTRL_RESTRICT container );

SIXTRL_STATIC SIXTRL_GLOBAL_DEC NS(BlockInfo) const* 
NS(BlocksContainer_get_const_block_infos_begin)(
    const NS(BlocksContainer) *const SIXTRL_RESTRICT container );

SIXTRL_STATIC SIXTRL_GLOBAL_DEC NS(BlockInfo) const* 
NS(BlocksContainer_get_const_block_infos_end)(
    const NS(BlocksContainer) *const SIXTRL_RESTRICT container );

SIXTRL_STATIC SIXTRL_GLOBAL_DEC NS(BlockInfo)* 
NS(BlocksContainer_get_block_infos_begin)(
    NS(BlocksContainer)* SIXTRL_RESTRICT container );

SIXTRL_STATIC SIXTRL_GLOBAL_DEC NS(BlockInfo)* 
NS(BlocksContainer_get_block_infos_end)(
    NS(BlocksContainer)* SIXTRL_RESTRICT container );

SIXTRL_STATIC NS(BlockInfo) 
NS(BlocksContainer_get_block_info_by_index)(
    const NS(BlocksContainer) *const SIXTRL_RESTRICT container, 
    NS(block_size_t) const block_index );

SIXTRL_STATIC SIXTRL_GLOBAL_DEC NS(BlockInfo) const* 
NS(BlocksContainer_get_const_ptr_to_block_info_by_index)(
    const NS(BlocksContainer) *const SIXTRL_RESTRICT container, 
    NS(block_size_t) const block_index );

SIXTRL_STATIC SIXTRL_GLOBAL_DEC NS(BlockInfo)* 
NS(BlocksContainer_get_ptr_to_block_info_by_index)(
    NS(BlocksContainer)* SIXTRL_RESTRICT container, 
    NS(block_size_t) const block_index );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC NS(block_alignment_t) NS(BlocksContainer_get_info_alignment)(
    const NS(BlocksContainer) *const SIXTRL_RESTRICT container );

SIXTRL_STATIC NS(block_alignment_t) NS(BlocksContainer_get_data_alignment)(
    const NS(BlocksContainer) *const SIXTRL_RESTRICT container );

SIXTRL_STATIC NS(block_alignment_t) 
NS(BlocksContainer_get_info_begin_alignment)(
    const NS(BlocksContainer) *const SIXTRL_RESTRICT container );

SIXTRL_STATIC NS(block_alignment_t) 
NS(BlocksContainer_get_data_begin_alignment)(
    const NS(BlocksContainer) *const SIXTRL_RESTRICT container );

SIXTRL_STATIC NS(block_size_t) NS(BlocksContainer_get_data_capacity)(
    const NS(BlocksContainer) *const SIXTRL_RESTRICT container );

SIXTRL_STATIC NS(block_size_t) NS(BlocksContainer_get_data_size)( 
    const NS(BlocksContainer) *const SIXTRL_RESTRICT container );

SIXTRL_STATIC NS(block_size_t) NS(BlocksContainer_get_block_capacity)(
    const NS(BlocksContainer) *const SIXTRL_RESTRICT container );

SIXTRL_STATIC NS(block_size_t) NS(BlocksContainer_get_num_of_blocks)(
    const NS(BlocksContainer) *const SIXTRL_RESTRICT container );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC int NS(BlocksContainer_has_info_store)(
    const NS(BlocksContainer) *const SIXTRL_RESTRICT container );

SIXTRL_STATIC int NS(BlocksContainer_has_data_store)(
    const NS(BlocksContainer) *const SIXTRL_RESTRICT container );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC int NS(BlocksContainer_assemble)(    
    NS(BlocksContainer)* SIXTRL_RESTRICT container,
    SIXTRL_GLOBAL_DEC NS(BlockInfo)* SIXTRL_RESTRICT block_infos_begin,
    NS(block_size_t) const num_of_blocks,
    SIXTRL_GLOBAL_DEC unsigned char* SIXTRL_RESTRICT data_mem_begin,
    NS(block_size_t) const data_num_of_bytes );

/* ========================================================================= */
/* ======             Implementation of inline functions            ======== */
/* ========================================================================= */

SIXTRL_INLINE NS(block_type_num_t) NS(BlockType_to_number)(
    NS(BlockType) const type_id )
{
    NS(block_type_num_t) type_id_num;
    
    switch( type_id )
    {
        case NS(BLOCK_TYPE_NONE):
        case NS(BLOCK_TYPE_EXT_MAP_INFO):
        case NS(BLOCK_TYPE_PARTICLE):
        case NS(BLOCK_TYPE_DRIFT):
        case NS(BLOCK_TYPE_DRIFT_EXACT):
        case NS(BLOCK_TYPE_MULTIPOLE):
        case NS(BLOCK_TYPE_CAVITY):
        case NS(BLOCK_TYPE_ALIGN):
        case NS(BLOCK_TYPE_USERDEFINED):
        {
            type_id_num = ( NS(block_type_num_t) )type_id;
            break;
        }
        
        default:
        {
            type_id_num = 
                ( NS(block_type_num_t) )NS(BLOCK_TYPE_INVALID);
        }
    };
    
    return type_id_num;
}
    
SIXTRL_INLINE NS(BlockType) NS(BlockType_from_number)(
    NS(block_type_num_t) const type_id_num )
{
    NS(BlockType) type_id;

    switch( type_id_num )
    {
        case ( NS(block_type_num_t) )NS(BLOCK_TYPE_NONE):
        case ( NS(block_type_num_t) )NS(BLOCK_TYPE_EXT_MAP_INFO):
        case ( NS(block_type_num_t) )NS(BLOCK_TYPE_PARTICLE):
        case ( NS(block_type_num_t) )NS(BLOCK_TYPE_DRIFT):
        case ( NS(block_type_num_t) )NS(BLOCK_TYPE_DRIFT_EXACT):
        case ( NS(block_type_num_t) )NS(BLOCK_TYPE_MULTIPOLE):
        case ( NS(block_type_num_t) )NS(BLOCK_TYPE_CAVITY):
        case ( NS(block_type_num_t) )NS(BLOCK_TYPE_ALIGN):
        case ( NS(block_type_num_t) )NS(BLOCK_TYPE_USERDEFINED):
        {
            type_id = ( NS(BlockType) )type_id_num;
            break;
        }
        
        default:
        {
            type_id = NS(BLOCK_TYPE_INVALID);            
        }        
    };
        
    return type_id;
}

SIXTRL_INLINE int NS(BlockType_is_valid_number)(
    NS(block_type_num_t) const type_id_num )
{
    return ( NS(BlockType_from_number)( type_id_num ) !=
             NS(BLOCK_TYPE_INVALID) );
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE SIXTRL_GLOBAL_DEC NS(BlockInfo)* NS(BlockInfo_preset)( 
    SIXTRL_GLOBAL_DEC NS(BlockInfo)* SIXTRL_RESTRICT info )
{
    if( info != 0 )
    {
        info->mem_offset   = ( NS(block_size_t) )0u;
        info->num_bytes    = ( NS(block_size_t) )0u;
        info->num_elements = ( NS(block_num_elements_t) )0u;
        info->type_id_num  = NS(BlockType_to_number)( NS(BLOCK_TYPE_INVALID) );
        info->store_info   = ( NS(block_alignment_t) )0u;
    }
    
    return info;
}

SIXTRL_INLINE NS(block_size_t) NS(BlockInfo_get_total_storage_size)(
    const SIXTRL_GLOBAL_DEC NS(BlockInfo) *const SIXTRL_RESTRICT infos, 
    NS(block_size_t) const num_of_blocks )
{
    NS(block_size_t) const last = ( num_of_blocks > 0u )
        ? ( num_of_blocks - 1u ) : 0u;
    
    NS(block_size_t) total_num_bytes = infos[ last ].num_bytes; 
        
    SIXTRL_ASSERT( 
        ( infos != 0 ) && 
        ( infos[ 0 ].mem_offset <= infos[ last ].mem_offset ) &&
        ( infos[ 0 ].num_bytes  <= ( ( infos[ last ].mem_offset + 
            infos[ last ].num_bytes ) - infos[ 0 ].mem_offset ) ) );
    
    total_num_bytes += ( infos[ last ].mem_offset - infos[ 0 ].mem_offset );
    
    #if !defined( NDEBUG )
    if( num_of_blocks > 1u )
    {
        NS(block_size_t) calculated_size = 0u;
        NS(block_size_t) prev_offset     = infos[ 0 ].mem_offset;
        NS(block_size_t) prev_num_bytes  = infos[ 0 ].num_bytes;
        NS(block_size_t) ii = 1u;
        
        for( ; ii < num_of_blocks ; ++ii )
        {
            NS(block_size_t) const num_bytes = infos[ ii ].num_bytes;
            NS(block_size_t) const offset    = infos[ ii ].mem_offset;
            NS(block_size_t) const length    = offset - prev_offset;
            
            SIXTRL_ASSERT( length == prev_num_bytes );
            
            prev_num_bytes   = num_bytes;
            prev_offset      = offset;
            calculated_size += length;            
        }
        
        calculated_size += infos[ last ].num_bytes;
        SIXTRL_ASSERT( total_num_bytes == calculated_size );        
    }
    #endif /* !defined( NDEBUG ) */
    
    return total_num_bytes;
}

SIXTRL_INLINE NS(block_size_t) 
NS(BlockInfo_get_total_num_of_elements_in_blocks)(
    const SIXTRL_GLOBAL_DEC NS(BlockInfo) *const SIXTRL_RESTRICT infos, 
    NS(block_size_t) const num_of_blocks )
{
    NS(block_size_t) total_num_elements = 0u;
    NS(block_size_t) ii = 0u;
    
    for( ; ii < num_of_blocks ; ++ii )
    {
        if( ( infos[ ii ].type_id_num == 
                NS(BLOCK_TYPE_EXT_MAP_INFO) ) ||
            ( infos[ ii ].num_elements < 0 ) )
        {
            continue;
        }
        
        SIXTRL_ASSERT( infos[ ii ].num_elements > 0 );
        total_num_elements += infos[ ii ].num_elements;
    }
    
    return total_num_elements;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE NS(block_size_t) NS(BlockInfo_get_mem_offset)(
    const SIXTRL_GLOBAL_DEC NS(BlockInfo) *const SIXTRL_RESTRICT info )
{
    SIXTRL_ASSERT( info != 0 );
    return info->mem_offset;
}

SIXTRL_INLINE NS(block_size_t) NS(BlockInfo_get_next_mem_offset)(
    const SIXTRL_GLOBAL_DEC NS(BlockInfo) *const SIXTRL_RESTRICT info )
{
    SIXTRL_ASSERT( info != 0 );
    return ( info->mem_offset + info->num_bytes );
}

SIXTRL_INLINE NS(block_size_t) NS(BlockInfo_get_num_of_bytes)(
    const SIXTRL_GLOBAL_DEC NS(BlockInfo) *const SIXTRL_RESTRICT info )
{
    SIXTRL_ASSERT( info != 0 );
    return info->num_bytes;
}

SIXTRL_INLINE SIXTRL_GLOBAL_DEC unsigned char* 
    NS(BlockInfo_get_ptr_to_data_begin)(
        const SIXTRL_GLOBAL_DEC NS(BlockInfo) *const SIXTRL_RESTRICT info, 
        SIXTRL_GLOBAL_DEC unsigned char* SIXTRL_RESTRICT memory_begin )
{
    typedef SIXTRL_GLOBAL_DEC unsigned char*  g_ptr_uchar_t;
    
    return ( g_ptr_uchar_t )NS(BlockInfo_get_const_ptr_to_data_begin)( 
        info, memory_begin );
}

SIXTRL_INLINE SIXTRL_GLOBAL_DEC unsigned char const* 
    NS(BlockInfo_get_const_ptr_to_data_begin)(
        const SIXTRL_GLOBAL_DEC NS(BlockInfo) *const SIXTRL_RESTRICT info, 
        SIXTRL_GLOBAL_DEC unsigned char const* SIXTRL_RESTRICT memory_begin )
{
    SIXTRL_ASSERT( ( info != 0 ) && ( memory_begin != 0 ) );
    return ( memory_begin + info->mem_offset );
}

SIXTRL_INLINE SIXTRL_GLOBAL_DEC unsigned char* 
    NS(BlockInfo_get_ptr_to_next_data_begin)(
        const SIXTRL_GLOBAL_DEC NS(BlockInfo) *const SIXTRL_RESTRICT info, 
        SIXTRL_GLOBAL_DEC unsigned char* SIXTRL_RESTRICT mem_begin )
{
    typedef SIXTRL_GLOBAL_DEC unsigned char*  g_ptr_uchar_t;
    
    return ( g_ptr_uchar_t 
    )NS(BlockInfo_get_const_ptr_to_next_data_begin)( info, mem_begin );
}

SIXTRL_INLINE SIXTRL_GLOBAL_DEC unsigned char const* 
    NS(BlockInfo_get_const_ptr_to_next_data_begin)(
        const SIXTRL_GLOBAL_DEC NS(BlockInfo) *const SIXTRL_RESTRICT info, 
        SIXTRL_GLOBAL_DEC unsigned char const* SIXTRL_RESTRICT memory_begin )
{
    SIXTRL_ASSERT( ( info != 0 ) && ( memory_begin != 0 ) );
    return ( memory_begin + info->mem_offset + info->num_bytes );
}
    
SIXTRL_INLINE void NS(BlockInfo_set_mem_offset)(
    SIXTRL_GLOBAL_DEC NS(BlockInfo)* SIXTRL_RESTRICT info, 
    NS(block_size_t) const offset )
{
    SIXTRL_ASSERT( info != 0 );
    info->mem_offset = offset;
    return;
}

SIXTRL_INLINE void NS(BlockInfo_set_num_of_bytes)(
    SIXTRL_GLOBAL_DEC NS(BlockInfo)* SIXTRL_RESTRICT info, 
    NS(block_size_t) const num_bytes )
{
    SIXTRL_ASSERT( info != 0 );
    info->num_bytes = num_bytes;
    return;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE NS(BlockType)  NS(BlockInfo_get_type_id)(
    const SIXTRL_GLOBAL_DEC NS(BlockInfo) *const SIXTRL_RESTRICT info )
{
    SIXTRL_ASSERT( info != 0 );
    return NS(BlockType_from_number)( info->type_id_num );
}

SIXTRL_INLINE NS(block_type_num_t) NS(BlockInfo_get_type_id_num)(
    const SIXTRL_GLOBAL_DEC NS(BlockInfo) *const SIXTRL_RESTRICT info )
{
    SIXTRL_ASSERT( info != 0 );
    return info->type_id_num;
}

SIXTRL_INLINE void NS(BlockInfo_set_type_id)(
    SIXTRL_GLOBAL_DEC NS(BlockInfo)* SIXTRL_RESTRICT info, 
    NS(BlockType) const type_id )
{
    SIXTRL_ASSERT( info != 0 );
    info->type_id_num = NS(BlockType_to_number)( type_id );
}

SIXTRL_INLINE void NS(BlockInfo_set_type_id_num)(
    SIXTRL_GLOBAL_DEC NS(BlockInfo)* SIXTRL_RESTRICT info, 
    NS(block_type_num_t) const type_id_num )
{
    SIXTRL_ASSERT( info );
    info->type_id_num = type_id_num;
}

SIXTRL_INLINE int NS(BlockInfo_is_a_particles_block)(
    const SIXTRL_GLOBAL_DEC NS(BlockInfo) *const SIXTRL_RESTRICT info )
{
    NS(BlockType) const type_id = NS(BlockInfo_get_type_id)( info );
    return ( type_id == NS(BLOCK_TYPE_PARTICLE) );
}

SIXTRL_INLINE int NS(BlockInfo_is_a_beam_element_block)(
    const SIXTRL_GLOBAL_DEC NS(BlockInfo) *const SIXTRL_RESTRICT info )
{
    NS(BlockType) const type_id = NS(BlockInfo_get_type_id)( info );
    return ( ( type_id == NS(BLOCK_TYPE_DRIFT) ) ||
             ( type_id == NS(BLOCK_TYPE_DRIFT_EXACT) ) ||
             ( type_id == NS(BLOCK_TYPE_MULTIPOLE) ) ||
             ( type_id == NS(BLOCK_TYPE_CAVITY) ) ||
             ( type_id == NS(BLOCK_TYPE_ALIGN) ) );
}

SIXTRL_INLINE int NS(BlockInfo_is_a_mapping_info_block)(
    const SIXTRL_GLOBAL_DEC NS(BlockInfo) *const SIXTRL_RESTRICT info )
{
    NS(BlockType) const type_id = NS(BlockInfo_get_type_id)( info );
    return ( type_id == NS(BLOCK_TYPE_EXT_MAP_INFO) );
}

SIXTRL_INLINE int NS(BlockInfo_is_a_userdefined_block)(
    const SIXTRL_GLOBAL_DEC NS(BlockInfo) *const SIXTRL_RESTRICT info )
{
    NS(BlockType) const type_id = NS(BlockInfo_get_type_id)( info );
    return ( type_id == NS(BLOCK_TYPE_USERDEFINED) );
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE NS(block_num_elements_t) NS(BlockInfo_get_num_elements)(
    const SIXTRL_GLOBAL_DEC NS(BlockInfo) *const SIXTRL_RESTRICT info )
{
    SIXTRL_ASSERT( info != 0 );
    return info->num_elements;
}

SIXTRL_INLINE void NS(BlockInfo_set_num_elements)(
    SIXTRL_GLOBAL_DEC NS(BlockInfo)* SIXTRL_RESTRICT info, 
    NS(block_num_elements_t) const neleme )
{
    SIXTRL_ASSERT( info != 0 );
    info->num_elements = neleme;
    return;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE int NS(BlockInfo_has_common_alignment)(
    const SIXTRL_GLOBAL_DEC NS(BlockInfo) *const SIXTRL_RESTRICT info )
{
    return ( 
        ( info != 0 ) && ( info->store_info > 0 ) && 
        ( info->type_id_num != NS(BlockType_to_number)( 
            NS(BLOCK_TYPE_EXT_MAP_INFO) ) ) );
}

SIXTRL_INLINE int NS(BlockInfo_has_associated_mapping_info)(
    const SIXTRL_GLOBAL_DEC NS(BlockInfo) *const SIXTRL_RESTRICT info )
{
    return ( ( info != 0 ) && ( info->store_info < 0 ) &&
             ( info->type_id_num != NS(BlockType_to_number)(
                 NS(BLOCK_TYPE_EXT_MAP_INFO) ) ) );
}

SIXTRL_INLINE NS(block_map_info_id_t) 
NS(BlockInfo_get_associated_map_info_id)(
    const SIXTRL_GLOBAL_DEC NS(BlockInfo) *const SIXTRL_RESTRICT info )
{
    return ( info != 0 ) ? ( -( info->store_info ) ) : 0;
}

SIXTRL_INLINE NS(block_alignment_t) NS(BlockInfo_get_common_alignment)(
    const SIXTRL_GLOBAL_DEC NS(BlockInfo) *const SIXTRL_RESTRICT info )
{
    return ( info != 0 ) ? ( info->store_info ) : 0;
}

SIXTRL_INLINE void NS(BlockInfo_set_common_alignment)(
    SIXTRL_GLOBAL_DEC NS(BlockInfo)* SIXTRL_RESTRICT info, 
    NS(block_alignment_t) const alignment )
{
    if( ( info != 0 ) && ( alignment >= 0 ) )
    {
        info->store_info = alignment;
    }
    
    return;
}

SIXTRL_INLINE void NS(BlockInfo_set_associated_mapping_header_id)(
    SIXTRL_GLOBAL_DEC NS(BlockInfo)* SIXTRL_RESTRICT info,
    NS(block_map_info_id_t) const map_info_id )
{
    if( ( info != 0 ) && ( map_info_id >= 0 ) )
    {
        info->store_info = -( map_info_id );
    }
    
    return;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE SIXTRL_GLOBAL_DEC unsigned char*
NS(Block_map_begin_to_memory_aligned)(
    NS(block_size_t)* SIXTRL_RESTRICT ptr_mem_offset,
    NS(block_alignment_t) const alignment, 
    SIXTRL_GLOBAL_DEC unsigned char* SIXTRL_RESTRICT mem_begin,
    NS(block_size_t) const max_num_of_bytes_in_buffer )
{
    typedef SIXTRL_GLOBAL_DEC unsigned char* g_ptr_uchar_t;
    typedef NS(block_alignment_t) align_t;
    
    SIXTRL_STATIC NS(block_size_t) const ZERO = ( NS(block_size_t) )0u;    
    
    g_ptr_uchar_t ptr = mem_begin;
    uintptr_t addr_mod;
        
    SIXTRL_ASSERT( 
        ( mem_begin != 0 ) && ( ptr_mem_offset != 0 ) &&
        ( *ptr_mem_offset <= max_num_of_bytes_in_buffer ) &&
        ( alignment > ( align_t )0u ) && 
        ( ( alignment % ( align_t )2u )  == ( align_t )0u ) );
    
    ptr = ptr + *ptr_mem_offset;
    addr_mod   = ( ( uintptr_t )ptr ) % alignment;
    
    if( addr_mod != ZERO )
    {
        align_t const offset = alignment - addr_mod;
        *ptr_mem_offset += offset;
        ptr = ptr + offset;
    }
    
    return ( *ptr_mem_offset <= max_num_of_bytes_in_buffer ) ? ptr : 0;
}

SIXTRL_INLINE SIXTRL_GLOBAL_DEC unsigned char* 
NS(Block_map_attribute_to_memory_aligned)(
    NS(block_size_t)* SIXTRL_RESTRICT ptr_num_bytes_in_block,
    NS(block_size_t) const mem_offset,
    NS(block_size_t) const num_bytes_for_attribute, 
    NS(block_alignment_t) const alignment,
    SIXTRL_GLOBAL_DEC unsigned char* SIXTRL_RESTRICT mem_begin,
    NS(block_size_t) const max_num_of_bytes_in_buffer )
{
    typedef SIXTRL_GLOBAL_DEC unsigned char* g_ptr_uchar_t;
    typedef NS(block_alignment_t) align_t;
    
    SIXTRL_STATIC NS(block_size_t) const ZERO = ( NS(block_size_t) )0u;
    
    g_ptr_uchar_t next_ptr = 0;
    g_ptr_uchar_t ptr      = mem_begin + mem_offset;  
        
    uintptr_t addr_mod;
        
    SIXTRL_ASSERT( ( mem_begin != 0 ) && ( num_bytes_for_attribute > ZERO ) &&
        ( ptr_num_bytes_in_block != 0 ) && ( alignment > ( align_t )ZERO ) &&
        ( ( alignment % ( align_t )2u )  == ( align_t )ZERO ) );
    
    ptr = ptr + *ptr_num_bytes_in_block;
    SIXTRL_ASSERT( ( ( ( uintptr_t )ptr ) % alignment ) == ZERO );
    
    next_ptr = ptr + num_bytes_for_attribute;
    addr_mod = ( ( uintptr_t )next_ptr ) % alignment;

    *ptr_num_bytes_in_block += ( addr_mod == ( align_t )ZERO )
        ? num_bytes_for_attribute 
        : num_bytes_for_attribute + ( alignment - addr_mod );
    
    return ( ( mem_offset + *ptr_num_bytes_in_block ) <= 
        max_num_of_bytes_in_buffer ) ? ptr : 0;
}

SIXTRL_INLINE SIXTRL_GLOBAL_DEC unsigned char*
NS(Block_map_attribute_from_memory_aligned)(
    NS(block_size_t)* SIXTRL_RESTRICT ptr_num_bytes_in_block, 
    NS(block_size_t) const total_num_bytes_in_block,
    NS(block_size_t) const mem_offset, 
    NS(block_size_t) const num_bytes_for_attribute, 
    NS(block_alignment_t) const alignment,
    SIXTRL_GLOBAL_DEC unsigned char* SIXTRL_RESTRICT mem_begin,
    NS(block_size_t) const max_num_of_bytes_in_buffer )
{
    return ( SIXTRL_GLOBAL_DEC unsigned char* 
        )NS(Block_map_attribute_from_const_memory_aligned)(
            ptr_num_bytes_in_block, total_num_bytes_in_block, mem_offset, 
            num_bytes_for_attribute, alignment, mem_begin, 
                max_num_of_bytes_in_buffer );
}

SIXTRL_INLINE SIXTRL_GLOBAL_DEC unsigned char const* 
NS(Block_map_attribute_from_const_memory_aligned)(
    NS(block_size_t)* SIXTRL_RESTRICT ptr_num_bytes_in_block,
    NS(block_size_t) const total_num_bytes_in_block,
    NS(block_size_t) const mem_offset, 
    NS(block_size_t) const num_bytes_for_attribute, 
    NS(block_alignment_t) const alignment,
    SIXTRL_GLOBAL_DEC unsigned char* SIXTRL_RESTRICT mem_begin,
    NS(block_size_t) const max_num_of_bytes_in_buffer )
{
    typedef SIXTRL_GLOBAL_DEC unsigned char const* g_ptr_uchar_t;
    typedef NS(block_alignment_t) align_t;
    
    SIXTRL_STATIC NS(block_size_t) const ZERO = ( NS(block_size_t) )0u;    
    
    g_ptr_uchar_t next_ptr = 0;
    g_ptr_uchar_t ptr      = mem_begin + mem_offset;  
        
    uintptr_t addr_mod;
        
    SIXTRL_ASSERT( ( mem_begin != 0 ) && ( num_bytes_for_attribute > ZERO ) &&
        ( ptr_num_bytes_in_block != 0 ) && 
        ( total_num_bytes_in_block >= *ptr_num_bytes_in_block ) &&
        ( alignment > ( align_t )ZERO ) &&
        ( ( alignment % ( align_t )2u )  == ( align_t )ZERO ) );
    
    ptr = ptr + *ptr_num_bytes_in_block;
    SIXTRL_ASSERT( ( ( ( uintptr_t )ptr ) % alignment ) == ZERO );
    
    next_ptr = ptr + num_bytes_for_attribute;
    addr_mod = ( ( uintptr_t )next_ptr ) % alignment;

    *ptr_num_bytes_in_block += ( addr_mod == ( align_t )ZERO )
        ? num_bytes_for_attribute 
        : num_bytes_for_attribute + ( alignment - addr_mod );
    
    return ( ( *ptr_num_bytes_in_block <= total_num_bytes_in_block ) &&
             ( ( mem_offset + *ptr_num_bytes_in_block ) <= 
                 max_num_of_bytes_in_buffer ) ) ? ptr : 0;
}

SIXTRL_INLINE int NS(BlockInfo_map_to_memory_aligned)(
    SIXTRL_GLOBAL_DEC NS(BlockInfo)* SIXTRL_RESTRICT block_info, 
    SIXTRL_GLOBAL_DEC unsigned char** SIXTRL_RESTRICT attrs_ptr,
    NS(block_size_t)* SIXTRL_RESTRICT num_bytes_for_attrs,
    NS(block_size_t) const num_of_attributes,
    NS(block_num_elements_t) const num_elements, NS(BlockType) const type_id,
    SIXTRL_GLOBAL_DEC unsigned char* SIXTRL_RESTRICT mem_begin, 
    NS(block_size_t) const max_num_of_bytes_in_buffer )
{
    int success = -1;
    SIXTRL_STATIC NS(block_size_t) const ZERO = ( NS(block_size_t) )0u;
    
    NS(block_size_t) mem_offset = 
        NS(BlockInfo_get_mem_offset)( block_info );
    
    NS(block_alignment_t) const alignment = 
        NS(BlockInfo_get_common_alignment)( block_info );
        
    SIXTRL_ASSERT( 
        ( mem_begin != 0 )  && ( block_info != 0 ) && ( attrs_ptr != 0 ) &&
        ( num_bytes_for_attrs != 0 ) && ( num_of_attributes > ZERO ) &&
        ( type_id != NS(BLOCK_TYPE_INVALID) ) &&
        ( type_id != NS(BLOCK_TYPE_EXT_MAP_INFO) ) &&
        ( num_elements > ( NS(block_num_elements_t) )0u ) &&
        ( NS(BlockInfo_has_common_alignment)( block_info ) ) &&
        ( mem_offset <= max_num_of_bytes_in_buffer ) );
        
    if( NS(Block_map_begin_to_memory_aligned)( &mem_offset, alignment, 
            mem_begin, max_num_of_bytes_in_buffer ) != 0 )
    {
        NS(block_size_t) prev_num_bytes_in_blk;
        NS(block_size_t) num_of_bytes_in_blk   = ZERO;
        NS(block_size_t) ii = ZERO;
        
        #if !defined( _GPUCODE ) && !defined( NDEBUG )
        NS(block_size_t) sum_bytes_for_attributes = ZERO;
        #endif /* !defined( _GPUCODE ) && !defined( NDEBUG ) */
        
        success = 0;
        
        for( ; ii < num_of_attributes ; ++ii )
        {
            NS(block_size_t) const bytes_for_attr = num_bytes_for_attrs[ ii ];            
            prev_num_bytes_in_blk = num_of_bytes_in_blk;
            
            #if !defined( _GPUCODE ) && !defined( NDEBUG )
            sum_bytes_for_attributes += bytes_for_attr;
            #endif /* !defined( _GPUCODE ) && !defined( NDEBUG ) */
            
            attrs_ptr[ ii ] = NS(Block_map_attribute_to_memory_aligned)(
                &num_of_bytes_in_blk, mem_offset, bytes_for_attr, 
                alignment, mem_begin, max_num_of_bytes_in_buffer );
            
            SIXTRL_ASSERT( 
                ( ( attrs_ptr[ ii ] != 0 ) && 
                  ( num_of_bytes_in_blk >= prev_num_bytes_in_blk ) ) ||
                ( attrs_ptr[ ii ] == 0 ) );
            
            if( attrs_ptr[ ii ] != 0 )
            {
                num_bytes_for_attrs[ ii ] = 
                    num_of_bytes_in_blk - prev_num_bytes_in_blk;
            }
            else
            {
                success = -1;
                break;
            }
        }
        
        #if !defined( _GPUCODE ) && !defined( NDEBUG )
        SIXTRL_ASSERT( sum_bytes_for_attributes <= num_of_bytes_in_blk );
        #endif /* !defined( _GPUCODE ) && !defined( NDEBUG ) */
        
        if( ( success == 0 ) &&
            ( ( mem_offset + num_of_bytes_in_blk ) <= 
                max_num_of_bytes_in_buffer ) )
        {
            NS(BlockInfo_set_mem_offset)(   block_info, mem_offset );
            NS(BlockInfo_set_num_of_bytes)( block_info, num_of_bytes_in_blk );
            NS(BlockInfo_set_num_elements)( block_info, num_elements );
            NS(BlockInfo_set_type_id)(      block_info, type_id );
        }        
    }
    
    return success;
}
                                                                          
SIXTRL_INLINE int NS(BlockInfo_remap_from_memory_aligned)(
    const SIXTRL_GLOBAL_DEC NS(BlockInfo) *const SIXTRL_RESTRICT block_info,
    SIXTRL_GLOBAL_DEC unsigned char** SIXTRL_RESTRICT attrs_ptr, 
    NS(block_size_t)* SIXTRL_RESTRICT num_bytes_for_attrs, 
    NS(block_size_t) const num_of_attributes, 
    SIXTRL_GLOBAL_DEC unsigned char* SIXTRL_RESTRICT mem_begin,
    NS(block_size_t) const max_num_of_bytes_in_buffer )
{
    int success = -1;
    SIXTRL_STATIC  NS(block_size_t) const ZERO = ( NS(block_size_t) )0u;
    
    NS(block_size_t) prev_num_bytes_in_blk;
    NS(block_size_t) num_of_bytes_in_blk = ZERO;
        
    NS(block_size_t) const mem_offset = 
        NS(BlockInfo_get_mem_offset)( block_info );
    
    NS(block_size_t) const total_num_bytes_in_block =
        NS(BlockInfo_get_num_of_bytes)( block_info );
        
    NS(block_alignment_t) const alignment = 
        NS(BlockInfo_get_common_alignment)( block_info );
        
    NS(block_size_t) ii = ZERO;
        
    #if !defined( _GPUCODE ) && !defined( NDEBUG )
    NS(block_size_t) sum_bytes_for_attributes = ZERO;
    #endif /* !defined( _GPUCODE ) && !defined( NDEBUG ) */
    
    SIXTRL_ASSERT( 
        ( mem_begin != 0 )  && ( block_info != 0 ) && ( attrs_ptr != 0 ) &&
        ( num_bytes_for_attrs != 0 ) && ( num_of_attributes > ZERO ) && 
        ( total_num_bytes_in_block > ZERO ) && 
        ( NS(BlockInfo_get_type_id)( block_info ) != 
            NS(BLOCK_TYPE_INVALID) ) &&
        ( NS(BlockInfo_get_type_id)( block_info ) != 
            NS(BLOCK_TYPE_EXT_MAP_INFO) ) &&
        ( NS(BlockInfo_get_num_elements)( block_info ) > 
            ( NS(block_num_elements_t) )0u ) &&
        ( NS(BlockInfo_has_common_alignment)( block_info ) ) );
    
    success = 0;
    
    for( ; ii < num_of_attributes ; ++ii )
    {
        prev_num_bytes_in_blk = num_of_bytes_in_blk;
        
        #if !defined( _GPUCODE ) && !defined( NDEBUG )
        sum_bytes_for_attributes += num_bytes_for_attrs[ ii ];
        #endif /* !defined( _GPUCODE ) && !defined( NDEBUG ) */
        
        attrs_ptr[ ii ] = NS(Block_map_attribute_from_memory_aligned)(
            &num_of_bytes_in_blk, total_num_bytes_in_block, mem_offset, 
            num_bytes_for_attrs[ ii ], alignment, mem_begin, 
            max_num_of_bytes_in_buffer );
        
        SIXTRL_ASSERT( 
            ( ( attrs_ptr[ ii ] != 0 ) && 
                ( num_of_bytes_in_blk >= prev_num_bytes_in_blk ) ) ||
            ( attrs_ptr[ ii ] == 0 ) );
        
        if( attrs_ptr[ ii ] != 0 )
        {
            SIXTRL_ASSERT( ( ( ( uintptr_t )attrs_ptr[ ii ] ) % alignment ) ==
                ( uintptr_t )0u );
            
            num_bytes_for_attrs[ ii ] = 
                num_of_bytes_in_blk - prev_num_bytes_in_blk;
        }
        else
        {
            success = -1;
            break;
        }
    }
        
    #if !defined( _GPUCODE ) && !defined( NDEBUG )
    SIXTRL_ASSERT( sum_bytes_for_attributes <= num_of_bytes_in_blk );
    #endif /* !defined( _GPUCODE ) && !defined( NDEBUG ) */
    
    SIXTRL_ASSERT( 
        ( ( success == 0 ) && 
          ( total_num_bytes_in_block == num_of_bytes_in_blk ) ) ||
        ( success != 0 ) );
    
    return success;
}
    
/* ========================================================================= */

SIXTRL_INLINE NS(BlocksContainer)* NS(BlocksContainer_preset)( 
    NS(BlocksContainer)* SIXTRL_RESTRICT container )
{
    SIXTRL_STATIC NS(block_size_t) const 
        REAL_SIZE = sizeof( SIXTRL_REAL_T );
        
    SIXTRL_STATIC NS(block_size_t) const 
        ELEMENT_ID_SIZE = sizeof( NS(element_id_t) );
        
    SIXTRL_STATIC NS(block_size_t) const ZERO_SIZE = ( NS(block_size_t) )0u;
    
    if( container != 0 )
    {
        NS(block_size_t) const DEFAULT_DATA_ALIGNMENT = 
            NS(Alignment_calculate_common)( REAL_SIZE, ELEMENT_ID_SIZE );
        
        SIXTRL_STATIC NS(block_size_t) const 
            DEFAULT_INFO_ALIGNMENT = sizeof( NS(BlockInfo) );
        
        container->info_begin           = 0;
        container->data_begin           = 0;
        
        container->ptr_data_store       = 0;
        container->ptr_info_store       = 0;
        
        container->num_blocks           = ZERO_SIZE;
        container->blocks_capacity      = ZERO_SIZE;
        
        container->data_raw_size        = ZERO_SIZE;
        container->data_raw_capacity    = ZERO_SIZE;
        
        container->info_begin_alignment = DEFAULT_INFO_ALIGNMENT;
        container->info_alignment       = DEFAULT_INFO_ALIGNMENT;
        
        container->data_begin_alignment = DEFAULT_DATA_ALIGNMENT;
        container->data_alignment       = DEFAULT_DATA_ALIGNMENT;        
    }
    
    return container;
}
    
SIXTRL_INLINE int NS(BlocksContainer_set_info_begin_alignment)(
    NS(BlocksContainer)* SIXTRL_RESTRICT container, 
    NS(block_alignment_t) const begin_alignment )
{
    int success = -1;
    
    typedef NS(block_alignment_t)   align_t;
    typedef NS(block_size_t)        bl_size_t;
    
    if( ( container != 0 ) && 
        ( begin_alignment > ( align_t )0u ) &&
        ( container->num_blocks == ( bl_size_t )0u ) )
    {
        container->info_begin_alignment = begin_alignment;
        success = 0;
    }
    
    return success;
}

SIXTRL_INLINE int NS(BlocksContainer_set_data_begin_alignment)(
    NS(BlocksContainer)* SIXTRL_RESTRICT container, 
    NS(block_alignment_t) const begin_alignment )
{
    int success = -1;
    
    typedef NS(block_alignment_t)   align_t;
    typedef NS(block_size_t)        bl_size_t;
    
    if( ( container != 0 ) && 
        ( begin_alignment > ( align_t )0u ) &&
        ( container->data_raw_size == ( bl_size_t )0u ) )
    {
        container->data_begin_alignment = begin_alignment;
        success = 0;
    }
    
    return success;
}

SIXTRL_INLINE int NS(BlocksContainer_set_data_alignment)(
    NS(BlocksContainer)* SIXTRL_RESTRICT container, 
    NS(block_alignment_t) const alignment )
{
    int success = -1;
    
    typedef NS(block_alignment_t)   align_t;
    typedef NS(block_size_t)        bl_size_t;
    
    if( ( container != 0 ) && 
        ( alignment > ( align_t )0u ) &&
        ( container->data_raw_size == ( bl_size_t )0u ) )
    {
        container->data_alignment = alignment;
        success = 0;
    }
    
    return success;
}

SIXTRL_INLINE int NS(BlocksContainer_set_info_alignment )(
    NS(BlocksContainer)* SIXTRL_RESTRICT container, 
    NS(block_alignment_t) const alignment )
{
    int success = -1;
    
    typedef NS(block_alignment_t)   align_t;
    typedef NS(block_size_t)        bl_size_t;
    
    if( ( container != 0 ) && 
        ( alignment > ( align_t )0u ) &&
        ( container->num_blocks == ( bl_size_t )0u ) )
    {
        container->info_alignment = alignment;
        success = 0;
    }
    
    return success;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE NS(block_alignment_t) NS(BlocksContainer_get_info_alignment)(
    const NS(BlocksContainer) *const SIXTRL_RESTRICT container )
{
    SIXTRL_ASSERT( container != 0 );
    return container->info_alignment;
}

SIXTRL_INLINE NS(block_alignment_t) NS(BlocksContainer_get_data_alignment)(
    const NS(BlocksContainer) *const SIXTRL_RESTRICT container )
{
    SIXTRL_ASSERT( container != 0 );
    return container->data_alignment;
}

SIXTRL_INLINE NS(block_alignment_t) 
NS(BlocksContainer_get_info_begin_alignment)(
    const NS(BlocksContainer) *const SIXTRL_RESTRICT container )
{
    SIXTRL_ASSERT( container != 0 );
    return container->info_begin_alignment;
}

SIXTRL_INLINE NS(block_alignment_t) 
NS(BlocksContainer_get_data_begin_alignment)(
    const NS(BlocksContainer) *const SIXTRL_RESTRICT container )
{
    SIXTRL_ASSERT( container != 0 );
    return container->data_begin_alignment;
}

SIXTRL_INLINE NS(block_size_t) NS(BlocksContainer_get_data_capacity)(
    const NS(BlocksContainer) *const SIXTRL_RESTRICT container )
{
    SIXTRL_ASSERT( container != 0 ) ;
    return container->data_raw_capacity;
}

SIXTRL_INLINE NS(block_size_t) NS(BlocksContainer_get_data_size)(
    const NS(BlocksContainer) *const SIXTRL_RESTRICT container )
{
    SIXTRL_ASSERT( container != 0 );
    return container->data_raw_size;
}

SIXTRL_INLINE NS(block_size_t) NS(BlocksContainer_get_block_capacity)(
    const NS(BlocksContainer) *const SIXTRL_RESTRICT container )
{
    SIXTRL_ASSERT( container != 0 );
    return container->blocks_capacity;
}

SIXTRL_INLINE NS(block_size_t) NS(BlocksContainer_get_num_of_blocks)(
    const NS(BlocksContainer) *const SIXTRL_RESTRICT container )
{
    SIXTRL_ASSERT( container != 0 );
    return container->num_blocks;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE unsigned char const* 
NS(BlocksContainer_get_const_ptr_data_begin)(
    const NS(BlocksContainer) *const SIXTRL_RESTRICT container )
{
    SIXTRL_ASSERT( container != 0 );
    return container->data_begin;
}

SIXTRL_INLINE unsigned char* NS(BlocksContainer_get_ptr_data_begin)(
    NS(BlocksContainer)* SIXTRL_RESTRICT container )
{
    return ( unsigned char* )NS(BlocksContainer_get_const_ptr_data_begin)( 
        container );
}

SIXTRL_INLINE NS(BlockInfo) const* 
NS(BlocksContainer_get_const_block_infos_begin)(
    const NS(BlocksContainer) *const SIXTRL_RESTRICT container )
{
    SIXTRL_ASSERT( container != 0 );
    return container->info_begin;
}

SIXTRL_INLINE NS(BlockInfo) const* 
NS(BlocksContainer_get_const_block_infos_end)(
    const NS(BlocksContainer) *const SIXTRL_RESTRICT container )
{
    NS(BlockInfo) const* end_ptr = 
        NS(BlocksContainer_get_const_block_infos_begin)( container );
        
    if( end_ptr != 0 ) 
    {
        end_ptr = end_ptr + 
            NS(BlocksContainer_get_num_of_blocks)( container );
    }
    
    return end_ptr;
}


SIXTRL_INLINE NS(BlockInfo)* NS(BlocksContainer_get_block_infos_begin)(
    NS(BlocksContainer)* SIXTRL_RESTRICT container )
{
    return ( NS(BlockInfo)* )NS(BlocksContainer_get_const_block_infos_begin)(
        container );
}

SIXTRL_INLINE NS(BlockInfo)* NS(BlocksContainer_get_block_infos_end)(
    NS(BlocksContainer)* SIXTRL_RESTRICT container )
{
    return ( NS(BlockInfo)* )NS(BlocksContainer_get_const_block_infos_end)( 
        container );
}

SIXTRL_INLINE NS(BlockInfo) NS(BlocksContainer_get_block_info_by_index)(
    const NS(BlocksContainer) *const SIXTRL_RESTRICT container, 
    NS(block_size_t) const block_index )
{
    NS(BlockInfo) const* ptr_to_info = 
        NS(BlocksContainer_get_const_block_infos_end)( container );
        
    SIXTRL_ASSERT( ptr_to_info != 0 );
    return *ptr_to_info;
}

SIXTRL_INLINE NS(BlockInfo) const* 
NS(BlocksContainer_get_const_ptr_to_block_info_by_index)(
    const NS(BlocksContainer) *const SIXTRL_RESTRICT container, 
    NS(block_size_t) const block_index )
{
    SIXTRL_ASSERT( container != 0 );
    return ( ( container != 0 ) && ( container->info_begin != 0 ) &&
             ( NS(BlocksContainer_get_block_capacity)( container ) >
               block_index ) )
        ? &container->info_begin[ block_index ] : 0;
}

SIXTRL_INLINE NS(BlockInfo)* 
NS(BlocksContainer_get_ptr_to_block_info_by_index)(
    NS(BlocksContainer)* SIXTRL_RESTRICT container, 
    NS(block_size_t) const block_index )
{
    return ( NS(BlockInfo)* 
        )NS(BlocksContainer_get_const_ptr_to_block_info_by_index)(
            container, block_index );
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE int NS(BlocksContainer_has_info_store)(
    const NS(BlocksContainer) *const SIXTRL_RESTRICT container )
{
    return ( container != 0 ) ? ( container->ptr_info_store != 0 ) : 0;
}

SIXTRL_INLINE int NS(BlocksContainer_has_data_store)(
    const NS(BlocksContainer) *const SIXTRL_RESTRICT container )
{
    return ( container != 0 ) ? ( container->ptr_data_store != 0 ) : 0;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE int NS(BlocksContainer_assemble)(
    NS(BlocksContainer)* SIXTRL_RESTRICT container,
    SIXTRL_GLOBAL_DEC NS(BlockInfo)* SIXTRL_RESTRICT blk_infos_begin,
    NS(block_size_t) const num_of_blocks,
    SIXTRL_GLOBAL_DEC unsigned char* SIXTRL_RESTRICT data_mem_begin,
    NS(block_size_t) const data_num_of_bytes )
{
    typedef NS(block_alignment_t) align_t;
    
    int success = -1;
    
    NS(BlocksContainer_preset)( container );
    
    if( ( container != 0 ) &&
        ( blk_infos_begin != 0 ) && ( data_mem_begin != 0 ) &&
        ( num_of_blocks     > ( NS( block_size_t ) )0u ) &&
        ( data_num_of_bytes > ( NS( block_size_t ) )0u ) )
    {
        SIXTRL_ASSERT( 
            ( NS(BlockInfo_has_common_alignment)( blk_infos_begin ) ) &&
            ( NS(BlockInfo_get_common_alignment)( blk_infos_begin ) > 
                ( align_t )0u ) &&
            ( ( NS(BlockInfo_get_common_alignment)( 
                blk_infos_begin ) % ( align_t )2u ) == ( align_t )0u ) &&
            ( ( ( ( uintptr_t )data_mem_begin ) % 
                NS(BlockInfo_get_common_alignment)( blk_infos_begin ) ) == 
                ( uintptr_t )0u ) );
        
        container->data_alignment = 
            NS(BlockInfo_get_common_alignment)( blk_infos_begin );
        
        container->data_begin_alignment =
            NS(BlockInfo_get_common_alignment)( blk_infos_begin );
            
        container->info_alignment = ( align_t )sizeof( NS(BlockInfo) );
        container->info_begin_alignment = ( align_t )sizeof( NS(BlockInfo) );
        
        container->info_begin = blk_infos_begin;
        container->data_begin = data_mem_begin;
        
        container->num_blocks = num_of_blocks;
        container->blocks_capacity = num_of_blocks;
        
        container->data_raw_capacity = data_num_of_bytes;
        container->data_raw_size     = data_num_of_bytes;        
        
        success = 0;
    }
    
    return success;
}

#if !defined( _GPUCODE )
    
#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* !defined( _GPUCODE ) */

#endif /* SIXTRACKLIB_COMMON_IMPL_BLOCK_INFO_IMPL_H__ */

/* end: sixtracklib/common/impl/block_info_impl.h */
