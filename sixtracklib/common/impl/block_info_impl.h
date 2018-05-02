#ifndef SIXTRACKLIB_COMMON_IMPL_BLOCK_INFO_IMPL_H__
#define SIXTRACKLIB_COMMON_IMPL_BLOCK_INFO_IMPL_H__

#if !defined( _GPUCODE )

#include "sixtracklib/_impl/definitions.h"

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

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

SIXTRL_STATIC NS(BlockInfo)* NS(BlockInfo_preset)( 
    NS(BlockInfo)* SIXTRL_RESTRICT info );

SIXTRL_STATIC NS(block_size_t) 
NS(BlockInfo_get_total_num_of_elements_in_blocks)(
    const NS(BlockInfo) *const SIXTRL_RESTRICT infos, 
    NS(block_size_t) const num_of_blocks );

SIXTRL_STATIC NS(block_size_t) NS(BlockInfo_get_total_storage_size)(
    const NS(BlockInfo) *const SIXTRL_RESTRICT infos, 
    NS(block_size_t) const num_of_blocks );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC NS(block_size_t) NS(BlockInfo_get_mem_offset)(
    const NS(BlockInfo) *const SIXTRL_RESTRICT info );

SIXTRL_STATIC NS(block_size_t) NS(BlockInfo_get_next_mem_offset)(
    const NS(BlockInfo) *const SIXTRL_RESTRICT info );

SIXTRL_STATIC NS(block_size_t) NS(BlockInfo_get_num_of_bytes)(
    const NS(BlockInfo) *const SIXTRL_RESTRICT info );

SIXTRL_STATIC SIXTRL_GLOBAL_DEC unsigned char* 
    NS(BlockInfo_get_ptr_to_data_begin)(
        const NS(BlockInfo) *const SIXTRL_RESTRICT info, 
        SIXTRL_GLOBAL_DEC unsigned char* SIXTRL_RESTRICT memory_begin );

SIXTRL_STATIC SIXTRL_GLOBAL_DEC unsigned char const* 
    NS(BlockInfo_get_const_ptr_to_data_begin)(
        const NS(BlockInfo) *const SIXTRL_RESTRICT info, 
        SIXTRL_GLOBAL_DEC unsigned char const* SIXTRL_RESTRICT memory_begin );

SIXTRL_STATIC SIXTRL_GLOBAL_DEC unsigned char* 
    NS(BlockInfo_get_ptr_to_next_data_begin)(
        const NS(BlockInfo) *const SIXTRL_RESTRICT info, 
        SIXTRL_GLOBAL_DEC unsigned char* SIXTRL_RESTRICT memory_begin );

SIXTRL_STATIC SIXTRL_GLOBAL_DEC unsigned char const* 
    NS(BlockInfo_get_const_ptr_to_next_data_begin)(
        const NS(BlockInfo) *const SIXTRL_RESTRICT info, 
        SIXTRL_GLOBAL_DEC unsigned char const* SIXTRL_RESTRICT memory_begin );
    
SIXTRL_STATIC void NS(BlockInfo_set_mem_offset)(
    NS(BlockInfo)* SIXTRL_RESTRICT info, NS(block_size_t) const offset );

SIXTRL_STATIC void NS(BlockInfo_set_num_of_bytes)(
    NS(BlockInfo)* SIXTRL_RESTRICT info, NS(block_size_t) const num_bytes );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC NS(BlockType) NS(BlockInfo_get_type_id)(
    const NS(BlockInfo) *const SIXTRL_RESTRICT info );

SIXTRL_STATIC NS(block_type_num_t) NS(BlockInfo_get_type_id_num)(
    const NS(BlockInfo) *const SIXTRL_RESTRICT info );

SIXTRL_STATIC void NS(BlockInfo_set_type_id)(
    NS(BlockInfo)* SIXTRL_RESTRICT info, NS(BlockType) const type_id );

SIXTRL_STATIC void NS(BlockInfo_set_type_id_num)(
    NS(BlockInfo)* SIXTRL_RESTRICT info, 
    NS(block_type_num_t) const type_id );

SIXTRL_STATIC int NS(BlockInfo_is_a_particles_block)(
    const NS(BlockInfo) *const SIXTRL_RESTRICT info );

SIXTRL_STATIC int NS(BlockInfo_is_a_beam_element_block)(
    const NS(BlockInfo) *const SIXTRL_RESTRICT info );

SIXTRL_STATIC int NS(BlockInfo_is_a_mapping_info_block)(
    const NS(BlockInfo) *const SIXTRL_RESTRICT info );

SIXTRL_STATIC int NS(BlockInfo_is_a_userdefined_block)(
    const NS(BlockInfo) *const SIXTRL_RESTRICT info );


/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC NS(block_num_elements_t) NS(BlockInfo_get_num_elements)(
    const NS(BlockInfo) *const SIXTRL_RESTRICT info );

SIXTRL_STATIC void NS(BlockInfo_set_num_elements)(
    NS(BlockInfo)* SIXTRL_RESTRICT info, 
    NS(block_num_elements_t) const num_elements );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC int NS(BlockInfo_has_common_alignment)(
    const NS(BlockInfo) *const SIXTRL_RESTRICT info );

SIXTRL_STATIC int NS(BlockInfo_has_associated_mapping_info)(
    const NS(BlockInfo) *const SIXTRL_RESTRICT info );

SIXTRL_STATIC NS(block_map_info_id_t) 
NS(BlockInfo_get_associated_map_info_id)(
    const NS(BlockInfo) *const SIXTRL_RESTRICT info );

SIXTRL_STATIC NS(block_alignment_t) NS(BlockInfo_get_common_alignment)(
    const NS(BlockInfo) *const SIXTRL_RESTRICT info );

SIXTRL_STATIC void NS(BlockInfo_set_common_alignment)(
    NS(BlockInfo)* SIXTRL_RESTRICT info, 
    NS(block_alignment_t) const alignment );

SIXTRL_STATIC void NS(BlockInfo_set_associated_mapping_header_id)(
    NS(BlockInfo)* SIXTRL_RESTRICT info,
    NS(block_map_info_id_t) const map_info_id );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC int NS(BlockInfo_generic_map_to_memory_for_writing_aligned)(
    NS(BlockInfo)* SIXTRL_RESTRICT block_info, 
    SIXTRL_GLOBAL_DEC unsigned char** SIXTRL_RESTRICT attrs_ptr,
    NS(block_size_t)* SIXTRL_RESTRICT num_bytes_for_attrs,
    NS(block_size_t) const num_of_attributes,
    NS(block_num_elements_t) const num_elements, NS(BlockType) const type_id,
    SIXTRL_GLOBAL_DEC unsigned char* SIXTRL_RESTRICT mem_begin, 
    NS(block_size_t) const max_num_of_bytes_in_buffer );    

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC int NS(BlockInfo_generic_map_from_memory_for_reading_aligned)(
    const NS(BlockInfo) *const SIXTRL_RESTRICT block_info,
    SIXTRL_GLOBAL_DEC unsigned char** SIXTRL_RESTRICT attrs_ptr, 
    NS(block_size_t)* SIXTRL_RESTRICT num_bytes_for_attrs, 
    NS(block_size_t) const num_of_attributes, 
    SIXTRL_GLOBAL_DEC unsigned char* SIXTRL_RESTRICT mem_begin,
    NS(block_size_t) const max_num_of_bytes_in_buffer );

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

SIXTRL_INLINE NS(BlockInfo)* NS(BlockInfo_preset)( 
    NS(BlockInfo)* SIXTRL_RESTRICT info )
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
    const NS(BlockInfo) *const SIXTRL_RESTRICT infos, 
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
    const NS(BlockInfo) *const SIXTRL_RESTRICT infos, 
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
    const NS(BlockInfo) *const SIXTRL_RESTRICT info )
{
    SIXTRL_ASSERT( info != 0 );
    return info->mem_offset;
}

SIXTRL_INLINE NS(block_size_t) NS(BlockInfo_get_next_mem_offset)(
    const NS(BlockInfo) *const SIXTRL_RESTRICT info )
{
    SIXTRL_ASSERT( info != 0 );
    return ( info->mem_offset + info->num_bytes );
}

SIXTRL_INLINE NS(block_size_t) NS(BlockInfo_get_num_of_bytes)(
    const NS(BlockInfo) *const SIXTRL_RESTRICT info )
{
    SIXTRL_ASSERT( info != 0 );
    return info->num_bytes;
}

SIXTRL_INLINE SIXTRL_GLOBAL_DEC unsigned char* 
    NS(BlockInfo_get_ptr_to_data_begin)(
        const NS(BlockInfo) *const SIXTRL_RESTRICT info, 
        SIXTRL_GLOBAL_DEC unsigned char* SIXTRL_RESTRICT memory_begin )
{
    typedef SIXTRL_GLOBAL_DEC unsigned char*  g_ptr_uchar_t;
    
    return ( g_ptr_uchar_t )NS(BlockInfo_get_const_ptr_to_data_begin)( 
        info, memory_begin );
}

SIXTRL_INLINE SIXTRL_GLOBAL_DEC unsigned char const* 
    NS(BlockInfo_get_const_ptr_to_data_begin)(
        const NS(BlockInfo) *const SIXTRL_RESTRICT info, 
        SIXTRL_GLOBAL_DEC unsigned char const* SIXTRL_RESTRICT memory_begin )
{
    SIXTRL_ASSERT( ( info != 0 ) && ( memory_begin != 0 ) );
    return ( memory_begin + info->mem_offset );
}

SIXTRL_INLINE SIXTRL_GLOBAL_DEC unsigned char* 
    NS(BlockInfo_get_ptr_to_next_data_begin)(
        const NS(BlockInfo) *const SIXTRL_RESTRICT info, 
        SIXTRL_GLOBAL_DEC unsigned char* SIXTRL_RESTRICT mem_begin )
{
    typedef SIXTRL_GLOBAL_DEC unsigned char*  g_ptr_uchar_t;
    
    return ( g_ptr_uchar_t 
    )NS(BlockInfo_get_const_ptr_to_next_data_begin)( info, mem_begin );
}

SIXTRL_INLINE SIXTRL_GLOBAL_DEC unsigned char const* 
    NS(BlockInfo_get_const_ptr_to_next_data_begin)(
        const NS(BlockInfo) *const SIXTRL_RESTRICT info, 
        SIXTRL_GLOBAL_DEC unsigned char const* SIXTRL_RESTRICT memory_begin )
{
    SIXTRL_ASSERT( ( info != 0 ) && ( memory_begin != 0 ) );
    return ( memory_begin + info->mem_offset + info->num_bytes );
}
    
SIXTRL_INLINE void NS(BlockInfo_set_mem_offset)(
    NS(BlockInfo)* SIXTRL_RESTRICT info, 
    NS(block_size_t) const offset )
{
    SIXTRL_ASSERT( info != 0 );
    info->mem_offset = offset;
    return;
}

SIXTRL_INLINE void NS(BlockInfo_set_num_of_bytes)(
    NS(BlockInfo)* SIXTRL_RESTRICT info, 
    NS(block_size_t) const num_bytes )
{
    SIXTRL_ASSERT( info != 0 );
    info->num_bytes = num_bytes;
    return;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE NS(BlockType)  NS(BlockInfo_get_type_id)(
    const NS(BlockInfo) *const SIXTRL_RESTRICT info )
{
    SIXTRL_ASSERT( info != 0 );
    return NS(BlockType_from_number)( info->type_id_num );
}

SIXTRL_INLINE NS(block_type_num_t) NS(BlockInfo_get_type_id_num)(
    const NS(BlockInfo) *const SIXTRL_RESTRICT info )
{
    SIXTRL_ASSERT( info != 0 );
    return info->type_id_num;
}

SIXTRL_INLINE void NS(BlockInfo_set_type_id)(
    NS(BlockInfo)* SIXTRL_RESTRICT info, 
    NS(BlockType) const type_id )
{
    SIXTRL_ASSERT( info != 0 );
    info->type_id_num = NS(BlockType_to_number)( type_id );
}

SIXTRL_INLINE void NS(BlockInfo_set_type_id_num)(
    NS(BlockInfo)* SIXTRL_RESTRICT info, 
    NS(block_type_num_t) const type_id_num )
{
    SIXTRL_ASSERT( info );
    info->type_id_num = type_id_num;
}

SIXTRL_INLINE int NS(BlockInfo_is_a_particles_block)(
    const NS(BlockInfo) *const SIXTRL_RESTRICT info )
{
    NS(BlockType) const type_id = NS(BlockInfo_get_type_id)( info );
    return ( type_id == NS(BLOCK_TYPE_PARTICLE) );
}

SIXTRL_INLINE int NS(BlockInfo_is_a_beam_element_block)(
    const NS(BlockInfo) *const SIXTRL_RESTRICT info )
{
    NS(BlockType) const type_id = NS(BlockInfo_get_type_id)( info );
    return ( ( type_id == NS(BLOCK_TYPE_DRIFT) ) ||
             ( type_id == NS(BLOCK_TYPE_DRIFT_EXACT) ) ||
             ( type_id == NS(BLOCK_TYPE_MULTIPOLE) ) ||
             ( type_id == NS(BLOCK_TYPE_CAVITY) ) ||
             ( type_id == NS(BLOCK_TYPE_ALIGN) ) );
}

SIXTRL_INLINE int NS(BlockInfo_is_a_mapping_info_block)(
    const NS(BlockInfo) *const SIXTRL_RESTRICT info )
{
    NS(BlockType) const type_id = NS(BlockInfo_get_type_id)( info );
    return ( type_id == NS(BLOCK_TYPE_EXT_MAP_INFO) );
}

SIXTRL_INLINE int NS(BlockInfo_is_a_userdefined_block)(
    const NS(BlockInfo) *const SIXTRL_RESTRICT info )
{
    NS(BlockType) const type_id = NS(BlockInfo_get_type_id)( info );
    return ( type_id == NS(BLOCK_TYPE_USERDEFINED) );
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE NS(block_num_elements_t) NS(BlockInfo_get_num_elements)(
    const NS(BlockInfo) *const SIXTRL_RESTRICT info )
{
    SIXTRL_ASSERT( info != 0 );
    return info->num_elements;
}

SIXTRL_INLINE void NS(BlockInfo_set_num_elements)(
    NS(BlockInfo)* SIXTRL_RESTRICT info, 
    NS(block_num_elements_t) const neleme )
{
    SIXTRL_ASSERT( info != 0 );
    info->num_elements = neleme;
    return;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE int NS(BlockInfo_has_common_alignment)(
    const NS(BlockInfo) *const SIXTRL_RESTRICT info )
{
    return ( 
        ( info != 0 ) && ( info->store_info > 0 ) && 
        ( info->type_id_num != NS(BlockType_to_number)( 
            NS(BLOCK_TYPE_EXT_MAP_INFO) ) ) );
}

SIXTRL_INLINE int NS(BlockInfo_has_associated_mapping_info)(
    const NS(BlockInfo) *const SIXTRL_RESTRICT info )
{
    return ( ( info != 0 ) && ( info->store_info < 0 ) &&
             ( info->type_id_num != NS(BlockType_to_number)(
                 NS(BLOCK_TYPE_EXT_MAP_INFO) ) ) );
}

SIXTRL_INLINE NS(block_map_info_id_t) 
NS(BlockInfo_get_associated_map_info_id)(
    const NS(BlockInfo) *const SIXTRL_RESTRICT info )
{
    return ( info != 0 ) ? ( -( info->store_info ) ) : 0;
}

SIXTRL_INLINE NS(block_alignment_t) NS(BlockInfo_get_common_alignment)(
    const NS(BlockInfo) *const SIXTRL_RESTRICT info )
{
    return ( info != 0 ) ? ( info->store_info ) : 0;
}

SIXTRL_INLINE void NS(BlockInfo_set_common_alignment)(
    NS(BlockInfo)* SIXTRL_RESTRICT info, 
    NS(block_alignment_t) const alignment )
{
    if( ( info != 0 ) && ( alignment >= 0 ) )
    {
        info->store_info = alignment;
    }
    
    return;
}

SIXTRL_INLINE void NS(BlockInfo_set_associated_mapping_header_id)(
    NS(BlockInfo)* SIXTRL_RESTRICT info,
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

SIXTRL_INLINE int NS(BlockInfo_generic_map_to_memory_for_writing_aligned)(
    NS(BlockInfo)* SIXTRL_RESTRICT block_info, 
    SIXTRL_GLOBAL_DEC unsigned char** SIXTRL_RESTRICT attrs_ptr,
    NS(block_size_t)* SIXTRL_RESTRICT num_bytes_for_attrs,
    NS(block_size_t) const num_of_attributes,
    NS(block_num_elements_t) const num_elements, NS(BlockType) const type_id,
    SIXTRL_GLOBAL_DEC unsigned char* SIXTRL_RESTRICT mem_begin, 
    NS(block_size_t) const max_num_of_bytes_in_buffer )
{
    int success = -1;
    static NS(block_size_t) const ZERO = ( NS(block_size_t) )0u;
    
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
                                                                          
SIXTRL_INLINE int NS(BlockInfo_generic_map_from_memory_for_reading_aligned)(
    const NS(BlockInfo) *const SIXTRL_RESTRICT block_info,
    SIXTRL_GLOBAL_DEC unsigned char** SIXTRL_RESTRICT attrs_ptr, 
    NS(block_size_t)* SIXTRL_RESTRICT num_bytes_for_attrs, 
    NS(block_size_t) const num_of_attributes, 
    SIXTRL_GLOBAL_DEC unsigned char* SIXTRL_RESTRICT mem_begin,
    NS(block_size_t) const max_num_of_bytes_in_buffer )
{
    int success = -1;
    static NS(block_size_t) const ZERO = ( NS(block_size_t) )0u;
    
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
    

#if !defined( _GPUCODE )
    
#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* !defined( _GPUCODE ) */

#endif /* SIXTRACKLIB_COMMON_IMPL_BLOCK_INFO_IMPL_H__ */

/* end: sixtracklib/common/impl/block_info_impl.h */
