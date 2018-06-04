#ifndef SIXTRACKLIB_COMMON_BLOCKS_H__
#define SIXTRACKLIB_COMMON_BLOCKS_H__

#if !defined( _GPUCODE )

#include "sixtracklib/_impl/definitions.h"

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include "sixtracklib/common/impl/alignment_impl.h"
#include "sixtracklib/common/mem_pool.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#endif /* !defined( _GPUCODE ) */

/* ------------------------------------------------------------------------- */

typedef SIXTRL_UINT64_T NS(block_type_num_t);

typedef enum NS(BlockType)
{
    NS(BLOCK_TYPE_NONE)             = 0x0000000,
    NS(BLOCK_TYPE_PARTICLE)         = 0x0000001,
    NS(BLOCK_TYPE_DRIFT)            = 0x0000002,
    NS(BLOCK_TYPE_DRIFT_EXACT)      = 0x0000003,
    NS(BLOCK_TYPE_MULTIPOLE)        = 0x0000004,
    NS(BLOCK_TYPE_CAVITY)           = 0x0000005,
    NS(BLOCK_TYPE_ALIGN)            = 0x0000006,
    NS(BLOCK_TYPE_USERDEFINED)      = 0x0000007,
    NS(BLOCK_TYPE_INVALID)          = 0xfffffff
}
NS(BlockType);

SIXTRL_STATIC NS(block_type_num_t) NS(BlockType_to_number)(
    NS(BlockType) const type_id );
    
SIXTRL_STATIC NS(BlockType) NS(BlockType_from_number)(
    NS(block_type_num_t) const type_id_num );

SIXTRL_STATIC int NS(BlockType_is_valid_number)(
    NS(block_type_num_t) const type_id_num );

/* ------------------------------------------------------------------------- */

typedef SIXTRL_UINT64_T NS(block_size_t);
typedef SIXTRL_INT64_T  NS(block_num_elements_t);
typedef SIXTRL_INT64_T  NS(block_element_id_t);

typedef struct NS(BlockInfo)
{
    NS(block_type_num_t) type_id_num  __attribute__(( aligned( 8 ) ));            
    
    SIXTRL_GLOBAL_DEC void* SIXTRL_RESTRICT 
        begin __attribute__(( aligned( 8 ) ));    
        
    NS(block_size_t) length __attribute__(( aligned( 8 ) ));
    
    SIXTRL_GLOBAL_DEC void* SIXTRL_RESTRICT 
        ptr_metadata __attribute__(( aligned( 8 ) ));        
}
NS(BlockInfo);

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC SIXTRL_GLOBAL_DEC NS(BlockInfo)* 
NS(BlockInfo_preset)( SIXTRL_GLOBAL_DEC NS(BlockInfo)* SIXTRL_RESTRICT info );

SIXTRL_STATIC NS(BlockType) NS(BlockInfo_get_type_id)( 
    const NS(BlockInfo) *const SIXTRL_RESTRICT info );

SIXTRL_STATIC NS(block_type_num_t) NS(BlockInfo_get_type_id_num)(
    const NS(BlockInfo) *const SIXTRL_RESTRICT info );

SIXTRL_STATIC void NS(BlockInfo_set_type_id)(
    NS(BlockInfo)* SIXTRL_RESTRICT info, NS(BlockType) const type_id );

SIXTRL_STATIC void NS(BlockInfo_set_type_id_num)(
    NS(BlockInfo)* SIXTRL_RESTRICT info, NS(block_type_num_t) const num );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC SIXTRL_GLOBAL_DEC void const* 
NS(BlockInfo_get_const_ptr_begin)(
    const NS(BlockInfo) *const SIXTRL_RESTRICT info );

SIXTRL_STATIC SIXTRL_GLOBAL_DEC void* NS(BlockInfo_get_ptr_begin)(
    NS(BlockInfo)* SIXTRL_RESTRICT info );

SIXTRL_STATIC void NS(BlockInfo_set_ptr_begin)( 
    NS(BlockInfo)* SIXTRL_RESTRICT info, 
    SIXTRL_GLOBAL_DEC void* SIXTRL_RESTRICT ptr_begin );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC SIXTRL_GLOBAL_DEC void const* 
NS(BlockInfo_get_const_ptr_metadata)(
    const NS(BlockInfo) *const SIXTRL_RESTRICT info );

SIXTRL_STATIC SIXTRL_GLOBAL_DEC void* NS(BlockInfo_get_ptr_metadata)(
    NS(BlockInfo)* SIXTRL_RESTRICT info );

SIXTRL_STATIC void NS(BlockInfo_set_ptr_metadata)( 
    NS(BlockInfo)* SIXTRL_RESTRICT info, 
    SIXTRL_GLOBAL_DEC void* SIXTRL_RESTRICT ptr_metadata );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC NS(block_size_t) NS(BlockInfo_get_block_size)(
    const NS(BlockInfo) *const SIXTRL_RESTRICT info );

SIXTRL_STATIC void NS(BlockInfo_set_block_size)(
    NS(BlockInfo)* SIXTRL_RESTRICT info, NS(block_size_t) const length );

/* ------------------------------------------------------------------------- */

typedef struct NS(Blocks)
{
    SIXTRL_GLOBAL_DEC unsigned char* 
        SIXTRL_RESTRICT ptr_data_begin       __attribute__(( aligned( 8 ) ));
    
    SIXTRL_GLOBAL_DEC NS(BlockInfo)*         
        SIXTRL_RESTRICT ptr_block_infos      __attribute__(( aligned( 8 ) ));
    
    SIXTRL_GLOBAL_DEC void*** SIXTRL_RESTRICT
        ptr_data_pointers_begin              __attribute__(( aligned( 8 ) ));
    
    SIXTRL_GLOBAL_DEC SIXTRL_UINT64_T*       
        SIXTRL_RESTRICT ptr_total_num_bytes  __attribute__(( aligned( 8 ) ));
    
    void* SIXTRL_RESTRICT data_store         __attribute__(( aligned( 8 ) ));        
    void* SIXTRL_RESTRICT index_store        __attribute__(( aligned( 8 ) ));        
    void* SIXTRL_RESTRICT data_ptrs_store    __attribute__(( aligned( 8 ) ));
    
    NS(block_size_t) data_size               __attribute__(( aligned( 8 ) ));
    NS(block_size_t) num_blocks              __attribute__(( aligned( 8 ) ));    
    NS(block_size_t) num_data_pointers       __attribute__(( aligned( 8 ) ));
    
    NS(block_size_t) begin_alignment         __attribute__(( aligned( 8 ) ));
    NS(block_size_t) alignment               __attribute__(( aligned( 8 ) ));
    int is_serialized                        __attribute__(( aligned( 8 ) ));
}
NS(Blocks);

SIXTRL_STATIC NS(Blocks)* NS(Blocks_preset)( 
    NS(Blocks)* SIXTRL_RESTRICT blocks );

SIXTRL_STATIC int NS(Blocks_are_serialized)( 
    const NS(Blocks) *const SIXTRL_RESTRICT blocks );

SIXTRL_STATIC SIXTRL_GLOBAL_DEC unsigned char* 
NS(Blocks_get_data_begin)( NS(Blocks)* SIXTRL_RESTRICT blocks );

SIXTRL_STATIC SIXTRL_GLOBAL_DEC unsigned char* 
NS(Blocks_get_data_end)( NS(Blocks)* SIXTRL_RESTRICT blocks );

SIXTRL_STATIC SIXTRL_GLOBAL_DEC unsigned char const* 
NS(Blocks_get_const_data_begin)( NS(Blocks)* SIXTRL_RESTRICT blocks );

SIXTRL_STATIC SIXTRL_GLOBAL_DEC unsigned char const* 
NS(Blocks_get_const_data_end)( NS(Blocks)* SIXTRL_RESTRICT blocks );

SIXTRL_STATIC NS(block_size_t) NS(Blocks_predict_data_capacity_for_num_blocks)(
    const NS(Blocks) *const SIXTRL_RESTRICT blocks, 
    NS(block_size_t) const num_of_blocks );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC NS(block_size_t) NS(Blocks_get_total_num_bytes)(
    const NS(Blocks) *const SIXTRL_RESTRICT blocks );

SIXTRL_STATIC int NS(Blocks_remap)( NS(Blocks)* SIXTRL_RESTRICT blocks );

SIXTRL_STATIC int NS(Blocks_unserialize)( NS(Blocks)* SIXTRL_RESTRICT blocks,
    SIXTRL_GLOBAL_DEC unsigned char* SIXTRL_RESTRICT data_mem_begin );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC NS(block_size_t) NS(Blocks_get_max_num_of_blocks)( 
    const NS(Blocks) *const SIXTRL_RESTRICT blocks );

SIXTRL_STATIC NS(block_size_t) NS(Blocks_get_num_of_blocks)(
    const NS(Blocks) *const SIXTRL_RESTRICT blocks );

SIXTRL_STATIC NS(block_size_t) NS(Blocks_get_blocks_write_capacity)(
    const NS(Blocks) *const SIXTRL_RESTRICT blocks );

SIXTRL_STATIC NS(block_size_t) NS(Blocks_get_data_capacity)(
    const NS(Blocks) *const SIXTRL_RESTRICT blocks );

SIXTRL_STATIC NS(block_size_t) NS(Blocks_get_data_size)(
    const NS(Blocks) *const SIXTRL_RESTRICT blocks );

SIXTRL_STATIC NS(block_size_t) NS(Blocks_get_data_write_capacity)(
    const NS(Blocks) *const SIXTRL_RESTRICT blocks );

SIXTRL_STATIC NS(block_size_t) NS(Blocks_get_max_num_data_pointers)(
    const NS(Blocks) *const SIXTRL_RESTRICT blocks );

SIXTRL_STATIC NS(block_size_t) NS(Blocks_get_num_data_pointers)(
    const NS(Blocks) *const SIXTRL_RESTRICT blocks );

SIXTRL_STATIC NS(block_size_t) NS(Blocks_get_data_pointers_write_capacity)(
    const NS(Blocks) *const SIXTRL_RESTRICT blocks );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC void NS(Blocks_set_data_alignment)( 
    NS(Blocks)* SIXTRL_RESTRICT blocks, NS(block_size_t) const alignment );
                
SIXTRL_STATIC NS(block_size_t) NS(Blocks_get_data_alignment)( 
    const NS(Blocks) *const SIXTRL_RESTRICT blocks );

SIXTRL_STATIC void NS(Blocks_set_begin_alignment)(
    NS(Blocks)* SIXTRL_RESTRICT blocks, NS(block_size_t) const alignment );

SIXTRL_STATIC NS(block_size_t) NS(Blocks_get_begin_alignment)(
    const NS(Blocks) *const SIXTRL_RESTRICT blocks );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC SIXTRL_GLOBAL_DEC NS(BlockInfo) const*
NS(Blocks_get_const_block_infos_begin)( 
    const NS(Blocks) *const SIXTRL_RESTRICT blocks );

SIXTRL_STATIC SIXTRL_GLOBAL_DEC NS(BlockInfo) const* 
NS(Blocks_get_const_block_infos_end)( 
    const NS(Blocks) *const SIXTRL_RESTRICT blocks );

SIXTRL_STATIC SIXTRL_GLOBAL_DEC NS(BlockInfo) const* 
NS(Blocks_get_const_block_info_by_index)(
    const NS(Blocks) *const SIXTRL_RESTRICT blocks, 
    NS(block_size_t) const index );


SIXTRL_STATIC SIXTRL_GLOBAL_DEC NS(BlockInfo)*
NS(Blocks_get_block_infos_begin)( NS(Blocks)* SIXTRL_RESTRICT blocks );

SIXTRL_STATIC SIXTRL_GLOBAL_DEC NS(BlockInfo)* 
NS(Blocks_get_block_infos_end)( NS(Blocks)* SIXTRL_RESTRICT blocks );

SIXTRL_STATIC SIXTRL_GLOBAL_DEC NS(BlockInfo)* 
NS(Blocks_get_block_info_by_index)( 
    NS(Blocks)* SIXTRL_RESTRICT blocks, NS(block_size_t) const index );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC int NS(Blocks_has_data_store)(
    const NS(Blocks) *const SIXTRL_RESTRICT blocks );

SIXTRL_STATIC int NS(Blocks_has_index_store)(
    const NS(Blocks) *const SIXTRL_RESTRICT blocks );

SIXTRL_STATIC int NS(Blocks_has_data_pointers_store)(
    const NS(Blocks) *const SIXTRL_RESTRICT blocks );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

#if !defined( _GPUCODE )

int NS(Blocks_init)( NS(Blocks)* SIXTRL_RESTRICT blocks,
    NS(block_size_t) max_num_blocks, NS(block_size_t) const data_capacity );

void NS(Blocks_clear)( NS(Blocks)* SIXTRL_RESTRICT blocks );
void NS(Blocks_free)(  NS(Blocks)* SIXTRL_RESTRICT blocks );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_GLOBAL_DEC NS(BlockInfo)* NS(Blocks_add_block)( 
    NS(Blocks)* SIXTRL_RESTRICT blocks, NS(BlockType) const type_id,
    NS(block_size_t) const block_handle_size,
    const void *const SIXTRL_RESTRICT block_handle,
    NS(block_size_t) const num_attr_data_pointers, 
    const NS(block_size_t) *const SIXTRL_RESTRICT attr_data_pointer_offsets,
    const NS(block_size_t) *const SIXTRL_RESTRICT attr_type_sizes,
    const NS(block_size_t) *const SIXTRL_RESTRICT attr_type_counts );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

int NS(Blocks_serialize)( NS(Blocks)* SIXTRL_RESTRICT blocks );

#endif /* !defined( _GPUCODE ) */
    
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

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE NS(BlockType) NS(BlockType_from_number)(
    NS(block_type_num_t) const type_id_num )
{
    NS(BlockType) type_id;

    switch( type_id_num )
    {
        case ( NS(block_type_num_t) )NS(BLOCK_TYPE_NONE):
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

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE int NS(BlockType_is_valid_number)(
    NS(block_type_num_t) const type_id_num )
{
    return ( NS(BlockType_from_number)( type_id_num ) !=
             NS(BLOCK_TYPE_INVALID) );
}

/* ========================================================================= */

SIXTRL_INLINE SIXTRL_GLOBAL_DEC NS(BlockInfo)* NS(BlockInfo_preset)( 
    SIXTRL_GLOBAL_DEC NS(BlockInfo)* SIXTRL_RESTRICT info )
{
    NS(BlockInfo_set_ptr_begin)( info, 0 );
    NS(BlockInfo_set_type_id)( info, NS(BLOCK_TYPE_INVALID) );
    NS(BlockInfo_set_ptr_metadata)( info, 0 );
    NS(BlockInfo_set_block_size)( info, 0u );
    
    return info;
}


SIXTRL_INLINE NS(BlockType) NS(BlockInfo_get_type_id)( 
    const NS(BlockInfo) *const SIXTRL_RESTRICT info )
{
    return ( info != 0 ) 
        ? NS(BlockType_from_number)( info->type_id_num ) 
        : NS(BLOCK_TYPE_INVALID);
}

SIXTRL_INLINE NS(block_type_num_t) NS(BlockInfo_get_type_id_num)(
    const NS(BlockInfo) *const SIXTRL_RESTRICT info )
{
    return ( info != 0 )
        ? info->type_id_num 
        : NS(BlockType_to_number)( NS(BLOCK_TYPE_INVALID ) );
}

SIXTRL_INLINE void NS(BlockInfo_set_type_id)(
    NS(BlockInfo)* SIXTRL_RESTRICT info, NS(BlockType) const type_id )
{
    if( info != 0 ) info->type_id_num = NS(BlockType_to_number)( type_id );
    return;
}

SIXTRL_INLINE void NS(BlockInfo_set_type_id_num)(
    NS(BlockInfo)* SIXTRL_RESTRICT info, NS(block_type_num_t) const num )
{
    if( info != 0 ) info->type_id_num = num;
    return;
}

SIXTRL_INLINE SIXTRL_GLOBAL_DEC void const* NS(BlockInfo_get_const_ptr_begin)(
    const NS(BlockInfo) *const SIXTRL_RESTRICT info )
{
    return ( info != 0 ) ? info->begin : 0;
}

SIXTRL_INLINE SIXTRL_GLOBAL_DEC void* NS(BlockInfo_get_ptr_begin)(
    NS(BlockInfo)* SIXTRL_RESTRICT info )
{
    return ( void* )NS(BlockInfo_get_const_ptr_begin)( info );
}

SIXTRL_INLINE void NS(BlockInfo_set_ptr_begin)( 
    NS(BlockInfo)* SIXTRL_RESTRICT info, 
    SIXTRL_GLOBAL_DEC void* SIXTRL_RESTRICT ptr_begin )
{
    if( info != 0 ) info->begin = ptr_begin;
    return;
}

SIXTRL_INLINE SIXTRL_GLOBAL_DEC void const* 
NS(BlockInfo_get_const_ptr_metadata)(
    const NS(BlockInfo) *const SIXTRL_RESTRICT info )
{
    return ( info != 0 ) ? info->ptr_metadata : 0;
}

SIXTRL_INLINE SIXTRL_GLOBAL_DEC void* NS(BlockInfo_get_ptr_metadata)(
    NS(BlockInfo)* SIXTRL_RESTRICT info )
{
    return ( void* )NS(BlockInfo_get_const_ptr_metadata)( info );
}

SIXTRL_INLINE void NS(BlockInfo_set_ptr_metadata)( 
    NS(BlockInfo)* SIXTRL_RESTRICT info, 
    SIXTRL_GLOBAL_DEC void* SIXTRL_RESTRICT ptr_metadata )
{
    if( info != 0 ) info->ptr_metadata = ptr_metadata;
    return;
}

SIXTRL_INLINE NS(block_size_t) NS(BlockInfo_get_block_size)(
    const NS(BlockInfo) *const SIXTRL_RESTRICT info )
{
    return ( info != 0 ) ? info->length : ( NS(block_size_t) )0u;
}

SIXTRL_INLINE void NS(BlockInfo_set_block_size)(
    NS(BlockInfo)* SIXTRL_RESTRICT info, NS(block_size_t) const length )
{
    if( info != 0 ) info->length = length;
    return;
}

/* ========================================================================= */

SIXTRL_INLINE NS(Blocks)* NS(Blocks_preset)( 
    NS(Blocks)* SIXTRL_RESTRICT blocks )
{
    if( blocks != 0 )
    {
        static NS(block_size_t) const ZERO = ( NS(block_size_t) )0u;
        
        blocks->ptr_data_begin          = 0;        
        blocks->ptr_block_infos         = 0;
        blocks->ptr_data_pointers_begin = 0;
        blocks->ptr_total_num_bytes     = 0;
        
        blocks->data_store              = 0;        
        blocks->index_store             = 0; 
        blocks->data_ptrs_store         = 0;
                                        
        blocks->num_blocks              = ZERO;        
        blocks->data_size               = ZERO;
        blocks->num_data_pointers       = ZERO;
                                        
        blocks->begin_alignment         = ( NS(block_size_t) )8u;
        blocks->alignment               = ( NS(block_size_t) )8u;
                                        
        blocks->is_serialized            = 0;
    }
    
    return blocks;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE int NS(Blocks_are_serialized)( 
    const NS(Blocks) *const SIXTRL_RESTRICT blocks )
{
    return ( ( blocks != 0 ) && ( blocks->is_serialized == 1 ) );
}

SIXTRL_INLINE SIXTRL_GLOBAL_DEC unsigned char* 
NS(Blocks_get_data_begin)( NS(Blocks)* SIXTRL_RESTRICT blocks )
{
    typedef SIXTRL_GLOBAL_DEC unsigned char* g_uchar_ptr_t;
    return ( g_uchar_ptr_t )NS(Blocks_get_const_data_begin)( blocks );
}

SIXTRL_INLINE SIXTRL_GLOBAL_DEC unsigned char* 
NS(Blocks_get_data_end)( NS(Blocks)* SIXTRL_RESTRICT blocks )
{
    typedef SIXTRL_GLOBAL_DEC unsigned char* g_uchar_ptr_t;
    return ( g_uchar_ptr_t )NS(Blocks_get_const_data_end)( blocks );
}

SIXTRL_INLINE SIXTRL_GLOBAL_DEC unsigned char const* 
NS(Blocks_get_const_data_begin)( NS(Blocks)* SIXTRL_RESTRICT blocks )
{
    return ( NS(Blocks_are_serialized)( blocks ) ) 
        ? blocks->ptr_data_begin : 0;
}

SIXTRL_INLINE SIXTRL_GLOBAL_DEC unsigned char const* 
NS(Blocks_get_const_data_end)( NS(Blocks)* SIXTRL_RESTRICT blocks )
{
    SIXTRL_GLOBAL_DEC unsigned char const* end_ptr = 
        NS(Blocks_get_const_data_begin)( blocks );
        
    if( ( end_ptr != 0 ) && ( NS(Blocks_get_total_num_bytes)( blocks ) > 0u ) )
    {
        end_ptr = end_ptr + NS(Blocks_get_total_num_bytes)( blocks );
    }
    
    return end_ptr;
}

SIXTRL_INLINE NS(block_size_t) NS(Blocks_predict_data_capacity_for_num_blocks)(
    const NS(Blocks) *const SIXTRL_RESTRICT blocks, 
    NS(block_size_t) const num_of_blocks )
{
    NS(block_size_t) align = NS(Blocks_get_data_alignment)( blocks );
    NS(block_size_t) data_capacity = 
        4u * align + num_of_blocks * sizeof( NS(BlockInfo) );
        
    return data_capacity;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE NS(block_size_t) NS(Blocks_get_total_num_bytes)(
    const NS(Blocks) *const SIXTRL_RESTRICT blocks )
{
    return ( ( NS(Blocks_are_serialized)( blocks ) ) &&
             ( blocks->ptr_total_num_bytes != 0 ) )
          ? *( blocks->ptr_total_num_bytes ) : ( NS(block_size_t) )0u;    
}

SIXTRL_INLINE int NS(Blocks_remap)( NS(Blocks)* SIXTRL_RESTRICT blocks )
{
    ( void )blocks;
    return -1;
}

SIXTRL_INLINE int NS(Blocks_unserialize)( 
    NS(Blocks)* SIXTRL_RESTRICT blocks,
    SIXTRL_GLOBAL_DEC unsigned char* SIXTRL_RESTRICT data_mem_begin )
{
    int success = -1;
    
    if( ( blocks != 0 ) && ( data_mem_begin != 0 ) )
    {
        typedef SIXTRL_GLOBAL_DEC unsigned char*    g_uchar_ptr_t;
        typedef SIXTRL_GLOBAL_DEC unsigned char**   g_ptr_uchar_ptr_t;
        
        typedef SIXTRL_GLOBAL_DEC void*             g_void_ptr_t;
        typedef SIXTRL_GLOBAL_DEC void**            g_ptr_void_ptr_t;
        typedef SIXTRL_GLOBAL_DEC void***           g_ptr2_void_ptr_t;
        typedef SIXTRL_GLOBAL_DEC void****          g_ptr3_void_ptr_t;
        
        typedef SIXTRL_GLOBAL_DEC NS(BlockInfo)     g_info_t;
        typedef SIXTRL_GLOBAL_DEC NS(BlockInfo)*    g_info_ptr_t;
        typedef SIXTRL_GLOBAL_DEC NS(BlockInfo)**   g_ptr_info_ptr_t;
        
        typedef SIXTRL_GLOBAL_DEC SIXTRL_UINT64_T*  g_u64_ptr_t;
        
        NS(block_size_t) const align = NS(Blocks_get_data_alignment)( blocks );
        NS(block_size_t) num_of_blocks = 0u;        
        NS(block_size_t) num_of_data_ptrs = 0u;
        
        intptr_t offset = 0;
        
        g_ptr_uchar_ptr_t begin = ( g_ptr_uchar_ptr_t )data_mem_begin;
        g_uchar_ptr_t data_mem_end = 0;        
        g_ptr_info_ptr_t ptr_block_infos = 0;
        g_ptr3_void_ptr_t ptr_data_ptrs = 0;
        g_u64_ptr_t ptr_total_num_bytes = 0;
        
        SIXTRL_ASSERT( ( begin != 0 ) && ( align != 0u ) );
        
        SIXTRL_ASSERT( align == sizeof( g_u64_ptr_t ) );
        SIXTRL_ASSERT( align == sizeof( g_uchar_ptr_t ) );
        SIXTRL_ASSERT( align == sizeof( g_ptr_uchar_ptr_t ) );
        SIXTRL_ASSERT( align == sizeof( g_void_ptr_t ) );
        SIXTRL_ASSERT( align == sizeof( g_ptr_void_ptr_t ) );
        SIXTRL_ASSERT( align == sizeof( g_ptr2_void_ptr_t ) );
        SIXTRL_ASSERT( align == sizeof( g_info_ptr_t ) );
        SIXTRL_ASSERT( align == sizeof( g_ptr_info_ptr_t ) );
        
        SIXTRL_ASSERT( (   sizeof( g_info_t ) >= align ) &&
                       ( ( sizeof( g_info_t ) %  align ) == 0u ) );
                
        SIXTRL_ASSERT( ( ( ( uintptr_t ) begin ) % align ) == 0u );
        SIXTRL_ASSERT( ( ( ( uintptr_t )*begin ) % align ) == 0u );
        
        ptr_block_infos = ( g_ptr_info_ptr_t  )( data_mem_begin + align );
        ptr_data_ptrs   = ( g_ptr3_void_ptr_t )( data_mem_begin + align * 2u );
        ptr_total_num_bytes = ( g_u64_ptr_t   )( data_mem_begin + align * 3u );
        
        SIXTRL_ASSERT( *ptr_total_num_bytes >= 4u * align );
                
        data_mem_end = data_mem_begin + *ptr_total_num_bytes;
        
        offset = ( ( intptr_t )begin ) - ( ( intptr_t )*begin );
        
        if( *ptr_block_infos != 0 )
        {
            ptrdiff_t dist = 0;
                
            if( *ptr_data_ptrs != 0 )
            {
                dist = ( ptrdiff_t )( 
                    ( ( g_uchar_ptr_t )*ptr_data_ptrs ) - 
                    ( ( g_uchar_ptr_t )*ptr_block_infos ) );
            }
            else
            {
                SIXTRL_ASSERT( ( offset > 0 ) || 
                    ( ( ( intptr_t )*ptr_block_infos ) > offset ) );
                
                dist = ( ptrdiff_t )( data_mem_end - 
                    ( ( ( g_uchar_ptr_t )*ptr_block_infos ) + offset ) );
            }
            
            num_of_blocks = ( ( dist > 0 ) && 
                ( 0 == ( dist % align ) ) &&
                ( 0 == ( dist % sizeof( NS(BlockInfo) ) ) ) )
                    ?  ( dist / sizeof( NS(BlockInfo) ) ) : ( 0u );
        }
        
        if( *ptr_data_ptrs != 0 )
        {
            SIXTRL_ASSERT( ( offset > 0 ) || 
                ( ( intptr_t )ptr_data_ptrs ) > offset );
            
            ptrdiff_t const dist = ( ptrdiff_t )( data_mem_end - 
                ( ( ( g_uchar_ptr_t )*ptr_data_ptrs ) + offset ) );
            
            num_of_data_ptrs = ( ( dist > 0 ) &&
                ( 0 == ( dist % align ) ) &&
                ( 0 == ( dist % sizeof( g_ptr_void_ptr_t ) ) ) )
                    ? (  dist / sizeof( g_ptr_void_ptr_t ) ) : ( 0u );
        }
        
        if( offset != 0 )
        {
            *begin = *begin + offset;
            
            SIXTRL_ASSERT( ( ( uintptr_t )*begin ) == 
                           ( ( uintptr_t )begin  ) );
            
            if( *ptr_block_infos != 0 )
            {
                g_info_ptr_t info_it = 0;
                NS(block_size_t) ii = 0;
                
                SIXTRL_ASSERT( ( offset > 0 ) ||
                    ( ( ( intptr_t )*ptr_block_infos ) > -offset ) );
                
                *ptr_block_infos = ( g_info_ptr_t )( 
                    ( ( g_uchar_ptr_t )*ptr_block_infos ) + offset );
                
                info_it = *ptr_block_infos;
                
                for( ; ii < num_of_blocks ; ++ii, ++info_it )
                {
                    g_void_ptr_t  ptr_begin    = 0;
                    g_void_ptr_t  ptr_metadata = 0;
                    
                    ptr_begin    = NS(BlockInfo_get_ptr_begin)( info_it );
                    
                    SIXTRL_ASSERT( ( ptr_begin != 0 ) && ( ( offset > 0 ) || 
                        ( ( ( intptr_t )ptr_begin ) > -offset ) ) );
                    
                    ptr_begin = ( g_void_ptr_t )( 
                        ( ( g_uchar_ptr_t )ptr_begin ) + offset );
                    
                    NS(BlockInfo_set_ptr_begin)( info_it, ptr_begin );
                    
                    ptr_metadata = NS(BlockInfo_get_ptr_metadata)( info_it );
                    
                    if( ptr_metadata != 0 )
                    {
                        SIXTRL_ASSERT( ( ptr_metadata != 0 ) && 
                            ( ( offset > 0 ) || 
                            ( ( ( intptr_t )ptr_metadata ) > offset ) ) );
                        
                        ptr_metadata = ( g_void_ptr_t )(
                            ( ( g_uchar_ptr_t )ptr_metadata ) + offset );
                        
                        NS(BlockInfo_set_ptr_metadata)( 
                            info_it, ptr_metadata );
                    }
                }
            }
            
            if( *ptr_data_ptrs != 0 )
            {                
                NS(block_size_t) ii = 0;                
                g_ptr2_void_ptr_t ptr_to_data_pointers_begin = 0;
                
                SIXTRL_ASSERT( ( offset > 0 ) ||
                    ( ( ( intptr_t )*ptr_data_ptrs ) > -offset ) );
                
                ptr_to_data_pointers_begin = ( g_ptr2_void_ptr_t 
                    )( ( ( g_uchar_ptr_t )*ptr_data_ptrs ) + offset );
                
                for( ; ii < num_of_data_ptrs ; ++ii )
                {
                    g_ptr_void_ptr_t ptr_to_data_ptr = 0;
                    g_void_ptr_t     ptr_to_data     = 0;
                    
                    SIXTRL_ASSERT( ( offset > 0 ) || ( -offset <
                        ( ( intptr_t )ptr_to_data_pointers_begin[ ii ] ) ) );
                    
                    ptr_to_data_ptr = ( g_ptr_void_ptr_t )( offset +
                        ( ( g_uchar_ptr_t )ptr_to_data_pointers_begin[ ii ] ) );
                    
                    SIXTRL_ASSERT( ( offset > 0 ) || ( -offset <
                        ( ( intptr_t )*ptr_to_data_ptr ) ) );
                    
                    ptr_to_data = ( g_void_ptr_t )( offset +
                        ( ( g_uchar_ptr_t )*ptr_to_data_ptr ) );
                    
                    *ptr_to_data_ptr = ptr_to_data;
                    ptr_to_data_pointers_begin[ ii ] = ptr_to_data_ptr;
                }
                
                *ptr_data_ptrs = ptr_to_data_pointers_begin;
            }
        }
        
        blocks->ptr_data_begin          = *begin;
        blocks->ptr_block_infos         = *ptr_block_infos;
        blocks->ptr_data_pointers_begin = *ptr_data_ptrs;
        blocks->ptr_total_num_bytes     = ptr_total_num_bytes;
        
        blocks->num_blocks              = num_of_blocks;            
        blocks->num_data_pointers       = num_of_data_ptrs;
        blocks->data_size               = *ptr_total_num_bytes;
                                        
        blocks->data_store              = 0;
        blocks->index_store             = 0;
        blocks->data_ptrs_store         = 0;            
        blocks->is_serialized            = 1;

        success = 0;
    }
        
    return success;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE NS(block_size_t) NS(Blocks_get_max_num_of_blocks)( 
    const NS(Blocks) *const SIXTRL_RESTRICT blocks )
{
    NS(block_size_t) max_num_blocks = ( NS(block_size_t) )0u;
    
    if( NS(Blocks_are_serialized)( blocks ) )
    {
        max_num_blocks = blocks->num_blocks;
    }
    #if !defined( _GPUCODE )
    else
    {
        max_num_blocks = NS(MemPool_get_capacity)( 
            ( NS(MemPool)* )blocks->index_store ) / sizeof( NS(BlockInfo) );
    }
    #endif /* !defined( _GPUCODE ) */
    
    return max_num_blocks; 
}

SIXTRL_INLINE NS(block_size_t) NS(Blocks_get_num_of_blocks)(
    const NS(Blocks) *const SIXTRL_RESTRICT blocks )
{
    NS(block_size_t) num_of_blocks = 0u;
    
    if( NS(Blocks_are_serialized)( blocks ) ) 
    {
        num_of_blocks = blocks->num_blocks;                
    }    
    #if !defined( _GPUCODE )
    else if( NS(Blocks_has_index_store)( blocks ) )
    {
        num_of_blocks = NS(MemPool_get_size)( 
            ( NS(MemPool)* )blocks->index_store ) / sizeof( NS(BlockInfo) );
    }
    #endif /* defined( _GPUCODE ) */
    
    return num_of_blocks;
}

SIXTRL_INLINE NS(block_size_t) NS(Blocks_get_blocks_write_capacity)(
    const NS(Blocks) *const SIXTRL_RESTRICT blocks )
{
    #if !defined( _GPUCODE )
    if( !NS(Blocks_are_serialized)( blocks ) )
    {
        NS(MemPool)* pool = ( NS(MemPool)* )blocks->index_store;
        
        NS(block_size_t) const capacity = NS(MemPool_get_capacity)( pool );
        NS(block_size_t) const size     = NS(MemPool_get_size)( pool );
        
        NS(block_size_t) const max_num_blocks = 
            capacity / sizeof( NS(BlockInfo ) );
            
        NS(block_size_t) const current_num_blocks =
            size / sizeof( NS(BlockInfo) );
                
        return ( current_num_blocks <= max_num_blocks )
            ? max_num_blocks - current_num_blocks : 0u;
    }
    #endif /* !defined( _GPUCODE ) */
    
    return 0u;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE NS(block_size_t) NS(Blocks_get_data_capacity)(
    const NS(Blocks) *const SIXTRL_RESTRICT blocks )
{
    NS(block_size_t) data_capacity = ( NS(block_size_t) )0u;
    
    if( NS(Blocks_are_serialized)( blocks ) )
    {
        data_capacity = blocks->data_size;
    }
    #if !defined( _GPUCODE )
    else
    {
        NS(MemPool)* pool = ( NS(MemPool)* )blocks->data_store;
        data_capacity = NS(MemPool_get_capacity)( pool );
    }
    #endif /* !defined( _GPUCODE ) */
    
    return data_capacity;
}

SIXTRL_INLINE NS(block_size_t) NS(Blocks_get_data_size)(
    const NS(Blocks) *const SIXTRL_RESTRICT blocks )
{
    NS(block_size_t) data_size = ( NS(block_size_t) )0u;
    
    if( NS(Blocks_are_serialized)( blocks ) )
    {
        data_size = blocks->data_size;
    }
    #if !defined( _GPUCODE )
    else
    {
        NS(MemPool)* pool = ( NS(MemPool)* )blocks->data_store;
        data_size = NS(MemPool_get_size)( pool );
    }
    #endif /* !defined( _GPUCODE ) */
    
    return data_size;
}

SIXTRL_INLINE NS(block_size_t) NS(Blocks_get_data_write_capacity)(
    const NS(Blocks) *const SIXTRL_RESTRICT blocks )
{
    #if !defined( _GPUCODE )
    if( !NS(Blocks_are_serialized)( blocks ) )
    {
        NS(MemPool)* pool = ( NS(MemPool)* )blocks->data_store;
        return NS(MemPool_get_remaining_bytes)( pool );
    }
    #endif /* !defined( _GPUCODE ) */
    
    return 0u;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE NS(block_size_t) NS(Blocks_get_max_num_data_pointers)(
    const NS(Blocks) *const SIXTRL_RESTRICT blocks )
{
    NS(block_size_t) max_num_data_pointers = ( NS(block_size_t) )0u;
    
    if( NS(Blocks_are_serialized)( blocks ) )
    {
        max_num_data_pointers = blocks->num_data_pointers;
    }
    #if !defined( _GPUCODE )
    else
    {
        max_num_data_pointers = NS(MemPool_get_capacity)(
            ( NS(MemPool)* )blocks->data_ptrs_store ) / 
                sizeof( SIXTRL_GLOBAL_DEC void** );
    }
    #endif /* !defined( _GPUCODE ) */
    
    return max_num_data_pointers;
}

SIXTRL_INLINE NS(block_size_t) NS(Blocks_get_num_data_pointers)(
    const NS(Blocks) *const SIXTRL_RESTRICT blocks )
{
    NS(block_size_t) num_of_data_pointers = 0u;
    
    if( NS(Blocks_are_serialized)( blocks ) )
    {
        num_of_data_pointers = blocks->num_data_pointers;
    }
    #if !defined( _GPUCODE )
    else
    {
        num_of_data_pointers = NS(MemPool_get_size)(
            ( NS(MemPool)* )blocks->data_ptrs_store ) / 
                sizeof( SIXTRL_GLOBAL_DEC void** );
    }
    #endif /* !defined( _GPUCODE ) */
    
    return num_of_data_pointers;
}

SIXTRL_INLINE NS(block_size_t) NS(Blocks_get_data_pointers_write_capacity)(
    const NS(Blocks) *const SIXTRL_RESTRICT blocks )
{
    #if !defined( _GPUCODE )
    if( !NS(Blocks_are_serialized)( blocks ) )
    {
        typedef SIXTRL_GLOBAL_DEC void** g_ptr_void_ptr_t;
        
        NS(MemPool)* pool = ( NS(MemPool)* )blocks->index_store;        
        
        NS(block_size_t) const max_num_data_ptrs = 
            NS(MemPool_get_capacity)( pool ) / sizeof( g_ptr_void_ptr_t );
        
        NS(block_size_t) const current_num_data_ptrs =
            NS(MemPool_get_size)( pool ) / sizeof( g_ptr_void_ptr_t );
                
        return ( current_num_data_ptrs <= max_num_data_ptrs )
            ? max_num_data_ptrs - current_num_data_ptrs : 0u;
    }
    #endif /* !defined( _GPUCODE ) */
    
    return 0u;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE void NS(Blocks_set_data_alignment)( 
    NS(Blocks)* SIXTRL_RESTRICT blocks, NS(block_size_t) const alignment )
{
    static NS(block_size_t) const ZERO = ( NS(block_size_t) )0u;
    
    SIXTRL_ASSERT( NS(Blocks_get_num_of_blocks)( blocks ) == ZERO );
    
    if( ( blocks != 0 ) && ( alignment > ZERO ) )
    {
        blocks->alignment = alignment;
    }
    
    return;
}
                
SIXTRL_INLINE NS(block_size_t) NS(Blocks_get_data_alignment)( 
    const NS(Blocks) *const SIXTRL_RESTRICT blocks )
{
    return ( blocks != 0 ) ? blocks->alignment : ( NS(block_size_t) )8u; 
}

SIXTRL_INLINE void NS(Blocks_set_begin_alignment)(
    NS(Blocks)* SIXTRL_RESTRICT blocks, NS(block_size_t) const alignment )
{
    static NS(block_size_t) const ZERO = ( NS(block_size_t) )0u;
    
    SIXTRL_ASSERT( NS(Blocks_get_num_of_blocks)( blocks ) == ZERO );
    
    if( ( blocks != 0 ) && ( alignment > ZERO ) )
    {
        blocks->begin_alignment = alignment;
    }
    
    return;
}

SIXTRL_INLINE NS(block_size_t) NS(Blocks_get_begin_alignment)(
    const NS(Blocks) *const SIXTRL_RESTRICT blocks )
{
    return ( blocks != 0 ) ? blocks->begin_alignment : ( NS(block_size_t) )8u;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE SIXTRL_GLOBAL_DEC NS(BlockInfo) const*
NS(Blocks_get_const_block_infos_begin)( 
    const NS(Blocks) *const SIXTRL_RESTRICT blocks )
{
    return ( blocks != 0 ) ? blocks->ptr_block_infos : 0;
}

SIXTRL_INLINE SIXTRL_GLOBAL_DEC NS(BlockInfo) const* 
NS(Blocks_get_const_block_infos_end)( 
    const NS(Blocks) *const SIXTRL_RESTRICT blocks )
{
    SIXTRL_GLOBAL_DEC NS(BlockInfo) const* end_ptr = 
        NS(Blocks_get_const_block_infos_begin)( blocks );
        
    if( end_ptr != 0 )
    {
        end_ptr = end_ptr + NS(Blocks_get_num_of_blocks)( blocks );
    }
    
    return end_ptr;
}

SIXTRL_INLINE SIXTRL_GLOBAL_DEC NS(BlockInfo) const* 
NS(Blocks_get_const_block_info_by_index)(
    const NS(Blocks) *const SIXTRL_RESTRICT blocks, 
    NS(block_size_t) const index )
{
    SIXTRL_GLOBAL_DEC NS(BlockInfo) const* ptr_to_block_info = 
        NS(Blocks_get_const_block_infos_begin)( blocks );
        
    if( ptr_to_block_info != 0 )
    {
        SIXTRL_ASSERT( index < NS(Blocks_get_num_of_blocks)( blocks ) );
        ptr_to_block_info = ptr_to_block_info + index;
    }
    
    return ptr_to_block_info;
}


SIXTRL_INLINE SIXTRL_GLOBAL_DEC NS(BlockInfo)*
NS(Blocks_get_block_infos_begin)( NS(Blocks)* SIXTRL_RESTRICT blocks )
{
    return ( NS(BlockInfo)* )NS(Blocks_get_const_block_infos_begin)( blocks );
}

SIXTRL_INLINE SIXTRL_GLOBAL_DEC NS(BlockInfo)* 
NS(Blocks_get_block_infos_end)( NS(Blocks)* SIXTRL_RESTRICT blocks )
{
    return ( NS(BlockInfo)* )NS(Blocks_get_const_block_infos_end)( blocks );
}

SIXTRL_INLINE SIXTRL_GLOBAL_DEC NS(BlockInfo)* 
NS(Blocks_get_block_info_by_index)( 
    NS(Blocks)* SIXTRL_RESTRICT blocks, NS(block_size_t) const index )
{
    return ( NS(BlockInfo)* 
        )NS(Blocks_get_const_block_info_by_index)( blocks, index );
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE int NS(Blocks_has_data_store)(
    const NS(Blocks) *const SIXTRL_RESTRICT blocks )
{
    return ( blocks != 0 ) ? ( blocks->data_store != 0 ) : 0;
}

SIXTRL_INLINE int NS(Blocks_has_index_store)(
    const NS(Blocks) *const SIXTRL_RESTRICT blocks )
{
    return ( blocks != 0 ) ? ( blocks->index_store != 0 ) : 0;
}

SIXTRL_INLINE int NS(Blocks_has_data_pointers_store)(
    const NS(Blocks) *const SIXTRL_RESTRICT blocks )
{
    return ( blocks != 0 ) ? ( blocks->data_ptrs_store != 0 ) : 0;
}

/* ------------------------------------------------------------------------- */

#if !defined( _GPUCODE )
    
#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* !defined( _GPUCODE ) */

#endif /* SIXTRACKLIB_COMMON_BLOCKS_H__ */

/* end: sixtracklib/common/blocks.h */
