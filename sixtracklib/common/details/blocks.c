#include "sixtracklib/common/blocks.h"

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "sixtracklib/common/mem_pool.h"

extern int NS(Blocks_init)( NS(Blocks)* SIXTRL_RESTRICT blocks,
    NS(block_size_t) max_num_blocks, NS(block_size_t) const data_capacity );

extern int NS(Blocks_init_with_same_structure_as)(
    NS(Blocks)* SIXTRL_RESTRICT blocks, 
    const NS(Blocks) *const SIXTRL_RESTRICT ref_blocks );

extern int NS(Blocks_init_from_serialized_data)(
    NS(Blocks)* SIXTRL_RESTRICT blocks, 
    SIXTRL_GLOBAL_DEC unsigned char const* SIXTRL_RESTRICT data_mem_begin,
    NS(block_size_t) const total_num_of_bytes );

extern void NS(Blocks_clear)( NS(Blocks)* SIXTRL_RESTRICT blocks );
extern void NS(Blocks_free)(  NS(Blocks)* SIXTRL_RESTRICT blocks );

extern NS(BlockInfo)* NS(Blocks_add_block)( 
    NS(Blocks)* SIXTRL_RESTRICT blocks, NS(BlockType) const type_id,
    NS(block_size_t) const block_handle_size,
    const void *const SIXTRL_RESTRICT block_handle,
    NS(block_size_t) const num_attr_data_pointers, 
    const NS(block_size_t) *const SIXTRL_RESTRICT attr_data_pointer_offsets,
    const NS(block_size_t) *const SIXTRL_RESTRICT attr_type_sizes,
    const NS(block_size_t) *const SIXTRL_RESTRICT attr_type_counts );

extern int NS(Blocks_serialize)( NS(Blocks)* SIXTRL_RESTRICT blocks );

/* ------------------------------------------------------------------------- */

//SIXTRL_STATIC NS(block_size_t) const DEFAULT_ALIGN = ( NS(block_size_t) )8u;
SIXTRL_STATIC NS(block_size_t) const ZERO_SIZE     = ( NS(block_size_t) )0u;

SIXTRL_STATIC NS(block_size_t) const MAX_DATA_PTRS_PER_BLOCK = 
    ( NS(block_size_t) )20u;

    
int NS(Blocks_init)( NS(Blocks)* SIXTRL_RESTRICT blocks,
    NS(block_size_t) max_num_blocks, NS(block_size_t) const data_capacity )
{
    int success = -1;
    
    if( ( blocks != 0 ) && ( max_num_blocks > ZERO_SIZE ) && 
        ( data_capacity > ZERO_SIZE ) )
    {
        NS(block_size_t) begin_align = NS(Blocks_get_begin_alignment)( blocks );
        NS(block_size_t) align = NS(Blocks_get_data_alignment)( blocks );
        
        if( ( begin_align != ZERO_SIZE ) && 
            ( align != ZERO_SIZE ) && ( begin_align >= align ) &&
            ( ( begin_align % align ) == ZERO_SIZE ) )
        {
            success = 0;
        }
        else if( align != ZERO_SIZE )
        {
            begin_align = align;
            success = 0;
        }
        else if( begin_align != ZERO_SIZE )
        {
            align = begin_align;
            success = 0;
        }
        
        if( success == 0 )
        {
            NS(block_size_t) const info_store_capacity = 
                sizeof( NS(BlockInfo) ) * max_num_blocks;
            
            NS(block_size_t) const data_ptrs_capacity =
                sizeof( SIXTRL_GLOBAL_DEC void** ) * 
                    max_num_blocks *  MAX_DATA_PTRS_PER_BLOCK;
                
            if( blocks->data_store == 0 )
            {
                blocks->data_store = NS(MemPool_preset)( 
                    ( NS(MemPool)* )malloc( sizeof( NS(MemPool) ) ) );
                
                NS(MemPool_init_aligned)( 
                    blocks->data_store, data_capacity, align, begin_align );
            }
            else
            {
                NS(MemPool_clear)( blocks->data_store );
                success = ( NS(MemPool_reserve_aligned)( 
                    blocks->data_store, data_capacity, begin_align ) )
                        ? 0 : -1;
            }
            
            if( blocks->index_store == 0 )
            {
                blocks->index_store = NS(MemPool_preset)(
                    ( NS(MemPool)* )malloc( sizeof( NS(MemPool ) ) ) );
                
                NS(MemPool_init_aligned)( blocks->index_store, 
                    info_store_capacity, align, begin_align );
            }
            else
            {
                NS(MemPool_clear)( blocks->index_store );
                success = ( NS(MemPool_reserve_aligned)(
                    blocks->index_store, info_store_capacity, begin_align ) )
                        ? 0 : -1;
            }
            
            if( blocks->data_ptrs_store == 0 )
            {
                blocks->data_ptrs_store = NS(MemPool_preset)(
                    ( NS(MemPool)* )malloc( sizeof( NS(MemPool) ) ) );
                
                NS(MemPool_init_aligned)( blocks->data_ptrs_store, 
                    data_ptrs_capacity, align, begin_align );
            }
            else
            {
                NS(MemPool_clear)( blocks->data_ptrs_store );
                success = ( NS(MemPool_reserve_aligned)(
                    blocks->data_ptrs_store, data_ptrs_capacity, begin_align ) )
                        ? 0 : -1;
            }
        }
        
        if( success == 0 )
        {
            SIXTRL_ASSERT( 
                ( blocks->data_store      != 0 ) && 
                ( blocks->index_store     != 0 ) &&
                ( blocks->data_ptrs_store != 0 ) );
            
            blocks->begin_alignment         = begin_align;
            blocks->alignment               = align;
            
            NS(Blocks_clear)( blocks );
        }
        else
        {
            NS(Blocks_free)( blocks );
        }
    }
    
    return success;
}

int NS(Blocks_init_with_same_structure_as)(
    NS(Blocks)* SIXTRL_RESTRICT blocks, 
    const NS(Blocks) *const SIXTRL_RESTRICT ref_blocks )
{
    ( void )blocks;
    ( void )ref_blocks;

    return -1;
}

int NS(Blocks_init_from_serialized_data)(
    NS(Blocks)* SIXTRL_RESTRICT blocks, 
    SIXTRL_GLOBAL_DEC unsigned char const* SIXTRL_RESTRICT data_mem_begin,
    NS(block_size_t) const total_num_of_bytes )
{
    int success = -1;
    
    if( ( blocks != 0 ) && ( data_mem_begin != 0 ) &&
        ( total_num_of_bytes > 0u ) &&
        ( !NS(Blocks_are_serialized)( blocks ) ) &&
        ( !NS(Blocks_has_data_pointers_store)( blocks ) ) &&
        ( !NS(Blocks_has_index_store)( blocks ) ) &&
        ( !NS(Blocks_has_data_pointers_store)( blocks ) ) )
    {
        NS(block_size_t) const begin_align = 
            NS(Blocks_get_begin_alignment)( blocks );
        
        NS(block_size_t) const align = 
            NS(Blocks_get_data_alignment)( blocks );
        
        NS(MemPool)* ptr_data_store = NS(MemPool_preset)(
            ( NS(MemPool)* )malloc( sizeof( NS(MemPool) ) ) );
            
        if( ptr_data_store != 0 )
        {
            NS(AllocResult) result;
            unsigned char* serialized_data_begin = 0;
            
            NS(MemPool_init_aligned)( ptr_data_store, total_num_of_bytes, 
                                      align, begin_align );
            
            result = NS(MemPool_append_aligned)( 
                ptr_data_store, total_num_of_bytes, align );
            
            if( NS(AllocResult_valid)( &result ) )
            {
                serialized_data_begin = 
                    ( unsigned char* )NS(AllocResult_get_pointer)( &result );
                
                memcpy( serialized_data_begin, data_mem_begin, 
                        total_num_of_bytes );                
            }
            
            if( ( serialized_data_begin != 0 ) && ( NS(Blocks_unserialize)( 
                    blocks, serialized_data_begin ) == 0 ) )
            {
                blocks->data_store = ( void* )ptr_data_store;                
                success = 0;
            }
        }
        
        if( success != 0 )
        {
            NS(MemPool_free)( ptr_data_store );
            free( ptr_data_store );
            ptr_data_store = 0;
        }
    }
    
    return success;
}

void NS(Blocks_clear)( NS(Blocks)* SIXTRL_RESTRICT blocks )
{
    if( blocks != 0 )
    {
        int success = 0;
        
        NS(Blocks) const rollback_blocks = *blocks;
        
        NS(block_size_t) const begin_align = 
            NS(Blocks_get_begin_alignment)( blocks );
        
        NS(block_size_t) const align = 
            NS(Blocks_get_data_alignment)( blocks );
            
        SIXTRL_ASSERT( align != 0 );
        SIXTRL_ASSERT( (   begin_align >= align ) && 
                       ( ( begin_align %  align ) == 0u ) );
                
        blocks->ptr_data_begin          = 0;
        blocks->ptr_block_infos         = 0;
        blocks->ptr_data_pointers_begin = 0;
        blocks->ptr_total_num_bytes     = 0;
        
        blocks->data_size         = 0u;
        blocks->num_blocks        = 0u;
        blocks->num_data_pointers = 0u;
        blocks->is_serialized     = 0;
        
        if( NS(Blocks_has_data_store)( blocks ) )
        {
            typedef SIXTRL_GLOBAL_DEC unsigned char**   g_ptr_uchar_ptr_t;
            typedef SIXTRL_GLOBAL_DEC NS(BlockInfo)*    g_info_ptr_t;
            typedef SIXTRL_GLOBAL_DEC void***           g_ptr2_void_ptr_t;
            
            NS(block_size_t) const MIN_DATA_SIZE = 
                sizeof( g_ptr_uchar_ptr_t ) + sizeof( g_info_ptr_t ) +
                sizeof( g_ptr2_void_ptr_t ) + sizeof( SIXTRL_UINT64_T );
            
            NS(MemPool)* data_store = ( NS(MemPool)* )blocks->data_store;            
            NS(MemPool_clear_to_aligned_position)( data_store, begin_align );
                        
            SIXTRL_ASSERT( 
                ( align   == NS(MemPool_get_chunk_size)( data_store ) ) ||
                ( (   align >= NS(MemPool_get_chunk_size)( data_store ) ) &&
                  ( ( align %  NS(MemPool_get_chunk_size)( data_store ) ) 
                    == 0u ) ) );
                
            SIXTRL_ASSERT( align == sizeof( g_ptr_uchar_ptr_t ) );
            SIXTRL_ASSERT( align == sizeof( g_ptr2_void_ptr_t ) );
            SIXTRL_ASSERT( align == sizeof( g_info_ptr_t      ) );
            SIXTRL_ASSERT( align == sizeof( SIXTRL_UINT64_T   ) );
            
            if( ( data_store != 0 ) && ( MIN_DATA_SIZE <= 
                    NS(MemPool_get_remaining_bytes)( data_store ) ) )
            {
                NS(MemPool) const rollback_data_store = *data_store;
                
                NS(AllocResult) result = NS(MemPool_append_aligned)( 
                    data_store, MIN_DATA_SIZE, align );
                
                if( NS(AllocResult_valid)( &result ) )
                {
                    memset( ( void* )NS(AllocResult_get_pointer)( &result ), 
                            ( int )0, NS(AllocResult_get_length)( &result ) );
                    
                    success = 0;
                }
                
                if( success != 0 )
                {
                    *data_store = rollback_data_store;                    
                }
            }
        }
        
        if( NS(Blocks_has_index_store)( blocks ) )
        {
            NS(MemPool)* ptr_store = ( NS(MemPool)* )blocks->index_store;
            
            SIXTRL_ASSERT( 
                ( align   == NS(MemPool_get_chunk_size)( ptr_store ) ) ||
                ( (   align >= NS(MemPool_get_chunk_size)( ptr_store ) ) &&
                  ( ( align %  NS(MemPool_get_chunk_size)( ptr_store ) ) 
                    == 0u ) ) );
                
            NS(MemPool_clear_to_aligned_position)( ptr_store, begin_align );
        }
        
        if( NS(Blocks_has_data_pointers_store)( blocks ) )
        {
            NS(MemPool)* ptr_store = ( NS(MemPool)* )blocks->data_ptrs_store;
            
            SIXTRL_ASSERT( 
                ( align   == NS(MemPool_get_chunk_size)( ptr_store ) ) ||
                ( (   align >= NS(MemPool_get_chunk_size)( ptr_store ) ) &&
                  ( ( align %  NS(MemPool_get_chunk_size)( ptr_store ) ) 
                    == 0u ) ) );
                
            NS(MemPool_clear_to_aligned_position)( ptr_store, begin_align );
        }
        
        if( success != 0 )
        {
            *blocks = rollback_blocks;
        }
    }
    
    return;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

void NS(Blocks_free)(  NS(Blocks)* SIXTRL_RESTRICT blocks )
{
    if( blocks != 0 )
    {
        NS(MemPool_free)( ( NS(MemPool)* )blocks->data_store );
        NS(MemPool_free)( ( NS(MemPool)* )blocks->index_store );
        NS(MemPool_free)( ( NS(MemPool)* )blocks->data_ptrs_store );
        
        free( blocks->data_store );
        free( blocks->index_store );
        free( blocks->data_ptrs_store );
        
        NS(Blocks_preset)( blocks );
    }
    
    return;
}


/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

NS(BlockInfo)* NS(Blocks_add_block)( 
    NS(Blocks)* SIXTRL_RESTRICT blocks, NS(BlockType) const type_id,
    NS(block_size_t) const block_handle_size,
    const void *const SIXTRL_RESTRICT block_handle,
    NS(block_size_t) const num_attr_data_pointers, 
    const NS(block_size_t) *const SIXTRL_RESTRICT attr_data_pointer_offsets,
    const NS(block_size_t) *const SIXTRL_RESTRICT attr_type_sizes,
    const NS(block_size_t) *const SIXTRL_RESTRICT attr_type_counts )
{
    SIXTRL_GLOBAL_DEC NS(BlockInfo)* result_block_infos = 0;
    
    SIXTRL_STATIC NS(block_size_t) const ZERO = ( NS(block_size_t) )0u;    
    NS(block_size_t) const alignment = NS(Blocks_get_data_alignment)( blocks );    
    
    if( ( blocks != 0 ) && ( type_id != NS(BLOCK_TYPE_INVALID) ) &&
        ( block_handle != 0 ) && ( block_handle_size > ZERO ) &&
        ( alignment != ZERO ) &&         
        ( ZERO == ( ( ( uintptr_t )block_handle ) % alignment ) ) &&
        ( !NS(Blocks_are_serialized)( blocks ) ) &&
        ( NS(Blocks_has_data_store)(  blocks ) ) &&
        ( NS(Blocks_has_index_store)( blocks ) ) &&
        ( NS(Blocks_has_data_pointers_store)( blocks ) ) )
    {
        typedef SIXTRL_GLOBAL_DEC unsigned char*    g_uchar_ptr_t;
        typedef SIXTRL_GLOBAL_DEC NS(BlockInfo)     g_info_t;
        typedef SIXTRL_GLOBAL_DEC NS(BlockInfo)*    g_info_ptr_t;
        typedef SIXTRL_GLOBAL_DEC void*             g_void_ptr_t;
        typedef SIXTRL_GLOBAL_DEC void**            g_ptr_void_ptr_t;
        typedef SIXTRL_GLOBAL_DEC void***           g_ptr2_void_ptr_t;
        typedef NS(MemPool)                         pool_t;
                
        NS(block_size_t) block_length = ZERO;
        
        g_uchar_ptr_t ptr_block_begin = 0;
        g_info_ptr_t  ptr_block_info  = 0;
        g_ptr2_void_ptr_t attr_data_ptrs = 0;
        
        pool_t* ptr_data_store           = ( pool_t* )blocks->data_store;
        pool_t* ptr_index_store          = ( pool_t* )blocks->index_store;
        pool_t* ptr_data_ptrs_store      = ( pool_t* )blocks->data_ptrs_store;
        
        pool_t const rollback_data_store      = *ptr_data_store;            
        pool_t const rollback_index_store     = *ptr_index_store;
        pool_t const rollback_data_ptrs_store = *ptr_data_ptrs_store;
        
        int success = -1;
        
        NS(AllocResult) result = NS(MemPool_append_aligned)( 
            ptr_data_store, block_handle_size, alignment );
        
        if( NS(AllocResult_valid)( &result ) )
        {
            ptr_block_begin = 
                ( g_uchar_ptr_t )NS(AllocResult_get_pointer)( &result );
            
            memcpy( ptr_block_begin, block_handle, block_handle_size );  
            
            block_length += NS(AllocResult_get_length)( &result );
            success = 0;
        }
        
        if( success == 0 )
        {
            result = NS(MemPool_append_aligned)(
                ptr_index_store, sizeof( g_info_t ), alignment );
            
            if( NS(AllocResult_valid)( &result ) )
            {
                ptr_block_info = ( g_info_ptr_t 
                    )NS(AllocResult_get_pointer)( &result );
                
                SIXTRL_ASSERT( ptr_block_info != 0 );                
            }
            else
            {
                success = -1;
            }
        }
        
        if( ( success == 0 ) && ( num_attr_data_pointers != ZERO ) )
        {
            NS(block_size_t) attr_data_size = ZERO;
            NS(block_size_t) const attr_data_ptrs_size = 
                    sizeof( g_void_ptr_t ) * num_attr_data_pointers;
            
            SIXTRL_ASSERT( attr_data_pointer_offsets != 0 );
            SIXTRL_ASSERT( attr_type_sizes  != 0 );
            SIXTRL_ASSERT( attr_type_counts != 0 );
            
            success = -1;
            
            if( attr_data_ptrs_size > ZERO )
            {
                NS(block_size_t) offset = ZERO;
                
                result = NS(MemPool_append_aligned)( 
                    ptr_data_ptrs_store, attr_data_ptrs_size, alignment );
                
                if( NS(AllocResult_valid)( &result ) )
                {
                    NS(block_size_t) ii = ZERO;
                    
                    attr_data_ptrs = ( g_ptr2_void_ptr_t 
                        )NS(AllocResult_get_pointer)( &result );
                        
                    SIXTRL_ASSERT( attr_data_ptrs != 0 );
                    
                    success = 0;
                    
                    for( ; ii < num_attr_data_pointers; ++ii )
                    {
                        NS(block_size_t) const attr_size = 
                            attr_type_sizes[ ii ] * attr_type_counts[ ii ];
                            
                        NS(block_size_t) const attr_offset = 
                            attr_data_pointer_offsets[ ii ];
                            
                        if( ( attr_size > ZERO ) && 
                            ( ( attr_offset > offset ) || 
                              ( ( attr_offset == offset ) &&
                                ( attr_offset == 0 ) ) ) &&
                            ( attr_offset < block_handle_size ) )
                        {
                            attr_data_size += attr_size;
                            offset = attr_offset;
                            attr_data_ptrs[ ii ] = ( g_ptr_void_ptr_t )(
                               ( ptr_block_begin + attr_offset ) );
                        }
                        else
                        {
                            success = -1;
                            break;
                        }
                    }
                }
            }
            
            if( ( success == 0 ) && ( attr_data_size > ZERO ) )
            {
                NS(block_size_t) ii = ZERO;
                
                for( ; ii < num_attr_data_pointers ; ++ii )
                {
                    NS(block_size_t) const attr_size = 
                        attr_type_sizes[ ii ] * attr_type_counts[ ii ];
                    
                    result = NS(MemPool_append_aligned)( 
                        ptr_data_store, attr_size, alignment );
                    
                    if( NS(AllocResult_valid)( &result ) )
                    {
                        *attr_data_ptrs[ ii ] = ( g_void_ptr_t )(
                            NS(AllocResult_get_pointer)( &result ) );
                        
                        SIXTRL_ASSERT( *attr_data_ptrs[ ii ] != 0 );
                    }
                    else
                    {
                        success = -1;
                        break;
                    }
                }
                
                block_length += attr_data_size;
            }
            else
            {
                success = -1;
            }
        }
        
        if( success == 0 )
        {
            SIXTRL_ASSERT( ptr_block_begin != 0 );
            SIXTRL_ASSERT( ptr_block_info  != 0 );            
            SIXTRL_ASSERT( block_length >= block_handle_size );
            
            NS(BlockInfo_set_type_id)( ptr_block_info, type_id );
            NS(BlockInfo_set_block_size)( ptr_block_info, block_length );
            NS(BlockInfo_set_ptr_metadata)( ptr_block_info, 0 );
            NS(BlockInfo_set_ptr_begin)( 
                ptr_block_info, ( g_void_ptr_t )ptr_block_begin );
            
            result_block_infos = ptr_block_info;
        }
        else
        {
            *ptr_data_store      = rollback_data_store;
            *ptr_index_store     = rollback_index_store;
            *ptr_data_ptrs_store = rollback_data_ptrs_store;
        }
    }
    
    return result_block_infos;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

int NS(Blocks_serialize)( NS(Blocks)* SIXTRL_RESTRICT blocks )
{
    int success = -1;
    
    SIXTRL_STATIC NS(block_size_t) const ZERO = ( NS(block_size_t) )0u;
    
    if( (  blocks != 0 ) && 
        ( !NS(Blocks_are_serialized)( blocks ) ) &&
        (  NS(Blocks_has_data_store)( blocks ) ) &&
        (  NS(Blocks_has_index_store)( blocks ) ) &&
        (  NS(Blocks_has_data_pointers_store)( blocks ) ) &&
        (  NS(Blocks_get_num_of_blocks)( blocks ) > ZERO ) )
    {
        typedef NS(MemPool) pool_t;
        
        typedef SIXTRL_GLOBAL_DEC NS(BlockInfo)*    g_info_ptr_t;
        typedef SIXTRL_GLOBAL_DEC NS(BlockInfo)**   g_ptr_info_ptr_t;
        
        typedef SIXTRL_GLOBAL_DEC void**            g_ptr_void_ptr_t;
        typedef SIXTRL_GLOBAL_DEC void***           g_ptr2_void_ptr_t;
        
        typedef SIXTRL_GLOBAL_DEC unsigned char*    g_uchar_ptr_t;
        typedef SIXTRL_GLOBAL_DEC unsigned char**   g_ptr_uchar_ptr_t;
        typedef SIXTRL_GLOBAL_DEC SIXTRL_UINT64_T*  g_u64_ptr_t;
        
        NS(Blocks) const rollback_blocks = *blocks;
        
        pool_t* ptr_data_store      = ( pool_t* )blocks->data_store;
        pool_t* ptr_index_store     = ( pool_t* )blocks->index_store;
        pool_t* ptr_data_ptrs_store = ( pool_t* )blocks->data_ptrs_store;
        
        NS(block_size_t) num_of_blocks        = ZERO;
        NS(block_size_t) num_of_data_pointers = ZERO;
        
        NS(block_size_t) const index_size = 
            NS(MemPool_get_size)( ptr_index_store );
        
        NS(block_size_t) const data_ptrs_size =
            NS(MemPool_get_size)( ptr_data_ptrs_store );
            
        NS(block_size_t) const begin_alignment =
            NS(Blocks_get_begin_alignment)( blocks );
            
        NS(block_size_t) const align = 
            NS(Blocks_get_data_alignment)( blocks );
            
        pool_t const rollback_data_store = *ptr_data_store;
            
        g_uchar_ptr_t data_mem_begin = 
            ( g_uchar_ptr_t )NS(MemPool_get_begin_pos)( ptr_data_store );
                    
        SIXTRL_ASSERT( data_mem_begin != 0 );
        
        SIXTRL_ASSERT( align != ZERO );
        SIXTRL_ASSERT( begin_alignment >= align );
        SIXTRL_ASSERT( ( begin_alignment % align ) == ZERO );
        
        SIXTRL_ASSERT( align == sizeof( g_info_ptr_t ) );
        SIXTRL_ASSERT( align == sizeof( g_ptr_info_ptr_t ) );
        
        SIXTRL_ASSERT( align == sizeof( g_ptr_void_ptr_t ) );
        SIXTRL_ASSERT( align == sizeof( g_ptr2_void_ptr_t ) );
        
        SIXTRL_ASSERT( align == sizeof( g_uchar_ptr_t ) );
        SIXTRL_ASSERT( align == sizeof( g_ptr_uchar_ptr_t ) );
            
        if( ( index_size >= sizeof( NS(BlockInfo) ) ) &&
            ( ( data_ptrs_size >= sizeof( g_ptr_void_ptr_t ) ) ||
              ( data_ptrs_size == ZERO ) ) &&
            ( NS(MemPool_get_remaining_bytes)( ptr_data_store ) ) >=
            ( index_size + data_ptrs_size ) )
        {
            NS(AllocResult) result = NS(MemPool_append_aligned)( 
                ptr_data_store, index_size, align );
            
            num_of_blocks = index_size / sizeof( NS(BlockInfo) );
                
            if( NS(AllocResult_valid)( &result ) )
            {
                g_ptr_info_ptr_t ptr_to_block_infos = 
                    ( g_ptr_info_ptr_t )( data_mem_begin + align );
                
                g_info_ptr_t ptr_to_stored_infos = 
                    ( g_info_ptr_t )NS(AllocResult_get_pointer)( &result );
                
                memcpy( ptr_to_stored_infos, 
                        NS(MemPool_get_begin_pos)( ptr_index_store ),
                        index_size );
                    
                SIXTRL_ASSERT( ( ( uintptr_t )ptr_to_block_infos ) > align );
                *ptr_to_block_infos = ptr_to_stored_infos;
                
                blocks->ptr_block_infos = ptr_to_stored_infos;    
                blocks->num_blocks = num_of_blocks;
                
                success = 0;
            }
            
            if( success == 0 )
            {
                typedef SIXTRL_GLOBAL_DEC void**** g_ptr3_void_ptr_t;
                
                NS(block_size_t) const d = align * ( NS(block_size_t) )2u;
                
                g_ptr3_void_ptr_t ptr_to_data_ptrs_begin = 
                    ( g_ptr3_void_ptr_t )( data_mem_begin + d );
                    
                SIXTRL_ASSERT( ( ( uintptr_t )ptr_to_data_ptrs_begin ) > d );
                
                num_of_data_pointers = 
                    data_ptrs_size / sizeof( g_ptr_void_ptr_t );
                    
                if( num_of_data_pointers > ZERO )
                {
                    result = NS(MemPool_append_aligned)(
                        ptr_data_store, data_ptrs_size, align );
                    
                    if( NS(AllocResult_valid)( &result ) )
                    {
                        g_ptr2_void_ptr_t ptr_to_data_ptrs = 
                            ( g_ptr2_void_ptr_t )NS(AllocResult_get_pointer)( 
                                &result );
                        
                        memcpy( ptr_to_data_ptrs, 
                            NS(MemPool_get_begin_pos)( ptr_data_ptrs_store ),
                            data_ptrs_size );
                            
                        blocks->ptr_data_pointers_begin = ptr_to_data_ptrs;
                        *ptr_to_data_ptrs_begin         = ptr_to_data_ptrs;                            
                        blocks->num_data_pointers = num_of_data_pointers;
                    }
                    else
                    {
                        success = -1;
                    }
                }
                else
                {
                    *ptr_to_data_ptrs_begin = 0;
                    blocks->ptr_data_pointers_begin = 0;
                    blocks->num_data_pointers = ZERO;
                }
            }
            
            if( success == 0 )
            {
                g_ptr_uchar_ptr_t begin = ( g_ptr_uchar_ptr_t )data_mem_begin;
                
                g_u64_ptr_t ptr_total_num_bytes = 
                    ( g_u64_ptr_t )( data_mem_begin + align * 3u );
                
                SIXTRL_ASSERT( begin != 0 );
                SIXTRL_ASSERT( ( ( uintptr_t )ptr_total_num_bytes ) > 
                    3u * align );
                
                *begin = ( g_uchar_ptr_t 
                    )NS(MemPool_get_begin_pos)( ptr_data_store );
                
                blocks->ptr_data_begin = *begin;
                    
                 blocks->ptr_total_num_bytes = ptr_total_num_bytes;
                *blocks->ptr_total_num_bytes = 
                    NS(MemPool_get_next_begin_offset)( ptr_data_store, align );
                    
                blocks->data_size = 
                    *blocks->ptr_total_num_bytes;
                    
                NS(MemPool_clear_to_aligned_position)( 
                    ptr_index_store, begin_alignment );
                
                NS(MemPool_clear_to_aligned_position)( 
                    ptr_data_ptrs_store, begin_alignment );
                
                blocks->is_serialized = 1;
            }
            else
            {
                NS(block_size_t) const MIN_DATA_SIZE = 4u * align;
                
                if( ( data_mem_begin != 0 ) && ( align > ZERO ) )
                {
                    memset( data_mem_begin, ( int )0, MIN_DATA_SIZE );
                }
                
                *blocks = rollback_blocks;
                *( ( NS(MemPool)* )blocks->data_store ) = rollback_data_store;
            }
        }
    }
    
    return success;
}

/* sixtracklib/common/details/blocks.c */
