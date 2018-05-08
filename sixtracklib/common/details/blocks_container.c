#include "sixtracklib/common/blocks_container.h"

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "sixtracklib/_impl/definitions.h"
#include "sixtracklib/_impl/namespace_begin.h"

#include "sixtracklib/common/alignment.h"
#include "sixtracklib/common/impl/alignment_impl.h"
#include "sixtracklib/common/impl/block_info_impl.h"
#include "sixtracklib/common/mem_pool.h"

extern void NS(BlocksContainer_clear)(
    NS(BlocksContainer)* SIXTRL_RESTRICT container );

extern void NS(BlocksContainer_free)(
    NS(BlocksContainer)* SIXTRL_RESTRICT container );

extern int NS(BlocksContainer_init)(
    NS(BlocksContainer)* SIXTRL_RESTRICT container, 
    NS(block_size_t) const blocks_capacity, 
    NS(block_size_t) const data_capacity );

/* ------------------------------------------------------------------------- */

extern void NS(BlocksContainer_reserve_num_blocks)(
    NS(BlocksContainer)* SIXTRL_RESTRICT container, 
    NS(block_size_t) const new_block_capacity );

extern void NS(BlocksContainer_reserve_for_data)(
    NS(BlocksContainer)* SIXTRL_RESTRICT container, 
    NS(block_size_t) const new_data_capacity );

/* ------------------------------------------------------------------------- */

static NS(block_size_t) const NS(DEFAULT_INFO_BEGIN_ALIGNMENT) = 
    sizeof( NS(BlockInfo) );
    
static NS(block_size_t) const NS(DEFAULT_INFO_ALIGNMENT) = 
    sizeof( NS(BlockInfo) );

static NS(block_size_t) const NS(DEFAULT_DATA_BEGIN_ALIGNMENT) = 
    ( NS(block_size_t) )8u;
    
static NS(block_size_t) const NS(DEFAULT_DATA_ALIGNMENT) = 
    ( NS(block_size_t) )8u;

/* ------------------------------------------------------------------------- */
 
void NS(BlocksContainer_clear)(
    NS(BlocksContainer)* SIXTRL_RESTRICT container )
{
    if( container != 0 )
    {
        typedef NS(block_size_t)      bl_size_t;
        
        static bl_size_t const ZERO_SIZE = ( bl_size_t )0u;
        
        typedef SIXTRL_GLOBAL_DEC NS(BlockInfo)* g_ptr_info_block_t;
        typedef SIXTRL_GLOBAL_DEC unsigned char* g_ptr_uchar_t;
        
        #if !defined( NDEBUG ) && ( !defined( _GPUCODE ) ) 
        typedef NS(block_alignment_t) align_t;
        static align_t   const ZERO      = ( align_t )0u;
        static align_t   const TWO       = ( align_t )2u;
        static align_t   const INFO_SIZE = ( align_t )sizeof( NS(BlockInfo ) );
        #endif /* !defined( NDEBUG ) && ( !defined( _GPUCODE ) )  */
        
        container->num_blocks    = ZERO_SIZE;
        container->data_raw_size = ZERO_SIZE;
        
        SIXTRL_ASSERT( 
            ( container->info_begin_alignment > ZERO_SIZE ) &&
            ( ( container->info_begin_alignment % TWO ) == ZERO ) &&
            ( ( container->info_begin_alignment % INFO_SIZE ) == ZERO ) &&
            ( container->info_alignment > ZERO_SIZE ) &&
            ( ( container->info_alignment % TWO ) == ZERO ) &&
            ( ( container->info_alignment % INFO_SIZE ) == ZERO ) &&
            ( container->data_begin_alignment > ZERO_SIZE ) &&
            ( ( container->data_begin_alignment % TWO ) == ZERO ) &&
            ( container->data_alignment > ZERO_SIZE ) &&
            ( ( container->data_alignment % TWO ) == ZERO ) );
                
        if( ( NS(MemPool_clear_to_aligned_position)( 
                NS(BlocksContainer_get_ptr_info_store)( container ),
                NS(BlocksContainer_get_info_begin_alignment)( container ) ) ) &&
            ( NS(MemPool_clear_to_aligned_position)( 
                NS(BlocksContainer_get_ptr_data_store)( container ),
                NS(BlocksContainer_get_data_begin_alignment)( container ) ) ) )
        {
            container->info_begin = 
            ( g_ptr_info_block_t )NS(MemPool_get_next_begin_pointer)( 
                NS(BlocksContainer_get_ptr_info_store)( container ),
                NS(BlocksContainer_get_info_begin_alignment)( container ) );
            
            container->data_begin = 
            ( g_ptr_uchar_t )NS(MemPool_get_next_begin_pointer)(
                NS(BlocksContainer_get_ptr_data_store)( container ),
                NS(BlocksContainer_get_data_begin_alignment)( container ) );
        }        
    }
    
    return;
}

void NS(BlocksContainer_free)(
    NS(BlocksContainer)* SIXTRL_RESTRICT container )
{
    if( container != 0 )
    {
        if( NS(BlocksContainer_has_info_store)( container ) )
        {
            NS(MemPool)* pool = 
                NS(BlocksContainer_get_ptr_info_store)( container );
            
            NS(MemPool_free)( pool );
        }
        
        if( NS(BlocksContainer_has_data_store)( container ) )
        {
            NS(MemPool)* pool =
                NS(BlocksContainer_get_ptr_data_store)( container );
                
            NS(MemPool_free)( pool );
        }
        
        NS(BlocksContainer_preset)( container );
    }
        
    return;    
}

int NS(BlocksContainer_init)(
    NS(BlocksContainer)* SIXTRL_RESTRICT container, 
    NS(block_size_t) const blocks_capacity, 
    NS(block_size_t) const data_capacity )
{
    int success = -1;
    
    NS(BlocksContainer_reserve_num_blocks)( container, blocks_capacity );
    NS(BlocksContainer_reserve_for_data)(   container, data_capacity );
    
    if( ( NS(BlocksContainer_has_data_store)( container ) ) &&
        ( NS(BlocksContainer_has_info_store)( container ) ) )
    {
        success = 0;
    }
    
    return success;
}

/* ------------------------------------------------------------------------- */

void NS(BlocksContainer_reserve_num_blocks)(
    NS(BlocksContainer)* SIXTRL_RESTRICT container, 
    NS(block_size_t) const new_block_capacity )
{
    typedef NS(block_size_t)                 bl_size_t;    
    typedef SIXTRL_GLOBAL_DEC NS(BlockInfo)* g_ptr_info_block_t;
    
    static bl_size_t const INFO_SIZE = sizeof( NS(BlockInfo) );
    NS(block_size_t) const requ_mem_capacity = INFO_SIZE * new_block_capacity;
    
    int success = 0;
    
    if( NS(BlocksContainer_has_info_store)( container ) )
    {
        NS(MemPool_reserve_aligned)( 
            NS(BlocksContainer_get_ptr_info_store)( container ),
            requ_mem_capacity, 
            NS(BlocksContainer_get_info_begin_alignment)( container ) );
        
        if( ( NS(BlocksContainer_get_ptr_info_store)( container ) != 0 ) &&
            ( NS(MemPool_is_begin_aligned_with)( 
                NS(BlocksContainer_get_ptr_info_store)( container ), 
                NS(BlocksContainer_get_info_begin_alignment)( container ) ) ) &&
            ( NS(MemPool_get_capacity)( 
                NS(BlocksContainer_get_ptr_info_store)( container ) ) >=
                requ_mem_capacity ) )
        {
            success = 1;
        }
    }
    else
    {
        NS(MemPool)* ptr_info_store = NS(MemPool_preset)( 
            ( NS(MemPool)* )malloc( sizeof( NS(MemPool) ) ) );
        
        SIXTRL_ASSERT( 
            NS(BlocksContainer_get_ptr_info_store)( container ) == 0 );
        
        NS(MemPool_init_aligned)( 
            ptr_info_store, requ_mem_capacity, INFO_SIZE, 
            NS(BlocksContainer_get_info_begin_alignment)( container ) );
        
        if( ( NS(MemPool_get_const_begin_pos)( ptr_info_store ) != 0 ) &&
            ( NS(MemPool_is_begin_aligned_with)( 
                NS(BlocksContainer_get_ptr_data_store( container ) ),
                NS(BlocksContainer_get_info_begin_alignment)( container ) ) 
            ) &&
            ( NS(MemPool_get_capacity)( ptr_info_store ) >= 
                requ_mem_capacity ) )
        {
            container->ptr_info_store = ptr_info_store;
            success = 1;
        }
        else
        {
            NS(MemPool_free)( ptr_info_store );
            free( ptr_info_store );
            ptr_info_store = 0;
        }
    }
    
    if( success )
    {
        NS(MemPool)* ptr_info_store = 
            NS(BlocksContainer_get_ptr_info_store)( container );
        
        container->blocks_capacity = 
            new_block_capacity;
        
        container->info_begin = 
            ( g_ptr_info_block_t )NS(MemPool_get_begin_pos)( ptr_info_store );
    }
    
    return;
}

void NS(BlocksContainer_reserve_for_data)(
     NS(BlocksContainer)* SIXTRL_RESTRICT container, 
     NS(block_size_t) const new_data_capacity )
{
    typedef SIXTRL_GLOBAL_DEC unsigned char* g_ptr_uchar_t;
    
    int success = 0;
    
    if( NS(BlocksContainer_has_data_store)( container ) )
    {
        NS(MemPool_reserve_aligned)( 
            NS(BlocksContainer_get_ptr_data_store)( container ),
            new_data_capacity, 
            NS(BlocksContainer_get_data_begin_alignment)( container ) );
        
        if( ( NS(BlocksContainer_get_ptr_data_store)( container ) != 0 ) &&
            ( NS(MemPool_is_begin_aligned_with)( 
                NS(BlocksContainer_get_ptr_data_store)( container ), 
                NS(BlocksContainer_get_data_begin_alignment)( container ) ) 
            ) &&
            ( NS(MemPool_get_capacity)( 
                NS(BlocksContainer_get_ptr_data_store)( container ) ) >=
                new_data_capacity ) )
        {
            success = 1;
        }
    }
    else
    {
        NS(MemPool)* ptr_data_store = NS(MemPool_preset)( 
            ( NS(MemPool)* )malloc( sizeof( NS(MemPool) ) ) );
        
        SIXTRL_ASSERT( 
            NS(BlocksContainer_get_ptr_data_store)( container ) == 0 );
        
        NS(MemPool_init_aligned)( ptr_data_store, new_data_capacity,
            NS(BlocksContainer_get_data_alignment)( container ),
            NS(BlocksContainer_get_data_begin_alignment)( container ) );
        
        if( ( NS(MemPool_get_const_begin_pos)( ptr_data_store ) != 0 ) &&
            ( NS(MemPool_is_begin_aligned_with)( ptr_data_store, 
                NS(BlocksContainer_get_data_begin_alignment)( container ) ) 
            ) &&
            ( NS(MemPool_get_capacity)( ptr_data_store ) >= 
                new_data_capacity ) )
        {
            container->ptr_data_store = ptr_data_store;
            container->data_raw_size  = ( NS(block_size_t) )0u;
            success = 1;
        }
        else
        {
            NS(MemPool_free)( ptr_data_store );
            free( ptr_data_store );
            ptr_data_store = 0;
        }
    }
    
    if( success )
    {
        NS(MemPool)* ptr_data_store = 
            NS(BlocksContainer_get_ptr_data_store)( container );
        
        container->data_raw_capacity = new_data_capacity;
        
        container->data_begin = 
            ( g_ptr_uchar_t )NS(MemPool_get_begin_pos)( ptr_data_store );
    }
    
    return;
}

/* ------------------------------------------------------------------------- */

/* end: sixtracklib/common/details/blocks_container.c */
