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

extern NS(BlocksContainer)* NS(BlocksContainer_preset)( 
    NS(BlocksContainer)* SIXTRL_RESTRICT container );

extern void NS(BlocksContainer_clear)(
    NS(BlocksContainer)* SIXTRL_RESTRICT container );

extern void NS(BlocksContainer_free)(
    NS(BlocksContainer)* SIXTRL_RESTRICT container );

extern int NS(BlocksContainer_init)(
    NS(BlocksContainer)* SIXTRL_RESTRICT container, 
    NS(block_size_t) const blocks_capacity, 
    NS(block_size_t) const data_capacity );

/* ------------------------------------------------------------------------- */

extern int NS(BlocksContainer_set_info_begin_alignment)(
    NS(BlocksContainer)* SIXTRL_RESTRICT container, 
    NS(block_alignment_t) const begin_alignment );

extern int NS(BlocksContainer_set_data_begin_alignment)(
    NS(BlocksContainer)* SIXTRL_RESTRICT container, 
    NS(block_alignment_t) const begin_alignment );

extern int NS(BlocksContainer_set_data_alignment)(
    NS(BlocksContainer)* SIXTRL_RESTRICT container, 
    NS(block_alignment_t) const alignment );

extern int NS(BlocksContainer_set_info_alignment )(
    NS(BlocksContainer)* SIXTRL_RESTRICT container, 
    NS(block_alignment_t) const alignment );

extern void NS(BlocksContainer_reserve_num_blocks)(
    NS(BlocksContainer)* SIXTRL_RESTRICT container, 
    NS(block_size_t) const new_block_capacity );

extern void NS(BlocksContainer_reserve_for_data)(
    NS(BlocksContainer)* SIXTRL_RESTRICT container, 
    NS(block_size_t) const new_data_capacity );

/* ------------------------------------------------------------------------- */

extern NS(block_alignment_t) NS(BlocksContainer_get_info_alignment)(
    const NS(BlocksContainer) *const SIXTRL_RESTRICT container );

extern NS(block_alignment_t) NS(BlocksContainer_get_data_alignment)(
    const NS(BlocksContainer) *const SIXTRL_RESTRICT container );

extern NS(block_alignment_t) NS(BlocksContainer_get_info_begin_alignment)(
    const NS(BlocksContainer) *const SIXTRL_RESTRICT container );

extern NS(block_alignment_t) NS(BlocksContainer_get_data_begin_alignment)(
    const NS(BlocksContainer) *const SIXTRL_RESTRICT container );

extern NS(block_size_t) NS(BlocksContainer_get_data_capacity)(
    const NS(BlocksContainer) *const SIXTRL_RESTRICT container );

extern NS(block_size_t) NS(BlocksContainer_get_data_size)(
    const NS(BlocksContainer) *const SIXTRL_RESTRICT container );

extern NS(block_size_t) NS(BlocksContainer_get_block_capacity)(
    const NS(BlocksContainer) *const SIXTRL_RESTRICT container );

extern NS(block_size_t) NS(BlocksContainer_get_num_of_blocks)(
    const NS(BlocksContainer) *const SIXTRL_RESTRICT container );

/* ------------------------------------------------------------------------- */

extern unsigned char const* NS(BlocksContainer_get_const_ptr_data_begin)(
    const NS(BlocksContainer) *const SIXTRL_RESTRICT container );

extern unsigned char* NS(BlocksContainer_get_ptr_data_begin)(
    NS(BlocksContainer)* SIXTRL_RESTRICT container );

extern NS(BlockInfo) const* NS(BlocksContainer_get_const_block_infos_begin)(
    const NS(BlocksContainer) *const SIXTRL_RESTRICT container );

extern NS(BlockInfo) const* NS(BlocksContainer_get_const_block_infos_end)(
    const NS(BlocksContainer) *const SIXTRL_RESTRICT container );


extern NS(BlockInfo)* NS(BlocksContainer_get_block_infos_begin)(
    NS(BlocksContainer)* SIXTRL_RESTRICT container );

extern NS(BlockInfo)* NS(BlocksContainer_get_block_infos_end)(
    NS(BlocksContainer)* SIXTRL_RESTRICT container );

extern NS(BlockInfo) NS(BlocksContainer_get_block_info_by_index)(
    const NS(BlocksContainer) *const SIXTRL_RESTRICT container, 
    NS(block_size_t) const block_index );

extern NS(BlockInfo) const* 
NS(BlocksContainer_get_const_ptr_to_block_info_by_index)(
    const NS(BlocksContainer) *const SIXTRL_RESTRICT container, 
    NS(block_size_t) const block_index );

extern NS(BlockInfo)* NS(BlocksContainer_get_ptr_to_block_info_by_index)(
    NS(BlocksContainer)* SIXTRL_RESTRICT container, 
    NS(block_size_t) const block_index );

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
    
NS(BlocksContainer)* NS(BlocksContainer_preset)( 
    NS(BlocksContainer)* SIXTRL_RESTRICT container )
{
    if( container != 0 )
    {
        container->info_begin = 0;
        container->data_begin = 0;
        
        NS(MemPool_preset)( &container->info_store );
        NS(MemPool_preset)( &container->data_store );
        
        container->num_blocks           = ( NS(block_size_t) )0u;
        container->blocks_capacity      = ( NS(block_size_t) )0u;
        
        container->data_raw_size        = ( NS(block_size_t) )0u;
        container->data_raw_capacity    = ( NS(block_size_t) )0u;
        
        container->info_begin_alignment = NS(DEFAULT_INFO_BEGIN_ALIGNMENT);
        container->info_alignment       = NS(DEFAULT_INFO_ALIGNMENT);
        
        container->data_begin_alignment = NS(DEFAULT_DATA_BEGIN_ALIGNMENT);
        container->data_alignment       = NS(DEFAULT_DATA_ALIGNMENT);        
    }
    
    return container;
}

void NS(BlocksContainer_clear)(
    NS(BlocksContainer)* SIXTRL_RESTRICT container )
{
    if( container != 0 )
    {
        typedef NS(block_size_t)      bl_size_t;
        
        static bl_size_t const ZERO_SIZE = ( bl_size_t )0u;
        
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
                &container->info_store, container->info_begin_alignment ) ) &&
            ( NS(MemPool_clear_to_aligned_position)( 
            &container->data_store, container->data_begin_alignment ) ) )
        {
            container->info_begin = 
            ( NS(BlockInfo)* )NS(MemPool_get_next_begin_pointer)( 
                &container->info_store, container->info_begin_alignment );
            
            container->data_begin = NS(MemPool_get_next_begin_pointer)(
                &container->data_store, container->data_begin_alignment );            
        }        
    }
    
    return;
}

void NS(BlocksContainer_free)(
    NS(BlocksContainer)* SIXTRL_RESTRICT container )
{
    if( container != 0 )
    {
        container->num_blocks    = ( NS(block_size_t) )0u;
        container->data_raw_size = ( NS(block_size_t) )0u;
        
        NS(MemPool_free)( &container->info_store );
        NS(MemPool_free)( &container->data_store );
        
        NS(BlocksContainer_preset)( container );
    }
        
    return;    
}

int NS(BlocksContainer_init)(
    NS(BlocksContainer)* SIXTRL_RESTRICT container, 
    NS(block_size_t) const blocks_capacity, 
    NS(block_size_t) const data_capacity )
{
    typedef NS(block_alignment_t) align_t;
    typedef NS(block_size_t)      bl_size_t;
    
    int success = -1;
    
    align_t const info_begin_align = 
        NS(BlocksContainer_get_info_begin_alignment)( container );
        
    align_t const info_align =
        NS(BlocksContainer_get_info_alignment)( container );
        
    align_t const data_begin_align =
        NS(BlocksContainer_get_data_begin_alignment)( container );
        
    align_t const data_align =
        NS(BlocksContainer_get_data_alignment)( container );
    
    bl_size_t const info_capacity = sizeof( NS(BlockInfo) ) * blocks_capacity;
        
    if( ( data_capacity > ( NS(block_size_t) )data_align ) &&
        ( info_capacity > ( NS(block_size_t) )info_align ) &&
        ( info_begin_align > ( align_t )0u ) && 
        ( info_align > ( align_t )0u ) &&
        ( data_begin_align > ( align_t )0u ) &&
        ( data_align > ( align_t )0u ) )
    {
        NS(MemPool_free)( &container->info_store );
        NS(MemPool_free)( &container->data_store );
        
        container->info_begin = 0;
        container->data_begin = 0;
        
        container->num_blocks        = ( bl_size_t )0u;
        container->blocks_capacity   = ( bl_size_t )0u;
        
        container->data_raw_size     = ( bl_size_t )0u;
        container->data_raw_capacity = ( bl_size_t )0u;
        
        SIXTRL_ASSERT( 
            ( container != 0 ) && 
            ( ( info_begin_align % ( align_t )2u ) == ( align_t )0u ) &&
            ( ( info_align       % ( align_t )2u ) == ( align_t )0u ) &&
            ( ( data_begin_align % ( align_t )2u ) == ( align_t )0u ) &&
            ( ( info_align       % ( align_t )2u ) == ( align_t )0u ) );
        
        NS(MemPool_init)( &container->info_store, 
                          info_capacity, info_align );
        
        NS(MemPool_init)( &container->data_store, 
                          data_capacity, data_align );
        
        if( ( NS(MemPool_clear_to_aligned_position)( 
                &container->info_store, info_begin_align ) ) &&
            ( NS(MemPool_clear_to_aligned_position)(
                &container->data_store, data_begin_align ) ) )
        {
            container->info_begin = 
            ( NS(BlockInfo)* )NS(MemPool_get_next_begin_pointer)( 
                &container->info_store, container->info_begin_alignment );
            
            container->data_begin = NS(MemPool_get_next_begin_pointer)(
                &container->data_store, container->data_begin_alignment );   
            
            container->blocks_capacity   = blocks_capacity;
            container->data_raw_capacity = data_capacity;
            
            success = 0;
        }                   
    }
    
    return success;
}

/* ------------------------------------------------------------------------- */

int NS(BlocksContainer_set_info_begin_alignment)(
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

int NS(BlocksContainer_set_data_begin_alignment)(
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

int NS(BlocksContainer_set_data_alignment)(
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

int NS(BlocksContainer_set_info_alignment )(
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

void NS(BlocksContainer_reserve_num_blocks)(
    NS(BlocksContainer)* SIXTRL_RESTRICT container, 
    NS(block_size_t) const new_block_capacity )
{
    typedef NS(block_alignment_t)   align_t;
    typedef NS(block_size_t)        bl_size_t;
    
    static bl_size_t const INFO_SIZE = sizeof( NS(BlockInfo) );
    
    if( new_block_capacity > 
           NS(BlocksContainer_get_block_capacity)( container ) )
    {
        align_t const begin_alignment =
            NS(BlocksContainer_get_info_begin_alignment)( container );
        
        bl_size_t const alignment = 
            NS(BlocksContainer_get_info_begin_alignment)( container );
            
        bl_size_t const required_size = 
            NS(BlocksContainer_get_info_begin_alignment)( container ) +
            new_block_capacity * INFO_SIZE;
            
        NS(MemPool) new_store;
        NS(MemPool_preset)( &new_store );
    
        SIXTRL_ASSERT( 
            (   alignment >= INFO_SIZE ) &&
            ( ( alignment %  INFO_SIZE ) == ( bl_size_t )0u ) &&
            ( ( alignment % ( bl_size_t )2u ) == ( bl_size_t )0u ) );
        
        NS(MemPool_init)( &new_store, required_size, alignment );
        
        if( ( NS(MemPool_get_buffer)( &new_store ) != 0 ) &&
            ( NS(MemPool_get_capacity)( &new_store ) >= required_size ) &&
            ( NS(MemPool_clear_to_aligned_position) ( 
                &new_store, begin_alignment ) ) )
        {
            NS(MemPool) temp = container->info_store;
            
            NS(BlockInfo)* info_begin = 
            ( NS(BlockInfo)* )NS(MemPool_get_next_begin_pointer)(
                &new_store, begin_alignment );
            
            if( NS(BlocksContainer_get_num_of_blocks)( container ) > 
                ( bl_size_t )0u )
            {
                SIXTRL_ASSERT( container->info_begin != 0 );
                memcpy( info_begin, container->info_begin, INFO_SIZE * 
                        NS(BlocksContainer_get_num_of_blocks)( container ) );
            }
            
            SIXTRL_ASSERT( NS(MemPool_get_remaining_bytes)( &new_store ) >=
                INFO_SIZE * ( bl_size_t )new_block_capacity );
            
            container->info_store = new_store;
            container->info_begin = info_begin;
            container->blocks_capacity = new_block_capacity;
            
            NS(MemPool_free)( &temp );
            NS(MemPool_preset)( &new_store );
        }
        else
        {
            NS(MemPool_free)( &new_store );
        }        
    }
    
    return;
}

void NS(BlocksContainer_reserve_for_data)(
     NS(BlocksContainer)* SIXTRL_RESTRICT container, 
     NS(block_size_t) const new_data_capacity )
{
    typedef NS(block_alignment_t)   align_t;
    typedef NS(block_size_t)        bl_size_t;
    
    if( new_data_capacity > 
           NS(BlocksContainer_get_data_capacity)( container ) )
    {
        align_t const begin_alignment =
            NS(BlocksContainer_get_data_begin_alignment)( container );
        
        bl_size_t const alignment = 
            NS(BlocksContainer_get_data_begin_alignment)( container );
            
        bl_size_t const required_size = new_data_capacity +
            NS(BlocksContainer_get_info_begin_alignment)( container );
            
        NS(MemPool) new_store;
        NS(MemPool_preset)( &new_store );
    
        SIXTRL_ASSERT( 
            (   alignment > ( bl_size_t)0u ) &&
            ( ( alignment % ( bl_size_t )2u ) == ( bl_size_t )0u ) );
        
        NS(MemPool_init)( &new_store, required_size, alignment );
        
        if( ( NS(MemPool_get_buffer)( &new_store ) != 0 ) &&
            ( NS(MemPool_get_capacity)( &new_store ) >= required_size ) &&
            ( NS(MemPool_clear_to_aligned_position) ( 
                &new_store, begin_alignment ) ) )
        {
            NS(MemPool) temp = container->data_store;
            
            unsigned char* data_begin = NS(MemPool_get_next_begin_pointer)(
                &new_store, begin_alignment );
            
            if( NS(BlocksContainer_get_num_of_blocks)( container ) > 
                ( bl_size_t )0u )
            {
                SIXTRL_ASSERT( container->info_begin != 0 );
                memcpy( data_begin, container->data_begin, 
                        NS(BlocksContainer_get_data_size)( container ) );
            }
            
            container->data_store = new_store;
            container->data_begin = data_begin;
            container->data_raw_capacity = new_data_capacity;
            
            NS(MemPool_free)( &temp );
            NS(MemPool_preset)( &new_store );
        }
        else
        {
            NS(MemPool_free)( &new_store );
        }        
    }
    
    return;
}

/* ------------------------------------------------------------------------- */

NS(block_alignment_t) NS(BlocksContainer_get_info_alignment)(
    const NS(BlocksContainer) *const SIXTRL_RESTRICT container )
{
    SIXTRL_ASSERT( container != 0 );
    return container->info_alignment;
}

NS(block_alignment_t) NS(BlocksContainer_get_data_alignment)(
    const NS(BlocksContainer) *const SIXTRL_RESTRICT container )
{
    SIXTRL_ASSERT( container != 0 );
    return container->data_alignment;
}

NS(block_alignment_t) NS(BlocksContainer_get_info_begin_alignment)(
    const NS(BlocksContainer) *const SIXTRL_RESTRICT container )
{
    SIXTRL_ASSERT( container != 0 );
    return container->info_begin_alignment;
}

NS(block_alignment_t) NS(BlocksContainer_get_data_begin_alignment)(
    const NS(BlocksContainer) *const SIXTRL_RESTRICT container )
{
    SIXTRL_ASSERT( container != 0 );
    return container->data_begin_alignment;
}

NS(block_size_t) NS(BlocksContainer_get_data_capacity)(
    const NS(BlocksContainer) *const SIXTRL_RESTRICT container )
{
    SIXTRL_ASSERT( container != 0 ) ;
    return container->data_raw_capacity;
}

NS(block_size_t) NS(BlocksContainer_get_data_size)(
    const NS(BlocksContainer) *const SIXTRL_RESTRICT container )
{
    SIXTRL_ASSERT( container != 0 );
    return container->data_raw_size;
}

NS(block_size_t) NS(BlocksContainer_get_block_capacity)(
    const NS(BlocksContainer) *const SIXTRL_RESTRICT container )
{
    SIXTRL_ASSERT( container != 0 );
    return container->blocks_capacity;
}

NS(block_size_t) NS(BlocksContainer_get_num_of_blocks)(
    const NS(BlocksContainer) *const SIXTRL_RESTRICT container )
{
    SIXTRL_ASSERT( container != 0 );
    return container->num_blocks;
}

/* ------------------------------------------------------------------------- */

unsigned char const* NS(BlocksContainer_get_const_ptr_data_begin)(
    const NS(BlocksContainer) *const SIXTRL_RESTRICT container )
{
    SIXTRL_ASSERT( container != 0 );
    return container->data_begin;
}

unsigned char* NS(BlocksContainer_get_ptr_data_begin)(
    NS(BlocksContainer)* SIXTRL_RESTRICT cont )
{
    return ( unsigned char* )NS(BlocksContainer_get_const_ptr_data_begin)( cont );
}

NS(BlockInfo) const* NS(BlocksContainer_get_const_block_infos_begin)(
    const NS(BlocksContainer) *const SIXTRL_RESTRICT container )
{
    SIXTRL_ASSERT( container != 0 );
    return container->info_begin;
}

NS(BlockInfo) const* NS(BlocksContainer_get_const_block_infos_end)(
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


NS(BlockInfo)* NS(BlocksContainer_get_block_infos_begin)(
    NS(BlocksContainer)* SIXTRL_RESTRICT container )
{
    return ( NS(BlockInfo)* )NS(BlocksContainer_get_const_block_infos_begin)(
        container );
}

NS(BlockInfo)* NS(BlocksContainer_get_block_infos_end)(
    NS(BlocksContainer)* SIXTRL_RESTRICT container )
{
    return ( NS(BlockInfo)* )NS(BlocksContainer_get_const_block_infos_end)( 
        container );
}

NS(BlockInfo) NS(BlocksContainer_get_block_info_by_index)(
    const NS(BlocksContainer) *const SIXTRL_RESTRICT container, 
    NS(block_size_t) const block_index )
{
    NS(BlockInfo) const* ptr_to_info = 
        NS(BlocksContainer_get_const_block_infos_end)( container );
        
    SIXTRL_ASSERT( ptr_to_info != 0 );
    return *ptr_to_info;
}

NS(BlockInfo) const* NS(BlocksContainer_get_const_ptr_to_block_info_by_index)(
    const NS(BlocksContainer) *const SIXTRL_RESTRICT container, 
    NS(block_size_t) const block_index )
{
    SIXTRL_ASSERT( container != 0 );
    return ( ( container != 0 ) && ( container->info_begin != 0 ) &&
             ( NS(BlocksContainer_get_block_capacity)( container ) >
               block_index ) )
        ? &container->info_begin[ block_index ] : 0;
}

NS(BlockInfo)* NS(BlocksContainer_get_ptr_to_block_info_by_index)(
    NS(BlocksContainer)* SIXTRL_RESTRICT container, 
    NS(block_size_t) const block_index )
{
    return ( NS(BlockInfo)* 
        )NS(BlocksContainer_get_const_ptr_to_block_info_by_index)(
            container, block_index );
}

/* ------------------------------------------------------------------------- */

/* end: sixtracklib/common/details/blocks_container.c */
