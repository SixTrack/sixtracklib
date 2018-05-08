#ifndef SIXTRACKLIB_COMMON_PARTICLES_H__
#define SIXTRACKLIB_COMMON_PARTICLES_H__

#if !defined( _GPUCODE )

#include "sixtracklib/_impl/definitions.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

#include "sixtracklib/common/impl/block_info_impl.h"
#include "sixtracklib/common/blocks_container.h"
#include "sixtracklib/common/impl/particles_impl.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */
    
#endif /* !defined( _GPUCODE ) */
   
/* ------------------------------------------------------------------------- */

typedef NS(BlocksContainer) NS(ParticlesContainer);

/* ------------------------------------------------------------------------- */

SIXTRL_STATIC NS(ParticlesContainer)* NS(ParticlesContainer_preset)( 
    NS(ParticlesContainer)* particle_buffer );

SIXTRL_STATIC void NS(ParticlesContainer_clear)( 
    NS(ParticlesContainer)* SIXTRL_RESTRICT particle_buffer );

SIXTRL_STATIC void NS(ParticlesContainer_free)( 
    NS(ParticlesContainer)* SIXTRL_RESTRICT particle_buffer );

SIXTRL_STATIC int NS(ParticlesContainer_init)( 
    NS(ParticlesContainer)* SIXTRL_RESTRICT particle_buffer, 
    NS(block_size_t) const blocks_capacity, 
    NS(block_size_t) const data_capacity );

/* ------------------------------------------------------------------------- */

SIXTRL_STATIC int NS(ParticlesContainer_set_info_begin_alignment)(
    NS(ParticlesContainer)* SIXTRL_RESTRICT particle_buffer, 
    NS(block_alignment_t) const begin_alignment );

SIXTRL_STATIC int NS(ParticlesContainer_set_data_begin_alignment)(
    NS(ParticlesContainer)* SIXTRL_RESTRICT particle_buffer, 
    NS(block_alignment_t) const begin_alignment );

SIXTRL_STATIC int NS(ParticlesContainer_set_data_alignment)(
    NS(ParticlesContainer)* SIXTRL_RESTRICT particle_buffer, 
    NS(block_alignment_t) const alignment );

SIXTRL_STATIC int NS(ParticlesContainer_set_info_alignment )(
    NS(ParticlesContainer)* SIXTRL_RESTRICT particle_buffer, 
    NS(block_alignment_t) const alignment );

SIXTRL_STATIC void NS(ParticlesContainer_reserve_num_blocks)(
    NS(ParticlesContainer)* SIXTRL_RESTRICT particle_buffer, 
    NS(block_size_t) const new_block_capacity );

SIXTRL_STATIC void NS(ParticlesContainer_reserve_for_data)(
    NS(ParticlesContainer)* SIXTRL_RESTRICT particle_buffer, 
    NS(block_size_t) const new_data_capacity );

/* ------------------------------------------------------------------------- */

SIXTRL_STATIC int NS(ParticlesContainer_has_info_store)(
    const NS(ParticlesContainer) *const SIXTRL_RESTRICT container );

SIXTRL_STATIC int NS(ParticlesContainer_has_data_store)(
    const NS(ParticlesContainer) *const SIXTRL_RESTRICT container );

SIXTRL_STATIC NS(MemPool) const* 
NS(ParticlesContainer_get_const_ptr_info_store)(
    NS(ParticlesContainer)* SIXTRL_RESTRICT container );

SIXTRL_STATIC NS(MemPool) const* 
NS(ParticlesContainer_get_const_ptr_data_store)(
    NS(ParticlesContainer)* SIXTRL_RESTRICT container );

SIXTRL_STATIC NS(MemPool)* 
NS(ParticlesContainer_get_ptr_info_store)(
    NS(ParticlesContainer)* SIXTRL_RESTRICT container );

SIXTRL_STATIC NS(MemPool)* 
NS(ParticlesContainer_get_ptr_data_store)(
    NS(ParticlesContainer)* SIXTRL_RESTRICT container );

/* ------------------------------------------------------------------------- */

SIXTRL_STATIC NS(block_alignment_t) NS(ParticlesContainer_get_info_alignment)(
    const NS(ParticlesContainer) *const SIXTRL_RESTRICT particle_buffer );

SIXTRL_STATIC NS(block_alignment_t) NS(ParticlesContainer_get_data_alignment)(
    const NS(ParticlesContainer) *const SIXTRL_RESTRICT particle_buffer );

SIXTRL_STATIC NS(block_alignment_t) 
NS(ParticlesContainer_get_info_begin_alignment)(
    const NS(ParticlesContainer) *const SIXTRL_RESTRICT particle_buffer );

SIXTRL_STATIC NS(block_alignment_t) 
NS(ParticlesContainer_get_data_begin_alignment)(
    const NS(ParticlesContainer) *const SIXTRL_RESTRICT particle_buffer );

SIXTRL_STATIC NS(block_size_t) NS(ParticlesContainer_get_data_capacity)(
    const NS(ParticlesContainer) *const SIXTRL_RESTRICT particle_buffer );

SIXTRL_STATIC NS(block_size_t) NS(ParticlesContainer_get_data_size)(
    const NS(ParticlesContainer) *const SIXTRL_RESTRICT particle_buffer );

SIXTRL_STATIC NS(block_size_t) NS(ParticlesContainer_get_block_capacity)(
    const NS(ParticlesContainer) *const SIXTRL_RESTRICT particle_buffer );

SIXTRL_STATIC NS(block_size_t) NS(ParticlesContainer_get_num_of_blocks)(
    const NS(ParticlesContainer) *const SIXTRL_RESTRICT particle_buffer );

/* ------------------------------------------------------------------------- */

SIXTRL_STATIC unsigned char const* 
NS(ParticlesContainer_get_const_ptr_data_begin)(
    const NS(ParticlesContainer) *const SIXTRL_RESTRICT particle_buffer );

SIXTRL_STATIC unsigned char* 
NS(ParticlesContainer_get_ptr_data_begin)(
    NS(ParticlesContainer)* SIXTRL_RESTRICT particle_buffer );

SIXTRL_STATIC NS(BlockInfo) const* 
NS(ParticlesContainer_get_const_block_infos_begin)(
    const NS(ParticlesContainer) *const SIXTRL_RESTRICT particle_buffer );

SIXTRL_STATIC NS(BlockInfo) const* 
NS(ParticlesContainer_get_const_block_infos_end)(
    const NS(ParticlesContainer) *const SIXTRL_RESTRICT particle_buffer );


SIXTRL_STATIC NS(BlockInfo)* NS(ParticlesContainer_get_block_infos_begin)(
    NS(ParticlesContainer)* SIXTRL_RESTRICT particle_buffer );

SIXTRL_STATIC NS(BlockInfo)* NS(ParticlesContainer_get_block_infos_end)(
    NS(ParticlesContainer)* SIXTRL_RESTRICT particle_buffer );

SIXTRL_STATIC NS(BlockInfo) NS(ParticlesContainer_get_block_info_by_index)(
    const NS(ParticlesContainer) *const SIXTRL_RESTRICT particle_buffer, 
    NS(block_size_t) const block_index );

SIXTRL_STATIC NS(BlockInfo) const* 
NS(ParticlesContainer_get_const_ptr_to_block_info_by_index)(
    const NS(ParticlesContainer) *const SIXTRL_RESTRICT particle_buffer, 
    NS(block_size_t) const block_index );

SIXTRL_STATIC NS(BlockInfo)* 
NS(ParticlesContainer_get_ptr_to_block_info_by_index)(
    NS(ParticlesContainer)* SIXTRL_RESTRICT particle_buffer, 
    NS(block_size_t) const block_index );

/* ------------------------------------------------------------------------- */

SIXTRL_STATIC int NS(ParticlesContainer_add_particles)( 
    NS(ParticlesContainer)* SIXTRL_RESTRICT particles_buffer, 
    NS(Particles)* SIXTRL_RESTRICT particle_block,
    NS(block_num_elements_t) const num_of_particles );

SIXTRL_STATIC int NS(ParticlesContainer_get_particles)(
    NS(Particles)* SIXTRL_RESTRICT particles,
    NS(ParticlesContainer)* SIXTRL_RESTRICT particles_buffer,
    NS(block_size_t) const block_index );


int NS(ParticlesContainer_add_blocks_of_particles)(
    NS(ParticlesContainer)* SIXTRL_RESTRICT particles_buffer,
    NS(Particles)* SIXTRL_RESTRICT particle_blocks,
    NS(block_size_t) const num_of_blocks, 
    NS(block_num_elements_t) const* SIXTRL_RESTRICT num_of_particles_vec );

/* ************************************************************************ */
/* *********     Implementation of inline functions and methods     ******* */
/* ************************************************************************ */

SIXTRL_INLINE NS(ParticlesContainer)* NS(ParticlesContainer_preset)( 
    NS(ParticlesContainer)* particle_buffer )
{
    return NS(BlocksContainer_preset)( particle_buffer );
}

SIXTRL_INLINE void NS(ParticlesContainer_clear)( 
    NS(ParticlesContainer)* SIXTRL_RESTRICT particle_buffer )
{
    NS(BlocksContainer_clear)( particle_buffer );
    return;
}

SIXTRL_INLINE void NS(ParticlesContainer_free)( 
    NS(ParticlesContainer)* SIXTRL_RESTRICT particle_buffer )
{
    NS(BlocksContainer_free)( particle_buffer );
    return;
}

SIXTRL_INLINE int NS(ParticlesContainer_init)( 
    NS(ParticlesContainer)* SIXTRL_RESTRICT particle_buffer, 
    NS(block_size_t) const blocks_capacity, 
    NS(block_size_t) const data_capacity )
{
    return NS(BlocksContainer_init)( 
        particle_buffer, blocks_capacity, data_capacity );
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE int NS(ParticlesContainer_set_info_begin_alignment)(
    NS(ParticlesContainer)* SIXTRL_RESTRICT particle_buffer, 
    NS(block_alignment_t) const begin_alignment )
{
    return NS(BlocksContainer_set_info_begin_alignment)( 
        particle_buffer, begin_alignment );
}

SIXTRL_INLINE int NS(ParticlesContainer_set_data_begin_alignment)(
    NS(ParticlesContainer)* SIXTRL_RESTRICT particle_buffer, 
    NS(block_alignment_t) const begin_alignment )
{
    return NS(BlocksContainer_set_data_begin_alignment)( 
        particle_buffer, begin_alignment );
}

SIXTRL_INLINE int NS(ParticlesContainer_set_data_alignment)(
    NS(ParticlesContainer)* SIXTRL_RESTRICT particle_buffer, 
    NS(block_alignment_t) const alignment )
{
    return NS(BlocksContainer_set_data_alignment)( 
        particle_buffer, alignment );
}

SIXTRL_INLINE int NS(ParticlesContainer_set_info_alignment )(
    NS(ParticlesContainer)* SIXTRL_RESTRICT particle_buffer, 
    NS(block_alignment_t) const alignment )
{
    return NS(BlocksContainer_set_info_alignment)( 
        particle_buffer, alignment );
}

SIXTRL_INLINE void NS(ParticlesContainer_reserve_num_blocks)(
    NS(ParticlesContainer)* SIXTRL_RESTRICT particle_buffer, 
    NS(block_size_t) const new_block_capacity )
{
    NS(BlocksContainer_reserve_num_blocks)( 
        particle_buffer, new_block_capacity );
    
    return;
}

SIXTRL_INLINE void NS(ParticlesContainer_reserve_for_data)(
    NS(ParticlesContainer)* SIXTRL_RESTRICT particle_buffer, 
    NS(block_size_t) const new_data_capacity )
{
    NS(BlocksContainer_reserve_for_data)( 
        particle_buffer, new_data_capacity );
    
    return;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE int NS(ParticlesContainer_has_info_store)(
    const NS(ParticlesContainer) *const SIXTRL_RESTRICT container )
{
    return ( ( container != 0 ) && ( container->ptr_info_store != 0 ) );
}

SIXTRL_INLINE int NS(ParticlesContainer_has_data_store)(
    const NS(ParticlesContainer) *const SIXTRL_RESTRICT container )
{
    return ( ( container != 0 ) && ( container->ptr_data_store != 0 ) );
}

SIXTRL_INLINE NS(MemPool) const* 
NS(ParticlesContainer_get_const_ptr_info_store)(
    NS(ParticlesContainer)* SIXTRL_RESTRICT container )
{
    return NS(BlocksContainer_get_const_ptr_info_store)( container );
}

SIXTRL_INLINE NS(MemPool) const* 
NS(ParticlesContainer_get_const_ptr_data_store)(
    NS(ParticlesContainer)* SIXTRL_RESTRICT container )
{
    return NS(BlocksContainer_get_const_ptr_data_store)( container );
}

SIXTRL_INLINE NS(MemPool)* NS(ParticlesContainer_get_ptr_info_store)(
    NS(ParticlesContainer)* SIXTRL_RESTRICT container )
{
    return NS(BlocksContainer_get_ptr_info_store)( container );
}

SIXTRL_INLINE NS(MemPool)* NS(ParticlesContainer_get_ptr_data_store)(
    NS(ParticlesContainer)* SIXTRL_RESTRICT container )
{
    return NS(BlocksContainer_get_ptr_data_store)( container );
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE NS(block_alignment_t) 
NS(ParticlesContainer_get_info_alignment)(
    const NS(ParticlesContainer) *const SIXTRL_RESTRICT particle_buffer )
{
    return NS(BlocksContainer_get_info_alignment)( particle_buffer );
}

SIXTRL_INLINE NS(block_alignment_t) NS(ParticlesContainer_get_data_alignment)(
    const NS(ParticlesContainer) *const SIXTRL_RESTRICT particle_buffer )
{
    return NS(BlocksContainer_get_data_alignment)( particle_buffer );
}

SIXTRL_INLINE NS(block_alignment_t) 
NS(ParticlesContainer_get_info_begin_alignment)(
    const NS(ParticlesContainer) *const SIXTRL_RESTRICT particle_buffer )
{
    return NS(BlocksContainer_get_info_begin_alignment)( particle_buffer );
}

SIXTRL_INLINE NS(block_alignment_t) 
NS(ParticlesContainer_get_data_begin_alignment)(
    const NS(ParticlesContainer) *const SIXTRL_RESTRICT particle_buffer )
{
    return NS(BlocksContainer_get_data_begin_alignment)( particle_buffer );
}

SIXTRL_INLINE NS(block_size_t) NS(ParticlesContainer_get_data_capacity)(
    const NS(ParticlesContainer) *const SIXTRL_RESTRICT particle_buffer )
{
    return NS(BlocksContainer_get_data_capacity)( particle_buffer );
}

SIXTRL_INLINE NS(block_size_t) NS(ParticlesContainer_get_data_size)(
    const NS(ParticlesContainer) *const SIXTRL_RESTRICT particle_buffer )
{
    return NS(BlocksContainer_get_data_size)( particle_buffer );
}

SIXTRL_INLINE NS(block_size_t) NS(ParticlesContainer_get_block_capacity)(
    const NS(ParticlesContainer) *const SIXTRL_RESTRICT particle_buffer )
{
    return NS(BlocksContainer_get_block_capacity)( particle_buffer );
}

SIXTRL_INLINE NS(block_size_t) NS(ParticlesContainer_get_num_of_blocks)(
    const NS(ParticlesContainer) *const SIXTRL_RESTRICT particle_buffer )
{
    return NS(BlocksContainer_get_num_of_blocks)( particle_buffer );
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE unsigned char const* 
NS(ParticlesContainer_get_const_ptr_data_begin)(
    const NS(ParticlesContainer) *const SIXTRL_RESTRICT particle_buffer )
{
    return NS(BlocksContainer_get_const_ptr_data_begin)( particle_buffer );
}

SIXTRL_INLINE unsigned char* NS(ParticlesContainer_get_ptr_data_begin)(
    NS(ParticlesContainer)* SIXTRL_RESTRICT particle_buffer )
{
    return NS(BlocksContainer_get_ptr_data_begin)( particle_buffer );
}

SIXTRL_INLINE NS(BlockInfo) const* 
NS(ParticlesContainer_get_const_block_infos_begin)(
    const NS(ParticlesContainer) *const SIXTRL_RESTRICT particle_buffer )
{
    return NS(BlocksContainer_get_const_block_infos_begin)( particle_buffer );
}

SIXTRL_INLINE NS(BlockInfo) const* 
NS(ParticlesContainer_get_const_block_infos_end)(
    const NS(ParticlesContainer) *const SIXTRL_RESTRICT particle_buffer )
{
    return NS(BlocksContainer_get_const_block_infos_end)( particle_buffer );
}


SIXTRL_INLINE NS(BlockInfo)* NS(ParticlesContainer_get_block_infos_begin)(
    NS(ParticlesContainer)* SIXTRL_RESTRICT particle_buffer )
{
    return NS(BlocksContainer_get_block_infos_begin)( particle_buffer );
}

SIXTRL_INLINE NS(BlockInfo)* NS(ParticlesContainer_get_block_infos_end)(
    NS(ParticlesContainer)* SIXTRL_RESTRICT particle_buffer )
{
    return NS(BlocksContainer_get_block_infos_end)( particle_buffer );
}

SIXTRL_INLINE NS(BlockInfo) NS(ParticlesContainer_get_block_info_by_index)(
    const NS(ParticlesContainer) *const SIXTRL_RESTRICT particle_buffer, 
    NS(block_size_t) const block_index )
{
    return NS(BlocksContainer_get_block_info_by_index)( 
        particle_buffer, block_index );
}

SIXTRL_INLINE NS(BlockInfo) const* 
NS(ParticlesContainer_get_const_ptr_to_block_info_by_index)(
    const NS(ParticlesContainer) *const SIXTRL_RESTRICT particle_buffer, 
    NS(block_size_t) const block_index )
{
    return NS(BlocksContainer_get_const_ptr_to_block_info_by_index)(
        particle_buffer, block_index );
}

SIXTRL_INLINE NS(BlockInfo)* 
NS(ParticlesContainer_get_ptr_to_block_info_by_index)(
    NS(ParticlesContainer)* SIXTRL_RESTRICT particle_buffer, 
    NS(block_size_t) const block_index )
{
    return NS(BlocksContainer_get_ptr_to_block_info_by_index)(
        particle_buffer, block_index );
}
    

SIXTRL_INLINE int NS(ParticlesContainer_add_particles)( 
    NS(ParticlesContainer)* SIXTRL_RESTRICT particles_buffer, 
    NS(Particles)* SIXTRL_RESTRICT particle_block,
    NS(block_num_elements_t) const num_of_particles )
{
    int success = -1;
    
    NS(MemPool)* ptr_info_store = 0;
    NS(MemPool)* ptr_data_store = 0;
    
    NS(MemPool) rollback_info_store;
    NS(MemPool) rollback_data_store;
        
    if( ( !NS(ParticlesContainer_has_data_store)( particles_buffer ) ) ||
        ( !NS(ParticlesContainer_has_info_store)( particles_buffer ) ) )
    {
        return success;
    }
    
    ptr_info_store = 
        NS(ParticlesContainer_get_ptr_info_store)( particles_buffer );
    
    ptr_data_store = 
        NS(ParticlesContainer_get_ptr_data_store)( particles_buffer );
    
    SIXTRL_ASSERT( ( ptr_info_store != 0 ) && ( ptr_data_store != 0 ) );
        
    rollback_info_store = *ptr_data_store;
    rollback_data_store = *ptr_info_store;    
    
    if( ( particles_buffer != 0 ) && 
        ( NS(ParticlesContainer_get_block_capacity)( particles_buffer ) >
          NS(ParticlesContainer_get_num_of_blocks)( particles_buffer ) ) )
    {
        static NS(block_size_t) const INFO_SIZE = sizeof( NS(BlockInfo ) );
        
        NS(block_alignment_t) const info_align = 
            NS(ParticlesContainer_get_info_alignment)( particles_buffer );
            
        NS(block_alignment_t) const data_align =
            NS(ParticlesContainer_get_data_alignment)( particles_buffer );
        
        NS(AllocResult) info_result = NS(MemPool_append_aligned)(
            ptr_info_store, INFO_SIZE, info_align );
        
        NS(block_size_t) const mem_offset = NS(MemPool_get_next_begin_offset)( 
            ptr_data_store, data_align );
        
        NS(block_size_t) const max_num_bytes_on_mem = 
            NS(MemPool_get_capacity)( ptr_data_store );
        
        if( ( NS(AllocResult_valid)( &info_result ) ) &&
            ( max_num_bytes_on_mem > mem_offset ) )
        {
            NS(BlockType) const type_id = NS(BLOCK_TYPE_PARTICLE);
            
            NS(BlockInfo)* block_info = 
                ( NS(BlockInfo)* )NS(AllocResult_get_pointer)( &info_result );
            
            unsigned char* data_mem_begin = 
                NS(ParticlesContainer_get_ptr_data_begin)( particles_buffer );
                
            NS(BlockInfo_set_mem_offset)( block_info, mem_offset );
            NS(BlockInfo_set_type_id)( block_info, type_id );
            NS(BlockInfo_set_common_alignment)( block_info, data_align );
            NS(BlockInfo_set_num_elements)( block_info, 
                                            ( NS(block_num_elements_t) )1u );
            
            NS(Particles_preset)( particle_block );
            NS(Particles_set_num_particles)( particle_block, num_of_particles );
            NS(Particles_set_type_id)( particle_block, NS(BLOCK_TYPE_PARTICLE ) );
            
            if( 0 == NS(Particles_create_on_memory)( particle_block, block_info, 
                    data_mem_begin, max_num_bytes_on_mem ) )
            {
                ++particles_buffer->num_blocks;
                
                NS(MemPool_increment_size)( ptr_data_store, 
                    NS(BlockInfo_get_mem_offset)( block_info ) + 
                    NS(BlockInfo_get_num_of_bytes)( block_info ) );
                
                success = 0;
                
                return success;
            }
        }
        
        /* if we are here, something went wrong -> rollback and return an 
         * invalid drift! */
        
        SIXTRL_ASSERT( NS(Particles_get_type_id)( particle_block ) == 
                       NS(BLOCK_TYPE_INVALID ) );
        
        SIXTRL_ASSERT( ( ptr_info_store != 0 ) && ( ptr_data_store != 0 ) );
        
        *ptr_data_store = rollback_data_store;        
        *ptr_info_store = rollback_info_store;
    }
    
    NS(Particles_preset)( particle_block );
    return success;
}


SIXTRL_INLINE int NS(ParticlesContainer_get_particles)(
    NS(Particles)* SIXTRL_RESTRICT particles,
    NS(ParticlesContainer)* SIXTRL_RESTRICT particles_buffer,
    NS(block_size_t) const block_index )
{
    int status = -1;
    
    NS(Particles_preset)( particles );
    
    if( block_index < 
        NS(ParticlesContainer_get_num_of_blocks)( particles_buffer ) )
    {
        status = NS(Particles_remap_from_memory)( particles, 
            NS(ParticlesContainer_get_const_ptr_to_block_info_by_index)(
                particles_buffer, block_index ), 
            NS(ParticlesContainer_get_ptr_data_begin)( particles_buffer ),
            NS(ParticlesContainer_get_data_capacity)( particles_buffer ) );
    }
    
    return status;
}


    
#if !defined( _GPUCODE )
#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* !defined( _GPUCODE ) */

#endif /* SIXTRACKLIB_COMMON_PARTICLES_H__ */

/* end: sixtracklib/common/particles.h */
