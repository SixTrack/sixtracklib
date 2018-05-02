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

SIXTRL_STATIC NS(BlockInfo)* NS(ParticlesContainer_get_infos_end)(
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

int NS(ParticlesContainer_add_particles)( 
    NS(ParticlesContainer)* SIXTRL_RESTRICT particles_buffer, 
    NS(Particles)* SIXTRL_RESTRICT particle_block,
    NS(block_num_elements_t) const num_of_particles );

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

SIXTRL_INLINE NS(BlockInfo)* NS(ParticlesContainer_get_infos_end)(
    NS(ParticlesContainer)* SIXTRL_RESTRICT particle_buffer )
{
    return NS(BlocksContainer_get_infos_end)( particle_buffer );
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
    
    
#if !defined( _GPUCODE )
#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* !defined( _GPUCODE ) */

#endif /* SIXTRACKLIB_COMMON_PARTICLES_H__ */

/* end: sixtracklib/common/particles.h */
