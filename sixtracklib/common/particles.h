#ifndef SIXTRACKLIB_COMMON_PARTICLES_H__
#define SIXTRACKLIB_COMMON_PARTICLES_H__

#if !defined( _GPUCODE )

#include "sixtracklib/_impl/definitions.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "sixtracklib/common/impl/particles_type.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */
    
#endif /* !defined( _GPUCODE ) */

/* ------------------------------------------------------------------------- */

struct NS(Blocks);
    
/* ------------------------------------------------------------------------- */

SIXTRL_FN SIXTRL_STATIC NS(BlockType) NS(Particles_get_type_id)(
    const NS(Particles) *const SIXTRL_RESTRICT particles );

SIXTRL_FN SIXTRL_STATIC NS(block_type_num_t) NS(Particles_get_type_id_num)(
    const NS(Particles) *const SIXTRL_RESTRICT particles );

SIXTRL_FN SIXTRL_STATIC NS(block_type_num_t) NS(Particles_predict_blocks_data_capacity)(
    const NS(Blocks) *const SIXTRL_RESTRICT blocks, 
    NS(block_size_t) const num_of_blocks, 
    NS(block_size_t) const num_of_particles );

/* ------------------------------------------------------------------------- */

SIXTRL_FN SIXTRL_STATIC int NS(Particles_is_valid)(
    const NS(Particles) *const SIXTRL_RESTRICT particles );

SIXTRL_FN SIXTRL_STATIC int NS(Particles_has_mapping)(
    const NS(Particles) *const SIXTRL_RESTRICT particles );

SIXTRL_FN SIXTRL_STATIC int NS(Particles_is_aligned_with)( 
    const NS(Particles) *const SIXTRL_RESTRICT particles, 
    NS(block_size_t) const alignment );

/* ------------------------------------------------------------------------- */

#if !defined( _GPUCODE )

SIXTRL_HOST_FN SIXTRL_GLOBAL_DEC NS(Particles)* 
NS(Blocks_add_particles)( NS(Blocks)* SIXTRL_RESTRICT blocks, 
    NS(block_num_elements_t) const num_of_particles );

#endif /* !defiend( _GPUCODE ) */

/* ========================================================================= */
/* =====             Implementation of inline functions                ===== */
/* ========================================================================= */

#if !defined( _GPUCODE )

#include "sixtracklib/common/blocks.h"
#include "sixtracklib/common/impl/particles_api.h"

#endif /* !defined( _GPUCODE ) */

SIXTRL_INLINE NS(BlockType) NS(Particles_get_type_id)(
    const NS(Particles) *const SIXTRL_RESTRICT particles )
{
    return NS(BLOCK_TYPE_PARTICLE);
}

SIXTRL_INLINE NS(block_type_num_t) NS(Particles_get_type_id_num)(
    const NS(Particles) *const SIXTRL_RESTRICT particles )
{
    return NS(BlockType_to_number)( NS(BLOCK_TYPE_PARTICLE) );
}

SIXTRL_INLINE NS(block_type_num_t) NS(Particles_predict_blocks_data_capacity)(
    const NS(Blocks) *const SIXTRL_RESTRICT blocks, 
    NS(block_size_t) const num_of_blocks,
    NS(block_size_t) const num_of_particles )
{
    NS(block_size_t) attr_data_capacity = ( NS(block_size_t) )0u;
    
    if( ( blocks != 0 ) && ( num_of_particles > 0u ) && 
        ( num_of_blocks > 0u ) )
    {
        SIXTRL_STATIC NS(block_size_t) const NUM_REAL_ATTRIBUTES = 16u;        
        SIXTRL_STATIC NS(block_size_t) const NUM_I64_ATTRIBUTES  =  4u;
        
        NS(block_size_t) const alignment = 
            NS(Blocks_get_data_alignment)( blocks );
            
        NS(block_size_t) const REAL_ATTRIBUTE_SIZE =
                num_of_particles * sizeof( SIXTRL_REAL_T );
                
        NS(block_size_t) const I64_ATTRIBUTE_SIZE =
            num_of_particles * sizeof( SIXTRL_INT64_T );
            
        attr_data_capacity = alignment + num_of_blocks * (
            NUM_REAL_ATTRIBUTES * REAL_ATTRIBUTE_SIZE +
            NUM_I64_ATTRIBUTES  * I64_ATTRIBUTE_SIZE  +
            alignment + sizeof( NS(Particles) ) +
            alignment + num_of_particles * ( 
                NUM_REAL_ATTRIBUTES + NUM_I64_ATTRIBUTES ) * 
                    sizeof( SIXTRL_GLOBAL_DEC void** ) );
    }
        
    return attr_data_capacity;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE int NS(Particles_is_valid)(
    const NS(Particles) *const SIXTRL_RESTRICT particles )
{
    return ( 
        ( particles != 0 ) &&
        ( NS(Particles_get_type_id)( particles ) == 
          NS(BLOCK_TYPE_PARTICLE) ) &&
        ( NS(Particles_get_num_particles)( particles) >
          ( NS(block_num_elements_t) )0u ) &&
        ( NS(Particles_has_mapping)( particles ) ) ) ? 1 : 0;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE int NS(Particles_has_mapping)(
    const NS(Particles) *const SIXTRL_RESTRICT particles )
{
    return ( 
        ( particles != 0 ) && 
        ( NS(Particles_get_const_q0)( particles )                 != 0 ) &&
        ( NS(Particles_get_const_mass0)( particles )              != 0 ) &&
        ( NS(Particles_get_const_beta0)( particles )              != 0 ) &&
        ( NS(Particles_get_const_gamma0)( particles )             != 0 ) &&
        ( NS(Particles_get_const_p0c)( particles )                != 0 ) &&
        ( NS(Particles_get_const_s)( particles )                  != 0 ) &&
        ( NS(Particles_get_const_x)( particles )                  != 0 ) &&
        ( NS(Particles_get_const_y)( particles )                  != 0 ) &&
        ( NS(Particles_get_const_px)( particles )                 != 0 ) &&
        ( NS(Particles_get_const_py)( particles )                 != 0 ) &&
        ( NS(Particles_get_const_sigma)( particles )              != 0 ) &&
        ( NS(Particles_get_const_psigma)( particles )             != 0 ) &&
        ( NS(Particles_get_const_delta)( particles )              != 0 ) &&
        ( NS(Particles_get_const_rpp)( particles )                != 0 ) &&
        ( NS(Particles_get_const_rvv)( particles )                != 0 ) &&
        ( NS(Particles_get_const_chi)( particles )                != 0 ) &&
        ( NS(Particles_get_const_particle_id)( particles )        != 0 ) &&
        ( NS(Particles_get_const_lost_at_element_id)( particles ) != 0 ) &&
        ( NS(Particles_get_const_lost_at_turn)( particles )       != 0 ) &&
        ( NS(Particles_get_const_state)( particles )              != 0 ) ) ? 1 : 0;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE int NS(Particles_is_aligned_with)( 
    const NS(Particles)  *const SIXTRL_RESTRICT particles, 
    NS(block_type_num_t) const align )
{
    typedef NS(block_type_num_t) align_t;
    
    SIXTRL_STATIC uintptr_t const ZERO = ( uintptr_t )0u;
    
    return ( 
        ( particles != 0 ) &&
        ( align > ( align_t )0u ) &&
        ( ( align % ( align_t )2u ) == ( align_t)0u ) &&
        ( NS(Particles_has_mapping)( particles ) ) &&
        ( ( ( ( uintptr_t )NS(Particles_get_const_q0)( particles ) ) 
            % align ) == ZERO ) &&
        ( ( ( ( uintptr_t )NS(Particles_get_const_mass0)( particles ) ) 
            % align ) == ZERO ) &&
        ( ( ( ( uintptr_t )NS(Particles_get_const_beta0)( particles ) ) 
            % align ) == ZERO ) &&
        ( ( ( ( uintptr_t )NS(Particles_get_const_gamma0)( particles ) ) 
            % align ) == ZERO ) &&
        ( ( ( ( uintptr_t )NS(Particles_get_const_p0c)( particles ) ) 
            % align ) == ZERO ) &&
        ( ( ( ( uintptr_t )NS(Particles_get_const_s)( particles ) ) 
            % align ) == ZERO ) &&
        ( ( ( ( uintptr_t )NS(Particles_get_const_x)( particles ) ) 
            % align ) == ZERO ) &&
        ( ( ( ( uintptr_t )NS(Particles_get_const_y)( particles ) ) 
            % align ) == ZERO ) &&
        ( ( ( ( uintptr_t )NS(Particles_get_const_px)( particles ) ) 
            % align ) == ZERO ) &&
        ( ( ( ( uintptr_t )NS(Particles_get_const_py)( particles ) ) 
            % align ) == ZERO ) &&
        ( ( ( ( uintptr_t )NS(Particles_get_const_sigma)( particles ) ) 
            % align ) == ZERO ) &&
        ( ( ( ( uintptr_t )NS(Particles_get_const_psigma)( particles ) ) 
            % align ) == ZERO ) &&
        ( ( ( ( uintptr_t )NS(Particles_get_const_delta)( particles ) ) 
            % align ) == ZERO ) &&
        ( ( ( ( uintptr_t )NS(Particles_get_const_rpp)( particles ) ) 
            % align ) == ZERO ) &&
        ( ( ( ( uintptr_t )NS(Particles_get_const_rvv)( particles ) ) 
            % align ) == ZERO ) &&
        ( ( ( ( uintptr_t )NS(Particles_get_const_chi)( particles ) ) 
            % align ) == ZERO ) &&
        ( ( ( ( uintptr_t )NS(Particles_get_const_particle_id)( particles ) ) 
            % align ) == ZERO ) &&
        ( ( ( ( uintptr_t )NS(Particles_get_const_lost_at_element_id)( 
            particles ) ) % align ) == ZERO ) &&
        ( ( ( ( uintptr_t )NS(Particles_get_const_lost_at_turn)( particles ) ) 
            % align ) == ZERO ) &&
        ( ( ( ( uintptr_t )NS(Particles_get_const_state)( particles ) ) 
            % align ) == ZERO ) ) ? 1 : 0;
}

/* ------------------------------------------------------------------------- */

#if !defined( _GPUCODE )
#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* !defined( _GPUCODE ) */

#endif /* SIXTRACKLIB_COMMON_PARTICLES_H__ */

/* end: sixtracklib/common/particles.h */
