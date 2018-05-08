#ifndef SIXTRACKLIB_MPFR4_IMPL_PARTICLES_IMPL_H__
#define SIXTRACKLIB_MPFR4_IMPL_PARTICLES_IMPL_H__

#include <stdint.h>
#include <stdio.h>
#include <stdarg.h>

#include <mpfr.h>

#include "sixtracklib/mpfr4/track.h"
#include "sixtracklib/common/impl/particles_impl.h"
#include "sixtracklib/common/particles.h"
#include "sixtracklib/common/impl/block_info_impl.h"
    
SIXTRL_STATIC int NS(Particles_init_mpfr4)(
    NS(Particles)* SIXTRL_RESTRICT particles, 
    mpfr_prec_t const prec );

SIXTRL_STATIC void NS(Particles_clear_mpfr4)(
    NS(Particles)* SIXTRL_RESTRICT particles );

SIXTRL_STATIC void NS( Particles_copy_single_unchecked_mpfr4 )( 
    struct NS( Particles ) * SIXTRL_RESTRICT des, 
    NS(block_num_elements_t) const des_id,
    const struct NS( Particles ) *const SIXTRL_RESTRICT src, 
    NS(block_num_elements_t) const src_id, 
    mpfr_rnd_t const rnd );

SIXTRL_STATIC void NS( Particles_copy_all_unchecked_mpfr4 )(
    struct NS( Particles ) * SIXTRL_RESTRICT des,
    const struct NS( Particles ) *const SIXTRL_RESTRICT src, 
    mpfr_rnd_t const rnd );

SIXTRL_STATIC int NS(ParticlesContainer_init_num_of_blocks_mpfr4)(
    NS(ParticlesContainer)* SIXTRL_RESTRICT particle_buffer, 
    NS(block_num_elements_t) const num_of_blocks, 
    NS(block_num_elements_t) const num_particles_per_block,
    mpfr_prec_t const prec );

SIXTRL_STATIC void NS(ParticlesContainer_free_mpfr4)(
    NS(ParticlesContainer)* SIXTRL_RESTRICT particle_buffer );

/* ========================================================================= */
/* =====             Implementation of inline functions                ===== */
/* ========================================================================= */

SIXTRL_INLINE int NS(Particles_init_mpfr4)(
    NS(Particles)* SIXTRL_RESTRICT particles, mpfr_prec_t const prec )
{
    int success = -1;
    
    NS(block_num_elements_t) const NUM_PARTICLES = 
        NS(Particles_get_num_particles)( particles );
    
    if( ( particles != 0 ) && 
        ( NUM_PARTICLES > ( NS(block_num_elements_t) )0u ) )
    {
        NS(block_num_elements_t) ii = 0;
        
        SIXTRL_ASSERT( NS(Particles_has_mapping )( particles ) );
        SIXTRL_ASSERT( NS(Particles_is_aligned_with)( 
            particles, sizeof( SIXTRL_REAL_T ) ) );
        
        for( ii = 0 ; ii < NUM_PARTICLES ; ++ii )
        {
            mpfr_init2( NS(Particles_get_q0)( particles )[ ii ].value, prec );
            mpfr_init2( NS(Particles_get_mass0)( particles )[ ii ].value, prec );
            mpfr_init2( NS(Particles_get_beta0)( particles )[ ii ].value, prec );
            mpfr_init2( NS(Particles_get_gamma0)( particles )[ ii ].value, prec );
            mpfr_init2( NS(Particles_get_p0c)( particles )[ ii ].value, prec );
            mpfr_init2( NS(Particles_get_s)( particles )[ ii ].value, prec );
            mpfr_init2( NS(Particles_get_x)( particles )[ ii ].value, prec );
            mpfr_init2( NS(Particles_get_y)( particles )[ ii ].value, prec );
            mpfr_init2( NS(Particles_get_px)( particles )[ ii ].value, prec );
            mpfr_init2( NS(Particles_get_py)( particles )[ ii ].value, prec );
            mpfr_init2( NS(Particles_get_sigma)( particles )[ ii ].value, prec );
            mpfr_init2( NS(Particles_get_psigma)( particles )[ ii ].value, prec );
            mpfr_init2( NS(Particles_get_delta)( particles )[ ii ].value, prec );
            mpfr_init2( NS(Particles_get_rpp)( particles )[ ii ].value, prec );
            mpfr_init2( NS(Particles_get_rvv)( particles )[ ii ].value, prec );
            mpfr_init2( NS(Particles_get_chi)( particles )[ ii ].value, prec );
        }
        
        success = 0;
    }
    
    return success;
}


SIXTRL_INLINE void NS(Particles_clear_mpfr4)( 
    NS(Particles)* SIXTRL_RESTRICT particles )
{
    NS(block_num_elements_t) const NUM_PARTICLES = 
        NS(Particles_get_num_particles)( particles );
    
    if( ( particles != 0 ) && 
        ( NUM_PARTICLES > ( NS(block_num_elements_t) )0u ) )
    {
        NS(block_num_elements_t) ii = 0;
        
        SIXTRL_ASSERT( NS(Particles_has_mapping )( particles ) );
        SIXTRL_ASSERT( NS(Particles_is_aligned_with)( 
            particles, sizeof( SIXTRL_REAL_T ) ) );
        
        for( ii = 0 ; ii < NUM_PARTICLES ; ++ii )
        {
            mpfr_clear( st_Particles_get_q0( particles )[ ii ].value );
            mpfr_clear( st_Particles_get_mass0( particles )[ ii ].value );
            mpfr_clear( st_Particles_get_beta0( particles )[ ii ].value );
            mpfr_clear( st_Particles_get_gamma0( particles )[ ii ].value );
            mpfr_clear( st_Particles_get_p0c( particles )[ ii ].value );
            mpfr_clear( st_Particles_get_s( particles )[ ii ].value );
            mpfr_clear( st_Particles_get_x( particles )[ ii ].value );
            mpfr_clear( st_Particles_get_y( particles )[ ii ].value );
            mpfr_clear( st_Particles_get_px( particles )[ ii ].value );
            mpfr_clear( st_Particles_get_py( particles )[ ii ].value );
            mpfr_clear( st_Particles_get_sigma( particles )[ ii ].value );
            mpfr_clear( st_Particles_get_psigma( particles )[ ii ].value );
            mpfr_clear( st_Particles_get_delta( particles )[ ii ].value );
            mpfr_clear( st_Particles_get_rpp( particles )[ ii ].value );
            mpfr_clear( st_Particles_get_rvv( particles )[ ii ].value );
            mpfr_clear( st_Particles_get_chi( particles )[ ii ].value );
        }
    }
    
    return;
}

SIXTRL_INLINE void NS( Particles_copy_single_unchecked_mpfr4 )( 
    struct NS( Particles ) * SIXTRL_RESTRICT des, 
    NS(block_num_elements_t) const des_id,
    const struct NS( Particles ) *const SIXTRL_RESTRICT src, 
    NS(block_num_elements_t) const src_id, 
    mpfr_rnd_t const rnd )
{
    #if !defined( NDEBUG )
    
    NS(block_num_elements_t) const DEST_NUM_PARTICLES = 
        NS(Particles_get_num_particles)( des );
        
    NS(block_num_elements_t) const SRC_NUM_PARTICLES =
        NS(Particles_get_num_particles)( src );
    
    SIXTRL_ASSERT( ( des != 0 ) && ( src != 0 ) &&
                   ( DEST_NUM_PARTICLES > des_id ) &&
                   ( SRC_NUM_PARTICLES  > src_id ) );
    
    SIXTRL_ASSERT( NS(Particles_has_mapping )( des ) );
    SIXTRL_ASSERT( NS(Particles_is_aligned_with)( des, sizeof( SIXTRL_REAL_T ) ) );
    SIXTRL_ASSERT( NS(Particles_has_mapping )( src ) );
    SIXTRL_ASSERT( NS(Particles_is_aligned_with)( src, sizeof( SIXTRL_REAL_T ) ) );
    
    #endif /* !defined( NDEBUG ) */
    
    mpfr_set( NS(Particles_get_q0)( des )[ des_id ].value, 
              NS(Particles_get_const_q0)( src )[ src_id ].value, rnd );
    
    mpfr_set( NS(Particles_get_mass0)( des )[ des_id ].value, 
              NS(Particles_get_const_mass0)( src )[ src_id ].value, rnd );
    
    mpfr_set( NS(Particles_get_beta0)( des )[ des_id ].value, 
              NS(Particles_get_const_beta0)( src )[ src_id ].value, rnd );
    
    mpfr_set( NS(Particles_get_gamma0)( des )[ des_id ].value, 
              NS(Particles_get_const_gamma0)( src )[ src_id ].value, rnd );
    
    mpfr_set( NS(Particles_get_p0c)( des )[ des_id ].value, 
              NS(Particles_get_const_p0c)( src )[ src_id ].value, rnd );
    
    mpfr_set( NS(Particles_get_s)( des )[ des_id ].value, 
              NS(Particles_get_const_s)( src )[ src_id ].value, rnd );
    
    mpfr_set( NS(Particles_get_x)( des )[ des_id ].value, 
              NS(Particles_get_const_x)( src )[ src_id ].value, rnd );
    
    mpfr_set( NS(Particles_get_y)( des )[ des_id ].value, 
              NS(Particles_get_const_y)( src )[ src_id ].value, rnd );
    
    mpfr_set( NS(Particles_get_px)( des )[ des_id ].value, 
              NS(Particles_get_const_px)( src )[ src_id ].value, rnd );
    
    mpfr_set( NS(Particles_get_py)( des )[ des_id ].value, 
              NS(Particles_get_const_py)( src )[ src_id ].value, rnd );
    
    mpfr_set( NS(Particles_get_sigma)( des )[ des_id ].value, 
              NS(Particles_get_const_sigma)( src )[ src_id ].value, rnd );
    
    mpfr_set( NS(Particles_get_psigma)( des )[ des_id ].value, 
              NS(Particles_get_const_psigma)( src )[ src_id ].value, rnd );
    
    mpfr_set( NS(Particles_get_delta)( des )[ des_id ].value, 
              NS(Particles_get_const_delta)( src )[ src_id ].value, rnd );
    
    mpfr_set( NS(Particles_get_rpp)( des )[ des_id ].value, 
              NS(Particles_get_const_rpp)( src )[ src_id ].value, rnd );
    
    mpfr_set( NS(Particles_get_rvv)( des )[ des_id ].value, 
              NS(Particles_get_const_rvv)( src )[ src_id ].value, rnd );
    
    mpfr_set( NS(Particles_get_chi)( des )[ des_id ].value, 
              NS(Particles_get_const_chi)( src )[ src_id ].value, rnd );
    
    NS( Particles_set_particle_id_value )
    ( des, des_id, NS( Particles_get_particle_id_value )( src, src_id ) );

    NS( Particles_set_lost_at_element_id_value )
    ( des, des_id, 
      NS( Particles_get_lost_at_element_id_value )( src, src_id ) );

    NS( Particles_set_lost_at_turn_value )
    ( des, des_id, NS( Particles_get_lost_at_turn_value )( src, src_id ) );

    NS( Particles_set_state_value )
    ( des, des_id, NS( Particles_get_state_value )( src, src_id ) );
    
    
    return;
}

SIXTRL_INLINE void NS( Particles_copy_all_unchecked_mpfr4 )(
    struct NS( Particles ) * SIXTRL_RESTRICT des,
    const struct NS( Particles ) *const SIXTRL_RESTRICT src,
    mpfr_rnd_t const rnd )
{
    NS(block_num_elements_t) ii = 0;
    NS(block_num_elements_t) const NUM_PARTICLES = 
        NS(Particles_get_num_particles)( des );
        
    SIXTRL_ASSERT( 
        ( NUM_PARTICLES > ( NS( block_num_elements_t ) )0u ) &&
        ( NS(Particles_get_num_particles)( src ) == NUM_PARTICLES ) );
    
    for( ; ii < NUM_PARTICLES ; ++ii )
    {
        NS(Particles_copy_single_unchecked_mpfr4)( des, ii, src, ii, rnd );        
    }
    
    return;
}



SIXTRL_INLINE int NS(ParticlesContainer_init_num_of_blocks_mpfr4)(
    NS(ParticlesContainer)* SIXTRL_RESTRICT particle_buffer, 
    NS(block_num_elements_t) const num_of_blocks, 
    NS(block_num_elements_t) const num_particles_per_block,
    mpfr_prec_t const prec )
{
    int status = 0;
    
    NS(block_num_elements_t) ii = 0;
    
    static NS(block_size_t) const REAL_SIZE = sizeof( SIXTRL_REAL_T );
    static NS(block_size_t) const I64_SIZE  = sizeof( SIXTRL_INT64_T );
    NS(block_size_t) const PARTICLE_BLOCK_SIZE = 16 * REAL_SIZE + 4 * I64_SIZE;
    
    NS(block_size_t) const BLOCK_CAPACITY = num_of_blocks;
    NS(block_size_t) const DATA_CAPACITY  = 
        num_of_blocks * num_particles_per_block * PARTICLE_BLOCK_SIZE;
        
    if( ( BLOCK_CAPACITY > 0 ) && ( DATA_CAPACITY > 0 ) )
    {
        NS(Particles) particles;
        NS(Particles_preset)( &particles );
        
        NS(ParticlesContainer_clear)( particle_buffer );
        
        NS(ParticlesContainer_set_data_begin_alignment)( 
            particle_buffer, REAL_SIZE );
        
        NS(ParticlesContainer_set_data_alignment)( 
            particle_buffer, REAL_SIZE );
        
        NS(ParticlesContainer_reserve_num_blocks)( 
            particle_buffer, BLOCK_CAPACITY );
        
        NS(ParticlesContainer_reserve_for_data)( 
            particle_buffer, DATA_CAPACITY );
        
        for( ; ii < num_of_blocks ; ++ii )
        {
            status |= NS(ParticlesContainer_add_particles)( 
                particle_buffer, &particles, num_particles_per_block );
        
            if( status == 0 )
            {
                status = NS(Particles_init_mpfr4)( &particles, prec );
            }
            
            if( status != 0 )
            {
                break;
            }
        }
    }
    
    return status;
}

SIXTRL_INLINE void NS(ParticlesContainer_free_mpfr4)(
    NS(ParticlesContainer)* SIXTRL_RESTRICT particle_buffer )
{
    NS(BlockInfo)* block_info_it = 
        NS(ParticlesContainer_get_block_infos_begin)( particle_buffer );
        
    NS(BlockInfo)* block_info_end =
        NS(ParticlesContainer_get_block_infos_end)( particle_buffer );
        
    NS(Particles) particles;
        
    for( ; block_info_it != block_info_end ; ++block_info_it )
    {
        NS(Particles_preset)( &particles );
        
        int const ret = NS(Particles_remap_from_memory)(
            &particles, block_info_it, 
            NS(ParticlesContainer_get_ptr_data_begin)( particle_buffer ),
            NS(ParticlesContainer_get_data_capacity)( particle_buffer ) );
        
        if( ret == 0 )
        {
            NS(Particles_clear_mpfr4)( &particles );
        }
    }
    
    NS(ParticlesContainer_free)( particle_buffer );    
    return;
}


#if defined( __cplusplus )
}
#endif /* defined( __cplusplus ) */

#endif /* SIXTRACKLIB_MPFR4_IMPL_PARTICLES_IMPL_H__ */

/* end: sixtracklib/mpfr4/impl/particles_impl.h */
