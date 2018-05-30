#ifndef SIXTRACKLIB_COMMON_PARTICLES_H__
#define SIXTRACKLIB_COMMON_PARTICLES_H__

#if !defined( _GPUCODE )

#include "sixtracklib/_impl/definitions.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "sixtracklib/common/blocks.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */
    
#endif /* !defined( _GPUCODE ) */

/* ------------------------------------------------------------------------- */
    
typedef struct NS(Particles)
{
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT 
        q0     __attribute__(( aligned( 8 ) ));     /* C */
    
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT 
        mass0  __attribute__(( aligned( 8 ) ));  /* eV */
        
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT 
        beta0  __attribute__(( aligned( 8 ) ));  /* nounit */
        
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT 
        gamma0 __attribute__(( aligned( 8 ) )); /* nounit */
        
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT 
        p0c    __attribute__(( aligned( 8 ) ));    /* eV */

    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT 
        s      __attribute__(( aligned( 8 ) ));     /* [m] */
        
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT 
        x      __attribute__(( aligned( 8 ) ));     /* [m] */
    
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT 
        y      __attribute__(( aligned( 8 ) ));     /* [m] */
        
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT 
        px     __attribute__(( aligned( 8 ) ));    /* Px/P0 */
        
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT 
        py     __attribute__(( aligned( 8 ) ));    /* Py/P0 */
        
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT 
        sigma  __attribute__(( aligned( 8 ) )); 
            /* s-beta0*c*t  where t is the time
                      since the beginning of the simulation */

    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT 
        psigma __attribute__(( aligned( 8 ) )); /* (E-E0) / (beta0 P0c) 
            conjugate of sigma */
            
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT 
        delta  __attribute__(( aligned( 8 ) ));  /* P/P0-1 = 1/rpp-1 */
        
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT 
        rpp    __attribute__(( aligned( 8 ) ));    /* ratio P0 /P */
        
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT 
        rvv    __attribute__(( aligned( 8 ) ));    /* ratio beta / beta0 */
        
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT 
        chi    __attribute__(( aligned( 8 ) ));    /* q/q0 * m/m0  */

    SIXTRL_GLOBAL_DEC SIXTRL_INT64_T* SIXTRL_RESTRICT 
        particle_id __attribute__(( aligned( 8 ) ));
    
    SIXTRL_GLOBAL_DEC SIXTRL_INT64_T* SIXTRL_RESTRICT 
        lost_at_element_id __attribute__(( aligned( 8 ) )); /* element at 
            which the particle was lost */
        
    SIXTRL_GLOBAL_DEC SIXTRL_INT64_T* SIXTRL_RESTRICT 
        lost_at_turn __attribute__(( aligned( 8 ) )); /* turn at which the 
            particle was lost */
            
    SIXTRL_GLOBAL_DEC SIXTRL_INT64_T* SIXTRL_RESTRICT 
        state __attribute__(( aligned( 8 ) )); /* negative means particle */
    
    NS(block_num_elements_t) num_of_particles  __attribute__(( aligned( 8 ) ));   
}
NS(Particles);
    
/* ------------------------------------------------------------------------- */

SIXTRL_STATIC NS(Particles)* NS(Particles_preset)( 
    NS(Particles)* SIXTRL_RESTRICT particles );

SIXTRL_STATIC NS(block_num_elements_t) NS(Particles_get_num_particles)( 
    const NS(Particles) *const SIXTRL_RESTRICT particles );

SIXTRL_STATIC void NS(Particles_set_num_particles)( 
    NS(Particles)* SIXTRL_RESTRICT particles,
    NS(block_num_elements_t) const num_of_particles );

SIXTRL_STATIC NS(BlockType) NS(Particles_get_type_id)(
    const NS(Particles) *const SIXTRL_RESTRICT particles );

SIXTRL_STATIC NS(block_type_num_t) NS(Particles_get_type_id_num)(
    const NS(Particles) *const SIXTRL_RESTRICT particles );

SIXTRL_STATIC NS(block_type_num_t) NS(Particles_predict_blocks_data_capacity)(
    const NS(Blocks) *const SIXTRL_RESTRICT blocks,     
    NS(block_size_t) const num_of_particles );

/* ------------------------------------------------------------------------- */

SIXTRL_STATIC void NS( Particles_copy_single_unchecked )( 
    struct NS( Particles ) * SIXTRL_RESTRICT des, 
    NS(block_num_elements_t) const des_id,
    const struct NS( Particles ) *const SIXTRL_RESTRICT src, 
    NS(block_num_elements_t) const src_id );

SIXTRL_STATIC void NS( Particles_copy_all_unchecked )(
    struct NS( Particles ) * SIXTRL_RESTRICT des,
    const struct NS( Particles ) *const SIXTRL_RESTRICT src );

/* ------------------------------------------------------------------------- */

SIXTRL_STATIC SIXTRL_GLOBAL_DEC NS(Particles) const* 
NS(Blocks_get_const_particles)(
    const NS(BlockInfo) *const SIXTRL_RESTRICT block_info );

SIXTRL_STATIC SIXTRL_GLOBAL_DEC NS(Particles)* NS(Blocks_get_particles)(
    NS(BlockInfo)* SIXTRL_RESTRICT block_info );

/* ------------------------------------------------------------------------- */

SIXTRL_STATIC int NS(Particles_is_valid)(
    const NS(Particles) *const SIXTRL_RESTRICT particles );

SIXTRL_STATIC int NS(Particles_has_mapping)(
    const NS(Particles) *const SIXTRL_RESTRICT particles );

SIXTRL_STATIC int NS(Particles_is_aligned_with)( 
    const NS(Particles) *const SIXTRL_RESTRICT particles, 
    NS(block_size_t) const alignment );

/* ------------------------------------------------------------------------- */

#if !defined( _GPUCODE )

SIXTRL_GLOBAL_DEC NS(Particles)* 
NS(Blocks_add_particles)( NS(Blocks)* SIXTRL_RESTRICT blocks, 
    NS(block_num_elements_t) const num_of_particles );

#endif /* !defiend( _GPUCODE ) */

/* ========================================================================= */
/* =====             Implementation of inline functions                ===== */
/* ========================================================================= */

#include "sixtracklib/common/impl/particles_api.h"

SIXTRL_INLINE NS(Particles)* NS(Particles_preset)( 
    NS(Particles)* SIXTRL_RESTRICT particles )
{
    if( particles != 0 )
    {
        particles->q0                 = 0;    
        particles->mass0              = 0; 
        particles->beta0              = 0; 
        particles->gamma0             = 0;
        particles->p0c                = 0;   
        particles->s                  = 0;     
        particles->x                  = 0;     
        particles->px                 = 0;    
        particles->y                  = 0;     
        particles->py                 = 0;    
        particles->sigma              = 0; 
        particles->psigma             = 0;
        particles->delta              = 0;
        particles->rpp                = 0;
        particles->rvv                = 0;
        particles->chi                = 0;

        particles->particle_id        = 0;
        particles->lost_at_element_id = 0;
        particles->lost_at_turn       = 0;
        particles->state              = 0;
        
        NS(Particles_set_num_particles)( 
            particles, ( NS(block_num_elements_t) )0 );        
    }
    
    return particles;
}

SIXTRL_INLINE NS(block_num_elements_t) NS(Particles_get_num_particles)( 
    const NS(Particles) *const SIXTRL_RESTRICT particles )
{
    return ( particles != 0 ) ? particles->num_of_particles : 0;
}

SIXTRL_INLINE void NS(Particles_set_num_particles)( 
    NS(Particles)* SIXTRL_RESTRICT particles, 
    NS(block_num_elements_t) const num_of_particles )
{
    if( particles != 0 ) particles->num_of_particles = num_of_particles;
    return;
}

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
    NS(block_size_t) const num_of_particles )
{
    NS(block_size_t) attr_data_capacity = ( NS(block_size_t) )0u;
    
    if( ( blocks != 0 ) && ( num_of_particles > 0u ) )
    {
        SIXTRL_STATIC NS(block_size_t) const NUM_REAL_ATTRIBUTES = 16u;        
        SIXTRL_STATIC NS(block_size_t) const NUM_I64_ATTRIBUTES  =  4u;
        
        NS(block_size_t) const alignment = 
            NS(Blocks_get_data_alignment)( blocks );
            
        NS(block_size_t) const REAL_ATTRIBUTE_SIZE =
                num_of_particles * sizeof( SIXTRL_REAL_T );
                
        NS(block_size_t) const I64_ATTRIBUTE_SIZE =
            num_of_particles * sizeof( SIXTRL_INT64_T );
            
        attr_data_capacity = alignment +
            NUM_REAL_ATTRIBUTES * REAL_ATTRIBUTE_SIZE +
            NUM_I64_ATTRIBUTES  * I64_ATTRIBUTE_SIZE  +
            alignment + sizeof( NS(Particles) ) +
            alignment + num_of_particles * ( 
                NUM_REAL_ATTRIBUTES + NUM_I64_ATTRIBUTES ) * 
                    sizeof( SIXTRL_GLOBAL_DEC void** );
    }
        
    return attr_data_capacity;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE void NS( Particles_copy_single_unchecked )( 
    struct NS( Particles ) * SIXTRL_RESTRICT des, 
    NS(block_num_elements_t) const des_id,
    const struct NS( Particles ) *const SIXTRL_RESTRICT src, 
    NS(block_num_elements_t) const src_id )
{
    SIXTRL_ASSERT( ( des != 0 ) && ( src != 0 ) &&
                   ( NS(Particles_get_num_particles)( des ) > des_id ) &&
                   ( NS(Particles_get_num_particles)( src ) > src_id ) );
    
    NS( Particles_set_q0_value )
        ( des, des_id, NS( Particles_get_q0_value )( src, src_id ) );

    NS( Particles_set_mass0_value )
        ( des, des_id, NS( Particles_get_mass0_value )( src, src_id ) );

    NS( Particles_set_beta0_value )
        ( des, des_id, NS( Particles_get_beta0_value )( src, src_id ) );

    NS( Particles_set_gamma0_value )
        ( des, des_id, NS( Particles_get_gamma0_value )( src, src_id ) );

    NS( Particles_set_p0c_value )
        ( des, des_id, NS( Particles_get_p0c_value )( src, src_id ) );

    NS( Particles_set_s_value )
        ( des, des_id, NS( Particles_get_s_value )( src, src_id ) );

    NS( Particles_set_x_value )
        ( des, des_id, NS( Particles_get_x_value )( src, src_id ) );

    NS( Particles_set_y_value )
        ( des, des_id, NS( Particles_get_y_value )( src, src_id ) );

    NS( Particles_set_px_value )
        ( des, des_id, NS( Particles_get_px_value )( src, src_id ) );

    NS( Particles_set_py_value )
        ( des, des_id, NS( Particles_get_py_value )( src, src_id ) );

    NS( Particles_set_sigma_value )
        ( des, des_id, NS( Particles_get_sigma_value )( src, src_id ) );

    NS( Particles_set_psigma_value )
        ( des, des_id, NS( Particles_get_psigma_value )( src, src_id ) );

    NS( Particles_set_delta_value )
        ( des, des_id, NS( Particles_get_delta_value )( src, src_id ) );

    NS( Particles_set_rpp_value )
        ( des, des_id, NS( Particles_get_rpp_value )( src, src_id ) );

    NS( Particles_set_rvv_value )
        ( des, des_id, NS( Particles_get_rvv_value )( src, src_id ) );

    NS( Particles_set_chi_value )
        ( des, des_id, NS( Particles_get_chi_value )( src, src_id ) );
    
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

SIXTRL_INLINE void NS( Particles_copy_all_unchecked )(
    struct NS( Particles ) * SIXTRL_RESTRICT des,
    const struct NS( Particles ) *const SIXTRL_RESTRICT src )
{
    SIXTRL_ASSERT( 
        ( des != 0 ) && ( src != 0 ) &&
        ( NS(Particles_get_num_particles)( des ) == 
          NS(Particles_get_num_particles( src ) ) ) );
    
    NS( Particles_set_q0     )( des, NS( Particles_get_const_q0     )( src ) );
    NS( Particles_set_mass0  )( des, NS( Particles_get_const_mass0  )( src ) );
    NS( Particles_set_beta0  )( des, NS( Particles_get_const_beta0  )( src ) );
    NS( Particles_set_gamma0 )( des, NS( Particles_get_const_gamma0 )( src ) );
    NS( Particles_set_p0c    )( des, NS( Particles_get_const_p0c    )( src ) );
    
    NS( Particles_set_s      )( des, NS( Particles_get_const_s      )( src ) );
    NS( Particles_set_x      )( des, NS( Particles_get_const_x      )( src ) );
    NS( Particles_set_y      )( des, NS( Particles_get_const_y      )( src ) );
    NS( Particles_set_px     )( des, NS( Particles_get_const_px     )( src ) );
    NS( Particles_set_py     )( des, NS( Particles_get_const_py     )( src ) );
    NS( Particles_set_sigma  )( des, NS( Particles_get_const_sigma  )( src ) );
    
    NS( Particles_set_state  )( des, NS( Particles_get_const_state  )( src ) );        
    NS( Particles_set_psigma )( des, NS( Particles_get_const_psigma )( src ) );
    NS( Particles_set_delta  )( des, NS( Particles_get_const_delta  )( src ) );
    NS( Particles_set_rpp    )( des, NS( Particles_get_const_rpp    )( src ) );
    NS( Particles_set_rvv    )( des, NS( Particles_get_const_rvv    )( src ) );
    NS( Particles_set_chi    )( des, NS( Particles_get_const_chi    )( src ) );
    
    NS( Particles_set_particle_id )( des, 
        NS( Particles_get_const_particle_id )( src ) );
    
    NS( Particles_set_lost_at_element_id )( des, 
        NS( Particles_get_const_lost_at_element_id)( src ) );
    
    NS( Particles_set_lost_at_turn)( des, 
        NS(Particles_get_const_lost_at_turn)( src ) );
    
    return;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE NS(Particles) const* NS(Blocks_get_const_particles)(
    const NS(BlockInfo) *const SIXTRL_RESTRICT block_info )
{
    return ( ( block_info != 0 ) && 
             ( NS(BlockInfo_get_const_ptr_begin)( block_info ) != 0 ) &&
             ( ( ( ( uintptr_t )NS(BlockInfo_get_const_ptr_begin)( block_info ) 
                 ) % 8u ) == 0u ) ) 
        ? ( SIXTRL_GLOBAL_DEC NS(Particles) const*                
            )NS(BlockInfo_get_const_ptr_begin)( block_info )
        : 0;
}

SIXTRL_INLINE SIXTRL_GLOBAL_DEC NS(Particles)* NS(Blocks_get_particles)(
    NS(BlockInfo)* SIXTRL_RESTRICT block_info )
{
    return ( SIXTRL_GLOBAL_DEC NS(Particles)*            
        )NS(Blocks_get_const_particles)( block_info );
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
