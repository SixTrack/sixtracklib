#ifndef SIXTRACKLIB_COMMON_IMPL_PARTICLES_IMPL_H__
#define SIXTRACKLIB_COMMON_IMPL_PARTICLES_IMPL_H__

#if !defined( _GPUCODE )

#include "sixtracklib/_impl/definitions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "sixtracklib/_impl/namespace_begin.h"
#include "sixtracklib/common/impl/block_info_impl.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#endif /* !defined( _GPUCODE ) */

struct __attribute__ (( packed )) NS( Particles )
{
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT q0;     /* C */
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT mass0;  /* eV */
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT beta0;  /* nounit */
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT gamma0; /* nounit */
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT p0c;    /* eV */

    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT s;     /* [m] */
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT x;     /* [m] */
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT px;    /* Px/P0 */
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT y;     /* [m] */
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT py;    /* Py/P0 */
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT sigma; /* s-beta0*c*t  where t is the time
                      since the beginning of the simulation */

    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT psigma; /* (E-E0) / (beta0 P0c) conjugate of sigma */
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT delta;  /* P/P0-1 = 1/rpp-1 */
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT rpp;    /* ratio P0 /P */
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT rvv;    /* ratio beta / beta0 */
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT chi;    /* q/q0 * m/m0  */

    SIXTRL_GLOBAL_DEC SIXTRL_INT64_T* SIXTRL_RESTRICT particle_id;
    SIXTRL_GLOBAL_DEC SIXTRL_INT64_T* SIXTRL_RESTRICT lost_at_element_id; /* element at which the particle was lost */
    SIXTRL_GLOBAL_DEC SIXTRL_INT64_T* SIXTRL_RESTRICT lost_at_turn;   /* turn at which the particle was lost */
    SIXTRL_GLOBAL_DEC SIXTRL_INT64_T* SIXTRL_RESTRICT state;  /* negative means particle lost */
    
    NS(block_num_elements_t)  num_particles;
    NS(block_type_num_t)      type_id_num;    
};

typedef struct NS(Particles)
        NS( Particles ) __attribute__ ( ( aligned ) );

/* ------------------------------------------------------------------------- */

SIXTRL_STATIC NS(Particles)* NS(Particles_preset)( 
    NS(Particles)* SIXTRL_RESTRICT particles );

SIXTRL_STATIC NS(block_num_elements_t) NS(Particles_get_num_particles)( 
    const NS(Particles) *const SIXTRL_RESTRICT particles );

SIXTRL_STATIC void NS(Particles_set_num_particles)(
    NS(Particles)* SIXTRL_RESTRICT particles, 
    NS(block_num_elements_t) const num_particles_in_block );

SIXTRL_STATIC NS(block_type_num_t) NS(Particles_get_type_id_num)( 
    const NS(Particles) *const SIXTRL_RESTRICT particles );

SIXTRL_STATIC NS(BlockType) NS(Particles_get_type_id)( 
    const NS(Particles) *const SIXTRL_RESTRICT particles );

SIXTRL_STATIC void NS(Particles_set_type_id_num)(
    NS(Particles)* SIXTRL_RESTRICT particles,
    NS(block_type_num_t) const type_id_num );

SIXTRL_STATIC void NS(Particles_set_type_id)(
    NS(Particles)* SIXTRL_RESTRICT particles, NS(BlockType) const type_id );

SIXTRL_STATIC int NS(Particles_is_valid)(
    const NS(Particles) *const SIXTRL_RESTRICT particles );

SIXTRL_STATIC int NS(Particles_has_mapping)(
    const NS(Particles) *const SIXTRL_RESTRICT particles );

SIXTRL_STATIC int NS(Particles_is_aligned_with)( 
    const NS(Particles) *const SIXTRL_RESTRICT particles, 
    NS(block_alignment_t) const alignment );

SIXTRL_STATIC int NS(Particles_is_consistent)(
    const NS(Particles) *const SIXTRL_RESTRICT particles );

/* ------------------------------------------------------------------------ */

SIXTRL_STATIC void NS( Particles_copy_single_unchecked )( 
    struct NS( Particles ) * SIXTRL_RESTRICT des, 
    NS(block_num_elements_t) const des_id,
    const struct NS( Particles ) *const SIXTRL_RESTRICT src, 
    NS(block_num_elements_t) const src_id );

SIXTRL_STATIC void NS( Particles_copy_all_unchecked )(
    struct NS( Particles ) * SIXTRL_RESTRICT des,
    const struct NS( Particles ) *const SIXTRL_RESTRICT src );

/* ------------------------------------------------------------------------ */

SIXTRL_STATIC int NS(Particles_assign_all_ptrs)(
    struct NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_GLOBAL_DEC unsigned char** SIXTRL_RESTRICT attribute_ptrs, 
    NS(block_size_t) const num_attributes );

SIXTRL_STATIC int NS(Particles_create_on_memory)(
    struct NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_GLOBAL_DEC struct NS(BlockInfo)* SIXTRL_RESTRICT block_info, 
    SIXTRL_GLOBAL_DEC unsigned char* SIXTRL_RESTRICT mem_begin, 
    NS(block_size_t) max_num_bytes_on_mem );

SIXTRL_STATIC int NS(Particles_remap_from_memory)(
    struct NS(Particles)* SIXTRL_RESTRICT particles, 
    const SIXTRL_GLOBAL_DEC struct NS(BlockInfo) *const SIXTRL_RESTRICT block_info,
    SIXTRL_GLOBAL_DEC unsigned char* SIXTRL_RESTRICT mem_begin, 
    NS(block_size_t) const max_num_bytes_on_mem );

/* ------------------------------------------------------------------------- */

SIXTRL_STATIC SIXTRL_REAL_T NS(Particles_get_q0_value)(
    const NS(Particles) *const SIXTRL_RESTRICT particles, 
    NS(block_num_elements_t) const ii );

SIXTRL_STATIC SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const* 
NS(Particles_get_const_q0)( 
    const NS(Particles) *const SIXTRL_RESTRICT particles );

SIXTRL_STATIC SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* NS(Particles_get_q0)( 
    NS(Particles)* SIXTRL_RESTRICT particles );

SIXTRL_STATIC void NS(Particles_set_q0)(
    NS(Particles)* SIXTRL_RESTRICT particles, 
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT ptr_to_q0s );

SIXTRL_STATIC void NS(Particles_set_q0_value)(
    NS(Particles)* SIXTRL_RESTRICT particles, NS(block_num_elements_t) const ii, SIXTRL_REAL_T const q0_value );

SIXTRL_STATIC void NS(Particles_assign_ptr_to_q0)(
    NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* ptr_to_q0s );

/* ------------------------------------------------------------------------- */

SIXTRL_STATIC SIXTRL_REAL_T NS(Particles_get_mass0_value)(
    const NS(Particles) *const SIXTRL_RESTRICT particles, NS(block_num_elements_t) const ii );

SIXTRL_STATIC SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const* 
NS(Particles_get_const_mass0)( 
    const NS(Particles) *const SIXTRL_RESTRICT particles );

SIXTRL_STATIC SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* NS(Particles_get_mass0)( 
    NS(Particles)* SIXTRL_RESTRICT particles );

SIXTRL_STATIC void NS(Particles_set_mass0)(
    NS(Particles)* SIXTRL_RESTRICT particles, 
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT ptr_to_mass0s );

SIXTRL_STATIC void NS(Particles_set_mass0_value)(
    NS(Particles)* SIXTRL_RESTRICT particles, NS(block_num_elements_t) const ii, SIXTRL_REAL_T const mass0_value );

SIXTRL_STATIC void NS(Particles_assign_ptr_to_mass0)(
    NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* ptr_to_mass0s );

/* ------------------------------------------------------------------------- */

SIXTRL_STATIC SIXTRL_REAL_T NS(Particles_get_beta0_value)(
    const NS(Particles) *const SIXTRL_RESTRICT particles, NS(block_num_elements_t) const ii );

SIXTRL_STATIC SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const* 
NS(Particles_get_const_beta0)( 
    const NS(Particles) *const SIXTRL_RESTRICT particles );

SIXTRL_STATIC SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* NS(Particles_get_beta0)( 
    NS(Particles)* SIXTRL_RESTRICT particles );

SIXTRL_STATIC void NS(Particles_set_beta0)(
    NS(Particles)* SIXTRL_RESTRICT particles, 
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT ptr_to_beta0s );

SIXTRL_STATIC void NS(Particles_set_beta0_value)(
    NS(Particles)* SIXTRL_RESTRICT particles, 
    NS(block_num_elements_t) const ii, 
    SIXTRL_REAL_T const beta0_value );

SIXTRL_STATIC void NS(Particles_assign_ptr_to_beta0)(
    NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* ptr_to_beta0s );

/* ------------------------------------------------------------------------- */

SIXTRL_STATIC SIXTRL_REAL_T NS(Particles_get_gamma0_value)(
    const NS(Particles) *const SIXTRL_RESTRICT particles, 
    NS(block_num_elements_t) const ii );

SIXTRL_STATIC SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const* 
NS(Particles_get_const_gamma0)( 
    const NS(Particles) *const SIXTRL_RESTRICT particles );

SIXTRL_STATIC SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* NS(Particles_get_gamma0)( 
    NS(Particles)* SIXTRL_RESTRICT particles );

SIXTRL_STATIC void NS(Particles_set_gamma0)(
    NS(Particles)* SIXTRL_RESTRICT particles, 
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT ptr_to_gamma0s );

SIXTRL_STATIC void NS(Particles_set_gamma0_value)(
    NS(Particles)* SIXTRL_RESTRICT particles, 
    NS(block_num_elements_t) const ii, 
    SIXTRL_REAL_T const gamma0_value );

SIXTRL_STATIC void NS(Particles_assign_ptr_to_gamma0)(
    NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* ptr_to_gamma0s );


/* ------------------------------------------------------------------------- */

SIXTRL_STATIC SIXTRL_REAL_T NS(Particles_get_p0c_value)(
    const NS(Particles) *const SIXTRL_RESTRICT particles, 
    NS(block_num_elements_t) const ii );

SIXTRL_STATIC SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const* 
NS(Particles_get_const_p0c)( 
    const NS(Particles) *const SIXTRL_RESTRICT particles );

SIXTRL_STATIC SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* NS(Particles_get_p0c)( 
    NS(Particles)* SIXTRL_RESTRICT particles );

SIXTRL_STATIC void NS(Particles_set_p0c)(
    NS(Particles)* SIXTRL_RESTRICT particles, 
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT ptr_to_p0cs );

SIXTRL_STATIC void NS(Particles_set_p0c_value)(
    NS(Particles)* SIXTRL_RESTRICT particles, 
    NS(block_num_elements_t) const ii, SIXTRL_REAL_T const p0c_value );

SIXTRL_STATIC void NS(Particles_assign_ptr_to_p0c)(
    NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* ptr_to_p0cs );

/* ------------------------------------------------------------------------- */

SIXTRL_STATIC SIXTRL_REAL_T NS(Particles_get_s_value)(
    const NS(Particles) *const SIXTRL_RESTRICT particles, 
    NS(block_num_elements_t) const ii );

SIXTRL_STATIC SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const* 
NS(Particles_get_const_s)( 
    const NS(Particles) *const SIXTRL_RESTRICT particles );

SIXTRL_STATIC SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* NS(Particles_get_s)( 
    NS(Particles)* SIXTRL_RESTRICT particles );

SIXTRL_STATIC void NS(Particles_set_s)(
    NS(Particles)* SIXTRL_RESTRICT particles, 
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT ptr_to_ss );

SIXTRL_STATIC void NS(Particles_set_s_value)(
    NS(Particles)* SIXTRL_RESTRICT particles, 
    NS(block_num_elements_t) const ii, SIXTRL_REAL_T const s_value );

SIXTRL_STATIC void NS(Particles_assign_ptr_to_s)(
    NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* ptr_to_ss );

/* ------------------------------------------------------------------------- */

SIXTRL_STATIC SIXTRL_REAL_T NS(Particles_get_x_value)(
    const NS(Particles) *const SIXTRL_RESTRICT particles, 
    NS(block_num_elements_t) const ii );

SIXTRL_STATIC SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const* 
NS(Particles_get_const_x)( 
    const NS(Particles) *const SIXTRL_RESTRICT particles );

SIXTRL_STATIC SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* NS(Particles_get_x)( 
    NS(Particles)* SIXTRL_RESTRICT particles );

SIXTRL_STATIC void NS(Particles_set_x)(
    NS(Particles)* SIXTRL_RESTRICT particles, 
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT ptr_to_xs );

SIXTRL_STATIC void NS(Particles_set_x_value)(
    NS(Particles)* SIXTRL_RESTRICT particles, 
    NS(block_num_elements_t) const ii, SIXTRL_REAL_T const x_value );

SIXTRL_STATIC void NS(Particles_assign_ptr_to_x)(
    NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* ptr_to_xs );

/* ------------------------------------------------------------------------- */

SIXTRL_STATIC SIXTRL_REAL_T NS(Particles_get_y_value)(
    const NS(Particles) *const SIXTRL_RESTRICT particles, 
    NS(block_num_elements_t) const ii );

SIXTRL_STATIC SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const* 
NS(Particles_get_const_y)( 
    const NS(Particles) *const SIXTRL_RESTRICT particles );

SIXTRL_STATIC SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* NS(Particles_get_y)( 
    NS(Particles)* SIXTRL_RESTRICT particles );

SIXTRL_STATIC void NS(Particles_set_y)(
    NS(Particles)* SIXTRL_RESTRICT particles, 
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT ptr_to_ys );

SIXTRL_STATIC void NS(Particles_set_y_value)(
    NS(Particles)* SIXTRL_RESTRICT particles, 
    NS(block_num_elements_t) const ii, SIXTRL_REAL_T const y_value );

SIXTRL_STATIC void NS(Particles_assign_ptr_to_y)(
    NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* ptr_to_ys );

/* ------------------------------------------------------------------------- */

SIXTRL_STATIC SIXTRL_REAL_T NS(Particles_get_px_value)(
    const NS(Particles) *const SIXTRL_RESTRICT particles, 
    NS(block_num_elements_t) const ii );

SIXTRL_STATIC SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const* 
NS(Particles_get_const_px)( 
    const NS(Particles) *const SIXTRL_RESTRICT particles );

SIXTRL_STATIC SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* NS(Particles_get_px)( 
    NS(Particles)* SIXTRL_RESTRICT particles );

SIXTRL_STATIC void NS(Particles_set_px)(
    NS(Particles)* SIXTRL_RESTRICT particles, 
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT ptr_to_pxs );

SIXTRL_STATIC void NS(Particles_set_px_value)(
    NS(Particles)* SIXTRL_RESTRICT particles, NS(block_num_elements_t) const ii, 
    SIXTRL_REAL_T const px_value );

SIXTRL_STATIC void NS(Particles_assign_ptr_to_px)(
    NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* ptr_to_pxs );

/* ------------------------------------------------------------------------- */

SIXTRL_STATIC SIXTRL_REAL_T NS(Particles_get_py_value)(
    const NS(Particles) *const SIXTRL_RESTRICT particles, 
    NS(block_num_elements_t) const ii );

SIXTRL_STATIC SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const* 
NS(Particles_get_const_py)( 
    const NS(Particles) *const SIXTRL_RESTRICT particles );

SIXTRL_STATIC SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* NS(Particles_get_py)( 
    NS(Particles)* SIXTRL_RESTRICT particles );

SIXTRL_STATIC void NS(Particles_set_py)(
    NS(Particles)* SIXTRL_RESTRICT particles, 
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT ptr_to_pys );

SIXTRL_STATIC void NS(Particles_set_py_value)(
    NS(Particles)* SIXTRL_RESTRICT particles, 
    NS(block_num_elements_t) const ii, SIXTRL_REAL_T const py_value );

SIXTRL_STATIC void NS(Particles_assign_ptr_to_py)(
    NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* ptr_to_pys );

/* ------------------------------------------------------------------------- */

SIXTRL_STATIC SIXTRL_REAL_T NS(Particles_get_sigma_value)(
    const NS(Particles) *const SIXTRL_RESTRICT particles, 
    NS(block_num_elements_t) const ii );

SIXTRL_STATIC SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const* 
NS(Particles_get_const_sigma)( 
    const NS(Particles) *const SIXTRL_RESTRICT particles );

SIXTRL_STATIC SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* NS(Particles_get_sigma)( 
    NS(Particles)* SIXTRL_RESTRICT particles );

SIXTRL_STATIC void NS(Particles_set_sigma)(
    NS(Particles)* SIXTRL_RESTRICT particles, 
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT ptr_to_sigmas );

SIXTRL_STATIC void NS(Particles_set_sigma_value)(
    NS(Particles)* SIXTRL_RESTRICT particles,
    NS(block_num_elements_t) const ii, SIXTRL_REAL_T const sigma_value );

SIXTRL_STATIC void NS(Particles_assign_ptr_to_sigma)(
    NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* ptr_to_sigmas );

/* ------------------------------------------------------------------------- */

SIXTRL_STATIC SIXTRL_REAL_T NS(Particles_get_psigma_value)(
    const NS(Particles) *const SIXTRL_RESTRICT particles, NS(block_num_elements_t) const ii );

SIXTRL_STATIC SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const* 
NS(Particles_get_const_psigma)( 
    const NS(Particles) *const SIXTRL_RESTRICT particles );

SIXTRL_STATIC SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* NS(Particles_get_psigma)( 
    NS(Particles)* SIXTRL_RESTRICT particles );

SIXTRL_STATIC void NS(Particles_set_psigma)(
    NS(Particles)* SIXTRL_RESTRICT particles, 
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT ptr_to_psigmas );

SIXTRL_STATIC void NS(Particles_set_psigma_value)(
    NS(Particles)* SIXTRL_RESTRICT particles, 
    NS(block_num_elements_t) const ii, SIXTRL_REAL_T const psigma_value );

SIXTRL_STATIC void NS(Particles_assign_ptr_to_psigma)(
    NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* ptr_to_psigmas );

/* ------------------------------------------------------------------------- */

SIXTRL_STATIC SIXTRL_REAL_T NS(Particles_get_delta_value)(
    const NS(Particles) *const SIXTRL_RESTRICT particles, 
    NS(block_num_elements_t) const ii );

SIXTRL_STATIC SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const* 
NS(Particles_get_const_delta)( 
    const NS(Particles) *const SIXTRL_RESTRICT particles );

SIXTRL_STATIC SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* NS(Particles_get_delta)( 
    NS(Particles)* SIXTRL_RESTRICT particles );

SIXTRL_STATIC void NS(Particles_set_delta)(
    NS(Particles)* SIXTRL_RESTRICT particles, 
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT ptr_to_deltas );

SIXTRL_STATIC void NS(Particles_set_delta_value)(
    NS(Particles)* SIXTRL_RESTRICT particles, 
    NS(block_num_elements_t) const ii, SIXTRL_REAL_T const delta_value );

SIXTRL_STATIC void NS(Particles_assign_ptr_to_delta)(
    NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* ptr_to_deltas );

/* ------------------------------------------------------------------------- */

SIXTRL_STATIC SIXTRL_REAL_T NS(Particles_get_rpp_value)(
    const NS(Particles) *const SIXTRL_RESTRICT particles, 
    NS(block_num_elements_t) const ii );

SIXTRL_STATIC SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const* 
NS(Particles_get_const_rpp)( 
    const NS(Particles) *const SIXTRL_RESTRICT particles );

SIXTRL_STATIC SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* NS(Particles_get_rpp)( 
    NS(Particles)* SIXTRL_RESTRICT particles );

SIXTRL_STATIC void NS(Particles_set_rpp)(
    NS(Particles)* SIXTRL_RESTRICT particles, 
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT ptr_to_rpps );

SIXTRL_STATIC void NS(Particles_set_rpp_value)(
    NS(Particles)* SIXTRL_RESTRICT particles, 
    NS(block_num_elements_t) const ii, SIXTRL_REAL_T const rpp_value );

SIXTRL_STATIC void NS(Particles_assign_ptr_to_rpp)(
    NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* ptr_to_rpps );

/* ------------------------------------------------------------------------- */

SIXTRL_STATIC SIXTRL_REAL_T NS(Particles_get_rvv_value)(
    const NS(Particles) *const SIXTRL_RESTRICT particles, 
    NS(block_num_elements_t) const ii );

SIXTRL_STATIC SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const* 
NS(Particles_get_const_rvv)( 
    const NS(Particles) *const SIXTRL_RESTRICT particles );

SIXTRL_STATIC SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* NS(Particles_get_rvv)( 
    NS(Particles)* SIXTRL_RESTRICT particles );

SIXTRL_STATIC void NS(Particles_set_rvv)(
    NS(Particles)* SIXTRL_RESTRICT particles, 
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT ptr_to_rvvs );

SIXTRL_STATIC void NS(Particles_set_rvv_value)(
    NS(Particles)* SIXTRL_RESTRICT particles, 
    NS(block_num_elements_t) const ii, SIXTRL_REAL_T const rvv_value );

SIXTRL_STATIC void NS(Particles_assign_ptr_to_rvv)(
    NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* ptr_to_rvvs );

/* ------------------------------------------------------------------------- */

SIXTRL_STATIC SIXTRL_REAL_T NS(Particles_get_chi_value)(
    const NS(Particles) *const SIXTRL_RESTRICT particles, 
    NS(block_num_elements_t) const ii );

SIXTRL_STATIC SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const* 
NS(Particles_get_const_chi)( 
    const NS(Particles) *const SIXTRL_RESTRICT particles );

SIXTRL_STATIC SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* NS(Particles_get_chi)( 
    NS(Particles)* SIXTRL_RESTRICT particles );

SIXTRL_STATIC void NS(Particles_set_chi)(
    NS(Particles)* SIXTRL_RESTRICT particles, 
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT ptr_to_chis );

SIXTRL_STATIC void NS(Particles_set_chi_value)(
    NS(Particles)* SIXTRL_RESTRICT particles, 
    NS(block_num_elements_t) const ii, SIXTRL_REAL_T const chi_value );

SIXTRL_STATIC void NS(Particles_assign_ptr_to_chi)(
    NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* ptr_to_chis );

/* ------------------------------------------------------------------------- */

SIXTRL_STATIC SIXTRL_INT64_T NS(Particles_get_particle_id_value)(
    const NS(Particles) *const SIXTRL_RESTRICT particles, 
    NS(block_num_elements_t) const ii );

SIXTRL_STATIC SIXTRL_GLOBAL_DEC SIXTRL_INT64_T const* 
NS(Particles_get_const_particle_id)( 
    const NS(Particles) *const SIXTRL_RESTRICT particles );

SIXTRL_STATIC SIXTRL_GLOBAL_DEC SIXTRL_INT64_T* NS(Particles_get_particle_id)( 
    NS(Particles)* SIXTRL_RESTRICT particles );

SIXTRL_STATIC void NS(Particles_set_particle_id)(
    NS(Particles)* SIXTRL_RESTRICT particles, 
    SIXTRL_GLOBAL_DEC SIXTRL_INT64_T const* SIXTRL_RESTRICT ptr_to_particle_ids );

SIXTRL_STATIC void NS(Particles_set_particle_id_value)(
    NS(Particles)* SIXTRL_RESTRICT particles, 
    NS(block_num_elements_t) const ii, 
    SIXTRL_INT64_T const particle_id_value );

SIXTRL_STATIC void NS(Particles_assign_ptr_to_particle_id)(
    NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_GLOBAL_DEC SIXTRL_INT64_T* ptr_to_particle_ids );

/* ------------------------------------------------------------------------- */

SIXTRL_STATIC SIXTRL_INT64_T NS(Particles_get_lost_at_element_id_value)(
    const NS(Particles) *const SIXTRL_RESTRICT particles, 
    NS(block_num_elements_t) const ii );

SIXTRL_STATIC SIXTRL_GLOBAL_DEC SIXTRL_INT64_T const* 
NS(Particles_get_const_lost_at_element_id)( 
    const NS(Particles) *const SIXTRL_RESTRICT particles );

SIXTRL_STATIC SIXTRL_GLOBAL_DEC SIXTRL_INT64_T* 
NS(Particles_get_lost_at_element_id)( 
    NS(Particles)* SIXTRL_RESTRICT particles );

SIXTRL_STATIC void NS(Particles_set_lost_at_element_id)(
    NS(Particles)* SIXTRL_RESTRICT particles, 
    SIXTRL_GLOBAL_DEC SIXTRL_INT64_T const* SIXTRL_RESTRICT ptr_to_lost_at_element_ids );

SIXTRL_STATIC void NS(Particles_set_lost_at_element_id_value)(
    NS(Particles)* SIXTRL_RESTRICT particles, 
    NS(block_num_elements_t) const ii, 
    SIXTRL_INT64_T const lost_at_element_id_value );

SIXTRL_STATIC void NS(Particles_assign_ptr_to_lost_at_element_id)(
    NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_GLOBAL_DEC SIXTRL_INT64_T* ptr_to_lost_at_element_ids );

/* ------------------------------------------------------------------------- */

SIXTRL_STATIC SIXTRL_INT64_T NS(Particles_get_lost_at_turn_value)(
    const NS(Particles) *const SIXTRL_RESTRICT particles, 
    NS(block_num_elements_t) const ii );

SIXTRL_STATIC SIXTRL_GLOBAL_DEC SIXTRL_INT64_T const* 
NS(Particles_get_const_lost_at_turn)( 
    const NS(Particles) *const SIXTRL_RESTRICT particles );

SIXTRL_STATIC SIXTRL_GLOBAL_DEC 
SIXTRL_INT64_T* NS(Particles_get_lost_at_turn)( 
    NS(Particles)* SIXTRL_RESTRICT particles );

SIXTRL_STATIC void NS(Particles_set_lost_at_turn)(
    NS(Particles)* SIXTRL_RESTRICT particles, 
    SIXTRL_GLOBAL_DEC SIXTRL_INT64_T const* SIXTRL_RESTRICT ptr_to_lost_at_turns );

SIXTRL_STATIC void NS(Particles_set_lost_at_turn_value)(
    NS(Particles)* SIXTRL_RESTRICT particles, 
    NS(block_num_elements_t) const ii, 
    SIXTRL_INT64_T const lost_at_turn_value );

SIXTRL_STATIC void NS(Particles_assign_ptr_to_lost_at_turn)(
    NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_GLOBAL_DEC SIXTRL_INT64_T* ptr_to_lost_at_turns );

/* ------------------------------------------------------------------------- */

SIXTRL_STATIC SIXTRL_INT64_T NS(Particles_get_state_value)(
    const NS(Particles) *const SIXTRL_RESTRICT particles, 
    NS(block_num_elements_t) const ii );

SIXTRL_STATIC SIXTRL_GLOBAL_DEC SIXTRL_INT64_T const* 
NS(Particles_get_const_state)( 
    const NS(Particles) *const SIXTRL_RESTRICT particles );

SIXTRL_STATIC SIXTRL_GLOBAL_DEC SIXTRL_INT64_T* NS(Particles_get_state)( 
    NS(Particles)* SIXTRL_RESTRICT particles );

SIXTRL_STATIC void NS(Particles_set_state)(
    NS(Particles)* SIXTRL_RESTRICT particles, 
    SIXTRL_GLOBAL_DEC SIXTRL_INT64_T const* SIXTRL_RESTRICT ptr_to_states );

SIXTRL_STATIC void NS(Particles_set_state_value)(
    NS(Particles)* SIXTRL_RESTRICT particles, 
    NS(block_num_elements_t) const ii, SIXTRL_INT64_T const state_value );

SIXTRL_STATIC void NS(Particles_assign_ptr_to_state)(
    NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_GLOBAL_DEC SIXTRL_INT64_T* ptr_to_states );

/* ************************************************************************ */
/* *****             Implementation of inline functions             ******* */
/* ************************************************************************ */

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
        
        NS(Particles_set_type_id_num)(
            particles, NS(BlockType_to_number)( NS(BLOCK_TYPE_PARTICLE) ) );        
    }
    
    return particles;
}


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

SIXTRL_INLINE NS(block_num_elements_t) NS(Particles_get_num_particles)( 
    const NS(Particles) *const SIXTRL_RESTRICT particles )
{
    SIXTRL_ASSERT( particles != 0 );
    return particles->num_particles;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE SIXTRL_REAL_T NS(Particles_get_q0_value)(
    const NS(Particles) *const SIXTRL_RESTRICT p, 
    NS(block_num_elements_t) const ii )
{
    SIXTRL_ASSERT( ( p != 0 ) && ( ii < NS(Particles_get_num_particles)( p ) ) );
    return p->q0[ ii ];
}

SIXTRL_INLINE SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const* 
NS(Particles_get_const_q0)( const NS(Particles) *const SIXTRL_RESTRICT p )
{
    SIXTRL_ASSERT( p != 0 );
    return p->q0;
}

SIXTRL_INLINE SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* NS(Particles_get_q0)( 
    NS(Particles)* SIXTRL_RESTRICT particles )
{
    typedef SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* g_ptr_real_t;
    return ( g_ptr_real_t )NS(Particles_get_const_q0)( particles );
}

SIXTRL_INLINE void NS(Particles_set_q0)(
    NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT ptr_to_q0s )
{
    SIXTRL_ASSERT( ( particles != 0 ) && 
                   ( NS(Particles_get_num_particles)( particles ) != 0u ) &&
                   ( ptr_to_q0s != 0 ) );

    SIXTRACKLIB_COPY_VALUES( SIXTRL_REAL_T, particles->q0, ptr_to_q0s, 
                             NS(Particles_get_num_particles)( particles ) );

    return;
}

SIXTRL_INLINE void NS(Particles_set_q0_value)(
    NS(Particles)* SIXTRL_RESTRICT p, 
    NS(block_num_elements_t) const ii, SIXTRL_REAL_T const q0_value )
{
    SIXTRL_ASSERT( ( p != 0 ) && 
                   ( ii < NS(Particles_get_num_particles)( p ) ) );

    p->q0[ ii ] = q0_value;
    return;
}

SIXTRL_INLINE void NS(Particles_assign_ptr_to_q0)(
    NS(Particles)* SIXTRL_RESTRICT p, 
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* ptr_to_q0 )
{
    SIXTRL_ASSERT( p != 0 );
    p->q0 = ptr_to_q0;

    return;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE SIXTRL_REAL_T NS(Particles_get_mass0_value)(
    const NS(Particles) *const SIXTRL_RESTRICT p, 
    NS(block_num_elements_t) const ii )
{
    SIXTRL_ASSERT( ( p != 0 ) && ( ii < NS(Particles_get_num_particles)( p ) ) );
    return p->mass0[ ii ];
}

SIXTRL_INLINE SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const* 
NS(Particles_get_const_mass0)( const NS(Particles) *const SIXTRL_RESTRICT p )
{
    SIXTRL_ASSERT( p != 0 );
    return p->mass0;
}

SIXTRL_INLINE SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* NS(Particles_get_mass0)( 
    NS(Particles)* SIXTRL_RESTRICT p )
{
    typedef SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* g_ptr_real_t;
    return ( g_ptr_real_t )NS(Particles_get_const_mass0)( p );
}

SIXTRL_INLINE void NS(Particles_set_mass0)(
    NS(Particles)* SIXTRL_RESTRICT p, 
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT ptr_to_mass0s )
{
    SIXTRL_ASSERT( ( p != 0 ) && 
                   ( NS(Particles_get_num_particles)( p ) != 0u ) &&
                   ( ptr_to_mass0s != 0 ) );

    SIXTRACKLIB_COPY_VALUES( SIXTRL_REAL_T, p->mass0, ptr_to_mass0s, 
                             NS(Particles_get_num_particles)( p ) );

    return;
}

SIXTRL_INLINE void NS(Particles_set_mass0_value)(
    NS(Particles)* SIXTRL_RESTRICT p, 
    NS(block_num_elements_t) const ii, SIXTRL_REAL_T const mass0_value )
{
    SIXTRL_ASSERT( ( p != 0 ) && 
                   ( ii < NS(Particles_get_num_particles)( p ) ) );

    p->mass0[ ii ] = mass0_value;
    return;
}

SIXTRL_INLINE void NS(Particles_assign_ptr_to_mass0)(
    NS(Particles)* SIXTRL_RESTRICT p, 
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* ptr_to_mass0 )
{
    SIXTRL_ASSERT( p != 0 );
    p->mass0 = ptr_to_mass0;

    return;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE SIXTRL_REAL_T NS(Particles_get_beta0_value)(
    const NS(Particles) *const SIXTRL_RESTRICT p, 
    NS(block_num_elements_t) const ii  )
{
    SIXTRL_ASSERT( 
        ( p != 0 ) && 
        ( ii < NS(Particles_get_num_particles)( p ) ) );
    
    return p->beta0[ ii ];
}

SIXTRL_INLINE SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const* 
NS(Particles_get_const_beta0)( const NS(Particles) *const SIXTRL_RESTRICT p )
{
    SIXTRL_ASSERT( p != 0 );
    return p->beta0;
}

SIXTRL_INLINE SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* NS(Particles_get_beta0)( 
    NS(Particles)* SIXTRL_RESTRICT p )
{
    typedef SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* g_ptr_real_t;
    return ( g_ptr_real_t )NS(Particles_get_const_beta0)( p );
}

SIXTRL_INLINE void NS(Particles_set_beta0)(
    NS(Particles)* SIXTRL_RESTRICT p, 
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT ptr_to_beta0s )
{
    SIXTRL_ASSERT( ( p != 0 ) && 
                   ( NS(Particles_get_num_particles)( p ) != 0u ) &&
                   ( ptr_to_beta0s != 0 ) );

    SIXTRACKLIB_COPY_VALUES( SIXTRL_REAL_T, p->beta0, ptr_to_beta0s, 
                             NS(Particles_get_num_particles)( p ) );

    return;
}

SIXTRL_INLINE void NS(Particles_set_beta0_value)(
    NS(Particles)* SIXTRL_RESTRICT p, 
    NS(block_num_elements_t) const ii, 
    SIXTRL_REAL_T const beta0_value )
{
    SIXTRL_ASSERT( ( p != 0 ) && 
                   ( ii < NS(Particles_get_num_particles)( p ) ) );

    p->beta0[ ii ] = beta0_value;
    return;
}

SIXTRL_INLINE void NS(Particles_assign_ptr_to_beta0)(
    NS(Particles)* SIXTRL_RESTRICT p, 
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* ptr_to_beta0 )
{
    SIXTRL_ASSERT( p != 0 );
    p->beta0 = ptr_to_beta0;

    return;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE SIXTRL_REAL_T NS(Particles_get_gamma0_value)(
    const NS(Particles) *const SIXTRL_RESTRICT p, 
    NS(block_num_elements_t) const ii )
{
    SIXTRL_ASSERT( ( p != 0 ) && 
                   ( ii < NS(Particles_get_num_particles)( p ) ) );
    return p->gamma0[ ii ];
}

SIXTRL_INLINE SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const* 
NS(Particles_get_const_gamma0)( const NS(Particles) *const SIXTRL_RESTRICT p )
{
    SIXTRL_ASSERT( p != 0 );
    return p->gamma0;
}

SIXTRL_INLINE SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* NS(Particles_get_gamma0)( 
    NS(Particles)* SIXTRL_RESTRICT p )
{
    typedef SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* g_ptr_real_t;
    return ( g_ptr_real_t )NS(Particles_get_const_gamma0)( p );
}

SIXTRL_INLINE void NS(Particles_set_gamma0)(
    NS(Particles)* SIXTRL_RESTRICT p, 
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT ptr_to_gamma0s )
{
    SIXTRL_ASSERT( ( p != 0 ) && 
                   ( NS(Particles_get_num_particles)( p ) != 0u ) &&
                   ( ptr_to_gamma0s != 0 ) );

    SIXTRACKLIB_COPY_VALUES( SIXTRL_REAL_T, p->gamma0, ptr_to_gamma0s, 
                             NS(Particles_get_num_particles)( p ) );

    return;
}

SIXTRL_INLINE void NS(Particles_set_gamma0_value)(
    NS(Particles)* SIXTRL_RESTRICT p, 
    NS(block_num_elements_t) const ii, 
    SIXTRL_REAL_T const gamma0_value )
{
    SIXTRL_ASSERT( ( p != 0 ) && 
                   ( ii < NS(Particles_get_num_particles)( p ) ) );

    p->gamma0[ ii ] = gamma0_value;
    return;
}

SIXTRL_INLINE void NS(Particles_assign_ptr_to_gamma0)(
    NS(Particles)* SIXTRL_RESTRICT p, 
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* ptr_to_gamma0 )
{
    SIXTRL_ASSERT( p != 0 );
    p->gamma0 = ptr_to_gamma0;

    return;
}


/* ------------------------------------------------------------------------- */

SIXTRL_INLINE SIXTRL_REAL_T NS(Particles_get_p0c_value)(
    const NS(Particles) *const SIXTRL_RESTRICT p, 
    NS(block_num_elements_t) const ii )
{
    SIXTRL_ASSERT( ( p != 0 ) && 
                   ( ii < NS(Particles_get_num_particles)( p ) ) );
    
    return p->p0c[ ii ];
}

SIXTRL_INLINE SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const* 
NS(Particles_get_const_p0c)( const NS(Particles) *const SIXTRL_RESTRICT p )
{
    SIXTRL_ASSERT( p != 0 );
    return p->p0c;
}

SIXTRL_INLINE SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* NS(Particles_get_p0c)( 
    NS(Particles)* SIXTRL_RESTRICT p )
{
    typedef SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* g_ptr_real_t;
    return ( g_ptr_real_t )NS(Particles_get_const_p0c)( p );
}

SIXTRL_INLINE void NS(Particles_set_p0c)(
    NS(Particles)* SIXTRL_RESTRICT p, 
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT ptr_to_p0cs )
{
    SIXTRL_ASSERT( ( p != 0 ) && 
                   ( NS(Particles_get_num_particles)( p ) != 0u ) &&
                   ( ptr_to_p0cs != 0 ) );

    SIXTRACKLIB_COPY_VALUES( SIXTRL_REAL_T, p->p0c, ptr_to_p0cs, 
                             NS(Particles_get_num_particles)( p ) );

    return;
}

SIXTRL_INLINE void NS(Particles_set_p0c_value)(
    NS(Particles)* SIXTRL_RESTRICT p, 
    NS(block_num_elements_t) const ii,
    SIXTRL_REAL_T const p0c_value )
{
    SIXTRL_ASSERT( ( p != 0 ) && 
                   ( ii < NS(Particles_get_num_particles)( p ) ) );

    p->p0c[ ii ] = p0c_value;
    return;
}

SIXTRL_INLINE void NS(Particles_assign_ptr_to_p0c)(
    NS(Particles)* SIXTRL_RESTRICT p, 
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* ptr_to_p0c )
{
    SIXTRL_ASSERT( p != 0 );
    p->p0c = ptr_to_p0c;

    return;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE SIXTRL_REAL_T NS(Particles_get_s_value)(
    const NS(Particles) *const SIXTRL_RESTRICT p, NS(block_num_elements_t) const ii )
{
    SIXTRL_ASSERT( ( p != 0 ) && 
                   ( ii < NS(Particles_get_num_particles)( p ) ) );
    
    return p->s[ ii ];
}

SIXTRL_INLINE SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const* 
NS(Particles_get_const_s)( const NS(Particles) *const SIXTRL_RESTRICT p )
{
    SIXTRL_ASSERT( p != 0 );
    return p->s;
}

SIXTRL_INLINE SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* NS(Particles_get_s)( 
    NS(Particles)* SIXTRL_RESTRICT p )
{
    typedef SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* g_ptr_real_t;
    return ( g_ptr_real_t )NS(Particles_get_const_s)( p );
}

SIXTRL_INLINE void NS(Particles_set_s)(
    NS(Particles)* SIXTRL_RESTRICT p, 
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT ptr_to_ss )
{
    SIXTRL_ASSERT( ( p != 0 ) && 
                   ( NS(Particles_get_num_particles)( p ) != 0u ) &&
                   ( ptr_to_ss != 0 ) );

    SIXTRACKLIB_COPY_VALUES( SIXTRL_REAL_T, p->s, ptr_to_ss, 
                             NS(Particles_get_num_particles)( p ) );

    return;
}

SIXTRL_INLINE void NS(Particles_set_s_value)(
    NS(Particles)* SIXTRL_RESTRICT p, NS(block_num_elements_t) const ii, 
    SIXTRL_REAL_T const s_value )
{
    SIXTRL_ASSERT( ( p != 0 ) && 
                   ( ii < NS(Particles_get_num_particles)( p ) ) );

    p->s[ ii ] = s_value;
    return;
}

SIXTRL_INLINE void NS(Particles_assign_ptr_to_s)(
    NS(Particles)* SIXTRL_RESTRICT p, 
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* ptr_to_s )
{
    SIXTRL_ASSERT( p != 0 );
    p->s = ptr_to_s;

    return;
}


/* ------------------------------------------------------------------------- */

SIXTRL_INLINE SIXTRL_REAL_T NS(Particles_get_x_value)(
    const NS(Particles) *const SIXTRL_RESTRICT p, NS(block_num_elements_t) const ii )
{
    SIXTRL_ASSERT( ( p != 0 ) && 
                   ( ii < NS(Particles_get_num_particles)( p ) ) );
    
    return p->x[ ii ];
}

SIXTRL_INLINE SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const* 
NS(Particles_get_const_x)( const NS(Particles) *const SIXTRL_RESTRICT p )
{
    SIXTRL_ASSERT( p != 0 );
    return p->x;
}

SIXTRL_INLINE SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* NS(Particles_get_x)( 
    NS(Particles)* SIXTRL_RESTRICT p )
{
    typedef SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* g_ptr_real_t;
    return ( g_ptr_real_t )NS(Particles_get_const_x)( p );
}

SIXTRL_INLINE void NS(Particles_set_x)(
    NS(Particles)* SIXTRL_RESTRICT p, 
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT ptr_to_xs )
{
    SIXTRL_ASSERT( ( p != 0 ) && 
                   ( NS(Particles_get_num_particles)( p ) != 0u ) &&
                   ( ptr_to_xs != 0 ) );

    SIXTRACKLIB_COPY_VALUES( SIXTRL_REAL_T, p->x, ptr_to_xs, 
                             NS(Particles_get_num_particles)( p ) );

    return;
}

SIXTRL_INLINE void NS(Particles_set_x_value)(
    NS(Particles)* SIXTRL_RESTRICT p, NS(block_num_elements_t) const ii, 
    SIXTRL_REAL_T const x_value )
{
    SIXTRL_ASSERT( ( p != 0 ) && 
                   ( ii < NS(Particles_get_num_particles)( p ) ) );

    p->x[ ii ] = x_value;
    return;
}

SIXTRL_INLINE void NS(Particles_assign_ptr_to_x)(
    NS(Particles)* SIXTRL_RESTRICT p, 
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* ptr_to_x )
{
    SIXTRL_ASSERT( p != 0 );
    p->x = ptr_to_x;

    return;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE SIXTRL_REAL_T NS(Particles_get_y_value)(
    const NS(Particles) *const SIXTRL_RESTRICT p, NS(block_num_elements_t) const ii )
{
    SIXTRL_ASSERT( ( p != 0 ) && 
                   ( ii < NS(Particles_get_num_particles)( p ) ) );
    
    return p->y[ ii ];
}

SIXTRL_INLINE SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const* 
NS(Particles_get_const_y)( const NS(Particles) *const SIXTRL_RESTRICT p )
{
    SIXTRL_ASSERT( p != 0 );
    return p->y;
}

SIXTRL_INLINE SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* NS(Particles_get_y)( 
    NS(Particles)* SIXTRL_RESTRICT p )
{
    typedef SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* g_ptr_real_t;
    return ( g_ptr_real_t )NS(Particles_get_const_y)( p );
}

SIXTRL_INLINE void NS(Particles_set_y)(
    NS(Particles)* SIXTRL_RESTRICT p, 
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT ptr_to_ys )
{
    SIXTRL_ASSERT( ( p != 0 ) && 
                   ( NS(Particles_get_num_particles)( p ) != 0u ) &&
                   ( ptr_to_ys != 0 ) );

    SIXTRACKLIB_COPY_VALUES( SIXTRL_REAL_T, p->y, ptr_to_ys, 
                             NS(Particles_get_num_particles)( p ) );

    return;
}

SIXTRL_INLINE void NS(Particles_set_y_value)(
    NS(Particles)* SIXTRL_RESTRICT p, NS(block_num_elements_t) const ii, 
    SIXTRL_REAL_T const y_value )
{
    SIXTRL_ASSERT( ( p != 0 ) && 
                   ( ii < NS(Particles_get_num_particles)( p ) ) );

    p->y[ ii ] = y_value;
    return;
}

SIXTRL_INLINE void NS(Particles_assign_ptr_to_y)(
    NS(Particles)* SIXTRL_RESTRICT p, 
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* ptr_to_y )
{
    SIXTRL_ASSERT( p != 0 );
    p->y = ptr_to_y;

    return;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE SIXTRL_REAL_T NS(Particles_get_px_value)(
    const NS(Particles) *const SIXTRL_RESTRICT p, NS(block_num_elements_t) const ii )
{
    SIXTRL_ASSERT( ( p != 0 ) && 
                   ( ii < NS(Particles_get_num_particles)( p ) ) );
    
    return p->px[ ii ];
}

SIXTRL_INLINE SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const* 
NS(Particles_get_const_px)( const NS(Particles) *const SIXTRL_RESTRICT p )
{
    SIXTRL_ASSERT( p != 0 );
    return p->px;
}

SIXTRL_INLINE SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* NS(Particles_get_px)( 
    NS(Particles)* SIXTRL_RESTRICT p )
{
    typedef SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* g_ptr_real_t;
    return ( g_ptr_real_t )NS(Particles_get_const_px)( p );
}

SIXTRL_INLINE void NS(Particles_set_px)(
    NS(Particles)* SIXTRL_RESTRICT p, 
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT ptr_to_pxs )
{
    SIXTRL_ASSERT( ( p != 0 ) && 
                   ( NS(Particles_get_num_particles)( p ) != 0u ) &&
                   ( ptr_to_pxs != 0 ) );

    SIXTRACKLIB_COPY_VALUES( SIXTRL_REAL_T, p->px, ptr_to_pxs, 
                             NS(Particles_get_num_particles)( p ) );

    return;
}

SIXTRL_INLINE void NS(Particles_set_px_value)(
    NS(Particles)* SIXTRL_RESTRICT p, NS(block_num_elements_t) const ii, 
    SIXTRL_REAL_T const px_value )
{
    SIXTRL_ASSERT( ( p != 0 ) && 
                   ( ii < NS(Particles_get_num_particles)( p ) ) );

    p->px[ ii ] = px_value;
    return;
}

SIXTRL_INLINE void NS(Particles_assign_ptr_to_px)(
    NS(Particles)* SIXTRL_RESTRICT p, 
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* ptr_to_px )
{
    SIXTRL_ASSERT( p != 0 );
    p->px = ptr_to_px;

    return;
}


/* ------------------------------------------------------------------------- */

SIXTRL_INLINE SIXTRL_REAL_T NS(Particles_get_py_value)(
    const NS(Particles) *const SIXTRL_RESTRICT p, NS(block_num_elements_t) const ii )
{
    SIXTRL_ASSERT( ( p != 0 ) && 
                   ( ii < NS(Particles_get_num_particles)( p ) ) );
    
    return p->py[ ii ];
}

SIXTRL_INLINE SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const* 
NS(Particles_get_const_py)( const NS(Particles) *const SIXTRL_RESTRICT p )
{
    SIXTRL_ASSERT( p != 0 );
    return p->py;
}

SIXTRL_INLINE SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* NS(Particles_get_py)( 
    NS(Particles)* SIXTRL_RESTRICT p )
{
    typedef SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* g_ptr_real_t;
    return ( g_ptr_real_t )NS(Particles_get_const_py)( p );
}

SIXTRL_INLINE void NS(Particles_set_py)(
    NS(Particles)* SIXTRL_RESTRICT p, 
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT ptr_to_pys )
{
    SIXTRL_ASSERT( ( p != 0 ) && 
                   ( NS(Particles_get_num_particles)( p ) != 0u ) &&
                   ( ptr_to_pys != 0 ) );

    SIXTRACKLIB_COPY_VALUES( SIXTRL_REAL_T, p->py, ptr_to_pys, 
                             NS(Particles_get_num_particles)( p ) );

    return;
}

SIXTRL_INLINE void NS(Particles_set_py_value)(
    NS(Particles)* SIXTRL_RESTRICT p, NS(block_num_elements_t) const ii, 
    SIXTRL_REAL_T const py_value )
{
    SIXTRL_ASSERT( ( p != 0 ) && 
                   ( ii < NS(Particles_get_num_particles)( p ) ) );

    p->py[ ii ] = py_value;
    return;
}

SIXTRL_INLINE void NS(Particles_assign_ptr_to_py)(
    NS(Particles)* SIXTRL_RESTRICT p, 
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* ptr_to_py )
{
    SIXTRL_ASSERT( p != 0 );
    p->py = ptr_to_py;

    return;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE SIXTRL_REAL_T NS(Particles_get_sigma_value)(
    const NS(Particles) *const SIXTRL_RESTRICT p, NS(block_num_elements_t) const ii )
{
    SIXTRL_ASSERT( ( p != 0 ) && 
                   ( ii < NS(Particles_get_num_particles)( p ) ) );
    
    return p->sigma[ ii ];
}

SIXTRL_INLINE SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const* 
NS(Particles_get_const_sigma)( const NS(Particles) *const SIXTRL_RESTRICT p )
{
    SIXTRL_ASSERT( p != 0 );
    return p->sigma;
}

SIXTRL_INLINE SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* NS(Particles_get_sigma)( 
    NS(Particles)* SIXTRL_RESTRICT p )
{
    typedef SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* g_ptr_real_t;
    return ( g_ptr_real_t )NS(Particles_get_const_sigma)( p );
}

SIXTRL_INLINE void NS(Particles_set_sigma)(
    NS(Particles)* SIXTRL_RESTRICT p, 
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT ptr_to_sigmas )
{
    SIXTRL_ASSERT( ( p != 0 ) && 
                   ( NS(Particles_get_num_particles)( p ) != 0u ) &&
                   ( ptr_to_sigmas != 0 ) );

    SIXTRACKLIB_COPY_VALUES( SIXTRL_REAL_T, p->sigma, ptr_to_sigmas, 
                             NS(Particles_get_num_particles)( p ) );

    return;
}

SIXTRL_INLINE void NS(Particles_set_sigma_value)(
    NS(Particles)* SIXTRL_RESTRICT p, NS(block_num_elements_t) const ii, 
    SIXTRL_REAL_T const sigma_value )
{
    SIXTRL_ASSERT( ( p != 0 ) && 
                   ( ii < NS(Particles_get_num_particles)( p ) ) );

    p->sigma[ ii ] = sigma_value;
    return;
}

SIXTRL_INLINE void NS(Particles_assign_ptr_to_sigma)(
    NS(Particles)* SIXTRL_RESTRICT p, 
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* ptr_to_sigma )
{
    SIXTRL_ASSERT( p != 0 );
    p->sigma = ptr_to_sigma;

    return;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE SIXTRL_REAL_T NS(Particles_get_psigma_value)(
    const NS(Particles) *const SIXTRL_RESTRICT p, NS(block_num_elements_t) const ii )
{
    SIXTRL_ASSERT( ( p != 0 ) && 
                   ( ii < NS(Particles_get_num_particles)( p ) ) );
    
    return p->psigma[ ii ];
}

SIXTRL_INLINE SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const* 
NS(Particles_get_const_psigma)( const NS(Particles) *const SIXTRL_RESTRICT p )
{
    SIXTRL_ASSERT( p != 0 );
    return p->psigma;
}

SIXTRL_INLINE SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* NS(Particles_get_psigma)( 
    NS(Particles)* SIXTRL_RESTRICT p )
{
    typedef SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* g_ptr_real_t;
    return ( g_ptr_real_t )NS(Particles_get_const_psigma)( p );
}

SIXTRL_INLINE void NS(Particles_set_psigma)(
    NS(Particles)* SIXTRL_RESTRICT p, 
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT ptr_to_psigmas )
{
    SIXTRL_ASSERT( ( p != 0 ) && 
                   ( NS(Particles_get_num_particles)( p ) != 0u ) &&
                   ( ptr_to_psigmas != 0 ) );

    SIXTRACKLIB_COPY_VALUES( SIXTRL_REAL_T, p->psigma, ptr_to_psigmas, 
                             NS(Particles_get_num_particles)( p ) );

    return;
}

SIXTRL_INLINE void NS(Particles_set_psigma_value)(
    NS(Particles)* SIXTRL_RESTRICT p, NS(block_num_elements_t) const ii, 
    SIXTRL_REAL_T const psigma_value )
{
    SIXTRL_ASSERT( ( p != 0 ) && 
                   ( ii < NS(Particles_get_num_particles)( p ) ) );

    p->psigma[ ii ] = psigma_value;
    return;
}

SIXTRL_INLINE void NS(Particles_assign_ptr_to_psigma)(
    NS(Particles)* SIXTRL_RESTRICT p, 
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* ptr_to_psigma )
{
    SIXTRL_ASSERT( p != 0 );
    p->psigma = ptr_to_psigma;

    return;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE SIXTRL_REAL_T NS(Particles_get_delta_value)(
    const NS(Particles) *const SIXTRL_RESTRICT p, NS(block_num_elements_t) const ii )
{
    SIXTRL_ASSERT( ( p != 0 ) && 
                   ( ii < NS(Particles_get_num_particles)( p ) ) );
    
    return p->delta[ ii ];
}

SIXTRL_INLINE SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const* 
NS(Particles_get_const_delta)( const NS(Particles) *const SIXTRL_RESTRICT p )
{
    SIXTRL_ASSERT( p != 0 );
    return p->delta;
}

SIXTRL_INLINE SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* NS(Particles_get_delta)( 
    NS(Particles)* SIXTRL_RESTRICT p )
{
    typedef SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* g_ptr_real_t;
    return ( g_ptr_real_t )NS(Particles_get_const_delta)( p );
}

SIXTRL_INLINE void NS(Particles_set_delta)(
    NS(Particles)* SIXTRL_RESTRICT p, 
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT ptr_to_deltas )
{
    SIXTRL_ASSERT( ( p != 0 ) && 
                   ( NS(Particles_get_num_particles)( p ) != 0u ) &&
                   ( ptr_to_deltas != 0 ) );

    SIXTRACKLIB_COPY_VALUES( SIXTRL_REAL_T, p->delta, ptr_to_deltas, 
                             NS(Particles_get_num_particles)( p ) );

    return;
}

SIXTRL_INLINE void NS(Particles_set_delta_value)(
    NS(Particles)* SIXTRL_RESTRICT p, NS(block_num_elements_t) const ii, 
    SIXTRL_REAL_T const delta_value )
{
    SIXTRL_ASSERT( ( p != 0 ) && 
                   ( ii < NS(Particles_get_num_particles)( p ) ) );

    p->delta[ ii ] = delta_value;
    return;
}

SIXTRL_INLINE void NS(Particles_assign_ptr_to_delta)(
    NS(Particles)* SIXTRL_RESTRICT p, 
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* ptr_to_delta )
{
    SIXTRL_ASSERT( p != 0 );
    p->delta = ptr_to_delta;

    return;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE SIXTRL_REAL_T NS(Particles_get_rpp_value)(
    const NS(Particles) *const SIXTRL_RESTRICT p, 
    NS(block_num_elements_t) const ii )
{
    SIXTRL_ASSERT( ( p != 0 ) && 
                   ( ii < NS(Particles_get_num_particles)( p ) ) );
                   
    return p->rpp[ ii ];
}

SIXTRL_INLINE SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const* 
NS(Particles_get_const_rpp)( const NS(Particles) *const SIXTRL_RESTRICT p )
{
    SIXTRL_ASSERT( p != 0 );
    return p->rpp;
}

SIXTRL_INLINE SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* NS(Particles_get_rpp)( 
    NS(Particles)* SIXTRL_RESTRICT p )
{
    typedef SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* g_ptr_real_t;
    return ( g_ptr_real_t )NS(Particles_get_const_rpp)( p );
}

SIXTRL_INLINE void NS(Particles_set_rpp)(
    NS(Particles)* SIXTRL_RESTRICT p, 
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT ptr_to_rpps )
{
    SIXTRL_ASSERT( ( p != 0 ) && 
                   ( NS(Particles_get_num_particles)( p ) != 0u ) &&
                   ( ptr_to_rpps != 0 ) );

    SIXTRACKLIB_COPY_VALUES( SIXTRL_REAL_T, p->rpp, ptr_to_rpps, 
                             NS(Particles_get_num_particles)( p ) );

    return;
}

SIXTRL_INLINE void NS(Particles_set_rpp_value)(
    NS(Particles)* SIXTRL_RESTRICT p, NS(block_num_elements_t) const ii, 
    SIXTRL_REAL_T const rpp_value )
{
    SIXTRL_ASSERT( ( p != 0 ) && 
                   ( ii < NS(Particles_get_num_particles)( p ) ) );

    p->rpp[ ii ] = rpp_value;
    return;
}

SIXTRL_INLINE void NS(Particles_assign_ptr_to_rpp)(
    NS(Particles)* SIXTRL_RESTRICT p, 
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* ptr_to_rpp )
{
    SIXTRL_ASSERT( p != 0 );
    p->rpp = ptr_to_rpp;

    return;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE SIXTRL_REAL_T NS(Particles_get_rvv_value)(
    const NS(Particles) *const SIXTRL_RESTRICT p, 
    NS(block_num_elements_t) const ii )
{
    SIXTRL_ASSERT( ( p != 0 ) && 
                   ( ii < NS(Particles_get_num_particles)( p ) ) );
    
    return p->rvv[ ii ];
}

SIXTRL_INLINE SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const* 
NS(Particles_get_const_rvv)( const NS(Particles) *const SIXTRL_RESTRICT p )
{
    SIXTRL_ASSERT( p != 0 );
    return p->rvv;
}

SIXTRL_INLINE SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* NS(Particles_get_rvv)( 
    NS(Particles)* SIXTRL_RESTRICT p )
{
    typedef SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* g_ptr_real_t;
    return ( g_ptr_real_t )NS(Particles_get_const_rvv)( p );
}

SIXTRL_INLINE void NS(Particles_set_rvv)(
    NS(Particles)* SIXTRL_RESTRICT p, 
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT ptr_to_rvvs )
{
    SIXTRL_ASSERT( ( p != 0 ) && 
                   ( NS(Particles_get_num_particles)( p ) != 0u ) &&
                   ( ptr_to_rvvs != 0 ) );

    SIXTRACKLIB_COPY_VALUES( SIXTRL_REAL_T, p->rvv, ptr_to_rvvs, 
                             NS(Particles_get_num_particles)( p ) );

    return;
}

SIXTRL_INLINE void NS(Particles_set_rvv_value)(
    NS(Particles)* SIXTRL_RESTRICT p, NS(block_num_elements_t) const ii, 
    SIXTRL_REAL_T const rvv_value )
{
    SIXTRL_ASSERT( ( p != 0 ) && 
                   ( ii < NS(Particles_get_num_particles)( p ) ) );

    p->rvv[ ii ] = rvv_value;
    return;
}

SIXTRL_INLINE void NS(Particles_assign_ptr_to_rvv)(
    NS(Particles)* SIXTRL_RESTRICT p, 
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* ptr_to_rvv )
{
    SIXTRL_ASSERT( p != 0 );
    p->rvv = ptr_to_rvv;

    return;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE SIXTRL_REAL_T NS(Particles_get_chi_value)(
    const NS(Particles) *const SIXTRL_RESTRICT p, 
    NS(block_num_elements_t) const ii )
{
    SIXTRL_ASSERT( ( p != 0 ) && 
                   ( ii < NS(Particles_get_num_particles)( p ) ) );
    
    return p->chi[ ii ];
}

SIXTRL_INLINE SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const* 
NS(Particles_get_const_chi)( const NS(Particles) *const SIXTRL_RESTRICT p )
{
    SIXTRL_ASSERT( p != 0 );
    return p->chi;
}

SIXTRL_INLINE SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* NS(Particles_get_chi)( 
    NS(Particles)* SIXTRL_RESTRICT p )
{
    typedef SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* g_ptr_real_t;
    return ( g_ptr_real_t )NS(Particles_get_const_chi)( p );
}

SIXTRL_INLINE void NS(Particles_set_chi)(
    NS(Particles)* SIXTRL_RESTRICT p, 
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT ptr_to_chis )
{
    SIXTRL_ASSERT( ( p != 0 ) && 
                   ( NS(Particles_get_num_particles)( p ) != 0u ) &&
                   ( ptr_to_chis != 0 ) );

    SIXTRACKLIB_COPY_VALUES( SIXTRL_REAL_T, p->chi, ptr_to_chis, 
                             NS(Particles_get_num_particles)( p ) );

    return;
}

SIXTRL_INLINE void NS(Particles_set_chi_value)(
    NS(Particles)* SIXTRL_RESTRICT p, NS(block_num_elements_t) const ii, 
    SIXTRL_REAL_T const chi_value )
{
    SIXTRL_ASSERT( ( p != 0 ) && 
                   ( ii < NS(Particles_get_num_particles)( p ) ) );

    p->chi[ ii ] = chi_value;
    return;
}

SIXTRL_INLINE void NS(Particles_assign_ptr_to_chi)(
    NS(Particles)* SIXTRL_RESTRICT p, 
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* ptr_to_chi )
{
    SIXTRL_ASSERT( p != 0 );
    p->chi = ptr_to_chi;

    return;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE SIXTRL_INT64_T NS(Particles_get_particle_id_value)(
    const NS(Particles) *const SIXTRL_RESTRICT p, 
    NS(block_num_elements_t) const ii )
{
    SIXTRL_ASSERT( ( p != 0 ) && 
                   ( ii < NS(Particles_get_num_particles)( p ) ) );
    
    return p->particle_id[ ii ];
}

SIXTRL_INLINE SIXTRL_GLOBAL_DEC SIXTRL_INT64_T const* 
NS(Particles_get_const_particle_id)( 
    const NS(Particles) *const SIXTRL_RESTRICT p )
{
    SIXTRL_ASSERT( p != 0 );
    return p->particle_id;
}

SIXTRL_INLINE SIXTRL_GLOBAL_DEC SIXTRL_INT64_T* NS(Particles_get_particle_id)( 
    NS(Particles)* SIXTRL_RESTRICT p )
{
    typedef SIXTRL_GLOBAL_DEC SIXTRL_INT64_T* g_ptr_real_t;
    return ( g_ptr_real_t )NS(Particles_get_const_particle_id)( p );
}

SIXTRL_INLINE void NS(Particles_set_particle_id)(
    NS(Particles)* SIXTRL_RESTRICT p, 
    SIXTRL_GLOBAL_DEC SIXTRL_INT64_T const* SIXTRL_RESTRICT ptr_to_particle_ids )
{
    SIXTRL_ASSERT( ( p != 0 ) && 
                   ( NS(Particles_get_num_particles)( p ) != 0u ) &&
                   ( ptr_to_particle_ids != 0 ) );

    SIXTRACKLIB_COPY_VALUES( 
        SIXTRL_INT64_T, p->particle_id, ptr_to_particle_ids, 
        NS(Particles_get_num_particles)( p ) );

    return;
}

SIXTRL_INLINE void NS(Particles_set_particle_id_value)(
    NS(Particles)* SIXTRL_RESTRICT p, NS(block_num_elements_t) const ii, 
    SIXTRL_INT64_T const particle_id_value )
{
    SIXTRL_ASSERT( ( p != 0 ) && 
                   ( ii < NS(Particles_get_num_particles)( p ) ) );

    p->particle_id[ ii ] = particle_id_value;
    return;
}

SIXTRL_INLINE void NS(Particles_assign_ptr_to_particle_id)(
    NS(Particles)* SIXTRL_RESTRICT p, 
    SIXTRL_GLOBAL_DEC SIXTRL_INT64_T* ptr_to_particle_id )
{
    SIXTRL_ASSERT( p != 0 );
    p->particle_id = ptr_to_particle_id;

    return;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE SIXTRL_INT64_T NS(Particles_get_lost_at_element_id_value)(
    const NS(Particles) *const SIXTRL_RESTRICT p, 
    NS(block_num_elements_t) const ii )
{
    SIXTRL_ASSERT( ( p != 0 ) && 
                   ( ii < NS(Particles_get_num_particles)( p ) ) );
    
    return p->lost_at_element_id[ ii ];
}

SIXTRL_INLINE SIXTRL_GLOBAL_DEC SIXTRL_INT64_T const* 
NS(Particles_get_const_lost_at_element_id)( 
    const NS(Particles) *const SIXTRL_RESTRICT p )
{
    SIXTRL_ASSERT( p != 0 );
    return p->lost_at_element_id;
}

SIXTRL_INLINE SIXTRL_GLOBAL_DEC SIXTRL_INT64_T* 
NS(Particles_get_lost_at_element_id)( 
    NS(Particles)* SIXTRL_RESTRICT p )
{
    typedef SIXTRL_GLOBAL_DEC SIXTRL_INT64_T* g_ptr_real_t;
    return ( g_ptr_real_t )NS(Particles_get_const_lost_at_element_id)( p );
}

SIXTRL_INLINE void NS(Particles_set_lost_at_element_id)(
    NS(Particles)* SIXTRL_RESTRICT p, 
    SIXTRL_GLOBAL_DEC SIXTRL_INT64_T const* SIXTRL_RESTRICT ptr_to_lost_at_element_ids )
{
    SIXTRL_ASSERT( ( p != 0 ) && 
                   ( NS(Particles_get_num_particles)( p ) != 0u ) &&
                   ( ptr_to_lost_at_element_ids != 0 ) );

    SIXTRACKLIB_COPY_VALUES( 
        SIXTRL_INT64_T, p->lost_at_element_id, ptr_to_lost_at_element_ids, 
        NS(Particles_get_num_particles)( p ) );

    return;
}

SIXTRL_INLINE void NS(Particles_set_lost_at_element_id_value)(
    NS(Particles)* SIXTRL_RESTRICT p, NS(block_num_elements_t) const ii, 
    SIXTRL_INT64_T const lost_at_element_id_value )
{
    SIXTRL_ASSERT( ( p != 0 ) && 
                   ( ii < NS(Particles_get_num_particles)( p ) ) );

    p->lost_at_element_id[ ii ] = lost_at_element_id_value;
    return;
}

SIXTRL_INLINE void NS(Particles_assign_ptr_to_lost_at_element_id)(
    NS(Particles)* SIXTRL_RESTRICT p, 
    SIXTRL_GLOBAL_DEC SIXTRL_INT64_T* ptr_to_lost_at_element_id )
{
    SIXTRL_ASSERT( p != 0 );
    p->lost_at_element_id = ptr_to_lost_at_element_id;

    return;
}


/* ------------------------------------------------------------------------- */

SIXTRL_INLINE SIXTRL_INT64_T NS(Particles_get_lost_at_turn_value)(
    const NS(Particles) *const SIXTRL_RESTRICT p, 
    NS(block_num_elements_t) const ii )
{
    SIXTRL_ASSERT( ( p != 0 ) && 
                   ( ii < NS(Particles_get_num_particles)( p ) ) );
    
    return p->lost_at_turn[ ii ];
}

SIXTRL_INLINE SIXTRL_GLOBAL_DEC SIXTRL_INT64_T const* 
NS(Particles_get_const_lost_at_turn)( 
    const NS(Particles) *const SIXTRL_RESTRICT p )
{
    SIXTRL_ASSERT( p != 0 );
    return p->lost_at_turn;
}

SIXTRL_INLINE SIXTRL_GLOBAL_DEC SIXTRL_INT64_T* 
NS(Particles_get_lost_at_turn)( 
    NS(Particles)* SIXTRL_RESTRICT p )
{
    typedef SIXTRL_GLOBAL_DEC SIXTRL_INT64_T* g_ptr_real_t;
    return ( g_ptr_real_t )NS(Particles_get_const_lost_at_turn)( p );
}

SIXTRL_INLINE void NS(Particles_set_lost_at_turn)(
    NS(Particles)* SIXTRL_RESTRICT p, 
    SIXTRL_GLOBAL_DEC SIXTRL_INT64_T const* SIXTRL_RESTRICT ptr_to_lost_at_turns )
{
    SIXTRL_ASSERT( ( p != 0 ) && 
                   ( NS(Particles_get_num_particles)( p ) != 0u ) &&
                   ( ptr_to_lost_at_turns != 0 ) );

    SIXTRACKLIB_COPY_VALUES( 
        SIXTRL_INT64_T, p->lost_at_turn, ptr_to_lost_at_turns, 
        NS(Particles_get_num_particles)( p ) );

    return;
}

SIXTRL_INLINE void NS(Particles_set_lost_at_turn_value)(
    NS(Particles)* SIXTRL_RESTRICT p, NS(block_num_elements_t) const ii, 
    SIXTRL_INT64_T const lost_at_turn_value )
{
    SIXTRL_ASSERT( ( p != 0 ) && 
                   ( ii < NS(Particles_get_num_particles)( p ) ) );

    p->lost_at_turn[ ii ] = lost_at_turn_value;
    return;
}

SIXTRL_INLINE void NS(Particles_assign_ptr_to_lost_at_turn)(
    NS(Particles)* SIXTRL_RESTRICT p, 
    SIXTRL_GLOBAL_DEC SIXTRL_INT64_T* ptr_to_lost_at_turn )
{
    SIXTRL_ASSERT( p != 0 );
    p->lost_at_turn = ptr_to_lost_at_turn;

    return;
}


/* ------------------------------------------------------------------------- */

SIXTRL_INLINE SIXTRL_INT64_T NS(Particles_get_state_value)(
    const NS(Particles) *const SIXTRL_RESTRICT p, NS(block_num_elements_t) const ii )
{
    SIXTRL_ASSERT( ( p != 0 ) && 
                   ( ii < NS(Particles_get_num_particles)( p ) ) );
    
    return p->state[ ii ];
}

SIXTRL_INLINE SIXTRL_GLOBAL_DEC SIXTRL_INT64_T const* 
NS(Particles_get_const_state)( const NS(Particles) *const SIXTRL_RESTRICT p )
{
    SIXTRL_ASSERT( p != 0 );
    return p->state;
}

SIXTRL_INLINE SIXTRL_GLOBAL_DEC SIXTRL_INT64_T* NS(Particles_get_state)( 
    NS(Particles)* SIXTRL_RESTRICT p )
{
    typedef SIXTRL_GLOBAL_DEC SIXTRL_INT64_T* g_ptr_real_t;
    return ( g_ptr_real_t )NS(Particles_get_const_state)( p );
}

SIXTRL_INLINE void NS(Particles_set_state)(
    NS(Particles)* SIXTRL_RESTRICT p, 
    SIXTRL_GLOBAL_DEC SIXTRL_INT64_T const* SIXTRL_RESTRICT ptr_to_states )
{
    SIXTRL_ASSERT( ( p != 0 ) && 
                   ( NS(Particles_get_num_particles)( p ) != 0u ) &&
                   ( ptr_to_states != 0 ) );

    SIXTRACKLIB_COPY_VALUES( SIXTRL_INT64_T, p->state, ptr_to_states, 
                             NS(Particles_get_num_particles)( p ) );

    return;
}

SIXTRL_INLINE void NS(Particles_set_state_value)(
    NS(Particles)* SIXTRL_RESTRICT p, NS(block_num_elements_t) const ii, 
    SIXTRL_INT64_T const state_value )
{
    SIXTRL_ASSERT( ( p != 0 ) && 
                   ( ii < NS(Particles_get_num_particles)( p ) ) );

    p->state[ ii ] = state_value;
    return;
}

SIXTRL_INLINE void NS(Particles_assign_ptr_to_state)(
    NS(Particles)* SIXTRL_RESTRICT p, 
    SIXTRL_GLOBAL_DEC SIXTRL_INT64_T* ptr_to_state )
{
    SIXTRL_ASSERT( p != 0 );
    p->state = ptr_to_state;

    return;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE void NS(Particles_set_num_particles)(
    NS(Particles)* SIXTRL_RESTRICT particles, 
    NS(block_num_elements_t) const num_particles_in_block )
{
    SIXTRL_ASSERT( 
        ( particles != 0 ) &&
        ( NS(Particles_get_q0)( particles ) == 0 ) &&
        ( NS(Particles_get_mass0)( particles ) == 0 ) &&
        ( NS(Particles_get_beta0)( particles ) == 0 ) &&
        ( NS(Particles_get_gamma0)( particles ) == 0 ) &&
        ( NS(Particles_get_p0c)( particles ) == 0 ) &&
        ( NS(Particles_get_s)( particles ) == 0 ) &&
        ( NS(Particles_get_x)( particles ) == 0 ) &&
        ( NS(Particles_get_y)( particles ) == 0 ) &&
        ( NS(Particles_get_px)( particles ) == 0 ) &&
        ( NS(Particles_get_py)( particles ) == 0 ) &&
        ( NS(Particles_get_sigma)( particles ) == 0 ) &&
        ( NS(Particles_get_psigma)( particles ) == 0 ) &&
        ( NS(Particles_get_delta)( particles ) == 0 ) &&
        ( NS(Particles_get_rpp)( particles ) == 0 ) &&
        ( NS(Particles_get_rvv)( particles ) == 0 ) &&
        ( NS(Particles_get_chi)( particles ) == 0 ) &&
        ( NS(Particles_get_particle_id)( particles ) == 0 ) &&
        ( NS(Particles_get_lost_at_element_id)( particles ) == 0 ) &&
        ( NS(Particles_get_lost_at_turn)( particles ) == 0 ) &&
        ( NS(Particles_get_state)( particles ) == 0 ) );
    
    particles->num_particles = num_particles_in_block;
    return;
}

SIXTRL_INLINE NS(block_type_num_t) NS(Particles_get_type_id_num)( 
    const NS(Particles) *const SIXTRL_RESTRICT particles )
{
    SIXTRL_ASSERT( particles != 0 );
    return particles->type_id_num;
}

SIXTRL_INLINE NS(BlockType) NS(Particles_get_type_id)( 
    const NS(Particles) *const SIXTRL_RESTRICT particles )
{
    SIXTRL_ASSERT( particles != 0 );
    return NS(BlockType_from_number)( particles->type_id_num );    
}

SIXTRL_INLINE void NS(Particles_set_type_id_num)(
    NS(Particles)* SIXTRL_RESTRICT particles,
    NS(block_type_num_t) const type_id_num )
{
    SIXTRL_ASSERT( 
        ( particles != 0 ) &&
        ( NS(Particles_get_q0)( particles ) == 0 ) &&
        ( NS(Particles_get_mass0)( particles ) == 0 ) &&
        ( NS(Particles_get_beta0)( particles ) == 0 ) &&
        ( NS(Particles_get_gamma0)( particles ) == 0 ) &&
        ( NS(Particles_get_p0c)( particles ) == 0 ) &&
        ( NS(Particles_get_s)( particles ) == 0 ) &&
        ( NS(Particles_get_x)( particles ) == 0 ) &&
        ( NS(Particles_get_y)( particles ) == 0 ) &&
        ( NS(Particles_get_px)( particles ) == 0 ) &&
        ( NS(Particles_get_py)( particles ) == 0 ) &&
        ( NS(Particles_get_sigma)( particles ) == 0 ) &&
        ( NS(Particles_get_psigma)( particles ) == 0 ) &&
        ( NS(Particles_get_delta)( particles ) == 0 ) &&
        ( NS(Particles_get_rpp)( particles ) == 0 ) &&
        ( NS(Particles_get_rvv)( particles ) == 0 ) &&
        ( NS(Particles_get_chi)( particles ) == 0 ) &&
        ( NS(Particles_get_particle_id)( particles ) == 0 ) &&
        ( NS(Particles_get_lost_at_element_id)( particles ) == 0 ) &&
        ( NS(Particles_get_lost_at_turn)( particles ) == 0 ) &&
        ( NS(Particles_get_state)( particles ) == 0 ) );
    
    particles->type_id_num = type_id_num;
    return;
}

SIXTRL_INLINE void NS(Particles_set_type_id)(
    NS(Particles)* SIXTRL_RESTRICT particles, NS(BlockType) const type_id )
{
    NS(Particles_set_type_id_num)( 
        particles, NS(BlockType_to_number)( type_id ) );
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

SIXTRL_INLINE int NS(Particles_is_aligned_with)( 
    const NS(Particles)  *const SIXTRL_RESTRICT particles, 
    NS(block_alignment_t) const align )
{
    typedef NS(block_alignment_t) align_t;
    
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


SIXTRL_INLINE int NS(Particles_is_consistent)(
    const NS(Particles) *const SIXTRL_RESTRICT particles )
{
    int success = 0;
    
    if( ( particles != 0 ) &&
        ( NS(Particles_get_type_id)( particles ) == NS(BLOCK_TYPE_PARTICLE) ) )
    {
        NS(block_num_elements_t) const nn = 
            NS(Particles_get_num_particles)( particles );
        
        success = 1;
            
        if( nn > ( NS(block_num_elements_t) )0u )
        {
            typedef SIXTRL_GLOBAL_DEC unsigned char const* g_ptr_uchar_t;
            
            NS(block_size_t) const REAL_SIZE = sizeof( SIXTRL_REAL_T  );
            NS(block_size_t) const I64_SIZE  = sizeof( SIXTRL_INT64_T );
            
            NS(block_size_t) const REAL_BLOCK_SIZE = REAL_SIZE * nn;
            NS(block_size_t) const I64_BLOCK_SIZE  = I64_SIZE  * nn;
            
            /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
            
            g_ptr_uchar_t prev_ptr = 
                ( g_ptr_uchar_t  )NS(Particles_get_const_q0)( particles );                
                
            g_ptr_uchar_t ptr = 
                ( g_ptr_uchar_t  )NS(Particles_get_const_mass0)( particles );
                
            ptrdiff_t temp_dist = ( ptrdiff_t )( ptr - prev_ptr );
            
            if( ( temp_dist <= 0 ) || ( ptr == 0 ) || ( prev_ptr == 0 ) ||
                ( ( ( NS(block_size_t) )temp_dist ) < REAL_BLOCK_SIZE ) ||
                ( ( ( ( uintptr_t )ptr ) % REAL_SIZE ) != 0 ) )
            {
                success = 0;                
            }
                
            /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
                
            if( success )
            {
                prev_ptr = ptr;
                ptr = ( g_ptr_uchar_t )NS(Particles_get_const_beta0)( 
                    particles );                    
                
                temp_dist = ( ptrdiff_t )( ptr - prev_ptr );
                
                if( ( temp_dist <= 0 ) || ( ptr == 0 ) || 
                    ( ( ( NS(block_size_t) )temp_dist ) < REAL_BLOCK_SIZE ) ||
                    ( ( ( ( uintptr_t )ptr ) % REAL_SIZE ) != 0 ) )
                {
                    success = 0;                
                }
            }
            
            /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
                
            if( success )
            {
                prev_ptr = ptr;
                ptr = ( g_ptr_uchar_t )NS(Particles_get_const_gamma0)( 
                    particles );
                
                temp_dist = ( ptrdiff_t )( ptr - prev_ptr );
                
                if( ( temp_dist <= 0 ) || ( ptr == 0 ) || 
                    ( ( ( NS(block_size_t) )temp_dist ) < REAL_BLOCK_SIZE ) ||
                    ( ( ( ( uintptr_t )ptr ) % REAL_SIZE ) != 0 ) )
                {
                    success = 0;                
                }
            }
            
            /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
                
            if( success )
            {
                prev_ptr = ptr;
                ptr = ( g_ptr_uchar_t  )NS(Particles_get_const_p0c)( 
                    particles );                    
                
                temp_dist = ( ptrdiff_t )( ptr - prev_ptr );
                
                if( ( temp_dist <= 0 ) || ( ptr == 0 ) || 
                    ( ( ( NS(block_size_t) )temp_dist ) < REAL_BLOCK_SIZE ) ||
                    ( ( ( ( uintptr_t )ptr ) % REAL_SIZE ) != 0 ) )
                {
                    success = 0;                
                }
            }
            
            /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
                
            if( success )
            {
                prev_ptr = ptr;
                ptr = ( g_ptr_uchar_t )NS(Particles_get_const_s)( particles );                    
                temp_dist = ( ptrdiff_t )( ptr - prev_ptr );
                
                if( ( temp_dist <= 0 ) || ( ptr == 0 ) || 
                    ( ( ( NS(block_size_t) )temp_dist ) < REAL_BLOCK_SIZE ) ||
                    ( ( ( ( uintptr_t )ptr ) % REAL_SIZE ) != 0 ) )
                {
                    success = 0;                
                }
            }
            
            /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
                
            if( success )
            {
                prev_ptr = ptr;
                ptr = ( g_ptr_uchar_t )NS(Particles_get_const_x)( particles );                    
                temp_dist = ( ptrdiff_t )( ptr - prev_ptr );
                
                if( ( temp_dist <= 0 ) || ( ptr == 0 ) || 
                    ( ( ( NS(block_size_t) )temp_dist ) < REAL_BLOCK_SIZE ) ||
                    ( ( ( ( uintptr_t )ptr ) % REAL_SIZE ) != 0 ) )
                {
                    success = 0;                
                }
            }
            
            /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
                
            if( success )
            {
                prev_ptr = ptr;
                ptr = ( g_ptr_uchar_t )NS(Particles_get_const_y)( particles );
                temp_dist = ( ptrdiff_t )( ptr - prev_ptr );
                
                if( ( temp_dist <= 0 ) || ( ptr == 0 ) || 
                    ( ( ( NS(block_size_t) )temp_dist ) < REAL_BLOCK_SIZE ) ||
                    ( ( ( ( uintptr_t )ptr ) % REAL_SIZE ) != 0 ) )
                {
                    success = 0;                
                }
            }
            
            /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
                
            if( success )
            {
                prev_ptr = ptr;
                ptr = ( g_ptr_uchar_t )NS(Particles_get_const_px)( particles );                    
                temp_dist = ( ptrdiff_t )( ptr - prev_ptr );
                
                if( ( temp_dist <= 0 ) || ( ptr == 0 ) || 
                    ( ( ( NS(block_size_t) )temp_dist ) < REAL_BLOCK_SIZE ) ||
                    ( ( ( ( uintptr_t )ptr ) % REAL_SIZE ) != 0 ) )
                {
                    success = 0;                
                }
            }
            
            /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
                
            if( success )
            {
                prev_ptr = ptr;
                ptr = ( g_ptr_uchar_t )NS(Particles_get_const_py)( particles );
                temp_dist = ( ptrdiff_t )( ptr - prev_ptr );
                
                if( ( temp_dist <= 0 ) || ( ptr == 0 ) || 
                    ( ( ( NS(block_size_t) )temp_dist ) < REAL_BLOCK_SIZE ) ||
                    ( ( ( ( uintptr_t )ptr ) % REAL_SIZE ) != 0 ) )
                {
                    success = 0;                
                }
            }
            
            /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
                
            if( success )
            {
                prev_ptr = ptr;
                ptr = ( g_ptr_uchar_t )NS(Particles_get_const_sigma)( 
                    particles );                    
                
                temp_dist = ( ptrdiff_t )( ptr - prev_ptr );
                
                if( ( temp_dist <= 0 ) || ( ptr == 0 ) || 
                    ( ( ( NS(block_size_t) )temp_dist ) < REAL_BLOCK_SIZE ) ||
                    ( ( ( ( uintptr_t )ptr ) % REAL_SIZE ) != 0 ) )
                {
                    success = 0;                
                }
            }
            
            /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
                
            if( success )
            {
                prev_ptr = ptr;
                ptr = ( g_ptr_uchar_t )NS(Particles_get_const_psigma)( 
                    particles );                    
                
                temp_dist = ( ptrdiff_t )( ptr - prev_ptr );
                
                if( ( temp_dist <= 0 ) || ( ptr == 0 ) || 
                    ( ( ( NS(block_size_t) )temp_dist ) < REAL_BLOCK_SIZE ) ||
                    ( ( ( ( uintptr_t )ptr ) % REAL_SIZE ) != 0 ) )
                {
                    success = 0;                
                }
            }
            
            /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
                
            if( success )
            {
                prev_ptr = ptr;
                ptr = ( g_ptr_uchar_t )NS(Particles_get_const_delta)( 
                    particles );                    
                
                temp_dist = ( ptrdiff_t )( ptr - prev_ptr );
                
                if( ( temp_dist <= 0 ) || ( ptr == 0 ) || 
                    ( ( ( NS(block_size_t) )temp_dist ) < REAL_BLOCK_SIZE ) ||
                    ( ( ( ( uintptr_t )ptr ) % REAL_SIZE ) != 0 ) )
                {
                    success = 0;                
                }
            }
            
            /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
                
            if( success )
            {
                prev_ptr = ptr;
                ptr = ( g_ptr_uchar_t )NS(Particles_get_const_rpp)( 
                    particles );                    
                
                temp_dist = ( ptrdiff_t )( ptr - prev_ptr );
                
                if( ( temp_dist <= 0 ) || ( ptr == 0 ) || 
                    ( ( ( NS(block_size_t) )temp_dist ) < REAL_BLOCK_SIZE ) ||
                    ( ( ( ( uintptr_t )ptr ) % REAL_SIZE ) != 0 ) )
                {
                    success = 0;                
                }
            }
            
            /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
                
            if( success )
            {
                prev_ptr = ptr;
                ptr = ( g_ptr_uchar_t )NS(Particles_get_const_rvv)( 
                    particles );                    
                
                temp_dist = ( ptrdiff_t )( ptr - prev_ptr );
                
                if( ( temp_dist <= 0 ) || ( ptr == 0 ) || 
                    ( ( ( NS(block_size_t) )temp_dist ) < REAL_BLOCK_SIZE ) ||
                    ( ( ( ( uintptr_t )ptr ) % REAL_SIZE ) != 0 ) )
                {
                    success = 0;                
                }
            }
            
            /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
                
            if( success )
            {
                prev_ptr = ptr;
                ptr = ( g_ptr_uchar_t )NS(Particles_get_const_chi)( 
                    particles );                    
                
                temp_dist = ( ptrdiff_t )( ptr - prev_ptr );
                
                if( ( temp_dist <= 0 ) || ( ptr == 0 ) || 
                    ( ( ( NS(block_size_t) )temp_dist ) < REAL_BLOCK_SIZE ) ||
                    ( ( ( ( uintptr_t )ptr ) % REAL_SIZE ) != 0 ) )
                {
                    success = 0;                
                }
            }
            
            /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
                
            if( success )
            {
                prev_ptr = ptr;
                ptr = ( g_ptr_uchar_t )NS(Particles_get_const_particle_id)( 
                    particles );                    
                
                temp_dist = ( ptrdiff_t )( ptr - prev_ptr );
                
                if( ( temp_dist <= 0 ) || ( ptr == 0 ) || 
                    ( ( ( NS(block_size_t) )temp_dist ) < I64_BLOCK_SIZE ) ||
                    ( ( ( ( uintptr_t )ptr ) % I64_SIZE ) != 0 ) )
                {
                    success = 0;                
                }
            }
            
            /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
                
            if( success )
            {
                prev_ptr = ptr;
                ptr = ( g_ptr_uchar_t                     
                    )NS(Particles_get_const_lost_at_element_id)( particles );
                
                temp_dist = ( ptrdiff_t )( ptr - prev_ptr );
                
                if( ( temp_dist <= 0 ) || ( ptr == 0 ) || 
                    ( ( ( NS(block_size_t) )temp_dist ) < I64_BLOCK_SIZE ) ||
                    ( ( ( ( uintptr_t )ptr ) % I64_SIZE ) != 0 ) )
                {
                    success = 0;                
                }
            }
            
            /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
                
            if( success )
            {
                prev_ptr = ptr;
                ptr = ( g_ptr_uchar_t )NS(Particles_get_const_lost_at_turn)( 
                    particles );
                
                temp_dist = ( ptrdiff_t )( ptr - prev_ptr );
                
                if( ( temp_dist <= 0 ) || ( ptr == 0 ) || 
                    ( ( ( NS(block_size_t) )temp_dist ) < I64_BLOCK_SIZE ) ||
                    ( ( ( ( uintptr_t )ptr ) % I64_SIZE ) != 0 ) )
                {
                    success = 0;                
                }
            }
            
            /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
                
            if( success )
            {
                prev_ptr = ptr;
                ptr = ( g_ptr_uchar_t )NS(Particles_get_const_state)( 
                    particles );
                
                temp_dist = ( ptrdiff_t )( ptr - prev_ptr );
                
                if( ( temp_dist <= 0 ) || ( ptr == 0 ) || 
                    ( ( ( NS(block_size_t) )temp_dist ) < I64_BLOCK_SIZE ) ||
                    ( ( ( ( uintptr_t )ptr ) % I64_SIZE ) != 0 ) )
                {
                    success = 0;                
                }
            }
        }
        else
        {
            success &= (
                ( ( NS(Particles_get_const_q0)(     particles ) ) == 0 ) &&
                ( ( NS(Particles_get_const_mass0)(  particles ) ) == 0 ) &&
                ( ( NS(Particles_get_const_beta0)(  particles ) ) == 0 ) &&
                ( ( NS(Particles_get_const_gamma0)( particles ) ) == 0 ) &&
                ( ( NS(Particles_get_const_p0c)(    particles ) ) == 0 ) &&
                ( ( NS(Particles_get_const_s)(      particles ) ) == 0 ) &&
                ( ( NS(Particles_get_const_x)(      particles ) ) == 0 ) &&
                ( ( NS(Particles_get_const_y)(      particles ) ) == 0 ) &&
                ( ( NS(Particles_get_const_px)(     particles ) ) == 0 ) &&
                ( ( NS(Particles_get_const_py)(     particles ) ) == 0 ) &&
                ( ( NS(Particles_get_const_sigma)(  particles ) ) == 0 ) &&
                ( ( NS(Particles_get_const_psigma)( particles ) ) == 0 ) &&
                ( ( NS(Particles_get_const_delta)(  particles ) ) == 0 ) &&
                ( ( NS(Particles_get_const_rpp)(    particles ) ) == 0 ) &&
                ( ( NS(Particles_get_const_rvv)(    particles ) ) == 0 ) &&
                ( ( NS(Particles_get_const_chi)(    particles ) ) == 0 ) &&
                ( ( NS(Particles_get_const_particle_id)( 
                    particles ) ) == 0 ) &&
                ( ( NS(Particles_get_const_lost_at_element_id)( 
                    particles ) ) == 0 ) &&
                ( ( NS(Particles_get_const_lost_at_turn)( 
                    particles ) ) == 0 ) &&
                ( ( NS(Particles_get_const_state)( particles ) ) == 0 ) ) 
            ? 1 : 0;
        }        
    }
    
    return success;
}

/* ------------------------------------------------------------------------ */

SIXTRL_INLINE int NS(Particles_assign_all_ptrs)(
    NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_GLOBAL_DEC unsigned char** SIXTRL_RESTRICT attribute_ptrs, 
    NS(block_size_t) const num_attributes )
{
    int success = -1;
    
    if( ( particles != 0 ) && ( attribute_ptrs != 0 ) && 
        ( num_attributes == ( NS(block_size_t) )20u ) )
    {        
        typedef SIXTRL_GLOBAL_DEC SIXTRL_REAL_T*    g_ptr_real_t;
        typedef SIXTRL_GLOBAL_DEC NS(element_id_t)* g_ptr_elemid_t;
        
        NS(Particles_assign_ptr_to_q0)( 
            particles, ( g_ptr_real_t )attribute_ptrs[  0 ] );
        
        NS(Particles_assign_ptr_to_mass0)( 
            particles, ( g_ptr_real_t )attribute_ptrs[  1 ] );
        
        NS(Particles_assign_ptr_to_beta0)( 
            particles, ( g_ptr_real_t )attribute_ptrs[  2 ] );
        
        NS(Particles_assign_ptr_to_gamma0)( 
            particles, ( g_ptr_real_t )attribute_ptrs[  3 ] );
        
        NS(Particles_assign_ptr_to_p0c)( 
            particles, ( g_ptr_real_t )attribute_ptrs[  4 ] );
        
        NS(Particles_assign_ptr_to_s)( 
            particles, ( g_ptr_real_t )attribute_ptrs[  5 ] );
        
        NS(Particles_assign_ptr_to_x)( 
            particles, ( g_ptr_real_t )attribute_ptrs[  6 ] );
        
        NS(Particles_assign_ptr_to_y)( 
            particles, ( g_ptr_real_t )attribute_ptrs[  7 ] );
        
        NS(Particles_assign_ptr_to_px)( 
            particles, ( g_ptr_real_t )attribute_ptrs[  8 ] );
        
        NS(Particles_assign_ptr_to_py)( 
            particles, ( g_ptr_real_t )attribute_ptrs[  9 ] );
        
        NS(Particles_assign_ptr_to_sigma)( 
            particles, ( g_ptr_real_t )attribute_ptrs[ 10 ] );
        
        NS(Particles_assign_ptr_to_psigma)( 
            particles, ( g_ptr_real_t )attribute_ptrs[ 11 ] );
        
        NS(Particles_assign_ptr_to_delta)( 
            particles, ( g_ptr_real_t )attribute_ptrs[ 12 ] );
        
        NS(Particles_assign_ptr_to_rpp)( 
            particles, ( g_ptr_real_t )attribute_ptrs[ 13 ] );
        
        NS(Particles_assign_ptr_to_rvv)( 
            particles, ( g_ptr_real_t )attribute_ptrs[ 14 ] );
        
        NS(Particles_assign_ptr_to_chi)( 
            particles, ( g_ptr_real_t )attribute_ptrs[ 15 ] );
        
        NS(Particles_assign_ptr_to_particle_id)( 
            particles, ( g_ptr_elemid_t )attribute_ptrs[ 16 ] );
        
        NS(Particles_assign_ptr_to_lost_at_element_id)( 
            particles, ( g_ptr_elemid_t )attribute_ptrs[ 17 ] );
        
        NS(Particles_assign_ptr_to_lost_at_turn)( 
            particles, ( g_ptr_elemid_t )attribute_ptrs[ 18 ] );
        
        NS(Particles_assign_ptr_to_state)( 
            particles, ( g_ptr_elemid_t )attribute_ptrs[ 19 ] );
        
        success = 0;
    }
    
    return success;
}

SIXTRL_INLINE int NS(Particles_create_on_memory)(
    NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_GLOBAL_DEC NS(BlockInfo)* SIXTRL_RESTRICT block_info, 
    SIXTRL_GLOBAL_DEC unsigned char* SIXTRL_RESTRICT mem_begin, 
    NS(block_size_t) max_num_bytes_on_mem )
{
    typedef SIXTRL_GLOBAL_DEC unsigned char* g_ptr_uchar_t;
    
    int success = -1;
    
    NS(block_num_elements_t) const num_elem = 
        NS(Particles_get_num_particles)( particles );
        
    NS(block_size_t) const num_attributes = ( NS(block_size_t) )20u;
    
    NS(block_size_t) const real_attr_size = 
        sizeof( SIXTRL_REAL_T ) * num_elem;
    
    NS(block_size_t) const elem_attr_size = 
        sizeof( NS(element_id_t ) ) * num_elem;
    
    g_ptr_uchar_t attributes_ptr[] = 
    { 
        0, 0, 0, 0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 0, 0, 0, 
        0, 0, 0, 0 
    };
        
    NS(block_size_t) num_bytes_for_attribute[] = 
    { 
        real_attr_size, real_attr_size, real_attr_size, real_attr_size,
        real_attr_size, real_attr_size, real_attr_size, real_attr_size,
        real_attr_size, real_attr_size, real_attr_size, real_attr_size,
        real_attr_size, real_attr_size, real_attr_size, real_attr_size,
        elem_attr_size, elem_attr_size, elem_attr_size, elem_attr_size
    };
            
    NS(BlockInfo_set_type_id)( block_info, NS(BLOCK_TYPE_PARTICLE) );
    NS(BlockInfo_set_num_elements)( block_info, num_elem );
    
    if( 0 == NS(BlockInfo_map_to_memory_aligned)(
        block_info, &attributes_ptr[ 0 ], &num_bytes_for_attribute[ 0 ], 
        num_attributes, NS(Particles_get_num_particles)( particles ), 
        NS(Particles_get_type_id)( particles ), mem_begin, 
        max_num_bytes_on_mem ) )
    {
        SIXTRL_ASSERT(
            ( NS(BlockInfo_has_common_alignment)( block_info ) ) &&
            ( NS(BlockInfo_get_common_alignment)( block_info ) != 0 ) &&            
            ( NS(BlockInfo_get_type_id_num)( block_info ) == 
              NS(Particles_get_type_id_num)( particles ) ) &&
            ( NS(BlockInfo_get_num_elements)( block_info ) ==
              NS(Particles_get_num_particles)( particles ) ) );
        
        NS(Particles_assign_all_ptrs)( particles, attributes_ptr, 
                                       num_attributes );
        
        SIXTRL_ASSERT( 
            ( NS(Particles_has_mapping)( particles ) ) &&
            ( NS(Particles_is_aligned_with)( 
                particles, NS(BlockInfo_get_common_alignment)( 
                    block_info ) ) ) );
        
        success = 0;
    }
    
    return success;
}

SIXTRL_INLINE int NS(Particles_remap_from_memory)(
    NS(Particles)* SIXTRL_RESTRICT particles, 
    const SIXTRL_GLOBAL_DEC NS(BlockInfo) *const SIXTRL_RESTRICT block_info,
    SIXTRL_GLOBAL_DEC unsigned char* SIXTRL_RESTRICT mem_begin, 
    NS(block_size_t) const max_num_bytes_on_mem )
{
    typedef SIXTRL_GLOBAL_DEC unsigned char* g_ptr_uchar_t;
    
    int success = -1;
    
    NS(BlockType) const type_id = 
        NS(BlockInfo_get_type_id)( block_info );
    
    NS(block_num_elements_t) const num_particles = 
        NS(BlockInfo_get_num_elements)( block_info );
        
    NS(block_size_t) const num_attributes = ( NS(block_size_t) )20u;
    
    NS(block_size_t) const real_attr_size = 
        sizeof( SIXTRL_REAL_T ) * num_particles;
    
    NS(block_size_t) const elem_attr_size = 
        sizeof( NS(element_id_t ) ) * num_particles;
    
    g_ptr_uchar_t attributes_ptr[] = 
    { 
        0, 0, 0, 0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 0, 0, 0, 
        0, 0, 0, 0 
    };
        
    NS(block_size_t) num_bytes_for_attribute[] = 
    { 
        real_attr_size, real_attr_size, real_attr_size, real_attr_size,
        real_attr_size, real_attr_size, real_attr_size, real_attr_size,
        real_attr_size, real_attr_size, real_attr_size, real_attr_size,
        real_attr_size, real_attr_size, real_attr_size, real_attr_size,
        elem_attr_size, elem_attr_size, elem_attr_size, elem_attr_size
    };
    
    if( ( num_particles > ( NS(block_num_elements_t) )0u ) &&
        ( type_id == NS(BLOCK_TYPE_PARTICLE) ) &&
        ( 0 == NS(BlockInfo_remap_from_memory_aligned)(
            block_info, &attributes_ptr[ 0 ], &num_bytes_for_attribute[ 0 ],
            num_attributes, mem_begin, max_num_bytes_on_mem ) ) )
    {
        NS(Particles_preset)( particles );
        
        SIXTRL_ASSERT(
            ( NS(BlockInfo_get_num_elements)( block_info ) == 
                num_particles ) &&
            ( NS(BlockInfo_get_type_id)( block_info ) == type_id ) &&
            ( NS(BlockInfo_has_common_alignment)( block_info ) ) &&
            ( NS(BlockInfo_get_common_alignment)( block_info ) != 0 ) );          
        
        NS(Particles_set_type_id)( particles, type_id );
        NS(Particles_set_num_particles)( particles, num_particles );
        NS(Particles_assign_all_ptrs)( particles, attributes_ptr, 
                                       num_attributes );
        
        SIXTRL_ASSERT( NS(Particles_has_mapping)( particles ) );
        SIXTRL_ASSERT( NS(Particles_is_aligned_with)( 
            particles, NS(BlockInfo_get_common_alignment)( block_info ) ) );
        
        success = 0;
    }
    
    return success;
}

/* ------------------------------------------------------------------------- */

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* SIXTRACKLIB_COMMON_IMPL_PARTICLES_IMPL_H__ */

/* end: sixtracklib/common/impl/particles_impl.h */
