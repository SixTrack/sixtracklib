#ifndef SIXTRACKLIB_COMMON_IMPL_PARTICLES_API_H__
#define SIXTRACKLIB_COMMON_IMPL_PARTICLES_API_H__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <assert.h>
    #include <stdbool.h>
    #include <stdint.h>
    #include <stdlib.h>
    #include <string.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/_impl/definitions.h"
    #include "sixtracklib/common/impl/particles_type.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined( _GPUCODE ) && defined( __cplusplus ) */

struct NS(Particles);

SIXTRL_FN SIXTRL_STATIC NS(Particles)* NS(Particles_preset)(
    NS(Particles)* SIXTRL_RESTRICT particles );

SIXTRL_FN SIXTRL_STATIC void NS(Particles_preset_values)(
    struct NS(Particles)* SIXTRL_RESTRICT particles );

SIXTRL_FN SIXTRL_STATIC NS(block_num_elements_t)
NS(Particles_get_num_particles)(
    const NS(Particles) *const SIXTRL_RESTRICT particles );

SIXTRL_FN SIXTRL_STATIC void NS(Particles_set_num_particles)(
    NS(Particles)* SIXTRL_RESTRICT particles,
    NS(block_num_elements_t) const num_of_particles );

/* ------------------------------------------------------------------------- */

SIXTRL_FN SIXTRL_STATIC SIXTRL_GLOBAL_DEC NS(Particles) const*
NS(Blocks_get_const_particles)(
    const NS(BlockInfo) *const SIXTRL_RESTRICT block_info );

SIXTRL_FN SIXTRL_STATIC SIXTRL_GLOBAL_DEC NS(Particles)* NS(Blocks_get_particles)(
    NS(BlockInfo)* SIXTRL_RESTRICT block_info );

SIXTRL_FN SIXTRL_STATIC void NS(Particles_buffer_preset_values)(
    NS(Blocks)* SIXTRL_RESTRICT blocks );

/* ------------------------------------------------------------------------- */

SIXTRL_FN SIXTRL_STATIC void NS( Particles_copy_single_unchecked )(
    struct NS( Particles ) * SIXTRL_RESTRICT des,
    NS(block_num_elements_t) const des_id,
    const struct NS( Particles ) *const SIXTRL_RESTRICT src,
    NS(block_num_elements_t) const src_id );

SIXTRL_FN SIXTRL_STATIC void NS( Particles_copy_range_unchecked )(
    struct NS(Particles)* SIXTRL_RESTRICT destination,
    const struct NS(Particles) *const SIXTRL_RESTRICT source,
    NS(block_num_elements_t) const start_index,
    NS(block_num_elements_t) const end_index );

SIXTRL_FN SIXTRL_STATIC void NS( Particles_copy_all_unchecked )(
    struct NS( Particles ) * SIXTRL_RESTRICT des,
    const struct NS( Particles ) *const SIXTRL_RESTRICT src );

SIXTRL_FN SIXTRL_STATIC void NS( Particles_calculate_difference)(
    const struct NS(Particles) *const SIXTRL_RESTRICT lhs,
    const struct NS(Particles) *const SIXTRL_RESTRICT rhs,
    struct NS(Particles)* SIXTRL_RESTRICT diff );

SIXTRL_FN SIXTRL_STATIC void NS( Particles_get_max_value)(
    struct NS(Particles)* SIXTRL_RESTRICT destination,
    SIXTRL_GLOBAL_DEC NS(block_size_t)* SIXTRL_RESTRICT max_value_index,
    const struct NS(Particles) *const SIXTRL_RESTRICT source );

SIXTRL_FN SIXTRL_STATIC void NS( Particles_buffer_calculate_difference)(
    const struct NS(Blocks) *const SIXTRL_RESTRICT lhs,
    const struct NS(Blocks) *const SIXTRL_RESTRICT rhs,
    struct NS(Blocks)* SIXTRL_RESTRICT diff );

SIXTRL_FN SIXTRL_STATIC void NS( Particles_buffer_get_max_value )(
    struct NS(Blocks)* SIXTRL_RESTRICT destination,
    SIXTRL_GLOBAL_DEC NS(block_size_t)* SIXTRL_RESTRICT max_value_index,
    const struct NS(Blocks) *const SIXTRL_RESTRICT source );

/* ------------------------------------------------------------------------- */

SIXTRL_FN SIXTRL_STATIC SIXTRL_REAL_T NS(Particles_get_q0_value)(
    const NS(Particles) *const SIXTRL_RESTRICT particles,
    NS(block_num_elements_t) const ii );

SIXTRL_FN SIXTRL_STATIC SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const*
NS(Particles_get_const_q0)(
    const NS(Particles) *const SIXTRL_RESTRICT particles );

SIXTRL_FN SIXTRL_STATIC SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* NS(Particles_get_q0)(
    NS(Particles)* SIXTRL_RESTRICT particles );

SIXTRL_FN SIXTRL_STATIC void NS(Particles_set_q0)(
    NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT ptr_to_q0s );

SIXTRL_FN SIXTRL_STATIC void NS(Particles_set_q0_value)(
    NS(Particles)* SIXTRL_RESTRICT particles,
    NS(block_num_elements_t) const ii,
    SIXTRL_REAL_T const q0_value );

SIXTRL_FN SIXTRL_STATIC void NS(Particles_assign_ptr_to_q0)(
    NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* ptr_to_q0s );

/* ------------------------------------------------------------------------- */

SIXTRL_FN SIXTRL_STATIC SIXTRL_REAL_T NS(Particles_get_mass0_value)(
    const NS(Particles) *const SIXTRL_RESTRICT particles,
    NS(block_num_elements_t) const ii );

SIXTRL_FN SIXTRL_STATIC SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const*
NS(Particles_get_const_mass0)(
    const NS(Particles) *const SIXTRL_RESTRICT particles );

SIXTRL_FN SIXTRL_STATIC SIXTRL_GLOBAL_DEC SIXTRL_REAL_T*
NS(Particles_get_mass0)( NS(Particles)* SIXTRL_RESTRICT particles );

SIXTRL_FN SIXTRL_STATIC void NS(Particles_set_mass0)(
    NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT ptr_to_mass0s );

SIXTRL_FN SIXTRL_STATIC void NS(Particles_set_mass0_value)(
    NS(Particles)* SIXTRL_RESTRICT particles,
    NS(block_num_elements_t) const ii, SIXTRL_REAL_T const mass0_value );

SIXTRL_FN SIXTRL_STATIC void NS(Particles_assign_ptr_to_mass0)(
    NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* ptr_to_mass0s );

/* ------------------------------------------------------------------------- */

SIXTRL_FN SIXTRL_STATIC SIXTRL_REAL_T NS(Particles_get_beta0_value)(
    const NS(Particles) *const SIXTRL_RESTRICT particles,
    NS(block_num_elements_t) const ii );

SIXTRL_FN SIXTRL_STATIC SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const*
NS(Particles_get_const_beta0)(
    const NS(Particles) *const SIXTRL_RESTRICT particles );

SIXTRL_FN SIXTRL_STATIC SIXTRL_GLOBAL_DEC SIXTRL_REAL_T*
NS(Particles_get_beta0)( NS(Particles)* SIXTRL_RESTRICT particles );

SIXTRL_FN SIXTRL_STATIC void NS(Particles_set_beta0)(
    NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT ptr_to_beta0s );

SIXTRL_FN SIXTRL_STATIC void NS(Particles_set_beta0_value)(
    NS(Particles)* SIXTRL_RESTRICT particles,
    NS(block_num_elements_t) const ii,
    SIXTRL_REAL_T const beta0_value );

SIXTRL_FN SIXTRL_STATIC void NS(Particles_assign_ptr_to_beta0)(
    NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* ptr_to_beta0s );

/* ------------------------------------------------------------------------- */

SIXTRL_FN SIXTRL_STATIC SIXTRL_REAL_T NS(Particles_get_gamma0_value)(
    const NS(Particles) *const SIXTRL_RESTRICT particles,
    NS(block_num_elements_t) const ii );

SIXTRL_FN SIXTRL_STATIC SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const*
NS(Particles_get_const_gamma0)(
    const NS(Particles) *const SIXTRL_RESTRICT particles );

SIXTRL_FN SIXTRL_STATIC SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* NS(Particles_get_gamma0)(
    NS(Particles)* SIXTRL_RESTRICT particles );

SIXTRL_FN SIXTRL_STATIC void NS(Particles_set_gamma0)(
    NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT ptr_to_gamma0s );

SIXTRL_FN SIXTRL_STATIC void NS(Particles_set_gamma0_value)(
    NS(Particles)* SIXTRL_RESTRICT particles,
    NS(block_num_elements_t) const ii,
    SIXTRL_REAL_T const gamma0_value );

SIXTRL_FN SIXTRL_STATIC void NS(Particles_assign_ptr_to_gamma0)(
    NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* ptr_to_gamma0s );


/* ------------------------------------------------------------------------- */

SIXTRL_FN SIXTRL_STATIC SIXTRL_REAL_T NS(Particles_get_p0c_value)(
    const NS(Particles) *const SIXTRL_RESTRICT particles,
    NS(block_num_elements_t) const ii );

SIXTRL_FN SIXTRL_STATIC SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const*
NS(Particles_get_const_p0c)(
    const NS(Particles) *const SIXTRL_RESTRICT particles );

SIXTRL_FN SIXTRL_STATIC SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* NS(Particles_get_p0c)(
    NS(Particles)* SIXTRL_RESTRICT particles );

SIXTRL_FN SIXTRL_STATIC void NS(Particles_set_p0c)(
    NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT ptr_to_p0cs );

SIXTRL_FN SIXTRL_STATIC void NS(Particles_set_p0c_value)(
    NS(Particles)* SIXTRL_RESTRICT particles,
    NS(block_num_elements_t) const ii, SIXTRL_REAL_T const p0c_value );

SIXTRL_FN SIXTRL_STATIC void NS(Particles_assign_ptr_to_p0c)(
    NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* ptr_to_p0cs );

/* ------------------------------------------------------------------------- */

SIXTRL_FN SIXTRL_STATIC SIXTRL_REAL_T NS(Particles_get_s_value)(
    const NS(Particles) *const SIXTRL_RESTRICT particles,
    NS(block_num_elements_t) const ii );

SIXTRL_FN SIXTRL_STATIC SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const*
NS(Particles_get_const_s)(
    const NS(Particles) *const SIXTRL_RESTRICT particles );

SIXTRL_FN SIXTRL_STATIC SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* NS(Particles_get_s)(
    NS(Particles)* SIXTRL_RESTRICT particles );

SIXTRL_FN SIXTRL_STATIC void NS(Particles_set_s)(
    NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT ptr_to_ss );

SIXTRL_FN SIXTRL_STATIC void NS(Particles_set_s_value)(
    NS(Particles)* SIXTRL_RESTRICT particles,
    NS(block_num_elements_t) const ii, SIXTRL_REAL_T const s_value );

SIXTRL_FN SIXTRL_STATIC void NS(Particles_assign_ptr_to_s)(
    NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* ptr_to_ss );

/* ------------------------------------------------------------------------- */

SIXTRL_FN SIXTRL_STATIC SIXTRL_REAL_T NS(Particles_get_x_value)(
    const NS(Particles) *const SIXTRL_RESTRICT particles,
    NS(block_num_elements_t) const ii );

SIXTRL_FN SIXTRL_STATIC SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const*
NS(Particles_get_const_x)(
    const NS(Particles) *const SIXTRL_RESTRICT particles );

SIXTRL_FN SIXTRL_STATIC SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* NS(Particles_get_x)(
    NS(Particles)* SIXTRL_RESTRICT particles );

SIXTRL_FN SIXTRL_STATIC void NS(Particles_set_x)(
    NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT ptr_to_xs );

SIXTRL_FN SIXTRL_STATIC void NS(Particles_set_x_value)(
    NS(Particles)* SIXTRL_RESTRICT particles,
    NS(block_num_elements_t) const ii, SIXTRL_REAL_T const x_value );

SIXTRL_FN SIXTRL_STATIC void NS(Particles_assign_ptr_to_x)(
    NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* ptr_to_xs );

/* ------------------------------------------------------------------------- */

SIXTRL_FN SIXTRL_STATIC SIXTRL_REAL_T NS(Particles_get_y_value)(
    const NS(Particles) *const SIXTRL_RESTRICT particles,
    NS(block_num_elements_t) const ii );

SIXTRL_FN SIXTRL_STATIC SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const*
NS(Particles_get_const_y)(
    const NS(Particles) *const SIXTRL_RESTRICT particles );

SIXTRL_FN SIXTRL_STATIC SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* NS(Particles_get_y)(
    NS(Particles)* SIXTRL_RESTRICT particles );

SIXTRL_FN SIXTRL_STATIC void NS(Particles_set_y)(
    NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT ptr_to_ys );

SIXTRL_FN SIXTRL_STATIC void NS(Particles_set_y_value)(
    NS(Particles)* SIXTRL_RESTRICT particles,
    NS(block_num_elements_t) const ii, SIXTRL_REAL_T const y_value );

SIXTRL_FN SIXTRL_STATIC void NS(Particles_assign_ptr_to_y)(
    NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* ptr_to_ys );

/* ------------------------------------------------------------------------- */

SIXTRL_FN SIXTRL_STATIC SIXTRL_REAL_T NS(Particles_get_px_value)(
    const NS(Particles) *const SIXTRL_RESTRICT particles,
    NS(block_num_elements_t) const ii );

SIXTRL_FN SIXTRL_STATIC SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const*
NS(Particles_get_const_px)(
    const NS(Particles) *const SIXTRL_RESTRICT particles );

SIXTRL_FN SIXTRL_STATIC SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* NS(Particles_get_px)(
    NS(Particles)* SIXTRL_RESTRICT particles );

SIXTRL_FN SIXTRL_STATIC void NS(Particles_set_px)(
    NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT ptr_to_pxs );

SIXTRL_FN SIXTRL_STATIC void NS(Particles_set_px_value)(
    NS(Particles)* SIXTRL_RESTRICT particles, NS(block_num_elements_t) const ii,
    SIXTRL_REAL_T const px_value );

SIXTRL_FN SIXTRL_STATIC void NS(Particles_assign_ptr_to_px)(
    NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* ptr_to_pxs );

/* ------------------------------------------------------------------------- */

SIXTRL_FN SIXTRL_STATIC SIXTRL_REAL_T NS(Particles_get_py_value)(
    const NS(Particles) *const SIXTRL_RESTRICT particles,
    NS(block_num_elements_t) const ii );

SIXTRL_FN SIXTRL_STATIC SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const*
NS(Particles_get_const_py)(
    const NS(Particles) *const SIXTRL_RESTRICT particles );

SIXTRL_FN SIXTRL_STATIC SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* NS(Particles_get_py)(
    NS(Particles)* SIXTRL_RESTRICT particles );

SIXTRL_FN SIXTRL_STATIC void NS(Particles_set_py)(
    NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT ptr_to_pys );

SIXTRL_FN SIXTRL_STATIC void NS(Particles_set_py_value)(
    NS(Particles)* SIXTRL_RESTRICT particles,
    NS(block_num_elements_t) const ii, SIXTRL_REAL_T const py_value );

SIXTRL_FN SIXTRL_STATIC void NS(Particles_assign_ptr_to_py)(
    NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* ptr_to_pys );

/* ------------------------------------------------------------------------- */

SIXTRL_FN SIXTRL_STATIC SIXTRL_REAL_T NS(Particles_get_sigma_value)(
    const NS(Particles) *const SIXTRL_RESTRICT particles,
    NS(block_num_elements_t) const ii );

SIXTRL_FN SIXTRL_STATIC SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const*
NS(Particles_get_const_sigma)(
    const NS(Particles) *const SIXTRL_RESTRICT particles );

SIXTRL_FN SIXTRL_STATIC SIXTRL_GLOBAL_DEC SIXTRL_REAL_T*
NS(Particles_get_sigma)( NS(Particles)* SIXTRL_RESTRICT particles );

SIXTRL_FN SIXTRL_STATIC void NS(Particles_set_sigma)(
    NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT ptr_to_sigmas );

SIXTRL_FN SIXTRL_STATIC void NS(Particles_set_sigma_value)(
    NS(Particles)* SIXTRL_RESTRICT particles,
    NS(block_num_elements_t) const ii, SIXTRL_REAL_T const sigma_value );

SIXTRL_FN SIXTRL_STATIC void NS(Particles_assign_ptr_to_sigma)(
    NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* ptr_to_sigmas );

/* ------------------------------------------------------------------------- */

SIXTRL_FN SIXTRL_STATIC SIXTRL_REAL_T NS(Particles_get_psigma_value)(
    const NS(Particles) *const SIXTRL_RESTRICT particles,
    NS(block_num_elements_t) const ii );

SIXTRL_FN SIXTRL_STATIC SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const*
NS(Particles_get_const_psigma)(
    const NS(Particles) *const SIXTRL_RESTRICT particles );

SIXTRL_FN SIXTRL_STATIC SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* NS(Particles_get_psigma)(
    NS(Particles)* SIXTRL_RESTRICT particles );

SIXTRL_FN SIXTRL_STATIC void NS(Particles_set_psigma)(
    NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT ptr_to_psigmas );

SIXTRL_FN SIXTRL_STATIC void NS(Particles_set_psigma_value)(
    NS(Particles)* SIXTRL_RESTRICT particles,
    NS(block_num_elements_t) const ii, SIXTRL_REAL_T const psigma_value );

SIXTRL_FN SIXTRL_STATIC void NS(Particles_assign_ptr_to_psigma)(
    NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* ptr_to_psigmas );

/* ------------------------------------------------------------------------- */

SIXTRL_FN SIXTRL_STATIC SIXTRL_REAL_T NS(Particles_get_delta_value)(
    const NS(Particles) *const SIXTRL_RESTRICT particles,
    NS(block_num_elements_t) const ii );

SIXTRL_FN SIXTRL_STATIC SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const*
NS(Particles_get_const_delta)(
    const NS(Particles) *const SIXTRL_RESTRICT particles );

SIXTRL_FN SIXTRL_STATIC SIXTRL_GLOBAL_DEC SIXTRL_REAL_T*
NS(Particles_get_delta)( NS(Particles)* SIXTRL_RESTRICT particles );

SIXTRL_FN SIXTRL_STATIC void NS(Particles_set_delta)(
    NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT ptr_to_deltas );

SIXTRL_FN SIXTRL_STATIC void NS(Particles_set_delta_value)(
    NS(Particles)* SIXTRL_RESTRICT particles,
    NS(block_num_elements_t) const ii, SIXTRL_REAL_T const delta_value );

SIXTRL_FN SIXTRL_STATIC void NS(Particles_assign_ptr_to_delta)(
    NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* ptr_to_deltas );

/* ------------------------------------------------------------------------- */

SIXTRL_FN SIXTRL_STATIC SIXTRL_REAL_T NS(Particles_get_rpp_value)(
    const NS(Particles) *const SIXTRL_RESTRICT particles,
    NS(block_num_elements_t) const ii );

SIXTRL_FN SIXTRL_STATIC SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const*
NS(Particles_get_const_rpp)(
    const NS(Particles) *const SIXTRL_RESTRICT particles );

SIXTRL_FN SIXTRL_STATIC SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* NS(Particles_get_rpp)(
    NS(Particles)* SIXTRL_RESTRICT particles );

SIXTRL_FN SIXTRL_STATIC void NS(Particles_set_rpp)(
    NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT ptr_to_rpps );

SIXTRL_FN SIXTRL_STATIC void NS(Particles_set_rpp_value)(
    NS(Particles)* SIXTRL_RESTRICT particles,
    NS(block_num_elements_t) const ii, SIXTRL_REAL_T const rpp_value );

SIXTRL_FN SIXTRL_STATIC void NS(Particles_assign_ptr_to_rpp)(
    NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* ptr_to_rpps );

/* ------------------------------------------------------------------------- */

SIXTRL_FN SIXTRL_STATIC SIXTRL_REAL_T NS(Particles_get_rvv_value)(
    const NS(Particles) *const SIXTRL_RESTRICT particles,
    NS(block_num_elements_t) const ii );

SIXTRL_FN SIXTRL_STATIC SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const*
NS(Particles_get_const_rvv)(
    const NS(Particles) *const SIXTRL_RESTRICT particles );

SIXTRL_FN SIXTRL_STATIC SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* NS(Particles_get_rvv)(
    NS(Particles)* SIXTRL_RESTRICT particles );

SIXTRL_FN SIXTRL_STATIC void NS(Particles_set_rvv)(
    NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT ptr_to_rvvs );

SIXTRL_FN SIXTRL_STATIC void NS(Particles_set_rvv_value)(
    NS(Particles)* SIXTRL_RESTRICT particles,
    NS(block_num_elements_t) const ii, SIXTRL_REAL_T const rvv_value );

SIXTRL_FN SIXTRL_STATIC void NS(Particles_assign_ptr_to_rvv)(
    NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* ptr_to_rvvs );

/* ------------------------------------------------------------------------- */

SIXTRL_FN SIXTRL_STATIC SIXTRL_REAL_T NS(Particles_get_chi_value)(
    const NS(Particles) *const SIXTRL_RESTRICT particles,
    NS(block_num_elements_t) const ii );

SIXTRL_FN SIXTRL_STATIC SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const*
NS(Particles_get_const_chi)(
    const NS(Particles) *const SIXTRL_RESTRICT particles );

SIXTRL_FN SIXTRL_STATIC SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* NS(Particles_get_chi)(
    NS(Particles)* SIXTRL_RESTRICT particles );

SIXTRL_FN SIXTRL_STATIC void NS(Particles_set_chi)(
    NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT ptr_to_chis );

SIXTRL_FN SIXTRL_STATIC void NS(Particles_set_chi_value)(
    NS(Particles)* SIXTRL_RESTRICT particles,
    NS(block_num_elements_t) const ii, SIXTRL_REAL_T const chi_value );

SIXTRL_FN SIXTRL_STATIC void NS(Particles_assign_ptr_to_chi)(
    NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* ptr_to_chis );

/* ------------------------------------------------------------------------- */

SIXTRL_FN SIXTRL_STATIC SIXTRL_INT64_T NS(Particles_get_particle_id_value)(
    const NS(Particles) *const SIXTRL_RESTRICT particles,
    NS(block_num_elements_t) const ii );

SIXTRL_FN SIXTRL_STATIC SIXTRL_GLOBAL_DEC SIXTRL_INT64_T const*
NS(Particles_get_const_particle_id)(
    const NS(Particles) *const SIXTRL_RESTRICT particles );

SIXTRL_FN SIXTRL_STATIC SIXTRL_GLOBAL_DEC SIXTRL_INT64_T*
NS(Particles_get_particle_id)( NS(Particles)* SIXTRL_RESTRICT particles );

SIXTRL_FN SIXTRL_STATIC void NS(Particles_set_particle_id)(
    NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_GLOBAL_DEC SIXTRL_INT64_T const* SIXTRL_RESTRICT
        ptr_to_particle_ids );

SIXTRL_FN SIXTRL_STATIC void NS(Particles_set_particle_id_value)(
    NS(Particles)* SIXTRL_RESTRICT particles,
    NS(block_num_elements_t) const ii,
    SIXTRL_INT64_T const particle_id_value );

SIXTRL_FN SIXTRL_STATIC void NS(Particles_assign_ptr_to_particle_id)(
    NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_GLOBAL_DEC SIXTRL_INT64_T* ptr_to_particle_ids );

/* ------------------------------------------------------------------------- */

SIXTRL_FN SIXTRL_STATIC SIXTRL_INT64_T
NS(Particles_get_lost_at_element_id_value)(
    const NS(Particles) *const SIXTRL_RESTRICT particles,
    NS(block_num_elements_t) const ii );

SIXTRL_FN SIXTRL_STATIC SIXTRL_GLOBAL_DEC SIXTRL_INT64_T const*
NS(Particles_get_const_lost_at_element_id)(
    const NS(Particles) *const SIXTRL_RESTRICT particles );

SIXTRL_FN SIXTRL_STATIC SIXTRL_GLOBAL_DEC SIXTRL_INT64_T*
NS(Particles_get_lost_at_element_id)(
    NS(Particles)* SIXTRL_RESTRICT particles );

SIXTRL_FN SIXTRL_STATIC void NS(Particles_set_lost_at_element_id)(
    NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_GLOBAL_DEC SIXTRL_INT64_T const* SIXTRL_RESTRICT ptr_to_lost_at_element_ids );

SIXTRL_FN SIXTRL_STATIC void NS(Particles_set_lost_at_element_id_value)(
    NS(Particles)* SIXTRL_RESTRICT particles,
    NS(block_num_elements_t) const ii,
    SIXTRL_INT64_T const lost_at_element_id_value );

SIXTRL_FN SIXTRL_STATIC void NS(Particles_assign_ptr_to_lost_at_element_id)(
    NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_GLOBAL_DEC SIXTRL_INT64_T* ptr_to_lost_at_element_ids );

/* ------------------------------------------------------------------------- */

SIXTRL_FN SIXTRL_STATIC SIXTRL_INT64_T NS(Particles_get_lost_at_turn_value)(
    const NS(Particles) *const SIXTRL_RESTRICT particles,
    NS(block_num_elements_t) const ii );

SIXTRL_FN SIXTRL_STATIC SIXTRL_GLOBAL_DEC SIXTRL_INT64_T const*
NS(Particles_get_const_lost_at_turn)(
    const NS(Particles) *const SIXTRL_RESTRICT particles );

SIXTRL_FN SIXTRL_STATIC SIXTRL_GLOBAL_DEC
SIXTRL_INT64_T* NS(Particles_get_lost_at_turn)(
    NS(Particles)* SIXTRL_RESTRICT particles );

SIXTRL_FN SIXTRL_STATIC void NS(Particles_set_lost_at_turn)(
    NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_GLOBAL_DEC SIXTRL_INT64_T const* SIXTRL_RESTRICT ptr_to_lost_at_turns );

SIXTRL_FN SIXTRL_STATIC void NS(Particles_set_lost_at_turn_value)(
    NS(Particles)* SIXTRL_RESTRICT particles,
    NS(block_num_elements_t) const ii,
    SIXTRL_INT64_T const lost_at_turn_value );

SIXTRL_FN SIXTRL_STATIC void NS(Particles_assign_ptr_to_lost_at_turn)(
    NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_GLOBAL_DEC SIXTRL_INT64_T* ptr_to_lost_at_turns );

/* ------------------------------------------------------------------------- */

SIXTRL_FN SIXTRL_STATIC SIXTRL_INT64_T NS(Particles_get_state_value)(
    const NS(Particles) *const SIXTRL_RESTRICT particles,
    NS(block_num_elements_t) const ii );

SIXTRL_FN SIXTRL_STATIC SIXTRL_GLOBAL_DEC SIXTRL_INT64_T const*
NS(Particles_get_const_state)(
    const NS(Particles) *const SIXTRL_RESTRICT particles );

SIXTRL_FN SIXTRL_STATIC SIXTRL_GLOBAL_DEC SIXTRL_INT64_T* NS(Particles_get_state)(
    NS(Particles)* SIXTRL_RESTRICT particles );

SIXTRL_FN SIXTRL_STATIC void NS(Particles_set_state)(
    NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_GLOBAL_DEC SIXTRL_INT64_T const* SIXTRL_RESTRICT ptr_to_states );

SIXTRL_FN SIXTRL_STATIC void NS(Particles_set_state_value)(
    NS(Particles)* SIXTRL_RESTRICT particles,
    NS(block_num_elements_t) const ii, SIXTRL_INT64_T const state_value );

SIXTRL_FN SIXTRL_STATIC void NS(Particles_assign_ptr_to_state)(
    NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_GLOBAL_DEC SIXTRL_INT64_T* ptr_to_states );

#if !defined ( _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined ( _GPUCODE ) && defined( __cplusplus ) */

/* ************************************************************************ */
/* *****             Implementation of inline functions             ******* */
/* ************************************************************************ */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/particles.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined ( _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined ( _GPUCODE ) && defined( __cplusplus ) */

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

SIXTRL_INLINE void NS(Particles_preset_values)(
    NS(Particles)* SIXTRL_RESTRICT particles )
{
    NS(block_size_t) const num_of_particles =
        NS(Particles_get_num_particles)( particles );

    if( ( particles != 0 ) && ( num_of_particles > 0 ) )
    {
        NS(block_size_t) ii = 0;
        SIXTRL_REAL_T  const ZERO_REAL = ( SIXTRL_REAL_T )0.0;

        SIXTRL_INT64_T const PARTICLE_ID    = ( SIXTRL_INT64_T )-1;
        SIXTRL_INT64_T const ELEMENT_ID     = ( SIXTRL_INT64_T )-1;
        SIXTRL_INT64_T const TURN_ID        = ( SIXTRL_INT64_T )-1;
        SIXTRL_INT64_T const PARTICLE_STATE = ( SIXTRL_INT64_T )0;

        SIXTRL_ASSERT( NS(Particles_get_const_q0)(     particles ) != 0 );
        SIXTRL_ASSERT( NS(Particles_get_const_beta0)(  particles ) != 0 );
        SIXTRL_ASSERT( NS(Particles_get_const_mass0)(  particles ) != 0 );
        SIXTRL_ASSERT( NS(Particles_get_const_gamma0)( particles ) != 0 );
        SIXTRL_ASSERT( NS(Particles_get_const_p0c)(    particles ) != 0 );
        SIXTRL_ASSERT( NS(Particles_get_const_s)(      particles ) != 0 );
        SIXTRL_ASSERT( NS(Particles_get_const_x)(      particles ) != 0 );
        SIXTRL_ASSERT( NS(Particles_get_const_y)(      particles ) != 0 );
        SIXTRL_ASSERT( NS(Particles_get_const_px)(     particles ) != 0 );
        SIXTRL_ASSERT( NS(Particles_get_const_py)(     particles ) != 0 );
        SIXTRL_ASSERT( NS(Particles_get_const_sigma)(  particles ) != 0 );
        SIXTRL_ASSERT( NS(Particles_get_const_psigma)( particles ) != 0 );
        SIXTRL_ASSERT( NS(Particles_get_const_delta)(  particles ) != 0 );
        SIXTRL_ASSERT( NS(Particles_get_const_rpp)(    particles ) != 0 );
        SIXTRL_ASSERT( NS(Particles_get_const_rvv)(    particles ) != 0 );
        SIXTRL_ASSERT( NS(Particles_get_const_chi)(    particles ) != 0 );
        SIXTRL_ASSERT( NS(Particles_get_const_state)(  particles ) != 0 );
        SIXTRL_ASSERT( NS(Particles_get_lost_at_turn)( particles ) != 0 );

        SIXTRL_ASSERT( NS(Particles_get_const_particle_id)(
            particles ) != 0 );

        SIXTRL_ASSERT( NS(Particles_get_const_lost_at_element_id)(
            particles ) != 0 );

        for( ; ii < num_of_particles ; ++ii )
        {
            NS(Particles_set_q0_value)(     particles, ii, ZERO_REAL );
            NS(Particles_set_beta0_value)(  particles, ii, ZERO_REAL );
            NS(Particles_set_mass0_value)(  particles, ii, ZERO_REAL );
            NS(Particles_set_gamma0_value)( particles, ii, ZERO_REAL );
            NS(Particles_set_p0c_value)(    particles, ii, ZERO_REAL );

            NS(Particles_set_s_value)(      particles, ii, ZERO_REAL );
            NS(Particles_set_x_value)(      particles, ii, ZERO_REAL );
            NS(Particles_set_y_value)(      particles, ii, ZERO_REAL );
            NS(Particles_set_px_value)(     particles, ii, ZERO_REAL );
            NS(Particles_set_py_value)(     particles, ii, ZERO_REAL );
            NS(Particles_set_sigma_value)(  particles, ii, ZERO_REAL );

            NS(Particles_set_psigma_value)( particles, ii, ZERO_REAL );
            NS(Particles_set_delta_value)(  particles, ii, ZERO_REAL );
            NS(Particles_set_rpp_value)(    particles, ii, ZERO_REAL );
            NS(Particles_set_rvv_value)(    particles, ii, ZERO_REAL );
            NS(Particles_set_chi_value)(    particles, ii, ZERO_REAL );

            NS(Particles_set_particle_id_value)(
                particles, ii, PARTICLE_ID );

            NS(Particles_set_lost_at_element_id_value)(
                particles, ii, ELEMENT_ID );

            NS(Particles_set_lost_at_turn_value)( particles, ii, TURN_ID );
            NS(Particles_set_state_value)( particles, ii, PARTICLE_STATE );
        }
    }

    return;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE SIXTRL_GLOBAL_DEC NS(Particles) const*
NS(Blocks_get_const_particles)( const NS(BlockInfo)
    *const SIXTRL_RESTRICT block_info )
{
    NS(BlockType) const type_id =
        NS(BlockInfo_get_type_id)( block_info );

    SIXTRL_GLOBAL_DEC void const* ptr_begin =
        NS(BlockInfo_get_const_ptr_begin)( block_info );

    SIXTRL_ASSERT( ( ptr_begin == 0 ) ||
                   ( ( ( ( uintptr_t )ptr_begin ) % 8u ) == 0u ) );

    return ( type_id == NS(BLOCK_TYPE_PARTICLE) )
        ? ( SIXTRL_GLOBAL_DEC NS(Particles) const* )ptr_begin : 0;
}

SIXTRL_INLINE SIXTRL_GLOBAL_DEC NS(Particles)* NS(Blocks_get_particles)(
    NS(BlockInfo)* SIXTRL_RESTRICT block_info )
{
    return ( SIXTRL_GLOBAL_DEC NS(Particles)*
        )NS(Blocks_get_const_particles)( block_info );
}

SIXTRL_INLINE void NS(Particles_buffer_preset_values)(
    NS(Blocks)* SIXTRL_RESTRICT blocks )
{
    SIXTRL_GLOBAL_DEC NS(BlockInfo)* blocks_it =
        NS(Blocks_get_block_infos_begin)( blocks );

    if( blocks_it != 0 )
    {
        SIXTRL_GLOBAL_DEC NS(BlockInfo)* blocks_end =
            NS(Blocks_get_block_infos_end)( blocks );

        for( ; blocks_it != blocks_end ; ++blocks_it )
        {
            NS(Particles)* particles = 0;

            #if !defined( _GPUCODE )
            particles = NS(Blocks_get_particles)( blocks_it );
            #else /* defined( _GPUCODE ) */
            NS(Particles) temp_particles;

            NS(BlockInfo) info = *blocks_it;
            SIXTRL_GLOBAL_DEC NS(Particles)* ptr_to_particles =
                NS(Blocks_get_particles)( &info );

            if( ptr_to_particles != 0 )
            {
                temp_particles = *ptr_to_particles;
                particles = &temp_particles;
            }
            #endif /* !defined( _GPUCODE ) */

             NS(Particles_preset_values)( particles );
        }
    }

    return;
}

/* ------------------------------------------------------------------------- */

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

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE void NS( Particles_copy_single_unchecked )(
    struct NS( Particles ) * SIXTRL_RESTRICT des,
    NS(block_num_elements_t) const des_id,
    const struct NS( Particles ) *const SIXTRL_RESTRICT src,
    NS(block_num_elements_t) const src_id )
{
    SIXTRL_ASSERT(
        ( des != 0 ) && ( src != 0 ) &&
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

SIXTRL_INLINE void NS( Particles_copy_range_unchecked )(
    struct NS(Particles)* SIXTRL_RESTRICT destination,
    const struct NS(Particles) *const SIXTRL_RESTRICT source,
    NS(block_num_elements_t) const start_index,
    NS(block_num_elements_t) const end_index )
{
    NS(block_num_elements_t) num_to_copy = ( NS(block_num_elements_t ) )0u;

    SIXTRL_ASSERT(
        ( destination != 0 ) && ( source != 0 ) &&
        ( start_index >= 0 ) && ( start_index <= end_index ) &&
        ( NS(Particles_get_num_particles)( destination ) >= end_index ) &&
        ( NS(Particles_get_num_particles)( source ) >= end_index ) );

    num_to_copy = end_index - start_index;

    {
        SIXTRACKLIB_COPY_VALUES( SIXTRL_REAL_T, &destination->q0[ start_index ],
                             &source->q0[ start_index ], num_to_copy );
    }

    {
        SIXTRACKLIB_COPY_VALUES( SIXTRL_REAL_T,
                             &destination->beta0[ start_index ],
                             &source->beta0[ start_index ], num_to_copy );
    }

    {
        SIXTRACKLIB_COPY_VALUES( SIXTRL_REAL_T,
                             &destination->mass0[ start_index ],
                             &source->mass0[ start_index ], num_to_copy );
    }

    {
        SIXTRACKLIB_COPY_VALUES( SIXTRL_REAL_T,
                                 &destination->gamma0[ start_index ],
                                 &source->gamma0[ start_index ], num_to_copy );
    }

    {
        SIXTRACKLIB_COPY_VALUES( SIXTRL_REAL_T,
                             &destination->p0c[ start_index ],
                             &source->p0c[ start_index ], num_to_copy );
    }

    {
        SIXTRACKLIB_COPY_VALUES( SIXTRL_REAL_T,
                             &destination->s[ start_index ],
                             &source->s[ start_index ], num_to_copy );
    }

    {
        SIXTRACKLIB_COPY_VALUES( SIXTRL_REAL_T,
                                 &destination->x[ start_index ],
                                 &source->x[ start_index ], num_to_copy );
    }

    {
        SIXTRACKLIB_COPY_VALUES( SIXTRL_REAL_T,
                                 &destination->y[ start_index ],
                                 &source->y[ start_index ], num_to_copy );
    }

    {
        SIXTRACKLIB_COPY_VALUES( SIXTRL_REAL_T,
                                 &destination->px[ start_index ],
                                 &source->px[ start_index ], num_to_copy );
    }

    {
        SIXTRACKLIB_COPY_VALUES( SIXTRL_REAL_T,
                                 &destination->py[ start_index ],
                                 &source->py[ start_index ], num_to_copy );
    }

    {
        SIXTRACKLIB_COPY_VALUES( SIXTRL_REAL_T,
                                 &destination->sigma[ start_index ],
                                 &source->sigma[ start_index ], num_to_copy );
    }

    {
        SIXTRACKLIB_COPY_VALUES( SIXTRL_REAL_T,
                                 &destination->psigma[ start_index ],
                                 &source->psigma[ start_index ], num_to_copy );
    }

    {
        SIXTRACKLIB_COPY_VALUES( SIXTRL_REAL_T,
                                 &destination->delta[ start_index ],
                                 &source->delta[ start_index ], num_to_copy );
    }

    {
        SIXTRACKLIB_COPY_VALUES( SIXTRL_REAL_T,
                                 &destination->rpp[ start_index ],
                                 &source->rpp[ start_index ], num_to_copy );
    }

    {
        SIXTRACKLIB_COPY_VALUES( SIXTRL_REAL_T,
                                 &destination->rvv[ start_index ],
                                 &source->rvv[ start_index ], num_to_copy );
    }

    {
        SIXTRACKLIB_COPY_VALUES( SIXTRL_REAL_T,
                                 &destination->chi[ start_index ],
                                 &source->chi[ start_index ], num_to_copy );
    }

    {
        SIXTRACKLIB_COPY_VALUES( SIXTRL_INT64_T,
                                 &destination->particle_id[ start_index ],
                                 &source->particle_id[ start_index ],
                                 num_to_copy );
    }

    {
        SIXTRACKLIB_COPY_VALUES( SIXTRL_INT64_T,
                                 &destination->lost_at_element_id[ start_index ],
                                 &source->lost_at_element_id[ start_index ],
                                 num_to_copy );
    }

    {
        SIXTRACKLIB_COPY_VALUES( SIXTRL_INT64_T,
                                 &destination->lost_at_turn[ start_index ],
                                 &source->lost_at_turn[ start_index ],
                                 num_to_copy );
    }

    {
        SIXTRACKLIB_COPY_VALUES( SIXTRL_INT64_T,
                                 &destination->state[ start_index ],
                                 &source->state[ start_index ], num_to_copy );
    }

    return;
}

SIXTRL_INLINE void NS( Particles_copy_all_unchecked )(
    struct NS( Particles ) * SIXTRL_RESTRICT destination,
    const struct NS( Particles ) *const SIXTRL_RESTRICT source )
{
    NS(block_num_elements_t) const num =
        NS(Particles_get_num_particles)( source );

    SIXTRL_ASSERT(
        ( destination != 0 ) &&  ( source != 0 ) &&
        ( num >  ( NS(block_num_elements_t) )0u ) &&
        ( num == NS(Particles_get_num_particles)( destination ) ) );

    NS(Particles_copy_range_unchecked)( destination, source, 0, num );

    return;
}

SIXTRL_INLINE void NS( Particles_calculate_difference)(
    const NS(Particles) *const SIXTRL_RESTRICT lhs,
    const NS(Particles) *const SIXTRL_RESTRICT rhs,
    NS(Particles)* SIXTRL_RESTRICT diff )
{
    NS(block_num_elements_t) const num_particles =
        NS(Particles_get_num_particles)( lhs );

    if( ( lhs != 0 ) && ( rhs != 0 ) && ( diff != 0 ) &&
        ( num_particles > ( NS(block_num_elements_t ) )0u ) &&
        ( num_particles == NS(Particles_get_num_particles)( rhs  ) ) &&
        ( num_particles == NS(Particles_get_num_particles)( diff ) ) )
    {
        NS(block_num_elements_t) ii = 0;

        for( ; ii < num_particles ; ++ii )
        {
            NS(Particles_set_q0_value)( diff, ii,
                NS(Particles_get_q0_value)( lhs, ii ) -
                NS(Particles_get_q0_value)( rhs, ii ) );

            NS(Particles_set_mass0_value)( diff, ii,
                NS(Particles_get_mass0_value)( lhs, ii ) -
                NS(Particles_get_mass0_value)( rhs, ii ) );

            NS(Particles_set_beta0_value)( diff, ii,
                NS(Particles_get_beta0_value)( lhs, ii ) -
                NS(Particles_get_beta0_value)( rhs, ii ) );

            NS(Particles_set_gamma0_value)( diff, ii,
                NS(Particles_get_gamma0_value)( lhs, ii ) -
                NS(Particles_get_gamma0_value)( rhs, ii ) );

            NS(Particles_set_p0c_value)( diff, ii,
                NS(Particles_get_p0c_value)( lhs, ii ) -
                NS(Particles_get_p0c_value)( rhs, ii ) );

            NS(Particles_set_s_value)( diff, ii,
                NS(Particles_get_s_value)( lhs, ii ) -
                NS(Particles_get_s_value)( rhs, ii ) );

            NS(Particles_set_x_value)( diff, ii,
                NS(Particles_get_x_value)( lhs, ii ) -
                NS(Particles_get_x_value)( rhs, ii ) );

            NS(Particles_set_y_value)( diff, ii,
                NS(Particles_get_y_value)( lhs, ii ) -
                NS(Particles_get_y_value)( rhs, ii ) );

            NS(Particles_set_px_value)( diff, ii,
                NS(Particles_get_px_value)( lhs, ii ) -
                NS(Particles_get_px_value)( rhs, ii ) );

            NS(Particles_set_py_value)( diff, ii,
                NS(Particles_get_py_value)( lhs, ii ) -
                NS(Particles_get_py_value)( rhs, ii ) );

            NS(Particles_set_sigma_value)( diff, ii,
                NS(Particles_get_sigma_value)( lhs, ii ) -
                NS(Particles_get_sigma_value)( rhs, ii ) );

            NS(Particles_set_psigma_value)( diff, ii,
                NS(Particles_get_psigma_value)( lhs, ii ) -
                NS(Particles_get_psigma_value)( rhs, ii ) );

            NS(Particles_set_delta_value)( diff, ii,
                NS(Particles_get_delta_value)( lhs, ii ) -
                NS(Particles_get_delta_value)( rhs, ii ) );

            NS(Particles_set_rpp_value)( diff, ii,
                NS(Particles_get_rpp_value)( lhs, ii ) -
                NS(Particles_get_rpp_value)( rhs, ii ) );

            NS(Particles_set_rvv_value)( diff, ii,
                NS(Particles_get_rvv_value)( lhs, ii ) -
                NS(Particles_get_rvv_value)( rhs, ii ) );

            NS(Particles_set_chi_value)( diff, ii,
                NS(Particles_get_chi_value)( lhs, ii ) -
                NS(Particles_get_chi_value)( rhs, ii ) );

            NS(Particles_set_particle_id_value)( diff, ii,
                NS(Particles_get_particle_id_value)( lhs, ii ) -
                NS(Particles_get_particle_id_value)( rhs, ii ) );

            NS(Particles_set_lost_at_element_id_value)( diff, ii,
                NS(Particles_get_lost_at_element_id_value)( lhs, ii ) -
                NS(Particles_get_lost_at_element_id_value)( rhs, ii ) );

            NS(Particles_set_lost_at_turn_value)( diff, ii,
                NS(Particles_get_lost_at_turn_value)( lhs, ii ) -
                NS(Particles_get_lost_at_turn_value)( rhs, ii ) );

            NS(Particles_set_state_value)( diff, ii,
                NS(Particles_get_state_value)( lhs, ii ) -
                NS(Particles_get_state_value)( rhs, ii ) );
        }
    }

    return;
}

SIXTRL_INLINE void NS( Particles_get_max_value)(
    NS(Particles)* SIXTRL_RESTRICT destination,
    SIXTRL_GLOBAL_DEC NS(block_size_t)* SIXTRL_RESTRICT max_value_index,
    const NS(Particles) *const SIXTRL_RESTRICT source )
{
    SIXTRL_STATIC_VAR SIXTRL_REAL_T const ZERO = ( SIXTRL_REAL_T )0.0L;

    if( ( destination     != 0 ) && ( source != 0 ) &&
        ( NS(Particles_get_num_particles)( destination ) > 0u ) &&
        ( NS(Particles_get_num_particles)( source      ) > 0u ) )
    {
        typedef SIXTRL_GLOBAL_DEC SIXTRL_REAL_T*  g_real_ptr_t;
        typedef SIXTRL_GLOBAL_DEC SIXTRL_INT64_T* g_int64_ptr_t;

        NS(block_size_t) dummy_max_value_indices[ 20 ] =
        {
            0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
            0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u
        };

        g_real_ptr_t out_real_values_begin[ 16 ] =
        {
            NS(Particles_get_q0)(     destination ),
            NS(Particles_get_beta0)(  destination ),
            NS(Particles_get_mass0)(  destination ),
            NS(Particles_get_gamma0)( destination ),
            NS(Particles_get_p0c)(    destination ),
            NS(Particles_get_s)(      destination ),
            NS(Particles_get_x)(      destination ),
            NS(Particles_get_y)(      destination ),
            NS(Particles_get_px)(     destination ),
            NS(Particles_get_py)(     destination ),
            NS(Particles_get_sigma)(  destination ),
            NS(Particles_get_psigma)( destination ),
            NS(Particles_get_delta)(  destination ),
            NS(Particles_get_rpp)(    destination ),
            NS(Particles_get_rvv)(    destination ),
            NS(Particles_get_chi)(    destination )
        };

        g_int64_ptr_t out_int64_values_begin[ 4 ] =
        {
            NS(Particles_get_particle_id)(        destination ),
            NS(Particles_get_lost_at_element_id)( destination ),
            NS(Particles_get_lost_at_turn)(       destination ),
            NS(Particles_get_state)(              destination )
        };

        g_real_ptr_t in_real_values_begin[ 16 ] =
        {
            ( g_real_ptr_t )NS(Particles_get_const_q0)(     source ),
            ( g_real_ptr_t )NS(Particles_get_const_beta0)(  source ),
            ( g_real_ptr_t )NS(Particles_get_const_mass0)(  source ),
            ( g_real_ptr_t )NS(Particles_get_const_gamma0)( source ),
            ( g_real_ptr_t )NS(Particles_get_const_p0c)(    source ),
            ( g_real_ptr_t )NS(Particles_get_const_s)(      source ),
            ( g_real_ptr_t )NS(Particles_get_const_x)(      source ),
            ( g_real_ptr_t )NS(Particles_get_const_y)(      source ),
            ( g_real_ptr_t )NS(Particles_get_const_px)(     source ),
            ( g_real_ptr_t )NS(Particles_get_const_py)(     source ),
            ( g_real_ptr_t )NS(Particles_get_const_sigma)(  source ),
            ( g_real_ptr_t )NS(Particles_get_const_psigma)( source ),
            ( g_real_ptr_t )NS(Particles_get_const_delta)(  source ),
            ( g_real_ptr_t )NS(Particles_get_const_rpp)(    source ),
            ( g_real_ptr_t )NS(Particles_get_const_rvv)(    source ),
            ( g_real_ptr_t )NS(Particles_get_const_chi)(    source )
        };

        g_int64_ptr_t in_int64_values_begin[ 4 ] =
        {
            ( g_int64_ptr_t )NS(Particles_get_const_particle_id)( source ),
            ( g_int64_ptr_t )NS(Particles_get_const_lost_at_element_id)(
                source ),
            ( g_int64_ptr_t )NS(Particles_get_const_lost_at_turn)( source ),
            ( g_int64_ptr_t )NS(Particles_get_const_state)( source )
        };

        NS(block_size_t) ii = 0;
        NS(block_size_t) jj = 0;
        NS(block_size_t) const num_particles =
            NS(Particles_get_num_particles)( destination );

        g_real_ptr_t in_real_values_end[ 16 ] =
        {
            in_real_values_begin[  0 ] + num_particles,
            in_real_values_begin[  1 ] + num_particles,
            in_real_values_begin[  2 ] + num_particles,
            in_real_values_begin[  3 ] + num_particles,
            in_real_values_begin[  4 ] + num_particles,
            in_real_values_begin[  5 ] + num_particles,
            in_real_values_begin[  6 ] + num_particles,
            in_real_values_begin[  7 ] + num_particles,
            in_real_values_begin[  8 ] + num_particles,
            in_real_values_begin[  9 ] + num_particles,
            in_real_values_begin[ 10 ] + num_particles,
            in_real_values_begin[ 11 ] + num_particles,
            in_real_values_begin[ 12 ] + num_particles,
            in_real_values_begin[ 13 ] + num_particles,
            in_real_values_begin[ 14 ] + num_particles,
            in_real_values_begin[ 15 ] + num_particles
        };

        g_int64_ptr_t in_int64_values_end[ 4 ] =
        {
            in_int64_values_begin[  0 ] + num_particles,
            in_int64_values_begin[  1 ] + num_particles,
            in_int64_values_begin[  2 ] + num_particles,
            in_int64_values_begin[  3 ] + num_particles
        };

        for( ; ii < 16 ; ++ii )
        {
            g_real_ptr_t in_it  = in_real_values_begin[ ii ];
            g_real_ptr_t in_end = in_real_values_end[ ii ];

            SIXTRL_REAL_T max_value     = ( SIXTRL_REAL_T )0.0;
            SIXTRL_REAL_T cmp_max_value = max_value;

            NS(block_size_t) kk = ( NS(block_size_t) )0u;
            dummy_max_value_indices[ ii ] = ( NS(block_size_t) )0u;

            for( ; in_it != in_end ; ++in_it, ++kk )
            {
                SIXTRL_REAL_T const value = *in_it;
                SIXTRL_REAL_T const cmp_value =
                    ( value >= ZERO ) ? value : -value;

                if( cmp_max_value < cmp_value )
                {
                    max_value = value;
                    cmp_max_value = cmp_value;
                    dummy_max_value_indices[ ii ] = kk;
                }
            }

            *out_real_values_begin[ ii ] = max_value;
        }

        for( ii = 0, jj = 16 ; ii < 4 ; ++ii, ++jj )
        {
            g_int64_ptr_t in_it  = in_int64_values_begin[ ii ];
            g_int64_ptr_t in_end = in_int64_values_end[ ii ];

            SIXTRL_INT64_T max_value = ( SIXTRL_INT64_T )0;
            SIXTRL_INT64_T cmp_max_value = max_value;

            NS(block_size_t)      kk = ( NS(block_size_t ) )0u;
            dummy_max_value_indices[ jj ] = ( NS(block_size_t ) )0u;

            for( ; in_it != in_end ; ++in_it, ++kk )
            {
                SIXTRL_INT64_T const value = *in_it;
                SIXTRL_INT64_T const cmp_value = ( value > 0 ) ? value : -value;

                if( cmp_max_value < cmp_value )
                {
                    cmp_max_value = cmp_value;
                    max_value     = value;
                    dummy_max_value_indices[ jj ] = kk;
                }
            }

            *out_int64_values_begin[ ii ] = max_value;
        }

        if( max_value_index != 0 )
        {
            SIXTRACKLIB_COPY_VALUES( NS(block_size_t), max_value_index,
                                     &dummy_max_value_indices[ 0 ], 20 );

            max_value_index = max_value_index + 20;
        }
    }

    return;
}

SIXTRL_INLINE void NS( Particles_buffer_get_max_value )(
    NS(Blocks)* SIXTRL_RESTRICT destination,
    SIXTRL_GLOBAL_DEC NS(block_size_t)* SIXTRL_RESTRICT max_value_index,
    const NS(Blocks) *const SIXTRL_RESTRICT source )
{
    if( ( destination != 0 ) && ( source != 0 ) &&
        ( NS(Blocks_get_num_of_blocks)( destination   ) ==
          NS(Blocks_get_num_of_blocks)( source ) ) &&
        ( NS(Blocks_get_num_of_blocks)( destination ) > 0u ) )
    {
        SIXTRL_GLOBAL_DEC NS(BlockInfo)* dest_it =
            NS(Blocks_get_block_infos_begin)( destination );

        SIXTRL_GLOBAL_DEC NS(BlockInfo)* dest_end =
            NS(Blocks_get_block_infos_end)( destination );

        SIXTRL_GLOBAL_DEC NS(BlockInfo) const* src_it =
            NS(Blocks_get_const_block_infos_begin)( source );

        for( ; dest_it != dest_end ; ++dest_it, ++src_it )
        {
            NS(Particles)* dest_particles = 0;
            NS(Particles) const* source_particles = 0;

            #if defined( _GPUCODE ) && !defined( __CUDACC__ )

            NS(BlockInfo) temp_dest_info   = *dest_it;
            NS(BlockInfo) temp_source_info = *src_it;

            SIXTRL_GLOBAL_DEC NS(Particles)* ptr_dest_particles =
                NS(Blocks_get_particles)( &temp_dest_info );

            SIXTRL_GLOBAL_DEC NS(Particles) const* ptr_source_particles =
                NS(Blocks_get_const_particles)( &temp_source_info );

            NS(Particles) temp_dest_particles;
            NS(Particles) temp_source_particles;

            if( ptr_dest_particles != 0 )
            {
                temp_dest_particles = *ptr_dest_particles;
                dest_particles = &temp_dest_particles;
            }

            if( ptr_source_particles != 0 )
            {
                temp_source_particles = *ptr_source_particles;
                source_particles = &temp_source_particles;
            }

            #else

            dest_particles   = NS(Blocks_get_particles)( dest_it );
            source_particles = NS(Blocks_get_const_particles)( src_it );

            #endif /* defined( _GPUCODE ) && !defined( __CUDACC__ ) */

            NS(Particles_get_max_value)(
                dest_particles, max_value_index, source_particles );

            if( max_value_index != 0 ) max_value_index = max_value_index + 20;
        }
    }

    return;
}


SIXTRL_INLINE void NS( Particles_buffer_calculate_difference)(
    const NS(Blocks) *const SIXTRL_RESTRICT lhs,
    const NS(Blocks) *const SIXTRL_RESTRICT rhs,
    NS(Blocks)* SIXTRL_RESTRICT diff )
{
    SIXTRL_GLOBAL_DEC NS(BlockInfo) const* lhs_it  =
        NS(Blocks_get_const_block_infos_begin)( lhs );

    SIXTRL_GLOBAL_DEC NS(BlockInfo) const* lhs_end =
        NS(Blocks_get_const_block_infos_end)( lhs );

    SIXTRL_GLOBAL_DEC NS(BlockInfo) const* rhs_it  =
        NS(Blocks_get_const_block_infos_begin)( rhs );

    SIXTRL_GLOBAL_DEC NS(BlockInfo)*      diff_it  =
        NS(Blocks_get_block_infos_begin)( diff );


    if( ( lhs_it  != 0 ) && ( lhs_end != 0 ) &&
        ( rhs_it  != 0 ) && ( rhs_it  != lhs_it ) &&
        ( diff_it != 0 ) && ( diff_it != lhs_it ) && ( diff_it != rhs_it ) )
    {
        #if !defined( NDEBUG ) && !defined( _GPUCODE ) && !defined( __CUDACC__ )
        ptrdiff_t const num_blocks = lhs_end - lhs_it;

        SIXTRL_ASSERT(
            ( num_blocks == ( ptrdiff_t
                )NS(Blocks_get_num_of_blocks)( rhs  ) ) &&
            ( num_blocks == ( ptrdiff_t
                )NS(Blocks_get_num_of_blocks)( diff ) ) );
        
        #endif /* !defined( NDEBUG ) */

        for( ; lhs_it != lhs_end ; ++lhs_it, ++rhs_it, ++diff_it )
        {
            NS(Particles) const* lhs_particles = 0;
            NS(Particles) const* rhs_particles = 0;
            NS(Particles)* diff_particles = 0;

            #if defined( _GPUCODE ) && !defined( __CUDACC__ )

            NS(BlockInfo) temp_lhs_info  = *lhs_it;
            NS(BlockInfo) temp_rhs_info  = *rhs_it;
            NS(BlockInfo) temp_diff_info = *diff_it;

            SIXTRL_GLOBAL_DEC NS(Particles) const* ptr_lhs_particles =
                NS(Blocks_get_const_particles)( &temp_lhs_info );

            SIXTRL_GLOBAL_DEC NS(Particles) const* ptr_rhs_particles =
                NS(Blocks_get_const_particles)( &temp_rhs_info );

            SIXTRL_GLOBAL_DEC NS(Particles)* ptr_diff_particles =
                NS(Blocks_get_particles)( &temp_diff_info );

            NS(Particles) temp_lhs_particles;
            NS(Particles) temp_rhs_particles;
            NS(Particles) temp_diff_particles;

            if( ptr_lhs_particles != 0 )
            {
                temp_lhs_particles = *ptr_lhs_particles;
                lhs_particles = &temp_lhs_particles;
            }

            if( ptr_rhs_particles != 0 )
            {
                temp_rhs_particles = *ptr_rhs_particles;
                rhs_particles = &temp_rhs_particles;
            }

            if( ptr_diff_particles != 0 )
            {
                temp_diff_particles = *ptr_diff_particles;
                diff_particles = &temp_diff_particles;
            }

            #else

            lhs_particles  = NS(Blocks_get_const_particles)( lhs_it );
            rhs_particles  = NS(Blocks_get_const_particles)( rhs_it );
            diff_particles = NS(Blocks_get_particles)( diff_it );

            #endif /* defined( _GPUCODE ) && !defined( __CUDACC__ ) */

            NS(Particles_calculate_difference)(
                lhs_particles, rhs_particles, diff_particles );
        }
    }

    return;
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

#if  !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined( _GPUCODE ) && defined( __cplusplus ) */

#endif /* SIXTRACKLIB_COMMON_IMPL_PARTICLES_API_H__ */

/* end: sixtracklib/common/impl/particles_api.h */
