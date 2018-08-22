#ifndef SIXTRACKLIB_COMMON_PARTICLES_H__
#define SIXTRACKLIB_COMMON_PARTICLES_H__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stdbool.h>
    #include <stdint.h>
    #include <stdio.h>
    #include <stdlib.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/_impl/definitions.h"
    #include "sixtracklib/common/impl/buffer_type.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

/* ========================================================================= */

typedef SIXTRL_INT64_T  NS(particle_index_t);
typedef SIXTRL_REAL_T   NS(particle_real_t);
typedef SIXTRL_INT64_T  NS(particle_num_elements_t);

typedef SIXTRL_DATAPTR_DEC NS(particle_real_t)*
        NS(particle_real_ptr_t);

typedef SIXTRL_DATAPTR_DEC NS(particle_real_t) const*
        NS(particle_real_const_ptr_t);

typedef SIXTRL_DATAPTR_DEC NS(particle_index_t)*
        NS(particle_index_ptr_t);

typedef SIXTRL_DATAPTR_DEC NS(particle_index_t) const*
        NS(particle_index_const_ptr_t);

SIXTRL_STATIC_VAR NS(buffer_size_t) const
    NS(PARTICLES_NUM_DATAPTRS) = ( NS(buffer_size_t) )20u;

typedef struct NS(Particles)
{
    NS(particle_num_elements_t) num_particles              SIXTRL_ALIGN( 8 );

    NS(particle_real_ptr_t)  SIXTRL_RESTRICT q0            SIXTRL_ALIGN( 8 ); /* C */
    NS(particle_real_ptr_t)  SIXTRL_RESTRICT mass0         SIXTRL_ALIGN( 8 ); /* eV */
    NS(particle_real_ptr_t)  SIXTRL_RESTRICT beta0         SIXTRL_ALIGN( 8 ); /* nounit */
    NS(particle_real_ptr_t)  SIXTRL_RESTRICT gamma0        SIXTRL_ALIGN( 8 ); /* nounit */
    NS(particle_real_ptr_t)  SIXTRL_RESTRICT p0c           SIXTRL_ALIGN( 8 ); /* eV */

    NS(particle_real_ptr_t)  SIXTRL_RESTRICT s             SIXTRL_ALIGN( 8 ); /* [m] */
    NS(particle_real_ptr_t)  SIXTRL_RESTRICT x             SIXTRL_ALIGN( 8 ); /* [m] */
    NS(particle_real_ptr_t)  SIXTRL_RESTRICT y             SIXTRL_ALIGN( 8 ); /* [m] */
    NS(particle_real_ptr_t)  SIXTRL_RESTRICT px            SIXTRL_ALIGN( 8 ); /* Px/P0 */
    NS(particle_real_ptr_t)  SIXTRL_RESTRICT py            SIXTRL_ALIGN( 8 ); /* Py/P0 */
    NS(particle_real_ptr_t)  SIXTRL_RESTRICT zeta          SIXTRL_ALIGN( 8 ); /* */

    NS(particle_real_ptr_t)  SIXTRL_RESTRICT psigma        SIXTRL_ALIGN( 8 ); /* (E-E0) / (beta0 P0c) conjugate of sigma */
    NS(particle_real_ptr_t)  SIXTRL_RESTRICT delta         SIXTRL_ALIGN( 8 ); /* P/P0-1 = 1/rpp-1 */
    NS(particle_real_ptr_t)  SIXTRL_RESTRICT rpp           SIXTRL_ALIGN( 8 ); /* ratio P0 /P */
    NS(particle_real_ptr_t)  SIXTRL_RESTRICT rvv           SIXTRL_ALIGN( 8 ); /* ratio beta / beta0 */
    NS(particle_real_ptr_t)  SIXTRL_RESTRICT chi           SIXTRL_ALIGN( 8 ); /* q/q0 * m/m0  */

    NS(particle_index_ptr_t) SIXTRL_RESTRICT particle_id   SIXTRL_ALIGN( 8 );
    NS(particle_index_ptr_t) SIXTRL_RESTRICT at_element_id SIXTRL_ALIGN( 8 );
    NS(particle_index_ptr_t) SIXTRL_RESTRICT at_turn       SIXTRL_ALIGN( 8 );
    NS(particle_index_ptr_t) SIXTRL_RESTRICT state         SIXTRL_ALIGN( 8 );
}
NS(Particles);

/* ========================================================================= */

SIXTRL_FN SIXTRL_STATIC bool NS(Particles_can_be_added)(
    SIXTRL_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t) const num_particles,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_objects,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_slots,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_dataptrs );

SIXTRL_FN SIXTRL_STATIC SIXTRL_ARGPTR_DEC NS(Particles)*
NS(Particles_new)( SIXTRL_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
                   NS(buffer_size_t) const num_particles );

SIXTRL_FN SIXTRL_STATIC SIXTRL_ARGPTR_DEC NS(Particles)*
NS(Particles_add)( SIXTRL_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const  num_particles,
    NS(particle_real_ptr_t)  q0_ptr,        NS(particle_real_ptr_t)  mass0_ptr,
    NS(particle_real_ptr_t)  beta0_ptr,     NS(particle_real_ptr_t)  gamma0_ptr,
    NS(particle_real_ptr_t)  p0c_ptr,       NS(particle_real_ptr_t)  s_ptr,
    NS(particle_real_ptr_t)  x_ptr,         NS(particle_real_ptr_t)  y_ptr,
    NS(particle_real_ptr_t)  px_ptr,        NS(particle_real_ptr_t)  py_ptr,
    NS(particle_real_ptr_t)  zeta_ptr,      NS(particle_real_ptr_t)  psigma_ptr,
    NS(particle_real_ptr_t)  delta_ptr,     NS(particle_real_ptr_t)  rpp_ptr,
    NS(particle_real_ptr_t)  rvv_ptr,       NS(particle_real_ptr_t)  chi_ptr,
    NS(particle_index_ptr_t) particle_id_ptr,
    NS(particle_index_ptr_t) at_element_id_ptr,
    NS(particle_index_ptr_t) at_turn_ptr,
    NS(particle_index_ptr_t) state_ptr );

/* ------------------------------------------------------------------------- */

SIXTRL_FN SIXTRL_STATIC SIXTRL_ARGPTR_DEC NS(Particles)* NS(Particles_preset)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles );

SIXTRL_FN SIXTRL_STATIC void NS(Particles_preset_values)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles );

SIXTRL_FN SIXTRL_STATIC NS(particle_num_elements_t)
NS(Particles_get_num_of_particles)(
    const SIXTRL_ARGPTR_DEC NS(Particles) *const SIXTRL_RESTRICT particles );

SIXTRL_FN SIXTRL_STATIC void NS(Particles_set_num_of_particles)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const num_of_particles );

/* ------------------------------------------------------------------------- */

SIXTRL_FN SIXTRL_STATIC SIXTRL_DATAPTR_DEC NS(Particles) const*
NS(BufferIndex_get_const_particles)( SIXTRL_ARGPTR_DEC const
    NS(Object) *const SIXTRL_RESTRICT index );

SIXTRL_FN SIXTRL_STATIC SIXTRL_DATAPTR_DEC NS(Particles)*
NS(BufferIndex_get_particles)( SIXTRL_ARGPTR_DEC const
    NS(Object) *const SIXTRL_RESTRICT index );

/* ------------------------------------------------------------------------- */

SIXTRL_FN SIXTRL_STATIC void NS(Particles_copy_single_unchecked)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT destination,
    NS(particle_num_elements_t) const destination_index,
    SIXTRL_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT source,
    NS(particle_num_elements_t) const source_index );

SIXTRL_FN SIXTRL_STATIC void NS(Particles_copy_range_unchecked)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT destination,
    SIXTRL_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT source,
    NS(particle_num_elements_t) const start_index,
    NS(particle_num_elements_t) const end_index );

SIXTRL_FN SIXTRL_STATIC void NS(Particles_copy_all_unchecked)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT destination,
    SIXTRL_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT source );

SIXTRL_FN SIXTRL_STATIC void NS(Particles_calculate_difference)(
    SIXTRL_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT lhs,
    SIXTRL_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT rhs,
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT diff );

SIXTRL_FN SIXTRL_STATIC void NS( Particles_get_max_value)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT destination,
    SIXTRL_DATAPTR_DEC NS(particle_num_elements_t)* SIXTRL_RESTRICT max_value_index,
    SIXTRL_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT source );

SIXTRL_FN SIXTRL_STATIC void NS( Particles_buffer_calculate_difference)(
    SIXTRL_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT lhs,
    SIXTRL_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT rhs,
    SIXTRL_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT diff );

SIXTRL_FN SIXTRL_STATIC void NS( Particles_buffer_get_max_value )(
    SIXTRL_ARGPTR_DEC  NS(Buffer)* SIXTRL_RESTRICT destination,
    SIXTRL_DATAPTR_DEC NS(particle_num_elements_t)* SIXTRL_RESTRICT max_value_index,
    SIXTRL_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT source );

/* ------------------------------------------------------------------------- */

SIXTRL_FN SIXTRL_STATIC NS(particle_real_t) NS(Particles_get_q0_value)(
    SIXTRL_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const ii );

SIXTRL_FN SIXTRL_STATIC NS(particle_real_const_ptr_t) NS(Particles_get_const_q0)(
    SIXTRL_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT particles );

SIXTRL_FN SIXTRL_STATIC NS(particle_real_ptr_t) NS(Particles_get_q0)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles );

SIXTRL_FN SIXTRL_STATIC void NS(Particles_set_q0)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_real_const_ptr_t) SIXTRL_RESTRICT ptr_to_q0s );

SIXTRL_FN SIXTRL_STATIC void NS(Particles_set_q0_value)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const ii,
    NS(particle_real_t) const q0_value );

SIXTRL_FN SIXTRL_STATIC void NS(Particles_assign_ptr_to_q0)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_real_ptr_t) ptr_to_q0s );

/* ------------------------------------------------------------------------- */

SIXTRL_FN SIXTRL_STATIC NS(particle_real_t) NS(Particles_get_mass0_value)(
    SIXTRL_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const ii );

SIXTRL_FN SIXTRL_STATIC NS(particle_real_const_ptr_t)
NS(Particles_get_const_mass0)(
    SIXTRL_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT particles );

SIXTRL_FN SIXTRL_STATIC NS(particle_real_ptr_t)
NS(Particles_get_mass0)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles );

SIXTRL_FN SIXTRL_STATIC void NS(Particles_set_mass0)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_real_const_ptr_t) SIXTRL_RESTRICT ptr_to_mass0s );

SIXTRL_FN SIXTRL_STATIC void NS(Particles_set_mass0_value)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const ii,
    NS(particle_real_t) const mass0_value );

SIXTRL_FN SIXTRL_STATIC void NS(Particles_assign_ptr_to_mass0)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_real_ptr_t) ptr_to_mass0s );

/* ------------------------------------------------------------------------- */

SIXTRL_FN SIXTRL_STATIC NS(particle_real_t) NS(Particles_get_beta0_value)(
    SIXTRL_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const ii );

SIXTRL_FN SIXTRL_STATIC NS(particle_real_const_ptr_t)
NS(Particles_get_const_beta0)(
    SIXTRL_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT particles );

SIXTRL_FN SIXTRL_STATIC NS(particle_real_ptr_t)
NS(Particles_get_beta0)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles );

SIXTRL_FN SIXTRL_STATIC void NS(Particles_set_beta0)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_real_const_ptr_t) SIXTRL_RESTRICT ptr_to_beta0s );

SIXTRL_FN SIXTRL_STATIC void NS(Particles_set_beta0_value)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const ii,
    NS(particle_real_t) const beta0_value );

SIXTRL_FN SIXTRL_STATIC void NS(Particles_assign_ptr_to_beta0)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_real_ptr_t) ptr_to_beta0s );

/* ------------------------------------------------------------------------- */

SIXTRL_FN SIXTRL_STATIC NS(particle_real_t) NS(Particles_get_gamma0_value)(
    SIXTRL_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const ii );

SIXTRL_FN SIXTRL_STATIC NS(particle_real_const_ptr_t)
NS(Particles_get_const_gamma0)(
    SIXTRL_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT particles );

SIXTRL_FN SIXTRL_STATIC NS(particle_real_ptr_t) NS(Particles_get_gamma0)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles );

SIXTRL_FN SIXTRL_STATIC void NS(Particles_set_gamma0)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_real_const_ptr_t) SIXTRL_RESTRICT ptr_to_gamma0s );

SIXTRL_FN SIXTRL_STATIC void NS(Particles_set_gamma0_value)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const ii,
    NS(particle_real_t) const gamma0_value );

SIXTRL_FN SIXTRL_STATIC void NS(Particles_assign_ptr_to_gamma0)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_real_ptr_t) ptr_to_gamma0s );


/* ------------------------------------------------------------------------- */

SIXTRL_FN SIXTRL_STATIC NS(particle_real_t) NS(Particles_get_p0c_value)(
    SIXTRL_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const ii );

SIXTRL_FN SIXTRL_STATIC NS(particle_real_const_ptr_t) NS(Particles_get_const_p0c)(
    SIXTRL_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT particles );

SIXTRL_FN SIXTRL_STATIC NS(particle_real_ptr_t) NS(Particles_get_p0c)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles );

SIXTRL_FN SIXTRL_STATIC void NS(Particles_set_p0c)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_real_const_ptr_t) SIXTRL_RESTRICT ptr_to_p0cs );

SIXTRL_FN SIXTRL_STATIC void NS(Particles_set_p0c_value)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const ii,
    NS(particle_real_t) const p0c_value );

SIXTRL_FN SIXTRL_STATIC void NS(Particles_assign_ptr_to_p0c)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_real_ptr_t) ptr_to_p0cs );

/* ------------------------------------------------------------------------- */

SIXTRL_FN SIXTRL_STATIC NS(particle_real_t) NS(Particles_get_s_value)(
    SIXTRL_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const ii );

SIXTRL_FN SIXTRL_STATIC NS(particle_real_const_ptr_t) NS(Particles_get_const_s)(
    SIXTRL_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT particles );

SIXTRL_FN SIXTRL_STATIC NS(particle_real_ptr_t) NS(Particles_get_s)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles );

SIXTRL_FN SIXTRL_STATIC void NS(Particles_set_s)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_real_const_ptr_t) SIXTRL_RESTRICT ptr_to_ss );

SIXTRL_FN SIXTRL_STATIC void NS(Particles_set_s_value)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const ii,
    NS(particle_real_t) const s_value );

SIXTRL_FN SIXTRL_STATIC void NS(Particles_assign_ptr_to_s)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_real_ptr_t) ptr_to_ss );

/* ------------------------------------------------------------------------- */

SIXTRL_FN SIXTRL_STATIC NS(particle_real_t) NS(Particles_get_x_value)(
    SIXTRL_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const ii );

SIXTRL_FN SIXTRL_STATIC NS(particle_real_const_ptr_t)
NS(Particles_get_const_x)(
    SIXTRL_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT particles );

SIXTRL_FN SIXTRL_STATIC NS(particle_real_ptr_t) NS(Particles_get_x)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles );

SIXTRL_FN SIXTRL_STATIC void NS(Particles_set_x)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_real_const_ptr_t) SIXTRL_RESTRICT ptr_to_xs );

SIXTRL_FN SIXTRL_STATIC void NS(Particles_set_x_value)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const ii,
    NS(particle_real_t) const x_value );

SIXTRL_FN SIXTRL_STATIC void NS(Particles_assign_ptr_to_x)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_real_ptr_t) ptr_to_xs );

/* ------------------------------------------------------------------------- */

SIXTRL_FN SIXTRL_STATIC NS(particle_real_t) NS(Particles_get_y_value)(
    SIXTRL_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const ii );

SIXTRL_FN SIXTRL_STATIC NS(particle_real_const_ptr_t) NS(Particles_get_const_y)(
    SIXTRL_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT particles );

SIXTRL_FN SIXTRL_STATIC NS(particle_real_ptr_t) NS(Particles_get_y)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles );

SIXTRL_FN SIXTRL_STATIC void NS(Particles_set_y)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_real_const_ptr_t) SIXTRL_RESTRICT ptr_to_ys );

SIXTRL_FN SIXTRL_STATIC void NS(Particles_set_y_value)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const ii,
    NS(particle_real_t) const y_value );

SIXTRL_FN SIXTRL_STATIC void NS(Particles_assign_ptr_to_y)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_real_ptr_t) ptr_to_ys );

/* ------------------------------------------------------------------------- */

SIXTRL_FN SIXTRL_STATIC NS(particle_real_t) NS(Particles_get_px_value)(
    SIXTRL_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const ii );

SIXTRL_FN SIXTRL_STATIC NS(particle_real_const_ptr_t)
NS(Particles_get_const_px)(
    SIXTRL_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT particles );

SIXTRL_FN SIXTRL_STATIC NS(particle_real_ptr_t) NS(Particles_get_px)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles );

SIXTRL_FN SIXTRL_STATIC void NS(Particles_set_px)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_real_const_ptr_t) SIXTRL_RESTRICT ptr_to_pxs );

SIXTRL_FN SIXTRL_STATIC void NS(Particles_set_px_value)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const ii,
    NS(particle_real_t) const px_value );

SIXTRL_FN SIXTRL_STATIC void NS(Particles_assign_ptr_to_px)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_real_ptr_t) ptr_to_pxs );

/* ------------------------------------------------------------------------- */

SIXTRL_FN SIXTRL_STATIC NS(particle_real_t) NS(Particles_get_py_value)(
    SIXTRL_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const ii );

SIXTRL_FN SIXTRL_STATIC NS(particle_real_const_ptr_t)
NS(Particles_get_const_py)(
    SIXTRL_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT particles );

SIXTRL_FN SIXTRL_STATIC NS(particle_real_ptr_t) NS(Particles_get_py)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles );

SIXTRL_FN SIXTRL_STATIC void NS(Particles_set_py)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_real_const_ptr_t) SIXTRL_RESTRICT ptr_to_pys );

SIXTRL_FN SIXTRL_STATIC void NS(Particles_set_py_value)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const ii,
    NS(particle_real_t) const py_value );

SIXTRL_FN SIXTRL_STATIC void NS(Particles_assign_ptr_to_py)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_real_ptr_t) ptr_to_pys );

/* ------------------------------------------------------------------------- */

SIXTRL_FN SIXTRL_STATIC NS(particle_real_t) NS(Particles_get_zeta_value)(
    SIXTRL_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const ii );

SIXTRL_FN SIXTRL_STATIC NS(particle_real_const_ptr_t)
NS(Particles_get_const_zeta)(
    SIXTRL_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT particles );

SIXTRL_FN SIXTRL_STATIC NS(particle_real_ptr_t)
NS(Particles_get_zeta)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles );

SIXTRL_FN SIXTRL_STATIC void NS(Particles_set_zeta)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_real_const_ptr_t) SIXTRL_RESTRICT ptr_to_zetas );

SIXTRL_FN SIXTRL_STATIC void NS(Particles_set_zeta_value)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const ii,
    NS(particle_real_t) const zeta_value );

SIXTRL_FN SIXTRL_STATIC void NS(Particles_assign_ptr_to_zeta)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_real_ptr_t) ptr_to_zetas );

/* ------------------------------------------------------------------------- */

SIXTRL_FN SIXTRL_STATIC SIXTRL_REAL_T NS(Particles_get_psigma_value)(
    SIXTRL_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const ii );

SIXTRL_FN SIXTRL_STATIC NS(particle_real_const_ptr_t)
NS(Particles_get_const_psigma)(
    SIXTRL_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT particles );

SIXTRL_FN SIXTRL_STATIC NS(particle_real_ptr_t) NS(Particles_get_psigma)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles );

SIXTRL_FN SIXTRL_STATIC void NS(Particles_set_psigma)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_real_const_ptr_t) SIXTRL_RESTRICT ptr_to_psigmas );

SIXTRL_FN SIXTRL_STATIC void NS(Particles_set_psigma_value)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const ii,
    NS(particle_real_t) const psigma_value );

SIXTRL_FN SIXTRL_STATIC void NS(Particles_assign_ptr_to_psigma)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_real_ptr_t) ptr_to_psigmas );

/* ------------------------------------------------------------------------- */

SIXTRL_FN SIXTRL_STATIC NS(particle_real_t) NS(Particles_get_delta_value)(
    SIXTRL_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const ii );

SIXTRL_FN SIXTRL_STATIC NS(particle_real_const_ptr_t)
NS(Particles_get_const_delta)(
    SIXTRL_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT particles );

SIXTRL_FN SIXTRL_STATIC NS(particle_real_ptr_t)
NS(Particles_get_delta)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles );

SIXTRL_FN SIXTRL_STATIC void NS(Particles_set_delta)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_real_const_ptr_t) SIXTRL_RESTRICT ptr_to_deltas );

SIXTRL_FN SIXTRL_STATIC void NS(Particles_set_delta_value)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const ii,
    NS(particle_real_t) const delta_value );

SIXTRL_FN SIXTRL_STATIC void NS(Particles_assign_ptr_to_delta)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_real_ptr_t) ptr_to_deltas );

/* ------------------------------------------------------------------------- */

SIXTRL_FN SIXTRL_STATIC NS(particle_real_t) NS(Particles_get_rpp_value)(
    SIXTRL_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const ii );

SIXTRL_FN SIXTRL_STATIC NS(particle_real_const_ptr_t)
NS(Particles_get_const_rpp)(
    SIXTRL_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT particles );

SIXTRL_FN SIXTRL_STATIC NS(particle_real_ptr_t) NS(Particles_get_rpp)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles );

SIXTRL_FN SIXTRL_STATIC void NS(Particles_set_rpp)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_real_const_ptr_t) SIXTRL_RESTRICT ptr_to_rpps );

SIXTRL_FN SIXTRL_STATIC void NS(Particles_set_rpp_value)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const ii,
    NS(particle_real_t) const rpp_value );

SIXTRL_FN SIXTRL_STATIC void NS(Particles_assign_ptr_to_rpp)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_real_ptr_t) ptr_to_rpps );

/* ------------------------------------------------------------------------- */

SIXTRL_FN SIXTRL_STATIC NS(particle_real_t) NS(Particles_get_rvv_value)(
    SIXTRL_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const ii );

SIXTRL_FN SIXTRL_STATIC NS(particle_real_const_ptr_t)
NS(Particles_get_const_rvv)(
    SIXTRL_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT particles );

SIXTRL_FN SIXTRL_STATIC NS(particle_real_ptr_t) NS(Particles_get_rvv)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles );

SIXTRL_FN SIXTRL_STATIC void NS(Particles_set_rvv)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_real_const_ptr_t) SIXTRL_RESTRICT ptr_to_rvvs );

SIXTRL_FN SIXTRL_STATIC void NS(Particles_set_rvv_value)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const ii,
    NS(particle_real_t) const rvv_value );

SIXTRL_FN SIXTRL_STATIC void NS(Particles_assign_ptr_to_rvv)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_real_ptr_t) ptr_to_rvvs );

/* ------------------------------------------------------------------------- */

SIXTRL_FN SIXTRL_STATIC NS(particle_real_t) NS(Particles_get_chi_value)(
    SIXTRL_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const ii );

SIXTRL_FN SIXTRL_STATIC NS(particle_real_const_ptr_t)
NS(Particles_get_const_chi)(
    SIXTRL_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT particles );

SIXTRL_FN SIXTRL_STATIC NS(particle_real_ptr_t) NS(Particles_get_chi)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles );

SIXTRL_FN SIXTRL_STATIC void NS(Particles_set_chi)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_real_const_ptr_t) SIXTRL_RESTRICT ptr_to_chis );

SIXTRL_FN SIXTRL_STATIC void NS(Particles_set_chi_value)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const ii,
    NS(particle_real_t) const chi_value );

SIXTRL_FN SIXTRL_STATIC void NS(Particles_assign_ptr_to_chi)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_real_ptr_t) ptr_to_chis );

/* ------------------------------------------------------------------------- */

SIXTRL_FN SIXTRL_STATIC NS(particle_index_t) NS(Particles_get_particle_id_value)(
    SIXTRL_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const ii );

SIXTRL_FN SIXTRL_STATIC NS(particle_index_const_ptr_t)
NS(Particles_get_const_particle_id)(
    SIXTRL_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT particles );

SIXTRL_FN SIXTRL_STATIC NS(particle_index_ptr_t)
NS(Particles_get_particle_id)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles );

SIXTRL_FN SIXTRL_STATIC void NS(Particles_set_particle_id)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_index_const_ptr_t) SIXTRL_RESTRICT ptr_to_particle_ids );

SIXTRL_FN SIXTRL_STATIC void NS(Particles_set_particle_id_value)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const ii,
    NS(particle_index_t) const particle_id_value );

SIXTRL_FN SIXTRL_STATIC void NS(Particles_assign_ptr_to_particle_id)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_index_ptr_t) ptr_to_particle_ids );

/* ------------------------------------------------------------------------- */

SIXTRL_FN SIXTRL_STATIC NS(particle_index_t)
NS(Particles_get_at_element_id_value)(
    SIXTRL_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const ii );

SIXTRL_FN SIXTRL_STATIC NS(particle_index_const_ptr_t)
NS(Particles_get_const_at_element_id)(
    SIXTRL_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT particles );

SIXTRL_FN SIXTRL_STATIC NS(particle_index_ptr_t)
NS(Particles_get_at_element_id)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles );

SIXTRL_FN SIXTRL_STATIC void NS(Particles_set_at_element_id)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_index_const_ptr_t) SIXTRL_RESTRICT ptr_to_at_element_ids );

SIXTRL_FN SIXTRL_STATIC void NS(Particles_set_at_element_id_value)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const ii,
    NS(particle_index_t) const at_element_id_value );

SIXTRL_FN SIXTRL_STATIC void NS(Particles_assign_ptr_to_at_element_id)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_index_ptr_t) ptr_to_at_element_ids );

/* ------------------------------------------------------------------------- */

SIXTRL_FN SIXTRL_STATIC NS(particle_index_t)
NS(Particles_get_at_turn_value)(
    SIXTRL_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const ii );

SIXTRL_FN SIXTRL_STATIC NS(particle_index_const_ptr_t)
NS(Particles_get_const_at_turn)(
    SIXTRL_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT particles );

SIXTRL_FN SIXTRL_STATIC NS(particle_index_ptr_t)
NS(Particles_get_at_turn)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles );

SIXTRL_FN SIXTRL_STATIC void NS(Particles_set_at_turn)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_index_const_ptr_t) SIXTRL_RESTRICT ptr_to_at_turns );

SIXTRL_FN SIXTRL_STATIC void NS(Particles_set_at_turn_value)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const ii,
    NS(particle_index_t) const at_turn_value );

SIXTRL_FN SIXTRL_STATIC void NS(Particles_assign_ptr_to_at_turn)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_index_ptr_t) ptr_to_at_turns );

/* ------------------------------------------------------------------------- */

SIXTRL_FN SIXTRL_STATIC NS(particle_index_t) NS(Particles_get_state_value)(
    SIXTRL_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const ii );

SIXTRL_FN SIXTRL_STATIC NS(particle_index_const_ptr_t)
NS(Particles_get_const_state)(
    SIXTRL_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT particles );

SIXTRL_FN SIXTRL_STATIC NS(particle_index_ptr_t) NS(Particles_get_state)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles );

SIXTRL_FN SIXTRL_STATIC void NS(Particles_set_state)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_index_const_ptr_t) SIXTRL_RESTRICT ptr_to_states );

SIXTRL_FN SIXTRL_STATIC void NS(Particles_set_state_value)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const ii,
    NS(particle_index_t) const state_value );

SIXTRL_FN SIXTRL_STATIC void NS(Particles_assign_ptr_to_state)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_index_ptr_t) ptr_to_states );

#if !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

#if defined( _GPUCODE ) && !defined( __cplusplus )
    #if !defined( NS(PARTICLES_NUM_DATAPTRS) ) \
        #define   NS(PARTICLES_NUM_DATAPTRS) 20u
    #endif /* !defined( NS(PARTICLES_NUM_DATAPTRS) ) */
#endif /* defined( _GPUCODE ) && !defined( __cplusplus ) */

#if defined( __cplusplus )

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <type_traits>
    #include <iterator>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

namespace SIXTRL_NAMESPACE
{
    template< typename T >
    struct TParticles
    {
        using num_elements_t = NS(particle_num_elements_t);

        using real_t = typename std::iterator_traits< T* >::value_type;

        using real_pointer_t       = real_t*;
        using const_real_pointer_t = real_t const*;

        using index_t = typename std::iterator_traits<
            NS(particle_index_ptr_t) >::value_type;

        using index_pointer_t       = index_t*;
        using const_index_pointer_t = index_t const*;

        using type_id_t     = NS(object_type_id_t);
        using size_type     = NS(buffer_size_t);
        using buffer_t   	= ::NS(Buffer);

        SIXTRL_FN TParticles() = default;
        SIXTRL_FN TParticles( TParticles< T > const& other ) = default;
        SIXTRL_FN TParticles( TParticles< T >&& other ) = default;

        SIXTRL_FN TParticles< T >& operator=(
            TParticles< T > const& rhs ) = default;

        SIXTRL_FN TParticles< T >& operator=(
            TParticles< T >&& rhs ) = default;

        SIXTRL_FN ~TParticles() = default;

        SIXTRL_FN type_id_t getTypeId() const SIXTRL_NOEXCEPT
        {
            return NS(OBJECT_TYPE_PARTICLE);
        }

        /* ----------------------------------------------------------------- */

        SIXTRL_FN static bool CanAddToBuffer(
            buffer_t& SIXTRL_RESTRICT_REF buffer,
            size_type const num_particles,
            size_type* SIXTRL_RESTRICT ptr_requ_objects = nullptr,
            size_type* SIXTRL_RESTRICT ptr_requ_slots = nullptr,
            size_type* SIXTRL_RESTRICT ptr_requ_dataptrs = nullptr ) SIXTRL_NOEXCEPT
        {
            using _this_t = TParticles< T >;

            static_assert( std::is_trivial< _this_t >::value, "" );
            static_assert( std::is_standard_layout< _this_t >::value, "" );

            using size_t  = typename _this_t::size_type;

            size_t const real_size    = sizeof( real_t );
            size_t const index_size   = sizeof( index_t );
            size_t const num_dataptrs = NS(PARTICLES_NUM_DATAPTRS);

            size_t const sizes[] =
            {
                real_size,  real_size,  real_size,  real_size,
                real_size,  real_size,  real_size,  real_size,
                real_size,  real_size,  real_size,  real_size,
                real_size,  real_size,  real_size,  real_size,
                index_size, index_size, index_size, index_size
            };

            size_t const counts[] =
            {
                num_particles, num_particles, num_particles, num_particles,
                num_particles, num_particles, num_particles, num_particles,
                num_particles, num_particles, num_particles, num_particles,
                num_particles, num_particles, num_particles, num_particles,
                num_particles, num_particles, num_particles, num_particles
            };

            return ::NS(Buffer_can_add_object)( &buffer, sizeof( _this_t ),
                num_dataptrs, sizes, counts, ptr_requ_objects, ptr_requ_slots,
                    ptr_requ_dataptrs );
        }

        template< typename RetPtr = TParticles< T >* >
        SIXTRL_FN static RetPtr CreateNewOnBuffer(
            buffer_t& SIXTRL_RESTRICT_REF buffer, size_type const num_particles )
        {
            using _this_t = TParticles< T >;

            static_assert( std::is_trivial< _this_t >::value, "" );
            static_assert( std::is_standard_layout< _this_t >::value, "" );

            size_t const real_size    = sizeof( real_t );
            size_t const index_size   = sizeof( index_t );
            size_t const num_dataptrs = NS(PARTICLES_NUM_DATAPTRS);

            size_t const offsets[] =
            {
                offsetof( _this_t, q0 ),
                offsetof( _this_t, mass0 ),
                offsetof( _this_t, beta0 ),
                offsetof( _this_t, gamma0 ),
                offsetof( _this_t, p0c ),

                offsetof( _this_t, s ),
                offsetof( _this_t, x ),
                offsetof( _this_t, y ),
                offsetof( _this_t, px ),
                offsetof( _this_t, py ),
                offsetof( _this_t, zeta ),

                offsetof( _this_t, psigma ),
                offsetof( _this_t, delta ),
                offsetof( _this_t, rpp ),
                offsetof( _this_t, rvv ),
                offsetof( _this_t, chi ),

                offsetof( _this_t, particle_id ),
                offsetof( _this_t, at_element_id ),
                offsetof( _this_t, at_turn ),
                offsetof( _this_t, state )
            };

            size_t const sizes[] =
            {
                real_size,  real_size,  real_size,  real_size,  real_size,
                real_size,  real_size,  real_size,
                real_size,  real_size,  real_size,
                real_size,  real_size,  real_size,  real_size,  real_size,
                index_size, index_size, index_size, index_size
            };

            size_t const counts[] =
            {
                num_particles, num_particles, num_particles, num_particles, num_particles,
                num_particles, num_particles, num_particles,
                num_particles, num_particles, num_particles,
                num_particles, num_particles, num_particles, num_particles, num_particles,
                num_particles, num_particles, num_particles, num_particles
            };

            _this_t temp;
            temp.preset();
            temp.setNumParticles( num_particles );
            type_id_t const type_id = temp.getTypeId();

            return reinterpret_cast< RetPtr >( static_cast< uintptr_t >(
                ::NS(Object_get_begin_addr)(
                    ::NS(Buffer_add_object)( &buffer, &temp, sizeof( _this_t ),
                        type_id, num_dataptrs, offsets, sizes, counts ) ) ) );
        }

        template< typename RetPtr = TParticles< T >* >
        SIXTRL_FN RetPtr AddToBuffer(
            buffer_t& SIXTRL_RESTRICT_REF buffer,
            size_type const num_particles,
            real_pointer_t  q0_ptr            = nullptr,
            real_pointer_t  mass0_ptr         = nullptr,
            real_pointer_t  beta0_ptr         = nullptr,
            real_pointer_t  gamma0_ptr        = nullptr,
            real_pointer_t  p0c_ptr           = nullptr,
            real_pointer_t  s_ptr             = nullptr,
            real_pointer_t  x_ptr             = nullptr,
            real_pointer_t  y_ptr             = nullptr,
            real_pointer_t  px_ptr            = nullptr,
            real_pointer_t  py_ptr            = nullptr,
            real_pointer_t  zeta_ptr          = nullptr,
            real_pointer_t  psigma_ptr        = nullptr,
            real_pointer_t  delta_ptr         = nullptr,
            real_pointer_t  rpp_ptr           = nullptr,
            real_pointer_t  rvv_ptr           = nullptr,
            real_pointer_t  chi_ptr           = nullptr,
            index_pointer_t particle_id_ptr   = nullptr,
            index_pointer_t at_element_id_ptr = nullptr,
            index_pointer_t at_turn_ptr       = nullptr,
            index_pointer_t state_ptr         = nullptr )
        {
            using _this_t = TParticles< T >;

            static_assert( std::is_trivial< _this_t >::value, "" );
            static_assert( std::is_standard_layout< _this_t >::value, "" );

            size_t const real_size    = sizeof( real_t );
            size_t const index_size   = sizeof( index_t );
            size_t const num_dataptrs = NS(PARTICLES_NUM_DATAPTRS);

            size_t const offsets[] =
            {
                offsetof( _this_t, q0 ),
                offsetof( _this_t, mass0 ),
                offsetof( _this_t, beta0 ),
                offsetof( _this_t, gamma0 ),
                offsetof( _this_t, p0c ),

                offsetof( _this_t, s ),
                offsetof( _this_t, x ),
                offsetof( _this_t, y ),
                offsetof( _this_t, px ),
                offsetof( _this_t, py ),
                offsetof( _this_t, zeta ),

                offsetof( _this_t, psigma ),
                offsetof( _this_t, delta ),
                offsetof( _this_t, rpp ),
                offsetof( _this_t, rvv ),
                offsetof( _this_t, chi ),

                offsetof( _this_t, particle_id ),
                offsetof( _this_t, at_element_id ),
                offsetof( _this_t, at_turn ),
                offsetof( _this_t, state )
            };

            size_t const sizes[] =
            {
                real_size,  real_size,  real_size,  real_size,  real_size,
                real_size,  real_size,  real_size,
                real_size,  real_size,  real_size,
                real_size,  real_size,  real_size,  real_size,  real_size,
                index_size, index_size, index_size, index_size
            };

            size_t const counts[] =
            {
                num_particles, num_particles, num_particles, num_particles, num_particles,
                num_particles, num_particles, num_particles,
                num_particles, num_particles, num_particles,
                num_particles, num_particles, num_particles, num_particles, num_particles,
                num_particles, num_particles, num_particles, num_particles
            };

            _this_t temp;
            temp.setNumParticles( num_particles );
            temp.assignQ0Ptr( q0_ptr );
            temp.assignMass0Ptr( mass0_ptr );
            temp.assignBeta0Ptr( beta0_ptr );
            temp.assignGamma0Ptr( gamma0_ptr );
            temp.assignP0cPtr( p0c_ptr );
            temp.assignSPtr( s_ptr );
            temp.assignXPtr( x_ptr );
            temp.assignYPtr( y_ptr );
            temp.assignPxPtr( px_ptr );
            temp.assignPyPtr( py_ptr );
            temp.assignZetaPtr( zeta_ptr );
            temp.assignPSigmaPtr( psigma_ptr );
            temp.assignDeltaPtr( delta_ptr );
            temp.assignRppPtr( rpp_ptr );
            temp.assignRvvPtr( rvv_ptr );
            temp.assignChiPtr( chi_ptr );
            temp.assignParticleIdPtr( particle_id_ptr );
            temp.assignAtElementIdPtr( at_element_id_ptr );
            temp.assignAtTurnPtr( at_turn_ptr );
            temp.assignStatePtr( state_ptr );

            type_id_t const type_id = temp.getTypeId();

            return reinterpret_cast< RetPtr >( static_cast< uintptr_t >(
                ::NS(Object_get_begin_addr)(
                    ::NS(Buffer_add_object)( &buffer, &temp, sizeof( _this_t ),
                        type_id, num_dataptrs, offsets, sizes, counts ) ) ) );
        }

        /* ----------------------------------------------------------------- */

        num_elements_t getNumParticles() const SIXTRL_NOEXCEPT
        {
            return this->num_particles;
        }

        void setNumParticles( num_elements_t const num_particles ) SIXTRL_NOEXCEPT
        {
            this->num_particles = num_particles;
            return;
        }

        void preset() SIXTRL_NOEXCEPT
        {
            this->setNumParticles( num_elements_t{ 0 } );

            this->assignQ0Ptr( nullptr );
            this->assignMass0Ptr( nullptr );
            this->assignBeta0Ptr( nullptr );
            this->assignGamma0Ptr( nullptr );
            this->assignP0cPtr( nullptr );

            this->assignSPtr( nullptr );
            this->assignXPtr( nullptr );
            this->assignYPtr( nullptr );
            this->assignPxPtr( nullptr );
            this->assignPyPtr( nullptr );
            this->assignZetaPtr( nullptr );

            this->assignPSigmaPtr( nullptr );
            this->assignDeltaPtr( nullptr );
            this->assignRppPtr( nullptr );
            this->assignRvvPtr( nullptr );
            this->assignChiPtr( nullptr );

            this->assignParticleIdPtr( nullptr );
            this->assignAtElementIdPtr( nullptr );
            this->assignAtTurnPtr( nullptr );
            this->assignStatePtr( nullptr );
        }

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        const_real_pointer_t getQ0() const SIXTRL_NOEXCEPT
        {
            return this->q0;
        }

        real_pointer_t getQ0() SIXTRL_NOEXCEPT
        {
            return this->q0;
        }

        real_t getQ0Value( index_t const index ) const SIXTRL_NOEXCEPT
        {
            return this->realValueGetterFn( this->q0, index );
        }

        template< typename Iter >
        void setQ0( Iter begin, Iter end ) SIXTRL_NOEXCEPT
        {
            using elem_t = typename TParticles< T >::num_elements_t;
            elem_t const in_num_particles = std::distance( begin, end );

            if( in_num_particles <= this->getNumParticles() )
            {
                std::copy( begin, end, this->getQ0() );
            }

            return;
        }

        void setQ0Value( index_t const index, real_t const value ) SIXTRL_NOEXCEPT
        {
            this->realValueSetterFn( this->q0, index, value );
            return;
        }

        void assignQ0Ptr( real_pointer_t ptr ) SIXTRL_NOEXCEPT
        {
            this->q0 = ptr;
            return;
        }

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        const_real_pointer_t getMass0() const SIXTRL_NOEXCEPT
        {
            return this->mass0;
        }

        real_pointer_t getMass0() SIXTRL_NOEXCEPT
        {
            return this->mass0;
        }

        real_t getMass0Value( index_t const index ) const SIXTRL_NOEXCEPT
        {
            return this->realValueGetterFn( this->mass0, index );
        }

        template< typename Iter >
        void setMass0( Iter begin, Iter end ) SIXTRL_NOEXCEPT
        {
            using elem_t = typename TParticles< T >::num_elements_t;
            elem_t const in_num_particles = std::distance( begin, end );

            if( in_num_particles <= this->getNumParticles() )
            {
                std::copy( begin, end, this->getMass0() );
            }

            return;
        }

        void setMass0Value( index_t const index, real_t const value ) SIXTRL_NOEXCEPT
        {
            this->realValueSetterFn( this->mass0, index, value );
            return;
        }

        void assignMass0Ptr( real_pointer_t ptr ) SIXTRL_NOEXCEPT
        {
            this->mass0 = ptr;
            return;
        }

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        const_real_pointer_t getBeta0() const SIXTRL_NOEXCEPT
        {
            return this->beta0;
        }

        real_pointer_t getBeta0() SIXTRL_NOEXCEPT
        {
            return this->beta0;
        }

        real_t getBeta0Value( index_t const index ) const SIXTRL_NOEXCEPT
        {
            return this->realValueGetterFn( this->beta0, index );
        }

        template< typename Iter >
        void setBeta0( Iter begin, Iter end ) SIXTRL_NOEXCEPT
        {
            using elem_t = typename TParticles< T >::num_elements_t;
            elem_t const in_num_particles = std::distance( begin, end );

            if( in_num_particles <= this->getNumParticles() )
            {
                std::copy( begin, end, this->getBeta0() );
            }

            return;
        }

        void setBeta0Value( index_t const index, real_t const value ) SIXTRL_NOEXCEPT
        {
            this->realValueSetterFn( this->beta0, index, value );
            return;
        }

        void assignBeta0Ptr( real_pointer_t ptr ) SIXTRL_NOEXCEPT
        {
            this->beta0 = ptr;
            return;
        }

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        const_real_pointer_t getGamma0() const SIXTRL_NOEXCEPT
        {
            return this->gamma0;
        }

        real_pointer_t getGamma0() SIXTRL_NOEXCEPT
        {
            return this->gamma0;
        }

        real_t getGamma0Value( index_t const index ) const SIXTRL_NOEXCEPT
        {
            return this->realValueGetterFn( this->gamma0, index );
        }

        template< typename Iter >
        void setGamma0( Iter begin, Iter end ) SIXTRL_NOEXCEPT
        {
            using elem_t = typename TParticles< T >::num_elements_t;
            elem_t const in_num_particles = std::distance( begin, end );

            if( in_num_particles <= this->getNumParticles() )
            {
                std::copy( begin, end, this->getGamma0() );
            }

            return;
        }

        void setGamma0Value( index_t const index, real_t const value ) SIXTRL_NOEXCEPT
        {
            this->realValueSetterFn( this->gamma0, index, value );
            return;
        }

        void assignGamma0Ptr( real_pointer_t ptr ) SIXTRL_NOEXCEPT
        {
            this->gamma0 = ptr;
            return;
        }

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        const_real_pointer_t getP0c() const SIXTRL_NOEXCEPT
        {
            return this->p0c;
        }

        real_pointer_t getP0c() SIXTRL_NOEXCEPT
        {
            return this->p0c;
        }

        real_t getP0cValue( index_t const index ) const SIXTRL_NOEXCEPT
        {
            return this->realValueGetterFn( this->p0c, index );
        }

        template< typename Iter >
        void setP0c( Iter begin, Iter end ) SIXTRL_NOEXCEPT
        {
            using elem_t = typename TParticles< T >::num_elements_t;
            elem_t const in_num_particles = std::distance( begin, end );

            if( in_num_particles <= this->getNumParticles() )
            {
                std::copy( begin, end, this->getP0c() );
            }

            return;
        }

        void setP0cValue( index_t const index, real_t const value ) SIXTRL_NOEXCEPT
        {
            this->realValueSetterFn( this->p0c, index, value );
            return;
        }

        void assignP0cPtr( real_pointer_t ptr ) SIXTRL_NOEXCEPT
        {
            this->p0c = ptr;
            return;
        }

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        const_real_pointer_t getS() const SIXTRL_NOEXCEPT
        {
            return this->s;
        }

        real_pointer_t getS() SIXTRL_NOEXCEPT
        {
            return this->s;
        }

        real_t getSValue( index_t const index ) const SIXTRL_NOEXCEPT
        {
            return this->realValueGetterFn( this->s, index );
        }

        template< typename Iter >
        void setS( Iter begin, Iter end ) SIXTRL_NOEXCEPT
        {
            using elem_t = typename TParticles< T >::num_elements_t;
            elem_t const in_num_particles = std::distance( begin, end );

            if( in_num_particles <= this->getNumParticles() )
            {
                std::copy( begin, end, this->getS() );
            }

            return;
        }

        void setSValue( index_t const index, real_t const value ) SIXTRL_NOEXCEPT
        {
            this->realValueSetterFn( this->s, index, value );
            return;
        }

        void assignSPtr( real_pointer_t ptr ) SIXTRL_NOEXCEPT
        {
            this->s = ptr;
            return;
        }

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        const_real_pointer_t getX() const SIXTRL_NOEXCEPT
        {
            return this->x;
        }

        real_pointer_t getX() SIXTRL_NOEXCEPT
        {
            return this->x;
        }

        real_t getXValue( index_t const index ) const SIXTRL_NOEXCEPT
        {
            return this->realValueGetterFn( this->x, index );
        }

        template< typename Iter >
        void setX( Iter begin, Iter end ) SIXTRL_NOEXCEPT
        {
            using elem_t = typename TParticles< T >::num_elements_t;
            elem_t const in_num_particles = std::distance( begin, end );

            if( in_num_particles <= this->getNumParticles() )
            {
                std::copy( begin, end, this->getX() );
            }

            return;
        }

        void setXValue( index_t const index, real_t const value ) SIXTRL_NOEXCEPT
        {
            this->realValueSetterFn( this->x, index, value );
            return;
        }

        void assignXPtr( real_pointer_t ptr ) SIXTRL_NOEXCEPT
        {
            this->x = ptr;
            return;
        }

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        const_real_pointer_t getY() const SIXTRL_NOEXCEPT
        {
            return this->y;
        }

        real_pointer_t getY() SIXTRL_NOEXCEPT
        {
            return this->y;
        }

        real_t getYValue( index_t const index ) const SIXTRL_NOEXCEPT
        {
            return this->realValueGetterFn( this->y, index );
        }

        template< typename Iter >
        void setY( Iter begin, Iter end ) SIXTRL_NOEXCEPT
        {
            using elem_t = typename TParticles< T >::num_elements_t;
            elem_t const in_num_particles = std::distance( begin, end );

            if( in_num_particles <= this->getNumParticles() )
            {
                std::copy( begin, end, this->getY() );
            }

            return;
        }

        void setYValue( index_t const index, real_t const value ) SIXTRL_NOEXCEPT
        {
            this->realValueSetterFn( this->y, index, value );
            return;
        }

        void assignYPtr( real_pointer_t ptr ) SIXTRL_NOEXCEPT
        {
            this->y = ptr;
            return;
        }

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        const_real_pointer_t getPx() const SIXTRL_NOEXCEPT
        {
            return this->px;
        }

        real_pointer_t getPx() SIXTRL_NOEXCEPT
        {
            return this->px;
        }

        real_t getPxValue( index_t const index ) const SIXTRL_NOEXCEPT
        {
            return this->realValueGetterFn( this->px, index );
        }

        template< typename Iter >
        void setPx( Iter begin, Iter end ) SIXTRL_NOEXCEPT
        {
            using elem_t = typename TParticles< T >::num_elements_t;
            elem_t const in_num_particles = std::distance( begin, end );

            if( in_num_particles <= this->getNumParticles() )
            {
                std::copy( begin, end, this->getPx() );
            }

            return;
        }

        void setPxValue( index_t const index, real_t const value ) SIXTRL_NOEXCEPT
        {
            this->realValueSetterFn( this->px, index, value );
            return;
        }

        void assignPxPtr( real_pointer_t ptr ) SIXTRL_NOEXCEPT
        {
            this->px = ptr;
            return;
        }

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        const_real_pointer_t getPy() const SIXTRL_NOEXCEPT
        {
            return this->py;
        }

        real_pointer_t getPy() SIXTRL_NOEXCEPT
        {
            return this->py;
        }

        real_t getPyValue( index_t const index ) const SIXTRL_NOEXCEPT
        {
            return this->realValueGetterFn( this->py, index );
        }

        template< typename Iter >
        void setPy( Iter begin, Iter end ) SIXTRL_NOEXCEPT
        {
            using elem_t = typename TParticles< T >::num_elements_t;
            elem_t const in_num_particles = std::distance( begin, end );

            if( in_num_particles <= this->getNumParticles() )
            {
                std::copy( begin, end, this->getPy() );
            }

            return;
        }

        void setPyValue( index_t const index, real_t const value ) SIXTRL_NOEXCEPT
        {
            this->realValueSetterFn( this->py, index, value );
            return;
        }

        void assignPyPtr( real_pointer_t ptr ) SIXTRL_NOEXCEPT
        {
            this->py = ptr;
            return;
        }

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        const_real_pointer_t getZeta() const SIXTRL_NOEXCEPT
        {
            return this->zeta;
        }

        real_pointer_t getZeta() SIXTRL_NOEXCEPT
        {
            return this->zeta;
        }

        real_t getZetaValue( index_t const index ) const SIXTRL_NOEXCEPT
        {
            return this->realValueGetterFn( this->zeta, index );
        }

        template< typename Iter >
        void setZeta( Iter begin, Iter end ) SIXTRL_NOEXCEPT
        {
            using elem_t = typename TParticles< T >::num_elements_t;
            elem_t const in_num_particles = std::distance( begin, end );

            if( in_num_particles <= this->getNumParticles() )
            {
                std::copy( begin, end, this->getZeta() );
            }

            return;
        }

        void setZetaValue( index_t const index, real_t const value ) SIXTRL_NOEXCEPT
        {
            this->realValueSetterFn( this->zeta, index, value );
            return;
        }

        void assignZetaPtr( real_pointer_t ptr ) SIXTRL_NOEXCEPT
        {
            this->zeta = ptr;
            return;
        }

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        const_real_pointer_t getPSigma() const SIXTRL_NOEXCEPT
        {
            return this->psigma;
        }

        real_pointer_t getPSigma() SIXTRL_NOEXCEPT
        {
            return this->psigma;
        }

        real_t getPSigmaValue( index_t const index ) const SIXTRL_NOEXCEPT
        {
            return this->realValueGetterFn( this->psigma, index );
        }

        template< typename Iter >
        void setPSigma( Iter begin, Iter end ) SIXTRL_NOEXCEPT
        {
            using elem_t = typename TParticles< T >::num_elements_t;
            elem_t const in_num_particles = std::distance( begin, end );

            if( in_num_particles <= this->getNumParticles() )
            {
                std::copy( begin, end, this->getPSigma() );
            }

            return;
        }

        void setPSigmaValue( index_t const index, real_t const value ) SIXTRL_NOEXCEPT
        {
            this->realValueSetterFn( this->psigma, index, value );
            return;
        }

        void assignPSigmaPtr( real_pointer_t ptr ) SIXTRL_NOEXCEPT
        {
            this->psigma = ptr;
            return;
        }

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        const_real_pointer_t getDelta() const SIXTRL_NOEXCEPT
        {
            return this->delta;
        }

        real_pointer_t getDelta() SIXTRL_NOEXCEPT
        {
            return this->delta;
        }

        real_t getDeltaValue( index_t const index ) const SIXTRL_NOEXCEPT
        {
            return this->realValueGetterFn( this->delta, index );
        }

        template< typename Iter >
        void setDelta( Iter begin, Iter end ) SIXTRL_NOEXCEPT
        {
            using elem_t = typename TParticles< T >::num_elements_t;
            elem_t const in_num_particles = std::distance( begin, end );

            if( in_num_particles <= this->getNumParticles() )
            {
                std::copy( begin, end, this->getDelta() );
            }

            return;
        }

        void setDeltaValue( index_t const index, real_t const value ) SIXTRL_NOEXCEPT
        {
            this->realValueSetterFn( this->delta, index, value );
            return;
        }

        void assignDeltaPtr( real_pointer_t ptr ) SIXTRL_NOEXCEPT
        {
            this->delta = ptr;
            return;
        }

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        const_real_pointer_t getRpp() const SIXTRL_NOEXCEPT
        {
            return this->rpp;
        }

        real_pointer_t getRpp() SIXTRL_NOEXCEPT
        {
            return this->rpp;
        }

        real_t getRppValue( index_t const index ) const SIXTRL_NOEXCEPT
        {
            return this->realValueGetterFn( this->rpp, index );
        }

        template< typename Iter >
        void setRpp( Iter begin, Iter end ) SIXTRL_NOEXCEPT
        {
            using elem_t = typename TParticles< T >::num_elements_t;
            elem_t const in_num_particles = std::distance( begin, end );

            if( in_num_particles <= this->getNumParticles() )
            {
                std::copy( begin, end, this->getRpp() );
            }

            return;
        }

        void setRppValue( index_t const index, real_t const value ) SIXTRL_NOEXCEPT
        {
            this->realValueSetterFn( this->rpp, index, value );
            return;
        }

        void assignRppPtr( real_pointer_t ptr ) SIXTRL_NOEXCEPT
        {
            this->rpp = ptr;
            return;
        }

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        const_real_pointer_t getRvv() const SIXTRL_NOEXCEPT
        {
            return this->rvv;
        }

        real_pointer_t getRvv() SIXTRL_NOEXCEPT
        {
            return this->rvv;
        }

        real_t getRvvValue( index_t const index ) const SIXTRL_NOEXCEPT
        {
            return this->realValueGetterFn( this->rvv, index );
        }

        template< typename Iter >
        void setRvv( Iter begin, Iter end ) SIXTRL_NOEXCEPT
        {
            using elem_t = typename TParticles< T >::num_elements_t;
            elem_t const in_num_particles = std::distance( begin, end );

            if( in_num_particles <= this->getNumParticles() )
            {
                std::copy( begin, end, this->getRvv() );
            }

            return;
        }

        void setRvvValue( index_t const index, real_t const value ) SIXTRL_NOEXCEPT
        {
            this->realValueSetterFn( this->rvv, index, value );
            return;
        }

        void assignRvvPtr( real_pointer_t ptr ) SIXTRL_NOEXCEPT
        {
            this->rvv = ptr;
            return;
        }

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        const_real_pointer_t getChi() const SIXTRL_NOEXCEPT
        {
            return this->chi;
        }

        real_pointer_t getChi() SIXTRL_NOEXCEPT
        {
            return this->chi;
        }

        real_t getChiValue( index_t const index ) const SIXTRL_NOEXCEPT
        {
            return this->realValueGetterFn( this->chi, index );
        }

        template< typename Iter >
        void setChi( Iter begin, Iter end ) SIXTRL_NOEXCEPT
        {
            using elem_t = typename TParticles< T >::num_elements_t;
            elem_t const in_num_particles = std::distance( begin, end );

            if( in_num_particles <= this->getNumParticles() )
            {
                std::copy( begin, end, this->getChi() );
            }

            return;
        }

        void setChiValue( index_t const index, real_t const value ) SIXTRL_NOEXCEPT
        {
            this->realValueSetterFn( this->chi, index, value );
            return;
        }

        void assignChiPtr( real_pointer_t ptr ) SIXTRL_NOEXCEPT
        {
            this->chi = ptr;
            return;
        }

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        const_index_pointer_t getParticleId() const SIXTRL_NOEXCEPT
        {
            return this->particle_id;
        }

        index_pointer_t getParticleId() SIXTRL_NOEXCEPT
        {
            return this->particle_id;
        }

        index_t getParticleIdValue( index_t const index ) const SIXTRL_NOEXCEPT
        {
            return this->indexValueGetterFn( this->particle_id, index );
        }

        template< typename Iter >
        void setParticleId( Iter begin, Iter end ) SIXTRL_NOEXCEPT
        {
            using elem_t = typename TParticles< T >::num_elements_t;
            elem_t const in_num_particles = std::distance( begin, end );

            if( in_num_particles <= this->getNumParticles() )
            {
                std::copy( begin, end, this->getParticleId() );
            }

            return;
        }

        void setParticleIdValue( index_t const index, index_t const value ) SIXTRL_NOEXCEPT
        {
            this->indexValueSetterFn( this->particle_id, index, value );
            return;
        }

        void assignParticleIdPtr( index_pointer_t ptr ) SIXTRL_NOEXCEPT
        {
            this->particle_id = ptr;
            return;
        }

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        const_index_pointer_t getAtElementId() const SIXTRL_NOEXCEPT
        {
            return this->at_element_id;
        }

        index_pointer_t getAtElementId() SIXTRL_NOEXCEPT
        {
            return this->at_element_id;
        }

        index_t getAtElementIdValue( index_t const index ) const SIXTRL_NOEXCEPT
        {
            return this->indexValueGetterFn( this->at_element_id, index );
        }

        template< typename Iter >
        void setAtElementId( Iter begin, Iter end ) SIXTRL_NOEXCEPT
        {
            using elem_t = typename TParticles< T >::num_elements_t;
            elem_t const in_num_particles = std::distance( begin, end );

            if( in_num_particles <= this->getNumParticles() )
            {
                std::copy( begin, end, this->getAtElementId() );
            }

            return;
        }

        void setAtElementIdValue( index_t const index, index_t const value ) SIXTRL_NOEXCEPT
        {
            this->indexValueSetterFn( this->at_element_id, index, value );
            return;
        }

        void assignAtElementIdPtr( index_pointer_t ptr ) SIXTRL_NOEXCEPT
        {
            this->at_element_id = ptr;
            return;
        }

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        const_index_pointer_t getAtTurn() const SIXTRL_NOEXCEPT
        {
            return this->at_turn;
        }

        index_pointer_t getAtTurn() SIXTRL_NOEXCEPT
        {
            return this->at_turn;
        }

        index_t getAtTurnValue( index_t const index ) const SIXTRL_NOEXCEPT
        {
            return this->indexValueGetterFn( this->at_turn, index );
        }

        template< typename Iter >
        void setAtTurn( Iter begin, Iter end ) SIXTRL_NOEXCEPT
        {
            using elem_t = typename TParticles< T >::num_elements_t;
            elem_t const in_num_particles = std::distance( begin, end );

            if( in_num_particles <= this->getNumParticles() )
            {
                std::copy( begin, end, this->getAtTurn() );
            }

            return;
        }

        void setAtTurnValue( index_t const index, index_t const value ) SIXTRL_NOEXCEPT
        {
            this->indexValueSetterFn( this->at_turn, index, value );
            return;
        }

        void assignAtTurnPtr( index_pointer_t ptr ) SIXTRL_NOEXCEPT
        {
            this->at_turn = ptr;
            return;
        }

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        const_index_pointer_t getState() const SIXTRL_NOEXCEPT
        {
            return this->state;
        }

        index_pointer_t getState() SIXTRL_NOEXCEPT
        {
            return this->state;
        }

        index_t getStateValue( index_t const index ) const SIXTRL_NOEXCEPT
        {
            return this->indexValueGetterFn( this->state, index );
        }

        template< typename Iter >
        void setState( Iter begin, Iter end ) SIXTRL_NOEXCEPT
        {
            using elem_t = typename TParticles< T >::num_elements_t;
            elem_t const in_num_particles = std::distance( begin, end );

            if( in_num_particles <= this->getNumParticles() )
            {
                std::copy( begin, end, this->getState() );
            }

            return;
        }

        void setStateValue( index_t const index, index_t const value ) SIXTRL_NOEXCEPT
        {
            this->indexValueSetterFn( this->state, index, value );
            return;
        }

        void assignStatePtr( index_pointer_t ptr ) SIXTRL_NOEXCEPT
        {
            this->state = ptr;
            return;
        }

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        real_t realValueGetterFn(
            const_real_pointer_t SIXTRL_RESTRICT ptr,
            index_t const index ) SIXTRL_NOEXCEPT
        {
            SIXTRL_ASSERT( ptr != nullptr );
            SIXTRL_ASSERT( index < this->num_particles );
            return ptr[ index ];
        }

        index_t indexValueGetterFn(
            const_index_pointer_t SIXTRL_RESTRICT ptr,
            index_t const index ) SIXTRL_NOEXCEPT
        {
            SIXTRL_ASSERT( ptr != nullptr );
            SIXTRL_ASSERT( index < this->num_particles );
            return ptr[ index ];
        }

        template< typename V, typename Ptr >
        void valueSetterFn( Ptr* SIXTRL_RESTRICT ptr,
                            index_t const index,
                            V const value ) SIXTRL_NOEXCEPT
        {
            SIXTRL_ASSERT( ptr != nullptr );
            SIXTRL_ASSERT( index < this->num_particles );
            ptr[ index ] = value;
            return;
        }

        /* ----------------------------------------------------------------- */

        num_elements_t  num_of_particles SIXTRL_ALIGN( 8 );
        real_pointer_t  q0               SIXTRL_ALIGN( 8 );
        real_pointer_t  mass0            SIXTRL_ALIGN( 8 );
        real_pointer_t  beta0            SIXTRL_ALIGN( 8 );
        real_pointer_t  gamma0           SIXTRL_ALIGN( 8 );
        real_pointer_t  p0c              SIXTRL_ALIGN( 8 );
        real_pointer_t  s                SIXTRL_ALIGN( 8 );
        real_pointer_t  x                SIXTRL_ALIGN( 8 );
        real_pointer_t  y                SIXTRL_ALIGN( 8 );
        real_pointer_t  px               SIXTRL_ALIGN( 8 );
        real_pointer_t  py               SIXTRL_ALIGN( 8 );
        real_pointer_t  zeta             SIXTRL_ALIGN( 8 );
        real_pointer_t  psigma           SIXTRL_ALIGN( 8 );
        real_pointer_t  delta            SIXTRL_ALIGN( 8 );
        real_pointer_t  rpp              SIXTRL_ALIGN( 8 );
        real_pointer_t  rvv              SIXTRL_ALIGN( 8 );
        real_pointer_t  chi              SIXTRL_ALIGN( 8 );
        index_pointer_t particle_id      SIXTRL_ALIGN( 8 );
        index_pointer_t at_element_id    SIXTRL_ALIGN( 8 );
        index_pointer_t at_turn          SIXTRL_ALIGN( 8 );
        index_pointer_t state            SIXTRL_ALIGN( 8 );
    };


    template<> struct TParticles< NS(particle_real_t) > :
        public ::NS(Particles)
    {
        using num_elements_t = NS(particle_num_elements_t);

        using real_t = typename std::iterator_traits<
            NS(particle_real_ptr_t) >::value_type;

        using real_pointer_t        = real_t*;
        using const_real_pointer_t  = real_t const*;

        using index_t = typename std::iterator_traits<
            NS(particle_index_ptr_t) >::value_type;

        using index_pointer_t       = index_t*;
        using const_index_pointer_t = index_t const*;

        using type_id_t             = NS(object_type_id_t);
        using size_type             = NS(buffer_size_t);
        using buffer_t   	        = ::NS(Buffer);
        using c_api_t               = ::NS(Particles);

        SIXTRL_FN TParticles() = default;

        SIXTRL_FN TParticles(
            TParticles< NS(particle_real_t) > const& other ) = default;

        SIXTRL_FN TParticles(
            TParticles< NS(particle_real_t) >&& other ) = default;

        SIXTRL_FN TParticles< NS(particle_real_t) >& operator=(
            TParticles< NS(particle_real_t) > const& rhs ) = default;

        SIXTRL_FN TParticles< NS(particle_real_t) >& operator=(
            TParticles< NS(particle_real_t) >&& rhs ) = default;

        SIXTRL_FN ~TParticles() = default;

        SIXTRL_FN type_id_t getTypeId() const SIXTRL_NOEXCEPT
        {
            return NS(OBJECT_TYPE_PARTICLE);
        }

        /* ----------------------------------------------------------------- */

        SIXTRL_FN static bool CanAddToBuffer(
            buffer_t& SIXTRL_RESTRICT_REF buffer,
            size_type const num_particles,
            size_type* SIXTRL_RESTRICT ptr_requ_objects = nullptr,
            size_type* SIXTRL_RESTRICT ptr_requ_slots = nullptr,
            size_type* SIXTRL_RESTRICT ptr_requ_dataptrs = nullptr ) SIXTRL_NOEXCEPT
        {
            return ::NS(Particles_can_be_added)( &buffer, num_particles,
                    ptr_requ_objects, ptr_requ_slots, ptr_requ_dataptrs );
        }

        template< typename RetPtr = TParticles< SIXTRL_REAL_T >* >
        SIXTRL_FN static RetPtr CreateNewOnBuffer(
            buffer_t& SIXTRL_RESTRICT_REF buffer, size_type const num_particles )
        {
            return reinterpret_cast< RetPtr >(
                ::NS(Particles_new)( &buffer, num_particles ) );
        }

        template< typename RetPtr = TParticles< SIXTRL_REAL_T >* >
        SIXTRL_FN RetPtr AddToBuffer(
            buffer_t& SIXTRL_RESTRICT_REF buffer,
            size_type const num_particles,
            real_pointer_t  q0_ptr            = nullptr,
            real_pointer_t  mass0_ptr         = nullptr,
            real_pointer_t  beta0_ptr         = nullptr,
            real_pointer_t  gamma0_ptr        = nullptr,
            real_pointer_t  p0c_ptr           = nullptr,
            real_pointer_t  s_ptr             = nullptr,
            real_pointer_t  x_ptr             = nullptr,
            real_pointer_t  y_ptr             = nullptr,
            real_pointer_t  px_ptr            = nullptr,
            real_pointer_t  py_ptr            = nullptr,
            real_pointer_t  zeta_ptr          = nullptr,
            real_pointer_t  psigma_ptr        = nullptr,
            real_pointer_t  delta_ptr         = nullptr,
            real_pointer_t  rpp_ptr           = nullptr,
            real_pointer_t  rvv_ptr           = nullptr,
            real_pointer_t  chi_ptr           = nullptr,
            index_pointer_t particle_id_ptr   = nullptr,
            index_pointer_t at_element_id_ptr = nullptr,
            index_pointer_t at_turn_ptr       = nullptr,
            index_pointer_t state_ptr         = nullptr )
        {
            return reinterpret_cast< RetPtr >( ::NS(Particles_add)(
                &buffer, num_particles,
                q0_ptr, mass0_ptr, beta0_ptr, gamma0_ptr, p0c_ptr,
                s_ptr, x_ptr, y_ptr, px_ptr, py_ptr, zeta_ptr,
                psigma_ptr, delta_ptr, rpp_ptr, rvv_ptr, chi_ptr,
                particle_id_ptr, at_element_id_ptr, at_turn_ptr, state_ptr ) );
        }

        /* ----------------------------------------------------------------- */

        c_api_t const* getCApiPtr() const SIXTRL_NOEXCEPT
        {
            using ptr_t = typename TParticles< SIXTRL_REAL_T >::c_api_t const*;
            return static_cast< ptr_t >( this );
        }

        c_api_t* getCApiPtr() SIXTRL_NOEXCEPT
        {
            using _this_t = TParticles< SIXTRL_REAL_T >;
            using   ptr_t = typename _this_t::c_api_t*;

            return const_cast< ptr_t >( static_cast< _this_t const& >(
                *this ).getCApiPtr() );
        }

        /* ----------------------------------------------------------------- */

        num_elements_t getNumParticles() const SIXTRL_NOEXCEPT
        {
            return ::NS(Particles_get_num_of_particles)( this->getCApiPtr() );
        }

        void setNumParticles( size_type const num_particles ) SIXTRL_NOEXCEPT
        {
            using _this_t    = TParticles< SIXTRL_REAL_T >;
            using num_elem_t = typename _this_t::num_elements_t;

            ::NS(Particles_set_num_of_particles)( this->getCApiPtr(),
                    static_cast< num_elem_t >( num_particles ) );
            return;
        }

        void preset() SIXTRL_NOEXCEPT
        {
            ::NS(Particles_preset)( this->getCApiPtr() );
        }

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        const_real_pointer_t getQ0() const SIXTRL_NOEXCEPT
        {
            return ::NS(Particles_get_const_q0)( this->getCApiPtr() );
        }

        real_pointer_t getQ0() SIXTRL_NOEXCEPT
        {
            return ::NS(Particles_get_q0)( this->getCApiPtr() );
        }

        real_t getQ0Value( index_t const index ) const SIXTRL_NOEXCEPT
        {
            return ::NS(Particles_get_q0_value)( this->getCApiPtr(), index );
        }

        template< typename Iter >
        void setQ0( Iter begin, Iter end ) SIXTRL_NOEXCEPT
        {
            using elem_t = typename TParticles< SIXTRL_REAL_T >::num_elements_t;
            elem_t const in_num_particles = std::distance( begin, end );

            if( in_num_particles <= this->getNumParticles() )
            {
                std::copy( begin, end, this->getQ0() );
            }

            return;
        }

        void setQ0Value( index_t const index, real_t const value ) SIXTRL_NOEXCEPT
        {
            ::NS(Particles_set_q0_value)( this->getCApiPtr(), index, value );
            return;
        }

        void assignQ0Ptr( real_pointer_t ptr ) SIXTRL_NOEXCEPT
        {
            ::NS(Particles_assign_ptr_to_q0)( this->getCApiPtr(), ptr );
            return;
        }

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        const_real_pointer_t getMass0() const SIXTRL_NOEXCEPT
        {
            return ::NS(Particles_get_const_mass0)( this->getCApiPtr() );
        }

        real_pointer_t getMass0() SIXTRL_NOEXCEPT
        {
            return ::NS(Particles_get_mass0)( this->getCApiPtr() );
        }

        real_t getMass0Value( index_t const index ) const SIXTRL_NOEXCEPT
        {
            return ::NS(Particles_get_mass0_value)( this->getCApiPtr(), index );
        }

        template< typename Iter >
        void setMass0( Iter begin, Iter end ) SIXTRL_NOEXCEPT
        {
            using elem_t = typename TParticles< SIXTRL_REAL_T >::num_elements_t;
            elem_t const in_num_particles = std::distance( begin, end );

            if( in_num_particles <= this->getNumParticles() )
            {
                std::copy( begin, end, this->getMass0() );
            }

            return;
        }

        void setMass0Value( index_t const index, real_t const value ) SIXTRL_NOEXCEPT
        {
            ::NS(Particles_set_mass0_value)( this->getCApiPtr(), index, value );
            return;
        }

        void assignMass0Ptr( real_pointer_t ptr ) SIXTRL_NOEXCEPT
        {
            ::NS(Particles_assign_ptr_to_mass0)( this->getCApiPtr(), ptr );
            return;
        }

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        const_real_pointer_t getBeta0() const SIXTRL_NOEXCEPT
        {
            return ::NS(Particles_get_const_beta0)( this->getCApiPtr() );
        }

        real_pointer_t getBeta0() SIXTRL_NOEXCEPT
        {
            return ::NS(Particles_get_beta0)( this->getCApiPtr() );
        }

        real_t getBeta0Value( index_t const index ) const SIXTRL_NOEXCEPT
        {
            return ::NS(Particles_get_beta0_value)( this->getCApiPtr(), index );
        }

        template< typename Iter >
        void setBeta0( Iter begin, Iter end ) SIXTRL_NOEXCEPT
        {
            using elem_t = typename TParticles< SIXTRL_REAL_T >::num_elements_t;
            elem_t const in_num_particles = std::distance( begin, end );

            if( in_num_particles <= this->getNumParticles() )
            {
                std::copy( begin, end, this->getBeta0() );
            }

            return;
        }

        void setBeta0Value( index_t const index, real_t const value ) SIXTRL_NOEXCEPT
        {
            ::NS(Particles_set_beta0_value)( this->getCApiPtr(), index, value );
            return;
        }

        void assignBeta0Ptr( real_pointer_t ptr ) SIXTRL_NOEXCEPT
        {
            ::NS(Particles_assign_ptr_to_beta0)( this->getCApiPtr(), ptr );
            return;
        }

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        const_real_pointer_t getGamma0() const SIXTRL_NOEXCEPT
        {
            return ::NS(Particles_get_const_gamma0)( this->getCApiPtr() );
        }

        real_pointer_t getGamma0() SIXTRL_NOEXCEPT
        {
            return ::NS(Particles_get_gamma0)( this->getCApiPtr() );
        }

        real_t getGamma0Value( index_t const index ) const SIXTRL_NOEXCEPT
        {
            return ::NS(Particles_get_gamma0_value)( this->getCApiPtr(), index );
        }

        template< typename Iter >
        void setGamma0( Iter begin, Iter end ) SIXTRL_NOEXCEPT
        {
            using elem_t = typename TParticles< SIXTRL_REAL_T >::num_elements_t;
            elem_t const in_num_particles = std::distance( begin, end );

            if( in_num_particles <= this->getNumParticles() )
            {
                std::copy( begin, end, this->getGamma0() );
            }

            return;
        }

        void setGamma0Value( index_t const index, real_t const value ) SIXTRL_NOEXCEPT
        {
            ::NS(Particles_set_gamma0_value)( this->getCApiPtr(), index, value );
            return;
        }

        void assignGamma0Ptr( real_pointer_t ptr ) SIXTRL_NOEXCEPT
        {
            ::NS(Particles_assign_ptr_to_gamma0)( this->getCApiPtr(), ptr );
            return;
        }

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        const_real_pointer_t getP0c() const SIXTRL_NOEXCEPT
        {
            return ::NS(Particles_get_const_p0c)( this->getCApiPtr() );
        }

        real_pointer_t getP0c() SIXTRL_NOEXCEPT
        {
            return ::NS(Particles_get_p0c)( this->getCApiPtr() );
        }

        real_t getP0cValue( index_t const index ) const SIXTRL_NOEXCEPT
        {
            return ::NS(Particles_get_p0c_value)( this->getCApiPtr(), index );
        }

        template< typename Iter >
        void setP0c( Iter begin, Iter end ) SIXTRL_NOEXCEPT
        {
            using elem_t = typename TParticles< SIXTRL_REAL_T >::num_elements_t;
            elem_t const in_num_particles = std::distance( begin, end );

            if( in_num_particles <= this->getNumParticles() )
            {
                std::copy( begin, end, this->getP0c() );
            }

            return;
        }

        void setP0cValue( index_t const index, real_t const value ) SIXTRL_NOEXCEPT
        {
            ::NS(Particles_set_p0c_value)( this->getCApiPtr(), index, value );
            return;
        }

        void assignP0cPtr( real_pointer_t ptr ) SIXTRL_NOEXCEPT
        {
            ::NS(Particles_assign_ptr_to_p0c)( this->getCApiPtr(), ptr );
            return;
        }

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        const_real_pointer_t getS() const SIXTRL_NOEXCEPT
        {
            return ::NS(Particles_get_const_s)( this->getCApiPtr() );
        }

        real_pointer_t getS() SIXTRL_NOEXCEPT
        {
            return ::NS(Particles_get_s)( this->getCApiPtr() );
        }

        real_t getSValue( index_t const index ) const SIXTRL_NOEXCEPT
        {
            return ::NS(Particles_get_s_value)( this->getCApiPtr(), index );
        }

        template< typename Iter >
        void setS( Iter begin, Iter end ) SIXTRL_NOEXCEPT
        {
            using elem_t = typename TParticles< SIXTRL_REAL_T >::num_elements_t;
            elem_t const in_num_particles = std::distance( begin, end );

            if( in_num_particles <= this->getNumParticles() )
            {
                std::copy( begin, end, this->getS() );
            }

            return;
        }

        void setSValue( index_t const index, real_t const value ) SIXTRL_NOEXCEPT
        {
            ::NS(Particles_set_s_value)( this->getCApiPtr(), index, value );
            return;
        }

        void assignSPtr( real_pointer_t ptr ) SIXTRL_NOEXCEPT
        {
            ::NS(Particles_assign_ptr_to_s)( this->getCApiPtr(), ptr );
            return;
        }

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        const_real_pointer_t getX() const SIXTRL_NOEXCEPT
        {
            return ::NS(Particles_get_const_x)( this->getCApiPtr() );
        }

        real_pointer_t getX() SIXTRL_NOEXCEPT
        {
            return ::NS(Particles_get_x)( this->getCApiPtr() );
        }

        real_t getXValue( index_t const index ) const SIXTRL_NOEXCEPT
        {
            return ::NS(Particles_get_x_value)( this->getCApiPtr(), index );
        }

        template< typename Iter >
        void setX( Iter begin, Iter end ) SIXTRL_NOEXCEPT
        {
            using elem_t = typename TParticles< SIXTRL_REAL_T >::num_elements_t;
            elem_t const in_num_particles = std::distance( begin, end );

            if( in_num_particles <= this->getNumParticles() )
            {
                std::copy( begin, end, this->getX() );
            }

            return;
        }

        void setXValue( index_t const index, real_t const value ) SIXTRL_NOEXCEPT
        {
            ::NS(Particles_set_x_value)( this->getCApiPtr(), index, value );
            return;
        }

        void assignXPtr( real_pointer_t ptr ) SIXTRL_NOEXCEPT
        {
            ::NS(Particles_assign_ptr_to_x)( this->getCApiPtr(), ptr );
            return;
        }

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        const_real_pointer_t getY() const SIXTRL_NOEXCEPT
        {
            return ::NS(Particles_get_const_y)( this->getCApiPtr() );
        }

        real_pointer_t getY() SIXTRL_NOEXCEPT
        {
            return ::NS(Particles_get_y)( this->getCApiPtr() );
        }

        real_t getYValue( index_t const index ) const SIXTRL_NOEXCEPT
        {
            return ::NS(Particles_get_y_value)( this->getCApiPtr(), index );
        }

        template< typename Iter >
        void setY( Iter begin, Iter end ) SIXTRL_NOEXCEPT
        {
            using elem_t = typename TParticles< SIXTRL_REAL_T >::num_elements_t;
            elem_t const in_num_particles = std::distance( begin, end );

            if( in_num_particles <= this->getNumParticles() )
            {
                std::copy( begin, end, this->getY() );
            }

            return;
        }

        void setYValue( index_t const index, real_t const value ) SIXTRL_NOEXCEPT
        {
            ::NS(Particles_set_y_value)( this->getCApiPtr(), index, value );
            return;
        }

        void assignYPtr( real_pointer_t ptr ) SIXTRL_NOEXCEPT
        {
            ::NS(Particles_assign_ptr_to_y)( this->getCApiPtr(), ptr );
            return;
        }

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        const_real_pointer_t getPx() const SIXTRL_NOEXCEPT
        {
            return ::NS(Particles_get_const_px)( this->getCApiPtr() );
        }

        real_pointer_t getPx() SIXTRL_NOEXCEPT
        {
            return ::NS(Particles_get_px)( this->getCApiPtr() );
        }

        real_t getPxValue( index_t const index ) const SIXTRL_NOEXCEPT
        {
            return ::NS(Particles_get_px_value)( this->getCApiPtr(), index );
        }

        template< typename Iter >
        void setPx( Iter begin, Iter end ) SIXTRL_NOEXCEPT
        {
            using elem_t = typename TParticles< SIXTRL_REAL_T >::num_elements_t;
            elem_t const in_num_particles = std::distance( begin, end );

            if( in_num_particles <= this->getNumParticles() )
            {
                std::copy( begin, end, this->getPx() );
            }

            return;
        }

        void setPxValue( index_t const index, real_t const value ) SIXTRL_NOEXCEPT
        {
            ::NS(Particles_set_px_value)( this->getCApiPtr(), index, value );
            return;
        }

        void assignPxPtr( real_pointer_t ptr ) SIXTRL_NOEXCEPT
        {
            ::NS(Particles_assign_ptr_to_px)( this->getCApiPtr(), ptr );
            return;
        }

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        const_real_pointer_t getPy() const SIXTRL_NOEXCEPT
        {
            return ::NS(Particles_get_const_py)( this->getCApiPtr() );
        }

        real_pointer_t getPy() SIXTRL_NOEXCEPT
        {
            return ::NS(Particles_get_py)( this->getCApiPtr() );
        }

        real_t getPyValue( index_t const index ) const SIXTRL_NOEXCEPT
        {
            return ::NS(Particles_get_py_value)( this->getCApiPtr(), index );
        }

        template< typename Iter >
        void setPy( Iter begin, Iter end ) SIXTRL_NOEXCEPT
        {
            using elem_t = typename TParticles< SIXTRL_REAL_T >::num_elements_t;
            elem_t const in_num_particles = std::distance( begin, end );

            if( in_num_particles <= this->getNumParticles() )
            {
                std::copy( begin, end, this->getPy() );
            }

            return;
        }

        void setPyValue( index_t const index, real_t const value ) SIXTRL_NOEXCEPT
        {
            ::NS(Particles_set_py_value)( this->getCApiPtr(), index, value );
            return;
        }

        void assignPyPtr( real_pointer_t ptr ) SIXTRL_NOEXCEPT
        {
            ::NS(Particles_assign_ptr_to_py)( this->getCApiPtr(), ptr );
            return;
        }

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        const_real_pointer_t getZeta() const SIXTRL_NOEXCEPT
        {
            return ::NS(Particles_get_const_zeta)( this->getCApiPtr() );
        }

        real_pointer_t getZeta() SIXTRL_NOEXCEPT
        {
            return ::NS(Particles_get_zeta)( this->getCApiPtr() );
        }

        real_t getZetaValue( index_t const index ) const SIXTRL_NOEXCEPT
        {
            return ::NS(Particles_get_zeta_value)( this->getCApiPtr(), index );
        }

        template< typename Iter >
        void setZeta( Iter begin, Iter end ) SIXTRL_NOEXCEPT
        {
            using elem_t = typename TParticles< SIXTRL_REAL_T >::num_elements_t;
            elem_t const in_num_particles = std::distance( begin, end );

            if( in_num_particles <= this->getNumParticles() )
            {
                std::copy( begin, end, this->getZeta() );
            }

            return;
        }

        void setZetaValue( index_t const index, real_t const value ) SIXTRL_NOEXCEPT
        {
            ::NS(Particles_set_zeta_value)( this->getCApiPtr(), index, value );
            return;
        }

        void assignZetaPtr( real_pointer_t ptr ) SIXTRL_NOEXCEPT
        {
            ::NS(Particles_assign_ptr_to_zeta)( this->getCApiPtr(), ptr );
            return;
        }

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        const_real_pointer_t getPSigma() const SIXTRL_NOEXCEPT
        {
            return ::NS(Particles_get_const_psigma)( this->getCApiPtr() );
        }

        real_pointer_t getPSigma() SIXTRL_NOEXCEPT
        {
            return ::NS(Particles_get_psigma)( this->getCApiPtr() );
        }

        real_t getPSigmaValue( index_t const index ) const SIXTRL_NOEXCEPT
        {
            return ::NS(Particles_get_psigma_value)( this->getCApiPtr(), index );
        }

        template< typename Iter >
        void setPSigma( Iter begin, Iter end ) SIXTRL_NOEXCEPT
        {
            using elem_t = typename TParticles< SIXTRL_REAL_T >::num_elements_t;
            elem_t const in_num_particles = std::distance( begin, end );

            if( in_num_particles <= this->getNumParticles() )
            {
                std::copy( begin, end, this->getPSigma() );
            }

            return;
        }

        void setPSigmaValue( index_t const index, real_t const value ) SIXTRL_NOEXCEPT
        {
            ::NS(Particles_set_psigma_value)( this->getCApiPtr(), index, value );
            return;
        }

        void assignPSigmaPtr( real_pointer_t ptr ) SIXTRL_NOEXCEPT
        {
            ::NS(Particles_assign_ptr_to_psigma)( this->getCApiPtr(), ptr );
            return;
        }

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        const_real_pointer_t getDelta() const SIXTRL_NOEXCEPT
        {
            return ::NS(Particles_get_const_delta)( this->getCApiPtr() );
        }

        real_pointer_t getDelta() SIXTRL_NOEXCEPT
        {
            return ::NS(Particles_get_delta)( this->getCApiPtr() );
        }

        real_t getDeltaValue( index_t const index ) const SIXTRL_NOEXCEPT
        {
            return ::NS(Particles_get_delta_value)( this->getCApiPtr(), index );
        }

        template< typename Iter >
        void setDelta( Iter begin, Iter end ) SIXTRL_NOEXCEPT
        {
            using elem_t = typename TParticles< SIXTRL_REAL_T >::num_elements_t;
            elem_t const in_num_particles = std::distance( begin, end );

            if( in_num_particles <= this->getNumParticles() )
            {
                std::copy( begin, end, this->getDelta() );
            }

            return;
        }

        void setDeltaValue( index_t const index, real_t const value ) SIXTRL_NOEXCEPT
        {
            ::NS(Particles_set_delta_value)( this->getCApiPtr(), index, value );
            return;
        }

        void assignDeltaPtr( real_pointer_t ptr ) SIXTRL_NOEXCEPT
        {
            ::NS(Particles_assign_ptr_to_delta)( this->getCApiPtr(), ptr );
            return;
        }

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        const_real_pointer_t getRpp() const SIXTRL_NOEXCEPT
        {
            return ::NS(Particles_get_const_rpp)( this->getCApiPtr() );
        }

        real_pointer_t getRpp() SIXTRL_NOEXCEPT
        {
            return ::NS(Particles_get_rpp)( this->getCApiPtr() );
        }

        real_t getRppValue( index_t const index ) const SIXTRL_NOEXCEPT
        {
            return ::NS(Particles_get_rpp_value)( this->getCApiPtr(), index );
        }

        template< typename Iter >
        void setRpp( Iter begin, Iter end ) SIXTRL_NOEXCEPT
        {
            using elem_t = typename TParticles< SIXTRL_REAL_T >::num_elements_t;
            elem_t const in_num_particles = std::distance( begin, end );

            if( in_num_particles <= this->getNumParticles() )
            {
                std::copy( begin, end, this->getRpp() );
            }

            return;
        }

        void setRppValue( index_t const index, real_t const value ) SIXTRL_NOEXCEPT
        {
            ::NS(Particles_set_rpp_value)( this->getCApiPtr(), index, value );
            return;
        }

        void assignRppPtr( real_pointer_t ptr ) SIXTRL_NOEXCEPT
        {
            ::NS(Particles_assign_ptr_to_rpp)( this->getCApiPtr(), ptr );
            return;
        }

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        const_real_pointer_t getRvv() const SIXTRL_NOEXCEPT
        {
            return ::NS(Particles_get_const_rvv)( this->getCApiPtr() );
        }

        real_pointer_t getRvv() SIXTRL_NOEXCEPT
        {
            return ::NS(Particles_get_rvv)( this->getCApiPtr() );
        }

        real_t getRvvValue( index_t const index ) const SIXTRL_NOEXCEPT
        {
            return ::NS(Particles_get_rvv_value)( this->getCApiPtr(), index );
        }

        template< typename Iter >
        void setRvv( Iter begin, Iter end ) SIXTRL_NOEXCEPT
        {
            using elem_t = typename TParticles< SIXTRL_REAL_T >::num_elements_t;
            elem_t const in_num_particles = std::distance( begin, end );

            if( in_num_particles <= this->getNumParticles() )
            {
                std::copy( begin, end, this->getRvv() );
            }

            return;
        }

        void setRvvValue( index_t const index, real_t const value ) SIXTRL_NOEXCEPT
        {
            ::NS(Particles_set_rvv_value)( this->getCApiPtr(), index, value );
            return;
        }

        void assignRvvPtr( real_pointer_t ptr ) SIXTRL_NOEXCEPT
        {
            ::NS(Particles_assign_ptr_to_rvv)( this->getCApiPtr(), ptr );
            return;
        }

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        const_real_pointer_t getChi() const SIXTRL_NOEXCEPT
        {
            return ::NS(Particles_get_const_chi)( this->getCApiPtr() );
        }

        real_pointer_t getChi() SIXTRL_NOEXCEPT
        {
            return ::NS(Particles_get_chi)( this->getCApiPtr() );
        }

        real_t getChiValue( index_t const index ) const SIXTRL_NOEXCEPT
        {
            return ::NS(Particles_get_chi_value)( this->getCApiPtr(), index );
        }

        template< typename Iter >
        void setChi( Iter begin, Iter end ) SIXTRL_NOEXCEPT
        {
            using elem_t = typename TParticles< SIXTRL_REAL_T >::num_elements_t;
            elem_t const in_num_particles = std::distance( begin, end );

            if( in_num_particles <= this->getNumParticles() )
            {
                std::copy( begin, end, this->getChi() );
            }

            return;
        }

        void setChiValue( index_t const index, real_t const value ) SIXTRL_NOEXCEPT
        {
            ::NS(Particles_set_chi_value)( this->getCApiPtr(), index, value );
            return;
        }

        void assignChiPtr( real_pointer_t ptr ) SIXTRL_NOEXCEPT
        {
            ::NS(Particles_assign_ptr_to_chi)( this->getCApiPtr(), ptr );
            return;
        }

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        const_index_pointer_t getParticleId() const SIXTRL_NOEXCEPT
        {
            return ::NS(Particles_get_const_particle_id)( this->getCApiPtr() );
        }

        index_pointer_t getParticleId() SIXTRL_NOEXCEPT
        {
            return ::NS(Particles_get_particle_id)( this->getCApiPtr() );
        }

        index_t getParticleIdValue( index_t const index ) const SIXTRL_NOEXCEPT
        {
            return ::NS(Particles_get_particle_id_value)(
                this->getCApiPtr(), index );
        }

        template< typename Iter >
        void setParticleId( Iter begin, Iter end ) SIXTRL_NOEXCEPT
        {
            using elem_t = typename TParticles< SIXTRL_REAL_T >::num_elements_t;
            elem_t const in_num_particles = std::distance( begin, end );

            if( in_num_particles <= this->getNumParticles() )
            {
                std::copy( begin, end, this->getParticleId() );
            }

            return;
        }

        void setParticleIdValue( index_t const index, index_t const value ) SIXTRL_NOEXCEPT
        {
            ::NS(Particles_set_particle_id_value)(
                this->getCApiPtr(), index, value );

            return;
        }

        void assignParticleIdPtr( index_pointer_t ptr ) SIXTRL_NOEXCEPT
        {
            ::NS(Particles_assign_ptr_to_particle_id)( this->getCApiPtr(), ptr );
            return;
        }

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        const_index_pointer_t getAtElementId() const SIXTRL_NOEXCEPT
        {
            return ::NS(Particles_get_const_at_element_id)( this->getCApiPtr() );
        }

        index_pointer_t getAtElementId() SIXTRL_NOEXCEPT
        {
            return ::NS(Particles_get_at_element_id)( this->getCApiPtr() );
        }

        index_t getAtElementIdValue( index_t const index ) const SIXTRL_NOEXCEPT
        {
            return ::NS(Particles_get_at_element_id_value)(
                this->getCApiPtr(), index );
        }

        template< typename Iter >
        void setAtElementId( Iter begin, Iter end ) SIXTRL_NOEXCEPT
        {
            using elem_t = typename TParticles< SIXTRL_REAL_T >::num_elements_t;
            elem_t const in_num_particles = std::distance( begin, end );

            if( in_num_particles <= this->getNumParticles() )
            {
                std::copy( begin, end, this->getAtElementId() );
            }

            return;
        }

        void setAtElementIdValue( index_t const index, index_t const value ) SIXTRL_NOEXCEPT
        {
            ::NS(Particles_set_at_element_id_value)(
                this->getCApiPtr(), index, value );

            return;
        }

        void assignAtElementIdPtr( index_pointer_t ptr ) SIXTRL_NOEXCEPT
        {
            ::NS(Particles_assign_ptr_to_at_element_id)( this->getCApiPtr(), ptr );
            return;
        }

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        const_index_pointer_t getAtTurn() const SIXTRL_NOEXCEPT
        {
            return ::NS(Particles_get_const_at_turn)( this->getCApiPtr() );
        }

        index_pointer_t getAtTurn() SIXTRL_NOEXCEPT
        {
            return ::NS(Particles_get_at_turn)( this->getCApiPtr() );
        }

        index_t getAtTurnValue( index_t const index ) const SIXTRL_NOEXCEPT
        {
            return ::NS(Particles_get_at_turn_value)(
                this->getCApiPtr(), index );
        }

        template< typename Iter >
        void setAtTurn( Iter begin, Iter end ) SIXTRL_NOEXCEPT
        {
            using elem_t = typename TParticles< SIXTRL_REAL_T >::num_elements_t;
            elem_t const in_num_particles = std::distance( begin, end );

            if( in_num_particles <= this->getNumParticles() )
            {
                std::copy( begin, end, this->getAtTurn() );
            }

            return;
        }

        void setAtTurnValue( index_t const index, index_t const value ) SIXTRL_NOEXCEPT
        {
            ::NS(Particles_set_at_turn_value)(
                this->getCApiPtr(), index, value );

            return;
        }

        void assignAtTurnPtr( index_pointer_t ptr ) SIXTRL_NOEXCEPT
        {
            ::NS(Particles_assign_ptr_to_at_turn)( this->getCApiPtr(), ptr );
            return;
        }

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        const_index_pointer_t getState() const SIXTRL_NOEXCEPT
        {
            return ::NS(Particles_get_const_state)( this->getCApiPtr() );
        }

        index_pointer_t getState() SIXTRL_NOEXCEPT
        {
            return ::NS(Particles_get_state)( this->getCApiPtr() );
        }

        index_t getStateValue( index_t const index ) const SIXTRL_NOEXCEPT
        {
            return ::NS(Particles_get_state_value)(
                this->getCApiPtr(), index );
        }

        template< typename Iter >
        void setState( Iter begin, Iter end ) SIXTRL_NOEXCEPT
        {
            using elem_t = typename TParticles< SIXTRL_REAL_T >::num_elements_t;
            elem_t const in_num_particles = std::distance( begin, end );

            if( in_num_particles <= this->getNumParticles() )
            {
                std::copy( begin, end, this->getState() );
            }

            return;
        }

        void setStateValue( index_t const index, index_t const value ) SIXTRL_NOEXCEPT
        {
            ::NS(Particles_set_state_value)( this->getCApiPtr(), index, value );
            return;
        }

        void assignStatePtr( index_pointer_t ptr ) SIXTRL_NOEXCEPT
        {
            ::NS(Particles_assign_ptr_to_state)( this->getCApiPtr(), ptr );
            return;
        }
    };

    using Particles = TParticles< SIXTRL_REAL_T >;


}

#endif /* defined( __cplusplus ) */

/* ========================================================================= */
/* =====             Implementation of inline functions                ===== */
/* ========================================================================= */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/buffer.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

SIXTRL_INLINE bool NS(Particles_can_be_added)(
    SIXTRL_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const num_particles,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_objects,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_slots,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_dataptrs )
{
    typedef NS(buffer_size_t) buf_size_t;

    buf_size_t const num_dataptrs = NS(PARTICLES_NUM_DATAPTRS);
    buf_size_t const real_size    = sizeof( NS(particle_real_t) );
    buf_size_t const index_size   = sizeof( NS(particle_index_t) );

    buf_size_t const sizes[] =
    {
        real_size,  real_size,  real_size,  real_size,  real_size,
        real_size,  real_size,  real_size,  real_size,  real_size,  real_size,
        real_size,  real_size,  real_size,  real_size,  real_size,
        index_size, index_size, index_size, index_size
    };

    buf_size_t const counts[] =
    {
        num_particles, num_particles, num_particles, num_particles, num_particles,
        num_particles, num_particles, num_particles,
        num_particles, num_particles, num_particles,
        num_particles, num_particles, num_particles, num_particles, num_particles,
        num_particles, num_particles, num_particles, num_particles
    };

    return NS(Buffer_can_add_object)( buffer, sizeof( NS(Particles) ),
        num_dataptrs, sizes, counts, ptr_requ_objects, ptr_requ_slots,
            ptr_requ_dataptrs );
}

SIXTRL_INLINE SIXTRL_ARGPTR_DEC NS(Particles)* NS(Particles_new)(
    SIXTRL_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const num_particles )
{
    typedef SIXTRL_ARGPTR_DEC NS(Particles)* ptr_to_particles_t;
    typedef NS(Particles)                    particles_t;

    NS(buffer_size_t) const real_size  = sizeof( SIXTRL_REAL_T  );
    NS(buffer_size_t) const int64_size = sizeof( SIXTRL_INT64_T );

    NS(buffer_size_t) const offsets[] =
    {
        offsetof( particles_t, q0 ),
        offsetof( particles_t, mass0 ),
        offsetof( particles_t, beta0 ),
        offsetof( particles_t, gamma0 ),
        offsetof( particles_t, p0c ),
        offsetof( particles_t, s ),
        offsetof( particles_t, x ),
        offsetof( particles_t, y ),
        offsetof( particles_t, px ),
        offsetof( particles_t, py ),
        offsetof( particles_t, zeta ),
        offsetof( particles_t, psigma ),
        offsetof( particles_t, delta ),
        offsetof( particles_t, rpp ),
        offsetof( particles_t, rvv ),
        offsetof( particles_t, chi ),
        offsetof( particles_t, particle_id ),
        offsetof( particles_t, at_element_id ),
        offsetof( particles_t, at_turn ),
        offsetof( particles_t, state )
    };

    NS(buffer_size_t) const sizes[] =
    {
        real_size,  real_size,  real_size,  real_size,
        real_size,  real_size,  real_size,  real_size,
        real_size,  real_size,  real_size,  real_size,
        real_size,  real_size,  real_size,  real_size,
        int64_size, int64_size, int64_size, int64_size
    };

    NS(buffer_size_t) const counts[] =
    {
        num_particles, num_particles, num_particles, num_particles,
        num_particles, num_particles, num_particles, num_particles,
        num_particles, num_particles, num_particles, num_particles,
        num_particles, num_particles, num_particles, num_particles,
        num_particles, num_particles, num_particles, num_particles
    };

    SIXTRL_ASSERT( buffer != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( NS(Buffer_has_datastore)( buffer ) );
    SIXTRL_ASSERT( NS(Buffer_allow_append_objects)( buffer ) );
    SIXTRL_ASSERT( NS(PARTICLES_NUM_DATAPTRS) == 20u );

    NS(Particles) particles;
    NS(Particles_preset)( &particles );
    NS(Particles_set_num_of_particles)( &particles, num_particles );

    return ( ptr_to_particles_t )( uintptr_t )NS(Object_get_begin_addr)(
        NS(Buffer_add_object)( buffer, &particles, sizeof( NS(Particles) ),
            NS(OBJECT_TYPE_PARTICLE), NS(PARTICLES_NUM_DATAPTRS),
                offsets, sizes, counts ) );
}

SIXTRL_INLINE SIXTRL_ARGPTR_DEC NS(Particles)* NS(Particles_add)(
    SIXTRL_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const num_particles,
    NS(particle_real_ptr_t)  q0_ptr,
    NS(particle_real_ptr_t)  mass0_ptr,
    NS(particle_real_ptr_t)  beta0_ptr,
    NS(particle_real_ptr_t)  gamma0_ptr,
    NS(particle_real_ptr_t)  p0c_ptr,
    NS(particle_real_ptr_t)  s_ptr,
    NS(particle_real_ptr_t)  x_ptr,
    NS(particle_real_ptr_t)  y_ptr,
    NS(particle_real_ptr_t)  px_ptr,
    NS(particle_real_ptr_t)  py_ptr,
    NS(particle_real_ptr_t)  zeta_ptr,
    NS(particle_real_ptr_t)  psigma_ptr,
    NS(particle_real_ptr_t)  delta_ptr,
    NS(particle_real_ptr_t)  rpp_ptr,
    NS(particle_real_ptr_t)  rvv_ptr,
    NS(particle_real_ptr_t)  chi_ptr,
    NS(particle_index_ptr_t) particle_id_ptr,
    NS(particle_index_ptr_t) at_element_id_ptr,
    NS(particle_index_ptr_t) at_turn_ptr,
    NS(particle_index_ptr_t) state_ptr )
{
    typedef SIXTRL_ARGPTR_DEC NS(Particles)* ptr_to_particles_t;
    typedef NS(Particles)                    particles_t;

    NS(buffer_size_t) const real_size  = sizeof( SIXTRL_REAL_T  );
    NS(buffer_size_t) const int64_size = sizeof( SIXTRL_INT64_T );

    NS(buffer_size_t) const offsets[] =
    {
        offsetof( particles_t, q0 ),
        offsetof( particles_t, mass0 ),
        offsetof( particles_t, beta0 ),
        offsetof( particles_t, gamma0 ),
        offsetof( particles_t, p0c ),
        offsetof( particles_t, s ),
        offsetof( particles_t, x ),
        offsetof( particles_t, y ),
        offsetof( particles_t, px ),
        offsetof( particles_t, py ),
        offsetof( particles_t, zeta ),
        offsetof( particles_t, psigma ),
        offsetof( particles_t, delta ),
        offsetof( particles_t, rpp ),
        offsetof( particles_t, rvv ),
        offsetof( particles_t, chi ),
        offsetof( particles_t, particle_id ),
        offsetof( particles_t, at_element_id ),
        offsetof( particles_t, at_turn ),
        offsetof( particles_t, state )
    };

    NS(buffer_size_t) const sizes[] =
    {
        real_size,  real_size,  real_size,  real_size,
        real_size,  real_size,  real_size,  real_size,
        real_size,  real_size,  real_size,  real_size,
        real_size,  real_size,  real_size,  real_size,
        int64_size, int64_size, int64_size, int64_size
    };

    NS(buffer_size_t) const counts[] =
    {
        num_particles, num_particles, num_particles, num_particles,
        num_particles, num_particles, num_particles, num_particles,
        num_particles, num_particles, num_particles, num_particles,
        num_particles, num_particles, num_particles, num_particles,
        num_particles, num_particles, num_particles, num_particles
    };

    SIXTRL_ASSERT( buffer != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( NS(Buffer_has_datastore)( buffer ) );
    SIXTRL_ASSERT( NS(Buffer_allow_append_objects)( buffer ) );
    SIXTRL_ASSERT( NS(PARTICLES_NUM_DATAPTRS) == 20u );

    NS(Particles) particles;
    NS(Particles_set_num_of_particles)(           &particles, num_particles );

    NS(Particles_assign_ptr_to_q0)(            &particles, q0_ptr );
    NS(Particles_assign_ptr_to_mass0)(         &particles, mass0_ptr );
    NS(Particles_assign_ptr_to_beta0)(         &particles, beta0_ptr );
    NS(Particles_assign_ptr_to_gamma0)(        &particles, gamma0_ptr );
    NS(Particles_assign_ptr_to_p0c)(           &particles, p0c_ptr );

    NS(Particles_assign_ptr_to_s)(             &particles, s_ptr );
    NS(Particles_assign_ptr_to_x)(             &particles, x_ptr );
    NS(Particles_assign_ptr_to_y)(             &particles, y_ptr );
    NS(Particles_assign_ptr_to_px)(            &particles, px_ptr );
    NS(Particles_assign_ptr_to_py)(            &particles, py_ptr );
    NS(Particles_assign_ptr_to_zeta)(          &particles, zeta_ptr );

    NS(Particles_assign_ptr_to_psigma)(        &particles, psigma_ptr );
    NS(Particles_assign_ptr_to_delta)(         &particles, delta_ptr );
    NS(Particles_assign_ptr_to_rpp)(           &particles, rpp_ptr );
    NS(Particles_assign_ptr_to_rvv)(           &particles, rvv_ptr );
    NS(Particles_assign_ptr_to_chi)(           &particles, chi_ptr );

    NS(Particles_assign_ptr_to_particle_id)(   &particles, particle_id_ptr );
    NS(Particles_assign_ptr_to_at_element_id)( &particles, at_element_id_ptr );
    NS(Particles_assign_ptr_to_at_turn)(       &particles, at_turn_ptr );
    NS(Particles_assign_ptr_to_state)(         &particles, state_ptr );

    return ( ptr_to_particles_t )( uintptr_t )NS(Object_get_begin_addr)(
        NS(Buffer_add_object)( buffer, &particles, sizeof( NS(Particles) ),
            NS(OBJECT_TYPE_PARTICLE), NS(PARTICLES_NUM_DATAPTRS),
                offsets, sizes, counts ) );
}


/* ------------------------------------------------------------------------- */

SIXTRL_INLINE SIXTRL_ARGPTR_DEC NS(Particles)* NS(Particles_preset)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles )
{
    if( particles != SIXTRL_NULLPTR )
    {
        particles->q0            = SIXTRL_NULLPTR;
        particles->mass0         = SIXTRL_NULLPTR;
        particles->beta0         = SIXTRL_NULLPTR;
        particles->gamma0        = SIXTRL_NULLPTR;
        particles->p0c           = SIXTRL_NULLPTR;

        particles->s             = SIXTRL_NULLPTR;
        particles->x             = SIXTRL_NULLPTR;
        particles->px            = SIXTRL_NULLPTR;
        particles->y             = SIXTRL_NULLPTR;
        particles->py            = SIXTRL_NULLPTR;
        particles->zeta          = SIXTRL_NULLPTR;

        particles->psigma        = SIXTRL_NULLPTR;
        particles->delta         = SIXTRL_NULLPTR;
        particles->rpp           = SIXTRL_NULLPTR;
        particles->rvv           = SIXTRL_NULLPTR;
        particles->chi           = SIXTRL_NULLPTR;

        particles->particle_id   = SIXTRL_NULLPTR;
        particles->at_element_id = SIXTRL_NULLPTR;
        particles->at_turn       = SIXTRL_NULLPTR;
        particles->state         = SIXTRL_NULLPTR;

        NS(Particles_set_num_of_particles)(
            particles, ( NS(particle_num_elements_t) )0 );
    }

    return particles;
}

SIXTRL_INLINE void NS(Particles_preset_values)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT p )
{
    typedef NS(buffer_size_t)           buf_size_t;
    typedef NS(particle_num_elements_t) elem_size_t;

    elem_size_t const temp_num_particles =
        NS(Particles_get_num_of_particles)( p );

    buf_size_t const num_of_particles = ( temp_num_particles >= 0 )
        ? ( buf_size_t )temp_num_particles : ( buf_size_t )0u;

    if( ( p != SIXTRL_NULLPTR ) &&
        ( num_of_particles > ( buf_size_t )0u ) )
    {
        buf_size_t ii = 0;

        NS(particle_real_t)  const ZERO_REAL      = ( NS(particle_real_t)  )0;
        NS(particle_index_t) const PARTICLE_ID    = ( NS(particle_index_t) )-1;
        NS(particle_index_t) const ELEMENT_ID     = ( NS(particle_index_t) )-1;
        NS(particle_index_t) const TURN_ID        = ( NS(particle_index_t) )-1;
        NS(particle_index_t) const PARTICLE_STATE = ( NS(particle_index_t) )-1;

        SIXTRL_ASSERT( NS(Particles_get_const_q0)(     p ) != SIXTRL_NULLPTR );
        SIXTRL_ASSERT( NS(Particles_get_const_beta0)(  p ) != SIXTRL_NULLPTR );
        SIXTRL_ASSERT( NS(Particles_get_const_mass0)(  p ) != SIXTRL_NULLPTR );
        SIXTRL_ASSERT( NS(Particles_get_const_gamma0)( p ) != SIXTRL_NULLPTR );
        SIXTRL_ASSERT( NS(Particles_get_const_p0c)(    p ) != SIXTRL_NULLPTR );

        SIXTRL_ASSERT( NS(Particles_get_const_s)(      p ) != SIXTRL_NULLPTR );
        SIXTRL_ASSERT( NS(Particles_get_const_x)(      p ) != SIXTRL_NULLPTR );
        SIXTRL_ASSERT( NS(Particles_get_const_y)(      p ) != SIXTRL_NULLPTR );
        SIXTRL_ASSERT( NS(Particles_get_const_px)(     p ) != SIXTRL_NULLPTR );
        SIXTRL_ASSERT( NS(Particles_get_const_py)(     p ) != SIXTRL_NULLPTR );
        SIXTRL_ASSERT( NS(Particles_get_const_zeta)(   p ) != SIXTRL_NULLPTR );

        SIXTRL_ASSERT( NS(Particles_get_const_psigma)( p ) != SIXTRL_NULLPTR );
        SIXTRL_ASSERT( NS(Particles_get_const_delta)(  p ) != SIXTRL_NULLPTR );
        SIXTRL_ASSERT( NS(Particles_get_const_rpp)(    p ) != SIXTRL_NULLPTR );
        SIXTRL_ASSERT( NS(Particles_get_const_rvv)(    p ) != SIXTRL_NULLPTR );
        SIXTRL_ASSERT( NS(Particles_get_const_chi)(    p ) != SIXTRL_NULLPTR );

        SIXTRL_ASSERT( SIXTRL_NULLPTR !=
            NS(Particles_get_const_particle_id)( p ) );

        SIXTRL_ASSERT( SIXTRL_NULLPTR !=
            NS(Particles_get_const_at_element_id)( p ) );

        SIXTRL_ASSERT( SIXTRL_NULLPTR !=
            NS(Particles_get_const_at_turn)( p ) );

        SIXTRL_ASSERT( SIXTRL_NULLPTR !=
            NS(Particles_get_const_state)( p ) );


        for( ; ii < num_of_particles ; ++ii )
        {
            NS(Particles_set_q0_value)(            p, ii, ZERO_REAL );
            NS(Particles_set_beta0_value)(         p, ii, ZERO_REAL );
            NS(Particles_set_mass0_value)(         p, ii, ZERO_REAL );
            NS(Particles_set_gamma0_value)(        p, ii, ZERO_REAL );
            NS(Particles_set_p0c_value)(           p, ii, ZERO_REAL );

            NS(Particles_set_s_value)(             p, ii, ZERO_REAL );
            NS(Particles_set_x_value)(             p, ii, ZERO_REAL );
            NS(Particles_set_y_value)(             p, ii, ZERO_REAL );
            NS(Particles_set_px_value)(            p, ii, ZERO_REAL );
            NS(Particles_set_py_value)(            p, ii, ZERO_REAL );
            NS(Particles_set_zeta_value)(          p, ii, ZERO_REAL );

            NS(Particles_set_psigma_value)(        p, ii, ZERO_REAL );
            NS(Particles_set_delta_value)(         p, ii, ZERO_REAL );
            NS(Particles_set_rpp_value)(           p, ii, ZERO_REAL );
            NS(Particles_set_rvv_value)(           p, ii, ZERO_REAL );
            NS(Particles_set_chi_value)(           p, ii, ZERO_REAL );

            NS(Particles_set_particle_id_value)(   p, ii, PARTICLE_ID );
            NS(Particles_set_at_element_id_value)( p, ii, ELEMENT_ID  );
            NS(Particles_set_at_turn_value)(       p, ii, TURN_ID );
            NS(Particles_set_state_value)(         p, ii, PARTICLE_STATE );
        }
    }

    return;
}

SIXTRL_INLINE NS(particle_num_elements_t)
NS(Particles_get_num_of_particles)(
    const SIXTRL_ARGPTR_DEC NS(Particles) *const SIXTRL_RESTRICT particles )
{
    return ( particles != SIXTRL_NULLPTR )
        ? particles->num_particles
        : ( NS(particle_num_elements_t) )0;
}

SIXTRL_INLINE void NS(Particles_set_num_of_particles)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const num_of_particles )
{
    SIXTRL_ASSERT( particles != SIXTRL_NULLPTR );
    particles->num_particles = num_of_particles;
    return;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE SIXTRL_DATAPTR_DEC NS(Particles) const*
NS(BufferIndex_get_const_particles)( SIXTRL_ARGPTR_DEC const
    NS(Object) *const SIXTRL_RESTRICT obj_index )
{
    typedef SIXTRL_DATAPTR_DEC NS(Particles) const* ptr_to_particles_t;
    ptr_to_particles_t ptr_to_particles = SIXTRL_NULLPTR;

    if( ( obj_index != SIXTRL_NULLPTR ) &&
        ( NS(Object_get_type_id)( obj_index ) == NS(OBJECT_TYPE_PARTICLE) ) &&
        ( NS(Object_get_size)( obj_index ) > sizeof( NS(Particles) ) ) )
    {
        ptr_to_particles = ( ptr_to_particles_t )( uintptr_t
            )NS(Object_get_begin_addr)( obj_index );
    }

    return ptr_to_particles;
}

SIXTRL_INLINE SIXTRL_DATAPTR_DEC NS(Particles)*
NS(BufferIndex_get_particles)( SIXTRL_ARGPTR_DEC const
    NS(Object) *const SIXTRL_RESTRICT index )
{
    typedef SIXTRL_DATAPTR_DEC NS(Particles)* ptr_to_particles_t;
    return ( ptr_to_particles_t )NS(BufferIndex_get_const_particles)( index );
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE void NS(Particles_copy_single_unchecked)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT destination,
    NS(particle_num_elements_t) const destination_idx,
    SIXTRL_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT source,
    NS(particle_num_elements_t) const source_idx )
{
    SIXTRL_ASSERT(
        ( destination != 0 ) && ( source != 0 ) &&
        ( NS(Particles_get_num_of_particles)( destination ) > destination_idx ) &&
        ( NS(Particles_get_num_of_particles)( source )      > source_idx      ) );

    NS(Particles_set_q0_value)( destination, destination_idx,
          NS(Particles_get_q0_value)( source, source_idx ) );

    NS(Particles_set_mass0_value)( destination, destination_idx,
        NS(Particles_get_mass0_value)( source, source_idx ) );

    NS(Particles_set_beta0_value)( destination, destination_idx,
        NS(Particles_get_beta0_value)( source, source_idx ) );

    NS(Particles_set_gamma0_value)( destination, destination_idx,
        NS(Particles_get_gamma0_value)( source, source_idx ) );

    NS(Particles_set_p0c_value)( destination, destination_idx,
        NS(Particles_get_p0c_value)( source, source_idx ) );

    NS(Particles_set_s_value)( destination, destination_idx,
        NS(Particles_get_s_value)( source, source_idx ) );

    NS(Particles_set_x_value)( destination, destination_idx,
        NS(Particles_get_x_value)( source, source_idx ) );

    NS(Particles_set_y_value)( destination, destination_idx,
        NS(Particles_get_y_value)( source, source_idx ) );

    NS(Particles_set_px_value)( destination, destination_idx,
        NS(Particles_get_px_value)( source, source_idx ) );

    NS(Particles_set_py_value)( destination, destination_idx,
        NS(Particles_get_py_value)( source, source_idx ) );

    NS(Particles_set_zeta_value)( destination, destination_idx,
        NS(Particles_get_zeta_value)( source, source_idx ) );

    NS(Particles_set_psigma_value)( destination, destination_idx,
        NS(Particles_get_psigma_value)( source, source_idx ) );

    NS(Particles_set_delta_value)( destination, destination_idx,
        NS(Particles_get_delta_value)( source, source_idx ) );

    NS(Particles_set_rpp_value )( destination, destination_idx,
        NS(Particles_get_rpp_value)( source, source_idx ) );

    NS(Particles_set_rvv_value)( destination, destination_idx,
        NS( Particles_get_rvv_value)( source, source_idx ) );

    NS(Particles_set_chi_value)( destination, destination_idx,
        NS(Particles_get_chi_value)( source, source_idx ) );

    NS(Particles_set_particle_id_value)( destination, destination_idx,
        NS(Particles_get_particle_id_value)( source, source_idx ) );

    NS(Particles_set_at_element_id_value)( destination, destination_idx,
        NS( Particles_get_at_element_id_value)( source, source_idx ) );

    NS(Particles_set_at_turn_value)( destination, destination_idx,
        NS(Particles_get_at_turn_value)( source, source_idx ) );

    NS(Particles_set_state_value)( destination, destination_idx,
        NS(Particles_get_state_value)( source, source_idx ) );

    return;
}

SIXTRL_INLINE void NS(Particles_copy_range_unchecked)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT destination,
    SIXTRL_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT source,
    NS(particle_num_elements_t) const start_index,
    NS(particle_num_elements_t) const end_index )
{
    typedef NS(particle_num_elements_t) num_elem_t;

    num_elem_t const num_to_copy = ( start_index <= end_index )
        ? ( end_index - start_index ) : ( num_elem_t )0;

    SIXTRL_ASSERT(
        ( destination != SIXTRL_NULLPTR ) &&
        ( source      != SIXTRL_NULLPTR ) && ( start_index >= 0 ) &&
        ( NS(Particles_get_num_of_particles)( destination ) >= end_index ) &&
        ( NS(Particles_get_num_of_particles)( source )      >= end_index ) );

    {
        SIXTRACKLIB_COPY_VALUES( NS(particle_real_t),
            &destination->q0[ start_index ], &source->q0[ start_index ],
                num_to_copy );
    }

    {
        SIXTRACKLIB_COPY_VALUES( NS(particle_real_t),
            &destination->beta0[ start_index ], &source->beta0[ start_index ],
                num_to_copy );
    }

    {
        SIXTRACKLIB_COPY_VALUES( NS(particle_real_t),
            &destination->mass0[ start_index ], &source->mass0[ start_index ],
                num_to_copy );
    }

    {
        SIXTRACKLIB_COPY_VALUES( NS(particle_real_t),
            &destination->gamma0[ start_index ], &source->gamma0[ start_index ],
                num_to_copy );
    }

    {
        SIXTRACKLIB_COPY_VALUES( NS(particle_real_t),
            &destination->p0c[ start_index ], &source->p0c[ start_index ],
                num_to_copy );
    }

    {
        SIXTRACKLIB_COPY_VALUES( NS(particle_real_t),
            &destination->s[ start_index ], &source->s[ start_index ],
                num_to_copy );
    }

    {
        SIXTRACKLIB_COPY_VALUES( NS(particle_real_t),
            &destination->x[ start_index ], &source->x[ start_index ],
                num_to_copy );
    }

    {
        SIXTRACKLIB_COPY_VALUES( NS(particle_real_t),
            &destination->y[ start_index ], &source->y[ start_index ],
                num_to_copy );
    }

    {
        SIXTRACKLIB_COPY_VALUES( NS(particle_real_t),
            &destination->px[ start_index ], &source->px[ start_index ],
                num_to_copy );
    }

    {
        SIXTRACKLIB_COPY_VALUES( NS(particle_real_t),
            &destination->py[ start_index ], &source->py[ start_index ],
                num_to_copy );
    }

    {
        SIXTRACKLIB_COPY_VALUES( NS(particle_real_t),
            &destination->zeta[ start_index ], &source->zeta[ start_index ],
                num_to_copy );
    }

    {
        SIXTRACKLIB_COPY_VALUES( NS(particle_real_t),
            &destination->psigma[ start_index ], &source->psigma[ start_index ],
                num_to_copy );
    }

    {
        SIXTRACKLIB_COPY_VALUES( NS(particle_real_t),
            &destination->delta[ start_index ], &source->delta[ start_index ],
                num_to_copy );
    }

    {
        SIXTRACKLIB_COPY_VALUES( NS(particle_real_t),
            &destination->rpp[ start_index ], &source->rpp[ start_index ],
                num_to_copy );
    }

    {
        SIXTRACKLIB_COPY_VALUES( NS(particle_real_t),
            &destination->rvv[ start_index ], &source->rvv[ start_index ],
                num_to_copy );
    }

    {
        SIXTRACKLIB_COPY_VALUES( NS(particle_real_t),
            &destination->chi[ start_index ], &source->chi[ start_index ],
                num_to_copy );
    }

    {
        SIXTRACKLIB_COPY_VALUES( NS(particle_index_t),
            &destination->particle_id[ start_index ],
                &source->particle_id[ start_index ], num_to_copy );
    }

    {
        SIXTRACKLIB_COPY_VALUES( NS(particle_index_t),
            &destination->at_element_id[ start_index ],
                &source->at_element_id[ start_index ], num_to_copy );
    }

    {
        SIXTRACKLIB_COPY_VALUES( NS(particle_index_t),
            &destination->at_turn[ start_index ],
                &source->at_turn[ start_index ], num_to_copy );
    }

    {
        SIXTRACKLIB_COPY_VALUES( NS(particle_index_t),
            &destination->state[ start_index ], &source->state[ start_index ],
                num_to_copy );
    }

    return;
}

SIXTRL_INLINE void NS(Particles_copy_all_unchecked)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT destination,
    SIXTRL_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT source )
{
    typedef NS(particle_num_elements_t) num_elem_t;

    num_elem_t const num = NS(Particles_get_num_of_particles)( source );

    SIXTRL_ASSERT(
        ( destination != SIXTRL_NULLPTR ) &&
        ( source      != SIXTRL_NULLPTR ) &&
        ( num >  ( num_elem_t )0u ) &&
        ( num == ( NS(Particles_get_num_of_particles)( destination ) ) ) );

    NS(Particles_copy_range_unchecked)( destination, source, 0, num );
    return;
}

SIXTRL_INLINE void NS(Particles_calculate_difference)(
    SIXTRL_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT lhs,
    SIXTRL_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT rhs,
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT diff )
{
    typedef NS(particle_num_elements_t) num_elem_t;

    num_elem_t const num_particles = NS(Particles_get_num_of_particles)( lhs );

    if( ( lhs  != SIXTRL_NULLPTR ) &&
        ( rhs  != SIXTRL_NULLPTR ) &&
        ( diff != SIXTRL_NULLPTR ) &&
        ( num_particles >  ( num_elem_t )0u ) &&
        ( num_particles == NS(Particles_get_num_of_particles)( rhs  ) ) &&
        ( num_particles == NS(Particles_get_num_of_particles)( diff ) ) )
    {
        num_elem_t ii = 0;

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

            NS(Particles_set_zeta_value)( diff, ii,
                NS(Particles_get_zeta_value)( lhs, ii ) -
                NS(Particles_get_zeta_value)( rhs, ii ) );

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

            NS(Particles_set_at_element_id_value)( diff, ii,
                NS(Particles_get_at_element_id_value)( lhs, ii ) -
                NS(Particles_get_at_element_id_value)( rhs, ii ) );

            NS(Particles_set_at_turn_value)( diff, ii,
                NS(Particles_get_at_turn_value)( lhs, ii ) -
                NS(Particles_get_at_turn_value)( rhs, ii ) );

            NS(Particles_set_state_value)( diff, ii,
                NS(Particles_get_state_value)( lhs, ii ) -
                NS(Particles_get_state_value)( rhs, ii ) );
        }
    }

    return;
}

SIXTRL_INLINE void NS(Particles_get_max_value)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT destination,
    SIXTRL_ARGPTR_DEC NS(particle_num_elements_t)* SIXTRL_RESTRICT max_value_index,
    SIXTRL_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT source )
{
    typedef NS(buffer_size_t)        buf_size_t;
    typedef NS(particle_real_ptr_t)  real_ptr_t;
    typedef NS(particle_real_t)      real_t;
    typedef NS(particle_index_ptr_t) index_ptr_t;
    typedef NS(particle_index_t)     index_t;

    SIXTRL_STATIC_VAR real_t const     ZERO      = ( real_t )0;
    SIXTRL_STATIC_VAR buf_size_t const ZERO_SIZE = ( buf_size_t )0u;

    if( ( destination != SIXTRL_NULLPTR ) &&
        ( source      != SIXTRL_NULLPTR ) &&
        ( NS(Particles_get_num_of_particles)( destination ) > 0u ) &&
        ( NS(Particles_get_num_of_particles)( source      ) > 0u ) )
    {
        buf_size_t dummy_max_value_indices[ 20 ] =
        {
            ZERO_SIZE, ZERO_SIZE, ZERO_SIZE, ZERO_SIZE, ZERO_SIZE,
            ZERO_SIZE, ZERO_SIZE, ZERO_SIZE, ZERO_SIZE, ZERO_SIZE, ZERO_SIZE,
            ZERO_SIZE, ZERO_SIZE, ZERO_SIZE, ZERO_SIZE, ZERO_SIZE,
            ZERO_SIZE, ZERO_SIZE, ZERO_SIZE, ZERO_SIZE
        };

        real_ptr_t out_real_values_begin[ 16 ] =
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
            NS(Particles_get_zeta)(   destination ),
            NS(Particles_get_psigma)( destination ),
            NS(Particles_get_delta)(  destination ),
            NS(Particles_get_rpp)(    destination ),
            NS(Particles_get_rvv)(    destination ),
            NS(Particles_get_chi)(    destination )
        };

        index_ptr_t out_index_values_begin[ 4 ] =
        {
            ( index_ptr_t )NS(Particles_get_particle_id)(   destination ),
            ( index_ptr_t )NS(Particles_get_at_element_id)( destination ),
            ( index_ptr_t )NS(Particles_get_at_turn)(       destination ),
            ( index_ptr_t )NS(Particles_get_state)(         destination )
        };

        real_ptr_t in_real_values_begin[ 16 ] =
        {
            ( real_ptr_t )NS(Particles_get_const_q0)(     source ),
            ( real_ptr_t )NS(Particles_get_const_beta0)(  source ),
            ( real_ptr_t )NS(Particles_get_const_mass0)(  source ),
            ( real_ptr_t )NS(Particles_get_const_gamma0)( source ),
            ( real_ptr_t )NS(Particles_get_const_p0c)(    source ),
            ( real_ptr_t )NS(Particles_get_const_s)(      source ),
            ( real_ptr_t )NS(Particles_get_const_x)(      source ),
            ( real_ptr_t )NS(Particles_get_const_y)(      source ),
            ( real_ptr_t )NS(Particles_get_const_px)(     source ),
            ( real_ptr_t )NS(Particles_get_const_py)(     source ),
            ( real_ptr_t )NS(Particles_get_const_zeta)(   source ),
            ( real_ptr_t )NS(Particles_get_const_psigma)( source ),
            ( real_ptr_t )NS(Particles_get_const_delta)(  source ),
            ( real_ptr_t )NS(Particles_get_const_rpp)(    source ),
            ( real_ptr_t )NS(Particles_get_const_rvv)(    source ),
            ( real_ptr_t )NS(Particles_get_const_chi)(    source )
        };

        index_ptr_t in_index_values_begin[ 4 ] =
        {
            ( index_ptr_t )NS(Particles_get_const_particle_id)(   source ),
            ( index_ptr_t )NS(Particles_get_const_at_element_id)( source ),
            ( index_ptr_t )NS(Particles_get_const_at_turn)(       source ),
            ( index_ptr_t )NS(Particles_get_const_state)(         source )
        };

        buf_size_t ii = 0;
        buf_size_t jj = 0;

        buf_size_t const num_particles =
            NS(Particles_get_num_of_particles)( destination );

        real_ptr_t in_real_values_end[ 16 ] =
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

        index_ptr_t in_index_values_end[ 4 ] =
        {
            in_index_values_begin[  0 ] + num_particles,
            in_index_values_begin[  1 ] + num_particles,
            in_index_values_begin[  2 ] + num_particles,
            in_index_values_begin[  3 ] + num_particles
        };

        for( ; ii < 16 ; ++ii )
        {
            real_ptr_t in_it     = in_real_values_begin[ ii ];
            real_ptr_t in_end    = in_real_values_end[ ii ];

            real_t max_value     = ( real_t )0.0;
            real_t cmp_max_value = max_value;

            buf_size_t kk = ZERO_SIZE;
            dummy_max_value_indices[ ii ] = ZERO_SIZE;

            for( ; in_it != in_end ; ++in_it, ++kk )
            {
                real_t const value     = *in_it;
                real_t const cmp_value = ( value >= ZERO ) ? value : -value;

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
            index_ptr_t in_it     = in_index_values_begin[ ii ];
            index_ptr_t in_end    = in_index_values_end[ ii ];

            index_t max_value     = ( index_t )0;
            index_t cmp_max_value = max_value;

            buf_size_t kk = ZERO_SIZE;
            dummy_max_value_indices[ jj ] = ZERO_SIZE;

            for( ; in_it != in_end ; ++in_it, ++kk )
            {
                index_t const value     = *in_it;
                index_t const cmp_value = ( value > 0 ) ? value : -value;

                if( cmp_max_value < cmp_value )
                {
                    cmp_max_value = cmp_value;
                    max_value     = value;
                    dummy_max_value_indices[ jj ] = kk;
                }
            }

            *out_index_values_begin[ ii ] = max_value;
        }

        if( max_value_index != 0 )
        {
            SIXTRACKLIB_COPY_VALUES( buf_size_t, max_value_index,
                                     &dummy_max_value_indices[ 0 ], 20 );

            max_value_index = max_value_index + 20;
        }
    }

    return;
}

SIXTRL_INLINE void NS(Particles_buffer_calculate_difference)(
    SIXTRL_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT lhs,
    SIXTRL_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT rhs,
    SIXTRL_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT diff )
{
    typedef NS(particle_num_elements_t) num_elem_t;

    num_elem_t const lhs_num_objects  = NS(Buffer_get_num_of_objects)( lhs );
    num_elem_t const rhs_num_objects  = NS(Buffer_get_num_of_objects)( rhs );
    num_elem_t const diff_num_objects = NS(Buffer_get_num_of_objects)( diff );

    if( ( lhs_num_objects == rhs_num_objects  ) &&
        ( lhs_num_objects == diff_num_objects ) &&
        ( lhs_num_objects >  ( num_elem_t )0u ) )
    {
        typedef SIXTRL_DATAPTR_DEC NS(Object) const* obj_const_ptr_t;
        typedef SIXTRL_DATAPTR_DEC NS(Object)*       obj_ptr_t;

        obj_const_ptr_t lhs_it  = ( obj_const_ptr_t )( uintptr_t
            )NS(Buffer_get_objects_begin_addr)( lhs );

        obj_const_ptr_t lhs_end = ( obj_const_ptr_t )( uintptr_t
            )NS(Buffer_get_objects_end_addr)( lhs );

        obj_const_ptr_t rhs_it  = ( obj_const_ptr_t )( uintptr_t
            )NS(Buffer_get_objects_begin_addr)( rhs );

        obj_ptr_t diff_it = ( obj_ptr_t )( uintptr_t
            )NS(Buffer_get_objects_begin_addr)( diff );

        for( ; lhs_it != lhs_end ; ++lhs_it, ++rhs_it, ++diff_it )
        {
            SIXTRL_ARGPTR_DEC NS(Particles) const* lhs_particles =
                NS(BufferIndex_get_const_particles)( lhs_it );

            SIXTRL_ARGPTR_DEC NS(Particles) const* rhs_particles =
                NS(BufferIndex_get_const_particles)( rhs_it );

            SIXTRL_ARGPTR_DEC NS(Particles)* diff_particles =
                NS(BufferIndex_get_particles)( diff_it );

            NS(Particles_calculate_difference)(
                lhs_particles, rhs_particles, diff_particles );
        }
    }

    return;
}

SIXTRL_INLINE void NS(Particles_buffer_get_max_value)(
    SIXTRL_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT destination,
    SIXTRL_ARGPTR_DEC NS(particle_num_elements_t)* SIXTRL_RESTRICT max_value_index,
    SIXTRL_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT source )
{
    typedef NS(particle_num_elements_t) num_elem_t;

    num_elem_t const num_source_objects =
        NS(Buffer_get_num_of_objects)( source );

    num_elem_t const num_destination_objects =
        NS(Buffer_get_num_of_objects)( destination );

    if( (  num_source_objects == num_destination_objects ) &&
        (  num_source_objects >  ( num_elem_t )0 ) )
    {
        typedef SIXTRL_DATAPTR_DEC NS(Object) const* obj_const_ptr_t;
        typedef SIXTRL_DATAPTR_DEC NS(Object)*       obj_ptr_t;

        obj_const_ptr_t src_it  = ( obj_const_ptr_t )( uintptr_t
            )NS(Buffer_get_objects_begin_addr)( source );

        obj_const_ptr_t src_end = ( obj_const_ptr_t )( uintptr_t
            )NS(Buffer_get_objects_end_addr)( source );

        obj_ptr_t dest_it = ( obj_ptr_t )( uintptr_t
            )NS(Buffer_get_objects_begin_addr)( destination );

        for( ; src_it != src_end ; ++src_end, ++dest_it )
        {
            SIXTRL_ARGPTR_DEC NS(Particles) const* source_particles =
                NS(BufferIndex_get_const_particles)( src_it );

            SIXTRL_ARGPTR_DEC NS(Particles)* dest_particles =
                NS(BufferIndex_get_particles)( dest_it );

            NS(Particles_get_max_value)(
                dest_particles, max_value_index, source_particles );

            if( max_value_index != SIXTRL_NULLPTR )
            {
                max_value_index = max_value_index + NS(PARTICLES_NUM_DATAPTRS);
            }
        }
    }

    return;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE NS(particle_real_t) NS(Particles_get_q0_value)(
    SIXTRL_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const ii )
{
    SIXTRL_ASSERT( ( particles != SIXTRL_NULLPTR ) &&
                   ( ii < NS(Particles_get_num_of_particles)( particles ) ) );

    return particles->q0[ ii ];
}

SIXTRL_INLINE NS(particle_real_const_ptr_t) NS(Particles_get_const_q0)(
    SIXTRL_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT particles )
{
    SIXTRL_ASSERT( particles != SIXTRL_NULLPTR );
    return particles->q0;
}

SIXTRL_INLINE NS(particle_real_ptr_t) NS(Particles_get_q0)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles )
{
    return ( NS(particle_real_ptr_t) )NS(Particles_get_const_q0)( particles );
}

SIXTRL_INLINE void NS(Particles_set_q0)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_real_const_ptr_t) SIXTRL_RESTRICT ptr_to_q0s )
{
    typedef NS(particle_num_elements_t) num_elem_t;

    num_elem_t const num_particles =
        NS(Particles_get_num_of_particles)( particles );

    SIXTRL_ASSERT( ( particles  != SIXTRL_NULLPTR ) &&
                   ( ptr_to_q0s != SIXTRL_NULLPTR ) &&
                   ( num_particles > ( num_elem_t )0u ) );

    SIXTRACKLIB_COPY_VALUES(
        NS(particle_real_t), particles->q0, ptr_to_q0s, num_particles );

    return;
}

SIXTRL_INLINE void NS(Particles_set_q0_value)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const ii,
    NS(particle_real_t) const q0_value )
{
    SIXTRL_ASSERT( ( particles != SIXTRL_NULLPTR ) &&
                   ( ii < NS(Particles_get_num_of_particles)( particles ) ) );

    particles->q0[ ii ] = q0_value;
    return;
}

SIXTRL_INLINE void NS(Particles_assign_ptr_to_q0)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_real_ptr_t) ptr_to_q0s )
{
    SIXTRL_ASSERT( particles != SIXTRL_NULLPTR );
    particles->q0 = ptr_to_q0s;
    return;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE NS(particle_real_t) NS(Particles_get_mass0_value)(
    SIXTRL_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const ii )
{
    SIXTRL_ASSERT( ( particles != SIXTRL_NULLPTR ) &&
                   ( ii < NS(Particles_get_num_of_particles)( particles ) ) );

    return particles->mass0[ ii ];
}

SIXTRL_INLINE NS(particle_real_const_ptr_t) NS(Particles_get_const_mass0)(
    SIXTRL_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT particles )
{
    SIXTRL_ASSERT( particles != SIXTRL_NULLPTR );
    return particles->mass0;
}

SIXTRL_INLINE NS(particle_real_ptr_t) NS(Particles_get_mass0)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles )
{
    return ( NS(particle_real_ptr_t) )NS(Particles_get_const_mass0)( particles );
}

SIXTRL_INLINE void NS(Particles_set_mass0)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_real_const_ptr_t) SIXTRL_RESTRICT ptr_to_mass0s )
{
    typedef NS(particle_num_elements_t) num_elem_t;

    num_elem_t const num_particles =
        NS(Particles_get_num_of_particles)( particles );

    SIXTRL_ASSERT( ( particles  != SIXTRL_NULLPTR ) &&
                   ( ptr_to_mass0s != SIXTRL_NULLPTR ) &&
                   ( num_particles > ( num_elem_t )0u ) );

    SIXTRACKLIB_COPY_VALUES(
        NS(particle_real_t), particles->mass0, ptr_to_mass0s, num_particles );

    return;
}

SIXTRL_INLINE void NS(Particles_set_mass0_value)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const ii,
    NS(particle_real_t) const mass0_value )
{
    SIXTRL_ASSERT( ( particles != SIXTRL_NULLPTR ) &&
                   ( ii < NS(Particles_get_num_of_particles)( particles ) ) );

    particles->mass0[ ii ] = mass0_value;
    return;
}

SIXTRL_INLINE void NS(Particles_assign_ptr_to_mass0)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_real_ptr_t) ptr_to_mass0s )
{
    SIXTRL_ASSERT( particles != SIXTRL_NULLPTR );
    particles->mass0 = ptr_to_mass0s;
    return;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE NS(particle_real_t) NS(Particles_get_beta0_value)(
    SIXTRL_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const ii )
{
    SIXTRL_ASSERT( ( particles != SIXTRL_NULLPTR ) &&
                   ( ii < NS(Particles_get_num_of_particles)( particles ) ) );

    return particles->beta0[ ii ];
}

SIXTRL_INLINE NS(particle_real_const_ptr_t) NS(Particles_get_const_beta0)(
    SIXTRL_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT particles )
{
    SIXTRL_ASSERT( particles != SIXTRL_NULLPTR );
    return particles->beta0;
}

SIXTRL_INLINE NS(particle_real_ptr_t) NS(Particles_get_beta0)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles )
{
    return ( NS(particle_real_ptr_t) )NS(Particles_get_const_beta0)( particles );
}

SIXTRL_INLINE void NS(Particles_set_beta0)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_real_const_ptr_t) SIXTRL_RESTRICT ptr_to_beta0s )
{
    typedef NS(particle_num_elements_t) num_elem_t;

    num_elem_t const num_particles =
        NS(Particles_get_num_of_particles)( particles );

    SIXTRL_ASSERT( ( particles  != SIXTRL_NULLPTR ) &&
                   ( ptr_to_beta0s != SIXTRL_NULLPTR ) &&
                   ( num_particles > ( num_elem_t )0u ) );

    SIXTRACKLIB_COPY_VALUES(
        NS(particle_real_t), particles->beta0, ptr_to_beta0s, num_particles );

    return;
}

SIXTRL_INLINE void NS(Particles_set_beta0_value)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const ii,
    NS(particle_real_t) const beta0_value )
{
    SIXTRL_ASSERT( ( particles != SIXTRL_NULLPTR ) &&
                   ( ii < NS(Particles_get_num_of_particles)( particles ) ) );

    particles->beta0[ ii ] = beta0_value;
    return;
}

SIXTRL_INLINE void NS(Particles_assign_ptr_to_beta0)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_real_ptr_t) ptr_to_beta0s )
{
    SIXTRL_ASSERT( particles != SIXTRL_NULLPTR );
    particles->beta0 = ptr_to_beta0s;
    return;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE NS(particle_real_t) NS(Particles_get_gamma0_value)(
    SIXTRL_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const ii )
{
    SIXTRL_ASSERT( ( particles != SIXTRL_NULLPTR ) &&
                   ( ii < NS(Particles_get_num_of_particles)( particles ) ) );

    return particles->gamma0[ ii ];
}

SIXTRL_INLINE NS(particle_real_const_ptr_t) NS(Particles_get_const_gamma0)(
    SIXTRL_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT particles )
{
    SIXTRL_ASSERT( particles != SIXTRL_NULLPTR );
    return particles->gamma0;
}

SIXTRL_INLINE NS(particle_real_ptr_t) NS(Particles_get_gamma0)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles )
{
    return ( NS(particle_real_ptr_t) )NS(Particles_get_const_gamma0)( particles );
}

SIXTRL_INLINE void NS(Particles_set_gamma0)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_real_const_ptr_t) SIXTRL_RESTRICT ptr_to_gamma0s )
{
    typedef NS(particle_num_elements_t) num_elem_t;

    num_elem_t const num_particles =
        NS(Particles_get_num_of_particles)( particles );

    SIXTRL_ASSERT( ( particles  != SIXTRL_NULLPTR ) &&
                   ( ptr_to_gamma0s != SIXTRL_NULLPTR ) &&
                   ( num_particles > ( num_elem_t )0u ) );

    SIXTRACKLIB_COPY_VALUES(
        NS(particle_real_t), particles->gamma0, ptr_to_gamma0s, num_particles );

    return;
}

SIXTRL_INLINE void NS(Particles_set_gamma0_value)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const ii,
    NS(particle_real_t) const gamma0_value )
{
    SIXTRL_ASSERT( ( particles != SIXTRL_NULLPTR ) &&
                   ( ii < NS(Particles_get_num_of_particles)( particles ) ) );

    particles->gamma0[ ii ] = gamma0_value;
    return;
}

SIXTRL_INLINE void NS(Particles_assign_ptr_to_gamma0)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_real_ptr_t) ptr_to_gamma0s )
{
    SIXTRL_ASSERT( particles != SIXTRL_NULLPTR );
    particles->gamma0 = ptr_to_gamma0s;
    return;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE NS(particle_real_t) NS(Particles_get_p0c_value)(
    SIXTRL_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const ii )
{
    SIXTRL_ASSERT( ( particles != SIXTRL_NULLPTR ) &&
                   ( ii < NS(Particles_get_num_of_particles)( particles ) ) );

    return particles->p0c[ ii ];
}

SIXTRL_INLINE NS(particle_real_const_ptr_t) NS(Particles_get_const_p0c)(
    SIXTRL_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT particles )
{
    SIXTRL_ASSERT( particles != SIXTRL_NULLPTR );
    return particles->p0c;
}

SIXTRL_INLINE NS(particle_real_ptr_t) NS(Particles_get_p0c)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles )
{
    return ( NS(particle_real_ptr_t) )NS(Particles_get_const_p0c)( particles );
}

SIXTRL_INLINE void NS(Particles_set_p0c)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_real_const_ptr_t) SIXTRL_RESTRICT ptr_to_p0cs )
{
    typedef NS(particle_num_elements_t) num_elem_t;

    num_elem_t const num_particles =
        NS(Particles_get_num_of_particles)( particles );

    SIXTRL_ASSERT( ( particles  != SIXTRL_NULLPTR ) &&
                   ( ptr_to_p0cs != SIXTRL_NULLPTR ) &&
                   ( num_particles > ( num_elem_t )0u ) );

    SIXTRACKLIB_COPY_VALUES(
        NS(particle_real_t), particles->p0c, ptr_to_p0cs, num_particles );

    return;
}

SIXTRL_INLINE void NS(Particles_set_p0c_value)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const ii,
    NS(particle_real_t) const p0c_value )
{
    SIXTRL_ASSERT( ( particles != SIXTRL_NULLPTR ) &&
                   ( ii < NS(Particles_get_num_of_particles)( particles ) ) );

    particles->p0c[ ii ] = p0c_value;
    return;
}

SIXTRL_INLINE void NS(Particles_assign_ptr_to_p0c)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_real_ptr_t) ptr_to_p0cs )
{
    SIXTRL_ASSERT( particles != SIXTRL_NULLPTR );
    particles->p0c = ptr_to_p0cs;
    return;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE NS(particle_real_t) NS(Particles_get_s_value)(
    SIXTRL_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const ii )
{
    SIXTRL_ASSERT( ( particles != SIXTRL_NULLPTR ) &&
                   ( ii < NS(Particles_get_num_of_particles)( particles ) ) );

    return particles->s[ ii ];
}

SIXTRL_INLINE NS(particle_real_const_ptr_t) NS(Particles_get_const_s)(
    SIXTRL_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT particles )
{
    SIXTRL_ASSERT( particles != SIXTRL_NULLPTR );
    return particles->s;
}

SIXTRL_INLINE NS(particle_real_ptr_t) NS(Particles_get_s)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles )
{
    return ( NS(particle_real_ptr_t) )NS(Particles_get_const_s)( particles );
}

SIXTRL_INLINE void NS(Particles_set_s)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_real_const_ptr_t) SIXTRL_RESTRICT ptr_to_ss )
{
    typedef NS(particle_num_elements_t) num_elem_t;

    num_elem_t const num_particles =
        NS(Particles_get_num_of_particles)( particles );

    SIXTRL_ASSERT( ( particles  != SIXTRL_NULLPTR ) &&
                   ( ptr_to_ss != SIXTRL_NULLPTR ) &&
                   ( num_particles > ( num_elem_t )0u ) );

    SIXTRACKLIB_COPY_VALUES(
        NS(particle_real_t), particles->s, ptr_to_ss, num_particles );

    return;
}

SIXTRL_INLINE void NS(Particles_set_s_value)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const ii,
    NS(particle_real_t) const s_value )
{
    SIXTRL_ASSERT( ( particles != SIXTRL_NULLPTR ) &&
                   ( ii < NS(Particles_get_num_of_particles)( particles ) ) );

    particles->s[ ii ] = s_value;
    return;
}

SIXTRL_INLINE void NS(Particles_assign_ptr_to_s)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_real_ptr_t) ptr_to_ss )
{
    SIXTRL_ASSERT( particles != SIXTRL_NULLPTR );
    particles->s = ptr_to_ss;
    return;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE NS(particle_real_t) NS(Particles_get_x_value)(
    SIXTRL_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const ii )
{
    SIXTRL_ASSERT( ( particles != SIXTRL_NULLPTR ) &&
                   ( ii < NS(Particles_get_num_of_particles)( particles ) ) );

    return particles->x[ ii ];
}

SIXTRL_INLINE NS(particle_real_const_ptr_t) NS(Particles_get_const_x)(
    SIXTRL_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT particles )
{
    SIXTRL_ASSERT( particles != SIXTRL_NULLPTR );
    return particles->x;
}

SIXTRL_INLINE NS(particle_real_ptr_t) NS(Particles_get_x)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles )
{
    return ( NS(particle_real_ptr_t) )NS(Particles_get_const_x)( particles );
}

SIXTRL_INLINE void NS(Particles_set_x)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_real_const_ptr_t) SIXTRL_RESTRICT ptr_to_xs )
{
    typedef NS(particle_num_elements_t) num_elem_t;

    num_elem_t const num_particles =
        NS(Particles_get_num_of_particles)( particles );

    SIXTRL_ASSERT( ( particles  != SIXTRL_NULLPTR ) &&
                   ( ptr_to_xs != SIXTRL_NULLPTR ) &&
                   ( num_particles > ( num_elem_t )0u ) );

    SIXTRACKLIB_COPY_VALUES(
        NS(particle_real_t), particles->x, ptr_to_xs, num_particles );

    return;
}

SIXTRL_INLINE void NS(Particles_set_x_value)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const ii,
    NS(particle_real_t) const x_value )
{
    SIXTRL_ASSERT( ( particles != SIXTRL_NULLPTR ) &&
                   ( ii < NS(Particles_get_num_of_particles)( particles ) ) );

    particles->x[ ii ] = x_value;
    return;
}

SIXTRL_INLINE void NS(Particles_assign_ptr_to_x)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_real_ptr_t) ptr_to_xs )
{
    SIXTRL_ASSERT( particles != SIXTRL_NULLPTR );
    particles->x = ptr_to_xs;
    return;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE NS(particle_real_t) NS(Particles_get_y_value)(
    SIXTRL_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const ii )
{
    SIXTRL_ASSERT( ( particles != SIXTRL_NULLPTR ) &&
                   ( ii < NS(Particles_get_num_of_particles)( particles ) ) );

    return particles->y[ ii ];
}

SIXTRL_INLINE NS(particle_real_const_ptr_t) NS(Particles_get_const_y)(
    SIXTRL_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT particles )
{
    SIXTRL_ASSERT( particles != SIXTRL_NULLPTR );
    return particles->y;
}

SIXTRL_INLINE NS(particle_real_ptr_t) NS(Particles_get_y)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles )
{
    return ( NS(particle_real_ptr_t) )NS(Particles_get_const_y)( particles );
}

SIXTRL_INLINE void NS(Particles_set_y)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_real_const_ptr_t) SIXTRL_RESTRICT ptr_to_ys )
{
    typedef NS(particle_num_elements_t) num_elem_t;

    num_elem_t const num_particles =
        NS(Particles_get_num_of_particles)( particles );

    SIXTRL_ASSERT( ( particles  != SIXTRL_NULLPTR ) &&
                   ( ptr_to_ys != SIXTRL_NULLPTR ) &&
                   ( num_particles > ( num_elem_t )0u ) );

    SIXTRACKLIB_COPY_VALUES(
        NS(particle_real_t), particles->y, ptr_to_ys, num_particles );

    return;
}

SIXTRL_INLINE void NS(Particles_set_y_value)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const ii,
    NS(particle_real_t) const y_value )
{
    SIXTRL_ASSERT( ( particles != SIXTRL_NULLPTR ) &&
                   ( ii < NS(Particles_get_num_of_particles)( particles ) ) );

    particles->y[ ii ] = y_value;
    return;
}

SIXTRL_INLINE void NS(Particles_assign_ptr_to_y)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_real_ptr_t) ptr_to_ys )
{
    SIXTRL_ASSERT( particles != SIXTRL_NULLPTR );
    particles->y = ptr_to_ys;
    return;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE NS(particle_real_t) NS(Particles_get_px_value)(
    SIXTRL_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const ii )
{
    SIXTRL_ASSERT( ( particles != SIXTRL_NULLPTR ) &&
                   ( ii < NS(Particles_get_num_of_particles)( particles ) ) );

    return particles->px[ ii ];
}

SIXTRL_INLINE NS(particle_real_const_ptr_t) NS(Particles_get_const_px)(
    SIXTRL_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT particles )
{
    SIXTRL_ASSERT( particles != SIXTRL_NULLPTR );
    return particles->px;
}

SIXTRL_INLINE NS(particle_real_ptr_t) NS(Particles_get_px)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles )
{
    return ( NS(particle_real_ptr_t) )NS(Particles_get_const_px)( particles );
}

SIXTRL_INLINE void NS(Particles_set_px)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_real_const_ptr_t) SIXTRL_RESTRICT ptr_to_pxs )
{
    typedef NS(particle_num_elements_t) num_elem_t;

    num_elem_t const num_particles =
        NS(Particles_get_num_of_particles)( particles );

    SIXTRL_ASSERT( ( particles  != SIXTRL_NULLPTR ) &&
                   ( ptr_to_pxs != SIXTRL_NULLPTR ) &&
                   ( num_particles > ( num_elem_t )0u ) );

    SIXTRACKLIB_COPY_VALUES(
        NS(particle_real_t), particles->px, ptr_to_pxs, num_particles );

    return;
}

SIXTRL_INLINE void NS(Particles_set_px_value)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const ii,
    NS(particle_real_t) const px_value )
{
    SIXTRL_ASSERT( ( particles != SIXTRL_NULLPTR ) &&
                   ( ii < NS(Particles_get_num_of_particles)( particles ) ) );

    particles->px[ ii ] = px_value;
    return;
}

SIXTRL_INLINE void NS(Particles_assign_ptr_to_px)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_real_ptr_t) ptr_to_pxs )
{
    SIXTRL_ASSERT( particles != SIXTRL_NULLPTR );
    particles->px = ptr_to_pxs;
    return;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE NS(particle_real_t) NS(Particles_get_py_value)(
    SIXTRL_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const ii )
{
    SIXTRL_ASSERT( ( particles != SIXTRL_NULLPTR ) &&
                   ( ii < NS(Particles_get_num_of_particles)( particles ) ) );

    return particles->py[ ii ];
}

SIXTRL_INLINE NS(particle_real_const_ptr_t) NS(Particles_get_const_py)(
    SIXTRL_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT particles )
{
    SIXTRL_ASSERT( particles != SIXTRL_NULLPTR );
    return particles->py;
}

SIXTRL_INLINE NS(particle_real_ptr_t) NS(Particles_get_py)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles )
{
    return ( NS(particle_real_ptr_t) )NS(Particles_get_const_py)( particles );
}

SIXTRL_INLINE void NS(Particles_set_py)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_real_const_ptr_t) SIXTRL_RESTRICT ptr_to_pys )
{
    typedef NS(particle_num_elements_t) num_elem_t;

    num_elem_t const num_particles =
        NS(Particles_get_num_of_particles)( particles );

    SIXTRL_ASSERT( ( particles  != SIXTRL_NULLPTR ) &&
                   ( ptr_to_pys != SIXTRL_NULLPTR ) &&
                   ( num_particles > ( num_elem_t )0u ) );

    SIXTRACKLIB_COPY_VALUES(
        NS(particle_real_t), particles->py, ptr_to_pys, num_particles );

    return;
}

SIXTRL_INLINE void NS(Particles_set_py_value)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const ii,
    NS(particle_real_t) const py_value )
{
    SIXTRL_ASSERT( ( particles != SIXTRL_NULLPTR ) &&
                   ( ii < NS(Particles_get_num_of_particles)( particles ) ) );

    particles->py[ ii ] = py_value;
    return;
}

SIXTRL_INLINE void NS(Particles_assign_ptr_to_py)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_real_ptr_t) ptr_to_pys )
{
    SIXTRL_ASSERT( particles != SIXTRL_NULLPTR );
    particles->py = ptr_to_pys;
    return;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE NS(particle_real_t) NS(Particles_get_zeta_value)(
    SIXTRL_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const ii )
{
    SIXTRL_ASSERT( ( particles != SIXTRL_NULLPTR ) &&
                   ( ii < NS(Particles_get_num_of_particles)( particles ) ) );

    return particles->zeta[ ii ];
}

SIXTRL_INLINE NS(particle_real_const_ptr_t) NS(Particles_get_const_zeta)(
    SIXTRL_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT particles )
{
    SIXTRL_ASSERT( particles != SIXTRL_NULLPTR );
    return particles->zeta;
}

SIXTRL_INLINE NS(particle_real_ptr_t) NS(Particles_get_zeta)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles )
{
    return ( NS(particle_real_ptr_t) )NS(Particles_get_const_zeta)( particles );
}

SIXTRL_INLINE void NS(Particles_set_zeta)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_real_const_ptr_t) SIXTRL_RESTRICT ptr_to_zetas )
{
    typedef NS(particle_num_elements_t) num_elem_t;

    num_elem_t const num_particles =
        NS(Particles_get_num_of_particles)( particles );

    SIXTRL_ASSERT( ( particles  != SIXTRL_NULLPTR ) &&
                   ( ptr_to_zetas != SIXTRL_NULLPTR ) &&
                   ( num_particles > ( num_elem_t )0u ) );

    SIXTRACKLIB_COPY_VALUES(
        NS(particle_real_t), particles->zeta, ptr_to_zetas, num_particles );

    return;
}

SIXTRL_INLINE void NS(Particles_set_zeta_value)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const ii,
    NS(particle_real_t) const zeta_value )
{
    SIXTRL_ASSERT( ( particles != SIXTRL_NULLPTR ) &&
                   ( ii < NS(Particles_get_num_of_particles)( particles ) ) );

    particles->zeta[ ii ] = zeta_value;
    return;
}

SIXTRL_INLINE void NS(Particles_assign_ptr_to_zeta)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_real_ptr_t) ptr_to_zetas )
{
    SIXTRL_ASSERT( particles != SIXTRL_NULLPTR );
    particles->zeta = ptr_to_zetas;
    return;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE NS(particle_real_t) NS(Particles_get_psigma_value)(
    SIXTRL_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const ii )
{
    SIXTRL_ASSERT( ( particles != SIXTRL_NULLPTR ) &&
                   ( ii < NS(Particles_get_num_of_particles)( particles ) ) );

    return particles->psigma[ ii ];
}

SIXTRL_INLINE NS(particle_real_const_ptr_t) NS(Particles_get_const_psigma)(
    SIXTRL_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT particles )
{
    SIXTRL_ASSERT( particles != SIXTRL_NULLPTR );
    return particles->psigma;
}

SIXTRL_INLINE NS(particle_real_ptr_t) NS(Particles_get_psigma)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles )
{
    return ( NS(particle_real_ptr_t) )NS(Particles_get_const_psigma)( particles );
}

SIXTRL_INLINE void NS(Particles_set_psigma)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_real_const_ptr_t) SIXTRL_RESTRICT ptr_to_psigmas )
{
    typedef NS(particle_num_elements_t) num_elem_t;

    num_elem_t const num_particles =
        NS(Particles_get_num_of_particles)( particles );

    SIXTRL_ASSERT( ( particles  != SIXTRL_NULLPTR ) &&
                   ( ptr_to_psigmas != SIXTRL_NULLPTR ) &&
                   ( num_particles > ( num_elem_t )0u ) );

    SIXTRACKLIB_COPY_VALUES(
        NS(particle_real_t), particles->psigma, ptr_to_psigmas, num_particles );

    return;
}

SIXTRL_INLINE void NS(Particles_set_psigma_value)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const ii,
    NS(particle_real_t) const psigma_value )
{
    SIXTRL_ASSERT( ( particles != SIXTRL_NULLPTR ) &&
                   ( ii < NS(Particles_get_num_of_particles)( particles ) ) );

    particles->psigma[ ii ] = psigma_value;
    return;
}

SIXTRL_INLINE void NS(Particles_assign_ptr_to_psigma)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_real_ptr_t) ptr_to_psigmas )
{
    SIXTRL_ASSERT( particles != SIXTRL_NULLPTR );
    particles->psigma = ptr_to_psigmas;
    return;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE NS(particle_real_t) NS(Particles_get_delta_value)(
    SIXTRL_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const ii )
{
    SIXTRL_ASSERT( ( particles != SIXTRL_NULLPTR ) &&
                   ( ii < NS(Particles_get_num_of_particles)( particles ) ) );

    return particles->delta[ ii ];
}

SIXTRL_INLINE NS(particle_real_const_ptr_t) NS(Particles_get_const_delta)(
    SIXTRL_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT particles )
{
    SIXTRL_ASSERT( particles != SIXTRL_NULLPTR );
    return particles->delta;
}

SIXTRL_INLINE NS(particle_real_ptr_t) NS(Particles_get_delta)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles )
{
    return ( NS(particle_real_ptr_t) )NS(Particles_get_const_delta)( particles );
}

SIXTRL_INLINE void NS(Particles_set_delta)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_real_const_ptr_t) SIXTRL_RESTRICT ptr_to_deltas )
{
    typedef NS(particle_num_elements_t) num_elem_t;

    num_elem_t const num_particles =
        NS(Particles_get_num_of_particles)( particles );

    SIXTRL_ASSERT( ( particles  != SIXTRL_NULLPTR ) &&
                   ( ptr_to_deltas != SIXTRL_NULLPTR ) &&
                   ( num_particles > ( num_elem_t )0u ) );

    SIXTRACKLIB_COPY_VALUES(
        NS(particle_real_t), particles->delta, ptr_to_deltas, num_particles );

    return;
}

SIXTRL_INLINE void NS(Particles_set_delta_value)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const ii,
    NS(particle_real_t) const delta_value )
{
    SIXTRL_ASSERT( ( particles != SIXTRL_NULLPTR ) &&
                   ( ii < NS(Particles_get_num_of_particles)( particles ) ) );

    particles->delta[ ii ] = delta_value;
    return;
}

SIXTRL_INLINE void NS(Particles_assign_ptr_to_delta)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_real_ptr_t) ptr_to_deltas )
{
    SIXTRL_ASSERT( particles != SIXTRL_NULLPTR );
    particles->delta = ptr_to_deltas;
    return;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE NS(particle_real_t) NS(Particles_get_rpp_value)(
    SIXTRL_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const ii )
{
    SIXTRL_ASSERT( ( particles != SIXTRL_NULLPTR ) &&
                   ( ii < NS(Particles_get_num_of_particles)( particles ) ) );

    return particles->rpp[ ii ];
}

SIXTRL_INLINE NS(particle_real_const_ptr_t) NS(Particles_get_const_rpp)(
    SIXTRL_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT particles )
{
    SIXTRL_ASSERT( particles != SIXTRL_NULLPTR );
    return particles->rpp;
}

SIXTRL_INLINE NS(particle_real_ptr_t) NS(Particles_get_rpp)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles )
{
    return ( NS(particle_real_ptr_t) )NS(Particles_get_const_rpp)( particles );
}

SIXTRL_INLINE void NS(Particles_set_rpp)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_real_const_ptr_t) SIXTRL_RESTRICT ptr_to_rpps )
{
    typedef NS(particle_num_elements_t) num_elem_t;

    num_elem_t const num_particles =
        NS(Particles_get_num_of_particles)( particles );

    SIXTRL_ASSERT( ( particles  != SIXTRL_NULLPTR ) &&
                   ( ptr_to_rpps != SIXTRL_NULLPTR ) &&
                   ( num_particles > ( num_elem_t )0u ) );

    SIXTRACKLIB_COPY_VALUES(
        NS(particle_real_t), particles->rpp, ptr_to_rpps, num_particles );

    return;
}

SIXTRL_INLINE void NS(Particles_set_rpp_value)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const ii,
    NS(particle_real_t) const rpp_value )
{
    SIXTRL_ASSERT( ( particles != SIXTRL_NULLPTR ) &&
                   ( ii < NS(Particles_get_num_of_particles)( particles ) ) );

    particles->rpp[ ii ] = rpp_value;
    return;
}

SIXTRL_INLINE void NS(Particles_assign_ptr_to_rpp)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_real_ptr_t) ptr_to_rpps )
{
    SIXTRL_ASSERT( particles != SIXTRL_NULLPTR );
    particles->rpp = ptr_to_rpps;
    return;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE NS(particle_real_t) NS(Particles_get_rvv_value)(
    SIXTRL_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const ii )
{
    SIXTRL_ASSERT( ( particles != SIXTRL_NULLPTR ) &&
                   ( ii < NS(Particles_get_num_of_particles)( particles ) ) );

    return particles->rvv[ ii ];
}

SIXTRL_INLINE NS(particle_real_const_ptr_t) NS(Particles_get_const_rvv)(
    SIXTRL_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT particles )
{
    SIXTRL_ASSERT( particles != SIXTRL_NULLPTR );
    return particles->rvv;
}

SIXTRL_INLINE NS(particle_real_ptr_t) NS(Particles_get_rvv)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles )
{
    return ( NS(particle_real_ptr_t) )NS(Particles_get_const_rvv)( particles );
}

SIXTRL_INLINE void NS(Particles_set_rvv)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_real_const_ptr_t) SIXTRL_RESTRICT ptr_to_rvvs )
{
    typedef NS(particle_num_elements_t) num_elem_t;

    num_elem_t const num_particles =
        NS(Particles_get_num_of_particles)( particles );

    SIXTRL_ASSERT( ( particles  != SIXTRL_NULLPTR ) &&
                   ( ptr_to_rvvs != SIXTRL_NULLPTR ) &&
                   ( num_particles > ( num_elem_t )0u ) );

    SIXTRACKLIB_COPY_VALUES(
        NS(particle_real_t), particles->rvv, ptr_to_rvvs, num_particles );

    return;
}

SIXTRL_INLINE void NS(Particles_set_rvv_value)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const ii,
    NS(particle_real_t) const rvv_value )
{
    SIXTRL_ASSERT( ( particles != SIXTRL_NULLPTR ) &&
                   ( ii < NS(Particles_get_num_of_particles)( particles ) ) );

    particles->rvv[ ii ] = rvv_value;
    return;
}

SIXTRL_INLINE void NS(Particles_assign_ptr_to_rvv)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_real_ptr_t) ptr_to_rvvs )
{
    SIXTRL_ASSERT( particles != SIXTRL_NULLPTR );
    particles->rvv = ptr_to_rvvs;
    return;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE NS(particle_real_t) NS(Particles_get_chi_value)(
    SIXTRL_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const ii )
{
    SIXTRL_ASSERT( ( particles != SIXTRL_NULLPTR ) &&
                   ( ii < NS(Particles_get_num_of_particles)( particles ) ) );

    return particles->chi[ ii ];
}

SIXTRL_INLINE NS(particle_real_const_ptr_t) NS(Particles_get_const_chi)(
    SIXTRL_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT particles )
{
    SIXTRL_ASSERT( particles != SIXTRL_NULLPTR );
    return particles->chi;
}

SIXTRL_INLINE NS(particle_real_ptr_t) NS(Particles_get_chi)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles )
{
    return ( NS(particle_real_ptr_t) )NS(Particles_get_const_chi)( particles );
}

SIXTRL_INLINE void NS(Particles_set_chi)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_real_const_ptr_t) SIXTRL_RESTRICT ptr_to_chis )
{
    typedef NS(particle_num_elements_t) num_elem_t;

    num_elem_t const num_particles =
        NS(Particles_get_num_of_particles)( particles );

    SIXTRL_ASSERT( ( particles  != SIXTRL_NULLPTR ) &&
                   ( ptr_to_chis != SIXTRL_NULLPTR ) &&
                   ( num_particles > ( num_elem_t )0u ) );

    SIXTRACKLIB_COPY_VALUES(
        NS(particle_real_t), particles->chi, ptr_to_chis, num_particles );

    return;
}

SIXTRL_INLINE void NS(Particles_set_chi_value)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const ii,
    NS(particle_real_t) const chi_value )
{
    SIXTRL_ASSERT( ( particles != SIXTRL_NULLPTR ) &&
                   ( ii < NS(Particles_get_num_of_particles)( particles ) ) );

    particles->chi[ ii ] = chi_value;
    return;
}

SIXTRL_INLINE void NS(Particles_assign_ptr_to_chi)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_real_ptr_t) ptr_to_chis )
{
    SIXTRL_ASSERT( particles != SIXTRL_NULLPTR );
    particles->chi = ptr_to_chis;
    return;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE NS(particle_index_t) NS(Particles_get_particle_id_value)(
    SIXTRL_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const ii )
{
    SIXTRL_ASSERT( ( particles != SIXTRL_NULLPTR ) &&
                   ( ii < NS(Particles_get_num_of_particles)( particles ) ) );

    return particles->particle_id[ ii ];
}

SIXTRL_INLINE NS(particle_index_const_ptr_t)
NS(Particles_get_const_particle_id)(
    SIXTRL_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT particles )
{
    SIXTRL_ASSERT( particles != SIXTRL_NULLPTR );
    return particles->particle_id;
}

SIXTRL_INLINE NS(particle_index_ptr_t)
NS(Particles_get_particle_id)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT p )
{
    return ( NS(particle_index_ptr_t) )NS(Particles_get_particle_id)( p );
}

SIXTRL_INLINE void NS(Particles_set_particle_id)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_index_const_ptr_t) SIXTRL_RESTRICT ptr_to_particle_ids )
{
    typedef NS(particle_num_elements_t) num_elem_t;

    num_elem_t const num_particles =
        NS(Particles_get_num_of_particles)( particles );

    SIXTRL_ASSERT( ( particles  != SIXTRL_NULLPTR ) &&
                   ( ptr_to_particle_ids != SIXTRL_NULLPTR ) &&
                   ( num_particles > ( num_elem_t )0u ) );

    SIXTRACKLIB_COPY_VALUES( NS(particle_real_t), particles->particle_id,
                             ptr_to_particle_ids, num_particles );

    return;
}

SIXTRL_INLINE void NS(Particles_set_particle_id_value)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const ii,
    NS(particle_index_t) const particle_id_value )
{
    SIXTRL_ASSERT( ( particles != SIXTRL_NULLPTR ) &&
                   ( ii < NS(Particles_get_num_of_particles)( particles ) ) );

    particles->particle_id[ ii ] = particle_id_value;
    return;
}

SIXTRL_INLINE void NS(Particles_assign_ptr_to_particle_id)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_index_ptr_t) ptr_to_particle_ids )
{
    SIXTRL_ASSERT( particles != SIXTRL_NULLPTR );
    particles->particle_id = ptr_to_particle_ids;
    return;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE NS(particle_index_t) NS(Particles_get_at_element_id_value)(
    SIXTRL_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const ii )
{
    SIXTRL_ASSERT( ( particles != SIXTRL_NULLPTR ) &&
                   ( ii < NS(Particles_get_num_of_particles)( particles ) ) );

    return particles->at_element_id[ ii ];
}

SIXTRL_INLINE NS(particle_index_const_ptr_t)
NS(Particles_get_const_at_element_id)(
    SIXTRL_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT particles )
{
    SIXTRL_ASSERT( particles != SIXTRL_NULLPTR );
    return particles->at_element_id;
}

SIXTRL_INLINE NS(particle_index_ptr_t)
NS(Particles_get_at_element_id)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT p )
{
    return ( NS(particle_index_ptr_t) )NS(Particles_get_at_element_id)( p );
}

SIXTRL_INLINE void NS(Particles_set_at_element_id)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_index_const_ptr_t) SIXTRL_RESTRICT ptr_to_at_element_ids )
{
    typedef NS(particle_num_elements_t) num_elem_t;

    num_elem_t const num_particles =
        NS(Particles_get_num_of_particles)( particles );

    SIXTRL_ASSERT( ( particles  != SIXTRL_NULLPTR ) &&
                   ( ptr_to_at_element_ids != SIXTRL_NULLPTR ) &&
                   ( num_particles > ( num_elem_t )0u ) );

    SIXTRACKLIB_COPY_VALUES( NS(particle_real_t), particles->at_element_id,
                             ptr_to_at_element_ids, num_particles );

    return;
}

SIXTRL_INLINE void NS(Particles_set_at_element_id_value)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const ii,
    NS(particle_index_t) const at_element_id_value )
{
    SIXTRL_ASSERT( ( particles != SIXTRL_NULLPTR ) &&
                   ( ii < NS(Particles_get_num_of_particles)( particles ) ) );

    particles->at_element_id[ ii ] = at_element_id_value;
    return;
}

SIXTRL_INLINE void NS(Particles_assign_ptr_to_at_element_id)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_index_ptr_t) ptr_to_at_element_ids )
{
    SIXTRL_ASSERT( particles != SIXTRL_NULLPTR );
    particles->at_element_id = ptr_to_at_element_ids;
    return;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE NS(particle_index_t) NS(Particles_get_at_turn_value)(
    SIXTRL_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const ii )
{
    SIXTRL_ASSERT( ( particles != SIXTRL_NULLPTR ) &&
                   ( ii < NS(Particles_get_num_of_particles)( particles ) ) );

    return particles->at_turn[ ii ];
}

SIXTRL_INLINE NS(particle_index_const_ptr_t)
NS(Particles_get_const_at_turn)(
    SIXTRL_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT particles )
{
    SIXTRL_ASSERT( particles != SIXTRL_NULLPTR );
    return particles->at_turn;
}

SIXTRL_INLINE NS(particle_index_ptr_t)
NS(Particles_get_at_turn)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT p )
{
    return ( NS(particle_index_ptr_t) )NS(Particles_get_at_turn)( p );
}

SIXTRL_INLINE void NS(Particles_set_at_turn)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_index_const_ptr_t) SIXTRL_RESTRICT ptr_to_at_turns )
{
    typedef NS(particle_num_elements_t) num_elem_t;

    num_elem_t const num_particles =
        NS(Particles_get_num_of_particles)( particles );

    SIXTRL_ASSERT( ( particles  != SIXTRL_NULLPTR ) &&
                   ( ptr_to_at_turns != SIXTRL_NULLPTR ) &&
                   ( num_particles > ( num_elem_t )0u ) );

    SIXTRACKLIB_COPY_VALUES( NS(particle_real_t), particles->at_turn,
                             ptr_to_at_turns, num_particles );

    return;
}

SIXTRL_INLINE void NS(Particles_set_at_turn_value)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const ii,
    NS(particle_index_t) const at_turn_value )
{
    SIXTRL_ASSERT( ( particles != SIXTRL_NULLPTR ) &&
                   ( ii < NS(Particles_get_num_of_particles)( particles ) ) );

    particles->at_turn[ ii ] = at_turn_value;
    return;
}

SIXTRL_INLINE void NS(Particles_assign_ptr_to_at_turn)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_index_ptr_t) ptr_to_at_turns )
{
    SIXTRL_ASSERT( particles != SIXTRL_NULLPTR );
    particles->at_turn = ptr_to_at_turns;
    return;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE NS(particle_index_t) NS(Particles_get_state_value)(
    SIXTRL_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const ii )
{
    SIXTRL_ASSERT( ( particles != SIXTRL_NULLPTR ) &&
                   ( ii < NS(Particles_get_num_of_particles)( particles ) ) );

    return particles->state[ ii ];
}

SIXTRL_INLINE NS(particle_index_const_ptr_t)
NS(Particles_get_const_state)(
    SIXTRL_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT particles )
{
    SIXTRL_ASSERT( particles != SIXTRL_NULLPTR );
    return particles->state;
}

SIXTRL_INLINE NS(particle_index_ptr_t)
NS(Particles_get_state)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles )
{
    return ( NS(particle_index_ptr_t) )NS(Particles_get_state)( particles );
}

SIXTRL_INLINE void NS(Particles_set_state)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_index_const_ptr_t) SIXTRL_RESTRICT ptr_to_states )
{
    typedef NS(particle_num_elements_t) num_elem_t;

    num_elem_t const num_particles =
        NS(Particles_get_num_of_particles)( particles );

    SIXTRL_ASSERT( ( particles  != SIXTRL_NULLPTR ) &&
                   ( ptr_to_states != SIXTRL_NULLPTR ) &&
                   ( num_particles > ( num_elem_t )0u ) );

    SIXTRACKLIB_COPY_VALUES( NS(particle_real_t), particles->state,
                             ptr_to_states, num_particles );

    return;
}

SIXTRL_INLINE void NS(Particles_set_state_value)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const ii,
    NS(particle_index_t) const state_value )
{
    SIXTRL_ASSERT( ( particles != SIXTRL_NULLPTR ) &&
                   ( ii < NS(Particles_get_num_of_particles)( particles ) ) );

    particles->state[ ii ] = state_value;
    return;
}

SIXTRL_INLINE void NS(Particles_assign_ptr_to_state)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_index_ptr_t) ptr_to_states )
{
    SIXTRL_ASSERT( particles != SIXTRL_NULLPTR );
    particles->state = ptr_to_states;
    return;
}

/* ------------------------------------------------------------------------- */

#if !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

#endif /* SIXTRACKLIB_COMMON_PARTICLES_H__ */

/* end: sixtracklib/common/particles.h */
