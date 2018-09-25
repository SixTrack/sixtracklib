#ifndef SIXTRACKLIB_TESTS_SIXTRACKLIB_TESTLIB_TEST_PARTICLES_TOOLS_H__
#define SIXTRACKLIB_TESTS_SIXTRACKLIB_TESTLIB_TEST_PARTICLES_TOOLS_H__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stdio.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/_impl/definitions.h"
    #include "sixtracklib/common/buffer.h"
    #include "sixtracklib/common/particles.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

void NS(Particles_random_init)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT p );

void NS(Particles_realistic_init)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT p );

/* ------------------------------------------------------------------------- */

int NS(Particles_have_same_structure)(
    SIXTRL_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT lhs,
    SIXTRL_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT rhs );

int NS(Particles_map_to_same_memory)(
    SIXTRL_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT lhs,
    SIXTRL_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT rhs );

int NS(Particles_compare_values)(
    SIXTRL_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT lhs,
    SIXTRL_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT rhs );

int NS(Particles_compare_values_with_treshold)(
    SIXTRL_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT lhs,
    SIXTRL_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT rhs,
    NS(particle_real_t) const treshold );

void NS(Particles_get_max_difference)(
    SIXTRL_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT max_diff,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT max_diff_indices,
    SIXTRL_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT lhs,
    SIXTRL_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT rhs );

void NS(Particles_print)(
    FILE* SIXTRL_RESTRICT fp,
    SIXTRL_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT particles );

void NS(Particles_print_max_diff)(
    FILE* SIXTRL_RESTRICT fp,
    SIXTRL_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT max_diff,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t) const* max_diff_indices );

/* ------------------------------------------------------------------------- */

int NS(Particles_buffers_map_to_same_memory)(
    SIXTRL_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT lhs_buffer,
    SIXTRL_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT rhs_buffer );

int NS(Particles_buffers_compare_values)(
    SIXTRL_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT lhs_buffer,
    SIXTRL_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT rhs_buffer );

int NS(Particles_buffers_compare_values_with_treshold)(
    SIXTRL_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT lhs_buffer,
    SIXTRL_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT rhs_buffer,
    NS(particle_real_t) const treshold );

void NS(Particles_buffers_get_max_difference)(
    SIXTRL_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT max_diff,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT max_diff_indices,
    SIXTRL_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT lhs,
    SIXTRL_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT rhs );

void NS(Particles_buffer_print)(
    FILE* SIXTRL_RESTRICT fp,
    SIXTRL_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT particles );

void NS(Particles_buffer_print_max_diff)(
    FILE* SIXTRL_RESTRICT fp,
    SIXTRL_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT max_diff,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t) const* max_diff_indices );

/* ------------------------------------------------------------------------- */

#if !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

#endif /* SIXTRACKLIB_TESTS_SIXTRACKLIB_TESTLIB_TEST_PARTICLES_TOOLS_H__ */

/* end: tests/sixtracklib/testlib/test_particles_tools.h */
