#ifndef SIXTRACKLIB_COMMON_TESTS_TEST_PARTICLES_TOOLS_H__
#define SIXTRACKLIB_COMMON_TESTS_TEST_PARTICLES_TOOLS_H__

#if defined( __cplusplus )
extern "C" {
#endif /* defined( __cplusplus ) */

#include "sixtracklib/_impl/definitions.h"
#include "sixtracklib/common/blocks.h"
#include "sixtracklib/common/particles.h"
    
void NS(Particles_random_init)( NS(Particles)* SIXTRL_RESTRICT p );

/* ------------------------------------------------------------------------- */

int NS(Particles_have_same_structure)(
    const NS(Particles) *const SIXTRL_RESTRICT lhs, 
    const NS(Particles) *const SIXTRL_RESTRICT rhs );

int NS(Particles_map_to_same_memory)(
    const NS(Particles) *const SIXTRL_RESTRICT lhs, 
    const NS(Particles) *const SIXTRL_RESTRICT rhs );

int NS(Particles_compare_values)(
    const NS(Particles) *const SIXTRL_RESTRICT lhs, 
    const NS(Particles) *const SIXTRL_RESTRICT rhs );

/* ------------------------------------------------------------------------- */

int NS(Particles_buffers_have_same_structure)(
    const NS(Blocks) *const SIXTRL_RESTRICT lhs_buffer,
    const NS(Blocks) *const SIXTRL_RESTRICT rhs_buffer );

int NS(Particles_buffers_map_to_same_memory)(
    const NS(Blocks) *const SIXTRL_RESTRICT lhs_buffer,
    const NS(Blocks) *const SIXTRL_RESTRICT rhs_buffer );

int NS(Particles_buffer_compare_values)(
    const NS(Blocks) *const SIXTRL_RESTRICT lhs_buffer,
    const NS(Blocks) *const SIXTRL_RESTRICT rhs_buffer );

/* ------------------------------------------------------------------------- */

#if defined( __cplusplus )
}
#endif /* defined( __cplusplus ) */

#endif /* SIXTRACKLIB_COMMON_TESTS_TEST_PARTICLES_TOOLS_H__ */

/* end: sixtracklib/common/tests/test_particles_tools.h */
