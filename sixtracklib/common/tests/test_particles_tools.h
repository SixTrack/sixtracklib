#ifndef SIXTRACKLIB_COMMON_TESTS_TEST_PARTICLES_TOOLS_H__
#define SIXTRACKLIB_COMMON_TESTS_TEST_PARTICLES_TOOLS_H__

#if defined( __cplusplus )
extern "C" {
#endif /* defined( __cplusplus ) */

#include "sixtracklib/_impl/definitions.h"
#include "sixtracklib/common/blocks.h"
#include "sixtracklib/common/particles.h"
    
void NS(Particles_random_init)( NS(Particles)* SIXTRL_RESTRICT p );

int NS(Particles_have_same_structure)(
    const st_Particles *const SIXTRL_RESTRICT lhs, 
    const st_Particles *const SIXTRL_RESTRICT rhs );

int NS(Particles_map_to_same_memory)(
    const st_Particles *const SIXTRL_RESTRICT lhs, 
    const st_Particles *const SIXTRL_RESTRICT rhs );

int NS(Particles_compare_values)(
    const st_Particles *const SIXTRL_RESTRICT lhs, 
    const st_Particles *const SIXTRL_RESTRICT rhs );



#if defined( __cplusplus )
}
#endif /* defined( __cplusplus ) */

#endif /* SIXTRACKLIB_COMMON_TESTS_TEST_PARTICLES_TOOLS_H__ */

/* end: sixtracklib/common/tests/test_particles_tools.h */
