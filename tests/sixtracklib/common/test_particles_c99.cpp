#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>

#include <gtest/gtest.h>

#include "sixtracklib/testlib.h"

#include "sixtracklib/_impl/definitions.h"
#include "sixtracklib/common/buffer.h"
#include "sixtracklib/common/particles.h"

/* ========================================================================= */
/* ====  Test random initialization of particles                             */

TEST( C99_ParticlesTests, RandomInitParticlesCopyAndCompare )
{
    uint64_t seed = UINT64_C( 20180420 );
    st_Random_init_genrand64( seed );

    st_buffer_size_t const NUM_PARTICLES = ( st_buffer_size_t )1000u;

    /* --------------------------------------------------------------------- */

    st_Buffer* pb = st_Buffer_new( ( st_buffer_size_t )( 1u << 20u ) );
    ASSERT_TRUE( pb != SIXTRL_NULLPTR );

    st_Particles* p = st_Particles_new( pb, NUM_PARTICLES );
    ASSERT_TRUE( p != SIXTRL_NULLPTR );
    ASSERT_TRUE( st_Buffer_get_num_of_objects( pb ) == st_buffer_size_t{ 1 } );

    ASSERT_TRUE( st_Particles_get_num_of_particles( p ) == NUM_PARTICLES );

    ASSERT_TRUE( st_Particles_get_const_q0( p )     != SIXTRL_NULLPTR );
    ASSERT_TRUE( st_Particles_get_const_mass0( p )  != SIXTRL_NULLPTR );
    ASSERT_TRUE( st_Particles_get_const_beta0( p )  != SIXTRL_NULLPTR );
    ASSERT_TRUE( st_Particles_get_const_gamma0( p ) != SIXTRL_NULLPTR );
    ASSERT_TRUE( st_Particles_get_const_p0c( p )    != SIXTRL_NULLPTR );
    ASSERT_TRUE( st_Particles_get_const_s( p )      != SIXTRL_NULLPTR );
    ASSERT_TRUE( st_Particles_get_const_x( p )      != SIXTRL_NULLPTR );
    ASSERT_TRUE( st_Particles_get_const_y( p )      != SIXTRL_NULLPTR );
    ASSERT_TRUE( st_Particles_get_const_px( p )     != SIXTRL_NULLPTR );
    ASSERT_TRUE( st_Particles_get_const_py( p )     != SIXTRL_NULLPTR );
    ASSERT_TRUE( st_Particles_get_const_zeta( p )   != SIXTRL_NULLPTR );
    ASSERT_TRUE( st_Particles_get_const_psigma( p ) != SIXTRL_NULLPTR );
    ASSERT_TRUE( st_Particles_get_const_delta( p )  != SIXTRL_NULLPTR );
    ASSERT_TRUE( st_Particles_get_const_rpp( p )    != SIXTRL_NULLPTR );
    ASSERT_TRUE( st_Particles_get_const_rvv( p )    != SIXTRL_NULLPTR );
    ASSERT_TRUE( st_Particles_get_const_chi( p )    != SIXTRL_NULLPTR );

    ASSERT_TRUE( st_Particles_get_const_particle_id( p )   != SIXTRL_NULLPTR );
    ASSERT_TRUE( st_Particles_get_const_at_element_id( p ) != SIXTRL_NULLPTR );
    ASSERT_TRUE( st_Particles_get_const_at_turn( p )       != SIXTRL_NULLPTR );
    ASSERT_TRUE( st_Particles_get_const_state( p )         != SIXTRL_NULLPTR );

    /* --------------------------------------------------------------------- */

    st_Particles* p_copy =  st_Particles_new( pb, NUM_PARTICLES );
    ASSERT_TRUE(  p_copy != SIXTRL_NULLPTR );
    ASSERT_TRUE( st_Buffer_get_num_of_objects( pb ) == st_buffer_size_t{ 2 } );

    ASSERT_TRUE( st_Particles_get_num_of_particles( p_copy ) == NUM_PARTICLES );

    ASSERT_TRUE( st_Particles_get_const_q0( p_copy )     != SIXTRL_NULLPTR );
    ASSERT_TRUE( st_Particles_get_const_mass0( p_copy )  != SIXTRL_NULLPTR );
    ASSERT_TRUE( st_Particles_get_const_beta0( p_copy )  != SIXTRL_NULLPTR );
    ASSERT_TRUE( st_Particles_get_const_gamma0( p_copy ) != SIXTRL_NULLPTR );
    ASSERT_TRUE( st_Particles_get_const_p0c( p_copy )    != SIXTRL_NULLPTR );
    ASSERT_TRUE( st_Particles_get_const_s( p_copy )      != SIXTRL_NULLPTR );
    ASSERT_TRUE( st_Particles_get_const_x( p_copy )      != SIXTRL_NULLPTR );
    ASSERT_TRUE( st_Particles_get_const_y( p_copy )      != SIXTRL_NULLPTR );
    ASSERT_TRUE( st_Particles_get_const_px( p_copy )     != SIXTRL_NULLPTR );
    ASSERT_TRUE( st_Particles_get_const_py( p_copy )     != SIXTRL_NULLPTR );
    ASSERT_TRUE( st_Particles_get_const_zeta( p_copy )   != SIXTRL_NULLPTR );
    ASSERT_TRUE( st_Particles_get_const_psigma( p_copy ) != SIXTRL_NULLPTR );
    ASSERT_TRUE( st_Particles_get_const_delta( p_copy )  != SIXTRL_NULLPTR );
    ASSERT_TRUE( st_Particles_get_const_rpp( p_copy )    != SIXTRL_NULLPTR );
    ASSERT_TRUE( st_Particles_get_const_rvv( p_copy )    != SIXTRL_NULLPTR );
    ASSERT_TRUE( st_Particles_get_const_chi( p_copy )    != SIXTRL_NULLPTR );

    ASSERT_TRUE( st_Particles_get_const_particle_id( p_copy ) !=
                 SIXTRL_NULLPTR );

    ASSERT_TRUE( st_Particles_get_const_at_element_id( p_copy ) !=
                 SIXTRL_NULLPTR );

    ASSERT_TRUE( st_Particles_get_const_at_turn( p_copy ) !=
                 SIXTRL_NULLPTR );

    ASSERT_TRUE( st_Particles_get_const_state( p_copy ) !=
                 SIXTRL_NULLPTR );

    /* --------------------------------------------------------------------- */

    st_Particles_random_init( p );
    st_Particles_copy( p_copy, p );

    /* --------------------------------------------------------------------- */

    ASSERT_TRUE( st_Particles_have_same_structure( p_copy, p ) );
    ASSERT_TRUE( !st_Particles_map_to_same_memory( p_copy, p ) );
    ASSERT_TRUE( 0 == st_Particles_compare_values( p_copy, p ) );

    /* --------------------------------------------------------------------- */

    p      = SIXTRL_NULLPTR;
    p_copy = SIXTRL_NULLPTR;

    st_Buffer_delete( pb );
    pb = SIXTRL_NULLPTR;
}


/* end: tests/sixtracklib/common/test_particles_c99.cpp */
