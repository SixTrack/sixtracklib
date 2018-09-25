#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <numeric>

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


TEST( C99_ParticlesTests, ParticlesBufferAddressByGlobalParticleIndex )
{
    using size_t     = ::st_buffer_size_t;
    using index_t    = ::st_particle_index_t;
    using num_elem_t = ::st_particle_num_elements_t;
    using object_t   = ::st_Object;

    std::vector< num_elem_t > num_particles_list =
        std::vector< num_elem_t >{ 1, 2, 4, 6, 8, 12, 1000, 1, 2, 6, 8, 128 };

    ::st_Buffer* pb = ::st_Buffer_new( size_t{ 1u << 20u } );
    ASSERT_TRUE( pb != SIXTRL_NULLPTR );

    index_t global_particle_index = index_t{ 0 };

    for( num_elem_t const NUM_PARTICLES : num_particles_list )
    {
        ::st_Particles* p = ::st_Particles_new( pb, NUM_PARTICLES );

        ASSERT_TRUE( p != nullptr );
        ASSERT_TRUE( ::st_Particles_get_num_of_particles( p ) == NUM_PARTICLES );

        for( num_elem_t ii = 0 ; ii < NUM_PARTICLES ; ++ii, ++global_particle_index )
        {
            ::st_Particles_set_particle_id_value( p, ii, global_particle_index );
        }
    }

    num_elem_t const total_num_particles =
        ::st_Particles_buffer_get_total_num_of_particles( pb );

    ASSERT_TRUE(  total_num_particles == std::accumulate(
        num_particles_list.begin(), num_particles_list.end(), num_elem_t{ 0 } ) );

    ASSERT_TRUE( ::st_Particles_buffer_get_num_of_particle_blocks( pb ) ==
                 num_particles_list.size() );

    /* --------------------------------------------------------------------- */
    /* Always search from the begin -> slow but should work all the time     */

    num_elem_t block_begin_index = num_elem_t{ 0 };

    global_particle_index = 0;

    object_t const* obj_begin = ::st_Buffer_get_const_objects_begin( pb );
    object_t const* obj_end   = ::st_Buffer_get_const_objects_end( pb );
    object_t const* idx       = nullptr;

    for( num_elem_t ii = 0 ; ii < total_num_particles ; ++ii, ++global_particle_index )
    {
        block_begin_index = num_elem_t{ 0 };

        idx = ::st_BufferIndex_get_index_object_by_global_index_from_range(
                ii, block_begin_index, obj_begin, obj_end, &block_begin_index );

        ASSERT_TRUE( idx != obj_end );
        ASSERT_TRUE( idx != nullptr );

        ::st_Particles const* p = ::st_BufferIndex_get_const_particles( idx );
        ASSERT_TRUE( p != nullptr );

        ASSERT_TRUE( ii < ( block_begin_index +
            ::st_Particles_get_num_of_particles( p ) ) );

        num_elem_t const local_index = ii - block_begin_index;
        ASSERT_TRUE( local_index < ::st_Particles_get_num_of_particles( p ) );

        ASSERT_TRUE( global_particle_index ==
            ::st_Particles_get_particle_id_value( p, local_index ) );
    }

    /* simulate what happens if we try to access a particle by an global
     * index that is too large, e.g. there are less particles than
     * required for such a index */

    block_begin_index = num_elem_t{ 0 };

    idx = ::st_BufferIndex_get_index_object_by_global_index_from_range(
        total_num_particles, block_begin_index, obj_begin, obj_end,
            &block_begin_index );

    ASSERT_TRUE( idx != nullptr );
    ASSERT_TRUE( idx == obj_end );
    ASSERT_TRUE( block_begin_index == total_num_particles );

    /* even larger! */

    block_begin_index = 0;
    idx = ::st_BufferIndex_get_index_object_by_global_index_from_range(
        2 * total_num_particles, block_begin_index, obj_begin, obj_end,
            &block_begin_index );

    ASSERT_TRUE( idx != nullptr );
    ASSERT_TRUE( idx == obj_end );
    ASSERT_TRUE( block_begin_index == total_num_particles );

    /* --------------------------------------------------------------------- */
    /* Search incrementally -> should be much faster, but the user is re-
     * sponsible for ensuring that the sequence of returned idx variables is
     * increasing and not decreasing */

    global_particle_index = 0;
    block_begin_index = 0;

    idx = obj_begin;

    for( num_elem_t ii = 0 ; ii < total_num_particles ; ++ii, ++global_particle_index )
    {
        num_elem_t found_block_begin_index = 0;
        object_t const* prev_idx = idx;

        idx = ::st_BufferIndex_get_index_object_by_global_index_from_range(
                ii, block_begin_index, idx, obj_end, &found_block_begin_index );

        ASSERT_TRUE( idx != obj_end );
        ASSERT_TRUE( idx != nullptr );
        ASSERT_TRUE( std::distance( prev_idx, idx ) >= 0 );
        ASSERT_TRUE( found_block_begin_index >= block_begin_index );

        ASSERT_TRUE( found_block_begin_index ==
            ::st_BufferIndex_get_total_num_of_particles_in_range(
                obj_begin, idx ) );

        ASSERT_TRUE( found_block_begin_index <= ii );

        ::st_Particles const* p = ::st_BufferIndex_get_const_particles( idx );
        ASSERT_TRUE( p != nullptr );

        ASSERT_TRUE( ii < ( found_block_begin_index +
            ::st_Particles_get_num_of_particles( p ) ) );

        num_elem_t const local_index = ii - found_block_begin_index;
        ASSERT_TRUE( local_index < ::st_Particles_get_num_of_particles( p ) );

        ASSERT_TRUE( global_particle_index ==
            ::st_Particles_get_particle_id_value( p, local_index ) );

        block_begin_index = found_block_begin_index;
    }

    ::st_Buffer_delete( pb );
}


/* end: tests/sixtracklib/common/test_particles_c99.cpp */
