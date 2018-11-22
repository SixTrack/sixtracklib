#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <set>
#include <vector>

#include <gtest/gtest.h>

#include "sixtracklib/testlib.h"

#include "sixtracklib/common/definitions.h"
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

TEST( C99_ParticlesTests, InitRandomParticleIdsCheckDuplicatesMinMaxParticleIds )
{
    using prng_t      = std::mt19937_64;
    using prng_seed_t = prng_t::result_type;
    using buf_size_t  = ::st_buffer_size_t;
    using num_elem_t  = ::st_particle_num_elements_t;
    using index_t     = ::st_particle_index_t;

    prng_seed_t const seed = prng_seed_t{ 20181120 };
    std::mt19937_64 prng( seed );

    std::vector< buf_size_t > num_particles_list = {
        buf_size_t{      1 },
        buf_size_t{      2 },
        buf_size_t{      3 },
        buf_size_t{     10 },
        buf_size_t{     32 },
        buf_size_t{    100 },
        buf_size_t{    509 },
        buf_size_t{   4297 },
        buf_size_t{  10000 },
        buf_size_t{  65535 },
        buf_size_t{ 100000 } };

    for( auto const num_particles : num_particles_list )
    {
        ASSERT_TRUE( num_particles > buf_size_t{ 0 } );

        ::st_Buffer* pb = ::st_Buffer_new( 0u );
        ::st_Particles* particles = ::st_Particles_new( pb, num_particles );

        ASSERT_TRUE( particles != nullptr );
        ASSERT_TRUE( static_cast< num_elem_t >( num_particles ) ==
                     ::st_Particles_get_num_of_particles( particles ) );

        std::vector< index_t > in_index( num_particles, index_t{ 0 } );
        std::iota( in_index.begin(), in_index.end(), index_t{ 0 } );

        ASSERT_TRUE( in_index.front() == index_t{ 0 } );
        ASSERT_TRUE( in_index.back()  == static_cast< index_t >(
            num_particles - buf_size_t{ 1 } ) );

        std::shuffle( in_index.begin(), in_index.end(), prng );

        ::st_Particles_set_particle_id( particles, in_index.data() );

        ASSERT_TRUE( 0 == std::memcmp( in_index.data(),
            ::st_Particles_get_const_particle_id( particles ), num_particles ) );

        index_t min_particle_id = std::numeric_limits< index_t >::max();
        index_t max_particle_id = std::numeric_limits< index_t >::min();

        ASSERT_TRUE( 0 == ::st_Particles_get_min_max_particle_id(
            particles, &min_particle_id, &max_particle_id ) );

        ASSERT_TRUE( static_cast< buf_size_t >( min_particle_id ) ==
                     index_t{ 0 } );

        ASSERT_TRUE( static_cast< buf_size_t >( max_particle_id ) ==
                     ( num_particles - buf_size_t{ 1 } ) );

        /* ----------------------------------------------------------------- */

        std::uniform_real_distribution< double > sign_dist(
            double{ 0 }, double{ 1 } );

        for( auto& particle_id : in_index )
        {
            if( sign_dist( prng ) > double{ 0.5 } )
            {
                particle_id = -particle_id;
            }
        }

        std::shuffle( in_index.begin(), in_index.end(), prng );

        ::st_Particles_set_particle_id( particles, in_index.data() );

        min_particle_id = std::numeric_limits< index_t >::max();
        max_particle_id = std::numeric_limits< index_t >::min();

        ASSERT_TRUE( 0 == ::st_Particles_get_min_max_particle_id(
            particles, &min_particle_id, &max_particle_id ) );

        ASSERT_TRUE( static_cast< buf_size_t >( min_particle_id ) ==
                     index_t{ 0 } );

        ASSERT_TRUE( static_cast< buf_size_t >( max_particle_id ) ==
                     ( num_particles - buf_size_t{ 1 } ) );

        /* ----------------------------------------------------------------- */

        index_t const MIN_DIST_VALUE = index_t{ 1000u };
        index_t const MAX_DIST_VALUE = MIN_DIST_VALUE + static_cast< index_t >(
            num_particles ) * index_t{ 64 };

        std::uniform_int_distribution< index_t >
            index_dist( MIN_DIST_VALUE, MAX_DIST_VALUE );

        std::set< index_t > temp_particle_ids;

        while( temp_particle_ids.size() < num_particles )
        {
            temp_particle_ids.insert( index_dist( prng ) );
        }

        ASSERT_TRUE( temp_particle_ids.size() > buf_size_t{ 0 } );

        index_t const min_assigned_particle_id = *temp_particle_ids.begin();
        index_t const max_assigned_particle_id = *temp_particle_ids.rbegin();

        ASSERT_TRUE( min_assigned_particle_id <= max_assigned_particle_id );

        in_index.assign( temp_particle_ids.begin(), temp_particle_ids.end() );
        std::shuffle( in_index.begin(), in_index.end(), prng );

        ::st_Particles_set_particle_id( particles, in_index.data() );

        min_particle_id = std::numeric_limits< index_t >::max();
        max_particle_id = std::numeric_limits< index_t >::min();

        ASSERT_TRUE( 0 == ::st_Particles_get_min_max_particle_id(
            particles, &min_particle_id, &max_particle_id ) );

        ASSERT_TRUE( min_particle_id == min_assigned_particle_id );
        ASSERT_TRUE( max_particle_id == max_assigned_particle_id );

        /* ----------------------------------------------------------------- */

        if( num_particles > buf_size_t{ 2 } )
        {
            std::uniform_int_distribution< num_elem_t > dupl_index_dist(
                num_elem_t{ 0 }, static_cast< num_elem_t >( num_particles - 1 ) );

            num_elem_t const dest_id = dupl_index_dist( prng );
            num_elem_t source_id = dupl_index_dist( prng );

            while( source_id == dest_id )
            {
                source_id = dupl_index_dist( prng );
            }

            ::st_Particles_set_particle_id_value( particles, dest_id,
                ::st_Particles_get_particle_id_value( particles, source_id ) );

            std::shuffle( ::st_Particles_get_particle_id( particles ),
                ::st_Particles_get_particle_id( particles ) + num_particles, prng );

            min_particle_id = std::numeric_limits< index_t >::max();
            max_particle_id = std::numeric_limits< index_t >::min();

            ASSERT_TRUE( 0 != ::st_Particles_get_min_max_particle_id(
                particles, &min_particle_id, &max_particle_id ) );

            ASSERT_TRUE( min_particle_id == std::numeric_limits< index_t >::max() );
            ASSERT_TRUE( max_particle_id == std::numeric_limits< index_t >::min() );
        }

        /* ----------------------------------------------------------------- */

        ::st_Buffer_delete( pb );

        particles = nullptr;
        pb = nullptr;
    }
}

/* end: tests/sixtracklib/common/test_particles_c99.cpp */
