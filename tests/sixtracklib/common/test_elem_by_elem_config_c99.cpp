#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <fstream>
#include <iterator>
#include <limits>
#include <random>
#include <vector>
#include <utility>

#include <gtest/gtest.h>

#include "sixtracklib/testlib.h"

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/buffer.h"
#include "sixtracklib/common/be_drift/be_drift.h"
#include "sixtracklib/common/output/elem_by_elem_config.h"
#include "sixtracklib/common/output/output_buffer.h"


TEST( C99_CommonElemByElemConfigTests, MinimalExampleInit )
{
    using config_t     = ::st_ElemByElemConfig;
    using order_t      = ::st_elem_by_elem_order_t;

    using particles_t  = ::st_Particles;
    using part_index_t = ::st_particle_index_t;

    using buf_size_t   = ::st_buffer_size_t;
    using num_elem_t   = ::st_particle_num_elements_t;

    ::st_Buffer* pb = ::st_Buffer_new( 0u );
    ::st_Buffer* eb = ::st_Buffer_new( 0u );

    buf_size_t const NUM_BEAM_ELEMENTS = buf_size_t{ 1000 };
    num_elem_t const NUM_PARTICLES     = num_elem_t{ 1000 };

    for( buf_size_t ii = buf_size_t{ 0 } ; ii < NUM_BEAM_ELEMENTS ; ++ii )
    {
        ::st_Drift*  drift = ::st_Drift_add( eb, double{ 0.1 } );
        ASSERT_TRUE( drift != nullptr );
    }

    particles_t* particles = ::st_Particles_new( pb, NUM_PARTICLES );
    ::st_Particles_realistic_init( particles );

    part_index_t const min_turn = part_index_t{  5 };
    part_index_t const max_turn = part_index_t{ 24 };

    part_index_t const min_element_id = part_index_t{ 0 };
    part_index_t const max_element_id = static_cast< part_index_t >(
        NUM_BEAM_ELEMENTS ) - part_index_t{ 1 };

    part_index_t max_particle_id = part_index_t{ 0 };
    part_index_t min_particle_id =
        static_cast< part_index_t >( NUM_PARTICLES );

    ASSERT_TRUE( 0 == ::st_Particles_get_min_max_particle_id(
        particles, &min_particle_id, &max_particle_id ) );

    std::vector< order_t > elem_by_elem_orders =
        { ::st_ELEM_BY_ELEM_ORDER_TURN_ELEM_PARTICLES };

    for( auto const order : elem_by_elem_orders )
    {
        config_t config;
        ::st_ElemByElemConfig_preset( &config );

        ASSERT_TRUE( 0 == ::st_ElemByElemConfig_init_detailed(
            &config, order, min_particle_id, max_particle_id,
                min_element_id, max_element_id, min_turn, max_turn, false ) );

        ASSERT_TRUE( ::st_ElemByElemConfig_get_order( &config ) == order );

        ASSERT_TRUE( ::st_ElemByElemConfig_get_min_particle_id( &config ) ==
                     min_particle_id );

        ASSERT_TRUE( ::st_ElemByElemConfig_get_max_particle_id( &config ) ==
                     max_particle_id );

        ASSERT_TRUE( ::st_ElemByElemConfig_get_min_element_id( &config ) ==
                     min_element_id );

        ASSERT_TRUE( ::st_ElemByElemConfig_get_max_element_id( &config ) ==
                     max_element_id );

        ASSERT_TRUE( ::st_ElemByElemConfig_get_min_turn( &config ) == min_turn );
        ASSERT_TRUE( ::st_ElemByElemConfig_get_max_turn( &config ) == max_turn );

        ASSERT_TRUE( ::st_ElemByElemConfig_get_num_particles_to_store(
            &config ) == NUM_PARTICLES );

        ASSERT_TRUE( ::st_ElemByElemConfig_get_num_elements_to_store(
            &config ) == NUM_BEAM_ELEMENTS );

        buf_size_t const cmp_num_turns_to_store = static_cast< buf_size_t >(
            max_turn - min_turn ) + buf_size_t{ 1 };

        ASSERT_TRUE( ::st_ElemByElemConfig_get_num_turns_to_store( &config )
            == cmp_num_turns_to_store );

        ASSERT_TRUE( ::st_ElemByElemConfig_get_out_store_num_particles(
            &config ) == ( NUM_PARTICLES * NUM_BEAM_ELEMENTS *
                cmp_num_turns_to_store ) );

        ASSERT_TRUE( ::st_ElemByElemConfig_get_output_store_address(
            &config ) == ::st_elem_by_elem_out_addr_t{ 0 } );

        ::st_Buffer* elem_by_elem_buffer = ::st_Buffer_new( 0u );

        buf_size_t out_buffer_index_offset = buf_size_t{ 1 };

        int ret = ::st_ElemByElemConfig_prepare_output_buffer_from_conf(
            &config, elem_by_elem_buffer, &out_buffer_index_offset );

        ASSERT_TRUE( ret == 0 );
        ASSERT_TRUE( out_buffer_index_offset == buf_size_t{ 0 } );

        ASSERT_TRUE( ::st_Buffer_get_num_of_objects(
            elem_by_elem_buffer ) == buf_size_t{ 1 } );

        particles_t* elem_by_elem_particles =
            ::st_Particles_buffer_get_particles(
                elem_by_elem_buffer, out_buffer_index_offset );

        ASSERT_TRUE( elem_by_elem_particles != nullptr );

        num_elem_t const store_size =
            ::st_ElemByElemConfig_get_num_particles_to_store( &config );

        ASSERT_TRUE( ::st_Particles_get_num_of_particles(
            elem_by_elem_particles ) >= store_size );

        for( num_elem_t ii = num_elem_t{ 0 } ; ii < store_size ; ++ii )
        {
            part_index_t const particle_id =
                ::st_ElemByElemConfig_get_particle_id_from_store_index(
                    &config, ii );

            part_index_t const at_element_id =
                ::st_ElemByElemConfig_get_at_element_id_from_store_index(
                    &config, ii );

            part_index_t const at_turn_id =
                ::st_ElemByElemConfig_get_at_turn_from_store_index(
                    &config, ii );

            ASSERT_TRUE( particle_id   >= part_index_t{ 0 } );
            ASSERT_TRUE( at_element_id >= part_index_t{ 0 } );
            ASSERT_TRUE( at_turn_id    >= part_index_t{ 0 } );

            num_elem_t const calc_store_index =
                ::st_ElemByElemConfig_get_particles_store_index_details(
                    &config, particle_id, at_element_id, at_turn_id );

            ASSERT_TRUE( calc_store_index >= num_elem_t{ 0 } );
            ASSERT_TRUE( calc_store_index <  store_size );
            ASSERT_TRUE( calc_store_index == ii );
        }

        ::st_Buffer_delete( elem_by_elem_buffer );
        elem_by_elem_buffer = nullptr;
    }

    if( ( NUM_PARTICLES     > num_elem_t{ 0 } ) &&
        ( NUM_BEAM_ELEMENTS > num_elem_t{ 0 } ) )
    {
        buf_size_t const until_turn_elem_by_elem = max_turn + buf_size_t{ 1 };
        SIXTRL_ASSERT( until_turn_elem_by_elem >
                       static_cast< buf_size_t >( min_turn ) );

        buf_size_t const cmp_num_turns_to_store =
            until_turn_elem_by_elem - static_cast< buf_size_t >( min_turn );

        ::st_Buffer* elem_by_elem_buffer = ::st_Buffer_new( 0u );
        ::NS(Particles_set_all_at_turn_value)( particles, min_turn );

        buf_size_t elem_by_elem_index_offset = buf_size_t{ 0 };

        int ret = ::st_ElemByElemConfig_prepare_output_buffer(
            eb, elem_by_elem_buffer, particles, until_turn_elem_by_elem,
            &elem_by_elem_index_offset );

        ASSERT_TRUE( ret == 0 );

        ASSERT_TRUE( ::st_Buffer_get_num_of_objects(
            elem_by_elem_buffer ) == buf_size_t{ 1 } );

        ASSERT_TRUE( elem_by_elem_index_offset == buf_size_t{ 0 } );

        particles_t* elem_by_elem_particles =
            ::st_Particles_buffer_get_particles(
                elem_by_elem_buffer, elem_by_elem_index_offset );

        ASSERT_TRUE( elem_by_elem_particles != nullptr );
        ASSERT_TRUE( ::st_Particles_get_num_of_particles(
            elem_by_elem_particles ) == static_cast< num_elem_t >(
                NUM_PARTICLES * NUM_BEAM_ELEMENTS * cmp_num_turns_to_store ) );

        ::st_Buffer_delete( elem_by_elem_buffer );
        elem_by_elem_buffer = nullptr;
    }

    ::st_Buffer_delete( eb );
    ::st_Buffer_delete( pb );
}

/* end: tests/sixtracklib/common/test_elem_by_elem_config_c99.cpp */
