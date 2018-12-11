#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <fstream>
#include <iterator>
#include <limits>
#include <map>
#include <random>
#include <vector>
#include <utility>

#include <gtest/gtest.h>

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/generated/path.h"
#include "sixtracklib/common/buffer.h"
#include "sixtracklib/common/be_drift/be_drift.h"
#include "sixtracklib/common/be_monitor/be_monitor.h"
#include "sixtracklib/common/be_monitor/track.h"
#include "sixtracklib/common/output/output_buffer.h"
#include "sixtracklib/common/track.h"

#include "sixtracklib/testlib/common/particles.h"

TEST( C99_CommonBeamMonitorTests, MinimalAddToBufferCopyRemapRead )
{
    using size_t        = ::st_buffer_size_t;
    using raw_t         = unsigned char;
    using nturn_t       = ::st_be_monitor_turn_t;
    using addr_t        = ::st_be_monitor_addr_t;
    using index_t       = ::st_be_monitor_index_t;
    using part_index_t  = ::st_particle_index_t;

    constexpr size_t NUM_PARTICLES = size_t{ 2 };

    std::vector< ::st_BeamMonitor > cmp_beam_monitors;

    ::st_Buffer* eb = ::st_Buffer_new( 0u );
    ::st_Buffer* pb = ::st_Buffer_new( 0u );
    ::st_Buffer* output_buffer = ::st_Buffer_new( 0u );

    ::st_Particles* out_particles = nullptr;
    ::st_Particles* particles     = ::st_Particles_new( pb, NUM_PARTICLES );
    ::st_Particles_init_particle_ids( particles );

    part_index_t min_particle_id = std::numeric_limits< part_index_t >::max();
    part_index_t max_particle_id = std::numeric_limits< part_index_t >::min();

    ASSERT_TRUE( 0 == ::st_Particles_get_min_max_particle_id(
        particles, &min_particle_id, &max_particle_id ) );

    ASSERT_TRUE( min_particle_id <= max_particle_id );

    std::vector< nturn_t > num_stores_list = { nturn_t{ 10 }, nturn_t{ 10 } };

    for( auto const nn : num_stores_list )
    {
        out_particles = ::st_Particles_new( output_buffer, NUM_PARTICLES * nn );
        ASSERT_TRUE( out_particles != nullptr );
    }

    addr_t const out_addr_0 = reinterpret_cast< uintptr_t >(
        ::st_Particles_buffer_get_particles( output_buffer, 0u ) );

    addr_t const out_addr_1 = reinterpret_cast< uintptr_t >(
        ::st_Particles_buffer_get_particles( output_buffer, 1u ) );

    ASSERT_TRUE( ( out_addr_0 != addr_t{ 0 } )&& ( out_addr_1 != addr_t{ 0 } ) );

    ::st_BeamMonitor* be_monitor = ::st_BeamMonitor_new( eb );

    ASSERT_TRUE( be_monitor != nullptr );
    ASSERT_TRUE(  ::st_BeamMonitor_get_num_stores( be_monitor ) == nturn_t{ 0 } );
    ASSERT_TRUE(  ::st_BeamMonitor_get_start( be_monitor ) == nturn_t{ 0 } );
    ASSERT_TRUE(  ::st_BeamMonitor_get_skip( be_monitor ) == nturn_t{ 1 } );
    ASSERT_TRUE( !::st_BeamMonitor_is_rolling( be_monitor ) );
    ASSERT_TRUE(  ::st_BeamMonitor_is_turn_ordered( be_monitor ) );
    ASSERT_TRUE(  ::st_BeamMonitor_get_out_address( be_monitor ) == addr_t{ 0 } );
    ASSERT_TRUE(  ::st_BeamMonitor_get_min_particle_id( be_monitor ) == index_t{ 0 } );
    ASSERT_TRUE(  ::st_BeamMonitor_get_max_particle_id( be_monitor ) == index_t{ 0 } );

    ::st_BeamMonitor_set_num_stores( be_monitor, 10u );
    ::st_BeamMonitor_set_start( be_monitor, 1000u );
    ::st_BeamMonitor_set_skip( be_monitor, 4u );
    ::st_BeamMonitor_set_is_rolling( be_monitor, true );
    ::st_BeamMonitor_setup_for_particles( be_monitor, particles );
    ::st_BeamMonitor_set_out_address( be_monitor, out_addr_0 );

    ASSERT_TRUE( be_monitor != nullptr );
    ASSERT_TRUE(  ::st_BeamMonitor_get_num_stores( be_monitor ) == nturn_t{ 10 } );
    ASSERT_TRUE(  ::st_BeamMonitor_get_start( be_monitor ) == nturn_t{ 1000 } );
    ASSERT_TRUE(  ::st_BeamMonitor_get_skip( be_monitor ) == nturn_t{ 4 } );
    ASSERT_TRUE(  ::st_BeamMonitor_is_rolling( be_monitor ) );
    ASSERT_TRUE(  ::st_BeamMonitor_is_turn_ordered( be_monitor ) );
    ASSERT_TRUE(  ::st_BeamMonitor_get_out_address( be_monitor ) == out_addr_0 );

    ASSERT_TRUE(  ::st_BeamMonitor_get_min_particle_id( be_monitor ) >=
                  index_t{ 0 } );

    ASSERT_TRUE(  ::st_BeamMonitor_get_min_particle_id( be_monitor ) <=
                  static_cast< index_t >( min_particle_id ) );

    ASSERT_TRUE( ::st_BeamMonitor_get_max_particle_id( be_monitor ) >=
                 ::st_BeamMonitor_get_min_particle_id( be_monitor ) );

    ASSERT_TRUE( ::st_BeamMonitor_get_max_particle_id( be_monitor ) >=
                 static_cast< index_t >( max_particle_id ) );

    cmp_beam_monitors.push_back( *be_monitor );

    ::st_BeamMonitor* copy_be_monitor =
        ::st_BeamMonitor_add_copy( eb, &cmp_beam_monitors.back() );

    ASSERT_TRUE( copy_be_monitor != nullptr );
    ASSERT_TRUE( copy_be_monitor != be_monitor );

    cmp_beam_monitors.push_back( *copy_be_monitor );

    ASSERT_TRUE(  ::st_BeamMonitor_get_num_stores( copy_be_monitor ) ==
                  ::st_BeamMonitor_get_num_stores( &cmp_beam_monitors[ 0 ] ) );

    ASSERT_TRUE(  ::st_BeamMonitor_get_start( copy_be_monitor ) ==
                  ::st_BeamMonitor_get_start( &cmp_beam_monitors[ 0 ] ) );

    ASSERT_TRUE(  ::st_BeamMonitor_get_skip( copy_be_monitor ) ==
                  ::st_BeamMonitor_get_skip( &cmp_beam_monitors[ 0 ] ) );

    ASSERT_TRUE(  ::st_BeamMonitor_is_rolling( copy_be_monitor ) ==
                  ::st_BeamMonitor_is_rolling( &cmp_beam_monitors[ 0 ] ) );

    ASSERT_TRUE(  ::st_BeamMonitor_is_turn_ordered( copy_be_monitor ) ==
                  ::st_BeamMonitor_is_turn_ordered( &cmp_beam_monitors[ 0 ] ) );

    ASSERT_TRUE(  ::st_BeamMonitor_is_turn_ordered( copy_be_monitor ) ==
                  ::st_BeamMonitor_is_turn_ordered( &cmp_beam_monitors[ 0 ] ) );

    ASSERT_TRUE(  ::st_BeamMonitor_get_out_address( copy_be_monitor ) ==
                  ::st_BeamMonitor_get_out_address( &cmp_beam_monitors[ 0 ] ) );

    ASSERT_TRUE(  ::st_BeamMonitor_get_min_particle_id( copy_be_monitor ) ==
                  ::st_BeamMonitor_get_min_particle_id( &cmp_beam_monitors[ 0 ] ) );

    ASSERT_TRUE(  ::st_BeamMonitor_get_max_particle_id( copy_be_monitor ) ==
                  ::st_BeamMonitor_get_max_particle_id( &cmp_beam_monitors[ 0 ] ) );

    be_monitor = ::st_BeamMonitor_add( eb, 10u, 5000u, 1u, out_addr_1,
        min_particle_id, max_particle_id, false, true );

    ASSERT_TRUE( be_monitor != nullptr );
    ASSERT_TRUE(  ::st_BeamMonitor_get_num_stores( be_monitor ) == nturn_t{ 10 } );
    ASSERT_TRUE(  ::st_BeamMonitor_get_start( be_monitor ) == nturn_t{ 5000 } );
    ASSERT_TRUE(  ::st_BeamMonitor_get_skip( be_monitor ) == nturn_t{ 1 } );
    ASSERT_TRUE( !::st_BeamMonitor_is_rolling( be_monitor ) );
    ASSERT_TRUE(  ::st_BeamMonitor_is_turn_ordered( be_monitor ) );
    ASSERT_TRUE(  ::st_BeamMonitor_get_out_address( be_monitor ) == out_addr_1 );

    ASSERT_TRUE(  ::st_BeamMonitor_get_min_particle_id( be_monitor ) ==
                  static_cast< index_t >( min_particle_id ) );

    ASSERT_TRUE(  ::st_BeamMonitor_get_max_particle_id( be_monitor ) ==
                  static_cast< index_t >( max_particle_id ) );

    cmp_beam_monitors.push_back( *be_monitor );

    ASSERT_TRUE( cmp_beam_monitors.size() ==
                 ::st_Buffer_get_num_of_objects( eb ) );

    std::vector< raw_t > copy_raw_buffer(
        ::st_Buffer_get_size( eb ), raw_t{ 0 } );

    copy_raw_buffer.assign( ::st_Buffer_get_const_data_begin( eb ),
                            ::st_Buffer_get_const_data_end( eb ) );

    ::st_Buffer cmp_eb;
    ::st_Buffer_preset( &cmp_eb );

    ASSERT_TRUE( 0 == ::st_Buffer_init_from_data( &cmp_eb,
        copy_raw_buffer.data(), copy_raw_buffer.size() ) );

    ASSERT_TRUE( !::st_Buffer_needs_remapping( &cmp_eb ) );
    ASSERT_TRUE(  ::st_Buffer_get_num_of_objects( &cmp_eb ) ==
                  ::st_Buffer_get_num_of_objects( eb ) );

    size_t jj = 0;

    for( auto const& cmp_be_monitor : cmp_beam_monitors )
    {
        ::st_Object const* ptr_obj = ::st_Buffer_get_const_object( eb, jj++ );

        ASSERT_TRUE( ptr_obj != nullptr );
        ASSERT_TRUE( ::st_Object_get_type_id( ptr_obj ) ==
                     ::st_OBJECT_TYPE_BEAM_MONITOR );

        ::st_BeamMonitor const* ptr_beam_monitor =
            reinterpret_cast< ::st_BeamMonitor const* >(
                ::st_Object_get_const_begin_ptr( ptr_obj ) );

        ASSERT_TRUE( ptr_beam_monitor != nullptr );
        ASSERT_TRUE( 0 == ::st_BeamMonitor_compare_values(
            ptr_beam_monitor, &cmp_be_monitor ) );
    }

    ASSERT_TRUE( ::st_BeamMonitor_get_num_elem_by_elem_objects( eb ) ==
                 cmp_beam_monitors.size() );

    ASSERT_TRUE( ::st_BeamMonitor_get_num_elem_by_elem_objects_from_managed_buffer(
        ::st_Buffer_get_const_data_begin( eb ), ::st_Buffer_get_slot_size( eb ) )
            == cmp_beam_monitors.size() );

    ASSERT_TRUE( ::st_BeamMonitor_get_num_of_beam_monitor_objects( eb ) ==
                 cmp_beam_monitors.size() );

    ASSERT_TRUE( ::st_BeamMonitor_get_num_of_beam_monitor_objects_from_managed_buffer(
        ::st_Buffer_get_const_data_begin( eb ), ::st_Buffer_get_slot_size( eb ) ) ==
                 cmp_beam_monitors.size() );

    ::st_BeamMonitor_clear_all( eb );

    jj = 0;

    for( auto const& cmp_be_monitor : cmp_beam_monitors )
    {
        ::st_Object const* ptr_obj = ::st_Buffer_get_const_object( eb, jj++ );

        ASSERT_TRUE( ptr_obj != nullptr );
        ASSERT_TRUE( ::st_Object_get_type_id( ptr_obj ) ==
                     ::st_OBJECT_TYPE_BEAM_MONITOR );

        ::st_BeamMonitor const* ptr_bemon =
            reinterpret_cast< ::st_BeamMonitor const* >(
                ::st_Object_get_const_begin_ptr( ptr_obj ) );

        ASSERT_TRUE( ptr_bemon != nullptr );

        ASSERT_TRUE( ::st_BeamMonitor_get_num_stores( ptr_bemon ) ==
                     ::st_BeamMonitor_get_num_stores( &cmp_be_monitor ) );

        ASSERT_TRUE( ::st_BeamMonitor_get_start( ptr_bemon ) ==
                     ::st_BeamMonitor_get_start( &cmp_be_monitor ) );

        ASSERT_TRUE( ::st_BeamMonitor_get_skip( ptr_bemon ) ==
                     ::st_BeamMonitor_get_skip( &cmp_be_monitor ) );

        ASSERT_TRUE( ::st_BeamMonitor_is_rolling( ptr_bemon ) ==
                     ::st_BeamMonitor_is_rolling( &cmp_be_monitor ) );

        ASSERT_TRUE( ::st_BeamMonitor_is_turn_ordered( ptr_bemon ) ==
                     ::st_BeamMonitor_is_turn_ordered( &cmp_be_monitor ) );

        ASSERT_TRUE( ::st_BeamMonitor_get_out_address( ptr_bemon ) !=
                     ::st_BeamMonitor_get_out_address( &cmp_be_monitor ) );

        ASSERT_TRUE( ::st_BeamMonitor_get_out_address( ptr_bemon ) ==
                     addr_t{ 0 } );

        ASSERT_TRUE( ::st_BeamMonitor_get_min_particle_id( ptr_bemon ) ==
                     index_t{ 0 } );

        ASSERT_TRUE( ::st_BeamMonitor_get_max_particle_id( ptr_bemon ) ==
                     index_t{ 0 } );
    }

    ::st_Buffer_delete( eb );
    ::st_Buffer_delete( pb );
    ::st_Buffer_delete( output_buffer );
    ::st_Buffer_free( &cmp_eb );
}

TEST( C99_CommonBeamMonitorTests, AssignIoBufferToBeamMonitors )
{
    using real_t        = ::st_particle_real_t;
    using size_t        = ::st_buffer_size_t;
    using nturn_t       = ::st_be_monitor_turn_t;
    using addr_t        = ::st_be_monitor_addr_t;
    using index_t       = ::st_be_monitor_index_t;
    using part_index_t  = ::st_particle_index_t;
    using turn_dist_t   = std::uniform_int_distribution< nturn_t >;
    using chance_dist_t = std::uniform_real_distribution< real_t >;

    std::mt19937_64::result_type const seed = 20181031u;

    std::mt19937_64 prng;
    prng.seed( seed );

    ::st_Buffer* eb = ::st_Buffer_new( 0u );
    ::st_Buffer* pb = ::st_Buffer_new( 0u );
    ::st_Buffer* out_buffer = ::st_Buffer_new( 0u );

    size_t const NUM_BEAM_MONITORS  = size_t{  10 };
    size_t const NUM_DRIFTS         = size_t{ 100 };
    size_t const NUM_PARTICLES      = size_t{   2 };
    size_t const DRIFT_SEQU_LEN     = NUM_DRIFTS / NUM_BEAM_MONITORS;
    size_t const NUM_BEAM_ELEMENTS  = NUM_DRIFTS + NUM_BEAM_MONITORS;

    std::vector< ::st_BeamMonitor > cmp_beam_monitors;

    turn_dist_t num_stores_dist( 1, 100 );
    turn_dist_t start_dist( 0, 1000 );
    turn_dist_t skip_dist( 1, 100 );

    chance_dist_t rolling_dist( 0., 1. );
    size_t sum_num_of_stores = size_t{ 0 };

    for( size_t ii = size_t{ 0 } ; ii < NUM_BEAM_MONITORS ; ++ii )
    {
        for( size_t jj = size_t{ 0 } ; jj < DRIFT_SEQU_LEN ; ++jj )
        {
            ::st_Drift*  drift = ::st_Drift_add( eb, real_t{ 1.0 } );
            ASSERT_TRUE( drift != nullptr );
        }

        ::st_BeamMonitor* be_monitor = ::st_BeamMonitor_add( eb,
            num_stores_dist( prng ), start_dist( prng ), skip_dist( prng ),
            addr_t{ 0 }, index_t{ 0 }, index_t{ 0 },
            bool{ rolling_dist( prng ) >= 0.5 }, true );

        ASSERT_TRUE( be_monitor != nullptr );

        sum_num_of_stores += ::st_BeamMonitor_get_num_stores( be_monitor );
        cmp_beam_monitors.push_back( *be_monitor );
    }

    ASSERT_TRUE( ::st_BeamMonitor_get_num_elem_by_elem_objects( eb ) ==
                 ( NUM_DRIFTS + NUM_BEAM_MONITORS ) );

    ASSERT_TRUE( ::st_BeamMonitor_get_num_of_beam_monitor_objects( eb ) ==
                 NUM_BEAM_MONITORS );

    /* --------------------------------------------------------------------- */
    /* reserve out_buffer buffer without element by element buffer */

    ::st_Particles* particles = ::st_Particles_new( pb, NUM_PARTICLES );
    ::st_Particles_init_particle_ids( particles );

    part_index_t min_particle_id = std::numeric_limits< part_index_t >::max();
    part_index_t max_particle_id = std::numeric_limits< part_index_t >::min();

    ASSERT_TRUE( 0 == ::st_Particles_get_min_max_particle_id(
        particles, &min_particle_id, &max_particle_id ) );

    ASSERT_TRUE( min_particle_id <= max_particle_id );
    ASSERT_TRUE( min_particle_id >= part_index_t{ 0 } );

    ASSERT_TRUE( 0 == ::st_BeamMonitor_prepare_particles_out_buffer(
        eb, out_buffer, particles, 0u ) );

    ASSERT_TRUE( NUM_BEAM_MONITORS ==
        ::st_Particles_buffer_get_num_of_particle_blocks( out_buffer ) );

    ASSERT_TRUE( ( sum_num_of_stores * NUM_PARTICLES ) == static_cast< size_t >(
        ::st_Particles_buffer_get_total_num_of_particles( out_buffer ) ) );

    ASSERT_TRUE( 0 == ::st_BeamMonitor_assign_particles_out_buffer(
        eb, out_buffer, 0u ) );

    ::st_Object const* eb_begin = ::st_Buffer_get_const_objects_begin( eb );
    ::st_Object const* eb_end   = ::st_Buffer_get_const_objects_end( eb );
    ::st_Object const* eb_it    = eb_begin;

    size_t out_particles_offset = size_t{ 0 };

    for( ; eb_it != eb_end ; ++eb_it )
    {
        if( ::st_Object_get_type_id( eb_it ) == ::st_OBJECT_TYPE_BEAM_MONITOR )
        {
            ::st_BeamMonitor const* be_monitor = reinterpret_cast<
                ::st_BeamMonitor const* >( ::st_Object_get_const_begin_ptr(
                    eb_it ) );

            ASSERT_TRUE( be_monitor != nullptr );
            ASSERT_TRUE( ::st_BeamMonitor_get_out_address( be_monitor ) !=
                         addr_t{ 0 } );

            addr_t const cmp_addr = static_cast< addr_t >( reinterpret_cast<
                uintptr_t >( ::st_Particles_buffer_get_particles(
                    out_buffer, out_particles_offset ) ) );

            ASSERT_TRUE( cmp_addr != addr_t{ 0 }  );
            ASSERT_TRUE( ::st_BeamMonitor_get_out_address( be_monitor ) == cmp_addr );

            ASSERT_TRUE( ::st_BeamMonitor_get_min_particle_id( be_monitor ) >=
                         index_t{ 0 } );

            ASSERT_TRUE( ::st_BeamMonitor_get_min_particle_id( be_monitor ) <=
                         static_cast< index_t >( min_particle_id ) );

            ASSERT_TRUE( ::st_BeamMonitor_get_max_particle_id( be_monitor ) >=
                         static_cast< index_t >( max_particle_id ) );

            ++out_particles_offset;
        }
    }

    ::st_BeamMonitor_clear_all( eb );

    eb_it = eb_begin;

    for( ; eb_it != eb_end ; ++eb_it )
    {
        if( ::st_Object_get_type_id( eb_it ) == ::st_OBJECT_TYPE_BEAM_MONITOR )
        {
            ::st_BeamMonitor const* be_monitor = reinterpret_cast<
                ::st_BeamMonitor const* >( ::st_Object_get_const_begin_ptr(
                    eb_it ) );

            ASSERT_TRUE( be_monitor != nullptr );
            ASSERT_TRUE( ::st_BeamMonitor_get_out_address( be_monitor ) ==
                         addr_t{ 0 } );
        }
    }

    /* --------------------------------------------------------------------- */
    /* reserve out_buffer buffer with element by element buffer */

    ASSERT_TRUE( 0 == ::st_BeamMonitor_prepare_particles_out_buffer(
        eb, out_buffer, particles, 1u ) );

    size_t num_out_particles_block =
        ::st_Particles_buffer_get_num_of_particle_blocks( out_buffer );

    ASSERT_TRUE( num_out_particles_block ==
        ( size_t{ 1 } + NUM_BEAM_MONITORS ) );

    ASSERT_TRUE( 0 == ::st_BeamMonitor_assign_particles_out_buffer(
        eb, out_buffer, 1u ) );

    eb_it = eb_begin;
    out_particles_offset = size_t{ 1 };

    for( size_t jj = size_t{ 0 } ; eb_it != eb_end ; ++eb_it )
    {
        if( ::st_Object_get_type_id( eb_it ) == ::st_OBJECT_TYPE_BEAM_MONITOR )
        {
            ::st_BeamMonitor const* be_monitor = reinterpret_cast<
                ::st_BeamMonitor const* >( ::st_Object_get_const_begin_ptr(
                    eb_it ) );

            ASSERT_TRUE( be_monitor != nullptr );
            ASSERT_TRUE( ::st_BeamMonitor_get_out_address( be_monitor ) !=
                         addr_t{ 0 } );

            addr_t const cmp_addr = static_cast< addr_t >( reinterpret_cast<
                uintptr_t >( ::st_Particles_buffer_get_particles(
                    out_buffer, out_particles_offset ) ) );

            ASSERT_TRUE( cmp_addr != addr_t{ 0 }  );
            ASSERT_TRUE( cmp_beam_monitors.size() > jj );

            ASSERT_TRUE( ::st_BeamMonitor_get_out_address( be_monitor ) ==
                         cmp_addr );

            ASSERT_TRUE( ::st_BeamMonitor_get_min_particle_id( be_monitor ) >=
                         index_t{ 0 } );

            ASSERT_TRUE( ::st_BeamMonitor_get_min_particle_id( be_monitor ) <=
                         static_cast< index_t >( min_particle_id ) );

            ASSERT_TRUE( ::st_BeamMonitor_get_max_particle_id( be_monitor ) >=
                         static_cast< index_t >( max_particle_id ) );

            ++out_particles_offset;
            ++jj;
        }
    }

    ::st_BeamMonitor_clear_all( eb );

    eb_it = eb_begin;

    for( ; eb_it != eb_end ; ++eb_it )
    {
        if( ::st_Object_get_type_id( eb_it ) == ::st_OBJECT_TYPE_BEAM_MONITOR )
        {
            ::st_BeamMonitor const* be_monitor = reinterpret_cast<
                ::st_BeamMonitor const* >( ::st_Object_get_const_begin_ptr(
                    eb_it ) );

            ASSERT_TRUE( be_monitor != nullptr );
            ASSERT_TRUE( ::st_BeamMonitor_get_out_address(
                be_monitor ) == addr_t{ 0 } );
        }
    }

    /* --------------------------------------------------------------------- */
    /* reserve out_buffer buffer with custom offset - only half the elements plus one
     * will be dumped element-by-element style  */

    ::st_Buffer_reset( out_buffer );

    size_t const num_elem_by_elem_blocks = ( NUM_BEAM_ELEMENTS / 2 ) + size_t{ 1 };
    size_t const target_num_out_blocks = NUM_BEAM_MONITORS + num_elem_by_elem_blocks;
    size_t const stored_num_particles  = static_cast< size_t >(
        max_particle_id + index_t{ 1 } );

    for( size_t jj = size_t{ 0 } ; jj < num_elem_by_elem_blocks ; ++jj )
    {
        ::st_Particles* out_particles =
            ::st_Particles_new( out_buffer, stored_num_particles );

        ASSERT_TRUE( out_particles != nullptr );
    }

    eb_it = eb_begin;

    for( ; eb_it != eb_end ; ++eb_it )
    {
        if( ::st_Object_get_type_id( eb_it ) == ::st_OBJECT_TYPE_BEAM_MONITOR )
        {
            ::st_BeamMonitor const* be_monitor = reinterpret_cast<
                ::st_BeamMonitor const* >( ::st_Object_get_const_begin_ptr(
                    eb_it ) );

            ASSERT_TRUE( be_monitor != nullptr );

            nturn_t const num_stores =
                ::st_BeamMonitor_get_num_stores( be_monitor );

            if( num_stores > nturn_t{ 0 } )
            {
                ::st_Particles* out_particles = ::st_Particles_new( out_buffer,
                        stored_num_particles * num_stores );

                ASSERT_TRUE( out_particles != nullptr );
            }
        }
    }

    num_out_particles_block =
        ::st_Particles_buffer_get_num_of_particle_blocks( out_buffer );

    ASSERT_TRUE( num_out_particles_block == target_num_out_blocks );

    ASSERT_TRUE( 0 == ::st_BeamMonitor_setup_for_particles_all( eb, particles ) );
    ASSERT_TRUE( 0 == ::st_BeamMonitor_assign_particles_out_buffer_from_offset(
        eb, out_buffer, num_elem_by_elem_blocks ) );

    eb_it = eb_begin;
    out_particles_offset = num_elem_by_elem_blocks;

    for( size_t jj = size_t{ 0 } ; eb_it != eb_end ; ++eb_it )
    {
        if( ::st_Object_get_type_id( eb_it ) == ::st_OBJECT_TYPE_BEAM_MONITOR )
        {
            ::st_BeamMonitor const* be_monitor = reinterpret_cast<
                ::st_BeamMonitor const* >( ::st_Object_get_const_begin_ptr(
                    eb_it ) );

            ASSERT_TRUE( be_monitor != nullptr );
            ASSERT_TRUE( ::st_BeamMonitor_get_out_address( be_monitor ) !=
                         addr_t{ 0 } );

            addr_t const cmp_addr = static_cast< addr_t >( reinterpret_cast<
                uintptr_t >( ::st_Particles_buffer_get_particles(
                    out_buffer, out_particles_offset ) ) );

            ASSERT_TRUE( cmp_addr != addr_t{ 0 }  );
            ASSERT_TRUE( cmp_beam_monitors.size() > jj );

            ASSERT_TRUE( ::st_BeamMonitor_get_out_address( be_monitor ) ==
                         cmp_addr );

            ASSERT_TRUE( ::st_BeamMonitor_get_min_particle_id( be_monitor ) >=
                         index_t{ 0 } );

            ASSERT_TRUE( ::st_BeamMonitor_get_min_particle_id( be_monitor ) <=
                         static_cast< index_t >( min_particle_id ) );

            ASSERT_TRUE( ::st_BeamMonitor_get_max_particle_id( be_monitor ) >=
                         static_cast< index_t >( max_particle_id ) );

            ++out_particles_offset;
            ++jj;
        }
    }

    ::st_BeamMonitor_clear_all( eb );

    eb_it = eb_begin;

    for( ; eb_it != eb_end ; ++eb_it )
    {
        if( ::st_Object_get_type_id( eb_it ) == ::st_OBJECT_TYPE_BEAM_MONITOR )
        {
            ::st_BeamMonitor const* be_monitor = reinterpret_cast<
                ::st_BeamMonitor const* >( ::st_Object_get_const_begin_ptr(
                    eb_it ) );

            ASSERT_TRUE( be_monitor != nullptr );
            ASSERT_TRUE( ::st_BeamMonitor_get_out_address(
                be_monitor ) == addr_t{ 0 } );
        }
    }

    /* --------------------------------------------------------------------- */

    ::st_Buffer_delete( eb );
    ::st_Buffer_delete( pb );
    ::st_Buffer_delete( out_buffer );
}

TEST( C99_CommonBeamMonitorTests, TrackingAndTurnByTurnIO )
{
    using real_t          = ::st_particle_real_t;
    using part_index_t    = ::st_particle_index_t;
    using mon_index_t     = ::st_be_monitor_index_t;
    using size_t          = ::st_buffer_size_t;
    using nturn_t         = ::st_be_monitor_turn_t;
    using addr_t          = ::st_be_monitor_addr_t;
    using turn_dist_t     = std::uniform_int_distribution< nturn_t >;
    using chance_dist_t   = std::uniform_real_distribution< real_t >;
    using type_id_t       = ::st_object_type_id_t;
    using beam_monitor_t  = ::st_BeamMonitor;
    using ptr_const_mon_t = beam_monitor_t const*;
    using ptr_particles_t = ::st_Particles const*;
    using num_elem_t      = ::st_particle_num_elements_t;

    std::mt19937_64::result_type const seed = 20181031u;

    std::mt19937_64 prng;
    prng.seed( seed );

    ::st_Buffer* eb = ::st_Buffer_new( 0u );
    ::st_Buffer* pb = ::st_Buffer_new( 0u );
    ::st_Buffer* out_buffer = ::st_Buffer_new( 0u );
    ::st_Buffer* elem_by_elem_buffer = ::st_Buffer_new( 0u );

    size_t const NUM_BEAM_MONITORS  = size_t{ 10 };
    size_t const NUM_DRIFTS         = size_t{ 40 };
    size_t const NUM_BEAM_ELEMENTS  = NUM_DRIFTS + NUM_BEAM_MONITORS;
    size_t const NUM_PARTICLES      = size_t{  2 };
    size_t const DRIFT_SEQU_LEN     = NUM_DRIFTS / NUM_BEAM_MONITORS;

    turn_dist_t num_stores_dist( 1, 8 );
    turn_dist_t start_dist( 0, 4 );
    turn_dist_t skip_dist( 1, 4 );

    static real_t const ABS_TOLERANCE = real_t{ 1e-13 };

    chance_dist_t rolling_dist( 0., 1. );

    nturn_t max_num_turns  = nturn_t{ 0 };
    nturn_t max_start_turn = nturn_t{ 0 };
    size_t  required_num_particle_blocks = size_t{ 0 };

    for( size_t ii = size_t{ 0 } ; ii < NUM_BEAM_MONITORS ; ++ii )
    {
        ::st_BeamMonitor* be_monitor = ::st_BeamMonitor_add( eb,
            num_stores_dist( prng ), start_dist( prng ), skip_dist( prng ),
            addr_t{ 0 }, mon_index_t{ 0 }, mon_index_t{ 0 },
            bool{ rolling_dist( prng ) >= 0.5 }, true );

        ASSERT_TRUE( be_monitor != nullptr );

        nturn_t const num_stores =
            ::st_BeamMonitor_get_num_stores( be_monitor );

        nturn_t const skip  = ::st_BeamMonitor_get_skip( be_monitor );
        nturn_t const start = ::st_BeamMonitor_get_start( be_monitor );
        nturn_t const n     = num_stores * skip;

        ASSERT_TRUE( num_stores > nturn_t{ 0 } );

        ++required_num_particle_blocks;

        if( max_num_turns  < n     ) max_num_turns  = n;
        if( max_start_turn < start ) max_start_turn = start;

        for( size_t jj = size_t{ 0 } ; jj < DRIFT_SEQU_LEN ; ++jj )
        {
            ::st_Drift*  drift = ::st_Drift_add( eb, real_t{ 1.0 } );
            ASSERT_TRUE( drift != nullptr );
        }
    }

    ASSERT_TRUE( ::st_Buffer_get_num_of_objects( eb ) == NUM_BEAM_ELEMENTS );

    /* --------------------------------------------------------------------- */

    ::st_Particles* particles = ::st_Particles_new( pb, NUM_PARTICLES );
    ::st_Particles_realistic_init( particles );

    part_index_t min_particle_id = std::numeric_limits< part_index_t >::max();
    part_index_t max_particle_id = std::numeric_limits< part_index_t >::min();

    ASSERT_TRUE( 0 == ::st_Particles_get_min_max_particle_id(
        particles, &min_particle_id, &max_particle_id ) );

    ASSERT_TRUE( min_particle_id >= part_index_t{ 0 } );
    ASSERT_TRUE( max_particle_id >= min_particle_id   );

    size_t const num_stored_particles = static_cast< size_t >(
        max_particle_id + part_index_t{ 1 } );

    ASSERT_TRUE( max_num_turns > nturn_t{ 0 } );

    nturn_t const NUM_TURNS = max_start_turn + 2 * max_num_turns;

    int const ret = ::st_BeamMonitor_prepare_particles_out_buffer(
        eb, out_buffer, particles, 0u );

    ASSERT_TRUE( ret == 0 );

    ::st_Particles* initial_state =
        ::st_Particles_add_copy( elem_by_elem_buffer, particles );

    ASSERT_TRUE( initial_state != nullptr );

    ::st_Particles* final_state =
        ::st_Particles_add_copy( elem_by_elem_buffer, particles );

    ASSERT_TRUE( final_state != nullptr );

    ::st_Particles* elem_by_elem_particles = ::st_Particles_new(
        elem_by_elem_buffer, NUM_TURNS * NUM_BEAM_ELEMENTS * NUM_PARTICLES );

    ASSERT_TRUE( elem_by_elem_particles != nullptr );

    initial_state = ::st_Particles_buffer_get_particles( elem_by_elem_buffer, 0u );
    final_state   = ::st_Particles_buffer_get_particles( elem_by_elem_buffer, 1u );

    ASSERT_TRUE( 0 == ::st_Track_all_particles_element_by_element_until_turn(
        particles, eb, NUM_TURNS, elem_by_elem_particles ) );

    /* --------------------------------------------------------------------- */

    ::st_Particles_copy( final_state, particles );
    ::st_Particles_copy( particles, initial_state );

    ASSERT_TRUE( 0 == ::st_BeamMonitor_assign_particles_out_buffer(
        eb, out_buffer, 0u ) );

    size_t out_particle_block_index = size_t{ 0 };

    for( size_t ii = size_t{ 0 } ; ii < NUM_BEAM_ELEMENTS ; ++ii )
    {
        ::st_Object const* obj_it = ::st_Buffer_get_const_object( eb, ii );
        type_id_t const type_id   = ::st_Object_get_type_id( obj_it );

        if( type_id == ::st_OBJECT_TYPE_BEAM_MONITOR )
        {
            ::st_BeamMonitor const* monitor = reinterpret_cast<
                ::st_BeamMonitor const* >(
                    ::st_Object_get_const_begin_ptr( obj_it ) );

            ASSERT_TRUE( monitor != nullptr );
            nturn_t const num_stores = ::st_BeamMonitor_get_num_stores( monitor );

            ASSERT_TRUE( out_particle_block_index <
                ::st_Buffer_get_num_of_objects( out_buffer ) );

            ::st_Particles const* out_particles =
                ::st_Particles_buffer_get_const_particles(
                    out_buffer, out_particle_block_index );

            ASSERT_TRUE( ::st_BeamMonitor_get_out_address( monitor ) != addr_t{ 0 } );
            ASSERT_TRUE( ::st_BeamMonitor_get_out_address( monitor ) == static_cast<
                addr_t >( reinterpret_cast< uintptr_t >( out_particles ) ) );

            ASSERT_TRUE( num_stores > nturn_t{ 0 } );

            num_elem_t const num_out_particles =
                ::st_Particles_get_num_of_particles( out_particles );

            ASSERT_TRUE( num_out_particles >= static_cast< num_elem_t >(
                num_stored_particles * num_stores ) );

            ++out_particle_block_index;
        }
    }

    /* --------------------------------------------------------------------- */
    ::st_Track_all_particles_until_turn( particles, eb, NUM_TURNS );

    ::st_Buffer* cmp_particles_buffer = ::st_Buffer_new( 0u );

    ::st_Particles* cmp_particles =
        ::st_Particles_new( cmp_particles_buffer, NUM_PARTICLES );

    ASSERT_TRUE( cmp_particles != nullptr );

    if( 0 != st_Particles_compare_values_with_treshold(
            particles, final_state, ABS_TOLERANCE ) )
    {
        ::st_Buffer* diff_buffer = ::st_Buffer_new( 0u );
        ::st_Particles* diff = ::st_Particles_new( diff_buffer, NUM_PARTICLES );

        ::st_Particles_calculate_difference( final_state, particles, diff );

        std::cout << "final_state: " << std::endl;
        ::st_Particles_print_out( final_state );

        std::cout << std::endl << "particles (tracked):" << std::endl;
        ::st_Particles_print_out( particles );

        std::cout << std::endl << "diff: " << std::endl;
        ::st_Particles_print_out( diff );

        diff = nullptr;
        ::st_Buffer_delete( diff_buffer );
    }

    ASSERT_TRUE( 0 == st_Particles_compare_values_with_treshold(
            particles, final_state, ABS_TOLERANCE ) );


    for( nturn_t kk = nturn_t{ 0 } ; kk < NUM_TURNS ; ++kk )
    {
        for( size_t jj = size_t{ 0 } ; jj < NUM_BEAM_ELEMENTS; ++jj )
        {
            ::st_Object const* obj_it = ::st_Buffer_get_const_object( eb, jj );

            if( ::st_Object_get_type_id( obj_it ) == ::st_OBJECT_TYPE_BEAM_MONITOR )
            {
                ptr_const_mon_t beam_monitor = reinterpret_cast< ptr_const_mon_t >(
                        ::st_Object_get_const_begin_ptr( obj_it ) );

                ASSERT_TRUE( ::st_BeamMonitor_get_out_address( beam_monitor )
                    != addr_t{ 0 } );

                if( !::st_BeamMonitor_has_turn_stored(
                        beam_monitor, kk, NUM_TURNS ) )
                {
                    continue;
                }

                ASSERT_TRUE( ::st_BeamMonitor_get_start( beam_monitor ) <= kk );
                ASSERT_TRUE( ( ( kk - ::st_BeamMonitor_get_start( beam_monitor ) )
                    % ::st_BeamMonitor_get_skip( beam_monitor ) ) == nturn_t{ 0 } );

                ptr_particles_t out_particles = reinterpret_cast< ptr_particles_t >(
                    static_cast< uintptr_t >( ::st_BeamMonitor_get_out_address(
                        beam_monitor ) ) );

                ASSERT_TRUE( elem_by_elem_particles != nullptr );

                for( size_t ll = size_t{ 0 } ; ll < NUM_PARTICLES ; ++ll )
                {
                    part_index_t const particle_id =
                        ::st_Particles_get_particle_id_value( particles, ll );

                    num_elem_t const elem_by_elem_index =
                        ::st_Track_element_by_element_get_out_particle_index(
                            min_particle_id, max_particle_id, particle_id,
                            0, NUM_BEAM_ELEMENTS - size_t{ 1 }, jj,
                            0, NUM_TURNS, kk, 0 );

                    ASSERT_TRUE( elem_by_elem_index >= num_elem_t{ 0 } );
                    ASSERT_TRUE( elem_by_elem_index <
                        ::st_Particles_get_num_of_particles(
                            elem_by_elem_particles ) );

                    ASSERT_TRUE( ::st_Particles_copy_single(
                        particles, ll, elem_by_elem_particles, elem_by_elem_index ) );

                    num_elem_t const stored_particle_id =
                        ::st_BeamMonitor_get_store_particle_index(
                            beam_monitor, kk, particle_id );

                    ASSERT_TRUE( stored_particle_id >= num_elem_t{ 0 } );
                    ASSERT_TRUE( ::st_Particles_copy_single( cmp_particles, ll,
                                    out_particles, stored_particle_id ) );
                }

                if( 0 != ::st_Particles_compare_values_with_treshold(
                        cmp_particles, particles, ABS_TOLERANCE ) )
                {
                    std::cout << "jj = " << jj << std::endl;

                    ::st_Buffer* diff_buffer = ::st_Buffer_new( 0u );
                    ::st_Particles* diff =
                        ::st_Particles_new( diff_buffer, NUM_PARTICLES );

                    ASSERT_TRUE( diff != nullptr );

                    ::st_Particles_calculate_difference(
                        cmp_particles, particles, diff );

                    std::cout << "cmp_particles: " << std::endl;
                    ::st_Particles_print_out( cmp_particles );

                    std::cout << std::endl << "elem_by_elem_particles: "
                                << std::endl;

                    ::st_Particles_print_out( particles );

                    std::cout << std::endl << "diff: " << std::endl;
                    ::st_Particles_print_out( diff );

                    ::st_Buffer_delete( diff_buffer );
                    diff_buffer = nullptr;
                }

                ASSERT_TRUE( ::st_Particles_compare_values_with_treshold(
                    cmp_particles, particles, ABS_TOLERANCE ) == 0 );
            }
        }
    }

    ::st_Buffer_delete( eb );
    ::st_Buffer_delete( pb );
    ::st_Buffer_delete( out_buffer );
    ::st_Buffer_delete( elem_by_elem_buffer );
    ::st_Buffer_delete( cmp_particles_buffer );
}

/* end: tests/sixtracklib/common/test_be_monitor_c99.cpp */
