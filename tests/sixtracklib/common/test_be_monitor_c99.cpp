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

#include <gtest/gtest.h>

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/generated/path.h"
#include "sixtracklib/common/buffer.h"
#include "sixtracklib/common/be_drift/be_drift.h"
#include "sixtracklib/common/be_monitor/be_monitor.h"
#include "sixtracklib/common/be_monitor/io_buffer.h"
#include "sixtracklib/common/be_monitor/track.h"
#include "sixtracklib/common/track.h"

#include "sixtracklib/testlib/common/particles.h"

TEST( C99_CommonBeamMonitorTests, MinimalAddToBufferCopyRemapRead )
{
    using size_t        = ::st_buffer_size_t;
    using raw_t         = unsigned char;
    using nturn_t       = ::st_be_monitor_turn_t;
    using addr_t        = ::st_be_monitor_addr_t;

    std::vector< ::st_BeamMonitor > cmp_beam_monitors;

    ::st_Buffer* eb = ::st_Buffer_new( 0u );
    ::st_Buffer* io = ::st_Buffer_new( 0u );

    for( size_t ii = size_t{ 0 } ; ii < size_t{ 20 } ; ++ii )
    {
        ::st_Particles* io_particles = ::st_Particles_new( io, 2u );
        ASSERT_TRUE( io_particles != nullptr );
    }

    addr_t const io_addr_0 = reinterpret_cast< uintptr_t >(
        ::st_Particles_buffer_get_particles( io, 0u ) );

    addr_t const io_addr_10 = reinterpret_cast< uintptr_t >(
        ::st_Particles_buffer_get_particles( io, 10u ) );

    size_t const io_stride  = ( size_t )0u;

    ASSERT_TRUE( nullptr != ::st_Particles_buffer_get_particles( io, 0u ) );

    ::st_BeamMonitor* be_monitor = ::st_BeamMonitor_new( eb );

    ASSERT_TRUE( be_monitor != nullptr );
    ASSERT_TRUE(  ::st_BeamMonitor_get_num_stores( be_monitor ) == nturn_t{ 0 } );
    ASSERT_TRUE(  ::st_BeamMonitor_get_start( be_monitor ) == nturn_t{ 0 } );
    ASSERT_TRUE(  ::st_BeamMonitor_get_skip( be_monitor ) == nturn_t{ 1 } );
    ASSERT_TRUE( !::st_BeamMonitor_is_rolling( be_monitor ) );
    ASSERT_TRUE(  ::st_BeamMonitor_are_attributes_continous( be_monitor ) );
    ASSERT_TRUE(  ::st_BeamMonitor_get_io_address( be_monitor ) == addr_t{ 0 } );

    ::st_BeamMonitor_set_num_stores( be_monitor, 10u );
    ::st_BeamMonitor_set_start( be_monitor, 1000u );
    ::st_BeamMonitor_set_skip( be_monitor, 4u );
    ::st_BeamMonitor_set_is_rolling( be_monitor, true );
    ::st_BeamMonitor_set_io_address( be_monitor, io_addr_0 );

    ASSERT_TRUE( be_monitor != nullptr );
    ASSERT_TRUE(  ::st_BeamMonitor_get_num_stores( be_monitor ) == nturn_t{ 10 } );
    ASSERT_TRUE(  ::st_BeamMonitor_get_start( be_monitor ) == nturn_t{ 1000 } );
    ASSERT_TRUE(  ::st_BeamMonitor_get_skip( be_monitor ) == nturn_t{ 4 } );
    ASSERT_TRUE(  ::st_BeamMonitor_is_rolling( be_monitor ) );
    ASSERT_TRUE(  ::st_BeamMonitor_are_attributes_continous( be_monitor ) );
    ASSERT_TRUE(  ::st_BeamMonitor_get_io_address( be_monitor ) == io_addr_0 );

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

    ASSERT_TRUE(  ::st_BeamMonitor_are_attributes_continous( copy_be_monitor ) ==
                  ::st_BeamMonitor_are_attributes_continous(
                      &cmp_beam_monitors[ 0 ] ) );

    ASSERT_TRUE(  ::st_BeamMonitor_get_io_address( copy_be_monitor ) ==
                  ::st_BeamMonitor_get_io_address( &cmp_beam_monitors[ 0 ] ) );

    be_monitor = ::st_BeamMonitor_add(
        eb, 10u, 5000u, 1u, io_addr_10, io_stride, false, true );

    ASSERT_TRUE( be_monitor != nullptr );
    ASSERT_TRUE(  ::st_BeamMonitor_get_num_stores( be_monitor ) == nturn_t{ 10 } );
    ASSERT_TRUE(  ::st_BeamMonitor_get_start( be_monitor ) == nturn_t{ 5000 } );
    ASSERT_TRUE(  ::st_BeamMonitor_get_skip( be_monitor ) == nturn_t{ 1 } );
    ASSERT_TRUE( !::st_BeamMonitor_is_rolling( be_monitor ) );
    ASSERT_TRUE(  ::st_BeamMonitor_are_attributes_continous( be_monitor ) );
    ASSERT_TRUE(  ::st_BeamMonitor_get_io_address( be_monitor ) == io_addr_10 );

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

        ASSERT_TRUE( ::st_BeamMonitor_are_attributes_continous( ptr_bemon ) ==
                     ::st_BeamMonitor_are_attributes_continous( &cmp_be_monitor ) );

        ASSERT_TRUE( ::st_BeamMonitor_get_io_address( ptr_bemon ) !=
                     ::st_BeamMonitor_get_io_address( &cmp_be_monitor ) );

        ASSERT_TRUE( ::st_BeamMonitor_get_io_address( ptr_bemon ) == addr_t{ 0 } );
    }

    ::st_Buffer_delete( eb );
    ::st_Buffer_delete( io );
    ::st_Buffer_free( &cmp_eb );
}

TEST( C99_CommonBeamMonitorTests, AssignIoBufferToBeamMonitors )
{
    using real_t        = ::st_particle_real_t;
    using size_t        = ::st_buffer_size_t;
    using nturn_t       = ::st_be_monitor_turn_t;
    using addr_t        = ::st_be_monitor_addr_t;
    using turn_dist_t   = std::uniform_int_distribution< nturn_t >;
    using chance_dist_t = std::uniform_real_distribution< real_t >;

    std::mt19937_64::result_type const seed = 20181031u;

    std::mt19937_64 prng;
    prng.seed( seed );

    ::st_Buffer* eb = ::st_Buffer_new( 0u );
    ::st_Buffer* io = ::st_Buffer_new( 0u );

    size_t const NUM_BEAM_MONITORS  = size_t{  10 };
    size_t const NUM_DRIFTS         = size_t{ 100 };
    size_t const NUM_PARTICLES      = size_t{   2 };
    size_t const DRIFT_SEQU_LEN     = NUM_DRIFTS / NUM_BEAM_MONITORS;

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
            addr_t{ 0 }, size_t{ 0 }, bool{ rolling_dist( prng ) >= 0.5 }, true );

        ASSERT_TRUE( be_monitor != nullptr );

        sum_num_of_stores += ::st_BeamMonitor_get_num_stores( be_monitor );
        cmp_beam_monitors.push_back( *be_monitor );
    }

    ASSERT_TRUE( ::st_BeamMonitor_get_num_elem_by_elem_objects( eb ) ==
                 ( NUM_DRIFTS + NUM_BEAM_MONITORS ) );

    ASSERT_TRUE( ::st_BeamMonitor_get_num_of_beam_monitor_objects( eb ) ==
                 NUM_BEAM_MONITORS );

    /* --------------------------------------------------------------------- */
    /* reserve io buffer without element by element buffer */

    ASSERT_TRUE( 0 == ::st_BeamMonitor_prepare_io_buffer(
        eb, io, NUM_PARTICLES, 0u ) );

    ASSERT_TRUE( sum_num_of_stores ==
        ::st_Particles_buffer_get_num_of_particle_blocks( io ) );

    ASSERT_TRUE( ( sum_num_of_stores * NUM_PARTICLES ) == static_cast< size_t >(
        ::st_Particles_buffer_get_total_num_of_particles( io ) ) );

    ASSERT_TRUE( 0 == ::st_BeamMonitor_assign_io_buffer(
        eb, io, NUM_PARTICLES, false ) );

    ::st_Object const* eb_begin = ::st_Buffer_get_const_objects_begin( eb );
    ::st_Object const* eb_end   = ::st_Buffer_get_const_objects_end( eb );
    ::st_Object const* eb_it    = eb_begin;

    size_t io_particles_offset = size_t{ 0 };

    for( size_t jj = size_t{ 0 } ; eb_it != eb_end ; ++eb_it )
    {
        if( ::st_Object_get_type_id( eb_it ) == ::st_OBJECT_TYPE_BEAM_MONITOR )
        {
            ::st_BeamMonitor const* be_monitor = reinterpret_cast<
                ::st_BeamMonitor const* >( ::st_Object_get_const_begin_ptr(
                    eb_it ) );

            ASSERT_TRUE( be_monitor != nullptr );
            ASSERT_TRUE( ::st_BeamMonitor_get_io_address( be_monitor ) !=
                         addr_t{ 0 } );

            addr_t const cmp_addr = static_cast< addr_t >( reinterpret_cast<
                uintptr_t >( ::st_Particles_buffer_get_particles(
                    io, io_particles_offset ) ) );

            ASSERT_TRUE( cmp_addr != addr_t{ 0 }  );
            ASSERT_TRUE( ::st_BeamMonitor_get_io_address( be_monitor ) == cmp_addr );
            ASSERT_TRUE( cmp_beam_monitors.size() > jj );

            io_particles_offset += ::st_BeamMonitor_get_num_stores(
                &cmp_beam_monitors[ jj++ ] );
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
            ASSERT_TRUE( ::st_BeamMonitor_get_io_address( be_monitor ) == addr_t{ 0 } );
        }
    }

    /* --------------------------------------------------------------------- */
    /* reserve io buffer with element by element buffer */

    ASSERT_TRUE( 0 == ::st_BeamMonitor_prepare_io_buffer(
        eb, io, NUM_PARTICLES, 1u ) );

    size_t num_io_particles_block =
        ::st_Particles_buffer_get_num_of_particle_blocks( io );

    ASSERT_TRUE(  num_io_particles_block ==
        ( sum_num_of_stores + NUM_DRIFTS + NUM_BEAM_MONITORS ) );

    ASSERT_TRUE( 0 == ::st_BeamMonitor_assign_io_buffer(
        eb, io, NUM_PARTICLES, true ) );

    eb_it = eb_begin;
    io_particles_offset = NUM_DRIFTS + NUM_BEAM_MONITORS;

    for( size_t jj = size_t{ 0 } ; eb_it != eb_end ; ++eb_it )
    {
        if( ::st_Object_get_type_id( eb_it ) == ::st_OBJECT_TYPE_BEAM_MONITOR )
        {
            ::st_BeamMonitor const* be_monitor = reinterpret_cast<
                ::st_BeamMonitor const* >( ::st_Object_get_const_begin_ptr(
                    eb_it ) );

            ASSERT_TRUE( be_monitor != nullptr );
            ASSERT_TRUE( ::st_BeamMonitor_get_io_address( be_monitor ) !=
                         addr_t{ 0 } );

            addr_t const cmp_addr = static_cast< addr_t >( reinterpret_cast<
                uintptr_t >( ::st_Particles_buffer_get_particles(
                    io, io_particles_offset ) ) );

            ASSERT_TRUE( cmp_addr != addr_t{ 0 }  );
            ASSERT_TRUE( cmp_beam_monitors.size() > jj );

            ASSERT_TRUE( ::st_BeamMonitor_get_io_address( be_monitor ) ==
                         cmp_addr );

            io_particles_offset += ::st_BeamMonitor_get_num_stores(
                &cmp_beam_monitors[ jj++ ] );
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
            ASSERT_TRUE( ::st_BeamMonitor_get_io_address(
                be_monitor ) == addr_t{ 0 } );
        }
    }

    /* --------------------------------------------------------------------- */
    /* reserve io buffer with custom offset - only half the elements plus one
     * will be dumped element-by-element style  */

    ::st_Buffer_reset( io );

    size_t const num_elem_by_elem_blocks = ( NUM_DRIFTS / 2 ) + size_t{ 1 };
    size_t const target_num_io_blocks = sum_num_of_stores + num_elem_by_elem_blocks;

    for( size_t jj = size_t{ 0 } ; jj < target_num_io_blocks ; ++jj )
    {
        ::st_Particles* io_particles = ::st_Particles_new( io, NUM_PARTICLES );
        ASSERT_TRUE( io_particles != nullptr );
    }

    num_io_particles_block = ::st_Particles_buffer_get_num_of_particle_blocks( io );
    ASSERT_TRUE(  num_io_particles_block == target_num_io_blocks );

    ASSERT_TRUE( 0 == ::st_BeamMonitor_assign_io_buffer_from_offset(
        eb, io, NUM_PARTICLES, num_elem_by_elem_blocks ) );

    eb_it = eb_begin;
    io_particles_offset = num_elem_by_elem_blocks;

    for( size_t jj = size_t{ 0 } ; eb_it != eb_end ; ++eb_it )
    {
        if( ::st_Object_get_type_id( eb_it ) == ::st_OBJECT_TYPE_BEAM_MONITOR )
        {
            ::st_BeamMonitor const* be_monitor = reinterpret_cast<
                ::st_BeamMonitor const* >( ::st_Object_get_const_begin_ptr(
                    eb_it ) );

            ASSERT_TRUE( be_monitor != nullptr );
            ASSERT_TRUE( ::st_BeamMonitor_get_io_address( be_monitor ) !=
                         addr_t{ 0 } );

            addr_t const cmp_addr = static_cast< addr_t >( reinterpret_cast<
                uintptr_t >( ::st_Particles_buffer_get_particles(
                    io, io_particles_offset ) ) );

            ASSERT_TRUE( cmp_addr != addr_t{ 0 }  );
            ASSERT_TRUE( cmp_beam_monitors.size() > jj );

            ASSERT_TRUE( ::st_BeamMonitor_get_io_address( be_monitor ) ==
                         cmp_addr );

            io_particles_offset += ::st_BeamMonitor_get_num_stores(
                &cmp_beam_monitors[ jj++ ] );
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
            ASSERT_TRUE( ::st_BeamMonitor_get_io_address(
                be_monitor ) == addr_t{ 0 } );
        }
    }

    /* --------------------------------------------------------------------- */

    ::st_Buffer_delete( eb );
    ::st_Buffer_delete( io );
}

TEST( C99_CommonBeamMonitorTests, TrackingAndTurnByTurnIO )
{
    using real_t          = ::st_particle_real_t;
    using index_t         = ::st_particle_index_t;
    using size_t          = ::st_buffer_size_t;
    using nturn_t         = ::st_be_monitor_turn_t;
    using addr_t          = ::st_be_monitor_addr_t;
    using turn_dist_t     = std::uniform_int_distribution< nturn_t >;
    using chance_dist_t   = std::uniform_real_distribution< real_t >;
    using type_id_t       = ::st_object_type_id_t;
    using beam_monitor_t  = ::st_BeamMonitor;
    using ptr_const_mon_t = beam_monitor_t const*;
    using ptr_particles_t = ::st_Particles const*;

    std::mt19937_64::result_type const seed = 20181031u;

    std::mt19937_64 prng;
    prng.seed( seed );

    ::st_Buffer* eb = ::st_Buffer_new( 0u );
    ::st_Buffer* io = ::st_Buffer_new( 0u );
    ::st_Buffer* pb = ::st_Buffer_new( 0u );
    ::st_Buffer* cmp_particles_buffer = ::st_Buffer_new( 0u );

    size_t const NUM_BEAM_MONITORS  = size_t{ 10 };
    size_t const NUM_DRIFTS         = size_t{ 40 };
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
            addr_t{ 0 }, size_t{ 0 }, bool{ rolling_dist( prng ) >= 0.5 }, true );

        ASSERT_TRUE( be_monitor != nullptr );

        nturn_t const num_stores =
            ::st_BeamMonitor_get_num_stores( be_monitor );

        nturn_t const skip  = ::st_BeamMonitor_get_skip( be_monitor );
        nturn_t const start = ::st_BeamMonitor_get_start( be_monitor );
        nturn_t const n     = num_stores * skip;

        required_num_particle_blocks += num_stores;

        if( max_num_turns  < n     ) max_num_turns  = n;
        if( max_start_turn < start ) max_start_turn = start;

        for( size_t jj = size_t{ 0 } ; jj < DRIFT_SEQU_LEN ; ++jj )
        {
            ::st_Drift*  drift = ::st_Drift_add( eb, real_t{ 1.0 } );
            ASSERT_TRUE( drift != nullptr );
        }
    }

    /* --------------------------------------------------------------------- */

    ASSERT_TRUE( max_num_turns > nturn_t{ 0 } );

    nturn_t const NUM_TURNS = max_start_turn + 2 * max_num_turns;

    int const ret = ::st_BeamMonitor_prepare_io_buffer(
        eb, io, NUM_PARTICLES, 0u );

    ASSERT_TRUE( 0 == ret );

    ASSERT_TRUE( ::st_Particles_buffer_get_num_of_particle_blocks( io ) >=
                 required_num_particle_blocks );

    ::st_Particles* particles = ::st_Particles_new( pb, NUM_PARTICLES );
    ::st_Particles_realistic_init( particles );

    ::st_Object const* obj_begin = ::st_Buffer_get_const_objects_begin( eb );
    ::st_Object const* obj_end   = ::st_Buffer_get_const_objects_end( eb );

    for( nturn_t ii = nturn_t{ 0 } ; ii < NUM_TURNS ; ++ii )
    {
        ASSERT_TRUE( 0 == ::st_Track_all_particles_append_element_by_element(
            particles, 0u, eb, cmp_particles_buffer ) );

        ::st_Track_all_particles_increment_at_turn( particles, 0u );
    }

    /* --------------------------------------------------------------------- */

    ::st_Particles_copy( particles, ::st_Particles_buffer_get_const_particles(
        cmp_particles_buffer, 0u ) );

    ASSERT_TRUE( 0 == ::st_BeamMonitor_assign_io_buffer(
        eb, io, NUM_PARTICLES, false ) );

    ::st_Object const* obj_it = obj_begin;

    size_t io_particle_block_index = size_t{ 0 };

    for( ; obj_it != obj_end ; ++obj_it )
    {
        type_id_t const type_id = ::st_Object_get_type_id( obj_it );

        if( type_id == ::st_OBJECT_TYPE_BEAM_MONITOR )
        {
            ::st_BeamMonitor const* monitor = reinterpret_cast<
                ::st_BeamMonitor const* >(
                    ::st_Object_get_const_begin_ptr( obj_it ) );

            ASSERT_TRUE( monitor != nullptr );
            nturn_t const num_stores = ::st_BeamMonitor_get_num_stores( monitor );

            ASSERT_TRUE( io_particle_block_index <
                ::st_Buffer_get_num_of_objects( io ) );

            ::st_Particles const* io_particles =
                ::st_Particles_buffer_get_const_particles(
                    io, io_particle_block_index );


            ASSERT_TRUE( ::st_BeamMonitor_get_io_address( monitor ) != addr_t{ 0 } );
            ASSERT_TRUE( ::st_BeamMonitor_get_io_address( monitor ) == static_cast<
                addr_t >( reinterpret_cast< uintptr_t >( io_particles ) ) );

            for( nturn_t kk = 0; kk < num_stores ; ++kk )
            {
                io_particles = ::st_Particles_buffer_get_const_particles(
                    io, io_particle_block_index + kk );

                ASSERT_TRUE( io_particles != nullptr );
                ASSERT_TRUE( ::st_Particles_get_num_of_particles(
                    io_particles ) == NUM_PARTICLES );
            }

            io_particle_block_index += num_stores;
        }
    }

    /* --------------------------------------------------------------------- */

    ::st_Track_all_particles_until_turn( particles, eb, NUM_TURNS );

    obj_it = obj_begin;

    for( ; obj_it != obj_end ; ++obj_it )
    {
        type_id_t const type_id = ::st_Object_get_type_id( obj_it );

        if( type_id == ::st_OBJECT_TYPE_BEAM_MONITOR )
        {
            ptr_const_mon_t beam_monitor = reinterpret_cast< ptr_const_mon_t >(
                    ::st_Object_get_const_begin_ptr( obj_it ) );

            ASSERT_TRUE( ::st_BeamMonitor_get_io_address( beam_monitor )
                != addr_t{ 0 } );

            nturn_t const num_stores =
                ::st_BeamMonitor_get_num_stores( beam_monitor );

            nturn_t const start_turn_value =
                ::st_BeamMonitor_get_start( beam_monitor );

            nturn_t const skip = ::st_BeamMonitor_get_skip( beam_monitor );

            ASSERT_TRUE( num_stores != nturn_t{ 0 } );
            ASSERT_TRUE( skip > nturn_t{ 0 } );
            ASSERT_TRUE( ::st_BeamMonitor_get_io_address( beam_monitor ) !=
                         addr_t{ 0 } );

            for( nturn_t kk = nturn_t{ 0 } ; kk < num_stores ; ++kk )
            {
                ptr_particles_t io_particles =
                    ::st_BeamMonitor_get_const_ptr_particles(
                        beam_monitor, kk );

                SIXTRL_ASSERT( io_particles != nullptr );
                SIXTRL_ASSERT( ::st_Particles_get_num_of_particles(
                    io_particles ) == NUM_PARTICLES );

                index_t prev_state      = index_t{ -1 };
                index_t prev_element_id = index_t{ -1 };

                for( size_t jj = size_t{ 0 } ; jj < NUM_PARTICLES ; ++jj )
                {
                    index_t const state = ::st_Particles_get_state_value(
                        io_particles, jj );

                    index_t const turn = ::st_Particles_get_at_turn_value(
                        io_particles, jj );

                    index_t const elem_id = ::st_Particles_get_at_element_id_value(
                            io_particles, jj );

                    if( !::st_BeamMonitor_is_rolling( beam_monitor ) )
                    {
                        ASSERT_TRUE( ( start_turn_value + ( skip * kk ) ) == turn );
                        ASSERT_TRUE( turn < ( start_turn_value + ( skip * num_stores ) ) );
                    }
                    else
                    {
                        ASSERT_TRUE( turn >= start_turn_value );
                        ASSERT_TRUE( ( ( turn - start_turn_value ) % skip ) == 0 );
                    }

                    size_t const cmp_particle_index =
                        turn * ( NUM_DRIFTS + NUM_BEAM_MONITORS ) + elem_id;

                    ASSERT_TRUE( cmp_particle_index <
                        ::st_Buffer_get_num_of_objects( cmp_particles_buffer ) );

                    ptr_particles_t cmp_particles =
                        ::st_Particles_buffer_get_const_particles(
                            cmp_particles_buffer, cmp_particle_index );

                    ASSERT_TRUE( cmp_particles != nullptr );

                    ASSERT_TRUE( state == index_t{ 1 } );
                    ASSERT_TRUE( ( prev_state == state ) ||
                                 ( prev_state == index_t{ -1 } ) );

                    ASSERT_TRUE( ( prev_element_id == elem_id ) ||
                                 ( prev_element_id == index_t{ -1 } ) );

                    ASSERT_TRUE(
                        ( 0 == ::st_Particles_compare_real_values(
                            cmp_particles, io_particles ) )  ||
                        ( 0 == ::st_Particles_compare_real_values_with_treshold(
                            cmp_particles, io_particles, ABS_TOLERANCE ) ) );

                    prev_state = state;
                    prev_element_id = elem_id;
                }
            }
        }
    }

    ::st_Buffer_delete( eb );
    ::st_Buffer_delete( io );
    ::st_Buffer_delete( pb );
    ::st_Buffer_delete( cmp_particles_buffer );
}

/* end: tests/sixtracklib/common/test_be_monitor_c99.cpp */
