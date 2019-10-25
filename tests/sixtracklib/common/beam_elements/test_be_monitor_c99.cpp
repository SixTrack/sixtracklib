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

#include "sixtracklib/testlib.h"

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/generated/path.h"
#include "sixtracklib/common/buffer.h"
#include "sixtracklib/common/be_drift/be_drift.h"
#include "sixtracklib/common/be_monitor/be_monitor.h"
#include "sixtracklib/common/be_monitor/output_buffer.h"
#include "sixtracklib/common/be_monitor/track.h"
#include "sixtracklib/common/output/elem_by_elem_config.h"
#include "sixtracklib/common/output/output_buffer.h"
#include "sixtracklib/common/track/track.h"


TEST( C99_CommonBeamMonitorTests, MinimalAddToBufferCopyRemapRead )
{
    using size_t        = ::NS(buffer_size_t);
    using raw_t         = unsigned char;
    using nturn_t       = ::NS(be_monitor_turn_t);
    using addr_t        = ::NS(be_monitor_addr_t);
    using index_t       = ::NS(be_monitor_index_t);
    using part_index_t  = ::NS(particle_index_t);

    constexpr size_t NUM_PARTICLES = size_t{ 2 };

    std::vector< ::NS(BeamMonitor) > cmp_beam_monitors;

    ::NS(Buffer)* eb = ::NS(Buffer_new)( 0u );
    ::NS(Buffer)* pb = ::NS(Buffer_new)( 0u );
    ::NS(Buffer)* output_buffer = ::NS(Buffer_new)( 0u );

    ::NS(Particles)* out_particles = nullptr;
    ::NS(Particles)* particles     = ::NS(Particles_new)( pb, NUM_PARTICLES );
    ::NS(Particles_init_particle_ids)( particles );

    part_index_t min_particle_id = std::numeric_limits< part_index_t >::max();
    part_index_t max_particle_id = std::numeric_limits< part_index_t >::min();

    ASSERT_TRUE( ::NS(ARCH_STATUS_SUCCESS) ==
        ::NS(Particles_get_min_max_particle_id)( particles,
            &min_particle_id, &max_particle_id ) );

    ASSERT_TRUE( min_particle_id <= max_particle_id );

    std::vector< nturn_t > num_stores_list = { nturn_t{ 10 }, nturn_t{ 10 } };

    for( auto const nn : num_stores_list )
    {
        out_particles = ::NS(Particles_new)( output_buffer, NUM_PARTICLES * nn );
        ASSERT_TRUE( out_particles != nullptr );
    }

    addr_t const out_addr_0 = reinterpret_cast< uintptr_t >(
        ::NS(Particles_buffer_get_particles)( output_buffer, 0u ) );

    addr_t const out_addr_1 = reinterpret_cast< uintptr_t >(
        ::NS(Particles_buffer_get_particles)( output_buffer, 1u ) );

    ASSERT_TRUE( ( out_addr_0 != addr_t{ 0 } )&& ( out_addr_1 != addr_t{ 0 } ) );

    ::NS(BeamMonitor)* be_monitor = ::NS(BeamMonitor_new)( eb );

    ASSERT_TRUE( be_monitor != nullptr );
    ASSERT_TRUE(  ::NS(BeamMonitor_get_num_stores)( be_monitor ) == nturn_t{ 0 } );
    ASSERT_TRUE(  ::NS(BeamMonitor_get_start)( be_monitor ) == nturn_t{ 0 } );
    ASSERT_TRUE(  ::NS(BeamMonitor_get_skip)( be_monitor ) == nturn_t{ 1 } );
    ASSERT_TRUE( !::NS(BeamMonitor_is_rolling)( be_monitor ) );
    ASSERT_TRUE(  ::NS(BeamMonitor_is_turn_ordered)( be_monitor ) );
    ASSERT_TRUE(  ::NS(BeamMonitor_get_out_address)( be_monitor ) == addr_t{ 0 } );
    ASSERT_TRUE(  ::NS(BeamMonitor_get_min_particle_id)( be_monitor ) == index_t{ 0 } );
    ASSERT_TRUE(  ::NS(BeamMonitor_get_max_particle_id)( be_monitor ) == index_t{ 0 } );

    ::NS(BeamMonitor_set_num_stores)( be_monitor, 10u );
    ::NS(BeamMonitor_set_start)( be_monitor, 1000u );
    ::NS(BeamMonitor_set_skip)( be_monitor, 4u );
    ::NS(BeamMonitor_set_is_rolling)( be_monitor, true );
    ::NS(BeamMonitor_setup_for_particles)( be_monitor, particles );
    ::NS(BeamMonitor_set_out_address)( be_monitor, out_addr_0 );

    ASSERT_TRUE( be_monitor != nullptr );
    ASSERT_TRUE(  ::NS(BeamMonitor_get_num_stores)( be_monitor ) == nturn_t{ 10 } );
    ASSERT_TRUE(  ::NS(BeamMonitor_get_start)( be_monitor ) == nturn_t{ 1000 } );
    ASSERT_TRUE(  ::NS(BeamMonitor_get_skip)( be_monitor ) == nturn_t{ 4 } );
    ASSERT_TRUE(  ::NS(BeamMonitor_is_rolling)( be_monitor ) );
    ASSERT_TRUE(  ::NS(BeamMonitor_is_turn_ordered)( be_monitor ) );
    ASSERT_TRUE(  ::NS(BeamMonitor_get_out_address)( be_monitor ) == out_addr_0 );

    ASSERT_TRUE(  ::NS(BeamMonitor_get_min_particle_id)( be_monitor ) >=
                  index_t{ 0 } );

    ASSERT_TRUE(  ::NS(BeamMonitor_get_min_particle_id)( be_monitor ) <=
                  static_cast< index_t >( min_particle_id ) );

    ASSERT_TRUE( ::NS(BeamMonitor_get_max_particle_id)( be_monitor ) >=
                 ::NS(BeamMonitor_get_min_particle_id)( be_monitor ) );

    ASSERT_TRUE( ::NS(BeamMonitor_get_max_particle_id)( be_monitor ) >=
                 static_cast< index_t >( max_particle_id ) );

    cmp_beam_monitors.push_back( *be_monitor );

    ::NS(BeamMonitor)* copy_be_monitor =
        ::NS(BeamMonitor_add_copy)( eb, &cmp_beam_monitors.back() );

    ASSERT_TRUE( copy_be_monitor != nullptr );
    ASSERT_TRUE( copy_be_monitor != be_monitor );

    cmp_beam_monitors.push_back( *copy_be_monitor );

    ASSERT_TRUE(  ::NS(BeamMonitor_get_num_stores)( copy_be_monitor ) ==
                  ::NS(BeamMonitor_get_num_stores)( &cmp_beam_monitors[ 0 ] ) );

    ASSERT_TRUE(  ::NS(BeamMonitor_get_start)( copy_be_monitor ) ==
                  ::NS(BeamMonitor_get_start)( &cmp_beam_monitors[ 0 ] ) );

    ASSERT_TRUE(  ::NS(BeamMonitor_get_skip)( copy_be_monitor ) ==
                  ::NS(BeamMonitor_get_skip)( &cmp_beam_monitors[ 0 ] ) );

    ASSERT_TRUE(  ::NS(BeamMonitor_is_rolling)( copy_be_monitor ) ==
                  ::NS(BeamMonitor_is_rolling)( &cmp_beam_monitors[ 0 ] ) );

    ASSERT_TRUE(  ::NS(BeamMonitor_is_turn_ordered)( copy_be_monitor ) ==
                  ::NS(BeamMonitor_is_turn_ordered)( &cmp_beam_monitors[ 0 ] ) );

    ASSERT_TRUE(  ::NS(BeamMonitor_is_turn_ordered)( copy_be_monitor ) ==
                  ::NS(BeamMonitor_is_turn_ordered)( &cmp_beam_monitors[ 0 ] ) );

    ASSERT_TRUE(  ::NS(BeamMonitor_get_out_address)( copy_be_monitor ) ==
                  ::NS(BeamMonitor_get_out_address)( &cmp_beam_monitors[ 0 ] ) );

    ASSERT_TRUE(  ::NS(BeamMonitor_get_min_particle_id)( copy_be_monitor ) ==
                  ::NS(BeamMonitor_get_min_particle_id)( &cmp_beam_monitors[ 0 ] ) );

    ASSERT_TRUE(  ::NS(BeamMonitor_get_max_particle_id)( copy_be_monitor ) ==
                  ::NS(BeamMonitor_get_max_particle_id)( &cmp_beam_monitors[ 0 ] ) );

    be_monitor = ::NS(BeamMonitor_add)( eb, 10u, 5000u, 1u, out_addr_1,
        min_particle_id, max_particle_id, false, true );

    ASSERT_TRUE( be_monitor != nullptr );
    ASSERT_TRUE(  ::NS(BeamMonitor_get_num_stores)( be_monitor ) == nturn_t{ 10 } );
    ASSERT_TRUE(  ::NS(BeamMonitor_get_start)( be_monitor ) == nturn_t{ 5000 } );
    ASSERT_TRUE(  ::NS(BeamMonitor_get_skip)( be_monitor ) == nturn_t{ 1 } );
    ASSERT_TRUE( !::NS(BeamMonitor_is_rolling)( be_monitor ) );
    ASSERT_TRUE(  ::NS(BeamMonitor_is_turn_ordered)( be_monitor ) );
    ASSERT_TRUE(  ::NS(BeamMonitor_get_out_address)( be_monitor ) == out_addr_1 );

    ASSERT_TRUE(  ::NS(BeamMonitor_get_min_particle_id)( be_monitor ) ==
                  static_cast< index_t >( min_particle_id ) );

    ASSERT_TRUE(  ::NS(BeamMonitor_get_max_particle_id)( be_monitor ) ==
                  static_cast< index_t >( max_particle_id ) );

    cmp_beam_monitors.push_back( *be_monitor );

    ASSERT_TRUE( cmp_beam_monitors.size() ==
                 ::NS(Buffer_get_num_of_objects)( eb ) );

    std::vector< raw_t > copy_raw_buffer(
        ::NS(Buffer_get_size)( eb ), raw_t{ 0 } );

    copy_raw_buffer.assign( ::NS(Buffer_get_const_data_begin)( eb ),
                            ::NS(Buffer_get_const_data_end)( eb ) );

    ::NS(Buffer) cmp_eb;
    ::NS(Buffer_preset)( &cmp_eb );

    ASSERT_TRUE( ::NS(ARCH_STATUS_SUCCESS) == ::NS(Buffer_init_from_data)(
        &cmp_eb, copy_raw_buffer.data(), copy_raw_buffer.size() ) );

    ASSERT_TRUE( !::NS(Buffer_needs_remapping)( &cmp_eb ) );
    ASSERT_TRUE(  ::NS(Buffer_get_num_of_objects)( &cmp_eb ) ==
                  ::NS(Buffer_get_num_of_objects)( eb ) );

    size_t jj = 0;

    for( auto const& cmp_be_monitor : cmp_beam_monitors )
    {
        ::NS(Object) const* ptr_obj = ::NS(Buffer_get_const_object)( eb, jj++ );

        ASSERT_TRUE( ptr_obj != nullptr );
        ASSERT_TRUE( ::NS(Object_get_type_id)( ptr_obj ) ==
                     ::NS(OBJECT_TYPE_BEAM_MONITOR) );

        ::NS(BeamMonitor) const* ptr_beam_monitor =
            reinterpret_cast< ::NS(BeamMonitor) const* >(
                ::NS(Object_get_const_begin_ptr)( ptr_obj ) );

        ASSERT_TRUE( ptr_beam_monitor != nullptr );
        ASSERT_TRUE( 0 == ::NS(BeamMonitor_compare_values)(
            ptr_beam_monitor, &cmp_be_monitor ) );
    }

    ASSERT_TRUE( ::NS(ElemByElemConfig_get_num_elem_by_elem_objects)( eb ) ==
                 cmp_beam_monitors.size() );

    ASSERT_TRUE( cmp_beam_monitors.size() ==
        ::NS(ElemByElemConfig_get_num_elem_by_elem_objects_from_managed_buffer)(
            ::NS(Buffer_get_const_data_begin)( eb ),
            ::NS(Buffer_get_slot_size)( eb ) ) );

    ASSERT_TRUE( ::NS(BeamMonitor_get_num_of_beam_monitor_objects)( eb ) ==
                 cmp_beam_monitors.size() );

    ASSERT_TRUE( ::NS(BeamMonitor_get_num_of_beam_monitor_objects_from_managed_buffer)(
        ::NS(Buffer_get_const_data_begin)( eb ), ::NS(Buffer_get_slot_size)( eb ) ) ==
                 cmp_beam_monitors.size() );

    ::NS(BeamMonitor_clear_all)( eb );

    jj = 0;

    for( auto const& cmp_be_monitor : cmp_beam_monitors )
    {
        ::NS(Object) const* ptr_obj = ::NS(Buffer_get_const_object)( eb, jj++ );

        ASSERT_TRUE( ptr_obj != nullptr );
        ASSERT_TRUE( ::NS(Object_get_type_id)( ptr_obj ) ==
                     ::NS(OBJECT_TYPE_BEAM_MONITOR) );

        ::NS(BeamMonitor) const* ptr_bemon =
            reinterpret_cast< ::NS(BeamMonitor) const* >(
                ::NS(Object_get_const_begin_ptr)( ptr_obj ) );

        ASSERT_TRUE( ptr_bemon != nullptr );

        ASSERT_TRUE( ::NS(BeamMonitor_get_num_stores)( ptr_bemon ) ==
                     ::NS(BeamMonitor_get_num_stores)( &cmp_be_monitor ) );

        ASSERT_TRUE( ::NS(BeamMonitor_get_start)( ptr_bemon ) ==
                     ::NS(BeamMonitor_get_start)( &cmp_be_monitor ) );

        ASSERT_TRUE( ::NS(BeamMonitor_get_skip)( ptr_bemon ) ==
                     ::NS(BeamMonitor_get_skip)( &cmp_be_monitor ) );

        ASSERT_TRUE( ::NS(BeamMonitor_is_rolling)( ptr_bemon ) ==
                     ::NS(BeamMonitor_is_rolling)( &cmp_be_monitor ) );

        ASSERT_TRUE( ::NS(BeamMonitor_is_turn_ordered)( ptr_bemon ) ==
                     ::NS(BeamMonitor_is_turn_ordered)( &cmp_be_monitor ) );

        ASSERT_TRUE( ::NS(BeamMonitor_get_out_address)( ptr_bemon ) !=
                     ::NS(BeamMonitor_get_out_address)( &cmp_be_monitor ) );

        ASSERT_TRUE( ::NS(BeamMonitor_get_out_address)( ptr_bemon ) ==
                     addr_t{ 0 } );

        ASSERT_TRUE( ::NS(BeamMonitor_get_min_particle_id)( ptr_bemon ) ==
                     index_t{ 0 } );

        ASSERT_TRUE( ::NS(BeamMonitor_get_max_particle_id)( ptr_bemon ) ==
                     index_t{ 0 } );
    }

    ::NS(Buffer_delete)( eb );
    ::NS(Buffer_delete)( pb );
    ::NS(Buffer_delete)( output_buffer );
    ::NS(Buffer_free)( &cmp_eb );
}

TEST( C99_CommonBeamMonitorTests, AssignIoBufferToBeamMonitors )
{
    using real_t        = ::NS(particle_real_t);
    using size_t        = ::NS(buffer_size_t);
    using nturn_t       = ::NS(be_monitor_turn_t);
    using addr_t        = ::NS(be_monitor_addr_t);
    using index_t       = ::NS(be_monitor_index_t);
    using part_index_t  = ::NS(particle_index_t);
    using turn_dist_t   = std::uniform_int_distribution< nturn_t >;
    using chance_dist_t = std::uniform_real_distribution< real_t >;

    std::mt19937_64::result_type const seed = 20181031u;

    std::mt19937_64 prng;
    prng.seed( seed );

    ::NS(Buffer)* eb = ::NS(Buffer_new)( 0u );
    ::NS(Buffer)* pb = ::NS(Buffer_new)( 0u );
    ::NS(Buffer)* out_buffer = ::NS(Buffer_new)( 0u );

    size_t const NUM_BEAM_MONITORS  = size_t{  10 };
    size_t const NUM_DRIFTS         = size_t{ 100 };
    size_t const NUM_PARTICLES      = size_t{   2 };
    size_t const DRIFT_SEQU_LEN     = NUM_DRIFTS / NUM_BEAM_MONITORS;
    size_t const NUM_BEAM_ELEMENTS  = NUM_DRIFTS + NUM_BEAM_MONITORS;

    std::vector< ::NS(BeamMonitor) > cmp_beam_monitors;

    turn_dist_t num_stores_dist( 1, 100 );
    turn_dist_t start_dist( 0, 1000 );
    turn_dist_t skip_dist( 1, 100 );

    chance_dist_t rolling_dist( 0., 1. );
    size_t sum_num_of_stores = size_t{ 0 };

    for( size_t ii = size_t{ 0 } ; ii < NUM_BEAM_MONITORS ; ++ii )
    {
        for( size_t jj = size_t{ 0 } ; jj < DRIFT_SEQU_LEN ; ++jj )
        {
            ::NS(Drift)*  drift = ::NS(Drift_add)( eb, real_t{ 1.0 } );
            ASSERT_TRUE( drift != nullptr );
        }

        ::NS(BeamMonitor)* be_monitor = ::NS(BeamMonitor_add)( eb,
            num_stores_dist( prng ), start_dist( prng ), skip_dist( prng ),
            addr_t{ 0 }, index_t{ 0 }, index_t{ 0 },
            bool{ rolling_dist( prng ) >= 0.5 }, true );

        ASSERT_TRUE( be_monitor != nullptr );

        sum_num_of_stores += ::NS(BeamMonitor_get_num_stores)( be_monitor );
        cmp_beam_monitors.push_back( *be_monitor );
    }

    ASSERT_TRUE( ::NS(ElemByElemConfig_get_num_elem_by_elem_objects)( eb ) ==
                 ( NUM_DRIFTS + NUM_BEAM_MONITORS ) );

    ASSERT_TRUE( ::NS(BeamMonitor_get_num_of_beam_monitor_objects)( eb ) ==
                 NUM_BEAM_MONITORS );

    /* --------------------------------------------------------------------- */
    /* reserve out_buffer buffer without element by element buffer */

    size_t  num_elem_by_elem_turns    = size_t{ 0 };
    size_t  elem_by_elem_index_offset = size_t{ 0 };
    size_t  beam_monitor_index_offset = size_t{ 0 };
    index_t min_turn_id               = index_t{ 0 };

    ::NS(Particles)* particles = ::NS(Particles_new)( pb, NUM_PARTICLES );
    ::NS(Particles_init_particle_ids)( particles );

    part_index_t min_particle_id = std::numeric_limits< part_index_t >::max();
    part_index_t max_particle_id = std::numeric_limits< part_index_t >::min();

    ASSERT_TRUE( ::NS(ARCH_STATUS_SUCCESS) ==
        ::NS(Particles_get_min_max_particle_id)( particles,
            &min_particle_id, &max_particle_id ) );

    ASSERT_TRUE( min_particle_id <= max_particle_id );
    ASSERT_TRUE( min_particle_id >= part_index_t{ 0 } );

    ASSERT_TRUE( ::NS(ARCH_STATUS_SUCCESS) == ::NS(OutputBuffer_prepare)(
        eb, out_buffer, particles, num_elem_by_elem_turns,
        &elem_by_elem_index_offset, &beam_monitor_index_offset,
        &min_turn_id ) );

    ASSERT_TRUE( elem_by_elem_index_offset == size_t{ 0 } );
    ASSERT_TRUE( beam_monitor_index_offset == size_t{ 0 } );
    ASSERT_TRUE( min_turn_id == index_t{ 0 } );

    ASSERT_TRUE( NUM_BEAM_MONITORS ==
        ::NS(Particles_buffer_get_num_of_particle_blocks)( out_buffer ) );

    ASSERT_TRUE( ( sum_num_of_stores * NUM_PARTICLES ) == static_cast< size_t >(
        ::NS(Particles_buffer_get_total_num_of_particles)( out_buffer ) ) );

    ASSERT_TRUE( ::NS(ARCH_STATUS_SUCCESS) ==
        ::NS(BeamMonitor_assign_output_buffer)( eb, out_buffer,
            min_turn_id, num_elem_by_elem_turns ) );

    ::NS(Object) const* eb_begin = ::NS(Buffer_get_const_objects_begin)( eb );
    ::NS(Object) const* eb_end   = ::NS(Buffer_get_const_objects_end)( eb );
    ::NS(Object) const* eb_it    = eb_begin;

    size_t out_particles_offset = size_t{ 0 };

    for( ; eb_it != eb_end ; ++eb_it )
    {
        if( ::NS(Object_get_type_id)( eb_it ) == ::NS(OBJECT_TYPE_BEAM_MONITOR) )
        {
            ::NS(BeamMonitor) const* be_monitor = reinterpret_cast<
                ::NS(BeamMonitor) const* >( ::NS(Object_get_const_begin_ptr)(
                    eb_it ) );

            ASSERT_TRUE( be_monitor != nullptr );
            ASSERT_TRUE( ::NS(BeamMonitor_get_out_address)( be_monitor ) !=
                         addr_t{ 0 } );

            addr_t const cmp_addr = static_cast< addr_t >( reinterpret_cast<
                uintptr_t >( ::NS(Particles_buffer_get_particles)(
                    out_buffer, out_particles_offset ) ) );

            ASSERT_TRUE( cmp_addr != addr_t{ 0 }  );
            ASSERT_TRUE( ::NS(BeamMonitor_get_out_address)( be_monitor ) == cmp_addr );

            ASSERT_TRUE( ::NS(BeamMonitor_get_min_particle_id)( be_monitor ) >=
                         index_t{ 0 } );

            ASSERT_TRUE( ::NS(BeamMonitor_get_min_particle_id)( be_monitor ) <=
                         static_cast< index_t >( min_particle_id ) );

            ASSERT_TRUE( ::NS(BeamMonitor_get_max_particle_id)( be_monitor ) >=
                         static_cast< index_t >( max_particle_id ) );

            ++out_particles_offset;
        }
    }

    ::NS(BeamMonitor_clear_all)( eb );

    eb_it = eb_begin;

    for( ; eb_it != eb_end ; ++eb_it )
    {
        if( ::NS(Object_get_type_id)( eb_it ) == ::NS(OBJECT_TYPE_BEAM_MONITOR) )
        {
            ::NS(BeamMonitor) const* be_monitor = reinterpret_cast<
                ::NS(BeamMonitor) const* >( ::NS(Object_get_const_begin_ptr)(
                    eb_it ) );

            ASSERT_TRUE( be_monitor != nullptr );
            ASSERT_TRUE( ::NS(BeamMonitor_get_out_address)( be_monitor ) ==
                         addr_t{ 0 } );
        }
    }

    /* --------------------------------------------------------------------- */
    /* reserve out_buffer buffer with element by element buffer */

    ::NS(Buffer_reset)( out_buffer );

    num_elem_by_elem_turns    = size_t{ 1 };
    elem_by_elem_index_offset = size_t{ 0 };
    beam_monitor_index_offset = size_t{ 0 };
    min_turn_id = index_t{ 0 };

    ASSERT_TRUE( ::NS(ARCH_STATUS_SUCCESS) == ::NS(OutputBuffer_prepare)(
        eb, out_buffer, particles, num_elem_by_elem_turns,
        &elem_by_elem_index_offset, &beam_monitor_index_offset,
        &min_turn_id ) );

    ASSERT_TRUE( elem_by_elem_index_offset == size_t{ 0 } );
    ASSERT_TRUE( beam_monitor_index_offset == size_t{ 1 } );
    ASSERT_TRUE( min_turn_id == index_t{ 0 } );

    size_t num_out_particles_block =
        ::NS(Particles_buffer_get_num_of_particle_blocks)( out_buffer );

    ASSERT_TRUE( num_out_particles_block ==
        ( size_t{ 1 } + NUM_BEAM_MONITORS ) );

    ASSERT_TRUE( ::NS(ARCH_STATUS_SUCCESS) ==
        ::NS(BeamMonitor_assign_output_buffer)( eb, out_buffer, min_turn_id,
                num_elem_by_elem_turns ) );

    eb_it = eb_begin;
    out_particles_offset = size_t{ 1 };

    for( size_t jj = size_t{ 0 } ; eb_it != eb_end ; ++eb_it )
    {
        if( ::NS(Object_get_type_id)( eb_it ) == ::NS(OBJECT_TYPE_BEAM_MONITOR) )
        {
            ::NS(BeamMonitor) const* be_monitor = reinterpret_cast<
                ::NS(BeamMonitor) const* >( ::NS(Object_get_const_begin_ptr)(
                    eb_it ) );

            ASSERT_TRUE( be_monitor != nullptr );
            ASSERT_TRUE( ::NS(BeamMonitor_get_out_address)( be_monitor ) !=
                         addr_t{ 0 } );

            addr_t const cmp_addr = static_cast< addr_t >( reinterpret_cast<
                uintptr_t >( ::NS(Particles_buffer_get_particles)(
                    out_buffer, out_particles_offset ) ) );

            ASSERT_TRUE( cmp_addr != addr_t{ 0 }  );
            ASSERT_TRUE( cmp_beam_monitors.size() > jj );

            ASSERT_TRUE( ::NS(BeamMonitor_get_out_address)( be_monitor ) ==
                         cmp_addr );

            ASSERT_TRUE( ::NS(BeamMonitor_get_min_particle_id)( be_monitor ) >=
                         index_t{ 0 } );

            ASSERT_TRUE( ::NS(BeamMonitor_get_min_particle_id)( be_monitor ) <=
                         static_cast< index_t >( min_particle_id ) );

            ASSERT_TRUE( ::NS(BeamMonitor_get_max_particle_id)( be_monitor ) >=
                         static_cast< index_t >( max_particle_id ) );

            ++out_particles_offset;
            ++jj;
        }
    }

    ::NS(BeamMonitor_clear_all)( eb );

    eb_it = eb_begin;

    for( ; eb_it != eb_end ; ++eb_it )
    {
        if( ::NS(Object_get_type_id)( eb_it ) == ::NS(OBJECT_TYPE_BEAM_MONITOR) )
        {
            ::NS(BeamMonitor) const* be_monitor = reinterpret_cast<
                ::NS(BeamMonitor) const* >( ::NS(Object_get_const_begin_ptr)(
                    eb_it ) );

            ASSERT_TRUE( be_monitor != nullptr );
            ASSERT_TRUE( ::NS(BeamMonitor_get_out_address)(
                be_monitor ) == addr_t{ 0 } );
        }
    }

    /* --------------------------------------------------------------------- */
    /* reserve out_buffer buffer with custom offset - only half the elements plus one
     * will be dumped element-by-element style  */

    ::NS(Buffer_reset)( out_buffer );

    size_t const num_elem_by_elem_blocks = ( NUM_BEAM_ELEMENTS / 2 ) + size_t{ 1 };
    size_t const target_num_out_blocks = NUM_BEAM_MONITORS + num_elem_by_elem_blocks;
    size_t const stored_num_particles  = static_cast< size_t >(
        max_particle_id + index_t{ 1 } );

    index_t max_turn_id = index_t{ -1 };

    ASSERT_TRUE( ::NS(ARCH_STATUS_SUCCESS) ==
        ::NS(Particles_get_min_max_at_turn_value)( particles,
            &min_turn_id, &max_turn_id ) );

    ASSERT_TRUE( min_turn_id >= index_t{ 0 } );
    ASSERT_TRUE( min_turn_id <= max_turn_id  );

    for( size_t jj = size_t{ 0 } ; jj < num_elem_by_elem_blocks ; ++jj )
    {
        ::NS(Particles)* out_particles =
            ::NS(Particles_new)( out_buffer, stored_num_particles );

        ASSERT_TRUE( out_particles != nullptr );
    }

    eb_it = eb_begin;

    for( ; eb_it != eb_end ; ++eb_it )
    {
        if( ::NS(Object_get_type_id)( eb_it ) == ::NS(OBJECT_TYPE_BEAM_MONITOR) )
        {
            ::NS(BeamMonitor) const* be_monitor = reinterpret_cast<
                ::NS(BeamMonitor) const* >( ::NS(Object_get_const_begin_ptr)(
                    eb_it ) );

            ASSERT_TRUE( be_monitor != nullptr );

            nturn_t const num_stores =
                ::NS(BeamMonitor_get_num_stores)( be_monitor );

            if( num_stores > nturn_t{ 0 } )
            {
                ::NS(Particles)* out_particles = ::NS(Particles_new)( out_buffer,
                        stored_num_particles * num_stores );

                ASSERT_TRUE( out_particles != nullptr );
            }
        }
    }

    num_out_particles_block =
        ::NS(Particles_buffer_get_num_of_particle_blocks)( out_buffer );

    ASSERT_TRUE( num_out_particles_block == target_num_out_blocks );

    ASSERT_TRUE( ::NS(ARCH_STATUS_SUCCESS) ==
        ::NS(BeamMonitor_setup_for_particles_all)( eb, particles ) );

    ASSERT_TRUE( ::NS(ARCH_STATUS_SUCCESS) ==
        ::NS(BeamMonitor_assign_output_buffer_from_offset)( eb, out_buffer,
            min_turn_id, num_elem_by_elem_blocks ) );

    eb_it = eb_begin;
    out_particles_offset = num_elem_by_elem_blocks;

    for( size_t jj = size_t{ 0 } ; eb_it != eb_end ; ++eb_it )
    {
        if( ::NS(Object_get_type_id)( eb_it ) == ::NS(OBJECT_TYPE_BEAM_MONITOR) )
        {
            ::NS(BeamMonitor) const* be_monitor = reinterpret_cast<
                ::NS(BeamMonitor) const* >( ::NS(Object_get_const_begin_ptr)(
                    eb_it ) );

            ASSERT_TRUE( be_monitor != nullptr );
            ASSERT_TRUE( ::NS(BeamMonitor_get_out_address)( be_monitor ) !=
                         addr_t{ 0 } );

            addr_t const cmp_addr = static_cast< addr_t >( reinterpret_cast<
                uintptr_t >( ::NS(Particles_buffer_get_particles)(
                    out_buffer, out_particles_offset ) ) );

            ASSERT_TRUE( cmp_addr != addr_t{ 0 }  );
            ASSERT_TRUE( cmp_beam_monitors.size() > jj );

            ASSERT_TRUE( ::NS(BeamMonitor_get_out_address)( be_monitor ) ==
                         cmp_addr );

            ASSERT_TRUE( ::NS(BeamMonitor_get_min_particle_id)( be_monitor ) >=
                         index_t{ 0 } );

            ASSERT_TRUE( ::NS(BeamMonitor_get_min_particle_id)( be_monitor ) <=
                         static_cast< index_t >( min_particle_id ) );

            ASSERT_TRUE( ::NS(BeamMonitor_get_max_particle_id)( be_monitor ) >=
                         static_cast< index_t >( max_particle_id ) );

            ++out_particles_offset;
            ++jj;
        }
    }

    ::NS(BeamMonitor_clear_all)( eb );

    eb_it = eb_begin;

    for( ; eb_it != eb_end ; ++eb_it )
    {
        if( ::NS(Object_get_type_id)( eb_it ) == ::NS(OBJECT_TYPE_BEAM_MONITOR) )
        {
            ::NS(BeamMonitor) const* be_monitor = reinterpret_cast<
                ::NS(BeamMonitor) const* >( ::NS(Object_get_const_begin_ptr)(
                    eb_it ) );

            ASSERT_TRUE( be_monitor != nullptr );
            ASSERT_TRUE( ::NS(BeamMonitor_get_out_address)(
                be_monitor ) == addr_t{ 0 } );
        }
    }

    /* --------------------------------------------------------------------- */

    ::NS(Buffer_delete)( eb );
    ::NS(Buffer_delete)( pb );
    ::NS(Buffer_delete)( out_buffer );
}

TEST( C99_CommonBeamMonitorTests, TrackingAndTurnByTurnIO )
{
    using real_t          = ::NS(particle_real_t);
    using status_t        = ::NS(arch_status_t);
    using part_index_t    = ::NS(particle_index_t);
    using mon_index_t     = ::NS(be_monitor_index_t);
    using size_t          = ::NS(buffer_size_t);
    using nturn_t         = ::NS(be_monitor_turn_t);
    using addr_t          = ::NS(be_monitor_addr_t);
    using turn_dist_t     = std::uniform_int_distribution< nturn_t >;
    using chance_dist_t   = std::uniform_real_distribution< real_t >;
    using type_id_t       = ::NS(object_type_id_t);
    using beam_monitor_t  = ::NS(BeamMonitor);
    using ptr_const_mon_t = beam_monitor_t const*;
    using ptr_particles_t = ::NS(Particles) const*;
    using num_elem_t      = ::NS(particle_num_elements_t);

    std::mt19937_64::result_type const seed = 20181031u;

    std::mt19937_64 prng;
    prng.seed( seed );

    ::NS(Buffer)* eb = ::NS(Buffer_new)( 0u );
    ::NS(Buffer)* pb = ::NS(Buffer_new)( 0u );
    ::NS(Buffer)* out_buffer = ::NS(Buffer_new)( 0u );
    ::NS(Buffer)* elem_by_elem_buffer = ::NS(Buffer_new)( 0u );

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
        ::NS(BeamMonitor)* be_monitor = ::NS(BeamMonitor_add)( eb,
            num_stores_dist( prng ), start_dist( prng ), skip_dist( prng ),
            addr_t{ 0 }, mon_index_t{ 0 }, mon_index_t{ 0 },
            bool{ rolling_dist( prng ) >= 0.5 }, true );

        ASSERT_TRUE( be_monitor != nullptr );

        nturn_t const num_stores =
            ::NS(BeamMonitor_get_num_stores)( be_monitor );

        nturn_t const skip  = ::NS(BeamMonitor_get_skip)( be_monitor );
        nturn_t const start = ::NS(BeamMonitor_get_start)( be_monitor );
        nturn_t const n     = num_stores * skip;

        ASSERT_TRUE( num_stores > nturn_t{ 0 } );

        ++required_num_particle_blocks;

        if( max_num_turns  < n     ) max_num_turns  = n;
        if( max_start_turn < start ) max_start_turn = start;

        for( size_t jj = size_t{ 0 } ; jj < DRIFT_SEQU_LEN ; ++jj )
        {
            ::NS(Drift)*  drift = ::NS(Drift_add)( eb, real_t{ 1.0 } );
            ASSERT_TRUE( drift != nullptr );
        }
    }

    ASSERT_TRUE( ::NS(Buffer_get_num_of_objects)( eb ) == NUM_BEAM_ELEMENTS );

    /* --------------------------------------------------------------------- */

    ::NS(Particles)* particles = ::NS(Particles_new)( pb, NUM_PARTICLES );
    ::NS(Particles_realistic_init)( particles );

    part_index_t min_particle_id = std::numeric_limits< part_index_t >::max();
    part_index_t max_particle_id = std::numeric_limits< part_index_t >::min();

    part_index_t min_turn_id = std::numeric_limits< part_index_t >::max();
    part_index_t max_turn_id = std::numeric_limits< part_index_t >::min();

    ASSERT_TRUE( ::NS(ARCH_STATUS_SUCCESS) ==
        ::NS(Particles_get_min_max_particle_id)( particles,
            &min_particle_id, &max_particle_id ) );

    ASSERT_TRUE( ::NS(ARCH_STATUS_SUCCESS) ==
        ::NS(Particles_get_min_max_at_turn_value)( particles,
            &min_turn_id, &max_turn_id ) );

    ASSERT_TRUE( min_particle_id <= max_particle_id );
    size_t num_stored_particles = size_t{ 1 } +
        static_cast< size_t >( max_particle_id - min_particle_id );

    ASSERT_TRUE( min_turn_id >= part_index_t{ 0 } );
    ASSERT_TRUE( max_turn_id >= min_turn_id );

    nturn_t const NUM_TURNS = max_start_turn + 2 * max_num_turns;

    ::NS(Particles)* initial_state =
        ::NS(Particles_add_copy)( elem_by_elem_buffer, particles );

    ASSERT_TRUE( initial_state != nullptr );

    ::NS(Particles)* final_state =
        ::NS(Particles_add_copy)( elem_by_elem_buffer, particles );

    ASSERT_TRUE( final_state != nullptr );

    /* --------------------------------------------------------------------- */

    ::NS(ElemByElemConfig) elem_by_elem_config;
    ::NS(ElemByElemConfig_preset)( &elem_by_elem_config );

    ASSERT_TRUE( ::NS(ARCH_STATUS_SUCCESS) == ::NS(ElemByElemConfig_init)(
        &elem_by_elem_config, particles, eb, part_index_t{ 0 }, NUM_TURNS ) );

    size_t elem_by_elem_index_offset = size_t{ 0 };

    status_t status = ::NS(ElemByElemConfig_prepare_output_buffer_from_conf)(
        &elem_by_elem_config, elem_by_elem_buffer,
            &elem_by_elem_index_offset );

    ASSERT_TRUE( status == ::NS(ARCH_STATUS_SUCCESS) );
    ASSERT_TRUE( elem_by_elem_index_offset == size_t{ 2 } );
    ASSERT_TRUE( ::NS(Buffer_get_num_of_objects)( elem_by_elem_buffer ) ==
                 size_t{ 3 } );

    ::NS(Particles)* elem_by_elem_particles =
        ::NS(Particles_buffer_get_particles)(
            elem_by_elem_buffer, elem_by_elem_index_offset );

    initial_state = ::NS(Particles_buffer_get_particles)(
        elem_by_elem_buffer, 0u );

    final_state   = ::NS(Particles_buffer_get_particles)(
        elem_by_elem_buffer, 1u );

    status = ::NS(ElemByElemConfig_assign_output_buffer)(
        &elem_by_elem_config, elem_by_elem_buffer, elem_by_elem_index_offset );

    ASSERT_TRUE( status == ::NS(ARCH_STATUS_SUCCESS) );

    ASSERT_TRUE( elem_by_elem_particles != nullptr );
    ASSERT_TRUE( initial_state != nullptr );
    ASSERT_TRUE( final_state   != nullptr );

    status = ::NS(Track_all_particles_element_by_element_until_turn)(
        particles, &elem_by_elem_config, eb, NUM_TURNS );

    ASSERT_TRUE( status == ::NS(ARCH_STATUS_SUCCESS) );

    ::NS(Particles_copy)( final_state, particles );
    ::NS(Particles_copy)( particles, initial_state );

    /* --------------------------------------------------------------------- */

    size_t num_elem_by_elem_turns    = size_t{ 0 };
    size_t beam_monitor_index_offset = size_t{ 0 };
    min_turn_id         = part_index_t{ -1 };

    ASSERT_TRUE( ::NS(ARCH_STATUS_SUCCESS) == ::NS(OutputBuffer_prepare)(
        eb, out_buffer, particles, num_elem_by_elem_turns, nullptr,
            &beam_monitor_index_offset, &min_turn_id ) );

    ::NS(Object) const* obj = ::NS(Buffer_get_const_object)( eb, 10 );

    ASSERT_TRUE( ::NS(Object_get_type_id)( obj ) == ::NS(OBJECT_TYPE_BEAM_MONITOR) );
    ::NS(BeamMonitor) const* monitor =
        ( ::NS(BeamMonitor) const* )::NS(Object_get_const_begin_ptr)( obj );

    ASSERT_TRUE( monitor != nullptr );
    ASSERT_TRUE( NS(ARCH_STATUS_SUCCESS) ==
        ::NS(BeamMonitor_assign_output_buffer_from_offset)( eb, out_buffer,
            min_turn_id, beam_monitor_index_offset ) );

    size_t out_particle_block_index = size_t{ 0 };

    for( size_t ii = size_t{ 0 } ; ii < NUM_BEAM_ELEMENTS ; ++ii )
    {
        ::NS(Object) const* obj_it = ::NS(Buffer_get_const_object)( eb, ii );
        type_id_t const type_id   = ::NS(Object_get_type_id)( obj_it );

        if( type_id == ::NS(OBJECT_TYPE_BEAM_MONITOR) )
        {
            ::NS(BeamMonitor) const* monitor = reinterpret_cast<
                ::NS(BeamMonitor) const* >(
                    ::NS(Object_get_const_begin_ptr)( obj_it ) );

            ASSERT_TRUE( monitor != nullptr );
            nturn_t const num_stores = ::NS(BeamMonitor_get_num_stores)( monitor );

            ASSERT_TRUE( out_particle_block_index <
                ::NS(Buffer_get_num_of_objects)( out_buffer ) );

            ::NS(Particles) const* out_particles =
                ::NS(Particles_buffer_get_const_particles)(
                    out_buffer, out_particle_block_index );

            ASSERT_TRUE( ::NS(BeamMonitor_get_out_address)( monitor ) != addr_t{ 0 } );
            ASSERT_TRUE( ::NS(BeamMonitor_get_out_address)( monitor ) == static_cast<
                addr_t >( reinterpret_cast< uintptr_t >( out_particles ) ) );

            ASSERT_TRUE( num_stores > nturn_t{ 0 } );

            num_elem_t const num_out_particles =
                ::NS(Particles_get_num_of_particles)( out_particles );

            ASSERT_TRUE( num_out_particles >= static_cast< num_elem_t >(
                num_stored_particles * num_stores ) );

            ++out_particle_block_index;
        }
    }

    /* --------------------------------------------------------------------- */

    ::NS(Track_all_particles_until_turn)( particles, eb, NUM_TURNS );

    ::NS(Buffer)* cmp_particles_buffer = ::NS(Buffer_new)( 0u );

    ::NS(Particles)* cmp_particles =
        ::NS(Particles_new)( cmp_particles_buffer, NUM_PARTICLES );

    ASSERT_TRUE( cmp_particles != nullptr );

    if( 0 != st_Particles_compare_values_with_treshold(
            particles, final_state, ABS_TOLERANCE ) )
    {
        ::NS(Buffer)* diff_buffer = ::NS(Buffer_new)( 0u );
        ::NS(Particles)* diff = ::NS(Particles_new)( diff_buffer, NUM_PARTICLES );

        ::NS(Particles_calculate_difference)( final_state, particles, diff );

        std::cout << "final_state: " << std::endl;
        ::NS(Particles_print_out)( final_state );

        std::cout << std::endl << "particles (tracked):" << std::endl;
        ::NS(Particles_print_out)( particles );

        std::cout << std::endl << "diff: " << std::endl;
        ::NS(Particles_print_out)( diff );

        diff = nullptr;
        ::NS(Buffer_delete)( diff_buffer );
    }

    ASSERT_TRUE( 0 == ::NS(Particles_compare_values_with_treshold)(
            particles, final_state, ABS_TOLERANCE ) );

    for( nturn_t kk = nturn_t{ 0 } ; kk < NUM_TURNS ; ++kk )
    {
        for( size_t jj = size_t{ 0 } ; jj < NUM_BEAM_ELEMENTS; ++jj )
        {
            ::NS(Object) const* obj_it = ::NS(Buffer_get_const_object)( eb, jj );

            if( ::NS(Object_get_type_id)( obj_it ) ==
                ::NS(OBJECT_TYPE_BEAM_MONITOR) )
            {
                ptr_const_mon_t mon = reinterpret_cast< ptr_const_mon_t >(
                        ::NS(Object_get_const_begin_ptr)( obj_it ) );

                ASSERT_TRUE( ::NS(BeamMonitor_get_out_address)(
                    mon ) != addr_t{ 0 } );

                if( !::NS(BeamMonitor_has_turn_stored)( mon, kk, NUM_TURNS ) )
                {
                    continue;
                }

                ASSERT_TRUE( ::NS(BeamMonitor_get_start)( mon ) <= kk );
                ASSERT_TRUE( ( ( kk - ::NS(BeamMonitor_get_start)( mon ) )
                    % ::NS(BeamMonitor_get_skip)( mon ) ) == nturn_t{ 0 } );

                ptr_particles_t out_particles = reinterpret_cast<
                    ptr_particles_t >( static_cast< uintptr_t >(
                        ::NS(BeamMonitor_get_out_address)( mon ) ) );

                ASSERT_TRUE( elem_by_elem_particles != nullptr );

                for( size_t ll = size_t{ 0 } ; ll < NUM_PARTICLES ; ++ll )
                {
                    part_index_t const particle_id =
                        ::NS(Particles_get_particle_id_value)( particles, ll );

                    num_elem_t const elem_by_elem_index =
                        st_ElemByElemConfig_get_particles_store_index_details(
                            &elem_by_elem_config, particle_id, jj, kk );

                    ASSERT_TRUE( elem_by_elem_index >= num_elem_t{ 0 } );
                    ASSERT_TRUE( elem_by_elem_index <
                        ::NS(ElemByElemConfig_get_out_store_num_particles)(
                            &elem_by_elem_config ) );

                    ASSERT_TRUE( elem_by_elem_index <
                        ::NS(Particles_get_num_of_particles)(
                            elem_by_elem_particles ) );

                    ASSERT_TRUE( ::NS(ARCH_STATUS_SUCCESS) ==
                    ::NS(Particles_copy_single)( particles, ll,
                        elem_by_elem_particles, elem_by_elem_index ) );

                    num_elem_t const stored_particle_id =
                        ::NS(BeamMonitor_get_store_particle_index)(
                            mon, kk, particle_id );

                    ASSERT_TRUE( stored_particle_id >= num_elem_t{ 0 } );
                    ASSERT_TRUE( ::NS(ARCH_STATUS_SUCCESS) ==
                        ::NS(Particles_copy_single)( cmp_particles, ll,
                                    out_particles, stored_particle_id ) );
                }

                if( 0 != ::NS(Particles_compare_values_with_treshold)(
                        cmp_particles, particles, ABS_TOLERANCE ) )
                {
                    std::cout << "jj = " << jj << std::endl;

                    ::NS(Buffer)* diff_buffer = ::NS(Buffer_new)( 0u );
                    ::NS(Particles)* diff =
                        ::NS(Particles_new)( diff_buffer, NUM_PARTICLES );

                    ASSERT_TRUE( diff != nullptr );

                    ::NS(Particles_calculate_difference)(
                        cmp_particles, particles, diff );

                    std::cout << "cmp_particles: " << std::endl;
                    ::NS(Particles_print_out)( cmp_particles );

                    std::cout << std::endl << "elem_by_elem_particles: "
                                << std::endl;

                    ::NS(Particles_print_out)( particles );

                    std::cout << std::endl << "diff: " << std::endl;
                    ::NS(Particles_print_out)( diff );

                    ::NS(Buffer_delete)( diff_buffer );
                    diff_buffer = nullptr;
                }

                ASSERT_TRUE( 0 == ::NS(Particles_compare_values_with_treshold)(
                    cmp_particles, particles, ABS_TOLERANCE ) );
            }
        }
    }

    ::NS(Buffer_delete)( eb );
    ::NS(Buffer_delete)( pb );
    ::NS(Buffer_delete)( out_buffer );
    ::NS(Buffer_delete)( elem_by_elem_buffer );
    ::NS(Buffer_delete)( cmp_particles_buffer );
}

/* end: tests/sixtracklib/common/test_be_monitor_c99.cpp */
