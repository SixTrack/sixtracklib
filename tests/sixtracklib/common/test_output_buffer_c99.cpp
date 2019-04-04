#include "sixtracklib/common/output/output_buffer.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <vector>
#include <utility>

#include <gtest/gtest.h>

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/buffer.h"
#include "sixtracklib/common/particles.h"
#include "sixtracklib/common/be_monitor/be_monitor.h"
// #include "sixtracklib/common/elem_by_elem_config.h"

#include "sixtracklib/testlib/testdata/testdata_files.h"

TEST( C99_CommonOutputBuffer, OutputBufferCalculateParameters )
{
    using buf_size_t   = ::NS(buffer_size_t);
    using part_index_t = ::NS(particle_index_t);
    using address_t    = ::NS(buffer_addr_t);

    ::NS(Buffer)* input_pb = ::NS(Buffer_new_from_file)(
        ::NS(PATH_TO_BEAMBEAM_PARTICLES_DUMP) );

    SIXTRL_ASSERT( input_pb != nullptr );
    SIXTRL_ASSERT( ::NS(Buffer_is_particles_buffer)( input_pb ) );

    ::NS(Buffer)* pb = ::NS(Buffer_new)( buf_size_t{ 0 } );

    ::NS(Buffer)* eb = ::NS(Buffer_new_from_file)(
        ::NS(PATH_TO_BEAMBEAM_BEAM_ELEMENTS) );

    SIXTRL_ASSERT( eb != nullptr );

    /* --------------------------------------------------------------------- */

    buf_size_t const NUM_PARTICLES = buf_size_t{ 100 };
    ::NS(Particles) const* in_particles =
        ::NS(Particles_buffer_get_const_particles)( input_pb, buf_size_t{ 0 } );

    buf_size_t const IN_NUM_PARTICLES =
        ::NS(Particles_get_num_of_particles)( in_particles );

    SIXTRL_ASSERT( IN_NUM_PARTICLES > buf_size_t{ 0 } );

    ::NS(Particles)* particles = ::NS(Particles_new)( pb, NUM_PARTICLES );
    SIXTRL_ASSERT( particles != nullptr );
    SIXTRL_ASSERT( ::NS(Buffer_get_num_of_objects)( pb ) == buf_size_t{ 1 } );

    for( buf_size_t ii = buf_size_t{ 0 } ; ii < NUM_PARTICLES ; ++ii )
    {
        buf_size_t const jj = ii % IN_NUM_PARTICLES;
        ::NS(Particles_copy_single)( particles, ii, in_particles, jj );
    }

    ::NS(Buffer_delete)( input_pb );
    in_particles = nullptr;
    input_pb = nullptr;

    ::NS(Particles_init_particle_ids)( particles );
    ::NS(Particles_set_all_at_element_id_value)( particles, part_index_t{ 0 } );
    ::NS(Particles_set_all_at_turn_value)( particles, part_index_t{ 0 } );

    part_index_t min_part_id, max_part_id, min_elem_id, max_elem_id,
                 min_turn_id, max_turn_id;

    buf_size_t pset_indices[] = { buf_size_t{ 0 } };

    int ret = ::NS(Particles_get_min_max_attributes)( particles, &min_part_id,
        &max_part_id, &min_elem_id, &max_elem_id, &min_turn_id, &max_turn_id );

    SIXTRL_ASSERT( ret == 0 );
    SIXTRL_ASSERT( min_part_id == part_index_t{ 0 } );
    SIXTRL_ASSERT( max_part_id >= min_part_id );
    SIXTRL_ASSERT( static_cast< buf_size_t >( max_part_id ) ==
                   NUM_PARTICLES - buf_size_t{ 1 }  );

    SIXTRL_ASSERT( min_turn_id >= part_index_t{ 0 } );
    SIXTRL_ASSERT( max_turn_id >= min_turn_id );

    SIXTRL_ASSERT( min_elem_id >= part_index_t{ 0 } );
    SIXTRL_ASSERT( max_elem_id >= min_elem_id );

    ret = ::NS(ElemByElemConfig_get_min_max_element_id_from_buffer)(
        eb, &min_elem_id, &max_elem_id, nullptr, buf_size_t{ 0 } );

    SIXTRL_ASSERT( ret == 0 );
    SIXTRL_ASSERT( min_turn_id >= part_index_t{ 0 } );
    SIXTRL_ASSERT( max_turn_id >= min_turn_id );

    /* --------------------------------------------------------------------- */

    ::NS(Buffer)* out_buffer = ::NS(Buffer_new)( buf_size_t{ 0 } );
    SIXTRL_ASSERT( out_buffer != nullptr );

    buf_size_t const slot_size = ::NS(Buffer_get_slot_size)( pb );
    SIXTRL_ASSERT( slot_size > buf_size_t{ 0 } );

    /* ===================================================================== */
    /* Test case 1: no output buffer at all */

    buf_size_t num_objects  = buf_size_t{ 0 };
    buf_size_t num_slots    = buf_size_t{ 0 };
    buf_size_t num_dataptrs = buf_size_t{ 0 };
    buf_size_t num_garbage  = buf_size_t{ 0 };

    buf_size_t num_elem_by_elem_turns = buf_size_t{ 0 };

    ASSERT_TRUE( buf_size_t{ 0 } ==
        ::NS(BeamMonitor_get_num_of_beam_monitor_objects)( eb ) );

    ret = NS(OutputBuffer_calculate_output_buffer_params)( eb, particles,
        num_elem_by_elem_turns, &num_objects, &num_slots, &num_dataptrs,
            &num_garbage, slot_size );

    ASSERT_TRUE( ret == 0 );
    ASSERT_TRUE( num_objects  == buf_size_t{ 0 } );
    ASSERT_TRUE( num_slots    == buf_size_t{ 0 } );
    ASSERT_TRUE( num_dataptrs == buf_size_t{ 0 } );
    ASSERT_TRUE( num_garbage  == buf_size_t{ 0 } );

    ret = NS(OutputBuffer_calculate_output_buffer_params_for_particles_sets)(
        eb, pb, buf_size_t{ 1 }, &pset_indices[ 0 ], num_elem_by_elem_turns,
            &num_objects, &num_slots, &num_dataptrs, &num_garbage, slot_size );

    ASSERT_TRUE( ret == 0 );
    ASSERT_TRUE( num_objects  == buf_size_t{ 0 } );
    ASSERT_TRUE( num_slots    == buf_size_t{ 0 } );
    ASSERT_TRUE( num_dataptrs == buf_size_t{ 0 } );
    ASSERT_TRUE( num_garbage  == buf_size_t{ 0 } );

    ret = NS(OutputBuffer_calculate_output_buffer_params_detailed)( eb,
        min_part_id, max_part_id, min_elem_id, max_elem_id, min_turn_id,
            max_turn_id, num_elem_by_elem_turns, &num_objects, &num_slots,
                &num_dataptrs, &num_garbage, slot_size );

    ASSERT_TRUE( ret == 0 );
    ASSERT_TRUE( num_objects  == buf_size_t{ 0 } );
    ASSERT_TRUE( num_slots    == buf_size_t{ 0 } );
    ASSERT_TRUE( num_dataptrs == buf_size_t{ 0 } );
    ASSERT_TRUE( num_garbage  == buf_size_t{ 0 } );

    /* ===================================================================== */
    /* Test case 2: only element by element output */

    num_elem_by_elem_turns = buf_size_t{ 5 };

    part_index_t max_elem_by_elem_turn_id =
        min_turn_id + num_elem_by_elem_turns;

    if( max_elem_by_elem_turn_id < max_turn_id )
    {
        max_elem_by_elem_turn_id = max_turn_id;
    }

    SIXTRL_ASSERT( ret == 0 );

    buf_size_t requ_store_particles =
        ::NS(ElemByElemConfig_get_stored_num_particles_detailed)(
            min_part_id, max_part_id, min_elem_id, max_elem_id,
                min_turn_id, max_elem_by_elem_turn_id );

    buf_size_t elem_by_elem_output_offset = buf_size_t{ 0 };
    buf_size_t beam_monitor_output_offset = buf_size_t{ 0 };
    part_index_t out_min_turn_id = part_index_t{ 0 };

    address_t begin_addr = address_t{ 0 };
    buf_size_t out_buffer_size = buf_size_t{ 0 };

    ret = ::NS(OutputBuffer_calculate_output_buffer_params)( eb, particles,
        num_elem_by_elem_turns, &num_objects, &num_slots, &num_dataptrs,
            &num_garbage, slot_size );

    ASSERT_TRUE( ret == 0 );
    ASSERT_TRUE( num_objects  != buf_size_t{ 0 } );
    ASSERT_TRUE( num_slots    != buf_size_t{ 0 } );
    ASSERT_TRUE( num_dataptrs != buf_size_t{ 0 } );

    ::NS(Buffer_delete)( out_buffer );
    out_buffer = ::NS(Buffer_new)( buf_size_t{ 0 } );
    ret = ::NS(Buffer_reserve)(
        out_buffer, num_objects, num_slots, num_dataptrs, num_garbage );

    SIXTRL_ASSERT( ret == 0 );
    begin_addr = ::NS(Buffer_get_data_begin_addr)( out_buffer );
    out_buffer_size = ::NS(Buffer_get_size)( out_buffer );

    ret = ::NS(OutputBuffer_prepare)( eb, out_buffer, particles,
        num_elem_by_elem_turns, &elem_by_elem_output_offset,
            &beam_monitor_output_offset, &out_min_turn_id );

    ASSERT_TRUE( ret == 0 );

    ASSERT_TRUE( elem_by_elem_output_offset == buf_size_t{ 0 } );
    ASSERT_TRUE( beam_monitor_output_offset >= elem_by_elem_output_offset );
    ASSERT_TRUE( out_min_turn_id == min_turn_id );

    ASSERT_TRUE( ::NS(Buffer_get_data_begin_addr)( out_buffer ) == begin_addr );
    ASSERT_TRUE( ::NS(Buffer_get_size)( out_buffer ) == out_buffer_size );
    ASSERT_TRUE( ::NS(Buffer_get_num_of_objects)( out_buffer ) == num_objects );
    ASSERT_TRUE( ::NS(Buffer_get_num_of_slots)( out_buffer ) <= num_slots );
    ASSERT_TRUE( ::NS(Buffer_get_num_of_dataptrs)( out_buffer ) <= num_dataptrs );

    ASSERT_TRUE( ::NS(Buffer_get_num_of_garbage_ranges)( out_buffer ) <=
                 num_garbage );

    ::NS(Particles) const* out_particles =
        ::NS(Particles_buffer_get_const_particles)(
            out_buffer, elem_by_elem_output_offset );

    ASSERT_TRUE( out_particles != nullptr );
    ASSERT_TRUE( requ_store_particles <= static_cast< buf_size_t >(
        ::NS(Particles_get_num_of_particles)( out_particles ) ) );

    /* -------------------------------------------------------------------- */

    num_objects = num_slots = num_dataptrs = num_garbage = buf_size_t{ 0 };
    beam_monitor_output_offset = elem_by_elem_output_offset = buf_size_t{ 0 };
    out_min_turn_id = part_index_t{ 0 };

    ret = ::NS(OutputBuffer_calculate_output_buffer_params_for_particles_sets)(
        eb, pb, buf_size_t{ 1 }, &pset_indices[ 0 ], num_elem_by_elem_turns,
            &num_objects, &num_slots, &num_dataptrs, &num_garbage, slot_size );

    ASSERT_TRUE( ret == 0 );
    ASSERT_TRUE( num_objects  != buf_size_t{ 0 } );
    ASSERT_TRUE( num_slots    != buf_size_t{ 0 } );
    ASSERT_TRUE( num_dataptrs != buf_size_t{ 0 } );

    ::NS(Buffer_delete)( out_buffer );
    out_buffer = NS(Buffer_new)( buf_size_t{ 0 } );

    ret = ::NS(Buffer_reserve)( out_buffer,
        num_objects, num_slots, num_dataptrs, num_garbage );

    SIXTRL_ASSERT( ret == 0 );
    begin_addr = ::NS(Buffer_get_data_begin_addr)( out_buffer );
    out_buffer_size = ::NS(Buffer_get_size)( out_buffer );

    ret = ::NS(OutputBuffer_prepare_for_particle_sets)( eb, out_buffer,
        pb, buf_size_t{ 1 }, &pset_indices[ 0 ], num_elem_by_elem_turns,
            &elem_by_elem_output_offset, &beam_monitor_output_offset,
                &out_min_turn_id );

    ASSERT_TRUE( ret == 0 );

    ASSERT_TRUE( elem_by_elem_output_offset == buf_size_t{ 0 } );
    ASSERT_TRUE( beam_monitor_output_offset >= elem_by_elem_output_offset );
    ASSERT_TRUE( out_min_turn_id == min_turn_id );

    ASSERT_TRUE( ::NS(Buffer_get_data_begin_addr)( out_buffer ) == begin_addr );
    ASSERT_TRUE( ::NS(Buffer_get_size)( out_buffer ) == out_buffer_size );
    ASSERT_TRUE( ::NS(Buffer_get_num_of_objects)( out_buffer ) == num_objects );
    ASSERT_TRUE( ::NS(Buffer_get_num_of_slots)( out_buffer ) <= num_slots );
    ASSERT_TRUE( ::NS(Buffer_get_num_of_dataptrs)( out_buffer ) <=
                 num_dataptrs );

    ASSERT_TRUE( ::NS(Buffer_get_num_of_garbage_ranges)( out_buffer ) <=
                 num_garbage );

    out_particles = ::NS(Particles_buffer_get_const_particles)(
            out_buffer, elem_by_elem_output_offset );

    ASSERT_TRUE( out_particles != nullptr );
    ASSERT_TRUE( requ_store_particles <= static_cast< buf_size_t >(
        ::NS(Particles_get_num_of_particles)( out_particles ) ) );

    /* -------------------------------------------------------------------- */

    num_objects = num_slots = num_dataptrs = num_garbage = buf_size_t{ 0 };
    beam_monitor_output_offset = elem_by_elem_output_offset = buf_size_t{ 0 };
    out_min_turn_id = part_index_t{ 0 };

    ret = ::NS(OutputBuffer_calculate_output_buffer_params_detailed)(
        eb, min_part_id, max_part_id, min_elem_id, max_elem_id, min_turn_id,
            max_turn_id, num_elem_by_elem_turns, &num_objects, &num_slots,
                &num_dataptrs, &num_garbage, slot_size );

    ASSERT_TRUE( ret == 0 );
    ASSERT_TRUE( num_objects  != buf_size_t{ 0 } );
    ASSERT_TRUE( num_slots    != buf_size_t{ 0 } );
    ASSERT_TRUE( num_dataptrs != buf_size_t{ 0 } );

    ::NS(Buffer_delete)( out_buffer );
    out_buffer = NS(Buffer_new)( buf_size_t{ 0 } );

    ret = ::NS(Buffer_reserve)( out_buffer,
        num_objects, num_slots, num_dataptrs, num_garbage );

    SIXTRL_ASSERT( ret == 0 );
    begin_addr = ::NS(Buffer_get_data_begin_addr)( out_buffer );
    out_buffer_size = ::NS(Buffer_get_size)( out_buffer );

    part_index_t cmp_max_elem_by_elem_turn_id = part_index_t{ 0 };

    ret = ::NS(OutputBuffer_prepare_detailed)( eb, out_buffer, min_part_id,
        max_part_id, min_elem_id, max_elem_id, min_turn_id, max_turn_id,
            num_elem_by_elem_turns, &elem_by_elem_output_offset,
                &beam_monitor_output_offset, &cmp_max_elem_by_elem_turn_id );

    ASSERT_TRUE( ret == 0 );

    ASSERT_TRUE( elem_by_elem_output_offset == buf_size_t{ 0 } );
    ASSERT_TRUE( beam_monitor_output_offset >= elem_by_elem_output_offset );
    ASSERT_TRUE( out_min_turn_id == min_turn_id );

    ASSERT_TRUE( ::NS(Buffer_get_data_begin_addr)( out_buffer ) == begin_addr );
    ASSERT_TRUE( ::NS(Buffer_get_size)( out_buffer ) == out_buffer_size );
    ASSERT_TRUE( ::NS(Buffer_get_num_of_objects)( out_buffer ) == num_objects );
    ASSERT_TRUE( ::NS(Buffer_get_num_of_slots)( out_buffer ) <= num_slots );
    ASSERT_TRUE( ::NS(Buffer_get_num_of_dataptrs)( out_buffer ) <=
                 num_dataptrs );

    ASSERT_TRUE( ::NS(Buffer_get_num_of_garbage_ranges)( out_buffer ) <=
                 num_garbage );

    ASSERT_TRUE( max_elem_by_elem_turn_id == cmp_max_elem_by_elem_turn_id );

    out_particles = ::NS(Particles_buffer_get_const_particles)(
            out_buffer, elem_by_elem_output_offset );

    ASSERT_TRUE( out_particles != nullptr );
    ASSERT_TRUE( requ_store_particles <= static_cast< buf_size_t >(
        ::NS(Particles_get_num_of_particles)( out_particles ) ) );

    /* ===================================================================== */
    /* Test case 3: only beam monitor output */

    num_elem_by_elem_turns   = buf_size_t{ 0 };
    max_elem_by_elem_turn_id = part_index_t{ 0 };

    buf_size_t num_turn_by_turn_turns = 10;
    buf_size_t target_num_turns = 500;
    buf_size_t skip_turns = 2;

    buf_size_t beam_monitor_index = ::NS(Buffer_get_num_of_objects)( eb );

    ret = ::NS(BeamMonitor_insert_end_of_turn_monitors)( eb,
        buf_size_t{ 0 }, num_turn_by_turn_turns, target_num_turns, skip_turns,
            ::NS(Buffer_get_objects_end)( eb ) );

    SIXTRL_ASSERT( ret == 0 );
    SIXTRL_ASSERT( ::NS(BeamMonitor_get_num_of_beam_monitor_objects)( eb ) ==
                   buf_size_t{ 2 } );

    SIXTRL_ASSERT( ::NS(Buffer_get_num_of_objects)( eb ) ==
        ( beam_monitor_index + buf_size_t{ 2 } ) );

    elem_by_elem_output_offset = buf_size_t{ 0 };
    beam_monitor_output_offset = buf_size_t{ 0 };
    out_min_turn_id = part_index_t{ 0 };

    begin_addr = address_t{ 0 };
    out_buffer_size = buf_size_t{ 0 };

    ret = ::NS(OutputBuffer_calculate_output_buffer_params)( eb, particles,
        num_elem_by_elem_turns, &num_objects, &num_slots, &num_dataptrs,
            &num_garbage, slot_size );

    ASSERT_TRUE( ret == 0 );
    ASSERT_TRUE( num_objects ==
        ::NS(BeamMonitor_get_num_of_beam_monitor_objects)( eb ) );

    ASSERT_TRUE( num_slots != buf_size_t{ 0 } );
    ASSERT_TRUE( num_dataptrs != buf_size_t{ 0 } );

    ::NS(Buffer_delete)( out_buffer );
    out_buffer = ::NS(Buffer_new)( buf_size_t{ 0 } );
    ret = ::NS(Buffer_reserve)( out_buffer,
        num_objects, num_slots, num_dataptrs, num_garbage );

    SIXTRL_ASSERT( ret == 0 );
    begin_addr = ::NS(Buffer_get_data_begin_addr)( out_buffer );
    out_buffer_size = ::NS(Buffer_get_size)( out_buffer );

    ret = ::NS(OutputBuffer_prepare)( eb, out_buffer, particles,
        num_elem_by_elem_turns, &elem_by_elem_output_offset,
            &beam_monitor_output_offset, &out_min_turn_id );

    ASSERT_TRUE( ret == 0 );

    ASSERT_TRUE( elem_by_elem_output_offset == buf_size_t{ 0 } );
    ASSERT_TRUE( beam_monitor_output_offset >= elem_by_elem_output_offset );
    ASSERT_TRUE( out_min_turn_id == min_turn_id );

    ASSERT_TRUE( ::NS(Buffer_get_data_begin_addr)( out_buffer ) == begin_addr );
    ASSERT_TRUE( ::NS(Buffer_get_size)( out_buffer ) == out_buffer_size );
    ASSERT_TRUE( ::NS(Buffer_get_num_of_objects)( out_buffer ) == num_objects );
    ASSERT_TRUE( ::NS(Buffer_get_num_of_slots)( out_buffer ) <= num_slots );
    ASSERT_TRUE( ::NS(Buffer_get_num_of_dataptrs)( out_buffer ) <=
                 num_dataptrs );

    ASSERT_TRUE( ::NS(Buffer_get_num_of_garbage_ranges)( out_buffer ) <=
                 num_garbage );

    for( buf_size_t jj = beam_monitor_index, ii = beam_monitor_output_offset ;
         ii < buf_size_t{ 2 } ; ++ii, ++jj )
    {
        ::NS(Object) const* obj = ::NS(Buffer_get_const_object)( eb, jj );

        SIXTRL_ASSERT( ::NS(Object_get_type_id)( obj ) ==
                       ::NS(OBJECT_TYPE_BEAM_MONITOR) );

        SIXTRL_ASSERT( ::NS(Object_get_size)( obj ) >=
                       sizeof( ::NS(BeamMonitor) ) );

        ::NS(BeamMonitor) const* monitor = reinterpret_cast<
            ::NS(BeamMonitor) const* >( static_cast< uintptr_t >(
                ::NS(Object_get_begin_addr)( obj ) ) );

        SIXTRL_ASSERT( monitor != nullptr );
        SIXTRL_ASSERT( ::NS(BeamMonitor_get_stored_num_particles)( monitor ) >
                     buf_size_t{ 0 } );

        out_particles = ::NS(Particles_buffer_get_const_particles)(
            out_buffer, ii );

        ASSERT_TRUE( out_particles != nullptr );

        ASSERT_TRUE( static_cast< buf_size_t >(
            ::NS(Particles_get_num_of_particles)( out_particles ) ) >=
                ::NS(BeamMonitor_get_stored_num_particles)( monitor ) );
    }

    /* -------------------------------------------------------------------- */

    num_objects = num_slots = num_dataptrs = num_garbage = buf_size_t{ 0 };
    beam_monitor_output_offset = elem_by_elem_output_offset = buf_size_t{ 0 };
    out_min_turn_id = part_index_t{ 0 };

    ret = ::NS(OutputBuffer_calculate_output_buffer_params_for_particles_sets)(
        eb, pb, buf_size_t{ 1 }, &pset_indices[ 0 ], num_elem_by_elem_turns,
            &num_objects, &num_slots, &num_dataptrs, &num_garbage, slot_size );

    ASSERT_TRUE( ret == 0 );
    ASSERT_TRUE( num_objects  != buf_size_t{ 0 } );
    ASSERT_TRUE( num_slots    != buf_size_t{ 0 } );
    ASSERT_TRUE( num_dataptrs != buf_size_t{ 0 } );

    ::NS(Buffer_delete)( out_buffer );
    out_buffer = NS(Buffer_new)( buf_size_t{ 0 } );

    ret = ::NS(Buffer_reserve)( out_buffer,
        num_objects, num_slots, num_dataptrs, num_garbage );

    SIXTRL_ASSERT( ret == 0 );
    begin_addr = ::NS(Buffer_get_data_begin_addr)( out_buffer );
    out_buffer_size = ::NS(Buffer_get_size)( out_buffer );

    ret = ::NS(OutputBuffer_prepare_for_particle_sets)( eb, out_buffer,
        pb, buf_size_t{ 1 }, &pset_indices[ 0 ], num_elem_by_elem_turns,
            &elem_by_elem_output_offset, &beam_monitor_output_offset,
                &out_min_turn_id );

    ASSERT_TRUE( ret == 0 );

    ASSERT_TRUE( elem_by_elem_output_offset == buf_size_t{ 0 } );
    ASSERT_TRUE( beam_monitor_output_offset >= elem_by_elem_output_offset );
    ASSERT_TRUE( out_min_turn_id == min_turn_id );

    ASSERT_TRUE( ::NS(Buffer_get_data_begin_addr)( out_buffer ) == begin_addr );
    ASSERT_TRUE( ::NS(Buffer_get_size)( out_buffer ) == out_buffer_size );
    ASSERT_TRUE( ::NS(Buffer_get_num_of_objects)( out_buffer ) == num_objects );
    ASSERT_TRUE( ::NS(Buffer_get_num_of_slots)( out_buffer ) <= num_slots );
    ASSERT_TRUE( ::NS(Buffer_get_num_of_dataptrs)( out_buffer ) <=
                 num_dataptrs );

    ASSERT_TRUE( ::NS(Buffer_get_num_of_garbage_ranges)( out_buffer ) <=
                 num_garbage );

    for( buf_size_t jj = beam_monitor_index, ii = beam_monitor_output_offset ;
         ii < buf_size_t{ 2 } ; ++ii, ++jj )
    {
        ::NS(Object) const* obj = ::NS(Buffer_get_const_object)( eb, jj );

        SIXTRL_ASSERT( ::NS(Object_get_type_id)( obj ) ==
                       ::NS(OBJECT_TYPE_BEAM_MONITOR) );

        SIXTRL_ASSERT( ::NS(Object_get_size)( obj ) >=
                       sizeof( ::NS(BeamMonitor) ) );

        ::NS(BeamMonitor) const* monitor = reinterpret_cast<
            ::NS(BeamMonitor) const* >( static_cast< uintptr_t >(
                ::NS(Object_get_begin_addr)( obj ) ) );

        SIXTRL_ASSERT( monitor != nullptr );
        SIXTRL_ASSERT( ::NS(BeamMonitor_get_stored_num_particles)( monitor ) >
                     buf_size_t{ 0 } );

        out_particles = ::NS(Particles_buffer_get_const_particles)(
            out_buffer, ii );

        ASSERT_TRUE( out_particles != nullptr );

        ASSERT_TRUE( static_cast< buf_size_t >(
            ::NS(Particles_get_num_of_particles)( out_particles ) ) >=
                ::NS(BeamMonitor_get_stored_num_particles)( monitor ) );
    }

    /* -------------------------------------------------------------------- */

    num_objects = num_slots = num_dataptrs = num_garbage = buf_size_t{ 0 };
    beam_monitor_output_offset = elem_by_elem_output_offset = buf_size_t{ 0 };
    out_min_turn_id = part_index_t{ 0 };

    ret = ::NS(OutputBuffer_calculate_output_buffer_params_detailed)(
        eb, min_part_id, max_part_id, min_elem_id, max_elem_id, min_turn_id,
            max_turn_id, num_elem_by_elem_turns, &num_objects, &num_slots,
                &num_dataptrs, &num_garbage, slot_size );

    ASSERT_TRUE( ret == 0 );
    ASSERT_TRUE( num_objects  != buf_size_t{ 0 } );
    ASSERT_TRUE( num_slots    != buf_size_t{ 0 } );
    ASSERT_TRUE( num_dataptrs != buf_size_t{ 0 } );

    ::NS(Buffer_delete)( out_buffer );
    out_buffer = NS(Buffer_new)( buf_size_t{ 0 } );

    ret = ::NS(Buffer_reserve)( out_buffer,
        num_objects, num_slots, num_dataptrs, num_garbage );

    SIXTRL_ASSERT( ret == 0 );
    begin_addr = ::NS(Buffer_get_data_begin_addr)( out_buffer );
    out_buffer_size = ::NS(Buffer_get_size)( out_buffer );

    cmp_max_elem_by_elem_turn_id = part_index_t{ 0 };

    ret = ::NS(OutputBuffer_prepare_detailed)( eb, out_buffer, min_part_id,
        max_part_id, min_elem_id, max_elem_id, min_turn_id, max_turn_id,
            num_elem_by_elem_turns, &elem_by_elem_output_offset,
                &beam_monitor_output_offset, &cmp_max_elem_by_elem_turn_id );

    ASSERT_TRUE( ret == 0 );

    ASSERT_TRUE( elem_by_elem_output_offset == buf_size_t{ 0 } );
    ASSERT_TRUE( beam_monitor_output_offset >= elem_by_elem_output_offset );
    ASSERT_TRUE( out_min_turn_id == min_turn_id );

    ASSERT_TRUE( ::NS(Buffer_get_data_begin_addr)( out_buffer ) == begin_addr );
    ASSERT_TRUE( ::NS(Buffer_get_size)( out_buffer ) == out_buffer_size );
    ASSERT_TRUE( ::NS(Buffer_get_num_of_objects)( out_buffer ) == num_objects );
    ASSERT_TRUE( ::NS(Buffer_get_num_of_slots)( out_buffer ) <= num_slots );
    ASSERT_TRUE( ::NS(Buffer_get_num_of_dataptrs)( out_buffer ) <=
                 num_dataptrs );

    ASSERT_TRUE( ::NS(Buffer_get_num_of_garbage_ranges)( out_buffer ) <=
                 num_garbage );

    ASSERT_TRUE( max_elem_by_elem_turn_id == cmp_max_elem_by_elem_turn_id );

    for( buf_size_t jj = beam_monitor_index, ii = beam_monitor_output_offset ;
         ii < buf_size_t{ 2 } ; ++ii, ++jj )
    {
        ::NS(Object) const* obj = ::NS(Buffer_get_const_object)( eb, jj );

        SIXTRL_ASSERT( ::NS(Object_get_type_id)( obj ) ==
                       ::NS(OBJECT_TYPE_BEAM_MONITOR) );

        SIXTRL_ASSERT( ::NS(Object_get_size)( obj ) >=
                       sizeof( ::NS(BeamMonitor) ) );

        ::NS(BeamMonitor) const* monitor = reinterpret_cast<
            ::NS(BeamMonitor) const* >( static_cast< uintptr_t >(
                ::NS(Object_get_begin_addr)( obj ) ) );

        SIXTRL_ASSERT( monitor != nullptr );
        SIXTRL_ASSERT( ::NS(BeamMonitor_get_stored_num_particles)( monitor ) >
                     buf_size_t{ 0 } );

        out_particles = ::NS(Particles_buffer_get_const_particles)(
            out_buffer, ii );

        ASSERT_TRUE( out_particles != nullptr );

        ASSERT_TRUE( static_cast< buf_size_t >(
            ::NS(Particles_get_num_of_particles)( out_particles ) ) >=
                ::NS(BeamMonitor_get_stored_num_particles)( monitor ) );
    }

    /* ===================================================================== */
    /* Test case 4: Both elem by elem output and beam monitors */

    num_elem_by_elem_turns = buf_size_t{ 5 };

    max_elem_by_elem_turn_id = min_turn_id + num_elem_by_elem_turns;

    if( max_elem_by_elem_turn_id < max_turn_id )
    {
        max_elem_by_elem_turn_id = max_turn_id;
    }

    num_turn_by_turn_turns = 10;
    target_num_turns = 500;
    skip_turns = 2;

    requ_store_particles =
        ::NS(ElemByElemConfig_get_stored_num_particles_detailed)(
            min_part_id, max_part_id, min_elem_id, max_elem_id, min_turn_id,
                max_elem_by_elem_turn_id );

    ::NS(Buffer_delete)( eb );
    eb = ::NS(Buffer_new_from_file)( ::NS(PATH_TO_BEAMBEAM_BEAM_ELEMENTS) );

    SIXTRL_ASSERT( eb != nullptr );
    beam_monitor_index = ::NS(Buffer_get_num_of_objects)( eb );

    ret = ::NS(BeamMonitor_insert_end_of_turn_monitors)( eb,
        num_elem_by_elem_turns, num_turn_by_turn_turns,
        target_num_turns, skip_turns, ::NS(Buffer_get_objects_end)( eb ) );

    SIXTRL_ASSERT( ret == 0 );
    SIXTRL_ASSERT( ::NS(BeamMonitor_get_num_of_beam_monitor_objects)( eb ) ==
                   buf_size_t{ 2 } );

    SIXTRL_ASSERT( ::NS(Buffer_get_num_of_objects)( eb ) ==
        ( beam_monitor_index + buf_size_t{ 2 } ) );

    elem_by_elem_output_offset = buf_size_t{ 0 };
    beam_monitor_output_offset = buf_size_t{ 0 };
    out_min_turn_id = part_index_t{ 0 };

    begin_addr = address_t{ 0 };
    out_buffer_size = buf_size_t{ 0 };

    ret = ::NS(OutputBuffer_calculate_output_buffer_params)( eb, particles,
        num_elem_by_elem_turns, &num_objects, &num_slots, &num_dataptrs,
            &num_garbage, slot_size );

    ASSERT_TRUE( ret == 0 );
    ASSERT_TRUE( num_objects == buf_size_t{ 1 } +
        ::NS(BeamMonitor_get_num_of_beam_monitor_objects)( eb ) );

    ASSERT_TRUE( num_slots != buf_size_t{ 0 } );
    ASSERT_TRUE( num_dataptrs != buf_size_t{ 0 } );

    ::NS(Buffer_delete)( out_buffer );
    out_buffer = ::NS(Buffer_new)( buf_size_t{ 0 } );
    ret = ::NS(Buffer_reserve)( out_buffer,
        num_objects, num_slots, num_dataptrs, num_garbage );

    SIXTRL_ASSERT( ret == 0 );
    begin_addr = ::NS(Buffer_get_data_begin_addr)( out_buffer );
    out_buffer_size = ::NS(Buffer_get_size)( out_buffer );

    ret = ::NS(OutputBuffer_prepare)( eb, out_buffer, particles,
        num_elem_by_elem_turns, &elem_by_elem_output_offset,
            &beam_monitor_output_offset, &out_min_turn_id );

    ASSERT_TRUE( ret == 0 );

    ASSERT_TRUE( elem_by_elem_output_offset == buf_size_t{ 0 } );
    ASSERT_TRUE( beam_monitor_output_offset > elem_by_elem_output_offset );
    ASSERT_TRUE( out_min_turn_id == min_turn_id );

    ASSERT_TRUE( ::NS(Buffer_get_data_begin_addr)( out_buffer ) == begin_addr );
    ASSERT_TRUE( ::NS(Buffer_get_size)( out_buffer ) == out_buffer_size );
    ASSERT_TRUE( ::NS(Buffer_get_num_of_objects)( out_buffer ) == num_objects );
    ASSERT_TRUE( ::NS(Buffer_get_num_of_slots)( out_buffer ) <= num_slots );
    ASSERT_TRUE( ::NS(Buffer_get_num_of_dataptrs)( out_buffer ) <=
                 num_dataptrs );

    ASSERT_TRUE( ::NS(Buffer_get_num_of_garbage_ranges)( out_buffer ) <=
                 num_garbage );

    out_particles = ::NS(Particles_buffer_get_const_particles)(
            out_buffer, elem_by_elem_output_offset );

    ASSERT_TRUE( out_particles != nullptr );
    ASSERT_TRUE( requ_store_particles <= static_cast< buf_size_t >(
        ::NS(Particles_get_num_of_particles)( out_particles ) ) );

    for( buf_size_t jj = beam_monitor_index, ii = beam_monitor_output_offset ;
         ii < buf_size_t{ 2 } ; ++ii, ++jj )
    {
        ::NS(Object) const* obj = ::NS(Buffer_get_const_object)( eb, jj );

        SIXTRL_ASSERT( ::NS(Object_get_type_id)( obj ) ==
                       ::NS(OBJECT_TYPE_BEAM_MONITOR) );

        SIXTRL_ASSERT( ::NS(Object_get_size)( obj ) >=
                       sizeof( ::NS(BeamMonitor) ) );

        ::NS(BeamMonitor) const* monitor = reinterpret_cast<
            ::NS(BeamMonitor) const* >( static_cast< uintptr_t >(
                ::NS(Object_get_begin_addr)( obj ) ) );

        SIXTRL_ASSERT( monitor != nullptr );
        SIXTRL_ASSERT( ::NS(BeamMonitor_get_stored_num_particles)( monitor ) >
                     buf_size_t{ 0 } );

        out_particles = ::NS(Particles_buffer_get_const_particles)(
            out_buffer, ii );

        ASSERT_TRUE( out_particles != nullptr );

        ASSERT_TRUE( static_cast< buf_size_t >(
            ::NS(Particles_get_num_of_particles)( out_particles ) ) >=
                ::NS(BeamMonitor_get_stored_num_particles)( monitor ) );
    }

    /* -------------------------------------------------------------------- */

    num_objects = num_slots = num_dataptrs = num_garbage = buf_size_t{ 0 };
    beam_monitor_output_offset = elem_by_elem_output_offset = buf_size_t{ 0 };
    out_min_turn_id = part_index_t{ 0 };

    ret = ::NS(OutputBuffer_calculate_output_buffer_params_for_particles_sets)(
        eb, pb, buf_size_t{ 1 }, &pset_indices[ 0 ], num_elem_by_elem_turns,
            &num_objects, &num_slots, &num_dataptrs, &num_garbage, slot_size );

    ASSERT_TRUE( ret == 0 );
    ASSERT_TRUE( num_objects  != buf_size_t{ 0 } );
    ASSERT_TRUE( num_slots    != buf_size_t{ 0 } );
    ASSERT_TRUE( num_dataptrs != buf_size_t{ 0 } );

    ::NS(Buffer_delete)( out_buffer );
    out_buffer = NS(Buffer_new)( buf_size_t{ 0 } );

    ret = ::NS(Buffer_reserve)( out_buffer,
        num_objects, num_slots, num_dataptrs, num_garbage );

    SIXTRL_ASSERT( ret == 0 );
    begin_addr = ::NS(Buffer_get_data_begin_addr)( out_buffer );
    out_buffer_size = ::NS(Buffer_get_size)( out_buffer );

    ret = ::NS(OutputBuffer_prepare_for_particle_sets)( eb, out_buffer,
        pb, buf_size_t{ 1 }, &pset_indices[ 0 ], num_elem_by_elem_turns,
            &elem_by_elem_output_offset, &beam_monitor_output_offset,
                &out_min_turn_id );

    ASSERT_TRUE( ret == 0 );

    ASSERT_TRUE( elem_by_elem_output_offset == buf_size_t{ 0 } );
    ASSERT_TRUE( beam_monitor_output_offset > elem_by_elem_output_offset );
    ASSERT_TRUE( out_min_turn_id == min_turn_id );

    ASSERT_TRUE( ::NS(Buffer_get_data_begin_addr)( out_buffer ) == begin_addr );
    ASSERT_TRUE( ::NS(Buffer_get_size)( out_buffer ) == out_buffer_size );
    ASSERT_TRUE( ::NS(Buffer_get_num_of_objects)( out_buffer ) == num_objects );
    ASSERT_TRUE( ::NS(Buffer_get_num_of_slots)( out_buffer ) <= num_slots );
    ASSERT_TRUE( ::NS(Buffer_get_num_of_dataptrs)( out_buffer ) <=
                 num_dataptrs );

    ASSERT_TRUE( ::NS(Buffer_get_num_of_garbage_ranges)( out_buffer ) <=
                 num_garbage );

    out_particles = ::NS(Particles_buffer_get_const_particles)(
            out_buffer, elem_by_elem_output_offset );

    ASSERT_TRUE( out_particles != nullptr );
    ASSERT_TRUE( requ_store_particles <= static_cast< buf_size_t >(
        ::NS(Particles_get_num_of_particles)( out_particles ) ) );

    for( buf_size_t jj = beam_monitor_index, ii = beam_monitor_output_offset ;
         ii < buf_size_t{ 2 } ; ++ii, ++jj )
    {
        ::NS(Object) const* obj = ::NS(Buffer_get_const_object)( eb, jj );

        SIXTRL_ASSERT( ::NS(Object_get_type_id)( obj ) ==
                       ::NS(OBJECT_TYPE_BEAM_MONITOR) );

        SIXTRL_ASSERT( ::NS(Object_get_size)( obj ) >=
                       sizeof( ::NS(BeamMonitor) ) );

        ::NS(BeamMonitor) const* monitor = reinterpret_cast<
            ::NS(BeamMonitor) const* >( static_cast< uintptr_t >(
                ::NS(Object_get_begin_addr)( obj ) ) );

        SIXTRL_ASSERT( monitor != nullptr );
        SIXTRL_ASSERT( ::NS(BeamMonitor_get_stored_num_particles)( monitor ) >
                     buf_size_t{ 0 } );

        out_particles = ::NS(Particles_buffer_get_const_particles)(
            out_buffer, ii );

        ASSERT_TRUE( out_particles != nullptr );

        ASSERT_TRUE( static_cast< buf_size_t >(
            ::NS(Particles_get_num_of_particles)( out_particles ) ) >=
                ::NS(BeamMonitor_get_stored_num_particles)( monitor ) );
    }

    /* -------------------------------------------------------------------- */

    num_objects = num_slots = num_dataptrs = num_garbage = buf_size_t{ 0 };
    beam_monitor_output_offset = elem_by_elem_output_offset = buf_size_t{ 0 };
    out_min_turn_id = part_index_t{ 0 };

    ret = ::NS(OutputBuffer_calculate_output_buffer_params_detailed)(
        eb, min_part_id, max_part_id, min_elem_id, max_elem_id, min_turn_id,
            max_turn_id, num_elem_by_elem_turns, &num_objects, &num_slots,
                &num_dataptrs, &num_garbage, slot_size );

    ASSERT_TRUE( ret == 0 );
    ASSERT_TRUE( num_objects  != buf_size_t{ 0 } );
    ASSERT_TRUE( num_slots    != buf_size_t{ 0 } );
    ASSERT_TRUE( num_dataptrs != buf_size_t{ 0 } );

    ::NS(Buffer_delete)( out_buffer );
    out_buffer = NS(Buffer_new)( buf_size_t{ 0 } );

    ret = ::NS(Buffer_reserve)( out_buffer,
        num_objects, num_slots, num_dataptrs, num_garbage );

    SIXTRL_ASSERT( ret == 0 );
    begin_addr = ::NS(Buffer_get_data_begin_addr)( out_buffer );
    out_buffer_size = ::NS(Buffer_get_size)( out_buffer );

    cmp_max_elem_by_elem_turn_id = part_index_t{ 0 };

    ret = ::NS(OutputBuffer_prepare_detailed)( eb, out_buffer, min_part_id,
        max_part_id, min_elem_id, max_elem_id, min_turn_id, max_turn_id,
            num_elem_by_elem_turns, &elem_by_elem_output_offset,
                &beam_monitor_output_offset, &cmp_max_elem_by_elem_turn_id );

    ASSERT_TRUE( ret == 0 );

    ASSERT_TRUE( elem_by_elem_output_offset == buf_size_t{ 0 } );
    ASSERT_TRUE( beam_monitor_output_offset > elem_by_elem_output_offset );
    ASSERT_TRUE( out_min_turn_id == min_turn_id );

    ASSERT_TRUE( ::NS(Buffer_get_data_begin_addr)( out_buffer ) == begin_addr );
    ASSERT_TRUE( ::NS(Buffer_get_size)( out_buffer ) == out_buffer_size );
    ASSERT_TRUE( ::NS(Buffer_get_num_of_objects)( out_buffer ) == num_objects );
    ASSERT_TRUE( ::NS(Buffer_get_num_of_slots)( out_buffer ) <= num_slots );
    ASSERT_TRUE( ::NS(Buffer_get_num_of_dataptrs)( out_buffer ) <=
                 num_dataptrs );

    ASSERT_TRUE( ::NS(Buffer_get_num_of_garbage_ranges)( out_buffer ) <=
                 num_garbage );

    out_particles = ::NS(Particles_buffer_get_const_particles)(
            out_buffer, elem_by_elem_output_offset );

    ASSERT_TRUE( out_particles != nullptr );
    ASSERT_TRUE( requ_store_particles <= static_cast< buf_size_t >(
        ::NS(Particles_get_num_of_particles)( out_particles ) ) );

    ASSERT_TRUE( max_elem_by_elem_turn_id == cmp_max_elem_by_elem_turn_id );

    for( buf_size_t jj = beam_monitor_index, ii = beam_monitor_output_offset ;
         ii < buf_size_t{ 2 } ; ++ii, ++jj )
    {
        ::NS(Object) const* obj = ::NS(Buffer_get_const_object)( eb, jj );

        SIXTRL_ASSERT( ::NS(Object_get_type_id)( obj ) ==
                       ::NS(OBJECT_TYPE_BEAM_MONITOR) );

        SIXTRL_ASSERT( ::NS(Object_get_size)( obj ) >=
                       sizeof( ::NS(BeamMonitor) ) );

        ::NS(BeamMonitor) const* monitor = reinterpret_cast<
            ::NS(BeamMonitor) const* >( static_cast< uintptr_t >(
                ::NS(Object_get_begin_addr)( obj ) ) );

        SIXTRL_ASSERT( monitor != nullptr );
        SIXTRL_ASSERT( ::NS(BeamMonitor_get_stored_num_particles)( monitor ) >
                     buf_size_t{ 0 } );

        out_particles = ::NS(Particles_buffer_get_const_particles)(
            out_buffer, ii );

        ASSERT_TRUE( out_particles != nullptr );

        ASSERT_TRUE( static_cast< buf_size_t >(
            ::NS(Particles_get_num_of_particles)( out_particles ) ) >=
                     ::NS(BeamMonitor_get_stored_num_particles)( monitor ) );
    }

    /* ===================================================================== */
    /* Cleanup */

    ::NS(Buffer_delete)( pb );
    ::NS(Buffer_delete)( eb );
    ::NS(Buffer_delete)( out_buffer );
}

/* end: tests/sixtracklib/common/test_output_buffer_c99.cpp */
