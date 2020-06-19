#include "sixtracklib/common/track/track_job_cpu.h"

#include <iomanip>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "sixtracklib/testlib.h"

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/control/definitions.h"
#include "sixtracklib/common/track/definitions.h"
#include "sixtracklib/common/buffer.h"
#include "sixtracklib/common/particles.h"
#include "sixtracklib/common/be_monitor/be_monitor.h"

TEST( C99_Cpu_CpuTrackJobTrackElemByElemTests,
      TrackElemByElemSingleParticleSetSimpleTest )
{
    namespace st = SIXTRL_CXX_NAMESPACE;

    using particles_t           = ::NS(Particles);
    using track_job_t           = ::NS(CpuTrackJob);
    using buffer_t              = ::NS(Buffer);
    using buf_size_t            = ::NS(buffer_size_t);
    using track_status_t        = ::NS(track_status_t);
    using ctrl_status_t         = ::NS(arch_status_t);
    using real_t                = ::NS(particle_real_t);
    using pindex_t              = ::NS(particle_index_t);
    using elem_by_elem_config_t = ::NS(ElemByElemConfig);

    real_t const ABS_TOLERANCE = real_t{ 1e-14 };

    buffer_t* in_particles = ::NS(Buffer_new_from_file)(
        ::NS(PATH_TO_BEAMBEAM_PARTICLES_DUMP) );

    buffer_t* beam_elem_buffer = ::NS(Buffer_new_from_file)(
        ::NS(PATH_TO_BEAMBEAM_BEAM_ELEMENTS) );

    buffer_t* cmp_track_pb = ::NS(Buffer_new)( buf_size_t{ 0 } );
    buffer_t* cmp_output_buffer = ::NS(Buffer_new)( buf_size_t{ 0 } );
    buffer_t* elem_by_elem_config_buffer = ::NS(Buffer_new)( buf_size_t{ 0 } );

    SIXTRL_ASSERT( ::NS(Particles_buffer_get_const_particles)(
        in_particles, buf_size_t{ 0 } ) != nullptr );

    particles_t* cmp_particles = ::NS(Particles_add_copy)( cmp_track_pb,
        ::NS(Particles_buffer_get_const_particles)(
            in_particles, buf_size_t{ 0 } ) );

    elem_by_elem_config_t* elem_by_elem_config =
        ::NS(ElemByElemConfig_new)( elem_by_elem_config_buffer );
    SIXTRL_ASSERT( elem_by_elem_config != nullptr );

    SIXTRL_ASSERT( cmp_particles != nullptr );

    /* --------------------------------------------------------------------- */
    /* Perform comparison tracking over lattice: */

    buf_size_t const NUM_PSETS  = buf_size_t{ 1 };
    buf_size_t track_pset_index = buf_size_t{ 0 };
    buf_size_t const UNTIL_TURN_ELEM_BY_ELEM = buf_size_t{ 5 };

    buf_size_t num_elem_by_elem_objects = buf_size_t{ 0 };
    pindex_t const start_be_index = pindex_t{ 0 };

    pindex_t min_particle_id, max_particle_id;
    pindex_t min_at_element_id, max_at_element_id;
    pindex_t min_at_turn, max_at_turn;

    ctrl_status_t status = ::NS(OutputBuffer_get_min_max_attributes)(
        cmp_particles, beam_elem_buffer, &min_particle_id,
        &max_particle_id, &min_at_element_id,
        &max_at_element_id, &min_at_turn, &max_at_turn,
        &num_elem_by_elem_objects, start_be_index );

    SIXTRL_ASSERT( status == st::ARCH_STATUS_SUCCESS );

    buf_size_t elem_by_elem_out_offset_index = buf_size_t{ 0 };
    buf_size_t beam_element_out_offset_index = buf_size_t{ 0 };
    pindex_t   max_elem_by_elem_turn_id      = pindex_t{ -1 };

    status = ::NS(OutputBuffer_prepare_detailed)(
        beam_elem_buffer, cmp_output_buffer,
        min_particle_id, max_particle_id, min_at_element_id, max_at_element_id,
        min_at_turn, max_at_turn, UNTIL_TURN_ELEM_BY_ELEM,
        &elem_by_elem_out_offset_index, &beam_element_out_offset_index,
        &max_elem_by_elem_turn_id );

    SIXTRL_ASSERT( status == st::ARCH_STATUS_SUCCESS );
    SIXTRL_ASSERT( max_elem_by_elem_turn_id >= pindex_t{ 0 } );

    ::NS(ElemByElemConfig_preset)( elem_by_elem_config );

    status = ::NS(ElemByElemConfig_init_detailed)( elem_by_elem_config,
        ::NS(ELEM_BY_ELEM_ORDER_DEFAULT), min_particle_id, max_particle_id,
        min_at_element_id, max_at_element_id, min_at_turn,
        max_elem_by_elem_turn_id, true );

    SIXTRL_ASSERT( status == st::ARCH_STATUS_SUCCESS );

    status = ::NS(ElemByElemConfig_assign_output_buffer)(
        elem_by_elem_config, cmp_output_buffer, elem_by_elem_out_offset_index );

    SIXTRL_ASSERT( status == st::ARCH_STATUS_SUCCESS );

    track_status_t track_status =
    ::NS(TestTrackCpu_track_particles_elem_by_elem_until_turn_cpu)(
        cmp_track_pb, NUM_PSETS, &track_pset_index,
            beam_elem_buffer, elem_by_elem_config, UNTIL_TURN_ELEM_BY_ELEM );

    SIXTRL_ASSERT( track_status == st::TRACK_SUCCESS );

    particles_t* cmp_elem_by_elem_particles =
        ::NS(Particles_buffer_get_particles)(
            cmp_output_buffer, elem_by_elem_out_offset_index );

    SIXTRL_ASSERT( cmp_elem_by_elem_particles != nullptr );

    /* -------------------------------------------------------------------- */
    /* Create a dedicated buffer for tracking & a track job instance        */

    buffer_t* track_pb = ::NS(Buffer_new)( buf_size_t{ 0 } );
    SIXTRL_ASSERT( track_pb != nullptr );

    particles_t* particles = ::NS(Particles_add_copy)(
        track_pb, ::NS(Particles_buffer_get_const_particles(
            in_particles, buf_size_t{ 0 } ) ) );

    SIXTRL_ASSERT( particles != nullptr );

    track_job_t* track_job = ::NS(CpuTrackJob_new_with_output)(
        track_pb, beam_elem_buffer, nullptr, UNTIL_TURN_ELEM_BY_ELEM );

    ASSERT_TRUE( track_job != nullptr );
    ASSERT_TRUE( ::NS(TrackJobNew_has_output_buffer)( track_job ) );
    ASSERT_TRUE( ::NS(TrackJobNew_owns_output_buffer)( track_job ) );
    ASSERT_TRUE( ::NS(TrackJobNew_has_elem_by_elem_output)( track_job ) );
    ASSERT_TRUE( ::NS(TrackJobNew_get_elem_by_elem_output_buffer_offset)(
        track_job ) == elem_by_elem_out_offset_index );

    if( !::NS(TrackJobNew_is_in_debug_mode)( track_job ) )
    {
        ::NS(TrackJobNew_enable_debug_mode)( track_job );
    }

    ASSERT_TRUE( ::NS(TrackJobNew_is_in_debug_mode)( track_job ) );
    ASSERT_TRUE( !::NS(TrackJobNew_requires_collecting)( track_job ) );

    track_status = ::NS(TrackJobNew_track_elem_by_elem)(
        track_job, UNTIL_TURN_ELEM_BY_ELEM );

    ASSERT_TRUE( track_status == ::NS(TRACK_SUCCESS) );

    status = ::NS(TrackJobNew_collect_particles)( track_job );
    ASSERT_TRUE( status == ::NS(ARCH_STATUS_SUCCESS) );
    ASSERT_TRUE( ::NS(TrackJobNew_get_particles_buffer)( track_job ) ==
                 track_pb );

    particles = ::NS(Particles_buffer_get_particles)(
        track_pb, buf_size_t{ 0 } );

    SIXTRL_ASSERT( particles != nullptr );

    status = ::NS(TrackJobNew_collect_output)( track_job );
    ASSERT_TRUE( status == ::NS(ARCH_STATUS_SUCCESS) );
    ASSERT_TRUE( ::NS(TrackJobNew_get_output_buffer)( track_job ) !=
                 nullptr );

    particles_t* elem_by_elem_particles =
        ::NS(Particles_buffer_get_particles)(
            ::NS(TrackJobNew_get_output_buffer)( track_job ),
             elem_by_elem_out_offset_index );

    SIXTRL_ASSERT( elem_by_elem_particles != nullptr );

    ASSERT_TRUE(
        ( cmp_particles != nullptr ) && ( particles != nullptr ) &&
        ( ( 0 == ::NS(Particles_compare_values)(
            cmp_particles, particles ) ) ||
          ( ( ABS_TOLERANCE > real_t{ 0 } ) &&
            ( 0 == ::NS(Particles_compare_values_with_treshold)(
                    cmp_particles, particles, ABS_TOLERANCE ) ) ) ) );

    ASSERT_TRUE( ( cmp_elem_by_elem_particles != nullptr ) &&
                 ( elem_by_elem_particles != nullptr ) &&
        ( ( 0 == ::NS(Particles_compare_values)(
            cmp_elem_by_elem_particles, elem_by_elem_particles ) ) ||
          ( ( ABS_TOLERANCE > real_t{ 0 } ) &&
            ( 0 == ::NS(Particles_compare_values_with_treshold)(
                cmp_elem_by_elem_particles, elem_by_elem_particles,
                    ABS_TOLERANCE ) ) ) ) );

    /* ***************************************************************** */
    /* Switch to non-debug mode: */

    status = ::NS(Particles_copy)( particles,
        ::NS(Particles_buffer_get_const_particles)( in_particles,
            buf_size_t{ 0 } ) );

    SIXTRL_ASSERT( status == ::NS(ARCH_STATUS_SUCCESS) );

    status = ::NS(TrackJobNew_disable_debug_mode)( track_job );
    ASSERT_TRUE( status == ::NS(ARCH_STATUS_SUCCESS) );
    ASSERT_TRUE( !::NS(TrackJobNew_is_in_debug_mode)( track_job ) );

    status = ::NS(TrackJobNew_reset_with_output)( track_job, track_pb,
        beam_elem_buffer, nullptr, UNTIL_TURN_ELEM_BY_ELEM );
    ASSERT_TRUE( status == ::NS(ARCH_STATUS_SUCCESS) );

    /* Check whether the update of the particles state has worked */
    status = ::NS(TrackJobNew_collect_particles)( track_job );
    SIXTRL_ASSERT( status == ::NS(ARCH_STATUS_SUCCESS) );

    particles = ::NS(Particles_buffer_get_particles)(
        track_pb, buf_size_t{ 0 } );

    SIXTRL_ASSERT( particles != nullptr );
    SIXTRL_ASSERT( 0 == ::NS(Particles_compare_values)(
        particles, ::NS(Particles_buffer_get_const_particles)(
            in_particles, buf_size_t{ 0 } ) ) );

    /* Perform tracking again, this time not in debug mode */
    track_status = ::NS(TrackJobNew_track_elem_by_elem)(
        track_job, UNTIL_TURN_ELEM_BY_ELEM );

    ASSERT_TRUE( track_status == ::NS(TRACK_SUCCESS) );

    /* Collect the results again and ... */
    status = ::NS(TrackJobNew_collect_particles)( track_job );
    ASSERT_TRUE( status == ::NS(ARCH_STATUS_SUCCESS) );
    SIXTRL_ASSERT( ::NS(TrackJobNew_get_particles_buffer)( track_job ) ==
        track_pb );

    particles = ::NS(Particles_buffer_get_particles)(
        track_pb, buf_size_t{ 0 } );

    SIXTRL_ASSERT( particles != nullptr );

    status = ::NS(TrackJobNew_collect_output)( track_job );
    ASSERT_TRUE( status == ::NS(ARCH_STATUS_SUCCESS) );
    ASSERT_TRUE( ::NS(TrackJobNew_get_output_buffer)( track_job ) !=
                 nullptr );

    elem_by_elem_particles = ::NS(Particles_buffer_get_particles)(
            ::NS(TrackJobNew_get_output_buffer)( track_job ),
            elem_by_elem_out_offset_index );

    SIXTRL_ASSERT( elem_by_elem_particles != nullptr );

    /* ... compare against the cpu tracking result */

    ASSERT_TRUE(
        ( cmp_particles != nullptr ) && ( particles != nullptr ) &&
        ( ( 0 == ::NS(Particles_compare_values)(
            cmp_particles, particles ) ) ||
          ( ( ABS_TOLERANCE > real_t{ 0 } ) &&
            ( 0 == ::NS(Particles_compare_values_with_treshold)(
                    cmp_particles, particles, ABS_TOLERANCE ) ) ) ) );

    ASSERT_TRUE( ( cmp_elem_by_elem_particles != nullptr ) &&
                 ( elem_by_elem_particles != nullptr ) &&
        ( ( 0 == ::NS(Particles_compare_values)(
            cmp_elem_by_elem_particles, elem_by_elem_particles ) ) ||
          ( ( ABS_TOLERANCE > real_t{ 0 } ) &&
            ( 0 == ::NS(Particles_compare_values_with_treshold)(
                cmp_elem_by_elem_particles, elem_by_elem_particles,
                    ABS_TOLERANCE ) ) ) ) );

    ::NS(TrackJobNew_delete)( track_job );

    ::NS(Buffer_delete)( track_pb );
    ::NS(Buffer_delete)( cmp_track_pb );
    ::NS(Buffer_delete)( cmp_output_buffer );
    ::NS(Buffer_delete)( beam_elem_buffer );
    ::NS(Buffer_delete)( in_particles );
    ::NS(Buffer_delete)( elem_by_elem_config_buffer );
}

/* end: tests/sixtracklib/common/track/test_track_job_track_elem_by_elem_c99.cpp */
