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

TEST( C99_Cpu_CpuTrackJobTrackUntilTests, TrackUntilSingleParticleSetSimpleTest )
{
    namespace st = SIXTRL_CXX_NAMESPACE;

    using particles_t    = ::NS(Particles);
    using track_job_t    = ::NS(CpuTrackJob);
    using buffer_t       = ::NS(Buffer);
    using buf_size_t     = ::NS(buffer_size_t);
    using track_status_t = ::NS(track_status_t);
    using ctrl_status_t  = ::NS(arch_status_t);
    using real_t         = ::NS(particle_real_t);

    real_t const ABS_TOLERANCE = real_t{ 1e-14 };

    buffer_t* in_particles = ::NS(Buffer_new_from_file)(
        ::NS(PATH_TO_BEAMBEAM_PARTICLES_DUMP) );

    buffer_t* beam_elem_buffer = ::NS(Buffer_new_from_file)(
        ::NS(PATH_TO_BEAMBEAM_BEAM_ELEMENTS) );

    buffer_t* cmp_track_pb = ::NS(Buffer_new)( buf_size_t{ 0 } );

    SIXTRL_ASSERT( ::NS(Particles_buffer_get_const_particles)(
        in_particles, buf_size_t{ 0 } ) != nullptr );

    particles_t* cmp_particles = ::NS(Particles_add_copy)( cmp_track_pb,
        ::NS(Particles_buffer_get_const_particles)(
            in_particles, buf_size_t{ 0 } ) );

    SIXTRL_ASSERT( cmp_particles != nullptr );

    /* --------------------------------------------------------------------- */
    /* Perform comparison tracking over lattice: */

    buf_size_t const NUM_PSETS  = buf_size_t{ 1 };
    buf_size_t track_pset_index = buf_size_t{ 0 };
    buf_size_t const UNTIL_TURN = buf_size_t{ 20 };

    track_status_t track_status =
    ::NS(TestTrackCpu_track_particles_until_turn_cpu)(
        cmp_track_pb, NUM_PSETS, &track_pset_index,
            beam_elem_buffer, UNTIL_TURN );

    SIXTRL_ASSERT( track_status == ::NS(TRACK_SUCCESS) );

    /* Create a dedicated buffer for tracking */
    buffer_t* track_pb = ::NS(Buffer_new)( buf_size_t{ 0 } );

    particles_t* particles = ::NS(Particles_add_copy)(
        track_pb, ::NS(Particles_buffer_get_const_particles(
            in_particles, buf_size_t{ 0 } ) ) );

    SIXTRL_ASSERT( particles != nullptr );

    track_job_t* track_job = NS(CpuTrackJob_new)( track_pb, beam_elem_buffer );
    ASSERT_TRUE( track_job != nullptr );

    if( !::NS(TrackJobNew_is_in_debug_mode)( track_job ) )
    {
        ::NS(TrackJobNew_enable_debug_mode)( track_job );
    }

    ASSERT_TRUE( ::NS(TrackJobNew_is_in_debug_mode)( track_job ) );
    ASSERT_TRUE( !::NS(TrackJobNew_requires_collecting)( track_job ) );

    track_status = ::NS(TrackJobNew_track_until)( track_job, UNTIL_TURN );
    ASSERT_TRUE( track_status == ::NS(TRACK_SUCCESS) );
    ASSERT_TRUE( ::NS(TrackJobNew_get_particles_buffer)( track_job ) ==
                 track_pb );

    particles = ::NS(Particles_buffer_get_particles)( track_pb,buf_size_t{ 0 });
    SIXTRL_ASSERT( particles != nullptr );

    ASSERT_TRUE( ( cmp_particles != nullptr ) && ( particles != nullptr ) &&
        ( ( 0 == ::NS(Particles_compare_values)(
                cmp_particles, particles ) ) ||
          ( ( ABS_TOLERANCE > real_t{ 0 } ) &&
            ( 0 == ::NS(Particles_compare_values_with_treshold)(
                cmp_particles, particles, ABS_TOLERANCE ) ) ) ) );

    ctrl_status_t status = ::NS(Particles_copy)( particles,
        ::NS(Particles_buffer_get_const_particles)( in_particles,
            buf_size_t{ 0 } ) );

    SIXTRL_ASSERT( status == ::NS(ARCH_STATUS_SUCCESS) );

    status = ::NS(TrackJobNew_disable_debug_mode)( track_job );
    ASSERT_TRUE( status == ::NS(ARCH_STATUS_SUCCESS) );
    ASSERT_TRUE( !::NS(TrackJobNew_is_in_debug_mode)( track_job ) );

    status = ::NS(TrackJobNew_reset)(
        track_job, track_pb, beam_elem_buffer, nullptr );
    ASSERT_TRUE( status == ::NS(ARCH_STATUS_SUCCESS) );

    /* Check whether the update of the particles state has worked */
    particles = ::NS(Particles_buffer_get_particles)(
        track_pb, buf_size_t{ 0 } );

    SIXTRL_ASSERT( particles != nullptr );
    SIXTRL_ASSERT( 0 == ::NS(Particles_compare_values)(
        particles, ::NS(Particles_buffer_get_const_particles)(
            in_particles, buf_size_t{ 0 } ) ) );

    /* Perform tracking again, this time not in debug mode */
    track_status = ::NS(TrackJobNew_track_until)( track_job, UNTIL_TURN );
    ASSERT_TRUE( track_status == ::NS(TRACK_SUCCESS) );

    /* Compare particles again against the cpu tracking result */
    particles = ::NS(Particles_buffer_get_particles)(
        track_pb, buf_size_t{ 0 } );


    ASSERT_TRUE( ( particles != nullptr ) && ( cmp_particles != nullptr ) &&
        ( ( 0 == ::NS(Particles_compare_values)(
                cmp_particles, particles ) ) ||
          ( ( ABS_TOLERANCE > real_t{ 0 } ) &&
            ( 0 == ::NS(Particles_compare_values_with_treshold)(
                cmp_particles, particles, ABS_TOLERANCE ) ) ) ) );


    ::NS(TrackJobNew_delete)( track_job );

    ::NS(Buffer_delete)( track_pb );
    ::NS(Buffer_delete)( cmp_track_pb );
    ::NS(Buffer_delete)( beam_elem_buffer );
    ::NS(Buffer_delete)( in_particles );
}

/* end: tests/sixtracklib/common/track/test_track_job_track_until_c99.cpp */
