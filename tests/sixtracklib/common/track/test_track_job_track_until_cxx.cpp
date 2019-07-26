#include "sixtracklib/common/track/track_job_cpu.hpp"

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
#include "sixtracklib/common/buffer.hpp"
#include "sixtracklib/common/particles.hpp"
#include "sixtracklib/common/be_monitor/be_monitor.hpp"

TEST( CXX_Cpu_CpuTrackJobTrackUntilTests, TrackUntilSingleParticleSetSimpleTest )
{
    namespace st = SIXTRL_CXX_NAMESPACE;

    using particles_t    = st::Particles;
    using track_job_t    = st::CpuTrackJob;
    using buffer_t       = track_job_t::buffer_t;
    using buf_size_t     = track_job_t::size_type;
    using track_status_t = track_job_t::track_status_t;
    using ctrl_status_t  = track_job_t::status_t;
    using real_t         = ::NS(particle_real_t);

    real_t const ABS_TOLERANCE = real_t{ 1e-14 };

    buffer_t in_particles( ::NS(PATH_TO_BEAMBEAM_PARTICLES_DUMP) );
    buffer_t beam_elem_buffer( ::NS(PATH_TO_BEAMBEAM_BEAM_ELEMENTS) );

    buffer_t cmp_track_pb;

    SIXTRL_ASSERT( in_particles.get< particles_t >(
        buf_size_t{ 0 } ) != nullptr );

    particles_t* cmp_particles = cmp_track_pb.addCopy(
        *in_particles.get< particles_t >( buf_size_t{ 0 } ) );

    SIXTRL_ASSERT( cmp_particles != nullptr );

    /* --------------------------------------------------------------------- */
    /* Perform comparison tracking over lattice: */

    buf_size_t const NUM_PSETS  = buf_size_t{ 1 };
    buf_size_t track_pset_index = buf_size_t{ 0 };
    buf_size_t const UNTIL_TURN = buf_size_t{ 20 };

    track_status_t track_status =
    ::NS(TestTrackCpu_track_particles_until_turn_cpu)(
        cmp_track_pb.getCApiPtr(), NUM_PSETS, &track_pset_index,
            beam_elem_buffer.getCApiPtr(), UNTIL_TURN );

    SIXTRL_ASSERT( track_status == st::TRACK_SUCCESS );

    /* -------------------------------------------------------------------- */
    /* Retrieve list of available nodes */

    /* Create a dedicated buffer for tracking */
    buffer_t track_pb;
    particles_t* particles = track_pb.addCopy(
        *in_particles.get< particles_t >( buf_size_t{ 0 } ) );

    SIXTRL_ASSERT( particles != nullptr );

    /* Create a track job on the current node */
    track_job_t track_job( track_pb, beam_elem_buffer );

    if( !track_job.isInDebugMode() )
    {
        track_job.enableDebugMode();
    }

    ASSERT_TRUE( track_job.isInDebugMode() );

    track_status = track_job.trackUntil( UNTIL_TURN );
    ASSERT_TRUE( track_status == st::TRACK_SUCCESS );

    ASSERT_TRUE( !track_job.requiresCollecting() );
    SIXTRL_ASSERT( track_job.ptrParticlesBuffer() == &track_pb );

    particles = track_pb.get< particles_t >( buf_size_t{ 0 } );
    SIXTRL_ASSERT( particles != nullptr );

    ASSERT_TRUE( ( cmp_particles != nullptr ) && ( particles != nullptr ) &&
        ( ( 0 == ::NS(Particles_compare_values)(
            cmp_particles->getCApiPtr(), particles->getCApiPtr() ) ) ||
          ( ( ABS_TOLERANCE > real_t{ 0 } ) &&
            ( 0 == ::NS(Particles_compare_values_with_treshold)(
                    cmp_particles->getCApiPtr(), particles->getCApiPtr(),
                        ABS_TOLERANCE ) ) ) ) );

    ctrl_status_t status = particles->copy(
        in_particles.get< particles_t >( buf_size_t{ 0 } ) );

    SIXTRL_ASSERT( status == st::ARCH_STATUS_SUCCESS );

    track_job.disableDebugMode();

    ASSERT_TRUE( !track_job.isInDebugMode() );

    status = track_job.reset( track_pb, beam_elem_buffer );
    ASSERT_TRUE( status == st::ARCH_STATUS_SUCCESS );

    /* Check whether the update of the particles state has worked */
    status = track_job.collectParticles();
    SIXTRL_ASSERT( status == st::ARCH_STATUS_SUCCESS );
    particles = track_pb.get< particles_t >( buf_size_t{ 0 } );
    SIXTRL_ASSERT( particles != nullptr );
    SIXTRL_ASSERT( 0 == ::NS(Particles_compare_values)(
        particles->getCApiPtr(),
        in_particles.get< particles_t >( buf_size_t{ 0 } )->getCApiPtr() ) );

    /* Perform tracking again, this time not in debug mode */
    track_status = track_job.trackUntil( UNTIL_TURN );
    ASSERT_TRUE( track_status == st::TRACK_SUCCESS );

    /* Access the results and compare against the cpu tracking result */

    SIXTRL_ASSERT( track_job.ptrParticlesBuffer() == &track_pb );
    particles = track_pb.get< particles_t >( buf_size_t{ 0 } );

    ASSERT_TRUE( ( particles != nullptr ) &&
        ( ( 0 == ::NS(Particles_compare_values)(
            cmp_particles->getCApiPtr(), particles->getCApiPtr() ) ) ||
          ( ( ABS_TOLERANCE > real_t{ 0 } ) &&
            ( 0 == ::NS(Particles_compare_values_with_treshold)(
                    cmp_particles->getCApiPtr(), particles->getCApiPtr(),
                        ABS_TOLERANCE ) ) ) ) );
}

/* end: tests/sixtracklib/common/track/test_track_job_track_until_cxx.cpp */
