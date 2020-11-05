#include "sixtracklib/common/track/track_job_cpu.hpp"

#include <algorithm>
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
#include "sixtracklib/common/be_drift/be_drift.hpp"
#include "sixtracklib/common/be_multipole/be_multipole.hpp"
#include "sixtracklib/common/be_limit/be_limit_rect.hpp"
#include "sixtracklib/common/control/definitions.h"
#include "sixtracklib/common/track/definitions.h"
#include "sixtracklib/common/buffer.hpp"
#include "sixtracklib/common/particles.hpp"
#include "sixtracklib/common/be_monitor/be_monitor.hpp"
#include "sixtracklib/common/track/track.h"

TEST( CXXCpuCpuTrackJobTrackLineTests, CmpWithTrackUntilTest )
{
    namespace st = SIXTRL_CXX_NAMESPACE;

    using particles_t    = st::Particles;
    using track_job_t    = st::CpuTrackJob;
    using buffer_t       = track_job_t::buffer_t;
    using buf_size_t     = track_job_t::size_type;
    using track_status_t = track_job_t::track_status_t;
    using ctrl_status_t  = track_job_t::status_t;
    using real_t         = particles_t::real_t;
    using pindex_t       = particles_t::index_t;

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

    /* Create a dedicated buffer for tracking */
    buffer_t track_pb;
    particles_t* particles = track_pb.addCopy(
        *in_particles.get< particles_t >( buf_size_t{ 0 } ) );

    SIXTRL_ASSERT( particles != nullptr );

    /* --------------------------------------------------------------------- */
    /* Create a track job */
    track_job_t track_job( track_pb, beam_elem_buffer );

    if( !track_job.isInDebugMode() )
    {
        track_job.enableDebugMode();
    }

    ASSERT_TRUE( !track_job.requiresCollecting() );
    ASSERT_TRUE( track_job.isInDebugMode() );

    /* ***************************************************************** */

    buf_size_t const num_beam_elements = beam_elem_buffer.getNumObjects();

    buf_size_t const chunk_length = buf_size_t{ 10 };
    buf_size_t num_chunks = num_beam_elements / chunk_length;

    if( ( num_beam_elements % chunk_length ) != buf_size_t{ 0 } )
    {
        ++num_chunks;
    }

    pindex_t min_particle_id, max_particle_id;
    pindex_t min_at_element_id, max_at_element_id;
    pindex_t min_at_turn_id, max_at_turn_id;

    ::NS(Particles_init_min_max_attributes_for_find)(
        &min_particle_id, &max_particle_id, &min_at_element_id,
        &max_at_element_id, &min_at_turn_id, &max_at_turn_id );

    ctrl_status_t status = ::NS(Particles_find_min_max_attributes)(
        particles, &min_particle_id, &max_particle_id,
            &min_at_element_id, &max_at_element_id, &min_at_turn_id,
                &max_at_turn_id );

    SIXTRL_ASSERT( status == st::ARCH_STATUS_SUCCESS );
    SIXTRL_ASSERT( min_at_turn_id == max_at_turn_id );
    SIXTRL_ASSERT( min_at_element_id == max_at_element_id );

    for( buf_size_t kk = min_at_turn_id ; kk < UNTIL_TURN ; ++kk )
    {
        for( buf_size_t jj = buf_size_t{ 0 } ; jj < num_chunks ; ++jj )
        {
            buf_size_t const be_begin_idx = jj * chunk_length;
            buf_size_t const be_end_idx = std::min(
                be_begin_idx + chunk_length, num_beam_elements );

            bool const finish_turn = be_end_idx >= num_beam_elements;

            track_status = track_job.trackLine(
                be_begin_idx, be_end_idx, finish_turn );

            ASSERT_TRUE( track_status == ::NS(TRACK_SUCCESS) );
        }
    }

    /* ***************************************************************** */

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

    status = particles->copy(
        in_particles.get< particles_t >( buf_size_t{ 0 } ) );

    SIXTRL_ASSERT( status == st::ARCH_STATUS_SUCCESS );

    track_job.disableDebugMode();

    ASSERT_TRUE( !track_job.isInDebugMode() );

    status = track_job.reset( track_pb, beam_elem_buffer );
    ASSERT_TRUE( status == st::ARCH_STATUS_SUCCESS );

    /* Check whether the update of the particles state has worked */
    particles = track_pb.get< particles_t >( buf_size_t{ 0 } );
    SIXTRL_ASSERT( particles != nullptr );
    SIXTRL_ASSERT( 0 == ::NS(Particles_compare_values)(
        particles->getCApiPtr(),
        in_particles.get< particles_t >( buf_size_t{ 0 } )->getCApiPtr() ) );

    /* ***************************************************************** */
    /* Perform tracking again, this time not in debug mode */

    ::NS(Particles_init_min_max_attributes_for_find)(
        &min_particle_id, &max_particle_id, &min_at_element_id,
        &max_at_element_id, &min_at_turn_id, &max_at_turn_id );

    status = ::NS(Particles_find_min_max_attributes)(
        particles, &min_particle_id, &max_particle_id,
            &min_at_element_id, &max_at_element_id, &min_at_turn_id,
                &max_at_turn_id );

    SIXTRL_ASSERT( status == st::ARCH_STATUS_SUCCESS );
    SIXTRL_ASSERT( min_at_turn_id == max_at_turn_id );
    SIXTRL_ASSERT( min_at_element_id == max_at_element_id );

    for( buf_size_t kk = min_at_turn_id ; kk < UNTIL_TURN ; ++kk )
    {
        for( buf_size_t jj = buf_size_t{ 0 } ; jj < num_chunks ; ++jj )
        {
            buf_size_t const be_begin_idx = jj * chunk_length;
            buf_size_t const be_end_idx = std::min(
                be_begin_idx + chunk_length, num_beam_elements );

            bool const finish_turn = be_end_idx >= num_beam_elements;

            track_status = track_job.trackLine(
                be_begin_idx, be_end_idx, finish_turn );

            ASSERT_TRUE( track_status == st::TRACK_SUCCESS );
        }
    }

    /* ***************************************************************** */
    /* Compare the results against the cpu tracking result */

    SIXTRL_ASSERT( track_job.ptrParticlesBuffer() == &track_pb );
    particles = track_pb.get< particles_t >( buf_size_t{ 0 } );

    ASSERT_TRUE( ( particles != nullptr ) && ( cmp_particles != nullptr ) &&
        ( ( 0 == ::NS(Particles_compare_values)(
                cmp_particles->getCApiPtr(), particles->getCApiPtr() ) ) ||
          ( ( ABS_TOLERANCE > real_t{ 0 } ) &&
            ( 0 == ::NS(Particles_compare_values_with_treshold)(
                cmp_particles->getCApiPtr(), particles->getCApiPtr(),
                        ABS_TOLERANCE ) ) ) ) );
}

TEST( CXXCpuCpuTrackJobTrackLineTests, LostParticleBehaviour )
{
    namespace  st = SIXTRL_CXX_NAMESPACE;
    using npart_t = st::particle_num_elements_t;

    st::Buffer elements;

    elements.createNew< st::DriftExact >();
    elements.createNew< st::Multipole >( 2 );
    elements.createNew< st::LimitRect >();
    elements.createNew< st::DriftExact >();

    st::DriftExact*  dr0 = elements.get< st::DriftExact >( 0 );
    st::Multipole*  quad = elements.get< st::Multipole >( 1 );
    st::LimitRect* limit = elements.get< st::LimitRect >( 2 );
    st::DriftExact*  dr1 = elements.get< st::DriftExact >( 3 );

    SIXTRL_ASSERT( dr0   != nullptr );
    SIXTRL_ASSERT( dr1   != nullptr );
    SIXTRL_ASSERT( quad  != nullptr );
    SIXTRL_ASSERT( limit != nullptr );

    dr0->setLength( double{ 5 } );

    quad->setKnlValue(  0.0, 0 );
    quad->setKnlValue( 1e-2, 1 );
    quad->setLength( 1.0 );

    limit->setMaxX(  0.1 );
    limit->setMinX( -0.1 );
    limit->setMaxY(  0.1 );
    limit->setMinY( -0.1 );

    dr1->setLength( double{ 5 } );

    st::Buffer cmp_pbuffer;
    st::Buffer track_pbuffer;

    constexpr npart_t NUM_PARTICLES = npart_t{ 100 };
    cmp_pbuffer.createNew< st::Particles >( NUM_PARTICLES );
    track_pbuffer.createNew< st::Particles >( NUM_PARTICLES );

    double const pc0   = double{ 10e9 };
    double const mass0 = double{ 1e9 };
    double const q0    = double{ 1 };

    double const energy0 = std::sqrt( mass0 * mass0 + pc0 * pc0 );
    double const gamma0  = energy0 / mass0;
    double const beta0   = std::sqrt( double{ 1 } -
        double{ 1 }  / ( gamma0 * gamma0 ) );

    st::Particles* cmp_particles =  cmp_pbuffer.get< st::Particles >( 0 );

    for( npart_t ii = npart_t{ 0 } ; ii < NUM_PARTICLES ; ++ii )
    {
        cmp_particles->setQ0Value( ii, q0 );
        cmp_particles->setMass0Value( ii, mass0 );
        cmp_particles->setBeta0Value( ii, beta0 );
        cmp_particles->setGamma0Value( ii, gamma0 );
        cmp_particles->setP0cValue( ii, pc0 );
        cmp_particles->setXValue( ii, static_cast< double >( ii * 0.002 ) );

        cmp_particles->setRvvValue( ii, 1.0 );
        cmp_particles->setRppValue( ii, 1.0);
        cmp_particles->setChiValue( ii, 1.0 );
        cmp_particles->setChargeRatioValue( ii, 1.0 );

        cmp_particles->setParticleIdValue( ii, ii );
        cmp_particles->setStateValue( ii, 1 );
    }

    st::Particles* track_particles = track_pbuffer.get< st::Particles >( 0 );
    ::NS(Particles_copy)( track_particles, cmp_particles );

    ::NS(track_status_t) track_status = ::NS(Track_all_particles_until_turn)(
        cmp_particles, elements.getCApiPtr(), 2 );

    SIXTRL_ASSERT( track_status == st::TRACK_SUCCESS );


    /* ********************************************************************* */
    /* Start track_line test */

    st::CpuTrackJob job( track_pbuffer, elements );
    track_status = job.trackLine( 0, 1, false );
    ASSERT_TRUE( track_status == st::TRACK_SUCCESS );

    auto states_begin = track_particles->getState();
    auto states_end   = states_begin;
    std::advance( states_end, NUM_PARTICLES );

    job.collectParticles();

    /* Check all states are equal to 1 after the first drift */
    ASSERT_TRUE( std::find_if( states_begin, states_end,
        []( st::particle_index_t const& SIXTRL_RESTRICT x )
        { return x != 1; } ) == states_end );

    /* Track until the end of turn */
    track_status = job.trackLine( 1, elements.getNumObjects(), true );
    ASSERT_TRUE( track_status == st::TRACK_SUCCESS );
    job.collectParticles();

    /* Verify that we now have lost particles due to the aperture check */
    ASSERT_TRUE( std::find_if( states_begin, states_end,
        []( st::particle_index_t const& SIXTRL_RESTRICT x )
        { return x != 1; } ) != states_end );

    std::vector< st::particle_index_t > const saved_states_after_first_turn(
        states_begin, states_end );

    SIXTRL_ASSERT( std::equal( states_begin, states_end,
                        saved_states_after_first_turn.begin() ) );

    /* track over the first drift for the second turn */
    track_status = job.trackLine( 0, 1, false );
    ASSERT_TRUE( track_status == st::TRACK_SUCCESS );
    job.collectParticles();

    /* Since no apertures have been encountered and the global aperture
     * limit should be much much larger than the current x/y values,
     * the states should not have changed compared to the end of turn 1 */

    ASSERT_TRUE( std::equal( states_begin, states_end,
                    saved_states_after_first_turn.begin() ) );

    /* Finish second turn */
    track_status = job.trackLine( 1, elements.getNumObjects(), true );
    ASSERT_TRUE( track_status == st::TRACK_SUCCESS );
    job.collectParticles();

    /* Compare against the results obtained by performing
     * NS(Track_all_particles_until_turn) for the whole two turns
     * in one go */

    double const ABS_TOLERANCE = double{ 1e-14 };

    ASSERT_TRUE( ::NS(Particles_compare_real_values_with_treshold)(
        cmp_particles, track_particles, ABS_TOLERANCE ) == 0 );
}

/* end: tests/sixtracklib/common/track/test_track_job_track_line_cxx.cpp */
