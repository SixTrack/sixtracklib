#include "sixtracklib/common/track/track_job_cpu.h"

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
#include "sixtracklib/common/be_drift/be_drift.h"
#include "sixtracklib/common/be_multipole/be_multipole.h"
#include "sixtracklib/common/be_limit/be_limit_rect.h"
#include "sixtracklib/common/control/definitions.h"
#include "sixtracklib/common/track/definitions.h"
#include "sixtracklib/common/buffer.h"
#include "sixtracklib/common/particles.h"
#include "sixtracklib/common/be_monitor/be_monitor.h"
#include "sixtracklib/common/track/track.h"

TEST( C99CpuCpuTrackJobTrackLineTests, CmpWithTrackUntilTest )
{
    namespace st = SIXTRL_CXX_NAMESPACE;

    using particles_t    = ::NS(Particles);
    using track_job_t    = ::NS(CpuTrackJob);
    using buffer_t       = ::NS(Buffer);
    using buf_size_t     = ::NS(buffer_size_t);
    using track_status_t = ::NS(track_status_t);
    using ctrl_status_t  = ::NS(arch_status_t);
    using real_t         = ::NS(particle_real_t);
    using pindex_t       = ::NS(particle_index_t);

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

    /* -------------------------------------------------------------------- */
    /* Create a dedicated buffer for tracking & create a track job instance */

    buffer_t* track_pb = ::NS(Buffer_new)( buf_size_t{ 0 } );
    particles_t* particles = ::NS(Particles_add_copy)(
        track_pb, ::NS(Particles_buffer_get_const_particles(
            in_particles, buf_size_t{ 0 } ) ) );

    SIXTRL_ASSERT( particles != nullptr );

    track_job_t* track_job = ::NS(CpuTrackJob_new)( track_pb, beam_elem_buffer );
    ASSERT_TRUE( track_job != nullptr );

    if( !::NS(TrackJobNew_is_in_debug_mode)( track_job ) )
    {
        ::NS(TrackJobNew_enable_debug_mode)( track_job );
    }

    ASSERT_TRUE( ::NS(TrackJobNew_is_in_debug_mode)( track_job ) );
    ASSERT_TRUE( !::NS(TrackJobNew_requires_collecting)( track_job ) );

    /* ************************************************************* */

    buf_size_t const num_beam_elements =
        ::NS(Buffer_get_num_of_objects)( beam_elem_buffer );

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

    SIXTRL_ASSERT( status == ::NS(ARCH_STATUS_SUCCESS) );
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

            track_status = ::NS(TrackJobNew_track_line)( track_job,
                be_begin_idx, be_end_idx, finish_turn );

            ASSERT_TRUE( track_status == ::NS(TRACK_SUCCESS) );
        }
    }

    /* ****************************************************************** */

    ASSERT_TRUE( ::NS(TrackJobNew_get_particles_buffer)( track_job ) ==
                 track_pb );

    particles = ::NS(Particles_buffer_get_particles)(
        track_pb, buf_size_t{ 0 } );

    SIXTRL_ASSERT( particles != nullptr );

    ASSERT_TRUE( ( particles != nullptr ) && ( cmp_particles != nullptr ) &&
        ( ( 0 == ::NS(Particles_compare_values)(
                cmp_particles, particles ) ) ||
          ( ( ABS_TOLERANCE > real_t{ 0 } ) &&
            ( 0 == ::NS(Particles_compare_values_with_treshold)(
                cmp_particles, particles, ABS_TOLERANCE ) ) ) ) );

    status = ::NS(Particles_copy)( particles,
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

    /* ***************************************************************** */
    /* Perform tracking again, this time not in debug mode */

    ::NS(Particles_init_min_max_attributes_for_find)(
        &min_particle_id, &max_particle_id, &min_at_element_id,
        &max_at_element_id, &min_at_turn_id, &max_at_turn_id );

    status = ::NS(Particles_find_min_max_attributes)(
        particles, &min_particle_id, &max_particle_id,
            &min_at_element_id, &max_at_element_id, &min_at_turn_id,
                &max_at_turn_id );

    SIXTRL_ASSERT( status == ::NS(ARCH_STATUS_SUCCESS) );
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

            track_status = ::NS(TrackJobNew_track_line)( track_job,
                be_begin_idx, be_end_idx, finish_turn );

            ASSERT_TRUE( track_status == ::NS(TRACK_SUCCESS) );
        }
    }

    /* Compare the results again against the cpu tracking result */

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


TEST( C99CpuCpuTrackJobTrackLineTests, LostParticleBehaviour )
{
    using npart_t = ::NS(particle_num_elements_t);

    ::NS(Buffer)* elements = ::NS(Buffer_new)( ::NS(buffer_size_t){ 0 } );

    ::NS(DriftExact)* drift0 = ::NS(DriftExact_new)( elements );
    ::NS(MultiPole)*  quad   = ::NS(MultiPole_new)( elements, 2 );
    ::NS(LimitRect)*  limit  = ::NS(LimitRect_new)( elements );
    ::NS(DriftExact)* drift1 = ::NS(DriftExact_new)( elements );

    drift0 = ::NS(DriftExact_from_buffer)( elements, 0 );
    quad   = ::NS(BeamElements_buffer_get_multipole)( elements, 1 );
    limit  = ::NS(BeamElements_buffer_get_limit_rect)( elements, 2 );
    drift1 = ::NS(DriftExact_from_buffer)( elements, 3 );

    SIXTRL_ASSERT( drift0 != nullptr );
    SIXTRL_ASSERT( quad   != nullptr );
    SIXTRL_ASSERT( limit  != nullptr );
    SIXTRL_ASSERT( drift1 != nullptr );

    ::NS(DriftExact_set_length)( drift0, double{ 5 } );

    ::NS(MultiPole_set_knl_value)( quad, 0, double{ 0 } );
    ::NS(MultiPole_set_knl_value)( quad, 1, double{ 1e-2 } );
    ::NS(MultiPole_set_length)( quad, double{ 1 } );

    ::NS(LimitRect_set_min_x)( limit, double{ -0.1 } );
    ::NS(LimitRect_set_max_x)( limit, double{  0.1 } );
    ::NS(LimitRect_set_min_y)( limit, double{ -0.1 } );
    ::NS(LimitRect_set_max_y)( limit, double{  0.1 } );

    ::NS(DriftExact_set_length)( drift1, double{ 5 } );

    ::NS(Buffer)* cmp_pbuffer = ::NS(Buffer_new)( ::NS(buffer_size_t){ 0 } );
    ::NS(Buffer)* track_pbuffer = ::NS(Buffer_new)( ::NS(buffer_size_t){ 0 } );

    constexpr npart_t NUM_PARTICLES = npart_t{ 100 };

    ::NS(Particles)* cmp_particles =
        ::NS(Particles_new)( cmp_pbuffer, NUM_PARTICLES );

    ::NS(Particles)* track_particles =
        ::NS(Particles_new)( track_pbuffer, NUM_PARTICLES );

    double const pc0   = double{ 10e9 };
    double const mass0 = double{ 1e9 };
    double const q0    = double{ 1 };

    double const energy0 = std::sqrt( mass0 * mass0 + pc0 * pc0 );
    double const gamma0  = energy0 / mass0;
    double const beta0   = std::sqrt( double{ 1 } -
        double{ 1 }  / ( gamma0 * gamma0 ) );

    for( npart_t ii = npart_t{ 0 } ; ii < NUM_PARTICLES ; ++ii )
    {
        ::NS(Particles_set_q0_value)( cmp_particles, ii, q0 );
        ::NS(Particles_set_mass0_value)( cmp_particles, ii, mass0 );
        ::NS(Particles_set_beta0_value)( cmp_particles, ii, beta0 );
        ::NS(Particles_set_gamma0_value)( cmp_particles, ii, gamma0 );
        ::NS(Particles_set_p0c_value)( cmp_particles, ii, pc0 );
        ::NS(Particles_set_x_value)(
            cmp_particles, ii, static_cast< double >( ii * 0.002 ) );

        ::NS(Particles_set_rvv_value)( cmp_particles, ii, 1.0 );
        ::NS(Particles_set_rpp_value)( cmp_particles, ii, 1.0);
        ::NS(Particles_set_chi_value)( cmp_particles, ii, 1.0 );
        ::NS(Particles_set_charge_ratio_value)( cmp_particles, ii, 1.0 );

        ::NS(Particles_set_particle_id_value)( cmp_particles, ii, ii );
        ::NS(Particles_set_state_value)( cmp_particles, ii, 1 );
    }

    ::NS(Particles_copy)( track_particles, cmp_particles );

    ::NS(track_status_t) track_status = ::NS(Track_all_particles_until_turn)(
        cmp_particles, elements, 2 );

    SIXTRL_ASSERT( track_status == ::NS(TRACK_SUCCESS) );


    /* ********************************************************************* */
    /* Start track_line test */

    ::NS(CpuTrackJob)* job = ::NS(CpuTrackJob_new)( track_pbuffer, elements );
    track_status = ::NS(TrackJobNew_track_line)( job, 0, 1, false );
    ASSERT_TRUE( track_status == ::NS(TRACK_SUCCESS) );

    auto states_begin = ::NS(Particles_get_const_state)( track_particles );
    auto states_end   = states_begin;
    std::advance( states_end, NUM_PARTICLES );

    ::NS(TrackJobNew_collect_particles)( job );

    /* Check all states are equal to 1 after the first drift */
    ASSERT_TRUE( std::find_if( states_begin, states_end,
        []( ::NS(particle_index_t) x ){ return x != 1; } ) == states_end );

    ::NS(buffer_size_t) const num_beam_elements =
        ::NS(Buffer_get_num_of_objects)( elements );

    /* Track until the end of turn */
    track_status = ::NS(TrackJobNew_track_line)(
        job, 1u, num_beam_elements, true );

    ASSERT_TRUE( track_status == ::NS(TRACK_SUCCESS) );
    ::NS(TrackJobNew_collect_particles)( job );

    /* Verify that we now have lost particles due to the aperture check */
    ASSERT_TRUE( std::find_if( states_begin, states_end,
        []( ::NS(particle_index_t) x ){ return x != 1; } ) != states_end );

    std::vector< ::NS(particle_index_t) > const saved_states_after_first_turn(
        states_begin, states_end );

    SIXTRL_ASSERT( std::equal( states_begin, states_end,
                        saved_states_after_first_turn.begin() ) );

    /* track over the first drift for the second turn */
    track_status = ::NS(TrackJobNew_track_line)( job, 0u, 1u, false );
    ASSERT_TRUE( track_status == ::NS(TRACK_SUCCESS) );
    ::NS(TrackJobNew_collect_particles)( job );

    /* Since no apertures have been encountered and the global aperture
     * limit should be much much larger than the current x/y values,
     * the states should not have changed compared to the end of turn 1 */

    ASSERT_TRUE( std::equal( states_begin, states_end,
                    saved_states_after_first_turn.begin() ) );

    /* Finish second turn */
    track_status = ::NS(TrackJobNew_track_line)(
        job, 1u, num_beam_elements, true );

    ASSERT_TRUE( track_status == ::NS(TRACK_SUCCESS) );
    ::NS(TrackJobNew_collect_particles)( job );

    /* Compare against the results obtained by performing
     * NS(Track_all_particles_until_turn) for the whole two turns
     * in one go */

    double const ABS_TOLERANCE = double{ 1e-14 };

    ASSERT_TRUE( ::NS(Particles_compare_real_values_with_treshold)(
        cmp_particles, track_particles, ABS_TOLERANCE ) == 0 );
}

/* end: tests/sixtracklib/common/track/test_track_job_track_until_c99.cpp */
