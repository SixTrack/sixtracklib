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

#include "sixtracklib/testlib.hpp"

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/particles.h"
#include "sixtracklib/common/particles/particles_addr.h"

TEST( C99_Cpu_CpuTrackJobFetchParticleAddrTests, BasicUsage )
{
    using track_job_t      = ::NS(CpuTrackJob);
    using size_t           = ::NS(buffer_size_t);
    using buffer_t         = ::NS(Buffer);
    using ctrl_status_t    = ::NS(arch_status_t);
    using particles_addr_t = ::NS(ParticlesAddr);
    using particles_t      = ::NS(Particles);

    buffer_t* eb = ::NS(Buffer_new_from_file)(
        ::NS(PATH_TO_BEAMBEAM_BEAM_ELEMENTS) );

    buffer_t* pb = ::NS(Buffer_new_from_file)(
        ::NS(PATH_TO_BEAMBEAM_PARTICLES_DUMP ) );

    size_t const num_particle_sets = ::NS(Buffer_get_num_of_objects)( pb );

    track_job_t* track_job = ::NS(CpuTrackJob_new)( pb, eb );
    ASSERT_TRUE( track_job != nullptr );

    ctrl_status_t status = ::NS(ARCH_STATUS_SUCCESS);

    if( !::NS(TrackJobNew_is_in_debug_mode)( track_job ) )
    {
        status = ::NS(TrackJobNew_enable_debug_mode)( track_job );
    }

    ASSERT_TRUE( status == ::NS(ARCH_STATUS_SUCCESS) );
    ASSERT_TRUE( ::NS(TrackJobNew_is_in_debug_mode)( track_job ) );

    ASSERT_TRUE( ::NS(TrackJobNew_can_fetch_particle_addresses)(
        track_job ) );

    if( !::NS(TrackJobNew_has_particle_addresses)( track_job ) )
    {
        status = ::NS(TrackJobNew_fetch_particle_addresses)( track_job );
    }

    ASSERT_TRUE( status == ::NS(ARCH_STATUS_SUCCESS) );
    ASSERT_TRUE( ::NS(TrackJobNew_has_particle_addresses)( track_job ) );
    ASSERT_TRUE( ::NS(TrackJobNew_get_ptr_particle_addresses_buffer)(
        track_job ) != nullptr );

    size_t const slot_size = ::NS(Buffer_get_slot_size)(
        ::NS(TrackJobNew_get_ptr_particle_addresses_buffer)( track_job ) );

    for( size_t ii = size_t{ 0 } ; ii < num_particle_sets ; ++ii )
    {
        particles_t const* cmp_particles =
            ::NS(Particles_buffer_get_const_particles)( pb, ii );

        ASSERT_TRUE( cmp_particles != nullptr );

        particles_addr_t const* particles_addr =
            ::NS(TrackJobNew_get_particle_addresses)( track_job, ii );

        ASSERT_TRUE( particles_addr != nullptr );
        ASSERT_TRUE( ::NS(Particles_get_num_of_particles)( cmp_particles )
                     == particles_addr->num_particles );

        bool const are_consistent =
            ::NS(TestParticlesAddr_are_addresses_consistent_with_particle)(
                particles_addr, cmp_particles, slot_size );

        ASSERT_TRUE( are_consistent );
    }

    status = ::NS(TrackJobNew_clear_all_particle_addresses)( track_job );
    ASSERT_TRUE( status == ::NS(ARCH_STATUS_SUCCESS) );
    ASSERT_TRUE( ::NS(TrackJobNew_can_fetch_particle_addresses)(
        track_job ) );

    ASSERT_TRUE( !::NS(TrackJobNew_has_particle_addresses)( track_job ) );

    status = ::NS(TrackJobNew_disable_debug_mode)( track_job );

    ASSERT_TRUE( status == ::NS(ARCH_STATUS_SUCCESS) );
    ASSERT_TRUE( !::NS(TrackJobNew_is_in_debug_mode)( track_job ) );

    status = ::NS(TrackJobNew_fetch_particle_addresses)( track_job );

    ASSERT_TRUE( status == ::NS(ARCH_STATUS_SUCCESS) );
    ASSERT_TRUE( ::NS(TrackJobNew_has_particle_addresses)( track_job ) );

    for( size_t ii = size_t{ 0 } ; ii < num_particle_sets ; ++ii )
    {
        particles_t const* cmp_particles =
            ::NS(Particles_buffer_get_const_particles)( pb, ii );

        ASSERT_TRUE( cmp_particles != nullptr );

        particles_addr_t const* particles_addr =
            ::NS(TrackJobNew_get_particle_addresses)( track_job, ii );

        ASSERT_TRUE( particles_addr != nullptr );
        ASSERT_TRUE( ::NS(Particles_get_num_of_particles)( cmp_particles )
                     == particles_addr->num_particles );

        bool const are_consistent =
            ::NS(TestParticlesAddr_are_addresses_consistent_with_particle)(
                particles_addr, cmp_particles, slot_size );

        ASSERT_TRUE( are_consistent );
    }

    ::NS(TrackJobNew_delete)( track_job );
    track_job = nullptr;

    ::NS(Buffer_delete)( pb );
    ::NS(Buffer_delete)( eb );
}

/* end: tests/sixtracklib/common/track/test_track_job_fetch_particle_addr_c99.cpp */
