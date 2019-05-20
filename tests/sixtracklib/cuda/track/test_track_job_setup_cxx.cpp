#include "sixtracklib/cuda/track_job.hpp"

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
#include "sixtracklib/common/particles.hpp"

TEST( CXX_CudaTrackJobSetupTests, CreateTrackJobNoOutput )
{
    namespace st_test   = SIXTRL_CXX_NAMESPACE::tests;
    namespace st        = SIXTRL_CXX_NAMESPACE;

    using track_job_t   = st::CudaTrackJob;
    using size_t        = st::CudaTrackJob::size_type;
    using buffer_t      = st::Buffer;
    using particles_t   = st::Particles;

    buffer_t eb( ::NS(PATH_TO_BEAMBEAM_BEAM_ELEMENTS) );
    buffer_t in_particle_buffer( ::NS(PATH_TO_BEAMBEAM_PARTICLES_DUMP) );

    buffer_t pb;
    buffer_t my_output_buffer;

    particles_t const* orig_particles =
        st::Particles::FromBuffer( in_particle_buffer, size_t{ 0u } );

    SIXTRL_ASSERT( orig_particles != nullptr );

    particles_t* particles = pb.createNew< particles_t >(
        orig_particles->getNumParticles() );

    SIXTRL_ASSERT( particles != nullptr );
    particles->copy( *orig_particles );

    /* ===================================================================== *
     * First set of tests:
     * No Beam Monitors
     * No Elem by Elem config
     * --------------------------------------------------------------------- */

    track_job_t job0;

    ASSERT_TRUE( job0.archId() == st::ARCHITECTURE_CUDA );
    ASSERT_TRUE( 0 == job0.archStr().compare( SIXTRL_ARCHITECTURE_CUDA_STR ) );

    bool success = job0.reset( pb, eb );
    ASSERT_TRUE( success );
    ASSERT_TRUE( st_test::TestTrackJob_setup_no_required_output(
        job0, pb, track_job_t::DefaultNumParticleSetIndices(),
        track_job_t::DefaultParticleSetIndicesBegin(), eb, nullptr ) );

    /* --------------------------------------------------------------------- */

    track_job_t  job1( "0.0", pb, eb, nullptr, size_t{ 0 } );
    ASSERT_TRUE( job1.archId() == st::ARCHITECTURE_CUDA );
    ASSERT_TRUE( job1.archStr().compare( SIXTRL_ARCHITECTURE_CUDA_STR ) == 0 );

    ASSERT_TRUE( st_test::TestTrackJob_setup_no_required_output(
        job1, pb, track_job_t::DefaultNumParticleSetIndices(),
        track_job_t::DefaultParticleSetIndicesBegin(), eb, nullptr ) );

    /* --------------------------------------------------------------------- */

    size_t const good_particle_sets[] = { size_t{ 0 } };

    track_job_t job2;

    ASSERT_TRUE( job2.archId() == st::ARCHITECTURE_CUDA );
    ASSERT_TRUE( job2.archStr().compare( SIXTRL_ARCHITECTURE_CUDA_STR ) == 0 );

    success = job2.reset( pb, good_particle_sets[ 0 ], eb, nullptr );

    ASSERT_TRUE( st_test::TestTrackJob_setup_no_required_output(
        job2, pb, size_t{ 1 }, &good_particle_sets[ 0 ], eb, nullptr ) );

    /* --------------------------------------------------------------------- */

    track_job_t job3;


    job3.reset( pb, track_job_t::DefaultParticleSetIndicesBegin(),
        track_job_t::DefaultParticleSetIndicesEnd(), eb, nullptr, size_t{ 0 } );

    ASSERT_TRUE( job3.archId() == st::ARCHITECTURE_CUDA );
    ASSERT_TRUE( job3.archStr().compare( SIXTRL_ARCHITECTURE_CUDA_STR ) == 0 );

    ASSERT_TRUE( st_test::TestTrackJob_setup_no_required_output(
        job3, pb, track_job_t::DefaultNumParticleSetIndices(),
            track_job_t::DefaultParticleSetIndicesBegin(), eb, nullptr ) );

    /* --------------------------------------------------------------------- */

    track_job_t job4( "0.0", pb, eb, &my_output_buffer );

    ASSERT_TRUE( job4.archId() == st::ARCHITECTURE_CUDA );
    ASSERT_TRUE( job4.archStr().compare( SIXTRL_ARCHITECTURE_CUDA_STR ) == 0 );

    ASSERT_TRUE( st_test::TestTrackJob_setup_no_required_output(
        job4, pb, track_job_t::DefaultNumParticleSetIndices(),
            track_job_t::DefaultParticleSetIndicesBegin(), eb,
                &my_output_buffer ) );
}

/* end: tests/sixtracklib/cuda/track/test_track_job_setup_cxx.cpp */
