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
#include "sixtracklib/common/beam_elements.h"
#include "sixtracklib/common/track/track.h"
#include "sixtracklib/common/generated/config.h"
#include "sixtracklib/testlib.h"

#if defined( SIXTRL_ENABLE_APERTURE_CHECK ) && \
    SIXTRL_ENABLE_APERTURE_CHECK == 1

TEST( C99_CommonParticlesApertureTests,
      TrackParticlesOverDriftEnabledApertureCheck )
{

#else

TEST( C99_CommonParticlesApertureTests,
      TrackParticlesOverDriftDisabledApertureCheck )
{

#endif /* SIXTRL_ENABLE_APERTURE_CHECK */

    using nelem_t = ::NS(particle_num_elements_t);
    using size_t  = size_t;

    size_t const NUM_PARTICLES = size_t{100u };
    size_t const NUM_DRIFTS = size_t{ 10000u };

    ::NS(Buffer)* pb = ::NS(Buffer_new)( size_t{ 0u } );
    ASSERT_TRUE( pb != nullptr );

    ::NS(Particles)* particles = ::NS(Particles_new)( pb, NUM_PARTICLES );
    ASSERT_TRUE( particles != nullptr );

    nelem_t const npart = ::NS(Particles_get_num_of_particles)( particles );

    ::NS(Particles_realistic_init)( particles );
    ASSERT_TRUE( ::NS(Particles_get_num_of_particles)( particles ) ==
                 NUM_PARTICLES );

    for( size_t ii = size_t{ 0 } ; ii < NUM_PARTICLES ; ++ii )
    {
        ASSERT_TRUE( ::NS(Particles_is_not_lost_value)( particles, ii ) );
    }

    ::NS(Buffer)* eb = ::NS(Buffer_new)( size_t{ 0u } );

    for( size_t ii = size_t{ 0 } ; ii < NUM_DRIFTS ;++ii )
    {
        ::NS(Drift)* drift = ::NS(Drift_add)( eb, ( double )10.0 );
        ASSERT_TRUE( drift != nullptr );
    }

    ASSERT_TRUE( ::NS(Buffer_get_num_of_objects)( eb ) == NUM_DRIFTS );

    for( size_t ii = size_t{ 0 } ; ii < NUM_DRIFTS ; ++ii )
    {
        for( nelem_t idx = nelem_t{ 0 } ; idx < npart ; ++idx )
        {
            ::NS(track_status_t) const status =
                ::NS(Track_particle_beam_element_obj)(
                    particles, idx, ::NS(Buffer_get_const_object)( eb, ii ) );

            ASSERT_TRUE( status == ::NS(TRACK_SUCCESS) );

            #if defined( SIXTRL_ENABLE_APERTURE_CHECK ) && \
                         SIXTRL_ENABLE_APERTURE_CHECK == 1

            double const x = ::NS(Particles_get_x_value)( particles, idx );
            double const y = ::NS(Particles_get_y_value)( particles, idx );

            if( ( fabs( x ) > SIXTRL_APERTURE_X_LIMIT ) ||
                ( fabs( y ) > SIXTRL_APERTURE_Y_LIMIT ) )
            {
                ASSERT_TRUE( ::NS(Particles_is_lost_value)( particles, idx ) );
            }
            else
            {
                ASSERT_TRUE( ::NS(Particles_is_not_lost_value)( particles, idx ) );
            }

            #else

            ASSERT_TRUE( ::NS(Particles_is_not_lost_value)( particles, idx ) );

            #endif /* SIXTRL_ENABLE_APERTURE_CHECK */
        }
    }

    ::NS(Buffer_delete)( eb );
    ::NS(Buffer_delete)( pb );

    pb = nullptr;
    eb = nullptr;
}

#if defined( SIXTRL_ENABLE_APERTURE_CHECK ) && \
             SIXTRL_ENABLE_APERTURE_CHECK == 1

TEST( C99_CommonParticlesApertureTests,
      TrackParticlesOverDriftExactEnabledApertureCheck )
{

#else

TEST( C99_CommonParticlesApertureTests,
      TrackParticlesOverDriftExactDisabledApertureCheck )
{
#endif /* SIXTRL_ENABLE_APERTURE_CHECK */
    using nelem_t = ::NS(particle_num_elements_t);
    using size_t  = size_t;

    size_t const NUM_PARTICLES = size_t{ 100u };
    size_t const NUM_DRIFTS    = size_t{ 10000u };

    ::NS(Buffer)* pb = ::NS(Buffer_new)( size_t{ 0u } );
    ASSERT_TRUE( pb != nullptr );

    ::NS(Particles)* particles = ::NS(Particles_new)( pb, NUM_PARTICLES );
    ASSERT_TRUE( particles != nullptr );

    nelem_t const npart = ::NS(Particles_get_num_of_particles)( particles );

    ::NS(Particles_realistic_init)( particles );
    ASSERT_TRUE( ::NS(Particles_get_num_of_particles)( particles ) == NUM_PARTICLES );

    for( size_t ii = size_t{ 0 } ; ii < NUM_PARTICLES ; ++ii )
    {
        ASSERT_TRUE( ::NS(Particles_is_not_lost_value)( particles, ii ) );
    }

    ::NS(Buffer)* eb = ::NS(Buffer_new)( size_t{ 0 } );

    for( size_t ii = size_t{ 0 } ; ii < NUM_DRIFTS ;++ii )
    {
        ::NS(DriftExact)* drift = ::NS(DriftExact_add)( eb, ( double )10.0 );
        ASSERT_TRUE( drift != nullptr );
    }

    ASSERT_TRUE( ::NS(Buffer_get_num_of_objects)( eb ) == NUM_DRIFTS );

    for( size_t ii = size_t{ 0 } ; ii < NUM_DRIFTS ; ++ii )
    {
        for( nelem_t idx = nelem_t{ 0 } ; idx < npart ; ++idx )
        {
            ::NS(track_status_t) const status =
                NS(Track_particle_beam_element_obj)(
                    particles, idx, ::NS(Buffer_get_const_object)( eb, ii ) );

            ASSERT_TRUE( status == ::NS(TRACK_SUCCESS) );

            #if defined( SIXTRL_ENABLE_APERTURE_CHECK ) && \
                SIXTRL_ENABLE_APERTURE_CHECK == 1

            double const x = ::NS(Particles_get_x_value)( particles, idx );
            double const y = ::NS(Particles_get_y_value)( particles, idx );

            if( ( fabs( x ) > SIXTRL_APERTURE_X_LIMIT ) ||
                ( fabs( y ) > SIXTRL_APERTURE_Y_LIMIT ) )
            {
                ASSERT_TRUE( ::NS(Particles_is_lost_value)( particles, idx ) );
            }
            else
            {
                ASSERT_TRUE( ::NS(Particles_is_not_lost_value)( particles, idx ) );
            }

            #else

            ASSERT_TRUE( ::NS(Particles_is_not_lost_value)( particles, idx ) );

            #endif /* SIXTRL_ENABLE_APERTURE_CHECK */
        }
    }

    ::NS(Buffer_delete)( eb );
    ::NS(Buffer_delete)( pb );

    pb = nullptr;
    eb = nullptr;
}

/* end: tests/sixtracklib/common/test_particles_aperture_c99.cpp */
