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
#include "sixtracklib/common/track.h"
#include "sixtracklib/common/generated/config.h"
#include "sixtracklib/testlib.h"

#if defined( SIXTRL_ENABLE_APERATURE_CHECK ) && \
    SIXTRL_ENABLE_APERATURE_CHECK == 1

TEST( C99_CommonParticlesAperatureTests,
      TrackParticlesOverDriftEnabledAperatureCheck )
{

#else

TEST( C99_CommonParticlesAperatureTests,
      TrackParticlesOverDriftDisabledAperatureCheck )
{

#endif /* SIXTRL_ENABLE_APERATURE_CHECK */
    ::st_buffer_size_t const NUM_PARTICLES = ( ::st_buffer_size_t )100u;
    ::st_buffer_size_t const NUM_DRIFTS    = ( ::st_buffer_size_t )10000u;

    ::st_Buffer* pb = ::st_Buffer_new( ( ::st_buffer_size_t )( 1u << 20u ) );
    ASSERT_TRUE( pb != nullptr );

    ::st_Particles* particles = ::st_Particles_new( pb, NUM_PARTICLES );
    ASSERT_TRUE( particles != nullptr );

    ::st_Particles_realistic_init( particles );
    ASSERT_TRUE( ::st_Particles_get_num_of_particles( particles ) == NUM_PARTICLES );

    for( ::st_buffer_size_t ii = ::st_buffer_size_t{ 0 } ; ii < NUM_PARTICLES ; ++ii )
    {
        ASSERT_TRUE( ::st_Particles_is_not_lost_value( particles, ii ) );
    }

    ::st_Buffer* eb = ::st_Buffer_new( ( ::st_buffer_size_t )( 1u << 20u ) );

    for( ::st_buffer_size_t ii = ::st_buffer_size_t{ 0 } ; ii < NUM_DRIFTS ;++ii )
    {
        ::st_Drift* drift = ::st_Drift_add( eb, ( double )10.0 );
        ASSERT_TRUE( drift != nullptr );
    }

    ASSERT_TRUE( ::st_Buffer_get_num_of_objects( eb ) == NUM_DRIFTS );

    for( ::st_buffer_size_t ii = ::st_buffer_size_t{ 0 } ; ii < NUM_DRIFTS ; ++ii )
    {
        int ret = ::st_Track_all_particles_beam_element_obj(
            particles, ::st_Buffer_get_const_object( eb, ii ) );

        ASSERT_TRUE( ret == 0 );

        for( ::st_buffer_size_t jj = ::st_buffer_size_t{ 0 } ; jj < NUM_PARTICLES ; ++jj )
        {
            #if defined( SIXTRL_ENABLE_APERATURE_CHECK ) && \
                SIXTRL_ENABLE_APERATURE_CHECK == 1

            double const x = ::st_Particles_get_x_value( particles, jj );
            double const y = ::st_Particles_get_y_value( particles, jj );

            if( ( fabs( x ) > SIXTRL_APERATURE_X_LIMIT ) ||
                ( fabs( y ) > SIXTRL_APERATURE_Y_LIMIT ) )
            {
                ASSERT_TRUE( ::st_Particles_is_lost_value( particles, jj ) );
            }
            else
            {
                ASSERT_TRUE( ::st_Particles_is_not_lost_value( particles, jj ) );
            }

            #else

            ASSERT_TRUE( ::st_Particles_is_not_lost_value( particles, jj ) );

            #endif /* SIXTRL_ENABLE_APERATURE_CHECK */
        }
    }

    ::st_Buffer_delete( eb );
    ::st_Buffer_delete( pb );

    pb = nullptr;
    eb = nullptr;
}

#if defined( SIXTRL_ENABLE_APERATURE_CHECK ) && \
    SIXTRL_ENABLE_APERATURE_CHECK == 1

TEST( C99_CommonParticlesAperatureTests,
      TrackParticlesOverDriftExactEnabledAperatureCheck )
{

#else

TEST( C99_CommonParticlesAperatureTests,
      TrackParticlesOverDriftExactDisabledAperatureCheck )
{

#endif /* SIXTRL_ENABLE_APERATURE_CHECK */
    ::st_buffer_size_t const NUM_PARTICLES = ( ::st_buffer_size_t )100u;
    ::st_buffer_size_t const NUM_DRIFTS    = ( ::st_buffer_size_t )10000u;

    ::st_Buffer* pb = ::st_Buffer_new( ( ::st_buffer_size_t )( 1u << 20u ) );
    ASSERT_TRUE( pb != nullptr );

    ::st_Particles* particles = ::st_Particles_new( pb, NUM_PARTICLES );
    ASSERT_TRUE( particles != nullptr );

    ::st_Particles_realistic_init( particles );
    ASSERT_TRUE( ::st_Particles_get_num_of_particles( particles ) == NUM_PARTICLES );

    for( ::st_buffer_size_t ii = ::st_buffer_size_t{ 0 } ; ii < NUM_PARTICLES ; ++ii )
    {
        ASSERT_TRUE( ::st_Particles_is_not_lost_value( particles, ii ) );
    }

    ::st_Buffer* eb = ::st_Buffer_new( ( ::st_buffer_size_t )( 1u << 20u ) );

    for( ::st_buffer_size_t ii = ::st_buffer_size_t{ 0 } ; ii < NUM_DRIFTS ;++ii )
    {
        ::st_DriftExact* drift = ::st_DriftExact_add( eb, ( double )10.0 );
        ASSERT_TRUE( drift != nullptr );
    }

    ASSERT_TRUE( ::st_Buffer_get_num_of_objects( eb ) == NUM_DRIFTS );

    for( ::st_buffer_size_t ii = ::st_buffer_size_t{ 0 } ; ii < NUM_DRIFTS ; ++ii )
    {
        int ret = ::st_Track_all_particles_beam_element_obj(
            particles, ::st_Buffer_get_const_object( eb, ii ) );

        ASSERT_TRUE( ret == 0 );

        for( ::st_buffer_size_t jj = ::st_buffer_size_t{ 0 } ; jj < NUM_PARTICLES ; ++jj )
        {
            #if defined( SIXTRL_ENABLE_APERATURE_CHECK ) && \
                SIXTRL_ENABLE_APERATURE_CHECK == 1

            double const x = ::st_Particles_get_x_value( particles, jj );
            double const y = ::st_Particles_get_y_value( particles, jj );

            if( ( fabs( x ) > SIXTRL_APERATURE_X_LIMIT ) ||
                ( fabs( y ) > SIXTRL_APERATURE_Y_LIMIT ) )
            {
                ASSERT_TRUE( ::st_Particles_is_lost_value( particles, jj ) );
            }
            else
            {
                ASSERT_TRUE( ::st_Particles_is_not_lost_value( particles, jj ) );
            }

            #else

            ASSERT_TRUE( ::st_Particles_is_not_lost_value( particles, jj ) );

            #endif /* SIXTRL_ENABLE_APERATURE_CHECK */
        }
    }

    ::st_Buffer_delete( eb );
    ::st_Buffer_delete( pb );

    pb = nullptr;
    eb = nullptr;
}

/* end: tests/sixtracklib/common/test_particles_aperature_c99.cpp */
