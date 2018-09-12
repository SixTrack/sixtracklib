#define _USE_MATH_DEFINES

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <fstream>
#include <iterator>
#include <sstream>
#include <vector>

#include <gtest/gtest.h>

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#include "sixtracklib/testlib.h"

#include "sixtracklib/_impl/definitions.h"
#include "sixtracklib/_impl/path.h"
#include "sixtracklib/common/buffer.h"
#include "sixtracklib/common/beam_elements.h"

/* ************************************************************************* */

TEST( C99_OpenCL_TrackParticlesTests, LHCReproduceSixTrackSingleTurnNoBeamBeam )
{
    using size_t          = ::st_buffer_size_t;
    using buffer_t        = ::st_Buffer;
    using object_t        = ::st_Object;
    using index_t         = ::st_particle_index_t;
    using real_t          = ::st_particle_real_t;
    using num_particles_t = ::st_particle_num_elements_t;

    /* ===================================================================== */

    ::st_Buffer* lhc_particles_buffer = ::st_Buffer_new_from_file(
        ::st_PATH_TO_TEST_LHC_PARTICLES_DATA_T1_P2_NO_BEAM_BEAM );

    ::st_Buffer* lhc_beam_elements_buffer = ::st_Buffer_new_from_file(
        ::st_PATH_TO_TEST_LHC_BEAM_ELEMENTS_DATA_NO_BEAM_BEAM );

    ::st_Buffer* cmp_particles_buffer = ::st_Buffer_new( size_t{ 1u << 20u } );
    ::st_Buffer* particles_buffer     = ::st_Buffer_new( size_t{ 1u << 20u } );
    ::st_Buffer* beam_elements_buffer = ::st_Buffer_new( size_t{ 1u << 20u } );

    /* --------------------------------------------------------------------- */

    ASSERT_TRUE( lhc_particles_buffer     != nullptr );
    ASSERT_TRUE( lhc_beam_elements_buffer != nullptr );

    ASSERT_TRUE( cmp_particles_buffer     != nullptr );
    ASSERT_TRUE( particles_buffer         != nullptr );
    ASSERT_TRUE( beam_elements_buffer     != nullptr );

    /* --------------------------------------------------------------------- */

    index_t const lhc_num_sequences =
        ::st_Buffer_get_num_of_objects( lhc_particles_buffer );

    index_t const lhc_num_beam_elements =
        ::st_Buffer_get_num_of_objects( lhc_beam_elements_buffer );

    ASSERT_TRUE( lhc_num_sequences     > index_t{ 0 } );
    ASSERT_TRUE( lhc_num_beam_elements > index_t{ 0 } );

    /* ===================================================================== */

    std::vector< cl::Platform > platforms;
    cl::Platform::get( &platforms );

    std::vector< cl::Device > devices;

    for( auto const& p : platforms )
    {
        std::vector< cl::Device > temp_devices;

        p.getDevices( CL_DEVICE_TYPE_ALL, &temp_devices );

        for( auto const& d : temp_devices )
        {
            if( !d.getInfo< CL_DEVICE_AVAILABLE >() ) continue;
            devices.push_back( d );
        }
    }

    if( !devices.empty() )
    {






    }

    /* ===================================================================== */

    ::st_Buffer_delete( lhc_particles_buffer );
    ::st_Buffer_delete( lhc_beam_elements_buffer );

    ::st_Buffer_delete( cmp_particles_buffer );
    ::st_Buffer_delete( particles_buffer );
    ::st_Buffer_delete( beam_elements_buffer );
}


/* ************************************************************************* */

/* end: tests/sixtracklib/opencl/test_track_opencl_c99.cpp */
