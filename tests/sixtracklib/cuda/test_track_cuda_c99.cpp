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

#include <cuda_runtime.h>
#include <cuda.h>

#include "sixtracklib/testlib.h"

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/generated/path.h"
#include "sixtracklib/common/buffer.h"
#include "sixtracklib/common/particles.h"
#include "sixtracklib/common/beam_elements.h"

#include "sixtracklib/cuda/track_particles_kernel_c_wrapper.h"

TEST( C99_Cuda_TrackParticlesTests, LHCReproduceSixTrackSingleTurnNoBeamBeam )
{
    using size_t          = ::st_buffer_size_t;
    using object_t        = ::st_Object;
    using particles_t     = ::st_Particles;
    using index_t         = ::st_particle_index_t;
    using real_t          = ::st_particle_real_t;
    using num_particles_t = ::st_particle_num_elements_t;

    /* ===================================================================== */

    static real_t const ABS_TOLERANCE = real_t{ 1e-13 };

    ::st_Buffer* lhc_particles_buffer = ::st_Buffer_new_from_file(
        ::st_PATH_TO_TEST_LHC_PARTICLES_DATA_T1_P2_NO_BEAM_BEAM );

    ::st_Buffer* lhc_beam_elements_buffer = ::st_Buffer_new_from_file(
        ::st_PATH_TO_TEST_LHC_BEAM_ELEMENTS_DATA_NO_BEAM_BEAM );

    ::st_Buffer* particles_buffer      = ::st_Buffer_new( size_t{ 1u << 20u } );
    ::st_Buffer* diff_particles_buffer = ::st_Buffer_new( size_t{ 1u << 20u } );
    ::st_Buffer* beam_elements_buffer  = ::st_Buffer_new( size_t{ 1u << 20u } );

    /* --------------------------------------------------------------------- */

    ASSERT_TRUE( lhc_particles_buffer     != nullptr );
    ASSERT_TRUE( lhc_beam_elements_buffer != nullptr );

    ASSERT_TRUE( particles_buffer         != nullptr );
    ASSERT_TRUE( diff_particles_buffer    != nullptr );
    ASSERT_TRUE( beam_elements_buffer     != nullptr );

    /* --------------------------------------------------------------------- */

    index_t const lhc_num_sequences =
        ::st_Buffer_get_num_of_objects( lhc_particles_buffer );

    index_t const lhc_num_beam_elements =
        ::st_Buffer_get_num_of_objects( lhc_beam_elements_buffer );

    ASSERT_TRUE( lhc_num_sequences     > index_t{ 0 } );
    ASSERT_TRUE( lhc_num_beam_elements > index_t{ 0 } );

    int num_devices = 0;
    cudaError_t cu_err = cudaGetDeviceCount( &num_devices );
    ASSERT_TRUE( cu_err == cudaSuccess );

    if( num_devices > 0 )
    {
         int device_id = 0;

        for( ; device_id < num_devices ; ++device_id )
        {
            cudaDeviceProp properties;

            cu_err = cudaSetDevice( device_id );
            ASSERT_TRUE( cu_err == cudaSuccess );

            cu_err = cudaGetDeviceProperties( &properties, device_id );
            ASSERT_TRUE( cu_err == cudaSuccess );

            std::cout << "Device # " << std::setw( 3 )
                      << device_id   << "\r\n"
                      << "Name   : " << std::setw( 20 )
                      << properties.name << "\r\n" << std::endl;

            /* ------------------------------------------------------------- */

            object_t const* be_begin =
                ::st_Buffer_get_const_objects_begin( lhc_beam_elements_buffer );

            object_t const* pb_begin =
                ::st_Buffer_get_const_objects_begin( lhc_particles_buffer );

            object_t const* pb_end   =
                ::st_Buffer_get_const_objects_end( lhc_particles_buffer );

            object_t const* pb_it = pb_begin;

            particles_t const* in_particles =
                ::st_BufferIndex_get_const_particles( pb_it );

            object_t const* prev_pb = pb_it++;

            particles_t const* prev_in_particles = nullptr;

            num_particles_t num_particles =
                ::st_Particles_get_num_of_particles( in_particles );

            num_particles_t prev_num_particles = num_particles_t{ 0 };
            size_t cnt = size_t{ 0 };

            uint64_t const NUM_TURNS = uint64_t{ 1 };

            for( ; pb_it != pb_end ; ++pb_it, ++prev_pb, ++cnt )
            {
                prev_in_particles = in_particles;
                in_particles      = ::st_BufferIndex_get_const_particles( pb_it );

                prev_num_particles = num_particles;
                num_particles = ::st_Particles_get_num_of_particles( in_particles );

                ASSERT_TRUE( num_particles == prev_num_particles );
                ASSERT_TRUE( in_particles != nullptr );

                ::st_Buffer_clear( particles_buffer, true );
                particles_t* particles = ::st_Particles_add_copy(
                    particles_buffer, prev_in_particles );

                ASSERT_TRUE( ::st_Buffer_get_num_of_objects( particles_buffer ) == 1u );
                ASSERT_TRUE( ::st_Buffer_get_size( particles_buffer ) > size_t{ 0 } );

                index_t const begin_elem_id = ::st_Particles_get_at_element_id_value(
                        particles, num_particles_t{ 0 } );

                index_t const end_elem_id = ::st_Particles_get_at_element_id_value(
                    in_particles, num_particles_t{ 0 } );

                object_t const* line_begin = be_begin;
                object_t const* line_end   = be_begin;

                std::advance( line_begin, begin_elem_id + index_t{ 1 } );
                std::advance( line_end,   end_elem_id   + index_t{ 1 } );

                ::st_Buffer_reset( beam_elements_buffer );
                ::st_BeamElements_copy_to_buffer(
                    beam_elements_buffer, line_begin, line_end );

                int const track_success ::st_Track_particles_in_place_on_cuda(
                    particles_buffer, beam_elements, NUM_TURNS );

                ASSERT_TRUE( track_success == 0 );


    }
    else
    {
        std::cout << "Skipping unit-test because no "
                  << "CUDA platforms have been found --> "
                  << "NEITHER PASSED NOR FAILED!"
                  << std::endl;
    }

    ::st_Buffer_delete( lhc_particles_buffer );
    ::st_Buffer_deletE( lhc_beam_elements_buffer );

    ::st_Buffer_delete( particles_buffer );
    ::st_Buffer_delete( beam_elements_buffer );
    ::st_Buffer_delete( diff_particles_buffer );
}

/* end: tests/sixtracklib/cuda/test_track_cuda_c99.cpp */
