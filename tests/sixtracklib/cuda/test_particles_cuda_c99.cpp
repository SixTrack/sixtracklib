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

#include "sixtracklib/_impl/definitions.h"
#include "sixtracklib/_impl/path.h"
#include "sixtracklib/common/buffer.h"

#include "sixtracklib/cuda/impl/cuda_particles_kernel_c_wrapper.h"

TEST( C99_Cuda_ParticlesTests, CopyParticlesHostToDeviceThenBackCompare )
{
    using buffer_t    = ::st_Buffer;
    using particles_t = ::st_Particles;
    using size_t      = ::st_buffer_size_t;

    std::string const path_to_data( ::st_PATH_TO_TEST_TRACKING_BE_DRIFT_DATA );

    /* --------------------------------------------------------------------- */

    buffer_t* orig_particles_buffer =
        ::st_TrackTestdata_extract_result_particles_buffer(
            path_to_data.c_str() );

    size_t const slot_size = ::st_Buffer_get_slot_size( orig_particles_buffer );
    ASSERT_TRUE( slot_size == size_t{ 8 } );


    ASSERT_TRUE( orig_particles_buffer != nullptr );

    ASSERT_TRUE( ::st_Buffer_get_num_of_objects( orig_particles_buffer ) >
                 size_t{ 0 } );

    size_t const particle_buffer_size =
        ::st_Buffer_get_size( orig_particles_buffer );

    ASSERT_TRUE( particle_buffer_size > size_t{ 0 } );

    buffer_t* copy_particles_buffer = ::st_Buffer_new( particle_buffer_size );


    ASSERT_TRUE( copy_particles_buffer != nullptr );

    int success = ::st_Buffer_reserve( copy_particles_buffer,
        ::st_Buffer_get_num_of_objects( orig_particles_buffer ),
        ::st_Buffer_get_num_of_slots( orig_particles_buffer ),
        ::st_Buffer_get_num_of_dataptrs( orig_particles_buffer ),
        ::st_Buffer_get_num_of_garbage_ranges( orig_particles_buffer ) );

    ASSERT_TRUE( success == 0 );

    /* --------------------------------------------------------------------- */

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

            ASSERT_TRUE( ::st_Buffer_reset( copy_particles_buffer ) == 0 );

            success = st_Run_test_particles_copy_buffer_kernel_on_cuda_grid(
                orig_particles_buffer, copy_particles_buffer, 1, 1 );

            ASSERT_TRUE( success == 0 );

            ASSERT_TRUE( !::st_Buffer_needs_remapping( copy_particles_buffer ) );
            ASSERT_TRUE( !::st_Buffer_needs_remapping( orig_particles_buffer ) );

            ASSERT_TRUE( ::st_Buffer_get_num_of_objects( copy_particles_buffer ) ==
                         ::st_Buffer_get_num_of_objects( orig_particles_buffer ) );

            ASSERT_TRUE( ::st_Particles_buffers_have_same_structure(
                copy_particles_buffer, orig_particles_buffer ) );

            ASSERT_TRUE( ::st_Particles_buffer_get_total_num_of_particles(
                            copy_particles_buffer ) ==
                        ::st_Particles_buffer_get_total_num_of_particles(
                            orig_particles_buffer ) );

            ASSERT_TRUE( ::st_Particles_buffers_compare_values(
                orig_particles_buffer, copy_particles_buffer ) == 0 );

            /* ------------------------------------------------------------- */

            ASSERT_TRUE( ::st_Buffer_reset( copy_particles_buffer ) == 0 );

            success = st_Run_test_particles_copy_buffer_kernel_on_cuda_grid(
                orig_particles_buffer, copy_particles_buffer, 64, 1 );

            ASSERT_TRUE( success == 0 );

            ASSERT_TRUE( !::st_Buffer_needs_remapping( copy_particles_buffer ) );
            ASSERT_TRUE( !::st_Buffer_needs_remapping( orig_particles_buffer ) );

            ASSERT_TRUE( ::st_Buffer_get_num_of_objects( copy_particles_buffer ) ==
                         ::st_Buffer_get_num_of_objects( orig_particles_buffer ) );

            ASSERT_TRUE( ::st_Particles_buffers_have_same_structure(
                copy_particles_buffer, orig_particles_buffer ) );

            ASSERT_TRUE( ::st_Particles_buffer_get_total_num_of_particles(
                            copy_particles_buffer ) ==
                        ::st_Particles_buffer_get_total_num_of_particles(
                            orig_particles_buffer ) );

            ASSERT_TRUE( ::st_Particles_buffers_compare_values(
                orig_particles_buffer, copy_particles_buffer ) == 0 );

            /* ------------------------------------------------------------- */

            ASSERT_TRUE( ::st_Buffer_reset( copy_particles_buffer ) == 0 );

            success = st_Run_test_particles_copy_buffer_kernel_on_cuda_grid(
                orig_particles_buffer, copy_particles_buffer, 1, 64 );

            ASSERT_TRUE( success == 0 );

            ASSERT_TRUE( !::st_Buffer_needs_remapping( copy_particles_buffer ) );
            ASSERT_TRUE( !::st_Buffer_needs_remapping( orig_particles_buffer ) );

            ASSERT_TRUE( ::st_Buffer_get_num_of_objects( copy_particles_buffer ) ==
                         ::st_Buffer_get_num_of_objects( orig_particles_buffer ) );

            ASSERT_TRUE( ::st_Particles_buffers_have_same_structure(
                copy_particles_buffer, orig_particles_buffer ) );

            ASSERT_TRUE( ::st_Particles_buffer_get_total_num_of_particles(
                            copy_particles_buffer ) ==
                        ::st_Particles_buffer_get_total_num_of_particles(
                            orig_particles_buffer ) );

            ASSERT_TRUE( ::st_Particles_buffers_compare_values(
                orig_particles_buffer, copy_particles_buffer ) == 0 );

            /* ------------------------------------------------------------- */

            ASSERT_TRUE( ::st_Buffer_reset( copy_particles_buffer ) == 0 );

            success = st_Run_test_particles_copy_buffer_kernel_on_cuda_grid(
                orig_particles_buffer, copy_particles_buffer, 128, 128 );

            ASSERT_TRUE( success == 0 );

            ASSERT_TRUE( !::st_Buffer_needs_remapping( copy_particles_buffer ) );
            ASSERT_TRUE( !::st_Buffer_needs_remapping( orig_particles_buffer ) );

            ASSERT_TRUE( ::st_Buffer_get_num_of_objects( copy_particles_buffer ) ==
                         ::st_Buffer_get_num_of_objects( orig_particles_buffer ) );

            ASSERT_TRUE( ::st_Particles_buffers_have_same_structure(
                copy_particles_buffer, orig_particles_buffer ) );

            ASSERT_TRUE( ::st_Particles_buffer_get_total_num_of_particles(
                            copy_particles_buffer ) ==
                        ::st_Particles_buffer_get_total_num_of_particles(
                            orig_particles_buffer ) );

            ASSERT_TRUE( ::st_Particles_buffers_compare_values(
                orig_particles_buffer, copy_particles_buffer ) == 0 );

            /* ------------------------------------------------------------- */

            ASSERT_TRUE( ::st_Buffer_reset( copy_particles_buffer ) == 0 );

            success = st_Run_test_particles_copy_buffer_kernel_on_cuda(
                orig_particles_buffer, copy_particles_buffer );

            ASSERT_TRUE( success == 0 );

            ASSERT_TRUE( !::st_Buffer_needs_remapping( copy_particles_buffer ) );
            ASSERT_TRUE( !::st_Buffer_needs_remapping( orig_particles_buffer ) );

            ASSERT_TRUE( ::st_Buffer_get_num_of_objects( copy_particles_buffer ) ==
                         ::st_Buffer_get_num_of_objects( orig_particles_buffer ) );

            ASSERT_TRUE( ::st_Particles_buffers_have_same_structure(
                copy_particles_buffer, orig_particles_buffer ) );

            ASSERT_TRUE( ::st_Particles_buffer_get_total_num_of_particles(
                            copy_particles_buffer ) ==
                        ::st_Particles_buffer_get_total_num_of_particles(
                            orig_particles_buffer ) );

            ASSERT_TRUE( ::st_Particles_buffers_compare_values(
                orig_particles_buffer, copy_particles_buffer ) == 0 );

            /* ------------------------------------------------------------- */


        }
    }

    ::st_Buffer_delete( orig_particles_buffer );
    ::st_Buffer_delete( copy_particles_buffer );
}

/* end: tests/sixtracklib/cuda/test_particles_cuda_c99.cpp */
