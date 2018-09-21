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

#include "sixtracklib/cuda/impl/cuda_beam_elements_kernel_c_wrapper.h"

TEST( C99_Cuda_BeamElementsDriftTests,
      CopyDriftsHostToDeviceThenBackCompare )
{
    using buffer_t    = ::st_Buffer;
    using size_t      = ::st_buffer_size_t;

    std::string const path_to_data( ::st_PATH_TO_TEST_TRACKING_BE_DRIFT_DATA );

    /* --------------------------------------------------------------------- */

    buffer_t* orig_buffer = ::st_TrackTestdata_extract_beam_elements_buffer(
            path_to_data.c_str() );

    size_t const slot_size = ::st_Buffer_get_slot_size( orig_buffer );
    ASSERT_TRUE( slot_size == size_t{ 8 } );

    ASSERT_TRUE( orig_buffer != nullptr );


    size_t const num_beam_elements =
        ::st_Buffer_get_num_of_objects( orig_buffer );

    ASSERT_TRUE( num_beam_elements > size_t{ 0 } );

    size_t const buffer_size = ::st_Buffer_get_size( orig_buffer );
    ASSERT_TRUE( buffer_size > size_t{ 0 } );


    buffer_t* copy_buffer = ::st_Buffer_new( buffer_size );
    ASSERT_TRUE( copy_buffer != nullptr );

    int success = ::st_Buffer_reserve( copy_buffer,
        num_beam_elements, ::st_Buffer_get_num_of_slots( orig_buffer ),
        ::st_Buffer_get_num_of_dataptrs( orig_buffer ),
        ::st_Buffer_get_num_of_garbage_ranges( orig_buffer ) );

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

            ASSERT_TRUE( ::st_Buffer_reset( copy_buffer ) == 0 );

            success = st_Run_test_copy_beam_elements_buffer_kernel_cuda_grid(
                orig_buffer, copy_buffer, 1, 1 );

            ASSERT_TRUE( success == 0 );

            ASSERT_TRUE( !::st_Buffer_needs_remapping( copy_buffer ) );
            ASSERT_TRUE( !::st_Buffer_needs_remapping( orig_buffer ) );

            ASSERT_TRUE( ::st_Buffer_get_size( copy_buffer ) ==
                         ::st_Buffer_get_size( orig_buffer ) );

            ASSERT_TRUE( ::st_Buffer_get_num_of_objects( copy_buffer ) ==
                         ::st_Buffer_get_num_of_objects( orig_buffer ) );

            ASSERT_TRUE( 0 == ::st_BeamElements_compare_lines(
                ::st_Buffer_get_const_objects_begin( orig_buffer ),
                ::st_Buffer_get_const_objects_end(   orig_buffer ),
                ::st_Buffer_get_const_objects_begin( copy_buffer ) ) );

            /* ------------------------------------------------------------- */

            ASSERT_TRUE( ::st_Buffer_reset( copy_buffer ) == 0 );

            success = st_Run_test_copy_beam_elements_buffer_kernel_cuda_grid(
                orig_buffer, copy_buffer, 32, 1 );

            ASSERT_TRUE( success == 0 );

            ASSERT_TRUE( !::st_Buffer_needs_remapping( copy_buffer ) );
            ASSERT_TRUE( !::st_Buffer_needs_remapping( orig_buffer ) );

            ASSERT_TRUE( ::st_Buffer_get_size( copy_buffer ) ==
                         ::st_Buffer_get_size( orig_buffer ) );

            ASSERT_TRUE( ::st_Buffer_get_num_of_objects( copy_buffer ) ==
                         ::st_Buffer_get_num_of_objects( orig_buffer ) );

            ASSERT_TRUE( 0 == ::st_BeamElements_compare_lines(
                ::st_Buffer_get_const_objects_begin( orig_buffer ),
                ::st_Buffer_get_const_objects_end(   orig_buffer ),
                ::st_Buffer_get_const_objects_begin( copy_buffer ) ) );

            /* ------------------------------------------------------------- */

            ASSERT_TRUE( ::st_Buffer_reset( copy_buffer ) == 0 );

            success = st_Run_test_copy_beam_elements_buffer_kernel_cuda_grid(
                orig_buffer, copy_buffer, 1, 32 );

            ASSERT_TRUE( success == 0 );

            ASSERT_TRUE( !::st_Buffer_needs_remapping( copy_buffer ) );
            ASSERT_TRUE( !::st_Buffer_needs_remapping( orig_buffer ) );

            ASSERT_TRUE( ::st_Buffer_get_size( copy_buffer ) ==
                         ::st_Buffer_get_size( orig_buffer ) );

            ASSERT_TRUE( ::st_Buffer_get_num_of_objects( copy_buffer ) ==
                         ::st_Buffer_get_num_of_objects( orig_buffer ) );

            ASSERT_TRUE( 0 == ::st_BeamElements_compare_lines(
                ::st_Buffer_get_const_objects_begin( orig_buffer ),
                ::st_Buffer_get_const_objects_end(   orig_buffer ),
                ::st_Buffer_get_const_objects_begin( copy_buffer ) ) );

            if( num_beam_elements < 65535 )
            {
                /* ------------------------------------------------------------- */

                ASSERT_TRUE( ::st_Buffer_reset( copy_buffer ) == 0 );

                success = st_Run_test_copy_beam_elements_buffer_kernel_cuda_grid(
                    orig_buffer, copy_buffer, num_beam_elements, 1 );

                ASSERT_TRUE( success == 0 );

                ASSERT_TRUE( !::st_Buffer_needs_remapping( copy_buffer ) );
                ASSERT_TRUE( !::st_Buffer_needs_remapping( orig_buffer ) );

                ASSERT_TRUE( ::st_Buffer_get_size( copy_buffer ) ==
                             ::st_Buffer_get_size( orig_buffer ) );

                ASSERT_TRUE( ::st_Buffer_get_num_of_objects( copy_buffer ) ==
                             ::st_Buffer_get_num_of_objects( orig_buffer ) );

                ASSERT_TRUE( 0 == ::st_BeamElements_compare_lines(
                    ::st_Buffer_get_const_objects_begin( orig_buffer ),
                    ::st_Buffer_get_const_objects_end(   orig_buffer ),
                    ::st_Buffer_get_const_objects_begin( copy_buffer ) ) );
            }
            else
            {
                ASSERT_TRUE( ::st_Buffer_reset( copy_buffer ) == 0 );

                success = st_Run_test_copy_beam_elements_buffer_kernel_cuda_grid(
                    orig_buffer, copy_buffer, 64, 64 );

                ASSERT_TRUE( success == 0 );

                ASSERT_TRUE( !::st_Buffer_needs_remapping( copy_buffer ) );
                ASSERT_TRUE( !::st_Buffer_needs_remapping( orig_buffer ) );

                ASSERT_TRUE( ::st_Buffer_get_size( copy_buffer ) ==
                             ::st_Buffer_get_size( orig_buffer ) );

                ASSERT_TRUE( ::st_Buffer_get_num_of_objects( copy_buffer ) ==
                             ::st_Buffer_get_num_of_objects( orig_buffer ) );

                ASSERT_TRUE( 0 == ::st_BeamElements_compare_lines(
                    ::st_Buffer_get_const_objects_begin( orig_buffer ),
                    ::st_Buffer_get_const_objects_end(   orig_buffer ),
                    ::st_Buffer_get_const_objects_begin( copy_buffer ) ) );
            }

            /* ------------------------------------------------------------- */

            ASSERT_TRUE( ::st_Buffer_reset( copy_buffer ) == 0 );

            success = st_Run_test_copy_beam_elements_buffer_kernel_cuda(
                orig_buffer, copy_buffer );

            ASSERT_TRUE( success == 0 );

            ASSERT_TRUE( !::st_Buffer_needs_remapping( copy_buffer ) );
            ASSERT_TRUE( !::st_Buffer_needs_remapping( orig_buffer ) );

            ASSERT_TRUE( ::st_Buffer_get_size( copy_buffer ) ==
                         ::st_Buffer_get_size( orig_buffer ) );

            ASSERT_TRUE( ::st_Buffer_get_num_of_objects( copy_buffer ) ==
                         ::st_Buffer_get_num_of_objects( orig_buffer ) );

            ASSERT_TRUE( 0 == ::st_BeamElements_compare_lines(
                ::st_Buffer_get_const_objects_begin( orig_buffer ),
                ::st_Buffer_get_const_objects_end(   orig_buffer ),
                ::st_Buffer_get_const_objects_begin( copy_buffer ) ) );
        }
    }

    ::st_Buffer_delete( copy_buffer );
    ::st_Buffer_delete( orig_buffer );
}

/* end: tests/sixtracklib/cuda/test_be_drift_cuda_c99.cpp */
