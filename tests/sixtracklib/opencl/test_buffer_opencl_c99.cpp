#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <vector>

#include <gtest/gtest.h>

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#include "sixtracklib/testlib.h"

#include "sixtracklib/_impl/definitions.h"
#include "sixtracklib/common/buffer.h"

TEST( C99_OpenCLBuffer, InitWithGenericObjDataCopyToDeviceCopyBackCmp )
{
    using buffer_t      = ::st_Buffer;
    using size_t        = ::st_buffer_size_t;

    buffer_t* orig_buffer = ::st_Buffer_new_from_file(
        ::st_PATH_TO_TEST_GENERIC_OBJ_BUFFER_DATA );
    ASSERT_TRUE( orig_buffer != nullptr );
    ASSERT_TRUE( ::st_Buffer_get_num_of_objects( orig_buffer ) > size_t{ 0 } );

    size_t const orig_buffer_size = ::st_Buffer_get_size( orig_buffer );
    ASSERT_TRUE( orig_buffer_size > size_t{ 0 } );

    buffer_t* copy_buffer = ::st_Buffer_new( orig_buffer_size );
    ASSERT_TRUE( copy_buffer != nullptr );

    int success = ::st_Buffer_reserve( copy_buffer,
        ::st_Buffer_get_max_num_of_objects( orig_buffer ),
        ::st_Buffer_get_max_num_of_slots( orig_buffer ),
        ::st_Buffer_get_max_num_of_dataptrs( orig_buffer ),
        ::st_Buffer_get_max_num_of_garbage_ranges( orig_buffer ) );

    ASSERT_TRUE( success == 0 );

    unsigned char const* orig_buffer_begin =
        ::st_Buffer_get_const_data_begin( orig_buffer );

    ASSERT_TRUE( orig_buffer_begin != nullptr );

    unsigned char* copy_buffer_begin =
        ::st_Buffer_get_data_begin( copy_buffer );

    ASSERT_TRUE( copy_buffer_begin != nullptr );

    /* --------------------------------------------------------------------- */

    std::vector< cl::Platform > platforms;
    cl::Platform::get( &platforms );

    if( platforms.empty() )
    {
        std::cout << "Unable to perform unit-test as no OpenCL "
                  << "platforms have been found."
                  << std::endl;
    }

    ASSERT_TRUE( !platforms.empty() );

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

    if( devices.empty() )
    {
        std::cout << "Unable to perform unit-test as no valid OpenCL "
                  << "devices have been found."
                  << std::endl;
    }

    ASSERT_TRUE( !devices.empty() );

    std::ostringstream a2str;

    std::string const PROGRAM_SOURCE_CODE =
        "#include \"test_buffer_generic_obj_kernel.cl\"\r\n";

    std::string const PATH_TO_BASE_DIR( "/home/martin/git/sixtracklib/" );

    a2str.str( "" );
    a2str << " -D_GPUCODE=1"
          << " -D__NAMESPACE=st_"
          << " -DSIXTRL_DATAPTR_DEC=__global"
          << " -I" << PATH_TO_BASE_DIR
          << " -I" << PATH_TO_BASE_DIR << "tests/sixtracklib/opencl"
          << " -I" << PATH_TO_BASE_DIR << "tests";

    std::string const COMPILE_OPTIONS = a2str.str();

    for( auto& device : devices )
    {
        cl_int cl_ret = CL_SUCCESS;

        cl::Context context( device );
        cl::CommandQueue queue( context, device, CL_QUEUE_PROFILING_ENABLE );
        cl::Program program( context, PROGRAM_SOURCE_CODE );

        try
        {
            cl_ret = program.build( COMPILE_OPTIONS.c_str() );
        }
        catch( cl::Error const& e )
        {
            std::cerr << "OpenCL Compilation Error -> Stopping Unit-Test \r\n"
                      << program.getBuildInfo< CL_PROGRAM_BUILD_LOG >( device )
                      << "\r\n"
                      << std::endl;

            cl_ret = CL_FALSE;
            throw;
        }

        ASSERT_TRUE( cl_ret == CL_SUCCESS );

        cl::Buffer cl_orig_buf( context, CL_MEM_READ_WRITE, orig_buffer_size );
        cl::Buffer cl_err_flag( context, CL_MEM_READ_WRITE, sizeof( int64_t ) );
        cl::Buffer cl_copy_buf( context, CL_MEM_READ_WRITE, orig_buffer_size );

        cl_ret = queue.enqueueWriteBuffer(
            cl_orig_buf, CL_TRUE, 0, orig_buffer_size, orig_buffer_begin );

        ASSERT_TRUE( cl_ret == CL_SUCCESS );

        int64_t error_flag = -1;
        cl_ret = queue.enqueueWriteBuffer(
            cl_err_flag, CL_TRUE, 0, sizeof( error_flag ), &error_flag );

        int num_threads = ( int )1;
        int block_size  = ( int )1;

        uint64_t n = orig_buffer_size;

        cl::Kernel remap_kernel;

        try
        {
            remap_kernel = cl::Kernel( program, "st_remap_orig_buffer" );
        }
        catch( cl::Error const& err )
        {
            std::cout << "ERROR: " << err.what() << std::endl
                      << err.err() << std::endl;

            throw;
        }

        remap_kernel.setArg( 0, cl_orig_buf );
        remap_kernel.setArg( 1, n );
        remap_kernel.setArg( 2, cl_err_flag );

        cl_ret = queue.enqueueNDRangeKernel( remap_kernel, cl::NullRange,
            cl::NDRange( num_threads ), cl::NDRange( block_size ) );

        cl_ret = queue.enqueueReadBuffer(
            cl_err_flag, CL_TRUE, 0, sizeof( error_flag ), &error_flag );

        ASSERT_TRUE( cl_ret == CL_SUCCESS );
        ASSERT_TRUE( error_flag == int64_t{ 0 } );

        cl::Kernel copy_kernel;

        try
        {
            copy_kernel = cl::Kernel( program, "st_copy_orig_buffer" );
        }
        catch( cl::Error const& err )
        {
            std::cout << "ERROR: " << err.what() << std::endl
                      << err.err() << std::endl;

            throw;
        }

        copy_kernel.setArg( 0, cl_orig_buf );
        copy_kernel.setArg( 1, cl_copy_buf );
        copy_kernel.setArg( 2, n );
        copy_kernel.setArg( 3, cl_err_flag );

        num_threads = ( int )1;
        block_size  = ( int )1;

        cl_ret = queue.enqueueNDRangeKernel( copy_kernel, cl::NullRange,
            cl::NDRange( num_threads ), cl::NDRange( block_size ) );

        ASSERT_TRUE( cl_ret == CL_SUCCESS );

        cl_ret = queue.enqueueReadBuffer(
            cl_err_flag, CL_TRUE, 0, sizeof( error_flag ), &error_flag );

        ASSERT_TRUE( cl_ret == CL_SUCCESS );

        cl_ret = queue.enqueueReadBuffer(
            cl_copy_buf, CL_TRUE, 0, orig_buffer_size, copy_buffer_begin );

        ASSERT_TRUE( cl_ret == CL_SUCCESS );

        /* ----------------------------------------------------------------- */

//         success = ::st_Buffer_remap( copy_buffer );
//         ASSERT_TRUE( success == 0 );
//         ASSERT_TRUE( ::NS(st_Buffer_get_num_of_objects)( copy_buffer ) ==
//                      ::NS(st_Buffer_get_num_of_objects)( orig_buffer ) );

        ::st_Buffer_delete( copy_buffer );
        ::st_Buffer_delete( orig_buffer );
    }



}

/* end: tests/sixtracklib/opencl/test_managed_buffer_opencl_c99.cpp */
