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

TEST( C99_OpenCL_Buffer, InitWithGenericObjDataCopyToDeviceCopyBackCmpSingleThread )
{
    using buffer_t      = ::st_Buffer;
    using size_t        = ::st_buffer_size_t;

    buffer_t* orig_buffer = ::st_Buffer_new_from_file(
        ::st_PATH_TO_TEST_GENERIC_OBJ_BUFFER_DATA );
    ASSERT_TRUE( orig_buffer != nullptr );
    ASSERT_TRUE( ::st_Buffer_get_num_of_objects( orig_buffer ) > size_t{ 0 } );

    size_t const orig_buffer_size = ::st_Buffer_get_size( orig_buffer );
    ASSERT_TRUE( orig_buffer_size > size_t{ 0 } );

    buffer_t* copy_buffer = ::st_Buffer_new( 4096u );
    ASSERT_TRUE( copy_buffer != nullptr );
    ASSERT_TRUE( copy_buffer != orig_buffer );
    ASSERT_TRUE( st_Buffer_get_data_begin_addr( orig_buffer ) !=
                 st_Buffer_get_data_begin_addr( copy_buffer ) );

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
    ASSERT_TRUE( copy_buffer_begin != orig_buffer_begin );

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

    std::string const PATH_TO_BASE_DIR = ::st_PATH_TO_BASE_DIR;

    a2str.str( "" );
    a2str << " -D_GPUCODE=1"
          << " -D__NAMESPACE=st_"
          << " -DSIXTRL_BUFFER_DATAPTR_DEC=__global"
          << " -I" << PATH_TO_BASE_DIR
          << " -I" << PATH_TO_BASE_DIR << "tests" ;

    std::string const COMPILE_OPTIONS = a2str.str();


    std::string path_to_kernel_source = PATH_TO_BASE_DIR;
    path_to_kernel_source += "tests/sixtracklib/opencl/opencl_buffer_generic_obj_kernel.cl";

    std::ifstream kernel_file( path_to_kernel_source, std::ios::in );

    std::string const PROGRAM_SOURCE_CODE(
        ( std::istreambuf_iterator< char >( kernel_file ) ),
          std::istreambuf_iterator< char >() );

    kernel_file.close();

    for( auto& device : devices )
    {
        std::cout << "Perform test for device : "
                  << device.getInfo< CL_DEVICE_NAME >() << "\r\n"
                  << "Platform                : "
                  << device.getInfo< CL_DEVICE_PLATFORM >()
                  << std::endl;

        /* ---------------------------------------------------------------- */
        /* prepare copy buffer */

        ::st_Buffer_clear( copy_buffer, true );

        auto obj_it  = st_Buffer_get_const_objects_begin( orig_buffer );
        auto obj_end = st_Buffer_get_const_objects_end( orig_buffer );

        for( ; obj_it != obj_end ; ++obj_it )
        {
            ::st_GenericObj const* orig_obj = reinterpret_cast<
                ::st_GenericObj const* >( static_cast< uintptr_t >(
                    ::st_Object_get_begin_addr( obj_it ) ) );

            ASSERT_TRUE( orig_obj != nullptr );
            ASSERT_TRUE( orig_obj->type_id ==
                         ::st_Object_get_type_id( obj_it ) );

            ::st_GenericObj* copy_obj = ::st_GenericObj_new( copy_buffer,
                orig_obj->type_id, orig_obj->num_d, orig_obj->num_e );

            ASSERT_TRUE( copy_obj != nullptr );
            ASSERT_TRUE( orig_obj->type_id == copy_obj->type_id );
            ASSERT_TRUE( orig_obj->num_d   == copy_obj->num_d );
            ASSERT_TRUE( orig_obj->num_e   == copy_obj->num_e );
            ASSERT_TRUE( copy_obj->d != nullptr );
            ASSERT_TRUE( copy_obj->e != nullptr );
        }

        ASSERT_TRUE( ::st_Buffer_get_num_of_objects( copy_buffer ) ==
                     ::st_Buffer_get_num_of_objects( orig_buffer ) );

        /* ---------------------------------------------------------------- */

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

        try
        {
            cl_ret = queue.enqueueWriteBuffer(
                cl_orig_buf, CL_TRUE, 0, orig_buffer_size, orig_buffer_begin );
        }
        catch( cl::Error const& e )
        {
            std::cout << "enqueueWriteBuffer( orig_buffer_begin ) :: line = "
                      << __LINE__
                      << " :: ERROR : " << e.what() << std::endl
                      << e.err() << std::endl;

            cl_ret = CL_FALSE;
            throw;
        }

        ASSERT_TRUE( cl_ret == CL_SUCCESS );

        try
        {
            cl_ret = queue.enqueueWriteBuffer(
                cl_copy_buf, CL_TRUE, 0, orig_buffer_size, copy_buffer_begin );
        }
        catch( cl::Error const& e )
        {
            std::cout << "enqueueWriteBuffer( copy_buffer_begin ) :: line = "
                      << __LINE__
                      << " :: ERROR : " << e.what() << std::endl
                      << e.err() << std::endl;

            cl_ret = CL_FALSE;
            throw;
        }

        ASSERT_TRUE( cl_ret == CL_SUCCESS );

        int64_t error_flag = -1;

        try
        {
            cl_ret = queue.enqueueWriteBuffer(
                cl_err_flag, CL_TRUE, 0, sizeof( error_flag ), &error_flag );
        }
        catch( cl::Error const& e )
        {
            std::cout << "enqueueWriteBuffer( error_flag ) :: line = "
                      << __LINE__
                      << " :: ERROR : " << e.what() << std::endl
                      << e.err() << std::endl;

            cl_ret = CL_FALSE;
            throw;
        }

        ASSERT_TRUE( cl_ret == CL_SUCCESS );

        int num_threads = ( int )1;
        int block_size  = ( int )1;

        cl::Kernel remap_kernel;

        try
        {
            remap_kernel = cl::Kernel( program, "st_remap_orig_buffer" );
        }
        catch( cl::Error const& e )
        {
            std::cout << "kernel remap_kernel :: line = " << __LINE__
                      << " :: ERROR : " << e.what() << std::endl
                      << e.err() << std::endl;
            throw;
        }

        auto num_remap_kernel_args = remap_kernel.getInfo< CL_KERNEL_NUM_ARGS >();

        ASSERT_TRUE( num_remap_kernel_args == 3u );

        remap_kernel.setArg( 0, cl_orig_buf );
        remap_kernel.setArg( 1, cl_copy_buf );
        remap_kernel.setArg( 2, cl_err_flag );


        try
        {
            cl_ret = queue.enqueueNDRangeKernel( remap_kernel, cl::NullRange,
                cl::NDRange( num_threads ), cl::NDRange( block_size ) );
        }
        catch( cl::Error const& e )
        {
            std::cout << "enqueueNDRangeKernel( remap_kernel) :: line = "
                      << __LINE__
                      << " :: ERROR : " << e.what() << std::endl
                      << e.err() << std::endl;

            throw;
        }

        try
        {
            cl_ret = queue.enqueueReadBuffer(
                cl_err_flag, CL_TRUE, 0, sizeof( error_flag ), &error_flag );
        }
        catch( cl::Error const& e )
        {
            std::cout << "enqueueReadBuffer( error_flag ) :: line = "
                      << __LINE__ << " :: ERROR : " << e.what() << std::endl
                      << e.err() << std::endl;

            cl_ret = CL_FALSE;
            throw;
        }

        ASSERT_TRUE( cl_ret == CL_SUCCESS );
        ASSERT_TRUE( error_flag == int64_t{ 0 } );

        cl::Kernel copy_kernel;

        try
        {
            copy_kernel = cl::Kernel( program, "st_copy_orig_buffer" );
        }
        catch( cl::Error const& e )
        {
            std::cout << "kernel copy_kernel :: line = " << __LINE__
                      << " :: ERROR : " << e.what() << std::endl
                      << e.err() << std::endl;

            throw;
        }

        copy_kernel.setArg( 0, cl_orig_buf );
        copy_kernel.setArg( 1, cl_copy_buf );
        copy_kernel.setArg( 2, cl_err_flag );

        block_size  = ( int )1;
        num_threads = ( int )1;

        cl_ret = queue.enqueueNDRangeKernel( copy_kernel, cl::NullRange,
            cl::NDRange( num_threads ), cl::NDRange( block_size ) );

        ASSERT_TRUE( cl_ret == CL_SUCCESS );

        try
        {
            cl_ret = queue.enqueueReadBuffer(
                cl_err_flag, CL_TRUE, 0, sizeof( error_flag ), &error_flag );
        }
        catch( cl::Error const& e )
        {
            std::cout << "enqueueReadBuffer( error_flag ) :: line = "
                      << __LINE__
                      << " :: ERROR : " << e.what() << std::endl
                      << e.err() << std::endl;

            cl_ret = CL_FALSE;
            throw;
        }

        ASSERT_TRUE( cl_ret == CL_SUCCESS );

        try
        {
            cl_ret = queue.enqueueReadBuffer(
                cl_copy_buf, CL_TRUE, 0, orig_buffer_size, copy_buffer_begin );
        }
        catch( cl::Error const& e )
        {
            std::cout << "enqueueReadBuffer( copy_buffer_begin ) :: line = "
                      << __LINE__
                      << " :: ERROR : " << e.what() << std::endl
                      << e.err() << std::endl;

            cl_ret = CL_FALSE;
            throw;
        }

        ASSERT_TRUE( cl_ret == CL_SUCCESS );

        /* ----------------------------------------------------------------- */

        success = ::st_Buffer_remap( copy_buffer );
        ASSERT_TRUE( success == 0 );

        ASSERT_TRUE( ::st_Buffer_get_num_of_objects( copy_buffer ) ==
                     ::st_Buffer_get_num_of_objects( orig_buffer ) );

        obj_it       = st_Buffer_get_const_objects_begin( orig_buffer );
        obj_end      = st_Buffer_get_const_objects_end( orig_buffer );
        auto cmp_it  = st_Buffer_get_const_objects_begin( copy_buffer );

        for( ; obj_it != obj_end ; ++obj_it, ++cmp_it )
        {
            ::st_GenericObj const* orig_obj = reinterpret_cast<
                ::st_GenericObj const* >( static_cast< uintptr_t >(
                    ::st_Object_get_begin_addr( obj_it ) ) );

            ::st_GenericObj const* cmp_obj = reinterpret_cast<
                ::st_GenericObj const* >( static_cast< uintptr_t >(
                    ::st_Object_get_begin_addr( cmp_it ) ) );

            ASSERT_TRUE( orig_obj != nullptr );
            ASSERT_TRUE( cmp_obj  != nullptr );
            ASSERT_TRUE( cmp_obj  != orig_obj );

            ASSERT_TRUE( orig_obj->type_id == cmp_obj->type_id );
            ASSERT_TRUE( orig_obj->num_d   == cmp_obj->num_d );
            ASSERT_TRUE( orig_obj->num_e   == cmp_obj->num_e );
            ASSERT_TRUE( orig_obj->a       == cmp_obj->a );

            ASSERT_TRUE( std::fabs( orig_obj->a - cmp_obj->a ) <
                std::numeric_limits< double >::epsilon() );

            for( std::size_t ii = 0 ; ii < 4u ; ++ii )
            {
                ASSERT_TRUE( std::fabs( orig_obj->c[ ii ] - cmp_obj->c[ ii ] ) <
                    std::numeric_limits< double >::epsilon() );
            }

            if( orig_obj->num_d > 0u )
            {
                for( std::size_t ii = 0u ; ii < orig_obj->num_d ; ++ii )
                {
                    ASSERT_TRUE( orig_obj->d[ ii ] == cmp_obj->d[ ii ] );
                }
            }

            if( orig_obj->num_e > 0u )
            {
                for( std::size_t ii = 0u ; ii < orig_obj->num_e ; ++ii )
                {
                    ASSERT_TRUE( std::fabs( orig_obj->e[ ii ] - cmp_obj->e[ ii ] )
                        < std::numeric_limits< double >::epsilon() );
                }
            }
        }
    }

    ::st_Buffer_delete( orig_buffer );
    ::st_Buffer_delete( copy_buffer );
}

/* end: tests/sixtracklib/opencl/test_buffer_opencl_c99.cpp */
