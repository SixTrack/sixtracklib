#define _USE_MATH_DEFINES

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <iterator>
#include <fstream>
#include <random>
#include <sstream>
#include <vector>

#include <gtest/gtest.h>

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#include "sixtracklib/testlib.h"

#include "sixtracklib/_impl/definitions.h"
#include "sixtracklib/_impl/path.h"
#include "sixtracklib/common/buffer.h"
#include "sixtracklib/common/impl/be_drift.h"

TEST( C99_OpenCL_BeamElementsDriftTests, CopyDriftsHostToDeviceThenBackCompare )
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

    ASSERT_TRUE( ::st_Buffer_get_num_of_objects( orig_buffer ) > size_t{ 0 } );
    size_t const buffer_size = ::st_Buffer_get_size( orig_buffer );
    ASSERT_TRUE( buffer_size > size_t{ 0 } );


    buffer_t* copy_buffer = ::st_Buffer_new( buffer_size );
    ASSERT_TRUE( copy_buffer != nullptr );

    int success = ::st_Buffer_reserve( copy_buffer,
        ::st_Buffer_get_num_of_objects( orig_buffer ),
        ::st_Buffer_get_num_of_slots( orig_buffer ),
        ::st_Buffer_get_num_of_dataptrs( orig_buffer ),
        ::st_Buffer_get_num_of_garbage_ranges( orig_buffer ) );

    ASSERT_TRUE( success == 0 );

    /* --------------------------------------------------------------------- */

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
        std::ostringstream a2str;

        std::string const PATH_TO_BASE_DIR = ::st_PATH_TO_BASE_DIR;

        /* ----------------------------------------------------------------- */

        a2str.str( "" );
        a2str << " -D_GPUCODE=1"
              << " -D__NAMESPACE=st_"
              << " -DSIXTRL_DATAPTR_DEC=__global"
              << " -DSIXTRL_BUFFER_DATAPTR_DEC=__global"
              << " -DSIXTRL_BUFFER_OBJ_ARGPTR_DEC=__global"
              << " -DISXTRL_BUFFER_OBJ_DATAPTR_DEC=__global"
              << " -DSIXTRL_BE_ARGPTR_DEC=__global"
              << " -DSIXTRL_BE_DATAPTR_DEC=__global"
              << " -w"
              << " -Werror"
              << " -I" << PATH_TO_BASE_DIR;

        std::string const REMAP_COMPILE_OPTIONS = a2str.str();

        std::string path_to_source = PATH_TO_BASE_DIR + std::string(
            "sixtracklib/opencl/impl/managed_buffer_remap_kernel.cl" );

        std::ifstream kernel_file( path_to_source, std::ios::in );

        std::string const REMAP_PROGRAM_SOURCE_CODE(
            ( std::istreambuf_iterator< char >( kernel_file ) ),
              std::istreambuf_iterator< char >() );

        kernel_file.close();

        /* ----------------------------------------------------------------- */

        path_to_source  = PATH_TO_BASE_DIR;
        path_to_source += "tests/sixtracklib/opencl/";
        path_to_source += "test_beam_elements_opencl_kernel.cl";

        kernel_file.open( path_to_source, std::ios::in );

        std::string const COPY_PROGRAM_SOURCE_CODE(
            ( std::istreambuf_iterator< char >( kernel_file ) ),
              std::istreambuf_iterator< char >() );

        a2str << " -I" << PATH_TO_BASE_DIR << "tests";

        std::string const COPY_COMPILE_OPTIONS = a2str.str();

        kernel_file.close();

        /* ----------------------------------------------------------------- */

        for( auto& device : devices )
        {
            cl::Platform platform( device.getInfo< CL_DEVICE_PLATFORM >() );

            std::cout << "--------------------------------------------------"
                      << "----------------------------------------------\r\n"
                      << "INFO  :: Perform test for device       : "
                      << device.getInfo< CL_DEVICE_NAME >() << "\r\n"
                      << "INFO  :: Platform                      : "
                      << platform.getInfo< CL_PLATFORM_NAME >() << "\r\n"
                      << "INFO  :: Platform Vendor               : "
                      << platform.getInfo< CL_PLATFORM_VENDOR >() << "\r\n"
                      << "INFO  :: Device Type                   : ";

            auto const device_type = device.getInfo< CL_DEVICE_TYPE >();

            switch( device_type )
            {
                case CL_DEVICE_TYPE_CPU:
                {
                    std::cout << "CPU";
                    break;
                }

                case CL_DEVICE_TYPE_GPU:
                {
                    std::cout << "GPU";
                    break;
                }

                case CL_DEVICE_TYPE_ACCELERATOR:
                {
                    std::cout << "Accelerator";
                    break;
                }

                case CL_DEVICE_TYPE_CUSTOM:
                {
                    std::cout << "Custom";
                    break;
                }

                default:
                {
                    std::cout << "Unknown";
                }
            };

            size_t const device_max_compute_units =
                device.getInfo< CL_DEVICE_MAX_COMPUTE_UNITS >();

            std::cout << "\r\n"
                      << "INFO  :: Max work-group size           : "
                      << device.getInfo< CL_DEVICE_MAX_WORK_GROUP_SIZE >()
                      << "\r\n"
                      << "INFO  :: Max num compute units         : "
                      << device_max_compute_units << "\r\n";

            /* ------------------------------------------------------------- */
            /* reset copy_buffer structure and re-init:            */
            /* ------------------------------------------------------------- */

            ::st_Buffer_clear( copy_buffer, true );

            auto in_obj_it  = ::st_Buffer_get_const_objects_begin( orig_buffer );
            auto in_obj_end = ::st_Buffer_get_const_objects_end( orig_buffer );

            for( ; in_obj_it != in_obj_end ; ++in_obj_it )
            {
                ::st_object_type_id_t const type_id =
                    NS(Object_get_type_id)( in_obj_it );

                switch( type_id )
                {
                    case NS(OBJECT_TYPE_DRIFT):
                    {
                        typedef NS(Drift) belem_t;
                        typedef SIXTRL_BE_ARGPTR_DEC belem_t* ptr_belem_t;

                        ptr_belem_t new_belem = NS(Drift_new)( copy_buffer );
                        ASSERT_TRUE( new_belem != nullptr );
                        break;
                    }

                    case NS(OBJECT_TYPE_DRIFT_EXACT):
                    {
                        typedef NS(DriftExact)   belem_t;
                        typedef SIXTRL_BE_ARGPTR_DEC belem_t* ptr_belem_t;

                        ptr_belem_t new_belem = NS(DriftExact_new)( copy_buffer );
                        ASSERT_TRUE( new_belem != nullptr );
                        break;
                    }

                    default:
                    {
                        success = -1;
                    }
                };
            }

            ASSERT_TRUE( success == 0 );

            size_t const total_num_beam_elements =
                ::st_Buffer_get_num_of_objects( orig_buffer );

            ASSERT_TRUE( total_num_beam_elements ==
                         ::st_Buffer_get_num_of_objects( copy_buffer ) );

            /* ------------------------------------------------------------- */

            cl_int cl_ret = CL_SUCCESS;

            cl::Context context( device );
            cl::CommandQueue queue( context, device, CL_QUEUE_PROFILING_ENABLE );
            cl::Program remap_program( context, REMAP_PROGRAM_SOURCE_CODE );
            cl::Program copy_program(  context, COPY_PROGRAM_SOURCE_CODE );

            try
            {
                cl_ret = remap_program.build( REMAP_COMPILE_OPTIONS.c_str() );
            }
            catch( cl::Error const& e )
            {
                std::cerr
                      << "ERROR :: remap_program :: "
                      << "OpenCL Compilation Error -> Stopping Unit-Test \r\n"
                      << remap_program.getBuildInfo< CL_PROGRAM_BUILD_LOG >( device )
                      << "\r\n"
                      << std::endl;

                cl_ret = CL_FALSE;
                throw;
            }

            ASSERT_TRUE( cl_ret == CL_SUCCESS );

            try
            {
                cl_ret = copy_program.build( COPY_COMPILE_OPTIONS.c_str() );
            }
            catch( cl::Error const& e )
            {
                std::cerr
                      << "ERROR :: copy_program :: "
                      << "OpenCL Compilation Error -> Stopping Unit-Test \r\n"
                      << copy_program.getBuildInfo< CL_PROGRAM_BUILD_LOG >( device )
                      << "\r\n"
                      << std::endl;

                cl_ret = CL_FALSE;
                throw;
            }

            ASSERT_TRUE( cl_ret == CL_SUCCESS );

            cl::Buffer cl_orig_buffer( context, CL_MEM_READ_WRITE, buffer_size );
            cl::Buffer cl_copy_buffer( context, CL_MEM_READ_WRITE, buffer_size );
            cl::Buffer cl_success_flag( context, CL_MEM_READ_WRITE, sizeof( int32_t ) );

            try
            {
                cl_ret = queue.enqueueWriteBuffer( cl_orig_buffer, CL_TRUE, 0,
                    ::st_Buffer_get_size( orig_buffer ),
                    ::st_Buffer_get_const_data_begin( orig_buffer ) );
            }
            catch( cl::Error const& e )
            {
                std::cout << "enqueueWriteBuffer( orig_buffer ) :: "
                          << "line = " << __LINE__
                          << " :: ERROR : " << e.what() << std::endl
                          << e.err() << std::endl;

                cl_ret = CL_FALSE;
                throw;
            }

            ASSERT_TRUE( cl_ret == CL_SUCCESS );

            try
            {
                cl_ret = queue.enqueueWriteBuffer( cl_copy_buffer, CL_TRUE, 0,
                   ::st_Buffer_get_size( copy_buffer ),
                   ::st_Buffer_get_const_data_begin( copy_buffer ) );
            }
            catch( cl::Error const& e )
            {
                std::cout << "enqueueWriteBuffer( copy_buffer ) :: "
                          << "line = " << __LINE__
                          << " :: ERROR : " << e.what() << std::endl
                          << e.err() << std::endl;

                cl_ret = CL_FALSE;
                throw;
            }

            ASSERT_TRUE( cl_ret == CL_SUCCESS );

            int32_t success_flag = int32_t{ 0 };

            try
            {
                cl_ret = queue.enqueueWriteBuffer( cl_success_flag, CL_TRUE, 0,
                    sizeof( int32_t ), &success_flag );
            }
            catch( cl::Error const& e )
            {
                std::cout << "enqueueWriteBuffer( success_flag ) :: "
                          << "line = " << __LINE__
                          << " :: ERROR : " << e.what() << std::endl
                          << e.err() << std::endl;

                cl_ret = CL_FALSE;
                throw;
            }

            ASSERT_TRUE( cl_ret == CL_SUCCESS );

            /* ============================================================= *
             * REMAP KERNEL *
             * ============================================================= */

            cl::Kernel remap_kernel;

            try
            {
                remap_kernel = cl::Kernel(
                    remap_program, "st_ManagedBuffer_remap_io_buffers_opencl" );
            }
            catch( cl::Error const& e )
            {
                std::cout << "kernel remap_kernel :: "
                          << "line  = " << __LINE__ << " :: "
                          << "ERROR : " << e.what() << "\r\n"
                          << e.err() << std::endl;

                cl_ret = CL_FALSE;
                throw;
            }

            ASSERT_TRUE( remap_kernel.getInfo< CL_KERNEL_NUM_ARGS >() == 3u );

            size_t remap_work_group_size = remap_kernel.getWorkGroupInfo<
                CL_KERNEL_WORK_GROUP_SIZE >( device );

            size_t const remap_work_group_size_prefered_multiple =
                remap_kernel.getWorkGroupInfo<
                    CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE >( device );

            size_t remap_num_threads = remap_work_group_size;

            std::cout << "INFO  :: remap kernel wg size          : "
                      << remap_work_group_size << "\r\n"
                      << "INFO  :: remap kernel wg size multi    : "
                      << remap_work_group_size_prefered_multiple << "\r\n"
                      << "INFO  :: remap kernel launch with      : "
                      << remap_num_threads << " threads \r\n"
                      << std::endl;

            ASSERT_TRUE( remap_work_group_size_prefered_multiple > size_t{ 0 } );
            ASSERT_TRUE( remap_work_group_size > size_t{ 0 } );
            ASSERT_TRUE( remap_num_threads > size_t{ 0 } );
            ASSERT_TRUE( ( remap_num_threads % remap_work_group_size ) ==
                size_t{ 0 } );

            remap_kernel.setArg( 0, cl_orig_buffer );
            remap_kernel.setArg( 1, cl_copy_buffer );
            remap_kernel.setArg( 2, cl_success_flag );

            try
            {
                cl_ret = queue.enqueueNDRangeKernel(
                    remap_kernel, cl::NullRange,
                    cl::NDRange( remap_num_threads ),
                    cl::NDRange( remap_work_group_size ) );
            }
            catch( cl::Error const& e )
            {
                std::cout << "enqueueNDRangeKernel( remap_kernel ) :: "
                          << "line = " << __LINE__
                          << " :: ERROR : " << e.what() << std::endl
                          << e.err() << std::endl;

                cl_ret = CL_FALSE;
                throw;
            }

            ASSERT_TRUE( cl_ret == CL_SUCCESS );

            try
            {
                cl_ret = queue.enqueueReadBuffer( cl_success_flag, CL_TRUE, 0,
                    sizeof( success_flag ), &success_flag );
            }
            catch( cl::Error const& e )
            {
                std::cout << "enqueueReadBuffer( success_flag ) :: "
                          << "line = " << __LINE__
                          << " :: ERROR : " << e.what() << std::endl
                          << e.err() << std::endl;

                cl_ret = CL_FALSE;
                throw;
            }

            ASSERT_TRUE( cl_ret == CL_SUCCESS );

            if( success_flag != int32_t{ 0 } )
            {
                std::cout << "ERROR :: remap kernel success flag     : "
                          << success_flag
                          << std::endl;
            }

            ASSERT_TRUE( success_flag == int32_t{ 0 } );

            /* ============================================================= *
             * TRACKING KERNEL *
             * ============================================================= */

            cl::Kernel copy_kernel;

            try
            {
                copy_kernel = cl::Kernel( copy_program,
                    "st_BeamElements_copy_beam_elements_opencl" );
            }
            catch( cl::Error const& e )
            {
                std::cout << "kernel copy_kernel :: "
                          << "line = " << __LINE__
                          << " :: ERROR : " << e.what() << std::endl
                          << e.err() << std::endl;
                cl_ret = CL_FALSE;
                throw;
            }

            ASSERT_TRUE( copy_kernel.getInfo< CL_KERNEL_NUM_ARGS >() == 3u );

            size_t copy_work_group_size = copy_kernel.getWorkGroupInfo<
                CL_KERNEL_WORK_GROUP_SIZE >( device );

            size_t const copy_work_group_size_prefered_multiple =
                copy_kernel.getWorkGroupInfo<
                    CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE >( device );

            ASSERT_TRUE( copy_work_group_size_prefered_multiple > size_t{ 0 } );
            ASSERT_TRUE( copy_work_group_size > size_t{ 0 } );

            size_t copy_num_threads = total_num_beam_elements / copy_work_group_size;

            copy_num_threads *= copy_work_group_size;

            if( total_num_beam_elements > copy_num_threads )
            {
                copy_num_threads += copy_work_group_size;
            }

            std::cout << "INFO  :: copy     kernel wg size       : "
                      << copy_work_group_size << "\r\n"
                      << "INFO  :: copy     kernel wg size multi : "
                      << copy_work_group_size_prefered_multiple << "\r\n"
                      << "INFO  :: copy     kernel launch with   : "
                      << copy_num_threads << " threads \r\n"
                      << "INFO  :: total num of beam_elements    : "
                      << total_num_beam_elements << "\r\n"
                      << std::endl;


            ASSERT_TRUE( copy_num_threads > size_t{ 0 } );
            ASSERT_TRUE( ( copy_num_threads % copy_work_group_size ) == size_t{ 0 } );

            copy_kernel.setArg( 0, cl_orig_buffer );
            copy_kernel.setArg( 1, cl_copy_buffer );
            copy_kernel.setArg( 2, cl_success_flag );

            try
            {
                cl_ret = queue.enqueueNDRangeKernel(
                    copy_kernel, cl::NullRange,
                    cl::NDRange( copy_num_threads ),
                    cl::NDRange( copy_work_group_size  ) );
            }
            catch( cl::Error const& e )
            {
                std::cout << "enqueueNDRangeKernel( copy_kernel ) :: "
                          << "line = " << __LINE__
                          << " :: ERROR : " << e.what() << std::endl
                          << e.err() << std::endl;

                cl_ret = CL_FALSE;
                throw;
            }

            try
            {
                cl_ret = queue.enqueueReadBuffer( cl_success_flag, CL_TRUE, 0,
                    sizeof( success_flag ), &success_flag );
            }
            catch( cl::Error const& e )
            {
                std::cout << "enqueueReadBuffer( success_flag ) :: "
                          << "line = " << __LINE__
                          << " :: ERROR : " << e.what() << std::endl
                          << e.err() << std::endl;

                cl_ret = CL_FALSE;
                throw;
            }

            ASSERT_TRUE( cl_ret == CL_SUCCESS );

            if( success_flag != int32_t{ 0 } )
            {
                std::cout << "ERROR :: tracking kernel success flag  : "
                          << success_flag
                          << std::endl;
            }

            ASSERT_TRUE( success_flag == int32_t{ 0 } );

            try
            {
                cl_ret = queue.enqueueReadBuffer( cl_copy_buffer, CL_TRUE, 0,
                    ::st_Buffer_get_size( copy_buffer ),
                    ::st_Buffer_get_data_begin( copy_buffer ) );
            }
            catch( cl::Error const& e )
            {
                std::cout << "enqueueReadBuffer( copy_buffer ) :: "
                          << "line = " << __LINE__
                          << " :: ERROR : " << e.what() << std::endl
                          << e.err() << std::endl;

                cl_ret = CL_FALSE;
                throw;
            }

            ASSERT_TRUE( cl_ret == CL_SUCCESS );

            /* ============================================================= */
            /* COMPARE COPIED DATA TO ORIGINAL DATA                          */
            /* ============================================================= */

            success = ::st_Buffer_remap( copy_buffer );

            ASSERT_TRUE( success == 0 );

            ASSERT_TRUE(
                ::st_Buffer_get_num_of_objects( copy_buffer ) ==
                ::st_Buffer_get_num_of_objects( orig_buffer ) );

            in_obj_it  = ::st_Buffer_get_const_objects_begin( orig_buffer );
            in_obj_end = ::st_Buffer_get_const_objects_end( orig_buffer );

            auto out_obj_it = ::st_Buffer_get_const_objects_begin( copy_buffer );

            for( ; in_obj_it != in_obj_end ; ++in_obj_it, ++out_obj_it )
            {
                ::st_object_type_id_t const type_id =
                    ::st_Object_get_type_id( in_obj_it );

                ASSERT_TRUE( ::st_Object_get_type_id( out_obj_it ) == type_id );

                switch( type_id )
                {
                    case NS(OBJECT_TYPE_DRIFT):
                    {
                        typedef NS(Drift) belem_t;
                        typedef SIXTRL_BE_ARGPTR_DEC belem_t const* ptr_belem_t;

                        ptr_belem_t orig_belem = ( ptr_belem_t )( uintptr_t
                            )NS(Object_get_begin_addr)( in_obj_it );

                        ptr_belem_t copy_belem = ( ptr_belem_t )( uintptr_t
                            )NS(Object_get_begin_addr)( out_obj_it );

                        ASSERT_TRUE( orig_belem != nullptr );
                        ASSERT_TRUE( copy_belem != nullptr );

                        ASSERT_TRUE( ::st_Drift_compare(
                            orig_belem, copy_belem ) == 0 );

                        break;
                    }

                    case NS(OBJECT_TYPE_DRIFT_EXACT):
                    {
                        typedef NS(DriftExact) belem_t;
                        typedef SIXTRL_BE_ARGPTR_DEC belem_t const* ptr_belem_t;

                        ptr_belem_t orig_belem = ( ptr_belem_t )( uintptr_t
                            )NS(Object_get_begin_addr)( in_obj_it );

                        ptr_belem_t copy_belem = ( ptr_belem_t )( uintptr_t
                            )NS(Object_get_begin_addr)( out_obj_it );

                        ASSERT_TRUE( orig_belem != nullptr );
                        ASSERT_TRUE( copy_belem != nullptr );

                        ASSERT_TRUE( ::st_DriftExact_compare(
                            orig_belem, copy_belem ) == 0 );

                        break;
                    }

                    default:
                    {
                        success = -1;
                    }
                };

                ASSERT_TRUE( success == 0 );
            }
        }
    }
    else
    {
        std::cerr << "!!! --> WARNING :: Unable to perform unit-test as "
                  << "no valid OpenCL nodes have been found. -> skipping "
                  << "this unit-test <-- !!!" << std::endl;
    }

    ::st_Buffer_delete( orig_buffer );
    ::st_Buffer_delete( copy_buffer );
}


/* end: tests/sixtracklib/opencl/test_be_drift_opencl_c99.cpp */
