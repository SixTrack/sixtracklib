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


TEST( C99_OpenCL_ParticlesTests, CopyParticlesHostToDeviceThenBackCompare )
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
              << " -DSIXTRL_PARTICLE_ARGPTR_DEC=__global"
              << " -DSIXTRL_PARTICLE_DATAPTR_DEC=__global"
              << " -DSIXTRL_BUFFER_OBJ_ARGPTR_DEC=__global"
              << " -DISXTRL_BUFFER_OBJ_DATAPTR_DEC=__global"
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
        path_to_source += "tests/sixtracklib/opencl/test_particles_kernel.cl";

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
            /* reset copy_particles_buffer structure and re-init:            */
            /* ------------------------------------------------------------- */

            ::st_Buffer_clear( copy_particles_buffer, true );

            size_t total_num_particles = size_t{ 0 };

            auto in_obj_it  = ::st_Buffer_get_const_objects_begin(
                orig_particles_buffer );

            auto in_obj_end = ::st_Buffer_get_const_objects_end(
                orig_particles_buffer );

            for( ; in_obj_it != in_obj_end ; ++in_obj_it )
            {
                particles_t const* orig = ( particles_t const* )( uintptr_t
                    )::st_Object_get_begin_addr( in_obj_it );

                size_t const num_particles =
                    ::st_Particles_get_num_of_particles( orig );

                particles_t* new_particle =
                    ::st_Particles_new( copy_particles_buffer, num_particles );

                total_num_particles += num_particles;

                ASSERT_TRUE( new_particle != nullptr );
                ASSERT_TRUE( ::st_Particles_have_same_structure( orig, new_particle ) );
                ASSERT_TRUE( !::st_Particles_map_to_same_memory( orig, new_particle ) );
            }

            ASSERT_TRUE( total_num_particles > size_t{ 0 } );

            ASSERT_TRUE(
                ::st_Buffer_get_num_of_objects( orig_particles_buffer ) ==
                ::st_Buffer_get_num_of_objects( copy_particles_buffer ) );

            ASSERT_TRUE( ::st_Particles_buffers_have_same_structure(
                orig_particles_buffer, copy_particles_buffer ) );

            ASSERT_TRUE( ::st_Particles_buffer_compare_values(
                orig_particles_buffer, copy_particles_buffer ) != 0 );

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

            cl::Buffer cl_orig_buffer(
                context, CL_MEM_READ_WRITE, particle_buffer_size );

            cl::Buffer cl_copy_buffer(
                context, CL_MEM_READ_WRITE, particle_buffer_size );

            cl::Buffer cl_success_flag(
                context, CL_MEM_READ_WRITE, sizeof( int32_t ) );

            try
            {
                cl_ret = queue.enqueueWriteBuffer( cl_orig_buffer, CL_TRUE, 0,
                    ::st_Buffer_get_size( orig_particles_buffer ),
                    ::st_Buffer_get_const_data_begin( orig_particles_buffer ) );
            }
            catch( cl::Error const& e )
            {
                std::cout << "enqueueWriteBuffer( orig_particles_buffer ) :: "
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
                   ::st_Buffer_get_size( copy_particles_buffer ),
                   ::st_Buffer_get_const_data_begin( copy_particles_buffer ) );
            }
            catch( cl::Error const& e )
            {
                std::cout << "enqueueWriteBuffer( copy_particles_buffer ) :: "
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
                copy_kernel =
                    cl::Kernel( copy_program, "st_Particles_copy_buffer_opencl" );
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

            size_t copy_num_threads = total_num_particles / copy_work_group_size;

            copy_num_threads *= copy_work_group_size;

            if( total_num_particles > copy_num_threads )
            {
                copy_num_threads += copy_work_group_size;
            }

            std::cout << "INFO  :: copy     kernel wg size       : "
                      << copy_work_group_size << "\r\n"
                      << "INFO  :: copy     kernel wg size multi : "
                      << copy_work_group_size_prefered_multiple << "\r\n"
                      << "INFO  :: copy     kernel launch with   : "
                      << copy_num_threads << " threads \r\n"
                      << "INFO  :: total num of particles        : "
                      << total_num_particles << "\r\n"
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
                    ::st_Buffer_get_size( copy_particles_buffer ),
                    ::st_Buffer_get_data_begin( copy_particles_buffer ) );
            }
            catch( cl::Error const& e )
            {
                std::cout << "enqueueReadBuffer( copy_particles_buffer ) :: "
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

            success = ::st_Buffer_remap( copy_particles_buffer );

            ASSERT_TRUE( success == 0 );

            ASSERT_TRUE(
                ::st_Buffer_get_num_of_objects( copy_particles_buffer ) ==
                ::st_Buffer_get_num_of_objects( orig_particles_buffer ) );

            ASSERT_TRUE(
                ::st_Particles_buffers_have_same_structure(
                orig_particles_buffer, copy_particles_buffer ) );

            ASSERT_TRUE( ::st_Particles_buffer_compare_values(
                orig_particles_buffer, copy_particles_buffer ) == 0 );
        }
    }
    else
    {
        std::cerr << "!!! --> WARNING :: Unable to perform unit-test as "
                  << "no valid OpenCL nodes have been found. -> skipping "
                  << "this unit-test <-- !!!" << std::endl;
    }

    ::st_Buffer_delete( orig_particles_buffer );
    ::st_Buffer_delete( copy_particles_buffer );
}


/* end: tests/sixtracklib/opencl/test_be_drift_opencl_c99.cpp */
