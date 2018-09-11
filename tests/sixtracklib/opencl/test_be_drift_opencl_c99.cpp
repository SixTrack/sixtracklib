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


TEST( C99_OpenCL_BeamElementsTests, TrackDriftsFromTestsDataSet )
{
    using buffer_t    = ::st_Buffer;
    using particles_t = ::st_Particles;
    using size_t      = ::st_buffer_size_t;

    std::string const path_to_data( ::st_PATH_TO_TEST_TRACKING_BE_DRIFT_DATA );

    /* --------------------------------------------------------------------- */

    buffer_t* init_particles_buffer =
        ::st_TrackTestdata_extract_initial_particles_buffer(
            path_to_data.c_str() );

    buffer_t* beam_elements_buffer =
        ::st_TrackTestdata_extract_beam_elements_buffer(
            path_to_data.c_str() );

    buffer_t* result_particles_buffer =
        ::st_TrackTestdata_extract_result_particles_buffer(
            path_to_data.c_str() );

    ASSERT_TRUE( init_particles_buffer   != nullptr );
    ASSERT_TRUE( result_particles_buffer != nullptr );
    ASSERT_TRUE( beam_elements_buffer    != nullptr );

    ASSERT_TRUE( ::st_Buffer_get_num_of_objects( init_particles_buffer ) ==
                 ::st_Buffer_get_num_of_objects( result_particles_buffer ) );

    ASSERT_TRUE( ::st_Particles_buffers_have_same_structure(
        init_particles_buffer, result_particles_buffer ) );

    ASSERT_TRUE( ::st_Buffer_get_num_of_objects( init_particles_buffer ) >
                 static_cast< size_t >( 0 ) );

    ASSERT_TRUE( ::st_Buffer_get_num_of_objects( beam_elements_buffer ) >
                 static_cast< size_t >( 0 ) );

    size_t const particle_buffer_size =
        ::st_Buffer_get_size( init_particles_buffer );

    size_t const beam_elements_buffer_size =
        ::st_Buffer_get_size( beam_elements_buffer );

    ASSERT_TRUE( particle_buffer_size > size_t{ 0 } );
    ASSERT_TRUE( beam_elements_buffer_size > size_t{ 0 } );

    buffer_t* particles_buffer = ::st_Buffer_new( particle_buffer_size );

    ASSERT_TRUE( particles_buffer != nullptr );

    int success = ::st_Buffer_reserve( particles_buffer,
        ::st_Buffer_get_num_of_objects( init_particles_buffer ),
        ::st_Buffer_get_num_of_slots( init_particles_buffer ),
        ::st_Buffer_get_num_of_dataptrs( init_particles_buffer ),
        ::st_Buffer_get_num_of_garbage_ranges( init_particles_buffer ) );

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

        a2str.str( "" );
        a2str << " -D_GPUCODE=1"
              << " -D__NAMESPACE=st_"
              << " -DSIXTRL_BUFFER_DATAPTR_DEC=__global"
              << " -w"
              << " -Werror"
              << " -I" << PATH_TO_BASE_DIR;

        std::string const COMPILE_OPTIONS = a2str.str();

        std::string path_to_source = PATH_TO_BASE_DIR + std::string(
            "sixtracklib/opencl/impl/track_particles_kernel.cl" );

        std::ifstream kernel_file( path_to_source, std::ios::in );

        std::string const PROGRAM_SOURCE_CODE(
            ( std::istreambuf_iterator< char >( kernel_file ) ),
              std::istreambuf_iterator< char >() );

        kernel_file.close();

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
                }

                case CL_DEVICE_TYPE_GPU:
                {
                    std::cout << "GPU";
                }

                case CL_DEVICE_TYPE_ACCELERATOR:
                {
                    std::cout << "Accelerator";
                }

                case CL_DEVICE_TYPE_CUSTOM:
                {
                    std::cout << "Custom";
                }

                case CL_DEVICE_TYPE_DEFAULT:
                {
                    std::cout << " [DEFAULT]";
                    break;
                }

                default:
                {
                    std::cout << "Unknown";
                }
            };

            std::cout << "\r\n"
                      << "INFO  :: Max work-group size           : "
                      << device.getInfo< CL_DEVICE_MAX_WORK_GROUP_SIZE >()
                      << "\r\n"
                      << "INFO  :: Max num compute units         : "
                      << device.getInfo< CL_DEVICE_MAX_COMPUTE_UNITS >()
                      << "\r\n";

            size_t const device_max_compute_units =
                device.getInfo< CL_DEVICE_MAX_COMPUTE_UNITS >();

            /* ------------------------------------------------------------- */
            /* reset particle data: */
            /* ------------------------------------------------------------- */

            ::st_Buffer_clear( particles_buffer, true );

            size_t total_num_particles = size_t{ 0 };

            auto init_obj_it  = ::st_Buffer_get_const_objects_begin(
                init_particles_buffer );

            auto init_obj_end = ::st_Buffer_get_const_objects_end(
                init_particles_buffer );

            for( ; init_obj_it != init_obj_end ; ++init_obj_it )
            {
                particles_t const* orig = ( particles_t const* )( uintptr_t
                    )::st_Object_get_begin_addr( init_obj_it );

                particles_t* copied_particle = ::st_Particles_add_copy(
                    particles_buffer, orig );

                total_num_particles +=
                    ::st_Particles_get_num_of_particles( orig );

                ASSERT_TRUE( copied_particle != nullptr );

                ASSERT_TRUE( ::st_Particles_have_same_structure(
                    orig, copied_particle ) );

                ASSERT_TRUE( !::st_Particles_map_to_same_memory(
                    orig, copied_particle ) );

                ASSERT_TRUE( ::st_Particles_compare_values(
                    orig, copied_particle ) == 0 );
            }

            ASSERT_TRUE( total_num_particles > size_t{ 0 } );

            ASSERT_TRUE( ::st_Buffer_get_num_of_objects( particles_buffer ) ==
                ::st_Buffer_get_num_of_objects( init_particles_buffer ) );

            ASSERT_TRUE( ::st_Particles_buffers_have_same_structure(
                init_particles_buffer, particles_buffer ) );

            ASSERT_TRUE( 0 == ::st_Particles_buffer_compare_values(
                init_particles_buffer, particles_buffer ) );

            /* ------------------------------------------------------------- */

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

            cl::Buffer cl_particles( context, CL_MEM_READ_WRITE,
                                     particle_buffer_size );

            cl::Buffer cl_beam_elements( context, CL_MEM_READ_WRITE,
                                         beam_elements_buffer_size );

            cl::Buffer cl_success_flag( context, CL_MEM_READ_WRITE,
                                        sizeof( int64_t ) );

            try
            {
                cl_ret = queue.enqueueWriteBuffer( cl_particles, CL_TRUE, 0,
                    ::st_Buffer_get_size( particles_buffer ),
                    ::st_Buffer_get_const_data_begin( particles_buffer ) );
            }
            catch( cl::Error const& e )
            {
                std::cout << "enqueueWriteBuffer( particles_buffer ) :: "
                          << "line = " << __LINE__
                          << " :: ERROR : " << e.what() << std::endl
                          << e.err() << std::endl;

                cl_ret = CL_FALSE;
                throw;
            }

            ASSERT_TRUE( cl_ret == CL_SUCCESS );

            try
            {
                cl_ret = queue.enqueueWriteBuffer(
                    cl_beam_elements, CL_TRUE, 0, beam_elements_buffer_size,
                    ::st_Buffer_get_const_data_begin( beam_elements_buffer ) );
            }
            catch( cl::Error const& e )
            {
                std::cout << "enqueueWriteBuffer( beam_elements_buffer ) :: "
                          << "line = " << __LINE__
                          << " :: ERROR : " << e.what() << std::endl
                          << e.err() << std::endl;

                cl_ret = CL_FALSE;
                throw;
            }

            ASSERT_TRUE( cl_ret == CL_SUCCESS );

            int64_t success_flag = int64_t{ 0 };

            try
            {
                cl_ret = queue.enqueueWriteBuffer( cl_success_flag, CL_TRUE, 0,
                    sizeof( int64_t ), &success_flag );
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
                    program, "st_Remap_particles_beam_elements_buffers_opencl" );
            }
            catch( cl::Error const& e )
            {
                std::cout << "kernel remap_kernel :: line = " << __LINE__
                          << " :: ERROR : " << e.what() << std::endl
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

            remap_kernel.setArg( 0, cl_particles );
            remap_kernel.setArg( 1, cl_beam_elements );
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
                std::cout << "enqueueNDRangeKernel( remap_kernel) :: "
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

            if( success_flag != int64_t{ 0 } )
            {
                std::cout << "ERROR :: remap kernel success flag     : "
                          << success_flag
                          << std::endl;
            }

            ASSERT_TRUE( success_flag == int64_t{ 0 } );

            /* ============================================================= *
             * TRACKING KERNEL *
             * ============================================================= */

            cl::Kernel tracking_kernel;

            try
            {
                tracking_kernel = cl::Kernel(
                    program, "st_Track_particles_beam_elements_opencl" );
            }
            catch( cl::Error const& e )
            {
                std::cout << "kernel tracking_kernel :: "
                          << "line = " << __LINE__
                          << " :: ERROR : " << e.what() << std::endl
                          << e.err() << std::endl;
                cl_ret = CL_FALSE;
                throw;
            }

            ASSERT_TRUE( tracking_kernel.getInfo< CL_KERNEL_NUM_ARGS >() == 4u );

            size_t tracking_work_group_size = tracking_kernel.getWorkGroupInfo<
                CL_KERNEL_WORK_GROUP_SIZE >( device );

            size_t const tracking_work_group_size_prefered_multiple =
                tracking_kernel.getWorkGroupInfo<
                    CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE >( device );

            size_t tracking_num_threads =
                device_max_compute_units * tracking_work_group_size;

            std::cout << "INFO  :: tracking kernel wg size       : "
                      << tracking_work_group_size << "\r\n"
                      << "INFO  :: tracking kernel wg size multi : "
                      << tracking_work_group_size_prefered_multiple << "\r\n"
                      << "INFO  :: tracking kernel launch with   : "
                      << tracking_num_threads << " threads \r\n"
                      << "INFO  :: total num of particles        : "
                      << total_num_particles << "\r\n"
                      << std::endl;

            ASSERT_TRUE( tracking_work_group_size_prefered_multiple > size_t{ 0 } );
            ASSERT_TRUE( tracking_work_group_size > size_t{ 0 } );
            ASSERT_TRUE( tracking_num_threads > size_t{ 0 } );
            ASSERT_TRUE( ( tracking_num_threads % tracking_work_group_size ) ==
                         size_t{ 0 } );

            SIXTRL_UINT64_T num_turns = SIXTRL_UINT64_T{ 1 };

            tracking_kernel.setArg( 0, cl_particles );
            tracking_kernel.setArg( 1, cl_beam_elements );
            tracking_kernel.setArg( 2, num_turns );
            tracking_kernel.setArg( 3, cl_success_flag );

            try
            {
                cl_ret = queue.enqueueNDRangeKernel(
                    tracking_kernel, cl::NullRange,
                    cl::NDRange( tracking_num_threads ),
                    cl::NDRange( tracking_work_group_size  ) );
            }
            catch( cl::Error const& e )
            {
                std::cout << "enqueueNDRangeKernel( tracking_kernel ) :: "
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

            if( success_flag != int64_t{ 0 } )
            {
                std::cout << "ERROR :: tracking kernel success flag  : "
                          << success_flag
                          << std::endl;
            }

            ASSERT_TRUE( success_flag == int64_t{ 0 } );

            try
            {
                cl_ret = queue.enqueueReadBuffer( cl_particles, CL_TRUE, 0,
                    ::st_Buffer_get_size( particles_buffer ),
                    ::st_Buffer_get_data_begin( particles_buffer ) );
            }
            catch( cl::Error const& e )
            {
                std::cout << "enqueueReadBuffer( particles_buffer ) :: "
                          << "line = " << __LINE__
                          << " :: ERROR : " << e.what() << std::endl
                          << e.err() << std::endl;

                cl_ret = CL_FALSE;
                throw;
            }

            ASSERT_TRUE( cl_ret == CL_SUCCESS );

            success = ::st_Buffer_remap( particles_buffer );

            ASSERT_TRUE( success == 0 );

            ASSERT_TRUE( ::st_Buffer_get_num_of_objects( particles_buffer ) ==
                ::st_Buffer_get_num_of_objects( result_particles_buffer ) );

            ASSERT_TRUE( ::st_Particles_buffers_have_same_structure(
                result_particles_buffer, particles_buffer ) );

            /* ============================================================= */
            /* COMPARE TRACKING RESULTS WITH STORED RESULT DATA              */
            /* ============================================================= */

            double const COMPARE_TRESHOLD = double{ 1e-13 };

            int compare_result = ::st_Particles_buffer_compare_values(
                    result_particles_buffer, particles_buffer );

            if( 0 == compare_result )
            {
                std::cout << "INFO  :: "
                          << "tracking kernel reproduced the results exactly!!!"
                          << std::endl;
            }
            else
            {
                compare_result =
                    ::st_Particles_buffer_compare_values_with_treshold(
                        result_particles_buffer, particles_buffer,
                            COMPARE_TRESHOLD );

                if( compare_result == 0 )
                {
                    std::cout << "INFO  :: "
                          << "tracking kernel reproduced the results within "
                          << "a tolerance of " << COMPARE_TRESHOLD
                          << std::endl;
                }
            }

            ASSERT_TRUE( compare_result == 0 );
        }
    }
    else
    {
        std::cerr << "!!! --> WARNING :: Unable to perform unit-test as "
                  << "no valid OpenCL nodes have been found. -> skipping "
                  << "this unit-test <-- !!!" << std::endl;
    }

    ASSERT_TRUE( !devices.empty() );


    /* --------------------------------------------------------------------- */

    ::st_Buffer_delete( init_particles_buffer   );
    ::st_Buffer_delete( result_particles_buffer );
    ::st_Buffer_delete( beam_elements_buffer    );
}


/* end: tests/sixtracklib/opencl/test_be_drift_opencl_c99.cpp */
