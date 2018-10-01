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
#include "sixtracklib/common/particles.h"

/* ************************************************************************* */

TEST( C99_OpenCL_TrackParticlesTests, LHCReproduceSixTrackSingleTurnNoBeamBeam )
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
        std::ostringstream a2str( "" );
        std::string const PATH_TO_BASE_DIR = ::st_PATH_TO_BASE_DIR;

        a2str << " -D_GPUCODE=1"
              << " -D__NAMESPACE=st_"
              << " -DSIXTRL_BUFFER_ARGPTR_DEC=__private"
              << " -DSIXTRL_BUFFER_DATAPTR_DEC=__global"
              << " -I" << PATH_TO_BASE_DIR;

        std::string const COMPILE_OPTIONS = a2str.str();

        /* ----------------------------------------------------------------- */

        std::string const path_to_source( PATH_TO_BASE_DIR +
            std::string( "sixtracklib/opencl/kernels/track_particles_kernel.cl" ) );

        std::ifstream kernel_file( path_to_source, std::ios::in );

        std::string const PROGRAM_SOURCE_CODE(
            ( std::istreambuf_iterator< char >( kernel_file ) ),
              std::istreambuf_iterator< char >() );

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
                std::cerr
                      << "ERROR :: remap_program :: "
                      << "OpenCL Compilation Error -> Stopping Unit-Test \r\n"
                      << program.getBuildInfo< CL_PROGRAM_BUILD_LOG >( device )
                      << "\r\n"
                      << std::endl;

                cl_ret = CL_FALSE;
                throw;
            }

            ASSERT_TRUE( cl_ret == CL_SUCCESS );

            /* ------------------------------------------------------------- */

             cl::Kernel remap_kernel;

            try
            {
                remap_kernel =
                    cl::Kernel( program, "st_Remap_particles_beam_elements_buffers_opencl" );
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

            size_t remap_num_threads = remap_work_group_size_prefered_multiple;
            size_t remap_group_size  = remap_work_group_size_prefered_multiple;

            std::cout << "INFO  :: remap kernel wg size          : "
                      << remap_work_group_size << "\r\n"
                      << "INFO  :: remap kernel wg size multi    : "
                      << remap_work_group_size_prefered_multiple << "\r\n"
                      << "INFO  :: remap kernel launch with      : "
                      << remap_num_threads << " threads \r\n"
                      << "INFO  :: remap_kernel local size       : "
                      << remap_group_size << " threads \r\n"
                      << std::endl;

            ASSERT_TRUE( remap_work_group_size_prefered_multiple > size_t{ 0 } );
            ASSERT_TRUE( remap_work_group_size > size_t{ 0 } );
            ASSERT_TRUE( remap_group_size  > size_t{ 0 } );
            ASSERT_TRUE( remap_num_threads > size_t{ 0 } );
            ASSERT_TRUE( ( remap_num_threads % remap_group_size ) == size_t{ 0 } );

            /* ------------------------------------------------------------- */

            cl::Kernel tracking_kernel;

            try
            {
                tracking_kernel =
                    cl::Kernel( program, "st_Track_particles_beam_elements_opencl" );
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

            ASSERT_TRUE( tracking_kernel.getInfo< CL_KERNEL_NUM_ARGS >() == 4u );

            size_t tracking_work_group_size = tracking_kernel.getWorkGroupInfo<
                CL_KERNEL_WORK_GROUP_SIZE >( device );

            size_t const tracking_work_group_size_prefered_multiple =
                tracking_kernel.getWorkGroupInfo<
                    CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE >( device );

            ASSERT_TRUE( tracking_work_group_size_prefered_multiple > size_t{ 0 } );
            ASSERT_TRUE( tracking_work_group_size > size_t{ 0 } );

            std::cout << "INFO  :: tracking kernel wg size       : "
                      << tracking_work_group_size << "\r\n"
                      << "INFO  :: tracking kernel wg size multi : "
                      << tracking_work_group_size_prefered_multiple << "\r\n"
                      << std::endl;

            /* ------------------------------------------------------------- */

            object_t const* be_begin =
                ::st_Buffer_get_const_objects_begin( lhc_beam_elements_buffer );

            object_t const* pb_begin =
                ::st_Buffer_get_const_objects_begin( lhc_particles_buffer );

            object_t const* pb_end   =
                ::st_Buffer_get_const_objects_end( lhc_particles_buffer );

            object_t const* pb_it = pb_begin;

            particles_t const* in_particles =
                reinterpret_cast< particles_t const* >(
                    ::st_Object_get_const_begin_ptr( pb_it ) );

            object_t const* prev_pb = pb_it++;

            particles_t const* prev_in_particles = nullptr;

            num_particles_t num_particles = ::st_Particles_get_num_of_particles(
                in_particles );

            num_particles_t prev_num_particles = num_particles_t{ 0 };
            size_t cnt = size_t{ 0 };

            size_t prev_tracking_num_threads = size_t{ 0 };
            uint64_t const NUM_TURNS = uint64_t{ 1 };

            cl::Buffer cl_success_flag( context, CL_MEM_READ_WRITE,
                                        sizeof( int32_t ) );

            for( ; pb_it != pb_end ; ++pb_it, ++prev_pb, ++cnt )
            {
                prev_in_particles = in_particles;

                in_particles = reinterpret_cast< particles_t const* >(
                    ::st_Object_get_const_begin_ptr( pb_it ) );

                prev_num_particles = num_particles;
                num_particles = ::st_Particles_get_num_of_particles( in_particles );

                ASSERT_TRUE( num_particles == prev_num_particles );
                ASSERT_TRUE( in_particles != nullptr );

                ::st_Buffer_clear( particles_buffer, true );
                particles_t* particles = ::st_Particles_add_copy(
                    particles_buffer, prev_in_particles );

                ASSERT_TRUE( ::st_Buffer_get_num_of_objects( particles_buffer ) == 1u );
                ASSERT_TRUE( ::st_Buffer_get_size( particles_buffer ) > size_t{ 0 } );

                size_t const particles_buffer_size =
                    ::st_Buffer_get_size( particles_buffer );

                cl::Buffer cl_particles( context, CL_MEM_READ_WRITE,
                                         particles_buffer_size );

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

                ASSERT_TRUE( static_cast< std::ptrdiff_t >(
                    ::st_Buffer_get_num_of_objects( beam_elements_buffer ) ) ==
                    std::distance( line_begin, line_end ) );

                ASSERT_TRUE( ::st_Buffer_get_size( beam_elements_buffer )
                    > size_t{ 0 } );

                cl::Buffer cl_beam_elements( context, CL_MEM_READ_WRITE,
                    ::st_Buffer_get_size( beam_elements_buffer ) );

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
                    cl_ret = queue.enqueueWriteBuffer( cl_beam_elements, CL_TRUE, 0,
                    ::st_Buffer_get_size( beam_elements_buffer ),
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

                int32_t success_flag = int32_t{ 0 };

                try
                {
                    cl_ret = queue.enqueueWriteBuffer( cl_success_flag, CL_TRUE, 0,
                        sizeof( success_flag ), &success_flag );
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

                /* --------------------------------------------------------- */
                /* remap particles and beam elements buffers: */

                remap_kernel.setArg( 0, cl_particles );
                remap_kernel.setArg( 1, cl_beam_elements );
                remap_kernel.setArg( 2, cl_success_flag );

                try
                {
                    cl_ret = queue.enqueueNDRangeKernel( remap_kernel,
                        cl::NullRange, cl::NDRange( remap_num_threads ),
                        cl::NDRange( remap_group_size ) );
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

                /* --------------------------------------------------------- */
                /* read back success_flag: */

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
                    std::cout << "ERROR :: remap_kernel success flag  : "
                              << success_flag
                              << std::endl;
                }

                ASSERT_TRUE( success_flag == int32_t{ 0 } );

                /* --------------------------------------------------------- */
                /* track particles over line: */

                size_t const tracking_group_size =
                    tracking_work_group_size_prefered_multiple;

                size_t tracking_num_threads =
                    num_particles / tracking_work_group_size_prefered_multiple;

                tracking_num_threads *=
                    tracking_work_group_size_prefered_multiple;

                if( tracking_num_threads <
                        static_cast< size_t >( num_particles ) )
                {
                    tracking_num_threads +=
                        tracking_work_group_size_prefered_multiple;
                }

                ASSERT_TRUE( tracking_num_threads >=
                    static_cast< size_t >( num_particles ) );

                if( prev_tracking_num_threads != tracking_num_threads )
                {
                    std::cout << "INFO  :: num_particles                 : "
                              << num_particles << "\r\n"
                              << "INFO  :: tracking_num_threads          : "
                              << tracking_num_threads << "\r\n"
                              << "INFO  :: tracking_group_size           : "
                              << tracking_group_size << "\r\n"
                              << std::endl;

                    prev_tracking_num_threads = tracking_num_threads;
                }

                tracking_kernel.setArg( 0, cl_particles );
                tracking_kernel.setArg( 1, cl_beam_elements );
                tracking_kernel.setArg( 2, NUM_TURNS );
                tracking_kernel.setArg( 3, cl_success_flag );

                try
                {
                    cl_ret = queue.enqueueNDRangeKernel( tracking_kernel,
                        cl::NullRange, cl::NDRange( tracking_num_threads ),
                        cl::NDRange( tracking_group_size ) );
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

                ASSERT_TRUE( cl_ret == CL_SUCCESS );

                /* --------------------------------------------------------- */
                /* read back success_flag: */

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
                    std::cout << "ERROR :: tracking_kernel success flag  : "
                              << success_flag
                              << std::endl;
                }

                ASSERT_TRUE( success_flag == int32_t{ 0 } );

                /* --------------------------------------------------------- */
                /* read back the tracked particles and compare them to the   *
                 * expected data from the data-file */

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

                int success = ::st_Buffer_remap( particles_buffer );
                ASSERT_TRUE( success == 0 );

                particles_t const* cmp_particles = in_particles;

                particles = reinterpret_cast< particles_t* >(
                    ::st_Object_get_begin_ptr( ::st_Buffer_get_objects_begin(
                        particles_buffer ) ) );

                ::st_Buffer_clear( diff_particles_buffer, true );

                particles_t* diff_particles = ::st_Particles_new(
                    diff_particles_buffer, num_particles );

                ASSERT_TRUE( diff_particles != nullptr );

                ::st_Particles_calculate_difference(
                    cmp_particles, particles, diff_particles );

                bool is_equal = true;

                for( num_particles_t ii = 0 ; ii < num_particles ; ++ii )
                {
                    if( ( ABS_TOLERANCE < std::fabs( ::st_Particles_get_s_value( diff_particles, ii ) ) ) ||
                        ( ABS_TOLERANCE < std::fabs( ::st_Particles_get_x_value( diff_particles, ii ) ) ) ||
                        ( ABS_TOLERANCE < std::fabs( ::st_Particles_get_y_value( diff_particles, ii ) ) ) ||
                        ( ABS_TOLERANCE < std::fabs( ::st_Particles_get_px_value( diff_particles, ii ) ) ) ||
                        ( ABS_TOLERANCE < std::fabs( ::st_Particles_get_py_value( diff_particles, ii ) ) ) ||
                        ( ABS_TOLERANCE < std::fabs( ::st_Particles_get_zeta_value( diff_particles, ii ) ) ) ||
                        ( ABS_TOLERANCE < std::fabs( ::st_Particles_get_psigma_value( diff_particles, ii ) ) ) ||
                        ( ABS_TOLERANCE < std::fabs( ::st_Particles_get_delta_value( diff_particles, ii ) ) ) ||
                        ( ABS_TOLERANCE < std::fabs( ::st_Particles_get_rpp_value( diff_particles, ii ) ) ) ||
                        ( ABS_TOLERANCE < std::fabs( ::st_Particles_get_rvv_value( diff_particles, ii ) ) ) ||
                        ( ABS_TOLERANCE < std::fabs( ::st_Particles_get_chi_value( diff_particles, ii ) ) ) )
                    {
                        is_equal = false;
                        break;
                    }
                }

                if( !is_equal )
                {
                    std::cout << "Difference between tracked particles and "
                                "reference particle data detected: \r\n"
                            << "at beam-element block #" << cnt
                            << ", concerning beam-elements [ "
                            << std::setw( 6 ) << begin_elem_id + 1 << " - "
                            << std::setw( 6 ) << end_elem_id + 1 << " ):\r\n"
                            << "absolute tolerance : " << ABS_TOLERANCE << "\r\n"
                            << "\r\n"
                            << "beam-elements: \r\n";

                    object_t const* line_it  = line_begin;
                    size_t jj = begin_elem_id + index_t{ 1 };

                    for( ; line_it != line_end ; ++line_it )
                    {
                        std::cout << "be id = " << std::setw( 6 ) << jj ;
                        ::st_BeamElement_print( line_it );
                    }

                    std::cout << "\r\n"
                            << "diff_particles = |cmp_particles - particles| :\r\n";

                    ::st_Particles_print( stdout, diff_particles );

                    std::cout << std::endl;
                }

                for( num_particles_t ii = 0 ; ii < num_particles ; ++ii )
                {
                    ASSERT_TRUE( ABS_TOLERANCE > std::fabs(
                        ::st_Particles_get_s_value( diff_particles, ii ) ) );

                    ASSERT_TRUE( ABS_TOLERANCE > std::fabs(
                        ::st_Particles_get_x_value( diff_particles, ii ) ) );

                    ASSERT_TRUE( ABS_TOLERANCE > std::fabs(
                        ::st_Particles_get_y_value( diff_particles, ii ) ) );

                    ASSERT_TRUE( ABS_TOLERANCE > std::fabs(
                        ::st_Particles_get_px_value( diff_particles, ii ) ) );

                    ASSERT_TRUE( ABS_TOLERANCE > std::fabs(
                        ::st_Particles_get_py_value( diff_particles, ii ) ) );

                    ASSERT_TRUE( ABS_TOLERANCE > std::fabs(
                        ::st_Particles_get_zeta_value( diff_particles, ii ) ) );

                    ASSERT_TRUE( ABS_TOLERANCE > std::fabs(
                        ::st_Particles_get_psigma_value( diff_particles, ii ) ) );

                    ASSERT_TRUE( ABS_TOLERANCE > std::fabs(
                        ::st_Particles_get_delta_value( diff_particles, ii ) ) );

                    ASSERT_TRUE( ABS_TOLERANCE > std::fabs(
                        ::st_Particles_get_rpp_value( diff_particles, ii ) ) );

                    ASSERT_TRUE( ABS_TOLERANCE > std::fabs(
                        ::st_Particles_get_rvv_value( diff_particles, ii ) ) );

                    ASSERT_TRUE( ABS_TOLERANCE > std::fabs(
                        ::st_Particles_get_chi_value( diff_particles, ii ) ) );

                    ASSERT_TRUE( ::st_Particles_get_particle_id_value(
                        diff_particles, ii ) == index_t{ 0 } );
                }
            }
        }
    }

    /* ===================================================================== */

    ::st_Buffer_delete( lhc_particles_buffer );
    ::st_Buffer_delete( lhc_beam_elements_buffer );

    ::st_Buffer_delete( particles_buffer );
    ::st_Buffer_delete( diff_particles_buffer );
    ::st_Buffer_delete( beam_elements_buffer );
}

/* ************************************************************************* */

TEST( C99_OpenCL_TrackParticlesTests,
      LHCReproduceSixTrackSingleTurnNoBeamBeamPrivParticlesOptimized )
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
        std::ostringstream a2str( "" );
        std::string const PATH_TO_BASE_DIR = ::st_PATH_TO_BASE_DIR;

        a2str << " -D_GPUCODE=1"
              << " -D__NAMESPACE=st_"
              << " -DSIXTRL_BUFFER_ARGPTR_DEC=__private"
              << " -DSIXTRL_BUFFER_DATAPTR_DEC=__global"
              << " -I" << PATH_TO_BASE_DIR;

        std::string const REMAP_COMPILE_OPTIONS = a2str.str();

        /* ----------------------------------------------------------------- */

        std::string path_to_source = PATH_TO_BASE_DIR;
        path_to_source += "sixtracklib/opencl/kernels/track_particles_kernel.cl";

        std::ifstream kernel_file( path_to_source, std::ios::in );

        std::string const REMAP_PROGRAM_SOURCE_CODE(
            ( std::istreambuf_iterator< char >( kernel_file ) ),
              std::istreambuf_iterator< char >() );

        kernel_file.close();

        path_to_source  = PATH_TO_BASE_DIR;
        path_to_source += "sixtracklib/opencl/kernels/";
        path_to_source += "track_particles_priv_particles_optimized_kernel.cl";

        kernel_file.open( path_to_source, std::ios::in );

        std::string const TRACKING_PRORGRAM_SOURCE_CODE(
            ( std::istreambuf_iterator< char >( kernel_file ) ),
              std::istreambuf_iterator< char >() );

        a2str.str( "" );

        a2str << " -D_GPUCODE=1"
              << " -D__NAMESPACE=st_"
              << " -DSIXTRL_BUFFER_ARGPTR_DEC=__private"
              << " -DSIXTRL_BUFFER_DATAPTR_DEC=__global"
              << " -DSIXTRL_PARTICLE_ARGPTR_DEC=__private"
              << " -DSIXTRL_PARTICLE_DATAPTR_DEC=__private"
              << " -I" << PATH_TO_BASE_DIR;

        std::string const TRACKING_COMPILE_OPTIONS = a2str.str();

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

            cl_int cl_ret = CL_SUCCESS;

            cl::Context context( device );
            cl::CommandQueue queue( context, device, CL_QUEUE_PROFILING_ENABLE );
            cl::Program remap_program( context, REMAP_PROGRAM_SOURCE_CODE );
            cl::Program tracking_program( context, TRACKING_PRORGRAM_SOURCE_CODE );

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
                cl_ret = tracking_program.build( TRACKING_COMPILE_OPTIONS.c_str() );
            }
            catch( cl::Error const& e )
            {
                std::cerr
                      << "ERROR :: tracking_program :: "
                      << "OpenCL Compilation Error -> Stopping Unit-Test \r\n"
                      << tracking_program.getBuildInfo< CL_PROGRAM_BUILD_LOG >( device )
                      << "\r\n"
                      << std::endl;

                cl_ret = CL_FALSE;
                throw;
            }

            ASSERT_TRUE( cl_ret == CL_SUCCESS );

            /* ------------------------------------------------------------- */

             cl::Kernel remap_kernel;

            try
            {
                remap_kernel =
                    cl::Kernel( remap_program, "st_Remap_particles_beam_elements_buffers_opencl" );
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

            size_t remap_num_threads = remap_work_group_size_prefered_multiple;
            size_t remap_group_size  = remap_work_group_size_prefered_multiple;

            std::cout << "INFO  :: remap kernel wg size          : "
                      << remap_work_group_size << "\r\n"
                      << "INFO  :: remap kernel wg size multi    : "
                      << remap_work_group_size_prefered_multiple << "\r\n"
                      << "INFO  :: remap kernel launch with      : "
                      << remap_num_threads << " threads \r\n"
                      << "INFO  :: remap_kernel local size       : "
                      << remap_group_size << " threads \r\n"
                      << std::endl;

            ASSERT_TRUE( remap_work_group_size_prefered_multiple > size_t{ 0 } );
            ASSERT_TRUE( remap_work_group_size > size_t{ 0 } );
            ASSERT_TRUE( remap_group_size  > size_t{ 0 } );
            ASSERT_TRUE( remap_num_threads > size_t{ 0 } );
            ASSERT_TRUE( ( remap_num_threads % remap_group_size ) == size_t{ 0 } );

            /* ------------------------------------------------------------- */

            cl::Kernel tracking_kernel;

            try
            {
                tracking_kernel = cl::Kernel( tracking_program,
                    "st_Track_particles_beam_elements_priv_particles_optimized_opencl" );
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

            ASSERT_TRUE( tracking_kernel.getInfo< CL_KERNEL_NUM_ARGS >() == 4u );

            size_t tracking_work_group_size = tracking_kernel.getWorkGroupInfo<
                CL_KERNEL_WORK_GROUP_SIZE >( device );

            size_t const tracking_work_group_size_prefered_multiple =
                tracking_kernel.getWorkGroupInfo<
                    CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE >( device );

            ASSERT_TRUE( tracking_work_group_size_prefered_multiple > size_t{ 0 } );
            ASSERT_TRUE( tracking_work_group_size > size_t{ 0 } );

            std::cout << "INFO  :: tracking kernel wg size       : "
                      << tracking_work_group_size << "\r\n"
                      << "INFO  :: tracking kernel wg size multi : "
                      << tracking_work_group_size_prefered_multiple << "\r\n"
                      << std::endl;

            /* ------------------------------------------------------------- */

            object_t const* be_begin =
                ::st_Buffer_get_const_objects_begin( lhc_beam_elements_buffer );

            object_t const* pb_begin =
                ::st_Buffer_get_const_objects_begin( lhc_particles_buffer );

            object_t const* pb_end   =
                ::st_Buffer_get_const_objects_end( lhc_particles_buffer );

            object_t const* pb_it = pb_begin;

            particles_t const* in_particles =
                reinterpret_cast< particles_t const* >(
                    ::st_Object_get_const_begin_ptr( pb_it ) );

            object_t const* prev_pb = pb_it++;

            particles_t const* prev_in_particles = nullptr;

            num_particles_t num_particles = ::st_Particles_get_num_of_particles(
                in_particles );

            num_particles_t prev_num_particles = num_particles_t{ 0 };
            size_t cnt = size_t{ 0 };

            size_t prev_tracking_num_threads = size_t{ 0 };
            uint64_t const NUM_TURNS = uint64_t{ 1 };

            cl::Buffer cl_success_flag( context, CL_MEM_READ_WRITE,
                                        sizeof( int32_t ) );

            for( ; pb_it != pb_end ; ++pb_it, ++prev_pb, ++cnt )
            {
                prev_in_particles = in_particles;

                in_particles = reinterpret_cast< particles_t const* >(
                    ::st_Object_get_const_begin_ptr( pb_it ) );

                prev_num_particles = num_particles;
                num_particles = ::st_Particles_get_num_of_particles( in_particles );

                ASSERT_TRUE( num_particles == prev_num_particles );
                ASSERT_TRUE( in_particles != nullptr );

                ::st_Buffer_clear( particles_buffer, true );
                particles_t* particles = ::st_Particles_add_copy(
                    particles_buffer, prev_in_particles );

                ASSERT_TRUE( ::st_Buffer_get_num_of_objects( particles_buffer ) == 1u );
                ASSERT_TRUE( ::st_Buffer_get_size( particles_buffer ) > size_t{ 0 } );

                size_t const particles_buffer_size =
                    ::st_Buffer_get_size( particles_buffer );

                cl::Buffer cl_particles( context, CL_MEM_READ_WRITE,
                                         particles_buffer_size );

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

                ASSERT_TRUE( static_cast< std::ptrdiff_t >(
                    ::st_Buffer_get_num_of_objects( beam_elements_buffer ) ) ==
                    std::distance( line_begin, line_end ) );

                ASSERT_TRUE( ::st_Buffer_get_size( beam_elements_buffer )
                    > size_t{ 0 } );

                cl::Buffer cl_beam_elements( context, CL_MEM_READ_WRITE,
                    ::st_Buffer_get_size( beam_elements_buffer ) );

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
                    cl_ret = queue.enqueueWriteBuffer( cl_beam_elements, CL_TRUE, 0,
                    ::st_Buffer_get_size( beam_elements_buffer ),
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

                int32_t success_flag = int32_t{ 0 };

                try
                {
                    cl_ret = queue.enqueueWriteBuffer( cl_success_flag, CL_TRUE, 0,
                        sizeof( success_flag ), &success_flag );
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

                /* --------------------------------------------------------- */
                /* remap particles and beam elements buffers: */

                remap_kernel.setArg( 0, cl_particles );
                remap_kernel.setArg( 1, cl_beam_elements );
                remap_kernel.setArg( 2, cl_success_flag );

                try
                {
                    cl_ret = queue.enqueueNDRangeKernel( remap_kernel,
                        cl::NullRange, cl::NDRange( remap_num_threads ),
                        cl::NDRange( remap_group_size ) );
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

                /* --------------------------------------------------------- */
                /* read back success_flag: */

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
                    std::cout << "ERROR :: remap_kernel success flag  : "
                              << success_flag
                              << std::endl;
                }

                ASSERT_TRUE( success_flag == int32_t{ 0 } );

                /* --------------------------------------------------------- */
                /* track particles over line: */

                size_t const tracking_group_size =
                    tracking_work_group_size_prefered_multiple;

                size_t tracking_num_threads =
                    num_particles / tracking_work_group_size_prefered_multiple;

                tracking_num_threads *=
                    tracking_work_group_size_prefered_multiple;

                if( tracking_num_threads <
                        static_cast< size_t >( num_particles ) )
                {
                    tracking_num_threads +=
                        tracking_work_group_size_prefered_multiple;
                }

                ASSERT_TRUE( tracking_num_threads >=
                    static_cast< size_t >( num_particles ) );

                if( prev_tracking_num_threads != tracking_num_threads )
                {
                    std::cout << "INFO  :: num_particles                 : "
                              << num_particles << "\r\n"
                              << "INFO  :: tracking_num_threads          : "
                              << tracking_num_threads << "\r\n"
                              << "INFO  :: tracking_group_size           : "
                              << tracking_group_size << "\r\n"
                              << std::endl;

                    prev_tracking_num_threads = tracking_num_threads;
                }

                tracking_kernel.setArg( 0, cl_particles );
                tracking_kernel.setArg( 1, cl_beam_elements );
                tracking_kernel.setArg( 2, NUM_TURNS );
                tracking_kernel.setArg( 3, cl_success_flag );

                try
                {
                    cl_ret = queue.enqueueNDRangeKernel( tracking_kernel,
                        cl::NullRange, cl::NDRange( tracking_num_threads ),
                        cl::NDRange( tracking_group_size ) );
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

                ASSERT_TRUE( cl_ret == CL_SUCCESS );

                /* --------------------------------------------------------- */
                /* read back success_flag: */

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
                    std::cout << "ERROR :: tracking_kernel success flag  : "
                              << success_flag
                              << std::endl;
                }

                ASSERT_TRUE( success_flag == int32_t{ 0 } );

                /* --------------------------------------------------------- */
                /* read back the tracked particles and compare them to the   *
                 * expected data from the data-file */

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

                int success = ::st_Buffer_remap( particles_buffer );
                ASSERT_TRUE( success == 0 );

                particles_t const* cmp_particles = in_particles;

                particles = reinterpret_cast< particles_t* >(
                    ::st_Object_get_begin_ptr( ::st_Buffer_get_objects_begin(
                        particles_buffer ) ) );

                ::st_Buffer_clear( diff_particles_buffer, true );

                particles_t* diff_particles = ::st_Particles_new(
                    diff_particles_buffer, num_particles );

                ASSERT_TRUE( diff_particles != nullptr );

                ::st_Particles_calculate_difference(
                    cmp_particles, particles, diff_particles );

                bool is_equal = true;

                for( num_particles_t ii = 0 ; ii < num_particles ; ++ii )
                {
                    if( ( ABS_TOLERANCE < std::fabs( ::st_Particles_get_s_value( diff_particles, ii ) ) ) ||
                        ( ABS_TOLERANCE < std::fabs( ::st_Particles_get_x_value( diff_particles, ii ) ) ) ||
                        ( ABS_TOLERANCE < std::fabs( ::st_Particles_get_y_value( diff_particles, ii ) ) ) ||
                        ( ABS_TOLERANCE < std::fabs( ::st_Particles_get_px_value( diff_particles, ii ) ) ) ||
                        ( ABS_TOLERANCE < std::fabs( ::st_Particles_get_py_value( diff_particles, ii ) ) ) ||
                        ( ABS_TOLERANCE < std::fabs( ::st_Particles_get_zeta_value( diff_particles, ii ) ) ) ||
                        ( ABS_TOLERANCE < std::fabs( ::st_Particles_get_psigma_value( diff_particles, ii ) ) ) ||
                        ( ABS_TOLERANCE < std::fabs( ::st_Particles_get_delta_value( diff_particles, ii ) ) ) ||
                        ( ABS_TOLERANCE < std::fabs( ::st_Particles_get_rpp_value( diff_particles, ii ) ) ) ||
                        ( ABS_TOLERANCE < std::fabs( ::st_Particles_get_rvv_value( diff_particles, ii ) ) ) ||
                        ( ABS_TOLERANCE < std::fabs( ::st_Particles_get_chi_value( diff_particles, ii ) ) ) )
                    {
                        is_equal = false;
                        break;
                    }
                }

                if( !is_equal )
                {
                    std::cout << "Difference between tracked particles and "
                                "reference particle data detected: \r\n"
                            << "at beam-element block #" << cnt
                            << ", concerning beam-elements [ "
                            << std::setw( 6 ) << begin_elem_id + 1 << " - "
                            << std::setw( 6 ) << end_elem_id + 1 << " ):\r\n"
                            << "absolute tolerance : " << ABS_TOLERANCE << "\r\n"
                            << "\r\n"
                            << "beam-elements: \r\n";

                    object_t const* line_it  = line_begin;
                    size_t jj = begin_elem_id + index_t{ 1 };

                    for( ; line_it != line_end ; ++line_it )
                    {
                        std::cout << "be id = " << std::setw( 6 ) << jj ;
                        ::st_BeamElement_print( line_it );
                    }

                    std::cout << "\r\n"
                            << "diff_particles = |cmp_particles - particles| :\r\n";

                    ::st_Particles_print( stdout, diff_particles );

                    std::cout << std::endl;
                }

                for( num_particles_t ii = 0 ; ii < num_particles ; ++ii )
                {
                    ASSERT_TRUE( ABS_TOLERANCE > std::fabs(
                        ::st_Particles_get_s_value( diff_particles, ii ) ) );

                    ASSERT_TRUE( ABS_TOLERANCE > std::fabs(
                        ::st_Particles_get_x_value( diff_particles, ii ) ) );

                    ASSERT_TRUE( ABS_TOLERANCE > std::fabs(
                        ::st_Particles_get_y_value( diff_particles, ii ) ) );

                    ASSERT_TRUE( ABS_TOLERANCE > std::fabs(
                        ::st_Particles_get_px_value( diff_particles, ii ) ) );

                    ASSERT_TRUE( ABS_TOLERANCE > std::fabs(
                        ::st_Particles_get_py_value( diff_particles, ii ) ) );

                    ASSERT_TRUE( ABS_TOLERANCE > std::fabs(
                        ::st_Particles_get_zeta_value( diff_particles, ii ) ) );

                    ASSERT_TRUE( ABS_TOLERANCE > std::fabs(
                        ::st_Particles_get_psigma_value( diff_particles, ii ) ) );

                    ASSERT_TRUE( ABS_TOLERANCE > std::fabs(
                        ::st_Particles_get_delta_value( diff_particles, ii ) ) );

                    ASSERT_TRUE( ABS_TOLERANCE > std::fabs(
                        ::st_Particles_get_rpp_value( diff_particles, ii ) ) );

                    ASSERT_TRUE( ABS_TOLERANCE > std::fabs(
                        ::st_Particles_get_rvv_value( diff_particles, ii ) ) );

                    ASSERT_TRUE( ABS_TOLERANCE > std::fabs(
                        ::st_Particles_get_chi_value( diff_particles, ii ) ) );

                    ASSERT_TRUE( ::st_Particles_get_particle_id_value(
                        diff_particles, ii ) == index_t{ 0 } );
                }
            }
        }
    }

    /* ===================================================================== */

    ::st_Buffer_delete( lhc_particles_buffer );
    ::st_Buffer_delete( lhc_beam_elements_buffer );

    ::st_Buffer_delete( particles_buffer );
    ::st_Buffer_delete( diff_particles_buffer );
    ::st_Buffer_delete( beam_elements_buffer );
}

/* end: tests/sixtracklib/opencl/test_track_opencl_c99.cpp */
