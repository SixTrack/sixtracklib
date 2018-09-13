#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>
#include <vector>

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#include "sixtracklib/sixtracklib.h"
#include "sixtracklib/testlib.h"

namespace sixtrack
{
    namespace benchmarks
    {
        struct TimingResult
        {
            explicit TimingResult(
                std::size_t id = std::size_t{ 0 },
                double const t = double{ 0.0 } ) :
                m_time( t ), m_run_id( id )
            {

            }

            TimingResult( TimingResult const& orig ) = default;
            TimingResult( TimingResult&& orig ) = default;

            TimingResult& operator=( TimingResult const& rhs ) = default;
            TimingResult& operator=( TimingResult&& rhs ) = default;

            ~TimingResult() = default;

            double       m_time;
            std::size_t  m_run_id;
        };
    }
}

int main()
{
    using size_t                = ::st_buffer_size_t;
    using num_particles_t       = ::st_particle_num_elements_t;
    using buffer_t              = ::st_Buffer;
    using object_t              = ::st_Object;
    using particles_t           = ::st_Particles;

    using timing_result_t       = sixtrack::benchmarks::TimingResult;
    size_t const NUM_TURNS      = size_t{ 20u };

    /* ===================================================================== */
    /* ==== Prepare Host Buffers                                             */

    double begin_time = ::st_Time_get_seconds_since_epoch();

    std::vector< size_t > num_particles_list =
    {
        20000u
    };

    std::sort( num_particles_list.begin(), num_particles_list.end() );

    /* --------------------------------------------------------------------- */

    buffer_t* lhc_beam_elements = ::st_Buffer_new_from_file(
        ::st_PATH_TO_TEST_LHC_BEAM_ELEMENTS_DATA_NO_BEAM_BEAM );

    object_t const* be_begin =
        ::st_Buffer_get_const_objects_begin( lhc_beam_elements );

    object_t const* be_end   =
        ::st_Buffer_get_const_objects_end( lhc_beam_elements );

    /* --------------------------------------------------------------------- */

    buffer_t* lhc_particles_buffer = ::st_Buffer_new_from_file(
        st_PATH_TO_TEST_LHC_PARTICLES_DATA_T1_P2_NO_BEAM_BEAM );

    particles_t const* lhc_particles = ( particles_t const* )( uintptr_t
        )::st_Object_get_begin_addr( ::st_Buffer_get_const_objects_begin(
            lhc_particles_buffer ) );

    size_t const lhc_num_particles =
        ::st_Particles_get_num_of_particles( lhc_particles );

    /* --------------------------------------------------------------------- */

    size_t const max_num_particles = num_particles_list.back();

    size_t const requ_num_slots = ::st_Particles_get_required_num_slots(
        lhc_particles_buffer, max_num_particles );

    size_t const requ_num_dataptrs = ::st_Particles_get_required_num_dataptrs(
        lhc_particles_buffer, max_num_particles );

    size_t const req_particles_buffer_size =
        ::st_Buffer_calculate_required_buffer_length( lhc_particles_buffer,
            max_num_particles, requ_num_slots, requ_num_dataptrs, size_t{ 0 } );

    buffer_t* particles_buffer = ::st_Buffer_new( req_particles_buffer_size );

    /* --------------------------------------------------------------------- */

    double now = ::st_Time_get_seconds_since_epoch();

    double const time_setup_host_buffers =
        ( now >= begin_time ) ? ( now - begin_time ) : double{ 0.0 };

    /* ===================================================================== */
    /* ==== Prepare OpenCL Environment Buffers                                             */

    begin_time = ::st_Time_get_seconds_since_epoch();

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

    now =  ::st_Time_get_seconds_since_epoch();

    double const time_get_platforms =
        ( now >= begin_time ) ? now - begin_time : double{ 0 };

    begin_time = ::st_Time_get_seconds_since_epoch();

    if( !devices.empty() )
    {
        std::ostringstream a2str( "" );
        std::string const PATH_TO_BASE_DIR = ::st_PATH_TO_BASE_DIR;

        a2str << " -D_GPUCODE=1"
              << " -D__NAMESPACE=st_"
              << " -DSIXTRL_DATAPTR_DEC=__global"
              << " -DSIXTRL_BUFFER_DATAPTR_DEC=__global"
              << " -DSIXTRL_BUFFER_OBJ_ARGPTR_DEC=__global"
              << " -DISXTRL_BUFFER_OBJ_DATAPTR_DEC=__global"
              << " -DSIXTRL_PARTICLE_ARGPTR_DEC=__global"
              << " -DSIXTRL_PARTICLE_DATAPTR_DEC=__global"
              << " -DSIXTRL_BE_ARGPTR_DEC=__global"
              << " -DSIXTRL_BE_DATAPTR_DEC=__global"
              << " -w"
              << " -Werror"
              << " -I" << PATH_TO_BASE_DIR;

        /* ----------------------------------------------------------------- */

        std::string path_to_source = PATH_TO_BASE_DIR;
        path_to_source += "sixtracklib/opencl/impl/track_particles_kernel.cl";

        std::ifstream kernel_file( path_to_source, std::ios::in );

        std::string const PROGRAM_SOURCE_CODE(
            ( std::istreambuf_iterator< char >( kernel_file ) ),
              std::istreambuf_iterator< char >() );

        kernel_file.close();

        std::string const COMPILER_OPTIONS = a2str.str();

        /* ----------------------------------------------------------------- */

        now = ::st_Time_get_seconds_since_epoch();

        double const time_prepare_source_code = ( now >= begin_time )
            ? ( now - begin_time ) : double{ 0 };

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

            begin_time = ::st_Time_get_seconds_since_epoch();

            size_t const device_max_compute_units =
                device.getInfo< CL_DEVICE_MAX_COMPUTE_UNITS >();

            std::cout << "\r\n"
                      << "INFO  :: Max work-group size           : "
                      << device.getInfo< CL_DEVICE_MAX_WORK_GROUP_SIZE >()
                      << "\r\n"
                      << "INFO  :: Max num compute units         : "
                      << device_max_compute_units << "\r\n";

            cl_int cl_ret = CL_SUCCESS;

            cl::Context context( device );
            cl::CommandQueue queue( context, device, CL_QUEUE_PROFILING_ENABLE );
            cl::Program program( context, PROGRAM_SOURCE_CODE );

            try
            {
                cl_ret = program.build( COMPILER_OPTIONS.c_str() );
            }
            catch( cl::Error const& e )
            {
                std::cerr
                      << "ERROR :: program :: "
                      << "OpenCL Compilation Error -> Stopping Unit-Test \r\n"
                      << program.getBuildInfo< CL_PROGRAM_BUILD_LOG >( device )
                      << "\r\n"
                      << std::endl;

                cl_ret = CL_FALSE;
                throw;
            }

            cl::Kernel remapping_kernel;

            try
            {
                remapping_kernel = cl::Kernel( program,
                    "st_Remap_particles_beam_elements_buffers_opencl" );
            }
            catch( cl::Error const& e )
            {
                std::cout << "kernel remapping_kernel :: "
                          << "line = " << __LINE__
                          << " :: ERROR : " << e.what() << std::endl
                          << e.err() << std::endl;
                cl_ret = CL_FALSE;
                throw;
            }

            size_t remap_work_group_size = remapping_kernel.getWorkGroupInfo<
                CL_KERNEL_WORK_GROUP_SIZE >( device );

            size_t const remap_work_group_size_prefered_multiple =
                remapping_kernel.getWorkGroupInfo<
                    CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE >( device );

            size_t remap_num_threads = remap_work_group_size_prefered_multiple;
            size_t remap_group_size  = remap_work_group_size_prefered_multiple;

            /* ------------------------------------------------------------- */

            cl::Kernel tracking_kernel;

            try
            {
                tracking_kernel = cl::Kernel( program,
                    "st_Track_particles_beam_elements_opencl" );
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

            size_t track_work_group_size = tracking_kernel.getWorkGroupInfo<
                CL_KERNEL_WORK_GROUP_SIZE >( device );

            size_t const track_work_group_size_prefered_multiple =
                tracking_kernel.getWorkGroupInfo<
                    CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE >( device );

            now = ::st_Time_get_seconds_since_epoch();

            double const time_cl_program_compile = ( now >= begin_time )
                ? ( now - begin_time ) : double{ 0.0 };

            /* ============================================================= */

            for( auto const NUM_PARTICLES : num_particles_list )
            {
                size_t tracking_num_threads = size_t{ 0 };
                size_t tracking_group_size  = track_work_group_size;

                tracking_num_threads  = NUM_PARTICLES / track_work_group_size;
                tracking_num_threads *= track_work_group_size;

                if( tracking_num_threads < NUM_PARTICLES )
                {
                    tracking_num_threads += track_work_group_size;
                }

                std::cout << "INFO  :: num_particles                 : "
                          << NUM_PARTICLES << "\r\n"
                          << "INFO  :: remap kernel wg size          : "
                          << remap_work_group_size << "\r\n"
                          << "INFO  :: remap kernel wg size multi    : "
                          << remap_work_group_size_prefered_multiple << "\r\n"
                          << "INFO  :: remap kernel launch with      : "
                          << remap_num_threads << " threads \r\n"
                          << "INFO  :: remap_kernel local size       : "
                          << remap_group_size << " threads \r\n\r\n"
                          << "INFO  :: num_turns                     : "
                          << NUM_TURNS     << "\r\n"
                          << "INFO  :: tracking kernel wg size       : "
                          << track_work_group_size << "\r\n"
                          << "INFO  :: tracking kernel wg size multi : "
                          << track_work_group_size_prefered_multiple << "\r\n"
                          << "INFO  :: tracking kernel launch with   : "
                          << tracking_num_threads << " threads\r\n"
                          << "INFO  :: tracking kernel local size    : "
                          << tracking_group_size  << " threads\r\n"
                          << std::endl;

                begin_time = ::st_Time_get_seconds_since_epoch();

                int success = ::st_Buffer_reset( particles_buffer );
                SIXTRL_ASSERT( success == 0 );

                particles_t* particles = ::st_Particles_new(
                    particles_buffer, NUM_PARTICLES );

                for( size_t ii = size_t{ 0 } ; ii < NUM_PARTICLES ; ++ii )
                {
                    size_t jj = ii % lhc_num_particles;
                    ::st_Particles_copy_single( particles, ii, lhc_particles, jj );
                }

                now = ::st_Time_get_seconds_since_epoch();

                double const time_setup_particle_buffer = ( now >= begin_time )
                    ? ( now - begin_time ) : double{ 0.0 };

                /* ========================================================= */

                std::vector< cl::Event > write_xfer_events( 3u, cl::Event{} );

                cl_ulong write_xfer_when_queued[]    = { 0, 0, 0 };
                cl_ulong write_xfer_when_submitted[] = { 0, 0, 0 };
                cl_ulong write_xfer_when_started[]   = { 0, 0, 0 };
                cl_ulong write_xfer_when_ended[]     = { 0, 0, 0 };

                begin_time = ::st_Time_get_seconds_since_epoch();

                int32_t success_flag = int32_t{ 0 };

                cl::Buffer cl_particles( context, CL_MEM_READ_WRITE,
                    ::st_Buffer_get_size( lhc_particles_buffer ) );

                cl::Buffer cl_beam_elements( context, CL_MEM_READ_WRITE,
                    ::st_Buffer_get_size( lhc_beam_elements ) );

                cl::Buffer cl_success_flag( context, CL_MEM_READ_WRITE,
                    sizeof( success_flag ) );

                try
                {
                    cl_ret = queue.enqueueWriteBuffer( cl_particles, CL_TRUE, 0,
                        ::st_Buffer_get_size( particles_buffer ),
                        ::st_Buffer_get_const_data_begin( particles_buffer ),
                        nullptr, &write_xfer_events[ 0 ] );

                    cl_ret |= write_xfer_events[ 0 ].getProfilingInfo< cl_ulong >(
                        CL_PROFILING_COMMAND_QUEUED, &write_xfer_when_queued[ 0 ] );

                    cl_ret |= write_xfer_events[ 0 ].getProfilingInfo< cl_ulong >(
                        CL_PROFILING_COMMAND_SUBMIT, &write_xfer_when_submitted[ 0 ] );

                    cl_ret |= write_xfer_events[ 0 ].getProfilingInfo< cl_ulong >(
                        CL_PROFILING_COMMAND_START, &write_xfer_when_started[ 0 ] );

                    cl_ret |= write_xfer_events[ 0 ].getProfilingInfo< cl_ulong >(
                        CL_PROFILING_COMMAND_END, &write_xfer_when_ended[ 0 ] );
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

                SIXTRL_ASSERT( cl_ret == CL_SUCCESS );

                try
                {
                    cl_ret = queue.enqueueWriteBuffer( cl_beam_elements, CL_TRUE, 0,
                        ::st_Buffer_get_size( lhc_beam_elements ),
                        ::st_Buffer_get_const_data_begin( lhc_beam_elements ),
                        nullptr, &write_xfer_events[ 1 ] );

                    cl_ret |= write_xfer_events[ 1 ].getProfilingInfo< cl_ulong >(
                        CL_PROFILING_COMMAND_QUEUED, &write_xfer_when_queued[ 1 ] );

                    cl_ret |= write_xfer_events[ 1 ].getProfilingInfo< cl_ulong >(
                        CL_PROFILING_COMMAND_SUBMIT, &write_xfer_when_submitted[ 1 ] );

                    cl_ret |= write_xfer_events[ 1 ].getProfilingInfo< cl_ulong >(
                        CL_PROFILING_COMMAND_START, &write_xfer_when_started[ 1 ] );

                    cl_ret |= write_xfer_events[ 1 ].getProfilingInfo< cl_ulong >(
                        CL_PROFILING_COMMAND_END, &write_xfer_when_ended[ 1 ] );
                }
                catch( cl::Error const& e )
                {
                    std::cout << "enqueueWriteBuffer( beam_elements ) :: "
                            << "line = " << __LINE__
                            << " :: ERROR : " << e.what() << std::endl
                            << e.err() << std::endl;

                    cl_ret = CL_FALSE;
                    throw;
                }

                success_flag = int32_t{ 0 };

                try
                {
                    cl_ret = queue.enqueueWriteBuffer( cl_success_flag, CL_TRUE, 0,
                        sizeof( success_flag ), &success_flag,
                        nullptr, &write_xfer_events[ 2 ] );

                    cl_ret |= write_xfer_events[ 2 ].getProfilingInfo< cl_ulong >(
                        CL_PROFILING_COMMAND_QUEUED, &write_xfer_when_queued[ 2 ] );

                    cl_ret |= write_xfer_events[ 2 ].getProfilingInfo< cl_ulong >(
                        CL_PROFILING_COMMAND_SUBMIT, &write_xfer_when_submitted[ 2 ] );

                    cl_ret |= write_xfer_events[ 2 ].getProfilingInfo< cl_ulong >(
                        CL_PROFILING_COMMAND_START, &write_xfer_when_started[ 2 ] );

                    cl_ret |= write_xfer_events[ 2 ].getProfilingInfo< cl_ulong >(
                        CL_PROFILING_COMMAND_END, &write_xfer_when_ended[ 2 ] );
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

                now = ::st_Time_get_seconds_since_epoch();

                double const time_write_xfer = ( now >= begin_time )
                    ? ( now - begin_time ) : double{ 0 };

                /*  ======================================================== */

                cl::Event run_remap_kernel_event;

                cl_ulong run_remap_kernel_when_queued    = cl_ulong{ 0 };
                cl_ulong run_remap_kernel_when_submitted = cl_ulong{ 0 };
                cl_ulong run_remap_kernel_when_started   = cl_ulong{ 0 };
                cl_ulong run_remap_kernel_when_ended     = cl_ulong{ 0 };

                begin_time = ::st_Time_get_seconds_since_epoch();

                remapping_kernel.setArg( 0, cl_particles );
                remapping_kernel.setArg( 1, cl_beam_elements );
                remapping_kernel.setArg( 2, cl_success_flag );

                try
                {
                    cl_ret = queue.enqueueNDRangeKernel( remapping_kernel,
                        cl::NullRange, cl::NDRange( remap_num_threads ),
                        cl::NDRange( remap_group_size ), nullptr,
                        &run_remap_kernel_event );
                }
                catch( cl::Error const& e )
                {
                    std::cout << "enqueueNDRangeKernel( remapping_kernel ) :: "
                            << "line = " << __LINE__
                            << " :: ERROR : " << e.what() << std::endl
                            << e.err() << std::endl;

                    cl_ret = CL_FALSE;
                    throw;
                }

                SIXTRL_ASSERT( cl_ret == CL_SUCCESS );

                queue.flush();
                run_remap_kernel_event.wait();

                cl_ret = run_remap_kernel_event.getProfilingInfo< cl_ulong >(
                    CL_PROFILING_COMMAND_QUEUED, &run_remap_kernel_when_queued );

                cl_ret |= run_remap_kernel_event.getProfilingInfo< cl_ulong >(
                    CL_PROFILING_COMMAND_SUBMIT, &run_remap_kernel_when_submitted );

                cl_ret |= run_remap_kernel_event.getProfilingInfo< cl_ulong >(
                    CL_PROFILING_COMMAND_START, &run_remap_kernel_when_started );

                cl_ret |= run_remap_kernel_event.getProfilingInfo< cl_ulong >(
                    CL_PROFILING_COMMAND_END, &run_remap_kernel_when_ended );

                now = ::st_Time_get_seconds_since_epoch();

                double const time_run_remapping_kernel = ( now >= begin_time )
                    ? ( now - begin_time ) : double{ 0 };

                /* ========================================================= */

                cl::Event xfer_after_remap_events;

                cl_ulong xfer_after_remap_when_queued    = cl_ulong{ 0 };
                cl_ulong xfer_after_remap_when_submitted = cl_ulong{ 0 };
                cl_ulong xfer_after_remap_when_started   = cl_ulong{ 0 };
                cl_ulong xfer_after_remap_when_ended     = cl_ulong{ 0 };

                begin_time = ::st_Time_get_seconds_since_epoch();

                try
                {
                    cl_ret = queue.enqueueReadBuffer( cl_success_flag, CL_TRUE, 0,
                        sizeof( success_flag ), &success_flag,
                        nullptr, &xfer_after_remap_events );


                    cl_ret = xfer_after_remap_events.getProfilingInfo< cl_ulong >(
                        CL_PROFILING_COMMAND_QUEUED, &xfer_after_remap_when_queued );

                    cl_ret |= xfer_after_remap_events.getProfilingInfo< cl_ulong >(
                        CL_PROFILING_COMMAND_SUBMIT, &xfer_after_remap_when_submitted );

                    cl_ret |= xfer_after_remap_events.getProfilingInfo< cl_ulong >(
                        CL_PROFILING_COMMAND_START, &xfer_after_remap_when_started );

                    cl_ret |= xfer_after_remap_events.getProfilingInfo< cl_ulong >(
                        CL_PROFILING_COMMAND_END, &xfer_after_remap_when_ended );
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

                now = ::st_Time_get_seconds_since_epoch();

                SIXTRL_ASSERT( cl_ret == CL_SUCCESS );
                SIXTRL_ASSERT( success_flag == int32_t{ 0 } );

                double const time_xfer_after_remap = ( now >= begin_time )
                    ? ( now - begin_time ) : double{ 0 };

                /* ========================================================== */

                uint64_t const turns = NUM_TURNS;

                cl::Event run_tracking_kernel_event;

                cl_ulong run_tracking_kernel_when_queued    = cl_ulong{ 0 };
                cl_ulong run_tracking_kernel_when_submitted = cl_ulong{ 0 };
                cl_ulong run_tracking_kernel_when_started   = cl_ulong{ 0 };
                cl_ulong run_tracking_kernel_when_ended     = cl_ulong{ 0 };

                begin_time = ::st_Time_get_seconds_since_epoch();

                tracking_kernel.setArg( 0, cl_particles );
                tracking_kernel.setArg( 1, cl_beam_elements );
                tracking_kernel.setArg( 2, turns );
                tracking_kernel.setArg( 3, cl_success_flag );

                try
                {
                    cl_ret = queue.enqueueNDRangeKernel( tracking_kernel,
                        cl::NullRange, cl::NDRange( tracking_num_threads ),
                        cl::NDRange( tracking_group_size ), nullptr,
                        &run_tracking_kernel_event );
                }
                catch( cl::Error const& e )
                {
                    std::cout << "enqueueNDRangeKernel( remapping_kernel ) :: "
                              << "line = " << __LINE__
                              << " :: ERROR : " << e.what() << std::endl
                              << e.err() << std::endl;

                    cl_ret = CL_FALSE;
                    throw;
                }

                cl_ret = queue.flush();
                run_tracking_kernel_event.wait();

                cl_ret |= run_tracking_kernel_event.getProfilingInfo< cl_ulong >(
                    CL_PROFILING_COMMAND_QUEUED, &run_tracking_kernel_when_queued );

                cl_ret |= run_tracking_kernel_event.getProfilingInfo< cl_ulong >(
                    CL_PROFILING_COMMAND_SUBMIT, &run_tracking_kernel_when_submitted );

                cl_ret |= run_tracking_kernel_event.getProfilingInfo< cl_ulong >(
                    CL_PROFILING_COMMAND_START, &run_tracking_kernel_when_started );

                cl_ret |= run_tracking_kernel_event.getProfilingInfo< cl_ulong >(
                    CL_PROFILING_COMMAND_END, &run_tracking_kernel_when_ended );

                now = ::st_Time_get_seconds_since_epoch();

                double const time_run_tracking_kernel = ( now >= begin_time )
                    ? ( now - begin_time ) : double{ 0 };

                double const time_tracking_until_submitted =
                    static_cast< double >( run_tracking_kernel_when_submitted -
                                           run_tracking_kernel_when_queued ) * 1e-9;

                double const time_tracking_until_start =
                    static_cast< double >( run_tracking_kernel_when_started -
                                           run_tracking_kernel_when_submitted ) * 1e-9;

                double const time_tracking_device_execution =
                    static_cast< double >( run_tracking_kernel_when_ended -
                                           run_tracking_kernel_when_started ) * 1e-9;

                /* ========================================================== */

                std::vector< cl::Event >
                    xfer_after_tracking_events( 2u, cl::Event{} );

                cl_ulong xfer_after_tracking_when_queued[]    = { 0, 0 };
                cl_ulong xfer_after_tracking_when_submitted[] = { 0, 0 };
                cl_ulong xfer_after_tracking_when_started[]   = { 0, 0 };
                cl_ulong xfer_after_tracking_when_ended[]     = { 0, 0 };

                begin_time = ::st_Time_get_seconds_since_epoch();

                try
                {
                    cl_ret = queue.enqueueReadBuffer( cl_particles, CL_TRUE, 0,
                        ::st_Buffer_get_size( particles_buffer ),
                        ::st_Buffer_get_data_begin( particles_buffer ),
                        nullptr, &xfer_after_tracking_events[ 0 ] );


                    cl_ret |= xfer_after_tracking_events[ 0 ].getProfilingInfo< cl_ulong >(
                        CL_PROFILING_COMMAND_QUEUED, &xfer_after_tracking_when_queued[ 0 ] );

                    cl_ret |= xfer_after_tracking_events[ 0 ].getProfilingInfo< cl_ulong >(
                        CL_PROFILING_COMMAND_SUBMIT, &xfer_after_tracking_when_submitted[ 0 ] );

                    cl_ret |= xfer_after_tracking_events[ 0 ].getProfilingInfo< cl_ulong >(
                        CL_PROFILING_COMMAND_START, &xfer_after_tracking_when_started[ 0 ] );

                    cl_ret |= xfer_after_tracking_events[ 0 ].getProfilingInfo< cl_ulong >(
                        CL_PROFILING_COMMAND_END, &xfer_after_tracking_when_ended[ 0 ] );
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

                SIXTRL_ASSERT( cl_ret == CL_SUCCESS );

                try
                {
                    cl_ret = queue.enqueueReadBuffer( cl_success_flag, CL_TRUE, 0,
                        sizeof( success_flag ), &success_flag,
                        nullptr, &xfer_after_tracking_events[ 1 ] );

                    cl_ret |= xfer_after_tracking_events[ 1 ].getProfilingInfo< cl_ulong >(
                        CL_PROFILING_COMMAND_QUEUED, &xfer_after_tracking_when_queued[ 1 ] );

                    cl_ret |= xfer_after_tracking_events[ 1 ].getProfilingInfo< cl_ulong >(
                        CL_PROFILING_COMMAND_SUBMIT, &xfer_after_tracking_when_submitted[ 1 ] );

                    cl_ret |= xfer_after_tracking_events[ 1 ].getProfilingInfo< cl_ulong >(
                        CL_PROFILING_COMMAND_START, &xfer_after_tracking_when_started[ 1 ] );

                    cl_ret |= xfer_after_tracking_events[ 1 ].getProfilingInfo< cl_ulong >(
                        CL_PROFILING_COMMAND_END, &xfer_after_tracking_when_ended[ 1 ] );
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

                now = ::st_Time_get_seconds_since_epoch();

                SIXTRL_ASSERT( cl_ret == CL_SUCCESS );
                SIXTRL_ASSERT( success_flag == int32_t{ 0 } );

                double const time_xfer_after_tracking = ( now >= begin_time )
                    ? ( now - begin_time ) : double{ 0 };

                /* ======================================================== */

                a2str.str( "" );

                double time_run_tracking_normalized =
                    time_run_tracking_kernel / static_cast< double >(
                        NUM_TURNS * NUM_PARTICLES );

                if( time_run_tracking_normalized >= 0.1 )
                {
                    a2str << "sec";
                }


                std::cout << std::endl
                          << "Reslts: \r\n"
                          << "------------------------------------------------"
                          << "------------------------------------------------"
                          << "--------------------------------------------\r\n"
                          << "      :: Tracking time                 : "
                          << std::setw( 20 ) << std::fixed
                          << time_run_tracking_kernel << " [sec] \r\n"
                          << "      :: Tracking time/particle/turn   : ";


                if( time_run_tracking_normalized >= 200e-3 )
                {
                    std::cout << std::setw( 20 ) << std::fixed
                              << time_run_tracking_normalized << "[sec]\r\n";
                }
                else if( time_run_tracking_normalized >= 200e-6 )
                {
                    std::cout << std::setw( 20 ) << std::fixed
                              << time_run_tracking_normalized * 1e3 << "[millisec]\r\n";
                }
                else
                {
                    std::cout << std::setw( 20 ) << std::fixed
                              << time_run_tracking_normalized * 1e6 << "[usec]\r\n";
                }

                std::cout << "      :: device_run_time               : "
                          << std::setw( 20 ) << std::fixed
                          << time_tracking_device_execution << "\r\n"
                          << "      :: device overhead               : "
                          << std::setw( 20 ) << std::fixed
                          << time_tracking_until_start << " + "
                          << time_tracking_until_submitted << "\r\n"
                          << "------------------------------------------------"
                          << "------------------------------------------------"
                          << "--------------------------------------------\r\n"
                          << "\r\n"
                          << std::endl;
            }

            int success = ::st_Buffer_remap( particles_buffer );
            SIXTRL_ASSERT( success == 0 );

            ::st_Buffer_reset( particles_buffer );
        }
    }

    ::st_Buffer_delete( lhc_particles_buffer );
    ::st_Buffer_delete( lhc_beam_elements );
    ::st_Buffer_delete( particles_buffer );

    return 0;
}

/* end: tests/benchmark/sixtracklib/opencl/benchmark_lhc_no_bb_opencl_c99.c */
