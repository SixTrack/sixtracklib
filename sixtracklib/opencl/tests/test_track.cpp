#define _USE_MATH_DEFINES 

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <cmath>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <CL/cl.hpp>

#if defined( __NAMESPACE )
    #define __SAVED_NAMESPACE __NAMESPACE
    #undef  __NAMESPACE     
#endif /* !defiend( __NAMESPACE ) */

#if !defined( __NAMESPACE )
    #define __NAMESPACE st_    
#endif /* !defiend( __NAMESPACE ) */

#include "sixtracklib/_impl/definitions.h"
#include "sixtracklib/_impl/path.h"

#include "sixtracklib/common/blocks.h"
#include "sixtracklib/common/beam_elements.h"
#include "sixtracklib/common/particles.h"
#include "sixtracklib/common/track.h"
#include "sixtracklib/common/details/gpu_kernel_tools.h"

#include "sixtracklib/common/tests/test_particles_tools.h"
#include "sixtracklib/common/tests/test_track_tools.h"
#include "sixtracklib/testdata/tracking_testfiles.h"

#if defined( __SAVED_NAMESPACE )
    #undef __NAMESPACE
    #define __NAMESPACE __SAVED_NAMESPACE
#endif /* defined( __SAVED_NAMESPACE ) */

/* ========================================================================= */
/* =====  track drifts of constant length                                    */
/* ========================================================================= */

TEST( OpenCLTrackTests, TrackDrifts )
{
    uint64_t dummy_elem_by_elem_buffer[ 4 ] = { 
            uint64_t{ 0 }, uint64_t{ 0 }, uint64_t{ 0 }, uint64_t{ 0 } };
    
    st_Blocks initial_particles_buffer;
    st_Blocks result_particles_buffer;
    st_Blocks beam_elements;
    st_Blocks elem_by_elem_buffer;
    
    st_Blocks_preset( &initial_particles_buffer );
    st_Blocks_preset( &result_particles_buffer );
    st_Blocks_preset( &elem_by_elem_buffer );
    st_Blocks_preset( &beam_elements );
    
    uint64_t NUM_OF_TURNS = uint64_t{ 0 };
    
    ASSERT_TRUE( st_Tracks_restore_testdata_from_binary_file(
        st_PATH_TO_TEST_TRACKING_DRIFT_DATA, &NUM_OF_TURNS,
        &initial_particles_buffer, 
        &result_particles_buffer, 
        &beam_elements, &elem_by_elem_buffer ) );
    
    /* --------------------------------------------------------------------- */
    
    st_block_size_t const NUM_OF_BEAM_ELEMENTS = 
        st_Blocks_get_num_of_blocks( &beam_elements );
        
    st_block_size_t const NUM_OF_PARTICLE_BLOCKS = 
        st_Blocks_get_num_of_blocks( &initial_particles_buffer );
        
    cl_ulong NUM_OF_PARTICLES = cl_ulong{ 0 };
    
    st_BlockInfo const* part_block_it  = 
        st_Blocks_get_const_block_infos_begin( &initial_particles_buffer );
        
    st_BlockInfo const* part_block_end = 
        st_Blocks_get_const_block_infos_end( &initial_particles_buffer );
        
    for( ; part_block_it != part_block_end ; ++part_block_it )
    {
        st_Particles const* particles = 
            st_Blocks_get_const_particles( part_block_it );
            
        NUM_OF_PARTICLES += st_Particles_get_num_particles( particles );
    }
        
    /* --------------------------------------------------------------------- */
    
    st_Blocks particles_buffer;
    st_Blocks_preset( &particles_buffer );
    
    ASSERT_TRUE( 0 == st_Blocks_init_from_serialized_data( &particles_buffer, 
       st_Blocks_get_const_data_begin( &initial_particles_buffer ),
       st_Blocks_get_total_num_bytes(  &initial_particles_buffer ) ) );
    
    st_Blocks calculated_elem_by_elem_buffer;
    st_Blocks_preset( &calculated_elem_by_elem_buffer );
    
    bool use_elem_by_elem_buffer = false;
        
    st_block_size_t const AVAILABLE_ELEM_BY_ELEM_BLOCKS = 
        st_Blocks_get_num_of_blocks( &elem_by_elem_buffer );
      
    st_block_size_t const NUM_IO_ELEMENTS_PER_TURN =
        NUM_OF_BEAM_ELEMENTS * NUM_OF_PARTICLE_BLOCKS;
        
    if( ( NUM_OF_TURNS > uint64_t{ 0 } ) &&
        ( ( NUM_OF_TURNS * NUM_IO_ELEMENTS_PER_TURN ) 
            <= AVAILABLE_ELEM_BY_ELEM_BLOCKS ) )
    {
        ASSERT_TRUE( 0 == st_Blocks_init_from_serialized_data( 
            &calculated_elem_by_elem_buffer,
            st_Blocks_get_const_data_begin( &calculated_elem_by_elem_buffer ),
            st_Blocks_get_total_num_bytes(  &calculated_elem_by_elem_buffer ) ) 
        );
        
        st_Particles_buffer_preset_values( &calculated_elem_by_elem_buffer );
        
        use_elem_by_elem_buffer = true;
    }
    
    /* ******************************************************************** */
    /* *****                 OpenCL based tracking                    ***** */
    /* ******************************************************************** */
    
    std::vector< cl::Platform >platforms;
    cl::Platform::get( &platforms );
    
    ASSERT_TRUE( !platforms.empty() );
    cl::Platform platform = platforms.front();
    
    cl_int ret = 0;
    
    if( !platforms.empty() )
    {
        std::string name;
        ret  = platform.getInfo( CL_PLATFORM_NAME, &name );
    
        std::string vendor;
        ret |= platform.getInfo( CL_PLATFORM_VENDOR, &vendor );
    
        std::string profile;
        ret |= platform.getInfo( CL_PLATFORM_PROFILE, &profile );
        
        ASSERT_TRUE( ret == CL_SUCCESS );
        
        
        std::cout << "selected platform  : \r\n" 
                  << " -> name           = " << name    << "\r\n"
                  << " -> vendor         = " << vendor  << "\r\n"
                  << " -> profile        = " << profile << "\r\n"
                  << "\r\n" << std::endl;
    }
    
    std::vector< cl::Device > devices;
    platform.getDevices( CL_DEVICE_TYPE_ALL, &devices );
    ASSERT_TRUE( !devices.empty() );
    
    cl::Device device = devices.front();
        
    if( !devices.empty() )        
    {
        std::string name;
        ret  = device.getInfo( CL_DEVICE_NAME, &name );
        
        std::string vendor;
        ret |= device.getInfo( CL_DEVICE_VENDOR, &vendor );
        
        std::string profile;
        ret |= device.getInfo( CL_DEVICE_PROFILE, &profile );
        
        std::string opencl_version;
        ret |= device.getInfo( CL_DEVICE_OPENCL_C_VERSION, &opencl_version );
        
        ASSERT_TRUE( ret == CL_SUCCESS );
        
        std::cout   << "selected device : \r\n"
                    << " -> name           = " <<  name    << "\r\n"
                    << " -> vendor         = " <<  vendor  << "\r\n"
                    << " -> profile        = " <<  profile << "\r\n"
                    << " -> OpenCL Version = " << opencl_version << "\r\n"
                    << std::endl;
    }
    
    
    cl::Context context( device );
    
    std::string PATH_TO_SOURCE_DIR( st_PATH_TO_BASE_DIR );
    
    PATH_TO_SOURCE_DIR += std::string( "sixtracklib/" );
    
    std::vector< std::string > const paths_to_kernel_files{
        PATH_TO_SOURCE_DIR + std::string{ "_impl/namespace_begin.h" },
        PATH_TO_SOURCE_DIR + std::string{ "_impl/definitions.h" },
        PATH_TO_SOURCE_DIR + std::string{ "common/blocks.h" },
        PATH_TO_SOURCE_DIR + std::string{ "common/impl/particles_type.h" },
        PATH_TO_SOURCE_DIR + std::string{ "common/impl/particles_api.h" },
        PATH_TO_SOURCE_DIR + std::string{ "common/particles.h" },
        PATH_TO_SOURCE_DIR + std::string{ "common/impl/beam_elements_type.h" },
        PATH_TO_SOURCE_DIR + std::string{ "common/impl/beam_elements_api.h" },
        PATH_TO_SOURCE_DIR + std::string{ "common/beam_elements.h" },
        PATH_TO_SOURCE_DIR + std::string{ "common/track.h" },
        PATH_TO_SOURCE_DIR + std::string{ "common/impl/track_api.h" },
        PATH_TO_SOURCE_DIR + std::string{ "opencl/track_particles_kernel.cl" },
        PATH_TO_SOURCE_DIR + std::string{ "_impl/namespace_end.h" }
    };
    
    std::string kernel_source( 1024 * 1024, '\0' );
    kernel_source.clear();
    
    for( auto const& path : paths_to_kernel_files )
    {
        std::ifstream const one_kernel_file( path, std::ios::in );
        
        std::istreambuf_iterator< char > one_kernel_file_begin( 
            one_kernel_file.rdbuf() );
        std::istreambuf_iterator< char > end_of_file;
        
        kernel_source.insert( kernel_source.end(), 
                              one_kernel_file_begin, end_of_file );
    }
    
    if( !kernel_source.empty() )
    {
        std::ofstream tmp( "/tmp/out.cl" );
        tmp << kernel_source << std::endl;
        tmp.flush();
        tmp.close();
    }
        
    cl::Program program( context, kernel_source );
    
    char compile_options[] = "-D _GPUCODE=1 -D __NAMESPACE=st_";
    
    if( program.build( compile_options ) != CL_SUCCESS )
    {  
        std::ofstream tmp( "/tmp/out.cl" );
        tmp << kernel_source << std::endl;
        tmp.flush();
        tmp.close();
        
        std::cout  << "Error building: " 
                   << program.getBuildInfo< CL_PROGRAM_BUILD_LOG >( device ) 
                   << "\n";
        exit(1);
    }
    
    std::size_t const PARTICLES_BUFFER_SIZE = 
        st_Blocks_get_total_num_bytes( &particles_buffer );
        
    std::size_t const BEAM_ELEMENTS_BUFFER_SIZE =
        st_Blocks_get_total_num_bytes( &beam_elements );
        
    std::size_t const ELEM_BY_ELEM_BUFFER_SIZE = ( use_elem_by_elem_buffer )
        ? st_Blocks_get_total_num_bytes( &calculated_elem_by_elem_buffer )
        : 32u;
        
    ret = CL_SUCCESS;
    
    /* --------------------------------------------------------------------- */
    
    cl::Buffer cl_particles_buffer( 
        context, CL_MEM_READ_WRITE, PARTICLES_BUFFER_SIZE, 0, &ret );
    
    ASSERT_TRUE( ret == CL_SUCCESS );
    
    cl::Buffer cl_beam_elements_buffer(
        context, CL_MEM_READ_WRITE, BEAM_ELEMENTS_BUFFER_SIZE, 0, &ret );
    
    ASSERT_TRUE( ret == CL_SUCCESS );
    
    cl::Buffer cl_elem_by_elem_buffer(
        context, CL_MEM_READ_WRITE, ELEM_BY_ELEM_BUFFER_SIZE, 0, &ret );
    
    ASSERT_TRUE( ret == CL_SUCCESS );
    
    cl::Buffer cl_success_flag_buffer(
        context, CL_MEM_READ_WRITE, sizeof( cl_long ), 0, &ret );
    
    ASSERT_TRUE( ret == CL_SUCCESS );
    
    /* --------------------------------------------------------------------- */
    
    cl::Kernel remap_kernel( 
        program, "Track_remap_serialized_blocks_buffer", &ret );
    
    ASSERT_TRUE( ret == CL_SUCCESS );
    
    ret  = remap_kernel.setArg( 0, cl_particles_buffer );
    ret |= remap_kernel.setArg( 1, cl_beam_elements_buffer );
    ret |= remap_kernel.setArg( 2, cl_elem_by_elem_buffer );
    ret |= remap_kernel.setArg( 3, cl_success_flag_buffer );
    
    ASSERT_TRUE( ret == CL_SUCCESS );
    
    cl::Kernel track_kernel( program, "Track_particles_kernel_opencl", &ret );
    
    ASSERT_TRUE( ret == CL_SUCCESS );
    
    ret  = track_kernel.setArg( 0, NUM_OF_TURNS );
    ret |= track_kernel.setArg( 1, cl_particles_buffer );
    ret |= track_kernel.setArg( 2, cl_beam_elements_buffer );
    ret |= track_kernel.setArg( 3, cl_elem_by_elem_buffer );
    ret |= track_kernel.setArg( 4, cl_success_flag_buffer );
    
    ASSERT_TRUE( ret == CL_SUCCESS );
    
    size_t const remap_work_group_multiple = remap_kernel.getWorkGroupInfo< 
            CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE >( device );
    
    size_t const local_work_group_size = 
        track_kernel.getWorkGroupInfo< CL_KERNEL_WORK_GROUP_SIZE >( device );
        
    ASSERT_TRUE( remap_work_group_multiple != 0u );
    ASSERT_TRUE( local_work_group_size     != 0u );
    
    size_t global_work_size = NUM_OF_PARTICLES;
    
    if( ( global_work_size % local_work_group_size ) != 0u )
    {
        global_work_size /= local_work_group_size;
        ++global_work_size;
        
        global_work_size *= local_work_group_size;
    }
    
    ASSERT_TRUE( ( global_work_size >= NUM_OF_PARTICLES ) &&
               ( ( global_work_size % local_work_group_size ) == 0u ) );
    
    /* --------------------------------------------------------------------- */
    
    cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE, &ret );
    ASSERT_TRUE( ret == CL_SUCCESS );
     
    /* --------------------------------------------------------------------- */
    
    unsigned char* particles_data_buffer = 
        st_Blocks_get_data_begin( &particles_buffer );
        
    unsigned char* beam_elements_data_buffer =
        st_Blocks_get_data_begin( &beam_elements );
        
    unsigned char* elem_by_elem_data_buffer = ( use_elem_by_elem_buffer )
         ? static_cast< unsigned char* >( 
             st_Blocks_get_data_begin( &calculated_elem_by_elem_buffer ) )
         : reinterpret_cast< unsigned char* >( 
             &dummy_elem_by_elem_buffer[ 0 ] );
         
    cl_long success_flag = 0;
        
    ASSERT_TRUE( particles_data_buffer     != nullptr );
    ASSERT_TRUE( beam_elements_data_buffer != nullptr );
    ASSERT_TRUE( elem_by_elem_data_buffer  != nullptr );
    
    ret =  queue.enqueueWriteBuffer( 
                cl_particles_buffer, CL_TRUE, 0, PARTICLES_BUFFER_SIZE, 
                particles_data_buffer );
    
    ret |= queue.enqueueWriteBuffer(
                cl_beam_elements_buffer, CL_TRUE, 0, BEAM_ELEMENTS_BUFFER_SIZE,
                beam_elements_data_buffer );
    
    ret |= queue.enqueueWriteBuffer(
                cl_elem_by_elem_buffer, CL_TRUE, 0, ELEM_BY_ELEM_BUFFER_SIZE,
                elem_by_elem_data_buffer );
    
    ret |= queue.enqueueWriteBuffer( cl_success_flag_buffer, CL_TRUE, 0, 
                sizeof( cl_long ), &success_flag );
    
    ASSERT_TRUE( ret == CL_SUCCESS );
    
    /* --------------------------------------------------------------------- */
    
    cl::Event event_remap;
    
    ret = queue.enqueueNDRangeKernel(
                remap_kernel, cl::NullRange, 
                cl::NDRange( remap_work_group_multiple ),
                cl::NDRange( remap_work_group_multiple ), 
                nullptr, &event_remap );    
    
    ASSERT_TRUE( ret == CL_SUCCESS );
    
    ret = queue.flush();
    ASSERT_TRUE( ret == CL_SUCCESS );
    
    ret = event_remap.wait();
    ASSERT_TRUE( ret == CL_SUCCESS );
    
    /* --------------------------------------------------------------------- */
    
    ret = queue.enqueueReadBuffer( cl_success_flag_buffer, CL_TRUE, 0,
                                   sizeof( cl_long ), &success_flag );
    
    ASSERT_TRUE( success_flag == 0 );
    ASSERT_TRUE( ret == CL_SUCCESS );
    
    /* --------------------------------------------------------------------- */
    
    cl::Event event_track;
    
    ret = queue.enqueueNDRangeKernel(
                track_kernel, cl::NullRange, cl::NDRange( global_work_size ),
                cl::NDRange( local_work_group_size ), nullptr, &event_track );
    
    ASSERT_TRUE( ret == CL_SUCCESS );
    
    ret |= queue.flush();
    ASSERT_TRUE( ret == CL_SUCCESS );
    
    ret = event_track.wait();
    ASSERT_TRUE( ret == CL_SUCCESS );
    
    cl_ulong when_kernel_queued    = 0;
    cl_ulong when_kernel_submitted = 0;
    cl_ulong when_kernel_started   = 0;
    cl_ulong when_kernel_ended     = 0;

    ret  = event_track.getProfilingInfo< cl_ulong >( 
      CL_PROFILING_COMMAND_QUEUED, &when_kernel_queued );

    ret |= event_track.getProfilingInfo< cl_ulong >( 
      CL_PROFILING_COMMAND_SUBMIT, &when_kernel_submitted );

    ret |= event_track.getProfilingInfo< cl_ulong >( 
      CL_PROFILING_COMMAND_START, &when_kernel_started );

    ret |= event_track.getProfilingInfo< cl_ulong >( 
      CL_PROFILING_COMMAND_END, &when_kernel_ended );

    ASSERT_TRUE( ret == CL_SUCCESS );
    
    double const kernel_time_elapsed = when_kernel_ended - when_kernel_started;
    std::cout << "kernel_time_elapsed: " << kernel_time_elapsed << std::endl;
    std::cout.flush();
    
    ret  = queue.enqueueReadBuffer( cl_success_flag_buffer, CL_TRUE, 
            0, sizeof( cl_long ), &success_flag );
    
    ASSERT_TRUE( success_flag == 0 );
    ASSERT_TRUE( ret == CL_SUCCESS );
    
    ret  = queue.enqueueReadBuffer(
            cl_particles_buffer, CL_TRUE, 0, PARTICLES_BUFFER_SIZE, 
            particles_data_buffer );
    
    ASSERT_TRUE( ret == CL_SUCCESS );
    
    ret = queue.enqueueReadBuffer(
            cl_elem_by_elem_buffer, CL_TRUE, 0, ELEM_BY_ELEM_BUFFER_SIZE,
            elem_by_elem_data_buffer );
    
    ASSERT_TRUE( ret == CL_SUCCESS );
    
    /* ******************************************************************** */
    /* *****             End of OpenCL based tracking                 ***** */
    /* ******************************************************************** */
        
    ASSERT_TRUE( 0 == st_Blocks_unserialize( 
        &particles_buffer, particles_data_buffer ) );
    
    /* --------------------------------------------------------------------- */
    
    ASSERT_TRUE( st_Particles_buffers_have_same_structure( 
        &initial_particles_buffer, &result_particles_buffer ) );
    
    ASSERT_TRUE( st_Particles_buffers_have_same_structure( 
        &initial_particles_buffer, &particles_buffer ) );
    
    ASSERT_TRUE( !st_Particles_buffers_map_to_same_memory(
        &initial_particles_buffer, &result_particles_buffer ) );
    
    ASSERT_TRUE( !st_Particles_buffers_map_to_same_memory(
        &initial_particles_buffer, &particles_buffer ) );
    
    if( 0 != st_Particles_buffer_compare_values(
        &result_particles_buffer, &particles_buffer ) )
    {
        st_Blocks max_diff_buffer;
        st_Blocks_preset( &max_diff_buffer );
        
        st_block_size_t const MAX_DIST_DATA_CAPACITY = 
            st_Blocks_predict_data_capacity_for_num_blocks(
                &max_diff_buffer, NUM_OF_PARTICLE_BLOCKS ) +
            st_Particles_predict_blocks_data_capacity(
                &max_diff_buffer, NUM_OF_PARTICLE_BLOCKS, 1u );
        
        st_Blocks_init( &max_diff_buffer, 
                        NUM_OF_PARTICLE_BLOCKS, MAX_DIST_DATA_CAPACITY );
        
        for( st_block_size_t ii = 0 ; ii < NUM_OF_PARTICLE_BLOCKS ; ++ii )
        {
            st_Particles* particles = st_Blocks_add_particles( 
                &max_diff_buffer, 1u );
            
            if( particles != nullptr )
            {
                st_Particles_preset_values( particles );
            }
        }
        
        std::vector< st_block_size_t > max_diff_index(
            NUM_OF_PARTICLE_BLOCKS * 20, st_block_size_t{ 0 } );
        
        st_Blocks_serialize( &max_diff_buffer );
        
        st_Particles_buffer_get_max_difference( 
            &max_diff_buffer, max_diff_index.data(),
            &result_particles_buffer, &particles_buffer );
        
        fprintf( stdout, "|Diff| = |precalculated result - calculated|\r\n" );
        
        st_Particles_buffer_print_max_diff( 
            stdout, &max_diff_buffer, max_diff_index.data() );
        
        st_Blocks_free( &max_diff_buffer );
    }
    
    if( use_elem_by_elem_buffer )
    {
        st_block_size_t ll = 0;
        
        st_block_size_t const num_elem_by_elem_per_turn = 
            NUM_OF_PARTICLE_BLOCKS * NUM_OF_BEAM_ELEMENTS;
        
        st_BlockInfo const* block_it = st_Blocks_get_const_block_infos_begin( 
            &calculated_elem_by_elem_buffer );
        
        st_BlockInfo const* block_end = st_Blocks_get_const_block_infos_end(
            &calculated_elem_by_elem_buffer );
        
        st_BlockInfo const* cmp_block_it = 
            st_Blocks_get_const_block_infos_begin( &elem_by_elem_buffer );
            
        st_BlockInfo const* cmp_block_end =
            st_Blocks_get_const_block_infos_end( &elem_by_elem_buffer );
        
        ASSERT_TRUE( ( block_it      != nullptr ) && 
                     ( block_end     != nullptr ) &&
                     ( cmp_block_it  != nullptr ) && 
                     ( cmp_block_end != nullptr ) );
            
        ASSERT_TRUE( std::distance( cmp_block_end, cmp_block_it ) >=
                     std::distance( block_end,     block_it     ) );
        
        for( ; block_it != block_end ; ++block_it, ++cmp_block_it, ++ll )
        {
            st_Particles const* particles = 
                st_Blocks_get_const_particles( block_it );
                
            st_Particles const* cmp_particles = 
                st_Blocks_get_const_particles( cmp_block_it );
            
            ASSERT_TRUE( st_Particles_have_same_structure( 
                particles, cmp_particles ) );
            
            ASSERT_TRUE( !st_Particles_map_to_same_memory(
                particles, cmp_particles ) );
            
            if( 0 != st_Particles_compare_values( particles, cmp_particles ) )
            {
                st_block_size_t const turn = ll / num_elem_by_elem_per_turn;
                st_block_size_t       temp = ll % num_elem_by_elem_per_turn;
                
                st_block_size_t const particle_block_index = 
                    temp / NUM_OF_BEAM_ELEMENTS;
                    
                st_block_size_t const beam_element_index =
                    temp % NUM_OF_BEAM_ELEMENTS;
                
                fprintf( stdout, "first deviation in elem_by_elem buffer @"
                         "elem_by_elem_block_index = %8lu :: "
                         "turn = %8lu / part_block_idx = %8ld / "
                         "beam_elem_id = %8ld\r\n", 
                         ll, turn, particle_block_index, beam_element_index );
                
                break;
            }
        }
    }
    
    /* --------------------------------------------------------------------- */
    
    st_Blocks_free( &calculated_elem_by_elem_buffer );
    st_Blocks_free( &particles_buffer );
    
    st_Blocks_free( &initial_particles_buffer );
    st_Blocks_free( &result_particles_buffer );
    st_Blocks_free( &beam_elements );
    st_Blocks_free( &elem_by_elem_buffer );    
}

/* ************************************************************************* */


/* end: sixtracklib/opencl/tests/test_track.cpp */
