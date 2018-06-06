#define _USE_MATH_DEFINES 

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <fstream>
#include <sstream>
#include <cmath>

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

#include "sixtracklib/common/details/random.h"
#include "sixtracklib/common/tests/test_particles_tools.h"

#if defined( __SAVED_NAMESPACE )
    #undef __NAMESPACE
    #define __NAMESPACE __SAVED_NAMESPACE
#endif /* defined( __SAVED_NAMESPACE ) */

/* ========================================================================= */
/* =====  track drifts of constant length                                    */
/* ========================================================================= */

TEST( CommonTrackTests, TrackDrifts )
{
    uint64_t seed = UINT64_C( 20180420 );
    st_Random_init_genrand64( seed );
    
    /* --------------------------------------------------------------------- */
    
    st_block_size_t const NUM_OF_TURNS         = st_block_size_t{ 100 };
    st_block_size_t const NUM_OF_PARTICLES     = st_block_size_t{ 100 };
    st_block_size_t const NUM_OF_BEAM_ELEMENTS = st_block_size_t{ 100 };
    
    /* --------------------------------------------------------------------- */
    
    st_Blocks beam_elements;
    st_Blocks_preset( &beam_elements );
    
    st_block_size_t const BEAM_ELEMENTS_DATA_CAPACITY =
        st_Blocks_predict_data_capacity_for_num_blocks( 
            &beam_elements, NUM_OF_BEAM_ELEMENTS ) +
        st_Drift_predict_blocks_data_capacity( 
            &beam_elements, NUM_OF_BEAM_ELEMENTS );
    
    int ret = st_Blocks_init( 
        &beam_elements, NUM_OF_BEAM_ELEMENTS, BEAM_ELEMENTS_DATA_CAPACITY );
    
    std::vector< st_Drift* > ptr_beam_elem( NUM_OF_BEAM_ELEMENTS, nullptr );
    
    SIXTRL_REAL_T const DRIFT_LEN = SIXTRL_REAL_T{ 0.5L };
    
    for( st_block_size_t ii = 0 ; ii < NUM_OF_BEAM_ELEMENTS ; ++ii )
    {
        ptr_beam_elem[ ii ] = st_Blocks_add_drift( &beam_elements, DRIFT_LEN );
        ASSERT_TRUE( ptr_beam_elem[ ii ] != nullptr );
    }
    
    ASSERT_TRUE( st_Blocks_serialize( &beam_elements ) == 0 );
    ASSERT_TRUE( st_Blocks_are_serialized( &beam_elements ) );
    ASSERT_TRUE( st_Blocks_get_num_of_blocks( &beam_elements ) == 
                 NUM_OF_BEAM_ELEMENTS );
    
    ASSERT_TRUE( ret == 0 );
    
    /* --------------------------------------------------------------------- */
    
    st_Blocks particles_buffer;
    st_Blocks_preset( &particles_buffer );
    
    st_Blocks copy_particles_buffer;
    st_Blocks_preset( &copy_particles_buffer );
    
    st_block_size_t const PARTICLES_DATA_CAPACITY = 
        st_Blocks_predict_data_capacity_for_num_blocks(
            &particles_buffer, 1u ) +
        st_Particles_predict_blocks_data_capacity(
            &particles_buffer, 1u, NUM_OF_PARTICLES );
        
    ret = st_Blocks_init( &particles_buffer, 1u, PARTICLES_DATA_CAPACITY );        
    ASSERT_TRUE( ret == 0 );
    
    ret = st_Blocks_init( &copy_particles_buffer, 1u, PARTICLES_DATA_CAPACITY );
    ASSERT_TRUE( ret == 0 );
    
    st_Particles* initial_particles = 
        st_Blocks_add_particles( &copy_particles_buffer, NUM_OF_PARTICLES );
    
    ASSERT_TRUE( initial_particles != nullptr );
    ASSERT_TRUE( st_Blocks_serialize( &copy_particles_buffer ) == 0 );
    ASSERT_TRUE( st_Blocks_are_serialized( &copy_particles_buffer ) );
    ASSERT_TRUE( st_Blocks_get_num_of_blocks( &copy_particles_buffer ) == 1u );
    
    st_Particles* particles = 
        st_Blocks_add_particles( &particles_buffer, NUM_OF_PARTICLES );
    
    ASSERT_TRUE( particles != nullptr );
    ASSERT_TRUE( st_Blocks_serialize( &particles_buffer ) == 0 );
    ASSERT_TRUE( st_Blocks_are_serialized( &particles_buffer ) );
    ASSERT_TRUE( st_Blocks_get_num_of_blocks( &particles_buffer ) == 1u );
    
    st_Particles_random_init( particles );
    st_Particles_copy_all_unchecked( initial_particles, particles );
    
    /* --------------------------------------------------------------------- */
    
    st_Blocks elem_by_elem;
    st_Blocks_preset( &elem_by_elem );
    
    st_block_size_t const NUM_OF_ELEM_BY_ELEM_BLOCKS =
        NUM_OF_TURNS * NUM_OF_BEAM_ELEMENTS;
    
    st_block_size_t const ELEM_BY_ELEM_DATA_CAPACITY =
        st_Blocks_predict_data_capacity_for_num_blocks(
            &elem_by_elem, NUM_OF_ELEM_BY_ELEM_BLOCKS ) +
        st_Particles_predict_blocks_data_capacity(
            &elem_by_elem, NUM_OF_ELEM_BY_ELEM_BLOCKS, NUM_OF_PARTICLES );

    ret = st_Blocks_init( &elem_by_elem, NUM_OF_ELEM_BY_ELEM_BLOCKS, 
                          ELEM_BY_ELEM_DATA_CAPACITY );
    
    ASSERT_TRUE( ret == 0 );
    
    std::vector< st_Particles* > elem_by_elem_particles( 
        NUM_OF_ELEM_BY_ELEM_BLOCKS, nullptr );
    
    for( st_block_size_t ii = 0 ; ii <  NUM_OF_ELEM_BY_ELEM_BLOCKS ; ++ii )
    {
        elem_by_elem_particles[ ii ] = st_Blocks_add_particles( 
            &elem_by_elem, NUM_OF_PARTICLES );
        
        ASSERT_TRUE( elem_by_elem_particles[ ii ] != nullptr );
        ASSERT_TRUE( elem_by_elem_particles[ ii ]->num_of_particles == 
                     NUM_OF_PARTICLES );
    }
    
    ASSERT_TRUE( st_Blocks_serialize( &elem_by_elem ) == 0 );
    ASSERT_TRUE( st_Blocks_are_serialized( &elem_by_elem ) );
    ASSERT_TRUE( st_Blocks_get_num_of_blocks( &elem_by_elem ) == 
        NUM_OF_ELEM_BY_ELEM_BLOCKS );
    
    /* ******************************************************************** */
    /* *****                 OpenCL based tracking                    ***** */
    /* ******************************************************************** */
    
    std::vector< cl::Platform >platforms;
    cl::Platform::get( &platforms );
    
    ASSERT_TRUE( !platforms.empty() );
    cl::Platform platform = platforms.front();
    
    std::vector< cl::Device > devices;
    platform.getDevices( CL_DEVICE_TYPE_ALL, &devices );
            
    ASSERT_TRUE( !devices.empty() );
    
    cl::Device device = devices.front();
    cl::Context context( device );
    
    std::string PATH_TO_KERNEL_DIR( st_PATH_TO_BASE_DIR );
    
    PATH_TO_KERNEL_DIR += std::string( "sixtracklib/" );
    
    std::vector< std::string > const paths_to_kernel_files{
        PATH_TO_KERNEL_DIR + std::string{ "_impl/namespace_begin.h" },
        PATH_TO_KERNEL_DIR + std::string{ "_impl/definitions.h" },
        PATH_TO_KERNEL_DIR + std::string{ "common/blocks.h" },
        PATH_TO_KERNEL_DIR + std::string{ "common/impl/particles_type.h" },
        PATH_TO_KERNEL_DIR + std::string{ "common/impl/particles_api.h" },
        PATH_TO_KERNEL_DIR + std::string{ "common/particles.h" },
        PATH_TO_KERNEL_DIR + std::string{ "common/impl/beam_elements_type.h" },
        PATH_TO_KERNEL_DIR + std::string{ "common/impl/beam_elements_api.h" },
        PATH_TO_KERNEL_DIR + std::string{ "common/beam_elements.h" },
        PATH_TO_KERNEL_DIR + std::string{ "opencl/track_particles_kernel.cl" },
        PATH_TO_KERNEL_DIR + std::string{ "_impl/namespace_end.h" }
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
        
    /* ******************************************************************** */
    /* *****             End of OpenCL based tracking                 ***** */
    /* ******************************************************************** */
        
    std::ostringstream a2str("");
    a2str << st_PATH_TO_BASE_DIR 
          << "sixtracklib/opencl/tests/testdata/test_track_drift_opencl"
          << "__nturn"
          << std::setfill( '0' ) << std::setw( 6 ) << NUM_OF_TURNS
          << "__npart" 
          << std::setw( 6 ) << NUM_OF_PARTICLES 
          << "__nelem" 
          << std::setw( 4 ) << NUM_OF_BEAM_ELEMENTS
          << "__driftlen" << std::setw( 4 )
          << static_cast< int >( 1000.0 * DRIFT_LEN + 0.5 )
          << "mm.bin";
          
    std::ofstream test_data_out( 
        a2str.str().c_str(), std::ios::binary | std::ios::out );
    
    test_data_out << NUM_OF_TURNS;
    test_data_out << NUM_OF_PARTICLES;
    test_data_out << NUM_OF_BEAM_ELEMENTS;
    
    uint64_t const beam_elements_bytes = 
        st_Blocks_get_total_num_bytes( &beam_elements );
    
    test_data_out << beam_elements_bytes;
    test_data_out.write( 
        ( char* )st_Blocks_get_const_data_begin( &beam_elements ), 
        beam_elements_bytes );
    
    uint64_t const initial_particles_size =
        st_Blocks_get_total_num_bytes( &copy_particles_buffer );
        
    test_data_out << initial_particles_size;
    test_data_out.write(
        ( char* )st_Blocks_get_const_data_begin( &copy_particles_buffer ),
        initial_particles_size );
    
    uint64_t const elem_by_elem_size =
        st_Blocks_get_total_num_bytes( &elem_by_elem );
        
    test_data_out << elem_by_elem_size;
    test_data_out.write(
        ( char* )st_Blocks_get_const_data_begin( &elem_by_elem ),
        elem_by_elem_size );
    
    uint64_t const end_particles_size =
        st_Blocks_get_total_num_bytes( &particles_buffer );
        
    test_data_out << end_particles_size;
    test_data_out.write(
        ( char* )st_Blocks_get_const_data_begin( &particles_buffer ),
        end_particles_size );
    
    test_data_out.close();
    
    /* --------------------------------------------------------------------- */
    
    particles = nullptr;
    initial_particles = nullptr;
    
    st_Blocks_free( &elem_by_elem );
    st_Blocks_free( &copy_particles_buffer );
    st_Blocks_free( &beam_elements );
    st_Blocks_free( &particles_buffer );
}

/* ************************************************************************* */


/* end: sixtracklib/common/tests/test_particles.cpp */
