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
#include "sixtracklib/opencl/ocl_environment.h"

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
        
    /* ******************************************************************** */
    /* *****                 OpenCL based tracking                    ***** */
    /* ******************************************************************** */
    
    st_OclEnvironment ocl_env;
    
    if( ocl_env.numAvailableNodes() > 0u )
    {
        std::size_t const num_nodes = ocl_env.numAvailableNodes();
        
        std::cout << "found " << num_nodes  
                  << "number of nodes -> repeat unit-test for each node!" 
                  << std::endl << std::endl;
                  
        auto node_it  = ocl_env.constAvailableNodesBegin();
        auto node_end = ocl_env.constAvailableNodesEnd();
        
        std::size_t ii = 0u;
        
        for( ; node_it != node_end ; ++node_it, ++ii )
        {
            st_ComputeNodeInfo const* info = 
                    ocl_env.getPtrNodeInfo( *node_it );
            
            if( info != nullptr )
            {
                std::cout 
                    << "running the test for node " 
                    << ii << " / " << num_nodes << ":" << "\r\n\r\n"
                    << "platform_id : " << st_ComputeNodeInfo_get_platform_id( info )
                    << "\r\n"
                    << "device_id   : " << st_ComputeNodeInfo_get_device_id( info )
                    << "\r\n"
                    << "arch        : " << st_ComputeNodeInfo_get_arch( info )
                    << "\r\n"
                    << "platform    : " << st_ComputeNodeInfo_get_platform( info )
                    << "\r\n"
                    << "device name : " << st_ComputeNodeInfo_get_name( info )
                    << "\r\n"
                    << "description : " << st_ComputeNodeInfo_get_description( info )
                    << "\r\n"
                    << "\r\n";
                
                
                st_Blocks particles_buffer;
                st_Blocks_preset( &particles_buffer );
                
                ASSERT_TRUE( 0 == st_Blocks_init_from_serialized_data( 
                    &particles_buffer, 
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
                    
                bool const prepare_success = ocl_env.prepareParticlesTracking( 
                    particles_buffer, beam_elements, &calculated_elem_by_elem_buffer, 
                    node_it, 1 );
                
                ASSERT_TRUE( prepare_success );
                
                bool const run_success = ocl_env.runParticlesTracking(
                    NUM_OF_TURNS, particles_buffer, beam_elements, 
                    &calculated_elem_by_elem_buffer );
                
                ASSERT_TRUE( run_success );
                
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
                
                st_Blocks_free( &calculated_elem_by_elem_buffer );
                st_Blocks_free( &particles_buffer );    
            }                    
        }
    }
    
    /* --------------------------------------------------------------------- */
    
    st_Blocks_free( &initial_particles_buffer );
    st_Blocks_free( &result_particles_buffer );
    st_Blocks_free( &beam_elements );
    st_Blocks_free( &elem_by_elem_buffer );    
}

/* ************************************************************************* */


/* end: sixtracklib/opencl/tests/test_track.cpp */
