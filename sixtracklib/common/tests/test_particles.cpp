#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>

#include <gtest/gtest.h>

#if defined( __NAMESPACE )
    #define __SAVED_NAMESPACE __NAMESPACE
    #undef  __NAMESPACE     
#endif /* !defiend( __NAMESPACE ) */

#if !defined( __NAMESPACE )
    #define __NAMESPACE st_    
#endif /* !defiend( __NAMESPACE ) */

#include "sixtracklib/_impl/definitions.h"
#include "sixtracklib/common/blocks.h"
#include "sixtracklib/common/particles.h"
#include "sixtracklib/common/details/random.h"
#include "sixtracklib/common/tests/test_particles_tools.h"

#if defined( __SAVED_NAMESPACE )
    #undef __NAMESPACE
    #define __NAMESPACE __SAVED_NAMESPACE
#endif /* defined( __SAVED_NAMESPACE ) */

/* ========================================================================= */
/* ====  Test random initialization of particles                             */

TEST( ParticlesTests, RandomInitParticlesCopyAndCompare )
{
    uint64_t seed = UINT64_C( 20180420 );
    st_Random_init_genrand64( seed );
    
    /* --------------------------------------------------------------------- */
    
    st_Blocks particles_buffer;
    st_Blocks_preset( &particles_buffer );
    
    st_block_size_t const NUM_BLOCKS = 2u;
    
    st_block_num_elements_t const NUM_PARTICLES = 
        ( st_block_num_elements_t )1000u;
        
    st_block_size_t const PARTICLES_DATA_CAPACITY = 
        st_Blocks_predict_data_capacity_for_num_blocks( 
            &particles_buffer, NUM_BLOCKS ) + 
        NUM_BLOCKS * st_Particles_predict_blocks_data_capacity( 
            &particles_buffer, NUM_PARTICLES );
    
    /* --------------------------------------------------------------------- */
    
    int ret = st_Blocks_init( 
        &particles_buffer, NUM_BLOCKS, PARTICLES_DATA_CAPACITY );
    
    ASSERT_TRUE( ret == 0 );
    
    st_Particles* particles = st_Blocks_add_particles(
        &particles_buffer, NUM_PARTICLES );
    
    ASSERT_TRUE( particles != nullptr );    
    
    ASSERT_TRUE( st_Blocks_get_num_of_blocks( &particles_buffer ) == 1u );
    
    ASSERT_TRUE( st_Particles_get_num_particles( particles ) == 
                 NUM_PARTICLES );    
    
    ASSERT_TRUE( st_Particles_get_type_id( particles ) == 
                 st_BLOCK_TYPE_PARTICLE );
    
    st_Particles_random_init( particles );
    
    /* --------------------------------------------------------------------- */
    
    st_Particles* particles_copy = st_Blocks_add_particles( 
        &particles_buffer, NUM_PARTICLES );
    
    ASSERT_TRUE( particles_copy != nullptr );
    ASSERT_TRUE( st_Blocks_get_num_of_blocks( &particles_buffer ) == 2 );    
    
    ASSERT_TRUE( st_Particles_get_num_particles( particles_copy ) == 
                 NUM_PARTICLES );
    
    ASSERT_TRUE( st_Particles_get_type_id( particles_copy ) == 
                 st_BLOCK_TYPE_PARTICLE );
    
    /* --------------------------------------------------------------------- */
    
    st_Particles_copy_all_unchecked( particles_copy, particles );
    
    ASSERT_TRUE( st_Particles_have_same_structure( 
        particles_copy, particles ) );
    
    ASSERT_TRUE( !st_Particles_map_to_same_memory( 
        particles_copy, particles ) );
    
    ASSERT_TRUE( 0 == st_Particles_compare_values( 
        particles_copy, particles ) );
    
    /* --------------------------------------------------------------------- */
    
    particles      = nullptr;
    particles_copy = nullptr;
    
    st_Blocks_free( &particles_buffer );
}

/* ========================================================================= */
/* ====  test: init, serialize and unserialize from same memory -> compare   */

TEST( ParticlesTests, RandomInitSerializationToUnserializationSameMemory )
{
    uint64_t seed = UINT64_C( 20180420 );
    st_Random_init_genrand64( seed );
    
    /* --------------------------------------------------------------------- */
        
    st_Blocks particles_buffer;
    st_Blocks_preset( &particles_buffer );
    
    st_block_size_t const NUM_BLOCKS = ( st_block_size_t )1u;
    
    st_block_num_elements_t const NUM_PARTICLES = 
        ( st_block_num_elements_t )1000u;
        
    st_block_size_t const PARTICLES_DATA_CAPACITY = 
        st_Blocks_predict_data_capacity_for_num_blocks( 
            &particles_buffer, NUM_BLOCKS ) +
        st_Particles_predict_blocks_data_capacity( 
            &particles_buffer, NUM_PARTICLES );
    
    int ret = st_Blocks_init( 
        &particles_buffer, NUM_BLOCKS, PARTICLES_DATA_CAPACITY );
    
    ASSERT_TRUE( ret == 0 );
    
    /* --------------------------------------------------------------------- */
    
    st_Particles* particles = st_Blocks_add_particles( 
        &particles_buffer, NUM_PARTICLES );
    
    ASSERT_TRUE( particles != nullptr );
    
    st_Particles_random_init( particles );
    
    ASSERT_TRUE( !st_Blocks_are_serialized( &particles_buffer ) );
    ASSERT_TRUE(  st_Blocks_serialize( &particles_buffer ) == 0 );
    ASSERT_TRUE(  st_Blocks_are_serialized( &particles_buffer ) );
    
    /* --------------------------------------------------------------------- */
    
    SIXTRL_GLOBAL_DEC unsigned char* serialized_particles_begin =
        st_Blocks_get_data_begin( &particles_buffer );
        
    ASSERT_TRUE( serialized_particles_begin != nullptr );
    
    st_Blocks ref_particles_buffer;
    st_Blocks_preset( &ref_particles_buffer );
    
    ASSERT_TRUE( st_Blocks_unserialize( 
        &ref_particles_buffer, serialized_particles_begin ) == 0 );    
    
    ASSERT_TRUE( st_Blocks_get_num_of_blocks( &ref_particles_buffer ) == 1u );
    
    SIXTRL_GLOBAL_DEC st_BlockInfo const* blocks_it  = 
        st_Blocks_get_const_block_infos_begin( &ref_particles_buffer );
        
    SIXTRL_GLOBAL_DEC st_BlockInfo const* blocks_end =
        st_Blocks_get_const_block_infos_end( &ref_particles_buffer );
    
    ASSERT_TRUE( std::distance( blocks_it, blocks_end ) == 1 );
        
    ASSERT_TRUE( st_BlockInfo_get_type_id( blocks_it ) == 
                 st_BLOCK_TYPE_PARTICLE );
    
    SIXTRL_GLOBAL_DEC st_Particles const* ref_particles = 
        st_Blocks_get_const_particles( blocks_it );
        
    ASSERT_TRUE( ref_particles != nullptr );
        
    /* --------------------------------------------------------------------- */
    
    ASSERT_TRUE( st_Particles_have_same_structure( 
        ref_particles, particles ) );
    
    ASSERT_TRUE( st_Particles_map_to_same_memory( 
        ref_particles, particles ) );
    
    ASSERT_TRUE( 0 == st_Particles_compare_values( 
        ref_particles, particles ) );
    
    /* --------------------------------------------------------------------- */
    
    particles = nullptr;
    ref_particles = nullptr;
    
    st_Blocks_free( &particles_buffer );
}

/* ========================================================================= */
/* ====  test: init, serialize, copy memory, unserialize -> compare   */

TEST( ParticlesTests, RandomInitSerializationCopyMemoryUnserializeCompare )
{
    uint64_t seed = UINT64_C( 20180420 );
    st_Random_init_genrand64( seed );
    
    /* --------------------------------------------------------------------- */
    
    st_Blocks particles_buffer;
    st_Blocks_preset( &particles_buffer );
    
    st_block_size_t const NUM_BLOCKS = ( st_block_size_t )1u;
    
    st_block_num_elements_t const NUM_PARTICLES = 
        ( st_block_num_elements_t )1000u;
        
    st_block_size_t const PARTICLES_DATA_CAPACITY = 
        st_Blocks_predict_data_capacity_for_num_blocks( 
            &particles_buffer, NUM_BLOCKS ) +
        st_Particles_predict_blocks_data_capacity( 
            &particles_buffer, NUM_PARTICLES );
        
    int ret = st_Blocks_init( 
        &particles_buffer, NUM_BLOCKS, PARTICLES_DATA_CAPACITY );
    
    ASSERT_TRUE( ret == 0 );
    
    /* --------------------------------------------------------------------- */
    
    st_Particles* particles = st_Blocks_add_particles( 
        &particles_buffer, NUM_PARTICLES );
    
    ASSERT_TRUE( particles != nullptr );
    
    st_Particles_random_init( particles );
    
    ASSERT_TRUE( !st_Blocks_are_serialized( &particles_buffer ) );
    ASSERT_TRUE(  st_Blocks_serialize( &particles_buffer ) == 0 );
    ASSERT_TRUE(  st_Blocks_are_serialized( &particles_buffer ) );
    
    /* --------------------------------------------------------------------- */
    
    std::vector< unsigned char > copied_raw_data(
        st_Blocks_get_const_data_begin( &particles_buffer ),
        st_Blocks_get_const_data_end( &particles_buffer ) );
    
    ASSERT_TRUE( copied_raw_data.size() == 
                 st_Blocks_get_total_num_bytes( &particles_buffer ) );
    
    st_Blocks copied_particles_buffer;
    st_Blocks_preset( &copied_particles_buffer );
    
    ret = st_Blocks_unserialize( 
        &copied_particles_buffer, copied_raw_data.data() );
    
    ASSERT_TRUE( ret == 0 );
    
    ASSERT_TRUE( st_Blocks_get_num_of_blocks( &copied_particles_buffer ) == 1u );
    
    SIXTRL_GLOBAL_DEC st_BlockInfo const* blocks_it  = 
        st_Blocks_get_const_block_infos_begin( 
            &copied_particles_buffer );
        
    SIXTRL_GLOBAL_DEC st_BlockInfo const* blocks_end =
        st_Blocks_get_const_block_infos_end( 
            &copied_particles_buffer );
    
    ASSERT_TRUE( std::distance( blocks_it, blocks_end ) == 1 );
        
    ASSERT_TRUE( st_BlockInfo_get_type_id( blocks_it ) == 
                 st_BLOCK_TYPE_PARTICLE );
    
    SIXTRL_GLOBAL_DEC st_Particles const* copied_particles = 
        st_Blocks_get_const_particles( blocks_it );
        
    ASSERT_TRUE( copied_particles != nullptr );
        
    /* --------------------------------------------------------------------- */
    
    ASSERT_TRUE( st_Particles_have_same_structure( 
        copied_particles, particles ) );
    
    ASSERT_TRUE( !st_Particles_map_to_same_memory( 
        copied_particles, particles ) );
    
    ASSERT_TRUE( 0 == st_Particles_compare_values( 
        copied_particles, particles ) );
    
    /* --------------------------------------------------------------------- */
    
    particles = nullptr;
    copied_particles = nullptr;
    
    st_Blocks_free( &particles_buffer );
    st_Blocks_free( &copied_particles_buffer );
}

/* end: sixtracklib/common/tests/test_particles.cpp */
