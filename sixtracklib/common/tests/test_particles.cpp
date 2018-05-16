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
#include "sixtracklib/common/particles.h"
#include "sixtracklib/common/tests/test_particles_tools.h"
#include "sixtracklib/common/impl/particles_impl.h"
#include "sixtracklib/common/particles.h"
#include "sixtracklib/common/details/random.h"

#if defined( __SAVED_NAMESPACE )
    #undef __NAMESPACE
    #define __NAMESPACE __SAVED_NAMESPACE
#endif /* defined( __SAVED_NAMESPACE ) */

/* ========================================================================= */
/* ====  Test random initialization of particles                             */

TEST( ParticlesTests, RandomInitializationAndCopy )
{
    st_block_num_elements_t const NUM_PARTICLES = 
        ( st_block_num_elements_t )1000u;
        
    st_block_size_t const REAL_ATTRIBUTE_SIZE = 
        NUM_PARTICLES * sizeof( SIXTRL_REAL_T );
        
    st_block_size_t const INT64_ATTRIBUTE_SIZE =
        NUM_PARTICLES * sizeof( SIXTRL_INT64_T );
        
    st_block_size_t const PARTICLES_DATA_CAPACITY = 
        2u * ( 16u * REAL_ATTRIBUTE_SIZE + 4u * INT64_ATTRIBUTE_SIZE );
        
    st_Particles particles;
    st_Particles particles_copy;    
    st_ParticlesContainer particles_buffer;
    
    /* --------------------------------------------------------------------- */
            
    st_ParticlesContainer_preset( &particles_buffer );
    st_ParticlesContainer_reserve_num_blocks( 
        &particles_buffer, 2 );
    
    st_ParticlesContainer_reserve_for_data( 
        &particles_buffer, PARTICLES_DATA_CAPACITY );
    
    st_Particles_preset( &particles );
    int ret = st_ParticlesContainer_add_particles( 
        &particles_buffer, &particles, NUM_PARTICLES );
    
    ASSERT_TRUE( ret == 0 );
    
    ASSERT_TRUE( st_Particles_is_aligned_with( &particles, 
                 st_ParticlesContainer_get_data_alignment( 
                    &particles_buffer ) ) );
    
    ASSERT_TRUE( st_ParticlesContainer_get_num_of_blocks( 
                 &particles_buffer ) == 1 );
    
    ASSERT_TRUE( st_Particles_get_num_particles( &particles ) == 
                 NUM_PARTICLES );
    
    ASSERT_TRUE( st_Particles_get_type_id( &particles ) == 
                 st_BLOCK_TYPE_PARTICLE );
    
    /* --------------------------------------------------------------------- */
    
    uint64_t seed = UINT64_C( 20180420 );
    st_Random_init_genrand64( seed );
    
    st_Particles_random_init( &particles );
    
    /* --------------------------------------------------------------------- */
    
    st_Particles_preset( &particles_copy );
    
    ret = st_ParticlesContainer_add_particles(
        &particles_buffer, &particles_copy, NUM_PARTICLES );
    
    ASSERT_TRUE( ret == 0 );
    
    ASSERT_TRUE( st_Particles_is_aligned_with( &particles_copy,
                 st_ParticlesContainer_get_data_alignment( 
                    &particles_buffer ) ) );
    
    ASSERT_TRUE( st_ParticlesContainer_get_num_of_blocks( 
                 &particles_buffer ) == 2 );
    
    ASSERT_TRUE( st_Particles_get_num_particles( &particles_copy ) == 
                 NUM_PARTICLES );
    
    ASSERT_TRUE( st_Particles_get_type_id( &particles_copy ) == 
                 st_BLOCK_TYPE_PARTICLE );
    
    st_Particles_copy_all_unchecked( &particles_copy, &particles );
    
    /* --------------------------------------------------------------------- */
    
    int cmp_result = std::memcmp( st_Particles_get_const_q0( &particles ), 
        st_Particles_get_const_q0( &particles_copy ), REAL_ATTRIBUTE_SIZE );
    
    ASSERT_TRUE( cmp_result == 0 );
    ASSERT_TRUE( st_Particles_get_const_q0( &particles ) !=
                 st_Particles_get_const_q0( &particles_copy ) );
    
    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    
    cmp_result = std::memcmp( st_Particles_get_const_mass0( &particles ), 
        st_Particles_get_const_mass0( &particles_copy ), REAL_ATTRIBUTE_SIZE );
    
    ASSERT_TRUE( cmp_result == 0 );
    ASSERT_TRUE( st_Particles_get_const_mass0( &particles ) !=
                 st_Particles_get_const_mass0( &particles_copy ) );
    
    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    
    cmp_result = std::memcmp( st_Particles_get_const_beta0( &particles ), 
        st_Particles_get_const_beta0( &particles_copy ), REAL_ATTRIBUTE_SIZE );
    
    ASSERT_TRUE( cmp_result == 0 );
    ASSERT_TRUE( st_Particles_get_const_beta0( &particles ) !=
                 st_Particles_get_const_beta0( &particles_copy ) );
    
    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    
    cmp_result = std::memcmp( st_Particles_get_const_gamma0( &particles ), 
        st_Particles_get_const_gamma0( &particles_copy ), REAL_ATTRIBUTE_SIZE );
    
    ASSERT_TRUE( cmp_result == 0 );
    ASSERT_TRUE( st_Particles_get_const_gamma0( &particles ) !=
                 st_Particles_get_const_gamma0( &particles_copy ) );
    
    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    
    cmp_result = std::memcmp( st_Particles_get_const_p0c( &particles ), 
        st_Particles_get_const_p0c( &particles_copy ), REAL_ATTRIBUTE_SIZE );
    
    ASSERT_TRUE( cmp_result == 0 );
    ASSERT_TRUE( st_Particles_get_const_p0c( &particles ) !=
                 st_Particles_get_const_p0c( &particles_copy ) );
    
    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    
    cmp_result = std::memcmp( st_Particles_get_const_s( &particles ), 
        st_Particles_get_const_s( &particles_copy ), REAL_ATTRIBUTE_SIZE );
    
    ASSERT_TRUE( cmp_result == 0 );
    ASSERT_TRUE( st_Particles_get_const_s( &particles ) !=
                 st_Particles_get_const_s( &particles_copy ) );
    
    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    
    cmp_result = std::memcmp( st_Particles_get_const_x( &particles ), 
        st_Particles_get_const_x( &particles_copy ), REAL_ATTRIBUTE_SIZE );
    
    ASSERT_TRUE( cmp_result == 0 );
    ASSERT_TRUE( st_Particles_get_const_x( &particles ) !=
                 st_Particles_get_const_x( &particles_copy ) );
    
    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    
    cmp_result = std::memcmp( st_Particles_get_const_y( &particles ), 
        st_Particles_get_const_y( &particles_copy ), REAL_ATTRIBUTE_SIZE );
    
    ASSERT_TRUE( cmp_result == 0 );
    ASSERT_TRUE( st_Particles_get_const_y( &particles ) !=
                 st_Particles_get_const_y( &particles_copy ) );
    
    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    
    cmp_result = std::memcmp( st_Particles_get_const_px( &particles ), 
        st_Particles_get_const_px( &particles_copy ), REAL_ATTRIBUTE_SIZE );
    
    ASSERT_TRUE( cmp_result == 0 );
    ASSERT_TRUE( st_Particles_get_const_px( &particles ) !=
                 st_Particles_get_const_px( &particles_copy ) );
    
    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    
    cmp_result = std::memcmp( st_Particles_get_const_py( &particles ), 
        st_Particles_get_const_py( &particles_copy ), REAL_ATTRIBUTE_SIZE );
    
    ASSERT_TRUE( cmp_result == 0 );
    ASSERT_TRUE( st_Particles_get_const_py( &particles ) !=
                 st_Particles_get_const_py( &particles_copy ) );
    
    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    
    cmp_result = std::memcmp( st_Particles_get_const_sigma( &particles ), 
        st_Particles_get_const_sigma( &particles_copy ), REAL_ATTRIBUTE_SIZE );
    
    ASSERT_TRUE( cmp_result == 0 );
    ASSERT_TRUE( st_Particles_get_const_sigma( &particles ) !=
                 st_Particles_get_const_sigma( &particles_copy ) );
    
    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    
    cmp_result = std::memcmp( st_Particles_get_const_psigma( &particles ), 
        st_Particles_get_const_psigma( &particles_copy ), REAL_ATTRIBUTE_SIZE );
    
    ASSERT_TRUE( cmp_result == 0 );
    ASSERT_TRUE( st_Particles_get_const_psigma( &particles ) !=
                 st_Particles_get_const_psigma( &particles_copy ) );
    
    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    
    cmp_result = std::memcmp( st_Particles_get_const_delta( &particles ), 
        st_Particles_get_const_delta( &particles_copy ), REAL_ATTRIBUTE_SIZE );
    
    ASSERT_TRUE( cmp_result == 0 );
    ASSERT_TRUE( st_Particles_get_const_delta( &particles ) !=
                 st_Particles_get_const_delta( &particles_copy ) );
    
    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    
    cmp_result = std::memcmp( st_Particles_get_const_rpp( &particles ), 
        st_Particles_get_const_rpp( &particles_copy ), REAL_ATTRIBUTE_SIZE );
    
    ASSERT_TRUE( cmp_result == 0 );
    ASSERT_TRUE( st_Particles_get_const_rpp( &particles ) !=
                 st_Particles_get_const_rpp( &particles_copy ) );
    
    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    
    cmp_result = std::memcmp( st_Particles_get_const_rvv( &particles ), 
        st_Particles_get_const_rvv( &particles_copy ), REAL_ATTRIBUTE_SIZE );
    
    ASSERT_TRUE( cmp_result == 0 );
    ASSERT_TRUE( st_Particles_get_const_rvv( &particles ) !=
                 st_Particles_get_const_rvv( &particles_copy ) );
    
    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    
    cmp_result = std::memcmp( st_Particles_get_const_chi( &particles ), 
        st_Particles_get_const_chi( &particles_copy ), REAL_ATTRIBUTE_SIZE );
    
    ASSERT_TRUE( cmp_result == 0 );
    ASSERT_TRUE( st_Particles_get_const_chi( &particles ) !=
                 st_Particles_get_const_chi( &particles_copy ) );
    
    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    
    cmp_result = std::memcmp( st_Particles_get_const_particle_id( &particles ), 
        st_Particles_get_const_particle_id( &particles_copy ), 
            INT64_ATTRIBUTE_SIZE );
    
    ASSERT_TRUE( cmp_result == 0 );
    ASSERT_TRUE( st_Particles_get_const_particle_id( &particles ) !=
                 st_Particles_get_const_particle_id( &particles_copy ) );
    
    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    
    cmp_result = std::memcmp( st_Particles_get_const_lost_at_element_id( 
        &particles ), st_Particles_get_const_lost_at_element_id( 
            &particles_copy ), INT64_ATTRIBUTE_SIZE );
    
    ASSERT_TRUE( cmp_result == 0 );
    ASSERT_TRUE( st_Particles_get_const_lost_at_element_id( &particles ) !=
                 st_Particles_get_const_lost_at_element_id( &particles_copy ) );
    
    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    
    cmp_result = std::memcmp( st_Particles_get_const_lost_at_turn( &particles ), 
        st_Particles_get_const_lost_at_turn( &particles_copy ), 
            INT64_ATTRIBUTE_SIZE );
    
    ASSERT_TRUE( cmp_result == 0 );
    ASSERT_TRUE( st_Particles_get_const_lost_at_turn( &particles ) !=
                 st_Particles_get_const_lost_at_turn( &particles_copy ) );
    
    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    
    cmp_result = std::memcmp( st_Particles_get_const_state( &particles ), 
        st_Particles_get_const_state( &particles_copy ), 
            INT64_ATTRIBUTE_SIZE );
    
    ASSERT_TRUE( cmp_result == 0 );
    ASSERT_TRUE( st_Particles_get_const_state( &particles ) !=
                 st_Particles_get_const_state( &particles_copy ) );
    
    /* --------------------------------------------------------------------- */
    
    st_ParticlesContainer_free( &particles_buffer );
}

/* ========================================================================= */
/* ====  test: test initialization and unmapping of particles                */

TEST( ParticlesTests, RandomInitializationAndUnmapping )
{
    st_block_num_elements_t const NUM_PARTICLES = 
        ( st_block_num_elements_t )1000u;
        
    st_block_size_t const REAL_ATTRIBUTE_SIZE = 
        NUM_PARTICLES * sizeof( SIXTRL_REAL_T );
        
    st_block_size_t const INT64_ATTRIBUTE_SIZE =
        NUM_PARTICLES * sizeof( SIXTRL_INT64_T );
        
    st_block_size_t const PARTICLES_DATA_CAPACITY = 
        ( 16u * REAL_ATTRIBUTE_SIZE + 4u * INT64_ATTRIBUTE_SIZE );
        
    st_Particles particles;
    st_Particles particles_unmapped;    
    st_ParticlesContainer particles_buffer;
    
    /* --------------------------------------------------------------------- */
            
    st_ParticlesContainer_preset( &particles_buffer );
    st_ParticlesContainer_reserve_num_blocks( 
        &particles_buffer, 1 );
    
    st_ParticlesContainer_reserve_for_data( 
        &particles_buffer, PARTICLES_DATA_CAPACITY );
    
    st_Particles_preset( &particles );
    int ret = st_ParticlesContainer_add_particles( 
        &particles_buffer, &particles, NUM_PARTICLES );
    
    ASSERT_TRUE( ret == 0 );
    
    ASSERT_TRUE( st_Particles_is_aligned_with( &particles, 
                 st_ParticlesContainer_get_data_alignment( 
                    &particles_buffer ) ) );
    
    ASSERT_TRUE( st_ParticlesContainer_get_num_of_blocks( 
                 &particles_buffer ) == 1 );
    
    ASSERT_TRUE( st_Particles_get_num_particles( &particles ) == 
                 NUM_PARTICLES );
    
    ASSERT_TRUE( st_Particles_get_type_id( &particles ) == 
                 st_BLOCK_TYPE_PARTICLE );
    
    /* --------------------------------------------------------------------- */
    
    uint64_t seed = UINT64_C( 20180420 );
    st_Random_init_genrand64( seed );
    
    st_Particles_random_init( &particles );
    
    /* --------------------------------------------------------------------- */
    
    st_Particles_preset( &particles_unmapped );
    
    ret = st_Particles_remap_from_memory(
        &particles_unmapped, 
        st_ParticlesContainer_get_block_infos_begin( &particles_buffer ),
        st_ParticlesContainer_get_ptr_data_begin( &particles_buffer ),
        st_ParticlesContainer_get_data_capacity( &particles_buffer ) );
    
    ASSERT_TRUE( ret == 0 );
    
    ASSERT_TRUE( st_Particles_has_mapping( &particles_unmapped ) );
    
    ASSERT_TRUE( st_Particles_is_aligned_with( &particles_unmapped,
                 st_ParticlesContainer_get_data_alignment( 
                    &particles_buffer ) ) );
    
    ASSERT_TRUE( st_ParticlesContainer_get_num_of_blocks( 
                 &particles_buffer ) == 1 );
    
    ASSERT_TRUE( st_Particles_get_num_particles( &particles_unmapped ) == 
                 NUM_PARTICLES );
    
    ASSERT_TRUE( st_Particles_get_type_id( &particles_unmapped ) == 
                 st_BLOCK_TYPE_PARTICLE );
    
    /* --------------------------------------------------------------------- */
    
    ASSERT_TRUE( st_Particles_get_const_q0( &particles ) ==
                 st_Particles_get_const_q0( &particles_unmapped ) );
    
    ASSERT_TRUE( st_Particles_get_const_mass0( &particles ) ==
                 st_Particles_get_const_mass0( &particles_unmapped ) );
    
    ASSERT_TRUE( st_Particles_get_const_beta0( &particles ) ==
                 st_Particles_get_const_beta0( &particles_unmapped ) );
    
    ASSERT_TRUE( st_Particles_get_const_gamma0( &particles ) ==
                 st_Particles_get_const_gamma0( &particles_unmapped ) );
    
    ASSERT_TRUE( st_Particles_get_const_p0c( &particles ) ==
                 st_Particles_get_const_p0c( &particles_unmapped ) );
    
    ASSERT_TRUE( st_Particles_get_const_s( &particles ) ==
                 st_Particles_get_const_s( &particles_unmapped ) );
    
    ASSERT_TRUE( st_Particles_get_const_x( &particles ) ==
                 st_Particles_get_const_x( &particles_unmapped ) );
    
    ASSERT_TRUE( st_Particles_get_const_y( &particles ) ==
                 st_Particles_get_const_y( &particles_unmapped ) );
    
    ASSERT_TRUE( st_Particles_get_const_px( &particles ) ==
                 st_Particles_get_const_px( &particles_unmapped ) );
    
    ASSERT_TRUE( st_Particles_get_const_py( &particles ) ==
                 st_Particles_get_const_py( &particles_unmapped ) );
    
    ASSERT_TRUE( st_Particles_get_const_sigma( &particles ) ==
                 st_Particles_get_const_sigma( &particles_unmapped ) );
    
    ASSERT_TRUE( st_Particles_get_const_psigma( &particles ) ==
                 st_Particles_get_const_psigma( &particles_unmapped ) );
    
    ASSERT_TRUE( st_Particles_get_const_delta( &particles ) ==
                 st_Particles_get_const_delta( &particles_unmapped ) );
    
    ASSERT_TRUE( st_Particles_get_const_rpp( &particles ) ==
                 st_Particles_get_const_rpp( &particles_unmapped ) );
    
    ASSERT_TRUE( st_Particles_get_const_rvv( &particles ) ==
                 st_Particles_get_const_rvv( &particles_unmapped ) );
    
    ASSERT_TRUE( st_Particles_get_const_chi( &particles ) ==
                 st_Particles_get_const_chi( &particles_unmapped ) );
    
    ASSERT_TRUE( st_Particles_get_const_particle_id( &particles ) ==
                 st_Particles_get_const_particle_id( &particles_unmapped ) );
    
    ASSERT_TRUE( st_Particles_get_const_lost_at_element_id( &particles ) ==
                 st_Particles_get_const_lost_at_element_id( &particles_unmapped ) );
    
    ASSERT_TRUE( st_Particles_get_const_lost_at_turn( &particles ) ==
                 st_Particles_get_const_lost_at_turn( &particles_unmapped ) );
    
    ASSERT_TRUE( st_Particles_get_const_state( &particles ) ==
                 st_Particles_get_const_state( &particles_unmapped ) );
    
    /* --------------------------------------------------------------------- */
    
    st_ParticlesContainer_free( &particles_buffer );
}

/* ========================================================================= */
/* ====  test: test writing and reading from binary file                     */

TEST( ParticlesTests, RandomInitializationAndWritingAndReadingBinaryFiles )
{
    int ret = 0;
    
    FILE* fp = nullptr;
    
    st_block_num_elements_t const NUM_PARTICLES = 
        ( st_block_num_elements_t )1000u;
        
    st_block_size_t const REAL_ATTRIBUTE_SIZE = 
        NUM_PARTICLES * sizeof( SIXTRL_REAL_T );
        
    st_block_size_t const INT64_ATTRIBUTE_SIZE =
        NUM_PARTICLES * sizeof( SIXTRL_INT64_T );
        
    st_block_size_t const PARTICLES_DATA_CAPACITY = 
        ( 16u * REAL_ATTRIBUTE_SIZE + 4u * INT64_ATTRIBUTE_SIZE );
        
    st_Particles particles;
    st_ParticlesContainer particles_buffer;
    
    st_Particles restored_particles;
    st_ParticlesContainer restored_buffer;
    
    /* --------------------------------------------------------------------- */
            
    st_ParticlesContainer_preset( &particles_buffer );
    
    ret = st_ParticlesContainer_init( 
        &particles_buffer, 1u, PARTICLES_DATA_CAPACITY );
    
    ASSERT_TRUE( ret == 0 );
    
    ret = st_ParticlesContainer_add_particles( 
        &particles_buffer, &particles, NUM_PARTICLES );
    
    ASSERT_TRUE( ret == 0 );
    
    /* --------------------------------------------------------------------- */
    
    uint64_t seed = UINT64_C( 20180420 );
    st_Random_init_genrand64( seed );
    
    st_Particles_random_init( &particles );
    
    /* --------------------------------------------------------------------- */
    
    fp = std::fopen( "./test_particles_dump.bin", "wb" );
    
    ASSERT_TRUE( fp != nullptr );
    
    ret = st_Particles_write_to_bin_file( fp, &particles );
    ASSERT_TRUE( ret == 0 );
    
    std::fclose( fp );
    fp = nullptr;
    
    /* --------------------------------------------------------------------- */
    
    fp = std::fopen( "./test_particles_dump.bin", "rb" );
    
    ASSERT_TRUE( fp != nullptr );
    
    st_block_num_elements_t const RESTORE_NUM_PARTICLES = 
        st_Particles_get_next_num_particles_from_bin_file( fp );
        
    ASSERT_TRUE( RESTORE_NUM_PARTICLES == NUM_PARTICLES );
    
    st_ParticlesContainer_preset( &restored_buffer );
    
    ret = st_ParticlesContainer_init(
        &restored_buffer, 1u, PARTICLES_DATA_CAPACITY );
    
    ASSERT_TRUE( ret == 0 );
    
    ret = st_ParticlesContainer_add_particles(
        &restored_buffer, &restored_particles, RESTORE_NUM_PARTICLES );
    
    ASSERT_TRUE( ret == 0 );
    
    ret = st_Particles_read_from_bin_file( fp, &restored_particles );
    
    std::fclose( fp );
    fp = nullptr;
    
    /* --------------------------------------------------------------------- */
    
    int cmp_result = std::memcmp( st_Particles_get_const_q0( &particles ), 
        st_Particles_get_const_q0( &restored_particles ), REAL_ATTRIBUTE_SIZE );
    
    ASSERT_TRUE( cmp_result == 0 );
    ASSERT_TRUE( st_Particles_get_const_q0( &particles ) !=
                 st_Particles_get_const_q0( &restored_particles ) );
    
    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    
    cmp_result = std::memcmp( st_Particles_get_const_mass0( &particles ), 
        st_Particles_get_const_mass0( &restored_particles ), 
            REAL_ATTRIBUTE_SIZE );
    
    ASSERT_TRUE( cmp_result == 0 );
    ASSERT_TRUE( st_Particles_get_const_mass0( &particles ) !=
                 st_Particles_get_const_mass0( &restored_particles ) );
    
    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    
    cmp_result = std::memcmp( st_Particles_get_const_beta0( &particles ), 
        st_Particles_get_const_beta0( &restored_particles ), 
            REAL_ATTRIBUTE_SIZE );
    
    ASSERT_TRUE( cmp_result == 0 );
    ASSERT_TRUE( st_Particles_get_const_beta0( &particles ) !=
                 st_Particles_get_const_beta0( &restored_particles ) );
    
    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    
    cmp_result = std::memcmp( st_Particles_get_const_gamma0( &particles ), 
        st_Particles_get_const_gamma0( &restored_particles ), 
            REAL_ATTRIBUTE_SIZE );
    
    ASSERT_TRUE( cmp_result == 0 );
    ASSERT_TRUE( st_Particles_get_const_gamma0( &particles ) !=
                 st_Particles_get_const_gamma0( &restored_particles ) );
    
    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    
    cmp_result = std::memcmp( st_Particles_get_const_p0c( &particles ), 
        st_Particles_get_const_p0c( &restored_particles ), 
            REAL_ATTRIBUTE_SIZE );
    
    ASSERT_TRUE( cmp_result == 0 );
    ASSERT_TRUE( st_Particles_get_const_p0c( &particles ) !=
                 st_Particles_get_const_p0c( &restored_particles ) );
    
    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    
    cmp_result = std::memcmp( st_Particles_get_const_s( &particles ), 
        st_Particles_get_const_s( &restored_particles ), REAL_ATTRIBUTE_SIZE );
    
    ASSERT_TRUE( cmp_result == 0 );
    ASSERT_TRUE( st_Particles_get_const_s( &particles ) !=
                 st_Particles_get_const_s( &restored_particles ) );
    
    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    
    cmp_result = std::memcmp( st_Particles_get_const_x( &particles ), 
        st_Particles_get_const_x( &restored_particles ), REAL_ATTRIBUTE_SIZE );
    
    ASSERT_TRUE( cmp_result == 0 );
    ASSERT_TRUE( st_Particles_get_const_x( &particles ) !=
                 st_Particles_get_const_x( &restored_particles ) );
    
    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    
    cmp_result = std::memcmp( st_Particles_get_const_y( &particles ), 
        st_Particles_get_const_y( &restored_particles ), REAL_ATTRIBUTE_SIZE );
    
    ASSERT_TRUE( cmp_result == 0 );
    ASSERT_TRUE( st_Particles_get_const_y( &particles ) !=
                 st_Particles_get_const_y( &restored_particles ) );
    
    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    
    cmp_result = std::memcmp( st_Particles_get_const_px( &particles ), 
        st_Particles_get_const_px( &restored_particles ), 
            REAL_ATTRIBUTE_SIZE );
    
    ASSERT_TRUE( cmp_result == 0 );
    ASSERT_TRUE( st_Particles_get_const_px( &particles ) !=
                 st_Particles_get_const_px( &restored_particles ) );
    
    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    
    cmp_result = std::memcmp( st_Particles_get_const_py( &particles ), 
        st_Particles_get_const_py( &restored_particles ), 
            REAL_ATTRIBUTE_SIZE );
    
    ASSERT_TRUE( cmp_result == 0 );
    ASSERT_TRUE( st_Particles_get_const_py( &particles ) !=
                 st_Particles_get_const_py( &restored_particles ) );
    
    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    
    cmp_result = std::memcmp( st_Particles_get_const_sigma( &particles ), 
        st_Particles_get_const_sigma( &restored_particles ), 
            REAL_ATTRIBUTE_SIZE );
    
    ASSERT_TRUE( cmp_result == 0 );
    ASSERT_TRUE( st_Particles_get_const_sigma( &particles ) !=
                 st_Particles_get_const_sigma( &restored_particles ) );
    
    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    
    cmp_result = std::memcmp( st_Particles_get_const_psigma( &particles ), 
        st_Particles_get_const_psigma( &restored_particles ), 
            REAL_ATTRIBUTE_SIZE );
    
    ASSERT_TRUE( cmp_result == 0 );
    ASSERT_TRUE( st_Particles_get_const_psigma( &particles ) !=
                 st_Particles_get_const_psigma( &restored_particles ) );
    
    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    
    cmp_result = std::memcmp( st_Particles_get_const_delta( &particles ), 
        st_Particles_get_const_delta( &restored_particles ), 
            REAL_ATTRIBUTE_SIZE );
    
    ASSERT_TRUE( cmp_result == 0 );
    ASSERT_TRUE( st_Particles_get_const_delta( &particles ) !=
                 st_Particles_get_const_delta( &restored_particles ) );
    
    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    
    cmp_result = std::memcmp( st_Particles_get_const_rpp( &particles ), 
        st_Particles_get_const_rpp( &restored_particles ), 
            REAL_ATTRIBUTE_SIZE );
    
    ASSERT_TRUE( cmp_result == 0 );
    ASSERT_TRUE( st_Particles_get_const_rpp( &particles ) !=
                 st_Particles_get_const_rpp( &restored_particles ) );
    
    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    
    cmp_result = std::memcmp( st_Particles_get_const_rvv( &particles ), 
        st_Particles_get_const_rvv( &restored_particles ), 
            REAL_ATTRIBUTE_SIZE );
    
    ASSERT_TRUE( cmp_result == 0 );
    ASSERT_TRUE( st_Particles_get_const_rvv( &particles ) !=
                 st_Particles_get_const_rvv( &restored_particles ) );
    
    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    
    cmp_result = std::memcmp( st_Particles_get_const_chi( &particles ), 
        st_Particles_get_const_chi( &restored_particles ), 
            REAL_ATTRIBUTE_SIZE );
    
    ASSERT_TRUE( cmp_result == 0 );
    ASSERT_TRUE( st_Particles_get_const_chi( &particles ) !=
                 st_Particles_get_const_chi( &restored_particles ) );
    
    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    
    cmp_result = std::memcmp( st_Particles_get_const_particle_id( &particles ), 
        st_Particles_get_const_particle_id( &restored_particles ), 
            INT64_ATTRIBUTE_SIZE );
    
    ASSERT_TRUE( cmp_result == 0 );
    ASSERT_TRUE( st_Particles_get_const_particle_id( &particles ) !=
                 st_Particles_get_const_particle_id( &restored_particles ) );
    
    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    
    cmp_result = std::memcmp( st_Particles_get_const_lost_at_element_id( 
        &particles ), st_Particles_get_const_lost_at_element_id( 
            &restored_particles ), INT64_ATTRIBUTE_SIZE );
    
    ASSERT_TRUE( cmp_result == 0 );
    ASSERT_TRUE( st_Particles_get_const_lost_at_element_id( 
                    &particles ) !=
                 st_Particles_get_const_lost_at_element_id( 
                    &restored_particles ) );
    
    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    
    cmp_result = std::memcmp( st_Particles_get_const_lost_at_turn( 
        &particles ), st_Particles_get_const_lost_at_turn( 
            &restored_particles ), INT64_ATTRIBUTE_SIZE );
    
    ASSERT_TRUE( cmp_result == 0 );
    ASSERT_TRUE( st_Particles_get_const_lost_at_turn( &particles ) !=
                 st_Particles_get_const_lost_at_turn( &restored_particles ) );
    
    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    
    cmp_result = std::memcmp( st_Particles_get_const_state( &particles ), 
        st_Particles_get_const_state( &restored_particles ), 
            INT64_ATTRIBUTE_SIZE );
    
    ASSERT_TRUE( cmp_result == 0 );
    ASSERT_TRUE( st_Particles_get_const_state( &particles ) !=
                 st_Particles_get_const_state( &restored_particles ) );
    
    /* --------------------------------------------------------------------- */
    
    st_ParticlesContainer_free( &particles_buffer );
    st_ParticlesContainer_free( &restored_buffer );
}

/* end: sixtracklib/common/tests/test_particles.cpp */
