#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>

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

/* end: sixtracklib/common/tests/test_particles.cpp */
