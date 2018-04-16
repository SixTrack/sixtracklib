#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>

#include <gtest/gtest.h>

#if defined( __NAMESPACE )
    #define __SAVED_NAMESPACE __NAMESPACE
    #undef  __NAMESPACE     
#endif /* !defiend( __NAMESPACE ) */

#if !defined( __NAMESPACE )
    #define __NAMESPACE st_    
#endif /* !defiend( __NAMESPACE ) */

#include "sixtracklib/_impl/definitions.h"
#include "sixtracklib/common/mem_pool.h"
#include "sixtracklib/common/particles.h"
#include "sixtracklib/common/impl/particles_type.h"
#include "sixtracklib/common/single_particle.h"

#if defined( __SAVED_NAMESPACE )
    #undef __NAMESPACE
    #define __NAMESPACE __SAVED_NAMESPACE
#endif /* defined( __SAVED_NAMESPACE ) */

/* ------------------------------------------------------------------------- */
/* --- Helper function for checking the consistency of a SingleParticle ---- */
/* --- instance with a SingleParticle based Particles container         ---- */

bool st_Particles_check_mapping_of_single_praticle(
     const st_SingleParticle *const SIXTRL_RESTRICT single_particle,
     const st_Particles *const SIXTRL_RESTRICT particles )
{
    bool is_mapped_correctly = false;
    
    if( ( single_particle != 0 ) && ( particles != 0 ) &&
        ( st_Particles_get_size( particles ) == std::size_t{ 1 } ) )
    {
        is_mapped_correctly   = ( &single_particle->q0 == 
            st_Particles_get_q0( particles ) );
        
        is_mapped_correctly  &= ( &single_particle->mass0 == 
                    st_Particles_get_mass0( particles ) );
        
        is_mapped_correctly  &= ( &single_particle->beta0 == 
                    st_Particles_get_beta0( particles ) );
        
        is_mapped_correctly  &= ( &single_particle->gamma0 == 
                    st_Particles_get_gamma0( particles ) );
        
        is_mapped_correctly  &= ( &single_particle->p0c == 
                    st_Particles_get_p0c( particles ) );
        
        is_mapped_correctly  &= ( &single_particle->partid == 
                    st_Particles_get_particle_id( particles ) );
        
        is_mapped_correctly  &= ( &single_particle->elemid == 
                    st_Particles_get_lost_at_element_id( particles ) );
        
        is_mapped_correctly  &= ( &single_particle->turn == 
                    st_Particles_get_lost_at_turn( particles ) );
        
        is_mapped_correctly  &= ( &single_particle->state == 
                    st_Particles_get_state( particles ) );
        
        is_mapped_correctly  &= ( &single_particle->s == 
                    st_Particles_get_s( particles ) );
        
        is_mapped_correctly  &= ( &single_particle->x == 
                    st_Particles_get_x( particles ) );
        
        is_mapped_correctly  &= ( &single_particle->y == 
                    st_Particles_get_y( particles ) );
        
        is_mapped_correctly  &= ( &single_particle->px == 
                    st_Particles_get_px( particles ) );
        
        is_mapped_correctly  &= ( &single_particle->py == 
                    st_Particles_get_py( particles ) );
        
        is_mapped_correctly  &= ( &single_particle->sigma == 
                    st_Particles_get_sigma( particles ) );
        
        is_mapped_correctly  &= ( &single_particle->psigma == 
                    st_Particles_get_psigma( particles ) );
        
        is_mapped_correctly  &= ( &single_particle->delta == 
                    st_Particles_get_delta( particles ) );
        
        is_mapped_correctly  &= ( &single_particle->rpp == 
                    st_Particles_get_rpp( particles ) );
        
        is_mapped_correctly  &= ( &single_particle->rvv == 
                    st_Particles_get_rvv( particles ) );
        
        is_mapped_correctly  &= ( &single_particle->chi == 
                    st_Particles_get_chi( particles ) );
    }
    
    return is_mapped_correctly;    
}


/* ========================================================================= */
/* ====  Test basic usage of Particles with self-managed mem-pool            */

TEST( ParticlesTests, InitOwnMemPoolBasic )
{
    std::size_t const NUM_OF_PARTICLES = std::size_t{ 4 };
    
    st_Particles* particles = st_Particles_new( NUM_OF_PARTICLES );
    ASSERT_TRUE( particles != nullptr );

    /* --------------------------------------------------------------------- */
    
    ASSERT_TRUE(  st_Particles_get_size( particles ) == NUM_OF_PARTICLES );
    ASSERT_TRUE(  st_Particles_is_consistent( particles ) );
    ASSERT_TRUE(  st_Particles_is_packed( particles ) );
    ASSERT_TRUE(  st_Particles_manages_own_memory( particles ) );
    ASSERT_TRUE(  st_Particles_uses_mempool( particles ) );
    ASSERT_TRUE( !st_Particles_uses_single_particle( particles ) );
    ASSERT_TRUE( !st_Particles_uses_flat_memory( particles ) );
        
    st_MemPool const* ptr_mem_pool = 
        st_Particles_get_const_mem_pool( particles );
        
    ASSERT_TRUE( ptr_mem_pool != nullptr );
    ASSERT_TRUE( st_MemPool_get_const_buffer( ptr_mem_pool ) != nullptr );
    ASSERT_TRUE( st_MemPool_get_chunk_size( ptr_mem_pool ) == 
                 st_PARTICLES_DEFAULT_MEMPOOL_CHUNK_SIZE );
    
    ASSERT_TRUE(  st_Particles_has_defined_alignment( particles ) );
    
    uint64_t const alignment  = st_Particles_alignment( particles );
    size_t   const chunk_size = st_MemPool_get_chunk_size( ptr_mem_pool );
    
    ASSERT_TRUE( alignment == st_PARTICLES_DEFAULT_MEMPOOL_ALIGNMENT );
    ASSERT_TRUE( ( size_t )alignment >= chunk_size );
    ASSERT_TRUE( st_Particles_check_alignment( particles, alignment ) );
    
    st_Particles_free( particles );
    free( particles );
    particles = nullptr;
}

/* ========================================================================= */
/* ====  Test basic usage of Particles with externally managed mem-pool      */

TEST( ParticlesTests, InitExtMemPoolBasic )
{
    std::size_t const NUM_OF_PARTICLES  = std::size_t{ 4 };
    
    std::size_t chunk_size = st_PARTICLES_DEFAULT_MEMPOOL_CHUNK_SIZE;
    std::size_t alignment  = st_PARTICLES_DEFAULT_MEMPOOL_ALIGNMENT;
    
    std::size_t const MEM_POOL_CAPACITY = st_Particles_predict_required_capacity( 
            NUM_OF_PARTICLES, &chunk_size, &alignment, true );
    
    st_MemPool mem_pool;
    st_MemPool_init( &mem_pool, MEM_POOL_CAPACITY, chunk_size );
    
    st_Particles* particles = 
        st_Particles_new_on_mempool( NUM_OF_PARTICLES, &mem_pool );
        
    ASSERT_TRUE( particles != nullptr );
    
    /* --------------------------------------------------------------------- */
    
    ASSERT_TRUE(  st_Particles_get_size( particles ) == NUM_OF_PARTICLES );
    ASSERT_TRUE(  st_Particles_is_consistent( particles ) );
    ASSERT_TRUE(  st_Particles_is_packed( particles ) );
    ASSERT_TRUE( !st_Particles_manages_own_memory( particles ) );
    ASSERT_TRUE(  st_Particles_uses_mempool( particles ) );
    ASSERT_TRUE( !st_Particles_uses_single_particle( particles ) );
    ASSERT_TRUE( !st_Particles_uses_flat_memory( particles ) );
        
    ASSERT_TRUE(  ( st_MemPool const* )&mem_pool == 
                  st_Particles_get_const_mem_pool( particles ) );
        
    ASSERT_TRUE(  st_Particles_has_defined_alignment( particles ) );
    
    uint64_t const def_alignment = st_Particles_alignment( particles );
    ASSERT_TRUE( def_alignment == st_PARTICLES_DEFAULT_MEMPOOL_ALIGNMENT );
    ASSERT_TRUE( ( size_t )def_alignment >= st_MemPool_get_chunk_size( &mem_pool ));
    ASSERT_TRUE( st_Particles_check_alignment( particles, def_alignment ) );
    
    st_Particles_free( particles );
    free( particles );
    particles = nullptr;
    
    st_MemPool_free( &mem_pool );    
}

/* ========================================================================= */
/* ====  Test basic usage of Particles with externally managed mem-pool &
 * ====  non-standard chunk_size to enforce a different alignment            */

TEST( ParticlesTests, InitExtMemPoolBasicForceAlignment32 )
{
    std::size_t const NUM_OF_PARTICLES  = std::size_t{ 5 };    
    /* to force alignment */
    std::size_t chunk_size = std::size_t{ 32 }; 
    std::size_t alignment  = st_PARTICLES_DEFAULT_MEMPOOL_ALIGNMENT;
    
    std::size_t const MEM_POOL_CAPACITY = st_Particles_predict_required_capacity( 
        NUM_OF_PARTICLES, &chunk_size, &alignment, true );
            
    ASSERT_TRUE(   alignment >= chunk_size );
    ASSERT_TRUE( ( alignment % chunk_size ) == ( size_t )0u );
    
    st_MemPool mem_pool;
    st_MemPool_init( &mem_pool, MEM_POOL_CAPACITY, chunk_size );
    
    st_Particles* particles = 
        st_Particles_new_on_mempool( NUM_OF_PARTICLES, &mem_pool );
        
    ASSERT_TRUE( particles != nullptr );
    
    /* --------------------------------------------------------------------- */
    
    ASSERT_TRUE(  st_Particles_get_size( particles ) == NUM_OF_PARTICLES );
    ASSERT_TRUE(  st_Particles_is_consistent( particles ) );
    ASSERT_TRUE(  st_Particles_is_packed( particles ) );
    ASSERT_TRUE( !st_Particles_manages_own_memory( particles ) );
    ASSERT_TRUE(  st_Particles_uses_mempool( particles ) );
    ASSERT_TRUE( !st_Particles_uses_single_particle( particles ) );
    ASSERT_TRUE( !st_Particles_uses_flat_memory( particles ) );
        
    ASSERT_TRUE(  ( st_MemPool const* )&mem_pool == 
                  st_Particles_get_const_mem_pool( particles ) );
        
    ASSERT_TRUE(  st_Particles_has_defined_alignment( particles ) );
    
    uint64_t const def_alignment = st_Particles_alignment( particles );
    ASSERT_TRUE( ( size_t )def_alignment == chunk_size );
    ASSERT_TRUE( st_Particles_check_alignment( particles, def_alignment ) );
    
    st_Particles_free( particles );
    free( particles );
    particles = nullptr;
    
    st_MemPool_free( &mem_pool );    
}


/* ========================================================================= */
/* ====  Test basic usage of Particles with self-managed single-particle     */

TEST( ParticlesTests, InitOwnSingleParticleBasic )
{
    st_Particles* particles = st_Particles_new_single();
    ASSERT_TRUE( particles != nullptr );

    /* --------------------------------------------------------------------- */
    
    ASSERT_TRUE(  st_Particles_get_size( particles ) == std::size_t{ 1 } );
    ASSERT_TRUE(  st_Particles_is_consistent( particles ) );
    ASSERT_TRUE( !st_Particles_is_packed( particles ) );
    ASSERT_TRUE(  st_Particles_manages_own_memory( particles ) );
    ASSERT_TRUE( !st_Particles_uses_mempool( particles ) );
    ASSERT_TRUE(  st_Particles_uses_single_particle( particles ) );
    ASSERT_TRUE( !st_Particles_uses_flat_memory( particles ) );
        
    st_SingleParticle const* ptr_single_particle = 
        st_Particles_get_const_base_single_particle( particles );
        
    ASSERT_TRUE( ptr_single_particle != nullptr );
    
    ASSERT_TRUE( st_Particles_check_mapping_of_single_praticle(
        ptr_single_particle, particles ) );
    
    ASSERT_TRUE( !st_Particles_has_defined_alignment( particles ) );
    
    uint64_t const alignment  = st_Particles_alignment( particles );
    ASSERT_TRUE( alignment == UINT64_C( 0 ) );
    
    st_Particles_free( particles );
    free( particles );
    particles = nullptr;
}

/* ========================================================================= */
/* ====  Test basic usage of Particles with external single-particle         */

TEST( ParticlesTests, InitExtSingleParticleBasic )
{
    st_SingleParticle single_particle;
    st_Particles* particles = st_Particles_new_on_single( &single_particle );
    ASSERT_TRUE( particles != nullptr );

    /* --------------------------------------------------------------------- */
    
    ASSERT_TRUE(  st_Particles_get_size( particles ) == std::size_t{ 1 } );
    ASSERT_TRUE(  st_Particles_is_consistent( particles ) );
    ASSERT_TRUE( !st_Particles_is_packed( particles ) );
    ASSERT_TRUE( !st_Particles_manages_own_memory( particles ) );
    ASSERT_TRUE( !st_Particles_uses_mempool( particles ) );
    ASSERT_TRUE(  st_Particles_uses_single_particle( particles ) );
    ASSERT_TRUE( !st_Particles_uses_flat_memory( particles ) );
        
    ASSERT_TRUE( st_Particles_get_const_base_single_particle( particles ) ==
                ( st_SingleParticle const* )&single_particle );
    
    ASSERT_TRUE( st_Particles_check_mapping_of_single_praticle(
        &single_particle, particles ) );
    
    ASSERT_TRUE( !st_Particles_has_defined_alignment( particles ) );
    
    uint64_t const alignment  = st_Particles_alignment( particles );
    ASSERT_TRUE( alignment == UINT64_C( 0 ) );
    
    st_Particles_free( particles );
    free( particles );
    particles = nullptr;
}

/* ========================================================================= */
/* ====  Test packing and subsequent unpacking of Particles                  */

TEST( ParticlesTests, TestPackingToMemPoolAndMapDuringUnpacking )
{
    std::size_t const NUM_OF_PARTICLES = std::size_t{ 8 };
    
    st_Particles* packed_particles = st_Particles_new( NUM_OF_PARTICLES );
    
    st_Particles unpacked_particles;
    st_Particles_preset( &unpacked_particles );
    
    ASSERT_TRUE(  st_Particles_get_size( packed_particles ) == NUM_OF_PARTICLES );
    ASSERT_TRUE(  st_Particles_is_consistent( packed_particles ) );
    ASSERT_TRUE(  st_Particles_is_packed( packed_particles ) );
    ASSERT_TRUE(  st_Particles_manages_own_memory( packed_particles ) );
    ASSERT_TRUE(  st_Particles_uses_mempool( packed_particles ) );
    ASSERT_TRUE( !st_Particles_uses_single_particle( packed_particles ) );
    ASSERT_TRUE( !st_Particles_uses_flat_memory( packed_particles ) );
    
    st_MemPool const* ptr_mem_pool = 
        st_Particles_get_const_mem_pool( packed_particles );
    
    ASSERT_TRUE(  ptr_mem_pool != nullptr );
    ASSERT_TRUE(  st_MemPool_get_const_buffer( ptr_mem_pool ) ==
                  st_Particles_get_const_mem_begin( packed_particles ) );
    
    bool success = st_Particles_unpack( &unpacked_particles, 
        ( unsigned char* )st_Particles_get_mem_begin( packed_particles ), 
        st_PARTICLES_UNPACK_MAP | st_PARTICLES_UNPACK_CHECK_CONSISTENCY );
        
    ASSERT_TRUE( success );
    ASSERT_TRUE(  st_Particles_uses_flat_memory( &unpacked_particles ) );
    ASSERT_TRUE( !st_Particles_manages_own_memory( &unpacked_particles ) );
    ASSERT_TRUE( !st_Particles_uses_mempool( &unpacked_particles ) );
    ASSERT_TRUE(  st_Particles_get_const_flat_memory( &unpacked_particles ) ==
                  st_Particles_get_const_mem_begin( packed_particles ) );
    
    ASSERT_TRUE(  st_Particles_get_q0( packed_particles ) ==
                  st_Particles_get_q0( &unpacked_particles ) );
    
    ASSERT_TRUE(  st_Particles_get_mass0( packed_particles ) ==
                  st_Particles_get_mass0( &unpacked_particles ) );
    
    ASSERT_TRUE(  st_Particles_get_beta0( packed_particles ) ==
                  st_Particles_get_beta0( &unpacked_particles ) );
    
    ASSERT_TRUE(  st_Particles_get_gamma0( packed_particles ) ==
                  st_Particles_get_gamma0( &unpacked_particles ) );
    
    ASSERT_TRUE(  st_Particles_get_particle_id( packed_particles ) ==
                  st_Particles_get_particle_id( &unpacked_particles ) );
    
    ASSERT_TRUE(  st_Particles_get_lost_at_element_id( packed_particles ) ==
                  st_Particles_get_lost_at_element_id( &unpacked_particles ) );
    
    ASSERT_TRUE(  st_Particles_get_lost_at_turn( packed_particles ) ==
                  st_Particles_get_lost_at_turn( &unpacked_particles ) );
    
    ASSERT_TRUE(  st_Particles_get_state( packed_particles ) ==
                  st_Particles_get_state( &unpacked_particles ) );
    
    ASSERT_TRUE(  st_Particles_get_s( packed_particles ) ==
                  st_Particles_get_s( &unpacked_particles ) );
    
    ASSERT_TRUE(  st_Particles_get_x( packed_particles ) ==
                  st_Particles_get_x( &unpacked_particles ) );
    
    ASSERT_TRUE(  st_Particles_get_y( packed_particles ) ==
                  st_Particles_get_y( &unpacked_particles ) );
    
    ASSERT_TRUE(  st_Particles_get_px( packed_particles ) ==
                  st_Particles_get_px( &unpacked_particles ) );
    
    ASSERT_TRUE(  st_Particles_get_py( packed_particles ) ==
                  st_Particles_get_py( &unpacked_particles ) );
    
    ASSERT_TRUE(  st_Particles_get_sigma( packed_particles ) ==
                  st_Particles_get_sigma( &unpacked_particles ) );
    
    ASSERT_TRUE(  st_Particles_get_psigma( packed_particles ) ==
                  st_Particles_get_psigma( &unpacked_particles ) );
    
    ASSERT_TRUE(  st_Particles_get_delta( packed_particles ) ==
                  st_Particles_get_delta( &unpacked_particles ) );
    
    ASSERT_TRUE(  st_Particles_get_rpp( packed_particles ) ==
                  st_Particles_get_rpp( &unpacked_particles ) );
    
    ASSERT_TRUE(  st_Particles_get_rvv( packed_particles ) ==
                  st_Particles_get_rvv( &unpacked_particles ) );
    
    ASSERT_TRUE(  st_Particles_get_chi( packed_particles ) ==
                  st_Particles_get_chi( &unpacked_particles ) );
    
    st_Particles_free( packed_particles );
    free( packed_particles );
    packed_particles = nullptr;    
}

/* end: sixtracklib/common/tests/test_particles.cpp */
