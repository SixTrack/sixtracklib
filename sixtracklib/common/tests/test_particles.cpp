#if !defined( __NAMESPACE )
    #define __NAMESPACE st_
    #define __UNDEF_NAMESPACE_AT_END 1
#endif /* !defiend( __NAMESPACE ) */

#include "sixtracklib/_impl/definitions.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>

#include "sixtracklib/common/mem_pool.h"
#include "sixtracklib/common/particles.h"
#include "sixtracklib/common/impl/particles_type.h"
#include "sixtracklib/common/single_particle.h"

#include <gtest/gtest.h>

#if defined( __NAMESPACE ) && defined( __UNDEF_NAMESPACE_AT_END )
    #undef __NAMESPACE
    #undef __UNDEF_NAMESPACE_AT_END
#endif /* !defined( __NAMESPACE ) && defined( __UNDEF_NAMESPACE_AT_END ) */

/* ------------------------------------------------------------------------- */
/* --- Helper function for checking the consistency of a SingleParticle ---- */
/* --- instance with a SingleParticle based Particles container         ---- */

bool NS(Particles_check_mapping_of_single_praticle)(
     const NS(SingleParticle) *const SIXTRL_RESTRICT single_particle,
     const NS(Particles) *const SIXTRL_RESTRICT particles )
{
    bool is_mapped_correctly = false;
    
    if( ( single_particle != 0 ) && ( particles != 0 ) &&
        ( NS(Particles_get_size)( particles ) == std::size_t{ 1 } ) )
    {
        is_mapped_correctly   = ( &single_particle->q0 == 
            NS(Particles_get_q0)( particles ) );
        
        is_mapped_correctly  &= ( &single_particle->mass0 == 
                    NS(Particles_get_mass0)( particles ) );
        
        is_mapped_correctly  &= ( &single_particle->beta0 == 
                    NS(Particles_get_beta0)( particles ) );
        
        is_mapped_correctly  &= ( &single_particle->gamma0 == 
                    NS(Particles_get_gamma0)( particles ) );
        
        is_mapped_correctly  &= ( &single_particle->p0c == 
                    NS(Particles_get_p0c)( particles ) );
        
        is_mapped_correctly  &= ( &single_particle->partid == 
                    NS(Particles_get_particle_id)( particles ) );
        
        is_mapped_correctly  &= ( &single_particle->elemid == 
                    NS(Particles_get_lost_at_element_id)( particles ) );
        
        is_mapped_correctly  &= ( &single_particle->turn == 
                    NS(Particles_get_lost_at_turn)( particles ) );
        
        is_mapped_correctly  &= ( &single_particle->state == 
                    NS(Particles_get_state)( particles ) );
        
        is_mapped_correctly  &= ( &single_particle->s == 
                    NS(Particles_get_s)( particles ) );
        
        is_mapped_correctly  &= ( &single_particle->x == 
                    NS(Particles_get_x)( particles ) );
        
        is_mapped_correctly  &= ( &single_particle->y == 
                    NS(Particles_get_y)( particles ) );
        
        is_mapped_correctly  &= ( &single_particle->px == 
                    NS(Particles_get_px)( particles ) );
        
        is_mapped_correctly  &= ( &single_particle->py == 
                    NS(Particles_get_py)( particles ) );
        
        is_mapped_correctly  &= ( &single_particle->sigma == 
                    NS(Particles_get_sigma)( particles ) );
        
        is_mapped_correctly  &= ( &single_particle->psigma == 
                    NS(Particles_get_psigma)( particles ) );
        
        is_mapped_correctly  &= ( &single_particle->delta == 
                    NS(Particles_get_delta)( particles ) );
        
        is_mapped_correctly  &= ( &single_particle->rpp == 
                    NS(Particles_get_rpp)( particles ) );
        
        is_mapped_correctly  &= ( &single_particle->rvv == 
                    NS(Particles_get_rvv)( particles ) );
        
        is_mapped_correctly  &= ( &single_particle->chi == 
                    NS(Particles_get_chi)( particles ) );
    }
    
    return is_mapped_correctly;    
}


/* ========================================================================= */
/* ====  Test basic usage of Particles with self-managed mem-pool            */

TEST( ParticlesTests, InitOwnMemPoolBasic )
{
    std::size_t const NUM_OF_PARTICLES = std::size_t{ 4 };
    
    NS(Particles)* particles = NS(Particles_new)( NUM_OF_PARTICLES );
    ASSERT_TRUE( particles != nullptr );

    /* --------------------------------------------------------------------- */
    
    ASSERT_TRUE(  NS(Particles_get_size)( particles ) == NUM_OF_PARTICLES );
    ASSERT_TRUE(  NS(Particles_is_consistent( particles ) ) );
    ASSERT_TRUE(  NS(Particles_is_packed)( particles ) );
    ASSERT_TRUE(  NS(Particles_manages_own_memory( particles ) ) );
    ASSERT_TRUE(  NS(Particles_uses_mempool)( particles ) );
    ASSERT_TRUE( !NS(Particles_uses_single_particle)( particles ) );
    ASSERT_TRUE( !NS(Particles_uses_flat_memory)( particles ) );
        
    NS(MemPool) const* ptr_mem_pool = 
        NS(Particles_get_const_mem_pool)( particles );
        
    ASSERT_TRUE( ptr_mem_pool != nullptr );
    ASSERT_TRUE( NS(MemPool_get_const_buffer)( ptr_mem_pool ) != nullptr );
    ASSERT_TRUE( NS(MemPool_get_chunk_size)( ptr_mem_pool ) == 
                 NS(PARTICLES_DEFAULT_MEMPOOL_CHUNK_SIZE) );
    
    ASSERT_TRUE(  NS(Particles_has_defined_alignment)( particles ) );
    
    uint64_t const alignment  = NS(Particles_alignment)( particles );
    size_t   const chunk_size = NS(MemPool_get_chunk_size)( ptr_mem_pool );
    
    ASSERT_TRUE( alignment == NS(PARTICLES_DEFAULT_MEMPOOL_ALIGNMENT) );
    ASSERT_TRUE( ( size_t )alignment >= chunk_size );
    ASSERT_TRUE( NS(Particles_check_alignment)( particles, alignment ) );
    
    NS(Particles_free)( particles );
    free( particles );
    particles = nullptr;
}

/* ========================================================================= */
/* ====  Test basic usage of Particles with externally managed mem-pool      */

TEST( ParticlesTests, InitExtMemPoolBasic )
{
    std::size_t const NUM_OF_PARTICLES  = std::size_t{ 4 };
    
    std::size_t chunk_size = NS(PARTICLES_DEFAULT_MEMPOOL_CHUNK_SIZE);
    std::size_t alignment  = NS(PARTICLES_DEFAULT_MEMPOOL_ALIGNMENT);
    
    std::size_t const MEM_POOL_CAPACITY = NS(Particles_predict_required_capacity)( 
            NUM_OF_PARTICLES, &chunk_size, &alignment, true );
    
    NS(MemPool) mem_pool;
    NS(MemPool_init)( &mem_pool, MEM_POOL_CAPACITY, chunk_size );
    
    NS(Particles)* particles = 
        NS(Particles_new_on_mempool)( NUM_OF_PARTICLES, &mem_pool );
        
    ASSERT_TRUE( particles != nullptr );
    
    /* --------------------------------------------------------------------- */
    
    ASSERT_TRUE(  NS(Particles_get_size)( particles ) == NUM_OF_PARTICLES );
    ASSERT_TRUE(  NS(Particles_is_consistent( particles ) ) );
    ASSERT_TRUE(  NS(Particles_is_packed)( particles ) );
    ASSERT_TRUE( !NS(Particles_manages_own_memory( particles ) ) );
    ASSERT_TRUE(  NS(Particles_uses_mempool)( particles ) );
    ASSERT_TRUE( !NS(Particles_uses_single_particle)( particles ) );
    ASSERT_TRUE( !NS(Particles_uses_flat_memory)( particles ) );
        
    ASSERT_TRUE(  ( NS(MemPool) const* )&mem_pool == 
                  NS(Particles_get_const_mem_pool)( particles ) );
        
    ASSERT_TRUE(  NS(Particles_has_defined_alignment)( particles ) );
    
    uint64_t const def_alignment = NS(Particles_alignment)( particles );
    ASSERT_TRUE( def_alignment == NS(PARTICLES_DEFAULT_MEMPOOL_ALIGNMENT) );
    ASSERT_TRUE( ( size_t )def_alignment >= NS(MemPool_get_chunk_size)( &mem_pool ));
    ASSERT_TRUE( NS(Particles_check_alignment)( particles, def_alignment ) );
    
    NS(Particles_free)( particles );
    free( particles );
    particles = nullptr;
    
    NS(MemPool_free)( &mem_pool );    
}

/* ========================================================================= */
/* ====  Test basic usage of Particles with externally managed mem-pool &
 * ====  non-standard chunk_size to enforce a different alignment            */

TEST( ParticlesTests, InitExtMemPoolBasicForceAlignment32 )
{
    std::size_t const NUM_OF_PARTICLES  = std::size_t{ 5 };    
    /* to force alignment */
    std::size_t chunk_size = std::size_t{ 32 }; 
    std::size_t alignment  = NS(PARTICLES_DEFAULT_MEMPOOL_ALIGNMENT);
    
    std::size_t const MEM_POOL_CAPACITY = NS(Particles_predict_required_capacity)( 
        NUM_OF_PARTICLES, &chunk_size, &alignment, true );
            
    ASSERT_TRUE(   alignment >= chunk_size );
    ASSERT_TRUE( ( alignment % chunk_size ) == ( size_t )0u );
    
    NS(MemPool) mem_pool;
    NS(MemPool_init)( &mem_pool, MEM_POOL_CAPACITY, chunk_size );
    
    NS(Particles)* particles = 
        NS(Particles_new_on_mempool)( NUM_OF_PARTICLES, &mem_pool );
        
    ASSERT_TRUE( particles != nullptr );
    
    /* --------------------------------------------------------------------- */
    
    ASSERT_TRUE(  NS(Particles_get_size)( particles ) == NUM_OF_PARTICLES );
    ASSERT_TRUE(  NS(Particles_is_consistent( particles ) ) );
    ASSERT_TRUE(  NS(Particles_is_packed)( particles ) );
    ASSERT_TRUE( !NS(Particles_manages_own_memory( particles ) ) );
    ASSERT_TRUE(  NS(Particles_uses_mempool)( particles ) );
    ASSERT_TRUE( !NS(Particles_uses_single_particle)( particles ) );
    ASSERT_TRUE( !NS(Particles_uses_flat_memory)( particles ) );
        
    ASSERT_TRUE(  ( NS(MemPool) const* )&mem_pool == 
                  NS(Particles_get_const_mem_pool)( particles ) );
        
    ASSERT_TRUE(  NS(Particles_has_defined_alignment)( particles ) );
    
    uint64_t const def_alignment = NS(Particles_alignment)( particles );
    ASSERT_TRUE( ( size_t )def_alignment == chunk_size );
    ASSERT_TRUE( NS(Particles_check_alignment)( particles, def_alignment ) );
    
    NS(Particles_free)( particles );
    free( particles );
    particles = nullptr;
    
    NS(MemPool_free)( &mem_pool );    
}


/* ========================================================================= */
/* ====  Test basic usage of Particles with self-managed single-particle     */

TEST( ParticlesTests, InitOwnSingleParticleBasic )
{
    NS(Particles)* particles = NS(Particles_new_single)();
    ASSERT_TRUE( particles != nullptr );

    /* --------------------------------------------------------------------- */
    
    ASSERT_TRUE(  NS(Particles_get_size)( particles ) == std::size_t{ 1 } );
    ASSERT_TRUE(  NS(Particles_is_consistent( particles ) ) );
    ASSERT_TRUE( !NS(Particles_is_packed)( particles ) );
    ASSERT_TRUE(  NS(Particles_manages_own_memory( particles ) ) );
    ASSERT_TRUE( !NS(Particles_uses_mempool)( particles ) );
    ASSERT_TRUE(  NS(Particles_uses_single_particle)( particles ) );
    ASSERT_TRUE( !NS(Particles_uses_flat_memory)( particles ) );
        
    NS(SingleParticle) const* ptr_single_particle = 
        NS(Particles_get_const_base_single_particle)( particles );
        
    ASSERT_TRUE( ptr_single_particle != nullptr );
    
    ASSERT_TRUE( NS(Particles_check_mapping_of_single_praticle)(
        ptr_single_particle, particles ) );
    
    ASSERT_TRUE( !NS(Particles_has_defined_alignment)( particles ) );
    
    uint64_t const alignment  = NS(Particles_alignment)( particles );
    ASSERT_TRUE( alignment == UINT64_C( 0 ) );
    
    NS(Particles_free)( particles );
    free( particles );
    particles = nullptr;
}

/* ========================================================================= */
/* ====  Test basic usage of Particles with external single-particle         */

TEST( ParticlesTests, InitExtSingleParticleBasic )
{
    NS(SingleParticle) single_particle;
    NS(Particles)* particles = NS(Particles_new_on_single)( &single_particle );
    ASSERT_TRUE( particles != nullptr );

    /* --------------------------------------------------------------------- */
    
    ASSERT_TRUE(  NS(Particles_get_size)( particles ) == std::size_t{ 1 } );
    ASSERT_TRUE(  NS(Particles_is_consistent( particles ) ) );
    ASSERT_TRUE( !NS(Particles_is_packed)( particles ) );
    ASSERT_TRUE( !NS(Particles_manages_own_memory( particles ) ) );
    ASSERT_TRUE( !NS(Particles_uses_mempool)( particles ) );
    ASSERT_TRUE(  NS(Particles_uses_single_particle)( particles ) );
    ASSERT_TRUE( !NS(Particles_uses_flat_memory)( particles ) );
        
    ASSERT_TRUE( NS(Particles_get_const_base_single_particle)( particles ) ==
                ( NS(SingleParticle) const* )&single_particle );
    
    ASSERT_TRUE( NS(Particles_check_mapping_of_single_praticle)(
        &single_particle, particles ) );
    
    ASSERT_TRUE( !NS(Particles_has_defined_alignment)( particles ) );
    
    uint64_t const alignment  = NS(Particles_alignment)( particles );
    ASSERT_TRUE( alignment == UINT64_C( 0 ) );
    
    NS(Particles_free)( particles );
    free( particles );
    particles = nullptr;
}

/* ========================================================================= */
/* ====  Test packing and subsequent unpacking of Particles                  */

TEST( ParticlesTests, TestPackingToMemPoolAndMapDuringUnpacking )
{
    std::size_t const NUM_OF_PARTICLES = std::size_t{ 8 };
    
    NS(Particles)* packed_particles = NS(Particles_new)( NUM_OF_PARTICLES );
    
    NS(Particles) unpacked_particles;
    NS(Particles_preset)( &unpacked_particles );
    
    ASSERT_TRUE(  NS(Particles_get_size)( packed_particles ) == NUM_OF_PARTICLES );
    ASSERT_TRUE(  NS(Particles_is_consistent( packed_particles ) ) );
    ASSERT_TRUE(  NS(Particles_is_packed)( packed_particles ) );
    ASSERT_TRUE(  NS(Particles_manages_own_memory( packed_particles ) ) );
    ASSERT_TRUE(  NS(Particles_uses_mempool)( packed_particles ) );
    ASSERT_TRUE( !NS(Particles_uses_single_particle)( packed_particles ) );
    ASSERT_TRUE( !NS(Particles_uses_flat_memory)( packed_particles ) );
    
    NS(MemPool) const* ptr_mem_pool = 
        NS(Particles_get_const_mem_pool)( packed_particles );
    
    ASSERT_TRUE(  ptr_mem_pool != nullptr );
    ASSERT_TRUE(  NS(MemPool_get_const_buffer)( ptr_mem_pool ) ==
                  NS(Particles_get_const_mem_begin)( packed_particles ) );
    
    bool success = NS(Particles_unpack)( &unpacked_particles, 
        ( unsigned char* )NS(Particles_get_mem_begin)( packed_particles ), 
        NS(PARTICLES_UNPACK_MAP) | NS(PARTICLES_UNPACK_CHECK_CONSISTENCY) );
        
    ASSERT_TRUE( success );
    ASSERT_TRUE(  NS(Particles_uses_flat_memory)( &unpacked_particles ) );
    ASSERT_TRUE( !NS(Particles_manages_own_memory )( &unpacked_particles ) );
    ASSERT_TRUE( !NS(Particles_uses_mempool)( &unpacked_particles ) );
    ASSERT_TRUE(  NS(Particles_get_const_flat_memory)( &unpacked_particles ) ==
                  NS(Particles_get_const_mem_begin)( packed_particles ) );
    
    ASSERT_TRUE(  NS(Particles_get_q0)( packed_particles ) ==
                  NS(Particles_get_q0)( &unpacked_particles ) );
    
    ASSERT_TRUE(  NS(Particles_get_mass0)( packed_particles ) ==
                  NS(Particles_get_mass0)( &unpacked_particles ) );
    
    ASSERT_TRUE(  NS(Particles_get_beta0)( packed_particles ) ==
                  NS(Particles_get_beta0)( &unpacked_particles ) );
    
    ASSERT_TRUE(  NS(Particles_get_gamma0)( packed_particles ) ==
                  NS(Particles_get_gamma0)( &unpacked_particles ) );
    
    ASSERT_TRUE(  NS(Particles_get_particle_id)( packed_particles ) ==
                  NS(Particles_get_particle_id)( &unpacked_particles ) );
    
    ASSERT_TRUE(  NS(Particles_get_lost_at_element_id)( packed_particles ) ==
                  NS(Particles_get_lost_at_element_id)( &unpacked_particles ) );
    
    ASSERT_TRUE(  NS(Particles_get_lost_at_turn)( packed_particles ) ==
                  NS(Particles_get_lost_at_turn)( &unpacked_particles ) );
    
    ASSERT_TRUE(  NS(Particles_get_state)( packed_particles ) ==
                  NS(Particles_get_state)( &unpacked_particles ) );
    
    ASSERT_TRUE(  NS(Particles_get_s)( packed_particles ) ==
                  NS(Particles_get_s)( &unpacked_particles ) );
    
    ASSERT_TRUE(  NS(Particles_get_x)( packed_particles ) ==
                  NS(Particles_get_x)( &unpacked_particles ) );
    
    ASSERT_TRUE(  NS(Particles_get_y)( packed_particles ) ==
                  NS(Particles_get_y)( &unpacked_particles ) );
    
    ASSERT_TRUE(  NS(Particles_get_px)( packed_particles ) ==
                  NS(Particles_get_px)( &unpacked_particles ) );
    
    ASSERT_TRUE(  NS(Particles_get_py)( packed_particles ) ==
                  NS(Particles_get_py)( &unpacked_particles ) );
    
    ASSERT_TRUE(  NS(Particles_get_sigma)( packed_particles ) ==
                  NS(Particles_get_sigma)( &unpacked_particles ) );
    
    ASSERT_TRUE(  NS(Particles_get_psigma)( packed_particles ) ==
                  NS(Particles_get_psigma)( &unpacked_particles ) );
    
    ASSERT_TRUE(  NS(Particles_get_delta)( packed_particles ) ==
                  NS(Particles_get_delta)( &unpacked_particles ) );
    
    ASSERT_TRUE(  NS(Particles_get_rpp)( packed_particles ) ==
                  NS(Particles_get_rpp)( &unpacked_particles ) );
    
    ASSERT_TRUE(  NS(Particles_get_rvv)( packed_particles ) ==
                  NS(Particles_get_rvv)( &unpacked_particles ) );
    
    ASSERT_TRUE(  NS(Particles_get_chi)( packed_particles ) ==
                  NS(Particles_get_chi)( &unpacked_particles ) );
    
    NS(Particles_free)( packed_particles );
    free( packed_particles );
    packed_particles = nullptr;    
}

/* end: sixtracklib/common/tests/test_particles.cpp */
