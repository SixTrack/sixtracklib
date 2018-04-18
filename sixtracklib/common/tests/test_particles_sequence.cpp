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

#include "sixtracklib/common/particles_sequence.h"

#include "sixtracklib/_impl/definitions.h"
#include "sixtracklib/common/mem_pool.h"
#include "sixtracklib/common/particles.h"
#include "sixtracklib/common/impl/particles_type.h"

#if defined( __SAVED_NAMESPACE )
    #undef __NAMESPACE
    #define __NAMESPACE __SAVED_NAMESPACE
#endif /* defined( __SAVED_NAMESPACE ) */

TEST( CommonParticlesSequenceTests, TestCreateAndDestruction )
{
    std::size_t const SEQUENCE_SIZE = std::size_t{ 20 };
    std::size_t const NUM_PARTICLES = std::size_t{ 1000 };
    std::size_t chunk_size = st_PARTICLES_DEFAULT_MEMPOOL_CHUNK_SIZE;
    std::size_t alignment  = st_PARTICLES_DEFAULT_MEMPOOL_ALIGNMENT;
    
    st_ParticlesSequence sequence;
    
    bool success = st_ParticlesSequence_init( 
        &sequence, SEQUENCE_SIZE, NUM_PARTICLES, &chunk_size, &alignment, true );
    
    ASSERT_TRUE( success );
    ASSERT_TRUE( st_ParticlesSequence_get_size( &sequence ) == SEQUENCE_SIZE );
    ASSERT_TRUE( st_ParticlesSequence_get_const_begin( &sequence ) != nullptr );
    ASSERT_TRUE( st_ParticlesSequence_get_const_end( &sequence ) != nullptr );
    ASSERT_TRUE( st_ParticlesSequence_get_const_begin( &sequence ) !=
                 st_ParticlesSequence_get_const_end( &sequence ) );
    
    ASSERT_TRUE( std::distance( st_ParticlesSequence_get_begin( &sequence ),
                                st_ParticlesSequence_get_end( &sequence ) ) ==
                    static_cast< std::ptrdiff_t >( SEQUENCE_SIZE ) );
    
    st_Particles* particles = st_Particles_new( NUM_PARTICLES );
    ASSERT_TRUE( particles != nullptr );
    
    /* --------------------------------------------------------------------- */
    
    for( std::size_t jj = 0 ; jj < SEQUENCE_SIZE ; ++jj )
    {
        SIXTRL_INT64_T element_id = SIXTRL_INT64_T{ 0 };
    
        for( std::size_t ii = 0 ; ii < NUM_PARTICLES ; ++ii, ++element_id )
        {
            st_Particles_set_s_value( particles, ii, static_cast< SIXTRL_REAL_T >( jj * ii )  );
            st_Particles_set_x_value( particles, ii, static_cast< SIXTRL_REAL_T >( jj * ii ) + 0.4 );
            st_Particles_set_y_value( particles, ii, static_cast< SIXTRL_REAL_T >( jj * ii ) - 0.4 );
            st_Particles_set_particle_id_value( particles, ii, element_id );
        }
        
        success = st_Particles_deep_copy_all( 
            st_ParticlesSequence_get_particles_by_index( &sequence, jj ), 
                particles );
        
        ASSERT_TRUE( success );
    }
    
    st_Particles const* particles_in_seq = 
        st_ParticlesSequence_get_const_begin( &sequence );
        
    st_Particles const* particles_seq_end =
        st_ParticlesSequence_get_const_end( &sequence );
    
    for( std::size_t jj = 0 ; particles_in_seq != particles_seq_end ; 
            ++jj, ++particles_in_seq )
    {
        SIXTRL_INT64_T element_id = SIXTRL_INT64_T{ 0 };
    
        ASSERT_TRUE( st_Particles_get_size( particles_in_seq ) == NUM_PARTICLES );
        
        for( std::size_t ii = 0 ; ii < NUM_PARTICLES ; ++ii, ++element_id )
        {
            st_Particles_set_s_value( particles, ii, static_cast< SIXTRL_REAL_T >( jj * ii )  );
            st_Particles_set_x_value( particles, ii, static_cast< SIXTRL_REAL_T >( jj * ii ) + 0.4 );
            st_Particles_set_y_value( particles, ii, static_cast< SIXTRL_REAL_T >( jj * ii ) - 0.4 );
            st_Particles_set_particle_id_value( particles, ii, element_id );
        }
        
        ASSERT_TRUE( st_Particles_get_s( particles_in_seq ) != 
                     st_Particles_get_s( particles ) );
        
        ASSERT_TRUE( st_Particles_get_x( particles_in_seq ) != 
                     st_Particles_get_x( particles ) );
        
        ASSERT_TRUE( st_Particles_get_y( particles_in_seq ) != 
                     st_Particles_get_y( particles ) );
        
        ASSERT_TRUE( st_Particles_get_particle_id( particles_in_seq ) != 
                     st_Particles_get_particle_id( particles ) );
        
        ASSERT_TRUE( memcmp( st_Particles_get_s( particles_in_seq ), 
                             st_Particles_get_s( particles ), 
                             NUM_PARTICLES * sizeof( SIXTRL_REAL_T ) ) == 0 );
        
        ASSERT_TRUE( memcmp( st_Particles_get_x( particles_in_seq ), 
                             st_Particles_get_x( particles ), 
                             NUM_PARTICLES * sizeof( SIXTRL_REAL_T ) ) == 0 );
        
        ASSERT_TRUE( memcmp( st_Particles_get_y( particles_in_seq ), 
                             st_Particles_get_y( particles ), 
                             NUM_PARTICLES * sizeof( SIXTRL_REAL_T ) ) == 0 );
        
        ASSERT_TRUE( memcmp( st_Particles_get_particle_id( particles_in_seq ), 
                             st_Particles_get_particle_id( particles ), 
                             NUM_PARTICLES * sizeof( SIXTRL_INT64_T ) ) == 0 );
    }
    
    /* --------------------------------------------------------------------- */
    
    st_Particles_free( particles );
    free( particles );
    particles = nullptr;
    
    st_ParticlesSequence_free( &sequence );
    
    ASSERT_TRUE( st_ParticlesSequence_get_size( &sequence ) == 
        std::size_t{ 0 } );
    
    ASSERT_TRUE( st_ParticlesSequence_get_const_begin( 
        &sequence ) == nullptr );
    
    ASSERT_TRUE( st_ParticlesSequence_get_const_end( 
        &sequence ) == nullptr );
    
    ASSERT_TRUE( st_ParticlesSequence_get_const_ptr_to_mem_pool( 
        &sequence ) == nullptr );    
}

