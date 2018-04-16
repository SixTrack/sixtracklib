#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <chrono>
#include <random>
#include <iostream>
#include <memory>
#include <vector>

#if defined( __NAMESPACE )
    #define __SAVED_NAMESPACE __NAMESPACE
    #undef  __NAMESPACE     
#endif /* !defiend( __NAMESPACE ) */

#if !defined( __NAMESPACE )
    #define __NAMESPACE st_    
#endif /* !defiend( __NAMESPACE ) */

#include "sixtracklib/common/block.h"
#include "sixtracklib/common/block_drift.h"
#include "sixtracklib/common/impl/block_drift_type.h"

#include "sixtracklib/common/particles.h"
#include "sixtracklib/common/impl/particles_type.h"

#include "sixtracklib/common/mem_pool.h"

#if defined( __SAVED_NAMESPACE )
    #undef __NAMESPACE
    #define __NAMESPACE __SAVED_NAMESPACE
#endif /* defined( __SAVED_NAMESPACE ) */

int main()
{
    using particles_t = ::st_Particles;
    
    std::vector< std::size_t > const NUM_OF_PARTICLES{ 1, 10, 100, 1000, 10000, 1000000 };
    std::size_t const NUM_OF_TURNS    = 1;
    std::size_t const NUM_OF_ELEMENTS = 1;
    
    auto particles_del = 
        []( particles_t* p ){ ::st_Particles_free( p ); free( p ); };
        
    using ptr_particles_t = 
          std::unique_ptr< particles_t, decltype( particles_del ) >;
          
    std::vector< ptr_particles_t > elem_by_elem;
    elem_by_elem.clear();
    std::size_t const ELEMS_BY_TURNS = NUM_OF_TURNS * NUM_OF_ELEMENTS;
    elem_by_elem.reserve( ELEMS_BY_TURNS );
    
        
    std::vector< ptr_particles_t > turn_by_turn;
    turn_by_turn.clear();
    turn_by_turn.reserve( NUM_OF_TURNS );
    
    for( std::size_t const npart : NUM_OF_PARTICLES )
    {
        std::size_t const CHUNK_SIZE = st_BLOCK_DEFAULT_MEMPOOL_CHUNK_SIZE;
        std::size_t const CAPACITY   = std::size_t{ 256 } * NUM_OF_ELEMENTS;
        std::size_t alignment = st_BLOCK_DEFAULT_MEMPOOL_ALIGNMENT;
        
        st_MemPool block;
        st_MemPool_init( &block, CAPACITY, CHUNK_SIZE );
        
        std::random_device rd;
        std::mt19937 prng( rd() );
        std::uniform_real_distribution<> random_lengths( 0.1, 10.0 );
                
        std::vector< st_DriftSingle > elements( NUM_OF_ELEMENTS );
        elements.clear();
        
        int64_t element_id = int64_t{ 0 };
        unsigned char* ptr_elements_begin = nullptr;
        
        for( std::size_t ii = 0 ; ii < NUM_OF_ELEMENTS ; ++ii )
        {
            st_DriftSingle single;
            st_DriftSingle_set_type_id( &single, st_ELEMENT_TYPE_DRIFT );
            st_DriftSingle_set_element_id( &single, element_id );
            st_DriftSingle_set_length( &single, random_lengths( prng ) );
            
            st_Drift drift;
            st_Drift_map_from_single_drift( &drift, &single );
            
            st_AllocResult result = st_Drift_pack_aligned( &drift, &block, alignment );
            unsigned char* ptr_begin = st_AllocResult_get_pointer( &result );
            
            if( ptr_elements_begin == nullptr ) ptr_elements_begin = ptr_begin;
        }
        
        if( ptr_elements_begin == nullptr ) break;
        
        /* ------------------------------------------------------------------ */
        
        std::vector< particles_t* > elem_by_elem_ptr( ELEMS_BY_TURNS, nullptr );
        
        elem_by_elem.clear();
        
        for( std::size_t ii = 0 ; ii < ELEMS_BY_TURNS ; ++ii )
        {
            elem_by_elem.emplace_back( st_Particles_new( npart ), particles_del );
            elem_by_elem_ptr[ ii ] = elem_by_elem.back().get();
        }
        
        std::cout << "elem_by_elem :: size = " << elem_by_elem.size() << std::endl;
        
        /* ------------------------------------------------------------------ */
        
        std::vector< particles_t* > turn_by_turn_ptr( NUM_OF_TURNS, nullptr );
        
        turn_by_turn.clear();
        
        for( std::size_t ii = 0 ; ii < NUM_OF_TURNS ; ++ii )
        {
            turn_by_turn.emplace_back( 
                st_Particles_new( npart ), particles_del );
            
            turn_by_turn_ptr[ ii ] = turn_by_turn.back().get();
        }
        
        std::cout << "turn_by_turn :: size = " << turn_by_turn.size() << std::endl;
        
        /* ------------------------------------------------------------------ */
        
        ptr_particles_t particles( st_Particles_new( npart ), particles_del );
        
        /* ------------------------------------------------------------------ */
        
        
        st_MemPool_free( &block );
    }
    
    return 0;
}

/* end: */
