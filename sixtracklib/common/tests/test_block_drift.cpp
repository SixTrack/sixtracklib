#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <random>

#include <gtest/gtest.h>

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
#include "sixtracklib/common/mem_pool.h"

#if defined( __SAVED_NAMESPACE )
    #undef __NAMESPACE
    #define __NAMESPACE __SAVED_NAMESPACE
#endif /* defined( __SAVED_NAMESPACE ) */

TEST( CommonTestsBlockDrift, PackSingleDriftToMemPoolAndUnpack )
{
    std::size_t const CHUNK_SIZE = st_BLOCK_DEFAULT_MEMPOOL_CHUNK_SIZE;
    std::size_t alignment        = st_BLOCK_DEFAULT_MEMPOOL_ALIGNMENT;
    
    st_MemPool mem_pool;
    
    std::size_t const REQUIRED_MIN_CAPACITY = 
        st_Drift_predict_required_num_bytes_on_mempool_for_packing(
            0, CHUNK_SIZE, &alignment );
        
    ASSERT_TRUE( ( alignment == CHUNK_SIZE ) ||
                 ( ( alignment >  CHUNK_SIZE ) && 
                   ( ( alignment % CHUNK_SIZE ) == 0 ) ) );
    
    ASSERT_TRUE( REQUIRED_MIN_CAPACITY > CHUNK_SIZE );
    
    st_MemPool_init( &mem_pool, REQUIRED_MIN_CAPACITY, CHUNK_SIZE );
    
    /* ===================================================================== */
    
    st_DriftSingle single_drift;
    
    double  const length  = double{ 1.23456 };
    int64_t const elem_id = int64_t{ 42 };
    
    st_DriftSingle_set_type_id( &single_drift, st_ELEMENT_TYPE_DRIFT );
    st_DriftSingle_set_length( &single_drift, length );
    st_DriftSingle_set_element_id( &single_drift, elem_id );
    
    /* --------------------------------------------------------------------- */
    
    st_Drift drift;
    st_Drift_map_from_single_drift( &drift, &single_drift );
    
    ASSERT_TRUE( st_Drift_get_type_id( &drift ) == st_ELEMENT_TYPE_DRIFT );
    ASSERT_TRUE( st_Drift_get_element_id( &drift ) == elem_id );
    ASSERT_TRUE( std::fabs( st_Drift_get_length( &drift ) - length ) < 1e-9 );
    
    /* ===================================================================== */
    
    st_AllocResult result = st_Drift_pack_aligned( &drift, &mem_pool, alignment );
    
    ASSERT_TRUE( st_AllocResult_valid( &result ) );
    ASSERT_TRUE( st_AllocResult_get_pointer( &result ) != nullptr );
    ASSERT_TRUE( st_AllocResult_get_offset( &result )  >= uint64_t{ 0 } );
           
    /* ===================================================================== */
    
    st_Drift unpacked_drift;
    
    unsigned char* ptr_next = st_Drift_unpack_from_flat_memory( 
        &unpacked_drift, st_AllocResult_get_pointer( &result ) );
    
    ASSERT_TRUE( ptr_next != nullptr );
    
    ASSERT_TRUE( st_Drift_get_type_id( &unpacked_drift ) ==
                 st_Drift_get_type_id( &drift ) );
    
    ASSERT_TRUE( st_Drift_get_element_id( &unpacked_drift ) ==
                 st_Drift_get_element_id( &drift ) );
    
    ASSERT_TRUE( st_Drift_get_element_id_ptr( &unpacked_drift ) !=
                 st_Drift_get_element_id_ptr( &drift ) );
     
    ASSERT_TRUE( std::fabs( st_Drift_get_length( &unpacked_drift ) -
                            st_Drift_get_length( &drift ) ) < 1e-9 );
    
    ASSERT_TRUE( st_Drift_get_length_ptr( &unpacked_drift ) !=
                 st_Drift_get_length_ptr( &drift ) );
    
    /* ===================================================================== */
    
    st_MemPool_free( &mem_pool );
}

TEST( CommonTestsBlockDrift, PackDriftsToMemPoolAndUnpack )
{
    std::size_t const CHUNK_SIZE     = st_BLOCK_DEFAULT_MEMPOOL_CHUNK_SIZE;
    std::size_t const NUM_ELEMENTS   = std::size_t{ 10 };    
    std::size_t alignment            = std::size_t{ 32 };
    
    std::size_t const SIZE_PER_DRIFT = 
        st_Drift_predict_required_size_on_mempool_for_packing( alignment );
    
    std::size_t const CAPACITY = 
        ( NUM_ELEMENTS + std::size_t{ 1 } ) * SIZE_PER_DRIFT;
        
    st_MemPool mem_pool;    
    st_MemPool_init( &mem_pool, CAPACITY, CHUNK_SIZE );
    
    /* ===================================================================== */
    
    std::vector< st_DriftSingle >   cmp_drifts;
    cmp_drifts.clear();
    cmp_drifts.reserve( NUM_ELEMENTS );
    
    st_DriftSingle single_drift;
    st_DriftSingle_set_type_id( &single_drift, st_ELEMENT_TYPE_DRIFT );
    
    st_Drift drift;
    st_Drift_map_from_single_drift( &drift, &single_drift );
        
    std::random_device rd;
    std::mt19937 prng( rd() );
    std::uniform_real_distribution<> distribution( 0.1, 10.0 );
    
    int64_t element_id = int64_t{ 0 };
    
    unsigned char* ptr_read_begin = nullptr;
    
    for( std::size_t ii = 0 ; ii < NUM_ELEMENTS ; ++ii, ++element_id )
    {
        st_Drift_set_element_id( &drift, element_id );
        st_Drift_set_length( &drift, distribution( prng ) );
        
        cmp_drifts.push_back( single_drift );
        
        st_AllocResult result = 
            st_Drift_pack_aligned( &drift, &mem_pool, alignment );
            
        ASSERT_TRUE( st_AllocResult_valid( &result ) );
        ASSERT_TRUE( st_AllocResult_get_pointer( &result ) != nullptr );
        ASSERT_TRUE( st_AllocResult_get_offset( &result )  >= uint64_t{ 0 } );
        
        if( ii == std::size_t{ 0 } )
        {
            ptr_read_begin = st_AllocResult_get_pointer( &result );
        }
    }
    
    ASSERT_TRUE( ptr_read_begin != nullptr );    
    ASSERT_TRUE( cmp_drifts.size() == NUM_ELEMENTS );
    
    /* ===================================================================== */
    
    std::vector< st_Drift > unpacked_drifts;
    unpacked_drifts.clear();
    unpacked_drifts.reserve( NUM_ELEMENTS );
    
    unsigned char* ptr_to_read_pos = ptr_read_begin;
    
    for( std::size_t ii = 0 ; ii < NUM_ELEMENTS ; ++ii )
    {
        std::uintptr_t const read_begin_addr = 
            reinterpret_cast< std::uintptr_t >( ptr_to_read_pos );        
            
        ASSERT_TRUE( ( read_begin_addr % alignment ) == std::size_t{ 0 } );
        
        st_Drift unpacked;
        unsigned char* ptr_next = 
            st_Drift_unpack_from_flat_memory( &unpacked, ptr_to_read_pos );
            
        ASSERT_TRUE( ptr_next != 0 );
        ASSERT_TRUE( std::distance( ptr_to_read_pos, ptr_next ) > 0 );
                
        unpacked_drifts.push_back( unpacked );
        ptr_to_read_pos = ptr_next;
    }   
    
    ASSERT_TRUE( unpacked_drifts.size() == NUM_ELEMENTS );
    
    /* ===================================================================== */
    
    for( std::size_t ii = 0 ; ii < NUM_ELEMENTS ; ++ii )
    {
        st_DriftSingle& cmp = cmp_drifts[ ii ];
        st_Drift& unpacked  = unpacked_drifts[ ii ];
        
        ASSERT_TRUE( st_Drift_get_length_ptr( &unpacked ) != nullptr );
        ASSERT_TRUE( st_Drift_get_element_id_ptr( &unpacked ) != nullptr );
        ASSERT_TRUE( st_Drift_get_type_id( &unpacked ) == st_ELEMENT_TYPE_DRIFT );
        
        ASSERT_TRUE( reinterpret_cast< unsigned char* >( 
                        st_Drift_get_element_id_ptr( &unpacked ) ) !=
                     reinterpret_cast< unsigned char* >(
                        st_Drift_get_length_ptr( &unpacked ) ) );
        
        ASSERT_TRUE( st_Drift_get_type_id( &unpacked ) == 
                     st_DriftSingle_get_type_id( &cmp ) );
        
        ASSERT_TRUE( st_Drift_get_element_id( &unpacked ) == 
                     st_DriftSingle_get_element_id( &cmp ) );
        
        ASSERT_TRUE( st_Drift_get_element_id_ptr( &unpacked ) !=
                     st_DriftSingle_get_element_id_ptr( &cmp ) );
        
        ASSERT_TRUE( st_Drift_get_length_ptr( &unpacked ) !=
                     st_DriftSingle_get_length_ptr( &cmp ) );
        
        std::ptrdiff_t const temp_dist_length = std::distance(
            st_MemPool_get_buffer( &mem_pool ), reinterpret_cast< unsigned char* >( 
                st_Drift_get_length_ptr( &unpacked ) ) );
        
        ASSERT_TRUE( temp_dist_length >= 0 );
        ASSERT_TRUE( static_cast< std::size_t >( temp_dist_length ) <= CAPACITY );
        
        std::ptrdiff_t const temp_dist_elemid = std::distance(
            st_MemPool_get_buffer( &mem_pool ), reinterpret_cast< unsigned char* >( 
                st_Drift_get_element_id_ptr( &unpacked ) ) );
        
        ASSERT_TRUE( temp_dist_elemid >= 0 );
        ASSERT_TRUE( static_cast< std::size_t >( temp_dist_elemid ) <= CAPACITY );
        
        double const delta_length = std::fabs(
            st_Drift_get_length( &unpacked ) - st_DriftSingle_get_length( &cmp ) );
        
        ASSERT_TRUE( delta_length <= 1e-16 );
    }
    
    /* ===================================================================== */
    
    st_MemPool_free( &mem_pool );
}

/* end: sixtracklib/common/tests/test_block_drift.cpp */
