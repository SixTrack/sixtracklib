#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <limits>

#include <gtest/gtest.h>

#include "sixtracklib/_impl/definitions.h"
#include "sixtracklib/common/blocks.h"

typedef struct st_TestDrift_s
{
    st_block_type_num_t   type_id    __attribute__ (( aligned( 8 ) ));
    st_block_element_id_t object_id  __attribute__ (( aligned( 8 ) ));
    SIXTRL_REAL_T         length     __attribute__ (( aligned( 8 ) ));
}
st_TestDrift;

typedef struct st_TestMultipole_s
{
    st_block_type_num_t   type_id    __attribute__ (( aligned( 8 ) ));
    st_block_element_id_t object_id  __attribute__ (( aligned( 8 ) ));
    SIXTRL_REAL_T         length     __attribute__ (( aligned( 8 ) ));
    SIXTRL_INT64_T        order      __attribute__ (( aligned( 8 ) ));
    SIXTRL_REAL_T         hxl        __attribute__ (( aligned( 8 ) ));
    SIXTRL_REAL_T         hyl        __attribute__ (( aligned( 8 ) ));
    SIXTRL_REAL_T*        bal        __attribute__ (( aligned( 8 ) ));
}
st_TestMultipole;

TEST( CommonBlocksTests, InitAssembleCommitFree )
{
    static st_block_size_t const BLOCKS_CAPACITY = ( st_block_size_t )2;
    static st_block_size_t const DATA_CAPACITY   = ( st_block_size_t )0x1FFFFF;

    const SIXTRL_REAL_T REAL_CMP_EPS =
        std::numeric_limits< SIXTRL_REAL_T >::epsilon();

    st_Blocks blocks;
    st_Blocks_preset( &blocks );

    ASSERT_TRUE( st_Blocks_get_data_alignment( &blocks ) > 0u );

    ASSERT_TRUE( st_Blocks_get_begin_alignment( &blocks ) >=
                 st_Blocks_get_data_alignment( &blocks  ) );

    ASSERT_TRUE( ( st_Blocks_get_begin_alignment( &blocks ) %
                   st_Blocks_get_data_alignment( &blocks ) ) == 0u );

    int success = st_Blocks_init( &blocks, BLOCKS_CAPACITY, DATA_CAPACITY );
    ASSERT_TRUE( success == 0 );

    ASSERT_TRUE( !st_Blocks_are_serialized( &blocks ) );
    ASSERT_TRUE(  st_Blocks_has_data_store( &blocks ) );
    ASSERT_TRUE(  st_Blocks_has_index_store( &blocks ) );
    ASSERT_TRUE(  st_Blocks_has_data_pointers_store( &blocks ) );
    ASSERT_TRUE(  st_Blocks_get_num_of_blocks( &blocks ) == 0u );
    ASSERT_TRUE(  st_Blocks_get_max_num_of_blocks( &blocks ) >=
                  BLOCKS_CAPACITY );

    ASSERT_TRUE(  st_Blocks_get_data_size( &blocks ) != 0u );
    ASSERT_TRUE(  st_Blocks_get_data_capacity( &blocks ) >= DATA_CAPACITY );

    ASSERT_TRUE(  st_Blocks_get_num_data_pointers( &blocks ) == 0u );
    ASSERT_TRUE(  st_Blocks_get_max_num_data_pointers( &blocks ) >=
                  BLOCKS_CAPACITY );

    /* --------------------------------------------------------------------- */
    /* Add st_TestDrift instance: */

    st_TestDrift test_drift;

    test_drift.type_id   = st_BlockType_to_number( st_BLOCK_TYPE_DRIFT );
    test_drift.object_id = 1;
    test_drift.length    = 1.0;

    st_BlockInfo* ptr_drift_info = st_Blocks_add_block(
        &blocks, st_BLOCK_TYPE_DRIFT,
        sizeof( test_drift ), &test_drift, 0u, 0, 0, 0 );

    ASSERT_TRUE( ptr_drift_info != nullptr );
    ASSERT_TRUE( st_BlockInfo_get_type_id( ptr_drift_info ) ==
                 st_BLOCK_TYPE_DRIFT );

    ASSERT_TRUE( st_BlockInfo_get_block_size( ptr_drift_info ) >=
                 sizeof( test_drift ) );

    ASSERT_TRUE( st_BlockInfo_get_const_ptr_begin( ptr_drift_info ) !=
                 nullptr );

    ASSERT_TRUE( st_BlockInfo_get_const_ptr_metadata( ptr_drift_info ) ==
                 nullptr );

    ASSERT_TRUE( st_Blocks_get_num_of_blocks( &blocks ) == 1u );
    ASSERT_TRUE( st_Blocks_get_num_data_pointers( &blocks ) == 0u );

    /* --------------------------------------------------------------------- */
    /* Add st_TestMultipole instance: */

    st_TestMultipole test_multipole;

    test_multipole.type_id   = st_BLOCK_TYPE_MULTIPOLE;
    test_multipole.object_id = 2;
    test_multipole.length    = 0.5;
    test_multipole.order     = 3;
    test_multipole.hxl       = 0.1;
    test_multipole.hyl       = 0.1;
    test_multipole.bal       = 0;

    st_block_size_t const mp_num_data_ptrs = ( st_block_size_t )1u;
    st_block_size_t const mp_offsets[] = { offsetof( st_TestMultipole, bal ) };
    st_block_size_t const mp_sizes[]   = { sizeof( SIXTRL_REAL_T ) };

    st_block_size_t const mp_counts[]  =
        { ( st_block_size_t )test_multipole.order * 2u };

    st_BlockInfo* mp_block_info = st_Blocks_add_block( &blocks, st_BLOCK_TYPE_MULTIPOLE,
        sizeof( test_multipole ), &test_multipole, mp_num_data_ptrs,
            &mp_offsets[ 0 ], &mp_sizes[ 0 ], &mp_counts[ 0 ] );

    ASSERT_TRUE( mp_block_info  != nullptr );

    ASSERT_TRUE( st_BlockInfo_get_type_id( mp_block_info  ) ==
                 st_BLOCK_TYPE_MULTIPOLE );

    ASSERT_TRUE( st_BlockInfo_get_block_size( mp_block_info  ) >=
                 sizeof( test_drift ) );

    ASSERT_TRUE( st_BlockInfo_get_const_ptr_begin( mp_block_info  ) !=
                 nullptr );

    ASSERT_TRUE( st_BlockInfo_get_const_ptr_metadata( mp_block_info ) ==
                 nullptr );

    ASSERT_TRUE( st_Blocks_get_num_of_blocks( &blocks ) == 2u );
    ASSERT_TRUE( st_Blocks_get_num_data_pointers( &blocks ) == 1u );

    /* ===================================================================== */
    /* Reconstruct from original pointer: */

    success = st_Blocks_serialize( &blocks );

    ASSERT_TRUE( success == 0 );

    SIXTRL_GLOBAL_DEC unsigned char*
        mem_data_begin = st_Blocks_get_data_begin( &blocks );

    if( mem_data_begin != 0 )
    {
        st_Blocks restored_blocks;
        st_Blocks_preset( &restored_blocks );

        success = st_Blocks_unserialize( &restored_blocks, mem_data_begin );

        ASSERT_TRUE( success == 0 );

        ASSERT_TRUE( st_Blocks_get_num_data_pointers( &blocks ) ==
                     st_Blocks_get_num_data_pointers( &restored_blocks ) );

        ASSERT_TRUE( st_Blocks_get_num_of_blocks( &blocks ) ==
                     st_Blocks_get_num_of_blocks( &restored_blocks ) );

        SIXTRL_GLOBAL_DEC st_BlockInfo const* blocks_it =
            st_Blocks_get_const_block_infos_begin( &restored_blocks );

        SIXTRL_GLOBAL_DEC st_BlockInfo const* cmp_blocks_it =
            st_Blocks_get_const_block_infos_begin( &blocks );

        ASSERT_TRUE( blocks_it     != nullptr );
        ASSERT_TRUE( cmp_blocks_it != nullptr );
        ASSERT_TRUE( cmp_blocks_it == blocks_it );

        /* ----------------------------------------------------------------- */
        /* Compare reconstructed drift: */

        ASSERT_TRUE( st_BlockInfo_get_type_id( blocks_it ) ==
                     st_BLOCK_TYPE_DRIFT );

        st_TestDrift* drift =
            ( st_TestDrift* )st_BlockInfo_get_const_ptr_begin( blocks_it );

        ASSERT_TRUE( drift != 0 );
        ASSERT_TRUE( test_drift.object_id == drift->object_id );
        ASSERT_TRUE( test_drift.type_id   == drift->type_id   );
        ASSERT_TRUE( std::fabs( test_drift.length - drift->length ) <
                     REAL_CMP_EPS );

        ++blocks_it;

        /* ----------------------------------------------------------------- */
        /* Compare reconstructed multipole: */

        ASSERT_TRUE( st_BlockInfo_get_type_id( blocks_it ) ==
                     st_BLOCK_TYPE_MULTIPOLE );

        st_TestMultipole* multipole =
            ( st_TestMultipole* )st_BlockInfo_get_const_ptr_begin( blocks_it );

        ASSERT_TRUE( multipole != 0 );
        ASSERT_TRUE( test_multipole.object_id == multipole->object_id );
        ASSERT_TRUE( test_multipole.order     == multipole->order );
        ASSERT_TRUE( test_multipole.type_id   == st_BLOCK_TYPE_MULTIPOLE );

        ASSERT_TRUE( std::fabs( test_multipole.hxl - multipole->hxl ) <=
                     REAL_CMP_EPS );

        ASSERT_TRUE( std::fabs( test_multipole.hyl - multipole->hyl ) <=
                     REAL_CMP_EPS );

        SIXTRL_INT64_T ii = 0;
        SIXTRL_INT64_T const dim = 2 * test_multipole.order;

        for( ; ii < dim ; ++ii )
        {
            multipole->bal[ ii ] = SIXTRL_REAL_T{ 1.0 };
        }
    }

    /* ===================================================================== */

    st_block_size_t const num_of_bytes =
        st_Blocks_get_total_num_bytes( &blocks );

    SIXTRL_GLOBAL_DEC unsigned char* data_mem_begin =
        new unsigned char[ num_of_bytes ];

    if( data_mem_begin != nullptr )
    {
        st_Blocks copy_blocks;
        st_Blocks_preset( &copy_blocks );

        SIXTRL_GLOBAL_DEC unsigned char const* orig_data_mem_begin =
            st_Blocks_get_const_data_begin( &blocks );

        SIXTRL_GLOBAL_DEC unsigned char const* orig_data_mem_end =
            st_Blocks_get_const_data_end( &blocks );

        std::copy( orig_data_mem_begin, orig_data_mem_end, data_mem_begin );

        success  = st_Blocks_unserialize( &copy_blocks, data_mem_begin );

         ASSERT_TRUE( st_Blocks_get_num_data_pointers( &blocks ) ==
                     st_Blocks_get_num_data_pointers( &copy_blocks ) );

        ASSERT_TRUE( st_Blocks_get_num_of_blocks( &blocks ) ==
                     st_Blocks_get_num_of_blocks( &copy_blocks ) );

        SIXTRL_GLOBAL_DEC st_BlockInfo const* blocks_it =
            st_Blocks_get_const_block_infos_begin( &copy_blocks );

        SIXTRL_GLOBAL_DEC st_BlockInfo const* cmp_blocks_it =
            st_Blocks_get_const_block_infos_begin( &blocks );

        ASSERT_TRUE( blocks_it     != nullptr );
        ASSERT_TRUE( cmp_blocks_it != nullptr );
        ASSERT_TRUE( cmp_blocks_it != blocks_it );

        /* ----------------------------------------------------------------- */
        /* Compare reconstructed drift: */

        ASSERT_TRUE( st_BlockInfo_get_type_id( cmp_blocks_it ) ==
                     st_BLOCK_TYPE_DRIFT );

        ASSERT_TRUE( st_BlockInfo_get_type_id( blocks_it ) ==
                     st_BLOCK_TYPE_DRIFT );

        st_TestDrift const* cmp_drift = ( st_TestDrift const*
            )st_BlockInfo_get_const_ptr_begin( cmp_blocks_it );

            ASSERT_TRUE( cmp_drift != nullptr );

        st_TestDrift const* drift = ( st_TestDrift const*
            )st_BlockInfo_get_const_ptr_begin( blocks_it );

        ASSERT_TRUE( drift != nullptr );
        ASSERT_TRUE( drift != cmp_drift );

        ASSERT_TRUE( cmp_drift->object_id == drift->object_id );
        ASSERT_TRUE( cmp_drift->type_id   == drift->type_id   );
        ASSERT_TRUE( std::fabs( cmp_drift->length - drift->length ) <=
                     REAL_CMP_EPS );

        ++blocks_it;
        ++cmp_blocks_it;

        /* ----------------------------------------------------------------- */
        /* Compare reconstructed multipole: */

        ASSERT_TRUE( st_BlockInfo_get_type_id( cmp_blocks_it ) ==
                     st_BLOCK_TYPE_MULTIPOLE );

        ASSERT_TRUE( st_BlockInfo_get_type_id( blocks_it ) ==
                     st_BLOCK_TYPE_MULTIPOLE );

        st_TestMultipole const* cmp_multipole = ( st_TestMultipole const*
            )st_BlockInfo_get_const_ptr_begin( cmp_blocks_it );

        st_TestMultipole const* multipole = ( st_TestMultipole const*
            )st_BlockInfo_get_const_ptr_begin( blocks_it );

        ASSERT_TRUE( cmp_multipole != multipole );

        ASSERT_TRUE( multipole != 0 );
        ASSERT_TRUE( test_multipole.object_id == multipole->object_id );
        ASSERT_TRUE( test_multipole.order     == multipole->order );
        ASSERT_TRUE( test_multipole.type_id   == st_BLOCK_TYPE_MULTIPOLE );

        ASSERT_TRUE( std::fabs( test_multipole.hxl - multipole->hxl ) <=
                     REAL_CMP_EPS );

        ASSERT_TRUE( std::fabs( test_multipole.hyl - multipole->hyl ) <=
                     REAL_CMP_EPS );

        SIXTRL_INT64_T ii = 0;
        SIXTRL_INT64_T const dim = 2 * test_multipole.order;

        for( ; ii < dim ; ++ii )
        {
            ASSERT_TRUE( std::fabs( multipole->bal[ ii ] -
                    cmp_multipole->bal[ ii ] ) <= REAL_CMP_EPS );
        }



        delete[] data_mem_begin;
        data_mem_begin = 0;
    }



    st_Blocks_free( &blocks );
}

/* end: tests/sixtracklib/common/test_blocks.cpp */

