#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>

#include <gtest/gtest.h>

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/buffer/mem_pool.h"

/* ========================================================================== */
/* ====  Test basic usage of MemPool: init and free operations */

TEST( C99_CommonMemPoolTests, InitFreeBasic )
{
    st_MemPool mem_pool;

    /* --------------------------------------------------------------------- */
    /* Test the "happy path": capacity and chunk_size are compatible */

    std::size_t const capacity = std::size_t{64};
    std::size_t const chunk_size = std::size_t{8};

    st_MemPool_init( &mem_pool, capacity, chunk_size );

    ASSERT_TRUE( st_MemPool_get_begin_pos( &mem_pool )  != nullptr );
    ASSERT_TRUE( st_MemPool_get_capacity( &mem_pool )   >= capacity );
    ASSERT_TRUE( st_MemPool_get_chunk_size( &mem_pool ) == chunk_size );
    ASSERT_TRUE( st_MemPool_get_size( &mem_pool )       == std::size_t{0} );

    st_MemPool_free( &mem_pool );

    ASSERT_TRUE( st_MemPool_get_begin_pos( &mem_pool )  == nullptr );
    ASSERT_TRUE( st_MemPool_get_capacity( &mem_pool )   == std::size_t{0} );
    ASSERT_TRUE( st_MemPool_get_chunk_size( &mem_pool ) == std::size_t{0} );
    ASSERT_TRUE( st_MemPool_get_size( &mem_pool )       == std::size_t{0} );
}

/* ========================================================================== */
/* ====  Test handling of odd-sized but admissible capacities */

TEST( C99_CommonMemPoolTests, InitFreeNonIntegerNumChunks )
{
    st_MemPool mem_pool;

    /* --------------------------------------------------------------------- */
    /* Test how the MemoryPool operates if the capacity is not an integer
     * multiple of the chunk size: */

    std::size_t const capacity = std::size_t{62};
    std::size_t const chunk_size = std::size_t{8};

    static std::size_t const ZERO_SIZE = std::size_t{0};

    st_MemPool_init( &mem_pool, capacity, chunk_size );

    ASSERT_TRUE( st_MemPool_get_begin_pos( &mem_pool ) != nullptr );
    ASSERT_TRUE( st_MemPool_get_capacity( &mem_pool ) >= capacity );

    ASSERT_TRUE( ZERO_SIZE ==
                 ( st_MemPool_get_capacity( &mem_pool ) % chunk_size ) );

    ASSERT_TRUE( st_MemPool_get_chunk_size( &mem_pool ) == chunk_size );
    ASSERT_TRUE( st_MemPool_get_size( &mem_pool ) == std::size_t{0} );

    st_MemPool_free( &mem_pool );
}

/* ========================================================================== */
/* ====  Test handling of pathological zero-sized capacities*/

TEST( C99_CommonMemPoolTests, InitFreeZeroCapacityNonZeroChunk )
{
    st_MemPool mem_pool;

    /* --------------------------------------------------------------------- */
    /* Test how the MemoryPool operates if the capacity is not an integer
     * multiple of the chunk size: */

    std::size_t const capacity = std::size_t{0};
    std::size_t const chunk_size = std::size_t{8};

    st_MemPool_init( &mem_pool, capacity, chunk_size );

    ASSERT_TRUE( st_MemPool_get_begin_pos( &mem_pool ) != nullptr );
    ASSERT_TRUE( st_MemPool_get_capacity( &mem_pool ) >= capacity );
    ASSERT_TRUE( st_MemPool_get_capacity( &mem_pool ) >= chunk_size );
    ASSERT_TRUE( st_MemPool_get_chunk_size( &mem_pool ) == chunk_size );
    ASSERT_TRUE( st_MemPool_get_size( &mem_pool ) == std::size_t{0} );

    st_MemPool_free( &mem_pool );
}

/* ========================================================================== */
/* ====  Happy-Path-Testing of adding blocks aligned and non-aligned */

TEST( C99_CommonMemPoolTests, AppendSuccess )
{
    st_MemPool mem_pool;

    std::size_t const chunk_size = std::size_t{8u};
    std::size_t const capacity = 12 * chunk_size;

    st_MemPool_init( &mem_pool, capacity, chunk_size );

    /* --------------------------------------------------------------------- */
    std::size_t num_bytes_to_add = std::size_t{2} * chunk_size;
    std::size_t expected_length = num_bytes_to_add;
    uint64_t expected_offset = uint64_t{0};

    st_AllocResult
    result = st_MemPool_append( &mem_pool, num_bytes_to_add );

    ASSERT_TRUE( st_AllocResult_valid( &result ) );
    ASSERT_TRUE( st_AllocResult_get_offset( &result ) == expected_offset );
    ASSERT_TRUE( st_AllocResult_get_length( &result ) == expected_length );

    /* --------------------------------------------------------------------- */

    num_bytes_to_add = ( chunk_size >> 1 );
    expected_length  =   chunk_size;
    expected_offset  = st_AllocResult_get_offset( &result )
                     + st_AllocResult_get_length( &result );

    result = st_MemPool_append( &mem_pool, num_bytes_to_add );

    ASSERT_TRUE( st_AllocResult_valid( &result ) );
    ASSERT_TRUE( st_AllocResult_get_offset( &result ) >= expected_offset );
    ASSERT_TRUE( st_AllocResult_get_length( &result ) == expected_length );

    /* --------------------------------------------------------------------- */

    std::size_t alignment = chunk_size << 1;
    num_bytes_to_add = chunk_size << 2;
    expected_length  = num_bytes_to_add;
    expected_offset  = st_AllocResult_get_offset( &result )
                     + st_AllocResult_get_length( &result );

    result = st_MemPool_append_aligned( &mem_pool, num_bytes_to_add,
                                        alignment );

    ASSERT_TRUE( st_AllocResult_valid( &result ) );
    ASSERT_TRUE( st_AllocResult_get_offset( &result ) >= expected_offset );
    ASSERT_TRUE( st_AllocResult_get_length( &result ) == expected_length );

    /* --------------------------------------------------------------------- */

    alignment = chunk_size;
    num_bytes_to_add = chunk_size << 1;
    expected_length  = num_bytes_to_add;
    expected_offset  = st_AllocResult_get_offset( &result )
                     + st_AllocResult_get_length( &result );

    result =
        st_MemPool_append_aligned( &mem_pool, num_bytes_to_add, alignment );

    ASSERT_TRUE( st_AllocResult_valid( &result ) );
    ASSERT_TRUE( st_AllocResult_get_offset( &result )  >= expected_offset );
    ASSERT_TRUE( st_AllocResult_get_length( &result )  == expected_length );

    /* --------------------------------------------------------------------- */

    expected_offset = st_AllocResult_get_offset( &result )
                    + st_AllocResult_get_length( &result );

    ASSERT_TRUE( st_MemPool_get_size( &mem_pool ) >= expected_offset );

    /* --------------------------------------------------------------------- */

    st_MemPool_clear( &mem_pool );

    ASSERT_TRUE( st_MemPool_get_begin_pos( &mem_pool )  != nullptr );
    ASSERT_TRUE( st_MemPool_get_capacity( &mem_pool )   >= capacity );
    ASSERT_TRUE( st_MemPool_get_chunk_size( &mem_pool ) == chunk_size );
    ASSERT_TRUE( st_MemPool_get_size( &mem_pool )       == std::size_t{0} );

    /* --------------------------------------------------------------------- */

    st_MemPool_free( &mem_pool );
}

/* ========================================================================== */

TEST( C99_CommonMemPoolTests, AppendAlignedWithPathologicalAlignment )
{
    st_MemPool mem_pool;

    static std::uintptr_t const ZERO_ALIGN = std::uintptr_t{ 0 };
    std::size_t const chunk_size = std::size_t{8u};

    /* we will use a pathological alignment here, so make sure the buffer is
     * large enough! */
    std::size_t const capacity = chunk_size * chunk_size * std::size_t{ 2 };

    st_MemPool_init( &mem_pool, capacity, chunk_size );

    unsigned char* ptr_buffer_begin = st_MemPool_get_begin_pos( &mem_pool );
    ASSERT_TRUE( ptr_buffer_begin != nullptr );

    std::uintptr_t const buffer_begin_addr =
        reinterpret_cast< std::uintptr_t >( ptr_buffer_begin );

    ASSERT_TRUE( buffer_begin_addr > ZERO_ALIGN );

    /* --------------------------------------------------------------------- */
    /* Try to add a non-zero-length block with a "strange" alignment -
     * it should align to the least common multiple of the provided
     * alignment and the chunk size to be divisible by both quantities.
     *
     * Note that this is potentially very wasteful with memory, so be
     * advised to avoid this! */

    std::size_t const alignment = chunk_size - std::size_t{ 1 };
    st_AllocResult const result =
        st_MemPool_append_aligned( &mem_pool, chunk_size, alignment );

    ASSERT_TRUE( st_AllocResult_valid( &result ) );

    unsigned char* ptr_begin    = st_AllocResult_get_pointer( &result );
    uint64_t const block_len    = st_AllocResult_get_length( &result );
    uint64_t const block_offset = st_AllocResult_get_offset( &result );

    ASSERT_TRUE( ptr_begin != nullptr );

    std::uintptr_t const begin_addr =
        reinterpret_cast< std::uintptr_t >( ptr_begin );

    ASSERT_TRUE( ( block_offset + buffer_begin_addr ) == begin_addr );
    ASSERT_TRUE( ( begin_addr % chunk_size ) == ZERO_ALIGN );
    ASSERT_TRUE( ( begin_addr % alignment  ) == ZERO_ALIGN );


    ASSERT_TRUE( block_len <= capacity );
    ASSERT_TRUE( st_MemPool_get_begin_pos( &mem_pool ) != nullptr  );
    ASSERT_TRUE( st_MemPool_get_capacity( &mem_pool )  >= capacity );

    st_MemPool_free( &mem_pool );
}

/* ========================================================================== */
/* ====  Test the failing of adding blocks with problematic properties        */

TEST( C99_CommonMemPoolTests, AppendFailures )
{
    st_MemPool mem_pool;

    std::size_t const chunk_size = std::size_t{8u};
    std::size_t const capacity = 8 * chunk_size;

    static std::size_t const ZERO_SIZE = std::size_t{0};

    st_MemPool_init( &mem_pool, capacity, chunk_size );

    /* --------------------------------------------------------------------- */
    /* Asked to add a block into an empty MemPool that would exceed capacity */

    std::size_t num_bytes_to_add = std::size_t{10} * chunk_size;

    st_AllocResult
    result = st_MemPool_append( &mem_pool, num_bytes_to_add );

    ASSERT_TRUE( !st_AllocResult_valid( &result ) );
    ASSERT_TRUE( st_AllocResult_get_pointer( &result ) == nullptr );
    ASSERT_TRUE( st_AllocResult_get_offset( &result ) == uint64_t{0} );
    ASSERT_TRUE( st_AllocResult_get_length( &result ) == uint64_t{0} );

    ASSERT_TRUE( st_MemPool_get_begin_pos( &mem_pool ) != nullptr );
    ASSERT_TRUE( st_MemPool_get_capacity( &mem_pool )   >= capacity );
    ASSERT_TRUE( st_MemPool_get_chunk_size( &mem_pool ) == chunk_size );
    ASSERT_TRUE( st_MemPool_get_size( &mem_pool ) == ZERO_SIZE );

    /* --------------------------------------------------------------------- */
    /* Add a block successfully - so we can check whether the MemPool keeps
     * its properties if we insert a block - nonsuccessfully */

    result = st_MemPool_append( &mem_pool, chunk_size );
    ASSERT_TRUE( st_AllocResult_valid( &result ) );

    std::size_t const current_size = st_MemPool_get_size( &mem_pool );

    /* --------------------------------------------------------------------- */
    /* Try to add a block with zero bytes length */

    result = st_MemPool_append( &mem_pool, ZERO_SIZE );

    ASSERT_TRUE( !st_AllocResult_valid( &result ) );
    ASSERT_TRUE( st_AllocResult_get_pointer( &result ) == nullptr );
    ASSERT_TRUE( st_AllocResult_get_offset( &result ) == uint64_t{0} );
    ASSERT_TRUE( st_AllocResult_get_length( &result ) == uint64_t{0} );

    ASSERT_TRUE( st_MemPool_get_begin_pos( &mem_pool ) != nullptr );
    ASSERT_TRUE( st_MemPool_get_capacity( &mem_pool )   >= capacity );
    ASSERT_TRUE( st_MemPool_get_chunk_size( &mem_pool ) == chunk_size );
    ASSERT_TRUE( st_MemPool_get_size( &mem_pool ) == current_size );

    /* --------------------------------------------------------------------- */
    /* Try to add a valid block exceeding the number of remaining bytes      */

    std::size_t remaining_bytes =
        st_MemPool_get_remaining_bytes( &mem_pool );

    std::size_t const too_large = remaining_bytes + std::size_t{1};

    result = st_MemPool_append( &mem_pool, too_large );

    ASSERT_TRUE( !st_AllocResult_valid( &result ) );
    ASSERT_TRUE( st_AllocResult_get_pointer( &result ) == nullptr );
    ASSERT_TRUE( st_AllocResult_get_offset( &result ) == uint64_t{0} );
    ASSERT_TRUE( st_AllocResult_get_length( &result ) == uint64_t{0} );

    ASSERT_TRUE( st_MemPool_get_begin_pos(  &mem_pool ) != nullptr );
    ASSERT_TRUE( st_MemPool_get_capacity(   &mem_pool ) >= capacity );
    ASSERT_TRUE( st_MemPool_get_chunk_size( &mem_pool ) == chunk_size );
    ASSERT_TRUE( st_MemPool_get_size( &mem_pool ) == current_size );

    /* --------------------------------------------------------------------- */
    /* Try to add a valid block fitting in memory but exceeding the capacity
     * due to too ambitious alignment requirements: */

    /* Step one: find the "too ambitious alignment" values */

    uint64_t const current_offset =
        st_MemPool_get_next_begin_offset( &mem_pool, chunk_size );

    ASSERT_TRUE( current_offset != UINT64_MAX );

    std::size_t alignment = ( chunk_size << 1 );
    std::size_t end_alignment = capacity;

    bool found_alignment_for_test = false;

    for( ; alignment < end_alignment; alignment += chunk_size )
    {
        if( st_MemPool_get_next_begin_offset( &mem_pool, alignment ) >
            current_offset )
        {
            result = st_MemPool_append_aligned(
                &mem_pool, remaining_bytes, alignment );

            found_alignment_for_test = true;

            break;
        }
    }

    if( found_alignment_for_test )
    {
        ASSERT_TRUE( !st_AllocResult_valid( &result ) );
        ASSERT_TRUE( st_AllocResult_get_pointer( &result ) == nullptr );
        ASSERT_TRUE( st_AllocResult_get_offset( &result )  == uint64_t{0} );
        ASSERT_TRUE( st_AllocResult_get_length( &result )  == uint64_t{0} );

        ASSERT_TRUE( st_MemPool_get_begin_pos( &mem_pool )  != nullptr );
        ASSERT_TRUE( st_MemPool_get_capacity(  &mem_pool )  >= capacity );
        ASSERT_TRUE( st_MemPool_get_chunk_size( &mem_pool ) == chunk_size );
        ASSERT_TRUE( st_MemPool_get_size( &mem_pool )       == current_size );
    }

    /* Verify that non-aligned insert would work, however: */

    result = st_MemPool_append( &mem_pool, remaining_bytes );

    ASSERT_TRUE( st_AllocResult_valid( &result ) );
    ASSERT_TRUE( st_AllocResult_get_pointer( &result ) != nullptr );
    ASSERT_TRUE( st_AllocResult_get_offset( &result ) == current_size );
    ASSERT_TRUE( st_AllocResult_get_length( &result ) == remaining_bytes );

    ASSERT_TRUE( st_MemPool_get_begin_pos( &mem_pool )  != nullptr );
    ASSERT_TRUE( st_MemPool_get_capacity( &mem_pool )   >= capacity );
    ASSERT_TRUE( st_MemPool_get_chunk_size( &mem_pool ) == chunk_size );
    ASSERT_TRUE( st_MemPool_get_size( &mem_pool )       >= capacity );
    ASSERT_TRUE( st_MemPool_get_size( &mem_pool )       ==
                 st_MemPool_get_capacity( &mem_pool )  );
    ASSERT_TRUE( st_MemPool_get_remaining_bytes( &mem_pool ) == ZERO_SIZE );

    /* --------------------------------------------------------------------- */

    st_MemPool_free( &mem_pool );
}

/* end: tests/sixtracklib/common/test_mem_pool.cpp */
