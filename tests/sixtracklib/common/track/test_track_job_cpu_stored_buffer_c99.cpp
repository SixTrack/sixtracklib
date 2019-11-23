#include "sixtracklib/common/track_job_cpu.h"

#include <cstddef>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "sixtracklib/testlib.hpp"

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/buffer.h"

TEST( C99_Cpu_CpuTrackJobStoredBufferTests, MinimalUsage )
{
    using track_job_t   = ::NS(TrackJobCpu);
    using c_buffer_t    = ::NS(Buffer);
    using size_t        = ::NS(buffer_size_t);
    using buffer_addr_t = ::NS(buffer_addr_t);

    /* Init track-job -> should have no ext stored buffers up front: */

    track_job_t* job = ::NS(TrackJobCpu_create)();
    SIXTRL_ASSERT( job != nullptr );

    ASSERT_TRUE( !::NS(TrackJob_has_stored_buffers)( job ) );

    ASSERT_TRUE(  ::NS(TrackJob_num_stored_buffers)( job ) ==
                  size_t{ 0 } );

    ASSERT_TRUE(  ::NS(TrackJob_min_stored_buffer_id)( job ) ==
                  ::NS(ARCH_ILLEGAL_BUFFER_ID) );

    ASSERT_TRUE(  ::NS(TrackJob_max_stored_buffer_id)( job ) ==
                  ::NS(ARCH_ILLEGAL_BUFFER_ID) );

    /* --------------------------------------------------------------------- */
    /* Create a buffer that is directly managed by the job */

    size_t const ext_buffer_01_capacity = size_t{ 1024 };
    size_t const ext_buffer_01_id = ::NS(TrackJob_create_stored_buffer)(
        job, ext_buffer_01_capacity );

    ASSERT_TRUE(  ext_buffer_01_id != ::NS(ARCH_ILLEGAL_BUFFER_ID) );

    ASSERT_TRUE(  ::NS(TrackJob_num_stored_buffers)( job ) == size_t{ 1 } );
    ASSERT_TRUE(  ::NS(TrackJob_min_stored_buffer_id)( job ) !=
                  ::NS(ARCH_ILLEGAL_BUFFER_ID) );

    ASSERT_TRUE(  ::NS(TrackJob_max_stored_buffer_id)( job ) !=
                  ::NS(ARCH_ILLEGAL_BUFFER_ID) );

    ASSERT_TRUE(  ext_buffer_01_id ==
                  ::NS(TrackJob_min_stored_buffer_id)( job ) );

    ASSERT_TRUE(  ext_buffer_01_id ==
                  ::NS(TrackJob_max_stored_buffer_id)( job ) );

    ASSERT_TRUE(  ::NS(TrackJob_owns_stored_buffer)(
        job, ext_buffer_01_id ) );

    ASSERT_TRUE(  ::NS(TrackJob_stored_buffer)(
        job, ext_buffer_01_id ) != nullptr );

    ASSERT_TRUE(  ::NS(TrackJob_stored_buffer)(
        job, ext_buffer_01_id ) != nullptr );

    ASSERT_TRUE(  ext_buffer_01_capacity == ::NS(Buffer_get_capacity)(
        ::NS(TrackJob_stored_buffer)( job, ext_buffer_01_id ) ) );

    /* --------------------------------------------------------------------- */
    /* Create an external C99 buffer that should not be managed/owned by
     * the track job */

    size_t ext_buffer_02_capacity = size_t{ 512 };
    c_buffer_t* ext_buffer_02 = ::NS(Buffer_new)( ext_buffer_02_capacity );
    SIXTRL_ASSERT( ext_buffer_02 != nullptr );

    SIXTRL_ASSERT( ::NS(Buffer_get_capacity)( ext_buffer_02 ) >=
                    ext_buffer_02_capacity );

    size_t const ext_buffer_02_id = ::NS(TrackJob_add_stored_buffer)(
        job, ext_buffer_02, false, false ); /* false == take no ownership */

    ASSERT_TRUE(  ext_buffer_02_id != ::NS(ARCH_ILLEGAL_BUFFER_ID) );
    ASSERT_TRUE(  ::NS(TrackJob_num_stored_buffers)( job ) == size_t{ 2 } );

    ASSERT_TRUE(  ::NS(TrackJob_min_stored_buffer_id)( job ) !=
                  ::NS(ARCH_ILLEGAL_BUFFER_ID) );

    ASSERT_TRUE(  ::NS(TrackJob_max_stored_buffer_id)( job ) !=
                  ::NS(ARCH_ILLEGAL_BUFFER_ID) );

    ASSERT_TRUE(  ext_buffer_01_id ==
                  ::NS(TrackJob_min_stored_buffer_id)( job ) );

    ASSERT_TRUE(  ext_buffer_02_id ==
                  ::NS(TrackJob_max_stored_buffer_id)( job ) );

    ASSERT_TRUE( !::NS(TrackJob_owns_stored_buffer)(
        job, ext_buffer_02_id ) );

    ASSERT_TRUE(  ::NS(TrackJob_stored_buffer)(
        job, ext_buffer_02_id ) != nullptr );

    ASSERT_TRUE(  ::NS(TrackJob_stored_buffer)(
        job, ext_buffer_02_id ) == ext_buffer_02 );

    /* --------------------------------------------------------------------- */
    /* Remove ext_buffer_02 again from the track job: */

    ASSERT_TRUE( ::NS(ARCH_STATUS_SUCCESS) ==
        ::NS(TrackJob_remove_stored_buffer)( job, ext_buffer_02_id ) );

    ASSERT_TRUE(  ::NS(TrackJob_num_stored_buffers)( job ) == size_t{ 1 } );
    ASSERT_TRUE(  ::NS(TrackJob_min_stored_buffer_id)( job ) !=
                  ::NS(ARCH_ILLEGAL_BUFFER_ID) );

    ASSERT_TRUE(  ::NS(TrackJob_max_stored_buffer_id)( job ) !=
                  ::NS(ARCH_ILLEGAL_BUFFER_ID) );

    ASSERT_TRUE(  ext_buffer_01_id ==
                  ::NS(TrackJob_min_stored_buffer_id)( job ) );

    ASSERT_TRUE(  ext_buffer_02_id ==
                  ::NS(TrackJob_max_stored_buffer_id)( job ) );

    ASSERT_TRUE( !::NS(TrackJob_owns_stored_buffer)(
        job, ext_buffer_02_id ) );

    ASSERT_TRUE(  ::NS(TrackJob_stored_buffer)(
        job, ext_buffer_02_id ) == nullptr );

    /* --------------------------------------------------------------------- */
    /* Add ext_buffer_02 again, but this time let the job take ownership: */

    buffer_addr_t const ext_buffer_02_begin_addr =
        ::NS(Buffer_get_data_begin_addr)( ext_buffer_02 );

    size_t const ext_buffer_02_size =
        ::NS(Buffer_get_size)( ext_buffer_02 );

    ext_buffer_02_capacity = ::NS(Buffer_get_capacity)( ext_buffer_02 );

    size_t const ext_buffer_02_id_b = ::NS(TrackJob_add_stored_buffer)(
        job, ext_buffer_02, true, true ); /* true == takes ownership */

    uintptr_t const ptr_ext_buffer_02_addr =
        reinterpret_cast< uintptr_t >( ext_buffer_02 );

    ASSERT_TRUE(  ext_buffer_02_id_b != ::NS(ARCH_ILLEGAL_BUFFER_ID) );
    ASSERT_TRUE(  ::NS(TrackJob_num_stored_buffers)( job ) == size_t{ 2 } );

    ASSERT_TRUE(  ::NS(TrackJob_min_stored_buffer_id)( job ) !=
                  ::NS(ARCH_ILLEGAL_BUFFER_ID) );

    ASSERT_TRUE(  ::NS(TrackJob_max_stored_buffer_id)( job ) !=
                  ::NS(ARCH_ILLEGAL_BUFFER_ID) );

    ASSERT_TRUE( ext_buffer_01_id ==
                 ::NS(TrackJob_min_stored_buffer_id)( job ) );

    ASSERT_TRUE(  ext_buffer_02_id_b ==
                  ::NS(TrackJob_max_stored_buffer_id)( job ) );

    ASSERT_TRUE( ::NS(TrackJob_owns_stored_buffer)(
        job, ext_buffer_02_id_b ) );

    /* After taking ownership, the buffer is accessible also as a C++ buffer;
       but taking ownership creates a new c99 pointer as well -> you can no
       longer access ext_buffer_02 via it's old handle */

    ext_buffer_02 = nullptr;

    ASSERT_TRUE(  ::NS(TrackJob_stored_buffer)(
        job, ext_buffer_02_id_b ) != nullptr );

    ASSERT_TRUE(  ::NS(TrackJob_stored_buffer)(
        job, ext_buffer_02_id_b ) != nullptr );

    ASSERT_TRUE(  reinterpret_cast< uintptr_t >(
        ::NS(TrackJob_stored_buffer)( job, ext_buffer_02_id_b ) ) !=
            ptr_ext_buffer_02_addr );

    /* Transfering of ownership not only invalidates the old ext_buffer_02
     * handle, it should also "preset" it with safe default values */

    ASSERT_TRUE( ::NS(Buffer_get_size)( ext_buffer_02 ) == size_t{ 0 } );
    ASSERT_TRUE( ::NS(Buffer_get_capacity)( ext_buffer_02 ) == size_t{ 0 } );

    /* The values, i.e. start address, capacity and size, should still be
     * available via the new handle */

    c_buffer_t* ext_buffer_02_b = ::NS(TrackJob_stored_buffer)(
        job, ext_buffer_02_id_b );

    ASSERT_TRUE( ::NS(Buffer_get_data_begin_addr)( ext_buffer_02_b ) ==
                 ext_buffer_02_begin_addr );

    ASSERT_TRUE( ::NS(Buffer_get_size)( ext_buffer_02_b ) ==
                 ext_buffer_02_size );

    ASSERT_TRUE( ::NS(Buffer_get_capacity)( ext_buffer_02_b ) ==
                 ext_buffer_02_capacity );

    ASSERT_TRUE( ::NS(TrackJob_stored_buffer)( job, ext_buffer_02_id_b )
                 == ext_buffer_02_b );

    /* --------------------------------------------------------------------- */
    /* Add ext_buffer_03, a C99 buffer on the stack: */

    size_t const ext_buffer_03_capacity = size_t{ 2048 };
    std::vector< unsigned char > ext_buffer_03_data_store(
        ext_buffer_03_capacity );

    c_buffer_t ext_buffer_03;
    ::NS(Buffer_preset)( &ext_buffer_03 );
    ::NS(arch_status_t) status = ::NS(Buffer_init_on_flat_memory)(
        &ext_buffer_03, ext_buffer_03_data_store.data(),
            ext_buffer_03_capacity );

    SIXTRL_ASSERT( status == ::NS(ARCH_STATUS_SUCCESS) );
    ( void )status;

    SIXTRL_ASSERT( ::NS(Buffer_get_const_data_begin)(
        &ext_buffer_03 ) != nullptr );

    SIXTRL_ASSERT( ::NS(Buffer_get_data_begin_addr)( &ext_buffer_03 ) ==
        reinterpret_cast< uintptr_t >( ext_buffer_03_data_store.data() ) );

    SIXTRL_ASSERT( ::NS(Buffer_get_capacity)( &ext_buffer_03 ) >=
        ext_buffer_03_capacity );

    size_t const ext_buffer_03_id = ::NS(TrackJob_add_stored_buffer)(
        job, &ext_buffer_03, false, false ); /* false == take no ownership */

    ASSERT_TRUE(  ext_buffer_03_id != ::NS(ARCH_ILLEGAL_BUFFER_ID) );
    ASSERT_TRUE(  ::NS(TrackJob_num_stored_buffers)( job ) == size_t{ 3 } );

    ASSERT_TRUE(  ::NS(TrackJob_min_stored_buffer_id)( job ) !=
                  ::NS(ARCH_ILLEGAL_BUFFER_ID) );

    ASSERT_TRUE(  ::NS(TrackJob_max_stored_buffer_id)( job ) !=
                  ::NS(ARCH_ILLEGAL_BUFFER_ID) );

    ASSERT_TRUE(  ext_buffer_01_id ==
                  ::NS(TrackJob_min_stored_buffer_id)( job ) );

    ASSERT_TRUE(  ext_buffer_03_id ==
                  ::NS(TrackJob_max_stored_buffer_id)( job ) );

    ASSERT_TRUE( !::NS(TrackJob_owns_stored_buffer)(
        job, ext_buffer_03_id ) );

    ASSERT_TRUE(  ::NS(TrackJob_stored_buffer)(
        job, ext_buffer_03_id ) != nullptr );

    ASSERT_TRUE(  ::NS(TrackJob_stored_buffer)( job, ext_buffer_03_id ) ==
                  &ext_buffer_03 );

    /* --------------------------------------------------------------------- */
    /* Add ext_buffer_04, a C99 buffer on the stack, but take ownership: */

    size_t ext_buffer_04_capacity = size_t{ 4096 };
    std::vector< unsigned char > ext_buffer_04_data_store(
        ext_buffer_04_capacity );

    c_buffer_t ext_buffer_04;
    ::NS(Buffer_preset)( &ext_buffer_04 );
    status = ::NS(Buffer_init_on_flat_memory)( &ext_buffer_04,
        ext_buffer_04_data_store.data(), ext_buffer_04_capacity );

    SIXTRL_ASSERT( status == ::NS(ARCH_STATUS_SUCCESS) );

    buffer_addr_t const ext_buffer_04_begin_addr =
        ::NS(Buffer_get_data_begin_addr)( &ext_buffer_04 );

    size_t const ext_buffer_04_size = ::NS(Buffer_get_size)( &ext_buffer_04 );
    ext_buffer_04_capacity = ::NS(Buffer_get_capacity)( &ext_buffer_04 );

    SIXTRL_ASSERT( ::NS(Buffer_get_const_data_begin)(
        &ext_buffer_04 ) != nullptr );

    SIXTRL_ASSERT( ::NS(Buffer_get_capacity)( &ext_buffer_04 ) >=
        ext_buffer_04_capacity );

    SIXTRL_ASSERT( reinterpret_cast< uintptr_t >(
        ext_buffer_04_data_store.data() ) == ext_buffer_04_begin_addr );

    size_t const ext_buffer_04_id = ::NS(TrackJob_add_stored_buffer)(
        job, &ext_buffer_04, true, false ); /* true: take ownership,
            false: do not delete as ext_buffer_04 is on the stack */

    ASSERT_TRUE(  ext_buffer_04_id != ::NS(ARCH_ILLEGAL_BUFFER_ID) );
    ASSERT_TRUE(  ::NS(TrackJob_num_stored_buffers)( job ) == size_t{ 4 } );

    ASSERT_TRUE(  ::NS(TrackJob_min_stored_buffer_id)( job ) !=
                  ::NS(ARCH_ILLEGAL_BUFFER_ID) );

    ASSERT_TRUE(  ::NS(TrackJob_max_stored_buffer_id)( job ) !=
                  ::NS(ARCH_ILLEGAL_BUFFER_ID) );

    ASSERT_TRUE(  ext_buffer_01_id ==
                  ::NS(TrackJob_min_stored_buffer_id)( job ) );

    ASSERT_TRUE(  ext_buffer_04_id ==
                  ::NS(TrackJob_max_stored_buffer_id)( job ) );

    ASSERT_TRUE(  ::NS(TrackJob_owns_stored_buffer)(
        job, ext_buffer_04_id ) );

    ASSERT_TRUE(  ::NS(TrackJob_stored_buffer)(
        job, ext_buffer_04_id ) != nullptr );

    ASSERT_TRUE(  ::NS(TrackJob_stored_buffer)(
        job, ext_buffer_04_id ) != &ext_buffer_04 );

    ASSERT_TRUE( ::NS(Buffer_get_data_begin_addr)( &ext_buffer_04 ) ==
                  buffer_addr_t{ 0 } );

    ASSERT_TRUE( ::NS(Buffer_get_capacity)( &ext_buffer_04 ) == size_t{ 0 } );
    ASSERT_TRUE( ::NS(Buffer_get_size)( &ext_buffer_04 ) == size_t{ 0 } );

    c_buffer_t* ptr_ext_buffer_04 =
        ::NS(TrackJob_stored_buffer)( job, ext_buffer_04_id );

    ASSERT_TRUE( ptr_ext_buffer_04 != nullptr );

    ASSERT_TRUE( ::NS(Buffer_get_capacity)( ptr_ext_buffer_04 ) ==
                 ext_buffer_04_capacity );

    ASSERT_TRUE( ::NS(Buffer_get_size)( ptr_ext_buffer_04 ) ==
                 ext_buffer_04_size );

    ASSERT_TRUE( ::NS(Buffer_get_data_begin_addr)( ptr_ext_buffer_04 ) ==
                 ext_buffer_04_begin_addr );

    ::NS(TrackJob_delete)( job );
}

/* end: tests/sixtracklib/common/track/test_track_job_cpu_stored_buffer_c99.cpp */
