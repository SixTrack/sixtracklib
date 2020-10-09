#include "sixtracklib/common/track_job_cpu.h"

#include <iomanip>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "sixtracklib/testlib.hpp"

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/buffer.hpp"
#include "sixtracklib/common/buffer.h"

TEST( CXX_Cpu_CpuTrackJob_StoredBufferTests, MinimalUsage )
{
    namespace st = SIXTRL_CXX_NAMESPACE;
    using track_job_t = st::TrackJobCpu;
    using buffer_t = track_job_t::buffer_t;
    using c_buffer_t = track_job_t::c_buffer_t;
    using size_t = track_job_t::size_type;
    using buffer_addr_t = ::NS(buffer_addr_t);

    /* Init track-job -> should have no ext stored buffers up front: */

    track_job_t job;

    ASSERT_TRUE( !job.has_stored_buffers() );
    ASSERT_TRUE(  job.num_stored_buffers() == size_t{ 0 } );
    ASSERT_TRUE(  job.min_stored_buffer_id() == st::ARCH_ILLEGAL_BUFFER_ID );
    ASSERT_TRUE(  job.max_stored_buffer_id() == st::ARCH_ILLEGAL_BUFFER_ID );

    /* --------------------------------------------------------------------- */
    /* Create a buffer that is directly managed by the job */

    size_t const ext_buffer_01_capacity = size_t{ 1024 };
    size_t const ext_buffer_01_id = job.add_stored_buffer( ext_buffer_01_capacity );

    ASSERT_TRUE(  ext_buffer_01_id != st::ARCH_ILLEGAL_BUFFER_ID );
    ASSERT_TRUE(  job.num_stored_buffers() == size_t{ 1 } );
    ASSERT_TRUE(  job.min_stored_buffer_id() != st::ARCH_ILLEGAL_BUFFER_ID );
    ASSERT_TRUE(  job.max_stored_buffer_id() != st::ARCH_ILLEGAL_BUFFER_ID );

    ASSERT_TRUE(  ext_buffer_01_id == job.min_stored_buffer_id() );
    ASSERT_TRUE(  ext_buffer_01_id == job.max_stored_buffer_id() );
    ASSERT_TRUE(  job.owns_stored_buffer( ext_buffer_01_id ) );

    ASSERT_TRUE(  job.ptr_stored_cxx_buffer( ext_buffer_01_id ) != nullptr );
    ASSERT_TRUE(  job.ptr_stored_buffer( ext_buffer_01_id ) != nullptr );

    ASSERT_TRUE(  job.ptr_stored_cxx_buffer( ext_buffer_01_id )->getCApiPtr()
                    == job.ptr_stored_buffer( ext_buffer_01_id ) );

    ASSERT_TRUE(  job.ptr_stored_cxx_buffer( ext_buffer_01_id )->getCapacity()
                    == ext_buffer_01_capacity );

    /* --------------------------------------------------------------------- */
    /* Create an external C99 buffer that should not be managed/owned by
     * the track job */

    size_t ext_buffer_02_capacity = size_t{ 512 };
    c_buffer_t* ext_buffer_02 = ::NS(Buffer_new)( ext_buffer_02_capacity );
    SIXTRL_ASSERT( ext_buffer_02 != nullptr );

    SIXTRL_ASSERT( ::NS(Buffer_get_capacity)( ext_buffer_02 ) >=
                    ext_buffer_02_capacity );

    size_t const ext_buffer_02_id = job.add_stored_buffer(
        ext_buffer_02, false ); /* false == take no ownership */

    ASSERT_TRUE(  ext_buffer_02_id != st::ARCH_ILLEGAL_BUFFER_ID );
    ASSERT_TRUE(  job.num_stored_buffers() == size_t{ 2 } );
    ASSERT_TRUE(  job.min_stored_buffer_id() != st::ARCH_ILLEGAL_BUFFER_ID );
    ASSERT_TRUE(  job.max_stored_buffer_id() != st::ARCH_ILLEGAL_BUFFER_ID );

    ASSERT_TRUE(  ext_buffer_01_id == job.min_stored_buffer_id() );
    ASSERT_TRUE(  ext_buffer_02_id == job.max_stored_buffer_id() );
    ASSERT_TRUE( !job.owns_stored_buffer( ext_buffer_02_id ) );

    /* This is a C99 buffer -> it is not possible to access it as a C++ buffer*/
    ASSERT_TRUE(  job.ptr_stored_cxx_buffer( ext_buffer_02_id ) == nullptr );
    ASSERT_TRUE(  job.ptr_stored_buffer( ext_buffer_02_id ) != nullptr );
    ASSERT_TRUE(  job.ptr_stored_buffer( ext_buffer_02_id ) == ext_buffer_02 );

    /* --------------------------------------------------------------------- */
    /* Remove ext_buffer_02 again from the track job: */

    ASSERT_TRUE( st::ARCH_STATUS_SUCCESS == job.remove_stored_buffer(
        ext_buffer_02_id ) );

    ASSERT_TRUE(  job.num_stored_buffers() == size_t{ 1 } );
    ASSERT_TRUE(  job.min_stored_buffer_id() != st::ARCH_ILLEGAL_BUFFER_ID );
    ASSERT_TRUE(  job.max_stored_buffer_id() != st::ARCH_ILLEGAL_BUFFER_ID );

    ASSERT_TRUE(  ext_buffer_01_id == job.min_stored_buffer_id() );
    ASSERT_TRUE(  ext_buffer_02_id == job.max_stored_buffer_id() );
    ASSERT_TRUE( !job.owns_stored_buffer( ext_buffer_02_id ) );

    /* This is a C99 buffer -> it is not possible to access it as a C++ buffer*/
    ASSERT_TRUE(  job.ptr_stored_cxx_buffer( ext_buffer_02_id ) == nullptr );
    ASSERT_TRUE(  job.ptr_stored_buffer( ext_buffer_02_id ) == nullptr );

    /* --------------------------------------------------------------------- */
    /* Add ext_buffer_02 again, but this time let the job take ownership: */

    buffer_addr_t const ext_buffer_02_begin_addr =
        ::NS(Buffer_get_data_begin_addr)( ext_buffer_02 );

    size_t const ext_buffer_02_size =
        ::NS(Buffer_get_size)( ext_buffer_02 );

    ext_buffer_02_capacity = ::NS(Buffer_get_capacity)( ext_buffer_02 );

    size_t const ext_buffer_02_id_b = job.add_stored_buffer(
        ext_buffer_02, true ); /* true == takes ownership */

    uintptr_t const ptr_ext_buffer_02_addr =
        reinterpret_cast< uintptr_t >( ext_buffer_02 );

    ASSERT_TRUE(  ext_buffer_02_id_b != st::ARCH_ILLEGAL_BUFFER_ID );
    ASSERT_TRUE(  job.num_stored_buffers() == size_t{ 2 } );
    ASSERT_TRUE(  job.min_stored_buffer_id() != st::ARCH_ILLEGAL_BUFFER_ID );
    ASSERT_TRUE(  job.max_stored_buffer_id() != st::ARCH_ILLEGAL_BUFFER_ID );

    ASSERT_TRUE(  ext_buffer_01_id   == job.min_stored_buffer_id() );
    ASSERT_TRUE(  ext_buffer_02_id_b == job.max_stored_buffer_id() );
    ASSERT_TRUE(  job.owns_stored_buffer( ext_buffer_02_id_b ) );

    /* After taking ownership, the buffer is accessible also as a C++ buffer;
       but taking ownership creates a new c99 pointer as well -> you can no
       longer access ext_buffer_02 via it's old handle */

    ext_buffer_02 = nullptr;

    ASSERT_TRUE(  job.ptr_stored_buffer( ext_buffer_02_id_b )  != nullptr );
    ASSERT_TRUE(  job.ptr_stored_buffer( ext_buffer_02_id_b ) != nullptr );
    ASSERT_TRUE(  reinterpret_cast< uintptr_t >( job.ptr_stored_buffer(
                    ext_buffer_02_id_b ) ) != ptr_ext_buffer_02_addr );

    /* Transfering of ownership not only invalidates the old ext_buffer_02
     * handle, it should also "preset" it with safe default values */

    ASSERT_TRUE( ::NS(Buffer_get_size)( ext_buffer_02 ) == size_t{ 0 } );
    ASSERT_TRUE( ::NS(Buffer_get_capacity)( ext_buffer_02 ) == size_t{ 0 } );

    /* The values, i.e. start address, capacity and size, should still be
     * available via the new handle */

    c_buffer_t* ext_buffer_02_b = job.ptr_stored_buffer( ext_buffer_02_id_b );
    ASSERT_TRUE( ::NS(Buffer_get_data_begin_addr)( ext_buffer_02_b ) ==
                 ext_buffer_02_begin_addr );

    ASSERT_TRUE( ::NS(Buffer_get_size)( ext_buffer_02_b ) ==
                 ext_buffer_02_size );

    ASSERT_TRUE( ::NS(Buffer_get_capacity)( ext_buffer_02_b ) ==
                 ext_buffer_02_capacity );

    ASSERT_TRUE( job.ptr_stored_cxx_buffer( ext_buffer_02_id_b )->getCApiPtr()
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

    size_t const ext_buffer_03_id = job.add_stored_buffer(
        &ext_buffer_03, false ); /* false == take no ownership */

    ASSERT_TRUE(  ext_buffer_03_id != st::ARCH_ILLEGAL_BUFFER_ID );
    ASSERT_TRUE(  job.num_stored_buffers() == size_t{ 3 } );
    ASSERT_TRUE(  job.min_stored_buffer_id() != st::ARCH_ILLEGAL_BUFFER_ID );
    ASSERT_TRUE(  job.max_stored_buffer_id() != st::ARCH_ILLEGAL_BUFFER_ID );

    ASSERT_TRUE(  ext_buffer_01_id == job.min_stored_buffer_id() );
    ASSERT_TRUE(  ext_buffer_03_id == job.max_stored_buffer_id() );
    ASSERT_TRUE( !job.owns_stored_buffer( ext_buffer_03_id ) );

    /* This is a C99 buffer -> it is not possible to access it as a C++ buffer*/
    ASSERT_TRUE(  job.ptr_stored_cxx_buffer( ext_buffer_03_id ) == nullptr );
    ASSERT_TRUE(  job.ptr_stored_buffer( ext_buffer_03_id ) != nullptr );
    ASSERT_TRUE(  job.ptr_stored_buffer( ext_buffer_03_id ) ==
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

    size_t const ext_buffer_04_id = job.add_stored_buffer(
        &ext_buffer_04, true, false ); /* true: take ownership,
            false: do not delete as ext_buffer_04 is on the stack */

    ASSERT_TRUE(  ext_buffer_04_id != st::ARCH_ILLEGAL_BUFFER_ID );
    ASSERT_TRUE(  job.num_stored_buffers() == size_t{ 4 } );
    ASSERT_TRUE(  job.min_stored_buffer_id() != st::ARCH_ILLEGAL_BUFFER_ID );
    ASSERT_TRUE(  job.max_stored_buffer_id() != st::ARCH_ILLEGAL_BUFFER_ID );

    ASSERT_TRUE(  ext_buffer_01_id == job.min_stored_buffer_id() );
    ASSERT_TRUE(  ext_buffer_04_id == job.max_stored_buffer_id() );
    ASSERT_TRUE(  job.owns_stored_buffer( ext_buffer_04_id ) );

    ASSERT_TRUE(  job.ptr_stored_cxx_buffer( ext_buffer_04_id ) != nullptr );
    ASSERT_TRUE(  job.ptr_stored_buffer( ext_buffer_04_id ) != nullptr );
    ASSERT_TRUE(  job.ptr_stored_buffer( ext_buffer_04_id ) !=
                  &ext_buffer_04 );

    ASSERT_TRUE( ::NS(Buffer_get_data_begin_addr)( &ext_buffer_04 ) ==
                  buffer_addr_t{ 0 } );

    ASSERT_TRUE( ::NS(Buffer_get_capacity)( &ext_buffer_04 ) == size_t{ 0 } );
    ASSERT_TRUE( ::NS(Buffer_get_size)( &ext_buffer_04 ) == size_t{ 0 } );

    c_buffer_t* ptr_ext_buffer_04 =
        job.ptr_stored_buffer( ext_buffer_04_id );

    ASSERT_TRUE( ptr_ext_buffer_04 != nullptr );

    ASSERT_TRUE( ::NS(Buffer_get_capacity)( ptr_ext_buffer_04 ) ==
                 ext_buffer_04_capacity );

    ASSERT_TRUE( ::NS(Buffer_get_size)( ptr_ext_buffer_04 ) ==
                 ext_buffer_04_size );

    ASSERT_TRUE( ::NS(Buffer_get_data_begin_addr)( ptr_ext_buffer_04 ) ==
                 ext_buffer_04_begin_addr );

    /* --------------------------------------------------------------------- */
    /* Add ext_buffer_05, a C++ buffer, not taking ownership: */

    size_t ext_buffer_05_capacity = size_t{ 8192 };
    buffer_t ext_buffer_05( ext_buffer_05_capacity );
    c_buffer_t* ext_buffer_05_cptr = ext_buffer_05.getCApiPtr();
    SIXTRL_ASSERT( ext_buffer_05_cptr != nullptr );

    ext_buffer_05_capacity = ext_buffer_05.getCapacity();

    size_t const ext_buffer_05_id = job.add_stored_buffer(
        &ext_buffer_05, false ); /* false == take no ownership */

    ASSERT_TRUE(  ext_buffer_05_id != st::ARCH_ILLEGAL_BUFFER_ID );
    ASSERT_TRUE(  job.num_stored_buffers() == size_t{ 5 } );
    ASSERT_TRUE(  job.min_stored_buffer_id() != st::ARCH_ILLEGAL_BUFFER_ID );
    ASSERT_TRUE(  job.max_stored_buffer_id() != st::ARCH_ILLEGAL_BUFFER_ID );

    ASSERT_TRUE(  ext_buffer_01_id == job.min_stored_buffer_id() );
    ASSERT_TRUE(  ext_buffer_05_id == job.max_stored_buffer_id() );
    ASSERT_TRUE( !job.owns_stored_buffer( ext_buffer_05_id ) );

    ASSERT_TRUE(  job.ptr_stored_cxx_buffer( ext_buffer_05_id ) ==
                  &ext_buffer_05 );

    ASSERT_TRUE(  job.ptr_stored_buffer( ext_buffer_05_id ) ==
                  ext_buffer_05_cptr );

    ASSERT_TRUE(  job.ptr_stored_buffer( ext_buffer_05_id ) ==
                  ext_buffer_05.getCApiPtr() );

    /* --------------------------------------------------------------------- */
    /* Add ext_buffer_06, a C++ buffer but adding as a C99 buffer,
     * not taking ownership: */

    size_t ext_buffer_06_capacity = size_t{ 6000 };
    buffer_t ext_buffer_06( ext_buffer_06_capacity );
    c_buffer_t* ext_buffer_06_cptr = ext_buffer_06.getCApiPtr();
    SIXTRL_ASSERT( ext_buffer_06_cptr != nullptr );

    ext_buffer_06_capacity = ext_buffer_06.getCapacity();

    size_t const ext_buffer_06_id = job.add_stored_buffer(
        ext_buffer_06_cptr, false ); /* false == take no ownership */

    ASSERT_TRUE(  ext_buffer_06_id != st::ARCH_ILLEGAL_BUFFER_ID );
    ASSERT_TRUE(  job.num_stored_buffers() == size_t{ 6 } );
    ASSERT_TRUE(  job.min_stored_buffer_id() != st::ARCH_ILLEGAL_BUFFER_ID );
    ASSERT_TRUE(  job.max_stored_buffer_id() != st::ARCH_ILLEGAL_BUFFER_ID );

    ASSERT_TRUE(  ext_buffer_01_id == job.min_stored_buffer_id() );
    ASSERT_TRUE(  ext_buffer_06_id == job.max_stored_buffer_id() );
    ASSERT_TRUE( !job.owns_stored_buffer( ext_buffer_06_id ) );

    /* Added as a non-owning C99 pointer -> can not access via C++ interface */
    ASSERT_TRUE(  job.ptr_stored_cxx_buffer( ext_buffer_06_id ) == nullptr );

    ASSERT_TRUE(  job.ptr_stored_buffer( ext_buffer_06_id ) ==
                  ext_buffer_06_cptr );

    ASSERT_TRUE(  job.ptr_stored_buffer( ext_buffer_06_id ) ==
                  ext_buffer_06.getCApiPtr() );

    /* --------------------------------------------------------------------- */
    /* Add ext_buffer_07, a C++ buffer, transfering ownership by moving in */

    size_t ext_buffer_07_capacity = size_t{ 14000 };
    buffer_t ext_buffer_07( ext_buffer_07_capacity );

    ext_buffer_07_capacity = ext_buffer_07.getCapacity();

    size_t const ext_buffer_07_size = ext_buffer_07.getSize();

    buffer_addr_t const ext_buffer_07_begin_addr =
        ext_buffer_07.getDataBeginAddr();

    size_t const ext_buffer_07_id = job.add_stored_buffer(
        std::move( ext_buffer_07 ) );

    ASSERT_TRUE(  ext_buffer_07_id != st::ARCH_ILLEGAL_BUFFER_ID );
    ASSERT_TRUE(  job.num_stored_buffers() == size_t{ 7 } );
    ASSERT_TRUE(  job.min_stored_buffer_id() != st::ARCH_ILLEGAL_BUFFER_ID );
    ASSERT_TRUE(  job.max_stored_buffer_id() != st::ARCH_ILLEGAL_BUFFER_ID );

    ASSERT_TRUE(  ext_buffer_01_id == job.min_stored_buffer_id() );
    ASSERT_TRUE(  ext_buffer_07_id == job.max_stored_buffer_id() );
    ASSERT_TRUE(  job.owns_stored_buffer( ext_buffer_07_id ) );

    /* Added as a non-owning C99 pointer -> can not access via C++ interface */
    ASSERT_TRUE(  job.ptr_stored_buffer( ext_buffer_07_id ) != nullptr );
    ASSERT_TRUE(  job.ptr_stored_buffer( ext_buffer_07_id ) != nullptr );

    buffer_t* ptr_ext_buffer_07 = job.ptr_stored_cxx_buffer( ext_buffer_07_id );
    ASSERT_TRUE( ptr_ext_buffer_07 != nullptr );
    ASSERT_TRUE( ptr_ext_buffer_07->getCapacity() == ext_buffer_07_capacity );
    ASSERT_TRUE( ptr_ext_buffer_07->getSize() == ext_buffer_07_size );
    ASSERT_TRUE( ptr_ext_buffer_07->getDataBeginAddr() ==
                 ext_buffer_07_begin_addr );

    /* --------------------------------------------------------------------- */
    /* Add ext_buffer_08, a C++ buffer, transfer ownership */

    size_t ext_buffer_08_capacity = size_t{ 856 };
    buffer_t* ext_buffer_08 = new buffer_t( ext_buffer_08_capacity );
    ASSERT_TRUE( ext_buffer_08 != nullptr );

    ext_buffer_08_capacity = ext_buffer_08->getCapacity();

    size_t const ext_buffer_08_id =
        job.add_stored_buffer( ext_buffer_08, true );

    ASSERT_TRUE(  ext_buffer_08_id != st::ARCH_ILLEGAL_BUFFER_ID );
    ASSERT_TRUE(  job.num_stored_buffers() == size_t{ 8 } );
    ASSERT_TRUE(  job.min_stored_buffer_id() != st::ARCH_ILLEGAL_BUFFER_ID );
    ASSERT_TRUE(  job.max_stored_buffer_id() != st::ARCH_ILLEGAL_BUFFER_ID );

    ASSERT_TRUE(  ext_buffer_01_id == job.min_stored_buffer_id() );
    ASSERT_TRUE(  ext_buffer_08_id == job.max_stored_buffer_id() );
    ASSERT_TRUE(  job.owns_stored_buffer( ext_buffer_08_id ) );

    ext_buffer_08 = nullptr;

    /* Added as a non-owning C99 pointer -> can not access via C++ interface */
    ASSERT_TRUE(  job.ptr_stored_cxx_buffer( ext_buffer_08_id ) != nullptr );
    ASSERT_TRUE(  job.ptr_stored_buffer( ext_buffer_08_id ) != nullptr );
}

/* end: tests/sixtracklib/common/track/test_track_job_cpu_stored_buffer_cxx.cpp */
