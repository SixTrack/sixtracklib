#include "sixtracklib/opencl/track_job_cl.h"

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

TEST( C99_OpenCL_TrackJobCl_StoredBufferTests, MinimalUsage )
{
    using track_job_t = ::NS(TrackJobCl);
    using context_t = ::NS(ClContext);
    using node_id_t = ::NS(ComputeNodeId);
    using node_info_t = ::NS(ComputeNodeInfo);
    using c_buffer_t = ::NS(Buffer);
    using size_t = ::NS(arch_size_t);
    using arg_t = ::NS(ClArgument);
    using status_t = ::NS(arch_status_t);
    using buffer_addr_t = ::NS(buffer_addr_t);

    /* Init track-job -> should have no ext stored buffers up front: */

    std::vector< node_id_t > node_ids;
    size_t const num_available_nodes = context_t::NUM_AVAILABLE_NODES();
    size_t num_nodes = size_t{ 0 };

    if(  num_available_nodes > size_t{ 0 } )
    {
        node_ids.resize( num_available_nodes, node_id_t{} );
        num_nodes = context_t::GET_AVAILABLE_NODES(
            node_ids.data(), node_ids.size() );

        ASSERT_TRUE( num_nodes <= num_available_nodes );
    }

    if( num_nodes == size_t{ 0 } )
    {
        std::cout << "No available OpenCL nodes -> skipping test\r\n";
    }

    bool first = true;

    for( size_t idx = size_t{ 0 } ; idx < num_nodes ; ++idx )
    {
        char node_id_str[ 32 ] =
        {
            '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0',
            '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0',
            '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0',
            '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0'
        };

        node_id_t const node_id = node_ids[ idx ];

        status_t status = ::NS(ComputeNodeId_to_string)(
            &node_id, &node_id_str[ 0 ], 32 );
        ASSERT_TRUE( status == ::NS(ARCH_STATUS_SUCCESS) );

        track_job_t* job = ::NS(TrackJobCl_create)( node_id_str );
        SIXTRL_ASSERT( job != nullptr );
        ASSERT_TRUE( ::NS(TrackJobCl_get_context)( job ) != nullptr );

        node_info_t const* node_info =
            ::NS(ClContextBase_get_available_node_info_by_node_id)(
                ::NS(TrackJobCl_get_context)( job ), &node_id );

        node_id_t const default_node_id =
            ::NS(ClContextBase_get_default_node_id)(
                ::NS(TrackJobCl_get_context)( job ) );

        if( node_info == nullptr ) continue;

        if( !first )
        {
            std::cout << "\r\n------------------------------------------------"
                      << "-------------------------------";

        }
        else
        {
            first = false;
        }

        ::NS(ComputeNodeInfo_print_out)( node_info, &default_node_id );
        std::cout << std::endl;

        /* Init track-job -> should have no ext stored buffers up front: */

        ASSERT_TRUE( !::NS(TrackJob_has_stored_buffers)( job ) );

        ASSERT_TRUE(  ::NS(TrackJob_num_stored_buffers)( job ) ==
                    size_t{ 0 } );

        ASSERT_TRUE(  ::NS(TrackJob_min_stored_buffer_id)( job ) ==
                    ::NS(ARCH_ILLEGAL_BUFFER_ID) );

        ASSERT_TRUE(  ::NS(TrackJob_max_stored_buffer_id)( job ) ==
                    ::NS(ARCH_ILLEGAL_BUFFER_ID) );

        /* ----------------------------------------------------------------- */
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


        arg_t* ext_buffer_01_arg = ::NS(TrackJobCl_argument_by_buffer_id)(
            job, ext_buffer_01_id );

        ASSERT_TRUE(  ext_buffer_01_arg != nullptr );
        ASSERT_TRUE(  ::NS(ClArgument_get_ptr_to_context)(
            ext_buffer_01_arg ) == ::NS(TrackJobCl_get_context)( job ) );

        ASSERT_TRUE(  ::NS(ClArgument_uses_cobj_buffer)( ext_buffer_01_arg ) );
        ASSERT_TRUE(  ::NS(ClArgument_get_ptr_cobj_buffer)(
            ext_buffer_01_arg ) == ::NS(TrackJob_stored_buffer)(
                job, ext_buffer_01_id ) );

        ASSERT_TRUE( ::NS(ClArgument_write)( ext_buffer_01_arg,
            ::NS(TrackJob_stored_buffer)( job, ext_buffer_01_id ) ) );

        ASSERT_TRUE( ::NS(ClArgument_read)( ext_buffer_01_arg,
            ::NS(TrackJob_stored_buffer)( job, ext_buffer_01_id ) ) );

        /* ----------------------------------------------------------------- */
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


        arg_t* ext_buffer_02_arg = ::NS(TrackJobCl_argument_by_buffer_id)(
            job, ext_buffer_02_id );

        ASSERT_TRUE(  ext_buffer_02_arg != nullptr );
        ASSERT_TRUE(  ::NS(ClArgument_get_ptr_to_context)(
            ext_buffer_02_arg ) == ::NS(TrackJobCl_get_context)( job ) );

        ASSERT_TRUE(  ::NS(ClArgument_uses_cobj_buffer)( ext_buffer_02_arg ) );
        ASSERT_TRUE(  ::NS(ClArgument_get_ptr_cobj_buffer)(
            ext_buffer_02_arg ) == ext_buffer_02 );

        ASSERT_TRUE( ::NS(ClArgument_write)(
            ext_buffer_02_arg, ext_buffer_02 ) );

        ASSERT_TRUE( ::NS(ClArgument_read)(
            ext_buffer_02_arg, ext_buffer_02 ) );

        /* ----------------------------------------------------------------- */
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

        ext_buffer_02_arg = ::NS(TrackJobCl_argument_by_buffer_id)(
            job, ext_buffer_02_id );

        ASSERT_TRUE( ext_buffer_02_arg == nullptr );

        /* ----------------------------------------------------------------- */
        /* Add ext_buffer_02 again, but this time let the job take ownership:*/

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

        arg_t* ext_buffer_02_b_arg = ::NS(TrackJobCl_argument_by_buffer_id)(
            job, ext_buffer_02_id_b );

        ASSERT_TRUE(  ext_buffer_02_b_arg != nullptr );
        ASSERT_TRUE(  ::NS(ClArgument_get_ptr_to_context)(
            ext_buffer_02_b_arg ) == ::NS(TrackJobCl_get_context)( job ) );

        ASSERT_TRUE(  ::NS(ClArgument_uses_cobj_buffer)( ext_buffer_02_b_arg ) );
        ASSERT_TRUE(  ::NS(ClArgument_get_ptr_cobj_buffer)(
            ext_buffer_02_b_arg ) == ext_buffer_02_b );

        ASSERT_TRUE( ::NS(ClArgument_write)(
            ext_buffer_02_b_arg, ext_buffer_02_b ) );

        ASSERT_TRUE( ::NS(ClArgument_read)(
            ext_buffer_02_b_arg, ext_buffer_02_b ) );

        /* ----------------------------------------------------------------- */
        /* Add ext_buffer_03, a C99 buffer on the stack: */

        size_t const ext_buffer_03_capacity = size_t{ 2048 };
        std::vector< unsigned char > ext_buffer_03_data_store(
            ext_buffer_03_capacity );

        c_buffer_t ext_buffer_03;
        ::NS(Buffer_preset)( &ext_buffer_03 );
        status = ::NS(Buffer_init_on_flat_memory)(
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

        arg_t* ext_buffer_03_arg = ::NS(TrackJobCl_argument_by_buffer_id)(
            job, ext_buffer_03_id );

        ASSERT_TRUE(  ext_buffer_03_arg != nullptr );
        ASSERT_TRUE(  ::NS(ClArgument_get_ptr_to_context)(
            ext_buffer_03_arg ) == ::NS(TrackJobCl_get_context)( job ) );

        ASSERT_TRUE(  ::NS(ClArgument_uses_cobj_buffer)( ext_buffer_03_arg ) );
        ASSERT_TRUE(  ::NS(ClArgument_get_ptr_cobj_buffer)(
            ext_buffer_03_arg ) == &ext_buffer_03 );

        ASSERT_TRUE( ::NS(ClArgument_write)(
            ext_buffer_03_arg, &ext_buffer_03 ) );

        ASSERT_TRUE( ::NS(ClArgument_read)(
            ext_buffer_03_arg, &ext_buffer_03 ) );


        /* ----------------------------------------------------------------- */
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

        arg_t* ext_buffer_04_arg = ::NS(TrackJobCl_argument_by_buffer_id)(
            job, ext_buffer_04_id );

        ASSERT_TRUE(  ext_buffer_04_arg != nullptr );
        ASSERT_TRUE(  ::NS(ClArgument_get_ptr_to_context)(
            ext_buffer_04_arg ) == ::NS(TrackJobCl_get_context)( job ) );

        ASSERT_TRUE(  ::NS(ClArgument_uses_cobj_buffer)( ext_buffer_04_arg ) );
        ASSERT_TRUE(  ::NS(ClArgument_get_ptr_cobj_buffer)(
            ext_buffer_04_arg ) == ptr_ext_buffer_04 );

        ASSERT_TRUE( ::NS(ClArgument_write)(
            ext_buffer_04_arg, ptr_ext_buffer_04 ) );

        ASSERT_TRUE( ::NS(ClArgument_read)(
            ext_buffer_04_arg, ptr_ext_buffer_04 ) );

        /* ----------------------------------------------------------------- */

        ::NS(TrackJob_delete)( job );
    }
}
