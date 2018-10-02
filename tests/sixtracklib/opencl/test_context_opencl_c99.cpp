#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <fstream>
#include <iterator>
#include <sstream>
#include <vector>

#include <gtest/gtest.h>

#include "sixtracklib/testlib.h"

#include "sixtracklib/_impl/definitions.h"
#include "sixtracklib/_impl/path.h"
#include "sixtracklib/common/buffer.h"
#include "sixtracklib/opencl/internal/base_context.h"
#include "sixtracklib/opencl/context.h"
#include "sixtracklib/opencl/argument.h"

TEST( C99_OpenCL_Context, BaseOpenCLContext )
{
    using context_t     = ::st_ClContextBase;
    using con_size_t    = ::st_context_size_t;
    using node_id_t     = ::st_context_node_id_t;
    using node_info_t   = ::st_context_node_info_t;
    using kernel_id_t   = int;
    using program_id_t  = int;
    using argument_t    = ::st_ClArgument;

    context_t* context = ::st_ClContextBase_create();

    con_size_t const num_nodes =
        ::st_ClContextBase_get_num_available_nodes( context );

    ASSERT_TRUE( context != nullptr );

    if( num_nodes > 0u )
    {
        node_info_t const* node_info_it  =
            ::st_ClContextBase_get_available_nodes_info_begin( context );

        node_info_t const* node_info_end =
            ::st_ClContextBase_get_available_nodes_info_end( context );

        ASSERT_TRUE( std::distance( node_info_it, node_info_end ) ==
                     static_cast< std::ptrdiff_t >( num_nodes ) );

        node_id_t const default_node_id =
            NS(ClContextBase_get_default_node_id)( context );

        ASSERT_TRUE( ::st_ComputeNodeId_is_valid( &default_node_id ) );

        for( ; node_info_it != node_info_end ; ++node_info_it )
        {
            char default_str[ 16 ];
            char node_id_str[ 16 ];

            std::memset( &default_str[ 0 ], ( int )'\0', 16 );
            std::memset( &node_id_str[ 0 ], ( int )'\0', 16 );

            node_id_t const node_id = st_ComputeNodeInfo_get_id( node_info_it );
            ASSERT_TRUE( ::st_ComputeNodeId_is_valid( &node_id ) );

            ASSERT_TRUE( 0 == ::st_ComputeNodeId_to_string(
                &node_id, &node_id_str[ 0 ], 16 ) );

            if( ::st_ClContextBase_is_node_id_default_node( context, &node_id ) )
            {
                strncpy( &default_str[ 0 ], " [DEFAULT]", 16 );
            }

            ASSERT_TRUE( ::st_ComputeNodeInfo_get_arch( node_info_it ) != nullptr );
            ASSERT_TRUE( ::st_ComputeNodeInfo_get_name( node_info_it ) != nullptr );
            ASSERT_TRUE( ::st_ComputeNodeInfo_get_platform( node_info_it ) != nullptr );
            ASSERT_TRUE( ::st_ComputeNodeInfo_get_description( node_info_it ) != nullptr );

            std::cout << "INFO  ::    Device Id = "
                      << node_id_str << default_str
                      << "\r\n         Architecture = "
                      << ::st_ComputeNodeInfo_get_arch( node_info_it )
                      << "\r\n             Platform = "
                      << ::st_ComputeNodeInfo_get_platform( node_info_it )
                      << "\r\n          Device Name = "
                      << ::st_ComputeNodeInfo_get_name( node_info_it )
                      << "\r\n          Description = "
                      << ::st_ComputeNodeInfo_get_description( node_info_it )
                      << "\r\n"
                      << std::endl;

            /* ------------------------------------------------------------- */
            /* Verify that the context has a remapping program but not a
             * remapping kernel before device selection */

            ASSERT_TRUE( !::st_ClContextBase_has_selected_node( context ) );
            ASSERT_TRUE(  ::st_ClContextBase_has_remapping_program( context ) );

            con_size_t const initial_num_programs =
                ::st_ClContextBase_get_num_available_programs( context );

            program_id_t remap_program_id =
                ::st_ClContextBase_get_remapping_program_id( context );

            ASSERT_TRUE( initial_num_programs >= con_size_t{ 0 } );

            ASSERT_TRUE(  ::st_ClContextBase_get_num_available_kernels(
                context ) == con_size_t{ 0 } );

            ASSERT_TRUE(  ::st_ClContextBase_get_remapping_program_id(
                context ) != -1 );

            ASSERT_TRUE( !::st_ClContextBase_has_remapping_kernel( context ) );
            ASSERT_TRUE(  ::st_ClContextBase_get_remapping_kernel_id(
                context ) == -1 );

            /* ------------------------------------------------------------- */
            /* Select current node by node_id */

            ASSERT_TRUE( ::st_ClContextBase_select_node_by_node_id(
                context, &node_id ) );

            ASSERT_TRUE( ::st_ClContextBase_has_selected_node( context ) );
            ASSERT_TRUE( ::st_ComputeNodeId_are_equal( &node_id,
                ::st_ClContextBase_get_selected_node_id( context ) ) );

            ASSERT_TRUE( ::st_ClContextBase_get_selected_node_info( context )
                         == node_info_it );

            ASSERT_TRUE( ::st_ClContextBase_get_num_available_programs(
                context ) == initial_num_programs );

            con_size_t const initial_num_kernels =
                ::st_ClContextBase_get_num_available_kernels( context );

            ASSERT_TRUE( initial_num_kernels >= initial_num_programs );

            ASSERT_TRUE( ::st_ClContextBase_has_remapping_program( context ) );
            ASSERT_TRUE( ::st_ClContextBase_get_remapping_program_id(
                         context ) == remap_program_id );

            ASSERT_TRUE( ::st_ClContextBase_has_remapping_kernel( context ) );

            kernel_id_t const remap_kernel_id =
                ::st_ClContextBase_get_remapping_kernel_id( context );

            ASSERT_TRUE( remap_kernel_id != kernel_id_t{ -1 } );
            ASSERT_TRUE( remap_program_id ==
                ::st_ClContextBase_get_program_id_by_kernel_id(
                    context, remap_kernel_id ) );

            /* ------------------------------------------------------------- */
            /* Create ClArgument from st::Buffer */

            ::st_Buffer* orig_buffer = ::st_Buffer_new_from_file(
                ::st_PATH_TO_TEST_GENERIC_OBJ_BUFFER_DATA );

            ::st_Buffer* copy_buffer = ::st_Buffer_new_detailed(
                ::st_Buffer_get_num_of_objects( orig_buffer ),
                ::st_Buffer_get_num_of_slots( orig_buffer ),
                ::st_Buffer_get_num_of_dataptrs( orig_buffer ),
                ::st_Buffer_get_num_of_garbage_ranges( orig_buffer ),
                ::st_Buffer_get_flags( orig_buffer ) );

            argument_t* orig_arg = ::st_ClArgument_new_from_buffer(
                orig_buffer, context );

            SIXTRL_ASSERT( ::st_ClArgument_get_ptr_to_context( orig_arg ) ==
                           context );

            SIXTRL_ASSERT( orig_arg != nullptr );
            SIXTRL_ASSERT( ::st_ClArgument_get_argument_size( orig_arg ) ==
                           ::st_Buffer_get_size( orig_buffer ) );

            SIXTRL_ASSERT( ::st_ClArgument_uses_cobj_buffer( orig_arg ) );
            SIXTRL_ASSERT( ::st_ClArgument_get_ptr_cobj_buffer( orig_arg ) ==
                           orig_buffer );

            argument_t* copy_arg =
                ::st_ClArgument_new_from_buffer( copy_buffer, context );

            SIXTRL_ASSERT( ::st_ClArgument_get_ptr_to_context( copy_arg ) ==
                           context );

            SIXTRL_ASSERT( orig_arg != nullptr );
            SIXTRL_ASSERT( ::st_ClArgument_get_argument_size( copy_arg ) ==
                           ::st_Buffer_get_size( copy_buffer ) );

            SIXTRL_ASSERT( ::st_ClArgument_uses_cobj_buffer( copy_arg ) );
            SIXTRL_ASSERT( ::st_ClArgument_get_ptr_cobj_buffer( copy_arg ) ==
                           copy_buffer );

            /* ------------------------------------------------------------- */
            /* Add copy generic obj buffer program */

            char path_to_copy_kernel_program[ 1024 ];
            std::memset( path_to_copy_kernel_program, ( int )'\0', 1024 );

            std::strncpy( path_to_copy_kernel_program, ::st_PATH_TO_BASE_DIR,
                          1023 );

            std::strncat( path_to_copy_kernel_program,
                          "tests/sixtracklib/opencl/",
                          1023 - std::strlen( path_to_copy_kernel_program ) );

            std::strncat( path_to_copy_kernel_program,
                          "opencl_buffer_generic_obj_kernel.cl",
                          1023 - std::strlen( path_to_copy_kernel_program ) );


            char copy_program_compile_options[ 1024 ];
            std::memset( copy_program_compile_options, ( int )'\0', 1024 );

            std::strncpy( copy_program_compile_options, "-D_GPUCODE=1", 1024 );

            std::strncat( copy_program_compile_options, " -D__NAMESPACE=st_",
                          1023 - std::strlen( copy_program_compile_options ) );

            std::strncat( copy_program_compile_options,
                          " -DSIXTRL_BUFFER_ARGPTR_DEC=__private",
                          1023 - std::strlen( copy_program_compile_options ) );

            std::strncat( copy_program_compile_options,
                          " -DSIXTRL_BUFFER_DATAPTR_DEC=__global",
                          1023 - std::strlen( copy_program_compile_options ) );

            std::strncat( copy_program_compile_options, " -I",
                          1023 - std::strlen( copy_program_compile_options ) );

            std::strncat( copy_program_compile_options, ::st_PATH_TO_BASE_DIR,
                          1023 - std::strlen( copy_program_compile_options ) );

            std::strncat( copy_program_compile_options, " -I",
                          1023 - std::strlen( copy_program_compile_options ) );

            std::strncat( copy_program_compile_options, ::st_PATH_TO_BASE_DIR,
                          1023 - std::strlen( copy_program_compile_options ) );

            std::strncat( copy_program_compile_options, "tests",
                          1023 - std::strlen( copy_program_compile_options ) );


            program_id_t copy_program_id = ::st_ClContextBase_add_program_file(
                context, path_to_copy_kernel_program,
                    copy_program_compile_options );

            ASSERT_TRUE( copy_program_id != program_id_t{ -1 } );

            ASSERT_TRUE( ::st_ClContextBase_get_num_available_programs(
                context ) == ( initial_num_programs + con_size_t{ 1 } ) );

            ASSERT_TRUE( copy_program_id != remap_program_id );

            ASSERT_TRUE( ::st_ClContextBase_has_program_file_path(
                context, copy_program_id ) );

            ASSERT_TRUE( 0 == std::strcmp( path_to_copy_kernel_program,
                ::st_ClContextBase_get_program_path_to_file(
                    context, copy_program_id ) ) );

            if( !::st_ClContextBase_is_program_compiled(
                    context, copy_program_id ) )
            {
                std::cout << "ERROR :: unable to compile copy program -> "
                          << "error report \r\n"
                          << ::st_ClContextBase_get_program_compile_report(
                              context, copy_program_id ) << std::endl;
            }

            ASSERT_TRUE( ::st_ClContextBase_is_program_compiled(
                    context, copy_program_id ) );

            ASSERT_TRUE( ::st_ClContextBase_get_num_available_kernels(
                    context ) == initial_num_kernels );

            kernel_id_t const copy_kernel_id = ::st_ClContextBase_enable_kernel(
                context, "st_copy_orig_buffer", copy_program_id );

            ASSERT_TRUE( copy_kernel_id != kernel_id_t{ -1 } );
            ASSERT_TRUE( copy_kernel_id != remap_kernel_id );
            ASSERT_TRUE( ::st_ClContextBase_get_num_available_kernels(
                    context ) == ( initial_num_kernels + con_size_t{ 1 } ) );

            ASSERT_TRUE( ::st_ClContextBase_get_program_id_by_kernel_id(
                context, copy_kernel_id ) == copy_program_id );

            ASSERT_TRUE( ::st_ClContextBase_find_kernel_id_by_name( context,
                "st_copy_orig_buffer" ) == copy_kernel_id );

            ASSERT_TRUE( ::st_ClContextBase_get_kernel_num_args(
                context, copy_kernel_id ) == con_size_t{ 3u } );

            /* ------------------------------------------------------------- */
            /* clear context to prepare it for the next node, if available   */

            ::st_ClContextBase_clear( context );

            ASSERT_TRUE( !::st_ClContextBase_has_selected_node( context ) );

            ASSERT_TRUE( !::st_ClContextBase_has_selected_node( context ) );
            ASSERT_TRUE(  ::st_ClContextBase_get_num_available_programs(
                context ) == initial_num_programs );

            ASSERT_TRUE( ::st_ClContextBase_get_num_available_kernels(
                context ) == con_size_t{ 0 } );

            ASSERT_TRUE( ::st_ClContextBase_get_remapping_program_id(
                context ) == remap_program_id );

            ASSERT_TRUE( ::st_ClContextBase_get_remapping_kernel_id(
                context ) == kernel_id_t{ -1 } );

            /* ------------------------------------------------------------- */
            /* Clean-up and ressource deallocation */

            ::st_ClArgument_delete( orig_arg );
            ::st_ClArgument_delete( copy_arg );

            orig_arg = nullptr;
            copy_arg = nullptr;

            ::st_Buffer_delete( orig_buffer );
            ::st_Buffer_delete( copy_buffer );

            orig_buffer = nullptr;
            copy_buffer = nullptr;
        }
    }
    else
    {
        std::cout << "INFO  :: No suitable OpenCL platforms found -> "
                  << "skipping unit-test"
                  << std::endl;
    }

    ::st_ClContextBase_delete( context );
}

/* end: tests/sixtracklib/opencl/test_opencl_context_c99.cpp */
