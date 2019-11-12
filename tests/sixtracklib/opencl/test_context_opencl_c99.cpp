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

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/generated/path.h"
#include "sixtracklib/common/buffer.h"
#include "sixtracklib/opencl/internal/base_context.h"
#include "sixtracklib/opencl/context.h"
#include "sixtracklib/opencl/argument.h"

TEST( C99_OpenCL_Context, BaseOpenCLContext )
{
    namespace st = SIXTRL_CXX_NAMESPACE;

    using context_t     = ::NS(ClContextBase);
    using con_size_t    = ::NS(arch_size_t);
    using node_id_t     = ::NS(context_node_id_t);
    using node_info_t   = ::NS(context_node_info_t);
    using kernel_id_t   = ::NS(arch_kernel_id_t);
    using program_id_t  = ::NS(arch_program_id_t);
    using argument_t    = ::NS(ClArgument);

    context_t* context = ::NS(ClContextBase_create)();

    con_size_t const num_nodes =
        ::NS(ClContextBase_get_num_available_nodes)( context );

    ASSERT_TRUE( context != nullptr );

    if( num_nodes > 0u )
    {
        node_info_t const* node_info_it  =
            ::NS(ClContextBase_get_available_nodes_info_begin)( context );

        node_info_t const* node_info_end =
            ::NS(ClContextBase_get_available_nodes_info_end)( context );

        ASSERT_TRUE( std::distance( node_info_it, node_info_end ) ==
                     static_cast< std::ptrdiff_t >( num_nodes ) );

        node_id_t const default_node_id =
            NS(ClContextBase_get_default_node_id)( context );

        ASSERT_TRUE( ::NS(ComputeNodeId_is_valid)( &default_node_id ) );

        for( ; node_info_it != node_info_end ; ++node_info_it )
        {
            char default_str[ 16 ];
            char node_id_str[ 16 ];

            std::memset( &default_str[ 0 ], ( int )'\0', 16 );
            std::memset( &node_id_str[ 0 ], ( int )'\0', 16 );

            node_id_t const node_id = NS(ComputeNodeInfo_get_id)( node_info_it );
            ASSERT_TRUE( ::NS(ComputeNodeId_is_valid)( &node_id ) );

            ASSERT_TRUE( 0 == ::NS(ComputeNodeId_to_string)(
                &node_id, &node_id_str[ 0 ], 16 ) );

            if( ::NS(ClContextBase_is_node_id_default_node)( context, &node_id ) )
            {
                strncpy( &default_str[ 0 ], " [DEFAULT]", 16 );
            }

            ASSERT_TRUE( ::NS(ComputeNodeInfo_get_arch)( node_info_it ) != nullptr );
            ASSERT_TRUE( ::NS(ComputeNodeInfo_get_name)( node_info_it ) != nullptr );
            ASSERT_TRUE( ::NS(ComputeNodeInfo_get_platform)( node_info_it ) != nullptr );
            ASSERT_TRUE( ::NS(ComputeNodeInfo_get_description)( node_info_it ) != nullptr );

            std::cout << "INFO  ::    Device Id = "
                      << node_id_str << default_str
                      << "\r\n         Architecture = "
                      << ::NS(ComputeNodeInfo_get_arch)( node_info_it )
                      << "\r\n             Platform = "
                      << ::NS(ComputeNodeInfo_get_platform)( node_info_it )
                      << "\r\n          Device Name = "
                      << ::NS(ComputeNodeInfo_get_name)( node_info_it )
                      << "\r\n          Description = "
                      << ::NS(ComputeNodeInfo_get_description)( node_info_it )
                      << "\r\n"
                      << std::endl;

            /* ------------------------------------------------------------- */
            /* Verify that the context has a remapping program but not a
             * remapping kernel before device selection */

            ASSERT_TRUE( !::NS(ClContextBase_has_selected_node)( context ) );
            ASSERT_TRUE(  ::NS(ClContextBase_has_remapping_program)( context ) );

            con_size_t const initial_num_programs =
                ::NS(ClContextBase_get_num_available_programs)( context );

            program_id_t remap_program_id =
                ::NS(ClContextBase_remapping_program_id)( context );

            ASSERT_TRUE( remap_program_id != st::ARCH_ILLEGAL_PROGRAM_ID );
            ASSERT_TRUE( initial_num_programs >= con_size_t{ 0 } );

            ASSERT_TRUE(  ::NS(ClContextBase_get_num_available_kernels)(
                context ) == con_size_t{ 0 } );

            ASSERT_TRUE( ::NS(ClContextBase_remapping_program_id)( context ) !=
                         st::ARCH_ILLEGAL_PROGRAM_ID );

            ASSERT_TRUE( !::NS(ClContextBase_has_remapping_kernel)( context ) );
            ASSERT_TRUE( ::NS(ClContextBase_remapping_kernel_id)( context ) ==
                         st::ARCH_ILLEGAL_KERNEL_ID );

            /* ------------------------------------------------------------- */
            /* Select current node by node_id */

            ASSERT_TRUE( ::NS(ClContextBase_select_node_by_node_id)(
                context, &node_id ) );

            ASSERT_TRUE( ::NS(ClContextBase_has_selected_node)( context ) );
            ASSERT_TRUE( ::NS(ComputeNodeId_are_equal)( &node_id,
                ::NS(ClContextBase_get_selected_node_id)( context ) ) );

            ASSERT_TRUE( ::NS(ClContextBase_get_selected_node_info)( context )
                         == node_info_it );

            ASSERT_TRUE( ::NS(ClContextBase_get_num_available_programs)(
                context ) == initial_num_programs );

            con_size_t const initial_num_kernels =
                ::NS(ClContextBase_get_num_available_kernels)( context );

            ASSERT_TRUE( initial_num_kernels >= initial_num_programs );

            ASSERT_TRUE( ::NS(ClContextBase_has_remapping_program)( context ) );
            ASSERT_TRUE( ::NS(ClContextBase_remapping_program_id)( context ) ==
                         remap_program_id );

            ASSERT_TRUE( ::NS(ClContextBase_has_remapping_kernel)( context ) );

            kernel_id_t const remap_kernel_id =
                ::NS(ClContextBase_remapping_kernel_id)( context );

            ASSERT_TRUE( remap_kernel_id != st::ARCH_ILLEGAL_KERNEL_ID );
            ASSERT_TRUE( ::NS(ClContextBase_get_program_id_by_kernel_id)(
                    context, remap_kernel_id ) == remap_program_id );

            /* ------------------------------------------------------------- */
            /* Create ClArgument from st::Buffer */

            ::NS(Buffer)* orig_buffer = ::NS(Buffer_new_from_file)(
                ::NS(PATH_TO_TEST_GENERIC_OBJ_BUFFER_DATA) );

            ::NS(Buffer)* copy_buffer = ::NS(Buffer_new_detailed)(
                ::NS(Buffer_get_num_of_objects)( orig_buffer ),
                ::NS(Buffer_get_num_of_slots)( orig_buffer ),
                ::NS(Buffer_get_num_of_dataptrs)( orig_buffer ),
                ::NS(Buffer_get_num_of_garbage_ranges)( orig_buffer ),
                ::NS(Buffer_get_flags)( orig_buffer ) );

            argument_t* orig_arg = ::NS(ClArgument_new_from_buffer)(
                orig_buffer, context );

            SIXTRL_ASSERT( ::NS(ClArgument_get_ptr_to_context)( orig_arg ) ==
                           context );

            SIXTRL_ASSERT( orig_arg != nullptr );
            SIXTRL_ASSERT( ::NS(ClArgument_get_argument_size)( orig_arg ) ==
                           ::NS(Buffer_get_size)( orig_buffer ) );

            SIXTRL_ASSERT( ::NS(ClArgument_uses_cobj_buffer)( orig_arg ) );
            SIXTRL_ASSERT( ::NS(ClArgument_get_ptr_cobj_buffer)( orig_arg ) ==
                           orig_buffer );

            argument_t* copy_arg =
                ::NS(ClArgument_new_from_buffer)( copy_buffer, context );

            SIXTRL_ASSERT( ::NS(ClArgument_get_ptr_to_context)( copy_arg ) ==
                           context );

            SIXTRL_ASSERT( orig_arg != nullptr );
            SIXTRL_ASSERT( ::NS(ClArgument_get_argument_size)( copy_arg ) ==
                           ::NS(Buffer_get_size)( copy_buffer ) );

            SIXTRL_ASSERT( ::NS(ClArgument_uses_cobj_buffer)( copy_arg ) );
            SIXTRL_ASSERT( ::NS(ClArgument_get_ptr_cobj_buffer)( copy_arg ) ==
                           copy_buffer );

            /* ------------------------------------------------------------- */
            /* Add copy generic obj buffer program */

            con_size_t const N = 1024;
            std::vector< char > path_to_copy_kernel_program( N + 1, '\0' );

            std::strncpy( path_to_copy_kernel_program.data(),
                          ::NS(PATH_TO_BASE_DIR), N );

            std::strncat( path_to_copy_kernel_program.data(),
                          "tests/sixtracklib/testlib/opencl/kernels/",
                          N - std::strlen( path_to_copy_kernel_program.data() ) );

            std::strncat( path_to_copy_kernel_program.data(),
                          "opencl_buffer_generic_obj_kernel.cl",
                          N - std::strlen( path_to_copy_kernel_program.data() ) );


            std::vector< char > copy_program_compile_options( N + 1, '\0' );

            std::strncat( copy_program_compile_options.data(),
                          "-D_GPUCODE=1"
                          " -DSIXTRL_BUFFER_ARGPTR_DEC=__private"
                          " -DSIXTRL_BUFFER_DATAPTR_DEC=__global"
                          " -I", N );

            std::strncat( copy_program_compile_options.data(),
                          ::NS(PATH_TO_SIXTRL_INCLUDE_DIR), N - std::strlen(
                              copy_program_compile_options.data() ) );

            if( std::strcmp( ::st_PATH_TO_SIXTRL_INCLUDE_DIR,
                             ::NS(PATH_TO_SIXTRL_TESTLIB_INCLUDE_DIR) ) != 0 )
            {
                std::strncat( copy_program_compile_options.data(), " -I",
                    N - std::strlen( copy_program_compile_options.data() ) );

                std::strncat( copy_program_compile_options.data(),
                    ::NS(PATH_TO_SIXTRL_TESTLIB_INCLUDE_DIR),
                    N - std::strlen( copy_program_compile_options.data() ) );
            }

            program_id_t copy_program_id = ::NS(ClContextBase_add_program_file)(
                context, path_to_copy_kernel_program.data(),
                    copy_program_compile_options.data() );

            ASSERT_TRUE( copy_program_id != st::ARCH_ILLEGAL_PROGRAM_ID );

            ASSERT_TRUE( ::NS(ClContextBase_get_num_available_programs)(
                context ) == ( initial_num_programs + con_size_t{ 1 } ) );

            ASSERT_TRUE( copy_program_id != remap_program_id );

            ASSERT_TRUE( ::NS(ClContextBase_has_program_file_path)(
                context, copy_program_id ) );

            ASSERT_TRUE( 0 == std::strcmp( path_to_copy_kernel_program.data(),
                ::NS(ClContextBase_get_program_path_to_file)(
                    context, copy_program_id ) ) );

            if( !::NS(ClContextBase_is_program_compiled)(
                    context, copy_program_id ) )
            {
                std::cout << "ERROR :: unable to compile copy program -> "
                          << "error report \r\n"
                          << ::NS(ClContextBase_get_program_compile_report)(
                              context, copy_program_id ) << std::endl;
            }

            ASSERT_TRUE( ::NS(ClContextBase_is_program_compiled)(
                    context, copy_program_id ) );

            ASSERT_TRUE( ::NS(ClContextBase_get_num_available_kernels)(
                    context ) == initial_num_kernels );

            std::string kernel_name( SIXTRL_C99_NAMESPACE_PREFIX_STR );
            kernel_name += "copy_orig_buffer";

            kernel_id_t const copy_kernel_id = ::NS(ClContextBase_enable_kernel)(
                context, kernel_name.c_str(), copy_program_id );

            ASSERT_TRUE( copy_kernel_id != st::ARCH_ILLEGAL_KERNEL_ID );
            ASSERT_TRUE( copy_kernel_id != remap_kernel_id );
            ASSERT_TRUE( ::NS(ClContextBase_get_num_available_kernels)(
                    context ) == ( initial_num_kernels + con_size_t{ 1 } ) );

            ASSERT_TRUE( ::NS(ClContextBase_get_program_id_by_kernel_id)(
                context, copy_kernel_id ) == copy_program_id );

            ASSERT_TRUE( ::NS(ClContextBase_find_kernel_id_by_name)( context,
                kernel_name.c_str() ) == copy_kernel_id );

            ASSERT_TRUE( ::NS(ClContextBase_get_kernel_num_args)(
                context, copy_kernel_id ) == con_size_t{ 3u } );

            /* ------------------------------------------------------------- */
            /* clear context to prepare it for the next node, if available   */

            ::NS(ClContextBase_clear)( context );

            ASSERT_TRUE( !::NS(ClContextBase_has_selected_node)( context ) );

            ASSERT_TRUE( !::NS(ClContextBase_has_selected_node)( context ) );
            ASSERT_TRUE(  ::NS(ClContextBase_get_num_available_programs)(
                context ) == initial_num_programs );

            ASSERT_TRUE( ::NS(ClContextBase_get_num_available_kernels)(
                context ) == con_size_t{ 0 } );

            ASSERT_TRUE( ::NS(ClContextBase_remapping_program_id)(
                context ) == remap_program_id );

            ASSERT_TRUE( ::NS(ClContextBase_remapping_kernel_id)(
                context ) == st::ARCH_ILLEGAL_KERNEL_ID );

            /* ------------------------------------------------------------- */
            /* Clean-up and ressource deallocation */

            ::NS(ClArgument_delete)( orig_arg );
            ::NS(ClArgument_delete)( copy_arg );

            orig_arg = nullptr;
            copy_arg = nullptr;

            ::NS(Buffer_delete)( orig_buffer );
            ::NS(Buffer_delete)( copy_buffer );

            orig_buffer = nullptr;
            copy_buffer = nullptr;
        }
    }
    else
    {
        std::cout << "INFO  :: No suitable OpenCL platforms found -> "
                  << "skipping unit-test" << std::endl;
    }

    ::NS(ClContextBase_delete)( context );
}

/* end: tests/sixtracklib/opencl/test_opencl_context_c99.cpp */
