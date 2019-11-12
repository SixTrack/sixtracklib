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

TEST( CXX_OpenCL_Context, BaseOpenCLContextClArgument )
{
    namespace st = sixtrack;

    using context_t     = st::ClContextBase;
    using con_size_t    = context_t::size_type;
    using node_id_t     = context_t::node_id_t;
    using kernel_id_t   = context_t::kernel_id_t;
    using program_id_t  = context_t::program_id_t;
    using cl_argument_t = st::ClArgument;

    context_t context;
    con_size_t const num_nodes = context.numAvailableNodes();

    if( num_nodes > con_size_t{ 0 } )
    {
        auto node_info_it  = context.availableNodesInfoBegin();
        auto node_info_end = context.availableNodesInfoEnd();

        ASSERT_TRUE( std::distance( node_info_it, node_info_end ) ==
                     static_cast< std::ptrdiff_t >( num_nodes ) );

        node_id_t const default_node_id = context.defaultNodeId();

        ASSERT_TRUE( ::st_ComputeNodeId_is_valid( &default_node_id ) );

        for( ; node_info_it != node_info_end ; ++node_info_it )
        {
            char node_id_str[ 16 ];
            std::memset( &node_id_str[ 0 ], ( int )'\0', 16 );

            std::string default_str( "" );

            node_id_t const node_id = ::st_ComputeNodeInfo_get_id( node_info_it );
            ASSERT_TRUE( ::st_ComputeNodeId_is_valid( &node_id ) );

            ASSERT_TRUE( 0 == ::st_ComputeNodeId_to_string(
                &node_id, &node_id_str[ 0 ], 16 ) );

            if( context.isDefaultNode( node_id ) )
            {
                default_str = " [DEFAULT]";
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

            ASSERT_TRUE( !context.hasSelectedNode() );

            con_size_t const initial_num_programs =
                context.numAvailablePrograms();

            ASSERT_TRUE(  initial_num_programs > con_size_t{ 0 } );
            ASSERT_TRUE(  context.numAvailableKernels() == con_size_t{ 0 } );

            ASSERT_TRUE(  context.has_remapping_program() );

            program_id_t const remap_program_id =
                context.remapping_program_id();

            ASSERT_TRUE( remap_program_id >= program_id_t{ 0 } );

            ASSERT_TRUE( !context.has_remapping_kernel() );
            ASSERT_TRUE(  context.remapping_kernel_id() ==
                          st::ARCH_ILLEGAL_KERNEL_ID );

            /* ------------------------------------------------------------- */
            /* Select current node by node_id */

            ASSERT_TRUE( context.selectNode( node_id ) );

            ASSERT_TRUE( node_info_it == context.ptrSelectedNodeInfo() );
            ASSERT_TRUE( ::st_ComputeNodeId_are_equal(
                context.ptrSelectedNodeId(), &node_id ) );

            ASSERT_TRUE( context.hasSelectedNode() );
            ASSERT_TRUE( context.numAvailablePrograms() == initial_num_programs );
            ASSERT_TRUE( context.numAvailableKernels()  >= initial_num_programs );

            con_size_t const initial_num_kernels =
                context.numAvailableKernels();

            ASSERT_TRUE( context.has_remapping_program() );
            ASSERT_TRUE( context.remapping_program_id() == remap_program_id );
            ASSERT_TRUE( context.has_remapping_kernel() );

            kernel_id_t const remap_kernel_id = context.remapping_kernel_id();
            ASSERT_TRUE(  remap_kernel_id >= kernel_id_t{ 0 } );

            ASSERT_TRUE( nullptr != context.openClKernel( remap_kernel_id ) );
            ASSERT_TRUE( nullptr != context.openClProgram( remap_program_id ) );
            ASSERT_TRUE( nullptr != context.openClContext() );
            ASSERT_TRUE( nullptr != context.openClQueue() );

            /* ------------------------------------------------------------- */
            /* Create ClArgument from st::Buffer */

            st::Buffer orig_buffer( ::st_PATH_TO_TEST_GENERIC_OBJ_BUFFER_DATA );

            st::Buffer copy_buffer( orig_buffer.getNumObjects(),
                                    orig_buffer.getNumSlots(),
                                    orig_buffer.getNumDataptrs(),
                                    orig_buffer.getNumGarbageRanges() );

            cl_argument_t orig_arg( orig_buffer, &context );
            cl_argument_t copy_arg( copy_buffer, &context );

            /* ------------------------------------------------------------- */
            /* Add copy generic obj buffer program */

            std::string path_to_copy_kernel_program( ::st_PATH_TO_BASE_DIR );
            path_to_copy_kernel_program += "tests/sixtracklib/testlib/opencl/kernels/";
            path_to_copy_kernel_program += "opencl_buffer_generic_obj_kernel.cl";

            std::string copy_program_compile_options = "-D_GPUCODE=1";
            copy_program_compile_options += " -DSIXTRL_BUFFER_ARGPTR_DEC=__private";
            copy_program_compile_options += " -DSIXTRL_BUFFER_DATAPTR_DEC=__global";
            copy_program_compile_options += " -I";
            copy_program_compile_options += NS(PATH_TO_SIXTRL_INCLUDE_DIR);

            if( std::strcmp( NS(PATH_TO_SIXTRL_INCLUDE_DIR),
                             NS(PATH_TO_SIXTRL_TESTLIB_INCLUDE_DIR) ) != 0 )
            {
                copy_program_compile_options += " -I";
                copy_program_compile_options +=
                    NS(PATH_TO_SIXTRL_TESTLIB_INCLUDE_DIR);
            }

            program_id_t copy_program_id =
                context.addProgramFile( path_to_copy_kernel_program,
                                        copy_program_compile_options );

            ASSERT_TRUE( copy_program_id != st::ARCH_ILLEGAL_PROGRAM_ID );
            ASSERT_TRUE( context.numAvailablePrograms() ==
                ( initial_num_programs + con_size_t{ 1 } ) );

            ASSERT_TRUE( copy_program_id != remap_program_id );
            ASSERT_TRUE( context.programHasFilePath( copy_program_id ) );
            ASSERT_TRUE( path_to_copy_kernel_program.compare(
                context.programPathToFile( copy_program_id ) ) == 0 );

            if( !context.isProgramCompiled( copy_program_id ) )
            {
                std::cout << "ERROR :: unable to compile copy program -> "
                          << "error report \r\n"
                          << context.programCompileReport( copy_program_id )
                          << std::endl;
            }

            ASSERT_TRUE( context.isProgramCompiled( copy_program_id ) );
            ASSERT_TRUE( context.numAvailableKernels() == initial_num_kernels );

            std::string kernel_name( SIXTRL_C99_NAMESPACE_PREFIX_STR );
            kernel_name += "copy_orig_buffer";

            kernel_id_t const copy_kernel_id = context.enableKernel(
                kernel_name.c_str(), copy_program_id );

            ASSERT_TRUE( copy_kernel_id != st::ARCH_ILLEGAL_KERNEL_ID );
            ASSERT_TRUE( copy_kernel_id != context.remapping_kernel_id() );
            ASSERT_TRUE( context.numAvailableKernels() ==
                ( initial_num_kernels + con_size_t{ 1 } ) );

            ASSERT_TRUE( context.programIdByKernelId( copy_kernel_id ) ==
                         copy_program_id );

            ASSERT_TRUE( context.findKernelByName( kernel_name.c_str() ) ==
                         copy_kernel_id );

            ASSERT_TRUE( context.kernelNumArgs( copy_kernel_id ) ==
                         con_size_t{ 3u } );

            /* ------------------------------------------------------------- */
            /* clear context to prepare it for the next node, if available   */

            context.clear();
            ASSERT_TRUE( !context.hasSelectedNode() );
            ASSERT_TRUE(  context.numAvailablePrograms() ==
                          initial_num_programs );

            ASSERT_TRUE(  context.numAvailableKernels()  == con_size_t{ 0 } );
            ASSERT_TRUE(  context.remapping_program_id() == remap_program_id );
            ASSERT_TRUE(  context.remapping_kernel_id() ==
                          st::ARCH_ILLEGAL_KERNEL_ID );

            ASSERT_TRUE( nullptr == context.openClKernel( remap_kernel_id ) );
            ASSERT_TRUE( nullptr == context.openClProgram( remap_program_id ) );
        }
    }
    else
    {
        std::cout << "INFO  :: No suitable OpenCL platforms found -> "
                  << "skipping unit-test"
                  << std::endl;
    }
}

/* end: tests/sixtracklib/opencl/test_opencl_context_c99.cpp */
