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
#include "sixtracklib/opencl/context.h"
#include "sixtracklib/opencl/cl.h"

TEST( C99_OpenCL_Buffer,
      InitWithGenericObjDataCopyToDeviceCopyBackCmpSingleThread )
{
    using buffer_t      = ::NS(Buffer);
    using size_t        = ::NS(buffer_size_t);
    using context_t     = SIXTRL_CXX_NAMESPACE::ClContextBase;

    size_t const NUM_OPENCL_NODES = ::NS(OpenCL_get_num_all_nodes)();

    if( NUM_OPENCL_NODES == size_t{ 0 } )
    {
        std::cout << "No OpenCL nodes -> skipping test" << std::endl;
        return;
    }

    size_t const NUM_NODES = context_t::NUM_AVAILABLE_NODES(
        nullptr, "SIXTRACKLIB_DEVICES" );

    if( NUM_NODES == size_t{ 0 } )
    {
        std::cout << "No OpenCL nodes available -> skipping test" << std::endl;
        return;
    }

    std::vector< context_t::node_id_t > node_ids(
        NUM_NODES, context_t::node_id_t{} );

    ASSERT_TRUE( NUM_NODES == context_t::GET_AVAILABLE_NODES( node_ids.data(),
        NUM_NODES, size_t{ 0 }, nullptr, "SIXTRACKLIB_DEVICES" ) );

    buffer_t* orig_buffer = ::NS(Buffer_new_from_file)(
        ::NS(PATH_TO_TEST_GENERIC_OBJ_BUFFER_DATA) );
    ASSERT_TRUE( orig_buffer != nullptr );
    ASSERT_TRUE( ::NS(Buffer_get_num_of_objects)( orig_buffer ) > size_t{ 0 } );

    size_t const orig_buffer_size = ::NS(Buffer_get_size)( orig_buffer );
    ASSERT_TRUE( orig_buffer_size > size_t{ 0 } );

    buffer_t* copy_buffer = ::NS(Buffer_new)( 4096u );
    ASSERT_TRUE( copy_buffer != nullptr );
    ASSERT_TRUE( copy_buffer != orig_buffer );
    ASSERT_TRUE( st_Buffer_get_data_begin_addr( orig_buffer ) !=
                 st_Buffer_get_data_begin_addr( copy_buffer ) );

    int success = ::NS(Buffer_reserve)( copy_buffer,
        ::NS(Buffer_get_max_num_of_objects)( orig_buffer ),
        ::NS(Buffer_get_max_num_of_slots)( orig_buffer ),
        ::NS(Buffer_get_max_num_of_dataptrs)( orig_buffer ),
        ::NS(Buffer_get_max_num_of_garbage_ranges)( orig_buffer ) );

    ASSERT_TRUE( success == 0 );

    unsigned char const* orig_buffer_begin =
        ::NS(Buffer_get_const_data_begin)( orig_buffer );

    ASSERT_TRUE( orig_buffer_begin != nullptr );

    unsigned char* copy_buffer_begin =
        ::NS(Buffer_get_data_begin)( copy_buffer );

    ASSERT_TRUE( copy_buffer_begin != nullptr );
    ASSERT_TRUE( copy_buffer_begin != orig_buffer_begin );

    /* --------------------------------------------------------------------- */

    for( auto const& node_id : node_ids )
    {
        context_t st_context( node_id );

        ASSERT_TRUE( st_context.hasSelectedNode() );
        ASSERT_TRUE( st_context.selectedNodeDevice() != nullptr );
        ASSERT_TRUE( st_context.openClContext() != nullptr );
        ASSERT_TRUE( st_context.openClQueue() != nullptr );
        ASSERT_TRUE( st_context.ptrSelectedNodeInfo() != nullptr );

        std::ostringstream a2str;

        a2str << " -D_GPUCODE=1"
            << " -DSIXTRL_BUFFER_ARGPTR_DEC=__private"
            << " -DSIXTRL_BUFFER_DATAPTR_DEC=__global"
            << " -I" << ::NS(PATH_TO_SIXTRL_INCLUDE_DIR);

        if( std::strcmp( ::NS(PATH_TO_SIXTRL_INCLUDE_DIR),
                        ::NS(PATH_TO_SIXTRL_TESTLIB_INCLUDE_DIR) ) != 0 )
        {
            a2str << " -I"
                << ::NS(PATH_TO_SIXTRL_TESTLIB_INCLUDE_DIR);
        }

        std::string const COMPILE_OPTIONS = a2str.str();

        std::string path_to_kernel_source = ::NS(PATH_TO_BASE_DIR);
        path_to_kernel_source += "tests/sixtracklib/testlib/opencl/kernels/";
        path_to_kernel_source += "opencl_buffer_generic_obj_kernel.cl";

        std::ifstream kernel_file( path_to_kernel_source, std::ios::in );

        std::string const PROGRAM_SOURCE_CODE(
            ( std::istreambuf_iterator< char >( kernel_file ) ),
            std::istreambuf_iterator< char >() );

        kernel_file.close();

        cl::Device& device = *st_context.selectedNodeDevice();
        cl::Context& context = *st_context.openClContext();
        cl::CommandQueue& queue = *st_context.openClQueue();

        auto ptr_node_info = st_context.ptrSelectedNodeInfo();
        auto ptr_default_info = ( st_context.isDefaultNode( node_id ) )
            ? st_context.ptrSelectedNodeId() : nullptr;

        std::cout << "Perform test for device: \r\n";
        ::NS(ComputeNodeInfo_print_out)( ptr_node_info, ptr_default_info );

        /* ---------------------------------------------------------------- */
        /* prepare copy buffer */

        ::NS(Buffer_clear)( copy_buffer, true );

        auto obj_it  = st_Buffer_get_const_objects_begin( orig_buffer );
        auto obj_end = st_Buffer_get_const_objects_end( orig_buffer );

        for( ; obj_it != obj_end ; ++obj_it )
        {
            ::NS(GenericObj) const* orig_obj = reinterpret_cast<
                ::NS(GenericObj) const* >( static_cast< uintptr_t >(
                    ::NS(Object_get_begin_addr)( obj_it ) ) );

            ASSERT_TRUE( orig_obj != nullptr );
            ASSERT_TRUE( orig_obj->type_id ==
                         ::NS(Object_get_type_id)( obj_it ) );

            ::NS(GenericObj)* copy_obj = ::NS(GenericObj_new)( copy_buffer,
                orig_obj->type_id, orig_obj->num_d, orig_obj->num_e );

            ASSERT_TRUE( copy_obj != nullptr );
            ASSERT_TRUE( orig_obj->type_id == copy_obj->type_id );
            ASSERT_TRUE( orig_obj->num_d   == copy_obj->num_d );
            ASSERT_TRUE( orig_obj->num_e   == copy_obj->num_e );
            ASSERT_TRUE( copy_obj->d != nullptr );
            ASSERT_TRUE( copy_obj->e != nullptr );
        }

        ASSERT_TRUE( ::NS(Buffer_get_num_of_objects)( copy_buffer ) ==
                     ::NS(Buffer_get_num_of_objects)( orig_buffer ) );

        /* ---------------------------------------------------------------- */

        cl_int cl_ret = CL_SUCCESS;

        cl::Program program( context, PROGRAM_SOURCE_CODE );

        try
        {
            cl_ret = program.build( COMPILE_OPTIONS.c_str() );
        }
        catch( cl::Error const& e )
        {
            std::cerr << "OpenCL Compilation Error -> Stopping Unit-Test \r\n"
                      << program.getBuildInfo< CL_PROGRAM_BUILD_LOG >( device )
                      << "\r\n"
                      << std::endl;

            cl_ret = CL_FALSE;
            throw;
        }

        ASSERT_TRUE( cl_ret == CL_SUCCESS );

        cl::Buffer cl_orig_buf( context, CL_MEM_READ_WRITE, orig_buffer_size );
        cl::Buffer cl_err_flag( context, CL_MEM_READ_WRITE, sizeof( int64_t ) );
        cl::Buffer cl_copy_buf( context, CL_MEM_READ_WRITE, orig_buffer_size );

        try
        {
            cl_ret = queue.enqueueWriteBuffer(
                cl_orig_buf, CL_TRUE, 0, orig_buffer_size, orig_buffer_begin );
        }
        catch( cl::Error const& e )
        {
            std::cout << "enqueueWriteBuffer( orig_buffer_begin ) :: line = "
                      << __LINE__
                      << " :: ERROR : " << e.what() << std::endl
                      << e.err() << std::endl;

            cl_ret = CL_FALSE;
            throw;
        }

        ASSERT_TRUE( cl_ret == CL_SUCCESS );

        try
        {
            cl_ret = queue.enqueueWriteBuffer(
                cl_copy_buf, CL_TRUE, 0, orig_buffer_size, copy_buffer_begin );
        }
        catch( cl::Error const& e )
        {
            std::cout << "enqueueWriteBuffer( copy_buffer_begin ) :: line = "
                      << __LINE__
                      << " :: ERROR : " << e.what() << std::endl
                      << e.err() << std::endl;

            cl_ret = CL_FALSE;
            throw;
        }

        ASSERT_TRUE( cl_ret == CL_SUCCESS );

        int64_t error_flag = -1;

        try
        {
            cl_ret = queue.enqueueWriteBuffer(
                cl_err_flag, CL_TRUE, 0, sizeof( error_flag ), &error_flag );
        }
        catch( cl::Error const& e )
        {
            std::cout << "enqueueWriteBuffer( error_flag ) :: line = "
                      << __LINE__
                      << " :: ERROR : " << e.what() << std::endl
                      << e.err() << std::endl;

            cl_ret = CL_FALSE;
            throw;
        }

        ASSERT_TRUE( cl_ret == CL_SUCCESS );

        int num_threads = ( int )1;
        int block_size  = ( int )1;

        std::string kernel_name( SIXTRL_C99_NAMESPACE_PREFIX_STR );
        kernel_name += "remap_orig_buffer";

        cl::Kernel remap_kernel;

        try
        {
            remap_kernel = cl::Kernel( program, kernel_name.c_str() );
        }
        catch( cl::Error const& e )
        {
            std::cout << "kernel remap_kernel :: line = " << __LINE__
                      << " :: ERROR : " << e.what() << std::endl
                      << e.err() << std::endl;
            throw;
        }

        auto num_remap_kernel_args = remap_kernel.getInfo< CL_KERNEL_NUM_ARGS >();

        ASSERT_TRUE( num_remap_kernel_args == 3u );

        remap_kernel.setArg( 0, cl_orig_buf );
        remap_kernel.setArg( 1, cl_copy_buf );
        remap_kernel.setArg( 2, cl_err_flag );


        try
        {
            cl_ret = queue.enqueueNDRangeKernel( remap_kernel, cl::NullRange,
                cl::NDRange( num_threads ), cl::NDRange( block_size ) );
        }
        catch( cl::Error const& e )
        {
            std::cout << "enqueueNDRangeKernel( remap_kernel) :: line = "
                      << __LINE__
                      << " :: ERROR : " << e.what() << std::endl
                      << e.err() << std::endl;

            throw;
        }

        try
        {
            cl_ret = queue.enqueueReadBuffer(
                cl_err_flag, CL_TRUE, 0, sizeof( error_flag ), &error_flag );
        }
        catch( cl::Error const& e )
        {
            std::cout << "enqueueReadBuffer( error_flag ) :: line = "
                      << __LINE__ << " :: ERROR : " << e.what() << std::endl
                      << e.err() << std::endl;

            cl_ret = CL_FALSE;
            throw;
        }

        ASSERT_TRUE( cl_ret == CL_SUCCESS );
        ASSERT_TRUE( error_flag == int64_t{ 0 } );

        cl::Kernel copy_kernel;

        try
        {
            copy_kernel = cl::Kernel( program, "st_copy_orig_buffer" );
        }
        catch( cl::Error const& e )
        {
            std::cout << "kernel copy_kernel :: line = " << __LINE__
                      << " :: ERROR : " << e.what() << std::endl
                      << e.err() << std::endl;

            throw;
        }

        copy_kernel.setArg( 0, cl_orig_buf );
        copy_kernel.setArg( 1, cl_copy_buf );
        copy_kernel.setArg( 2, cl_err_flag );

        block_size  = ( int )1;
        num_threads = ( int )1;

        cl_ret = queue.enqueueNDRangeKernel( copy_kernel, cl::NullRange,
            cl::NDRange( num_threads ), cl::NDRange( block_size ) );

        ASSERT_TRUE( cl_ret == CL_SUCCESS );

        try
        {
            cl_ret = queue.enqueueReadBuffer(
                cl_err_flag, CL_TRUE, 0, sizeof( error_flag ), &error_flag );
        }
        catch( cl::Error const& e )
        {
            std::cout << "enqueueReadBuffer( error_flag ) :: line = "
                      << __LINE__
                      << " :: ERROR : " << e.what() << std::endl
                      << e.err() << std::endl;

            cl_ret = CL_FALSE;
            throw;
        }

        ASSERT_TRUE( cl_ret == CL_SUCCESS );

        try
        {
            cl_ret = queue.enqueueReadBuffer(
                cl_copy_buf, CL_TRUE, 0, orig_buffer_size, copy_buffer_begin );
        }
        catch( cl::Error const& e )
        {
            std::cout << "enqueueReadBuffer( copy_buffer_begin ) :: line = "
                      << __LINE__
                      << " :: ERROR : " << e.what() << std::endl
                      << e.err() << std::endl;

            cl_ret = CL_FALSE;
            throw;
        }

        ASSERT_TRUE( cl_ret == CL_SUCCESS );

        /* ----------------------------------------------------------------- */

        success = ::NS(Buffer_remap)( copy_buffer );
        ASSERT_TRUE( success == 0 );

        ASSERT_TRUE( ::NS(Buffer_get_num_of_objects)( copy_buffer ) ==
                     ::NS(Buffer_get_num_of_objects)( orig_buffer ) );

        obj_it       = st_Buffer_get_const_objects_begin( orig_buffer );
        obj_end      = st_Buffer_get_const_objects_end( orig_buffer );
        auto cmp_it  = st_Buffer_get_const_objects_begin( copy_buffer );

        for( ; obj_it != obj_end ; ++obj_it, ++cmp_it )
        {
            ::NS(GenericObj) const* orig_obj = reinterpret_cast<
                ::NS(GenericObj) const* >( static_cast< uintptr_t >(
                    ::NS(Object_get_begin_addr)( obj_it ) ) );

            ::NS(GenericObj) const* cmp_obj = reinterpret_cast<
                ::NS(GenericObj) const* >( static_cast< uintptr_t >(
                    ::NS(Object_get_begin_addr)( cmp_it ) ) );

            ASSERT_TRUE( orig_obj != nullptr );
            ASSERT_TRUE( cmp_obj  != nullptr );
            ASSERT_TRUE( cmp_obj  != orig_obj );

            ASSERT_TRUE( orig_obj->type_id == cmp_obj->type_id );
            ASSERT_TRUE( orig_obj->num_d   == cmp_obj->num_d );
            ASSERT_TRUE( orig_obj->num_e   == cmp_obj->num_e );
            ASSERT_TRUE( orig_obj->a       == cmp_obj->a );

            ASSERT_TRUE( std::fabs( orig_obj->a - cmp_obj->a ) <
                std::numeric_limits< double >::epsilon() );

            for( std::size_t ii = 0 ; ii < 4u ; ++ii )
            {
                ASSERT_TRUE( std::fabs( orig_obj->c[ ii ] - cmp_obj->c[ ii ] ) <
                    std::numeric_limits< double >::epsilon() );
            }

            if( orig_obj->num_d > 0u )
            {
                for( std::size_t ii = 0u ; ii < orig_obj->num_d ; ++ii )
                {
                    ASSERT_TRUE( orig_obj->d[ ii ] == cmp_obj->d[ ii ] );
                }
            }

            if( orig_obj->num_e > 0u )
            {
                for( std::size_t ii = 0u ; ii < orig_obj->num_e ; ++ii )
                {
                    ASSERT_TRUE( std::fabs( orig_obj->e[ ii ] - cmp_obj->e[ ii ] )
                        < std::numeric_limits< double >::epsilon() );
                }
            }
        }
    }

    ::NS(Buffer_delete)( orig_buffer );
    ::NS(Buffer_delete)( copy_buffer );
}

/* end: tests/sixtracklib/opencl/test_buffer_opencl_c99.cpp */
