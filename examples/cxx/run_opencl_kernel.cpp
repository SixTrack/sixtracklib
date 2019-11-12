#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <iomanip>
#include <numeric>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "sixtracklib/testlib.h"
#include "sixtracklib/sixtracklib.hpp"

int main( int argc, char* argv[] )
{
    namespace st = sixtrack;
    using context_t     = st::ClContextBase;
    using buf_size_t    = ::NS(buffer_size_t);
    using program_id_t  = context_t::kernel_id_t;
    using kernel_id_t   = context_t::kernel_id_t;

    int64_t VECTOR_SIZE = int64_t{ 1000 };
    std::unique_ptr< context_t > ptr_ctx = nullptr;

    /* --------------------------------------------------------------------- */
    /* Handle command line arguments: */
    /* --------------------------------------------------------------------- */

    if( argc < 2  )
    {
        ptr_ctx.reset( new context_t );

        std::cout << "Usage: " << argv[ 0 ]
                  << " [ID] [VECTOR_SIZE] \r\n";

        ptr_ctx->printNodesInfo();

        if( ptr_ctx->numAvailableNodes() > buf_size_t{ 0 } )
        {
            std::cout << "INFO            :: "
                      << "Selecting default node\r\n";
        }
        else
        {
            std::cout << "Quitting program!\r\n"
                      << std::endl;

            return 0;
        }

        std::cout << "\r\n"
                  << "[VECTOR_SIZE]   :: Vector size\r\n"
                  << "                :: Default = "
                  << VECTOR_SIZE
                  << "\r\n";
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    /* Select the node based on the first command line param or
     * select the default node: */

    if( argc >= 2 )
    {
        ptr_ctx.reset( new context_t( argv[ 1 ] ) );

        if( !ptr_ctx->hasSelectedNode() )
        {
            std::cout << "Warning         : Provided ID " << argv[ 1 ]
                      << " not found -> use default device instead\r\n";
        }
    }

    if( argc >= 3 )
    {
        int64_t const temp = std::atoi( argv[ 2 ] );
        if( temp > 0 ) VECTOR_SIZE = temp;
    }

    if( (  ptr_ctx.get() != nullptr ) && ( !ptr_ctx->hasSelectedNode() ) )
    {
        context_t::node_id_t const default_node_id = ptr_ctx->defaultNodeId();
        ptr_ctx->selectNode( default_node_id );
    }

    if( (  ptr_ctx.get() == nullptr ) ||
        ( !ptr_ctx->hasSelectedNode() ) )
    {
        return 0;
    }

    context_t& ctx = *ptr_ctx;

    std::cout << "Selected Node     [ID] = "
              << ctx.selectedNodeIdStr()
              << " ( "
              << ::NS(ComputeNodeInfo_get_name)( ctx.ptrSelectedNodeInfo() )
              << " / "
              << ::NS(ComputeNodeInfo_get_platform)( ctx.ptrSelectedNodeInfo() )
              << " ) \r\n"
              << "Selected VECTOR_SIZE = "
              << std::setw( 10 ) << VECTOR_SIZE << "\r\n\r\n"
              << std::endl;

    /* --------------------------------------------------------------------- */

    buf_size_t const data_size = sizeof( double ) * VECTOR_SIZE;

    std::vector< double > vector_a( VECTOR_SIZE, 0.0 );
    std::vector< double > vector_b( VECTOR_SIZE, 0.0 );
    std::vector< double > result(   VECTOR_SIZE, 0.0 );

    double x = double{ 0 };

    std::iota( vector_a.begin(), vector_a.end(), x );

    x = double{ 0 };

    std::generate( vector_b.begin(), vector_b.end(),
                   [&x](){ return -( x++ ); } );

    std::copy( vector_a.begin(), vector_a.end(), result.begin() );

    /* Compile program */

    /* **** NOTE: The kernel is located in the examples/c99/ directory ->
     * ****       ensure that the C99 examples are present as well! */
    std::string path_to_program( ::NS(PATH_TO_BASE_DIR) );
    path_to_program += "examples/c99/run_opencl_kernel.cl";

    std::string compile_options( "-D_GPUCODE=1 -I" );
    compile_options += st_PATH_TO_BASE_DIR;

    program_id_t const program_id = ctx.addProgramFile(
        path_to_program, compile_options );

    if( program_id < program_id_t{ 0 } )
    {
        std::cout << "Error: unable to add program -> stopping"
                  << std::endl;

        return 0;
    }

    /* Enable addition kernel */

    std::string kernel_name( SIXTRL_C99_NAMESPACE_PREFIX_STR );
    kernel_name += "Add_vectors_kernel";

    kernel_id_t const add_kernel_id =
        ctx.enableKernel( kernel_name, program_id );

    if( add_kernel_id < kernel_id_t{ 0 } )
    {
        if( ctx.isProgramCompiled( program_id ) )
        {
            std::cout << "Error: unknown kernel name " << kernel_name
                      << " -> stopping" << std::endl;
        }
        else
        {
            std::cout << "Error: problems during compilation of program: "
                      << "\r\n\r\n"
                      << "Compile report:\r\n"
                      << ctx.programCompileReport( program_id )
                      << "\r\n"
                      << "Compile options: "
                      << ctx.programCompileOptions( program_id )
                      << "\r\n"
                      << "-> stopping"
                      << std::endl;

            return 0;
        }
    }

    /* Init kernel arguments */

    st::ClArgument vec_a_arg( vector_a.data(), data_size, &ctx );
    st::ClArgument vec_b_arg( vector_b.data(), data_size, &ctx );
    st::ClArgument result_arg( result.data(),  data_size, &ctx );

    /* Assign kernel arguments to addition kernel */

    ctx.assignKernelArgument( add_kernel_id, 0u, vec_a_arg );
    ctx.assignKernelArgument( add_kernel_id, 1u, vec_b_arg );
    ctx.assignKernelArgument( add_kernel_id, 2u, result_arg );

    ctx.assignKernelArgumentValue( add_kernel_id, 3u, VECTOR_SIZE );

    /* run addition kernel */

    bool success = ctx.runKernel( add_kernel_id, VECTOR_SIZE );

    /* Read-back the result and print it out */

    success &= result_arg.read( result.data(), data_size );

    if( success )
    {
        std::cout.precision( 6 );
        std::cout << std::fixed;

        for( int64_t ii = int64_t{ 0 } ; ii < VECTOR_SIZE ; ++ii )
        {
            std::cout << std::setw( 10 ) << ii
                      << " :: "
                      << "a = " << std::setw( 12 ) << vector_a[ ii ]
                      << " , "
                      << "b = " << std::setw( 12 ) << vector_b[ ii ]
                      << " , "
                      << "a + b = "
                      << std::fixed << std::setw( 12 ) << result[ ii ]
                      << "\r\n";
        }

        std::cout << std::endl;
    }
    else
    {
        std::cout << "Error running kernel -> stopping" << std::endl;
    }

    /* Run the same kernel again, but this time with an explicitly provided
     * work-group size; In this example, we use the maximum available
     * work-group size for the given kernel */

    if( success )
    {
        buf_size_t const max_wg_size =
            ctx.kernelMaxWorkGroupSize( add_kernel_id );

        success  = ctx.runKernel( add_kernel_id, VECTOR_SIZE, max_wg_size );
        success &= result_arg.read( result.data(), data_size );
    }

    if( success )
    {
        std::cout.precision( 6 );
        std::cout << std::fixed;

        for( int64_t ii = int64_t{ 0 } ; ii < VECTOR_SIZE ; ++ii )
        {
            std::cout << std::setw( 10 ) << ii
                      << " :: "
                      << "a = " << std::setw( 12 ) << vector_a[ ii ]
                      << " , "
                      << "b = " << std::setw( 12 ) << vector_b[ ii ]
                      << " , "
                      << "a + b = "
                      << std::fixed << std::setw( 12 ) << result[ ii ]
                      << "\r\n";
        }

        std::cout << std::endl;
    }
    else
    {
        std::cout << "Error running kernel -> stopping" << std::endl;
    }

    return 0;
}

/* end: examples/cxx/run_opencl_kernel.cpp */

