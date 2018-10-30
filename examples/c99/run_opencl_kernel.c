#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>

#include "sixtracklib/testlib.h"
#include "sixtracklib/sixtracklib.h"

int main( int argc, char* argv[] )
{
    typedef st_buffer_size_t buf_size_t;

    int64_t VECTOR_SIZE         = 1000;
    int64_t ii                  = 0;
    st_ClContextBase* context   = SIXTRL_NULLPTR;

    char* path_to_program        = SIXTRL_NULLPTR;
    char* compile_options       = SIXTRL_NULLPTR;
    char* kernel_name           = SIXTRL_NULLPTR;

    double* vector_a            = SIXTRL_NULLPTR;
    double* vector_b            = SIXTRL_NULLPTR;
    double* result              = SIXTRL_NULLPTR;

    st_ClArgument* vec_a_arg    = SIXTRL_NULLPTR;
    st_ClArgument* vec_b_arg    = SIXTRL_NULLPTR;
    st_ClArgument* result_arg   = SIXTRL_NULLPTR;

    buf_size_t data_size        = ( buf_size_t )0u;

    size_t const N = 1023;

    int program_id    = -1;
    int add_kernel_id = -1;

    /* --------------------------------------------------------------------- */
    /* Handle command line arguments: */
    /* --------------------------------------------------------------------- */

    if( argc < 2  )
    {
        unsigned int num_devices = 0u;

        printf( "Usage: %s [ID] [VECTOR_SIZE]\r\n", argv[ 0 ] );

        context = st_ClContext_create();
        num_devices = st_ClContextBase_get_num_available_nodes( context );

        st_ClContextBase_print_nodes_info( context );

        if( num_devices == 0u )
        {
            printf( "Quitting program!\r\n" );
            return 0;
        }

        printf( "\r\n"
                "[VECTOR_SIZE]   :: Vector size\r\n"
                "                :: Default = %d\r\n", ( int )VECTOR_SIZE );
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    /* Select the node based on the first command line param or
     * select the default node: */

    if( argc >= 2 )
    {
        context = st_ClContext_new( argv[ 1 ] );

        if( context == SIXTRL_NULLPTR )
        {
            printf( "Warning         : Provided ID %s not found "
                    "-> use default device instead\r\n",
                    argv[ 1 ] );
        }
    }

    if( !st_ClContextBase_has_selected_node( context ) )
    {
        /* select default node */
        st_context_node_id_t const default_node_id =
            st_ClContextBase_get_default_node_id( context );

        st_ClContextBase_select_node_by_node_id( context, &default_node_id );
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    /* take the number of particles from the third command line parameter,
     * otherwise keep using the default value */

    if( argc >= 3 )
    {
        int64_t const temp = atoi( argv[ 2 ] );
        if( temp > 0 ) VECTOR_SIZE = temp;
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    /* Give a summary of the provided parameters */

    if( ( context != SIXTRL_NULLPTR ) && ( st_ClContextBase_has_selected_node(
          context ) ) && ( VECTOR_SIZE > 0 ) )
    {
        st_context_node_id_t const* node_id =
            st_ClContextBase_get_selected_node_id( context );

        st_context_node_info_t const* node_info =
            st_ClContextBase_get_selected_node_info( context );

        char id_str[ 16 ];
        st_ComputeNodeId_to_string( node_id, &id_str[ 0 ], 16  );

        printf( "\r\n"
                "Selected [ID]          = %s (%s/%s)\r\n"
                "         [VECTOR_SIZE] = %d\r\n\r\n",
                id_str, st_ComputeNodeInfo_get_name( node_info ),
                st_ComputeNodeInfo_get_platform( node_info ),
                ( int )VECTOR_SIZE );
    }
    else
    {
        /* If we get here, something went wrong, most likely with the
         * selection of the device -> bailing out */

        st_ClContextBase_delete( context );

        return 0;
    }

    /* --------------------------------------------------------------------- */

    data_size = sizeof( double ) * VECTOR_SIZE;

    vector_a = ( double* )malloc( data_size );
    vector_b = ( double* )malloc( data_size );
    result   = ( double* )malloc( data_size );

    for( ; ii < VECTOR_SIZE ; ++ii )
    {
        vector_a[ ii ] =    ( double )ii;
        vector_b[ ii ] = -( ( double )ii );
        result[ ii ]   = vector_a[ ii ];
    }

    /* Compile program */

    path_to_program  = ( char* )malloc( sizeof( char ) * ( N + 1 ) );
    memset(  path_to_program,  ( int )'\0', N + 1 );
    strncpy( path_to_program, st_PATH_TO_BASE_DIR, N );
    strncat( path_to_program, "examples/c99/run_opencl_kernel.cl",
             1023 - strlen( path_to_program ) );

    compile_options = ( char* )malloc( sizeof( char ) * ( N + 1 ) );
    memset(  compile_options, ( int )'\0', N + 1 );
    strncpy( compile_options, "-D_GPUCODE=1 -I", N + 1 );
    strncat( compile_options, st_PATH_TO_BASE_DIR, N - strlen( compile_options ) );

    program_id = st_ClContextBase_add_program_file(
        context, path_to_program, compile_options );

    if( program_id < 0 )
    {
        printf( "Error: unable to add program -> stopping\r\n" );

        free( vector_a );
        free( vector_b );
        free( result );

        free( path_to_program );
        free( compile_options );

        NS(ClContextBase_delete)( context );
    }

    /* Enable addition kernel */

    kernel_name = ( char* )malloc( sizeof( char ) * ( N + 1 ) );
    memset(  kernel_name, ( int )'\0', N + 1 );
    strncpy( kernel_name, SIXTRL_C99_NAMESPACE_PREFIX_STR, N );
    strncat( kernel_name, "Add_vectors_kernel", N - strlen( kernel_name ) );

    add_kernel_id = st_ClContextBase_enable_kernel(
        context, kernel_name, program_id );

    if( add_kernel_id < 0 )
    {
        if( NS(ClContextBase_is_program_compiled)( context, program_id ) )
        {
            printf( "Error: unknown kernel name %s -> stopping\r\n",
                    kernel_name );
        }
        else
        {
            printf( "Error: problems during compilation of program:\r\n\r\n"
                    "Compile report:\r\n%s\r\n"
                    "Compile options: %s\r\n"
                    "-> Stopping"
                    "\r\n",
                    NS(ClContextBase_get_program_compile_report)( context, program_id ),
                    NS(ClContextBase_get_program_compile_options)( context, program_id ) );
        }

        free( vector_a );
        free( vector_b );
        free( result );

        free( path_to_program );
        free( compile_options );
        free( kernel_name );

        NS(ClContextBase_delete)( context );
    }

    /* Init kernel arguments */

    vec_a_arg  = st_ClArgument_new_from_memory( vector_a, data_size, context );
    vec_b_arg  = st_ClArgument_new_from_memory( vector_b, data_size, context );
    result_arg = st_ClArgument_new_from_memory( result,   data_size, context );

    /* Assign kernel arguments to addition kernel */

    NS(ClContextBase_assign_kernel_argument)( context, add_kernel_id, 0u, vec_a_arg  );
    NS(ClContextBase_assign_kernel_argument)( context, add_kernel_id, 1u, vec_b_arg  );
    NS(ClContextBase_assign_kernel_argument)( context, add_kernel_id, 2u, result_arg );

    NS(ClContextBase_assign_kernel_argument_value)(
        context, add_kernel_id, 3u, &VECTOR_SIZE, sizeof( VECTOR_SIZE ) );

    /* run addition kernel */

    bool success = NS(ClContextBase_run_kernel)( context, add_kernel_id, VECTOR_SIZE );

    /* Read-back the result and print it out */

    success &= NS(ClArgument_read_memory)( result_arg, result, data_size );

    if( success )
    {
        for( ii = 0 ; ii < VECTOR_SIZE ; ++ii )
        {
            printf( "%10d :: a = %12.6f , b = %12.6f , a + b = %12.6f\r\n",
                    ( int )ii, vector_a[ ii ], vector_b[ ii ], result[ ii ] );
        }

        printf( "\r\n" );
    }
    else
    {
        printf( "Error running kernel -> stopping\r\n" );
    }

    /* Run the same kernel again, but this time with an explicitly provided
     * work-group size; In this example, we use the maximum available
     * work-group size for the given kernel */

    if( success )
    {
        size_t const max_wg_size =
            NS(ClContextBase_get_kernel_max_work_group_size)(
                    context, add_kernel_id );

        success = NS(ClContextBase_run_kernel_wgsize)(
            context, add_kernel_id, VECTOR_SIZE, max_wg_size );

        success &= NS(ClArgument_read_memory)( result_arg, result, data_size );
    }

    if( success )
    {
        for( ii = 0 ; ii < VECTOR_SIZE ; ++ii )
        {
            printf( "%10d :: a = %12.6f , b = %12.6f , a + b = %12.6f\r\n",
                    ( int )ii, vector_a[ ii ], vector_b[ ii ], result[ ii ] );
        }

        printf( "\r\n" );
    }
    else
    {
        printf( "Error running kernel -> stopping\r\n" );
    }

    /* --------------------------------------------------------------------- */

    st_ClArgument_delete( vec_a_arg  );
    st_ClArgument_delete( vec_b_arg  );
    st_ClArgument_delete( result_arg );

    free( vector_a );
    free( vector_b );
    free( result );

    free( path_to_program );
    free( compile_options );
    free( kernel_name );

    st_ClContextBase_delete( context );

    return 0;
}

/* end: examples/c99/run_opencl_kernel.c */
