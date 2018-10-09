#if !defined( SIXTRL_DISABLE_BEAM_BEAM )
#pragma message "beam-beam elements enabled"
#else
#pragma message "beam-beam elements disabled"
#endif /* !defined( SIXTRL_DISABLE_BEAM_BEAM ) */

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "sixtracklib/sixtracklib.h"
#include "sixtracklib/testlib.h"

int main( int argc, char* argv[] )
{
    typedef  st_buffer_size_t   buf_size_t;
    typedef  st_Buffer          buffer_t;
    typedef  st_Particles       particles_t;

    buf_size_t const NUM_CONFIGURATIONS = 17;

    buf_size_t num_particles_list[] =
    {
        1, 16, 128, 1024, 2048, 4096, 8192, 10000, 16384, 20000, 32768, 40000,
        65536, 100000, 200000, 500000, 1000000
    };

    buf_size_t num_turns_list[] =
    {
        100, 100, 100, 100, 50, 50, 50, 20, 20, 20, 20, 20, 10, 10, 10, 5, 1
    };

    buf_size_t kk = ( buf_size_t )0u;

    char path_to_tracking_program[ 1024 ];
    char tracking_program_compile_options[ 1024 ];

    int tracking_program_id = -1;
    int tracking_kernel_id  = -1;

    memset( path_to_tracking_program,         ( int )'\0', 1024 );
    memset( tracking_program_compile_options, ( int )'\0', 1024 );

    /* ===================================================================== */
    /* ==== Prepare openCL Context -> select computing node                  */

    st_ClContext* context = SIXTRL_NULLPTR;

    if( argc < 2  )
    {
        context = st_ClContext_create();

        buf_size_t const num_nodes =
            st_ClContextBase_get_num_available_nodes( context );

        printf( "# Usage: %s [ID] \r\n",  argv[ 0 ] );

        st_ClContextBase_print_nodes_info( context );

        if( num_nodes > ( buf_size_t )0u )
        {
            printf( "# INFO            :: Selecting default node\r\n" );
        }
        else
        {
            printf( "# Quitting program!\r\n\r\n" );
            return 0;
        }
    }

    if( argc >= 2 )
    {
        context = st_ClContext_new( argv[ 1 ] );

        if( ( context == SIXTRL_NULLPTR ) ||
            ( !st_ClContextBase_has_selected_node( context ) ) )
        {
            printf( "# Warning         : Provided ID %s not found -> "
                    "use default device instead\r\n", argv[ 1 ] );
        }
    }

    if( !st_ClContextBase_has_selected_node( context ) )
    {
        /* select default node */
        st_context_node_id_t const default_node_id =
            st_ClContextBase_get_default_node_id( context );

        st_ClContextBase_select_node_by_node_id( context, &default_node_id );
    }

    if( ( context != SIXTRL_NULLPTR ) &&
        ( st_ClContextBase_has_selected_node( context ) ) )
    {
        char id_str[ 16 ];

        st_context_node_id_t const* node_id =
            st_ClContextBase_get_selected_node_id( context );

        st_context_node_info_t const* node_info =
            st_ClContextBase_get_selected_node_info( context );

        st_ClContextBase_get_selected_node_id_str( context, id_str, 16 );
        st_ComputeNodeId_to_string( node_id, &id_str[ 0 ], 16  );

        printf( "# Selected [ID]            = %s (%s/%s)\r\n"
                "# \r\n", id_str, st_ComputeNodeInfo_get_name( node_info ),
                st_ComputeNodeInfo_get_platform( node_info ) );

    }
    else
    {
        printf( "# Unable to create context and select node "
                "-> stopping program\r\n" );

        return 0;
    }

    /* ===================================================================== */
    /* ==== Prepare Host Buffers                                             */

    /* --------------------------------------------------------------------- */
    /* Setup non-optimized particle tracking: */

    strncpy( path_to_tracking_program, st_PATH_TO_BASE_DIR, 1023 );
    strncat( path_to_tracking_program,
             "sixtracklib/opencl/kernels/"
             "track_particles_priv_particles_optimized_kernel.cl",
             1023 - strlen( path_to_tracking_program ) );

    strncpy( tracking_program_compile_options,
             " -D_GPUCODE=1 -D__NAMESPACE=st_  -cl-strict-aliasing"
             " -DSIXTRL_DISABLE_BEAM_BEAM=1"
             " -DSIXTRL_BUFFER_ARGPTR_DEC=__private"
             " -DSIXTRL_BUFFER_DATAPTR_DEC=__global"
             " -DSIXTRL_PARTICLE_ARGPTR_DEC=__private"
             " -DSIXTRL_PARTICLE_DATAPTR_DEC=__private"
             " -I", 1023 - strlen( tracking_program_compile_options ) );
    strncat( tracking_program_compile_options,
             st_PATH_TO_BASE_DIR,
             1023 - strlen( tracking_program_compile_options ) );

    tracking_program_id = st_ClContextBase_add_program_file(
        context, path_to_tracking_program, tracking_program_compile_options );

    if( ( tracking_program_id < 0 ) || ( tracking_program_id >=
          ( int )st_ClContextBase_get_num_available_programs( context ) ) )
    {
        printf( "ERROR BUILDING TRACKING PROGRAM \r\n" );
        st_ClContextBase_delete( context );
        return 0;
    }

    tracking_kernel_id = st_ClContextBase_enable_kernel( context,
        "st_Track_particles_beam_elements_priv_particles_optimized_opencl",
        tracking_program_id );

    if( ( tracking_kernel_id < 0 ) || ( tracking_kernel_id >=
          ( int )st_ClContextBase_get_num_available_kernels( context ) ) )
    {
        printf( "ERROR ENABLING TRACKING KERNEL\r\n" );
        st_ClContextBase_delete( context );
        return 0;
    }

    if( !st_ClContext_set_tracking_kernel_id( context, tracking_kernel_id ) )
    {
        printf( "ERROR SETTING KERNEL AS TRACKING KERNEL\r\n" );
        st_ClContextBase_delete( context );
        return 0;
    }

    /* --------------------------------------------------------------------- */
    /* Perform NUM_CONFIGURATIONS benchmark runs: */

    printf( "                 NUM_PARTICLES"
            "                     NUM_TURNS"
            "               work group size"
            "                num_work_items"
            "             tracking time [s]"
            "        norm tracking time [s]"
            "\r\n" );

    for( ; kk < NUM_CONFIGURATIONS ; ++kk )
    {
        buf_size_t const NUM_PARTICLES = num_particles_list[ kk ];
        buf_size_t const NUM_TURNS     = num_turns_list[ kk ];

        buffer_t* lhc_particle_dump = st_Buffer_new_from_file(
            st_PATH_TO_TEST_LHC_PARTICLES_DATA_T1_P2_NO_BEAM_BEAM );

        buffer_t* lhc_beam_elements_buffer = st_Buffer_new_from_file(
            st_PATH_TO_TEST_LHC_BEAM_ELEMENTS_DATA_NO_BEAM_BEAM );

        buffer_t* pb = st_Buffer_new( ( buf_size_t )( 1u << 28u ) );

        particles_t* particles = st_Particles_new( pb, NUM_PARTICLES );

        particles_t const* input_particles =
            st_Particles_buffer_get_const_particles( lhc_particle_dump, 0u );

        buf_size_t const num_input_particles =
            st_Particles_get_num_of_particles( input_particles );

        int success = -1;

        if( ( NUM_PARTICLES      > ( buf_size_t )0u ) &&
            ( NUM_TURNS          > ( buf_size_t )0u ) &&
            ( particles          != SIXTRL_NULLPTR ) &&
            ( input_particles    != SIXTRL_NULLPTR ) &&
            ( num_input_particles > ( buf_size_t )0u ) )
        {
            buf_size_t ii = ( buf_size_t )0u;

            for( ; ii < NUM_PARTICLES ; ++ii )
            {
                buf_size_t const jj = ii % num_input_particles;
                st_Particles_copy_single( particles, ii, input_particles, jj );
            }

            success = 0;
        }

        st_ClArgument* particles_arg =
            st_ClArgument_new_from_buffer( pb, context );

        st_ClArgument* beam_elements_arg =
            st_ClArgument_new_from_buffer( lhc_beam_elements_buffer, context );

        if( ( success != 0 ) ||
            ( particles_arg == SIXTRL_NULLPTR ) ||
            ( beam_elements_arg == SIXTRL_NULLPTR ) )
        {
            printf( "ERROR ARGUMENT CREATION \r\n" );
            success = -1;
        }

        /* ----------------------------------------------------------------- */
        /* Perform tracking over NUM_TURNS */
        /* ----------------------------------------------------------------- */

        if( success == 0 )
        {
            success = st_ClContext_track_num_turns(
                context, particles_arg, beam_elements_arg, NUM_TURNS );

            if( success != 0 )
            {
                printf( "ERROR TRACKING\r\n" );
            }
        }

        /* ----------------------------------------------------------------- */
        /* Printout timing */
        /* ----------------------------------------------------------------- */

        if( success == 0 )
        {
            double const tracking_time = st_ClContextBase_get_avg_exec_time(
                context, tracking_kernel_id );

            double const norm_tracking_time  =
                tracking_time  / ( double )( NUM_TURNS * NUM_PARTICLES );

            int const work_group_size =
                st_ClContextBase_get_last_exec_work_group_size(
                    context, tracking_kernel_id );

            int const num_work_items =
                st_ClContextBase_get_last_exec_num_work_items(
                    context, tracking_kernel_id );

            printf( "%30d" "%30d" "%30d" "%30d" "%30.8f" "%30.8f" "\r\n",
                    ( int )NUM_PARTICLES, ( int )NUM_TURNS,
                    work_group_size, num_work_items,
                    tracking_time, norm_tracking_time );
        }

        st_ClContextBase_reset_kernel_exec_timing( context, tracking_kernel_id );

        /* ----------------------------------------------------------------- */
        /* Clean-up */
        /* ----------------------------------------------------------------- */

        st_ClArgument_delete( particles_arg );
        st_ClArgument_delete( beam_elements_arg );

        st_Buffer_delete( lhc_particle_dump );
        st_Buffer_delete( lhc_beam_elements_buffer );
        st_Buffer_delete( pb );
    }

    st_ClContextBase_delete( context );

    return 0;
}

/* end: tests/benchmark/sixtracklib/opencl/benchmark_lhc_no_bb_opencl_c99.c */
