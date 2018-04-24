#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>

#include "sixtracklib/sixtracklib.h"

void handle_cmd_line_arguments( int argc, char* argv[], 
    const st_OpenCLEnv *const ocl_env, char device_id_str[], 
    size_t* ptr_num_particles, size_t* ptr_num_elements, 
    size_t* ptr_num_turns );


int main( int argc, char* argv[] )
{
    char device_id_str[ 16 ] = 
    {  '0',  '.',  '0', '\0', '\0', '\0', '\0', '\0', 
      '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0'         
    };
    
        
    size_t NUM_ELEMS     = 100;
    size_t NUM_PARTICLES = 100000;
    size_t NUM_TURNS     = 100;
    
    /* 
    size_t ii = 0;
    
    struct timeval tstart;
    struct timeval tstop;
    
    st_Particles* particles = 0;
    double* drift_lengths = 0;
    */
    
    /* first: init the pseudo random number generator for the particle
     * values randomization - choose a constant seed to have reproducible
     * results! */
    
    uint64_t seed = UINT64_C( 20180420 );
    st_Random_init_genrand64( seed );
    
    /* Init the OpenCL Environment -> this gives us a list of available 
     * devices which is displayed if no command line parameters have been 
     * provided */
    
    st_OpenCLEnv* ocl_env = st_OpenCLEnv_init();
    
    handle_cmd_line_arguments( argc, argv, ocl_env, device_id_str, 
                               &NUM_PARTICLES, &NUM_ELEMS, &NUM_TURNS );
    
    if( st_OpenCLEnv_get_num_node_devices( ocl_env ) > ( size_t )0u )
    {
        char kernel_files[] = 
            "sixtracklib/_impl/namespace_begin.h, "
            "sixtracklib/_impl/definitions.h, "            
            "sixtracklib/common/impl/particles_type.h, "
            "sixtracklib/common/impl/block_type.h, "
            "sixtracklib/common/impl/block_drift_type.h, "
            //"sixtracklib/common/track.h, "
            "sixtracklib/examples/test.cl, "
            "sixtracklib/_impl/namespace_end.h";
            
        char compile_options[] = "-D _GPUCODE=1 -D __NAMESPACE=st_";
        
        bool success = st_OpenCLEnv_prepare( 
            ocl_env, device_id_str, "Track_print", 
            kernel_files, compile_options, 0, 0, 0u );
        
        if( success )
        {
            printf( "Success!\r\n" );
        }
    }
    
    st_OpenCLEnv_free( ocl_env );
    free( ocl_env );
    ocl_env = 0;
    
    return 0;
}

/* ------------------------------------------------------------------------- */

void handle_cmd_line_arguments( int argc, char* argv[], 
    const st_OpenCLEnv *const ocl_env, char device_id_str[], 
    size_t* ptr_num_particles, size_t* ptr_num_elements, size_t* ptr_num_turns )
{
    if( argc >= 1 )
    {
        memset( &device_id_str[ 0 ], ( int )'\0', 16 );
        
        printf( "\r\n\r\n"
                "Usage: %s DEVICE_ID NUM_PARTICLES  NUM_ELEMS  NUM_TURNS\r\n",                
                argv[ 0 ] );        
        
        if( ( argc == 1 ) && ( ocl_env != 0 ) )
        {
            size_t const num_devices = 
                st_OpenCLEnv_get_num_node_devices( ocl_env );
            
            st_OpenCLEnvNodeDevice const* node = 
                st_OpenCLEnv_get_node_devices( ocl_env );
            
            printf( "\r\n Listing %4lu platform/device combinations found: \r\n"
                    " DEVICE_ID || PLATTFORM / DEVICE NAME / EXTENSIONS \r\n",                     
                    num_devices );
            
            while( node != 0 )
            {
                printf( "----------------------------------------------------"
                        "----------------------------------------------------\r\n" );
                
                printf( "%10s || %s\r\n" "           || %s\r\n" 
                        "           || %s\r\n", 
                        node->id_str, node->platform_name, node->device_name,
                        node->extensions );
                
                node = node->ptr_next;
            }
            
            printf( "\r\n" );
        }
        else if( argc > 1 )
        {
            strncpy( device_id_str, argv[ 1 ], 16 );
        }
    }
    
    if( ( argc == 2 ) && ( ptr_num_particles != 0 ) && 
        ( ptr_num_elements != 0 ) && ( ptr_num_turns != 0 ) )
    {
        printf( "run with default values .... "
                "(npart = %8lu, nelem = %8lu, nturns = %8lu)\r\n",
                *ptr_num_particles, *ptr_num_elements, *ptr_num_turns );
    }
    else if( argc >= 4 )
    {
        size_t const temp_num_part = atoi( argv[ 1 ] );
        size_t const temp_num_elem = atoi( argv[ 2 ] );
        size_t const temp_num_turn = atoi( argv[ 3 ] );
        
        if( ( temp_num_part > 0 ) && ( ptr_num_particles != 0 ) )
        {
            *ptr_num_particles = temp_num_part;
        }
        
        if( ( temp_num_elem > 0 ) && ( ptr_num_elements != 0 ) )
        {
            *ptr_num_elements = temp_num_elem;
        }
        
        if( ( temp_num_turn > 0 ) && ( ptr_num_turns != 0 ) )
        {
            *ptr_num_turns = temp_num_turn;                
        }
    }
    
    return;
}

/* end: sixtracklib/examples/simple_drift_opencl.c */
