#include "sixtracklib/opencl/tests/test_opencl_tools.h"

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "sixtracklib/_impl/definitions.h"
#include "sixtracklib/common/blocks.h"
#include "sixtracklib/opencl/ocl_environment.h"

extern void handle_cmd_line_arguments( int argc, char* argv[], 
    const NS(OpenCLEnv) *const ocl_env, char device_id_str[], 
    NS(block_num_elements_t)* ptr_num_particles, 
    NS(block_num_elements_t)* ptr_num_elements, 
    NS(block_size_t)* ptr_num_turns );

void handle_cmd_line_arguments( int argc, char* argv[], 
    const NS(OpenCLEnv) *const ocl_env, char device_id_str[], 
    NS(block_num_elements_t)* ptr_num_particles, 
    NS(block_num_elements_t)* ptr_num_elements, 
    NS(block_size_t)* ptr_num_turns )
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
                NS(OpenCLEnv_get_num_node_devices)( ocl_env );
            
            st_OpenCLEnvNodeDevice const* node = 
                NS(OpenCLEnv_get_node_devices)( ocl_env );
            
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
        NS(block_num_elements_t) const temp_num_part = atoi( argv[ 1 ] );
        NS(block_num_elements_t) const temp_num_elem = atoi( argv[ 2 ] );
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

/* end:  sixtracklib/opencl/tests/test_opencl_tools.c */
