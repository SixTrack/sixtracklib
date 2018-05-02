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
    st_block_num_elements_t* ptr_num_particles, 
    st_block_num_elements_t* ptr_num_elements, 
    size_t* ptr_num_turns );


int main( int argc, char* argv[] )
{
    char device_id_str[ 16 ] = 
    {  '0',  '.',  '0', '\0', '\0', '\0', '\0', '\0', 
      '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0'         
    };
        
    st_block_num_elements_t NUM_ELEMS     = 100;
    st_block_num_elements_t NUM_PARTICLES = 100000;
    size_t NUM_TURNS     = 100;
    
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
    
    if( ( NUM_ELEMS     > ( st_block_num_elements_t )0u ) && 
        ( NUM_PARTICLES > ( st_block_num_elements_t )0u ) &&
        ( NUM_TURNS > ( SIXTRL_SIZE_T )0u ) && 
        ( st_OpenCLEnv_get_num_node_devices( ocl_env ) > ( size_t )0u ) )
    {
        bool success = true;
        
        struct timeval tstart;
        struct timeval tstop;
        
        int ret = 0;
        st_block_num_elements_t jj = 0;
        
        char kernel_files[] = 
            "sixtracklib/_impl/definitions.h, "
            "sixtracklib/common/impl/particles_impl.h, "
            "sixtracklib/common/impl/block_info_impl.h, "
            "sixtracklib/common/impl/be_drift_impl.h, "
            "sixtracklib/opencl/track_particles_kernel.cl";
            
        char compile_options[] = "-D _GPUCODE=1 -D __NAMESPACE=st_";
        
        
        st_ParticlesContainer particles_buffer;
        st_Particles          particles;    
        st_BeamElements       beam_elements;
        
        st_ParticlesContainer_preset( &particles_buffer );  
        st_ParticlesContainer_reserve_num_blocks( &particles_buffer, 1u );
        st_ParticlesContainer_reserve_for_data( &particles_buffer, 20000000u );
                
        
        ret = st_ParticlesContainer_add_particles( 
            &particles_buffer, &particles, NUM_PARTICLES );
        
        if( ret == 0 )
        {
            st_Particles_random_init( &particles );
        }
        else
        {
            printf( "Error initializing particles!\r\n" );
            st_ParticlesContainer_free( &particles_buffer );
            success = false;
        }
        
        st_BeamElements_preset( &beam_elements );
        st_BeamElements_reserve_num_blocks( &beam_elements, NUM_ELEMS );
        st_BeamElements_reserve_for_data( &beam_elements, NUM_ELEMS * sizeof( st_Drift ) );
                
        for( jj = 0 ; jj < NUM_ELEMS ; ++jj )
        {
            double LENGTH = 0.05 + 0.005 * jj;
            st_element_id_t const ELEMENT_ID = ( st_element_id_t )jj;
            
            
            st_Drift next_drift = 
                st_BeamElements_add_drift( &beam_elements, LENGTH, ELEMENT_ID );
            
            if( st_Drift_get_type_id( &next_drift ) != st_BLOCK_TYPE_DRIFT )
            {
                printf( "Error initializing drift #%lu\r\n", jj );
                success = false;
                break;
            }
        }
        
        gettimeofday( &tstart, 0 );
        
        if( success )
        {
            success = st_OpenCLEnv_prepare( ocl_env, device_id_str, 
                "Track_particles_kernel_opencl", kernel_files, 
                compile_options, NUM_TURNS, 0, &beam_elements );
        }
        
        if( success )
        {
              success = st_OpenCLEnv_track_particles( ocl_env, 0, &beam_elements );
        }
        
        gettimeofday( &tstop, 0 );
    
        if( success )
        {
            printf( "Success!\r\n" );
        }
        
        double const usec_dist = 1e-6 * ( ( tstop.tv_sec >= tstart.tv_sec ) ?
            ( tstop.tv_usec - tstart.tv_usec ) : ( 1000000 - tstart.tv_usec ) );
        
        double const delta_time = ( double )( tstop.tv_sec - tstart.tv_sec ) + usec_dist;
        printf( "time elapsed    : %16.8fs \r\n", delta_time );
        printf( "time / particle : %16.8f [us/Part]\r\n", 
                    ( delta_time / NUM_PARTICLES ) * 1e6 );
        
        st_BeamElements_free( &beam_elements );
        st_ParticlesContainer_free( &particles_buffer );
    }
       
    st_OpenCLEnv_free( ocl_env );
    free( ocl_env );
    ocl_env = 0;
    
    return 0;
}

/* ------------------------------------------------------------------------- */

void handle_cmd_line_arguments( int argc, char* argv[], 
    const st_OpenCLEnv *const ocl_env, char device_id_str[], 
    st_block_num_elements_t* ptr_num_particles, st_block_num_elements_t* ptr_num_elements, 
    size_t* ptr_num_turns )
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
        st_block_num_elements_t const temp_num_part = atoi( argv[ 1 ] );
        st_block_num_elements_t const temp_num_elem = atoi( argv[ 2 ] );
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
