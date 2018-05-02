#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>

#include "sixtracklib/sixtracklib.h"

int main( int argc, char* argv[] )
{
    size_t ii = 0;
    st_block_num_elements_t jj = 0;
        
    st_block_num_elements_t NUM_ELEMS     = 100;
    st_block_num_elements_t NUM_PARTICLES = 100000;
    size_t NUM_TURNS     = 100;
    
    struct timeval tstart;
    struct timeval tstop;
    
    st_ParticlesContainer particles_buffer;
    st_Particles          particles;    
    st_BeamElements       beam_elements;
    
    int ret = 0;
    
    /* first: init the pseudo random number generator for the particle
     * values randomization - choose a constant seed to have reproducible
     * results! */
    
    uint64_t seed = UINT64_C( 20180420 );
    st_Random_init_genrand64( seed );
    
    if( argc >= 1 )
    {
        printf( "\r\n\r\nUsage: %s NUM_PARTICLES  NUM_ELEMS  NUM_TURNS\r\n", argv[ 0 ] );        
    }
    
    if( argc < 2 )
    {
        printf( "run with default values .... "
                "(npart = %8lu, nelem = %8lu, nturns = %8lu)\r\n",
                NUM_PARTICLES, NUM_ELEMS, NUM_TURNS );
    }
    else if( argc >= 4 )
    {
        st_block_num_elements_t const temp_num_part = atoi( argv[ 1 ] );
        st_block_num_elements_t const temp_num_elem = atoi( argv[ 2 ] );
        size_t const temp_num_turn = atoi( argv[ 3 ] );
        
        if( temp_num_part > 0 ) NUM_PARTICLES = temp_num_part;
        if( temp_num_elem > 0 ) NUM_ELEMS     = temp_num_elem;
        if( temp_num_turn > 0 ) NUM_TURNS     = temp_num_turn;                
    }
    
    printf( "\r\n" "starting :                   "
                "(npart = %8lu, nelem = %8lu, nturns = %8lu)\r\n",
                NUM_PARTICLES, NUM_ELEMS, NUM_TURNS );
    
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
        
        return 0;
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
            break;
        }
    }
    
    /* track over a number of turns and measure the wall-time for the tracking: */
    
    gettimeofday( &tstart, 0 );
    
    for( ii = 0 ; ii < NUM_TURNS ; ++ii )
    {
        st_BlockInfo* block_info_it = 
            st_BeamElements_get_block_infos_begin( &beam_elements );
            
        unsigned char* be_mem_begin = 
            st_BeamElements_get_ptr_data_begin( &beam_elements );            

        st_block_size_t const be_max_num_bytes =
            st_BeamElements_get_data_capacity( &beam_elements );
            
        for( jj = 0 ; jj < NUM_ELEMS ; ++jj, ++block_info_it )
        {
            st_block_num_elements_t pi = 0;
            st_block_num_elements_t kk = 0;
            
            st_BlockType const type_id = st_BlockInfo_get_type_id( block_info_it );
                        
            switch( type_id )
            {
                case st_BLOCK_TYPE_DRIFT: 
                {
                    st_Drift drift;
                    st_Drift_preset( &drift );
                    
                    ret = st_Drift_map_from_memory_for_reading_aligned(
                        &drift, block_info_it, be_mem_begin, be_max_num_bytes );
                    
                    for( ; pi < NUM_PARTICLES ; ++pi )
                    {
                        st_Drift_track_particle_over_single_elem( 
                            &particles, pi, &drift, kk );
                    }
                    
                    break;
                }
                
                case st_BLOCK_TYPE_DRIFT_EXACT:
                {
                    st_Drift drift;
                    st_Drift_preset( &drift );
                    
                    ret = st_Drift_map_from_memory_for_reading_aligned(
                        &drift, block_info_it, be_mem_begin, be_max_num_bytes );
                    
                    for( ; pi < NUM_PARTICLES ; ++pi )
                    {
                        st_DriftExact_track_particle_over_single_elem( 
                            &particles, pi, &drift, kk );
                    }
                    
                    break;
                }
                
                default:
                {
                    printf( "unknown block type_id = %u --> skipping!\r\n",
                            st_BlockType_to_number( type_id ) );
                }
            };
        }
    }
    
    gettimeofday( &tstop, 0 );
    
    double const usec_dist = 1e-6 * ( ( tstop.tv_sec >= tstart.tv_sec ) ?
        ( tstop.tv_usec - tstart.tv_usec ) : ( 1000000 - tstart.tv_usec ) );
    
    double const delta_time = ( double )( tstop.tv_sec - tstart.tv_sec ) + usec_dist;
    printf( "time elapsed    : %16.8fs \r\n", delta_time );
    printf( "time / particle : %16.8f [us/Part]\r\n", 
                ( delta_time / NUM_PARTICLES ) * 1e6 );
    
    /* cleanup: */
              
    st_ParticlesContainer_free( &particles_buffer );
    st_BeamElements_free( &beam_elements );
        
    return 0;
}

/* end:  sixtracklib/examples/simple_drift.c */
