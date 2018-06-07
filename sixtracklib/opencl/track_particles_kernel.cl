#ifndef SIXTRACKLIB_OPENCL_TRACK_PARTICLES_KERNEL_CL__
#define SIXTRACKLIB_OPENCL_TRACK_PARTICLES_KERNEL_CL__

#if defined( _GPUCODE )

void __kernel Track_particles_kernel_opencl(
    unsigned long const num_of_turns, 
    __global unsigned char* SIXTRL_RESTRICT particles_data_buffer,
    __global unsigned char* SIXTRL_RESTRICT beam_elements_data_buffer, 
    __global unsigned char* SIXTRL_RESTRICT elem_by_elem_data_buffer )
{
    size_t const global_id   = get_global_id( 0 );
    size_t const local_id    = get_local_id( 0 );
    size_t const group_id    = get_group_id( 0 );
    size_t const group_size  = get_num_groups( 0 );
    size_t const local_size  = get_local_size( 0 );
    size_t const global_size = get_global_size( 0 );
        
    size_t const gid_to_remap_particles = 0;
    
    size_t const gid_to_remap_beam_elements = 
        ( global_size > 1 ) ? 1 : gid_to_remap_particles;
        
    size_t const gid_to_remap_elem_by_elem  = ( global_size > 2 ) 
        ? 2 : gid_to_remap_beam_elements;
    
    NS(block_size_t) num_particle_blocks     = 0;
    NS(block_size_t) num_beam_elements       = 0;
    NS(block_size_t) num_elem_by_elem_blocks = 0;
    
    NS(Blocks) particles_buffer;
    NS(Blocks) beam_elements;
    NS(Blocks) elem_by_elem_buffer;
    
    NS(BlockInfo) particles_info;
    NS(Particles) particles;
    
    __global NS(BlockInfo)* ptr_particles_info = 0;
    __global NS(Particles)* ptr_particles = 0;
    
    int ret = 0;
    
    NS(Blocks_preset)( &particles_buffer );
    NS(Blocks_preset)( &beam_elements );
    NS(Blocks_preset)( &elem_by_elem_buffer );
    
    if( global_id == gid_to_remap_particles )
    {
        ret  = NS(Blocks_unserialize)( &particles_buffer, particles_data_buffer );
    }
    
    if( global_id == gid_to_remap_beam_elements )
    {        
        ret |= NS(Blocks_unserialize)( &beam_elements, beam_elements_data_buffer );
    }
    
    if( global_id == gid_to_remap_elem_by_elem )
    {
        ret |= NS(Blocks_unserialize)( &elem_by_elem_buffer, elem_by_elem_data_buffer );
    }
    
    /* All pointer offsets should be compensated by now -> so there should be 
     * no more race conditions during unserialization as the content of the 
     * data buffers should not be altered any more -> check with oclgrind! */
    
    barrier( CLK_GLOBAL_MEM_FENCE );
    
    if( global_id > gid_to_remap_elem_by_elem )
    {
        ret |= NS(Blocks_unserialize)( 
            &particles_buffer, particles_data_buffer );
        
        ret |= NS(Blocks_unserialize)(
            &beam_elements, beam_elements_data_buffer );
        
        ret |= NS(Blocks_unserialize)(
            &elem_by_elem_buffer, elem_by_elem_data_buffer );
    }
    
    barrier( CLK_GLOBAL_MEM_FENCE );
    
    /* Now all threads should have a properly unserialized instance of the 
     * data structures */
        
    num_particle_blocks = NS(Blocks_get_num_of_blocks)( &particles_buffer );    
    num_beam_elements = NS(Blocks_get_num_of_blocks)( &beam_elements );
    num_elem_by_elem_blocks = NS(Blocks_get_num_of_blocks)( &elem_by_elem_buffer );
    
    ptr_particles_info = NS(Blocks_get_block_infos_begin)( &particles_buffer );
    
    if( ptr_particles_info != 0 )
    {
        particles_info = *ptr_particles_info;
    }
    else
    {
        NS(BlockInfo_preset)( &particles_info );
    }
        
    ptr_particles = NS(Blocks_get_particles)( &particles_info );
    
    if( ptr_particles != 0 )
    {
        particles = *ptr_particles;
    }
    else
    {
        NS(Particles_preset)( &particles );
    }
        
    if( ( ret == 0 ) && ( num_beam_elements != 0u ) && 
        ( ptr_particles != 0 ) && ( num_of_turns != 0 ) && 
        ( num_particle_blocks == 1u ) &&
        ( global_id < NS(Particles_get_num_particles)( &particles ) ) )
    {
        NS(block_size_t) const required_num_elem_by_elem = 
            num_of_turns * num_beam_elements;
            
        if( ( elem_by_elem_data_buffer != 0 ) && 
            ( num_elem_by_elem_blocks >= required_num_elem_by_elem ) )
        {
            printf( "gid = %u :: HERE\r\n", global_id );
            
            unsigned long ii = 0;
            
            SIXTRL_GLOBAL_DEC NS(BlockInfo)* io_info_it =
                NS(Blocks_get_const_block_infos_begin)( &elem_by_elem_buffer );
            
            for( ; ii < num_of_turns ; ++ii, 
                    io_info_it = io_info_it + num_beam_elements )
            {
                ret |= NS(Track_beam_elements_particle)( 
                        &particles, global_id, &beam_elements, io_info_it );
            }
        }
        else
        {
            unsigned long ii = 0;
            
            for( ; ii < num_of_turns ; ++ii )
            {
                ret |= NS(Track_beam_elements_particle)( 
                        &particles, global_id, &beam_elements, 0 );
            }
        }
    }
    
    return;
}
    
#endif /* defined( _GPUCODE ) */
    
#endif /* SIXTRACKLIB_OPENCL_TRACK_PARTICLES_KERNEL_CL__ */

/* end: sixtracklib/opencl/track_particles_kernel.cl */
