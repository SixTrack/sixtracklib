#ifndef SIXTRACKLIB_OPENCL_TRACK_PARTICLES_KERNEL_CL__
#define SIXTRACKLIB_OPENCL_TRACK_PARTICLES_KERNEL_CL__

#if defined( _GPUCODE )

void __kernel Track_particles_kernel_opencl(
    unsigned long const num_of_turns, 
    unsigned long const num_of_be_blocks,
    __global NS(BlockInfo)* SIXTRL_RESTRICT beam_elem_info_begin,
    __global unsigned char* SIXTRL_RESTRICT beam_elem_data_begin, 
    unsigned long const beam_elem_max_num_bytes,                                            
    unsigned long const num_of_particle_blocks, 
    __global NS(BlockInfo)* SIXTRL_RESTRICT particle_block_infos_begin,
    __global unsigned char* SIXTRL_RESTRICT particle_data_begin,
    unsigned long const particles_max_num_bytes,    
    __global NS(BlockInfo)* SIXTRL_RESTRICT elem_by_elem_block_infos_begin,
    __global unsigned char* SIXTRL_RESTRICT elem_by_elem_data_begin,
    unsigned long const elem_by_elem_max_num_bytes,    
    __global NS(BlockInfo)* SIXTRL_RESTRICT turn_by_turn_block_infos_begin,
    __global unsigned char* SIXTRL_RESTRICT turn_by_turn_data_begin,
    unsigned long const turn_by_turn_max_num_bytes )
{
    size_t const global_id = get_global_id( 0 );
    
    NS(block_num_elements_t) const num_elem_by_elem_blocks =
        num_of_turns * num_of_be_blocks;
    
    NS(BeamElements) beam_elements = NS(BeamElements_assemble)(
        beam_elem_info_begin, num_of_be_blocks, beam_elem_data_begin, 
        beam_elem_max_num_bytes );
    
    NS(ParticlesContainer) particles_buffer;
    NS(ParticlesContainer) elem_by_elem_buffer;
    NS(ParticlesContainer) turn_by_turn_buffer;
    
    NS(ParticlesContainer_assemble)( &particles_buffer,
        particle_block_infos_begin, num_of_particle_blocks, 
        particle_data_begin, particles_max_num_bytes );
    
    NS(ParticlesContainer_assemble)( &elem_by_elem_buffer,
        elem_by_elem_block_infos_begin, num_elem_by_elem_blocks,
        elem_by_elem_data_begin, elem_by_elem_max_num_bytes );
        
    NS(ParticlesContainer_assemble)( &turn_by_turn_buffer,
        turn_by_turn_block_infos_begin, num_turn_by_turn_blocks,
        turn_by_turn_data_begin, turn_by_turn_max_num_bytes );
    
    NS(Particles) particles;
    
    int use_turn_by_turn_buffer = ( num_turns <= 
        NS(ParticlesContainer_get_num_of_blocks)( &turn_by_turn_buffer ) );
    
    unsigned long ii;
    unsigned long elem_by_elem_start_idx = 0;
    
    unsigned long const global_id = get_global_id();
    
    NS(ParticlesContainer_get_particles)( &particles, &particles_buffer, 0 );
    
    for( ii = 0 ; ii < num_of_turns ; 
            ++ii, elem_by_elem_start_idx += num_of_be_blocks )
    {       
        NS(Track_beam_elements)( 
            &particles, global_id, global_id + 1, 
                &beam_elements, elem_by_elem_start_idx, &elem_by_elem_buffer );
        
        if( use_turn_by_turn_buffer )
        {
            NS(Particles) particles_after_turn;
            
            status |= NS(ParticlesContainer_get_particles)( 
                &particles_after_turn, &turn_by_turn_buffer, ii );
            
            NS(Particles_copy_all_unchecked)( &particles_after_turn, 
                                              &particles );
        }
    }
    
    return;
}
    
#endif /* defined( _GPUCODE ) */
    
#endif /* SIXTRACKLIB_OPENCL_TRACK_PARTICLES_KERNEL_CL__ */

/* end: sixtracklib/opencl/track_particles_kernel.cl */
