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
        
    int use_elem_by_elem_buffer = 0;
    
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
    
    /* *****  SECTION FOR DEFUSING THE HEISENBUG *** */
    /* This printf section seems to defuse the Heisenbug on the AMDGPU 
     * implementation available to the author -> YMMV */
    
    /*
    if( global_id == gid_to_remap_particles )
    {
        __global ulong const* header = ( __global ulong const* )particles_data_buffer;
            
        printf( "before unserialization: \r\n" );
        printf( "global_id     = %u\r\n", global_id );
        printf( "header[ 0 ]   = %16x at %18x\r\n", header[ 0 ], 
            ( uintptr_t )( particles_data_buffer +  0 ) );
            
        printf( "header[ 1 ]   = %16x at %18x\r\n", header[ 1 ], 
            ( uintptr_t )( particles_data_buffer +  8 ) );
            
        printf( "header[ 2 ]   = %16x at %18x\r\n", header[ 2 ], 
            ( uintptr_t )( particles_data_buffer + 16 ) );
            
        printf( "header[ 3 ]   = %16x at %18x\r\n", header[ 3 ], 
            ( uintptr_t )( particles_data_buffer + 32 ) );
    }
    
    barrier( CLK_GLOBAL_MEM_FENCE );    
    */
    
    /* *****  END OF SECTION FOR DEFUSING THE HEISENBUG *** */
    
    NS(Blocks_preset)( &particles_buffer );
    NS(Blocks_preset)( &beam_elements );
    NS(Blocks_preset)( &elem_by_elem_buffer );
    
    if( global_id == gid_to_remap_particles )
    {
        ret  = NS(Blocks_unserialize)( &particles_buffer, particles_data_buffer );        
    }
    
    if( ( ret == 0 ) && ( global_id == gid_to_remap_beam_elements ) )
    {        
        ret |= NS(Blocks_unserialize)( &beam_elements, beam_elements_data_buffer );
    }
    
    if( ( ret == 0 ) && ( global_id == gid_to_remap_elem_by_elem ) )
    {
        ret |= NS(Blocks_unserialize)( 
            &elem_by_elem_buffer, elem_by_elem_data_buffer );
        
        if( ret == 0 )
        {
            use_elem_by_elem_buffer = 1;
        }
        else
        {
            __global unsigned long const* header = 
                ( __global unsigned long const* )elem_by_elem_data_buffer;
                
            ret = ( ( header[ 0 ] == ( unsigned long )0u ) && 
                    ( header[ 1 ] == ( unsigned long )0u ) &&
                    ( header[ 2 ] == ( unsigned long )0u ) &&
                    ( header[ 3 ] == ( unsigned long )0u ) ) ? 0 : -1;
        }
    }
        
    /* All pointer offsets should be compensated by now -> so there should be 
     * no more race conditions during unserialization as the content of the 
     * data buffers should not be altered any more -> check with oclgrind! */
    
    barrier( CLK_GLOBAL_MEM_FENCE );
    
    if( ret == 0 )
    {
        if( global_id != gid_to_remap_particles )
        {
            ret = NS(Blocks_unserialize)( 
                &particles_buffer, particles_data_buffer );
        }
        
        num_particle_blocks = NS(Blocks_get_num_of_blocks)( &particles_buffer );
        
        if( ( ret == 0 ) && ( num_particle_blocks == 1u ) )
        {
            ptr_particles_info = NS(Blocks_get_block_infos_begin)( 
                &particles_buffer );
        
            if( ptr_particles_info != 0 )
            {
                particles_info = *ptr_particles_info;
            }
            else
            {
                NS(BlockInfo_preset)( &particles_info );
                ret = -1;
            }
                
            ptr_particles = NS(Blocks_get_particles)( &particles_info );
            
            if( ptr_particles != 0 )
            {
                particles = *ptr_particles;
            }
            else
            {
                NS(Particles_preset)( &particles );
                ret = -1;
            }
        }
        
        if( ( ret == 0 ) && ( global_id != gid_to_remap_beam_elements ) )
        {
            ret |= NS(Blocks_unserialize)(
                &beam_elements, beam_elements_data_buffer );
        }
        
        num_beam_elements = NS(Blocks_get_num_of_blocks)( &beam_elements );
        
        if( ( ret == 0 ) && ( use_elem_by_elem_buffer == 1 ) )
        {
            NS(block_size_t) const required_num_elem_by_elem =
                    num_of_turns * num_beam_elements * num_particle_blocks;
            
            num_elem_by_elem_blocks = NS(Blocks_get_num_of_blocks)( &elem_by_elem_buffer );
            
            if( global_id != gid_to_remap_elem_by_elem )
            {
                ret = NS(Blocks_unserialize)(
                    &elem_by_elem_buffer, elem_by_elem_data_buffer );
            }
            
            num_elem_by_elem_blocks = NS(Blocks_get_num_of_blocks)( 
                &elem_by_elem_buffer );
            
            if( num_elem_by_elem_blocks < required_num_elem_by_elem )
            {
                use_elem_by_elem_buffer = 0;
            }
        }
    }
    
    barrier( CLK_GLOBAL_MEM_FENCE );
    
    /* Now all threads should have a properly unserialized instance of the 
     * data structures */
        
    if( global_id == 0 )
    {
        printf( "ret                 = %d\r\n", ret );
        printf( "num_beam_elements   = %u\r\n", num_beam_elements );
        printf( "ptr_particles       = %x\r\n", ptr_particles );
        printf( "num_of_turns        = %u\r\n", num_of_turns );
        printf( "num_particle_blocks = %u\r\n", num_particle_blocks );
        printf( "num_particles       = %u\r\n", 
            NS(Particles_get_num_particles)( &particles ) );
    }
    
    if( ( ret == 0 ) && ( num_beam_elements != 0u ) && 
        ( ptr_particles != 0 ) && ( num_of_turns != 0 ) && 
        ( num_particle_blocks == 1u ) &&
        ( global_id < NS(Particles_get_num_particles)( &particles ) ) )
    {
        NS(block_size_t) const required_num_elem_by_elem = 
            num_of_turns * num_beam_elements;
            
        if( use_elem_by_elem_buffer != 1 )
        {
            unsigned long ii = 0;
            
            for( ; ii < num_of_turns ; ++ii )
            {
                ret |= NS(Track_beam_elements_particle)( 
                        &particles, global_id, &beam_elements, 0 );
            }
        }
        else
        {
            unsigned long ii = 0;
            
            SIXTRL_GLOBAL_DEC NS(BlockInfo)* io_info_it =
                NS(Blocks_get_block_infos_begin)( &elem_by_elem_buffer );
            
            for( ; ii < num_of_turns ; ++ii, 
                    io_info_it = io_info_it + num_beam_elements )
            {
                ret |= NS(Track_beam_elements_particle)( 
                        &particles, global_id, &beam_elements, io_info_it );
            }
        }
    }
    
    return;
}
    
#endif /* defined( _GPUCODE ) */
    
#endif /* SIXTRACKLIB_OPENCL_TRACK_PARTICLES_KERNEL_CL__ */

/* end: sixtracklib/opencl/track_particles_kernel.cl */
