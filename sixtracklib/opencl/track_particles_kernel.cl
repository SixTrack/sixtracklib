#ifndef SIXTRACKLIB_OPENCL_TRACK_PARTICLES_KERNEL_CL__
#define SIXTRACKLIB_OPENCL_TRACK_PARTICLES_KERNEL_CL__

#if defined( _GPUCODE )

void __kernel Track_remap_serialized_blocks_buffer(
     __global unsigned char* restrict  particles_data_buffer,
     __global unsigned char* restrict beam_elements_data_buffer,
     __global unsigned char* restrict elem_by_elem_data_buffer, 
     __global long int*      restrict ptr_success_flag )
{
    size_t const global_id   = get_global_id( 0 );
    size_t const global_size = get_global_size( 0 );
    
    size_t const gid_to_remap_particles = 0;
    
    size_t const gid_to_remap_beam_elements = ( global_size > 1u )
        ? 1u : gid_to_remap_particles;
        
    size_t const gid_to_remap_elem_by_elem = ( global_size > 2u )
        ? 2u : gid_to_remap_beam_elements;
    
    long int  success_flag = 0;
        
    if( global_id <= gid_to_remap_elem_by_elem )
    {
        if( global_id == gid_to_remap_particles )
        {
            NS(Blocks) particles_buffer;
            NS(Blocks_preset)( &particles_buffer );
            
            if( 0 != NS(Blocks_unserialize)( &particles_buffer, 
                        particles_data_buffer ) )
            {
                success_flag |= -1;
            }
        }
        
        if( ( success_flag == 0 ) && 
            ( global_id == gid_to_remap_beam_elements ) )
        {        
            NS(Blocks) beam_elements;
            NS(Blocks_preset)( &beam_elements );
            
            if( 0 != NS(Blocks_unserialize)( &beam_elements, 
                        beam_elements_data_buffer ) )
            {
                success_flag = -2;
            }
        }
        
        if( ( success_flag == 0 ) && 
            ( global_id == gid_to_remap_elem_by_elem ) )
        {
            __global unsigned long const* header = 
                    ( __global unsigned long const* )elem_by_elem_data_buffer;
            
            if( ( header != 0 ) && ( header[ 0 ] != 0u ) )
            {
                NS(Blocks) elem_by_elem_buffer;
                NS(Blocks_preset)( &elem_by_elem_buffer );
                
                if( 0 != NS(Blocks_unserialize)( 
                        &elem_by_elem_buffer, elem_by_elem_data_buffer ) )
                {
                    success_flag = -4;
                }
            }
        }                
        
        if( ( success_flag != 0 ) && ( ptr_success_flag  != 0 ) )
        {
            *ptr_success_flag |= success_flag;
        }
    }

    return;
}

void __kernel Track_particles_kernel_opencl(
    unsigned long const num_of_turns, 
    __global unsigned char* restrict particles_data_buffer,
    __global unsigned char* restrict beam_elements_data_buffer, 
    __global unsigned char* restrict elem_by_elem_data_buffer, 
    __global long int*      restrict ptr_success_flag )
{
    typedef __global unsigned long const* g_ulong_cptr_t;
    typedef __global NS(BlockInfo)*       g_info_ptr_t;
    typedef __global NS(Particles)*       g_particles_ptr_t;
    
    size_t const global_id   = get_global_id( 0 );
    size_t const global_size = get_global_size( 0 );
        
    NS(block_size_t) num_particle_blocks              = 0u;
    NS(block_size_t) num_beam_elements                = 0u;
    NS(block_size_t) num_elem_by_elem_blocks          = 0u;
    NS(block_size_t) num_elem_by_elem_blocks_per_turn = 0u;
    NS(block_size_t) num_required_elem_by_elem_blocks = 0u;
    NS(block_size_t) num_of_particles                 = 0u;
    
    NS(Blocks) particles_buffer;
    NS(Blocks) beam_elements;
    NS(Blocks) elem_by_elem_buffer;
    
    NS(Particles) particles;
    
    long int success_flag = 0;
    bool use_elem_by_elem_buffer = false;
    
    /* --------------------------------------------------------------------- */
    /* *****  SECTION FOR DEFUSING THE HEISENBUG *** */
    /* This printf section seems to defuse the Heisenbug on the AMDGPU 
     * implementation available to the author -> YMMV */
    
    /*
    if( global_id == 0u )
    {
        g_particles_ptr_t particles_header = 
            ( g_particles_ptr_t )particles_data_buffer;
            
        printf( "before unserialization: \r\n" );
        printf( "global_id     = %u\r\n", global_id );
        
        printf( "particles_header[ 0 ]   = %16x at %18x\r\n", 
                 particles_header[ 0 ], 
                 ( uintptr_t )( particles_data_buffer +  0 ) );
            
        printf( "particles_header[ 1 ]   = %16x at %18x\r\n", 
                 particles_header[ 1 ], 
                 ( uintptr_t )( particles_data_buffer +  8 ) );
            
        printf( "particles_header[ 2 ]   = %16x at %18x\r\n", 
                 particles_header[ 2 ], 
                ( uintptr_t )( particles_data_buffer + 16 ) );
            
        printf( "particles_header[ 3 ]   = %16x at %18x\r\n", 
                 particles_header[ 3 ], 
                ( uintptr_t )( particles_data_buffer + 32 ) );
    }
    
    barrier( CLK_GLOBAL_MEM_FENCE );    
    */
    
    /* *****  END OF SECTION FOR DEFUSING THE HEISENBUG *** */    
    /* --------------------------------------------------------------------- */
    
    NS(Blocks_preset)( &particles_buffer );
        
    if( 0 == NS(Blocks_unserialize_without_remapping)( 
                &particles_buffer, particles_data_buffer ) )
    {
        num_particle_blocks = NS(Blocks_get_num_of_blocks)( &particles_buffer );
        
        if( num_particle_blocks == 1u )
        {
            g_particles_ptr_t ptr_particles  = 0;
            g_info_ptr_t  ptr_particles_info = 
                NS(Blocks_get_block_infos_begin)( &particles_buffer );
            
            NS(BlockInfo) particles_info;
            
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
            
            num_of_particles = NS(Particles_get_num_particles)( &particles );
        }
    }
    
    if( num_of_particles == 0u )
    {
        NS(Blocks_preset)( &particles_buffer );
        NS(Particles_preset)( &particles );
        success_flag |= -1;
    }
    
    NS(Blocks_preset)( &beam_elements );
    
    if( 0 == NS(Blocks_unserialize_without_remapping)( 
                &beam_elements, beam_elements_data_buffer ) )
    {
        num_beam_elements = NS(Blocks_get_num_of_blocks)( &beam_elements );
    }
    
    if( num_beam_elements == 0u )
    {
        NS(Blocks_preset)( &beam_elements );
        success_flag |= -2;
    }
    
    if( elem_by_elem_data_buffer != 0 )                
    {
        g_ulong_cptr_t elem_by_elem_header = 
            ( g_ulong_cptr_t )elem_by_elem_data_buffer;
        
        num_elem_by_elem_blocks_per_turn = 
            num_beam_elements * num_particle_blocks;
            
        num_required_elem_by_elem_blocks = 
            num_of_turns * num_elem_by_elem_blocks_per_turn;
        
        NS(Blocks_preset)( &elem_by_elem_buffer );
        
        if( 0 == NS(Blocks_unserialize_without_remapping)( 
                    &elem_by_elem_buffer, elem_by_elem_data_buffer ) )
        {
            num_elem_by_elem_blocks = 
                NS(Blocks_get_num_of_blocks)( &elem_by_elem_buffer );
                
            if( ( num_required_elem_by_elem_blocks > 0u ) &&
                ( num_elem_by_elem_blocks >= 
                    num_required_elem_by_elem_blocks ) )
            {
                use_elem_by_elem_buffer = true;
            }
        }
        else if( elem_by_elem_header[ 0 ] != ( unsigned long )0u )
        {
            NS(Blocks_preset)( &elem_by_elem_buffer );
            success_flag |= -4;
        }
    }
    
    if( ( success_flag        == 0  ) && ( num_of_turns     != 0u ) &&
        ( num_beam_elements   != 0u ) && ( num_of_particles != 0u ) &&
        ( num_particle_blocks == 1u ) )
    {
        int ret = 0;
        unsigned long ii = 0u;
        
        if( global_id < num_of_particles )
        {
            if( !use_elem_by_elem_buffer )
            {
                for( ; ii < num_of_turns ; ++ii )
                {
                    ret |= NS(Track_beam_elements_particle)( 
                            &particles, global_id, &beam_elements, 0 );
                }
                
                if( ret != 0 ) success_flag |= -8;
            }
            else
            {
                SIXTRL_GLOBAL_DEC NS(BlockInfo)* io_info_it =
                    NS(Blocks_get_block_infos_begin)( &elem_by_elem_buffer );
                
                for( ; ii < num_of_turns ; ++ii, 
                        io_info_it = io_info_it + num_beam_elements )
                {
                    ret |= NS(Track_beam_elements_particle)( 
                            &particles, global_id, &beam_elements, io_info_it );
                    
                    if( io_info_it != 0 )
                    {
                        io_info_it = io_info_it + 
                            num_elem_by_elem_blocks_per_turn;
                    }
                }
                
                if( ret != 0 ) success_flag |= -16;
            }
        }
    }
    else
    {
        success_flag |= -32;
    }
    
    if( ( success_flag != 0 ) && ( ptr_success_flag != 0 ) )
    {
        *ptr_success_flag |= success_flag;
    }
    
    return;
}
    
#endif /* defined( _GPUCODE ) */
    
#endif /* SIXTRACKLIB_OPENCL_TRACK_PARTICLES_KERNEL_CL__ */

/* end: sixtracklib/opencl/track_particles_kernel.cl */
