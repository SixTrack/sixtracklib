#ifndef SIXTRACKLIB_OPENCL_TRACK_PARTICLES_KERNEL_CL__
#define SIXTRACKLIB_OPENCL_TRACK_PARTICLES_KERNEL_CL__

#if defined( _GPUCODE )

void __kernel Track_particles_kernel_opencl(
    unsigned long const num_of_turns, 
    unsigned long const num_of_beam_elements,
    unsigned long const num_of_particles, 
    __global NS(BeamElemInfo)* SIXTRL_RESTRICT beam_elem_info_begin,
    __global unsigned char* SIXTRL_RESTRICT beam_elem_data_begin )
{
    typedef __global SIXTRL_REAL_T*     g_ptr_real_t;
    typedef __global SIXTRL_INT64_T*    g_ptr_i64_t;
    
    /* let's assume that work_dim == 1, otherwise the calls below would get 
     * get a bit more complicated than ( 0 ). */    
    size_t const work_dim    = get_work_dim(); 
    
    size_t const num_groups  = get_num_groups( 0 );
    size_t const group_id    = get_group_id( 0 );
    
    size_t const global_size = get_global_size( 0 );
    size_t const global_id   = get_global_id( 0 );
    
    size_t const local_size  = get_local_size( 0 );
    size_t const local_id    = get_local_id( 0 );
    
    size_t const SIZE_REAL = sizeof( SIXTRL_REAL_T );
    
    unsigned long ii;
    unsigned long jj;
    
    for( ii = 0 ; ii < num_of_turns ; ++ii )
    {        
        for( jj = 0 ; jj < num_of_beam_elements ; ++jj )
        {
            unsigned long const off     = beam_elem_info_begin[ jj ].mem_offset;
            unsigned long const type_id = beam_elem_info_begin[ jj ].type_id;
            
            switch( type_id )
            {
                case 2:            
                {
                    NS(Drift) drift;
                    drift.type_id = type_id;
                    
                    drift.length  = 
                        ( g_ptr_real_t )( beam_elem_data_begin + off );
                    
                    drift.element_id = 
                        ( g_ptr_i64_t )( beam_elem_data_begin + off + SIZE_REAL );
                        
                    printf( "global_id=%6lu :: ii = %2lu | jj = %4lu || "
                            "type_id = %2lu, length = %8.4f, element_id = %4ld\r\n",
                            global_id, ii, jj, type_id, *drift.length, *drift.element_id );
                    
                    break;
                }
                
                case 3:
                {
                    NS(Drift) drift;
                    drift.type_id = type_id;
                    
                    drift.length  = 
                        ( g_ptr_real_t )( beam_elem_data_begin + off );
                    
                    drift.element_id = 
                        ( g_ptr_i64_t )( beam_elem_data_begin + off + SIZE_REAL );
                    
                    printf( "global_id=%6lu :: %ii = %2lu | jj = %4lu || "
                            "type_id = %2lu, length = %8.4f, element_id = %4ld\r\n",
                            global_id, ii, jj, type_id, *drift.length, *drift.element_id );
                    
                    break;
                }
            };
        }
    }
    
    return;
}
    
#endif /* defined( _GPUCODE ) */
    
#endif /* SIXTRACKLIB_OPENCL_TRACK_PARTICLES_KERNEL_CL__ */

/* end: sixtracklib/opencl/track_particles_kernel.cl */
