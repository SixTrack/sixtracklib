#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/cuda/impl/track_particles_kernel.cuh"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stddef.h>
    #include <stdint.h>
    #include <stdlib.h>

    #include <cuda_runtime.h>
    #include <cuda.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/internal/buffer_main_defines.h"
    #include "sixtracklib/common/buffer/managed_buffer_minimal.h"
    #include "sixtracklib/common/buffer/managed_buffer_remap.h"
    #include "sixtracklib/common/internal/particles_defines.h"
    #include "sixtracklib/common/particles.h"
    #include "sixtracklib/common/track.h"
    #include "sixtracklib/cuda/impl/cuda_tools.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

__global__ void NS(Remap_particles_beam_elements_buffers_kernel_cuda)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT particles_buffer,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT beam_elem_buffer,
    SIXTRL_BUFFER_DATAPTR_DEC int32_t* SIXTRL_RESTRICT ptr_success_flag )
{
    typedef NS(buffer_size_t) buf_size_t;

    size_t const thread_id = NS(Cuda_get_1d_thread_id_in_kernel)();
    size_t const total_num_threads = NS(Cuda_get_total_num_threads_in_kernel)();

    size_t const thread_id_to_remap_particles_buffers = ( size_t )0u;

    size_t const thread_id_to_remap_beam_elements_buffers =
        ( total_num_threads > ( size_t )1u )
            ? ( thread_id_to_remap_particles_buffers + ( size_t )1u )
            : ( thread_id_to_remap_particles_buffers );

    if( thread_id <= thread_id_to_remap_beam_elements_buffers )
    {
        buf_size_t const slot_size = ( buf_size_t )8u;
        int success_flag = ( int )0u;

        if( thread_id == thread_id_to_remap_particles_buffers )
        {
            if( ( particles_buffer != SIXTRL_NULLPTR ) &&
                ( particles_buffer != beam_elem_buffer ) )
            {
                if( 0 != NS(ManagedBuffer_remap)( particles_buffer, slot_size ) )
                {
                    success_flag |= -2;
                }
            }
            else
            {
                success_flag |= -1;
            }
        }

        if( thread_id == thread_id_to_remap_beam_elements_buffers )
        {
            if( ( beam_elem_buffer != SIXTRL_NULLPTR ) &&
                ( beam_elem_buffer != particles_buffer ) )
            {
                if( 0 != NS(ManagedBuffer_remap)(
                        beam_elem_buffer, slot_size ) )
                {
                    success_flag |= -4;
                }
            }
            else
            {
                success_flag |= -1;
            }
        }

        if( ( success_flag != 0 ) && ( ptr_success_flag != SIXTRL_NULLPTR ) )
        {
            #if ( defined( __CUDA_ARCH__ ) && ( __CUDA_ARCH__ >= 120 ) )
            atomicOr( ptr_success_flag, success_flag );
            #else  /* defined( __CUDA_ARCH__ ) && ( __CUDA_ARCH__ >= 120 ) */
            *ptr_success_flag |= success_flag;
            #endif /* defined( __CUDA_ARCH__ ) && ( __CUDA_ARCH__ >= 120 ) */
        }
    }

    return;
}


__global__ void NS(Track_particles_beam_elements_kernel_cuda)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT particles_buffer,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT beam_elem_buffer,
    SIXTRL_UINT64_T const num_turns,
    SIXTRL_BUFFER_DATAPTR_DEC int32_t* SIXTRL_RESTRICT ptr_success_flag )
{
    typedef NS(buffer_size_t) buf_size_t;
    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC  NS(Object)*        obj_iter_t;
    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC  NS(Object) const*  obj_const_iter_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC     NS(Particles)*     ptr_particles_t;

    int32_t success_flag = ( int32_t )0u;
    buf_size_t const slot_size = ( buf_size_t )8u;

    size_t particle_index = NS(Cuda_get_1d_thread_id_in_kernel)();
    size_t const stride   = NS(Cuda_get_1d_thread_stride_in_kernel)();

    if( ( !NS(ManagedBuffer_needs_remapping( particles_buffer, slot_size ) ) ) &&
        ( !NS(ManagedBuffer_needs_remapping( beam_elem_buffer, slot_size ) ) ) &&
        (  NS(ManagedBuffer_get_num_objects)( particles_buffer, slot_size ) == 1u ) )
    {
        obj_const_iter_t be_begin = NS(ManagedBuffer_get_const_objects_index_begin)(
                beam_elem_buffer, slot_size );

        obj_const_iter_t be_end = NS(ManagedBuffer_get_const_objects_index_end)(
                beam_elem_buffer, slot_size );

        ptr_particles_t particles = NS(BufferIndex_get_particles)(
            NS(ManagedBuffer_get_objects_index_begin)(
                particles_buffer, slot_size ) );

        size_t num_particles  = NS(Particles_get_num_of_particles)( particles );

        while( particle_index < num_particles )
        {
            SIXTRL_UINT64_T turn = 0;

            for( ; turn < num_turns ; ++turn )
            {
                success_flag |= NS(Track_particle_beam_element_objs)(
                    particles, particle_index, be_begin, be_end );
            }

            NS(Particles_set_state_value)( particles, particle_index, 137 );
            particle_index += stride;
        }
    }
    else
    {
        if( NS(ManagedBuffer_needs_remapping( particles_buffer, slot_size ) ) )
        {
            success_flag |= -2;
        }

        if( NS(ManagedBuffer_needs_remapping( beam_elem_buffer, slot_size ) ) )
        {
            success_flag |= -4;
        }

        if( NS(ManagedBuffer_get_num_objects)( particles_buffer, slot_size )
            != 1u )
        {
            success_flag |= -8;
        }
    }

    if( ( success_flag != 0 ) && ( ptr_success_flag != SIXTRL_NULLPTR ) )
    {
        #if ( defined( __CUDA_ARCH__ ) && ( __CUDA_ARCH__ >= 120 ) )
        atomicOr( ptr_success_flag, success_flag );
        #else  /* defined( __CUDA_ARCH__ ) && ( __CUDA_ARCH__ >= 120 ) */
        *ptr_success_flag |= success_flag;
        #endif /* defined( __CUDA_ARCH__ ) && ( __CUDA_ARCH__ >= 120 ) */
    }

    return;
}

/* end: sixtracklib/cuda/details/track_particles_kernel.cu */
