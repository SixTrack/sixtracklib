#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/cuda/impl/cuda_particles_kernel.cuh"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <cuda_runtime.h>
    #include <cuda.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/_impl/definitions.h"
    #include "sixtracklib/common/impl/managed_buffer_minimal.h"
    #include "sixtracklib/common/impl/managed_buffer_remap.h"
    #include "sixtracklib/common/particles.h"
    #include "sixtracklib/cuda/impl/cuda_tools.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

__global__ void NS(Particles_copy_buffer_kernel_cuda)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT in_buffer_begin,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT out_buffer,
    SIXTRL_ARGPTR_DEC int32_t* SIXTRL_RESTRICT ptr_success_flag );


__global__ void NS(Particles_copy_buffer_kernel_cuda)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT in_buffer_begin,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT out_buffer,
    SIXTRL_ARGPTR_DEC int32_t* SIXTRL_RESTRICT ptr_success_flag )
{
    typedef NS(buffer_size_t)                               buf_size_t;
    typedef NS(particle_num_elements_t)                     num_elem_t;

    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)*        obj_iter_t;
    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const*  obj_const_iter_t;

    typedef SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles) const* ptr_const_particles_t;
    typedef SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)*       ptr_particles_t;

    int32_t success_flag = ( int32_t )-1;

    buf_size_t const slot_size = NS(BUFFER_DEFAULT_SLOT_SIZE);

    buf_size_t const num_objects =
        NS(ManagedBuffer_get_num_objects)( in_buffer_begin, slot_size );

    if( ( !NS(ManagedBuffer_needs_remapping(  in_buffer_begin,  slot_size ) ) ) &&
        ( !NS(ManagedBuffer_needs_remapping( out_buffer_begin,  slot_size ) ) ) &&
        (  NS(ManagedBuffer_get_num_objects)( out_buffer_begin, slot_size ) ==
           num_objects ) )
    {
        num_elem_t const stride = NS(Cuda_get_1d_thread_stride_in_kernel)();
        num_elem_t global_particle_id = NS(Cuda_get_1d_thread_id_in_kernel)();
        num_elem_t block_begin_index  = ( num_elem_t )0u;

        obj_const_iter_t in_begin = ( obj_const_iter_t
            )NS(ManagedBuffer_get_const_objects_index_begin(
                in_buffer_begin, slot_size );

        obj_const_iter_t in_end   = ( obj_const_iter_t
            )NS(ManagedBuffer_get_const_objects_index_end)(
                in_buffer_begin, slot_size );

        obj_iter_t out_begin = ( obj_iter_t
            )NS(ManagedBuffer_get_objects_index_begin)(
                out_buffer_begin, slot_size );

        obj_const_iter_t idx =
            NS(BufferIndex_get_index_object_by_global_index_from_range)(
                global_particle_id, 0u, in_begin, end, &block_begin_index );

        if( ( in_begin != SIXTRL_NULLPTR ) && ( in_end != SIXTRL_NULLPTR ) &&
            ( out_begin != SIXTRL_NULLPTR ) )
        {
            success_flag = 0;
        }

        while( ( idx != SIXTRL_NULLPTR ) && ( idx != end ) )
        {
            num_elem_t const block_offset = ( idx - in_begin );

            num_elem_t const local_particle_id =
                ( global_particle_id >= block_begin_index )
                    ? ( global_particle_id - block_begin_index ) : -1;

            ptr_const_particles_t in_particle = ( ptr_const_particles_t )(
                uintptr_t )NS(Object_get_begin_addr)( idx );

            ptr_particles_t out_particle = ( ptr_particle_t )( uintptr_t
                )NS(Object_get_begin_addr)( out_begin + block_offset );

            if( ( in_particle  != SIXTRL_NULLPTR ) &&
                ( out_particle != SIXTRL_NULLPTR ) &&
                ( in_particle  != out_particle   ) &&
                ( block_offset >= ( num_elem_t )0u ) &&
                ( local_particle_id >= ( num_elem_t )0u ) &&
                ( local_particle_id < NS(Particles_get_num_of_particles)(
                    in_particle ) ) &&
                ( local_particle_id < NS(Particles_get_num_of_particles)(
                    out_particle ) ) )
            {
                SIXTRL_ASSERT(
                    NS(Particles_get_num_of_particles)( in_particle ) ==
                    NS(Particles_get_num_of_particles)( out_particle ) );

                NS(Particles_copy_single)( out_particle, local_particle_id,
                                           in_particle, local_particle_id );
            }
            else
            {
                success_flag = -2;
                break;
            }

            global_particle_id += stride;

            idx = NS(BufferIndex_get_index_object_by_global_index_from_range)(
                global_particle_id, block_begin_index,
                    idx, end, &block_begin_index );
        }

        if( idx != end )
        {
            success_flag |= -4;
        }
    }
    else
    {
        if( NS(ManagedBuffer_needs_remapping(  in_buffer_begin, slot_size ) ) )
        {
            success_flag |= -8;
        }

        if( NS(ManagedBuffer_needs_remapping( out_buffer_begin, slot_size ) ) )
        {
            success_flag |= -8;
        }

        if(  NS(ManagedBuffer_get_num_objects)( out_buffer_begin, slot_size ) !=
             NS(ManagedBuffer_get_num_objects)(  in_buffer_begin, slot_size ) )
        {
            success_flag |= -16;
        }
    }

    if( ( success_flag != 0 ) && ( ptr_success_flag != SIXTRL_NULLPTR ) )
    {
        #if defined( __CUDA_ARCH__ ) && ( __CUDA_ARCH__ > 120 )
        atomicOr( ptr_success_flag, success_flag );
        #else /* defined( __CUDA_ARCH__ ) && ( __CUDA_ARCH__ > 120 ) */
        *ptr_success_flag |= success_flag;
        #endif /* defined( __CUDA_ARCH__ ) && ( __CUDA_ARCH__ > 120 ) */
    }

    return;
}

/* end: tests/sixtracklib/cuda/details/cuda_particles_kernel.cu */
