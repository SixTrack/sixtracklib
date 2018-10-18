#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/testlib/cuda/kernels/cuda_beam_elements_kernel.cuh"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <cuda_runtime.h>
    #include <cuda.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/internal/buffer_main_defines.h"
    #include "sixtracklib/common/buffer/buffer_type.h"
    #include "sixtracklib/common/buffer/managed_buffer_minimal.h"
    #include "sixtracklib/common/buffer/managed_buffer_remap.h"
    #include "sixtracklib/common/beam_elements.h"
    #include "sixtracklib/cuda/cuda_tools.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

__global__ void NS(BeamElements_copy_beam_elements_buffer_kernel_cuda)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT in_buffer_begin,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT out_buffer_begin,
    SIXTRL_DATAPTR_DEC int* SIXTRL_RESTRICT ptr_success_flag )
{
    typedef NS(buffer_size_t) buf_size_t;
    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)*        obj_iter_t;
    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const*  obj_const_iter_t;

    int32_t success_flag = ( int32_t )-1;

    buf_size_t const slot_size = SIXTRL_BUFFER_DEFAULT_SLOT_SIZE;

    buf_size_t const num_beam_elements =
        NS(ManagedBuffer_get_num_objects)( in_buffer_begin, slot_size );

    if( ( !NS(ManagedBuffer_needs_remapping)(  in_buffer_begin, slot_size ) ) &&
        ( !NS(ManagedBuffer_needs_remapping)( out_buffer_begin, slot_size ) ) &&
        (  NS(ManagedBuffer_get_num_objects)( out_buffer_begin, slot_size ) ==
           num_beam_elements ) )
    {
        obj_const_iter_t in_begin = NS(ManagedBuffer_get_const_objects_index_begin)(
            in_buffer_begin, slot_size );

        obj_iter_t out_begin = NS(ManagedBuffer_get_objects_index_begin)(
            out_buffer_begin, slot_size );

        buf_size_t const stride        = NS(Cuda_get_1d_thread_stride_in_kernel)();
        buf_size_t global_beam_elem_id = NS(Cuda_get_1d_thread_id_in_kernel)();

        if( ( in_begin  != SIXTRL_NULLPTR ) &&
            ( out_begin != SIXTRL_NULLPTR ) )
        {
            success_flag = 0;
        }

        while( global_beam_elem_id < num_beam_elements )
        {
            obj_const_iter_t in_obj =  in_begin + global_beam_elem_id;
            obj_iter_t      out_obj = out_begin + global_beam_elem_id;

            global_beam_elem_id += stride;
            success_flag |= NS(BeamElements_copy_object)( out_obj, in_obj );
        }
    }
    else
    {
        if( NS(ManagedBuffer_needs_remapping)(  in_buffer_begin, slot_size ) )
        {
            success_flag |= -4;
        }

        if( NS(ManagedBuffer_needs_remapping)( out_buffer_begin, slot_size ) )
        {
            success_flag |= -8;
        }

        if( NS(ManagedBuffer_get_num_objects)( out_buffer_begin, slot_size ) !=
            NS(ManagedBuffer_get_num_objects)( in_buffer_begin,  slot_size ) )
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

/* end: tests/sixtracklib/testlib/cuda/kernels/cuda_beam_elements_kernel.cu */
