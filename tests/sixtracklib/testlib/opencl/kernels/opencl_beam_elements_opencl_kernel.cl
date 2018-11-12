#ifndef SIXTRL_TESTS_SIXTRACKLIB_OPENCL_TEST_TEST_BEAM_ELEMENTS_KERNEL_CL__
#define SIXTRL_TESTS_SIXTRACKLIB_OPENCL_TEST_TEST_BEAM_ELEMENTS_KERNEL_CL__

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/opencl/internal/default_compile_options.h"

    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/buffer/managed_buffer_minimal.h"
    #include "sixtracklib/common/buffer/managed_buffer_remap.h"
    #include "sixtracklib/common/beam_elements.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#pragma OPENCL_EXTENSION cl_khr_int32_extended_atomics

__kernel void NS(BeamElements_copy_beam_elements_opencl)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT in_buffer_begin,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT out_buffer_begin,
    SIXTRL_DATAPTR_DEC SIXTRL_INT32_T* SIXTRL_RESTRICT ptr_success_flag )
{
    typedef NS(buffer_size_t)                               buf_size_t;
    typedef NS(object_type_id_t)                            type_id_t;
    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)*        obj_iter_t;
    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const*  obj_const_iter_t;

    buf_size_t const  slot_size = ( buf_size_t )8u;
    SIXTRL_INT32_T success_flag = ( SIXTRL_INT32_T )-1;

    if( ( !NS(ManagedBuffer_needs_remapping)(  in_buffer_begin, slot_size ) ) &&
        ( !NS(ManagedBuffer_needs_remapping)( out_buffer_begin, slot_size ) ) &&
        (  NS(ManagedBuffer_get_num_objects)( out_buffer_begin, slot_size ) ==
           NS(ManagedBuffer_get_num_objects)(  in_buffer_begin, slot_size ) ) )
    {
        size_t beam_element_id = get_global_id( 0 );
        size_t const stride    = get_global_size( 0 );

        size_t const num_beam_elements =
            NS(ManagedBuffer_get_num_objects)( in_buffer_begin, slot_size );

        success_flag = ( SIXTRL_INT32_T )0u;

        for( ; beam_element_id < num_beam_elements ; beam_element_id += stride )
        {
            obj_const_iter_t in_obj_it  =
                NS(ManagedBuffer_get_const_objects_index_begin)(
                    in_buffer_begin, slot_size );

            obj_iter_t out_obj_it = NS(ManagedBuffer_get_objects_index_begin)(
                out_buffer_begin, slot_size );

            type_id_t const type_id = NS(Object_get_type_id)( in_obj_it );

            in_obj_it  = in_obj_it  + beam_element_id;
            out_obj_it = out_obj_it + beam_element_id;

            success_flag |= NS(BeamElements_copy_object)( out_obj_it, in_obj_it );
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

    if( ( success_flag != ( SIXTRL_INT32_T )0u ) &&
        ( ptr_success_flag != SIXTRL_NULLPTR ) )
    {
        atomic_or( ptr_success_flag, success_flag );
    }

    return;
}

#endif /* SIXTRL_TESTS_SIXTRACKLIB_OPENCL_TEST_TEST_BEAM_ELEMENTS_KERNEL_CL__ */

/* end: tests/sixtracklib/opencl/test_beam_elements_opencl_kernel.cl */
