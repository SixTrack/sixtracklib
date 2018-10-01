#ifndef SIXTRL_TESTS_SIXTRACKLIB_OPENCL_TEST_TEST_BEAM_ELEMENTS_KERNEL_CL__
#define SIXTRL_TESTS_SIXTRACKLIB_OPENCL_TEST_TEST_BEAM_ELEMENTS_KERNEL_CL__

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/opencl/internal/default_compile_options.h"

    #include "sixtracklib/_impl/definitions.h"
    #include "sixtracklib/common/impl/managed_buffer_minimal.h"
    #include "sixtracklib/common/impl/managed_buffer_remap.h"
    #include "sixtracklib/common/beam_elements.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#pragma OPENCL_EXTENSION cl_khr_int32_extended_atomics

__kernel void NS(BeamElements_copy_beam_elements_opencl)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT in_buffer_begin,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT out_buffer_begin,
    SIXTRL_DATAPTR_DEC int* SIXTRL_RESTRICT ptr_success_flag )
{
    typedef NS(buffer_size_t)                               buf_size_t;
    typedef NS(object_type_id_t)                            type_id_t;
    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)*        obj_iter_t;
    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const*  obj_const_iter_t;

    buf_size_t const slot_size = ( buf_size_t )8u;
    int success_flag = ( int )0u;

    if( ( !NS(ManagedBuffer_needs_remapping)(  in_buffer_begin, slot_size ) ) &&
        ( !NS(ManagedBuffer_needs_remapping)( out_buffer_begin, slot_size ) ) &&
        (  NS(ManagedBuffer_get_num_objects)( out_buffer_begin, slot_size ) ==
           NS(ManagedBuffer_get_num_objects)(  in_buffer_begin, slot_size ) ) )
    {
        size_t const global_id = get_global_id( 0 );

        if( global_id <
            NS(ManagedBuffer_get_num_objects)( in_buffer_begin, slot_size ) )
        {
            obj_const_iter_t in_obj_it  =
                NS(ManagedBuffer_get_const_objects_index_begin)(
                    in_buffer_begin, slot_size );

            obj_iter_t out_obj_it = NS(ManagedBuffer_get_objects_index_begin)(
                out_buffer_begin, slot_size );

            type_id_t const type_id = NS(Object_get_type_id)( in_obj_it );

            in_obj_it  = in_obj_it  + global_id;
            out_obj_it = out_obj_it + global_id;

            if( type_id == NS(Object_get_type_id)( out_obj_it ) )
            {
                switch( type_id )
                {
                    case NS(OBJECT_TYPE_DRIFT):
                    {
                        typedef NS(Drift) belem_t;
                        typedef SIXTRL_BE_ARGPTR_DEC belem_t*       ptr_belem_t;
                        typedef SIXTRL_BE_ARGPTR_DEC belem_t const* ptr_const_belem_t;

                        ptr_const_belem_t in_belem = ( ptr_const_belem_t )(
                            uintptr_t )NS(Object_get_begin_addr)( in_obj_it );

                        ptr_belem_t out_belem = ( ptr_belem_t )(
                            uintptr_t )NS(Object_get_begin_addr)( out_obj_it );

                        if( ( in_belem  != SIXTRL_NULLPTR ) &&
                            ( out_belem != SIXTRL_NULLPTR ) )
                        {
                            NS(Drift_copy)( out_belem, in_belem );
                        }
                        else
                        {
                            success_flag |= -1;
                        }

                        break;
                    }

                    case NS(OBJECT_TYPE_DRIFT_EXACT):
                    {
                        typedef NS(DriftExact) belem_t;
                        typedef SIXTRL_BE_ARGPTR_DEC belem_t*       ptr_belem_t;
                        typedef SIXTRL_BE_ARGPTR_DEC belem_t const* ptr_const_belem_t;

                        ptr_const_belem_t in_belem = ( ptr_const_belem_t )(
                            uintptr_t )NS(Object_get_begin_addr)( in_obj_it );

                        ptr_belem_t out_belem = ( ptr_belem_t )(
                            uintptr_t )NS(Object_get_begin_addr)( out_obj_it );

                        if( ( in_belem  != SIXTRL_NULLPTR ) &&
                            ( out_belem != SIXTRL_NULLPTR ) )
                        {
                            NS(DriftExact_copy)( out_belem, in_belem );
                        }
                        else
                        {
                            success_flag |= -1;
                        }

                        break;
                    }

                    default:
                    {
                        success_flag |= -1;
                    }
                };
            }
            else
            {
                success_flag |= -2;
            }
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

    if( ( success_flag     != ( int )0u ) &&
        ( ptr_success_flag != SIXTRL_NULLPTR ) )
    {
        atomic_or( ptr_success_flag, success_flag );
    }

    return;
}

#endif /* SIXTRL_TESTS_SIXTRACKLIB_OPENCL_TEST_TEST_BEAM_ELEMENTS_KERNEL_CL__ */

/* end: tests/sixtracklib/opencl/test_beam_elements_opencl_kernel.cl */
