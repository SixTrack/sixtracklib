#ifndef SIXTRL_TESTS_SIXTRACKLIB_OPENCL_TEST_BUFFER_GENERIC_OBJ_KERNEL_CL__
#define SIXTRL_TESTS_SIXTRACKLIB_OPENCL_TEST_BUFFER_GENERIC_OBJ_KERNEL_CL__

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/opencl/internal/default_compile_options.h"

    #include "sixtracklib/_impl/definitions.h"
    #include "sixtracklib/common/impl/managed_buffer_minimal.h"
    #include "sixtracklib/common/impl/managed_buffer_remap.h"
    #include "sixtracklib/testlib/generic_buffer_obj.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

__kernel void NS(remap_orig_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT orig_begin,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT copy_begin,
    __global SIXTRL_INT64_T* SIXTRL_RESTRICT ptr_err_flag )
{
    typedef NS(buffer_size_t) buf_size_t;

    buf_size_t const global_id           = get_global_id( 0 );
    buf_size_t const gid_to_remap_buffer = ( buf_size_t )0u;

    if( gid_to_remap_buffer == global_id )
    {
        long int error_flag = 0;

        NS(buffer_size_t) const slot_size = ( NS(buffer_size_t) )8u;
        int success = NS(ManagedBuffer_remap)( orig_begin, slot_size );

        if( ( success != 0 ) ||
            ( NS(ManagedBuffer_needs_remapping)( orig_begin, slot_size ) ) )
        {
            error_flag |= -1;
        }

        success = NS(ManagedBuffer_remap)( copy_begin, slot_size );

        if( ( success != 0 ) ||
            ( NS(ManagedBuffer_needs_remapping)( copy_begin, slot_size ) ) )
        {
            error_flag |= -2;
        }

        if( ptr_err_flag != SIXTRL_NULLPTR )
        {
            *ptr_err_flag = error_flag;
        }
    }

    return;
}


__kernel void NS(copy_orig_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT orig_begin,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char*  SIXTRL_RESTRICT copy_begin,
    __global SIXTRL_INT64_T* SIXTRL_RESTRICT ptr_err_flag )
{
    size_t const global_id   = get_global_id( 0 );
    size_t const gid_to_copy = ( size_t )0;

    if( gid_to_copy == global_id )
    {
        typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const*      in_index_ptr_t;
        typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)*            out_index_ptr_t;

        typedef SIXTRL_BUFFER_OBJ_DATAPTR_DEC NS(GenericObj) const* in_obj_ptr_t;
        typedef SIXTRL_BUFFER_OBJ_DATAPTR_DEC NS(GenericObj)*       out_obj_ptr_t;

        long int error_flag = 0;
        NS(buffer_size_t) const slot_size = ( NS(buffer_size_t) )8u;

        if( ( !NS(ManagedBuffer_needs_remapping)( orig_begin, slot_size ) ) &&
            ( !NS(ManagedBuffer_needs_remapping)( copy_begin, slot_size ) ) )
        {
            unsigned int obj_index = ( unsigned int )0u;

            in_index_ptr_t in_it = ( in_index_ptr_t )( uintptr_t
                )NS(ManagedBuffer_get_const_objects_index_begin)(
                    orig_begin, slot_size );

            in_index_ptr_t in_end = ( in_index_ptr_t )( uintptr_t
                )NS(ManagedBuffer_get_const_objects_index_end)(
                    orig_begin, slot_size );

            out_index_ptr_t out_it = ( out_index_ptr_t )( uintptr_t
               )NS(ManagedBuffer_get_objects_index_begin)( copy_begin, slot_size );

            for( ; in_it != in_end ; ++in_it, ++out_it, ++obj_index )
            {
                in_obj_ptr_t in_obj = ( in_obj_ptr_t )( uintptr_t
                    )NS(Object_get_const_begin_ptr)( in_it );

                out_obj_ptr_t out_obj = ( out_obj_ptr_t )( uintptr_t
                    )NS(Object_get_begin_ptr)( out_it );

                if( ( out_obj != SIXTRL_NULLPTR ) &&
                    ( in_obj  != SIXTRL_NULLPTR ) &&
                    ( out_obj != in_obj ) &&
                    ( out_obj->type_id == in_obj->type_id ) &&
                    ( out_obj->num_d   == in_obj->num_d   ) &&
                    ( out_obj->num_e   == in_obj->num_e   ) &&
                    ( out_obj->d != in_obj->d ) &&
                    ( out_obj->d != SIXTRL_NULLPTR ) &&
                    ( in_obj->d  != SIXTRL_NULLPTR ) &&
                    ( out_obj->e != SIXTRL_NULLPTR ) &&
                    ( in_obj->e  != SIXTRL_NULLPTR ) )
                {
                    out_obj->a = in_obj->a;
                    out_obj->b = in_obj->b;

                    SIXTRACKLIB_COPY_VALUES( SIXTRL_REAL_T,
                         &out_obj->c[ 0 ], &in_obj->c[ 0 ], ( size_t )4u );

                    SIXTRACKLIB_COPY_VALUES( SIXTRL_UINT8_T,
                         out_obj->d, in_obj->d, in_obj->num_d );

                    SIXTRACKLIB_COPY_VALUES( SIXTRL_REAL_T,
                         out_obj->e, in_obj->e, in_obj->num_e );
                }
                else
                {
                    printf( "missmatch between %u th input/output object\r\n",
                            obj_index );

                    error_flag |= -( int )( obj_index << 2 );
                }
            }
        }
        else
        {
            error_flag |= -1;
        }

        if( ptr_err_flag != SIXTRL_NULLPTR )
        {
            *ptr_err_flag = error_flag;
        }
    }

    return;
}

#endif /* SIXTRL_TESTS_SIXTRACKLIB_OPENCL_TEST_BUFFER_GENERIC_OBJ_KERNEL_CL__ */

/* end: tests/sixtracklib/opencl/test_buffer_generic_obj_kernel.cl */
