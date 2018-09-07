#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/_impl/definitions.h"
    #include "sixtracklib/common/impl/managed_buffer_minimal.h"
    #include "sixtracklib/common/impl/managed_buffer_remap.h"
    #include "sixtracklib/testlib/generic_buffer_obj.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

__kernel void NS(remap_orig_buffer)(
    __global unsigned char* SIXTRL_RESTRICT orig_begin,
    SIXTRL_UINT64_T const orig_buffer_length,
    __global SIXTRL_INT64_T* SIXTRL_RESTRICT ptr_err_flag )
{
    size_t const global_id = get_global_id( 0 );
    size_t const gid_to_remap_buffer = ( size_t )0u;

    if( gid_to_remap_buffer == global_id )
    {
        long int error_flag = -1;
        NS(buffer_size_t) const slot_size = ( NS(buffer_size_t) )8u;
        int success = NS(ManagedBuffer_remap)( orig_begin, slot_size );

        if( ( success == 0 ) &&
            ( !NS(ManagedBuffer_needs_remapping)( orig_begin, slot_size ) ) )
        {
            error_flag = 0;
        }

        if( ptr_err_flag != SIXTRL_NULLPTR )
        {
            *ptr_err_flag = error_flag;
        }
    }

    return;
}


__kernel void NS(copy_orig_buffer)(
    __global unsigned char const* SIXTRL_RESTRICT orig_begin,
    __global unsigned char* SIXTRL_RESTRICT copy_begin,
    SIXTRL_UINT64_T const buffer_length,
    __global SIXTRL_INT64_T* SIXTRL_RESTRICT ptr_err_flag )
{
    size_t const global_id   = get_global_id( 0 );
    size_t const gid_to_copy = ( size_t )0;

    if( gid_to_copy == global_id )
    {
        typedef __global NS(Object) const* in_index_ptr_t;

        long int error_flag = -1;
        NS(buffer_size_t) const slot_size = ( NS(buffer_size_t) )8u;

        if( !NS(ManagedBuffer_needs_remapping)( copy_begin, slot_size ) )
        {
            in_index_ptr_t in_it = ( in_index_ptr_t )( uintptr_t
                )NS(ManagedBuffer_get_const_objects_index_begin)(
                    orig_begin, slot_size );

            in_index_ptr_t in_end = ( in_index_ptr_t )( uintptr_t
                )NS(ManagedBuffer_get_const_objects_index_end)(
                    orig_begin, slot_size );

            for( ; in_it != in_end ; ++in_it )
            {
                NS(Object) info = *in_it;
                __global NS(GenericObj)* ptr_obj = ( __global NS(GenericObj)* )(
                    uintptr_t )NS(Object_get_begin_ptr)( &info );

                printf( "type_id: %6d\r\n", NS(Object_get_type_id)( &info ) );
                printf( "size   : %6d\r\n", NS(Object_get_size)( &info ) );
                printf( "\r\n" );
                printf( "object_info: \r\n" );
                printf( "obj->type_id:   %8d\r\n", ( int )ptr_obj->type_id );
                printf( "obj->a      :   %8d\r\n", ( int )ptr_obj->a );
                printf( "obj->b      :   %8.2f\r\n", ( int )ptr_obj->b );
                printf( "obj->c      :  [ %8.2f , %8.2f , %8.2f , %8.2f  ]\r\n",
                        ptr_obj->c[ 0 ], ptr_obj->c[ 1 ],
                        ptr_obj->c[ 2 ], ptr_obj->c[ 3 ] );

                printf( "obj->num_d  :  %8d\r\n", ( int )ptr_obj->num_d );
                printf( "obj->num_e  :  %8d\r\n", ( int )ptr_obj->num_e );

                if( ptr_obj->num_d < 10 )
                {
                    int ii = 0;

                    printf( "obj->d      : [ " );

                    for( ; ii < ( int )ptr_obj->num_d ; ++ii )
                    {
                        printf( "%4d, ", ( int )ptr_obj->d[ ii ] );
                    }

                    printf( " ]\r\n" );
                }

                if( ptr_obj->num_e < 10 )
                {
                    int ii = 0;

                    printf( "obj->e      : [ " );

                    for( ; ii < ( int )ptr_obj->num_d ; ++ii )
                    {
                        printf( "%8.2f, ", ( int )ptr_obj->e[ ii ] );
                    }

                    printf( "] \r\n" );
                }

                printf( "\r\n" );

            }

            error_flag = 0;
        }

        /*
        error_flag |= ( 0 == NS(ManagedBuffer_remap)(
            copy_begin, slot_size ) ) ? 0 : -1; */

        if( ptr_err_flag != SIXTRL_NULLPTR )
        {
            *ptr_err_flag = error_flag;
        }
    }

    return;
}

/* end: tests/sixtracklib/opencl/test_buffer_generic_obj_kernel.cl */
