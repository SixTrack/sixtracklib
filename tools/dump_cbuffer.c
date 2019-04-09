#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "sixtracklib/testlib.h"
#include "sixtracklib/sixtracklib.h"

int main( int argc, char* argv[] )
{
    st_Buffer* buffer = SIXTRL_NULLPTR;

    if( argc < 2 )
    {
        printf( "Usage: %s PATH_TO_DUMP_FILE \r\n", argv[ 0 ] );
        return 0;
    }

    buffer = st_Buffer_new_from_file( argv[ 1 ] );

    if( buffer != SIXTRL_NULLPTR )
    {
        st_buffer_size_t ii = ( st_buffer_size_t )0u;

        st_buffer_size_t const num_objects =
            st_Buffer_get_num_of_objects( buffer );

        st_buffer_size_t const num_slots =
            st_Buffer_get_num_of_slots( buffer );

        st_buffer_size_t const num_dataptrs =
            st_Buffer_get_num_of_dataptrs( buffer );

        st_buffer_size_t const num_garbage =
            st_Buffer_get_num_of_garbage_ranges( buffer );

        st_buffer_size_t const buffer_size =
            st_Buffer_get_size( buffer );

        st_buffer_size_t const buffer_capacity =
            st_Buffer_get_capacity( buffer );

        st_buffer_addr_t const addr =
            st_Buffer_get_data_begin_addr( buffer );

        st_Object const* it  = st_Buffer_get_const_objects_begin( buffer );
        st_Object const* end = st_Buffer_get_const_objects_end( buffer );

        printf( "Contents of %s\r\n", argv[ 1 ] );
        printf( "  num_objects  = %16lu\r\n"
                "  num_slots    = %16lu\r\n"
                "  num_dataptrs = %16lu\r\n"
                "  num_garbage  = %16lu\r\n"
                "  buf size     = %16lu\r\n"
                "  buf capacity = %16lu\r\n"
                "  begin addr   = %16p\r\n\r\n",
                ( uint64_t )num_objects,  ( uint64_t )num_slots,
                ( uint64_t )num_dataptrs, ( uint64_t )num_garbage,
                ( uint64_t )buffer_size,  ( uint64_t )buffer_capacity,
                ( void* )( uintptr_t )addr );

        for( ; it != end ; ++it, ++ii )
        {
            printf( "Object %9lu / %9lu:\r\n",
                    ( uint64_t )ii, ( uint64_t )num_objects );

            st_Buffer_object_print_out_typeid( st_Object_get_type_id( it ) );
            printf( "\r\n" );

            st_Buffer_object_print_out( it );
            printf( "\r\n" );
        }
    }

    st_Buffer_delete( buffer );
    buffer = SIXTRL_NULLPTR;

    return 0;
}

/* end: tools/dump_cbuffer.c */
