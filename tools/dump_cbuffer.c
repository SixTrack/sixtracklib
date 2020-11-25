#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "sixtracklib/testlib.h"
#include "sixtracklib/sixtracklib.h"

int main( int argc, char* argv[] )
{
    NS(Buffer)* buffer = SIXTRL_NULLPTR;

    if( argc < 2 )
    {
        printf( "Usage: %s PATH_TO_DUMP_FILE \r\n", argv[ 0 ] );
        return 0;
    }

    buffer = NS(Buffer_new_from_file)( argv[ 1 ] );

    if( buffer != SIXTRL_NULLPTR )
    {
        NS(buffer_size_t) ii = ( NS(buffer_size_t) )0u;

        NS(buffer_size_t) const num_objects =
            NS(Buffer_get_num_of_objects)( buffer );

        NS(buffer_size_t) const num_slots =
            NS(Buffer_get_num_of_slots)( buffer );

        NS(buffer_size_t) const num_dataptrs =
            NS(Buffer_get_num_of_dataptrs)( buffer );

        NS(buffer_size_t) const num_garbage =
            NS(Buffer_get_num_of_garbage_ranges)( buffer );

        NS(buffer_size_t) const buffer_size =
            NS(Buffer_get_size)( buffer );

        NS(buffer_size_t) const buffer_capacity =
            NS(Buffer_get_capacity)( buffer );

        NS(buffer_addr_t) const addr =
            NS(Buffer_get_data_begin_addr)( buffer );

        NS(Object) const* it  = NS(Buffer_get_const_objects_begin)( buffer );
        NS(Object) const* end = NS(Buffer_get_const_objects_end)( buffer );

        printf( "Contents of %s\r\n", argv[ 1 ] );
        printf( "  num_objects  = %16lu\r\n"
                "  num_slots    = %16lu\r\n"
                "  num_dataptrs = %16lu\r\n"
                "  num_garbage  = %16lu\r\n"
                "  buf size     = %16lu\r\n"
                "  buf capacity = %16lu\r\n"
                "  begin addr   = %16p\r\n\r\n",
                ( unsigned long )num_objects,  ( unsigned long )num_slots,
                ( unsigned long )num_dataptrs, ( unsigned long )num_garbage,
                ( unsigned long )buffer_size,  ( unsigned long )buffer_capacity,
                ( void* )( uintptr_t )addr );

        for( ; it != end ; ++it, ++ii )
        {
            printf( "Object %9lu / %9lu:\r\n",
                    ( unsigned long )ii, ( unsigned long )num_objects );

            NS(Buffer_object_print_out_typeid)( NS(Object_get_type_id)( it ) );
            printf( "\r\n" );

            NS(Buffer_object_print_out)( it );
            printf( "\r\n" );
        }
    }

    NS(Buffer_delete)( buffer );
    buffer = SIXTRL_NULLPTR;

    return 0;
}

/* end: tools/dump_cbuffer.c */
