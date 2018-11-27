#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "sixtracklib/testlib.h"
#include "sixtracklib/sixtracklib.h"

int main( int const argc, char* argv[] )
{
    st_Buffer* buffer = SIXTRL_NULLPTR;
    st_buffer_addr_t target_base_addr = ( st_buffer_addr_t )0x1000;
    char path_to_output_file[ 1024 ];

    memset( path_to_output_file, ( int )'\0', 1024 );

    if( argc < 2 )
    {
        printf( "Usage: %s PATH_TO_DUMP_FILE [BASE_ADDR=%lu] "
                "[PATH_TO_OUTPUT_FILE=PATH_TO_DUMP_FILE]\r\n",
                argv[ 0 ], target_base_addr );
    }

    if( argc == 2 )
    {
        buffer = st_Buffer_new_from_file( argv[ 1 ] );
    }

    if( argc >= 3 )
    {
        int64_t const temp = atoi( argv[ 2 ] );
        if( temp > 0 ) target_base_addr = ( st_buffer_addr_t )temp;
    }

    if( argc >= 4 )
    {
        strncpy( path_to_output_file, argv[ 3 ], 1023 );
    }
    else
    {
        strncpy( path_to_output_file, argv[ 1 ], 1023 );
    }

    if( st_Buffer_write_to_file_normalized_addr(
            buffer, path_to_output_file, target_base_addr ) )
    {
        printf( "Successfully normalized and written to %s\r\n",
                path_to_output_file );
    }
    else
    {
        printf( "Error -> stopping\r\n" );
    }

    st_Buffer_delete( buffer );

    return 0;
}

/* end: tools/normalize_cobject_dump.c */
