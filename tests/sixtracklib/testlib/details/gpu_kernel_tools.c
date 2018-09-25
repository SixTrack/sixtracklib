#include "sixtracklib/testlib/gpu_kernel_tools.h"

#include <assert.h>
#include <ctype.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "sixtracklib/_impl/definitions.h"

extern SIXTRL_SIZE_T NS(File_get_size)(
    const char *const path_to_file );

extern char* NS(File_read_into_string)(
    const char *const path_to_file, char* buffer_begin,
        SIXTRL_SIZE_T const max_num_chars );

extern bool NS(File_exists)(  const char *const path_to_file );

/* ------------------------------------------------------------------------- */

extern char** NS(GpuKernel_create_file_list)(
    char* SIXTRL_RESTRICT filenames,
    SIXTRL_SIZE_T* SIXTRL_RESTRICT ptr_num_of_files,
    char const* SIXTRL_RESTRICT prefix,
    char const* SIXTRL_RESTRICT separator );

extern void NS(GpuKernel_free_file_list)(
    char** SIXTRL_RESTRICT paths_to_kernel_files,
    SIXTRL_SIZE_T const num_of_files );

extern char* NS(GpuKernel_collect_source_string)(
    char**  file_list, SIXTRL_SIZE_T const num_of_files,
    SIXTRL_SIZE_T const max_line_length,
    SIXTRL_SIZE_T* SIXTRL_RESTRICT lines_offset );

/* ------------------------------------------------------------------------- */

bool NS(File_exists)( const char *const path_to_file )
{
    bool exists = false;

    if( path_to_file != 0 )
    {
        FILE* fp = fopen( path_to_file, "rb" );

        if( fp != 0 )
        {
            exists = true;
            fclose( fp );
            fp = 0;
        }
    }

    return exists;
}

/* ------------------------------------------------------------------------- */

SIXTRL_SIZE_T NS(File_get_size)( const char *const path_to_file )
{
    SIXTRL_SIZE_T num_of_bytes = ( SIXTRL_SIZE_T )0u;

    if( path_to_file != 0 )
    {
        FILE* fp = fopen( path_to_file, "rb" );

        if( fp != 0 )
        {
            long length = ( long )0u;

            fseek( fp, 0, SEEK_END );
            length = ftell( fp );
            fclose( fp );
            fp = 0;

            if( length > 0 )
            {
                num_of_bytes = ( SIXTRL_SIZE_T )length;
            }
        }
    }

    return num_of_bytes;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

char* NS(File_read_into_string)(
    const char *const path_to_file, char* buffer_begin,
    SIXTRL_SIZE_T const max_num_chars )
{
    char* write_pos = 0;

    if( ( path_to_file != 0 ) && ( buffer_begin != 0 ) )
    {
        FILE* fp = fopen( path_to_file, "rb" );

        if( fp != 0 )
        {
            long length = ( long )0u;

            fseek( fp, 0, SEEK_END );
            length = ftell( fp );
            fseek( fp, 0, SEEK_SET );

            if( length > 0 )
            {
                SIXTRL_SIZE_T const required_size = ( SIXTRL_SIZE_T )length;

                if( ( required_size > ( SIXTRL_SIZE_T )0u ) &&
                    ( required_size < max_num_chars ) )
                {
                    SIXTRL_SIZE_T bytes_to_read = required_size;

                    write_pos = buffer_begin;

                    while( bytes_to_read > ( SIXTRL_SIZE_T )0u )
                    {
                        SIXTRL_SIZE_T const bytes_read_now =
                            fread( write_pos, 1, bytes_to_read, fp );

                        if( bytes_to_read == bytes_read_now )
                        {
                            write_pos  = write_pos + bytes_to_read;
                            *write_pos = '\0';

                            bytes_to_read = ( SIXTRL_SIZE_T )0u;
                        }
                        else if( bytes_read_now < bytes_to_read )
                        {
                            write_pos = write_pos + bytes_read_now;
                            bytes_to_read -= bytes_read_now;
                        }
                        else
                        {
                            write_pos = 0;
                            break;
                        }
                    }
                }
            }

            fclose( fp );
            fp = 0;
        }
    }

    return write_pos;
}

/* ------------------------------------------------------------------------- */

char** NS(GpuKernel_create_file_list)(
    char* SIXTRL_RESTRICT filenames,
    SIXTRL_SIZE_T* SIXTRL_RESTRICT ptr_num_of_files,
    char const* SIXTRL_RESTRICT prefix, char const* SIXTRL_RESTRICT separator )
{
    char** ptr_list = 0;

    static SIXTRL_SIZE_T const ZERO_SIZE = ( SIXTRL_SIZE_T )0u;
    SIXTRL_SIZE_T filenames_strlen = ( SIXTRL_SIZE_T )0u;
    SIXTRL_SIZE_T num_of_files     = ZERO_SIZE;

    bool success = false;

    if( ( filenames != 0 ) && ( ptr_num_of_files != 0 ) &&
        ( ( filenames_strlen = strlen( filenames ) ) > ZERO_SIZE ) )
    {
        SIXTRL_SIZE_T prefix_length    = ZERO_SIZE;
        SIXTRL_SIZE_T separator_length = ZERO_SIZE;

        char* filenames_begin = filenames;
        char* filenames_end   = filenames;
        char* end_pos         = filenames + filenames_strlen;

        char default_separator[] = ",";
        char default_prefix[] = "";

        if( prefix    == 0 ) prefix = &default_prefix[ 0 ];
        if( separator == 0 ) separator = &default_separator[ 0 ];

        prefix_length    = strlen( prefix );
        separator_length = strlen( separator );

        assert( strlen( separator ) > ZERO_SIZE );

        *ptr_num_of_files = ZERO_SIZE;

        while( ( filenames_begin != 0 ) &&
               ( ( end_pos - filenames_begin ) > 0 ) &&
               ( strlen( filenames_begin ) > ZERO_SIZE ) )
        {
            filenames_end = strstr( filenames_begin, separator );

            if( filenames_end != 0 )
            {
                SIXTRL_SIZE_T ii = ZERO_SIZE;

                for( ; ii < separator_length ; ++ii )
                {
                    *filenames_end = '\0';
                }

                filenames_begin = filenames_end + 1;
            }
            else
            {
                filenames_begin = 0;
            }

            ++num_of_files;
        }

        if( num_of_files > ZERO_SIZE )
        {
            SIXTRL_SIZE_T jj = ZERO_SIZE;
            SIXTRL_SIZE_T ii = ZERO_SIZE;
            filenames_begin = filenames;

            ptr_list = ( char** )malloc( sizeof( char* ) * num_of_files );
            success  = ( ptr_list != 0 );

            if( ptr_list != 0 )
            {
                for( ; ii < num_of_files ; ++ii )
                {
                    ptr_list[ ii ] = 0;
                }

                success = true;
            }

            for( ii = ZERO_SIZE ; ii < num_of_files ; ++ii )
            {
                char* sep_begin   = 0;
                char* last_chr_it = 0;
                char* next_filenames_begin = 0;
                char  saved_end   = '\0';

                SIXTRL_SIZE_T in_length  = ZERO_SIZE;;
                SIXTRL_SIZE_T out_length = ( SIXTRL_SIZE_T )1u + prefix_length;

                ptr_list[ ii ]  = 0;

                if( ( filenames_begin == 0 ) ||
                    ( ( end_pos - filenames_begin ) < 0 ) ||
                    ( strlen( filenames_begin ) > filenames_strlen ) )
                {
                    success = false;
                    break;
                }

                in_length       = strlen( filenames_begin );
                sep_begin       = filenames_begin + in_length;
                filenames_end   = sep_begin;
                next_filenames_begin = sep_begin + separator_length;

                while( isspace( filenames_begin[ 0 ] ) && ( in_length > ZERO_SIZE ) )
                {
                    ++filenames_begin;
                    --in_length;
                }

                if( in_length == ZERO_SIZE )
                {
                    filenames_begin = next_filenames_begin;
                    continue;
                }

                last_chr_it = filenames_begin + ( in_length - 1 );

                while( ( isspace( *last_chr_it ) ) && ( in_length > ZERO_SIZE ) &&
                    (   last_chr_it != filenames_begin ) &&
                    ( ( last_chr_it -  filenames_begin ) > 0 ) )
                {
                    --last_chr_it;
                    --in_length;
                }

                if( in_length == ZERO_SIZE )
                {
                    filenames_begin = next_filenames_begin;
                    continue;
                }

                assert( ( filenames_end - last_chr_it ) >= 0 );
                filenames_end  = last_chr_it + 1;
                saved_end      = *filenames_end;
                *filenames_end = '\0';

                out_length += in_length;
                ptr_list[ jj ] = ( char* )malloc( sizeof( char ) * out_length );

                if( ptr_list[ jj ] == 0 )
                {
                    success = false;
                    break;
                }

                memset( ptr_list[ jj ], ( int )'\0', sizeof( char ) * out_length );
                strncpy( ptr_list[ jj ], prefix, out_length );

                if( ( in_length > ZERO_SIZE ) &&
                    ( ( strlen( ptr_list[ jj ] ) + in_length ) < out_length ) )
                {
                    strcat( ptr_list[ jj ], filenames_begin );
                    ++jj;
                }
                else
                {
                    success = false;
                    break;
                }

                memcpy( sep_begin, separator, sizeof( char ) * separator_length );
                *filenames_end  = saved_end;
                filenames_begin = next_filenames_begin;
            }

            if( ( success ) && ( jj <= num_of_files ) )
            {
                if( jj < num_of_files ) num_of_files = jj;
            }
            else if( ( jj == ZERO_SIZE ) || ( num_of_files < jj ) )
            {
                success = false;
            }
        }
    }

    if( success )
    {
        *ptr_num_of_files = num_of_files;
    }
    else
    {
        if( ptr_list != 0 )
        {
            SIXTRL_SIZE_T ii = ZERO_SIZE;

            for( ; ii < num_of_files ; ++ii )
            {
                free( ptr_list[ ii ] );
                ptr_list[ ii ] = 0;
            }

            free( ptr_list );
            ptr_list = 0;
        }

        if( ptr_num_of_files != 0 )
        {
            *ptr_num_of_files = ZERO_SIZE;
        }
    }

    return ptr_list;
}

void NS(GpuKernel_free_file_list)(
    char** SIXTRL_RESTRICT paths_to_kernel_files,
    SIXTRL_SIZE_T const num_of_files )
{
    static SIXTRL_SIZE_T const ZERO_SIZE = ( SIXTRL_SIZE_T )0u;

    if( ( paths_to_kernel_files != 0 ) && ( num_of_files > ZERO_SIZE ) )
    {
        SIXTRL_SIZE_T ii = ZERO_SIZE;

        for( ; ii < num_of_files ; ++ii )
        {
            free( paths_to_kernel_files[ ii ] );
            paths_to_kernel_files[ ii ] = 0;
        }

        free( paths_to_kernel_files );
        paths_to_kernel_files = 0;
    }

    return;
}

char* NS(GpuKernel_collect_source_string)(
    char**  file_list, SIXTRL_SIZE_T const num_of_files,
    SIXTRL_SIZE_T const max_line_length,
    SIXTRL_SIZE_T* SIXTRL_RESTRICT lines_offset )
{
    char* source = 0;
    bool success = false;

    static SIXTRL_SIZE_T const ZERO_SIZE   = ( SIXTRL_SIZE_T )0u;

    if( ( file_list != 0 ) && ( num_of_files > ZERO_SIZE ) )
    {
        SIXTRL_SIZE_T source_num_bytes = sizeof( char );
        SIXTRL_SIZE_T ii = ZERO_SIZE;

        for( ; ii < num_of_files ; ++ii )
        {
            char const* path_to_file = file_list[ ii ];
            SIXTRL_SIZE_T const file_size = NS(File_get_size)( path_to_file );

            if( path_to_file == 0 )
            {
                success = false;
                break;
            }

            source_num_bytes += file_size;
        }

        source  = ( char* )malloc( source_num_bytes );
        success = ( ( source_num_bytes > ZERO_SIZE ) && ( source != 0 ) );

        if( success )
        {
            memset( source, ( int )'\0', source_num_bytes );

            if( ( success ) && ( lines_offset != 0 ) &&
                ( max_line_length > ZERO_SIZE ) )
            {
                SIXTRL_SIZE_T line_cnt           = ZERO_SIZE;
                SIXTRL_SIZE_T ii                 = ZERO_SIZE;
                SIXTRL_SIZE_T num_bytes_to_write = source_num_bytes;

                SIXTRL_SIZE_T const line_buffer_bytes =
                    sizeof( char ) * ( max_line_length + 2 );

                char* write_pos = source;

                char* line_buffer = ( char* )malloc( line_buffer_bytes );
                if( line_buffer == 0 ) return source;

                memset( line_buffer, ( int )'\0', line_buffer_bytes );

                for( ; ii < num_of_files ; ++ii )
                {
                    FILE* fp = fopen( file_list[ ii ], "rb" );
                    SIXTRL_SIZE_T lines_in_current_file = ZERO_SIZE;

                    if( fp == 0 )
                    {
                        success = false;
                        break;
                    }

                    lines_offset[ ii ] = ZERO_SIZE;

                    while( 0 != fgets( line_buffer, max_line_length, fp ) )
                    {
                        SIXTRL_SIZE_T const line_length = strlen( line_buffer );

                        if( ( ( line_length + 1 ) >= max_line_length ) ||
                            ( line_length >  num_bytes_to_write ) )
                        {
                            success = false;
                            break;
                        }

                        strncpy( write_pos, line_buffer, num_bytes_to_write );
                        num_bytes_to_write   -= line_length;
                        write_pos = write_pos + line_length;

                        ++lines_in_current_file;
                    }

                    if( success )
                    {
                        lines_offset[ ii ] = line_cnt;
                        line_cnt += lines_in_current_file;
                    }
                    else
                    {
                        break;
                    }
                }

                free( line_buffer );
                line_buffer = 0;
            }
            else
            {
                char* end_pos   = 0;
                char* write_pos = source;

                SIXTRL_SIZE_T num_bytes_to_write = source_num_bytes;

                end_pos = source + source_num_bytes;

                for( ii = ZERO_SIZE ; ii < num_of_files ; ++ii )
                {
                    SIXTRL_SIZE_T num_bytes_written = ZERO_SIZE;

                    char* next_write_pos = NS(File_read_into_string)(
                        file_list[ ii ], write_pos, num_bytes_to_write );

                    if( ( next_write_pos == 0 ) ||
                        ( ( next_write_pos - write_pos ) < 0 ) ||
                        ( ( end_pos - next_write_pos   ) < 0 ) )
                    {
                        success = false;
                        break;
                    }

                    num_bytes_written = strlen( write_pos );

                    if( num_bytes_written > num_bytes_to_write )
                    {
                        success = false;
                        break;
                    }

                    num_bytes_to_write -= num_bytes_written;
                    write_pos = next_write_pos;
                }
            }
        }
    }

    if( !success )
    {
        free( source );
        source = 0;
    }

    return source;
}

/* end: tests/sixtracklib/testlib/details/gpu_kernel_tools.c */
