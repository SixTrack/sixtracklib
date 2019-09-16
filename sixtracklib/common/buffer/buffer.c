#include "sixtracklib/common/buffer.h"

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/buffer/mem_pool.h"

#include "sixtracklib/common/buffer/buffer_generic.h"
#include "sixtracklib/common/buffer/managed_buffer_minimal.h"
#include "sixtracklib/common/buffer/managed_buffer_remap.h"
#include "sixtracklib/common/buffer/managed_buffer.h"
#include "sixtracklib/common/buffer/buffer_object.h"
#include "sixtracklib/common/buffer/buffer_type.h"

#if defined( SIXTRACKLIB_ENABLE_MODULE_OPENCL ) && \
        ( SIXTRACKLIB_ENABLE_MODULE_OPENCL == 1 )
#endif /* OpenCL */

#if defined( SIXTRACKLIB_ENABLE_MODULE_CUDA ) && \
        ( SIXTRACKLIB_ENABLE_MODULE_CUDA == 1 )
#endif /* Cuda */

#if !defined( _GPUCODE )

SIXTRL_HOST_FN SIXTRL_STATIC SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)*
NS(Buffer_allocate_generic)(
    NS(buffer_size_t) const buffer_capacity,
    NS(buffer_flags_t) const buffer_flags );

/* ------------------------------------------------------------------------- */

SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* NS(Buffer_allocate_generic)(
    NS(buffer_size_t)  const buffer_capacity,
    NS(buffer_flags_t) const buffer_flags )
{
    typedef NS(Buffer)              buffer_t;
    typedef NS(buffer_size_t)       buf_size_t;
    typedef NS(buffer_addr_t)       address_t;
    typedef unsigned char           raw_t;

    int success = -1;

    buffer_t* ptr_buffer = NS(Buffer_preset)(
        ( buffer_t* )malloc( sizeof( buffer_t ) ) );

    buf_size_t const slot_size = NS(Buffer_get_slot_size)( ptr_buffer );

    buf_size_t const addr_size = NS(ManagedBuffer_get_slot_based_length)(
        sizeof( address_t ), slot_size );

    buf_size_t const section_hd_size = ( buf_size_t )2u * addr_size;

    buf_size_t const min_required_capacity =
        NS(BUFFER_DEFAULT_HEADER_SIZE) + ( buf_size_t )4u * section_hd_size;

    buf_size_t const requ_capacity =
        ( min_required_capacity <= buffer_capacity )
            ? buffer_capacity : min_required_capacity;

    if( ptr_buffer != SIXTRL_NULLPTR )
    {
        if( ( buffer_flags & NS(BUFFER_DATASTORE_MEMPOOL) ) ==
            NS(BUFFER_DATASTORE_MEMPOOL) )
        {
            typedef NS(MemPool)     mem_pool_t;
            typedef mem_pool_t*     ptr_to_mem_pool_t;

            ptr_to_mem_pool_t ptr_mem_pool = NS(MemPool_preset)(
                ( mem_pool_t* )malloc( sizeof( mem_pool_t ) ) );

            NS(MemPool_set_chunk_size)( ptr_mem_pool, slot_size );

            if( NS(MemPool_reserve_aligned)( ptr_mem_pool,
                    requ_capacity, slot_size ) )
            {
                NS(AllocResult) result = NS(MemPool_append_aligned)(
                    ptr_mem_pool, requ_capacity, slot_size );

                if( NS(AllocResult_valid)( &result ) )
                {
                    raw_t const z = ( raw_t )0u;

                    SIXTRACKLIB_SET_VALUES( raw_t,
                        NS(AllocResult_get_pointer)( &result ),
                        NS(AllocResult_get_length)(  &result ), z );

                    success = NS(Buffer_init_on_flat_memory)( ptr_buffer,
                        NS(AllocResult_get_pointer)( &result ),
                        NS(AllocResult_get_length)(  &result ) );

                    if( success == 0 )
                    {
                        ptr_buffer->datastore_addr =
                            ( address_t )( uintptr_t )ptr_mem_pool;

                        ptr_buffer->datastore_flags = buffer_flags |
                            NS(BUFFER_USES_DATASTORE) |
                            NS(BUFFER_OWNS_DATASTORE) |
                            NS(BUFFER_DATASTORE_MEMPOOL) |
                            NS(BUFFER_DATASTORE_ALLOW_APPENDS) |
                            NS(BUFFER_DATASTORE_ALLOW_CLEAR)   |
                            NS(BUFFER_DATASTORE_ALLOW_REMAPPING) |
                            NS(BUFFER_DATASTORE_ALLOW_RESIZE);
                    }
                }
            }
        }
    }

    if( success != 0 )
    {
        if( ptr_buffer != SIXTRL_NULLPTR )
        {
            if( ( buffer_flags & NS(BUFFER_DATASTORE_MEMPOOL) ) ==
                NS(BUFFER_DATASTORE_MEMPOOL) )
            {
                typedef NS(MemPool)     mem_pool_t;
                typedef mem_pool_t*     ptr_to_mem_pool_t;

                ptr_to_mem_pool_t ptr_mem_pool = ( ptr_to_mem_pool_t )(
                    uintptr_t )NS(Buffer_get_datastore_begin_addr)(
                        ptr_buffer );

                NS(MemPool_free)( ptr_mem_pool );
                free( ptr_mem_pool );

                ptr_buffer->datastore_addr = ( address_t )0u;
            }

            NS(Buffer_preset)( ptr_buffer );
            free( ptr_buffer );
            ptr_buffer = SIXTRL_NULLPTR;
        }
    }

    return ptr_buffer;
}

/* ------------------------------------------------------------------------- */

NS(buffer_size_t) NS(Buffer_get_slot_size_ext)(
   SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer )
{
    return NS(Buffer_get_slot_size)( buffer );
}

/* -------------------------------------------------------------------------- */

SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* NS(Buffer_preset_ext)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer )
{
    return NS(Buffer_preset)( buffer );
}

/* ------------------------------------------------------------------------- */

NS(buffer_addr_t) NS(Buffer_get_datastore_begin_addr_ext)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer )
{
    return NS(Buffer_get_datastore_begin_addr)( buffer );
}

/* ------------------------------------------------------------------------- */

NS(buffer_size_t) NS(Buffer_get_size_ext)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer )
{
    return NS(Buffer_get_size)( buffer );
}

 NS(buffer_size_t) NS(Buffer_get_capacity_ext)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer )
 {
     return NS(Buffer_get_capacity)( buffer );
 }

 NS(buffer_size_t) NS(Buffer_get_header_size_ext)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer )
 {
     return NS(Buffer_get_size)( buffer );
 }

NS(buffer_addr_t) NS(Buffer_get_data_begin_addr_ext)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer )
{
    return NS(Buffer_get_data_begin_addr)( buffer );
}

NS(buffer_addr_t) NS(Buffer_get_data_end_addr_ext)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer )
{
    return NS(Buffer_get_data_end_addr)( buffer );
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

NS(buffer_addr_t) NS(Buffer_get_objects_begin_addr_ext)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer )
{
    return NS(Buffer_get_objects_begin_addr)( buffer );
}

NS(buffer_addr_t) NS(Buffer_get_objects_end_addr_ext)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer )
{
    return NS(Buffer_get_objects_end_addr)( buffer );
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

NS(buffer_size_t) NS(Buffer_get_slots_size_ext)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer )
{
    return NS(Buffer_get_slot_size)( buffer );
}

NS(buffer_size_t) NS(Buffer_get_num_of_slots_ext)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer )
{
    return NS(Buffer_get_num_of_slots)( buffer );
}

NS(buffer_size_t) NS(Buffer_get_max_num_of_slots_ext)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer )
{
    return NS(Buffer_get_max_num_of_slots)( buffer );
}

NS(buffer_size_t) NS(Buffer_get_objects_size_ext)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buf );

NS(buffer_size_t) NS(Buffer_get_num_of_objects_ext)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer )
{
    return NS(Buffer_get_num_of_objects)( buffer );
}

NS(buffer_size_t) NS(Buffer_get_max_num_of_objects_ext)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer )
{
    return  NS(Buffer_get_max_num_of_objects)( buffer );
}

NS(buffer_size_t) NS(Buffer_get_dataptrs_size_ext)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer )
{
    return NS(Buffer_get_dataptrs_size)( buffer );
}

NS(buffer_size_t) NS(Buffer_get_num_of_dataptrs_ext)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer )
{
    return NS(Buffer_get_num_of_dataptrs)( buffer );
}

NS(buffer_size_t) NS(Buffer_get_max_num_of_dataptrs_ext)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer )
{
   return NS(Buffer_get_max_num_of_dataptrs)( buffer );
}

NS(buffer_size_t) NS(Buffer_get_garbage_size_ext)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer )
{
    return NS(Buffer_get_garbage_size)( buffer );
}

NS(buffer_size_t) NS(Buffer_get_num_of_garbage_ranges_ext)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer )
{
    return NS(Buffer_get_num_of_garbage_ranges)( buffer );
}

NS(buffer_size_t) NS(Buffer_get_max_num_of_garbage_ranges_ext)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer )
{
    return NS(Buffer_get_max_num_of_garbage_ranges)( buffer );
}

/* ------------------------------------------------------------------------- */

SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* NS(Buffer_new)(
    NS(buffer_size_t) const buffer_capacity )
{
    NS(Buffer)* ptr_buffer = NS(Buffer_allocate_generic)(
        buffer_capacity, NS(BUFFER_DATASTORE_MEMPOOL) );

    if( ptr_buffer != SIXTRL_NULLPTR )
    {
        if( 0 != NS(Buffer_reset_generic)( ptr_buffer ) )
        {
            NS(Buffer_delete)( ptr_buffer );
            ptr_buffer = SIXTRL_NULLPTR;
        }
    }

    return ptr_buffer;
}

SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* NS(Buffer_new_detailed)(
    NS(buffer_size_t)  const initial_max_num_objects,
    NS(buffer_size_t)  const initial_max_num_slots,
    NS(buffer_size_t)  const initial_max_num_dataptrs,
    NS(buffer_size_t)  const initial_max_num_garbage_elements,
    NS(buffer_flags_t) const buffer_flags )
{
    typedef NS(buffer_size_t) buf_size_t;
    typedef NS(buffer_addr_t) address_t;

    buf_size_t const slot_size = NS(BUFFER_DEFAULT_SLOT_SIZE);

    buf_size_t const addr_size = NS(ManagedBuffer_get_slot_based_length)(
        sizeof( address_t ), slot_size );

    buf_size_t const obj_info_size = NS(ManagedBuffer_get_slot_based_length)(
        sizeof( NS(Object) ), slot_size );

    buf_size_t const dataptrs_size = NS(ManagedBuffer_get_slot_based_length)(
        sizeof( address_t* ), slot_size );

    buf_size_t const garbage_size = slot_size;

    buf_size_t const section_hd_size = ( buf_size_t )2u * addr_size;

    buf_size_t const required_capacity =
        ( buf_size_t )8u * addr_size +
        section_hd_size + initial_max_num_slots    * slot_size +
        section_hd_size + initial_max_num_objects  * obj_info_size +
        section_hd_size + initial_max_num_dataptrs * dataptrs_size +
        section_hd_size + initial_max_num_garbage_elements * garbage_size +
        slot_size;

    NS(Buffer)* ptr_buffer = NS(Buffer_allocate_generic)(
        required_capacity, buffer_flags );

    if( ptr_buffer != SIXTRL_NULLPTR )
    {
        if( 0 != NS(Buffer_reset_detailed_generic)( ptr_buffer,
            initial_max_num_objects, initial_max_num_slots,
            initial_max_num_dataptrs, initial_max_num_garbage_elements ) )
        {
            NS(Buffer_delete)( ptr_buffer );
            ptr_buffer = SIXTRL_NULLPTR;
        }
    }

    return ptr_buffer;
}

SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* NS(Buffer_new_on_memory)(
    SIXTRL_ARGPTR_DEC unsigned char* SIXTRL_RESTRICT data_buffer_begin,
    NS(buffer_size_t) const buffer_capacity )
{
    typedef SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* ptr_buffer_t;

    ptr_buffer_t ptr_buffer = SIXTRL_NULLPTR;

    if( ( data_buffer_begin != SIXTRL_NULLPTR ) &&
        ( buffer_capacity > ( NS(buffer_size_t) )0u ) )
    {
        ptr_buffer = NS(Buffer_preset)(
            ( ptr_buffer_t )malloc( sizeof( NS(Buffer) ) ) );

        if( ptr_buffer != SIXTRL_NULLPTR )
        {
            if( 0 != NS(Buffer_init)(
                    ptr_buffer, data_buffer_begin, buffer_capacity ) )
            {
                NS(Buffer_delete)( ptr_buffer );
                ptr_buffer = SIXTRL_NULLPTR;
            }
        }
    }

    return ptr_buffer;
}

SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* NS(Buffer_new_on_data)(
    SIXTRL_ARGPTR_DEC unsigned char* SIXTRL_RESTRICT data_buffer_begin,
    NS(buffer_size_t) const max_data_buffer_length )
{
    typedef NS(Buffer) buf_t;
    typedef SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* ptr_t;

    ptr_t ptr_buffer = SIXTRL_NULLPTR;

    if( ( data_buffer_begin != SIXTRL_NULLPTR ) &&
        ( max_data_buffer_length > ( NS(buffer_size_t) )0u ) )
    {
        ptr_buffer = NS(Buffer_preset)( ( ptr_t )malloc( sizeof( buf_t ) ) );

        if( 0 != NS(Buffer_init_from_data)( ptr_buffer, data_buffer_begin,
                max_data_buffer_length ) )
        {
            NS(Buffer_delete)( ptr_buffer );
            ptr_buffer = SIXTRL_NULLPTR;
        }
    }

    return ptr_buffer;
}

SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* NS(Buffer_new_from_copy)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT original )
{
    typedef NS(buffer_size_t) size_t;
    typedef unsigned char raw_t;

    NS(Buffer)* buffer = SIXTRL_NULLPTR;
    size_t const orig_capacity = NS(Buffer_get_capacity)( original );

    if( ( original != SIXTRL_NULLPTR ) &&
        ( NS(Buffer_get_slot_size)( original ) != ( size_t )0u ) &&
        ( orig_capacity > ( size_t )0u ) &&
        ( NS(Buffer_get_data_begin_addr)( original ) != 0u ) )
    {
        int status = -1;
        size_t const orig_size = NS(Buffer_get_size)( original );
        buffer = NS(Buffer_new)( orig_capacity );

        if( ( buffer != SIXTRL_NULLPTR ) && ( orig_size > ( size_t )0u ) )
        {
            typedef SIXTRL_BUFFER_DATAPTR_DEC raw_t const* ptr_const_raw_t;
            typedef SIXTRL_BUFFER_DATAPTR_DEC raw_t* ptr_raw_t;

            ptr_const_raw_t orig_begin = ( ptr_const_raw_t )( uintptr_t
                )NS(Buffer_get_data_begin_addr)( original );

            ptr_raw_t dest_begin = ( ptr_raw_t )( uintptr_t
                )NS(Buffer_get_data_begin_addr)( buffer );

            if( ( orig_begin != SIXTRL_NULLPTR ) &&
                ( dest_begin != SIXTRL_NULLPTR ) )
            {
                SIXTRACKLIB_COPY_VALUES(
                    unsigned char, dest_begin, orig_begin, orig_size );

                status = NS(Buffer_refresh)( buffer );
            }

            if( status != 0 )
            {
                NS(Buffer_delete)( buffer );
                buffer = SIXTRL_NULLPTR;
            }
        }
    }

    return buffer;
}

int NS(Buffer_init_from_data_ext)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_ARGPTR_DEC unsigned char* SIXTRL_RESTRICT data_buffer_begin,
    NS(buffer_size_t) const max_data_buffer_length )
{
    return NS(Buffer_init_from_data)(
        buffer, data_buffer_begin, max_data_buffer_length );
}

bool NS(Buffer_read_from_file)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_BUFFER_ARGPTR_DEC char const* SIXTRL_RESTRICT path_to_file )
{
    typedef NS(buffer_size_t) buf_size_t;
    bool success = false;

    if( ( NS(Buffer_allow_clear)( buffer ) ) &&
        ( NS(Buffer_allow_modify_datastore_contents)( buffer ) ) &&
        ( path_to_file != SIXTRL_NULLPTR ) &&
        ( strlen( path_to_file ) > ( buf_size_t )0u ) )
    {
        buf_size_t size_of_file = ( buf_size_t )0u;
        FILE* fp = fopen( path_to_file, "rb" );

        if( fp != SIXTRL_NULLPTR )
        {
            long length = ( long )0u;

            fseek( fp, 0, SEEK_END );
            length = ftell( fp );
            fclose( fp );
            fp = SIXTRL_NULLPTR;

            if( length > 0 )
            {
                size_of_file = ( buf_size_t )length;
            }
        }

        if( size_of_file > ( buf_size_t )0u )
        {
            buf_size_t buffer_capacity = NS(Buffer_get_capacity)( buffer );

            if( size_of_file <= buffer_capacity )
            {
                success = 0;
            }
            else
            {
                if( NS(Buffer_reserve_capacity)( buffer, size_of_file ) == 0 )
                {
                    buffer_capacity = NS(Buffer_get_capacity)( buffer );
                    success = 0;
                }
            }

            if( ( success == 0 ) && ( size_of_file <= buffer_capacity ) )
            {
                success = -1;
                fp = fopen( path_to_file, "rb" );

                if( fp != SIXTRL_NULLPTR )
                {
                    typedef SIXTRL_BUFFER_DATAPTR_DEC char* ptr_read_buffer_t;

                    ptr_read_buffer_t read_buffer = ( ptr_read_buffer_t )(
                        uintptr_t )NS(Buffer_get_data_begin_addr)( buffer );

                    if( ( fread( read_buffer, size_of_file, 1u, fp ) == 1u ) &&
                        ( 0 == NS(Buffer_remap)( buffer ) ) )
                    {
                        success = 0;
                    }
                }
            }
        }

        if( fp != SIXTRL_NULLPTR )
        {
            fclose( fp );
            fp = SIXTRL_NULLPTR;
        }
    }

    return success;
}

NS(Buffer)* NS(Buffer_new_from_file)(
    SIXTRL_ARGPTR_DEC char const* SIXTRL_RESTRICT path_to_file )
{
    typedef NS(buffer_size_t) buf_size_t;

    NS(Buffer)*  ptr_buffer = SIXTRL_NULLPTR;
    buf_size_t size_of_file = ( buf_size_t )0u;

    if( path_to_file != SIXTRL_NULLPTR )
    {
        FILE* fp = fopen( path_to_file, "rb" );

        if( fp != SIXTRL_NULLPTR )
        {
            long length = ( long )0u;

            fseek( fp, 0, SEEK_END );
            length = ftell( fp );
            fclose( fp );
            fp = SIXTRL_NULLPTR;

            if( length > 0 )
            {
                size_of_file = ( buf_size_t )length;
            }
        }
    }

    if( size_of_file > ( buf_size_t )0u )
    {
        ptr_buffer = NS(Buffer_new)( size_of_file );

        if( ptr_buffer != SIXTRL_NULLPTR )
        {
            int success = -1;
            FILE* fp = fopen( path_to_file, "rb" );

            if( fp != SIXTRL_NULLPTR )
            {
                buf_size_t const cnt = fread(
                    ( SIXTRL_BUFFER_DATAPTR_DEC char* )( uintptr_t
                        )NS(Buffer_get_data_begin_addr)( ptr_buffer ),
                    size_of_file, ( buf_size_t )1u, fp );

                if( ( cnt == ( buf_size_t )1u ) &&
                    ( 0   == NS(Buffer_remap)( ptr_buffer ) ) )
                {
                    success = 0;
                }

                fclose( fp );
                fp = SIXTRL_NULLPTR;
            }

            if( success != 0 )
            {
                NS(Buffer_delete)( ptr_buffer );
                ptr_buffer = SIXTRL_NULLPTR;
            }
        }
    }

    return ptr_buffer;
}

bool NS(Buffer_write_to_file)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    SIXTRL_ARGPTR_DEC char const* SIXTRL_RESTRICT path_to_file )
{
    bool success = false;

    SIXTRL_ARGPTR_DEC FILE* fp = SIXTRL_NULLPTR;

    if( ( path_to_file != SIXTRL_NULLPTR ) &&
        ( strlen( path_to_file ) > 0 ) )
    {
        fp = fopen( path_to_file, "wb" );
    }

    success = NS(Buffer_write_to_fp)( buffer, fp );

    if( fp != SIXTRL_NULLPTR )
    {
        fclose( fp );
        fp = SIXTRL_NULLPTR;
    }

    return success;
}

bool NS(Buffer_write_to_file_normalized_addr)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_ARGPTR_DEC char const* SIXTRL_RESTRICT path_to_file,
    NS(buffer_addr_t) const norm_base_addr )
{
    bool success = false;

    SIXTRL_ARGPTR_DEC FILE* fp = SIXTRL_NULLPTR;

    if( ( path_to_file != SIXTRL_NULLPTR ) &&
        ( strlen( path_to_file ) > 0 ) )
    {
        fp = fopen( path_to_file, "wb" );
    }

    success = NS(Buffer_write_to_fp_normalized_addr)(
        buffer, fp, norm_base_addr );

    if( fp != SIXTRL_NULLPTR )
    {
        fclose( fp );
        fp = SIXTRL_NULLPTR;
    }

    return success;
}

bool NS(Buffer_write_to_fp)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    SIXTRL_ARGPTR_DEC FILE* fp )
{
    bool success = false;

    if( ( fp != SIXTRL_NULLPTR ) && ( buffer != SIXTRL_NULLPTR ) &&
        ( NS(Buffer_get_data_begin_addr)( buffer ) != ( NS(buffer_addr_t) )0 ) &&
        ( NS(Buffer_get_size)( buffer ) > ( NS(buffer_size_t) )0u ) )
    {
        size_t const cnt = fwrite( ( SIXTRL_BUFFER_DATAPTR_DEC char* )( uintptr_t
            )NS(Buffer_get_data_begin_addr)( buffer ),
            NS(Buffer_get_size)( buffer ), 1u, fp );

        success = ( cnt == 1u );

    }

    return success;
}

bool NS(Buffer_write_to_fp_normalized_addr)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_ARGPTR_DEC FILE* fp, NS(buffer_addr_t) const norm_base_addr )
{
    bool success = false;

    typedef NS(buffer_size_t)       buf_size_t;
    typedef NS(buffer_size_t)       address_t;
    typedef NS(buffer_addr_diff_t)  addr_diff_t;

    typedef SIXTRL_BUFFER_DATAPTR_DEC unsigned char* ptr_data_t;

    address_t  const cur_base_addr = NS(Buffer_get_data_begin_addr)( buffer );
    buf_size_t const data_size     = NS(Buffer_get_size)( buffer );

    if( ( fp != SIXTRL_NULLPTR ) && ( buffer != SIXTRL_NULLPTR ) &&
        ( cur_base_addr != ( address_t )0u ) && ( data_size > ( buf_size_t )0u ) )
    {
        buf_size_t const slot_size = NS(Buffer_get_slot_size)( buffer );
        ptr_data_t begin = ( ptr_data_t )( uintptr_t )cur_base_addr;

        if( cur_base_addr != norm_base_addr )
        {
            /* TODO: Check for integer overflow in a generic fashion, i.e.
             *       depending on the type NS(buffer_addr_t) is implemented as! */
            addr_diff_t const base_offset = ( norm_base_addr >= cur_base_addr )
                ?  ( addr_diff_t )( norm_base_addr - cur_base_addr )
                : -( addr_diff_t )( cur_base_addr - norm_base_addr );

            success = ( NS(ManagedBuffer_apply_addr_offset)(
                begin, base_offset, slot_size ) == 0 );
        }
        else
        {
            success = true;
        }

        if( success )
        {
            size_t const cnt = fwrite( begin, data_size, 1u, fp );
            success = ( cnt == 1u );

            if( NS(Buffer_needs_remapping)( buffer ) )
            {
                int const ret = NS(Buffer_remap)( buffer );
                success &= ( 0 == ret );
            }
        }
    }

    return success;
}

int NS(Buffer_reset_ext)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer )
{
    return NS(Buffer_reset)( buffer );
}

int NS(Buffer_reset_detailed_ext)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const max_num_objects,
    NS(buffer_size_t) const max_num_slots,
    NS(buffer_size_t) const max_num_dataptrs,
    NS(buffer_size_t) const max_num_garbage_ranges )
{
    return NS(Buffer_reset_detailed)( buffer, max_num_objects, max_num_slots,
        max_num_dataptrs, max_num_garbage_ranges );
}

int NS(Buffer_reserve_ext)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const new_max_num_objects,
    NS(buffer_size_t) const new_max_num_slots,
    NS(buffer_size_t) const new_max_num_dataptrs,
    NS(buffer_size_t) const new_max_num_garbage_ranges )
{
    return NS(Buffer_reserve)( buffer, new_max_num_objects, new_max_num_slots,
        new_max_num_dataptrs, new_max_num_garbage_ranges );
}

int NS(Buffer_reserve_capacity_ext)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const new_buffer_capacity )
{
    return NS(Buffer_reserve_capacity)( buffer, new_buffer_capacity );
}

bool NS(Buffer_needs_remapping_ext)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer )
{
    return NS(Buffer_needs_remapping)( buffer );
}

int NS(Buffer_remap_ext)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer )
{
    return NS(Buffer_remap)( buffer );
}

int NS(Buffer_refresh_ext)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer )
{
    return NS(Buffer_refresh)( buffer );
}

int NS(Buffer_clear_ext)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    bool const set_data_to_zero )
{
    return NS(Buffer_clear)( buffer, set_data_to_zero );
}

void NS(Buffer_free_ext)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer )
{
    NS(Buffer_free)( buffer );
    return;
}

void NS(Buffer_delete)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer )
{
    NS(Buffer_free)( buffer );
    free( buffer );
    buffer = SIXTRL_NULLPTR;

    return;
}

#endif /* !defined( _GPUCODE ) */

/* end: sixtracklib/common/buffer/buffer.c */
