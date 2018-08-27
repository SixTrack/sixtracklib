#include "sixtracklib/common/buffer.h"

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "sixtracklib/_impl/definitions.h"
#include "sixtracklib/common/mem_pool.h"

#include "sixtracklib/common/impl/buffer_generic.h"
#include "sixtracklib/common/impl/buffer_mem.h"
#include "sixtracklib/common/impl/buffer_mem_minimal.h"
#include "sixtracklib/common/impl/buffer_mem_remap.h"
#include "sixtracklib/common/impl/buffer_object.h"
#include "sixtracklib/common/impl/buffer_type.h"

#if defined( SIXTRACKLIB_ENABLE_MODULE_OPENCL ) && \
        ( SIXTRACKLIB_ENABLE_MODULE_OPENCL == 1 )
    #include "sixtracklib/opencl/buffer.h"
#endif /* OpenCL */

#if defined( SIXTRACKLIB_ENABLE_MODULE_CUDA ) && \
        ( SIXTRACKLIB_ENABLE_MODULE_CUDA == 1 )
    #include "sixtracklib/cuda/buffer.h"
#endif /* Cuda */

extern SIXTRL_HOST_FN NS(Buffer)* NS(Buffer_new)(
    NS(buffer_size_t) const buffer_capacity );

extern SIXTRL_HOST_FN NS(Buffer)* NS(Buffer_new_detailed)(
    NS(buffer_size_t)  const initial_max_num_objects,
    NS(buffer_size_t)  const initial_max_num_slots,
    NS(buffer_size_t)  const initial_max_num_dataptrs,
    NS(buffer_size_t)  const initial_max_num_garbage_elements,
    NS(buffer_flags_t) const buffer_flags );

SIXTRL_HOST_FN static NS(Buffer)* NS(Buffer_allocate_generic)(
    NS(buffer_size_t) const buffer_capacity,
    NS(buffer_flags_t) const buffer_flags );

/* ------------------------------------------------------------------------- */

NS(Buffer)* NS(Buffer_allocate_generic)(
    NS(buffer_size_t)  const buffer_capacity,
    NS(buffer_flags_t) const buffer_flags )
{
    typedef NS(Buffer)              buffer_t;
    typedef NS(buffer_size_t)       buf_size_t;
    typedef NS(buffer_addr_t)       address_t;

    int success = -1;

    buffer_t* ptr_buffer = NS(Buffer_preset)(
        ( buffer_t* )malloc( sizeof( buffer_t ) ) );

    buf_size_t const slot_size = NS(Buffer_get_slot_size)( ptr_buffer );

    buf_size_t const addr_size = NS(BufferMem_get_slot_based_length)(
        sizeof( address_t ), slot_size );

    buf_size_t const section_hd_size = ( buf_size_t )2u * addr_size;

    buf_size_t const min_required_capacity =
        NS(BUFFER_DEFAULT_HEADER_SIZE) + ( buf_size_t )4u * section_hd_size;

    if( ( ptr_buffer != SIXTRL_NULLPTR ) &&
        ( min_required_capacity <= buffer_capacity ) )
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
                    buffer_capacity, slot_size ) )
            {
                NS(AllocResult) result = NS(MemPool_append_aligned)(
                    ptr_mem_pool, buffer_capacity, slot_size );

                if( NS(AllocResult_valid)( &result ) )
                {
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

NS(Buffer)* NS(Buffer_new)( NS(buffer_size_t) const buffer_capacity )
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

NS(Buffer)* NS(Buffer_new_detailed)(
    NS(buffer_size_t)  const initial_max_num_objects,
    NS(buffer_size_t)  const initial_max_num_slots,
    NS(buffer_size_t)  const initial_max_num_dataptrs,
    NS(buffer_size_t)  const initial_max_num_garbage_elements,
    NS(buffer_flags_t) const buffer_flags )
{
    typedef NS(buffer_size_t) buf_size_t;
    typedef NS(buffer_addr_t) address_t;

    buf_size_t const slot_size = NS(BUFFER_DEFAULT_SLOT_SIZE);

    buf_size_t const addr_size = NS(BufferMem_get_slot_based_length)(
        sizeof( address_t ), slot_size );

    buf_size_t const obj_info_size = NS(BufferMem_get_slot_based_length)(
        sizeof( NS(Object) ), slot_size );

    buf_size_t const dataptrs_size = NS(BufferMem_get_slot_based_length)(
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


/* end: sixtracklib/common/details/buffer.c */
