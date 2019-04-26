#ifndef SIXTRACKLIB_COMMON_BUFFER_MANAGED_BUFFER_HANDLE_H__
#define SIXTRACKLIB_COMMON_BUFFER_MANAGED_BUFFER_HANDLE_H__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stdbool.h>
    #include <stddef.h>
    #include <stdlib.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/buffer/buffer_type.h"
    #if !defined( _GPUCODE )
        #include "sixtracklib/common/buffer.h"
    #endif /* !defined( _GPUCODE ) */
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus ) && !defined( _GPUCODE )
extern "C" {
#endif /* defined( __cplusplus ) && !defined( _GPUCODE ) */

typedef struct NS(ManagedBufferHandle)
{
    NS(buffer_addr_t)   begin_addr  SIXTRL_ALIGN( 8u );
    NS(buffer_size_t)   data_length SIXTRL_ALIGN( 8u );
    NS(buffer_size_t)   slot_size   SIXTRL_ALIGN( 8u );
}
NS(ManagedBufferHandle);

SIXTRL_STATIC SIXTRL_FN SIXTRL_BUFFER_DATAPTR_DEC NS(ManagedBufferHandle)*
NS(ManagedBufferHandle_preset)( SIXTRL_BUFFER_DATAPTR_DEC
    NS(ManagedBufferHandle)* SIXTRL_RESTRICT handle );

SIXTRL_STATIC SIXTRL_FN NS(buffer_addr_t)
NS(ManagedBufferHandle_get_begin_addr)( SIXTRL_BUFFER_DATAPTR_DEC
    const NS(ManagedBufferHandle) *const SIXTRL_RESTRICT handle );

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t)
NS(ManagedBufferHandle_get_length)( SIXTRL_BUFFER_DATAPTR_DEC
    const NS(ManagedBufferHandle) *const SIXTRL_RESTRICT handle );

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t)
NS(ManagedBufferHandle_get_slot_size)( SIXTRL_BUFFER_DATAPTR_DEC
    const NS(ManagedBufferHandle) *const SIXTRL_RESTRICT handle );

SIXTRL_STATIC SIXTRL_FN void NS(ManagedBufferHandle_set_begin_addr)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(ManagedBufferHandle)* SIXTRL_RESTRICT handle,
    NS(buffer_addr_t) const begin_addr );

SIXTRL_STATIC SIXTRL_FN void NS(ManagedBufferHandle_set_slot_size)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(ManagedBufferHandle)* SIXTRL_RESTRICT handle,
    NS(buffer_size_t) const slot_size );

SIXTRL_STATIC SIXTRL_FN void NS(ManagedBufferHandle_set_length)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(ManagedBufferHandle)* SIXTRL_RESTRICT handle,
    NS(buffer_size_t) const length );


SIXTRL_STATIC SIXTRL_FN SIXTRL_BUFFER_DATAPTR_DEC unsigned char*
NS(ManagedBufferHandle_get_data_begin)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(ManagedBufferHandle)* SIXTRL_RESTRICT handle );

SIXTRL_STATIC SIXTRL_FN SIXTRL_BUFFER_DATAPTR_DEC unsigned char*
NS(ManagedBufferHandle_get_data_end)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(ManagedBufferHandle)* SIXTRL_RESTRICT handle );


SIXTRL_STATIC SIXTRL_FN SIXTRL_BUFFER_DATAPTR_DEC unsigned char const*
NS(ManagedBufferHandle_get_const_data_begin)( SIXTRL_BUFFER_ARGPTR_DEC
    const NS(ManagedBufferHandle) *const SIXTRL_RESTRICT handle );

SIXTRL_STATIC SIXTRL_FN SIXTRL_BUFFER_DATAPTR_DEC unsigned char const*
NS(ManagedBufferHandle_get_const_data_end)( SIXTRL_BUFFER_ARGPTR_DEC
    const NS(ManagedBufferHandle) *const SIXTRL_RESTRICT handle );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

#if !defined( _GPUCODE )

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BUFFER_DATAPTR_DEC NS(ManagedBufferHandle)*
NS(ManagedBufferHandle_preset_ext)( SIXTRL_BUFFER_DATAPTR_DEC
    NS(ManagedBufferHandle)* SIXTRL_RESTRICT handle );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(buffer_addr_t)
NS(ManagedBufferHandle_get_begin_addr_ext)( SIXTRL_BUFFER_DATAPTR_DEC
    const NS(ManagedBufferHandle) *const SIXTRL_RESTRICT handle );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(buffer_size_t)
NS(ManagedBufferHandle_get_length_ext)( SIXTRL_BUFFER_DATAPTR_DEC
    const NS(ManagedBufferHandle) *const SIXTRL_RESTRICT handle );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(buffer_size_t)
NS(ManagedBufferHandle_get_slot_size_ext)( SIXTRL_BUFFER_DATAPTR_DEC
    const NS(ManagedBufferHandle) *const SIXTRL_RESTRICT handle );


SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BUFFER_DATAPTR_DEC unsigned char*
NS(ManagedBufferHandle_get_data_begin_ext)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(ManagedBufferHandle)* SIXTRL_RESTRICT handle );

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BUFFER_DATAPTR_DEC unsigned char*
NS(ManagedBufferHandle_get_data_end_ext)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(ManagedBufferHandle)* SIXTRL_RESTRICT handle );


SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BUFFER_DATAPTR_DEC unsigned char const*
NS(ManagedBufferHandle_get_const_data_begin_ext)( SIXTRL_BUFFER_ARGPTR_DEC
    const NS(ManagedBufferHandle) *const SIXTRL_RESTRICT handle );

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BUFFER_DATAPTR_DEC unsigned char const*
NS(ManagedBufferHandle_get_const_data_end_ext)( SIXTRL_BUFFER_ARGPTR_DEC
    const NS(ManagedBufferHandle) *const SIXTRL_RESTRICT handle );

#endif /* !defined( _GPUCODE ) */

/* ------------------------------------------------------------------------ */

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t)
NS(ManagedBufferHandle_get_required_num_slots_on_managed_buffer)(
    NS(buffer_size_t) const data_length_to_store,
    NS(buffer_size_t) const slot_size );

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t)
NS(ManagedBufferHandle_get_required_num_dataptrs_on_managed_buffer)(
    NS(buffer_size_t) const data_length_to_store,
    NS(buffer_size_t) const slot_size );

#if !defined( _GPUCODE )

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t)
NS(ManagedBufferHandle_get_required_num_slots)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const data_length_to_store );

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t)
NS(ManagedBufferHandle_get_required_num_dataptrs)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const data_length_to_store );

SIXTRL_STATIC SIXTRL_FN bool NS(ManagedBufferHandle_can_be_added)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const data_length_to_store,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_objects,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_slots,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_dataptr );

SIXTRL_STATIC SIXTRL_FN SIXTRL_BUFFER_DATAPTR_DEC NS(ManagedBufferHandle)*
NS(ManagedBufferHandle_new)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const data_length );

SIXTRL_STATIC SIXTRL_FN SIXTRL_BUFFER_DATAPTR_DEC NS(ManagedBufferHandle)*
NS(ManagedBufferHandle_add)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_addr_t) const begin_addr,
    NS(buffer_size_t) const data_length,
    NS(buffer_size_t) const stored_slot_size,
    bool const perform_deep_copy );

SIXTRL_STATIC SIXTRL_FN SIXTRL_BUFFER_DATAPTR_DEC NS(ManagedBufferHandle)*
NS(ManagedBufferHandle_new_from_existing_managed_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT buffer_begin,
    NS(buffer_size_t) const slot_size,
    bool const perform_deep_copy );

SIXTRL_STATIC SIXTRL_FN SIXTRL_BUFFER_DATAPTR_DEC NS(ManagedBufferHandle)*
NS(ManagedBufferHandle_add_copy)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(ManagedBufferHandle) *const
        SIXTRL_RESTRICT existing_handle,
    bool const perform_deep_copy );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_EXTERN SIXTRL_HOST_FN NS(buffer_size_t)
NS(ManagedBufferHandle_get_required_num_slots_ext)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const data_length_to_store );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(buffer_size_t)
NS(ManagedBufferHandle_get_required_num_dataptrs_ext)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const data_length_to_store );

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(ManagedBufferHandle_can_be_added_ext)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const data_length_to_store,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_objects,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_slots,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_dataptr );

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BUFFER_DATAPTR_DEC NS(ManagedBufferHandle)*
NS(ManagedBufferHandle_new_ext)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const data_length );

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BUFFER_DATAPTR_DEC NS(ManagedBufferHandle)*
NS(ManagedBufferHandle_add_ext)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_addr_t) const begin_addr,
    NS(buffer_size_t) const data_length,
    NS(buffer_size_t) const stored_slot_size,
    bool const perform_deep_copy );

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BUFFER_DATAPTR_DEC NS(ManagedBufferHandle)*
NS(ManagedBufferHandle_new_from_existing_managed_buffer_ext)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT buffer_begin,
    NS(buffer_size_t) const slot_size,
    bool const perform_deep_copy );

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BUFFER_DATAPTR_DEC NS(ManagedBufferHandle)*
NS(ManagedBufferHandle_add_copy_ext)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(ManagedBufferHandle) *const
        SIXTRL_RESTRICT existing_handle,
    bool const perform_deep_copy );

#endif /* !defined( _GPUCODE ) */

/* ************************************************************************* */
/* *********         Implementation of Inline Functions          *********** */
/* ************************************************************************* */

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(ManagedBufferHandle)*
NS(ManagedBufferHandle_preset)( SIXTRL_BUFFER_DATAPTR_DEC
    NS(ManagedBufferHandle)* SIXTRL_RESTRICT handle )
{
    if( handle != SIXTRL_NULLPTR )
    {
        handle->begin_addr  = ( NS(buffer_addr_t) )0u;
        handle->data_length = ( NS(buffer_size_t) )0u;
        handle->slot_size   = NS(BUFFER_DEFAULT_SLOT_SIZE);
    }

    return handle;
}

SIXTRL_INLINE NS(buffer_addr_t)
NS(ManagedBufferHandle_get_begin_addr)( SIXTRL_BUFFER_DATAPTR_DEC const
    NS(ManagedBufferHandle) *const SIXTRL_RESTRICT handle )
{
    SIXTRL_ASSERT( handle != SIXTRL_NULLPTR );
    return handle->begin_addr;
}

SIXTRL_INLINE NS(buffer_size_t)
NS(ManagedBufferHandle_get_length)( SIXTRL_BUFFER_DATAPTR_DEC
    const NS(ManagedBufferHandle) *const SIXTRL_RESTRICT handle )
{
    SIXTRL_ASSERT( handle != SIXTRL_NULLPTR );
    return handle->data_length;
}

SIXTRL_INLINE NS(buffer_size_t)
NS(ManagedBufferHandle_get_slot_size)( SIXTRL_BUFFER_DATAPTR_DEC
    const NS(ManagedBufferHandle) *const SIXTRL_RESTRICT handle )
{
    SIXTRL_ASSERT( handle != SIXTRL_NULLPTR );
    return handle->slot_size;
}

SIXTRL_INLINE void NS(ManagedBufferHandle_set_begin_addr)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(ManagedBufferHandle)* SIXTRL_RESTRICT handle,
    NS(buffer_addr_t) const begin_addr )
{
    SIXTRL_ASSERT( handle != SIXTRL_NULLPTR );
    handle->begin_addr = begin_addr;
}

SIXTRL_INLINE void NS(ManagedBufferHandle_set_slot_size)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(ManagedBufferHandle)* SIXTRL_RESTRICT handle,
    NS(buffer_size_t) const slot_size )
{
    SIXTRL_ASSERT( handle != SIXTRL_NULLPTR );
    handle->slot_size = slot_size;
}

SIXTRL_INLINE void NS(ManagedBufferHandle_set_length)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(ManagedBufferHandle)* SIXTRL_RESTRICT handle,
    NS(buffer_size_t) const length )
{
    SIXTRL_ASSERT( handle != SIXTRL_NULLPTR );
    handle->data_length = length;
}


SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC unsigned char*
NS(ManagedBufferHandle_get_data_begin)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(ManagedBufferHandle)* SIXTRL_RESTRICT handle )
{
    typedef SIXTRL_BUFFER_DATAPTR_DEC unsigned char* ptr_t;
    return ( ptr_t  )NS(ManagedBufferHandle_get_const_data_begin)( handle );
}

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC unsigned char*
NS(ManagedBufferHandle_get_data_end)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(ManagedBufferHandle)* SIXTRL_RESTRICT handle )
{
    typedef SIXTRL_BUFFER_DATAPTR_DEC unsigned char* ptr_t;
    return ( ptr_t  )NS(ManagedBufferHandle_get_const_data_end)( handle );
}


SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC unsigned char const*
NS(ManagedBufferHandle_get_const_data_begin)( SIXTRL_BUFFER_ARGPTR_DEC
    const NS(ManagedBufferHandle) *const SIXTRL_RESTRICT handle )
{
    typedef SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* ptr_t;

    return ( handle != SIXTRL_NULLPTR )
        ? ( ptr_t )( uintptr_t )handle->begin_addr
        : ( ptr_t )SIXTRL_NULLPTR;
}

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC unsigned char const*
NS(ManagedBufferHandle_get_const_data_end)( SIXTRL_BUFFER_ARGPTR_DEC
    const NS(ManagedBufferHandle) *const SIXTRL_RESTRICT handle )
{
    typedef SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* ptr_t;

    ptr_t end_ptr = NS(ManagedBufferHandle_get_const_data_begin)( handle );

    if( end_ptr != SIXTRL_NULLPTR )
    {
        end_ptr = end_ptr + NS(ManagedBufferHandle_get_length)( handle );
    }

    return end_ptr;
}


SIXTRL_INLINE NS(buffer_size_t)
NS(ManagedBufferHandle_get_required_num_slots_on_managed_buffer)(
    NS(buffer_size_t) const data_length_to_store,
    NS(buffer_size_t) const slot_size )
{
    typedef NS(buffer_size_t) buf_size_t;
    SIXTRL_STATIC_VAR buf_size_t const ZERO = ( buf_size_t )0u;

    buf_size_t num_slots = ZERO;
    buf_size_t required_size = NS(ManagedBuffer_get_slot_based_length)(
        sizeof( NS(ManagedBufferHandle) ), slot_size );

    if( data_length_to_store > ZERO )
    {
        required_size += NS(ManagedBuffer_get_slot_based_length)(
            data_length_to_store, slot_size );
    }

    if( slot_size > ZERO )
    {
        SIXTRL_ASSERT( ZERO == ( required_size % slot_size ) );
        num_slots = required_size / slot_size;
    }

    return num_slots;
}

SIXTRL_INLINE NS(buffer_size_t)
NS(ManagedBufferHandle_get_required_num_dataptrs_on_managed_buffer)(
    NS(buffer_size_t) const data_length_to_store,
    NS(buffer_size_t) const slot_size )
{
    typedef NS(buffer_size_t) buf_size_t;
    SIXTRL_STATIC_VAR buf_size_t const ZERO = ( buf_size_t )0u;

    buf_size_t num_dataptrs = ZERO;

    if( ( data_length_to_store > ZERO ) && ( slot_size > ZERO ) )
    {
        num_dataptrs = ( buf_size_t )1u;
    }

    return num_dataptrs;
}


SIXTRL_INLINE NS(buffer_size_t) NS(ManagedBufferHandle_get_required_num_slots)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const data_length_to_store )
{
    return NS(ManagedBufferHandle_get_required_num_slots_on_managed_buffer)(
        data_length_to_store, NS(Buffer_get_slot_size)( buffer ) );
}

SIXTRL_INLINE NS(buffer_size_t)
NS(ManagedBufferHandle_get_required_num_dataptrs)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const data_length_to_store )
{
    return NS(ManagedBufferHandle_get_required_num_dataptrs_on_managed_buffer)(
        data_length_to_store, NS(Buffer_get_slot_size)( buffer ) );
}

SIXTRL_INLINE bool NS(ManagedBufferHandle_can_be_added)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const data_length_to_store,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_objects,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_slots,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_dataptrs )
{
    typedef NS(buffer_size_t) buf_size_t;

    buf_size_t const num_dataptrs =
        NS(ManagedBufferHandle_get_required_num_dataptrs)( buffer,
            data_length_to_store );

    buf_size_t const sizes[]  = { data_length_to_store };
    buf_size_t const counts[] = { num_dataptrs };
    buf_size_t const obj_size = sizeof( NS(ManagedBufferHandle) );

    return NS(Buffer_can_add_object)( buffer, obj_size, num_dataptrs, sizes,
        counts, requ_objects, requ_slots, requ_dataptrs );
}

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(ManagedBufferHandle)*
NS(ManagedBufferHandle_new)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const data_length_to_store )
{
    typedef NS(ManagedBufferHandle) handle_t;
    typedef NS(buffer_size_t) buf_size_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC handle_t* ptr_handle_t;

    SIXTRL_STATIC_VAR buf_size_t const ZERO = ( buf_size_t )0u;
    SIXTRL_STATIC_VAR buf_size_t const ONE  = ( buf_size_t )1u;

    buf_size_t const nptrs = NS(ManagedBufferHandle_get_required_num_dataptrs)(
        buffer, data_length_to_store );

    buf_size_t const count_value = ( nptrs > ZERO ) ? ONE : ZERO;

    buf_size_t offsets[] = { offsetof( handle_t, begin_addr ) };
    buf_size_t sizes[]   = { data_length_to_store };
    buf_size_t counts[]  = { count_value };

    handle_t handle;
    handle.begin_addr  = ( NS(buffer_addr_t) )0u;
    handle.data_length = data_length_to_store;
    handle.slot_size   = NS(BUFFER_DEFAULT_SLOT_SIZE);

    return ( ptr_handle_t )( uintptr_t )NS(Object_get_begin_addr)(
        NS(Buffer_add_object)( buffer, &handle, sizeof( handle_t ),
            NS(OBJECT_TYPE_MANAGED_BUFFER_HANDLE),
                nptrs, &offsets[ 0 ], &sizes[ 0 ], &counts[ 0 ] ) );
}

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(ManagedBufferHandle)*
NS(ManagedBufferHandle_add)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_addr_t) const begin_addr, NS(buffer_size_t) const data_length,
    NS(buffer_size_t) const slot_size, bool const perform_deep_copy )
{
    typedef NS(buffer_size_t) buf_size_t;
    typedef NS(buffer_addr_t) addr_t;

    typedef NS(ManagedBufferHandle) handle_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC handle_t* ptr_handle_t;

    typedef unsigned char raw_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC raw_t*       ptr_raw_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC raw_t const* ptr_craw_t;

    ptr_handle_t handle = SIXTRL_NULLPTR;

    if( ( perform_deep_copy ) && ( data_length > ( buf_size_t )0u ) &&
        ( begin_addr > ( addr_t )0u ) && ( slot_size > ( buf_size_t )0u ) )
    {
        handle = NS(ManagedBufferHandle_new)( buffer, data_length );

        if( handle != SIXTRL_NULLPTR)
        {
            ptr_craw_t in = ( ptr_craw_t )( uintptr_t )begin_addr;
            ptr_raw_t out = NS(ManagedBufferHandle_get_data_begin)( handle );

            SIXTRL_ASSERT( out != SIXTRL_NULLPTR );
            SIXTRL_ASSERT( NS(ManagedBufferHandle_get_length)( handle ) ==
                           data_length );

            NS(ManagedBufferHandle_set_slot_size)( handle, slot_size );
            SIXTRACKLIB_COPY_VALUES( raw_t, out, in, data_length );

            if( NS(ManagedBuffer_needs_remapping)( out, slot_size ) )
            {
                int const ret = NS(ManagedBuffer_remap)( out, slot_size );
                SIXTRL_ASSERT( ret == 0 );

                ( void )ret;
            }
        }
    }
    else if( !perform_deep_copy )
    {
        handle = NS(ManagedBufferHandle_new)( buffer, ( buf_size_t )0u );

        if( handle != SIXTRL_NULLPTR )
        {
            NS(ManagedBufferHandle_set_begin_addr)( handle, begin_addr );
            NS(ManagedBufferHandle_set_length)( handle, data_length );
            NS(ManagedBufferHandle_set_slot_size)( handle, slot_size );
        }
    }

    return handle;
}

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(ManagedBufferHandle)*
NS(ManagedBufferHandle_new_from_existing_managed_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT buffer_begin,
    NS(buffer_size_t) const slot_size,
    bool const perform_deep_copy )
{
    return NS(ManagedBufferHandle_add)( buffer,
        ( NS(buffer_addr_t) )( uintptr_t )buffer_begin,
        NS(ManagedBuffer_get_buffer_length)( buffer_begin, slot_size ),
        slot_size, perform_deep_copy );
}

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(ManagedBufferHandle)*
NS(ManagedBufferHandle_add_copy)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(ManagedBufferHandle) *const
        SIXTRL_RESTRICT existing_handle,
    bool const perform_deep_copy )
{
    return NS(ManagedBufferHandle_add)( buffer,
        NS(ManagedBufferHandle_get_begin_addr)( existing_handle ),
        NS(ManagedBufferHandle_get_length)( existing_handle ),
        NS(ManagedBufferHandle_get_slot_size)( existing_handle ),
        perform_deep_copy );
}

#if defined( __cplusplus ) && !defined( _GPUCODE )
}
#endif /* defined( __cplusplus ) && !defined( _GPUCODE ) */

#endif /* SIXTRACKLIB_COMMON_BUFFER_MANAGED_BUFFER_HANDLE_H__ */

/* end: sixtracklib/common/buffer/managed_buffer_handle.h */
