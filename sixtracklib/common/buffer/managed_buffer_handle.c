#include "sixtracklib/common/buffer/managed_buffer_handle.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/buffer/buffer_type.h"
#include "sixtracklib/common/buffer.h"

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_BUFFER_DATAPTR_DEC NS(ManagedBufferHandle)*
NS(ManagedBufferHandle_preset_ext)( SIXTRL_BUFFER_DATAPTR_DEC
    NS(ManagedBufferHandle)* SIXTRL_RESTRICT handle )
{
    return NS(ManagedBufferHandle_preset)( handle );
}

NS(buffer_addr_t) NS(ManagedBufferHandle_get_begin_addr_ext)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(ManagedBufferHandle) *const
        SIXTRL_RESTRICT handle )
{
    return NS(ManagedBufferHandle_get_begin_addr)( handle );
}

NS(buffer_size_t) NS(ManagedBufferHandle_get_length_ext)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(ManagedBufferHandle) *const
        SIXTRL_RESTRICT handle )
{
    return NS(ManagedBufferHandle_get_length)( handle );
}

NS(buffer_size_t) NS(ManagedBufferHandle_get_slot_size_ext)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(ManagedBufferHandle) *const
        SIXTRL_RESTRICT handle )
{
    return NS(ManagedBufferHandle_get_slot_size)( handle );
}


SIXTRL_BUFFER_DATAPTR_DEC unsigned char*
NS(ManagedBufferHandle_get_data_begin_ext)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(ManagedBufferHandle)* SIXTRL_RESTRICT handle )
{
    return NS(ManagedBufferHandle_get_data_begin)( handle );
}

SIXTRL_BUFFER_DATAPTR_DEC unsigned char*
NS(ManagedBufferHandle_get_data_end_ext)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(ManagedBufferHandle)* SIXTRL_RESTRICT handle )
{
    return NS(ManagedBufferHandle_get_data_end)( handle );
}


SIXTRL_BUFFER_DATAPTR_DEC unsigned char const*
NS(ManagedBufferHandle_get_const_data_begin_ext)( SIXTRL_BUFFER_ARGPTR_DEC
    const NS(ManagedBufferHandle) *const SIXTRL_RESTRICT handle )
{
    return NS(ManagedBufferHandle_get_const_data_begin)( handle );
}

SIXTRL_BUFFER_DATAPTR_DEC unsigned char const*
NS(ManagedBufferHandle_get_const_data_end_ext)( SIXTRL_BUFFER_ARGPTR_DEC
    const NS(ManagedBufferHandle) *const SIXTRL_RESTRICT handle )
{
    return NS(ManagedBufferHandle_get_const_data_end)( handle );
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

NS(buffer_size_t) NS(ManagedBufferHandle_get_required_num_slots_ext)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const data_length_to_store )
{
    return NS(ManagedBufferHandle_get_required_num_slots)(
        buffer, data_length_to_store );
}

NS(buffer_size_t) NS(ManagedBufferHandle_get_requried_num_dataptrs_ext)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const data_length_to_store )
{
    return NS(ManagedBufferHandle_get_required_num_dataptrs)(
        buffer, data_length_to_store );
}

bool NS(ManagedBufferHandle_can_be_added_ext)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const data_length_to_store,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_objects,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_slots,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_dataptr )
{
    return NS(ManagedBufferHandle_can_be_added)( buffer, data_length_to_store,
        requ_objects, requ_slots, requ_dataptr );
}

SIXTRL_BUFFER_DATAPTR_DEC NS(ManagedBufferHandle)*
NS(ManagedBufferHandle_new_ext)( SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)*
    SIXTRL_RESTRICT buffer, NS(buffer_size_t) const data_length )
{
    return NS(ManagedBufferHandle_new)( buffer, data_length );
}

SIXTRL_BUFFER_DATAPTR_DEC NS(ManagedBufferHandle)*
NS(ManagedBufferHandle_add_ext)( SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)*
    SIXTRL_RESTRICT buffer, NS(buffer_addr_t) const begin_addr,
    NS(buffer_size_t) const data_length,
    NS(buffer_size_t) const stored_slot_size,
    bool const perform_deep_copy )
{
    return NS(ManagedBufferHandle_add)(
        buffer, begin_addr, data_length, stored_slot_size, perform_deep_copy );
}

SIXTRL_BUFFER_DATAPTR_DEC NS(ManagedBufferHandle)*
NS(ManagedBufferHandle_new_from_existing_managed_buffer_ext)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT buffer_begin,
    NS(buffer_size_t) const slot_size,
    bool const perform_deep_copy )
{
    return NS(ManagedBufferHandle_new_from_existing_managed_buffer)(
        buffer, buffer_begin, slot_size, perform_deep_copy );
}

SIXTRL_BUFFER_DATAPTR_DEC NS(ManagedBufferHandle)*
NS(ManagedBufferHandle_add_copy_ext)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(ManagedBufferHandle) *const
        SIXTRL_RESTRICT existing_handle,
    bool const perform_deep_copy )
{
    return NS(ManagedBufferHandle_add_copy)(
        buffer, existing_handle, perform_deep_copy );
}

/* end: sixtracklib/common/buffer/managed_buffer_handle.c */
