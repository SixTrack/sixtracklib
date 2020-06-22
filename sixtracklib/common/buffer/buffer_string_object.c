#include "sixtracklib/common/buffer/buffer_string_object.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/buffer/buffer_type.h"
#include "sixtracklib/common/buffer.h"

SIXTRL_BUFFER_DATAPTR_DEC NS(BufferStringObj) const*
NS(BufferStringObj_get_const_from_buffer_ext)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const index )
{
    return NS(BufferStringObj_get_const_from_buffer)( buffer, index );
}

SIXTRL_BUFFER_DATAPTR_DEC NS(BufferStringObj)*
NS(BufferStringObj_get_from_buffer_ext)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const index )
{
    return NS(BufferStringObj_get_from_buffer)( buffer, index );
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_BUFFER_DATAPTR_DEC NS(BufferStringObj) const*
NS(BufferStringObj_get_const_from_managed_buffer_ext)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT begin,
    NS(buffer_size_t) const index, NS(buffer_size_t) const slot_size )
{
    return NS(BufferStringObj_get_const_from_managed_buffer)(
        begin, index, slot_size );
}

SIXTRL_BUFFER_DATAPTR_DEC NS(BufferStringObj)*
NS(BufferStringObj_get_from_managed_buffer_ext)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT begin,
    NS(buffer_size_t) const index, NS(buffer_size_t) const slot_size )
{
    return NS(BufferStringObj_get_from_managed_buffer)(
        begin, index, slot_size );
}

/* ------------------------------------------------------------------------- */

SIXTRL_BUFFER_DATAPTR_DEC NS(BufferStringObj)*
NS(BufferStringObj_preset_ext)( SIXTRL_BUFFER_DATAPTR_DEC
    NS(BufferStringObj)* SIXTRL_RESTRICT str_obj )
{
    return NS(BufferStringObj_preset)( str_obj );
}

NS(buffer_addr_t) NS(BufferStringObj_get_begin_addr_ext)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(BufferStringObj) *const
        SIXTRL_RESTRICT str_obj )
{
    return NS(BufferStringObj_get_begin_addr)( str_obj );
}

SIXTRL_BUFFER_DATAPTR_DEC char const*
NS(BufferStringObj_get_const_string_ext)( SIXTRL_BUFFER_DATAPTR_DEC const
    NS(BufferStringObj) *const SIXTRL_RESTRICT str_obj )
{
    return NS(BufferStringObj_get_const_string)( str_obj );
}

SIXTRL_BUFFER_DATAPTR_DEC char* NS(BufferStringObj_get_string_ext)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(BufferStringObj)* SIXTRL_RESTRICT str )
{
    return NS(BufferStringObj_get_string)( str );
}

NS(buffer_size_t) NS(BufferStringObj_get_capacity_ext)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(BufferStringObj)
        *const SIXTRL_RESTRICT str_obj )
{
    return NS(BufferStringObj_get_capacity)( str_obj );
}

NS(buffer_size_t) NS(BufferStringObj_get_length_ext)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(BufferStringObj) *const
        SIXTRL_RESTRICT str_obj )
{
    return NS(BufferStringObj_get_length)( str_obj );
}

NS(buffer_size_t) NS(BufferStringObj_get_max_length_ext)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(BufferStringObj) *const
        SIXTRL_RESTRICT str_obj )
{
    return NS(BufferStringObj_get_max_length)( str_obj );
}

NS(buffer_size_t) NS(BufferStringObj_get_available_length_ext)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(BufferStringObj) *const
        SIXTRL_RESTRICT str_obj )
{
    return NS(BufferStringObj_get_available_length)( str_obj );
}

void NS(BufferStringObj_clear_ext)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(BufferStringObj)* SIXTRL_RESTRICT str_obj )
{
    NS(BufferStringObj_clear)( str_obj );
}

void NS(BufferStringObj_sync_length_ext)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(BufferStringObj)* SIXTRL_RESTRICT str_obj )
{
    NS(BufferStringObj_sync_length)( str_obj );
}

SIXTRL_BUFFER_DATAPTR_DEC char* NS(BufferStringObj_assign_cstring_ext)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(BufferStringObj)* SIXTRL_RESTRICT str_obj,
    SIXTRL_ARGPTR_DEC const char *const SIXTRL_RESTRICT source_str )
{
    return NS(BufferStringObj_assign_cstring)( str_obj, source_str );
}

SIXTRL_BUFFER_DATAPTR_DEC char* NS(BufferStringObj_append_cstring_ext)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(BufferStringObj)* SIXTRL_RESTRICT str_obj,
    SIXTRL_ARGPTR_DEC const char *const SIXTRL_RESTRICT source_str )
{
    return NS(BufferStringObj_append_cstring)( str_obj, source_str );
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

NS(buffer_size_t) NS(BufferStringObj_get_required_num_slots_ext)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const max_length )
{
    return NS(BufferStringObj_get_required_num_slots)( buffer, max_length );
}

NS(buffer_size_t) NS(BufferStringObj_get_required_num_dataptrs_ext)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const max_length )
{
    return NS(BufferStringObj_get_required_num_dataptrs)( buffer, max_length );
}

bool NS(BufferStringObj_can_be_added_ext)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const max_length,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_objects,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_slots,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_dataptr )
{
    return NS(BufferStringObj_can_be_added)(
        buffer, max_length, requ_objects, requ_slots, requ_dataptr );
}

SIXTRL_BUFFER_DATAPTR_DEC NS(BufferStringObj)*
NS(BufferStringObj_new_ext)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const max_length )
{
    return NS(BufferStringObj_new)( buffer, max_length );
}

SIXTRL_BUFFER_DATAPTR_DEC NS(BufferStringObj)*
NS(BufferStringObj_add_ext)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_ARGPTR_DEC const char *const SIXTRL_RESTRICT source_str,
    NS(buffer_size_t) const max_length )
{
    return NS(BufferStringObj_add)( buffer, source_str, max_length );
}

SIXTRL_BUFFER_DATAPTR_DEC NS(BufferStringObj)*
NS(BufferStringObj_add_detailed_ext)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_addr_t) const begin_addr, NS(buffer_size_t) const length,
    NS(buffer_size_t) const capacity )
{
    return NS(BufferStringObj_add_detailed)(
        buffer, begin_addr, length, capacity );
}

SIXTRL_BUFFER_DATAPTR_DEC NS(BufferStringObj)*
NS(BufferStringObj_new_from_cstring_ext)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_ARGPTR_DEC const char *const SIXTRL_RESTRICT source_str )
{
    return NS(BufferStringObj_new_from_cstring)( buffer, source_str );
}

SIXTRL_BUFFER_DATAPTR_DEC NS(BufferStringObj)*
NS(BufferStringObj_add_assign_ext)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(BufferStringObj) *const
        SIXTRL_RESTRICT existing_str_obj )
{
    return NS(BufferStringObj_add_copy)( buffer, existing_str_obj );
}

/* end: sixtracklib/common/buffer/buffer_string_object.c */
