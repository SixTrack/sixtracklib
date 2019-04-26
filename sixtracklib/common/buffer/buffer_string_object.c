#include "sixtracklib/common/buffer/buffer_string_object.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/buffer/buffer_type.h"
#include "sixtracklib/common/buffer.h"

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_BUFFER_DATAPTR_DEC NS(BufferStringObject)*
NS(BufferStringObject_preset_ext)( SIXTRL_BUFFER_DATAPTR_DEC
    NS(BufferStringObject)* SIXTRL_RESTRICT str_obj )
{
    return NS(BufferStringObject_preset)( str_obj );
}

NS(buffer_addr_t) NS(BufferStringObject_get_begin_addr_ext)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(BufferStringObject) *const
        SIXTRL_RESTRICT str_obj )
{
    return NS(BufferStringObject_get_begin_addr)( str_obj );
}

SIXTRL_BUFFER_DATAPTR_DEC char const*
NS(BufferStringObject_get_const_string_ext)( SIXTRL_BUFFER_DATAPTR_DEC const
    NS(BufferStringObject) *const SIXTRL_RESTRICT str_obj )
{
    return NS(BufferStringObject_get_const_string)( str_obj );
}

SIXTRL_BUFFER_DATAPTR_DEC char* NS(BufferStringObject_get_string_ext)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(BufferStringObject)* SIXTRL_RESTRICT str )
{
    return NS(BufferStringObject_get_string)( str );
}

NS(buffer_size_t) NS(BufferStringObject_get_capacity_ext)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(BufferStringObject)
        *const SIXTRL_RESTRICT str_obj )
{
    return NS(BufferStringObject_get_capacity)( str_obj );
}

NS(buffer_size_t) NS(BufferStringObject_get_length_ext)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(BufferStringObject) *const
        SIXTRL_RESTRICT str_obj )
{
    return NS(BufferStringObject_get_length)( str_obj );
}


SIXTRL_BUFFER_DATAPTR_DEC char const* NS(BufferStringObject_copy_cstring_ext)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(BufferStringObject)* SIXTRL_RESTRICT str_obj,
    SIXTRL_ARGPTR_DEC const char *const SIXTRL_RESTRICT source_str )
{
    return NS(BufferStringObject_copy_cstring)( str_obj, source_str );
}

SIXTRL_BUFFER_DATAPTR_DEC char const* NS(BufferStringObject_append_cstring_ext)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(BufferStringObject)* SIXTRL_RESTRICT str_obj,
    SIXTRL_ARGPTR_DEC const char *const SIXTRL_RESTRICT source_str )
{
    return NS(BufferStringObject_append_cstring)( str_obj, source_str );
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

NS(buffer_size_t) NS(BufferStringObject_get_required_num_slots_ext)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const capacity )
{
    return NS(BufferStringObject_get_required_num_slots)( buffer, capacity );
}

NS(buffer_size_t) NS(BufferStringObject_get_required_num_dataptrs_ext)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const capacity )
{
    return NS(BufferStringObject_get_required_num_dataptrs)( buffer, capacity );
}

bool NS(BufferStringObject_can_be_added_ext)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const capacity,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_objects,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_slots,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_dataptr )
{
    return NS(BufferStringObject_can_be_added)(
        buffer, capacity, requ_objects, requ_slots, requ_dataptr );
}

SIXTRL_BUFFER_DATAPTR_DEC NS(BufferStringObject)*
NS(BufferStringObject_new_ext)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const capacity )
{
    return NS(BufferStringObject_new)( buffer, capacity );
}

SIXTRL_BUFFER_DATAPTR_DEC NS(BufferStringObject)*
NS(BufferStringObject_add_ext)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_ARGPTR_DEC const char *const SIXTRL_RESTRICT source_str,
    NS(buffer_size_t) const capacity )
{
    return NS(BufferStringObject_add)( buffer, source_str, capacity );
}

SIXTRL_BUFFER_DATAPTR_DEC NS(BufferStringObject)*
NS(BufferStringObject_new_from_cstring_ext)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_ARGPTR_DEC const char *const SIXTRL_RESTRICT source_str )
{
    return NS(BufferStringObject_new_from_cstring)( buffer, source_str );
}

SIXTRL_BUFFER_DATAPTR_DEC NS(BufferStringObject)*
NS(BufferStringObject_add_copy_ext)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(BufferStringObject) *const
        SIXTRL_RESTRICT existing_str_obj )
{
    return NS(BufferStringObject_add_copy)( buffer, existing_str_obj );
}

/* end: sixtracklib/common/buffer/buffer_string_object.c */
