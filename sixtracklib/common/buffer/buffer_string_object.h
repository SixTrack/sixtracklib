#ifndef SIXTRACKLIB_COMMON_BUFFER_BUFFER_STRING_OBJECT_H__
#define SIXTRACKLIB_COMMON_BUFFER_BUFFER_STRING_OBJECT_H__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stdbool.h>
    #include <stddef.h>
    #include <stdlib.h>
    #include <string.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/buffer/buffer_type.h"
    #include "sixtracklib/common/internal/objects_type_id.h"
    #if !defined( _GPUCODE )
        #include "sixtracklib/common/buffer.h"
    #endif /* !defined( _GPUCODE ) */
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus ) && !defined( _GPUCODE )
extern "C" {
#endif /* defined( __cplusplus ) && !defined( _GPUCODE ) */

typedef struct NS(BufferStringObj)
{
    NS(buffer_addr_t) begin_addr SIXTRL_ALIGN( 8u );
    NS(buffer_size_t) length     SIXTRL_ALIGN( 8u );
    NS(buffer_size_t) capacity   SIXTRL_ALIGN( 8u );
}
NS(BufferStringObj);

/* ------------------------------------------------------------------------- */

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t)
NS(BufferStringObj_get_cstring_length)(
    SIXTRL_ARGPTR_DEC const char *const SIXTRL_RESTRICT str );

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t)
NS(BufferStringObj_get_cstring_length_reverse)(
    SIXTRL_ARGPTR_DEC const char *const SIXTRL_RESTRICT str,
    NS(buffer_size_t) const capacity );

/* ------------------------------------------------------------------------- */

SIXTRL_STATIC SIXTRL_FN SIXTRL_BUFFER_DATAPTR_DEC NS(BufferStringObj) const*
NS(BufferStringObj_get_const_from_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT begin,
    NS(buffer_size_t) const index, NS(buffer_size_t) const slot_size );

SIXTRL_STATIC SIXTRL_FN SIXTRL_BUFFER_DATAPTR_DEC NS(BufferStringObj)*
NS(BufferStringObj_get_from_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT begin,
    NS(buffer_size_t) const index, NS(buffer_size_t) const slot_size );

/* ------------------------------------------------------------------------- */

SIXTRL_STATIC SIXTRL_FN SIXTRL_BUFFER_DATAPTR_DEC NS(BufferStringObj)*
NS(BufferStringObj_preset)( SIXTRL_BUFFER_DATAPTR_DEC
    NS(BufferStringObj)* SIXTRL_RESTRICT string );

SIXTRL_STATIC SIXTRL_FN NS(buffer_addr_t)
NS(BufferStringObj_get_begin_addr)( SIXTRL_BUFFER_DATAPTR_DEC
    const NS(BufferStringObj) *const SIXTRL_RESTRICT str_obj );

SIXTRL_STATIC SIXTRL_FN SIXTRL_BUFFER_DATAPTR_DEC char const*
NS(BufferStringObj_get_const_string)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(BufferStringObj) *const
        SIXTRL_RESTRICT str_obj );

SIXTRL_STATIC SIXTRL_FN SIXTRL_BUFFER_DATAPTR_DEC char*
NS(BufferStringObj_get_string)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(BufferStringObj)* SIXTRL_RESTRICT str );

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t)
NS(BufferStringObj_get_capacity)( SIXTRL_BUFFER_DATAPTR_DEC
    const NS(BufferStringObj) *const SIXTRL_RESTRICT str_obj );

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t)
NS(BufferStringObj_get_length)( SIXTRL_BUFFER_DATAPTR_DEC
    const NS(BufferStringObj) *const SIXTRL_RESTRICT str_obj );

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t)
NS(BufferStringObj_get_max_length)( SIXTRL_BUFFER_DATAPTR_DEC
    const NS(BufferStringObj) *const SIXTRL_RESTRICT str_obj );

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t)
NS(BufferStringObj_get_available_length)( SIXTRL_BUFFER_DATAPTR_DEC
    const NS(BufferStringObj) *const SIXTRL_RESTRICT str_obj );


SIXTRL_STATIC SIXTRL_FN void NS(BufferStringObj_clear)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(BufferStringObj)* SIXTRL_RESTRICT str_obj );

SIXTRL_STATIC SIXTRL_FN void NS(BufferStringObj_sync_length)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(BufferStringObj)* SIXTRL_RESTRICT str_obj );

SIXTRL_STATIC SIXTRL_FN SIXTRL_BUFFER_DATAPTR_DEC char*
NS(BufferStringObj_assign_cstring)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(BufferStringObj)* SIXTRL_RESTRICT str_obj,
    SIXTRL_ARGPTR_DEC const char *const SIXTRL_RESTRICT source_str );

SIXTRL_STATIC SIXTRL_FN SIXTRL_BUFFER_DATAPTR_DEC char*
NS(BufferStringObj_append_cstring)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(BufferStringObj)* SIXTRL_RESTRICT str_obj,
    SIXTRL_ARGPTR_DEC const char *const SIXTRL_RESTRICT source_str );


SIXTRL_STATIC SIXTRL_FN void NS(BufferStringObj_set_begin_addr)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(BufferStringObj)* SIXTRL_RESTRICT str_obj,
    NS(buffer_addr_t) const begin_addr );

SIXTRL_STATIC SIXTRL_FN void NS(BufferStringObj_set_max_length)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(BufferStringObj)* SIXTRL_RESTRICT str_obj,
    NS(buffer_size_t) const max_length );

SIXTRL_STATIC SIXTRL_FN void NS(BufferStringObj_set_capacity)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(BufferStringObj)* SIXTRL_RESTRICT str_obj,
    NS(buffer_size_t) const capacity );

SIXTRL_STATIC SIXTRL_FN void NS(BufferStringObj_set_length)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(BufferStringObj)* SIXTRL_RESTRICT str_obj,
    NS(buffer_size_t) const length );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

#if !defined( _GPUCODE )

SIXTRL_STATIC SIXTRL_FN SIXTRL_BUFFER_DATAPTR_DEC NS(BufferStringObj) const*
NS(BufferStringObj_get_const_from_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const index );

SIXTRL_STATIC SIXTRL_FN SIXTRL_BUFFER_DATAPTR_DEC NS(BufferStringObj)*
NS(BufferStringObj_get_from_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const index );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BUFFER_DATAPTR_DEC NS(BufferStringObj) const*
NS(BufferStringObj_get_const_from_buffer_ext)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const index );

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BUFFER_DATAPTR_DEC NS(BufferStringObj)*
NS(BufferStringObj_get_from_buffer_ext)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const index );

/* ------------------------------------------------------------------------- */

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BUFFER_DATAPTR_DEC NS(BufferStringObj) const*
NS(BufferStringObj_get_const_from_managed_buffer_ext)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT begin,
    NS(buffer_size_t) const index, NS(buffer_size_t) const slot_size );

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BUFFER_DATAPTR_DEC NS(BufferStringObj)*
NS(BufferStringObj_get_from_managed_buffer_ext)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT begin,
    NS(buffer_size_t) const index, NS(buffer_size_t) const slot_size );

/* ------------------------------------------------------------------------- */

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BUFFER_DATAPTR_DEC NS(BufferStringObj)*
NS(BufferStringObj_preset_ext)( SIXTRL_BUFFER_DATAPTR_DEC
    NS(BufferStringObj)* SIXTRL_RESTRICT string );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(buffer_addr_t)
NS(BufferStringObj_get_begin_addr_ext)( SIXTRL_BUFFER_DATAPTR_DEC
    const NS(BufferStringObj) *const SIXTRL_RESTRICT str_obj );

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BUFFER_DATAPTR_DEC char const*
NS(BufferStringObj_get_const_string_ext)( SIXTRL_BUFFER_DATAPTR_DEC const
    NS(BufferStringObj) *const SIXTRL_RESTRICT str_obj );

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BUFFER_DATAPTR_DEC char*
NS(BufferStringObj_get_string_ext)( SIXTRL_BUFFER_DATAPTR_DEC
    NS(BufferStringObj)* SIXTRL_RESTRICT str );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(buffer_size_t)
NS(BufferStringObj_get_capacity_ext)( SIXTRL_BUFFER_DATAPTR_DEC
    const NS(BufferStringObj) *const SIXTRL_RESTRICT str_obj );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(buffer_size_t)
NS(BufferStringObj_get_length_ext)( SIXTRL_BUFFER_DATAPTR_DEC
    const NS(BufferStringObj) *const SIXTRL_RESTRICT str_obj );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(buffer_size_t)
NS(BufferStringObj_get_max_length_ext)( SIXTRL_BUFFER_DATAPTR_DEC
    const NS(BufferStringObj) *const SIXTRL_RESTRICT str_obj );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(buffer_size_t)
NS(BufferStringObj_get_available_length_ext)( SIXTRL_BUFFER_DATAPTR_DEC
    const NS(BufferStringObj) *const SIXTRL_RESTRICT str_obj );

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(BufferStringObj_clear_ext)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(BufferStringObj)* SIXTRL_RESTRICT str_obj );

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(BufferStringObj_sync_length_ext)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(BufferStringObj)* SIXTRL_RESTRICT str_obj );

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BUFFER_DATAPTR_DEC char*
NS(BufferStringObj_assign_cstring_ext)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(BufferStringObj)* SIXTRL_RESTRICT str_obj,
    SIXTRL_ARGPTR_DEC const char *const SIXTRL_RESTRICT source_str );

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BUFFER_DATAPTR_DEC char*
NS(BufferStringObj_append_cstring_ext)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(BufferStringObj)* SIXTRL_RESTRICT str_obj,
    SIXTRL_ARGPTR_DEC const char *const SIXTRL_RESTRICT source_str );

#endif /* !defined( _GPUCODE ) */

/* ------------------------------------------------------------------------ */

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t)
NS(BufferStringObj_get_required_num_slots_on_managed_buffer)(
    NS(buffer_size_t) const max_length, NS(buffer_size_t) const slot_size );

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t)
NS(BufferStringObj_get_required_num_dataptrs_on_managed_buffer)(
    NS(buffer_size_t) const max_length, NS(buffer_size_t) const slot_size );

#if !defined( _GPUCODE )

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t)
NS(BufferStringObj_get_required_num_slots)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const max_length );

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t)
NS(BufferStringObj_get_required_num_dataptrs)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const max_length );

SIXTRL_STATIC SIXTRL_FN bool NS(BufferStringObj_can_be_added)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const max_length,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_objects,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_slots,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_dataptr );

SIXTRL_STATIC SIXTRL_FN SIXTRL_BUFFER_DATAPTR_DEC NS(BufferStringObj)*
NS(BufferStringObj_new)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const max_length );

SIXTRL_STATIC SIXTRL_FN SIXTRL_BUFFER_DATAPTR_DEC NS(BufferStringObj)*
NS(BufferStringObj_add)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_ARGPTR_DEC const char *const SIXTRL_RESTRICT source_str,
    NS(buffer_size_t) const max_length );

SIXTRL_STATIC SIXTRL_FN SIXTRL_BUFFER_DATAPTR_DEC NS(BufferStringObj)*
NS(BufferStringObj_add_detailed)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_addr_t) const begin_addr, NS(buffer_size_t) const length,
    NS(buffer_size_t) const capacity );

SIXTRL_STATIC SIXTRL_FN SIXTRL_BUFFER_DATAPTR_DEC NS(BufferStringObj)*
NS(BufferStringObj_new_from_cstring)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_ARGPTR_DEC const char *const SIXTRL_RESTRICT source_str );

SIXTRL_STATIC SIXTRL_FN SIXTRL_BUFFER_DATAPTR_DEC NS(BufferStringObj)*
NS(BufferStringObj_add_copy)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(BufferStringObj) *const
        SIXTRL_RESTRICT existing_str_obj );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_EXTERN SIXTRL_HOST_FN NS(buffer_size_t)
NS(BufferStringObj_get_required_num_slots_on_managed_buffer_ext)(
    NS(buffer_size_t) const max_length, NS(buffer_size_t) const slot_size );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(buffer_size_t)
NS(BufferStringObj_get_required_num_dataptrs_on_managed_buffer_ext)(
    NS(buffer_size_t) const max_length, NS(buffer_size_t) const slot_size );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(buffer_size_t)
NS(BufferStringObj_get_required_num_slots_ext)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const max_length );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(buffer_size_t)
NS(BufferStringObj_get_required_num_dataptrs_ext)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const max_length );

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(BufferStringObj_can_be_added_ext)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const max_length,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_objects,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_slots,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_dataptr );

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BUFFER_DATAPTR_DEC NS(BufferStringObj)*
NS(BufferStringObj_new_ext)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const max_length );

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BUFFER_DATAPTR_DEC NS(BufferStringObj)*
NS(BufferStringObj_add_ext)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_ARGPTR_DEC const char *const SIXTRL_RESTRICT source_str,
    NS(buffer_size_t) const max_length );

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BUFFER_DATAPTR_DEC NS(BufferStringObj)*
NS(BufferStringObj_add_detailed_ext)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_addr_t) const begin_addr, NS(buffer_size_t) const length,
    NS(buffer_size_t) const capacity );

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BUFFER_DATAPTR_DEC NS(BufferStringObj)*
NS(BufferStringObj_new_from_cstring_ext)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_ARGPTR_DEC const char *const SIXTRL_RESTRICT source_str );

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BUFFER_DATAPTR_DEC NS(BufferStringObj)*
NS(BufferStringObj_add_assign_ext)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(BufferStringObj) *const
        SIXTRL_RESTRICT existing_str_obj );

#endif /* !defined( _GPUCODE ) */

/* ************************************************************************* */
/* *********         Implementation of Inline Functions          *********** */
/* ************************************************************************* */

SIXTRL_INLINE NS(buffer_size_t) NS(BufferStringObj_get_cstring_length)(
    SIXTRL_BUFFER_DATAPTR_DEC const char *const SIXTRL_RESTRICT input_str )
{
    typedef NS(buffer_size_t) buf_size_t;

    buf_size_t len = ( buf_size_t )0u;

    if( input_str != SIXTRL_NULLPTR )
    {
        #if !defined( _GPUCODE )
        len = strlen( input_str );

        #else /* defined( _GPUCODE ) */
        SIXTRL_BUFFER_DATAPTR_DEC char const* it = input_str;
        while( *it != '\0' ) ++it;
        len = ( buf_size_t )( it - input_str );

        #endif /* !defined( _GPUCODE ) */
    }

    return len;
}

SIXTRL_INLINE NS(buffer_size_t)
NS(BufferStringObj_get_cstring_length_reverse)(
    SIXTRL_ARGPTR_DEC const char *const SIXTRL_RESTRICT input_str,
    NS(buffer_size_t) const capacity )
{
    typedef NS(buffer_size_t) buf_size_t;

    buf_size_t len = ( buf_size_t )0u;

    if( ( input_str != SIXTRL_NULLPTR ) && ( capacity > ( buf_size_t )0u ) )
    {
        buf_size_t current_length = capacity - ( buf_size_t )1u;

        while( ( current_length > ( buf_size_t )0u ) &&
               ( input_str[ current_length ]  == '\0' ) )
        {
            --current_length;
        }

        len = current_length;
    }

    return len;
}

/* ------------------------------------------------------------------------- */


SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(BufferStringObj) const*
NS(BufferStringObj_get_const_from_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT begin,
    NS(buffer_size_t) const index, NS(buffer_size_t) const slot_size )
{
    typedef SIXTRL_BUFFER_DATAPTR_DEC NS(BufferStringObj) const* ptr_strobj_t;

    return ( ptr_strobj_t )( uintptr_t
        )NS(ManagedBuffer_get_object_begin_addr_by_index_filter_by_type_id)(
            begin, index, NS(OBJECT_TYPE_CSTRING), slot_size );
}

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(BufferStringObj)*
NS(BufferStringObj_get_from_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT begin,
    NS(buffer_size_t) const index, NS(buffer_size_t) const slot_size )
{
    typedef SIXTRL_BUFFER_DATAPTR_DEC NS(BufferStringObj)* ptr_strobj_t;
    return ( ptr_strobj_t )NS(BufferStringObj_get_const_from_managed_buffer)(
            begin, index, slot_size );
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(BufferStringObj)*
NS(BufferStringObj_preset)( SIXTRL_BUFFER_DATAPTR_DEC
    NS(BufferStringObj)* SIXTRL_RESTRICT str_obj )
{
    if( str_obj != SIXTRL_NULLPTR )
    {
        str_obj->begin_addr = ( NS(buffer_addr_t) )0u;
        str_obj->length     = ( NS(buffer_size_t) )0u;
        str_obj->capacity   = ( NS(buffer_size_t) )0u;
    }

    return str_obj;
}

SIXTRL_INLINE NS(buffer_addr_t) NS(BufferStringObj_get_begin_addr)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(BufferStringObj) *const
        SIXTRL_RESTRICT str_obj )
{
    SIXTRL_ASSERT( str_obj != SIXTRL_NULLPTR );
    return str_obj->begin_addr;
}

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC char const*
NS(BufferStringObj_get_const_string)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(BufferStringObj) *const
        SIXTRL_RESTRICT str_obj )
{
    typedef SIXTRL_BUFFER_DATAPTR_DEC char const* ptr_cstring_t;
    SIXTRL_ASSERT( str_obj != SIXTRL_NULLPTR );
    return ( ptr_cstring_t )( uintptr_t )str_obj->begin_addr;
}

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC char*
NS(BufferStringObj_get_string)( SIXTRL_BUFFER_DATAPTR_DEC
    NS(BufferStringObj)* SIXTRL_RESTRICT str_obj )
{
    return ( SIXTRL_BUFFER_DATAPTR_DEC char*
        )NS(BufferStringObj_get_const_string)( str_obj );
}

SIXTRL_INLINE NS(buffer_size_t) NS(BufferStringObj_get_capacity)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(BufferStringObj) *const
        SIXTRL_RESTRICT str_obj )
{
    SIXTRL_ASSERT( str_obj != SIXTRL_NULLPTR );
    return str_obj->capacity;
}

SIXTRL_INLINE NS(buffer_size_t) NS(BufferStringObj_get_length)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(BufferStringObj) *const
        SIXTRL_RESTRICT str_obj )
{
    typedef NS(buffer_size_t) buf_size_t;

    buf_size_t length = ( buf_size_t )0u;

    if( NS(BufferStringObj_get_begin_addr)( str_obj ) >
        ( NS(buffer_addr_t) )0u )
    {
        SIXTRL_ASSERT( str_obj != SIXTRL_NULLPTR );
        SIXTRL_ASSERT( NS(BufferStringObj_get_capacity)( str_obj ) >
            ( buf_size_t )0u );

        if( str_obj->length < str_obj->capacity )
        {
            length = str_obj->length;

            SIXTRL_ASSERT(
                ( ( length < ( buf_size_t )1024 ) &&
                  ( NS(BufferStringObj_get_cstring_length)(
                    NS(BufferStringObj_get_const_string)( str_obj ) ) ==
                        length ) ) ||
                ( ( length >= ( buf_size_t )1024 ) &&
                  ( NS(BufferStringObj_get_cstring_length_reverse)(
                      NS(BufferStringObj_get_const_string)( str_obj ),
                      NS(BufferStringObj_get_capacity)( str_obj ) ) ==
                        length ) ) );
        }
    }

    return length;
}

SIXTRL_INLINE NS(buffer_size_t) NS(BufferStringObj_get_max_length)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(BufferStringObj) *const
        SIXTRL_RESTRICT str_obj )
{
    NS(buffer_size_t) max_length = ( NS(buffer_size_t) )0u;
    NS(buffer_size_t) const capacity =
        NS(BufferStringObj_get_capacity)( str_obj );

    SIXTRL_ASSERT( capacity > NS(BufferStringObj_get_length)( str_obj ) );

    if( ( capacity > ( NS(buffer_size_t) )1u ) &&
        ( NS(BufferStringObj_get_begin_addr)( str_obj ) >
            ( NS(buffer_addr_t) )0u ) )
    {
        max_length = capacity - ( NS(buffer_size_t) )1u;
    }

    return max_length;
}

SIXTRL_INLINE NS(buffer_size_t)NS(BufferStringObj_get_available_length)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(BufferStringObj) *const
        SIXTRL_RESTRICT str_obj )
{
    typedef NS(buffer_size_t) size_t;
    typedef NS(buffer_addr_t) addr_t;

    size_t available_length = ( size_t )0u;
    size_t const capacity = NS(BufferStringObj_get_capacity)( str_obj );
    size_t const current_requ_capacity =
        NS(BufferStringObj_get_length)( str_obj ) + ( size_t )1u;

    if( ( capacity > current_requ_capacity ) &&
        ( ( addr_t )0u < NS(BufferStringObj_get_begin_addr)( str_obj ) ) )
    {
        available_length = capacity - current_requ_capacity;
    }

    return available_length;
}

SIXTRL_INLINE void NS(BufferStringObj_clear)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(BufferStringObj)* SIXTRL_RESTRICT str_obj )
{
    typedef NS(buffer_size_t) size_t;
    typedef NS(buffer_addr_t) addr_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC char* ptr_cstring_t;

    size_t const length = NS(BufferStringObj_get_length)( str_obj );
    size_t const capacity = NS(BufferStringObj_get_capacity)( str_obj );

    if( ( str_obj != SIXTRL_NULLPTR ) &&
        ( NS(BufferStringObj_get_begin_addr)( str_obj ) > ( addr_t )0u ) &&
        ( length > ( size_t )0u ) && ( length < capacity ) )
    {
        ptr_cstring_t str_begin = NS(BufferStringObj_get_string)( str_obj );
        SIXTRL_ASSERT( str_begin != SIXTRL_NULLPTR );

        SIXTRACKLIB_SET_VALUES( char, str_begin, length, '\0' );
        NS(BufferStringObj_set_length)( str_obj, ( size_t )0u );
    }
}

SIXTRL_INLINE void NS(BufferStringObj_sync_length)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(BufferStringObj)* SIXTRL_RESTRICT str_obj )
{
    typedef NS(buffer_size_t) size_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC char const* ptr_cstring_t;

    size_t const capacity = NS(BufferStringObj_get_capacity)( str_obj );
    ptr_cstring_t cstr = NS(BufferStringObj_get_const_string)( str_obj );

    if( ( str_obj != SIXTRL_NULLPTR ) && ( cstr != SIXTRL_NULLPTR ) &&
        ( capacity > ( size_t )0u ) )
    {
        size_t const updated_length =
            NS(BufferStringObj_get_cstring_length)( cstr );

        if( updated_length < capacity )
        {
            NS(BufferStringObj_set_length)( str_obj, updated_length );
        }
    }
}

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC char*
NS(BufferStringObj_assign_cstring)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(BufferStringObj)* SIXTRL_RESTRICT str_obj,
    SIXTRL_ARGPTR_DEC const char *const SIXTRL_RESTRICT source_str )
{
    typedef NS(buffer_size_t) buf_size_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC char* ptr_cstring_t;

    ptr_cstring_t copied_str = SIXTRL_NULLPTR;
    ptr_cstring_t stored_str = NS(BufferStringObj_get_string)( str_obj );
    buf_size_t const capacity = NS(BufferStringObj_get_capacity)( str_obj );

    if( ( source_str != SIXTRL_NULLPTR ) && ( stored_str != SIXTRL_NULLPTR ) &&
        ( capacity > ( buf_size_t )1u ) )
    {
        buf_size_t const source_str_len =
            NS(BufferStringObj_get_cstring_length)( source_str );

        if( source_str_len < capacity )
        {
            buf_size_t const remaining_chars = capacity - source_str_len;

            if( source_str_len > ( buf_size_t )0u )
            {
                SIXTRACKLIB_COPY_VALUES(
                    char, stored_str, source_str, source_str_len );
            }

            SIXTRACKLIB_SET_VALUES( char, stored_str + source_str_len,
                                    remaining_chars, '\0' );

            NS(BufferStringObj_set_length)( str_obj, source_str_len );
            copied_str = stored_str;
        }
    }

    return copied_str;
}

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC char*
NS(BufferStringObj_append_cstring)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(BufferStringObj)* SIXTRL_RESTRICT str_obj,
    SIXTRL_ARGPTR_DEC const char *const SIXTRL_RESTRICT source_str )
{
    typedef NS(buffer_size_t) buf_size_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC char* ptr_cstring_t;

    ptr_cstring_t appended_str = SIXTRL_NULLPTR;
    ptr_cstring_t stored_str = NS(BufferStringObj_get_string)( str_obj );
    buf_size_t const capacity = NS(BufferStringObj_get_capacity)( str_obj );

    buf_size_t const current_length =
        NS(BufferStringObj_get_length)( str_obj );

    if( ( source_str != SIXTRL_NULLPTR ) && ( stored_str != SIXTRL_NULLPTR ) &&
        ( capacity > current_length ) )
    {
        buf_size_t const source_str_len =
            NS(BufferStringObj_get_cstring_length)( source_str );

        buf_size_t const new_length = current_length + source_str_len;

        if( new_length < capacity )
        {
            buf_size_t const remaining_chars = capacity - new_length;

            if( source_str_len > ( buf_size_t )0u )
            {
                SIXTRACKLIB_COPY_VALUES( char, stored_str + current_length,
                                         source_str, source_str_len );
            }

            SIXTRACKLIB_SET_VALUES( char, stored_str + new_length,
                                    remaining_chars, '\0' );

            NS(BufferStringObj_set_length)( str_obj, new_length );
            appended_str = stored_str;
        }
    }

    return appended_str;
}


SIXTRL_INLINE void NS(BufferStringObj_set_begin_addr)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(BufferStringObj)* SIXTRL_RESTRICT str_obj,
    NS(buffer_addr_t) const begin_addr )
{
    SIXTRL_ASSERT( str_obj != SIXTRL_NULLPTR );
    str_obj->begin_addr = begin_addr;
}

SIXTRL_INLINE void NS(BufferStringObj_set_capacity)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(BufferStringObj)* SIXTRL_RESTRICT str_obj,
    NS(buffer_size_t) const capacity )
{
    SIXTRL_ASSERT( str_obj != SIXTRL_NULLPTR );
    str_obj->capacity = capacity;
}

SIXTRL_INLINE void NS(BufferStringObj_set_length)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(BufferStringObj)* SIXTRL_RESTRICT str_obj,
    NS(buffer_size_t) const length )
{
    SIXTRL_ASSERT( str_obj != SIXTRL_NULLPTR );
    str_obj->length = length;
}

SIXTRL_INLINE void NS(BufferStringObj_set_max_length)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(BufferStringObj)* SIXTRL_RESTRICT str_obj,
    NS(buffer_size_t) const max_length )
{
    SIXTRL_ASSERT( str_obj != SIXTRL_NULLPTR );
    str_obj->capacity = max_length + ( NS(buffer_size_t) )1u;
}

/* ------------------------------------------------------------------------ */

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(BufferStringObj) const*
NS(BufferStringObj_get_const_from_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const index )
{
    return NS(BufferStringObj_get_const_from_managed_buffer)(
        NS(Buffer_get_const_data_begin)( buffer ), index,
        NS(Buffer_get_slot_size)( buffer ) );
}

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(BufferStringObj)*
NS(BufferStringObj_get_from_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const index )
{
    return NS(BufferStringObj_get_from_managed_buffer)(
        NS(Buffer_get_data_begin)( buffer ), index,
        NS(Buffer_get_slot_size)( buffer ) );
}

/* ------------------------------------------------------------------------ */

SIXTRL_INLINE NS(buffer_size_t)
NS(BufferStringObj_get_required_num_slots_on_managed_buffer)(
    NS(buffer_size_t) const max_length, NS(buffer_size_t) const slot_size )
{
    typedef NS(buffer_size_t) buf_size_t;
    SIXTRL_STATIC_VAR buf_size_t const ZERO = ( buf_size_t )0u;

    buf_size_t const obj_size = sizeof( NS(BufferStringObj) );
    buf_size_t const capacity = max_length + ( buf_size_t )1u;

    buf_size_t const required_size =
        NS(ManagedBuffer_get_slot_based_length)( obj_size, slot_size ) +
        NS(ManagedBuffer_get_slot_based_length)( capacity, slot_size );

    SIXTRL_ASSERT( ( slot_size == ZERO ) ||
                   ( ( required_size % slot_size ) == ZERO ) );

    return ( slot_size > ZERO ) ? required_size / slot_size : ZERO;
}

SIXTRL_INLINE NS(buffer_size_t)
NS(BufferStringObj_get_required_num_dataptrs_on_managed_buffer)(
    NS(buffer_size_t) const max_length, NS(buffer_size_t) const slot_size )
{
    ( void )max_length;

    return ( slot_size > ( NS(buffer_size_t) )0u )
        ? ( NS(buffer_size_t) )1u : ( NS(buffer_size_t) )0u;
}


SIXTRL_INLINE NS(buffer_size_t) NS(BufferStringObj_get_required_num_slots)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const max_length )
{
    return NS(BufferStringObj_get_required_num_slots_on_managed_buffer)(
        max_length, NS(Buffer_get_slot_size)( buffer ) );
}

SIXTRL_INLINE NS(buffer_size_t)
NS(BufferStringObj_get_required_num_dataptrs)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const max_length )
{
    return NS(BufferStringObj_get_required_num_dataptrs_on_managed_buffer)(
        max_length, NS(Buffer_get_slot_size)( buffer ) );
}

SIXTRL_INLINE bool NS(BufferStringObj_can_be_added)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const max_length,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_objects,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_slots,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_dataptrs )
{
    typedef NS(buffer_size_t) buf_size_t;

    buf_size_t const num_dataptrs =
        NS(BufferStringObj_get_required_num_dataptrs)( buffer, max_length );

    buf_size_t const capacity = max_length + ( buf_size_t )1u;
    buf_size_t const counts[] = { capacity };
    buf_size_t const sizes[]  = { sizeof( char ) };
    buf_size_t const obj_size = sizeof( NS(BufferStringObj) );

    return NS(Buffer_can_add_object)( buffer, obj_size, num_dataptrs,
        &sizes[ 0 ], &counts[ 0 ], requ_objects, requ_slots, requ_dataptrs );
}

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(BufferStringObj)*
NS(BufferStringObj_new)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const max_length )
{
    typedef NS(BufferStringObj) str_obj_t;
    typedef NS(buffer_size_t) buf_size_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC str_obj_t* ptr_str_obj_t;

    SIXTRL_STATIC_VAR buf_size_t const ZERO = ( buf_size_t )0u;

    buf_size_t const num_dataptrs =
        NS(BufferStringObj_get_required_num_dataptrs)( buffer, max_length );

    buf_size_t const capacity  = max_length + ( buf_size_t )1u;
    buf_size_t const counts[]  = { capacity };
    buf_size_t const sizes[]   = { sizeof( char ) };
    buf_size_t const offsets[] = { offsetof( str_obj_t, begin_addr ) };

    str_obj_t str_obj;
    str_obj.begin_addr = ( NS(buffer_addr_t) )0u;
    str_obj.length     = ZERO;
    str_obj.capacity   = capacity;

    return ( ptr_str_obj_t )( uintptr_t )NS(Object_get_begin_addr)(
        NS(Buffer_add_object)( buffer, &str_obj, sizeof( str_obj_t ),
            NS(OBJECT_TYPE_CSTRING), num_dataptrs,
                &offsets[ 0 ], &sizes[ 0 ], &counts[ 0 ] ) );
}

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(BufferStringObj)*
NS(BufferStringObj_add)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_ARGPTR_DEC const char *const SIXTRL_RESTRICT input_str,
    NS(buffer_size_t) const max_length )
{
    typedef NS(buffer_size_t) buf_size_t;

    typedef NS(BufferStringObj) str_obj_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC str_obj_t* ptr_str_obj_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC char* ptr_cstring_t;

    ptr_str_obj_t str_obj = SIXTRL_NULLPTR;

    buf_size_t const capacity = max_length + ( buf_size_t )1u;
    buf_size_t const input_str_len = ( input_str != SIXTRL_NULLPTR )
        ? NS(BufferStringObj_get_cstring_length)( input_str )
        : ( buf_size_t )0u;

    if( ( input_str != SIXTRL_NULLPTR ) &&
        ( input_str_len > ( buf_size_t )0u ) && ( capacity >= input_str_len ) )
    {
        str_obj = NS(BufferStringObj_new)( buffer, max_length );

        if( str_obj != SIXTRL_NULLPTR )
        {
            buf_size_t const remaining_chars = capacity - input_str_len;
            ptr_cstring_t out = NS(BufferStringObj_get_string)( str_obj );

            SIXTRL_ASSERT( out != SIXTRL_NULLPTR );
            SIXTRACKLIB_COPY_VALUES( char, out, input_str, input_str_len );

            SIXTRACKLIB_SET_VALUES(  char, out + input_str_len,
                                     remaining_chars, '\0' );

            NS(BufferStringObj_set_length)( str_obj, input_str_len );
        }
    }
    else if( ( capacity > ( buf_size_t )0u ) &&
             ( ( input_str == SIXTRL_NULLPTR ) ||
               ( input_str_len == ( buf_size_t )0u ) ) )
    {
        str_obj = NS(BufferStringObj_new)( buffer, max_length );

        if( str_obj != SIXTRL_NULLPTR )
        {
            ptr_cstring_t out = NS(BufferStringObj_get_string)( str_obj );

            SIXTRL_ASSERT( out != SIXTRL_NULLPTR );
            SIXTRACKLIB_SET_VALUES(  char, out, capacity, '\0' );
        }
    }

    if( str_obj != SIXTRL_NULLPTR )
    {
        NS(BufferStringObj_set_capacity)( str_obj, capacity );
    }

    return str_obj;
}

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(BufferStringObj)*
NS(BufferStringObj_add_detailed)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_addr_t) const begin_addr, NS(buffer_size_t) const length,
    NS(buffer_size_t) const capacity )
{
    typedef NS(BufferStringObj) str_obj_t;
    typedef NS(buffer_size_t) buf_size_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC str_obj_t* ptr_str_obj_t;

    SIXTRL_STATIC_VAR buf_size_t const ZERO = ( buf_size_t )0u;

    buf_size_t const max_length = ( capacity > ZERO )
        ? capacity - ( buf_size_t )1u : ZERO;

    buf_size_t const num_dataptrs =
        NS(BufferStringObj_get_required_num_dataptrs)( buffer, max_length );

    buf_size_t const counts[]  = { capacity };
    buf_size_t const sizes[]   = { sizeof( char ) };
    buf_size_t const offsets[] = { offsetof( str_obj_t, begin_addr ) };

    str_obj_t str_obj;
    str_obj.begin_addr = begin_addr;
    str_obj.length     = length;
    str_obj.capacity   = capacity;

    return ( ptr_str_obj_t )( uintptr_t )NS(Object_get_begin_addr)(
        NS(Buffer_add_object)( buffer, &str_obj, sizeof( str_obj_t ),
            NS(OBJECT_TYPE_CSTRING), num_dataptrs,
                &offsets[ 0 ], &sizes[ 0 ], &counts[ 0 ] ) );
}

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(BufferStringObj)*
NS(BufferStringObj_new_from_cstring)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_ARGPTR_DEC const char *const SIXTRL_RESTRICT input_str )
{
    SIXTRL_STATIC_VAR NS(buffer_size_t) ZERO = ( NS(buffer_size_t) )0u;

    NS(buffer_size_t) const max_length = ( input_str != SIXTRL_NULLPTR )
        ? NS(BufferStringObj_get_cstring_length)( input_str ) : ZERO;

    return NS(BufferStringObj_add)( buffer, input_str, max_length );
}

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(BufferStringObj)*
NS(BufferStringObj_add_copy)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(BufferStringObj) *const
        SIXTRL_RESTRICT str_obj )
{
    return NS(BufferStringObj_add)( buffer,
        NS(BufferStringObj_get_const_string)( str_obj ),
        NS(BufferStringObj_get_capacity)( str_obj ) );
}

#if defined( __cplusplus ) && !defined( _GPUCODE )
}
#endif /* defined( __cplusplus ) && !defined( _GPUCODE ) */

#endif /* SIXTRACKLIB_COMMON_BUFFER_BUFFER_STRING_OBJECT_H__ */

/* end: sixtracklib/common/buffer/buffer_string_object.h */
