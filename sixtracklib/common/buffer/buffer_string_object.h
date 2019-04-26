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

typedef struct NS(BufferStringObject)
{
    NS(buffer_addr_t) begin_addr SIXTRL_ALIGN( 8u );
    NS(buffer_size_t) length     SIXTRL_ALIGN( 8u );
    NS(buffer_size_t) capacity   SIXTRL_ALIGN( 8u );
}
NS(BufferStringObject);

/* ------------------------------------------------------------------------- */

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t)
NS(BufferStringObject_get_cstring_length)(
    SIXTRL_ARGPTR_DEC const char *const SIXTRL_RESTRICT str );

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t)
NS(BufferStringObject_get_cstring_length_reverse)(
    SIXTRL_ARGPTR_DEC const char *const SIXTRL_RESTRICT str,
    NS(buffer_size_t) const capacity );

/* ------------------------------------------------------------------------- */

SIXTRL_STATIC SIXTRL_FN SIXTRL_BUFFER_DATAPTR_DEC NS(BufferStringObject)*
NS(BufferStringObject_preset)( SIXTRL_BUFFER_DATAPTR_DEC
    NS(BufferStringObject)* SIXTRL_RESTRICT string );

SIXTRL_STATIC SIXTRL_FN NS(buffer_addr_t)
NS(BufferStringObject_get_begin_addr)( SIXTRL_BUFFER_DATAPTR_DEC
    const NS(BufferStringObject) *const SIXTRL_RESTRICT str_obj );

SIXTRL_STATIC SIXTRL_FN SIXTRL_BUFFER_DATAPTR_DEC char const*
NS(BufferStringObject_get_const_string)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(BufferStringObject) *const
        SIXTRL_RESTRICT str_obj );

SIXTRL_STATIC SIXTRL_FN SIXTRL_BUFFER_DATAPTR_DEC char*
NS(BufferStringObject_get_string)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(BufferStringObject)* SIXTRL_RESTRICT str );

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t)
NS(BufferStringObject_get_capacity)( SIXTRL_BUFFER_DATAPTR_DEC
    const NS(BufferStringObject) *const SIXTRL_RESTRICT str_obj );

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t)
NS(BufferStringObject_get_length)( SIXTRL_BUFFER_DATAPTR_DEC
    const NS(BufferStringObject) *const SIXTRL_RESTRICT str_obj );


SIXTRL_STATIC SIXTRL_FN SIXTRL_BUFFER_DATAPTR_DEC char const*
NS(BufferStringObject_copy_cstring)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(BufferStringObject)* SIXTRL_RESTRICT str_obj,
    SIXTRL_ARGPTR_DEC const char *const SIXTRL_RESTRICT source_str );

SIXTRL_STATIC SIXTRL_FN SIXTRL_BUFFER_DATAPTR_DEC char const*
NS(BufferStringObject_append_cstring)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(BufferStringObject)* SIXTRL_RESTRICT str_obj,
    SIXTRL_ARGPTR_DEC const char *const SIXTRL_RESTRICT source_str );


SIXTRL_STATIC SIXTRL_FN void NS(BufferStringObject_set_begin_addr)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(BufferStringObject)* SIXTRL_RESTRICT str_obj,
    NS(buffer_addr_t) const begin_addr );

SIXTRL_STATIC SIXTRL_FN void NS(BufferStringObject_set_capacity)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(BufferStringObject)* SIXTRL_RESTRICT str_obj,
    NS(buffer_size_t) const capacity );

SIXTRL_STATIC SIXTRL_FN void NS(BufferStringObject_set_length)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(BufferStringObject)* SIXTRL_RESTRICT str_obj,
    NS(buffer_size_t) const length );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

#if !defined( _GPUCODE )

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BUFFER_DATAPTR_DEC NS(BufferStringObject)*
NS(BufferStringObject_preset_ext)( SIXTRL_BUFFER_DATAPTR_DEC
    NS(BufferStringObject)* SIXTRL_RESTRICT string );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(buffer_addr_t)
NS(BufferStringObject_get_begin_addr_ext)( SIXTRL_BUFFER_DATAPTR_DEC
    const NS(BufferStringObject) *const SIXTRL_RESTRICT str_obj );

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BUFFER_DATAPTR_DEC char const*
NS(BufferStringObject_get_const_string_ext)( SIXTRL_BUFFER_DATAPTR_DEC const
    NS(BufferStringObject) *const SIXTRL_RESTRICT str_obj );

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BUFFER_DATAPTR_DEC char*
NS(BufferStringObject_get_string_ext)( SIXTRL_BUFFER_DATAPTR_DEC
    NS(BufferStringObject)* SIXTRL_RESTRICT str );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(buffer_size_t)
NS(BufferStringObject_get_capacity_ext)( SIXTRL_BUFFER_DATAPTR_DEC
    const NS(BufferStringObject) *const SIXTRL_RESTRICT str_obj );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(buffer_size_t)
NS(BufferStringObject_get_length_ext)( SIXTRL_BUFFER_DATAPTR_DEC
    const NS(BufferStringObject) *const SIXTRL_RESTRICT str_obj );


SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BUFFER_DATAPTR_DEC char const*
NS(BufferStringObject_copy_cstring_ext)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(BufferStringObject)* SIXTRL_RESTRICT str_obj,
    SIXTRL_ARGPTR_DEC const char *const SIXTRL_RESTRICT source_str );

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BUFFER_DATAPTR_DEC char const*
NS(BufferStringObject_append_cstring_ext)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(BufferStringObject)* SIXTRL_RESTRICT str_obj,
    SIXTRL_ARGPTR_DEC const char *const SIXTRL_RESTRICT source_str );

#endif /* !defined( _GPUCODE ) */

/* ------------------------------------------------------------------------ */

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t)
NS(BufferStringObject_get_required_num_slots_on_managed_buffer)(
    NS(buffer_size_t) const capacity, NS(buffer_size_t) const slot_size );

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t)
NS(BufferStringObject_get_required_num_dataptrs_on_managed_buffer)(
    NS(buffer_size_t) const capacity, NS(buffer_size_t) const slot_size );

#if !defined( _GPUCODE )

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t)
NS(BufferStringObject_get_required_num_slots)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const capacity );

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t)
NS(BufferStringObject_get_required_num_dataptrs)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const capacity );

SIXTRL_STATIC SIXTRL_FN bool NS(BufferStringObject_can_be_added)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const capacity,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_objects,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_slots,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_dataptr );

SIXTRL_STATIC SIXTRL_FN SIXTRL_BUFFER_DATAPTR_DEC NS(BufferStringObject)*
NS(BufferStringObject_new)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const capacity );

SIXTRL_STATIC SIXTRL_FN SIXTRL_BUFFER_DATAPTR_DEC NS(BufferStringObject)*
NS(BufferStringObject_add)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_ARGPTR_DEC const char *const SIXTRL_RESTRICT source_str,
    NS(buffer_size_t) const capacity );

SIXTRL_STATIC SIXTRL_FN SIXTRL_BUFFER_DATAPTR_DEC NS(BufferStringObject)*
NS(BufferStringObject_new_from_cstring)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_ARGPTR_DEC const char *const SIXTRL_RESTRICT source_str );

SIXTRL_STATIC SIXTRL_FN SIXTRL_BUFFER_DATAPTR_DEC NS(BufferStringObject)*
NS(BufferStringObject_add_copy)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(BufferStringObject) *const
        SIXTRL_RESTRICT existing_str_obj );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_EXTERN SIXTRL_HOST_FN NS(buffer_size_t)
NS(BufferStringObject_get_required_num_slots_ext)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const capacity );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(buffer_size_t)
NS(BufferStringObject_get_required_num_dataptrs_ext)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const capacity );

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(BufferStringObject_can_be_added_ext)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const capacity,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_objects,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_slots,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_dataptr );

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BUFFER_DATAPTR_DEC NS(BufferStringObject)*
NS(BufferStringObject_new_ext)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const capacity );

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BUFFER_DATAPTR_DEC NS(BufferStringObject)*
NS(BufferStringObject_add_ext)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_ARGPTR_DEC const char *const SIXTRL_RESTRICT source_str,
    NS(buffer_size_t) const capacity );

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BUFFER_DATAPTR_DEC NS(BufferStringObject)*
NS(BufferStringObject_new_from_cstring_ext)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_ARGPTR_DEC const char *const SIXTRL_RESTRICT source_str );

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BUFFER_DATAPTR_DEC NS(BufferStringObject)*
NS(BufferStringObject_add_copy_ext)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(BufferStringObject) *const
        SIXTRL_RESTRICT existing_str_obj );

#endif /* !defined( _GPUCODE ) */

/* ************************************************************************* */
/* *********         Implementation of Inline Functions          *********** */
/* ************************************************************************* */

SIXTRL_INLINE NS(buffer_size_t) NS(BufferStringObject_get_cstring_length)(
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
NS(BufferStringObject_get_cstring_length_reverse)(
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

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(BufferStringObject)*
NS(BufferStringObject_preset)( SIXTRL_BUFFER_DATAPTR_DEC
    NS(BufferStringObject)* SIXTRL_RESTRICT str_obj )
{
    if( str_obj != SIXTRL_NULLPTR )
    {
        str_obj->begin_addr = ( NS(buffer_addr_t) )0u;
        str_obj->capacity   = ( NS(buffer_size_t) )0u;
    }

    return str_obj;
}

SIXTRL_INLINE NS(buffer_addr_t) NS(BufferStringObject_get_begin_addr)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(BufferStringObject) *const
        SIXTRL_RESTRICT str_obj )
{
    SIXTRL_ASSERT( str_obj != SIXTRL_NULLPTR );
    return str_obj->begin_addr;
}

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC char const*
NS(BufferStringObject_get_const_string)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(BufferStringObject) *const
        SIXTRL_RESTRICT str_obj )
{
    typedef SIXTRL_BUFFER_DATAPTR_DEC char const* ptr_cstring_t;
    SIXTRL_ASSERT( str_obj != SIXTRL_NULLPTR );
    return ( ptr_cstring_t )( uintptr_t )str_obj->begin_addr;
}

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC char*
NS(BufferStringObject_get_string)( SIXTRL_BUFFER_DATAPTR_DEC
    NS(BufferStringObject)* SIXTRL_RESTRICT str_obj )
{
    return ( SIXTRL_BUFFER_DATAPTR_DEC char*
        )NS(BufferStringObject_get_const_string)( str_obj );
}

SIXTRL_INLINE NS(buffer_size_t) NS(BufferStringObject_get_capacity)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(BufferStringObject) *const
        SIXTRL_RESTRICT str_obj )
{
    SIXTRL_ASSERT( str_obj != SIXTRL_NULLPTR );
    return str_obj->capacity;
}

SIXTRL_INLINE NS(buffer_size_t) NS(BufferStringObject_get_length)(
    SIXTRL_BUFFER_DATAPTR_DEC const NS(BufferStringObject) *const
        SIXTRL_RESTRICT str_obj )
{
    typedef NS(buffer_size_t) buf_size_t;

    buf_size_t length = ( buf_size_t )0u;

    if( NS(BufferStringObject_get_begin_addr)( str_obj ) >
        ( NS(buffer_addr_t) )0u )
    {
        SIXTRL_ASSERT( str_obj != SIXTRL_NULLPTR );
        SIXTRL_ASSERT( NS(BufferStringObject_get_capacity)( str_obj ) >
            ( buf_size_t )0u );

        if( str_obj->length < str_obj->capacity )
        {
            length = str_obj->length;

            SIXTRL_ASSERT(
                ( ( length < ( buf_size_t )1024 ) &&
                  ( NS(BufferStringObject_get_cstring_length)(
                    NS(BufferStringObject_get_const_string)( str_obj ) ) ==
                        length ) ) ||
                ( ( length >= ( buf_size_t )1024 ) &&
                  ( NS(BufferStringObject_get_cstring_length_reverse)(
                      NS(BufferStringObject_get_const_string)( str_obj ),
                      NS(BufferStringObject_get_capacity)( str_obj ) ) ==
                        length ) ) );
        }
    }

    return length;
}


SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC char const*
NS(BufferStringObject_copy_cstring)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(BufferStringObject)* SIXTRL_RESTRICT str_obj,
    SIXTRL_ARGPTR_DEC const char *const SIXTRL_RESTRICT source_str )
{
    typedef NS(buffer_size_t) buf_size_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC char const* ptr_const_cstring_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC char* ptr_cstring_t;

    ptr_const_cstring_t copied_str = SIXTRL_NULLPTR;
    ptr_cstring_t stored_str = NS(BufferStringObject_get_string)( str_obj );
    buf_size_t const capacity = NS(BufferStringObject_get_capacity)( str_obj );

    if( ( source_str != SIXTRL_NULLPTR ) && ( stored_str != SIXTRL_NULLPTR ) &&
        ( capacity > ( buf_size_t )1u ) )
    {
        buf_size_t const source_str_len =
            NS(BufferStringObject_get_cstring_length)( source_str );

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

            copied_str = stored_str;
        }
    }

    return copied_str;
}

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC char const*
NS(BufferStringObject_append_cstring)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(BufferStringObject)* SIXTRL_RESTRICT str_obj,
    SIXTRL_ARGPTR_DEC const char *const SIXTRL_RESTRICT source_str )
{
    typedef NS(buffer_size_t) buf_size_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC char const* ptr_const_cstring_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC char* ptr_cstring_t;

    ptr_const_cstring_t appended_str = SIXTRL_NULLPTR;
    ptr_cstring_t stored_str = NS(BufferStringObject_get_string)( str_obj );
    buf_size_t const capacity = NS(BufferStringObject_get_capacity)( str_obj );

    buf_size_t const current_length =
        NS(BufferStringObject_get_length)( str_obj );

    if( ( source_str != SIXTRL_NULLPTR ) && ( stored_str != SIXTRL_NULLPTR ) &&
        ( capacity > current_length ) )
    {
        buf_size_t const source_str_len =
            NS(BufferStringObject_get_cstring_length)( source_str );

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

            appended_str = stored_str;
        }
    }

    return appended_str;
}


SIXTRL_INLINE void NS(BufferStringObject_set_begin_addr)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(BufferStringObject)* SIXTRL_RESTRICT str_obj,
    NS(buffer_addr_t) const begin_addr )
{
    SIXTRL_ASSERT( str_obj != SIXTRL_NULLPTR );
    str_obj->begin_addr = begin_addr;
}

SIXTRL_INLINE void NS(BufferStringObject_set_capacity)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(BufferStringObject)* SIXTRL_RESTRICT str_obj,
    NS(buffer_size_t) const capacity )
{
    SIXTRL_ASSERT( str_obj != SIXTRL_NULLPTR );
    str_obj->capacity = capacity;
}

SIXTRL_INLINE void NS(BufferStringObject_set_length)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(BufferStringObject)* SIXTRL_RESTRICT str_obj,
    NS(buffer_size_t) const length )
{
    SIXTRL_ASSERT( str_obj != SIXTRL_NULLPTR );
    str_obj->length = length;
}

/* ------------------------------------------------------------------------ */

SIXTRL_INLINE NS(buffer_size_t)
NS(BufferStringObject_get_required_num_slots_on_managed_buffer)(
    NS(buffer_size_t) const capacity, NS(buffer_size_t) const slot_size )
{
    typedef NS(buffer_size_t) buf_size_t;
    SIXTRL_STATIC_VAR buf_size_t const ZERO = ( buf_size_t )0u;

    buf_size_t num_slots = ZERO;
    buf_size_t required_size = NS(ManagedBuffer_get_slot_based_length)(
        sizeof( NS(BufferStringObject) ), slot_size );

    if( capacity > ZERO )
    {
        required_size +=
            NS(ManagedBuffer_get_slot_based_length)( capacity, slot_size );
    }

    if( slot_size > ZERO )
    {
        SIXTRL_ASSERT( ZERO == ( required_size % slot_size ) );
        num_slots = required_size / slot_size;
    }

    return num_slots;
}

SIXTRL_INLINE NS(buffer_size_t)
NS(BufferStringObject_get_required_num_dataptrs_on_managed_buffer)(
    NS(buffer_size_t) const capacity, NS(buffer_size_t) const slot_size )
{
    ( void )capacity;

    return ( slot_size > ( NS(buffer_size_t) )0u )
        ? ( NS(buffer_size_t) )1u : ( NS(buffer_size_t) )0u;
}


SIXTRL_INLINE NS(buffer_size_t) NS(BufferStringObject_get_required_num_slots)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const capacity )
{
    return NS(BufferStringObject_get_required_num_slots_on_managed_buffer)(
        capacity, NS(Buffer_get_slot_size)( buffer ) );
}

SIXTRL_INLINE NS(buffer_size_t)
NS(BufferStringObject_get_required_num_dataptrs)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const capacity )
{
    return NS(BufferStringObject_get_required_num_dataptrs_on_managed_buffer)(
        capacity, NS(Buffer_get_slot_size)( buffer ) );
}

SIXTRL_INLINE bool NS(BufferStringObject_can_be_added)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const capacity,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_objects,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_slots,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_dataptrs )
{
    typedef NS(buffer_size_t) buf_size_t;

    buf_size_t const num_dataptrs =
        NS(BufferStringObject_get_required_num_dataptrs)( buffer, capacity );

    buf_size_t const sizes[]  = { sizeof( char ) };
    buf_size_t const counts[] = { capacity };
    buf_size_t const obj_size = sizeof( NS(BufferStringObject) );

    return NS(Buffer_can_add_object)( buffer, obj_size, num_dataptrs,
        &sizes[ 0 ], &counts[ 0 ], requ_objects, requ_slots, requ_dataptrs );
}

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(BufferStringObject)*
NS(BufferStringObject_new)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const capacity )
{
    typedef NS(BufferStringObject) str_obj_t;
    typedef NS(buffer_size_t) buf_size_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC str_obj_t* ptr_str_obj_t;

    SIXTRL_STATIC_VAR buf_size_t const ZERO = ( buf_size_t )0u;

    buf_size_t const num_dataptrs =
        NS(BufferStringObject_get_required_num_dataptrs)( buffer, capacity );

    buf_size_t offsets[] = { offsetof( str_obj_t, begin_addr ) };
    buf_size_t sizes[]   = { sizeof( char ) };
    buf_size_t counts[]  = { capacity };

    str_obj_t str_obj;
    str_obj.begin_addr = ( NS(buffer_addr_t) )0u;
    str_obj.length     = ZERO;
    str_obj.capacity   = capacity;

    return ( ptr_str_obj_t )( uintptr_t )NS(Object_get_begin_addr)(
        NS(Buffer_add_object)( buffer, &str_obj, sizeof( str_obj_t ),
            NS(OBJECT_TYPE_CSTRING), num_dataptrs,
                &offsets[ 0 ], &sizes[ 0 ], &counts[ 0 ] ) );
}

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(BufferStringObject)*
NS(BufferStringObject_add)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_ARGPTR_DEC const char *const SIXTRL_RESTRICT input_str,
    NS(buffer_size_t) const capacity )
{
    typedef NS(buffer_size_t) buf_size_t;

    typedef NS(BufferStringObject) str_obj_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC str_obj_t* ptr_str_obj_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC char* ptr_cstring_t;

    ptr_str_obj_t str_obj = SIXTRL_NULLPTR;

    buf_size_t const input_str_len = ( input_str != SIXTRL_NULLPTR )
        ? NS(BufferStringObject_get_cstring_length)( input_str )
        : ( buf_size_t )0u;

    if( ( input_str != SIXTRL_NULLPTR ) &&
        ( input_str_len > ( buf_size_t )0u ) && ( capacity > input_str_len ) )
    {
        str_obj = NS(BufferStringObject_new)( buffer, capacity );

        if( str_obj != SIXTRL_NULLPTR )
        {
            buf_size_t const remaining_chars = capacity - input_str_len;
            ptr_cstring_t out = NS(BufferStringObject_get_string)( str_obj );

            SIXTRL_ASSERT( out != SIXTRL_NULLPTR );
            SIXTRACKLIB_COPY_VALUES( char, out, input_str, input_str_len );

            SIXTRACKLIB_SET_VALUES(  char, out + input_str_len,
                                     remaining_chars, '\0' );

            NS(BufferStringObject_set_length)( str_obj, input_str_len );
        }
    }
    else if( ( capacity > ( buf_size_t )0u ) &&
             ( ( input_str == SIXTRL_NULLPTR ) ||
               ( input_str_len == ( buf_size_t )0u ) ) )
    {
        str_obj = NS(BufferStringObject_new)( buffer, capacity );

        if( str_obj != SIXTRL_NULLPTR )
        {
            ptr_cstring_t out = NS(BufferStringObject_get_string)( str_obj );

            SIXTRL_ASSERT( out != SIXTRL_NULLPTR );
            SIXTRACKLIB_SET_VALUES(  char, out, capacity, '\0' );
        }
    }

    if( str_obj != SIXTRL_NULLPTR )
    {
        NS(BufferStringObject_set_capacity)( str_obj, capacity );
    }

    return str_obj;
}

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(BufferStringObject)*
NS(BufferStringObject_new_from_cstring)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_ARGPTR_DEC const char *const SIXTRL_RESTRICT input_str )
{
    SIXTRL_STATIC_VAR NS(buffer_size_t) ONE = ( NS(buffer_size_t) )0u;

    NS(buffer_size_t) const capacity = ( input_str != SIXTRL_NULLPTR )
        ? NS(BufferStringObject_get_cstring_length)( input_str ) + ONE : ONE;

    return NS(BufferStringObject_add)( buffer, input_str, capacity );
}

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(BufferStringObject)*
NS(BufferStringObject_add_copy)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(BufferStringObject) *const
        SIXTRL_RESTRICT str_obj )
{
    return NS(BufferStringObject_add)( buffer,
        NS(BufferStringObject_get_const_string)( str_obj ),
        NS(BufferStringObject_get_capacity)( str_obj ) );
}

#if defined( __cplusplus ) && !defined( _GPUCODE )
}
#endif /* defined( __cplusplus ) && !defined( _GPUCODE ) */

#endif /* SIXTRACKLIB_COMMON_BUFFER_BUFFER_STRING_OBJECT_H__ */

/* end: sixtracklib/common/buffer/buffer_string_object.h */
