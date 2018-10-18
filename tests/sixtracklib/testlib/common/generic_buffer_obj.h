#ifndef SIXTRACKLIB_TESTS_TESTLIB_GENERIC_BUFFER_H__
#define SIXTRACKLIB_TESTS_TESTLIB_GENERIC_BUFFER_H__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stdbool.h>
    #include <stddef.h>
    #include <stdint.h>
    #include <stdlib.h>
    #include <string.h>
    #include <limits.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/internal/buffer_main_defines.h"
    #include "sixtracklib/common/buffer/buffer_object.h"
    #include "sixtracklib/common/buffer/buffer_type.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined( _GPUCODE ) && defined( __cplusplus ) */

typedef struct NS(GenericObj)
{
    NS(object_type_id_t) type_id                         SIXTRL_ALIGN( 8u );
    SIXTRL_INT32_T a                                     SIXTRL_ALIGN( 8u );
    SIXTRL_REAL_T b                                      SIXTRL_ALIGN( 8u );
    SIXTRL_REAL_T c[ 4 ]                                 SIXTRL_ALIGN( 8u );

    SIXTRL_UINT64_T num_d                                SIXTRL_ALIGN( 8u );
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_UINT8_T*
        SIXTRL_RESTRICT d                                SIXTRL_ALIGN( 8u );

    SIXTRL_UINT64_T num_e                                SIXTRL_ALIGN( 8u );
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T*
        SIXTRL_RESTRICT e                                SIXTRL_ALIGN( 8u );
}
NS(GenericObj);

#if !defined( _GPUCODE )

SIXTRL_FN SIXTRL_STATIC SIXTRL_BUFFER_DATAPTR_DEC NS(GenericObj)*
    NS(GenericObj_new)(
    SIXTRL_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(object_type_id_t) const type_id,
    NS(buffer_size_t) const num_d_values,
    NS(buffer_size_t) const num_e_values );

SIXTRL_FN SIXTRL_STATIC SIXTRL_BUFFER_DATAPTR_DEC NS(GenericObj)*
    NS(GenericObj_add)(
    SIXTRL_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(object_type_id_t) const type_id,
    NS(buffer_size_t) const num_d_values,
    NS(buffer_size_t) const num_e_values,
    SIXTRL_INT32_T const a_value, SIXTRL_REAL_T const b_value,
    SIXTRL_ARGPTR_DEC  SIXTRL_REAL_T const* SIXTRL_RESTRICT c_values,
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_UINT8_T* SIXTRL_RESTRICT d_values,
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T*  SIXTRL_RESTRICT e_values );


#if defined( __cplusplus )

namespace SIXTRL_NAMESPACE
{
    using GenericObj = ::NS(GenericObj);
}

#endif /* !defined( _GPUCODE ) */

#endif /* defined( __cplusplus ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined( _GPUCODE ) && defined( __cplusplus ) */

#if !defined( _GPUCODE )
#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/buffer.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */
#endif /* !defined( _GPUCODE ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined( _GPUCODE ) && defined( __cplusplus ) */

#if !defined( _GPUCODE )

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(GenericObj)* NS(GenericObj_new)(
    SIXTRL_ARGPTR_DEC struct NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(object_type_id_t) const type_id,
    NS(buffer_size_t)    const num_d_values,
    NS(buffer_size_t)    const num_e_values )
{
    typedef NS(buffer_size_t)                   buf_size_t;
    typedef SIXTRL_ARGPTR_DEC NS(GenericObj)*   ptr_gen_obj_t;

    ptr_gen_obj_t ptr_gen_obj  = SIXTRL_NULLPTR;

    if( ( num_d_values > ( buf_size_t )0u ) &&
        ( num_e_values > ( buf_size_t )0u ) )
    {
        buf_size_t const offsets[] =
        {
            offsetof( NS(GenericObj), d ),
            offsetof( NS(GenericObj), e )
        };

        buf_size_t const sizes[]  = { sizeof( SIXTRL_UINT8_T ), sizeof( SIXTRL_REAL_T ) };
        buf_size_t const counts[] = { num_d_values, num_e_values };

        NS(GenericObj) temp;
        memset( &temp, ( int )0, sizeof( temp ) );

        temp.type_id = type_id;
        temp.a       = ( SIXTRL_INT32_T )0;
        temp.b       = ( SIXTRL_REAL_T )0.0;

        temp.c[ 0 ]  =
        temp.c[ 1 ]  =
        temp.c[ 2 ]  =
        temp.c[ 3 ]  = ( SIXTRL_REAL_T )0.0;

        temp.num_d = num_d_values;
        temp.d     = SIXTRL_NULLPTR;

        temp.num_e = num_e_values;
        temp.e     = SIXTRL_NULLPTR;

        ptr_gen_obj = ( ptr_gen_obj_t )( uintptr_t )NS(Object_get_begin_addr)(
            NS(Buffer_add_object)( buffer, &temp, sizeof( temp ), temp.type_id,
            ( buf_size_t )2u, offsets, sizes, counts ) );
    }

    return ptr_gen_obj;
}

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(GenericObj)* NS(GenericObj_add)(
    SIXTRL_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(object_type_id_t) const type_id,
    NS(buffer_size_t) const num_d_values,
    NS(buffer_size_t) const num_e_values,
    SIXTRL_INT32_T const a_value, SIXTRL_REAL_T const b_value,
    SIXTRL_ARGPTR_DEC  SIXTRL_REAL_T const* SIXTRL_RESTRICT c_values,
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_UINT8_T* SIXTRL_RESTRICT d_values,
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T*  SIXTRL_RESTRICT e_values )
{
    typedef NS(buffer_size_t)                   buf_size_t;
    typedef SIXTRL_ARGPTR_DEC NS(GenericObj)*   ptr_gen_obj_t;

    ptr_gen_obj_t ptr_gen_obj  = SIXTRL_NULLPTR;

    if( ( num_d_values > ( buf_size_t )0u ) &&
        ( num_e_values > ( buf_size_t )0u ) )
    {
        buf_size_t const offsets[] =
        {
            offsetof( NS(GenericObj), d ),
            offsetof( NS(GenericObj), e )
        };

        buf_size_t const sizes[]  =
        {
            sizeof( SIXTRL_UINT8_T ),
            sizeof( SIXTRL_REAL_T )
        };

        buf_size_t const counts[] = { num_d_values, num_e_values };

        NS(GenericObj) temp;
        memset( &temp, ( int )0, sizeof( temp ) );

        temp.type_id = type_id;
        temp.a       = a_value;
        temp.b       = b_value;

        if( c_values != SIXTRL_NULLPTR )
        {
            memcpy( &temp.c[ 0 ], c_values, sizeof( SIXTRL_REAL_T ) * 4 );
        }
        else
        {
            temp.c[ 0 ] = temp.c[ 1 ]  =
            temp.c[ 2 ] = temp.c[ 3 ]  = ( SIXTRL_REAL_T )0.0;
        }

        temp.num_d = num_d_values;
        temp.d     = d_values;

        temp.num_e = num_e_values;
        temp.e     = e_values;

        ptr_gen_obj = ( ptr_gen_obj_t )( uintptr_t )NS(Object_get_begin_addr)(
            NS(Buffer_add_object)( buffer, &temp, sizeof( temp ), temp.type_id,
            ( buf_size_t )2u, offsets, sizes, counts ) );
    }

    return ptr_gen_obj;
}

#endif /* !defined( _GPUCODE ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined( _GPUCODE ) && defined( __cplusplus ) */


#endif /* SIXTRACKLIB_TESTS_TESTLIB_GENERIC_BUFFER_H__ */

/* end: tests/sixtracklib/testlib/common/generic_buffer_obj.h */
