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

SIXTRL_FN SIXTRL_STATIC int NS(GenericObj_compare_values)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(GenericObj) const* SIXTRL_RESTRICT lhs,
    SIXTRL_BUFFER_DATAPTR_DEC NS(GenericObj) const* SIXTRL_RESTRICT rhs );

SIXTRL_FN SIXTRL_STATIC int NS(GenericObj_compare_values_with_treshold)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(GenericObj) const* SIXTRL_RESTRICT lhs,
    SIXTRL_BUFFER_DATAPTR_DEC NS(GenericObj) const* SIXTRL_RESTRICT rhs,
    SIXTRL_REAL_T const treshold );

#if !defined( _GPUCODE )

SIXTRL_FN SIXTRL_STATIC NS(buffer_size_t)
NS(GenericObj_predict_required_num_dataptrs)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const num_d_values,
    NS(buffer_size_t) const num_e_values );

SIXTRL_FN SIXTRL_STATIC NS(buffer_size_t)
NS(GenericObj_predict_required_num_slots)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const num_d_values,
    NS(buffer_size_t) const num_e_values );

SIXTRL_FN SIXTRL_STATIC NS(buffer_size_t)
NS(GenericObj_predict_required_num_dataptrs_on_managed_buffer)(
    NS(buffer_size_t) const num_d_values,
    NS(buffer_size_t) const num_e_values, NS(buffer_size_t) const slot_size );

SIXTRL_FN SIXTRL_STATIC NS(buffer_size_t)
NS(GenericObj_predict_required_num_slots_on_managed_buffer)(
    NS(buffer_size_t) const num_d_values,
    NS(buffer_size_t) const num_e_values, NS(buffer_size_t) const slot_size );

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

SIXTRL_FN SIXTRL_STATIC SIXTRL_BUFFER_DATAPTR_DEC NS(GenericObj)*
    NS(GenericObj_add_copy)(
        SIXTRL_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
        SIXTRL_BUFFER_DATAPTR_DEC NS(GenericObj) const* original );

#if !defined( _GPUCODE )

SIXTRL_HOST_FN void NS(GenericObj_init_random)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(GenericObj)* obj );

#endif /* !defined( _GPUCODE ) */


#if defined( __cplusplus )

namespace SIXTRL_CXX_NAMESPACE
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

SIXTRL_INLINE NS(buffer_size_t) NS(GenericObj_predict_required_num_dataptrs)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const num_d_vals, NS(buffer_size_t) const num_e_vals )
{
    return NS(GenericObj_predict_required_num_dataptrs_on_managed_buffer)(
        num_d_vals, num_e_vals, NS(Buffer_get_slot_size)( buffer ) );
}

SIXTRL_INLINE NS(buffer_size_t) NS(GenericObj_predict_required_num_slots)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const num_d_vals, NS(buffer_size_t) const num_e_vals )
{
    return NS(GenericObj_predict_required_num_slots_on_managed_buffer)(
        num_d_vals, num_e_vals, NS(Buffer_get_slot_size)( buffer ) );
}


SIXTRL_INLINE NS(buffer_size_t)
NS(GenericObj_predict_required_num_dataptrs_on_managed_buffer)(
    NS(buffer_size_t) const num_d_vals, NS(buffer_size_t) const num_e_vals,
    NS(buffer_size_t) const slot_size )
{
    typedef NS(buffer_size_t) buf_size_t;

    ( void )num_d_vals;
    ( void )num_e_vals;
    ( void )slot_size;

    return ( buf_size_t )2u;
}

SIXTRL_INLINE NS(buffer_size_t)
NS(GenericObj_predict_required_num_slots_on_managed_buffer)(
    NS(buffer_size_t) const num_d_vals, NS(buffer_size_t) const num_e_vals,
    NS(buffer_size_t) const slot_size )
{
    typedef NS(buffer_size_t) buf_size_t;

    buf_size_t const num_dataptrs =
        NS(GenericObj_predict_required_num_dataptrs_on_managed_buffer)(
            num_d_vals, num_e_vals, slot_size );

    SIXTRL_ASSERT( num_dataptrs == ( buf_size_t )2u );

    buf_size_t const sizes[]  =
    {
        sizeof( SIXTRL_UINT8_T ), sizeof( SIXTRL_REAL_T )
    };

    buf_size_t const counts[] =
    {
        num_d_vals, num_e_vals
    };

    return NS(ManagedBuffer_predict_required_num_slots)(
        SIXTRL_NULLPTR, sizeof( NS(GenericObj) ), num_dataptrs,
            sizes, counts, slot_size );
}


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

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(GenericObj)* NS(GenericObj_add_copy)(
        SIXTRL_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
        SIXTRL_BUFFER_DATAPTR_DEC NS(GenericObj) const* original )
{
    if( original != SIXTRL_NULLPTR )
    {
        return NS(GenericObj_add)( buffer,
            original->type_id, original->num_d, original->num_e,
            original->a, original->b, &original->c[ 0 ],
            original->d, original->e );
    }

    return SIXTRL_NULLPTR;
}

#endif /* !defined( _GPUCODE ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined( _GPUCODE ) && defined( __cplusplus ) */

SIXTRL_INLINE int NS(GenericObj_compare_values)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(GenericObj) const* SIXTRL_RESTRICT lhs,
    SIXTRL_BUFFER_DATAPTR_DEC NS(GenericObj) const* SIXTRL_RESTRICT rhs )
{
    int cmp_result = -1;

    if( ( lhs != SIXTRL_NULLPTR ) && ( rhs != SIXTRL_NULLPTR ) )
    {
        if( lhs->type_id == rhs->type_id )
        {
            cmp_result = 0;
        }
        else if( lhs->type_id > rhs->type_id )
        {
            return -1;
        }
        else
        {
            return +1;
        }

        if( ( cmp_result == 0 ) && ( lhs->num_d != rhs->num_d ) )
        {
            if( lhs->num_d > rhs->num_d )
            {
                cmp_result = -1;
            }
            else if( lhs->num_d < rhs->num_d )
            {
                cmp_result = +1;
            }

            SIXTRL_ASSERT( ( cmp_result != 0 ) ||
                ( ( lhs->d != SIXTRL_NULLPTR ) &&
                  ( rhs->d != SIXTRL_NULLPTR ) ) );
        }

        if( ( cmp_result == 0 ) && ( lhs->num_e != rhs->num_e ) )
        {
            if( lhs->num_e > rhs->num_e )
            {
                cmp_result = -1;
            }
            else if( lhs->num_e < rhs->num_e )
            {
                cmp_result = +1;
            }

            SIXTRL_ASSERT( ( cmp_result != 0 ) ||
                ( ( lhs->e != SIXTRL_NULLPTR ) &&
                  ( rhs->e != SIXTRL_NULLPTR ) ) );
        }

        if( ( cmp_result == 0 ) && ( lhs->a != rhs->a ) )
        {
            if( lhs->a > rhs->a ) cmp_result = -1;
            else if( lhs->a < rhs->a ) cmp_result = +1;
        }

        if( cmp_result == 0 )
        {
            if( lhs->b > rhs->b )
            {
                cmp_result = -1;
            }
            else if( lhs->b > rhs->b )
            {
                cmp_result = +1;
            }
        }

        if( cmp_result == 0 )
        {
            int ii = 0;

            for( ; ii < 4 ; ++ii )
            {
                if( lhs->c[ ii ] > rhs->c[ ii ] )
                {
                    cmp_result = -1;
                }
                else if( lhs->c[ ii ] < rhs->c[ ii ] )
                {
                    cmp_result = +1;
                }

                if( cmp_result != 0 ) break;
            }
        }

        if( cmp_result == 0 )
        {
            SIXTRL_UINT64_T ii = 0;

            SIXTRL_ASSERT( ( lhs->num_d == rhs->num_d ) &&
                ( lhs->d != SIXTRL_NULLPTR ) &&
                ( rhs->d != SIXTRL_NULLPTR ) );

            for( ; ii < lhs->num_d ; ++ii )
            {
                if( lhs->d[ ii ] != rhs->d[ ii ] )
                {
                    if( lhs->d[ ii ] > rhs->d[ ii ] ) cmp_result = -1;
                    else if( lhs->d[ ii ] < rhs->d[ ii ] ) cmp_result = +1;

                    break;
                }
            }
        }

        if( cmp_result == 0 )
        {
            SIXTRL_UINT64_T ii = 0;

            SIXTRL_ASSERT( ( lhs->num_e == rhs->num_e ) &&
                ( lhs->e != SIXTRL_NULLPTR ) &&
                ( rhs->e != SIXTRL_NULLPTR ) );

            for( ; ii < lhs->num_e ; ++ii )
            {
                if( lhs->e[ ii ] > rhs->e[ ii ] )
                {
                    cmp_result = -1;
                }
                else if( lhs->e[ ii ] < rhs->e[ ii ] )
                {
                    cmp_result = +1;
                }

                if( cmp_result != 0 ) break;
            }
        }
    }
    else if( rhs != SIXTRL_NULLPTR )
    {
        cmp_result = +1;
    }

    return cmp_result;
}

SIXTRL_INLINE int NS(GenericObj_compare_values_with_treshold)(
    SIXTRL_BUFFER_DATAPTR_DEC NS(GenericObj) const* SIXTRL_RESTRICT lhs,
    SIXTRL_BUFFER_DATAPTR_DEC NS(GenericObj) const* SIXTRL_RESTRICT rhs,
    SIXTRL_REAL_T const treshold )
{
    int cmp_result = -1;

    SIXTRL_ASSERT( treshold >= ( SIXTRL_REAL_T )0.0 );

    if( ( lhs != SIXTRL_NULLPTR ) && ( rhs != SIXTRL_NULLPTR ) )
    {
        if( lhs->type_id == rhs->type_id )
        {
            cmp_result = 0;
        }
        else if( lhs->type_id > rhs->type_id )
        {
            return -1;
        }
        else
        {
            return +1;
        }

        if( ( cmp_result == 0 ) && ( lhs->num_d != rhs->num_d ) )
        {
            if( lhs->num_d > rhs->num_d )
            {
                cmp_result = -1;
            }
            else if( lhs->num_d < rhs->num_d )
            {
                cmp_result = +1;
            }

            SIXTRL_ASSERT( ( cmp_result != 0 ) ||
                ( ( lhs->d != SIXTRL_NULLPTR ) &&
                  ( rhs->d != SIXTRL_NULLPTR ) ) );
        }

        if( ( cmp_result == 0 ) && ( lhs->num_e != rhs->num_e ) )
        {
            if( lhs->num_e > rhs->num_e )
            {
                cmp_result = -1;
            }
            else if( lhs->num_e < rhs->num_e )
            {
                cmp_result = +1;
            }

            SIXTRL_ASSERT( ( cmp_result != 0 ) ||
                ( ( lhs->e != SIXTRL_NULLPTR ) &&
                  ( rhs->e != SIXTRL_NULLPTR ) ) );
        }

        if( ( cmp_result == 0 ) && ( lhs->a != rhs->a ) )
        {
            if( lhs->a > rhs->a )
            {
                cmp_result = -1;
            }
            else if( lhs->a < rhs->a )
            {
                cmp_result = +1;
            }
        }

        if( cmp_result == 0 )
        {
            SIXTRL_REAL_T const diff = ( lhs->b >= rhs->b )
                ? ( lhs->b - rhs->b ) : ( rhs->b - lhs->b );

            if( diff > treshold )
            {
                cmp_result = ( lhs->b > rhs->b ) ? -1 : +1;
            }
        }

        if( cmp_result == 0 )
        {
            int ii = 0;

            for( ; ii < 4 ; ++ii )
            {
                SIXTRL_REAL_T const diff = ( lhs->c[ ii ] >= rhs->c[ ii ] )
                    ? ( lhs->c[ ii ] - rhs->c[ ii ] )
                    : ( rhs->c[ ii ] - lhs->c[ ii ] );

                if( diff > treshold )
                {
                    cmp_result = ( lhs->c[ ii ] > rhs->c[ ii ] ) ? -1 : +1;
                }

                if( cmp_result != 0 ) break;
            }
        }

        if( cmp_result == 0 )
        {
            SIXTRL_UINT64_T ii = 0;

            SIXTRL_ASSERT( ( lhs->num_d == rhs->num_d ) &&
                ( lhs->d != SIXTRL_NULLPTR ) &&
                ( rhs->d != SIXTRL_NULLPTR ) );

            for( ; ii < lhs->num_d ; ++ii )
            {
                if( lhs->d[ ii ] != rhs->d[ ii ] )
                {
                    if( lhs->d[ ii ] > rhs->d[ ii ] ) cmp_result = -1;
                    else if( lhs->d[ ii ] < rhs->d[ ii ] ) cmp_result = +1;

                    break;
                }
            }
        }

        if( cmp_result == 0 )
        {
            SIXTRL_UINT64_T ii = 0;

            SIXTRL_ASSERT( ( lhs->num_e == rhs->num_e ) &&
                ( lhs->e != SIXTRL_NULLPTR ) &&
                ( rhs->e != SIXTRL_NULLPTR ) );

            for( ; ii < lhs->num_e ; ++ii )
            {
                SIXTRL_REAL_T const diff = ( lhs->e[ ii ] >= rhs->e[ ii ] )
                    ? ( lhs->e[ ii ] - rhs->e[ ii ] )
                    : ( rhs->e[ ii ] - lhs->e[ ii ] );

                if( diff > treshold )
                {
                    cmp_result = ( lhs->c[ ii ] > rhs->c[ ii ] ) ? -1 : +1;
                }

                if( cmp_result != 0 ) break;
            }
        }
    }
    else if( rhs != SIXTRL_NULLPTR )
    {
        cmp_result = +1;
    }

    return cmp_result;
}

#endif /* SIXTRACKLIB_TESTS_TESTLIB_GENERIC_BUFFER_H__ */

/* end: tests/sixtracklib/testlib/common/generic_buffer_obj.h */
