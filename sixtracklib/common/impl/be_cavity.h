#ifndef SIXTRACKLIB_COMMON_IMPL_BEAM_ELEMENT_CAVITY_H__
#define SIXTRACKLIB_COMMON_IMPL_BEAM_ELEMENT_CAVITY_H__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stdbool.h>
    #include <stdint.h>
    #include <stdlib.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/_impl/definitions.h"
    #include "sixtracklib/common/impl/buffer_defines.h"
    #include "sixtracklib/common/impl/beam_elements_defines.h"
    #include "sixtracklib/common/impl/buffer_type.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined(  _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

/* ========================================================================= */

typedef struct NS(Cavity)
{
    SIXTRL_REAL_T   voltage     SIXTRL_ALIGN( 8 );
    SIXTRL_REAL_T   frequency   SIXTRL_ALIGN( 8 );
    SIXTRL_REAL_T   lag         SIXTRL_ALIGN( 8 );
}
NS(Cavity);

/* ------------------------------------------------------------------------- */

SIXTRL_FN SIXTRL_STATIC NS(buffer_size_t) NS(Cavity_get_num_dataptrs)(
    SIXTRL_BE_ARGPTR_DEC  const NS(Cavity) *const SIXTRL_RESTRICT cavity );

SIXTRL_FN SIXTRL_STATIC SIXTRL_BE_ARGPTR_DEC NS(Cavity)* NS(Cavity_preset)(
    SIXTRL_BE_ARGPTR_DEC  NS(Cavity)* SIXTRL_RESTRICT cavity );

SIXTRL_FN SIXTRL_STATIC SIXTRL_REAL_T NS(Cavity_get_voltage)(
    SIXTRL_BE_ARGPTR_DEC const NS(Cavity) *const SIXTRL_RESTRICT cavity );

SIXTRL_FN SIXTRL_STATIC SIXTRL_REAL_T NS(Cavity_get_frequency)(
    SIXTRL_BE_ARGPTR_DEC  const NS(Cavity) *const SIXTRL_RESTRICT cavity );

SIXTRL_FN SIXTRL_STATIC SIXTRL_REAL_T NS(Cavity_get_lag)(
    SIXTRL_BE_ARGPTR_DEC  const NS(Cavity) *const SIXTRL_RESTRICT cavity );

SIXTRL_FN SIXTRL_STATIC void NS(Cavity_set_voltage)(
    SIXTRL_BE_ARGPTR_DEC NS(Cavity)* SIXTRL_RESTRICT cavity,
    SIXTRL_REAL_T const voltage );

SIXTRL_FN SIXTRL_STATIC void NS(Cavity_set_frequency)(
    SIXTRL_BE_ARGPTR_DEC  NS(Cavity)* SIXTRL_RESTRICT cavity,
    SIXTRL_REAL_T const frequency );

SIXTRL_FN SIXTRL_STATIC void NS(Cavity_set_lag)(
    SIXTRL_BE_ARGPTR_DEC  NS(Cavity)* SIXTRL_RESTRICT cavity,
    SIXTRL_REAL_T const lag );

SIXTRL_FN SIXTRL_STATIC void NS(Cavity_copy)(
    SIXTRL_BE_ARGPTR_DEC NS(Cavity)* SIXTRL_RESTRICT destination,
    SIXTRL_BE_ARGPTR_DEC const NS(Cavity) *const SIXTRL_RESTRICT source );

SIXTRL_FN SIXTRL_STATIC int NS(Cavity_compare_values)(
    SIXTRL_BE_ARGPTR_DEC const NS(Cavity) *const SIXTRL_RESTRICT lhs,
    SIXTRL_BE_ARGPTR_DEC const NS(Cavity) *const SIXTRL_RESTRICT rhs );

SIXTRL_FN SIXTRL_STATIC int NS(Cavity_compare_values_with_treshold)(
    SIXTRL_BE_ARGPTR_DEC const NS(Cavity) *const SIXTRL_RESTRICT lhs,
    SIXTRL_BE_ARGPTR_DEC const NS(Cavity) *const SIXTRL_RESTRICT rhs,
    SIXTRL_REAL_T const treshold );

#if !defined( _GPUCODE )

SIXTRL_FN SIXTRL_STATIC bool NS(Cavity_can_be_added)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_objects,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_slots,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_dataptrs );

SIXTRL_FN SIXTRL_STATIC SIXTRL_BUFFER_DATAPTR_DEC NS(Cavity)* NS(Cavity_new)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer );

SIXTRL_FN SIXTRL_STATIC SIXTRL_BUFFER_DATAPTR_DEC NS(Cavity)* NS(Cavity_add)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_REAL_T  const voltage, SIXTRL_REAL_T  const frequency,
    SIXTRL_REAL_T  const lag );

SIXTRL_FN SIXTRL_STATIC SIXTRL_BUFFER_DATAPTR_DEC NS(Cavity)*
NS(Cavity_add_copy)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_BE_ARGPTR_DEC const NS(Cavity) *const SIXTRL_RESTRICT cavity );

#endif /* !defined( _GPUCODE ) */

#if !defined(  _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

/* ************************************************************************* */
/* *****          Implementation of C inline functions                   *** */
/* ************************************************************************* */

#if !defined( SIXTRL_NO_INCLUDES )
    #if !defined( _GPUCODE )
        #include "sixtracklib/common/buffer.h"
    #endif /* !defined( _GPUCODE ) */
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined(  _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */


SIXTRL_INLINE NS(buffer_size_t) NS(Cavity_get_num_dataptrs)(
    SIXTRL_BE_ARGPTR_DEC const NS(Cavity) *const SIXTRL_RESTRICT cavity )
{
    return ( NS(buffer_size_t) )0u;
}

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(Cavity)* NS(Cavity_preset)(
    SIXTRL_BE_ARGPTR_DEC NS(Cavity)* SIXTRL_RESTRICT cavity )
{
    if( cavity != SIXTRL_NULLPTR )
    {
        NS(Cavity_set_voltage)(   cavity, ( SIXTRL_REAL_T )0 );
        NS(Cavity_set_frequency)( cavity, ( SIXTRL_REAL_T )0 );
        NS(Cavity_set_lag)( cavity, ( SIXTRL_REAL_T )0 );
    }

    return cavity;
}

SIXTRL_INLINE SIXTRL_REAL_T NS(Cavity_get_voltage)(
    SIXTRL_BE_ARGPTR_DEC const NS(Cavity) *const SIXTRL_RESTRICT cavity )
{
    return ( cavity != SIXTRL_NULLPTR ) ? cavity->voltage : ( SIXTRL_REAL_T )0;
}

SIXTRL_INLINE SIXTRL_REAL_T NS(Cavity_get_frequency)(
    SIXTRL_BE_ARGPTR_DEC const NS(Cavity) *const SIXTRL_RESTRICT cavity )
{
    return ( cavity != SIXTRL_NULLPTR )
        ? cavity->frequency : ( SIXTRL_REAL_T )0;
}

SIXTRL_INLINE SIXTRL_REAL_T NS(Cavity_get_lag)(
    SIXTRL_BE_ARGPTR_DEC const NS(Cavity) *const SIXTRL_RESTRICT cavity )
{
    return ( cavity != SIXTRL_NULLPTR ) ? cavity->lag : ( SIXTRL_REAL_T )0;
}

SIXTRL_INLINE void NS(Cavity_set_voltage)(
    SIXTRL_BE_ARGPTR_DEC NS(Cavity)* SIXTRL_RESTRICT cavity,
    SIXTRL_REAL_T const voltage )
{
    SIXTRL_ASSERT( cavity != SIXTRL_NULLPTR );
    cavity->voltage = voltage;
    return;
}

SIXTRL_INLINE void NS(Cavity_set_frequency)(
    SIXTRL_BE_ARGPTR_DEC NS(Cavity)* SIXTRL_RESTRICT cavity,
    SIXTRL_REAL_T const frequency )
{
    SIXTRL_ASSERT( cavity != SIXTRL_NULLPTR );
    cavity->frequency = frequency;
    return;
}

SIXTRL_INLINE void NS(Cavity_set_lag)(
    SIXTRL_BE_ARGPTR_DEC NS(Cavity)* SIXTRL_RESTRICT cavity,
    SIXTRL_REAL_T const lag )
{
    SIXTRL_ASSERT( cavity != SIXTRL_NULLPTR );
    cavity->lag = lag;
    return;
}

SIXTRL_INLINE void NS(Cavity_copy)(
    SIXTRL_BE_ARGPTR_DEC NS(Cavity)* SIXTRL_RESTRICT destination,
    SIXTRL_BE_ARGPTR_DEC const NS(Cavity) *const SIXTRL_RESTRICT source )
{
    if( ( destination != SIXTRL_NULLPTR ) && ( source != SIXTRL_NULLPTR ) )
    {
        NS(Cavity_set_voltage)(
            destination, NS(Cavity_get_voltage)( source ) );

        NS(Cavity_set_frequency)(
            destination, NS(Cavity_get_frequency)( source ) );

        NS(Cavity_set_lag)( destination, NS(Cavity_get_lag)( source ) );
    }

    return;
}

SIXTRL_INLINE int NS(Cavity_compare_values)(
    SIXTRL_BE_ARGPTR_DEC const NS(Cavity) *const SIXTRL_RESTRICT lhs,
    SIXTRL_BE_ARGPTR_DEC const NS(Cavity) *const SIXTRL_RESTRICT rhs )
{
    int compare_value = -1;

    if( ( lhs != SIXTRL_NULLPTR ) && ( rhs != SIXTRL_NULLPTR ) )
    {
        compare_value = 0;

        if( NS(Cavity_get_voltage)( lhs ) >
            NS(Cavity_get_voltage)( rhs ) )
        {
            compare_value = +1;
        }
        else if( NS(Cavity_get_voltage)( lhs ) >
                 NS(Cavity_get_voltage)( rhs ) )
        {
            compare_value = -1;
        }

        if( compare_value == 0 )
        {
            if( NS(Cavity_get_frequency)( lhs ) >
                NS(Cavity_get_frequency)( rhs ) )
            {
                compare_value = +1;
            }
            else if( NS(Cavity_get_frequency)( lhs ) <
                     NS(Cavity_get_frequency)( rhs ) )
            {
                compare_value = -1;
            }
        }

        if( compare_value == 0 )
        {
            if( NS(Cavity_get_lag)( lhs ) > NS(Cavity_get_lag)( rhs ) )
            {
                compare_value = +1;
            }
            else if( NS(Cavity_get_lag)( lhs ) < NS(Cavity_get_lag)( rhs ) )
            {
                compare_value = -1;
            }
        }
    }
    else if( lhs != SIXTRL_NULLPTR )
    {
        compare_value = +1;
    }

    return compare_value;
}

SIXTRL_INLINE int NS(Cavity_compare_values_with_treshold)(
    SIXTRL_BE_ARGPTR_DEC const NS(Cavity) *const SIXTRL_RESTRICT lhs,
    SIXTRL_BE_ARGPTR_DEC const NS(Cavity) *const SIXTRL_RESTRICT rhs,
    SIXTRL_REAL_T const treshold )
{
    typedef SIXTRL_REAL_T       real_t;

    SIXTRL_STATIC_VAR real_t const ZERO = ( real_t )0.0;

    int compare_value = -1;

    if( ( lhs != SIXTRL_NULLPTR ) && ( rhs != SIXTRL_NULLPTR ) &&
        ( treshold >= ZERO ) )
    {
        compare_value = 0;

        if( compare_value == 0 )
        {
            real_t const diff =
                NS(Cavity_get_voltage)( lhs ) - NS(Cavity_get_voltage)( rhs );

            real_t const abs_diff = ( diff >= ZERO ) ? diff : -diff;

            if( abs_diff > treshold )
            {
                if( diff > ZERO )
                {
                    compare_value = +1;
                }
                else if( diff < ZERO )
                {
                    compare_value = -1;
                }
            }
        }

        if( compare_value == 0 )
        {
            real_t const diff = NS(Cavity_get_frequency)( lhs ) -
                                NS(Cavity_get_frequency)( rhs );

            real_t const abs_diff = ( diff >= ZERO ) ? diff : -diff;

            if( abs_diff > treshold )
            {
                if( diff > ZERO )
                {
                    compare_value = +1;
                }
                else if( diff < ZERO )
                {
                    compare_value = -1;
                }
            }
        }

        if( compare_value == 0 )
        {
            real_t const diff =
                NS(Cavity_get_lag)( lhs ) - NS(Cavity_get_lag)( rhs );

            real_t const abs_diff = ( diff >= ZERO ) ? diff : -diff;

            if( abs_diff > treshold )
            {
                if( diff > ZERO )
                {
                    compare_value = +1;
                }
                else if( diff < ZERO )
                {
                    compare_value = -1;
                }
            }
        }
    }
    else if( ( lhs != SIXTRL_NULLPTR ) && ( treshold >= ZERO ) )
    {
        compare_value = +1;
    }

    return compare_value;
}

#if !defined( _GPUCODE )

SIXTRL_INLINE bool NS(Cavity_can_be_added)(
    SIXTRL_BE_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    SIXTRL_BE_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_objects,
    SIXTRL_BE_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_slots,
    SIXTRL_BE_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_dataptrs )
{
    typedef NS(buffer_size_t) buf_size_t;

    buf_size_t const num_dataptrs =
        NS(Cavity_get_num_dataptrs)( SIXTRL_NULLPTR );

    SIXTRL_BUFFER_ARGPTR_DEC buf_size_t const* sizes  = SIXTRL_NULLPTR;
    SIXTRL_BUFFER_ARGPTR_DEC buf_size_t const* counts = SIXTRL_NULLPTR;

    return NS(Buffer_can_add_object)( buffer, sizeof( NS(Cavity) ),
        num_dataptrs, sizes, counts, requ_objects, requ_slots, requ_dataptrs );
}

SIXTRL_INLINE SIXTRL_ARGPTR_DEC NS(Cavity)* NS(Cavity_new)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer )
{
    typedef NS(buffer_size_t)                       buf_size_t;
    typedef SIXTRL_REAL_T                           real_t;
    typedef NS(Cavity)                              elem_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC   elem_t*     ptr_elem_t;

    buf_size_t const num_dataptrs =
        NS(Cavity_get_num_dataptrs)( SIXTRL_NULLPTR );

    SIXTRL_BUFFER_ARGPTR_DEC buf_size_t const* offsets = SIXTRL_NULLPTR;
    SIXTRL_BUFFER_ARGPTR_DEC buf_size_t const* sizes   = SIXTRL_NULLPTR;
    SIXTRL_BUFFER_ARGPTR_DEC buf_size_t const* counts  = SIXTRL_NULLPTR;

    elem_t temp_obj;
    temp_obj.voltage   = ( real_t )0.0;
    temp_obj.frequency = ( real_t )0.0;
    temp_obj.lag       = ( real_t )0.0;

    return ( ptr_elem_t )( uintptr_t )NS(Object_get_begin_addr)(
        NS(Buffer_add_object)( buffer, &temp_obj, sizeof( temp_obj ),
        NS(OBJECT_TYPE_CAVITY), num_dataptrs, offsets, sizes, counts ) );
}

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(Cavity)* NS(Cavity_add)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_REAL_T  const voltage,
    SIXTRL_REAL_T  const frequency,
    SIXTRL_REAL_T  const lag )
{
    typedef NS(buffer_size_t)                       buf_size_t;
    typedef NS(Cavity)                              elem_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC   elem_t*     ptr_elem_t;

    buf_size_t const num_dataptrs =
        NS(Cavity_get_num_dataptrs)( SIXTRL_NULLPTR );

    SIXTRL_BUFFER_ARGPTR_DEC buf_size_t const* offsets = SIXTRL_NULLPTR;
    SIXTRL_BUFFER_ARGPTR_DEC buf_size_t const* sizes   = SIXTRL_NULLPTR;
    SIXTRL_BUFFER_ARGPTR_DEC buf_size_t const* counts  = SIXTRL_NULLPTR;

    elem_t temp_obj;
    temp_obj.voltage   = voltage;
    temp_obj.frequency = frequency;
    temp_obj.lag       = lag;

    return ( ptr_elem_t )( uintptr_t )NS(Object_get_begin_addr)(
        NS(Buffer_add_object)( buffer, &temp_obj, sizeof( temp_obj ),
            NS(OBJECT_TYPE_CAVITY), num_dataptrs, offsets, sizes, counts ) );
}

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(Cavity)* NS(Cavity_add_copy)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_BE_ARGPTR_DEC const NS(Cavity) *const SIXTRL_RESTRICT cavity )
{
    return NS(Cavity_add)( buffer, NS(Cavity_get_voltage)( cavity ),
        NS(Cavity_get_frequency)( cavity ), NS(Cavity_get_lag)( cavity ) );

}

#endif /* !defined( _GPUCODE ) */

#if !defined(  _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

#endif /* SIXTRACKLIB_COMMON_IMPL_BEAM_ELEMENT_CAVITY_H__ */

/*end: sixtracklib/common/impl/be_cavity.h */
