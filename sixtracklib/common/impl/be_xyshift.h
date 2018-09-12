#ifndef SIXTRACKLIB_COMMON_IMPL_BEAM_ELEMENT_XYSHIFT_H__
#define SIXTRACKLIB_COMMON_IMPL_BEAM_ELEMENT_XYSHIFT_H__

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

typedef SIXTRL_REAL_T NS(xyshift_real_t);

typedef struct NS(XYShift)
{
    NS(xyshift_real_t) dx SIXTRL_ALIGN( 8 );
    NS(xyshift_real_t) dy SIXTRL_ALIGN( 8 );
}
NS(XYShift);

/* ------------------------------------------------------------------------- */

SIXTRL_FN SIXTRL_STATIC NS(buffer_size_t) NS(XYShift_get_num_dataptrs)(
    SIXTRL_BE_ARGPTR_DEC const NS(XYShift) *const SIXTRL_RESTRICT xy_shift );

SIXTRL_FN SIXTRL_STATIC NS(buffer_size_t) NS(XYShift_get_num_slots)(
    SIXTRL_BE_ARGPTR_DEC  const NS(XYShift) *const SIXTRL_RESTRICT xy_shift,
    NS(buffer_size_t) const slot_size );

SIXTRL_FN SIXTRL_STATIC SIXTRL_BE_ARGPTR_DEC NS(XYShift)* NS(XYShift_preset)(
    SIXTRL_BE_ARGPTR_DEC NS(XYShift)* SIXTRL_RESTRICT xy_shift );

SIXTRL_FN SIXTRL_STATIC NS(xyshift_real_t) NS(XYShift_get_dx)(
    SIXTRL_BE_ARGPTR_DEC const NS(XYShift) *const SIXTRL_RESTRICT xy_shift );

SIXTRL_FN SIXTRL_STATIC NS(xyshift_real_t) NS(XYShift_get_dy)(
    SIXTRL_BE_ARGPTR_DEC const NS(XYShift) *const SIXTRL_RESTRICT xy_shift );

SIXTRL_FN SIXTRL_STATIC void NS(XYShift_set_dx)(
    SIXTRL_BE_ARGPTR_DEC NS(XYShift)* SIXTRL_RESTRICT xy_shift,
    NS(xyshift_real_t) const dx );

SIXTRL_FN SIXTRL_STATIC void NS(XYShift_set_dy)(
    SIXTRL_BE_ARGPTR_DEC NS(XYShift)* SIXTRL_RESTRICT xy_shift,
    NS(xyshift_real_t) const dy );

SIXTRL_FN SIXTRL_STATIC int NS(XYShift_copy)(
    SIXTRL_BE_ARGPTR_DEC NS(XYShift)* SIXTRL_RESTRICT destination,
    SIXTRL_BE_ARGPTR_DEC const NS(XYShift) *const SIXTRL_RESTRICT source );

SIXTRL_FN SIXTRL_STATIC int  NS(XYShift_compare_values)(
    SIXTRL_BE_ARGPTR_DEC const NS(XYShift) *const SIXTRL_RESTRICT lhs,
    SIXTRL_BE_ARGPTR_DEC const NS(XYShift) *const SIXTRL_RESTRICT rhs );

SIXTRL_FN SIXTRL_STATIC int NS(XYShift_compare_values_with_treshold)(
    SIXTRL_BE_ARGPTR_DEC const NS(XYShift) *const SIXTRL_RESTRICT lhs,
    SIXTRL_BE_ARGPTR_DEC const NS(XYShift) *const SIXTRL_RESTRICT rhs,
    SIXTRL_REAL_T const treshold );

#if !defined( _GPUCODE )

SIXTRL_FN SIXTRL_STATIC bool NS(XYShift_can_be_added)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_objects,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_slots,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_dataptrs );

SIXTRL_FN SIXTRL_STATIC SIXTRL_BUFFER_DATAPTR_DEC NS(XYShift)* NS(XYShift_new)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer );

SIXTRL_FN SIXTRL_STATIC SIXTRL_BUFFER_DATAPTR_DEC NS(XYShift)* NS(XYShift_add)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(xyshift_real_t)  const dx,
    NS(xyshift_real_t)  const dy );

SIXTRL_FN SIXTRL_STATIC SIXTRL_BUFFER_DATAPTR_DEC NS(XYShift)*
NS(XYShift_add_copy)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_BE_ARGPTR_DEC const NS(XYShift) *const SIXTRL_RESTRICT xyshift );

#endif /* !defined( _GPUCODE )*/

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

SIXTRL_INLINE NS(buffer_size_t) NS(XYShift_get_num_dataptrs)(
    SIXTRL_BE_ARGPTR_DEC const NS(XYShift) *const SIXTRL_RESTRICT xy_shift )
{
    ( void )xy_shift;
    return ( NS(buffer_size_t) )0u;
}

SIXTRL_INLINE NS(buffer_size_t) NS(XYShift_get_num_slots)(
    SIXTRL_BE_ARGPTR_DEC  const NS(XYShift) *const SIXTRL_RESTRICT xyshift,
    NS(buffer_size_t) const slot_size )
{
    typedef NS(buffer_size_t) buf_size_t;
    typedef NS(XYShift)     beam_element_t;

    SIXTRL_STATIC_VAR buf_size_t const ZERO = ( buf_size_t )0u;

    ( void )xyshift;

    buf_size_t extent = NS(ManagedBuffer_get_slot_based_length)(
        sizeof( beam_element_t ), slot_size );

    SIXTRL_ASSERT( ( slot_size == ZERO ) || ( ( extent % slot_size ) == ZERO ) );
    return ( slot_size > ZERO ) ? ( extent / slot_size ) : ( ZERO );
}

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(XYShift)* NS(XYShift_preset)(
    SIXTRL_BE_ARGPTR_DEC NS(XYShift)* SIXTRL_RESTRICT xy_shift )
{
    if( xy_shift != SIXTRL_NULLPTR )
    {
        NS(XYShift_set_dx)( xy_shift, ( NS(xyshift_real_t) )0 );
        NS(XYShift_set_dy)( xy_shift, ( NS(xyshift_real_t) )0 );
    }

    return xy_shift;
}

SIXTRL_INLINE NS(xyshift_real_t) NS(XYShift_get_dx)(
    SIXTRL_BE_ARGPTR_DEC const NS(XYShift) *const SIXTRL_RESTRICT xy_shift )
{
    return ( xy_shift != SIXTRL_NULLPTR )
        ? xy_shift->dx : ( NS(xyshift_real_t) )0;
}

SIXTRL_INLINE NS(xyshift_real_t) NS(XYShift_get_dy)(
    SIXTRL_BE_ARGPTR_DEC const NS(XYShift) *const SIXTRL_RESTRICT xy_shift )
{
    return ( xy_shift != SIXTRL_NULLPTR )
        ? xy_shift->dy : ( NS(xyshift_real_t) )0;
}

SIXTRL_INLINE void NS(XYShift_set_dx)(
    SIXTRL_BE_ARGPTR_DEC NS(XYShift)* SIXTRL_RESTRICT xy_shift,
    NS(xyshift_real_t) const dx )
{
    SIXTRL_ASSERT( xy_shift != SIXTRL_NULLPTR );
    xy_shift->dx = dx;
    return;
}

SIXTRL_INLINE void NS(XYShift_set_dy)(
    SIXTRL_BE_ARGPTR_DEC NS(XYShift)* SIXTRL_RESTRICT xy_shift,
    NS(xyshift_real_t) const dy )
{
    SIXTRL_ASSERT( xy_shift != SIXTRL_NULLPTR );
    xy_shift->dy = dy;
    return;
}


SIXTRL_INLINE int NS(XYShift_copy)(
    SIXTRL_BE_ARGPTR_DEC NS(XYShift)* SIXTRL_RESTRICT destination,
    SIXTRL_BE_ARGPTR_DEC const NS(XYShift) *const SIXTRL_RESTRICT source )
{
    int success = -1;

    if( ( destination != SIXTRL_NULLPTR ) && ( source != SIXTRL_NULLPTR ) )
    {
        NS(XYShift_set_dx)( destination, NS(XYShift_get_dx)( source ) );
        NS(XYShift_set_dx)( destination, NS(XYShift_get_dy)( source ) );
        success = 0;
    }

    return success;
}

SIXTRL_INLINE int NS(XYShift_compare_values)(
    SIXTRL_BE_ARGPTR_DEC const NS(XYShift) *const SIXTRL_RESTRICT lhs,
    SIXTRL_BE_ARGPTR_DEC const NS(XYShift) *const SIXTRL_RESTRICT rhs )
{
    int compare_value = -1;

    if( ( lhs != SIXTRL_NULLPTR ) && ( rhs != SIXTRL_NULLPTR ) )
    {
        compare_value = 0;

        if( NS(XYShift_get_dx)( lhs ) > NS(XYShift_get_dx)( rhs ) )
        {
            compare_value = +1;
        }
        else if( NS(XYShift_get_dx)( lhs ) < NS(XYShift_get_dx)( rhs ) )
        {
            compare_value = -1;
        }

        if( compare_value == 0 )
        {
            if( NS(XYShift_get_dy)( lhs ) > NS(XYShift_get_dy)( rhs ) )
            {
                compare_value = +1;
            }
            else if( NS(XYShift_get_dy)( lhs ) < NS(XYShift_get_dy)( rhs ) )
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

SIXTRL_INLINE int NS(XYShift_compare_values_with_treshold)(
    SIXTRL_BE_ARGPTR_DEC const NS(XYShift) *const SIXTRL_RESTRICT lhs,
    SIXTRL_BE_ARGPTR_DEC const NS(XYShift) *const SIXTRL_RESTRICT rhs,
    SIXTRL_REAL_T const treshold )
{
    typedef SIXTRL_REAL_T real_t;

    SIXTRL_STATIC_VAR real_t const ZERO = ( real_t )0.0;

    int compare_value = -1;

    if( ( lhs != SIXTRL_NULLPTR ) && ( rhs != SIXTRL_NULLPTR ) &&
        ( treshold >= ZERO ) )
    {
        compare_value = 0;

        if( compare_value == 0 )
        {
            real_t const diff =
                NS(XYShift_get_dx)( lhs ) - NS(XYShift_get_dx)( rhs );

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
                NS(XYShift_get_dy)( lhs ) - NS(XYShift_get_dy)( rhs );

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

SIXTRL_INLINE bool NS(XYShift_can_be_added)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_objects,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_slots,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_dataptrs )
{
    typedef NS(buffer_size_t)  buf_size_t;
    typedef NS(XYShift)        elem_t;

    buf_size_t const num_dataptrs =
        NS(XYShift_get_num_dataptrs)( SIXTRL_NULLPTR );

    SIXTRL_BUFFER_ARGPTR_DEC buf_size_t const* sizes   = SIXTRL_NULLPTR;
    SIXTRL_BUFFER_ARGPTR_DEC buf_size_t const* counts  = SIXTRL_NULLPTR;

    return NS(Buffer_can_add_object)( buffer, sizeof( elem_t ),
        num_dataptrs, sizes, counts, requ_objects, requ_slots, requ_dataptrs );
}

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(XYShift)* NS(XYShift_new)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer )
{
    typedef NS(buffer_size_t)  buf_size_t;
    typedef SIXTRL_REAL_T      real_t;
    typedef NS(XYShift)        elem_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC   elem_t*  ptr_to_elem_t;

    buf_size_t const num_dataptrs =
        NS(XYShift_get_num_dataptrs)( SIXTRL_NULLPTR );

    SIXTRL_BUFFER_ARGPTR_DEC buf_size_t const* offsets = SIXTRL_NULLPTR;
    SIXTRL_BUFFER_ARGPTR_DEC buf_size_t const* sizes   = SIXTRL_NULLPTR;
    SIXTRL_BUFFER_ARGPTR_DEC buf_size_t const* counts  = SIXTRL_NULLPTR;

    elem_t temp_obj;
    temp_obj.dx = ( real_t )0.0;
    temp_obj.dy = ( real_t )0.0;

    return ( ptr_to_elem_t )( uintptr_t )NS(Object_get_begin_addr)(
        NS(Buffer_add_object)( buffer, &temp_obj, sizeof( elem_t ),
            NS(OBJECT_TYPE_XYSHIFT), num_dataptrs, offsets, sizes, counts ) );
}

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(XYShift)* NS(XYShift_add)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(xyshift_real_t)  const dx, NS(xyshift_real_t)  const dy )
{
    typedef NS(buffer_size_t)  buf_size_t;
    typedef NS(XYShift)        elem_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC   elem_t*  ptr_to_elem_t;

    buf_size_t const num_dataptrs =
        NS(XYShift_get_num_dataptrs)( SIXTRL_NULLPTR );

    SIXTRL_BUFFER_ARGPTR_DEC buf_size_t const* offsets = SIXTRL_NULLPTR;
    SIXTRL_BUFFER_ARGPTR_DEC buf_size_t const* sizes   = SIXTRL_NULLPTR;
    SIXTRL_BUFFER_ARGPTR_DEC buf_size_t const* counts  = SIXTRL_NULLPTR;

    elem_t temp_obj;
    temp_obj.dx = dx;
    temp_obj.dy = dy;

    return ( ptr_to_elem_t )( uintptr_t )NS(Object_get_begin_addr)(
        NS(Buffer_add_object)( buffer, &temp_obj, sizeof( elem_t ),
            NS(OBJECT_TYPE_XYSHIFT), num_dataptrs, offsets, sizes, counts ) );
}

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(XYShift)* NS(XYShift_add_copy)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_BE_ARGPTR_DEC const NS(XYShift) *const SIXTRL_RESTRICT xyshift )
{
    return NS(XYShift_add)( buffer,
        NS(XYShift_get_dx)( xyshift ), NS(XYShift_get_dy)( xyshift ) );
}

#endif /* #if !defined( _GPUCODE ) */

#if !defined(  _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

#endif /* SIXTRACKLIB_COMMON_IMPL_BEAM_ELEMENT_XYSHIFT_H__ */

/*end: sixtracklib/common/impl/be_xyshift.h */
