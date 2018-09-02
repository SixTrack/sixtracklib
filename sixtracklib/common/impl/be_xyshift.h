#ifndef SIXTRACKLIB_COMMON_IMPL_BEAM_ELEMENT_XYSHIFT_H__
#define SIXTRACKLIB_COMMON_IMPL_BEAM_ELEMENT_XYSHIFT_H__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stdbool.h>
    #include <stdint.h>
    #include <stdlib.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/_impl/definitions.h"
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
    SIXTRL_ARGPTR_DEC const NS(XYShift) *const SIXTRL_RESTRICT xy_shift );

SIXTRL_FN SIXTRL_STATIC NS(XYShift)* NS(XYShift_preset)(
    SIXTRL_ARGPTR_DEC NS(XYShift)* SIXTRL_RESTRICT xy_shift );

SIXTRL_FN SIXTRL_STATIC NS(xyshift_real_t) NS(XYShift_get_dx)(
    SIXTRL_ARGPTR_DEC const NS(XYShift) *const SIXTRL_RESTRICT xy_shift );

SIXTRL_FN SIXTRL_STATIC NS(xyshift_real_t) NS(XYShift_get_dy)(
    SIXTRL_ARGPTR_DEC const NS(XYShift) *const SIXTRL_RESTRICT xy_shift );

SIXTRL_FN SIXTRL_STATIC void NS(XYShift_set_dx)(
    SIXTRL_ARGPTR_DEC NS(XYShift)* SIXTRL_RESTRICT xy_shift,
    NS(xyshift_real_t) const dx );

SIXTRL_FN SIXTRL_STATIC void NS(XYShift_set_dy)(
    SIXTRL_ARGPTR_DEC NS(XYShift)* SIXTRL_RESTRICT xy_shift,
    NS(xyshift_real_t) const dy );

SIXTRL_FN SIXTRL_STATIC bool NS(XYShift_can_be_added)(
    SIXTRL_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_objects,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_slots,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_dataptrs );

SIXTRL_FN SIXTRL_STATIC SIXTRL_ARGPTR_DEC NS(XYShift)* NS(XYShift_new)(
    SIXTRL_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer );

SIXTRL_FN SIXTRL_STATIC SIXTRL_ARGPTR_DEC NS(XYShift)* NS(XYShift_add)(
    SIXTRL_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(xyshift_real_t)  const dx,
    NS(xyshift_real_t)  const dy );

#if !defined(  _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

/* ************************************************************************* */
/* *****          Implementation of C inline functions                   *** */
/* ************************************************************************* */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/buffer.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined(  _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

SIXTRL_INLINE NS(buffer_size_t) NS(XYShift_get_num_dataptrs)(
    SIXTRL_ARGPTR_DEC const NS(XYShift) *const SIXTRL_RESTRICT xy_shift )
{
    ( void )xy_shift;
    return ( NS(buffer_size_t) )0u;
}

SIXTRL_INLINE NS(XYShift)* NS(XYShift_preset)(
    SIXTRL_ARGPTR_DEC NS(XYShift)* SIXTRL_RESTRICT xy_shift )
{
    if( xy_shift != SIXTRL_NULLPTR )
    {
        NS(XYShift_set_dx)( xy_shift, ( NS(xyshift_real_t) )0 );
        NS(XYShift_set_dy)( xy_shift, ( NS(xyshift_real_t) )0 );
    }

    return xy_shift;
}

SIXTRL_INLINE NS(xyshift_real_t) NS(XYShift_get_dx)(
    SIXTRL_ARGPTR_DEC const NS(XYShift) *const SIXTRL_RESTRICT xy_shift )
{
    return ( xy_shift != SIXTRL_NULLPTR )
        ? xy_shift->dx : ( NS(xyshift_real_t) )0;
}

SIXTRL_INLINE NS(xyshift_real_t) NS(XYShift_get_dy)(
    SIXTRL_ARGPTR_DEC const NS(XYShift) *const SIXTRL_RESTRICT xy_shift )
{
    return ( xy_shift != SIXTRL_NULLPTR )
        ? xy_shift->dy : ( NS(xyshift_real_t) )0;
}

SIXTRL_INLINE void NS(XYShift_set_dx)(
    SIXTRL_ARGPTR_DEC NS(XYShift)* SIXTRL_RESTRICT xy_shift,
    NS(xyshift_real_t) const dx )
{
    SIXTRL_ASSERT( xy_shift != SIXTRL_NULLPTR );
    xy_shift->dx = dx;
    return;
}

SIXTRL_INLINE void NS(XYShift_set_dy)(
    SIXTRL_ARGPTR_DEC NS(XYShift)* SIXTRL_RESTRICT xy_shift,
    NS(xyshift_real_t) const dy )
{
    SIXTRL_ASSERT( xy_shift != SIXTRL_NULLPTR );
    xy_shift->dy = dy;
    return;
}

SIXTRL_INLINE bool NS(XYShift_can_be_added)(
    SIXTRL_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_objects,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_slots,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_dataptrs )
{
    return NS(Buffer_can_add_object)( buffer, sizeof( NS(XYShift) ), 0u,
        SIXTRL_NULLPTR, SIXTRL_NULLPTR, ptr_requ_objects, ptr_requ_slots,
            ptr_requ_dataptrs );
}

SIXTRL_INLINE SIXTRL_ARGPTR_DEC NS(XYShift)* NS(XYShift_new)(
    SIXTRL_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer )
{
    typedef NS(XYShift) elem_t;
    typedef SIXTRL_ARGPTR_DEC NS(XYShift)* ptr_to_elem_t;
    elem_t temp_obj;

    return ( ptr_to_elem_t )( uintptr_t )NS(Object_get_begin_addr)(
        NS(Buffer_add_object)( buffer, NS(XYShift_preset)( &temp_obj ),
            sizeof( elem_t ), NS(OBJECT_TYPE_XYSHIFT), 0u,
                SIXTRL_NULLPTR, SIXTRL_NULLPTR, SIXTRL_NULLPTR ) );
}

SIXTRL_INLINE SIXTRL_ARGPTR_DEC NS(XYShift)* NS(XYShift_add)(
    SIXTRL_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(xyshift_real_t)  const dx, NS(xyshift_real_t)  const dy )
{
    typedef NS(XYShift)                 elem_t;
    typedef SIXTRL_ARGPTR_DEC elem_t*   ptr_to_elem_t;

    elem_t temp_obj;

    NS(XYShift_set_dx)( &temp_obj, dx );
    NS(XYShift_set_dy)( &temp_obj, dy );

    return ( ptr_to_elem_t )( uintptr_t )NS(Object_get_begin_addr)(
        NS(Buffer_add_object)( buffer, &temp_obj, sizeof( elem_t ),
            NS(OBJECT_TYPE_XYSHIFT), 0u, SIXTRL_NULLPTR, SIXTRL_NULLPTR,
                SIXTRL_NULLPTR ) );
}

#if !defined(  _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

#endif /* SIXTRACKLIB_COMMON_IMPL_BEAM_ELEMENT_XYSHIFT_H__ */

/*end: sixtracklib/common/impl/be_xyshift.h */
