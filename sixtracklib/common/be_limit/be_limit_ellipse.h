#ifndef SIXTRL_COMMON_BE_LIMIT_ELLIPSE_C99_H__
#define SIXTRL_COMMON_BE_LIMIT_ELLIPSE_C99_H__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stdbool.h>
    #include <stdint.h>
    #include <stdlib.h>
    #include <math.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/be_limit/definitions.h"
    #include "sixtracklib/common/internal/buffer_main_defines.h"
    #include "sixtracklib/common/internal/beam_elements_defines.h"
    #include "sixtracklib/common/internal/objects_type_id.h"
    #include "sixtracklib/common/control/definitions.h"
    #include "sixtracklib/common/particles.h"
    #include "sixtracklib/common/buffer/buffer_type.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined(  _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

typedef struct NS(LimitEllipse)
{
    NS(particle_real_t) a_squ       SIXTRL_ALIGN( 8 );
    NS(particle_real_t) b_squ       SIXTRL_ALIGN( 8 );
    NS(particle_real_t) a_b_squ     SIXTRL_ALIGN( 8 );
}
NS(LimitEllipse);

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t)
NS(LimitEllipse_get_required_num_dataptrs_on_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT buffer,
    SIXTRL_BE_ARGPTR_DEC const NS(LimitEllipse) *const SIXTRL_RESTRICT limit,
    NS(buffer_size_t) const slot_size );

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t)
NS(LimitEllipse_get_required_num_slots_on_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT buffer,
    SIXTRL_BE_ARGPTR_DEC const NS(LimitEllipse) *const SIXTRL_RESTRICT limit,
    NS(buffer_size_t) const slot_size );

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_ARGPTR_DEC NS(LimitEllipse)*
NS(LimitEllipse_preset)(
    SIXTRL_BE_ARGPTR_DEC NS(LimitEllipse)* SIXTRL_RESTRICT limit );

SIXTRL_STATIC SIXTRL_FN NS(particle_real_t) NS(LimitEllipse_get_x_half_axis)(
    SIXTRL_BE_ARGPTR_DEC const NS(LimitEllipse) *const SIXTRL_RESTRICT limit );

SIXTRL_STATIC SIXTRL_FN NS(particle_real_t)
NS(LimitEllipse_get_x_half_axis_squ)(
    SIXTRL_BE_ARGPTR_DEC const NS(LimitEllipse) *const SIXTRL_RESTRICT limit );

SIXTRL_STATIC SIXTRL_FN NS(particle_real_t) NS(LimitEllipse_get_y_half_axis)(
    SIXTRL_BE_ARGPTR_DEC const NS(LimitEllipse) *const SIXTRL_RESTRICT limit );

SIXTRL_STATIC SIXTRL_FN NS(particle_real_t)
NS(LimitEllipse_get_y_half_axis_squ)(
    SIXTRL_BE_ARGPTR_DEC const NS(LimitEllipse) *const SIXTRL_RESTRICT limit );

SIXTRL_STATIC SIXTRL_FN NS(particle_real_t)
NS(LimitEllipse_get_half_axes_product_squ)(
    SIXTRL_BE_ARGPTR_DEC const NS(LimitEllipse) *const SIXTRL_RESTRICT limit );


SIXTRL_STATIC SIXTRL_FN void NS(LimitEllipse_set_half_axes)(
    SIXTRL_BE_ARGPTR_DEC NS(LimitEllipse)* SIXTRL_RESTRICT limit,
    NS(particle_real_t) const x_half_axis,
    NS(particle_real_t) const y_half_axis );

SIXTRL_STATIC SIXTRL_FN void NS(LimitEllipse_set_half_axes_squ)(
    SIXTRL_BE_ARGPTR_DEC NS(LimitEllipse)* SIXTRL_RESTRICT limit,
    NS(particle_real_t) const x_half_axis_squ,
    NS(particle_real_t) const y_half_axis_squ );


SIXTRL_STATIC SIXTRL_FN void NS(LimitEllipse_clear)(
    SIXTRL_BE_ARGPTR_DEC NS(LimitEllipse)* SIXTRL_RESTRICT limit );

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(LimitEllipse_copy)(
    SIXTRL_BE_ARGPTR_DEC NS(LimitEllipse)* SIXTRL_RESTRICT destination,
    SIXTRL_BE_ARGPTR_DEC const NS(LimitEllipse) *const
        SIXTRL_RESTRICT source );

/* ------------------------------------------------------------------------- */

SIXTRL_FN SIXTRL_STATIC SIXTRL_BUFFER_DATAPTR_DEC NS(LimitEllipse) const*
NS(BufferIndex_get_const_limit_ellipse)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC const NS(Object) *const index_obj );

SIXTRL_FN SIXTRL_STATIC SIXTRL_BUFFER_DATAPTR_DEC NS(LimitEllipse)*
NS(BufferIndex_get_limit_ellipse)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)* index_obj );

SIXTRL_FN SIXTRL_STATIC SIXTRL_BUFFER_DATAPTR_DEC NS(LimitEllipse) const*
NS(BeamElements_managed_buffer_get_const_limit_ellipse)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT pbuffer,
    NS(buffer_size_t) const be_index, NS(buffer_size_t) const slot_size );

SIXTRL_FN SIXTRL_STATIC SIXTRL_BUFFER_DATAPTR_DEC NS(LimitEllipse)*
NS(BeamElements_managed_buffer_get_limit_ellipse)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT pbuffer,
    NS(buffer_size_t) const be_index, NS(buffer_size_t) const slot_size );

#if !defined( _GPUCODE )

SIXTRL_STATIC SIXTRL_HOST_FN SIXTRL_BUFFER_DATAPTR_DEC NS(LimitEllipse) const*
NS(BeamElements_buffer_get_const_limit_ellipse)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const be_index );

SIXTRL_STATIC SIXTRL_HOST_FN SIXTRL_BUFFER_DATAPTR_DEC NS(LimitEllipse)*
NS(BeamElements_buffer_get_limit_ellipse)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const be_index );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(buffer_size_t)
NS(LimitEllipse_get_required_num_dataptrs)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    SIXTRL_BE_ARGPTR_DEC const NS(LimitEllipse) *const SIXTRL_RESTRICT limit );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(buffer_size_t)
NS(LimitEllipse_get_required_num_slots)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    SIXTRL_BE_ARGPTR_DEC  const NS(LimitEllipse) *const SIXTRL_RESTRICT limit);

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(LimitEllipse_can_be_added)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT req_objects,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT req_slots,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT req_dataptrs );

SIXTRL_EXTERN SIXTRL_HOST_FN
SIXTRL_BUFFER_DATAPTR_DEC NS(LimitEllipse)* NS(LimitEllipse_new)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer );

SIXTRL_EXTERN SIXTRL_HOST_FN
SIXTRL_BUFFER_DATAPTR_DEC NS(LimitEllipse)* NS(LimitEllipse_add)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(particle_real_t) const x_semi_axis,
    NS(particle_real_t) const y_semi_axis );

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BUFFER_DATAPTR_DEC NS(LimitEllipse)*
NS(LimitEllipse_add_copy)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_BE_ARGPTR_DEC const NS(LimitEllipse) *const SIXTRL_RESTRICT limit );

#endif /* !defined( _GPUCODE )*/

#if !defined(  _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

/* ========================================================================= */
/*        Implementation of inline functions for NS(LimitEllipse)               */
/* ========================================================================= */

#if !defined( _GPUCODE )
    #if !defined( SIXTRL_NO_INCLUDES )
        #include "sixtracklib/common/buffer.h"
    #endif /* !defined( SIXTRL_NO_INCLUDES ) */
#endif /* !defined( _GPUCODE ) */

#if !defined(  _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

SIXTRL_INLINE NS(buffer_size_t)
NS(LimitEllipse_get_required_num_dataptrs_on_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT buffer,
    SIXTRL_BE_ARGPTR_DEC const NS(LimitEllipse) *const SIXTRL_RESTRICT limit,
    NS(buffer_size_t) const slot_size )
{
    SIXTRL_ASSERT( buffer != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( slot_size > ( NS(buffer_size_t) )0u );

    ( void )buffer;
    ( void )limit;
    ( void )slot_size;

    return ( NS(buffer_size_t) )0u;
}

SIXTRL_INLINE NS(buffer_size_t)
NS(LimitEllipse_get_required_num_slots_on_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT buffer,
    SIXTRL_BE_ARGPTR_DEC const NS(LimitEllipse) *const SIXTRL_RESTRICT limit,
    NS(buffer_size_t) const slot_size )
{
    SIXTRL_ASSERT( buffer != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( slot_size > ( NS(buffer_size_t) )0u );

    ( void )buffer;

    return ( limit != SIXTRL_NULLPTR )
        ? NS(ManagedBuffer_get_slot_based_length)(
            sizeof( *limit ), slot_size )
        : ( NS(buffer_size_t) )0u;
}

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(LimitEllipse)* NS(LimitEllipse_preset)(
    SIXTRL_BE_ARGPTR_DEC NS(LimitEllipse)* SIXTRL_RESTRICT limit )
{
    if( limit != SIXTRL_NULLPTR )
    {
        NS(LimitEllipse_clear)( limit );
    }

    return limit;
}


SIXTRL_INLINE NS(particle_real_t) NS(LimitEllipse_get_x_half_axis)(
    SIXTRL_BE_ARGPTR_DEC const NS(LimitEllipse) *const SIXTRL_RESTRICT limit )
{
    return sqrt( NS(LimitEllipse_get_x_half_axis_squ)( limit ) );
}

SIXTRL_INLINE NS(particle_real_t)
NS(LimitEllipse_get_x_half_axis_squ)(
    SIXTRL_BE_ARGPTR_DEC const NS(LimitEllipse) *const SIXTRL_RESTRICT limit )
{
    SIXTRL_ASSERT( limit != SIXTRL_NULLPTR );
    return limit->a_squ;
}

SIXTRL_INLINE NS(particle_real_t) NS(LimitEllipse_get_y_half_axis)(
    SIXTRL_BE_ARGPTR_DEC const NS(LimitEllipse) *const SIXTRL_RESTRICT limit )
{
    return sqrt( NS(LimitEllipse_get_y_half_axis_squ)( limit ) );
}

SIXTRL_INLINE NS(particle_real_t)
NS(LimitEllipse_get_y_half_axis_squ)(
    SIXTRL_BE_ARGPTR_DEC const NS(LimitEllipse) *const SIXTRL_RESTRICT limit )
{
    SIXTRL_ASSERT( limit != SIXTRL_NULLPTR );
    return limit->b_squ;
}

SIXTRL_INLINE NS(particle_real_t)
NS(LimitEllipse_get_half_axes_product_squ)(
    SIXTRL_BE_ARGPTR_DEC const NS(LimitEllipse) *const SIXTRL_RESTRICT limit )
{
    SIXTRL_ASSERT( limit != SIXTRL_NULLPTR );
    return limit->a_b_squ;
}


SIXTRL_INLINE void NS(LimitEllipse_set_half_axes)(
    SIXTRL_BE_ARGPTR_DEC NS(LimitEllipse)* SIXTRL_RESTRICT limit,
    NS(particle_real_t) const x_half_axis,
    NS(particle_real_t) const y_half_axis )
{
    NS(LimitEllipse_set_half_axes_squ)( limit,
        x_half_axis * x_half_axis, y_half_axis * y_half_axis );
}

SIXTRL_INLINE void NS(LimitEllipse_set_half_axes_squ)(
    SIXTRL_BE_ARGPTR_DEC NS(LimitEllipse)* SIXTRL_RESTRICT limit,
    NS(particle_real_t) const x_half_axis_squ,
    NS(particle_real_t) const y_half_axis_squ )
{
    if( limit != SIXTRL_NULLPTR )
    {
        limit->a_squ   = x_half_axis_squ;
        limit->b_squ   = y_half_axis_squ;
        limit->a_b_squ = x_half_axis_squ * y_half_axis_squ;
    }
}


SIXTRL_INLINE void NS(LimitEllipse_clear)(
    SIXTRL_BE_ARGPTR_DEC NS(LimitEllipse)* SIXTRL_RESTRICT limit )
{
    SIXTRL_STATIC_VAR NS(particle_real_t) const ONE_HALF =
        ( NS(particle_real_t) )0.5;

    NS(particle_real_t) const x_half_axis =
        ONE_HALF * ( SIXTRL_LIMIT_DEFAULT_MAX_X - SIXTRL_LIMIT_DEFAULT_MIN_X );

    NS(particle_real_t) const y_half_axis =
        ONE_HALF * ( SIXTRL_LIMIT_DEFAULT_MAX_Y - SIXTRL_LIMIT_DEFAULT_MIN_Y );

    SIXTRL_ASSERT( limit != SIXTRL_NULLPTR );

    NS(LimitEllipse_set_half_axes)( limit, x_half_axis, y_half_axis );
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE NS(arch_status_t) NS(LimitEllipse_copy)(
    SIXTRL_BE_ARGPTR_DEC NS(LimitEllipse)* SIXTRL_RESTRICT destination,
    SIXTRL_BE_ARGPTR_DEC const NS(LimitEllipse) *const SIXTRL_RESTRICT source )
{
    NS(arch_status_t) status = SIXTRL_ARCH_STATUS_GENERAL_FAILURE;

    if( ( destination != SIXTRL_NULLPTR ) &&
        ( source != SIXTRL_NULLPTR ) && ( destination != source ) )
    {
        if( destination != source )
        {
            destination->a_squ    = source->a_squ;
            destination->b_squ    = source->b_squ;
            destination->a_b_squ  = source->a_b_squ;
        }

        status = SIXTRL_ARCH_STATUS_SUCCESS;
    }

    return status;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(LimitEllipse) const*
NS(BufferIndex_get_const_limit_ellipse)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC const NS(Object) *const index_obj )
{
    typedef NS(LimitEllipse) beam_element_t;
    typedef SIXTRL_BUFFER_OBJ_DATAPTR_DEC beam_element_t const* ptr_to_be_t;
    ptr_to_be_t ptr_to_be = SIXTRL_NULLPTR;

    if( ( index_obj != SIXTRL_NULLPTR ) &&
        ( NS(Object_get_type_id)( index_obj ) == NS(OBJECT_TYPE_LIMIT_ELLIPSE) ) &&
        ( NS(Object_get_size)( index_obj ) >= sizeof( beam_element_t ) ) )
    {
        ptr_to_be = ( ptr_to_be_t )( uintptr_t
            )NS(Object_get_begin_addr)( index_obj );
    }

    return ptr_to_be;
}

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(LimitEllipse)*
NS(BufferIndex_get_limit_ellipse)( SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)* index_obj )
{
    return ( SIXTRL_BUFFER_DATAPTR_DEC NS(LimitEllipse)*
        )NS(BufferIndex_get_const_limit_ellipse)( index_obj );
}

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(LimitEllipse) const*
NS(BeamElements_managed_buffer_get_const_limit_ellipse)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT pbuffer,
    NS(buffer_size_t) const be_index, NS(buffer_size_t) const slot_size )
{
    return NS(BufferIndex_get_const_limit_ellipse)(
        NS(ManagedBuffer_get_const_object)( pbuffer, be_index, slot_size ) );
}

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(LimitEllipse)*
NS(BeamElements_managed_buffer_get_limit_ellipse)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT pbuffer,
    NS(buffer_size_t) const be_index, NS(buffer_size_t) const slot_size )
{
    return NS(BufferIndex_get_limit_ellipse)(
        NS(ManagedBuffer_get_object)( pbuffer, be_index, slot_size ) );
}

#if !defined( _GPUCODE )

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(LimitEllipse) const*
NS(BeamElements_buffer_get_const_limit_ellipse)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const be_index )
{
    typedef SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* ptr_raw_t;
    return NS(BeamElements_managed_buffer_get_const_limit_ellipse)(
        ( ptr_raw_t )( uintptr_t )NS(Buffer_get_data_begin_addr)( buffer ),
        be_index, NS(Buffer_get_slot_size)( buffer ) );
}

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(LimitEllipse)*
NS(BeamElements_buffer_get_limit_ellipse)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const be_index )
{
    typedef SIXTRL_BUFFER_DATAPTR_DEC unsigned char* ptr_raw_t;
    return NS(BeamElements_managed_buffer_get_limit_ellipse)(
        ( ptr_raw_t )( uintptr_t )NS(Buffer_get_data_begin_addr)( buffer ),
        be_index, NS(Buffer_get_slot_size)( buffer ) );
}

#endif /* !defined( _GPUCODE ) */

#if !defined(  _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

#endif /* SIXTRL_COMMON_BE_LIMIT_ELLIPSE_C99_H__ */
/*end: sixtracklib/common/be_/be_.h */
