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
    NS(particle_real_t) x_origin    SIXTRL_ALIGN( 8 );
    NS(particle_real_t) y_origin    SIXTRL_ALIGN( 8 );
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

SIXTRL_STATIC SIXTRL_FN NS(particle_real_t) NS(LimitEllipse_get_origin_x)(
    SIXTRL_BE_ARGPTR_DEC const NS(LimitEllipse) *const SIXTRL_RESTRICT limit );

SIXTRL_STATIC SIXTRL_FN NS(particle_real_t) NS(LimitEllipse_get_origin_y)(
    SIXTRL_BE_ARGPTR_DEC const NS(LimitEllipse) *const SIXTRL_RESTRICT limit );

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
NS(LimitEllipse_get_half_axis_product_squ)(
    SIXTRL_BE_ARGPTR_DEC const NS(LimitEllipse) *const SIXTRL_RESTRICT limit );


SIXTRL_STATIC SIXTRL_FN void NS(LimitEllipse_set_x_origin)(
    SIXTRL_BE_ARGPTR_DEC NS(LimitEllipse)* SIXTRL_RESTRICT limit,
    NS(particle_real_t) const x_origin );

SIXTRL_STATIC SIXTRL_FN void NS(LimitEllipse_set_y_origin)(
    SIXTRL_BE_ARGPTR_DEC NS(LimitEllipse)* SIXTRL_RESTRICT limit,
    NS(particle_real_t) const y_origin );


SIXTRL_STATIC SIXTRL_FN void NS(LimitEllipse_set_half_axis)(
    SIXTRL_BE_ARGPTR_DEC NS(LimitEllipse)* SIXTRL_RESTRICT limit,
    NS(particle_real_t) const x_half_axis,  
    NS(particle_real_t) const y_half_axis );

SIXTRL_STATIC SIXTRL_FN void NS(LimitEllipse_set_half_axis_squ)(
    SIXTRL_BE_ARGPTR_DEC NS(LimitEllipse)* SIXTRL_RESTRICT limit,
    NS(particle_real_t) const x_half_axis_squ, 
    NS(particle_real_t) const y_half_axis_squ );


SIXTRL_STATIC SIXTRL_FN void NS(LimitEllipse_clear)(
    SIXTRL_BE_ARGPTR_DEC NS(LimitEllipse)* SIXTRL_RESTRICT limit );

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(LimitEllipse_copy)(
    SIXTRL_BE_ARGPTR_DEC NS(LimitEllipse)* SIXTRL_RESTRICT destination,
    SIXTRL_BE_ARGPTR_DEC const NS(LimitEllipse) *const 
        SIXTRL_RESTRICT source );

#if !defined( _GPUCODE )

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
    NS(particle_real_t) const x_origin, 
    NS(particle_real_t) const y_origin,
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


SIXTRL_INLINE NS(particle_real_t) NS(LimitEllipse_get_origin_x)(
    SIXTRL_BE_ARGPTR_DEC const NS(LimitEllipse) *const SIXTRL_RESTRICT limit )
{
    SIXTRL_ASSERT( limit != SIXTRL_NULLPTR );
    return limit->x_origin;
}

SIXTRL_INLINE NS(particle_real_t) NS(LimitEllipse_get_origin_y)(
    SIXTRL_BE_ARGPTR_DEC const NS(LimitEllipse) *const SIXTRL_RESTRICT limit )
{
    SIXTRL_ASSERT( limit != SIXTRL_NULLPTR );
    return limit->y_origin;
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
NS(LimitEllipse_get_half_axis_product_squ)(
    SIXTRL_BE_ARGPTR_DEC const NS(LimitEllipse) *const SIXTRL_RESTRICT limit )
{
    SIXTRL_ASSERT( limit != SIXTRL_NULLPTR );
    return limit->a_b_squ;
}


SIXTRL_INLINE void NS(LimitEllipse_set_x_origin)(
    SIXTRL_BE_ARGPTR_DEC NS(LimitEllipse)* SIXTRL_RESTRICT limit,
    NS(particle_real_t) const x_origin )
{
    if( limit != SIXTRL_NULLPTR ) limit->x_origin = x_origin;
}

SIXTRL_INLINE void NS(LimitEllipse_set_y_origin)(
    SIXTRL_BE_ARGPTR_DEC NS(LimitEllipse)* SIXTRL_RESTRICT limit,
    NS(particle_real_t) const y_origin )
{
    if( limit != SIXTRL_NULLPTR ) limit->y_origin = y_origin;
}


SIXTRL_INLINE void NS(LimitEllipse_set_half_axis)(
    SIXTRL_BE_ARGPTR_DEC NS(LimitEllipse)* SIXTRL_RESTRICT limit,
    NS(particle_real_t) const x_half_axis,  
    NS(particle_real_t) const y_half_axis )
{
    NS(LimitEllipse_set_half_axis_squ)( limit, 
        x_half_axis * x_half_axis, y_half_axis * y_half_axis );
}

SIXTRL_INLINE void NS(LimitEllipse_set_half_axis_squ)(
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
    SIXTRL_ASSERT( limit != SIXTRL_NULLPTR );
    
    NS(LimitEllipse_set_x_origin)( limit, ( NS(particle_real_t) )0.0 );
    NS(LimitEllipse_set_y_origin)( limit, ( NS(particle_real_t) )0.0 );

    NS(LimitEllipse_set_half_axis)( 
        limit, SIXTRL_APERTURE_X_LIMIT, SIXTRL_APERTURE_Y_LIMIT );
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE NS(arch_status_t) NS(LimitEllipse_copy)(
    SIXTRL_BE_ARGPTR_DEC NS(LimitEllipse)* SIXTRL_RESTRICT destination,
    SIXTRL_BE_ARGPTR_DEC const NS(LimitEllipse) *const SIXTRL_RESTRICT source )
{
    NS(arch_status_t) status = NS(ARCH_STATUS_GENERAL_FAILURE);

    if( ( destination != SIXTRL_NULLPTR ) &&
        ( source != SIXTRL_NULLPTR ) && ( destination != source ) )
    {
        if( destination != source )
        {
            destination->x_origin = source->x_origin;
            destination->y_origin = source->y_origin;            
            destination->a_squ    = source->a_squ;
            destination->b_squ    = source->b_squ;
            destination->a_b_squ  = source->a_b_squ;
        }
        
        status = NS(ARCH_STATUS_SUCCESS);
    }

    return status;
}

#if !defined(  _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

#endif /* SIXTRL_COMMON_BE_LIMIT_ELLIPSE_C99_H__ */
/*end: sixtracklib/common/be_/be_.h */
