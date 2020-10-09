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
    #include "sixtracklib/common/control/definitions.h"
    #include "sixtracklib/common/particles/definitions.h"
    #include "sixtracklib/common/be_limit/definitions.h"
    #include "sixtracklib/common/internal/buffer_main_defines.h"
    #include "sixtracklib/common/internal/beam_elements_defines.h"
    #include "sixtracklib/common/internal/math_functions.h"
    #include "sixtracklib/common/internal/objects_type_id.h"
    #include "sixtracklib/common/buffer/buffer_type.h"
    #include "sixtracklib/common/buffer/buffer_object.h"
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

/* ------------------------------------------------------------------------- */

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_ARGPTR_DEC NS(LimitEllipse)*
NS(LimitEllipse_preset)( SIXTRL_BE_ARGPTR_DEC NS(LimitEllipse)*
    SIXTRL_RESTRICT limit ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t)
NS(LimitEllipse_clear)( SIXTRL_BE_ARGPTR_DEC NS(LimitEllipse)*
    SIXTRL_RESTRICT limit ) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t)
NS(LimitEllipse_num_dataptrs)( SIXTRL_BE_ARGPTR_DEC const NS(LimitEllipse)
    *const SIXTRL_RESTRICT limit ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t) NS(LimitEllipse_num_slots)(
    SIXTRL_BE_ARGPTR_DEC const NS(LimitEllipse) *const SIXTRL_RESTRICT limit,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(object_type_id_t) NS(LimitEllipse_type_id)(
    void ) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC SIXTRL_FN NS(particle_real_t) NS(LimitEllipse_x_half_axis)(
    SIXTRL_BE_ARGPTR_DEC const NS(LimitEllipse) *const
        SIXTRL_RESTRICT limit ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(particle_real_t) NS(LimitEllipse_x_half_axis_squ)(
    SIXTRL_BE_ARGPTR_DEC const NS(LimitEllipse) *const
        SIXTRL_RESTRICT limit ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(particle_real_t) NS(LimitEllipse_y_half_axis)(
    SIXTRL_BE_ARGPTR_DEC const NS(LimitEllipse) *const
        SIXTRL_RESTRICT limit ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(particle_real_t) NS(LimitEllipse_y_half_axis_squ)(
    SIXTRL_BE_ARGPTR_DEC const NS(LimitEllipse) *const
        SIXTRL_RESTRICT limit ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(particle_real_t)
NS(LimitEllipse_half_axes_product_squ)( SIXTRL_BE_ARGPTR_DEC const
    NS(LimitEllipse) *const SIXTRL_RESTRICT limit ) SIXTRL_NOEXCEPT;


SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(LimitEllipse_set_half_axes)(
    SIXTRL_BE_ARGPTR_DEC NS(LimitEllipse)* SIXTRL_RESTRICT limit,
    NS(particle_real_t) const x_half_axis,
    NS(particle_real_t) const y_half_axis ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(LimitEllipse_set_half_axes_squ)(
    SIXTRL_BE_ARGPTR_DEC NS(LimitEllipse)* SIXTRL_RESTRICT limit,
    NS(particle_real_t) const x_half_axis_squ,
    NS(particle_real_t) const y_half_axis_squ ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(LimitEllipse_copy)(
    SIXTRL_BE_ARGPTR_DEC NS(LimitEllipse)* SIXTRL_RESTRICT destination,
    SIXTRL_BE_ARGPTR_DEC const NS(LimitEllipse) *const
        SIXTRL_RESTRICT source ) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_ARGPTR_DEC NS(LimitEllipse) const*
NS(LimitEllipse_const_from_obj_index)( SIXTRL_BUFFER_OBJ_ARGPTR_DEC const
    NS(Object) *const SIXTRL_RESTRICT index_obj ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_ARGPTR_DEC NS(LimitEllipse)*
NS(LimitEllipse_from_obj_index)( SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)*
    SIXTRL_RESTRICT index_obj ) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_ARGPTR_DEC NS(LimitEllipse) const*
NS(LimitEllipse_const_from_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT buffer_begin,
    NS(buffer_size_t) const index,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_ARGPTR_DEC NS(LimitEllipse)*
NS(LimitEllipse_from_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT buffer_begin,
    NS(buffer_size_t) const index,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT;

#if !defined( _GPUCODE )

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_ARGPTR_DEC NS(LimitEllipse) const*
NS(LimitEllipse_const_from_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const index ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_ARGPTR_DEC NS(LimitEllipse)*
NS(LimitEllipse_from_buffer)( SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)*
    SIXTRL_RESTRICT buffer, NS(buffer_size_t) const index ) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t)
NS(LimitEllipse_attributes_offsets)(
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT offsets_begin,
    NS(buffer_size_t) const max_num_offsets,
    SIXTRL_BE_ARGPTR_DEC const NS(LimitEllipse) *const SIXTRL_RESTRICT data,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT;

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t)
NS(LimitEllipse_attributes_sizes)(
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT sizes_begin,
    NS(buffer_size_t) const max_num_sizes,
    SIXTRL_BE_ARGPTR_DEC const NS(LimitEllipse) *const SIXTRL_RESTRICT data,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT;

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t)
NS(LimitEllipse_attributes_counts)(
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT counts_begin,
    NS(buffer_size_t) const max_num_counts,
    SIXTRL_BE_ARGPTR_DEC const NS(LimitEllipse) *const SIXTRL_RESTRICT data,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_EXTERN SIXTRL_HOST_FN NS(object_type_id_t)
NS(LimitEllipse_type_id_ext)( void ) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(LimitEllipse_can_be_added)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT req_objects,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT req_slots,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)*
        SIXTRL_RESTRICT req_dataptrs ) SIXTRL_NOEXCEPT;

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BE_ARGPTR_DEC NS(LimitEllipse)*
NS(LimitEllipse_new)( SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)*
    SIXTRL_RESTRICT buffer );

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BE_ARGPTR_DEC NS(LimitEllipse)*
NS(LimitEllipse_add)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(particle_real_t) const x_semi_axis,
    NS(particle_real_t) const y_semi_axis );

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BE_ARGPTR_DEC NS(LimitEllipse)*
NS(LimitEllipse_add_copy)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_BE_ARGPTR_DEC const NS(LimitEllipse) *const SIXTRL_RESTRICT orig );

#endif /* !defined( _GPUCODE )*/

#if !defined(  _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

/* ========================================================================= */
/*        Implementation of inline functions for NS(LimitEllipse)               */
/* ========================================================================= */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/buffer/managed_buffer_minimal.h"
    #include "sixtracklib/common/internal/compiler_attributes.h"
    #if !defined( _GPUCODE )
    #include "sixtracklib/common/buffer.h"
    #endif /* !defined( _GPUCODE ) */
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined(  _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(LimitEllipse)* NS(LimitEllipse_preset)(
    SIXTRL_BE_ARGPTR_DEC NS(LimitEllipse)*
        SIXTRL_RESTRICT limit ) SIXTRL_NOEXCEPT
{
    if( limit != SIXTRL_NULLPTR ) NS(LimitEllipse_clear)( limit );
    return limit;
}

SIXTRL_INLINE NS(arch_status_t) NS(LimitEllipse_clear)(
    SIXTRL_BE_ARGPTR_DEC NS(LimitEllipse)* SIXTRL_RESTRICT limit ) SIXTRL_NOEXCEPT
{
    return NS(LimitEllipse_set_half_axes)( limit,
        ( NS(particle_real_t) )SIXTRL_LIMIT_DEFAULT_X_HALF_AXIS,
        ( NS(particle_real_t) )SIXTRL_LIMIT_DEFAULT_Y_HALF_AXIS );
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE NS(buffer_size_t) NS(LimitEllipse_num_dataptrs)( SIXTRL_BE_ARGPTR_DEC
    const NS(LimitEllipse) *const SIXTRL_RESTRICT SIXTRL_UNUSED( limit )
) SIXTRL_NOEXCEPT
{
    return ( NS(buffer_size_t) )0u;
}

SIXTRL_INLINE NS(buffer_size_t) NS(LimitEllipse_num_slots)( SIXTRL_BE_ARGPTR_DEC const
        NS(LimitEllipse) *const SIXTRL_RESTRICT SIXTRL_UNUSED( limit ),
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT
{
    NS(buffer_size_t) num_slots = ( NS(buffer_size_t) )0u;
    NS(buffer_size_t) const num_bytes = NS(ManagedBuffer_get_slot_based_length)(
                sizeof( NS(LimitEllipse) ), slot_size );

    SIXTRL_ASSERT( slot_size > ( NS(buffer_size_t) )0 );
    num_slots = num_bytes / slot_size;
    if( num_slots * slot_size < num_bytes ) ++num_slots;
    return num_slots;
}

SIXTRL_INLINE NS(object_type_id_t) NS(LimitEllipse_type_id)(
    void ) SIXTRL_NOEXCEPT
{
    return ( NS(object_type_id_t) )NS(OBJECT_TYPE_LIMIT_ELLIPSE);
}

SIXTRL_INLINE NS(particle_real_t) NS(LimitEllipse_x_half_axis)(
    SIXTRL_BE_ARGPTR_DEC const NS(LimitEllipse) *const
        SIXTRL_RESTRICT limit ) SIXTRL_NOEXCEPT
{
    return NS(sqrt)( NS(LimitEllipse_x_half_axis_squ)( limit ) );
}

SIXTRL_INLINE NS(particle_real_t) NS(LimitEllipse_x_half_axis_squ)(
    SIXTRL_BE_ARGPTR_DEC const NS(LimitEllipse) *const
        SIXTRL_RESTRICT limit ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( limit != SIXTRL_NULLPTR );
    return limit->a_squ;
}

SIXTRL_INLINE NS(particle_real_t) NS(LimitEllipse_y_half_axis)(
    SIXTRL_BE_ARGPTR_DEC const NS(LimitEllipse) *const
        SIXTRL_RESTRICT limit ) SIXTRL_NOEXCEPT
{
    return NS(sqrt)( NS(LimitEllipse_y_half_axis_squ)( limit ) );
}

SIXTRL_INLINE NS(particle_real_t) NS(LimitEllipse_y_half_axis_squ)(
    SIXTRL_BE_ARGPTR_DEC const NS(LimitEllipse) *const
        SIXTRL_RESTRICT limit ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( limit != SIXTRL_NULLPTR );
    return limit->b_squ;
}

SIXTRL_INLINE NS(particle_real_t) NS(LimitEllipse_half_axes_product_squ)(
    SIXTRL_BE_ARGPTR_DEC const NS(LimitEllipse) *const
        SIXTRL_RESTRICT limit ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( limit != SIXTRL_NULLPTR );
    return limit->a_b_squ;
}


SIXTRL_INLINE NS(arch_status_t) NS(LimitEllipse_set_half_axes)(
    SIXTRL_BE_ARGPTR_DEC NS(LimitEllipse)* SIXTRL_RESTRICT limit,
    NS(particle_real_t) const x_half_axis,
    NS(particle_real_t) const y_half_axis ) SIXTRL_NOEXCEPT
{
    return NS(LimitEllipse_set_half_axes_squ)( limit,
        x_half_axis * x_half_axis, y_half_axis * y_half_axis );
}

SIXTRL_INLINE NS(arch_status_t) NS(LimitEllipse_set_half_axes_squ)(
    SIXTRL_BE_ARGPTR_DEC NS(LimitEllipse)* SIXTRL_RESTRICT limit,
    NS(particle_real_t) const x_half_axis_squ,
    NS(particle_real_t) const y_half_axis_squ ) SIXTRL_NOEXCEPT
{
    NS(arch_status_t) status = ( NS(arch_status_t)
        )SIXTRL_ARCH_STATUS_GENERAL_FAILURE;

    if( ( limit != SIXTRL_NULLPTR ) &&
        ( x_half_axis_squ >= ( NS(particle_real_t) )0 ) &&
        ( y_half_axis_squ >= ( NS(particle_real_t) )0 ) )
    {
        limit->a_squ   = x_half_axis_squ;
        limit->b_squ   = y_half_axis_squ;
        limit->a_b_squ = x_half_axis_squ * y_half_axis_squ;
        status = ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
    }

    return status;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE NS(arch_status_t) NS(LimitEllipse_copy)(
    SIXTRL_BE_ARGPTR_DEC NS(LimitEllipse)* SIXTRL_RESTRICT dst,
    SIXTRL_BE_ARGPTR_DEC const NS(LimitEllipse) *const
        SIXTRL_RESTRICT src ) SIXTRL_NOEXCEPT
{
    NS(arch_status_t) status = ( NS(arch_status_t)
        )SIXTRL_ARCH_STATUS_GENERAL_FAILURE;

    if( ( dst != SIXTRL_NULLPTR ) && ( src != SIXTRL_NULLPTR ) )
    {
        if( dst != src )
        {
            status = NS(LimitEllipse_set_half_axes_squ)( dst,
                NS(LimitEllipse_x_half_axis_squ)( src ),
                NS(LimitEllipse_y_half_axis_squ)( src ) );
        }
        else
        {
            status = ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
        }
    }

    return status;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(LimitEllipse) const*
NS(LimitEllipse_const_from_obj_index)( SIXTRL_BUFFER_OBJ_ARGPTR_DEC const
    NS(Object) *const SIXTRL_RESTRICT obj ) SIXTRL_NOEXCEPT
{
    typedef SIXTRL_BE_ARGPTR_DEC NS(LimitEllipse) const* ptr_elem_t;
    ptr_elem_t elem = SIXTRL_NULLPTR;

    if( ( obj != SIXTRL_NULLPTR ) && ( NS(Object_get_type_id)( obj ) ==
          NS(OBJECT_TYPE_LIMIT_ELLIPSE) ) &&
        ( NS(Object_get_size)( obj ) >= sizeof( NS(LimitEllipse) ) ) )
    {
        elem = ( ptr_elem_t )( uintptr_t )NS(Object_get_begin_addr)( obj );
    }

    return elem;
}

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(LimitEllipse)*
NS(LimitEllipse_from_obj_index)( SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)*
    SIXTRL_RESTRICT obj ) SIXTRL_NOEXCEPT
{
    return ( SIXTRL_BE_ARGPTR_DEC NS(LimitEllipse)*
        )NS(LimitEllipse_const_from_obj_index)( obj );
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(LimitEllipse) const*
NS(LimitEllipse_const_from_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT buffer_begin,
    NS(buffer_size_t) const index,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT
{
    return NS(LimitEllipse_const_from_obj_index)(
        NS(ManagedBuffer_get_const_object)( buffer_begin, index, slot_size ) );
}

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(LimitEllipse)*
NS(LimitEllipse_from_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT buffer_begin,
    NS(buffer_size_t) const index,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT
{
    return NS(LimitEllipse_from_obj_index)(
        NS(ManagedBuffer_get_object)( buffer_begin, index, slot_size ) );
}

#if !defined( _GPUCODE )
SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(LimitEllipse) const*
NS(LimitEllipse_const_from_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const index ) SIXTRL_NOEXCEPT
{
    return NS(LimitEllipse_const_from_obj_index)( NS(Buffer_get_const_object)(
        buffer, index ) );
}

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(LimitEllipse)*
NS(LimitEllipse_from_buffer)( SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)*
    SIXTRL_RESTRICT buffer, NS(buffer_size_t) const index ) SIXTRL_NOEXCEPT
{
    return NS(LimitEllipse_from_obj_index)(
        NS(Buffer_get_object)( buffer, index ) );
}
#endif /* !defined( _GPUCODE ) */

#if !defined(  _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */
#endif /* SIXTRL_COMMON_BE_LIMIT_ELLIPSE_C99_H__ */
