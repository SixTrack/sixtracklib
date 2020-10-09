#ifndef SIXTRACKLIB_COMMON_BE_LIMIT_BE_LIMIT_RECT_ELLIPSE_C99_H__
#define SIXTRACKLIB_COMMON_BE_LIMIT_BE_LIMIT_RECT_ELLIPSE_C99_H__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stdbool.h>
    #include <stdint.h>
    #include <stdlib.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/control/definitions.h"
    #include "sixtracklib/common/particles/definitions.h"
    #include "sixtracklib/common/be_limit/definitions.h"
    #include "sixtracklib/common/internal/buffer_main_defines.h"
    #include "sixtracklib/common/internal/beam_elements_defines.h"
    #include "sixtracklib/common/internal/objects_type_id.h"
    #include "sixtracklib/common/buffer/buffer_type.h"
    #include "sixtracklib/common/buffer/buffer_object.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined(  _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

typedef struct NS(LimitRectEllipse)
{
    NS(particle_real_t)     max_x       SIXTRL_ALIGN( 8 );
    NS(particle_real_t)     max_y       SIXTRL_ALIGN( 8 );
    NS(particle_real_t)     a_squ       SIXTRL_ALIGN( 8 );
    NS(particle_real_t)     b_squ       SIXTRL_ALIGN( 8 );
    NS(particle_real_t)     a_squ_b_squ SIXTRL_ALIGN( 8 );
}
NS(LimitRectEllipse);


/* ------------------------------------------------------------------------- */

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_ARGPTR_DEC NS(LimitRectEllipse)*
NS(LimitRectEllipse_preset)( SIXTRL_BE_ARGPTR_DEC NS(LimitRectEllipse)*
    SIXTRL_RESTRICT limit ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(LimitRectEllipse_clear)(
    SIXTRL_BE_ARGPTR_DEC NS(LimitRectEllipse)*
        SIXTRL_RESTRICT limit ) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC SIXTRL_FN NS(object_type_id_t) NS(LimitRectEllipse_type_id)(
    void ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t) NS(LimitRectEllipse_num_dataptrs)(
    SIXTRL_BE_ARGPTR_DEC const NS(LimitRectEllipse) *const
        SIXTRL_RESTRICT limit ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t) NS(LimitRectEllipse_num_slots)(
    SIXTRL_BE_ARGPTR_DEC const NS(LimitRectEllipse) *const SIXTRL_RESTRICT l,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT;

/* ------------------------------------------------------------------------- */

SIXTRL_STATIC SIXTRL_FN NS(particle_real_t) NS(LimitRectEllipse_max_x)(
    SIXTRL_BE_ARGPTR_DEC const NS(LimitRectEllipse) *const
        SIXTRL_RESTRICT limit ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(particle_real_t) NS(LimitRectEllipse_max_y)(
    SIXTRL_BE_ARGPTR_DEC const NS(LimitRectEllipse) *const
        SIXTRL_RESTRICT limit ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(particle_real_t) NS(LimitRectEllipse_a_squ)(
    SIXTRL_BE_ARGPTR_DEC const NS(LimitRectEllipse) *const
        SIXTRL_RESTRICT limit ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(particle_real_t) NS(LimitRectEllipse_b_squ)(
    SIXTRL_BE_ARGPTR_DEC const NS(LimitRectEllipse) *const
        SIXTRL_RESTRICT limit )  SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(particle_real_t) NS(LimitRectEllipse_a_squ_b_squ)(
    SIXTRL_BE_ARGPTR_DEC const NS(LimitRectEllipse) *const
        SIXTRL_RESTRICT limit ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(particle_real_t) NS(LimitRectEllipse_a)(
    SIXTRL_BE_ARGPTR_DEC const NS(LimitRectEllipse) *const
        SIXTRL_RESTRICT limit ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(particle_real_t) NS(LimitRectEllipse_b)(
    SIXTRL_BE_ARGPTR_DEC const NS(LimitRectEllipse) *const
        SIXTRL_RESTRICT limit ) SIXTRL_NOEXCEPT;

/* ------------------------------------------------------------------------- */

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(LimitRectEllipse_set_max_x)(
    SIXTRL_BE_ARGPTR_DEC NS(LimitRectEllipse)* SIXTRL_RESTRICT l,
    NS(particle_real_t) const max_x ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(LimitRectEllipse_set_max_y)(
    SIXTRL_BE_ARGPTR_DEC NS(LimitRectEllipse)* SIXTRL_RESTRICT l,
    NS(particle_real_t) const max_y ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(LimitRectEllipse_set_a_squ)(
    SIXTRL_BE_ARGPTR_DEC NS(LimitRectEllipse)* SIXTRL_RESTRICT l,
    NS(particle_real_t) const a_squ ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(LimitRectEllipse_set_b_squ)(
    SIXTRL_BE_ARGPTR_DEC NS(LimitRectEllipse)* SIXTRL_RESTRICT l,
    NS(particle_real_t) const b_squ ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(LimitRectEllipse_set_a_squ_b_squ)(
    SIXTRL_BE_ARGPTR_DEC NS(LimitRectEllipse)* SIXTRL_RESTRICT l,
    NS(particle_real_t) const a_squ_b_squ ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(LimitRectEllipse_set_a)(
    SIXTRL_BE_ARGPTR_DEC NS(LimitRectEllipse)* SIXTRL_RESTRICT l,
    NS(particle_real_t) const a ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(LimitRectEllipse_set_b)(
    SIXTRL_BE_ARGPTR_DEC NS(LimitRectEllipse)* SIXTRL_RESTRICT l,
    NS(particle_real_t) const b ) SIXTRL_NOEXCEPT;

/* ------------------------------------------------------------------------- */

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_ARGPTR_DEC NS(LimitRectEllipse) const*
NS(LimitRectEllipse_const_from_obj_index)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC const NS(Object) *const obj ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_ARGPTR_DEC NS(LimitRectEllipse)*
NS(LimitRectEllipse_from_obj_index)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)* obj ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_ARGPTR_DEC NS(LimitRectEllipse) const*
NS(LimitRectEllipse_const_from_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT buffer_begin,
    NS(buffer_size_t) const index,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_ARGPTR_DEC NS(LimitRectEllipse)*
NS(LimitRectEllipse_from_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT buffer_begin,
    NS(buffer_size_t) const index,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(LimitRectEllipse_copy)(
    SIXTRL_BE_ARGPTR_DEC NS(LimitRectEllipse)* SIXTRL_RESTRICT dest,
    SIXTRL_BE_ARGPTR_DEC const NS(LimitRectEllipse) *const
        SIXTRL_RESTRICT src ) SIXTRL_NOEXCEPT;

#if !defined( _GPUCODE )

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BE_ARGPTR_DEC NS(LimitRectEllipse) const*
NS(LimitRectEllipse_const_from_buffer)( SIXTRL_BUFFER_ARGPTR_DEC const
    NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const index ) SIXTRL_NOEXCEPT;

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BE_ARGPTR_DEC NS(LimitRectEllipse)*
NS(LimitRectEllipse_from_buffer)( SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)*
        SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const index ) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t)
NS(LimitRectEllipse_attributes_offsets)(
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT offsets_begin,
    NS(buffer_size_t) const max_num_offsets,
    SIXTRL_BE_ARGPTR_DEC const NS(LimitRectEllipse) *const SIXTRL_RESTRICT l,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT;

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t)
NS(LimitRectEllipse_attributes_sizes)(
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT sizes_begin,
    NS(buffer_size_t) const max_num_sizes,
    SIXTRL_BE_ARGPTR_DEC const NS(LimitRectEllipse) *const SIXTRL_RESTRICT l,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT;

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t)
NS(LimitRectEllipse_attributes_counts)(
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT counts_begin,
    NS(buffer_size_t) const max_num_counts,
    SIXTRL_BE_ARGPTR_DEC const NS(LimitRectEllipse) *const
        SIXTRL_RESTRICT limit,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(LimitRectEllipse_can_be_added)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_num_objs,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_num_slots,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_num_dataptrs
) SIXTRL_NOEXCEPT;

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BE_ARGPTR_DEC NS(LimitRectEllipse)*
NS(LimitRectEllipse_new)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer );

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BE_ARGPTR_DEC NS(LimitRectEllipse)*
NS(LimitRectEllipse_add)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(particle_real_t) const max_x, NS(particle_real_t) const max_y,
    NS(particle_real_t) const a_squ, NS(particle_real_t) const b_squ );

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BE_ARGPTR_DEC NS(LimitRectEllipse)*
NS(LimitRectEllipse_add_copy)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_BE_ARGPTR_DEC const NS(LimitRectEllipse) *const
        SIXTRL_RESTRICT orig );

#endif /* !defined( _GPUCODE ) */
#if !defined(  _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

/* ************************************************************************* */
/* *****               Inline functions and methods                    ***** */
/* ************************************************************************* */

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #if !defined( _GPUCODE ) && defined( __cplusplus )
        #include <cmath>
    #else
        #include <math.h>
    #endif /* !defined( _GPUCODE ) && defined( __cplusplus ) */
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

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

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(LimitRectEllipse)*
NS(LimitRectEllipse_preset)( SIXTRL_BE_ARGPTR_DEC NS(LimitRectEllipse)*
    SIXTRL_RESTRICT limit ) SIXTRL_NOEXCEPT
{
    if( limit != SIXTRL_NULLPTR ) NS(LimitRectEllipse_clear)( limit );
    return limit;
}

SIXTRL_INLINE NS(arch_status_t) NS(LimitRectEllipse_clear)(
    SIXTRL_BE_ARGPTR_DEC NS(LimitRectEllipse)*
        SIXTRL_RESTRICT limit ) SIXTRL_NOEXCEPT
{
    NS(arch_status_t) status = NS(LimitRectEllipse_set_max_x)(
        limit, ( NS(particle_real_t) )SIXTRL_LIMIT_DEFAULT_MAX_X );

    status |= NS(LimitRectEllipse_set_max_y)(
        limit, ( NS(particle_real_t) )SIXTRL_LIMIT_DEFAULT_MAX_Y );

    status |= NS(LimitRectEllipse_set_a)(
        limit, ( NS(particle_real_t) )SIXTRL_APERTURE_X_LIMIT );

    status |= NS(LimitRectEllipse_set_b)(
        limit, ( NS(particle_real_t) )SIXTRL_APERTURE_Y_LIMIT );

    status |= NS(LimitRectEllipse_set_a_squ_b_squ)( limit,
        NS(LimitRectEllipse_a_squ)( limit ) *
        NS(LimitRectEllipse_b_squ)( limit ) );

    return status;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE NS(buffer_size_t) NS(LimitRectEllipse_num_dataptrs)(
    SIXTRL_BE_ARGPTR_DEC const NS(LimitRectEllipse) *const
        SIXTRL_RESTRICT SIXTRL_UNUSED( limit ) ) SIXTRL_NOEXCEPT
{
    return ( NS(buffer_size_t) )0u;
}

SIXTRL_INLINE NS(buffer_size_t) NS(LimitRectEllipse_num_slots)(
    SIXTRL_BE_ARGPTR_DEC const NS(LimitRectEllipse) *const SIXTRL_RESTRICT
        SIXTRL_UNUSED( limit ),
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT
{
    NS(buffer_size_t) num_slots = ( NS(buffer_size_t) )0u;
    NS(buffer_size_t) const num_bytes = NS(ManagedBuffer_get_slot_based_length)(
            sizeof( NS(LimitRectEllipse) ), slot_size );

    SIXTRL_ASSERT( slot_size > ( NS(buffer_size_t) )0 );
    num_slots = num_bytes / slot_size;
    if( num_slots * slot_size < num_bytes ) ++num_slots;
    return num_slots;
}

SIXTRL_INLINE NS(object_type_id_t) NS(LimitRectEllipse_type_id)(
    void ) SIXTRL_NOEXCEPT
{
    return ( NS(object_type_id_t) )NS(OBJECT_TYPE_LIMIT_RECT_ELLIPSE);
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE NS(particle_real_t) NS(LimitRectEllipse_max_x)(
    SIXTRL_BE_ARGPTR_DEC const NS(LimitRectEllipse) *const
        SIXTRL_RESTRICT limit ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( limit != SIXTRL_NULLPTR );
    return limit->max_x;
}

SIXTRL_INLINE NS(particle_real_t) NS(LimitRectEllipse_max_y)(
    SIXTRL_BE_ARGPTR_DEC const NS(LimitRectEllipse) *const
        SIXTRL_RESTRICT limit ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( limit != SIXTRL_NULLPTR );
    return limit->max_y;
}

SIXTRL_INLINE NS(particle_real_t) NS(LimitRectEllipse_a_squ)(
    SIXTRL_BE_ARGPTR_DEC const NS(LimitRectEllipse) *const
        SIXTRL_RESTRICT limit ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( limit != SIXTRL_NULLPTR );
    return limit->a_squ;
}

SIXTRL_INLINE NS(particle_real_t) NS(LimitRectEllipse_b_squ)(
    SIXTRL_BE_ARGPTR_DEC const NS(LimitRectEllipse) *const
        SIXTRL_RESTRICT limit )  SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( limit != SIXTRL_NULLPTR );
    return limit->b_squ;
}

SIXTRL_INLINE NS(particle_real_t) NS(LimitRectEllipse_a_squ_b_squ)(
    SIXTRL_BE_ARGPTR_DEC const NS(LimitRectEllipse) *const
        SIXTRL_RESTRICT limit ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( limit != SIXTRL_NULLPTR );
    return limit->a_squ_b_squ;
}

SIXTRL_INLINE NS(particle_real_t) NS(LimitRectEllipse_a)(
    SIXTRL_BE_ARGPTR_DEC const NS(LimitRectEllipse) *const
        SIXTRL_RESTRICT limit ) SIXTRL_NOEXCEPT
{
    #if defined( __cplusplus ) && !defined( _GPUCODE )
    using std::sqrt;
    #endif /* defined( __cplusplus ) && !defined( _GPUCODE ) */

    SIXTRL_ASSERT( limit != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( limit->a_squ >= ( NS(particle_real_t) )0 );
    return sqrt( limit->a_squ );
}

SIXTRL_INLINE NS(particle_real_t) NS(LimitRectEllipse_b)(
    SIXTRL_BE_ARGPTR_DEC const NS(LimitRectEllipse) *const
        SIXTRL_RESTRICT limit ) SIXTRL_NOEXCEPT
{
    #if defined( __cplusplus ) && !defined( _GPUCODE )
    using std::sqrt;
    #endif /* defined( __cplusplus ) && !defined( _GPUCODE ) */

    SIXTRL_ASSERT( limit != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( limit->b_squ >= ( NS(particle_real_t) )0 );
    return sqrt( limit->b_squ );
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE NS(arch_status_t) NS(LimitRectEllipse_set_max_x)(
    SIXTRL_BE_ARGPTR_DEC NS(LimitRectEllipse)* SIXTRL_RESTRICT limit,
    NS(particle_real_t) const max_x ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( limit != SIXTRL_NULLPTR );
    limit->max_x = max_x;
    return ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
}

SIXTRL_INLINE NS(arch_status_t) NS(LimitRectEllipse_set_max_y)(
    SIXTRL_BE_ARGPTR_DEC NS(LimitRectEllipse)* SIXTRL_RESTRICT limit,
    NS(particle_real_t) const max_y ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( limit != SIXTRL_NULLPTR );
    limit->max_y = max_y;
    return ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
}

SIXTRL_INLINE NS(arch_status_t) NS(LimitRectEllipse_set_a_squ)(
    SIXTRL_BE_ARGPTR_DEC NS(LimitRectEllipse)* SIXTRL_RESTRICT limit,
    NS(particle_real_t) const a_squ ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( limit != SIXTRL_NULLPTR );
    limit->a_squ = a_squ;
    return ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
}

SIXTRL_INLINE NS(arch_status_t) NS(LimitRectEllipse_set_b_squ)(
    SIXTRL_BE_ARGPTR_DEC NS(LimitRectEllipse)* SIXTRL_RESTRICT limit,
    NS(particle_real_t) const b_squ ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( limit != SIXTRL_NULLPTR );
    limit->b_squ = b_squ;
    return ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
}

SIXTRL_INLINE NS(arch_status_t) NS(LimitRectEllipse_set_a_squ_b_squ)(
    SIXTRL_BE_ARGPTR_DEC NS(LimitRectEllipse)* SIXTRL_RESTRICT limit,
    NS(particle_real_t) const a_squ_b_squ ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( limit != SIXTRL_NULLPTR );
    limit->a_squ_b_squ = a_squ_b_squ;
    return ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
}

SIXTRL_INLINE NS(arch_status_t) NS(LimitRectEllipse_set_a)(
    SIXTRL_BE_ARGPTR_DEC NS(LimitRectEllipse)* SIXTRL_RESTRICT limit,
    NS(particle_real_t) const a ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( limit != SIXTRL_NULLPTR );
    limit->a_squ = a * a;
    return ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
}

SIXTRL_INLINE NS(arch_status_t) NS(LimitRectEllipse_set_b)(
    SIXTRL_BE_ARGPTR_DEC NS(LimitRectEllipse)* SIXTRL_RESTRICT limit,
    NS(particle_real_t) const b ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( limit != SIXTRL_NULLPTR );
    limit->b_squ = b * b;
    return ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(LimitRectEllipse) const*
NS(LimitRectEllipse_const_from_obj_index)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC const NS(Object) *const obj ) SIXTRL_NOEXCEPT
{
    return ( ( NS(Object_get_type_id)( obj ) ==
               NS(OBJECT_TYPE_LIMIT_RECT_ELLIPSE) ) &&
             ( NS(Object_get_size)( obj ) >= sizeof( NS(LimitRectEllipse) ) ) )
        ? ( SIXTRL_BE_ARGPTR_DEC NS(LimitRectEllipse) const* )( uintptr_t
            )NS(Object_get_begin_addr)( obj )
        : SIXTRL_NULLPTR;
}

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(LimitRectEllipse)*
NS(LimitRectEllipse_from_obj_index)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)* obj ) SIXTRL_NOEXCEPT
{
    return ( SIXTRL_BE_ARGPTR_DEC NS(LimitRectEllipse)*
        )NS(LimitRectEllipse_const_from_obj_index)( obj );
}

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(LimitRectEllipse) const*
NS(LimitRectEllipse_const_from_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT buffer_begin,
    NS(buffer_size_t) const index,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT
{
    return NS(LimitRectEllipse_const_from_obj_index)(
        NS(ManagedBuffer_get_const_object)( buffer_begin, index, slot_size ) );
}

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(LimitRectEllipse)*
NS(LimitRectEllipse_from_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT buffer_begin,
    NS(buffer_size_t) const index,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT
{
    return NS(LimitRectEllipse_from_obj_index)(
        NS(ManagedBuffer_get_object)( buffer_begin, index, slot_size ) );
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE NS(arch_status_t) NS(LimitRectEllipse_copy)(
    SIXTRL_BE_ARGPTR_DEC NS(LimitRectEllipse)* SIXTRL_RESTRICT dst,
    SIXTRL_BE_ARGPTR_DEC const NS(LimitRectEllipse) *const
        SIXTRL_RESTRICT src ) SIXTRL_NOEXCEPT
{
    NS(arch_status_t) status = ( NS(arch_status_t)
        )SIXTRL_ARCH_STATUS_GENERAL_FAILURE;

    if( ( dst != SIXTRL_NULLPTR ) && ( src != SIXTRL_NULLPTR ) )
    {
        if( dst != src )
        {
            status  = NS(LimitRectEllipse_set_max_x)( dst,
                        NS(LimitRectEllipse_max_x)( src ) );

            status |= NS(LimitRectEllipse_set_max_y)( dst,
                        NS(LimitRectEllipse_max_y)( src ) );

            status |= NS(LimitRectEllipse_set_a_squ)( dst,
                        NS(LimitRectEllipse_a_squ)( src ) );

            status |= NS(LimitRectEllipse_set_b_squ)( dst,
                        NS(LimitRectEllipse_b_squ)( src ) );
        }
        else
        {
            status = ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
        }
    }

    return status;
}

#if !defined(  _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */
#endif /* SIXTRACKLIB_COMMON_BE_LIMIT_BE_LIMIT_RECT_ELLIPSE_C99_H__ */
