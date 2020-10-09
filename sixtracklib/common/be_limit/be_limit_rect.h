#ifndef SIXTRL_COMMON_BE_LIMIT_RECT_C99_H__
#define SIXTRL_COMMON_BE_LIMIT_RECT_C99_H__

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

typedef struct NS(LimitRect)
{
    NS(particle_real_t) min_x SIXTRL_ALIGN( 8 );
    NS(particle_real_t) max_x SIXTRL_ALIGN( 8 );

    NS(particle_real_t) min_y SIXTRL_ALIGN( 8 );
    NS(particle_real_t) max_y SIXTRL_ALIGN( 8 );
}
NS(LimitRect);

/* ------------------------------------------------------------------------- */

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_ARGPTR_DEC NS(LimitRect)*
NS(LimitRect_preset)( SIXTRL_BE_ARGPTR_DEC NS(LimitRect)*
    SIXTRL_RESTRICT limit ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(LimitRect_clear)(
    SIXTRL_BE_ARGPTR_DEC NS(LimitRect)* SIXTRL_RESTRICT limit ) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t) NS(LimitRect_num_dataptrs)(
    SIXTRL_BE_ARGPTR_DEC const NS(LimitRect) *const
        SIXTRL_RESTRICT limit ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t) NS(LimitRect_num_slots)(
    SIXTRL_BE_ARGPTR_DEC const NS(LimitRect) *const SIXTRL_RESTRICT limit,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(object_type_id_t) NS(LimitRect_type_id)(
    void ) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC SIXTRL_FN NS(particle_real_t) NS(LimitRect_min_x)(
    SIXTRL_BE_ARGPTR_DEC const NS(LimitRect) *const
        SIXTRL_RESTRICT limit ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(particle_real_t) NS(LimitRect_max_x)(
    SIXTRL_BE_ARGPTR_DEC const NS(LimitRect) *const
        SIXTRL_RESTRICT limit ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(particle_real_t) NS(LimitRect_min_y)(
    SIXTRL_BE_ARGPTR_DEC const NS(LimitRect) *const
        SIXTRL_RESTRICT limit ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(particle_real_t) NS(LimitRect_max_y)(
    SIXTRL_BE_ARGPTR_DEC const NS(LimitRect) *const
        SIXTRL_RESTRICT limit ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(LimitRect_set_x_limit)(
    SIXTRL_BE_ARGPTR_DEC NS(LimitRect)* SIXTRL_RESTRICT limit,
    NS(particle_real_t) const x_limit ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(LimitRect_set_y_limit)(
    SIXTRL_BE_ARGPTR_DEC NS(LimitRect)* SIXTRL_RESTRICT limit,
    NS(particle_real_t) const y_limit ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(LimitRect_set_min_x)(
    SIXTRL_BE_ARGPTR_DEC NS(LimitRect)* SIXTRL_RESTRICT limit,
    NS(particle_real_t) const min_x ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(LimitRect_set_max_x)(
    SIXTRL_BE_ARGPTR_DEC NS(LimitRect)* SIXTRL_RESTRICT limit,
    NS(particle_real_t) const max_x ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(LimitRect_set_min_y)(
    SIXTRL_BE_ARGPTR_DEC NS(LimitRect)* SIXTRL_RESTRICT limit,
    NS(particle_real_t) const min_y ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(LimitRect_set_max_y)(
    SIXTRL_BE_ARGPTR_DEC NS(LimitRect)* SIXTRL_RESTRICT limit,
    NS(particle_real_t) const max_y ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(LimitRect_copy)(
    SIXTRL_BE_ARGPTR_DEC NS(LimitRect)* SIXTRL_RESTRICT dst,
    SIXTRL_BE_ARGPTR_DEC const NS(LimitRect) *const
        SIXTRL_RESTRICT src ) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_ARGPTR_DEC NS(LimitRect) const*
NS(LimitRect_const_from_obj_index)( SIXTRL_BUFFER_OBJ_ARGPTR_DEC const NS(Object)
    *const SIXTRL_RESTRICT index_obj ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_ARGPTR_DEC NS(LimitRect)*
NS(LimitRect_from_obj_index)( SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)*
    SIXTRL_RESTRICT index_obj ) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_ARGPTR_DEC NS(LimitRect) const*
NS(LimitRect_const_from_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT buffer_begin,
    NS(buffer_size_t) const index,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_ARGPTR_DEC NS(LimitRect)*
NS(LimitRect_from_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT buffer_begin,
    NS(buffer_size_t) const index,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT;

#if !defined( _GPUCODE )

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_ARGPTR_DEC NS(LimitRect) const*
NS(LimitRect_const_from_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const index ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_ARGPTR_DEC NS(LimitRect)* NS(LimitRect_from_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const index ) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t)
NS(LimitRect_attributes_offsets)(
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT offsets_begin,
    NS(buffer_size_t) const max_num_offsets,
    SIXTRL_BE_ARGPTR_DEC const NS(LimitRect) *const SIXTRL_RESTRICT data,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT;

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t)
NS(LimitRect_attributes_sizes)(
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT sizes_begin,
    NS(buffer_size_t) const max_num_sizes,
    SIXTRL_BE_ARGPTR_DEC const NS(LimitRect) *const SIXTRL_RESTRICT data,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT;

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t)
NS(LimitRect_attributes_counts)(
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT counts_begin,
    NS(buffer_size_t) const max_num_counts,
    SIXTRL_BE_ARGPTR_DEC const NS(LimitRect) *const SIXTRL_RESTRICT data,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_EXTERN SIXTRL_HOST_FN NS(object_type_id_t)
NS(LimitRect_type_id_ext)( void ) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(LimitRect_can_be_added)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT req_objects,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT req_slots,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT req_dataptrs );

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BE_ARGPTR_DEC NS(LimitRect)*
NS(LimitRect_new)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer );

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BE_ARGPTR_DEC NS(LimitRect)*
NS(LimitRect_add)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(particle_real_t) const min_x, NS(particle_real_t) const max_x,
    NS(particle_real_t) const min_y, NS(particle_real_t) const max_y );

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BE_ARGPTR_DEC NS(LimitRect)*
NS(LimitRect_add_copy)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_BE_ARGPTR_DEC const NS(LimitRect) *const SIXTRL_RESTRICT limit );

#endif /* !defined( _GPUCODE )*/

#if !defined(  _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

/* ========================================================================= */
/*        Implementation of inline functions for NS(LimitRect)               */
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

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(LimitRect)* NS(LimitRect_preset)(
    SIXTRL_BE_ARGPTR_DEC NS(LimitRect)* SIXTRL_RESTRICT limit ) SIXTRL_NOEXCEPT
{
    if( limit != SIXTRL_NULLPTR ) NS(LimitRect_clear)( limit );
    return limit;
}

SIXTRL_INLINE NS(arch_status_t) NS(LimitRect_clear)(
    SIXTRL_BE_ARGPTR_DEC NS(LimitRect)* SIXTRL_RESTRICT limit ) SIXTRL_NOEXCEPT
{
    NS(arch_status_t) status = NS(LimitRect_set_min_x)(
        limit, ( NS(particle_real_t) )SIXTRL_LIMIT_DEFAULT_MIN_X );

    status |= NS(LimitRect_set_max_x)(
        limit, ( NS(particle_real_t) )SIXTRL_LIMIT_DEFAULT_MAX_X );

    status |= NS(LimitRect_set_min_y)(
        limit, ( NS(particle_real_t) )SIXTRL_LIMIT_DEFAULT_MIN_Y );

    status |= NS(LimitRect_set_max_y)(
        limit, ( NS(particle_real_t) )SIXTRL_LIMIT_DEFAULT_MAX_Y );

    return status;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE NS(buffer_size_t) NS(LimitRect_num_dataptrs)( SIXTRL_BE_ARGPTR_DEC
    const NS(LimitRect) *const SIXTRL_RESTRICT SIXTRL_UNUSED( limit )
) SIXTRL_NOEXCEPT
{
    return ( NS(buffer_size_t) )0u;
}

SIXTRL_INLINE NS(buffer_size_t) NS(LimitRect_num_slots)( SIXTRL_BE_ARGPTR_DEC
    const NS(LimitRect) *const SIXTRL_RESTRICT limit,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT
{
    NS(buffer_size_t) num_slots = ( NS(buffer_size_t) )0u;

    if( ( limit != SIXTRL_NULLPTR ) && ( slot_size > ( NS(buffer_size_t) )0u ) )
    {
        NS(buffer_size_t) const requ_bytes =
            NS(ManagedBuffer_get_slot_based_length)(
                sizeof( NS(LimitRect) ), slot_size );

        num_slots = requ_bytes / slot_size;
        if( num_slots * slot_size < requ_bytes ) ++num_slots;
    }

    return num_slots;
}

SIXTRL_INLINE NS(object_type_id_t) NS(LimitRect_type_id)(
    void ) SIXTRL_NOEXCEPT
{
    return ( NS(object_type_id_t) )NS(OBJECT_TYPE_LIMIT_RECT);
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE  NS(particle_real_t) NS(LimitRect_min_x)( SIXTRL_BE_ARGPTR_DEC
    const NS(LimitRect) *const SIXTRL_RESTRICT limit ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( limit != SIXTRL_NULLPTR );
    return limit->min_x;
}

SIXTRL_INLINE NS(particle_real_t) NS(LimitRect_max_x)( SIXTRL_BE_ARGPTR_DEC
    const NS(LimitRect) *const SIXTRL_RESTRICT limit ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( limit != SIXTRL_NULLPTR );
    return limit->max_x;
}

SIXTRL_INLINE NS(particle_real_t) NS(LimitRect_min_y)( SIXTRL_BE_ARGPTR_DEC
    const NS(LimitRect) *const SIXTRL_RESTRICT limit ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( limit != SIXTRL_NULLPTR );
    return limit->min_y;
}

SIXTRL_INLINE NS(particle_real_t) NS(LimitRect_max_y)( SIXTRL_BE_ARGPTR_DEC
    const NS(LimitRect) *const SIXTRL_RESTRICT limit ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( limit != SIXTRL_NULLPTR );
    return limit->max_y;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE NS(arch_status_t) NS(LimitRect_set_x_limit)(
    SIXTRL_BE_ARGPTR_DEC NS(LimitRect)* SIXTRL_RESTRICT limit,
    NS(particle_real_t) const x_limit ) SIXTRL_NOEXCEPT
{
    NS(arch_status_t) status = NS(LimitRect_set_min_x)( limit, -x_limit );
    status |= NS(LimitRect_set_max_x)( limit, +x_limit );

    SIXTRL_ASSERT( x_limit >= ( NS(particle_real_t) )0 );
    return status;
}

SIXTRL_INLINE NS(arch_status_t) NS(LimitRect_set_y_limit)(
    SIXTRL_BE_ARGPTR_DEC NS(LimitRect)* SIXTRL_RESTRICT limit,
    NS(particle_real_t) const y_limit ) SIXTRL_NOEXCEPT
{
    NS(arch_status_t) status = NS(LimitRect_set_min_y)( limit, -y_limit );
    status |= NS(LimitRect_set_max_y)( limit, +y_limit );

    SIXTRL_ASSERT( y_limit >= ( NS(particle_real_t) )0 );
    return status;
}

SIXTRL_INLINE NS(arch_status_t) NS(LimitRect_set_min_x)(
    SIXTRL_BE_ARGPTR_DEC NS(LimitRect)* SIXTRL_RESTRICT limit,
    NS(particle_real_t) const min_x ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( limit != SIXTRL_NULLPTR );
    limit->min_x = min_x;
    return ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
}

SIXTRL_INLINE NS(arch_status_t) NS(LimitRect_set_max_x)(
    SIXTRL_BE_ARGPTR_DEC NS(LimitRect)* SIXTRL_RESTRICT limit,
    NS(particle_real_t) const max_x ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( limit != SIXTRL_NULLPTR );
    limit->max_x = max_x;
    return ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
}

SIXTRL_INLINE NS(arch_status_t) NS(LimitRect_set_min_y)(
    SIXTRL_BE_ARGPTR_DEC NS(LimitRect)* SIXTRL_RESTRICT limit,
    NS(particle_real_t) const min_y ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( limit != SIXTRL_NULLPTR );
    limit->min_y = min_y;
    return ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
}

SIXTRL_INLINE NS(arch_status_t) NS(LimitRect_set_max_y)(
    SIXTRL_BE_ARGPTR_DEC NS(LimitRect)* SIXTRL_RESTRICT limit,
    NS(particle_real_t) const max_y ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( limit != SIXTRL_NULLPTR );
    limit->max_y = max_y;
    return ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE NS(arch_status_t) NS(LimitRect_copy)(
    SIXTRL_BE_ARGPTR_DEC NS(LimitRect)* SIXTRL_RESTRICT dst,
    SIXTRL_BE_ARGPTR_DEC const NS(LimitRect) *const
        SIXTRL_RESTRICT src ) SIXTRL_NOEXCEPT
{
    NS(arch_status_t) status = ( NS(arch_status_t)
        )SIXTRL_ARCH_STATUS_GENERAL_FAILURE;

    if( ( dst != SIXTRL_NULLPTR ) && ( src != SIXTRL_NULLPTR ) )
    {
        if( dst != src )
        {
            status  = NS(LimitRect_set_min_x)( dst, NS(LimitRect_min_x)( src ) );
            status |= NS(LimitRect_set_max_x)( dst, NS(LimitRect_max_x)( src ) );
            status |= NS(LimitRect_set_min_y)( dst, NS(LimitRect_min_y)( src ) );
            status |= NS(LimitRect_set_max_y)( dst, NS(LimitRect_max_y)( src ) );
        }
        else
        {
            status = ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
        }
    }

    return status;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(LimitRect) const*
NS(LimitRect_const_from_obj_index)( SIXTRL_BUFFER_OBJ_ARGPTR_DEC const NS(Object)
    *const SIXTRL_RESTRICT index_obj ) SIXTRL_NOEXCEPT
{
    typedef SIXTRL_BE_ARGPTR_DEC NS(LimitRect) const* ptr_limit_t;
    ptr_limit_t elem = SIXTRL_NULLPTR;

    if( ( index_obj != SIXTRL_NULLPTR ) &&
        ( NS(Object_get_type_id)( index_obj ) == NS(OBJECT_TYPE_LIMIT_RECT) ) &&
        ( NS(Object_get_size)( index_obj ) >= sizeof( NS(LimitRect) ) ) )
    {
        elem = ( ptr_limit_t )( uintptr_t )NS(Object_get_begin_addr)(
            index_obj );
    }

    return elem;
}

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(LimitRect)* NS(LimitRect_from_obj_index)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)* SIXTRL_RESTRICT index_obj
) SIXTRL_NOEXCEPT
{
    return ( SIXTRL_BE_ARGPTR_DEC NS(LimitRect)*
        )NS(LimitRect_const_from_obj_index)( index_obj );
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(LimitRect) const*
NS(LimitRect_const_from_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT buffer_begin,
    NS(buffer_size_t) const index,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT
{
    return NS(LimitRect_const_from_obj_index)(
        NS(ManagedBuffer_get_const_object)( buffer_begin, index, slot_size ) );
}

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(LimitRect)*
NS(LimitRect_from_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT buffer_begin,
    NS(buffer_size_t) const index,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT
{
    return NS(LimitRect_from_obj_index)( NS(ManagedBuffer_get_object)(
        buffer_begin, index, slot_size ) );
}

#if !defined( _GPUCODE )

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(LimitRect) const*
NS(LimitRect_const_from_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const index ) SIXTRL_NOEXCEPT
{
    return NS(LimitRect_const_from_obj_index)( NS(Buffer_get_const_object)(
        buffer, index ) );
}

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(LimitRect)* NS(LimitRect_from_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const index ) SIXTRL_NOEXCEPT
{
    return NS(LimitRect_from_obj_index)( NS(Buffer_get_object)(
        buffer, index ) );
}

#endif /* !defined( _GPUCODE ) */
#if !defined(  _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */
#endif /* SIXTRL_COMMON_BE_LIMIT_RECT_C99_H__ */
