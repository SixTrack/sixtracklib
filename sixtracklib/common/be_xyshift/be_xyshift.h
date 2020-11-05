#ifndef SIXTRACKLIB_COMMON_BE_DRIFT_BEAM_ELEMENT_XY_SHIFT_H__
#define SIXTRACKLIB_COMMON_BE_DRIFT_BEAM_ELEMENT_XY_SHIFT_H__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stdbool.h>
    #include <stdint.h>
    #include <stdlib.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/control/definitions.h"
    #include "sixtracklib/common/internal/buffer_main_defines.h"
    #include "sixtracklib/common/internal/beam_elements_defines.h"
    #include "sixtracklib/common/internal/objects_type_id.h"
    #include "sixtracklib/common/buffer/buffer_type.h"
    #include "sixtracklib/common/buffer/buffer_object.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined(  _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

/* ========================================================================= */

typedef struct NS(XYShift)
{
    SIXTRL_REAL_T dx SIXTRL_ALIGN( 8 );
    SIXTRL_REAL_T dy SIXTRL_ALIGN( 8 );
}
NS(XYShift);

/* ------------------------------------------------------------------------- */

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_ARGPTR_DEC NS(XYShift)* NS(XYShift_preset)(
    SIXTRL_BE_ARGPTR_DEC NS(XYShift)* SIXTRL_RESTRICT xy_shift ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(XYShift_clear)(
    SIXTRL_BE_ARGPTR_DEC NS(XYShift)* SIXTRL_RESTRICT xy_shift ) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t) NS(XYShift_num_dataptrs)(
    SIXTRL_BE_ARGPTR_DEC const NS(XYShift) *const
        SIXTRL_RESTRICT xy_shift ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t) NS(XYShift_num_slots)(
    SIXTRL_BE_ARGPTR_DEC const NS(XYShift) *const SIXTRL_RESTRICT xy_shift,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(object_type_id_t) NS(XYShift_type_id)(
    void ) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC SIXTRL_FN SIXTRL_REAL_T NS(XYShift_dx)(
    SIXTRL_BE_ARGPTR_DEC const NS(XYShift) *const
        SIXTRL_RESTRICT xy_shift ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_REAL_T NS(XYShift_dy)(
    SIXTRL_BE_ARGPTR_DEC const NS(XYShift) *const
        SIXTRL_RESTRICT xy_shift ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(XYShift_set_dx)(
    SIXTRL_BE_ARGPTR_DEC NS(XYShift)* SIXTRL_RESTRICT xy_shift,
    SIXTRL_REAL_T const dx ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(XYShift_set_dy)(
    SIXTRL_BE_ARGPTR_DEC NS(XYShift)* SIXTRL_RESTRICT xy_shift,
    SIXTRL_REAL_T const dy ) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(XYShift_copy)(
    SIXTRL_BE_ARGPTR_DEC NS(XYShift)* SIXTRL_RESTRICT destination,
    SIXTRL_BE_ARGPTR_DEC const NS(XYShift) *const
        SIXTRL_RESTRICT source ) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_ARGPTR_DEC NS(XYShift) const*
NS(XYShift_const_from_obj_index)( SIXTRL_BUFFER_OBJ_ARGPTR_DEC const NS(Object)
    *const SIXTRL_RESTRICT index_obj ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_ARGPTR_DEC NS(XYShift)*
NS(XYShift_from_obj_index)( SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)*
    SIXTRL_RESTRICT index_obj ) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_ARGPTR_DEC NS(XYShift) const*
NS(XYShift_const_from_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT buffer_begin,
    NS(buffer_size_t) const index,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_ARGPTR_DEC NS(XYShift)*
NS(XYShift_from_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT buffer_begin,
    NS(buffer_size_t) const index,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT;

#if !defined( _GPUCODE )

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_ARGPTR_DEC NS(XYShift) const*
NS(XYShift_const_from_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const index ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_ARGPTR_DEC NS(XYShift)* NS(XYShift_from_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const index ) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t)
NS(XYShift_attributes_offsets)(
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT offsets_begin,
    NS(buffer_size_t) const max_num_offsets,
    SIXTRL_BE_ARGPTR_DEC const NS(XYShift) *const SIXTRL_RESTRICT data,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT;

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t)
NS(XYShift_attributes_sizes)(
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT sizes_begin,
    NS(buffer_size_t) const max_num_sizes,
    SIXTRL_BE_ARGPTR_DEC const NS(XYShift) *const SIXTRL_RESTRICT data,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT;

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t)
NS(XYShift_attributes_counts)(
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT counts_begin,
    NS(buffer_size_t) const max_num_counts,
    SIXTRL_BE_ARGPTR_DEC const NS(XYShift) *const SIXTRL_RESTRICT data,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_EXTERN SIXTRL_HOST_FN NS(object_type_id_t)
NS(XYShift_type_id_ext)( void ) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(XYShift_can_be_added)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_objs,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_slots,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)*
        SIXTRL_RESTRICT ptr_requ_dataptrs ) SIXTRL_NOEXCEPT;

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BE_ARGPTR_DEC NS(XYShift)*
NS(XYShift_new)( SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer );

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BE_ARGPTR_DEC NS(XYShift)* NS(XYShift_add)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_REAL_T const dx, SIXTRL_REAL_T const dy );

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BE_ARGPTR_DEC NS(XYShift)*
NS(XYShift_add_copy)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_BE_ARGPTR_DEC const NS(XYShift) *const SIXTRL_RESTRICT xy_shift );

#endif /* !defined( _GPUCODE ) */
#if !defined(  _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

/* ************************************************************************* */
/* *****          Implementation of C inline functions                   *** */
/* ************************************************************************* */

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

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(XYShift)* NS(XYShift_preset)(
    SIXTRL_BE_ARGPTR_DEC NS(XYShift)* SIXTRL_RESTRICT xy_shift ) SIXTRL_NOEXCEPT
{
    if( xy_shift != SIXTRL_NULLPTR ) NS(XYShift_clear)( xy_shift );
    return xy_shift;
}

SIXTRL_INLINE NS(arch_status_t) NS(XYShift_clear)(
    SIXTRL_BE_ARGPTR_DEC NS(XYShift)* SIXTRL_RESTRICT xy_shift ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( xy_shift != SIXTRL_NULLPTR );
    return NS(XYShift_set_dx)( xy_shift, ( SIXTRL_REAL_T )0 ) |
           NS(XYShift_set_dy)( xy_shift, ( SIXTRL_REAL_T )0 );
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE NS(buffer_size_t) NS(XYShift_num_dataptrs)( SIXTRL_BE_ARGPTR_DEC
    const NS(XYShift) *const SIXTRL_RESTRICT SIXTRL_UNUSED( xy_shift )
) SIXTRL_NOEXCEPT
{
    return ( NS(buffer_size_t) )0u;
}

SIXTRL_INLINE NS(buffer_size_t) NS(XYShift_num_slots)( SIXTRL_BE_ARGPTR_DEC const
        NS(XYShift) *const SIXTRL_RESTRICT SIXTRL_UNUSED( xy_shift ),
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT
{
    NS(buffer_size_t) num_slots = ( NS(buffer_size_t) )0u;
    NS(buffer_size_t) const num_bytes = NS(ManagedBuffer_get_slot_based_length)(
                sizeof( NS(XYShift) ), slot_size );

    SIXTRL_ASSERT( slot_size > ( NS(buffer_size_t) )0 );
    num_slots = num_bytes / slot_size;
    if( num_slots * slot_size < num_bytes ) ++num_slots;
    return num_slots;
}

SIXTRL_INLINE NS(object_type_id_t) NS(XYShift_type_id)(
    void ) SIXTRL_NOEXCEPT
{
    return ( NS(object_type_id_t) )NS(OBJECT_TYPE_XYSHIFT);
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE SIXTRL_REAL_T NS(XYShift_dx)( SIXTRL_BE_ARGPTR_DEC const
    NS(XYShift) *const SIXTRL_RESTRICT xy_shift ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( xy_shift != SIXTRL_NULLPTR );
    return xy_shift->dx;
}

SIXTRL_INLINE SIXTRL_REAL_T NS(XYShift_dy)( SIXTRL_BE_ARGPTR_DEC const
    NS(XYShift) *const SIXTRL_RESTRICT xy_shift ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( xy_shift != SIXTRL_NULLPTR );
    return xy_shift->dy;
}

SIXTRL_INLINE NS(arch_status_t) NS(XYShift_set_dx)(
    SIXTRL_BE_ARGPTR_DEC NS(XYShift)* SIXTRL_RESTRICT xy_shift,
    SIXTRL_REAL_T const dx ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( xy_shift != SIXTRL_NULLPTR );
    xy_shift->dx = dx;
    return ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
}

SIXTRL_INLINE NS(arch_status_t) NS(XYShift_set_dy)(
    SIXTRL_BE_ARGPTR_DEC NS(XYShift)* SIXTRL_RESTRICT xy_shift,
    SIXTRL_REAL_T const dy ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( xy_shift != SIXTRL_NULLPTR );
    xy_shift->dy = dy;
    return ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
}

SIXTRL_INLINE NS(arch_status_t) NS(XYShift_copy)(
    SIXTRL_BE_ARGPTR_DEC NS(XYShift)* SIXTRL_RESTRICT destination,
    SIXTRL_BE_ARGPTR_DEC const NS(XYShift) *const SIXTRL_RESTRICT source
) SIXTRL_NOEXCEPT
{
    return NS(XYShift_set_dx)( destination, NS(XYShift_dx)( source ) ) |
           NS(XYShift_set_dy)( destination, NS(XYShift_dy)( source ) );
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(XYShift) const*
NS(XYShift_const_from_obj_index)( SIXTRL_BUFFER_OBJ_ARGPTR_DEC const NS(Object)
    *const SIXTRL_RESTRICT obj ) SIXTRL_NOEXCEPT
{
    typedef SIXTRL_BE_ARGPTR_DEC NS(XYShift) const* ptr_xy_shift_t;
    ptr_xy_shift_t elem = SIXTRL_NULLPTR;

    if( ( obj != SIXTRL_NULLPTR ) &&
        ( NS(Object_get_type_id)( obj ) == NS(OBJECT_TYPE_XYSHIFT) ) &&
        ( NS(Object_get_size)( obj ) >= sizeof( NS(XYShift) ) ) )
    {
        elem = ( ptr_xy_shift_t )( uintptr_t )NS(Object_get_begin_addr)( obj );
    }

    return elem;
}

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(XYShift)* NS(XYShift_from_obj_index)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)*
        SIXTRL_RESTRICT obj ) SIXTRL_NOEXCEPT
{
    return ( SIXTRL_BE_ARGPTR_DEC NS(XYShift)*
        )NS(XYShift_const_from_obj_index)( obj );
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(XYShift) const*
NS(XYShift_const_from_managed_buffer)( SIXTRL_BUFFER_DATAPTR_DEC unsigned char
        const* SIXTRL_RESTRICT buffer_begin, NS(buffer_size_t) const index,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT
{
    return NS(XYShift_const_from_obj_index)( NS(ManagedBuffer_get_const_object)(
        buffer_begin, index, slot_size ) );
}

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(XYShift)* NS(XYShift_from_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT buffer_begin,
    NS(buffer_size_t) const index, NS(buffer_size_t) const slot_size
) SIXTRL_NOEXCEPT
{
    return NS(XYShift_from_obj_index)( NS(ManagedBuffer_get_object)(
        buffer_begin, index, slot_size ) );
}

#if !defined( _GPUCODE )
SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(XYShift) const* NS(XYShift_const_from_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const index ) SIXTRL_NOEXCEPT
{
    return NS(XYShift_const_from_obj_index)( NS(Buffer_get_const_object)(
        buffer, index ) );
}

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(XYShift)* NS(XYShift_from_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const index ) SIXTRL_NOEXCEPT
{
    return NS(XYShift_from_obj_index)( NS(Buffer_get_object)(
        buffer, index ) );
}
#endif /* !defined( _GPUCODE ) */

#if !defined(  _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */
#endif /* SIXTRACKLIB_COMMON_BE_DRIFT_BEAM_ELEMENT_XY_SHIFT_H__ */
