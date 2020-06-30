#ifndef SIXTRACKLIB_COMMON_BE_SROTATION_BEAM_ELEMENT_SROTATION_H__
#define SIXTRACKLIB_COMMON_BE_SROTATION_BEAM_ELEMENT_SROTATION_H__

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

typedef struct NS(SRotation)
{
    SIXTRL_REAL_T cos_z SIXTRL_ALIGN( 8 );
    SIXTRL_REAL_T sin_z SIXTRL_ALIGN( 8 );
}
NS(SRotation);


SIXTRL_STATIC SIXTRL_FN NS(object_type_id_t) NS(SRotation_type_id)(
    void ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t) NS(SRotation_num_dataptrs)(
    SIXTRL_BE_ARGPTR_DEC const NS(SRotation) *const
        SIXTRL_RESTRICT srotation ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t) NS(SRotation_num_slots)(
    SIXTRL_BE_ARGPTR_DEC  const NS(SRotation) *const SIXTRL_RESTRICT srot,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT;

/* . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . */

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_ARGPTR_DEC NS(SRotation)*
NS(SRotation_preset)( SIXTRL_BE_ARGPTR_DEC NS(SRotation)*
    SIXTRL_RESTRICT srotation ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(SRotation_clear)(
    SIXTRL_BE_ARGPTR_DEC NS(SRotation)*
        SIXTRL_RESTRICT srotation ) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC SIXTRL_FN SIXTRL_REAL_T NS(SRotation_angle_deg)(
    SIXTRL_BE_ARGPTR_DEC const NS(SRotation) *const
        SIXTRL_RESTRICT srotation ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_REAL_T NS(SRotation_angle)(
    SIXTRL_BE_ARGPTR_DEC const NS(SRotation) *const
        SIXTRL_RESTRICT srotation ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_REAL_T NS(SRotation_cos_angle)(
    SIXTRL_BE_ARGPTR_DEC const NS(SRotation) *const
        SIXTRL_RESTRICT srotation ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_REAL_T NS(SRotation_sin_angle)(
    SIXTRL_BE_ARGPTR_DEC const NS(SRotation) *const
        SIXTRL_RESTRICT srotation ) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(SRotation_set_angle)(
    SIXTRL_BE_ARGPTR_DEC NS(SRotation)* SIXTRL_RESTRICT srotation,
    SIXTRL_REAL_T const angle ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(SRotation_set_angle_deg)(
    SIXTRL_BE_ARGPTR_DEC NS(SRotation)* SIXTRL_RESTRICT srotation,
    SIXTRL_REAL_T const angle_deg ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(SRotation_set_cos_angle)(
    SIXTRL_BE_ARGPTR_DEC NS(SRotation)* SIXTRL_RESTRICT srotation,
    SIXTRL_REAL_T const cos_angle ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(SRotation_set_sin_angle)(
    SIXTRL_BE_ARGPTR_DEC NS(SRotation)* SIXTRL_RESTRICT srotation,
    SIXTRL_REAL_T const sin_angle ) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC SIXTRL_FN int NS(SRotation_copy)(
    SIXTRL_BE_ARGPTR_DEC NS(SRotation)* SIXTRL_RESTRICT dst,
    SIXTRL_BE_ARGPTR_DEC const NS(SRotation) *const
        SIXTRL_RESTRICT src ) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_ARGPTR_DEC NS(SRotation) const*
NS(SRotation_const_from_obj_index)( SIXTRL_BUFFER_OBJ_ARGPTR_DEC const
    NS(Object) *const index_obj ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_ARGPTR_DEC NS(SRotation)*
NS(SRotation_from_obj_index)( SIXTRL_BUFFER_OBJ_ARGPTR_DEC
    NS(Object)* index_obj ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_ARGPTR_DEC NS(SRotation) const*
NS(SRotation_const_from_manged_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT pbuffer,
    NS(buffer_size_t) const be_index,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_ARGPTR_DEC NS(SRotation)*
NS(SRotation_from_manged_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT pbuffer,
    NS(buffer_size_t) const be_index,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT;

#if !defined( _GPUCODE )

SIXTRL_STATIC SIXTRL_HOST_FN SIXTRL_BE_ARGPTR_DEC NS(SRotation) const*
NS(SRotation_const_from_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const be_index ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_HOST_FN SIXTRL_BE_ARGPTR_DEC NS(SRotation)*
NS(SRotation_from_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const be_index ) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t) NS(SRotation_attributes_offsets)(
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT offsets_begin,
    NS(buffer_size_t) const max_num_offsets,
    SIXTRL_BE_ARGPTR_DEC const NS(SRotation) *const SIXTRL_RESTRICT data,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT;

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t) NS(SRotation_attributes_sizes)(
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT sizes_begin,
    NS(buffer_size_t) const max_num_sizes,
    SIXTRL_BE_ARGPTR_DEC const NS(SRotation) *const SIXTRL_RESTRICT data,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT;

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t) NS(SRotation_attributes_counts)(
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT counts_begin,
    NS(buffer_size_t) const max_num_counts,
    SIXTRL_BE_ARGPTR_DEC const NS(SRotation) *const SIXTRL_RESTRICT data,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_EXTERN SIXTRL_HOST_FN NS(object_type_id_t) NS(SRotation_type_id_ext)(
    void ) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(SRotation_can_be_added)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_objects,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_slots,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)*
        SIXTRL_RESTRICT requ_dataptrs ) SIXTRL_NOEXCEPT;

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BUFFER_DATAPTR_DEC NS(SRotation)*
NS(SRotation_new)( SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buf );

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BUFFER_DATAPTR_DEC NS(SRotation)*
NS(SRotation_add)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_REAL_T const cos_angle, SIXTRL_REAL_T const sin_angle );

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BUFFER_DATAPTR_DEC NS(SRotation)*
NS(SRotation_add_copy)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_BE_ARGPTR_DEC const NS(SRotation) *const SIXTRL_RESTRICT srotation );

#endif /* !defined( _GPUCODE ) */
#if !defined(  _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

/* ************************************************************************* */
/* *****          Implementation of C inline functions                   *** */
/* ************************************************************************* */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/buffer/managed_buffer.h"
    #include "sixtracklib/common/internal/compiler_attributes.h"
    #include "sixtracklib/common/internal/math_functions.h"
    #include "sixtracklib/common/internal/math_constants.h"
    #include "sixtracklib/common/internal/type_comparison_helpers.hpp"
    #if !defined( _GPUCODE )
        #include "sixtracklib/common/buffer.h"
    #endif /* !defined( _GPUCODE ) */
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined(  _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

SIXTRL_INLINE NS(object_type_id_t) NS(SRotation_type_id)( void ) SIXTRL_NOEXCEPT
{
    return ( NS(object_type_id_t) )NS(OBJECT_TYPE_SROTATION);
}

SIXTRL_INLINE NS(buffer_size_t) NS(SRotation_num_dataptrs)( SIXTRL_BE_ARGPTR_DEC
    const NS(SRotation) *const SIXTRL_RESTRICT
        SIXTRL_UNUSED( srotation ) ) SIXTRL_NOEXCEPT
{
    return ( NS(buffer_size_t) )0u;
}

SIXTRL_INLINE NS(buffer_size_t) NS(SRotation_num_slots)(
    SIXTRL_BE_ARGPTR_DEC  const NS(SRotation) *const SIXTRL_RESTRICT
        SIXTRL_UNUSED( srotation ),
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT
{
    typedef NS(buffer_size_t) st_size_t;
    st_size_t num_slots = ( st_size_t )0u;

    if( slot_size > ( st_size_t )0 )
    {
        st_size_t const num_bytes = NS(ManagedBuffer_get_slot_based_length)(
            sizeof( NS(SRotation) ), slot_size );
        num_slots = num_bytes / slot_size;
        if( num_slots * slot_size < num_bytes ) ++num_slots;
    }

    return num_slots;
}

/* . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . */

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(SRotation)* NS(SRotation_preset)(
    SIXTRL_BE_ARGPTR_DEC NS(SRotation)* SIXTRL_RESTRICT
        srotation ) SIXTRL_NOEXCEPT
{
    if( srotation != SIXTRL_NULLPTR ) NS(SRotation_clear)( srotation );
    return srotation;
}

SIXTRL_INLINE NS(arch_status_t) NS(SRotation_clear)( SIXTRL_BE_ARGPTR_DEC
    NS(SRotation)* SIXTRL_RESTRICT srotation ) SIXTRL_NOEXCEPT
{
    NS(arch_status_t) status = ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
    status |= NS(SRotation_set_cos_angle)( srotation, ( SIXTRL_REAL_T )1 );
    status |= NS(SRotation_set_sin_angle)( srotation, ( SIXTRL_REAL_T )0 );
    return status;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE SIXTRL_REAL_T NS(SRotation_angle_deg)( SIXTRL_BE_ARGPTR_DEC const
    NS(SRotation) *const SIXTRL_RESTRICT srotation ) SIXTRL_NOEXCEPT
{
    return NS(MathConst_rad2deg)() * NS(SRotation_angle)( srotation );
}

SIXTRL_INLINE SIXTRL_REAL_T NS(SRotation_angle)( SIXTRL_BE_ARGPTR_DEC const
    NS(SRotation) *const SIXTRL_RESTRICT srotation ) SIXTRL_NOEXCEPT
{
    return NS(atan2)( NS(SRotation_sin_angle)( srotation ),
                      NS(SRotation_cos_angle)( srotation ) );
}

SIXTRL_INLINE SIXTRL_REAL_T NS(SRotation_cos_angle)( SIXTRL_BE_ARGPTR_DEC const
    NS(SRotation) *const SIXTRL_RESTRICT srotation ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( srotation != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( NS(Type_comp_all_less_or_equal)( NS(abs)( srotation->cos_z ),
        ( SIXTRL_REAL_T )1 ) );
    SIXTRL_ASSERT( NS(Type_comp_all_less_or_equal)( NS(abs)( srotation->sin_z ),
        ( SIXTRL_REAL_T )1 ) );

    return srotation->cos_z;
}

SIXTRL_INLINE SIXTRL_REAL_T NS(SRotation_sin_angle)( SIXTRL_BE_ARGPTR_DEC const
    NS(SRotation) *const SIXTRL_RESTRICT srotation ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( srotation != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( NS(Type_comp_all_less_or_equal)( NS(abs)( srotation->cos_z ),
        ( SIXTRL_REAL_T )1 ) );
    SIXTRL_ASSERT( NS(Type_comp_all_less_or_equal)( NS(abs)( srotation->sin_z ),
        ( SIXTRL_REAL_T )1 ) );

    return srotation->sin_z;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE NS(arch_status_t) NS(SRotation_set_angle)(
    SIXTRL_BE_ARGPTR_DEC NS(SRotation)* SIXTRL_RESTRICT srotation,
    SIXTRL_REAL_T const angle ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( srotation != SIXTRL_NULLPTR );
    srotation->cos_z = NS(cos)( angle );
    srotation->sin_z = NS(sin)( angle );
    return ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
}

SIXTRL_INLINE NS(arch_status_t) NS(SRotation_set_angle_deg)(
    SIXTRL_BE_ARGPTR_DEC NS(SRotation)* SIXTRL_RESTRICT srotation,
    SIXTRL_REAL_T const angle_deg ) SIXTRL_NOEXCEPT
{
    return NS(SRotation_set_angle)( srotation,
        NS(MathConst_deg2rad)() * angle_deg );
}

SIXTRL_INLINE NS(arch_status_t) NS(SRotation_set_cos_angle)(
    SIXTRL_BE_ARGPTR_DEC NS(SRotation)* SIXTRL_RESTRICT srotation,
    SIXTRL_REAL_T const cos_angle ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( srotation != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( NS(Type_comp_all_less_or_equal)( NS(abs)( cos_angle ),
        ( SIXTRL_REAL_T )1 ) );

    srotation->cos_z = cos_angle;
    return ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
}

SIXTRL_INLINE NS(arch_status_t) NS(SRotation_set_sin_angle)(
    SIXTRL_BE_ARGPTR_DEC NS(SRotation)* SIXTRL_RESTRICT srotation,
    SIXTRL_REAL_T const sin_angle ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( srotation != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( NS(Type_comp_all_less_or_equal)( NS(abs)( sin_angle ),
        ( SIXTRL_REAL_T )1 ) );

    srotation->sin_z = sin_angle;
    return ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE int NS(SRotation_copy)(
    SIXTRL_BE_ARGPTR_DEC NS(SRotation)* SIXTRL_RESTRICT dst,
    SIXTRL_BE_ARGPTR_DEC const NS(SRotation) *const
        SIXTRL_RESTRICT src ) SIXTRL_NOEXCEPT
{
    NS(arch_status_t) status = ( NS(arch_status_t)
        )SIXTRL_ARCH_STATUS_GENERAL_FAILURE;

    if( ( dst != SIXTRL_NULLPTR ) && ( src != SIXTRL_NULLPTR ) )
    {
        if( dst != src )
        {
            status = NS(SRotation_set_cos_angle)(
                dst, NS(SRotation_cos_angle)( src ) );

            status |= NS(SRotation_set_sin_angle)(
                dst, NS(SRotation_sin_angle)( src ) );
        }
        else
        {
            status = ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
        }
    }

    return status;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(SRotation) const*
NS(SRotation_const_from_obj_index)( SIXTRL_BUFFER_OBJ_ARGPTR_DEC const
    NS(Object) *const obj ) SIXTRL_NOEXCEPT
{
    SIXTRL_BE_ARGPTR_DEC NS(SRotation) const* elem = SIXTRL_NULLPTR;
    if( ( obj != SIXTRL_NULLPTR ) &&
        ( NS(Object_get_type_id)( obj ) == NS(OBJECT_TYPE_SROTATION) ) &&
        ( NS(Object_get_size)( obj ) >= sizeof( NS(SRotation) ) ) )
    {
        elem = ( SIXTRL_BE_ARGPTR_DEC NS(SRotation) const* )(
            uintptr_t )NS(Object_get_begin_addr)( obj );
    }

    return elem;
}

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(SRotation)* NS(SRotation_from_obj_index)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)* obj ) SIXTRL_NOEXCEPT
{
    return ( SIXTRL_BE_ARGPTR_DEC NS(SRotation)*
        )NS(SRotation_const_from_obj_index)( obj );
}

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(SRotation) const*
NS(SRotation_const_from_manged_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const index,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT
{
    return NS(SRotation_const_from_obj_index)(
        NS(ManagedBuffer_get_const_object)( buffer, index, slot_size ) );
}

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(SRotation)*
NS(SRotation_from_manged_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const index,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT
{
    return NS(SRotation_from_obj_index)( NS(ManagedBuffer_get_object)(
        buffer, index, slot_size ) );
}

#if !defined( _GPUCODE )
SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(SRotation) const*
NS(SRotation_const_from_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const index ) SIXTRL_NOEXCEPT
{
    return NS(SRotation_const_from_obj_index)(
        NS(Buffer_get_const_object)( buffer, index ) );
}

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(SRotation)* NS(SRotation_from_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const index ) SIXTRL_NOEXCEPT
{
    return NS(SRotation_from_obj_index)(
        NS(Buffer_get_object)( buffer, index ) );
}
#endif /* Host */

#if !defined(  _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */
#endif /* SIXTRACKLIB_COMMON_BE_SROTATION_BEAM_ELEMENT_SROTATION_H__ */
