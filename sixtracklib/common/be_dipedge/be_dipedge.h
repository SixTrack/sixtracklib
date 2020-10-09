#ifndef SIXTRL_COMMON_BE_DIPEDGE_BE_DIPEDGE_C99_H__
#define SIXTRL_COMMON_BE_DIPEDGE_BE_DIPEDGE_C99_H__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stdbool.h>
    #include <stdint.h>
    #include <stdlib.h>
    #include <math.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/control/definitions.h"
    #include "sixtracklib/common/constants.h"
    #include "sixtracklib/common/internal/buffer_main_defines.h"
    #include "sixtracklib/common/internal/beam_elements_defines.h"
    #include "sixtracklib/common/internal/objects_type_id.h"
    #include "sixtracklib/common/control/definitions.h"
    #include "sixtracklib/common/buffer/buffer_type.h"
    #include "sixtracklib/common/buffer/buffer_object.h"
#endif /* !defined( SIXTGRL_NO_INCLUDES ) */

#if !defined(  _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

typedef SIXTRL_REAL_T NS(dipedge_real_t);

typedef struct NS(DipoleEdge)
{
    NS(dipedge_real_t) r21 SIXTRL_ALIGN( 8 );
    NS(dipedge_real_t) r43 SIXTRL_ALIGN( 8 );
}
NS(DipoleEdge);


/* ------------------------------------------------------------------------- */

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_ARGPTR_DEC NS(DipoleEdge)*
NS(DipoleEdge_preset)( SIXTRL_BE_ARGPTR_DEC NS(DipoleEdge)*
    SIXTRL_RESTRICT dipedge ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(DipoleEdge_clear)(
    SIXTRL_BE_ARGPTR_DEC NS(DipoleEdge)* SIXTRL_RESTRICT
        dipedge ) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t) NS(DipoleEdge_num_dataptrs)(
    SIXTRL_BE_ARGPTR_DEC const NS(DipoleEdge) *const
        SIXTRL_RESTRICT dipedge ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t) NS(DipoleEdge_num_slots)(
    SIXTRL_BE_ARGPTR_DEC const NS(DipoleEdge) *const SIXTRL_RESTRICT dipedge,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(object_type_id_t) NS(DipoleEdge_type_id)(
    void ) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC SIXTRL_FN NS(dipedge_real_t) NS(DipoleEdge_r21)(
    SIXTRL_BE_ARGPTR_DEC const NS(DipoleEdge) *const SIXTRL_RESTRICT
        dipedge ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(dipedge_real_t) NS(DipoleEdge_r43)(
    SIXTRL_BE_ARGPTR_DEC const NS(DipoleEdge) *const SIXTRL_RESTRICT
        dipedge ) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(DipoleEdge_set_r21)(
    SIXTRL_BE_ARGPTR_DEC NS(DipoleEdge)* SIXTRL_RESTRICT dipedge,
    NS(dipedge_real_t) const r21 ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(DipoleEdge_set_r43)(
    SIXTRL_BE_ARGPTR_DEC NS(DipoleEdge)* SIXTRL_RESTRICT dipedge,
    NS(dipedge_real_t) const r43 ) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(DipoleEdge_copy)(
    SIXTRL_BE_ARGPTR_DEC NS(DipoleEdge)* SIXTRL_RESTRICT destination,
    SIXTRL_BE_ARGPTR_DEC const NS(DipoleEdge) *const
        SIXTRL_RESTRICT source ) SIXTRL_NOEXCEPT;


SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_ARGPTR_DEC NS(DipoleEdge) const*
NS(DipoleEdge_const_from_obj_index)( SIXTRL_BUFFER_OBJ_ARGPTR_DEC const NS(Object)
    *const SIXTRL_RESTRICT index_obj ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_ARGPTR_DEC NS(DipoleEdge)*
NS(DipoleEdge_from_obj_index)( SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)*
    SIXTRL_RESTRICT index_obj ) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_ARGPTR_DEC NS(DipoleEdge) const*
NS(DipoleEdge_const_from_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT buffer_begin,
    NS(buffer_size_t) const index,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_ARGPTR_DEC NS(DipoleEdge)*
NS(DipoleEdge_from_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT buffer_begin,
    NS(buffer_size_t) const index,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT;

#if !defined( _GPUCODE )

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_ARGPTR_DEC NS(DipoleEdge) const*
NS(DipoleEdge_const_from_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const index ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_ARGPTR_DEC NS(DipoleEdge)*
NS(DipoleEdge_from_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const index ) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t)
NS(DipoleEdge_attributes_offsets)(
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT offsets_begin,
    NS(buffer_size_t) const max_num_offsets,
    SIXTRL_BE_ARGPTR_DEC const NS(DipoleEdge) *const SIXTRL_RESTRICT data,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT;

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t)
NS(DipoleEdge_attributes_sizes)(
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT sizes_begin,
    NS(buffer_size_t) const max_num_sizes,
    SIXTRL_BE_ARGPTR_DEC const NS(DipoleEdge) *const SIXTRL_RESTRICT data,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT;

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t)
NS(DipoleEdge_attributes_counts)(
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT counts_begin,
    NS(buffer_size_t) const max_num_counts,
    SIXTRL_BE_ARGPTR_DEC const NS(DipoleEdge) *const SIXTRL_RESTRICT data,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_EXTERN SIXTRL_HOST_FN NS(object_type_id_t)
NS(DipoleEdge_type_id_ext)( void ) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(DipoleEdge_can_be_added)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_objs,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_slots,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)*
        SIXTRL_RESTRICT ptr_requ_dataptrs ) SIXTRL_NOEXCEPT;

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BE_ARGPTR_DEC NS(DipoleEdge)*
NS(DipoleEdge_new)( SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)*
    SIXTRL_RESTRICT buffer );

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BE_ARGPTR_DEC NS(DipoleEdge)*
NS(DipoleEdge_add)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(dipedge_real_t) const r21, NS(dipedge_real_t) const r43 );

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BE_ARGPTR_DEC NS(DipoleEdge)*
NS(DipoleEdge_add_copy)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_BE_ARGPTR_DEC const NS(DipoleEdge) *const SIXTRL_RESTRICT dipedge );

#endif /* !defined( _GPUCODE ) */

#if !defined(  _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

/* ========================================================================= */
/*        Implementation of inline functions for NS(DipoleEdge)                   */
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

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(DipoleEdge)* NS(DipoleEdge_preset)(
    SIXTRL_BE_ARGPTR_DEC NS(DipoleEdge)*
        SIXTRL_RESTRICT dipedge ) SIXTRL_NOEXCEPT
{
    if( dipedge != SIXTRL_NULLPTR ) NS(DipoleEdge_clear)( dipedge );
    return dipedge;
}

SIXTRL_INLINE NS(arch_status_t) NS(DipoleEdge_clear)(
    SIXTRL_BE_ARGPTR_DEC NS(DipoleEdge)*
        SIXTRL_RESTRICT dipedge ) SIXTRL_NOEXCEPT
{
    NS(arch_status_t) status = NS(DipoleEdge_set_r21)(
        dipedge, ( NS(dipedge_real_t) )0 );
    status |= NS(DipoleEdge_set_r43)( dipedge, ( NS(dipedge_real_t) )1 );
    return status;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE NS(buffer_size_t) NS(DipoleEdge_num_dataptrs)(
    SIXTRL_BE_ARGPTR_DEC const NS(DipoleEdge) *const
        SIXTRL_RESTRICT SIXTRL_UNUSED( dipedge ) ) SIXTRL_NOEXCEPT
{
    return ( NS(buffer_size_t) )0u;
}

SIXTRL_INLINE NS(buffer_size_t) NS(DipoleEdge_num_slots)(
    SIXTRL_BE_ARGPTR_DEC const NS(DipoleEdge) *const SIXTRL_RESTRICT dipedge,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT
{
    NS(buffer_size_t) num_slots = ( NS(buffer_size_t) )0u;
    NS(buffer_size_t) const num_bytes =
        NS(ManagedBuffer_get_slot_based_length)(
            sizeof( NS(DipoleEdge) ), slot_size );

    SIXTRL_ASSERT( slot_size > ( NS(buffer_size_t) )0 );
    num_slots = num_bytes / slot_size;
    if( num_slots * slot_size < num_bytes ) ++num_slots;
    return num_slots;
}

SIXTRL_INLINE NS(object_type_id_t) NS(DipoleEdge_type_id)(
    void ) SIXTRL_NOEXCEPT
{
    return ( NS(object_type_id_t) )NS(OBJECT_TYPE_DIPEDGE);
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE NS(dipedge_real_t) NS(DipoleEdge_r21)( SIXTRL_BE_ARGPTR_DEC const
    NS(DipoleEdge) *const SIXTRL_RESTRICT dipedge ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( dipedge != SIXTRL_NULLPTR );
    return dipedge->r21;
}

SIXTRL_INLINE NS(dipedge_real_t) NS(DipoleEdge_r43)( SIXTRL_BE_ARGPTR_DEC const
    NS(DipoleEdge) *const SIXTRL_RESTRICT dipedge ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( dipedge != SIXTRL_NULLPTR );
    return dipedge->r43;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE NS(arch_status_t) NS(DipoleEdge_set_r21)(
    SIXTRL_BE_ARGPTR_DEC NS(DipoleEdge)* SIXTRL_RESTRICT dipedge,
    NS(dipedge_real_t) const r21 ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( dipedge != SIXTRL_NULLPTR );
    dipedge->r21 = r21;
    return ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
}

SIXTRL_INLINE NS(arch_status_t) NS(DipoleEdge_set_r43)(
    SIXTRL_BE_ARGPTR_DEC NS(DipoleEdge)* SIXTRL_RESTRICT dipedge,
    NS(dipedge_real_t) const r43 ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( dipedge != SIXTRL_NULLPTR );
    dipedge->r43 = r43;
    return ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE NS(arch_status_t) NS(DipoleEdge_copy)(
    SIXTRL_BE_ARGPTR_DEC NS(DipoleEdge)* SIXTRL_RESTRICT dst,
    SIXTRL_BE_ARGPTR_DEC const NS(DipoleEdge) *const SIXTRL_RESTRICT src
) SIXTRL_NOEXCEPT
{
    NS(arch_status_t) status = ( NS(arch_status_t)
        )SIXTRL_ARCH_STATUS_GENERAL_FAILURE;

    if( ( dst != SIXTRL_NULLPTR ) && ( src != SIXTRL_NULLPTR ) )
    {
        status = ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;

        if( dst != src )
        {
            status  = NS(DipoleEdge_set_r21)( dst, NS(DipoleEdge_r21)( src ) );
            status |= NS(DipoleEdge_set_r43)( dst, NS(DipoleEdge_r43)( src ) );
        }
    }

    return status;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(DipoleEdge) const*
NS(DipoleEdge_const_from_obj_index)( SIXTRL_BUFFER_OBJ_ARGPTR_DEC const
    NS(Object) *const SIXTRL_RESTRICT index_obj ) SIXTRL_NOEXCEPT
{
    typedef SIXTRL_BE_ARGPTR_DEC NS(DipoleEdge) const* ptr_dipedge_t;
    ptr_dipedge_t elem = SIXTRL_NULLPTR;

    if( ( index_obj != SIXTRL_NULLPTR ) &&
        ( NS(Object_get_type_id)( index_obj ) == NS(OBJECT_TYPE_DIPEDGE) ) &&
        ( NS(Object_get_size)( index_obj ) >= sizeof( NS(DipoleEdge) ) ) )
    {
        elem = ( ptr_dipedge_t )( uintptr_t )NS(Object_get_begin_addr)(
            index_obj );
    }

    return elem;
}

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(DipoleEdge)* NS(DipoleEdge_from_obj_index)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)* SIXTRL_RESTRICT index_obj
) SIXTRL_NOEXCEPT
{
    return ( SIXTRL_BE_ARGPTR_DEC NS(DipoleEdge)*
        )NS(DipoleEdge_const_from_obj_index)( index_obj );
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(DipoleEdge) const*
NS(DipoleEdge_const_from_managed_buffer)( SIXTRL_BUFFER_DATAPTR_DEC unsigned char
        const* SIXTRL_RESTRICT buffer_begin, NS(buffer_size_t) const index,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT
{
    return NS(DipoleEdge_const_from_obj_index)(
        NS(ManagedBuffer_get_const_object)( buffer_begin, index, slot_size ) );
}

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(DipoleEdge)* NS(DipoleEdge_from_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT buffer_begin,
    NS(buffer_size_t) const index,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT
{
    return NS(DipoleEdge_from_obj_index)( NS(ManagedBuffer_get_object)(
        buffer_begin, index, slot_size ) );
}

#if !defined( _GPUCODE )

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(DipoleEdge) const*
NS(DipoleEdge_const_from_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const index ) SIXTRL_NOEXCEPT
{
    return NS(DipoleEdge_const_from_obj_index)(
        NS(Buffer_get_const_object)( buffer, index ) );
}

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(DipoleEdge)*
NS(DipoleEdge_from_buffer)( SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)*
    SIXTRL_RESTRICT buffer, NS(buffer_size_t) const index ) SIXTRL_NOEXCEPT
{
    return NS(DipoleEdge_from_obj_index)( NS(Buffer_get_object)(
        buffer, index ) );
}

#endif /* !defined( _GPUCODE ) */

#if !defined(  _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */
#endif /* SIXTRL_COMMON_BE_DIPEDGE_BE_DIPEDGE_C99_H__ */
