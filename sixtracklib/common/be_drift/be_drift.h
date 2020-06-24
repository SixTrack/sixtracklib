#ifndef SIXTRACKLIB_COMMON_BE_DRIFT_BEAM_ELEMENT_DRIFT_H__
#define SIXTRACKLIB_COMMON_BE_DRIFT_BEAM_ELEMENT_DRIFT_H__

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
    #include "sixtracklib/common/internal/compiler_attributes.h"
    #include "sixtracklib/common/buffer/buffer_type.h"
    #include "sixtracklib/common/buffer/buffer_object.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined(  _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

/* ========================================================================= */

typedef SIXTRL_REAL_T NS(drift_real_t);

typedef struct NS(Drift)
{
    NS(drift_real_t) length SIXTRL_ALIGN( 8 );
}
NS(Drift);

/* ------------------------------------------------------------------------- */

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_ARGPTR_DEC NS(Drift)* NS(Drift_preset)(
    SIXTRL_BE_ARGPTR_DEC NS(Drift)* SIXTRL_RESTRICT drift ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN void NS(Drift_clear)(
    SIXTRL_BE_ARGPTR_DEC NS(Drift)* SIXTRL_RESTRICT drift ) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t) NS(Drift_num_dataptrs)(
    SIXTRL_BE_ARGPTR_DEC const NS(Drift) *const
        SIXTRL_RESTRICT drift ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t) NS(Drift_num_slots)(
    SIXTRL_BE_ARGPTR_DEC const NS(Drift) *const SIXTRL_RESTRICT drift,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(object_type_id_t) NS(Drift_type_id)(
    void ) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC SIXTRL_FN NS(drift_real_t) NS(Drift_length)(
    SIXTRL_BE_ARGPTR_DEC const NS(Drift) *const
        SIXTRL_RESTRICT drift ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(Drift_set_length)(
    SIXTRL_BE_ARGPTR_DEC NS(Drift)* SIXTRL_RESTRICT drift,
    NS(drift_real_t) const length ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(Drift_copy)(
    SIXTRL_BE_ARGPTR_DEC NS(Drift)* SIXTRL_RESTRICT destination,
    SIXTRL_BE_ARGPTR_DEC const NS(Drift) *const
        SIXTRL_RESTRICT source ) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_ARGPTR_DEC NS(Drift) const*
NS(Drift_const_from_obj_index)( SIXTRL_BUFFER_OBJ_ARGPTR_DEC const NS(Object)
    *const SIXTRL_RESTRICT index_obj ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_ARGPTR_DEC NS(Drift)*
NS(Drift_from_obj_index)( SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)*
    SIXTRL_RESTRICT index_obj ) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_ARGPTR_DEC NS(Drift) const*
NS(Drift_const_from_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT buffer_begin,
    NS(buffer_size_t) const index,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_ARGPTR_DEC NS(Drift)*
NS(Drift_from_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT buffer_begin,
    NS(buffer_size_t) const index,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT;

#if !defined( _GPUCODE )

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_ARGPTR_DEC NS(Drift) const*
NS(Drift_const_from_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const index ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_ARGPTR_DEC NS(Drift)* NS(Drift_from_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const index ) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_EXTERN SIXTRL_HOST_FN NS(object_type_id_t)
NS(Drift_type_id_ext)( void ) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(Drift_can_be_added)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_objs,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_slots,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)*
        SIXTRL_RESTRICT ptr_requ_dataptrs ) SIXTRL_NOEXCEPT;

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BE_ARGPTR_DEC NS(Drift)*
NS(Drift_new)( SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer );

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BE_ARGPTR_DEC NS(Drift)* NS(Drift_add)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(drift_real_t) const length );

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BE_ARGPTR_DEC NS(Drift)* NS(Drift_add_copy)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_BE_ARGPTR_DEC const NS(Drift) *const SIXTRL_RESTRICT drift );

#endif /* !defined( _GPUCODE ) */

/* ========================================================================= */

typedef struct NS(DriftExact)
{
    NS(drift_real_t) length SIXTRL_ALIGN( 8 );
}
NS(DriftExact);

/* ------------------------------------------------------------------------- */

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_ARGPTR_DEC NS(DriftExact)*
NS(DriftExact_preset)( SIXTRL_BE_ARGPTR_DEC NS(DriftExact)*
    SIXTRL_RESTRICT drift ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN void NS(DriftExact_clear)( SIXTRL_BE_ARGPTR_DEC
    NS(DriftExact)* SIXTRL_RESTRICT drift ) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t) NS(DriftExact_num_dataptrs)(
    SIXTRL_BE_ARGPTR_DEC const NS(DriftExact) *const
        SIXTRL_RESTRICT drift ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t) NS(DriftExact_num_slots)(
    SIXTRL_BE_ARGPTR_DEC const NS(DriftExact) *const SIXTRL_RESTRICT drift,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(object_type_id_t) NS(DriftExact_type_id)(
    void ) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC SIXTRL_FN NS(drift_real_t) NS(DriftExact_length)(
    SIXTRL_BE_ARGPTR_DEC const NS(DriftExact) *const
        SIXTRL_RESTRICT drift ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(DriftExact_set_length)(
    SIXTRL_BE_ARGPTR_DEC NS(DriftExact)* SIXTRL_RESTRICT drift,
    NS(drift_real_t) const length ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(DriftExact_copy)(
    SIXTRL_BE_ARGPTR_DEC NS(DriftExact)* SIXTRL_RESTRICT destination,
    SIXTRL_BE_ARGPTR_DEC const NS(DriftExact) *const
        SIXTRL_RESTRICT source ) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_ARGPTR_DEC NS(DriftExact) const*
NS(DriftExact_const_from_obj_index)( SIXTRL_BUFFER_OBJ_ARGPTR_DEC const NS(Object)
    *const SIXTRL_RESTRICT index_obj ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_ARGPTR_DEC NS(DriftExact)*
NS(DriftExact_from_obj_index)( SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)*
    SIXTRL_RESTRICT index_obj ) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_ARGPTR_DEC NS(DriftExact) const*
NS(DriftExact_const_from_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT buffer_begin,
    NS(buffer_size_t) const index,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_ARGPTR_DEC NS(DriftExact)*
NS(DriftExact_from_managed_buffer)( SIXTRL_BUFFER_DATAPTR_DEC unsigned char*
        SIXTRL_RESTRICT buffer_begin, NS(buffer_size_t) const index,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT;

#if !defined( _GPUCODE )

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_ARGPTR_DEC NS(DriftExact) const*
NS(DriftExact_const_from_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const index ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_ARGPTR_DEC NS(DriftExact)*
NS(DriftExact_from_buffer)( SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)*
        SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const index ) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_EXTERN SIXTRL_HOST_FN NS(object_type_id_t)
NS(DriftExact_type_id_ext)( void ) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(DriftExact_can_be_added)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_objs,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_slots,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)*
        SIXTRL_RESTRICT ptr_requ_dataptrs ) SIXTRL_NOEXCEPT;

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BE_ARGPTR_DEC NS(DriftExact)*
NS(DriftExact_new)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer );

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BE_ARGPTR_DEC NS(DriftExact)*
NS(DriftExact_add)( SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(drift_real_t) const length );

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BE_ARGPTR_DEC NS(DriftExact)*
NS(DriftExact_add_copy)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_BE_ARGPTR_DEC const NS(DriftExact) *const SIXTRL_RESTRICT drift );

#endif /* !defined( _GPUCODE ) */
#if !defined(  _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

/* ************************************************************************* */
/* *****          Implementation of C inline functions                   *** */
/* ************************************************************************* */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/buffer/managed_buffer_minimal.h"
    #if !defined( _GPUCODE )
    #include "sixtracklib/common/buffer.h"
    #endif /* !defined( _GPUCODE ) */
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined(  _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(Drift)* NS(Drift_preset)(
    SIXTRL_BE_ARGPTR_DEC NS(Drift)* SIXTRL_RESTRICT drift ) SIXTRL_NOEXCEPT
{
    if( drift != SIXTRL_NULLPTR ) NS(Drift_clear)( drift );
    return drift;
}

SIXTRL_INLINE void NS(Drift_clear)(
    SIXTRL_BE_ARGPTR_DEC NS(Drift)* SIXTRL_RESTRICT drift ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( drift != SIXTRL_NULLPTR );
    NS(Drift_set_length)( drift, ( NS(drift_real_t) )0 );
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE NS(buffer_size_t) NS(Drift_num_dataptrs)( SIXTRL_BE_ARGPTR_DEC
    const NS(Drift) *const SIXTRL_RESTRICT SIXTRL_UNUSED( drift )
) SIXTRL_NOEXCEPT
{
    return ( NS(buffer_size_t) )0u;
}

SIXTRL_INLINE NS(buffer_size_t) NS(Drift_num_slots)( SIXTRL_BE_ARGPTR_DEC const
        NS(Drift) *const SIXTRL_RESTRICT drift,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT
{
    NS(buffer_size_t) num_slots = ( NS(buffer_size_t) )0u;

    if( ( drift != SIXTRL_NULLPTR ) && ( slot_size > ( NS(buffer_size_t) )0u ) )
    {
        NS(buffer_size_t) const requ_bytes =
            NS(ManagedBuffer_get_slot_based_length)(
                sizeof( NS(Drift) ), slot_size );

        num_slots = requ_bytes / slot_size;
        if( num_slots * slot_size < requ_bytes ) ++num_slots;
    }

    return num_slots;
}

SIXTRL_INLINE NS(object_type_id_t) NS(Drift_type_id)(
    void ) SIXTRL_NOEXCEPT
{
    return ( NS(object_type_id_t) )NS(OBJECT_TYPE_DRIFT);
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE NS(drift_real_t) NS(Drift_length)( SIXTRL_BE_ARGPTR_DEC const
    NS(Drift) *const SIXTRL_RESTRICT drift ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( drift != SIXTRL_NULLPTR );
    return drift->length;
}

SIXTRL_INLINE NS(arch_status_t) NS(Drift_set_length)(
    SIXTRL_BE_ARGPTR_DEC NS(Drift)* SIXTRL_RESTRICT drift,
    NS(drift_real_t) const length ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( drift != SIXTRL_NULLPTR );
    drift->length = length;
    return ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
}

SIXTRL_INLINE NS(arch_status_t) NS(Drift_copy)(
    SIXTRL_BE_ARGPTR_DEC NS(Drift)* SIXTRL_RESTRICT destination,
    SIXTRL_BE_ARGPTR_DEC const NS(Drift) *const SIXTRL_RESTRICT source
) SIXTRL_NOEXCEPT
{
    return NS(Drift_set_length)( destination, NS(Drift_length)( source ) );
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(Drift) const*
NS(Drift_const_from_obj_index)( SIXTRL_BUFFER_OBJ_ARGPTR_DEC const NS(Object)
    *const SIXTRL_RESTRICT index_obj ) SIXTRL_NOEXCEPT
{
    typedef SIXTRL_BE_ARGPTR_DEC NS(Drift) const* ptr_drift_t;
    ptr_drift_t elem = SIXTRL_NULLPTR;

    if( ( index_obj != SIXTRL_NULLPTR ) &&
        ( NS(Object_get_type_id)( index_obj ) == NS(OBJECT_TYPE_DRIFT) ) &&
        ( NS(Object_get_size)( index_obj ) >= sizeof( NS(Drift) ) ) )
    {
        elem = ( ptr_drift_t )( uintptr_t )NS(Object_get_begin_addr)(
            index_obj );
    }

    return elem;
}

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(Drift)* NS(Drift_from_obj_index)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)* SIXTRL_RESTRICT index_obj
) SIXTRL_NOEXCEPT
{
    return ( SIXTRL_BE_ARGPTR_DEC NS(Drift)*
        )NS(Drift_const_from_obj_index)( index_obj );
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(Drift) const*
NS(Drift_const_from_managed_buffer)( SIXTRL_BUFFER_DATAPTR_DEC unsigned char
        const* SIXTRL_RESTRICT buffer_begin, NS(buffer_size_t) const index,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT
{
    return NS(Drift_const_from_obj_index)( NS(ManagedBuffer_get_const_object)(
        buffer_begin, index, slot_size ) );
}

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(Drift)* NS(Drift_from_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT buffer_begin,
    NS(buffer_size_t) const index, NS(buffer_size_t) const slot_size
) SIXTRL_NOEXCEPT
{
    return NS(Drift_from_obj_index)( NS(ManagedBuffer_get_object)(
        buffer_begin, index, slot_size ) );
}

#if !defined( _GPUCODE )

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(Drift) const* NS(Drift_const_from_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const index ) SIXTRL_NOEXCEPT
{
    return NS(Drift_const_from_obj_index)( NS(Buffer_get_const_object)(
        buffer, index ) );
}

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(Drift)* NS(Drift_from_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const index ) SIXTRL_NOEXCEPT
{
    return NS(Drift_from_obj_index)( NS(Buffer_get_object)(
        buffer, index ) );
}

#endif /* !defined( _GPUCODE ) */

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(DriftExact)* NS(DriftExact_preset)(
    SIXTRL_BE_ARGPTR_DEC NS(DriftExact)* SIXTRL_RESTRICT drift ) SIXTRL_NOEXCEPT
{
    if( drift != SIXTRL_NULLPTR ) NS(DriftExact_clear)( drift );
    return drift;
}

SIXTRL_INLINE void NS(DriftExact_clear)(
    SIXTRL_BE_ARGPTR_DEC NS(DriftExact)* SIXTRL_RESTRICT drift ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( drift != SIXTRL_NULLPTR );
    NS(DriftExact_set_length)( drift, ( NS(drift_real_t) )0 );
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE NS(buffer_size_t) NS(DriftExact_num_dataptrs)( SIXTRL_BE_ARGPTR_DEC
    const NS(DriftExact) *const SIXTRL_RESTRICT SIXTRL_UNUSED( drift )
) SIXTRL_NOEXCEPT
{
    return ( NS(buffer_size_t) )0u;
}

SIXTRL_INLINE NS(buffer_size_t) NS(DriftExact_num_slots)( SIXTRL_BE_ARGPTR_DEC
        const NS(DriftExact) *const SIXTRL_RESTRICT drift,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT
{
    NS(buffer_size_t) num_slots = ( NS(buffer_size_t) )0u;

    if( ( drift != SIXTRL_NULLPTR ) && ( slot_size > ( NS(buffer_size_t) )0u ) )
    {
        NS(buffer_size_t) const requ_bytes =
            NS(ManagedBuffer_get_slot_based_length)(
                sizeof( NS(DriftExact) ), slot_size );

        num_slots = requ_bytes / slot_size;
        if( num_slots * slot_size < requ_bytes ) ++num_slots;
    }

    return num_slots;
}

SIXTRL_INLINE NS(object_type_id_t) NS(DriftExact_type_id)(
    void ) SIXTRL_NOEXCEPT
{
    return ( NS(object_type_id_t) )NS(OBJECT_TYPE_DRIFT_EXACT);
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE NS(drift_real_t) NS(DriftExact_length)( SIXTRL_BE_ARGPTR_DEC const
    NS(DriftExact) *const SIXTRL_RESTRICT drift ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( drift != SIXTRL_NULLPTR );
    return drift->length;
}

SIXTRL_INLINE NS(arch_status_t) NS(DriftExact_set_length)(
    SIXTRL_BE_ARGPTR_DEC NS(DriftExact)* SIXTRL_RESTRICT drift,
    NS(drift_real_t) const length ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( drift != SIXTRL_NULLPTR );
    drift->length = length;
    return ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
}

SIXTRL_INLINE int NS(DriftExact_copy)(
    SIXTRL_BE_ARGPTR_DEC NS(DriftExact)* SIXTRL_RESTRICT destination,
    SIXTRL_BE_ARGPTR_DEC const NS(DriftExact) *const SIXTRL_RESTRICT source
) SIXTRL_NOEXCEPT
{
    return NS(DriftExact_set_length)(
        destination, NS(DriftExact_length)( source ) );
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(DriftExact) const*
NS(DriftExact_const_from_obj_index)( SIXTRL_BUFFER_OBJ_ARGPTR_DEC const
    NS(Object) *const SIXTRL_RESTRICT index_obj ) SIXTRL_NOEXCEPT
{
    typedef SIXTRL_BE_ARGPTR_DEC NS(DriftExact) const* ptr_drift_t;
    ptr_drift_t elem = SIXTRL_NULLPTR;

    if( ( index_obj != SIXTRL_NULLPTR ) &&
        ( NS(Object_get_type_id)( index_obj ) ==
          NS(OBJECT_TYPE_DRIFT_EXACT) ) &&
        ( NS(Object_get_size)( index_obj ) >= sizeof( NS(DriftExact) ) ) )
    {
        elem = ( ptr_drift_t )( uintptr_t )NS(Object_get_begin_addr)(
            index_obj );
    }

    return elem;
}

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(DriftExact)* NS(DriftExact_from_obj_index)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)* SIXTRL_RESTRICT index_obj
) SIXTRL_NOEXCEPT
{
    return ( SIXTRL_BE_ARGPTR_DEC NS(DriftExact)*
        )NS(DriftExact_const_from_obj_index)( index_obj );
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(DriftExact) const*
NS(DriftExact_const_from_managed_buffer)( SIXTRL_BUFFER_DATAPTR_DEC unsigned char
        const* SIXTRL_RESTRICT buffer_begin, NS(buffer_size_t) const index,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT
{
    return NS(DriftExact_const_from_obj_index)( NS(ManagedBuffer_get_const_object)(
        buffer_begin, index, slot_size ) );
}

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(DriftExact)*
NS(DriftExact_from_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT buffer_begin,
    NS(buffer_size_t) const index, NS(buffer_size_t) const slot_size
) SIXTRL_NOEXCEPT
{
    return NS(DriftExact_from_obj_index)( NS(ManagedBuffer_get_object)(
        buffer_begin, index, slot_size ) );
}

#if !defined( _GPUCODE )

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(DriftExact) const*
NS(DriftExact_const_from_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const index ) SIXTRL_NOEXCEPT
{
    return NS(DriftExact_const_from_obj_index)( NS(Buffer_get_const_object)(
        buffer, index ) );
}

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(DriftExact)* NS(DriftExact_from_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const index ) SIXTRL_NOEXCEPT
{
    return NS(DriftExact_from_obj_index)( NS(Buffer_get_object)(
        buffer, index ) );
}

#endif /* !defined( _GPUCODE ) */

#if !defined(  _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

#endif /* SIXTRACKLIB_COMMON_BE_DRIFT_BEAM_ELEMENT_DRIFT_H__ */

/* end: sixtracklib/common/be_drift/be_drift.h */
