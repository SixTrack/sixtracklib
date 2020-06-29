#ifndef SIXTRACKLIB_COMMON_BE_CAVITY_BEAM_ELEMENT_CAVITY_H__
#define SIXTRACKLIB_COMMON_BE_CAVITY_BEAM_ELEMENT_CAVITY_H__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stdbool.h>
    #include <stdint.h>
    #include <stdlib.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/control/definitions.h"
    #include "sixtracklib/common/internal/beam_elements_defines.h"
    #include "sixtracklib/common/internal/buffer_main_defines.h"
    #include "sixtracklib/common/internal/compiler_attributes.h"
    #include "sixtracklib/common/internal/objects_type_id.h"
    #include "sixtracklib/common/buffer/buffer_type.h"
    #include "sixtracklib/common/buffer/buffer_object.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined(  _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

/* ========================================================================= */

typedef struct NS(Cavity)
{
    SIXTRL_REAL_T   voltage     SIXTRL_ALIGN( 8 );
    SIXTRL_REAL_T   frequency   SIXTRL_ALIGN( 8 );
    SIXTRL_REAL_T   lag         SIXTRL_ALIGN( 8 );
}
NS(Cavity);

/* ------------------------------------------------------------------------- */

SIXTRL_STATIC SIXTRL_FN NS(object_type_id_t) NS(Cavity_type_id)(
    void ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t) NS(Cavity_num_dataptrs)(
    SIXTRL_BE_ARGPTR_DEC  const NS(Cavity) *const
        SIXTRL_RESTRICT cavity ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t) NS(Cavity_num_slots)(
    SIXTRL_BE_ARGPTR_DEC  const NS(Cavity) *const SIXTRL_RESTRICT cavity,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_ARGPTR_DEC NS(Cavity)* NS(Cavity_preset)(
    SIXTRL_BE_ARGPTR_DEC  NS(Cavity)* SIXTRL_RESTRICT cavity ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_REAL_T NS(Cavity_voltage)(
    SIXTRL_BE_ARGPTR_DEC const NS(Cavity) *const
        SIXTRL_RESTRICT cavity ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_REAL_T NS(Cavity_frequency)(
    SIXTRL_BE_ARGPTR_DEC  const NS(Cavity) *const
        SIXTRL_RESTRICT cavity ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_REAL_T NS(Cavity_lag)(
    SIXTRL_BE_ARGPTR_DEC  const NS(Cavity) *const
        SIXTRL_RESTRICT cavity ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(Cavity_set_voltage)(
    SIXTRL_BE_ARGPTR_DEC NS(Cavity)* SIXTRL_RESTRICT cavity,
    SIXTRL_REAL_T const voltage ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(Cavity_set_frequency)(
    SIXTRL_BE_ARGPTR_DEC  NS(Cavity)* SIXTRL_RESTRICT cavity,
    SIXTRL_REAL_T const frequency ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(Cavity_set_lag)(
    SIXTRL_BE_ARGPTR_DEC  NS(Cavity)* SIXTRL_RESTRICT cavity,
    SIXTRL_REAL_T const lag ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(Cavity_clear)(
    SIXTRL_BE_ARGPTR_DEC NS(Cavity)* SIXTRL_RESTRICT cavity ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(Cavity_copy)(
    SIXTRL_BE_ARGPTR_DEC NS(Cavity)* SIXTRL_RESTRICT destination,
    SIXTRL_BE_ARGPTR_DEC const NS(Cavity) *const
        SIXTRL_RESTRICT source ) SIXTRL_NOEXCEPT;

/* ------------------------------------------------------------------------- */

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_ARGPTR_DEC NS(Cavity) const*
NS(Cavity_const_from_obj_index)( SIXTRL_BUFFER_OBJ_ARGPTR_DEC const
    NS(Object) *const index_obj ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_ARGPTR_DEC NS(Cavity)*
NS(Cavity_from_obj_index)( SIXTRL_BUFFER_OBJ_ARGPTR_DEC
    NS(Object)* index_obj ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_ARGPTR_DEC NS(Cavity) const*
NS(Cavity_const_from_manged_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT pbuffer,
    NS(buffer_size_t) const be_index,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_ARGPTR_DEC NS(Cavity)*
NS(Cavity_from_manged_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT pbuffer,
    NS(buffer_size_t) const be_index,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT;

#if !defined( _GPUCODE )

SIXTRL_STATIC SIXTRL_HOST_FN SIXTRL_BE_ARGPTR_DEC NS(Cavity) const*
NS(Cavity_const_from_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const be_index ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_HOST_FN SIXTRL_BE_ARGPTR_DEC NS(Cavity)*
NS(Cavity_from_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const be_index ) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t) NS(Cavity_attributes_offsets)(
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT offsets_begin,
    NS(buffer_size_t) const max_num_offsets,
    SIXTRL_BE_ARGPTR_DEC const NS(Cavity) *const SIXTRL_RESTRICT data,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT;

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t) NS(Cavity_attributes_sizes)(
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT sizes_begin,
    NS(buffer_size_t) const max_num_sizes,
    SIXTRL_BE_ARGPTR_DEC const NS(Cavity) *const SIXTRL_RESTRICT data,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT;

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t) NS(Cavity_attributes_counts)(
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT counts_begin,
    NS(buffer_size_t) const max_num_counts,
    SIXTRL_BE_ARGPTR_DEC const NS(Cavity) *const SIXTRL_RESTRICT data,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_EXTERN SIXTRL_HOST_FN NS(object_type_id_t) NS(Cavity_type_id_ext)(
    void ) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(Cavity_can_be_added)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_objects,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_slots,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)*
        SIXTRL_RESTRICT requ_dataptrs ) SIXTRL_NOEXCEPT;

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BE_ARGPTR_DEC NS(Cavity)*
NS(Cavity_new)( SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer );

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BE_ARGPTR_DEC NS(Cavity)*
NS(Cavity_add)( SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_REAL_T  const voltage, SIXTRL_REAL_T  const frequency,
    SIXTRL_REAL_T  const lag );

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BE_ARGPTR_DEC NS(Cavity)*
NS(Cavity_add_copy)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_BE_ARGPTR_DEC const NS(Cavity) *const SIXTRL_RESTRICT cavity );

#endif /* !defined( _GPUCODE ) */

#if !defined(  _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

/* ************************************************************************* */
/* *****          Implementation of C inline functions                   *** */
/* ************************************************************************* */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/buffer/managed_buffer.h"
    #if !defined( _GPUCODE )
        #include "sixtracklib/common/buffer.h"
    #endif /* !defined( _GPUCODE ) */
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined(  _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

SIXTRL_INLINE NS(object_type_id_t) NS(Cavity_type_id)( void ) SIXTRL_NOEXCEPT
{
    return ( NS(object_type_id_t) )NS(OBJECT_TYPE_CAVITY);
}

SIXTRL_INLINE NS(buffer_size_t) NS(Cavity_num_dataptrs)( SIXTRL_BE_ARGPTR_DEC
    const NS(Cavity) *const SIXTRL_RESTRICT
        SIXTRL_UNUSED( cavity ) ) SIXTRL_NOEXCEPT
{
    return ( NS(buffer_size_t) )0u;
}

SIXTRL_INLINE NS(buffer_size_t) NS(Cavity_num_slots)( SIXTRL_BE_ARGPTR_DEC const
    NS(Cavity) *const SIXTRL_RESTRICT SIXTRL_UNUSED( cavity ),
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT
{
    NS(buffer_size_t) num_slots = ( NS(buffer_size_t) )0;
    NS(buffer_size_t) const num_bytes = NS(ManagedBuffer_get_slot_based_length)(
        sizeof( NS(Cavity) ), slot_size );

    SIXTRL_ASSERT( slot_size > ( NS(buffer_size_t) )0 );
    num_slots = num_bytes / slot_size;
    if( num_slots * slot_size < num_bytes ) ++num_slots;
    return num_slots;
}

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(Cavity)* NS(Cavity_preset)(
    SIXTRL_BE_ARGPTR_DEC NS(Cavity)* SIXTRL_RESTRICT cavity ) SIXTRL_NOEXCEPT
{
    if( cavity != SIXTRL_NULLPTR ) NS(Cavity_clear)( cavity );
    return cavity;
}

SIXTRL_INLINE NS(arch_status_t) NS(Cavity_clear)(
    SIXTRL_BE_ARGPTR_DEC NS(Cavity)* SIXTRL_RESTRICT cavity ) SIXTRL_NOEXCEPT
{
    NS(arch_status_t) status = NS(Cavity_set_voltage)(
        cavity, ( SIXTRL_REAL_T )0 );

    status |= NS(Cavity_set_frequency)( cavity, ( SIXTRL_REAL_T )0 );
    status |= NS(Cavity_set_lag)( cavity, ( SIXTRL_REAL_T )0 );
    return status;
}

SIXTRL_INLINE SIXTRL_REAL_T NS(Cavity_voltage)( SIXTRL_BE_ARGPTR_DEC const
    NS(Cavity) *const SIXTRL_RESTRICT cavity ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( cavity != SIXTRL_NULLPTR );
    return cavity->voltage;
}

SIXTRL_INLINE SIXTRL_REAL_T NS(Cavity_frequency)( SIXTRL_BE_ARGPTR_DEC const
    NS(Cavity) *const SIXTRL_RESTRICT cavity ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( cavity != SIXTRL_NULLPTR );
    return cavity->frequency;
}

SIXTRL_INLINE SIXTRL_REAL_T NS(Cavity_lag)( SIXTRL_BE_ARGPTR_DEC const
    NS(Cavity) *const SIXTRL_RESTRICT cavity ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( cavity != SIXTRL_NULLPTR );
    return cavity->lag;
}

SIXTRL_INLINE NS(arch_status_t) NS(Cavity_set_voltage)(
    SIXTRL_BE_ARGPTR_DEC NS(Cavity)* SIXTRL_RESTRICT cavity,
    SIXTRL_REAL_T const voltage ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( cavity != SIXTRL_NULLPTR );
    cavity->voltage = voltage;
    return ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
}

SIXTRL_INLINE NS(arch_status_t) NS(Cavity_set_frequency)(
    SIXTRL_BE_ARGPTR_DEC NS(Cavity)* SIXTRL_RESTRICT cavity,
    SIXTRL_REAL_T const frequency ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( cavity != SIXTRL_NULLPTR );
    cavity->frequency = frequency;
    return ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
}

SIXTRL_INLINE NS(arch_status_t) NS(Cavity_set_lag)(
    SIXTRL_BE_ARGPTR_DEC NS(Cavity)* SIXTRL_RESTRICT cavity,
    SIXTRL_REAL_T const lag ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( cavity != SIXTRL_NULLPTR );
    cavity->lag = lag;
    return ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
}

SIXTRL_INLINE NS(arch_status_t) NS(Cavity_copy)(
    SIXTRL_BE_ARGPTR_DEC NS(Cavity)* SIXTRL_RESTRICT destination,
    SIXTRL_BE_ARGPTR_DEC const NS(Cavity) *const
        SIXTRL_RESTRICT source ) SIXTRL_NOEXCEPT
{
    NS(arch_status_t) status = ( NS(arch_status_t)
        )SIXTRL_ARCH_STATUS_GENERAL_FAILURE;

    if( ( destination != SIXTRL_NULLPTR ) && ( source != SIXTRL_NULLPTR ) )
    {
        status = ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;

        if( destination != source )
        {
            status |= NS(Cavity_set_voltage)( destination,
                NS(Cavity_voltage)( source ) );

            status |= NS(Cavity_set_frequency)( destination,
                NS(Cavity_frequency)( source ) );

            status |= NS(Cavity_set_lag)( destination,
                NS(Cavity_lag)( source ) );
        }
    }

    return status;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(Cavity) const*
NS(Cavity_const_from_obj_index)( SIXTRL_BUFFER_OBJ_ARGPTR_DEC const NS(Object)
    *const SIXTRL_RESTRICT obj ) SIXTRL_NOEXCEPT
{
    SIXTRL_BE_ARGPTR_DEC NS(Cavity) const* elem = SIXTRL_NULLPTR;
    if( ( obj != SIXTRL_NULLPTR ) &&
        ( NS(Object_get_type_id)( obj ) == NS(OBJECT_TYPE_CAVITY) ) &&
        ( NS(Object_get_size)( obj ) >= sizeof( NS(Cavity) ) ) )
    {
        elem = ( SIXTRL_BE_ARGPTR_DEC NS(Cavity) const* )(
            uintptr_t )NS(Object_get_begin_addr)( obj );
    }

    return elem;
}

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(Cavity)* NS(Cavity_from_obj_index)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)* obj ) SIXTRL_NOEXCEPT
{
    return ( SIXTRL_BUFFER_DATAPTR_DEC NS(Cavity)*
        )NS(Cavity_const_from_obj_index)( obj );
}

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(Cavity) const*
NS(Cavity_const_from_manged_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT pbuffer,
    NS(buffer_size_t) const be_index,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT
{
    return NS(Cavity_const_from_obj_index)(
        NS(ManagedBuffer_get_const_object)( pbuffer, be_index, slot_size ) );
}

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(Cavity)* NS(Cavity_from_manged_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT pbuffer,
    NS(buffer_size_t) const be_index,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT
{
    return NS(Cavity_from_obj_index)(
        NS(ManagedBuffer_get_object)( pbuffer, be_index, slot_size ) );
}

#if !defined( _GPUCODE )

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(Cavity) const*
NS(Cavity_const_from_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const be_index ) SIXTRL_NOEXCEPT
{
    return NS(Cavity_const_from_obj_index)( NS(Buffer_get_const_object)(
        buffer, be_index ) );
}

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(Cavity)* NS(Cavity_from_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const be_index ) SIXTRL_NOEXCEPT
{
    return NS(Cavity_from_obj_index)(
        NS(Buffer_get_object)( buffer, be_index ) );
}

#endif /* !defined( _GPUCODE ) */

#if !defined(  _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */
#endif /* SIXTRACKLIB_COMMON_BE_CAVITY_BEAM_ELEMENT_CAVITY_H__ */
