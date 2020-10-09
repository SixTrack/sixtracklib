#ifndef SIXTRACKLIB_COMMON_BE_MULTIPOLE_BEAM_ELEMENT_MULTIPOLE_H__
#define SIXTRACKLIB_COMMON_BE_MULTIPOLE_BEAM_ELEMENT_MULTIPOLE_H__

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

typedef SIXTRL_INT64_T NS(multipole_order_t);
typedef SIXTRL_REAL_T  NS(multipole_real_t);

typedef struct NS(Multipole)
{
    NS(multipole_order_t) order     SIXTRL_ALIGN( 8 );
    NS(multipole_real_t)  length    SIXTRL_ALIGN( 8 );
    NS(multipole_real_t)  hxl       SIXTRL_ALIGN( 8 );
    NS(multipole_real_t)  hyl       SIXTRL_ALIGN( 8 );
    NS(buffer_addr_t)     bal_addr  SIXTRL_ALIGN( 8 );
}
NS(Multipole);


SIXTRL_STATIC SIXTRL_FN NS(object_type_id_t) NS(Multipole_type_id)(
    void ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t) NS(Multipole_num_dataptrs)(
    SIXTRL_BE_ARGPTR_DEC const NS(Multipole) *const
        SIXTRL_RESTRICT multipole ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t) NS(Multipole_num_slots)(
    SIXTRL_BE_ARGPTR_DEC const NS(Multipole) *const SIXTRL_RESTRICT multipole,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_ARGPTR_DEC NS(Multipole)*
NS(Multipole_preset)( SIXTRL_BE_ARGPTR_DEC NS(Multipole)*
    SIXTRL_RESTRICT multipole ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(Multipole_clear)(
    SIXTRL_BE_ARGPTR_DEC NS(Multipole)*
        SIXTRL_RESTRICT multipole ) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC SIXTRL_FN NS(multipole_real_t) NS(Multipole_length)(
    SIXTRL_BE_ARGPTR_DEC const NS(Multipole) *const
        SIXTRL_RESTRICT multipole ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(multipole_real_t) NS(Multipole_hxl)(
    SIXTRL_BE_ARGPTR_DEC const NS(Multipole) *const
        SIXTRL_RESTRICT multipole ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(multipole_real_t) NS(Multipole_hyl)(
    SIXTRL_BE_ARGPTR_DEC const NS(Multipole) *const
        SIXTRL_RESTRICT multipole ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(multipole_order_t) NS(Multipole_order)(
    SIXTRL_BE_ARGPTR_DEC const NS(Multipole) *const
        SIXTRL_RESTRICT multipole ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t) NS(Multipole_bal_length)(
    SIXTRL_BE_ARGPTR_DEC const NS(Multipole) *const
        SIXTRL_RESTRICT multipole ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(buffer_addr_t) NS(Multipole_bal_addr)(
    SIXTRL_BE_ARGPTR_DEC const NS(Multipole) *const
        SIXTRL_RESTRICT multipole ) SIXTRL_NOEXCEPT;

/* . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . */

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_DATAPTR_DEC NS(multipole_real_t) const*
NS(Multipole_const_bal_begin)( SIXTRL_BE_ARGPTR_DEC const NS(Multipole) *const
    SIXTRL_RESTRICT multipole ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_DATAPTR_DEC NS(multipole_real_t) const*
NS(Multipole_const_bal_end)( SIXTRL_BE_ARGPTR_DEC const NS(Multipole) *const
    SIXTRL_RESTRICT multipole ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(multipole_real_t) NS(Multipole_bal)(
    SIXTRL_BE_ARGPTR_DEC const NS(Multipole) *const SIXTRL_RESTRICT multipole,
    NS(buffer_size_t) const bal_index ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(multipole_real_t) NS(Multipole_knl)(
    SIXTRL_BE_ARGPTR_DEC const NS(Multipole) *const SIXTRL_RESTRICT multipole,
    NS(multipole_order_t) const knl_index ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(multipole_real_t) NS(Multipole_ksl)(
    SIXTRL_BE_ARGPTR_DEC const NS(Multipole) *const SIXTRL_RESTRICT multipole,
    NS(multipole_order_t) const ksl_index ) SIXTRL_NOEXCEPT;

/* . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . */

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_DATAPTR_DEC NS(multipole_real_t)*
NS(Multipole_bal_begin)( SIXTRL_BE_ARGPTR_DEC NS(Multipole)*
    SIXTRL_RESTRICT multipole ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_DATAPTR_DEC NS(multipole_real_t)*
NS(Multipole_bal_end)( SIXTRL_BE_ARGPTR_DEC NS(Multipole)*
    SIXTRL_RESTRICT multipole ) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(Multipole_set_length)(
    SIXTRL_BE_ARGPTR_DEC NS(Multipole)* SIXTRL_RESTRICT multipole,
    NS(multipole_real_t) const length ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(Multipole_set_hxl)(
    SIXTRL_BE_ARGPTR_DEC NS(Multipole)* SIXTRL_RESTRICT multipole,
    NS(multipole_real_t) const length ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(Multipole_set_hyl)(
    SIXTRL_BE_ARGPTR_DEC NS(Multipole)* SIXTRL_RESTRICT multipole,
    NS(multipole_real_t) const length ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(Multipole_set_order)(
    SIXTRL_BE_ARGPTR_DEC NS(Multipole)* SIXTRL_RESTRICT multipole,
    NS(multipole_order_t) const order ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(Multipole_set_bal_addr)(
    SIXTRL_BE_ARGPTR_DEC NS(Multipole)* SIXTRL_RESTRICT multipole,
    NS(buffer_addr_t) const bal_addr ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(Multipole_set_bal_value)(
    SIXTRL_BE_ARGPTR_DEC NS(Multipole)* SIXTRL_RESTRICT multipole,
    NS(buffer_size_t) const bal_index,
    NS(multipole_real_t) const bal_value ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(Multipole_set_all_bal_values)(
    SIXTRL_BE_ARGPTR_DEC NS(Multipole)* SIXTRL_RESTRICT multipole,
    NS(multipole_real_t) const bal_value ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(Multipole_set_bal)(
    SIXTRL_BE_ARGPTR_DEC NS(Multipole)* SIXTRL_RESTRICT multipole,
    SIXTRL_BE_DATAPTR_DEC NS(multipole_real_t) const*
        SIXTRL_RESTRICT bal_values_begin ) SIXTRL_NOEXCEPT;

/* . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . */

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(Multipole_set_knl_value)(
    SIXTRL_BE_ARGPTR_DEC NS(Multipole)* SIXTRL_RESTRICT multipole,
    NS(buffer_size_t) const knl_index,
    NS(multipole_real_t) const knl_value ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(Multipole_set_all_knl_values)(
    SIXTRL_BE_ARGPTR_DEC NS(Multipole)* SIXTRL_RESTRICT multipole,
    NS(multipole_real_t) const knl_value ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(Multipole_set_knl)(
    SIXTRL_BE_DATAPTR_DEC NS(Multipole)* SIXTRL_RESTRICT multipole,
    SIXTRL_BE_DATAPTR_DEC NS(multipole_real_t) const*
        SIXTRL_RESTRICT knl ) SIXTRL_NOEXCEPT;

/* . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . */

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(Multipole_set_ksl_value)(
    SIXTRL_BE_ARGPTR_DEC NS(Multipole)* SIXTRL_RESTRICT multipole,
    NS(buffer_size_t) const ksl_index,
    NS(multipole_real_t) const ksl_value ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(Multipole_set_all_ksl_values)(
    SIXTRL_BE_ARGPTR_DEC NS(Multipole)* SIXTRL_RESTRICT multipole,
    NS(multipole_real_t) const ksl_value ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(Multipole_set_ksl)(
    SIXTRL_BE_DATAPTR_DEC NS(Multipole)* SIXTRL_RESTRICT multipole,
    SIXTRL_BE_DATAPTR_DEC NS(multipole_real_t) const*
        SIXTRL_RESTRICT ksl ) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(Multipole_copy)(
    SIXTRL_BE_ARGPTR_DEC NS(Multipole)* SIXTRL_RESTRICT destination,
    SIXTRL_BE_ARGPTR_DEC const NS(Multipole) *const
        SIXTRL_RESTRICT src ) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_ARGPTR_DEC NS(Multipole) const*
NS(Multipole_const_from_obj_index)( SIXTRL_BUFFER_OBJ_ARGPTR_DEC const NS(Object)
    *const SIXTRL_RESTRICT index_obj ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_ARGPTR_DEC NS(Multipole)*
NS(Multipole_from_obj_index)( SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)*
    SIXTRL_RESTRICT index_obj ) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_ARGPTR_DEC NS(Multipole) const*
NS(Multipole_const_from_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT buffer_begin,
    NS(buffer_size_t) const index,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_ARGPTR_DEC NS(Multipole)*
NS(Multipole_from_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT buffer_begin,
    NS(buffer_size_t) const index,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT;

#if !defined( _GPUCODE )

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_ARGPTR_DEC NS(Multipole) const*
NS(Multipole_const_from_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const index ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_ARGPTR_DEC NS(Multipole)*
NS(Multipole_from_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const index ) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t)
NS(Multipole_attributes_offsets)(
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT offsets_begin,
    NS(buffer_size_t) const max_num_offsets,
    SIXTRL_BE_ARGPTR_DEC const NS(Multipole) *const SIXTRL_RESTRICT data,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT;

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t)
NS(Multipole_attributes_sizes)(
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT sizes_begin,
    NS(buffer_size_t) const max_num_sizes,
    SIXTRL_BE_ARGPTR_DEC const NS(Multipole) *const SIXTRL_RESTRICT data,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT;

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t)
NS(Multipole_attributes_counts)(
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT counts_begin,
    NS(buffer_size_t) const max_num_counts,
    SIXTRL_BE_ARGPTR_DEC const NS(Multipole) *const SIXTRL_RESTRICT data,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_EXTERN SIXTRL_HOST_FN NS(object_type_id_t)
NS(Multipole_type_id_ext)( void ) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(Multipole_can_be_added)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(multipole_order_t) const order,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_objs,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_slots,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)*
        SIXTRL_RESTRICT ptr_requ_dataptrs ) SIXTRL_NOEXCEPT;

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BE_ARGPTR_DEC NS(Multipole)*
NS(Multipole_new)( SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(multipole_order_t) const order );

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BE_ARGPTR_DEC NS(Multipole)*
NS(Multipole_add)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(multipole_order_t) const order,
    NS(multipole_real_t) const length,
    NS(multipole_real_t) const hxl,
    NS(multipole_real_t) const hyl,
    NS(buffer_addr_t) const bal_addr );

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BE_ARGPTR_DEC NS(Multipole)*
NS(Multipole_add_copy)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_BE_ARGPTR_DEC const NS(Multipole) *const SIXTRL_RESTRICT multipole );

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
    #include "sixtracklib/common/internal/math_factorial.h"
    #if !defined( _GPUCODE )
    #include "sixtracklib/common/buffer.h"
    #endif /* !defined( _GPUCODE ) */
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined(  _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

SIXTRL_INLINE NS(object_type_id_t) NS(Multipole_type_id)( void ) SIXTRL_NOEXCEPT
{
    return ( NS(object_type_id_t) )NS(OBJECT_TYPE_MULTIPOLE);
}

SIXTRL_INLINE NS(buffer_size_t) NS(Multipole_num_dataptrs)( SIXTRL_BE_ARGPTR_DEC
    const NS(Multipole) *const SIXTRL_RESTRICT
        SIXTRL_UNUSED( multipole ) ) SIXTRL_NOEXCEPT
{
    return ( NS(buffer_size_t) )1u;
}

SIXTRL_INLINE NS(buffer_size_t) NS(Multipole_num_slots)(
    SIXTRL_BE_ARGPTR_DEC const NS(Multipole) *const SIXTRL_RESTRICT multipole,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT
{
    NS(buffer_size_t) num_slots = ( NS(buffer_size_t) )0u;

    if( ( multipole != SIXTRL_NULLPTR ) &&
        ( slot_size > ( NS(buffer_size_t) )0 ) )
    {
        NS(buffer_size_t) num_bytes = NS(ManagedBuffer_get_slot_based_length)(
            sizeof( NS(Multipole) ), slot_size );

        NS(buffer_size_t) const bal_length = NS(Multipole_bal_length)( multipole );

        SIXTRL_ASSERT( bal_length >= ( NS(buffer_size_t) )0 );
        num_bytes += NS(ManagedBuffer_get_slot_based_length)(
            bal_length * sizeof( NS(multipole_real_t) ), slot_size );

        num_slots = num_bytes / slot_size;
        if( num_slots * slot_size < num_bytes ) ++num_slots;
    }

    return num_slots;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(Multipole)* NS(Multipole_preset)(
    SIXTRL_BE_ARGPTR_DEC NS(Multipole)*
        SIXTRL_RESTRICT multipole ) SIXTRL_NOEXCEPT
{
    if( multipole != SIXTRL_NULLPTR ) NS(Multipole_clear)( multipole );
    return multipole;
}

SIXTRL_INLINE NS(arch_status_t) NS(Multipole_clear)( SIXTRL_BE_ARGPTR_DEC
    NS(Multipole)* SIXTRL_RESTRICT multipole ) SIXTRL_NOEXCEPT
{
    NS(arch_status_t) status = NS(Multipole_set_length)(
        multipole, ( NS(multipole_real_t) )0 );

    status |= NS(Multipole_set_hxl)( multipole, ( NS(multipole_real_t) )0 );
    status |= NS(Multipole_set_hyl)( multipole, ( NS(multipole_real_t) )0 );
    status |= NS(Multipole_set_order)( multipole, ( NS(multipole_order_t) )0 );
    status |= NS(Multipole_set_bal_addr)( multipole, ( NS(buffer_addr_t) )0 );

    return status;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE NS(multipole_real_t) NS(Multipole_length)( SIXTRL_BE_ARGPTR_DEC
    const NS(Multipole) *const SIXTRL_RESTRICT multipole ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( multipole != SIXTRL_NULLPTR );
    return multipole->length;
}

SIXTRL_INLINE NS(multipole_real_t) NS(Multipole_hxl)( SIXTRL_BE_ARGPTR_DEC const
    NS(Multipole) *const SIXTRL_RESTRICT multipole ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( multipole != SIXTRL_NULLPTR );
    return multipole->hxl;
}

SIXTRL_INLINE NS(multipole_real_t) NS(Multipole_hyl)( SIXTRL_BE_ARGPTR_DEC const
    NS(Multipole) *const SIXTRL_RESTRICT multipole ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( multipole != SIXTRL_NULLPTR );
    return multipole->hyl;
}

SIXTRL_INLINE NS(multipole_order_t) NS(Multipole_order)( SIXTRL_BE_ARGPTR_DEC
    const NS(Multipole) *const SIXTRL_RESTRICT multipole ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( multipole != SIXTRL_NULLPTR );
    return multipole->order;
}

SIXTRL_INLINE NS(buffer_size_t) NS(Multipole_bal_length)( SIXTRL_BE_ARGPTR_DEC
    const NS(Multipole) *const SIXTRL_RESTRICT multipole ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( multipole != SIXTRL_NULLPTR );
    return ( multipole->order >= ( NS(multipole_order_t) )0 )
        ? ( NS(buffer_size_t) )( 2 * multipole->order + 2 )
        : ( NS(buffer_size_t) )0;
}

SIXTRL_INLINE NS(buffer_addr_t) NS(Multipole_bal_addr)( SIXTRL_BE_ARGPTR_DEC
    const NS(Multipole) *const SIXTRL_RESTRICT multipole ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( multipole != SIXTRL_NULLPTR );
    return multipole->bal_addr;
}

/* . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . */

SIXTRL_INLINE SIXTRL_BE_DATAPTR_DEC NS(multipole_real_t) const*
NS(Multipole_const_bal_begin)( SIXTRL_BE_ARGPTR_DEC const NS(Multipole) *const
    SIXTRL_RESTRICT multipole ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( multipole != SIXTRL_NULLPTR );
    return ( SIXTRL_BE_DATAPTR_DEC NS(multipole_real_t) const* )(
                uintptr_t )multipole->bal_addr;
}

SIXTRL_INLINE SIXTRL_BE_DATAPTR_DEC NS(multipole_real_t) const*
NS(Multipole_const_bal_end)( SIXTRL_BE_ARGPTR_DEC const NS(Multipole) *const
    SIXTRL_RESTRICT multipole ) SIXTRL_NOEXCEPT
{
    SIXTRL_BE_DATAPTR_DEC NS(multipole_real_t) const* bal_end =
        NS(Multipole_const_bal_begin)( multipole );

    if( bal_end != SIXTRL_NULLPTR )
        bal_end = bal_end + NS(Multipole_bal_length)( multipole );

    return bal_end;
}

SIXTRL_INLINE NS(multipole_real_t) NS(Multipole_bal)(
    SIXTRL_BE_ARGPTR_DEC const NS(Multipole) *const SIXTRL_RESTRICT multipole,
    NS(buffer_size_t) const bal_index ) SIXTRL_NOEXCEPT
{
    SIXTRL_BE_DATAPTR_DEC NS(multipole_real_t) const* bal_begin =
        NS(Multipole_const_bal_begin)( multipole );

    SIXTRL_ASSERT( bal_begin != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( NS(Multipole_bal_length)( multipole ) > bal_index );
    return bal_begin[ bal_index ];
}

SIXTRL_INLINE NS(multipole_real_t) NS(Multipole_knl)(
    SIXTRL_BE_ARGPTR_DEC const NS(Multipole) *const SIXTRL_RESTRICT multipole,
    NS(multipole_order_t) const knl_index ) SIXTRL_NOEXCEPT
{
    NS(buffer_size_t) const idx = ( NS(buffer_size_t) )( 2 * knl_index );
    SIXTRL_ASSERT( multipole != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( knl_index >= ( NS(multipole_order_t) )0 );
    SIXTRL_ASSERT( knl_index <= NS(Multipole_order)( multipole ) );
    return NS(Multipole_bal)( multipole, idx ) * NS(Math_factorial)( knl_index );
}

SIXTRL_INLINE NS(multipole_real_t) NS(Multipole_ksl)(
    SIXTRL_BE_ARGPTR_DEC const NS(Multipole) *const SIXTRL_RESTRICT multipole,
    NS(multipole_order_t) const ksl_index ) SIXTRL_NOEXCEPT
{
    NS(buffer_size_t) const idx = ( NS(buffer_size_t) )( 2 * ksl_index + 1 );
    SIXTRL_ASSERT( multipole != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( ksl_index >= ( NS(multipole_order_t) )0 );
    SIXTRL_ASSERT( ksl_index <= NS(Multipole_order)( multipole ) );
    return NS(Multipole_bal)( multipole, idx ) * NS(Math_factorial)( ksl_index );
}

/* . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . */

SIXTRL_INLINE SIXTRL_BE_DATAPTR_DEC NS(multipole_real_t)*
NS(Multipole_bal_begin)( SIXTRL_BE_ARGPTR_DEC NS(Multipole)*
    SIXTRL_RESTRICT multipole ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( multipole != SIXTRL_NULLPTR );
    return ( SIXTRL_BE_DATAPTR_DEC NS(multipole_real_t)* )(
                uintptr_t )multipole->bal_addr;
}

SIXTRL_INLINE SIXTRL_BE_DATAPTR_DEC NS(multipole_real_t)*
NS(Multipole_bal_end)( SIXTRL_BE_ARGPTR_DEC NS(Multipole)*
    SIXTRL_RESTRICT multipole ) SIXTRL_NOEXCEPT
{
    SIXTRL_BE_DATAPTR_DEC NS(multipole_real_t)* bal_end =
        NS(Multipole_bal_begin)( multipole );

    if( bal_end != SIXTRL_NULLPTR )
        bal_end = bal_end + NS(Multipole_bal_length)( multipole );

    return bal_end;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE NS(arch_status_t) NS(Multipole_set_length)(
    SIXTRL_BE_ARGPTR_DEC NS(Multipole)* SIXTRL_RESTRICT multipole,
    NS(multipole_real_t) const length ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( multipole != SIXTRL_NULLPTR );
    multipole->length = length;
    return ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
}

SIXTRL_INLINE NS(arch_status_t) NS(Multipole_set_hxl)(
    SIXTRL_BE_ARGPTR_DEC NS(Multipole)* SIXTRL_RESTRICT multipole,
    NS(multipole_real_t) const hxl ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( multipole != SIXTRL_NULLPTR );
    multipole->hxl = hxl;
    return ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
}

SIXTRL_INLINE NS(arch_status_t) NS(Multipole_set_hyl)(
    SIXTRL_BE_ARGPTR_DEC NS(Multipole)* SIXTRL_RESTRICT multipole,
    NS(multipole_real_t) const hyl ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( multipole != SIXTRL_NULLPTR );
    multipole->hyl = hyl;
    return ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
}

SIXTRL_INLINE NS(arch_status_t) NS(Multipole_set_order)(
    SIXTRL_BE_ARGPTR_DEC NS(Multipole)* SIXTRL_RESTRICT multipole,
    NS(multipole_order_t) const order ) SIXTRL_NOEXCEPT
{
    NS(arch_status_t) status = ( NS(arch_status_t)
        )SIXTRL_ARCH_STATUS_GENERAL_FAILURE;

    if( ( multipole != SIXTRL_NULLPTR ) &&
        ( order >= ( NS(multipole_order_t) )0 ) )
    {
        multipole->order = order;
        status = ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
    }

    return status;
}

SIXTRL_INLINE NS(arch_status_t) NS(Multipole_set_bal_addr)(
    SIXTRL_BE_ARGPTR_DEC NS(Multipole)* SIXTRL_RESTRICT multipole,
    NS(buffer_addr_t) const bal_addr ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( multipole != SIXTRL_NULLPTR );
    multipole->bal_addr = bal_addr;
    return ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
}

SIXTRL_INLINE NS(arch_status_t) NS(Multipole_set_bal_value)(
    SIXTRL_BE_ARGPTR_DEC NS(Multipole)* SIXTRL_RESTRICT multipole,
    NS(buffer_size_t) const bal_index,
    NS(multipole_real_t) const bal_value ) SIXTRL_NOEXCEPT
{
    NS(arch_status_t) status = (
        NS(arch_status_t) )SIXTRL_ARCH_STATUS_GENERAL_FAILURE;

    SIXTRL_BE_DATAPTR_DEC NS(multipole_real_t)* bal_begin =
        NS(Multipole_bal_begin)( multipole );

    if( ( multipole != SIXTRL_NULLPTR ) && ( bal_begin != SIXTRL_NULLPTR ) &&
        ( bal_index < NS(Multipole_bal_length)( multipole ) ) )
    {
        bal_begin[ bal_index ] = bal_value;
        status = ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
    }

    return status;
}

SIXTRL_INLINE NS(arch_status_t) NS(Multipole_set_all_bal_values)(
    SIXTRL_BE_ARGPTR_DEC NS(Multipole)* SIXTRL_RESTRICT multipole,
    NS(multipole_real_t) const bal_value ) SIXTRL_NOEXCEPT
{
    NS(arch_status_t) status = (
        NS(arch_status_t) )SIXTRL_ARCH_STATUS_GENERAL_FAILURE;

    SIXTRL_BE_DATAPTR_DEC NS(multipole_real_t)* bal_it =
            NS(Multipole_bal_begin)( multipole );

    SIXTRL_BE_DATAPTR_DEC NS(multipole_real_t)* bal_end =
        NS(Multipole_bal_begin)( multipole );

    if( ( multipole != SIXTRL_NULLPTR ) && ( bal_it != SIXTRL_NULLPTR ) &&
        ( bal_end != SIXTRL_NULLPTR ) )
    {
        SIXTRL_ASSERT( ( ( uintptr_t )bal_it ) <= ( uintptr_t )bal_end );
        for( ; bal_it != bal_end ; ++bal_it ) *bal_it = bal_value;
        status = ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
    }

    return status;
}

SIXTRL_INLINE NS(arch_status_t) NS(Multipole_set_bal)(
    SIXTRL_BE_ARGPTR_DEC NS(Multipole)* SIXTRL_RESTRICT multipole,
    SIXTRL_BE_DATAPTR_DEC NS(multipole_real_t) const*
        SIXTRL_RESTRICT in_bal_it ) SIXTRL_NOEXCEPT
{
    NS(arch_status_t) status = (
        NS(arch_status_t) )SIXTRL_ARCH_STATUS_GENERAL_FAILURE;

    SIXTRL_BE_DATAPTR_DEC NS(multipole_real_t)* bal_it =
            NS(Multipole_bal_begin)( multipole );

    SIXTRL_BE_DATAPTR_DEC NS(multipole_real_t)* bal_end =
        NS(Multipole_bal_end)( multipole );

    if( ( multipole != SIXTRL_NULLPTR ) && ( bal_it  != SIXTRL_NULLPTR ) &&
        ( in_bal_it != SIXTRL_NULLPTR ) && ( bal_end != SIXTRL_NULLPTR ) )
    {
        SIXTRL_ASSERT( ( ( uintptr_t )bal_it ) <= ( uintptr_t )bal_end );
        for( ; bal_it != bal_end ; ++bal_it, ++in_bal_it ) *bal_it = *in_bal_it;
        status = ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
    }

    return status;
}

/* . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . */

SIXTRL_INLINE NS(arch_status_t) NS(Multipole_set_knl_value)(
    SIXTRL_BE_ARGPTR_DEC NS(Multipole)* SIXTRL_RESTRICT multipole,
    NS(buffer_size_t) const knl_index,
    NS(multipole_real_t) const knl_value ) SIXTRL_NOEXCEPT
{
    NS(arch_status_t) status = (
        NS(arch_status_t) )SIXTRL_ARCH_STATUS_GENERAL_FAILURE;

    SIXTRL_BE_DATAPTR_DEC NS(multipole_real_t)* bal_values =
        NS(Multipole_bal_begin)( multipole );

    NS(buffer_size_t) const bal_index = ( NS(buffer_size_t) )( 2 * knl_index );

    if( ( multipole != SIXTRL_NULLPTR ) && ( bal_values != SIXTRL_NULLPTR ) &&
        ( bal_index < NS(Multipole_bal_length)( multipole ) ) )
    {
        bal_values[ bal_index ] = knl_value / NS(Math_factorial)( knl_index );
        status = ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
    }

    return status;
}

SIXTRL_INLINE NS(arch_status_t) NS(Multipole_set_all_knl_values)(
    SIXTRL_BE_ARGPTR_DEC NS(Multipole)* SIXTRL_RESTRICT multipole,
    NS(multipole_real_t) const knl_value ) SIXTRL_NOEXCEPT
{
    NS(arch_status_t) status = (
        NS(arch_status_t) )SIXTRL_ARCH_STATUS_GENERAL_FAILURE;

    SIXTRL_BE_DATAPTR_DEC NS(multipole_real_t)* bal_values =
        NS(Multipole_bal_begin)( multipole );

    if( ( multipole != SIXTRL_NULLPTR ) && ( bal_values != SIXTRL_NULLPTR ) )
    {
        NS(multipole_order_t) knl_index = ( NS(multipole_order_t) )0;
        NS(buffer_size_t) ii = ( NS(buffer_size_t) )0;
        NS(buffer_size_t) const nn = NS(Multipole_bal_length)( multipole );

        for( ; ii < nn ; ii += 2u, ++knl_index )
        {
            bal_values[ ii ] = knl_value / NS(Math_factorial)( knl_index );
        }

        status = ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
    }

    return status;
}

SIXTRL_INLINE NS(arch_status_t) NS(Multipole_set_knl)(
    SIXTRL_BE_DATAPTR_DEC NS(Multipole)* SIXTRL_RESTRICT multipole,
    SIXTRL_BE_DATAPTR_DEC NS(multipole_real_t) const*
        SIXTRL_RESTRICT knl_begin ) SIXTRL_NOEXCEPT
{
    NS(arch_status_t) status = (
        NS(arch_status_t) )SIXTRL_ARCH_STATUS_GENERAL_FAILURE;

    SIXTRL_BE_DATAPTR_DEC NS(multipole_real_t)* bal_values =
        NS(Multipole_bal_begin)( multipole );

    if( ( multipole != SIXTRL_NULLPTR ) && ( bal_values != SIXTRL_NULLPTR ) &&
        ( knl_begin != SIXTRL_NULLPTR ) )
    {
        NS(multipole_order_t) knl_index = ( NS(multipole_order_t) )0;
        NS(buffer_size_t) ii = ( NS(buffer_size_t) )0;
        NS(buffer_size_t) const nn = NS(Multipole_bal_length)( multipole );

        for( ; ii < nn ; ii += 2u, ++knl_index )
        {
            bal_values[ ii ] = knl_begin[ knl_index ] /
                NS(Math_factorial)( knl_index );
        }

        status = ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
    }

    return status;
}

/* . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . */

SIXTRL_INLINE NS(arch_status_t) NS(Multipole_set_ksl_value)(
    SIXTRL_BE_ARGPTR_DEC NS(Multipole)* SIXTRL_RESTRICT multipole,
    NS(buffer_size_t) const ksl_index,
    NS(multipole_real_t) const ksl_value ) SIXTRL_NOEXCEPT
{
    NS(arch_status_t) status = (
        NS(arch_status_t) )SIXTRL_ARCH_STATUS_GENERAL_FAILURE;

    SIXTRL_BE_DATAPTR_DEC NS(multipole_real_t)* bal_values =
        NS(Multipole_bal_begin)( multipole );

    NS(buffer_size_t) const bal_index =
        ( NS(buffer_size_t) )( 2 * ksl_index + 1 );

    if( ( multipole != SIXTRL_NULLPTR ) && ( bal_values != SIXTRL_NULLPTR ) &&
        ( bal_index < NS(Multipole_bal_length)( multipole ) ) )
    {
        bal_values[ bal_index ] = ksl_value / NS(Math_factorial)( ksl_index );
        status = ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
    }

    return status;
}

SIXTRL_INLINE NS(arch_status_t) NS(Multipole_set_all_ksl_values)(
    SIXTRL_BE_ARGPTR_DEC NS(Multipole)* SIXTRL_RESTRICT multipole,
    NS(multipole_real_t) const ksl_value ) SIXTRL_NOEXCEPT
{
    NS(arch_status_t) status = (
        NS(arch_status_t) )SIXTRL_ARCH_STATUS_GENERAL_FAILURE;

    SIXTRL_BE_DATAPTR_DEC NS(multipole_real_t)* bal_values =
        NS(Multipole_bal_begin)( multipole );

    if( ( multipole != SIXTRL_NULLPTR ) && ( bal_values != SIXTRL_NULLPTR ) )
    {
        NS(multipole_order_t) ksl_index = ( NS(multipole_order_t) )0;
        NS(buffer_size_t) ii = ( NS(buffer_size_t) )1;
        NS(buffer_size_t) const nn = NS(Multipole_bal_length)( multipole );

        for( ; ii < nn ; ii += 2u, ++ksl_index )
        {
            bal_values[ ii ] = ksl_value / NS(Math_factorial)( ksl_index );
        }

        status = ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
    }

    return status;
}

SIXTRL_INLINE NS(arch_status_t) NS(Multipole_set_ksl)(
    SIXTRL_BE_DATAPTR_DEC NS(Multipole)* SIXTRL_RESTRICT multipole,
    SIXTRL_BE_DATAPTR_DEC NS(multipole_real_t) const*
        SIXTRL_RESTRICT ksl_begin ) SIXTRL_NOEXCEPT
{
    NS(arch_status_t) status = (
        NS(arch_status_t) )SIXTRL_ARCH_STATUS_GENERAL_FAILURE;

    SIXTRL_BE_DATAPTR_DEC NS(multipole_real_t)* bal_values =
        NS(Multipole_bal_begin)( multipole );

    if( ( multipole != SIXTRL_NULLPTR ) && ( bal_values != SIXTRL_NULLPTR ) &&
        ( ksl_begin != SIXTRL_NULLPTR ) )
    {
        NS(multipole_order_t) ksl_index = ( NS(multipole_order_t) )0;
        NS(buffer_size_t) ii = ( NS(buffer_size_t) )1;
        NS(buffer_size_t) const nn = NS(Multipole_bal_length)( multipole );

        for( ; ii < nn ; ii += 2u, ++ksl_index )
        {
            bal_values[ ii ] = ksl_begin[ ksl_index ] /
                NS(Math_factorial)( ksl_index );
        }

        status = ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
    }

    return status;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE NS(arch_status_t) NS(Multipole_copy)(
    SIXTRL_BE_ARGPTR_DEC NS(Multipole)* SIXTRL_RESTRICT dst,
    SIXTRL_BE_ARGPTR_DEC const NS(Multipole) *const SIXTRL_RESTRICT src
) SIXTRL_NOEXCEPT
{
    NS(arch_status_t) status = (
        NS(arch_status_t) )SIXTRL_ARCH_STATUS_GENERAL_FAILURE;

    if( ( dst != SIXTRL_NULLPTR ) && ( src != SIXTRL_NULLPTR ) )
    {
        if( ( src != dst ) &&
            ( NS(Multipole_order)( src ) == NS(Multipole_order)( dst ) ) )
        {
            status  = NS(Multipole_set_length)(
                dst, NS(Multipole_length)( src ) );

            status |= NS(Multipole_set_hxl)(
                dst, NS(Multipole_hxl)( src ) );

            status |= NS(Multipole_set_hyl)(
                dst, NS(Multipole_hyl)( src ) );

            status |= NS(Multipole_set_bal)( dst,
                        NS(Multipole_const_bal_begin)( src ) );
        }
        else if( src == dst )
        {
            status = ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
        }
    }

    return status;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(Multipole) const*
NS(Multipole_const_from_obj_index)( SIXTRL_BUFFER_OBJ_ARGPTR_DEC const NS(Object)
    *const SIXTRL_RESTRICT obj ) SIXTRL_NOEXCEPT
{
    typedef SIXTRL_BE_ARGPTR_DEC NS(Multipole) const* ptr_multipole_t;
    ptr_multipole_t elem = SIXTRL_NULLPTR;

    if( ( obj != SIXTRL_NULLPTR ) &&
        ( NS(Object_get_type_id)( obj ) == NS(OBJECT_TYPE_MULTIPOLE) ) &&
        ( NS(Object_get_size)( obj ) >= sizeof( NS(Multipole) ) ) )
    {
        elem = ( ptr_multipole_t )( uintptr_t )NS(Object_get_begin_addr)( obj );
    }

    return elem;
}

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(Multipole)* NS(Multipole_from_obj_index)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)*
        SIXTRL_RESTRICT obj ) SIXTRL_NOEXCEPT
{
    return ( SIXTRL_BE_ARGPTR_DEC NS(Multipole)*
        )NS(Multipole_const_from_obj_index)( obj );
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(Multipole) const*
NS(Multipole_const_from_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT buffer_begin,
    NS(buffer_size_t) const index,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT
{
    return NS(Multipole_const_from_obj_index)(
        NS(ManagedBuffer_get_const_object)( buffer_begin, index, slot_size ) );
}

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(Multipole)*
NS(Multipole_from_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT buffer_begin,
    NS(buffer_size_t) const index,
    NS(buffer_size_t) const slot_size ) SIXTRL_NOEXCEPT
{
    return NS(Multipole_from_obj_index)(
        NS(ManagedBuffer_get_object)( buffer_begin, index, slot_size ) );
}

#if !defined( _GPUCODE )

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(Multipole) const*
NS(Multipole_const_from_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const index ) SIXTRL_NOEXCEPT
{
    return NS(Multipole_const_from_obj_index)(
        NS(Buffer_get_const_object)( buffer, index ) );
}

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(Multipole)* NS(Multipole_from_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const index ) SIXTRL_NOEXCEPT
{
    return NS(Multipole_from_obj_index)(
        NS(Buffer_get_object)( buffer, index ) );
}

#endif /* Host */
#if !defined(  _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */
#endif /* SIXTRACKLIB_COMMON_BE_MULTIPOLE_BEAM_ELEMENT_MULTIPOLE_H__ */
