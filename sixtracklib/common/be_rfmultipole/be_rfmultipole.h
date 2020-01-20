#ifndef SIXTRACKL_COMMON_BE_RFMULTIPOLE_BE_RFMULTIPOLE_C99_H__
#define SIXTRACKL_COMMON_BE_RFMULTIPOLE_BE_RFMULTIPOLE_C99_H__

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/control/definitions.h"
    #include "sixtracklib/common/internal/beam_elements_defines.h"
    #include "sixtracklib/common/buffer/buffer_type.h"
    #include "sixtracklib/common/buffer/buffer_object.h"
    #include "sixtracklib/common/internal/objects_type_id.h"
    #if !defined( _GPUCODE )
        #include "sixtracklib/common/buffer.h"
    #endif /* !defined( _GPUCODE ) */
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stddef.h>
    #include <stdlib.h>
    #include <stdint.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if defined( __cplusplus ) && !defined( _GPUCODE )
extern "C" {
#endif /* defined( __cplusplus ) && !defined( _GPUCODE ) */

typedef SIXTRL_REAL_T  NS(rf_multipole_real_t);
typedef SIXTRL_INT64_T NS(rf_multipole_int_t);

typedef struct NS(RFMultiPole)
{
    NS(rf_multipole_int_t)    order               SIXTRL_ALIGN( 8 );
    NS(rf_multipole_real_t)   voltage             SIXTRL_ALIGN( 8 );
    NS(rf_multipole_real_t)   frequency           SIXTRL_ALIGN( 8 );
    NS(rf_multipole_real_t)   lag                 SIXTRL_ALIGN( 8 );
    NS(buffer_addr_t)         bal_addr            SIXTRL_ALIGN( 8 );
    NS(buffer_addr_t)         phase_addr          SIXTRL_ALIGN( 8 );
}
NS(RFMultiPole);

/* ************************************************************************* */

SIXTRL_STATIC SIXTRL_FN NS(rf_multipole_int_t)
NS(RFMultiPole_calculate_factorial)( NS(rf_multipole_int_t) const n );

SIXTRL_STATIC SIXTRL_FN NS(rf_multipole_real_t)
NS(RFMultiPole_calculate_factorial_real)( NS(rf_multipole_int_t) const n );

/* ************************************************************************* */

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_ARGPTR_DEC NS(RFMultiPole)*
NS(RFMultiPole_preset)(
    SIXTRL_BE_ARGPTR_DEC NS(RFMultiPole)* SIXTRL_RESTRICT multipole );

SIXTRL_STATIC SIXTRL_FN void NS(RFMultiPole_clear)(
    SIXTRL_BE_ARGPTR_DEC NS(RFMultiPole)* SIXTRL_RESTRICT multipole );

/* ------------------------------------------------------------------- */

SIXTRL_STATIC SIXTRL_FN NS(object_type_id_t) NS(RFMultiPole_type_id)(
    SIXTRL_BE_ARGPTR_DEC const NS(RFMultiPole) *const SIXTRL_RESTRICT mpole );

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t) NS(RFMultiPole_num_dataptrs)(
    SIXTRL_BE_ARGPTR_DEC const NS(RFMultiPole) *const SIXTRL_RESTRICT mpole );

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t) NS(RFMultiPole_num_slots)(
    SIXTRL_BE_ARGPTR_DEC const NS(RFMultiPole) *const SIXTRL_RESTRICT mpole,
    NS(buffer_size_t) const slot_size );

/* ------------------------------------------------------------------- */

SIXTRL_STATIC SIXTRL_FN NS(rf_multipole_real_t) NS(RFMultiPole_voltage)(
    SIXTRL_BE_ARGPTR_DEC const NS(RFMultiPole) *const SIXTRL_RESTRICT mpole );

SIXTRL_STATIC SIXTRL_FN NS(rf_multipole_real_t) NS(RFMultiPole_frequency)(
    SIXTRL_BE_ARGPTR_DEC const NS(RFMultiPole) *const SIXTRL_RESTRICT mpole );

SIXTRL_STATIC SIXTRL_FN NS(rf_multipole_real_t) NS(RFMultiPole_lag)(
    SIXTRL_BE_ARGPTR_DEC const NS(RFMultiPole) *const SIXTRL_RESTRICT mpole );

SIXTRL_STATIC SIXTRL_FN NS(rf_multipole_int_t) NS(RFMultiPole_order )(
    SIXTRL_BE_ARGPTR_DEC const NS(RFMultiPole) *const SIXTRL_RESTRICT mpole );

SIXTRL_STATIC SIXTRL_FN NS(rf_multipole_int_t) NS(RFMultiPole_num_bal_elements)(
    SIXTRL_BE_ARGPTR_DEC const NS(RFMultiPole) *const SIXTRL_RESTRICT mpole );

SIXTRL_STATIC SIXTRL_FN NS(rf_multipole_int_t)
NS(RFMultiPole_num_phase_elements)( SIXTRL_BE_ARGPTR_DEC const
    NS(RFMultiPole) *const SIXTRL_RESTRICT mpole );

SIXTRL_STATIC SIXTRL_FN NS(buffer_addr_t) NS(RFMultiPole_bal_addr)(
    SIXTRL_BE_ARGPTR_DEC const NS(RFMultiPole) *const SIXTRL_RESTRICT mpole );

SIXTRL_STATIC SIXTRL_FN NS(buffer_addr_t) NS(RFMultiPole_phase_addr)(
    SIXTRL_BE_ARGPTR_DEC const NS(RFMultiPole) *const SIXTRL_RESTRICT mpole );



SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_DATAPTR_DEC NS(rf_multipole_real_t) const*
NS(RFMultiPole_const_bal)( SIXTRL_BE_ARGPTR_DEC const NS(RFMultiPole)
    *const SIXTRL_RESTRICT mpole );

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_DATAPTR_DEC NS(rf_multipole_real_t)*
NS(RFMultiPole_bal)( SIXTRL_BE_ARGPTR_DEC NS(RFMultiPole)*
    SIXTRL_RESTRICT mpole );

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_DATAPTR_DEC NS(rf_multipole_real_t) const*
NS(RFMultiPole_const_phase)( SIXTRL_BE_ARGPTR_DEC const NS(RFMultiPole)
    *const SIXTRL_RESTRICT mpole );

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_DATAPTR_DEC NS(rf_multipole_real_t)*
NS(RFMultiPole_phase)( SIXTRL_BE_ARGPTR_DEC NS(RFMultiPole)*
    SIXTRL_RESTRICT mpole );


SIXTRL_STATIC SIXTRL_FN NS(rf_multipole_real_t) NS(RFMultiPole_knl_value)(
    SIXTRL_BE_ARGPTR_DEC const NS(RFMultiPole) *const SIXTRL_RESTRICT mpole,
    NS(rf_multipole_int_t) const index );

SIXTRL_FN SIXTRL_STATIC NS(rf_multipole_real_t) NS(RFMultiPole_ksl_value)(
    SIXTRL_BE_ARGPTR_DEC const NS(RFMultiPole) *const SIXTRL_RESTRICT mpole,
    NS(rf_multipole_int_t) const index );

SIXTRL_STATIC SIXTRL_FN NS(rf_multipole_real_t) NS(RFMultiPole_pn_value)(
    SIXTRL_BE_ARGPTR_DEC const NS(RFMultiPole) *const SIXTRL_RESTRICT mpole,
    NS(rf_multipole_int_t) const index );

SIXTRL_FN SIXTRL_STATIC NS(rf_multipole_real_t) NS(RFMultiPole_ps_value)(
    SIXTRL_BE_ARGPTR_DEC const NS(RFMultiPole) *const SIXTRL_RESTRICT mpole,
    NS(rf_multipole_int_t) const index );

/* ------------------------------------------------------------------- */

SIXTRL_STATIC SIXTRL_FN void NS(RFMultiPole_set_voltage)(
    SIXTRL_BE_ARGPTR_DEC NS(RFMultiPole)* SIXTRL_RESTRICT mpole,
    NS(rf_multipole_real_t) const voltage );

SIXTRL_STATIC SIXTRL_FN void NS(RFMultiPole_set_frequency)(
    SIXTRL_BE_ARGPTR_DEC NS(RFMultiPole)* SIXTRL_RESTRICT mpole,
    NS(rf_multipole_real_t) const frequency );

SIXTRL_STATIC SIXTRL_FN void NS(RFMultiPole_set_lag)(
    SIXTRL_BE_ARGPTR_DEC NS(RFMultiPole)* SIXTRL_RESTRICT mpole,
    NS(rf_multipole_real_t) const lag );

SIXTRL_STATIC SIXTRL_FN void NS(RFMultiPole_set_order )(
    SIXTRL_BE_ARGPTR_DEC NS(RFMultiPole)* SIXTRL_RESTRICT mpole,
    NS(rf_multipole_int_t) const order );

SIXTRL_STATIC SIXTRL_FN void NS(RFMultiPole_set_bal_addr)(
    SIXTRL_BE_ARGPTR_DEC NS(RFMultiPole)* SIXTRL_RESTRICT mpole,
    NS(buffer_addr_t) const bal_addr );

SIXTRL_STATIC SIXTRL_FN void NS(RFMultiPole_set_phase_addr)(
    SIXTRL_BE_ARGPTR_DEC NS(RFMultiPole)* SIXTRL_RESTRICT mpole,
    NS(buffer_addr_t) const phase_addr );

/* ------------------------------------------------------------------- */

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_ARGPTR_DEC NS(RFMultiPole) const*
NS(RFMultiPole_const_from_obj_index)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC const NS(Object) *const obj );

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_ARGPTR_DEC NS(RFMultiPole)*
NS(RFMultiPole_from_obj_index)( SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)* obj );

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_ARGPTR_DEC NS(RFMultiPole) const*
NS(RFMultiPole_const_from_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT buffer_begin,
    NS(buffer_size_t) const index, NS(buffer_size_t) const slot_size );

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_ARGPTR_DEC NS(RFMultiPole)*
NS(RFMultiPole_from_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT buffer_begin,
    NS(buffer_size_t) const index, NS(buffer_size_t) const slot_size );

#if !defined( _GPUCODE )

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BE_ARGPTR_DEC NS(RFMultiPole) const*
NS(RFMultiPole_const_from_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const index );

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BE_ARGPTR_DEC NS(RFMultiPole)*
NS(RFMultiPole_from_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const index );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t) NS(RFMultiPole_attributes_offsets)(
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT offsets_begin,
    NS(buffer_size_t) const max_num_offsets,
    SIXTRL_BUFFER_DATAPTR_DEC const NS(RFMultiPole) *const SIXTRL_RESTRICT data,
    NS(buffer_size_t) const slot_size );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t) NS(RFMultiPole_attributes_sizes)(
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT sizes_begin,
    NS(buffer_size_t) const max_num_sizes,
    SIXTRL_BUFFER_DATAPTR_DEC const NS(RFMultiPole) *const SIXTRL_RESTRICT data,
    NS(buffer_size_t) const slot_size );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t) NS(RFMultiPole_attributes_counts)(
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT counts_begin,
    NS(buffer_size_t) const max_num_counts,
    SIXTRL_BUFFER_DATAPTR_DEC const NS(RFMultiPole) *const SIXTRL_RESTRICT data );

/* ------------------------------------------------------------------------- */

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(RFMultiPole_can_be_added)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(rf_multipole_int_t) const order,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_objects,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_slots,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_dataptrs );

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BE_ARGPTR_DEC NS(RFMultiPole)*
NS(RFMultiPole_new)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(rf_multipole_int_t) const order );

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BE_ARGPTR_DEC NS(RFMultiPole)*
NS(RFMultiPole_add)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(rf_multipole_int_t) const order,
    NS(rf_multipole_real_t) const voltage,
    NS(rf_multipole_real_t) const frequency,
    NS(rf_multipole_real_t) const lag,
    SIXTRL_ARGPTR_DEC NS(rf_multipole_real_t) const* SIXTRL_RESTRICT bal_values,
    SIXTRL_ARGPTR_DEC NS(rf_multipole_real_t) const*
        SIXTRL_RESTRICT phase_values );

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BE_ARGPTR_DEC NS(RFMultiPole)*
NS(RFMultiPole_add_copy)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(RFMultiPole) *const SIXTRL_RESTRICT mpole );

#endif /* !defined( _GPUCODE ) */

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(RFMultiPole_copy)(
    SIXTRL_BE_ARGPTR_DEC NS(RFMultiPole)* SIXTRL_RESTRICT dest,
    SIXTRL_BUFFER_DATAPTR_DEC const NS(RFMultiPole) *const SIXTRL_RESTRICT src );

#if defined( __cplusplus ) && !defined( _GPUCODE )
}
#endif /* defined( __cplusplus ) && !defined( _GPUCODE ) */

/* ****************************************************************** */

#if defined( __cplusplus ) && !defined( _GPUCODE )
extern "C" {
#endif /* defined( __cplusplus ) && !defined( _GPUCODE ) */

SIXTRL_INLINE NS(rf_multipole_int_t) NS(RFMultiPole_calculate_factorial)(
    NS(rf_multipole_int_t) const n )
{
    NS(rf_multipole_int_t) result = ( NS(rf_multipole_int_t) )1u;
    NS(rf_multipole_int_t) ii     = ( NS(rf_multipole_int_t) )1u;

    for( ; ii <= n ; ++ii ) result *= ii;
    return result;
}

SIXTRL_INLINE NS(rf_multipole_real_t) NS(RFMultiPole_calculate_factorial_real)(
    NS(rf_multipole_int_t) const n )
{
    NS(rf_multipole_real_t) result = ( NS(rf_multipole_real_t) )1.0;
    NS(rf_multipole_int_t) ii = ( NS(rf_multipole_int_t) )1u;

    for( ; ii <= n ; ++ii ) result *= ( NS(rf_multipole_real_t) )ii;
    return result;
}

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(RFMultiPole)*
NS(RFMultiPole_preset)(
    SIXTRL_BE_ARGPTR_DEC NS(RFMultiPole)* SIXTRL_RESTRICT multipole )
{
    if( multipole != SIXTRL_NULLPTR )
    {
        multipole->order = ( NS(rf_multipole_int_t) )0u;
        NS(RFMultiPole_clear)( multipole );
    }

    return multipole;
}

SIXTRL_INLINE void NS(RFMultiPole_clear)(
    SIXTRL_BE_ARGPTR_DEC NS(RFMultiPole)* SIXTRL_RESTRICT multipole )
{
    SIXTRL_ASSERT( multipole != SIXTRL_NULLPTR );

    multipole->voltage = ( NS(rf_multipole_real_t) )0.0;
    multipole->frequency = ( NS(rf_multipole_real_t) )0.0;
    multipole->lag = ( NS(rf_multipole_real_t) )0.0;

    if( multipole->order == ( NS(rf_multipole_int_t) )0u )
    {
        multipole->bal_addr = ( NS(buffer_addr_t) )0u;
        multipole->phase_addr = ( NS(buffer_addr_t) )0u;
    }
    else if( multipole->order >= ( NS(rf_multipole_int_t) )0u )
    {
        typedef NS(rf_multipole_real_t) real_t;
        SIXTRL_STATIC_VAR real_t const ZERO = ( real_t )0.0;

        NS(buffer_size_t) const bal_size =
            ( NS(buffer_size_t) )NS(RFMultiPole_num_bal_elements)( multipole );

        NS(buffer_size_t) const phase_size = ( NS(buffer_size_t)
            )NS(RFMultiPole_num_phase_elements)( multipole );

        SIXTRL_BE_DATAPTR_DEC real_t* bal = NS(RFMultiPole_bal)( multipole );
        SIXTRL_BE_DATAPTR_DEC real_t* phase =
            NS(RFMultiPole_phase)( multipole );

        SIXTRACKLIB_SET_VALUES( real_t, bal, bal_size, ZERO );
        SIXTRACKLIB_SET_VALUES( real_t, phase, phase_size, ZERO );
    }
}

/* ----------------------------------------------------------------------- */

SIXTRL_INLINE NS(object_type_id_t) NS(RFMultiPole_type_id)(
    SIXTRL_BE_ARGPTR_DEC const NS(RFMultiPole) *const SIXTRL_RESTRICT mpole )
{
    ( void )mpole;
    return NS(OBJECT_TYPE_RF_MULTIPOLE);
}

SIXTRL_INLINE NS(buffer_size_t) NS(RFMultiPole_num_dataptrs)(
    SIXTRL_BE_ARGPTR_DEC const NS(RFMultiPole) *const SIXTRL_RESTRICT mpole )
{
    ( void )mpole;
    return ( NS(buffer_size_t) )2u;
}

SIXTRL_INLINE NS(buffer_size_t) NS(RFMultiPole_num_slots)(
    SIXTRL_BE_ARGPTR_DEC const NS(RFMultiPole) *const SIXTRL_RESTRICT mpole,
    NS(buffer_size_t) const slot_size )
{
    NS(buffer_size_t) num_slots = ( NS(buffer_size_t) )0u;

    if( slot_size > ( NS(buffer_size_t) )0u )
    {
        NS(buffer_size_t) required_size =
            NS(ManagedBuffer_get_slot_based_length)(
                sizeof( NS(RFMultiPole) ), slot_size );

        SIXTRL_ASSERT( NS(RFMultiPole_num_dataptrs)( mpole ) ==
            ( NS(buffer_size_t) )2u );

        if( ( mpole != SIXTRL_NULLPTR ) && ( mpole->order > 0 ) )
        {
            NS(buffer_size_t) const bal_size = ( NS(buffer_size_t)
                )NS(RFMultiPole_num_bal_elements)( mpole );

            NS(buffer_size_t) const phase_size = ( NS(buffer_size_t)
                )NS(RFMultiPole_num_phase_elements)( mpole );

            required_size += NS(ManagedBuffer_get_slot_based_length)(
                sizeof( NS(rf_multipole_real_t) ) * bal_size, slot_size );

            required_size += NS(ManagedBuffer_get_slot_based_length)(
                sizeof( NS(rf_multipole_real_t) ) * phase_size, slot_size );
        }

        num_slots = required_size / slot_size;

        if( ( num_slots * slot_size ) < required_size )
        {
            ++num_slots;
        }

        SIXTRL_ASSERT( ( num_slots * slot_size ) >= required_size );
    }

    return num_slots;
}

/* ------------------------------------------------------------------- */

SIXTRL_INLINE NS(rf_multipole_real_t) NS(RFMultiPole_voltage)(
    SIXTRL_BE_ARGPTR_DEC const NS(RFMultiPole) *const SIXTRL_RESTRICT mpole )
{
    SIXTRL_ASSERT( mpole != SIXTRL_NULLPTR );
    return mpole->voltage;
}

SIXTRL_INLINE NS(rf_multipole_real_t) NS(RFMultiPole_frequency)(
    SIXTRL_BE_ARGPTR_DEC const NS(RFMultiPole) *const SIXTRL_RESTRICT mpole )
{
    SIXTRL_ASSERT( mpole != SIXTRL_NULLPTR );
    return mpole->frequency;
}

SIXTRL_INLINE NS(rf_multipole_real_t) NS(RFMultiPole_lag)(
    SIXTRL_BE_ARGPTR_DEC const NS(RFMultiPole) *const SIXTRL_RESTRICT mpole )
{
    SIXTRL_ASSERT( mpole != SIXTRL_NULLPTR );
    return mpole->lag;
}

SIXTRL_INLINE NS(rf_multipole_int_t) NS(RFMultiPole_order )(
    SIXTRL_BE_ARGPTR_DEC const NS(RFMultiPole) *const SIXTRL_RESTRICT mpole )
{
    SIXTRL_ASSERT( mpole != SIXTRL_NULLPTR );
    return mpole->order;
}

SIXTRL_INLINE NS(rf_multipole_int_t) NS(RFMultiPole_num_bal_elements)(
    SIXTRL_BE_ARGPTR_DEC const NS(RFMultiPole) *const SIXTRL_RESTRICT mpole )
{
    SIXTRL_ASSERT( mpole != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( mpole->order >= ( NS(rf_multipole_int_t) )0u );
    return ( NS(rf_multipole_int_t) )( 2u * mpole->order + 2u );
}

SIXTRL_INLINE NS(rf_multipole_int_t) NS(RFMultiPole_num_phase_elements)(
    SIXTRL_BE_ARGPTR_DEC const NS(RFMultiPole) *const SIXTRL_RESTRICT mpole )
{
    SIXTRL_ASSERT( mpole != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( mpole->order >= ( NS(rf_multipole_int_t) )0u );
    return ( NS(rf_multipole_int_t) )( 2u * mpole->order + 2u );
}

SIXTRL_INLINE NS(buffer_addr_t) NS(RFMultiPole_bal_addr)(
    SIXTRL_BE_ARGPTR_DEC const NS(RFMultiPole) *const SIXTRL_RESTRICT mpole )
{
    SIXTRL_ASSERT( mpole != SIXTRL_NULLPTR );
    return mpole->bal_addr;
}

SIXTRL_INLINE NS(buffer_addr_t) NS(RFMultiPole_phase_addr)(
    SIXTRL_BE_ARGPTR_DEC const NS(RFMultiPole) *const SIXTRL_RESTRICT mpole )
{
    SIXTRL_ASSERT( mpole != SIXTRL_NULLPTR );
    return mpole->phase_addr;
}

SIXTRL_INLINE SIXTRL_BE_DATAPTR_DEC NS(rf_multipole_real_t) const*
NS(RFMultiPole_const_bal)( SIXTRL_BE_ARGPTR_DEC const NS(RFMultiPole)
    *const SIXTRL_RESTRICT mpole )
{
    SIXTRL_ASSERT( mpole != SIXTRL_NULLPTR );
    return ( SIXTRL_BE_DATAPTR_DEC NS(rf_multipole_real_t) const* )(
        uintptr_t )mpole->bal_addr;
}

SIXTRL_INLINE SIXTRL_BE_DATAPTR_DEC NS(rf_multipole_real_t)*
NS(RFMultiPole_bal)( SIXTRL_BE_ARGPTR_DEC NS(RFMultiPole)* SIXTRL_RESTRICT mpole )
{
    return ( SIXTRL_BE_DATAPTR_DEC NS(rf_multipole_real_t)*
        )NS(RFMultiPole_const_bal)( mpole );
}

SIXTRL_INLINE SIXTRL_BE_DATAPTR_DEC NS(rf_multipole_real_t) const*
NS(RFMultiPole_const_phase)( SIXTRL_BE_ARGPTR_DEC const NS(RFMultiPole)
    *const SIXTRL_RESTRICT mpole )
{
    SIXTRL_ASSERT( mpole != SIXTRL_NULLPTR );
    return ( SIXTRL_BE_DATAPTR_DEC NS(rf_multipole_real_t) const* )(
        uintptr_t )mpole->phase_addr;
}

SIXTRL_INLINE SIXTRL_BE_DATAPTR_DEC NS(rf_multipole_real_t)*
NS(RFMultiPole_phase)( SIXTRL_BE_ARGPTR_DEC
    NS(RFMultiPole)* SIXTRL_RESTRICT mpole )
{
    return ( SIXTRL_BE_DATAPTR_DEC NS(rf_multipole_real_t)*
        )NS(RFMultiPole_const_phase)( mpole );
}

SIXTRL_INLINE NS(rf_multipole_real_t) NS(RFMultiPole_knl_value)(
    SIXTRL_BE_ARGPTR_DEC const NS(RFMultiPole) *const SIXTRL_RESTRICT mpole,
    NS(rf_multipole_int_t) const index )
{
    SIXTRL_ASSERT( mpole != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( ( mpole->order + ( NS(rf_multipole_int_t) )1u ) >= index );
    SIXTRL_ASSERT( mpole->bal_addr != ( NS(buffer_addr_t) )0u );

    return NS(RFMultiPole_const_bal)( mpole )[ 2 * index ] *
        NS(RFMultiPole_calculate_factorial_real)( index );
}

SIXTRL_INLINE NS(rf_multipole_real_t) NS(RFMultiPole_ksl_value)(
    SIXTRL_BE_ARGPTR_DEC const NS(RFMultiPole) *const SIXTRL_RESTRICT mpole,
    NS(rf_multipole_int_t) const index )
{
    SIXTRL_ASSERT( mpole != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( ( mpole->order + ( NS(rf_multipole_int_t) )1u ) >= index );
    SIXTRL_ASSERT( mpole->bal_addr != ( NS(buffer_addr_t) )0u );

    return NS(RFMultiPole_const_bal)( mpole )[ 2 * index + 1 ] *
        NS(RFMultiPole_calculate_factorial_real)( index );
}

SIXTRL_INLINE NS(rf_multipole_real_t) NS(RFMultiPole_pn_value)(
    SIXTRL_BE_ARGPTR_DEC const NS(RFMultiPole) *const SIXTRL_RESTRICT mpole,
    NS(rf_multipole_int_t) const index )
{
    /* TODO: Implement accessor function for pn: */
    SIXTRL_ASSERT( mpole != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( ( mpole->order + ( NS(rf_multipole_int_t) )1u ) >= index );
    SIXTRL_ASSERT( mpole->phase_addr != ( NS(buffer_addr_t) )0u );

    return NS(RFMultiPole_const_phase)( mpole )[ 2 * index ];
}

SIXTRL_INLINE NS(rf_multipole_real_t) NS(RFMultiPole_ps_value)(
    SIXTRL_BE_ARGPTR_DEC const NS(RFMultiPole) *const SIXTRL_RESTRICT mpole,
    NS(rf_multipole_int_t) const index )
{
    /* TODO: Implement accessor function for pn: */
    SIXTRL_ASSERT( mpole != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( ( mpole->order + ( NS(rf_multipole_int_t) )1u ) >= index );
    SIXTRL_ASSERT( mpole->phase_addr != ( NS(buffer_addr_t) )0u );

    return NS(RFMultiPole_const_phase)( mpole )[ 2 * index + 1 ];
}

/* ------------------------------------------------------------------- */

SIXTRL_INLINE void NS(RFMultiPole_set_voltage)(
    SIXTRL_BE_ARGPTR_DEC NS(RFMultiPole)* SIXTRL_RESTRICT mpole,
    NS(rf_multipole_real_t) const voltage )
{
    if( mpole != SIXTRL_NULLPTR ) mpole->voltage = voltage;
}

SIXTRL_INLINE void NS(RFMultiPole_set_frequency)(
    SIXTRL_BE_ARGPTR_DEC NS(RFMultiPole)* SIXTRL_RESTRICT mpole,
    NS(rf_multipole_real_t) const frequency )
{
    if( mpole != SIXTRL_NULLPTR ) mpole->frequency = frequency;
}

SIXTRL_INLINE void NS(RFMultiPole_set_lag)(
    SIXTRL_BE_ARGPTR_DEC NS(RFMultiPole)* SIXTRL_RESTRICT mpole,
    NS(rf_multipole_real_t) const lag )
{
    if( mpole != SIXTRL_NULLPTR ) mpole->lag = lag;
}

SIXTRL_INLINE void NS(RFMultiPole_set_order )(
    SIXTRL_BE_ARGPTR_DEC NS(RFMultiPole)* SIXTRL_RESTRICT mpole,
    NS(rf_multipole_int_t) const order )
{
    if( mpole != SIXTRL_NULLPTR ) mpole->order = order;
}


SIXTRL_INLINE void NS(RFMultiPole_set_bal_addr)(
    SIXTRL_BE_ARGPTR_DEC NS(RFMultiPole)* SIXTRL_RESTRICT mpole,
    NS(buffer_addr_t) const bal_addr )
{
    if( mpole != SIXTRL_NULLPTR ) mpole->bal_addr = bal_addr;
}

SIXTRL_INLINE void NS(RFMultiPole_set_phase_addr)(
    SIXTRL_BE_ARGPTR_DEC NS(RFMultiPole)* SIXTRL_RESTRICT mpole,
    NS(buffer_addr_t) const phase_addr )
{
    if( mpole != SIXTRL_NULLPTR ) mpole->phase_addr = phase_addr;
}

/* ------------------------------------------------------------------- */

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(RFMultiPole) const*
NS(RFMultiPole_const_from_obj_index)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC const NS(Object) *const obj )
{
    return (
        ( NS(Object_get_type_id)( obj ) == NS(OBJECT_TYPE_RF_MULTIPOLE) ) &&
        ( NS(Object_get_size)( obj ) >= sizeof( NS(RFMultiPole) ) ) )
        ? ( SIXTRL_BE_ARGPTR_DEC NS(RFMultiPole) const* )( uintptr_t
            )NS(Object_get_begin_addr)( obj )
        : SIXTRL_NULLPTR;
}

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(RFMultiPole)*
NS(RFMultiPole_from_obj_index)( SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)* obj )
{
    return ( SIXTRL_BE_ARGPTR_DEC NS(RFMultiPole)*
        )NS(RFMultiPole_const_from_obj_index)( obj );
}

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(RFMultiPole) const*
NS(RFMultiPole_const_from_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT buffer_begin,
    NS(buffer_size_t) const index, NS(buffer_size_t) const slot_size )
{
    return NS(RFMultiPole_const_from_obj_index)(
        NS(ManagedBuffer_get_const_object)( buffer_begin, index, slot_size ) );
}

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(RFMultiPole)*
NS(RFMultiPole_from_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT buffer_begin,
    NS(buffer_size_t) const index, NS(buffer_size_t) const slot_size )
{
    return NS(RFMultiPole_from_obj_index)( NS(ManagedBuffer_get_object)(
        buffer_begin, index, slot_size ) );
}

SIXTRL_INLINE NS(arch_status_t) NS(RFMultiPole_copy)(
    SIXTRL_BE_ARGPTR_DEC NS(RFMultiPole)* SIXTRL_RESTRICT dest,
    SIXTRL_BUFFER_DATAPTR_DEC const NS(RFMultiPole) *const SIXTRL_RESTRICT src )
{
    NS(arch_status_t) status = SIXTRL_ARCH_STATUS_GENERAL_FAILURE;

    if( ( dest != SIXTRL_NULLPTR ) && ( src != SIXTRL_NULLPTR ) &&
        ( dest->order == src->order ) )
    {
        if( dest != src )
        {
            dest->voltage = src->voltage;
            dest->frequency = src->frequency;
            dest->lag = src->lag;

            status = SIXTRL_ARCH_STATUS_SUCCESS;

            if( src->order > ( NS(rf_multipole_int_t) )0u )
            {
                typedef NS(rf_multipole_real_t) real_t;
                typedef SIXTRL_BE_DATAPTR_DEC real_t* dest_ptr_t;
                typedef SIXTRL_BE_DATAPTR_DEC real_t const* src_ptr_t;

                SIXTRL_STATIC_VAR real_t const ZERO = ( real_t )0.0;
                NS(buffer_size_t) const bal_size = ( NS(buffer_size_t)
                    )NS(RFMultiPole_num_bal_elements)( src );

                NS(buffer_size_t) const phase_size = ( NS(buffer_size_t)
                    )NS(RFMultiPole_num_phase_elements)( src );

                dest_ptr_t dest_bal = NS(RFMultiPole_bal)( dest );
                dest_ptr_t dest_p = NS(RFMultiPole_phase)( dest );

                src_ptr_t src_bal = NS(RFMultiPole_const_bal)( src );
                src_ptr_t src_p = NS(RFMultiPole_const_phase)( src );

                if( ( dest_bal != SIXTRL_NULLPTR ) &&
                    ( src_bal != SIXTRL_NULLPTR ) )
                {
                    SIXTRACKLIB_COPY_VALUES(
                        real_t, dest_bal, src_bal, bal_size );
                }
                else if( dest_bal != SIXTRL_NULLPTR )
                {
                    SIXTRACKLIB_SET_VALUES( real_t, dest_bal, bal_size, ZERO );
                }
                else
                {
                    status = SIXTRL_ARCH_STATUS_GENERAL_FAILURE;
                }

                if( ( dest_p != SIXTRL_NULLPTR ) &&
                    ( src_p != SIXTRL_NULLPTR ) )
                {
                    SIXTRACKLIB_COPY_VALUES( real_t, dest_p, src_p, phase_size );
                }
                else if( dest_p != SIXTRL_NULLPTR )
                {
                    SIXTRACKLIB_SET_VALUES( real_t, dest_p, phase_size, ZERO );
                }
                else
                {
                    status = SIXTRL_ARCH_STATUS_GENERAL_FAILURE;
                }
            }
        }
    }

    return status;
}

#if defined( __cplusplus ) && !defined( _GPUCODE )
}
#endif /* defined( __cplusplus ) && !defined( _GPUCODE ) */

#endif /* SIXTRACKL_COMMON_BE_RFMULTIPOLE_BE_RFMULTIPOLE_C99_H__ */
