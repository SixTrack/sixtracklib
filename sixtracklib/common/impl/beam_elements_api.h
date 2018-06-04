#ifndef SIXTRACKLIB_COMMON_IMPL_BEAM_ELEMENTS_API_H__
#define SIXTRACKLIB_COMMON_IMPL_BEAM_ELEMENTS_API_H__

#if !defined( _GPUCODE )

#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "sixtracklib/_impl/definitions.h"
#include "sixtracklib/common/beam_elements.h"
#include "sixtracklib/common/blocks.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#endif /* !defined( _GPUCODE ) */

/* ------------------------------------------------------------------------- */

struct NS(Drift);

SIXTRL_STATIC SIXTRL_GLOBAL_DEC NS(Drift)* NS(Drift_preset)(
    SIXTRL_GLOBAL_DEC NS(Drift)* drift );

SIXTRL_STATIC NS(BlockType) NS(Drift_get_type_id)(
    const NS(Drift) *const SIXTRL_RESTRICT drift );

SIXTRL_STATIC NS(block_type_num_t) NS(Drift_get_type_id_num)(
    const NS(Drift) *const SIXTRL_RESTRICT drift );

SIXTRL_STATIC SIXTRL_REAL_T NS(Drift_get_length)(
    const NS(Drift) *const SIXTRL_RESTRICT drift );

SIXTRL_STATIC void NS(Drift_set_length)(
    NS(Drift)* SIXTRL_RESTRICT drift, SIXTRL_REAL_T const length );
    
SIXTRL_STATIC SIXTRL_GLOBAL_DEC NS(Drift) const* 
NS(Blocks_get_const_drift)( 
    const NS(BlockInfo) *const SIXTRL_RESTRICT block_info );

SIXTRL_STATIC SIXTRL_GLOBAL_DEC NS(Drift)* 
NS(Blocks_get_drift)( NS(BlockInfo)* SIXTRL_RESTRICT block_info );

/* ------------------------------------------------------------------------- */

struct NS(DriftExact);

SIXTRL_STATIC SIXTRL_GLOBAL_DEC NS(DriftExact)* NS(DriftExact_preset)(
    SIXTRL_GLOBAL_DEC NS(DriftExact)* drift_exact );

SIXTRL_STATIC NS(BlockType) NS(DriftExact_get_type_id)(
    const NS(DriftExact) *const SIXTRL_RESTRICT drift_exact );

SIXTRL_STATIC NS(block_type_num_t) NS(DriftExact_get_type_id_num)(
    const NS(DriftExact) *const SIXTRL_RESTRICT drift_exact );

SIXTRL_STATIC SIXTRL_REAL_T NS(DriftExact_get_length)(
    const NS(DriftExact) *const SIXTRL_RESTRICT drift_exact );

SIXTRL_STATIC void NS(DriftExact_set_length)(
    NS(DriftExact)* SIXTRL_RESTRICT drift_exact, SIXTRL_REAL_T const length );
    
SIXTRL_STATIC SIXTRL_GLOBAL_DEC NS(DriftExact) const* 
NS(Blocks_get_const_drift_exact)( 
    const NS(BlockInfo) *const SIXTRL_RESTRICT block_info );

SIXTRL_STATIC SIXTRL_GLOBAL_DEC NS(DriftExact)* 
NS(Blocks_get_drift_exact)( NS(BlockInfo)* SIXTRL_RESTRICT block_info );

/* ------------------------------------------------------------------------- */
    
struct NS(MultiPole);

SIXTRL_STATIC SIXTRL_GLOBAL_DEC NS(MultiPole)* NS(MultiPole_preset)(
    SIXTRL_GLOBAL_DEC NS(MultiPole)* multipole );

SIXTRL_STATIC NS(BlockType) NS(MultiPole_get_type_id)(
    const NS(MultiPole) *const SIXTRL_RESTRICT multipole );

SIXTRL_STATIC NS(block_type_num_t) NS(MultiPole_get_type_id_num)(
    const NS(MultiPole) *const SIXTRL_RESTRICT multipole );

SIXTRL_STATIC SIXTRL_REAL_T NS(MultiPole_get_length)(
    const NS(MultiPole) *const SIXTRL_RESTRICT multipole );

SIXTRL_STATIC SIXTRL_REAL_T NS(MultiPole_get_hxl)(
    const NS(MultiPole) *const SIXTRL_RESTRICT multipole );

SIXTRL_STATIC SIXTRL_REAL_T NS(MultiPole_get_hyl)(
    const NS(MultiPole) *const SIXTRL_RESTRICT multipole );

SIXTRL_STATIC SIXTRL_INT64_T NS(MultiPole_get_order)(
    const NS(MultiPole) *const SIXTRL_RESTRICT multipole );

SIXTRL_STATIC SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const* 
NS(MultiPole_get_const_bal)( const NS(MultiPole) *const SIXTRL_RESTRICT mp );

SIXTRL_STATIC SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* 
NS(MultiPole_get_bal)( NS(MultiPole)* SIXTRL_RESTRICT mp );

SIXTRL_STATIC SIXTRL_REAL_T NS(MultiPole_get_bal_value)(
    const NS(MultiPole) *const SIXTRL_RESTRICT mp, 
    SIXTRL_INT64_T const index );

SIXTRL_STATIC void NS(MultiPole_set_length)(
    NS(MultiPole)* SIXTRL_RESTRICT multipole, SIXTRL_REAL_T const length );

SIXTRL_STATIC void NS(MultiPole_set_hxl)(
    NS(MultiPole)* SIXTRL_RESTRICT multipole, SIXTRL_REAL_T const hxl );

SIXTRL_STATIC void NS(MultiPole_set_hyl)(
    NS(MultiPole)* SIXTRL_RESTRICT multipole, SIXTRL_REAL_T const hyl );

SIXTRL_STATIC void NS(MultiPole_set_order)(
    NS(MultiPole)* SIXTRL_RESTRICT multipole, SIXTRL_INT64_T const order );

SIXTRL_STATIC void NS(MultiPole_set_bal_value)(
    NS(MultiPole)* SIXTRL_RESTRICT multipole, 
    SIXTRL_INT64_T const index, SIXTRL_REAL_T const value );

SIXTRL_STATIC void NS(MultiPole_set_bal)(
    NS(MultiPole)* SIXTRL_RESTRICT multipole, 
    SIXTRL_INT64_T const order, 
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT bal );

SIXTRL_STATIC void NS(MultiPole_assign_ptr_to_bal)(
    NS(MultiPole)* SIXTRL_RESTRICT multipole, 
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT ptr_to_bal );

SIXTRL_STATIC SIXTRL_GLOBAL_DEC NS(MultiPole) const* 
NS(Blocks_get_const_multipole)( 
    const NS(BlockInfo) *const SIXTRL_RESTRICT block_info );

SIXTRL_STATIC SIXTRL_GLOBAL_DEC NS(MultiPole)*
NS(Blocks_get_multipole)( NS(BlockInfo)* SIXTRL_RESTRICT block_info );

/* ========================================================================= */
/* ======             Implementation of inline functions            ======== */
/* ========================================================================= */

#include "sixtracklib/common/beam_elements.h"

SIXTRL_INLINE SIXTRL_GLOBAL_DEC NS(Drift)* NS(Drift_preset)(
    SIXTRL_GLOBAL_DEC NS(Drift)* drift )
{
    NS(Drift_set_length)( drift, ( SIXTRL_REAL_T )0.0 );
    return drift;
}

SIXTRL_INLINE NS(BlockType) NS(Drift_get_type_id)(
    const NS(Drift) *const SIXTRL_RESTRICT drift )
{
    return ( drift != 0 ) ? NS(BLOCK_TYPE_DRIFT) : NS(BLOCK_TYPE_INVALID);
}

SIXTRL_INLINE NS(block_type_num_t) NS(Drift_get_type_id_num)(
    const NS(Drift) *const SIXTRL_RESTRICT drift )
{
    return NS(BlockType_to_number)( NS(Drift_get_type_id)( drift ) );
        
}

SIXTRL_INLINE SIXTRL_REAL_T NS(Drift_get_length)(
    const NS(Drift) *const SIXTRL_RESTRICT drift )
{
    return ( drift ) ? drift->length : ( SIXTRL_REAL_T )0.0;
}

SIXTRL_INLINE void NS(Drift_set_length)(
    NS(Drift)* SIXTRL_RESTRICT drift, SIXTRL_REAL_T const length )
{
    if( ( drift != 0 ) && ( length >= ( SIXTRL_REAL_T )0.0 ) )
    {
        drift->length = length;
    }
    
    return;
}
    
SIXTRL_INLINE SIXTRL_GLOBAL_DEC NS(Drift) const* 
NS(Blocks_get_const_drift)( 
    const NS(BlockInfo) *const SIXTRL_RESTRICT block_info )
{
    return ( ( block_info != 0 ) && 
        ( NS(BlockInfo_get_type_id)( block_info ) == NS(BLOCK_TYPE_DRIFT ) ) )
            ? ( SIXTRL_GLOBAL_DEC NS(Drift) const* 
                )NS(BlockInfo_get_const_ptr_begin)( block_info ) 
            : 0;
}

SIXTRL_INLINE SIXTRL_GLOBAL_DEC NS(Drift)* 
NS(Blocks_get_drift)( NS(BlockInfo)* SIXTRL_RESTRICT block_info )
{
    return ( SIXTRL_GLOBAL_DEC NS(Drift)* 
        )NS(Blocks_get_const_drift)( block_info );
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE SIXTRL_GLOBAL_DEC NS(DriftExact)* NS(DriftExact_preset)(
    SIXTRL_GLOBAL_DEC NS(DriftExact)* drift )
{
    NS(DriftExact_set_length)( drift, ( SIXTRL_REAL_T )0.0 );
    return drift;
}

SIXTRL_INLINE NS(BlockType) NS(DriftExact_get_type_id)(
    const NS(DriftExact) *const SIXTRL_RESTRICT drift )
{
    return ( drift != 0 ) ? NS(BLOCK_TYPE_DRIFT_EXACT) : NS(BLOCK_TYPE_INVALID);
}

SIXTRL_INLINE NS(block_type_num_t) NS(DriftExact_get_type_id_num)(
    const NS(DriftExact) *const SIXTRL_RESTRICT drift )
{
    return NS(BlockType_to_number)( NS(DriftExact_get_type_id)( drift ) );
        
}

SIXTRL_INLINE SIXTRL_REAL_T NS(DriftExact_get_length)(
    const NS(DriftExact) *const SIXTRL_RESTRICT drift )
{
    return ( drift ) ? drift->length : ( SIXTRL_REAL_T )0.0;
}

SIXTRL_INLINE void NS(DriftExact_set_length)(
    NS(DriftExact)* SIXTRL_RESTRICT drift, SIXTRL_REAL_T const length )
{
    if( ( drift != 0 ) && ( length >= ( SIXTRL_REAL_T )0.0 ) )
    {
        drift->length = length;
    }
    
    return;
}
    
SIXTRL_INLINE SIXTRL_GLOBAL_DEC NS(DriftExact) const* 
NS(Blocks_get_const_drift_exact)( 
    const NS(BlockInfo) *const SIXTRL_RESTRICT block_info )
{
    return ( ( block_info != 0 ) && 
        ( NS(BlockInfo_get_type_id)( block_info ) == 
            NS(BLOCK_TYPE_DRIFT_EXACT ) ) )
            ? ( SIXTRL_GLOBAL_DEC NS(DriftExact) const* 
                )NS(BlockInfo_get_const_ptr_begin)( block_info )
            : 0;
}

SIXTRL_INLINE SIXTRL_GLOBAL_DEC NS(DriftExact)* 
NS(Blocks_get_drift_exact)( NS(BlockInfo)* SIXTRL_RESTRICT block_info )
{
    return ( SIXTRL_GLOBAL_DEC NS(DriftExact)* 
        )NS(Blocks_get_const_drift_exact)( block_info );
}

/* ------------------------------------------------------------------------- */
    
SIXTRL_INLINE SIXTRL_GLOBAL_DEC NS(MultiPole)* NS(MultiPole_preset)(
    SIXTRL_GLOBAL_DEC NS(MultiPole)* multipole )
{
    NS(MultiPole_set_order)(  multipole, ( SIXTRL_INT64_T )0 );
    NS(MultiPole_set_hxl)(    multipole, ( SIXTRL_REAL_T )0.0 );
    NS(MultiPole_set_hyl)(    multipole, ( SIXTRL_REAL_T )0.0 );
    NS(MultiPole_set_length)( multipole, ( SIXTRL_REAL_T )0.0 );
    NS(MultiPole_assign_ptr_to_bal)( multipole, 0 );
    
    return multipole;
}

SIXTRL_INLINE NS(BlockType) NS(MultiPole_get_type_id)(
    const NS(MultiPole) *const SIXTRL_RESTRICT multipole )
{
    return NS(BLOCK_TYPE_MULTIPOLE);
}

SIXTRL_INLINE NS(block_type_num_t) NS(MultiPole_get_type_id_num)(
    const NS(MultiPole) *const SIXTRL_RESTRICT multipole )
{
    return NS(BlockType_to_number)( NS(BLOCK_TYPE_MULTIPOLE) );
}

SIXTRL_INLINE SIXTRL_REAL_T NS(MultiPole_get_length)(
    const NS(MultiPole) *const SIXTRL_RESTRICT multipole )
{
    SIXTRL_ASSERT(  ( multipole != 0 ) && 
                    ( multipole->length >= ( SIXTRL_REAL_T )0.0 ) );
    
    return multipole->length;
}

SIXTRL_INLINE SIXTRL_REAL_T NS(MultiPole_get_hxl)(
    const NS(MultiPole) *const SIXTRL_RESTRICT multipole )
{
    SIXTRL_ASSERT(  ( multipole != 0 ) && 
                    ( multipole->hyl > ( SIXTRL_REAL_T )0.0 ) );
    
    return multipole->hxl;
}

SIXTRL_INLINE SIXTRL_REAL_T NS(MultiPole_get_hyl)(
    const NS(MultiPole) *const SIXTRL_RESTRICT multipole )
{
    SIXTRL_ASSERT(  ( multipole != 0 ) && 
                    ( multipole->hyl > ( SIXTRL_REAL_T )0.0 ) );
    
    return multipole->hyl;
}

SIXTRL_INLINE SIXTRL_INT64_T NS(MultiPole_get_order)(
    const NS(MultiPole) *const SIXTRL_RESTRICT multipole )
{
    SIXTRL_ASSERT( ( multipole != 0 ) && ( multipole->order >= 0 ) );
    return multipole->order;
}

SIXTRL_INLINE SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const* 
NS(MultiPole_get_const_bal)( const NS(MultiPole) *const SIXTRL_RESTRICT mp )
{
    SIXTRL_ASSERT( mp != 0 );
    return mp->bal;
}

SIXTRL_INLINE SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* 
NS(MultiPole_get_bal)( NS(MultiPole)* SIXTRL_RESTRICT mp )
{
    return ( SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* )NS(MultiPole_get_bal)( mp );
}

SIXTRL_INLINE SIXTRL_REAL_T NS(MultiPole_get_bal_value)(
    const NS(MultiPole) *const SIXTRL_RESTRICT mp, SIXTRL_INT64_T const index )
{
    SIXTRL_ASSERT( ( mp != 0 ) && ( index >= 0 ) && 
                   ( index < ( 2 * mp->order  ) ) );
    
    return ( mp->bal != 0 ) ? mp->bal[ index ] : ( SIXTRL_REAL_T )0.0;
}

SIXTRL_INLINE void NS(MultiPole_set_length)(
    NS(MultiPole)* SIXTRL_RESTRICT multipole, SIXTRL_REAL_T const length )
{
    SIXTRL_ASSERT( multipole != 0 );
    multipole->length = length;
    
    return;
}

SIXTRL_INLINE void NS(MultiPole_set_hxl)(
    NS(MultiPole)* SIXTRL_RESTRICT multipole, SIXTRL_REAL_T const hxl )
{
    SIXTRL_ASSERT( multipole != 0 );
    multipole->hxl = hxl;
    
    return;
}

SIXTRL_INLINE void NS(MultiPole_set_hyl)(
    NS(MultiPole)* SIXTRL_RESTRICT multipole, SIXTRL_REAL_T const hyl )
{
    SIXTRL_ASSERT( multipole != 0 );
    multipole->hyl = hyl;
    
    return;
}

SIXTRL_INLINE void NS(MultiPole_set_order)(
    NS(MultiPole)* SIXTRL_RESTRICT multipole, SIXTRL_INT64_T const order )
{
    SIXTRL_ASSERT( ( multipole != 0 ) && ( order >= 0 ) );
    multipole->order = order;
    
    return;
}

SIXTRL_INLINE void NS(MultiPole_set_bal_value)(
    NS(MultiPole)* SIXTRL_RESTRICT multipole, 
    SIXTRL_INT64_T const index, SIXTRL_REAL_T const value )
{
    SIXTRL_ASSERT( ( multipole != 0 ) && ( index >= 0 ) &&
                   ( index < ( 2 * multipole->order ) ) );
    
    multipole->bal[ index ] = value;
    return;
}

SIXTRL_INLINE void NS(MultiPole_set_bal)(
    NS(MultiPole)* SIXTRL_RESTRICT multipole, SIXTRL_INT64_T const order, 
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT bal )
{
    SIXTRL_ASSERT( ( multipole != 0 ) && ( order > 0 ) && ( bal != 0 ) );    
    SIXTRACKLIB_COPY_VALUES( SIXTRL_REAL_T, multipole->bal, bal, 2 * order );
    
    return;
}

SIXTRL_INLINE void NS(MultiPole_assign_ptr_to_bal)(
    NS(MultiPole)* SIXTRL_RESTRICT multipole, 
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT ptr_to_bal )
{
    SIXTRL_ASSERT( multipole != 0 );
    multipole->bal = ptr_to_bal;
    
    return;
}

SIXTRL_INLINE SIXTRL_GLOBAL_DEC NS(MultiPole) const* 
NS(Blocks_get_const_multipole)(
    const NS(BlockInfo) *const SIXTRL_RESTRICT block_info )
{
    NS(BlockType) const type_id = 
        NS(BlockInfo_get_type_id)( block_info );
    
    SIXTRL_GLOBAL_DEC void const* ptr_begin =
        NS(BlockInfo_get_const_ptr_begin)( block_info );
    
    return ( type_id == NS(BLOCK_TYPE_MULTIPOLE ) && ( ptr_begin != 0 ) )
        ? ( NS(MultiPole) const* )ptr_begin : 0;
}

SIXTRL_INLINE SIXTRL_GLOBAL_DEC NS(MultiPole)*
NS(Blocks_get_multipole)( NS(BlockInfo)* SIXTRL_RESTRICT block_info )
{
    return ( NS(MultiPole)* )NS(Blocks_get_const_multipole)( block_info );
}

#if !defined( _GPUCODE )
#ifdef __cplusplus
}
#endif /* __cplusplus */
#endif /* !defined( _GPUCODE ) */

#endif /* SIXTRACKLIB_COMMON_IMPL_BEAM_ELEMENTS_API_H__ */

/* end: sixtracklib/common/impl/beam_elements_api.h  */
