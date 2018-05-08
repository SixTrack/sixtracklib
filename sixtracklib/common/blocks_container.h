#ifndef SIXTRACKLIB_COMMON_BLOCKS_CONTAINER_H__
#define SIXTRACKLIB_COMMON_BLOCKS_CONTAINER_H__

#if !defined( _GPUCODE )

#include "sixtracklib/_impl/definitions.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

#include "sixtracklib/common/impl/block_info_impl.h"
#include "sixtracklib/common/mem_pool.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#endif /* !defined( _GPUCODE ) */

#if !defined( _GPUCODE )

/* ------------------------------------------------------------------------- */

void NS(BlocksContainer_clear)(
    NS(BlocksContainer)* SIXTRL_RESTRICT container );

void NS(BlocksContainer_free)(
    NS(BlocksContainer)* SIXTRL_RESTRICT container );

int NS(BlocksContainer_init)(
    NS(BlocksContainer)* SIXTRL_RESTRICT container, 
    NS(block_size_t) const blocks_capacity, 
    NS(block_size_t) const data_capacity );

/* ------------------------------------------------------------------------- */

void NS(BlocksContainer_reserve_num_blocks)(
    NS(BlocksContainer)* SIXTRL_RESTRICT container, 
    NS(block_size_t) const new_block_capacity );

void NS(BlocksContainer_reserve_for_data)(
    NS(BlocksContainer)* SIXTRL_RESTRICT container, 
    NS(block_size_t) const new_data_capacity );

/* ------------------------------------------------------------------------- */

SIXTRL_STATIC NS(MemPool) const* NS(BlocksContainer_get_const_ptr_info_store)(
    NS(BlocksContainer)* SIXTRL_RESTRICT container );

SIXTRL_STATIC NS(MemPool) const* NS(BlocksContainer_get_const_ptr_data_store)(
    NS(BlocksContainer)* SIXTRL_RESTRICT container );

SIXTRL_STATIC NS(MemPool)* NS(BlocksContainer_get_ptr_info_store)(
    NS(BlocksContainer)* SIXTRL_RESTRICT container );

SIXTRL_STATIC NS(MemPool)* NS(BlocksContainer_get_ptr_data_store)(
    NS(BlocksContainer)* SIXTRL_RESTRICT container );
    
/* ========================================================================= */
/* ======             Implementation of inline functions            ======== */
/* ========================================================================= */

SIXTRL_INLINE NS(MemPool) const* NS(BlocksContainer_get_const_ptr_info_store)(
    NS(BlocksContainer)* SIXTRL_RESTRICT container )
{
    return ( container != 0 ) 
        ? ( NS(MemPool) const* )container->ptr_info_store : 0;
}

SIXTRL_INLINE NS(MemPool) const* NS(BlocksContainer_get_const_ptr_data_store)(
    NS(BlocksContainer)* SIXTRL_RESTRICT container )
{
    return ( container != 0 ) 
        ? ( NS(MemPool) const* )container->ptr_data_store : 0;
}

SIXTRL_INLINE NS(MemPool)* NS(BlocksContainer_get_ptr_info_store)(
    NS(BlocksContainer)* SIXTRL_RESTRICT container )
{
    typedef NS(MemPool)* pool_t;    
    return ( pool_t )NS(BlocksContainer_get_const_ptr_info_store)( container );
}

SIXTRL_INLINE NS(MemPool)* NS(BlocksContainer_get_ptr_data_store)(
    NS(BlocksContainer)* SIXTRL_RESTRICT container )
{
    typedef NS(MemPool)* pool_t;    
    return ( pool_t )NS(BlocksContainer_get_const_ptr_data_store)( container );
}

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* !defined( _GPUCODE ) */

#endif /* SIXTRACKLIB_COMMON_BLOCKS_CONTAINER_H__ */



