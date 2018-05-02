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

typedef struct NS(BlocksContainer)
{
    NS(BlockInfo)*          info_begin;
    unsigned char*          data_begin;
        
    NS(MemPool)             info_store;
    NS(MemPool)             data_store;
    
    NS(block_size_t)        num_blocks;
    NS(block_size_t)        blocks_capacity;
    
    NS(block_size_t)        data_raw_size;
    NS(block_size_t)        data_raw_capacity;
    
    NS(block_alignment_t)   info_begin_alignment;
    NS(block_alignment_t)   info_alignment;
    
    NS(block_alignment_t)   data_begin_alignment;
    NS(block_alignment_t)   data_alignment;
}
NS(BlocksContainer);

/* ------------------------------------------------------------------------- */

NS(BlocksContainer)* NS(BlocksContainer_preset)( 
    NS(BlocksContainer)* SIXTRL_RESTRICT container );

void NS(BlocksContainer_clear)(
    NS(BlocksContainer)* SIXTRL_RESTRICT container );

void NS(BlocksContainer_free)(
    NS(BlocksContainer)* SIXTRL_RESTRICT container );

int NS(BlocksContainer_init)(
    NS(BlocksContainer)* SIXTRL_RESTRICT container, 
    NS(block_size_t) const blocks_capacity, 
    NS(block_size_t) const data_capacity );

/* ------------------------------------------------------------------------- */

int NS(BlocksContainer_set_info_begin_alignment)(
    NS(BlocksContainer)* SIXTRL_RESTRICT container, 
    NS(block_alignment_t) const begin_alignment );

int NS(BlocksContainer_set_data_begin_alignment)(
    NS(BlocksContainer)* SIXTRL_RESTRICT container, 
    NS(block_alignment_t) const begin_alignment );

int NS(BlocksContainer_set_data_alignment)(
    NS(BlocksContainer)* SIXTRL_RESTRICT container, 
    NS(block_alignment_t) const alignment );

int NS(BlocksContainer_set_info_alignment )(
    NS(BlocksContainer)* SIXTRL_RESTRICT container, 
    NS(block_alignment_t) const alignment );

void NS(BlocksContainer_reserve_num_blocks)(
    NS(BlocksContainer)* SIXTRL_RESTRICT container, 
    NS(block_size_t) const new_block_capacity );

void NS(BlocksContainer_reserve_for_data)(
    NS(BlocksContainer)* SIXTRL_RESTRICT container, 
    NS(block_size_t) const new_data_capacity );

/* ------------------------------------------------------------------------- */

NS(block_alignment_t) NS(BlocksContainer_get_info_alignment)(
    const NS(BlocksContainer) *const SIXTRL_RESTRICT container );

NS(block_alignment_t) NS(BlocksContainer_get_data_alignment)(
    const NS(BlocksContainer) *const SIXTRL_RESTRICT container );

NS(block_alignment_t) NS(BlocksContainer_get_info_begin_alignment)(
    const NS(BlocksContainer) *const SIXTRL_RESTRICT container );

NS(block_alignment_t) NS(BlocksContainer_get_data_begin_alignment)(
    const NS(BlocksContainer) *const SIXTRL_RESTRICT container );

NS(block_size_t) NS(BlocksContainer_get_data_capacity)(
    const NS(BlocksContainer) *const SIXTRL_RESTRICT container );

NS(block_size_t) NS(BlocksContainer_get_data_size)(
    const NS(BlocksContainer) *const SIXTRL_RESTRICT container );

NS(block_size_t) NS(BlocksContainer_get_block_capacity)(
    const NS(BlocksContainer) *const SIXTRL_RESTRICT container );

NS(block_size_t) NS(BlocksContainer_get_num_of_blocks)(
    const NS(BlocksContainer) *const SIXTRL_RESTRICT container );

/* ------------------------------------------------------------------------- */

unsigned char const* NS(BlocksContainer_get_const_ptr_data_begin)(
    const NS(BlocksContainer) *const SIXTRL_RESTRICT container );

unsigned char* NS(BlocksContainer_get_ptr_data_begin)(
    NS(BlocksContainer)* SIXTRL_RESTRICT container );

NS(BlockInfo) const* NS(BlocksContainer_get_const_block_infos_begin)(
    const NS(BlocksContainer) *const SIXTRL_RESTRICT container );

NS(BlockInfo) const* NS(BlocksContainer_get_const_block_infos_end)(
    const NS(BlocksContainer) *const SIXTRL_RESTRICT container );


NS(BlockInfo)* NS(BlocksContainer_get_block_infos_begin)(
    NS(BlocksContainer)* SIXTRL_RESTRICT container );

NS(BlockInfo)* NS(BlocksContainer_get_infos_end)(
    NS(BlocksContainer)* SIXTRL_RESTRICT container );

NS(BlockInfo) NS(BlocksContainer_get_block_info_by_index)(
    const NS(BlocksContainer) *const SIXTRL_RESTRICT container, 
    NS(block_size_t) const block_index );

NS(BlockInfo) const* NS(BlocksContainer_get_const_ptr_to_block_info_by_index)(
    const NS(BlocksContainer) *const SIXTRL_RESTRICT container, 
    NS(block_size_t) const block_index );

NS(BlockInfo)* NS(BlocksContainer_get_ptr_to_block_info_by_index)(
    NS(BlocksContainer)* SIXTRL_RESTRICT container, 
    NS(block_size_t) const block_index );

/* ------------------------------------------------------------------------- */

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* !defined( _GPUCODE ) */

#endif /* SIXTRACKLIB_COMMON_BLOCKS_CONTAINER_H__ */



