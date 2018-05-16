#ifndef SIXTRACKLIB_COMMON_BLOCK_INFO_H__
#define SIXTRACKLIB_COMMON_BLOCK_INFO_H__

#if !defined( _GPUCODE )

#include "sixtracklib/_impl/definitions.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "sixtracklib/common/impl/block_info_impl.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */
    
int NS(Block_write_to_binary_file)( FILE* fp,
    NS(BlockType) const type_id,
    NS(block_num_elements_t) const num_elements,
    NS(block_size_t) const num_attributes,
    void** SIXTRL_RESTRICT ptr_attr_begins,
    const NS(block_size_t) *const SIXTRL_RESTRICT attr_sizes, 
    const NS(block_size_t) *const SIXTRL_RESTRICT attr_counts );


int NS(Block_peak_at_next_block_in_binary_file)( FILE* fp, 
    SIXTRL_UINT64_T* SIXTRL_RESTRICT binary_length,
    SIXTRL_INT64_T*  SIXTRL_RESTRICT success_flag,
    NS(BlockType)*   SIXTRL_RESTRICT type_id,
    NS(block_num_elements_t)* SIXTRL_RESTRICT num_elements,
    NS(block_size_t)* SIXTRL_RESTRICT num_attributes, 
    NS(block_size_t)* SIXTRL_RESTRICT attr_sizes, 
    NS(block_size_t)* SIXTRL_RESTRICT attr_counts, 
    NS(block_size_t) const max_num_attributes 
);

int NS(Block_read_structure_from_binary_file)( FILE* fp,
    SIXTRL_UINT64_T* SIXTRL_RESTRICT binary_length,
    SIXTRL_INT64_T*  SIXTRL_RESTRICT success_flag,    
    NS(BlockType)* SIXTRL_RESTRICT type_id,
    NS(block_num_elements_t)* SIXTRL_RESTRICT num_elements,
    NS(block_size_t)* SIXTRL_RESTRICT num_attributes,
    void** SIXTRL_RESTRICT ptr_attr_begins,
    NS(block_size_t)* SIXTRL_RESTRICT attr_sizes,
    NS(block_size_t)* SIXTRL_RESTRICT attr_counts 
);   


#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* !defined( _GPUCODE ) */

    
#endif /* SIXTRACKLIB_COMMON_BLOCK_INFO_H__ */

/* end: sixtracklib/common/block_info.h */
