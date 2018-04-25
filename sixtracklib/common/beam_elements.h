#ifndef SIXTRACKLIB_COMMON_BEAM_ELEMENTS_H__
#define SIXTRACKLIB_COMMON_BEAM_ELEMENTS_H__

#if !defined( _GPUCODE )

#include "sixtracklib/_impl/definitions.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

#include "sixtracklib/common/mem_pool.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#endif /* !defined( _GPUCODE ) */

typedef struct NS(BeamElemInfo)
{
    SIXTRL_UINT64_T mem_offset;
    SIXTRL_UINT64_T type_id;
    SIXTRL_UINT64_T num_bytes;
    SIXTRL_INT64_T  element_id;
}
NS(BeamElemInfo);

#if !defined( _GPUCODE )

typedef struct NS(BeamElements)
{
    NS(BeamElemInfo)* info_begin;
    unsigned char*    data_begin;
        
    NS(MemPool)       info_store;
    NS(MemPool)       data_store;
    
    SIXTRL_SIZE_T     num_elements;
    SIXTRL_SIZE_T     elements_capacity;
    
    SIXTRL_SIZE_T     raw_size;
    SIXTRL_SIZE_T     raw_capacity;
    
    SIXTRL_SIZE_T     begin_alignment;
    SIXTRL_SIZE_T     element_alignment;
}
NS(BeamElements);

NS(BeamElements)* NS(BeamElements_preset)( NS(BeamElements)* beam_elements );

NS(BeamElements)* NS(BeamElements_new)( 
    SIXTRL_SIZE_T const elements_capacity, size_t const   raw_capacity, 
    SIXTRL_SIZE_T const element_alignment, SIXTRL_SIZE_T const begin_alignment );

void NS(BeamElements_free)( NS(BeamElements)* beam_elements );

bool NS(BeamElements_add_drift)( 
    NS(BeamElements)* beam_elements, SIXTRL_REAL_T  const length, 
    SIXTRL_INT64_T const element_id );

bool NS(BeamElements_add_drift_exact)( 
    NS(BeamElements)* beam_elements, SIXTRL_REAL_T const length, 
    SIXTRL_INT64_T const element_id );

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* !defined( _GPUCODE ) */

#endif /* SIXTRACKLIB_COMMON_BEAM_ELEMENTS_H__ */
