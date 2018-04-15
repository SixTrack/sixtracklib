#ifndef SIXTRACKLIB_COMMON_IMPL_BLOCK_TYPE_H__
#define SIXTRACKLIB_COMMON_IMPL_BLOCK_TYPE_H__

#if !defined( _GPUCODE )

#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "sixtracklib/_impl/definitions.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#endif /* !defined( _GPUCODE ) */

/* ========================================================================= */
    
typedef enum NS(BeamElementType)
{
    NS(ELEMENT_TYPE_NONE)        = 0,
    NS(ELEMENT_TYPE_DRIFT)       = 2,
    NS(ELEMENT_TYPE_DRIFT_EXACT) = 3,
    NS(ELEMENT_TYPE_MULTIPOLE)   = 4,
    NS(ELEMENT_TYPE_CAVITY)      = 5,
    NS(ELEMENT_TYPE_ALIGN)       = 6,
    NS(ELEMENT_TYPE_USERDEFINED) = 65535
}
NS(BeamElementType);
    
/* ========================================================================= */

typedef struct NS(BeamElementInfo)
{
    SIXTRL_INT64_T      element_id;
    NS(BeamElementType) type;
    void const*         ptr_element;
}
NS(BeamElementInfo);

static struct NS(BeamElementInfo)* NS(BeamElementInfo_preset)( 
    struct NS(BeamElementInfo)* SIXTRL_RESTRICT info );

static enum NS(BeamElementType) NS(BeamElementInfo_get_type)( 
    const struct NS(BeamElementInfo) * const SIXTRL_RESTRICT info );

static SIXTRL_INT64_T NS(BeamElementInfo_get_element_id)(
    const struct NS(BeamElementInfo) * const SIXTRL_RESTRICT info );

static bool NS(BeamElementInfo_is_available)(
    const struct NS(BeamElementInfo) * const SIXTRL_RESTRICT info );

static void const* NS(BeamElementInfo_get_const_pointer)(
    const struct NS(BeamElementInfo) * const SIXTRL_RESTRICT info );

static void* NS(BeamElementInfo_get_pointer)(
    struct NS(BeamElementInfo)* SIXTRL_RESTRICT info );

static SIXTRL_INT64_T const NS(PARTICLES_INVALID_BEAM_ELEMENT_ID) = INT64_C( -1 );

/* ========================================================================= */

struct NS(MemPool);

typedef struct NS(Block)
{
    SIXTRL_UINT64_T size;
    SIXTRL_UINT64_T capacity;
    
    SIXTRL_UINT64_T flags;
    SIXTRL_INT64_T  next_element_id;
    SIXTRL_SIZE_T   alignment;
    
    NS(BeamElementInfo)* elem_info;
    void* ptr_mem_context;
    void* ptr_mem_begin;
}
NS(Block);

/* -------------------------------------------------------------------------- */

static SIXTRL_UINT64_T const NS( BLOCK_FLAGS_NONE ) = ( SIXTRL_UINT64_T )0x0000;
static SIXTRL_UINT64_T const NS( BLOCK_FLAGS_PACKED ) = ( SIXTRL_UINT64_T )0x0001;
static SIXTRL_UINT64_T const NS( BLOCK_FLAGS_OWNS_MEMORY ) = ( SIXTRL_UINT64_T )0x0002;

static SIXTRL_UINT64_T const
    NS( BLOCK_FLAGS_MEM_CTX_MEMPOOL ) = ( SIXTRL_UINT64_T )0x0010;

static SIXTRL_UINT64_T const 
    NS( BLOCK_FLAGS_MEM_CTX_FLAT_MEMORY ) = ( SIXTRL_UINT64_T )0x0020;

static SIXTRL_UINT64_T const NS( BLOCK_FLAGS_ALIGN_MASK ) = ( SIXTRL_UINT64_T )0xFFFF00;

static SIXTRL_UINT64_T const NS( BLOCK_MAX_ALIGNMENT ) = ( SIXTRL_UINT64_T )0xFFFF;

static SIXTRL_UINT64_T const
    NS( BLOCK_FLAGS_ALIGN_MASK_OFFSET_BITS ) = ( SIXTRL_UINT64_T )8u;

static SIXTRL_SIZE_T const NS( BLOCK_DEFAULT_CAPACITY ) = ( SIXTRL_SIZE_T )1024u;
static SIXTRL_SIZE_T const NS( BLOCK_DEFAULT_ELEMENT_CAPACITY ) = (SIXTRL_SIZE_T)64u;
static SIXTRL_SIZE_T const NS( BLOCK_DEFAULT_MEMPOOL_CHUNK_SIZE ) = (SIXTRL_SIZE_T)8u;
static SIXTRL_SIZE_T const NS( BLOCK_DEFAULT_MEMPOOL_ALIGNMENT ) = (SIXTRL_SIZE_T)16u;

/* ------------------------------------------------------------------------- */

static NS(Block)* NS(Block_preset)( NS(Block)* SIXTRL_RESTRICT block );

static SIXTRL_SIZE_T NS(Block_get_capacity)(
    const struct NS(Block) *const SIXTRL_RESTRICT block );

static SIXTRL_SIZE_T NS(Block_get_size)(
    const struct NS(Block) *const SIXTRL_RESTRICT block );

static SIXTRL_UINT64_T NS(Block_get_flags)( 
    const struct NS(Block) *const SIXTRL_RESTRICT block );

static struct NS(BeamElementInfo)* NS(Block_get_elements_begin)(
    struct NS(Block)* SIXTRL_RESTRICT block );

static struct NS(BeamElementInfo)* NS(Block_get_elements_end)(
    struct NS(Block)* SIXTRL_RESTRICT block );

static struct NS(BeamElementInfo) const* NS(Block_get_const_elements_begin)(
    const struct NS(Block) *const SIXTRL_RESTRICT block );

static struct NS(BeamElementInfo) const* NS(Block_get_const_elements_end)(
    const struct NS(Block) *const SIXTRL_RESTRICT block );


/* ************************************************************************* */
/* *****               Implementation of inline functions              ***** */
/* ************************************************************************* */

SIXTRL_INLINE NS(BeamElementInfo)* NS(BeamElementInfo_preset)( 
    NS(BeamElementInfo)* SIXTRL_RESTRICT info )
{
    if( info != 0 )
    {
        info->element_id = NS(PARTICLES_INVALID_BEAM_ELEMENT_ID);
        info->type   = NS(ELEMENT_TYPE_NONE);
        info->ptr_element = 0;
    }
    
    return info;
}

SIXTRL_INLINE NS(BeamElementType) NS(BeamElementInfo_get_type)( 
    const NS(BeamElementInfo) * const SIXTRL_RESTRICT info )
{
    return ( info != 0 ) ? info->type : NS(ELEMENT_TYPE_NONE);
}

SIXTRL_INLINE SIXTRL_INT64_T NS(BeamElementInfo_get_element_id)(
    const NS(BeamElementInfo) * const SIXTRL_RESTRICT info )
{
    return ( info != 0 ) ? info->element_id : NS(PARTICLES_INVALID_BEAM_ELEMENT_ID);
}

SIXTRL_INLINE bool NS(BeamElementInfo_is_available)(
    const NS(BeamElementInfo) * const SIXTRL_RESTRICT info )
{
    return ( ( info != 0 ) && ( info->ptr_element != 0 ) &&
             ( info->element_id != NS(PARTICLES_INVALID_BEAM_ELEMENT_ID) ) );
}

SIXTRL_INLINE void const* NS(BeamElementInfo_get_const_pointer)(
    const NS(BeamElementInfo) * const SIXTRL_RESTRICT info )
{
    return ( info != 0 ) ? info->ptr_element : 0;
}

SIXTRL_INLINE void* NS(BeamElementInfo_get_pointer)(
    NS(BeamElementInfo)* SIXTRL_RESTRICT info )
{
    return ( void* )NS(BeamElementInfo_get_const_pointer)( info );
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE NS(Block)* NS(Block_preset)( NS(Block)* SIXTRL_RESTRICT block )
{
    if( block != 0 )
    {
        block->size            = ( SIXTRL_SIZE_T )0u;
        block->capacity        = ( SIXTRL_SIZE_T )0u;        
        block->flags           = NS(BLOCK_FLAGS_NONE);
        block->next_element_id     = ( SIXTRL_INT64_T )0;    
        block->alignment       = NS(BLOCK_DEFAULT_MEMPOOL_ALIGNMENT);
        block->elem_info       = 0;
        block->ptr_mem_context = 0;
        block->ptr_mem_begin   = 0;
    }
    
    return block;
}

SIXTRL_INLINE SIXTRL_SIZE_T NS(Block_get_capacity)(
    const NS(Block) *const SIXTRL_RESTRICT block )
{
    return ( block ) ? block->capacity : ( SIXTRL_SIZE_T )0u;
}

SIXTRL_INLINE SIXTRL_SIZE_T NS(Block_get_size)( 
    const NS(Block) *const SIXTRL_RESTRICT block )
{
    return ( block ) ? block->size : ( SIXTRL_SIZE_T )0u;
}

SIXTRL_INLINE SIXTRL_UINT64_T NS(Block_get_flags)( 
    const NS(Block) *const SIXTRL_RESTRICT block )
{
    return ( block ) ? block->flags : UINT64_C( 0 );
}

SIXTRL_INLINE NS(BeamElementInfo)* NS(Block_get_elements_begin)(
    NS(Block)* SIXTRL_RESTRICT block )
{
    return ( NS(BeamElementInfo)* )NS(Block_get_const_elements_begin)( block );
}

SIXTRL_INLINE NS(BeamElementInfo)* NS(Block_get_elements_end)(
    NS(Block)* SIXTRL_RESTRICT block )
{
    return ( NS(BeamElementInfo)* )NS(Block_get_const_elements_end)( block );
}

SIXTRL_INLINE NS(BeamElementInfo) const* NS(Block_get_const_elements_begin)(
    const NS(Block) *const SIXTRL_RESTRICT block )
{
    return ( block ) ? block->elem_info : 0;
}

SIXTRL_INLINE NS(BeamElementInfo) const* NS(Block_get_const_elements_end)(
    const NS(Block) *const SIXTRL_RESTRICT block )
{
    NS(BeamElementInfo) const* end_ptr = 
        NS(Block_get_const_elements_begin)( block );
        
    if( end_ptr != 0 )
    {
        SIXTRL_SIZE_T const num_of_elements = block->size;
        assert( num_of_elements <= block->capacity );
        end_ptr = end_ptr + num_of_elements;
    }
    
    return end_ptr;
}

/* ------------------------------------------------------------------------- */

#if !defined( _GPUCODE )

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* !defined( _GPUCODE ) */
    
#endif /* SIXTRACKLIB_COMMON_IMPL_BLOCK_TYPE_H__ */

/* end: sixtracklib/common/impl/block_type.h */
