#include "sixtracklib/common/block.h"

#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "sixtracklib/_impl/definitions.h"
#include "sixtracklib/common/impl/block_type.h"
#include "sixtracklib/common/mem_pool.h"
#include "sixtracklib/common/details/tools.h"

#include "sixtracklib/common/block_drift.h"
#include "sixtracklib/common/impl/block_drift_type.h"

/* ========================================================================= */

extern SIXTRL_INT64_T NS(Block_append_drift_aligned)(
    NS(Block)* SIXTRL_RESTRICT block, NS(BeamElementType) const type_id, 
    SIXTRL_REAL_T const length, 
    SIXTRL_SIZE_T alignment );

/* ========================================================================= */

extern SIXTRL_SIZE_T NS(Block_predict_required_mempool_capacity_for_packing)(
    unsigned char const* SIXTRL_RESTRICT ptr_mem_begin,
    SIXTRL_SIZE_T const chunk_size, 
    SIXTRL_SIZE_T* SIXTRL_RESTRICT ptr_alignment, 
    SIXTRL_SIZE_T const num_of_elements, 
    SIXTRL_SIZE_T const num_of_attributes,
    SIXTRL_SIZE_T const* ptr_attributes_sizes );

extern SIXTRL_SIZE_T NS(Block_predict_required_capacity_for_packing)(
    unsigned char const* SIXTRL_RESTRICT ptr_mem_begin,
    SIXTRL_SIZE_T const alignment, 
    SIXTRL_SIZE_T const num_of_elements,
    SIXTRL_SIZE_T const num_of_attributes,
    SIXTRL_SIZE_T const* ptr_attributes_sizes );

/* ------------------------------------------------------------------------- */


extern NS(Block)* NS(Block_new)( SIXTRL_SIZE_T const elements_capacity );

extern NS(Block)* NS(Block_new_on_mempool)( 
    SIXTRL_SIZE_T const capacity, struct NS(MemPool)* SIXTRL_RESTRICT pool );

extern void NS(Block_clear)( NS(Block)* SIXTRL_RESTRICT pool );

extern void NS(Block_free)( NS(Block)* SIXTRL_RESTRICT pool );

/* ------------------------------------------------------------------------- */


extern bool NS( Block_manages_own_memory )( const struct NS( Block )*
                                         const SIXTRL_RESTRICT p );

extern bool NS( Block_uses_mempool )( const struct NS( Block )*
                                   const SIXTRL_RESTRICT p );

extern bool NS( Block_uses_flat_memory )( 
    const struct NS(Block )* const SIXTRL_RESTRICT p );

extern NS( MemPool ) const* NS( Block_get_const_mem_pool )(
    const struct NS( Block )* const SIXTRL_RESTRICT p );

extern unsigned char const* NS( Block_get_const_flat_memory )(
    const struct NS( Block )* const SIXTRL_RESTRICT p );
    
/* ------------------------------------------------------------------------- */
    
SIXTRL_STATIC void NS(Block_set_size)( 
    NS(Block)* SIXTRL_RESTRICT block, size_t new_size );

SIXTRL_STATIC void NS(Block_set_capacity)(
    NS(Block)* SIXTRL_RESTRICT block, size_t new_capacity );

SIXTRL_STATIC void NS(Block_set_flags)(
    NS(Block)* SIXTRL_RESTRICT block, uint64_t flags );

SIXTRL_STATIC void NS(Block_set_next_element_id)( 
    NS(Block)* SIXTRL_RESTRICT block, int64_t next_element_id );

extern int64_t NS(Block_get_next_element_id)( 
    NS(Block)* SIXTRL_RESTRICT block );


SIXTRL_STATIC void const* NS( Block_get_const_ptr_mem_context )(
    const struct NS( Block ) * const SIXTRL_RESTRICT block );

SIXTRL_STATIC void* NS( Block_get_ptr_mem_context )( struct NS( Block ) *
                                                  SIXTRL_RESTRICT block );

SIXTRL_STATIC void const* NS( Block_get_const_mem_begin )(
    const NS(Block) *const SIXTRL_RESTRICT block );

extern void* NS( Block_get_mem_begin )(
    NS(Block)* SIXTRL_RESTRICT block );

SIXTRL_STATIC void NS( Block_set_ptr_elements_info)( 
    NS(Block)* SIXTRL_RESTRICT block, 
    struct NS(BeamElementInfo)* SIXTRL_RESTRICT ptr_info );

SIXTRL_STATIC void NS( Block_set_ptr_mem_context )( struct NS( Block ) *
                                                     SIXTRL_RESTRICT block,
                                                 void* ptr_mem_context );

SIXTRL_STATIC void NS( Block_set_ptr_mem_begin )( 
    NS(Block)* SIXTRL_RESTRICT block, void* ptr_mem_begin );

/* ------------------------------------------------------------------------- */

extern bool NS( Block_has_defined_alignment )( const struct NS( Block ) *
                                            const SIXTRL_RESTRICT block );

/* 
extern bool NS( Block_is_aligned )( const struct NS( Block ) *
                                     const SIXTRL_RESTRICT block,
                                 size_t alignment );

                           
extern bool NS( Block_check_alignment )( const struct NS( Block ) *
                                          const SIXTRL_RESTRICT block,
                                      size_t alignment ); */

extern size_t NS( Block_get_alignment )( const struct NS( Block ) *
                                    const SIXTRL_RESTRICT block );
                                    
extern void NS(Block_set_alignment)( 
    NS(Block)* SIXTRL_RESTRICT block, size_t new_alignment );

/* ************************************************************************* */

/* ========================================================================= */

SIXTRL_INT64_T NS(Block_append_drift_aligned)(
    NS(Block)* SIXTRL_RESTRICT block, NS(BeamElementType) const type_id, 
    SIXTRL_REAL_T const length, SIXTRL_SIZE_T alignment )
{
    typedef NS(BeamElementInfo) elem_info_t;    
    typedef NS(MemPool)         mem_pool_t;
    typedef NS(AllocResult)     result_t;
    
    SIXTRL_INT64_T element_id = NS(PARTICLES_INVALID_BEAM_ELEMENT_ID);
    
    SIXTRL_SIZE_T const current_size = NS(Block_get_size)( block );
    SIXTRL_SIZE_T const capacity     = NS(Block_get_capacity)( block );
    
    if( ( block != 0 ) && ( length >= ( SIXTRL_REAL_T )0 ) &&
        ( current_size < capacity ) &&
        ( ( type_id == NS(ELEMENT_TYPE_DRIFT) ) ||
          ( type_id == NS(ELEMENT_TYPE_DRIFT_EXACT) ) ) )
    {
        NS(MemPool) rollback_pool;
        
        SIXTRL_INT64_T const next_elem_id = 
            NS(Block_get_next_element_id)( block );
        
        mem_pool_t*  pool = 0;
        elem_info_t* ptr_elem_info = NS(Block_get_elements_begin)( block );
        
        result_t append_result;
        
        SIXTRL_ASSERT( ( ptr_elem_info != 0 ) && 
            ( next_elem_id != NS(PARTICLES_INVALID_BEAM_ELEMENT_ID) ) );
        
        /* TODO: Implement for flat memory storage backend as well! */        
        if( !NS(Block_uses_mempool)( block ) ) return element_id;
        
        pool = ( mem_pool_t* )NS(Block_get_const_mem_pool)( block );
        SIXTRL_ASSERT( pool != 0 );
        rollback_pool = *pool;
        
        append_result = NS(Drift_create_and_pack_aligned)( 
            type_id, next_elem_id, length, pool, alignment );
        
        if( NS(AllocResult_valid)( &append_result ) )
        {
            SIXTRL_SIZE_T const new_size = current_size + ( SIXTRL_SIZE_T )1u;
            
            ptr_elem_info = ptr_elem_info + current_size;
            NS(BeamElementInfo_set_type_id)( ptr_elem_info, type_id );
            NS(BeamElementInfo_set_element_id)( ptr_elem_info, next_elem_id );
            NS(BeamElementInfo_set_ptr_mem_begin)( ptr_elem_info, 
                NS(AllocResult_get_pointer)( &append_result ) );
            
            NS(Block_set_size)( block, new_size );
            element_id = next_elem_id;
        }
        else
        {
            *pool = rollback_pool;
            NS(Block_set_next_element_id)( block, next_elem_id );
        }
    }
    
    return element_id;
    
}

/* ========================================================================= */

SIXTRL_SIZE_T NS(Block_predict_required_num_bytes_on_mempool_for_packing)(
    unsigned char const* SIXTRL_RESTRICT ptr_mem_begin,
    SIXTRL_SIZE_T const chunk_size, 
    SIXTRL_SIZE_T* SIXTRL_RESTRICT ptr_alignment, 
    SIXTRL_SIZE_T const num_of_elements, 
    SIXTRL_SIZE_T const num_of_attributes,
    SIXTRL_SIZE_T const* ptr_attribute_sizes )
{
    static SIXTRL_SIZE_T const ZERO_SIZE = ( SIXTRL_SIZE_T )0u;
    SIXTRL_SIZE_T predicted_num_bytes = ZERO_SIZE;
    
    if( ( chunk_size > ZERO_SIZE ) && 
        ( ptr_attribute_sizes != 0 ) && ( num_of_elements > ZERO_SIZE ) && 
        ( num_of_attributes > ZERO_SIZE ) )
    {
        SIXTRL_SIZE_T alignment;
        
        if( ptr_alignment != 0 )
        {
            if( ( *ptr_alignment == chunk_size ) ||
                ( ( *ptr_alignment < chunk_size ) && 
                  ( ( chunk_size % *ptr_alignment ) == ZERO_SIZE ) ) )
            {
                alignment = chunk_size;                
            }
            else if( ( *ptr_alignment > chunk_size ) && 
                     ( ( *ptr_alignment % chunk_size ) == ZERO_SIZE ) )
            {
                alignment = *ptr_alignment;
            }
            else
            {
                alignment = NS(least_common_multiple)( chunk_size, *ptr_alignment );                
            }            
            
            assert( ( alignment >= chunk_size ) && 
                    ( alignment >= *ptr_alignment ) &&
                    ( ( alignment % *ptr_alignment ) == ZERO_SIZE ) &&
                    ( ( alignment % chunk_size ) == ZERO_SIZE ) );
        
            if( alignment != *ptr_alignment )
            {
                *ptr_alignment = alignment;
            }
        }
        else
        {
            alignment = chunk_size;
        }
                
        predicted_num_bytes = NS(Block_predict_required_num_bytes_for_packing)(
            ptr_mem_begin, alignment, num_of_elements, num_of_attributes,
            ptr_attribute_sizes );
    }
    
    return predicted_num_bytes;
}

/* ------------------------------------------------------------------------- */

SIXTRL_SIZE_T NS(Block_predict_required_num_bytes_for_packing)(
    unsigned char const* SIXTRL_RESTRICT ptr_mem_begin,
    SIXTRL_SIZE_T const alignment, SIXTRL_SIZE_T const num_of_elements,
    SIXTRL_SIZE_T const num_of_attributes, SIXTRL_SIZE_T const* attr_size_it )
{
    static SIXTRL_SIZE_T const ZERO_SIZE = ( SIXTRL_SIZE_T )0u;
    SIXTRL_SIZE_T predicted_num_bytes = ZERO_SIZE;
    
    if( ( alignment != ZERO_SIZE ) &&
        ( attr_size_it  != 0 ) && ( num_of_elements > ZERO_SIZE ) && 
        ( num_of_attributes > ZERO_SIZE ) )
    {
        uintptr_t const begin_addr     = ( uintptr_t )ptr_mem_begin;
        uintptr_t const begin_addr_mod = begin_addr % alignment;
                
        SIXTRL_SIZE_T ii;
        SIXTRL_SIZE_T temp;
        SIXTRL_SIZE_T pack_info_length;
        
        SIXTRL_SIZE_T const u64_size = sizeof( SIXTRL_UINT64_T );
        
        /* ----------------------------------------------------------------- */
        
        /* a null - pointer has been submitted, so we have to assume the worst -
         * add the full alignment  to the predicted capacity to be allow 
         * alignment under all circumstances;
         * Otherwise, use the right offset starting from ptr_mem_begin:
         */
        
        if( begin_addr == ZERO_SIZE )
        {
            predicted_num_bytes = alignment;
        }
        else if( begin_addr_mod != ZERO_SIZE )
        {
            predicted_num_bytes = alignment - begin_addr_mod;
        }
        
        /* Packing information: 
         * -----------------------------------------------------------------
         * - 1 x uint64_t .... length of the whole serialized slab of memory
         *                     including the length indicator itself. I.e. 
         *                     ( current_pos + length bytes ) == first byte
         *                     past the serialized item
         * -----------------------------------------------------------------
         * - 1 x uint64_t .... indicator, i.e. what type of element has been 
         *                     packed; note: Drift = 2, DriftExact = 3
         * -----------------------------------------------------------------
         * - 1 x uint64_t .... nelem, i.e. number of elements to be 
         *                     serialized. In the case of BeamElements, this
         *                     is always 1
         * -----------------------------------------------------------------
         * - 1 x uint64_t .... nattr, i.e. the number of attributes 
         *                     For Drift and DriftExact, this number is 2
         *                     (length, element_id);
         * -----------------------------------------------------------------
         * - num x uint64_t .. nattr x offsets, i.e. for each of the num 
         *                     elemens an offset in bytes on where the 
         *                     data is stored.
         * 
         *                     Note: the offset is calculated relative to 
         *                     the current_pos, i.e. the pointer pointing 
         *                     to the length indicator. The minimum 
         *                     offset for Particles is therefore 
         *                     NS(PARTICLES_PACK_BLOCK_LENGTH) 
         * ----------------------------------------------------------------- */
        
        pack_info_length  = u64_size + u64_size + u64_size + u64_size;
        pack_info_length += u64_size * num_of_attributes;
        
        temp = ( pack_info_length / alignment ) * alignment;
        
        if( pack_info_length > temp )
        {
            pack_info_length = temp + alignment;
        }
        
        assert( ( pack_info_length % alignment ) == ZERO_SIZE );
        predicted_num_bytes += pack_info_length;
        
        /* ----------------------------------------------------------------- */
        
        for( ii = ZERO_SIZE ; ii < num_of_attributes ; ++ii, ++attr_size_it )
        {
            SIXTRL_SIZE_T const attr_size = *attr_size_it;
            SIXTRL_SIZE_T attr_block_length = num_of_elements * attr_size;
            
            SIXTRL_SIZE_T const attr_block_length_mod = 
                attr_block_length % alignment;
            
            attr_block_length += ( attr_block_length_mod == ZERO_SIZE )
                ? ZERO_SIZE : ( alignment - attr_block_length_mod );
            
            assert( ( attr_block_length % alignment ) == ZERO_SIZE );
            predicted_num_bytes += attr_block_length;
        }
        
        /* ----------------------------------------------------------------- */
        
        assert( ( ( predicted_num_bytes + begin_addr ) % alignment ) == 
            ZERO_SIZE );        
    }
    
    return predicted_num_bytes;
}

/* ------------------------------------------------------------------------- */

NS(Block)* NS(Block_new)( SIXTRL_SIZE_T const capacity )
{
    bool success = false;
    
    typedef NS(Block)           block_t;
    typedef NS(MemPool)         pool_t;
    typedef NS(BeamElementInfo) elem_info_t;
    
    SIXTRL_STATIC SIXTRL_SIZE_T const ZERO_SIZE = ( SIXTRL_SIZE_T )0u;
    
    block_t*  block = 0;
    pool_t*    pool = 0;
    elem_info_t* element_infos = 0;
    
    if( capacity > ZERO_SIZE )
    {
        pool = NS(MemPool_preset)( ( pool_t* )malloc( sizeof( pool_t ) ) );
        block = NS(Block_preset)( ( block_t* )malloc( sizeof( block_t ) ) );
        element_infos = NS(BeamElementInfo_preset)( ( elem_info_t* )malloc( 
            sizeof( elem_info_t ) * capacity ) );
        
        if( ( block != 0 ) && ( pool != 0 ) && ( element_infos != 0 ) )
        {
            SIXTRL_SIZE_T const mem_pool_capacity = 
                capacity * NS(BLOCK_DEFAULT_ELEMENT_CAPACITY);
                
            SIXTRL_SIZE_T const chunk_size = NS(BLOCK_DEFAULT_MEMPOOL_CHUNK_SIZE);
        
            NS(MemPool_init)( pool, mem_pool_capacity, chunk_size );
        
            SIXTRL_ASSERT( 
                ( NS(MemPool_get_buffer)( pool ) != 0 ) &&
                ( NS(MemPool_get_chunk_size)( pool ) > ZERO_SIZE ) &&
                ( NS(MemPool_get_capacity)( pool ) >= mem_pool_capacity ) );
        
            SIXTRL_UINT64_T const flags = 
                NS(BLOCK_FLAGS_OWNS_MEMORY) | NS(BLOCK_FLAGS_PACKED) |
                NS(BLOCK_FLAGS_MEM_CTX_MEMPOOL);
            
            NS(Block_set_capacity)( block, capacity );
            NS(Block_set_size)( block, ZERO_SIZE );
            NS(Block_set_next_element_id)( block, ( SIXTRL_INT64_T )0 );
            NS(Block_set_flags)( block, flags );
            NS(Block_set_ptr_mem_context)( block, ( void* )pool );
            NS(Block_set_ptr_mem_begin)( block, NS(MemPool_get_buffer)( pool ) );
            NS(Block_set_ptr_elements_info)( block, element_infos );
            
            success = true;
        }        
    }
    
    if( !success )
    {
        if( block != 0 )
        {
            NS(Block_free)( block );
            free( block );
            element_infos = 0;
            block = 0;
        }
        
        if( pool != 0 )
        {
            NS(MemPool_free)( pool );
            free( pool );
            pool = 0;
        }
    }
    
    assert( ( (  success ) && ( block != 0 ) && ( pool != 0 ) && ( element_infos != 0 ) ) ||
            ( ( !success ) && ( block == 0 ) && ( pool == 0 ) && ( element_infos == 0 ) ) );
    
    return block;
}

NS(Block)* NS(Block_new_on_mempool)( SIXTRL_SIZE_T const capacity, 
    NS(MemPool)* SIXTRL_RESTRICT pool )
{
    bool success = false;
    
    typedef NS(Block)           block_t;
    typedef NS(BeamElementInfo) elem_info_t;
    
    SIXTRL_STATIC SIXTRL_SIZE_T const ZERO_SIZE = ( SIXTRL_SIZE_T )0u;
    
    block_t*  block = 0;    
    elem_info_t* element_infos = 0;
    
    SIXTRL_SIZE_T const mem_pool_capacity = 
        capacity * NS(BLOCK_DEFAULT_ELEMENT_CAPACITY);
    
    if( ( pool != 0 ) && ( capacity > ZERO_SIZE ) && 
        ( NS(MemPool_get_buffer)( pool ) != 0 ) &&
        ( NS(MemPool_get_remaining_bytes)( pool ) > mem_pool_capacity ) )
    {
        block = NS(Block_preset)( ( block_t* )malloc( sizeof( block_t ) ) );
        element_infos = NS(BeamElementInfo_preset)( ( elem_info_t* )malloc( 
            sizeof( elem_info_t ) * capacity ) );
        
        if( ( block != 0 ) && ( pool != 0 ) && ( element_infos != 0 ) )
        {
            SIXTRL_UINT64_T const flags = 
                NS(BLOCK_FLAGS_PACKED) | NS(BLOCK_FLAGS_MEM_CTX_MEMPOOL);
            
            NS(Block_set_capacity)( block, capacity );
            NS(Block_set_size)( block, ZERO_SIZE );
            NS(Block_set_next_element_id)( block, ( SIXTRL_INT64_T )0 );
            NS(Block_set_flags)( block, flags );
            NS(Block_set_ptr_mem_context)( block, ( void* )pool );
            NS(Block_set_ptr_mem_begin)( block, NS(MemPool_get_buffer)( pool ) );
            NS(Block_set_ptr_elements_info)( block, element_infos );
            
            success = true;
        }        
    }
    
    if( !success )
    {
        if( block != 0 )
        {
            NS(Block_free)( block );
            free( block );
            element_infos = 0;
            block = 0;
        }
    }
    
    assert( ( (  success ) && ( block != 0 ) && ( element_infos != 0 ) ) ||
            ( ( !success ) && ( block == 0 ) && ( element_infos == 0 ) ) );
    
    return block;
}

void NS(Block_free)( NS(Block)* SIXTRL_RESTRICT block  )
{
    free( NS(Block_get_elements_begin)( block ) );
    NS(Block_set_ptr_elements_info)( block, 0 );
        
    if( NS(Block_manages_own_memory)( block  ) )
    {
        if( NS(Block_uses_mempool)( block ) )
        {
            NS(MemPool)* pool = 
                ( NS(MemPool* ) )NS(Block_get_const_mem_pool)( block );
            
            NS(MemPool_free)( pool );
            free( pool );
            
            NS(Block_set_ptr_mem_context)( block, 0 );
            NS(Block_set_ptr_mem_begin)( block, 0 );            
        }
    }
    
    NS(Block_preset)( block );
    
    return;
}

void NS(Block_clear)( NS(Block)* SIXTRL_RESTRICT block )
{
    SIXTRL_STATIC SIXTRL_SIZE_T const ZERO = ( SIXTRL_SIZE_T )0u;
    SIXTRL_SIZE_T const num_elements = NS(Block_get_size)( block );
    
    if( num_elements > ZERO )
    {
        NS(BeamElementInfo)* it  = NS(Block_get_elements_begin)( block );
        NS(BeamElementInfo)* end = NS(Block_get_elements_end)( block );
        
        for( ; it != end ; ++it )
        {
            NS(BeamElementInfo_preset)( it );
        }
        
        NS(Block_set_size)( block, ZERO );
        
        if( ( NS(Block_manages_own_memory)( block ) ) &&
            ( NS(Block_uses_mempool)( block ) ) )
        {
            NS(MemPool)* pool = ( NS(MemPool)* )NS(Block_get_ptr_mem_context)( block );
            NS(MemPool_clear)( pool );
        }
    }
    
    return;
}

/* ------------------------------------------------------------------------- */

void NS(Block_set_size)( 
    NS(Block)* SIXTRL_RESTRICT block, size_t new_size )
{
    if( block != 0 ) block->size = new_size;
    return;
}

void NS(Block_set_capacity)(
    NS(Block)* SIXTRL_RESTRICT block, size_t new_capacity )
{
    if( block != 0 ) block->capacity = new_capacity;
    return;
}

void NS(Block_set_flags)( NS(Block)* SIXTRL_RESTRICT block, uint64_t flags )
{
    if( block != 0 ) block->flags = flags;
    return;
}

void NS(Block_set_next_element_id)( 
    NS(Block)* SIXTRL_RESTRICT block, int64_t next_element_id )
{
    if( block != 0 ) block->next_element_id = next_element_id;
}

int64_t NS(Block_get_next_element_id)( NS(Block)* SIXTRL_RESTRICT block )
{
    int64_t elem_id = NS(PARTICLES_INVALID_BEAM_ELEMENT_ID);
    
    static int64_t const MAX_ALLOWED_ELEMENT_ID = INT64_MAX - INT64_C( 1 );
    
    if( ( block != 0 ) && 
        ( block->next_element_id >= INT64_C( 0 ) ) &&
        ( block->next_element_id < MAX_ALLOWED_ELEMENT_ID ) )
    {
        elem_id = block->next_element_id++;
    }
    
    return elem_id;
}

/* ------------------------------------------------------------------------- */

bool NS( Block_manages_own_memory )( const struct NS( Block ) *
                                  const SIXTRL_RESTRICT block )
{
    return ( ( block != 0 ) && ( block->ptr_mem_context != 0 ) &&
             ( ( block->flags & NS(BLOCK_FLAGS_OWNS_MEMORY) ) ==
                NS(BLOCK_FLAGS_OWNS_MEMORY) ) );
}

/* ------------------------------------------------------------------------- */

bool NS( Block_uses_mempool )( const struct NS( Block ) *
                            const SIXTRL_RESTRICT block )
{
    return ( ( block != 0 ) && ( block->ptr_mem_context != 0 ) &&
             ( ( block->flags & NS(BLOCK_FLAGS_MEM_CTX_MEMPOOL ) ) ==
                NS(BLOCK_FLAGS_MEM_CTX_MEMPOOL) ) );
}

/* ------------------------------------------------------------------------- */

bool NS( Block_uses_flat_memory )( 
    const NS(Block)* const SIXTRL_RESTRICT block )
{
    return ( ( block != 0 ) && ( block->ptr_mem_context != 0 ) &&
             ( ( block->flags & NS(BLOCK_FLAGS_MEM_CTX_MEMPOOL ) ) ==
                NS(BLOCK_FLAGS_MEM_CTX_MEMPOOL) ) );
}

/* ------------------------------------------------------------------------- */

NS( MemPool ) const* NS( Block_get_const_mem_pool )(
    const struct NS( Block ) * const SIXTRL_RESTRICT block )
{
    NS( MemPool ) const* ptr_mem_pool = 0;

    if( ( block != 0 ) && ( block->ptr_mem_context != 0 ) &&
        ( ( block->flags & NS( BLOCK_FLAGS_MEM_CTX_MEMPOOL ) ) ==
          NS( BLOCK_FLAGS_MEM_CTX_MEMPOOL ) ) )
    {
        ptr_mem_pool = (NS( MemPool ) const*)block->ptr_mem_context;
    }
    
    return ptr_mem_pool;
}

/* ------------------------------------------------------------------------- */

unsigned char const* NS( Block_get_const_flat_memory )(
    const struct NS( Block ) * const SIXTRL_RESTRICT block )
{
    unsigned char const* ptr_flat_mem_block = 0;
    
    if( ( block != 0 ) && ( block->ptr_mem_context != 0 ) &&
        ( ( block->flags & NS( BLOCK_FLAGS_MEM_CTX_FLAT_MEMORY ) ) ==
          NS( BLOCK_FLAGS_MEM_CTX_FLAT_MEMORY ) ) )
    {
        ptr_flat_mem_block = ( unsigned char const* )block->ptr_mem_context;
    }
    
    return ptr_flat_mem_block;
}

/* ------------------------------------------------------------------------- */

bool NS( Block_has_defined_alignment )( 
    const struct NS( Block ) * const SIXTRL_RESTRICT block )
{
    return ( NS( Block_get_alignment )( block ) != ( SIXTRL_UINT64_T )0u );
}

/* ------------------------------------------------------------------------- */

/*
bool NS( Block_is_aligned )( 
    const struct NS( Block ) * const SIXTRL_RESTRICT block, size_t alignment )
{
    bool is_aligned = false;
    
    static size_t const ZERO_SIZE = ( size_t )0u;

    if( ( alignment != ZERO_SIZE ) && ( block != 0 ) )
    {
        if( block->alignment > ZERO_SIZE )
        {
            is_aligned = ( ( alignment == block->alignment ) ||
                ( (   alignment <  block->alignment ) &&
                  ( ( block->alignment % alignment  ) == ZERO_SIZE ) ) );
            
            assert( ( !is_aligned ) ||
                    ( NS( Block_check_alignment )( block, alignment ) ) );
        }
        else
        {
            is_aligned = NS( Particles_check_alignment )( block, alignment );
        }
    }
    
    return is_aligned;
}
*/

/* ------------------------------------------------------------------------- */

/*
bool NS( Block_check_alignment )( 
    const struct NS( Block ) * const SIXTRL_RESTRICT block, size_t alignment )
{
    bool is_aligned = false;
    
    static size_t const ZERO_SIZE = ( size_t )0u;
    
    if( ( block != 0 ) && ( alignment != ZERO_SIZE ) )
    {
        NS(BeamElementInfo) const* elem_it  = 
            NS(Block_get_const_elements_begin)( block );
            
        NS(BeamElementInfo) const* elem_end =
            NS(Block_get_const_elements_end)( block );
         
        if( ( elem_it != 0 ) && ( elem_it != elem_end ) )
        {
            is_aligned = true;
            
            for( ; elem_it != elem_end ; ++elem_it )
            {
                size_t num_of_inactive_elements = ZERO_SIZE;
                
                if( NS(BeamElementInfo_is_available)( elem_it ) )
                {
                    if( ( elem_it->ptr_element == 0 ) ||
                        ( ( ( ( uintptr_t )elem_it->ptr_element ) % alignment ) 
                            != ZERO_SIZE ) )
                    {
                        is_aligned = false;
                        break;
                    }
                }
                else
                {
                    ++num_of_inactive_elements;
                }
            }
        }
    }
    
    return is_aligned;
}
*/

/* ------------------------------------------------------------------------- */

size_t NS( Block_get_alignment )( 
    const struct NS( Block ) * const SIXTRL_RESTRICT block )
{
    return ( block != 0 ) ? block->alignment : ( size_t )0u;
}

void NS(Block_set_alignment)( NS(Block)* SIXTRL_RESTRICT block, 
                              size_t new_alignment )
{
    static size_t const ZERO_SIZE = ( size_t )0u;
    
    if( ( block != 0 ) && ( new_alignment != ZERO_SIZE ) &&
        ( NS(Block_get_size)( block ) == ZERO_SIZE ) )
    {
        block->alignment = new_alignment;
    }
    
    return;
}

/* ------------------------------------------------------------------------- */

void const* NS( Block_get_const_ptr_mem_context )(
    const struct NS( Block ) * const SIXTRL_RESTRICT block )
{
    return ( block != 0 ) ? block->ptr_mem_context : 0;
}

void* NS( Block_get_ptr_mem_context )( struct NS( Block ) *
                                                  SIXTRL_RESTRICT block )
{
   return ( void* )NS(Block_get_const_ptr_mem_context)( block );
}

void const* NS( Block_get_const_mem_begin )(
    const NS(Block) *const SIXTRL_RESTRICT block )
{
    return ( block != 0 ) ? block->ptr_mem_begin : 0;
}

void* NS( Block_get_mem_begin )( NS(Block)* SIXTRL_RESTRICT block )
{
    return ( void* )NS(Block_get_const_mem_begin)( block );
}

void NS( Block_set_ptr_mem_context )( 
    struct NS( Block ) * SIXTRL_RESTRICT block, void* ptr_mem_context )
{
    if( block != 0 )
    {
        block->ptr_mem_context = ptr_mem_context;
    }
    
    return;
}

void NS( Block_set_ptr_mem_begin )( 
    NS(Block)* SIXTRL_RESTRICT block, void* ptr_mem_begin )
{
    if( block != 0 )
    {
        block->ptr_mem_begin = ptr_mem_begin;
    }
    
    return;
}

void NS( Block_set_ptr_elements_info)( 
    NS(Block)* SIXTRL_RESTRICT block, 
    struct NS(BeamElementInfo)* SIXTRL_RESTRICT ptr_info )
{
    if( block != 0 )
    {
        block->elem_info = ptr_info;        
    }
    
    return;
}

/* end: sixtracklib/common/details/block.c */
