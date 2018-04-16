#include "sixtracklib/common/block.h"

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "sixtracklib/_impl/definitions.h"
#include "sixtracklib/common/details/tools.h"
#include "sixtracklib/common/mem_pool.h"

/* ------------------------------------------------------------------------- */

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

/*
extern struct NS(Block)* NS(Block_new)();

extern struct NS(Block)* NS(Block_new_on_mempool)( 
    struct NS(MemPool)* SIXTRL_RESTRICT pool );

extern void NS(Block_free)( struct NS(Block)* SIXTRL_RESTRICT pool );

extern void NS(Block_clear)( struct NS(Block)* SIXTRL_RESTRICT pool );
*/

/* ------------------------------------------------------------------------- */

/*
extern bool NS( Block_manages_own_memory )( const struct NS( Block )) *
                                         const SIXTRL_RESTRICT p );

extern bool NS( Block_uses_mempool )( const struct NS( Block )) *
                                   const SIXTRL_RESTRICT p );

extern bool NS( Block_uses_flat_memory )( 
    const struct NS(Particles )* const SIXTRL_RESTRICT p );

extern struct NS( MemPool ) const* NS( Block_get_const_mem_pool )(
    const struct NS( Block )) * const SIXTRL_RESTRICT p );

extern unsigned char const* NS( Block_get_const_flat_memory )(
    const struct NS( Block )) * const SIXTRL_RESTRICT p );
    
static void NS(Block_set_size)( 
    NS(Block)* SIXTRL_RESTRICT block, size_t new_size );

static void NS(Block_set_capacity)(
    NS(Block)* SIXTRL_RESTRICT block, size_t new_capacity );

static void NS(Block_set_flags)(
    NS(Block)* SIXTRL_RESTRICT block, uint64_t flags );

static void NS(Block_set_next_element_id)( 
    NS(Block)* SIXTRL_RESTRICT block, int64_t next_element_id );

static int64_t NS(Block_get_next_element_id)( NS(Block)* SIXTRL_RESTRICT block );
*/

/* ------------------------------------------------------------------------- */

/*
extern bool NS( Block_has_defined_alignment )( const struct NS( Block ) *
                                            const SIXTRL_RESTRICT block );

extern bool NS( Block_is_aligned )( const struct NS( Block ) *
                                     const SIXTRL_RESTRICT block,
                                 size_t alignment );

extern bool NS( Block_check_alignment )( const struct NS( Block ) *
                                          const SIXTRL_RESTRICT block,
                                      size_t alignment );

extern size_t NS( Block_get_alignment )( const struct NS( Block ) *
                                    const SIXTRL_RESTRICT block );
                                    
extern void NS(Block_set_alignment)( 
    struct NS(Block)* SIXTRL_RESTRICT block, size_t new_alignment );
*/   

/* ************************************************************************* */

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
         *                     packed; note: Particles = 1
         * -----------------------------------------------------------------
         * - 1 x uint64_t .... nelem, i.e. number of elements to be 
         *                     serialized 
         * -----------------------------------------------------------------
         * - 1 x uint64_t .... nattr, i.e. the number of attributes 
         *                     that have been packed per element -> should 
         *                     be NS(PARTICLES_NUM_OF_ATTRIBUTES), store for
         *                     format versioning reasons
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

/*
NS(Block)* NS(Block_new)()
{
    bool success = false;
    
    NS(Block)* block = NS(Block_preset)(
        ( NS(Block)* )malloc( sizeof( NS(Block) ) ) );
    
    NS(MemPool)* ptr_mem_pool = NS(MemPool_preset)(
        ( NS(MemPool)* )malloc( sizeof( NS(MemPool) ) ) );
    
    if( ( block != 0 ) && ( ptr_mem_pool != 0 ) )
    {
        size_t const elem_capacity = NS(BLOCK_DEFAULT_CAPACITY);
        
        size_t const mem_pool_capacity = 
            NS(BLOCK_DEFAULT_CAPACITY) * NS(BLOCK_DEFAULT_ELEMENT_CAPACITY);
        
        size_t chunk_size = NS(BLOCK_DEFAULT_MEMPOOL_CHUNK_SIZE);
            
        NS(MemPool_init)( ptr_mem_pool, mem_pool_capacity, NS
        
    }
    
    if( !success )
    {
        if( block != 0 )
        {
            NS(Block_free)( block );
            free( block );
            block = 0;
        }
        
        if( ptr_mem_pool != 0 )
        {
            NS(MemPool_free)( ptr_mem_pool );
            free( ptr_mem_pool );
            ptr_mem_pool = 0;
        }
    }
    
    assert( ( (  success ) && ( block != 0 ) && ( ptr_mem_pool != 0 ) ) ||
            ( ( !success ) && ( block == 0 ) && ( ptr_mem_pool == 0 ) ) );
    
    return block;
}

NS(Block)* NS(Block_new_on_mempool)( struct NS(MemPool)* SIXTRL_RESTRICT pool )
{
    
}

void NS(Block_free)( struct NS(Block)* SIXTRL_RESTRICT block  )
{
    free( block->elem_info );
        
    if( NS(Block_manages_own_memory)( block  ) )
    {
        if( NS(Block_uses_mempool)( block ) )
        {
            NS(MemPool)* ptr_mem_pool = NS(MemPool)*block->ptr_mem_context;
            NS(MemPool_free)( ptr_mem_pool );
            block->ptr_mem_context = 0;
            block->ptr_mem_begin   = 0;
        }
    }
    
    NS(Block_preset)( pool );
    
    return;
}

void NS(Block_clear)( struct NS(Block)* SIXTRL_RESTRICT block )
{
    size_t const num_elements = NS(Block_get_size)( block );
    
    if( num_elements > ( size_t )0u )
    {
        NS(BeamElementInfo)* it  = NS(Block_get_elements_begin)( block );
        NS(BeamElementInfo)* end = NS(Block_get_elements_end)( block );
        
        for( ; it != end ; ++it )
        {
            NS(BeamElementInfo_preset)( it );
        }
        
        NS(Block_set_size)( block, ( size_t )0u );
    }
    
    return;
}

SIXTRL_INLINE void NS(Block_set_size)( 
    NS(Block)* SIXTRL_RESTRICT block, size_t new_size )
{
    if( block != 0 ) block->size = new_size;
    return;
}

SIXTRL_INLINE void NS(Block_set_capacity)(
    NS(Block)* SIXTRL_RESTRICT block, size_t new_capacity )
{
    if( block != 0 ) block->capacity = new_capacity;
    return;
}

SIXTRL_INLINE void NS(Block_set_flags)(
    NS(Block)* SIXTRL_RESTRICT block, uint64_t flags )
{
    if( block != 0 ) block->flags = flags;
    return;
}

SIXTRL_INLINE void NS(Block_set_next_element_id)( 
    NS(Block)* SIXTRL_RESTRICT block, int64_t next_element_id )
{
    if( block != 0 ) block->next_elemid = next_element_id;
}

int64_t NS(Block_get_next_element_id)( NS(Block)* SIXTRL_RESTRICT block )
{
    int64_t elem_id = NS(PARTICLES_INVALID_BEAM_ELEMENT_ID);
    
    static int64_t const MAX_ALLOWED_ELEMENT_ID = INT64_MAX - INT64_C( 1 );
    
    if( ( block != 0 ) && 
        ( block->next_elemid >= INT64_C( 0 ) ) &&
        ( block->next_elemid < MAX_ALLOWED_ELEMENT_ID ) )
    {
        elem_id = block->next_elemid++;
    }
    
    return elem_id;
}
*/

/* ------------------------------------------------------------------------- */

/*
bool NS( Block_manages_own_memory )( const struct NS( Block )) *
                                  const SIXTRL_RESTRICT block )
{
    return ( ( block != 0 ) && ( block->ptr_mem_context != 0 ) &&
             ( ( block->flags & NS(BLOCK_FLAGS_OWN_MEMORY) ) ==
                NS(BLOCK_FLAGS_OWN_MEMORY) ) );
}
*/

/* ------------------------------------------------------------------------- */

/*
bool NS( Block_uses_mempool )( const struct NS( Block )) *
                            const SIXTRL_RESTRICT block )
{
    return ( ( block != 0 ) && ( block->ptr_mem_context != 0 ) &&
             ( ( block->flags & NS(BLOCK_FLAGS_MEM_CTX_MEMPOOL ) ) ==
                NS(BLOCK_FLAGS_MEM_CTX_MEMPOOL) ) );
}
*/

/* ------------------------------------------------------------------------- */

/*
bool NS( Block_uses_flat_memory )( 
    const struct NS(Particles )* const SIXTRL_RESTRICT block )
{
    return ( ( block != 0 ) && ( block->ptr_mem_context != 0 ) &&
             ( ( block->flags & NS(BLOCK_FLAGS_MEM_CTX_MEMPOOL ) ) ==
                NS(BLOCK_FLAGS_MEM_CTX_MEMPOOL) ) );
}
*/

/* ------------------------------------------------------------------------- */

/*
NS( MemPool ) const* NS( Block_get_const_mem_pool )(
    const struct NS( Block )) * const SIXTRL_RESTRICT block )
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
*/

/* ------------------------------------------------------------------------- */

/*
unsigned char const* NS( Block_get_const_flat_memory )(
    const struct NS( Block )) * const SIXTRL_RESTRICT block )
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
*/

/* ------------------------------------------------------------------------- */

/*
bool NS( Block_has_defined_alignment )( 
    const struct NS( Block ) * const SIXTRL_RESTRICT block )
{
    return ( NS( Block_alignment )( block ) != UINT64_C( 0 ) );
}
*/

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

/*
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
*/

/* ------------------------------------------------------------------------- */

/* end: sixtracklib/common/details/block.c */
