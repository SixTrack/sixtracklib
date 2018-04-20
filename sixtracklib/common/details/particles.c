#include "sixtracklib/common/particles.h"

#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "sixtracklib/_impl/definitions.h"

#include "sixtracklib/common/impl/particles_type.h"
#include "sixtracklib/common/mem_pool.h"
#include "sixtracklib/common/single_particle.h"

/* ------------------------------------------------------------------------- */

extern NS( Particles ) *
    NS( Particles_preset )( NS( Particles ) * SIXTRL_RESTRICT p );

/* ------------------------------------------------------------------------- */

extern SIXTRL_SIZE_T NS( Particles_predict_required_capacity )( 
    SIXTRL_SIZE_T num_particles, SIXTRL_SIZE_T* SIXTRL_RESTRICT ptr_chunk_size,
    SIXTRL_SIZE_T* SIXTRL_RESTRICT ptr_alignment, bool make_packed );
    
/* ------------------------------------------------------------------------- */

extern bool NS( Particles_is_packed )( const struct NS( Particles ) *
                                       const SIXTRL_RESTRICT p );

extern bool NS( Particles_manages_own_memory )( const struct NS( Particles ) *
                                                const SIXTRL_RESTRICT p );

extern bool NS( Particles_uses_mempool )( const struct NS( Particles ) *
                                          const SIXTRL_RESTRICT p );

extern bool NS( Particles_uses_single_particle )( const struct NS( Particles ) *
                                                  const SIXTRL_RESTRICT p );

extern bool NS( Particles_uses_flat_memory )( 
    const struct NS(Particles )* const SIXTRL_RESTRICT p );

extern struct NS( MemPool ) const* NS( Particles_get_const_mem_pool )(
    const struct NS( Particles ) * const SIXTRL_RESTRICT p );

extern struct NS( SingleParticle ) const* NS(
    Particles_get_const_base_single_particle )( const struct NS( Particles ) *
                                                const SIXTRL_RESTRICT p );
    
extern unsigned char const* NS( Particles_get_const_flat_memory )(
    const struct NS( Particles ) * const SIXTRL_RESTRICT p );

/* ------------------------------------------------------------------------- */

extern struct NS( Particles ) * NS( Particles_new )( SIXTRL_SIZE_T npart );

extern struct NS( Particles ) * NS( Particles_new_aligned )( 
    SIXTRL_SIZE_T const npart, SIXTRL_SIZE_T alignment );

extern struct NS( Particles ) *
    NS( Particles_new_on_mempool )( SIXTRL_SIZE_T npart,
                                    struct NS( MemPool ) *
                                        SIXTRL_RESTRICT pool );

extern struct NS( Particles ) * NS( Particles_new_single )();

extern struct NS( Particles ) *
    NS( Particles_new_on_single )( struct NS( SingleParticle ) *
                                   ptr_single_particle );

extern void NS( Particles_free )( struct NS( Particles ) *
                                  SIXTRL_RESTRICT particles );

static bool NS( Particles_map_to_mempool )( struct NS( Particles ) *
                                                SIXTRL_RESTRICT particles,
                                            struct NS( MemPool ) *
                                                SIXTRL_RESTRICT pool,
                                            SIXTRL_SIZE_T npart,
                                            SIXTRL_SIZE_T alignment, 
                                            bool make_packed );

static bool NS( Particles_map_to_single_particle )(
    struct NS( Particles ) * SIXTRL_RESTRICT particles,
    struct NS( SingleParticle ) * SIXTRL_RESTRICT single_particle );

/* -------------------------------------------------------------------------- */

extern bool NS(Particles_unpack)( struct NS(Particles)* SIXTRL_RESTRICT particles, 
    unsigned char* SIXTRL_RESTRICT mem, SIXTRL_UINT64_T flags ); 

static bool NS(Particles_unpack_read_header)(
    unsigned char const* SIXTRL_RESTRICT ptr_mem_begin, 
    SIXTRL_SIZE_T* ptr_length, SIXTRL_SIZE_T* ptr_num_particles, SIXTRL_SIZE_T* ptr_num_attrs );

static bool NS(Particles_unpack_check_consistency)(
    unsigned char const* SIXTRL_RESTRICT ptr_mem_begin,
    SIXTRL_SIZE_T  length, SIXTRL_SIZE_T num_of_particles, SIXTRL_SIZE_T num_of_attributes );

static bool NS(Particles_unpack_map_memory)(
    struct NS(Particles)* SIXTRL_RESTRICT particles, 
    unsigned char* ptr_mem_begin, SIXTRL_SIZE_T length, SIXTRL_SIZE_T num_of_particles,
    SIXTRL_SIZE_T num_of_attributes );

/* -------------------------------------------------------------------------- */

extern bool NS( Particles_has_defined_alignment )(
    const struct NS( Particles ) * const SIXTRL_RESTRICT p );

extern bool NS( Particles_is_aligned )( const struct NS( Particles ) *
                                            const SIXTRL_RESTRICT p,
                                        SIXTRL_SIZE_T alignment );

extern bool NS( Particles_check_alignment )( const struct NS( Particles ) *
                                                 const SIXTRL_RESTRICT p,
                                             SIXTRL_SIZE_T alignment );

extern SIXTRL_UINT64_T NS( Particles_alignment )( const struct NS( Particles ) *
                                           const SIXTRL_RESTRICT p );

/* ------------------------------------------------------------------------- */

extern bool NS( Particles_is_consistent )( const struct NS( Particles ) *
                                           const SIXTRL_RESTRICT p );

/* ------------------------------------------------------------------------- */

extern bool NS( Particles_deep_copy_one )( struct NS( Particles ) *
                                               SIXTRL_RESTRICT dest,
                                           SIXTRL_UINT64_T dest_id,
                                           struct NS( Particles )
                                               const* SIXTRL_RESTRICT src,
                                           SIXTRL_UINT64_T src_id );

extern bool NS( Particles_deep_copy_all )( struct NS( Particles ) *
                                               SIXTRL_RESTRICT dest,
                                           struct NS( Particles )
                                               const* SIXTRL_RESTRICT src );

/* ------------------------------------------------------------------------- */
/* ----- Implementation */
/* ------------------------------------------------------------------------- */

NS( Particles ) * NS( Particles_preset )( NS( Particles ) * SIXTRL_RESTRICT p )
{
    if( p != 0 )
    {
        NS( Particles_assign_ptr_to_q0 )( p, 0 );
        NS( Particles_assign_ptr_to_mass0 )( p, 0 );
        NS( Particles_assign_ptr_to_beta0 )( p, 0 );
        NS( Particles_assign_ptr_to_gamma0 )( p, 0 );
        NS( Particles_assign_ptr_to_p0c )( p, 0 );

        NS( Particles_assign_ptr_to_particle_id )( p, 0 );
        NS( Particles_assign_ptr_to_lost_at_element_id )( p, 0 );
        NS( Particles_assign_ptr_to_lost_at_turn )( p, 0 );
        NS( Particles_assign_ptr_to_state )( p, 0 );

        NS( Particles_assign_ptr_to_s )( p, 0 );
        NS( Particles_assign_ptr_to_x )( p, 0 );
        NS( Particles_assign_ptr_to_px )( p, 0 );
        NS( Particles_assign_ptr_to_y )( p, 0 );
        NS( Particles_assign_ptr_to_py )( p, 0 );
        NS( Particles_assign_ptr_to_sigma )( p, 0 );

        NS( Particles_assign_ptr_to_psigma )( p, 0 );
        NS( Particles_assign_ptr_to_delta )( p, 0 );
        NS( Particles_assign_ptr_to_rpp )( p, 0 );
        NS( Particles_assign_ptr_to_rvv )( p, 0 );
        NS( Particles_assign_ptr_to_chi )( p, 0 );

        NS( Particles_set_size )( p, ( SIXTRL_UINT64_T )0u );
        NS( Particles_set_flags )( p, NS( PARTICLES_FLAGS_NONE ) );
        NS( Particles_set_ptr_mem_context )( p, 0 );
        NS( Particles_set_ptr_mem_begin   )( p, 0 );
    }

    return p;
}

/* ------------------------------------------------------------------------ */

SIXTRL_SIZE_T NS( Particles_predict_required_capacity )( 
    SIXTRL_SIZE_T num_particles, SIXTRL_SIZE_T* SIXTRL_RESTRICT ptr_chunk_size,
    SIXTRL_SIZE_T* SIXTRL_RESTRICT ptr_alignment, bool make_packed )
{
    static SIXTRL_SIZE_T const ZERO_SIZE = (SIXTRL_SIZE_T)0u;

    SIXTRL_SIZE_T predicted_capacity = ZERO_SIZE;

    if( ( num_particles > ZERO_SIZE ) && ( ptr_chunk_size != 0 ) &&
        ( ptr_alignment != 0 ) )
    {
        SIXTRL_SIZE_T double_elem_length  = sizeof( double  ) * num_particles;
        SIXTRL_SIZE_T int64_elem_length   = sizeof( SIXTRL_INT64_T ) * num_particles;

        SIXTRL_SIZE_T chunk_size = *ptr_chunk_size;
        SIXTRL_SIZE_T alignment = *ptr_alignment;

        assert( ptr_chunk_size != ptr_alignment );

        if( chunk_size == ZERO_SIZE )
        {
            chunk_size = NS( PARTICLES_DEFAULT_MEMPOOL_CHUNK_SIZE );
        }

        assert( chunk_size <= NS( PARTICLES_MAX_ALIGNMENT ) );

        if( alignment == ZERO_SIZE )
        {
            alignment = NS( PARTICLES_DEFAULT_MEMPOOL_ALIGNMENT );
        }

        if( alignment < chunk_size )
        {
            alignment = chunk_size;
        }

        if( ( alignment % chunk_size ) != ZERO_SIZE )
        {
            alignment =
                chunk_size + ( ( alignment / chunk_size ) * chunk_size );
        }

        assert( ( alignment <= NS( PARTICLES_MAX_ALIGNMENT ) ) &&
                ( alignment >= chunk_size ) &&
                ( ( alignment % chunk_size ) == ZERO_SIZE ) );

        /* ----------------------------------------------------------------- */
        
        if( make_packed )
        {
            /* Packing information: 
             * - 1 x SIXTRL_UINT64_T .... length of the whole serialized slab of memory
             *                     including the length indicator itself. I.e. 
             *                     ( current_pos + length bytes ) == first byte
             *                     past the serialized item
             * - 1 x SIXTRL_UINT64_T .... indicator, i.e. what type of element has been 
             *                     packed; note: Particles = 1
             * - 1 x SIXTRL_UINT64_T .... nelem, i.e. number of elements to be serialized
             * - 1 x SIXTRL_UINT64_T .... nattr, i.e. the number of attributes 
             *                     that have been packed per element -> should 
             *                     be NS(PARTICLES_NUM_OF_ATTRIBUTES), store for
             *                     format versioning reasons
             * - num x SIXTRL_UINT64_T .. nattr x offsets, i.e. for each of the num 
             *                     elemens an offset in bytes on where the 
             *                     data is stored.
             *                     Note: the offset is calculated relative to 
             *                     the current_pos, i.e. the pointer pointing 
             *                     to the length indicator. The minimum 
             *                     offset for Particles is therefore 
             *                     NS(PARTICLES_PACK_BLOCK_LENGTH) */
            
            SIXTRL_SIZE_T pack_info_length = 
                sizeof( SIXTRL_UINT64_T ) + sizeof( SIXTRL_UINT64_T ) + sizeof( SIXTRL_UINT64_T ) +
                sizeof( SIXTRL_UINT64_T ) + 
                sizeof( SIXTRL_UINT64_T ) * NS(PARTICLES_NUM_OF_ATTRIBUTES);
            
            SIXTRL_SIZE_T const temp = ( pack_info_length / alignment ) * alignment;
            
            if( temp < pack_info_length )
            {
                pack_info_length = temp + alignment;
            }
            
            assert( ( pack_info_length % alignment ) == ZERO_SIZE );            
            assert( pack_info_length >= NS( PARTICLES_PACK_BLOCK_LENGTH ) );
            
            predicted_capacity += pack_info_length;
        }
                
        /* ----------------------------------------------------------------- */

        SIXTRL_SIZE_T temp = ( double_elem_length / ( alignment ) ) * ( alignment );

        if( temp < double_elem_length )
        {
            temp += alignment;
        }

        double_elem_length = temp;

        /* ----------------------------------------------------------------- */

        temp = ( int64_elem_length / ( alignment ) ) * ( alignment );

        if( temp < int64_elem_length )
        {
            temp += alignment;
        }

        int64_elem_length = temp;

        /* ----------------------------------------------------------------- */

        assert( ( double_elem_length > ZERO_SIZE ) &&
                ( ( double_elem_length % alignment ) == ZERO_SIZE ) &&
                ( int64_elem_length > ZERO_SIZE ) &&
                ( ( int64_elem_length % alignment ) == ZERO_SIZE ) );

        predicted_capacity +=
            NS( PARTICLES_NUM_OF_DOUBLE_ELEMENTS ) * double_elem_length +
            NS( PARTICLES_NUM_OF_INT64_ELEMENTS  ) * int64_elem_length;

        /* By aligning every part of the Particles struct to the
         * required alignment, we can ensure that the whole block used for
         * packing the data will be aligned internally. We have, however, to
         * account for the possibility that the initial address of the whole
         * memory region will be not properly aligned -> increase the capacity
         * by the alignment to allow for some wiggle room here */

        predicted_capacity += alignment;

        *ptr_chunk_size = chunk_size;
        *ptr_alignment = alignment;
    }

    return predicted_capacity;
}

/* ------------------------------------------------------------------------ */

bool NS( Particles_is_packed )( const struct NS( Particles ) *
                                const SIXTRL_RESTRICT p )
{
    return ( ( p != 0 ) && ( ( p->flags & NS( PARTICLES_FLAGS_PACKED ) ) ==
                             NS( PARTICLES_FLAGS_PACKED ) ) );
}

bool NS( Particles_manages_own_memory )( const struct NS( Particles ) *
                                         const SIXTRL_RESTRICT p )
{
    return ( ( p != 0 ) && ( p->ptr_mem_context != 0 ) &&
             ( ( p->flags & NS( PARTICLES_FLAGS_OWNS_MEMORY ) ) ==
               NS( PARTICLES_FLAGS_OWNS_MEMORY ) ) );
}

bool NS( Particles_uses_mempool )( const struct NS( Particles ) *
                                   const SIXTRL_RESTRICT p )
{
    return ( ( p != 0 ) && ( p->ptr_mem_context != 0 ) &&
             ( ( p->flags & NS( PARTICLES_FLAGS_MEM_CTX_MEMPOOL ) ) ==
               NS( PARTICLES_FLAGS_MEM_CTX_MEMPOOL ) ) );
}

bool NS( Particles_uses_single_particle )( const struct NS( Particles ) *
                                           const SIXTRL_RESTRICT p )
{
    return ( ( p != 0 ) && ( p->ptr_mem_context != 0 ) &&
             ( ( p->flags & NS( PARTICLES_FLAGS_MEM_CTX_SINGLEPARTICLE ) ) ==
               NS( PARTICLES_FLAGS_MEM_CTX_SINGLEPARTICLE ) ) );
}

bool NS( Particles_uses_flat_memory )( const NS(Particles )* const SIXTRL_RESTRICT p )
{
    return ( ( p != 0 ) && ( p->ptr_mem_context != 0 ) &&
             ( ( p->flags & NS( PARTICLES_FLAGS_MEM_CTX_FLAT_MEMORY ) ) ==
                 NS( PARTICLES_FLAGS_MEM_CTX_FLAT_MEMORY ) ) );
}

NS( MemPool )
const* NS( Particles_get_const_mem_pool )( const struct NS( Particles ) *
                                           const SIXTRL_RESTRICT p )
{
    NS( MemPool ) const* ptr_mem_pool = 0;

    if( ( p != 0 ) && ( p->ptr_mem_context != 0 ) &&
        ( ( p->flags & NS( PARTICLES_FLAGS_MEM_CTX_MEMPOOL ) ) ==
          NS( PARTICLES_FLAGS_MEM_CTX_MEMPOOL ) ) )
    {
        ptr_mem_pool = (NS( MemPool ) const*)p->ptr_mem_context;
    }

    return ptr_mem_pool;
}

NS( SingleParticle )
const* NS( Particles_get_const_base_single_particle )(
    const struct NS( Particles ) * const SIXTRL_RESTRICT p )
{
    NS( SingleParticle ) const* ptr_single_particle = 0;

    if( ( p != 0 ) && ( p->ptr_mem_context != 0 ) &&
        ( ( p->flags & NS( PARTICLES_FLAGS_MEM_CTX_SINGLEPARTICLE ) ) ==
          NS( PARTICLES_FLAGS_MEM_CTX_SINGLEPARTICLE ) ) )
    {
        ptr_single_particle = (NS( SingleParticle ) const*)p->ptr_mem_context;
    }

    return ptr_single_particle;
}

unsigned char const* NS( Particles_get_const_flat_memory )(
    const struct NS( Particles ) * const SIXTRL_RESTRICT p )
{
    unsigned char const* ptr_flat_mem_block = 0;
    
    if( ( p != 0 ) && ( p->ptr_mem_context != 0 ) &&
        ( ( p->flags & NS( PARTICLES_FLAGS_MEM_CTX_FLAT_MEMORY ) ) ==
          NS( PARTICLES_FLAGS_MEM_CTX_FLAT_MEMORY ) ) )
    {
        ptr_flat_mem_block = ( unsigned char const* )p->ptr_mem_context;
    }
    
    return ptr_flat_mem_block;
}

/* ------------------------------------------------------------------------ */

bool NS( Particles_map_to_single_particle )( struct NS( Particles ) *
                                                 SIXTRL_RESTRICT particles,
                                             struct NS( SingleParticle ) *
                                                 SIXTRL_RESTRICT single )
{
    bool success = false;

    if( ( single != 0 ) && ( particles != 0 ) )
    {
        NS( Particles_assign_ptr_to_q0 )( particles, &single->q0 );
        NS( Particles_assign_ptr_to_mass0 )( particles, &single->mass0 );
        NS( Particles_assign_ptr_to_beta0 )( particles, &single->beta0 );
        NS( Particles_assign_ptr_to_gamma0 )( particles, &single->gamma0 );
        NS( Particles_assign_ptr_to_p0c )( particles, &single->p0c );

        NS( Particles_assign_ptr_to_particle_id )( particles, &single->partid );
        NS( Particles_assign_ptr_to_lost_at_element_id )
        ( particles, &single->elemid );

        NS( Particles_assign_ptr_to_lost_at_turn )( particles, &single->turn );
        NS( Particles_assign_ptr_to_state )( particles, &single->state );

        NS( Particles_assign_ptr_to_s )( particles, &single->s );
        NS( Particles_assign_ptr_to_x )( particles, &single->x );
        NS( Particles_assign_ptr_to_y )( particles, &single->y );
        NS( Particles_assign_ptr_to_px )( particles, &single->px );
        NS( Particles_assign_ptr_to_py )( particles, &single->py );
        NS( Particles_assign_ptr_to_sigma )( particles, &single->sigma );

        NS( Particles_assign_ptr_to_psigma )( particles, &single->psigma );
        NS( Particles_assign_ptr_to_delta )( particles, &single->delta );
        NS( Particles_assign_ptr_to_rpp )( particles, &single->rpp );
        NS( Particles_assign_ptr_to_rvv )( particles, &single->rvv );
        NS( Particles_assign_ptr_to_chi )( particles, &single->chi );
        
        NS( Particles_set_ptr_mem_begin )( particles, single );
        
        success = true;
    }

    return success;
}

bool NS( Particles_map_to_mempool )( struct NS( Particles ) *
                                         SIXTRL_RESTRICT particles,
                                     struct NS( MemPool ) *
                                         SIXTRL_RESTRICT pool,
                                     SIXTRL_SIZE_T npart,
                                     SIXTRL_SIZE_T alignment, 
                                     bool make_packed )
{
    bool success = false;
    
    static SIXTRL_SIZE_T const ZERO_SIZE = (SIXTRL_SIZE_T)0u;

    if( ( pool != 0 ) && ( particles != 0 ) && ( npart > ZERO_SIZE ) &&
        ( npart <= ( SIXTRL_SIZE_T )UINT64_MAX ) &&
        ( alignment > ZERO_SIZE ) && 
        ( alignment <= NS(PARTICLES_MAX_ALIGNMENT) ) )
    {
        unsigned char* particles_begin = 0;
        
        NS( Particles ) rollback_particles = *particles;
        NS( MemPool ) rollback_mem_pool = *pool;

        /* try to not use goto's -> rewrite with a dummy do-while loop.
         * change back to plain goto's in case this is less
         * clear/more complicated */

        do
        {
            NS( AllocResult ) add_result;
            
            SIXTRL_UINT64_T const num_of_attributes =                 
                NS(PARTICLES_NUM_OF_DOUBLE_ELEMENTS) +
                NS(PARTICLES_NUM_OF_INT64_ELEMENTS);
            
            unsigned char* ptr_attr = 0;
            unsigned char* pack_info_it = 0;
            unsigned char* ptr_length_info = 0;
            
            SIXTRL_SIZE_T const u64_stride = sizeof( SIXTRL_UINT64_T );
            SIXTRL_SIZE_T const min_double_member_length = sizeof( double  ) * npart;
            SIXTRL_SIZE_T const min_int64_member_length  = sizeof( SIXTRL_INT64_T ) * npart;
            
            assert( ( ( alignment % sizeof( double   ) ) == ZERO_SIZE ) &&
                    ( ( alignment % sizeof( SIXTRL_INT64_T  ) ) == ZERO_SIZE ) );
            
            NS( AllocResult_preset )( &add_result );

            /* ------------------------------------------------------------- */
            /* Pack information: */
            
            if( make_packed )
            {
                SIXTRL_UINT64_T const pack_indicator = NS(PARTICLES_PACK_INDICATOR);
                SIXTRL_UINT64_T const num_of_particles = ( SIXTRL_UINT64_T )npart;
                
                SIXTRL_SIZE_T const offset_info_block_len = u64_stride + 
                    u64_stride + u64_stride + u64_stride + 
                    u64_stride * num_of_attributes;
                    
                add_result = NS(MemPool_append_aligned)(
                    pool, offset_info_block_len, alignment );
                
                if( ( !NS( AllocResult_valid )( &add_result ) ) ||
                    ( !NS( AllocResult_is_aligned)( &add_result, alignment ) ) )
                {
                    break;                
                }
                
                particles_begin = NS( AllocResult_get_pointer )( &add_result );
                ptr_length_info = particles_begin;
                pack_info_it    = particles_begin;
                
                memset( pack_info_it, ( int )0, u64_stride );
                pack_info_it    = pack_info_it + u64_stride;
                
                memcpy( pack_info_it, &pack_indicator, u64_stride );
                pack_info_it = pack_info_it + u64_stride;
                
                memcpy( pack_info_it, &num_of_particles, u64_stride );
                pack_info_it = pack_info_it + u64_stride;
                
                memcpy( pack_info_it, &num_of_attributes, u64_stride );
                pack_info_it = pack_info_it + u64_stride;
            }
            
            assert( ( pack_info_it != 0 ) || ( !make_packed ) );
            
            /* ------------------------------------------------------------- */
            /* q0: */

            add_result = NS( MemPool_append_aligned )(
                pool, min_double_member_length, alignment );
            
            assert( NS( AllocResult_valid )( &add_result ) );                
            
            ptr_attr = NS( AllocResult_get_pointer )( &add_result );
            
            /* If not packed, we have not yet written anything to the mempool,
             * i.e. we have to take the "beginning" of the memory section 
             * from the first member */
            
            if( !make_packed )  particles_begin = ptr_attr;            
            assert( NS(AllocResult_is_aligned)( &add_result, alignment ) );
            
            NS( Particles_assign_ptr_to_q0 )( particles, ( double*) ptr_attr );            
            
            if( make_packed )
            {
                ptrdiff_t const dist = ( ptr_attr - particles_begin );
                
                if( dist > 0 )
                {
                    SIXTRL_UINT64_T const attr_offset = ( SIXTRL_UINT64_T )dist;                    
                    memcpy( pack_info_it, &attr_offset, u64_stride );
                    pack_info_it = pack_info_it + u64_stride;
                }
                else
                {
                    break;
                }
            }

            /* ------------------------------------------------------------- */
            /* mass0: */

            add_result = NS( MemPool_append_aligned )(
                pool, min_double_member_length, alignment );
            
            assert( ( NS( AllocResult_valid )( &add_result ) ) &&
                    ( NS(AllocResult_is_aligned)( &add_result, alignment ) ) );
            
            ptr_attr = NS( AllocResult_get_pointer )( &add_result );
            NS( Particles_assign_ptr_to_mass0 )( particles, ( double*)ptr_attr );
            
            if( make_packed )
            {
                ptrdiff_t const dist = ( ptr_attr - particles_begin );
                
                if( dist > 0 )
                {
                    SIXTRL_UINT64_T const attr_offset = ( SIXTRL_UINT64_T )dist;                    
                    memcpy( pack_info_it, &attr_offset, u64_stride );
                    pack_info_it = pack_info_it + u64_stride;
                }
                else
                {
                    break;
                }
            }

            /* ------------------------------------------------------------- */
            /* beta0: */

            add_result = NS( MemPool_append_aligned )(
                pool, min_double_member_length, alignment );
            
            assert( ( NS( AllocResult_valid )( &add_result ) ) &&
                    ( NS(AllocResult_is_aligned)( &add_result, alignment ) ) );
            
            ptr_attr = NS( AllocResult_get_pointer )( &add_result );
            NS( Particles_assign_ptr_to_beta0 )( particles, ( double*)ptr_attr );
            
            if( make_packed )
            {
                ptrdiff_t const dist = ( ptr_attr - particles_begin );
                
                if( dist > 0 )
                {
                    SIXTRL_UINT64_T const attr_offset = ( SIXTRL_UINT64_T )dist;                    
                    memcpy( pack_info_it, &attr_offset, u64_stride );
                    pack_info_it = pack_info_it + u64_stride;
                }
                else
                {
                    break;
                }
            }
            
            /* ------------------------------------------------------------- */
            /* gamma0: */

            add_result = NS( MemPool_append_aligned )(
                pool, min_double_member_length, alignment );
            
            assert( ( NS( AllocResult_valid )( &add_result ) ) &&
                    ( NS(AllocResult_is_aligned)( &add_result, alignment ) ) );
            
            ptr_attr = NS( AllocResult_get_pointer )( &add_result );
            NS( Particles_assign_ptr_to_gamma0 )( particles, ( double*)ptr_attr );
            
            if( make_packed )
            {
                ptrdiff_t const dist = ( ptr_attr - particles_begin );
                
                if( dist > 0 )
                {
                    SIXTRL_UINT64_T const attr_offset = ( SIXTRL_UINT64_T )dist;                    
                    memcpy( pack_info_it, &attr_offset, u64_stride );
                    pack_info_it = pack_info_it + u64_stride;
                }
                else
                {
                    break;
                }
            }
            
            /* ------------------------------------------------------------- */
            /* p0c: */

            add_result = NS( MemPool_append_aligned )(
                pool, min_double_member_length, alignment );
            
            assert( ( NS( AllocResult_valid )( &add_result ) ) &&
                    ( NS(AllocResult_is_aligned)( &add_result, alignment ) ) );
            
            ptr_attr = NS( AllocResult_get_pointer )( &add_result );
            NS( Particles_assign_ptr_to_p0c )( particles, ( double*)ptr_attr );
            
            if( make_packed )
            {
                ptrdiff_t const dist = ( ptr_attr - particles_begin );
                
                if( dist > 0 )
                {
                    SIXTRL_UINT64_T const attr_offset = ( SIXTRL_UINT64_T )dist;                    
                    memcpy( pack_info_it, &attr_offset, u64_stride );
                    pack_info_it = pack_info_it + u64_stride;
                }
                else
                {
                    break;
                }
            }
            
            /* ------------------------------------------------------------- */
            /* partid: */

            add_result = NS( MemPool_append_aligned )(
                pool, min_int64_member_length, alignment );
            
            assert( ( NS( AllocResult_valid )( &add_result ) ) &&
                    ( NS(AllocResult_is_aligned)( &add_result, alignment ) ) );
            
            ptr_attr = NS( AllocResult_get_pointer )( &add_result );
            NS( Particles_assign_ptr_to_particle_id )( 
                particles, ( SIXTRL_INT64_T* )ptr_attr );
            
            if( make_packed )
            {
                ptrdiff_t const dist = ( ptr_attr - particles_begin );
                
                if( dist > 0 )
                {
                    SIXTRL_UINT64_T const attr_offset = ( SIXTRL_UINT64_T )dist;                    
                    memcpy( pack_info_it, &attr_offset, u64_stride );
                    pack_info_it = pack_info_it + u64_stride;
                }
                else
                {
                    break;
                }
            }

            /* ------------------------------------------------------------- */
            /* elemid: */

            add_result = NS( MemPool_append_aligned )(
                pool, min_int64_member_length, alignment );
            
            assert( ( NS( AllocResult_valid )( &add_result ) ) &&
                    ( NS(AllocResult_is_aligned)( &add_result, alignment ) ) );
            
            ptr_attr = NS( AllocResult_get_pointer )( &add_result );
            NS( Particles_assign_ptr_to_lost_at_element_id)( 
                particles, ( SIXTRL_INT64_T* )ptr_attr );
            
            if( make_packed )
            {
                ptrdiff_t const dist = ( ptr_attr - particles_begin );
                
                if( dist > 0 )
                {
                    SIXTRL_UINT64_T const attr_offset = ( SIXTRL_UINT64_T )dist;                    
                    memcpy( pack_info_it, &attr_offset, u64_stride );
                    pack_info_it = pack_info_it + u64_stride;
                }
                else
                {
                    break;
                }
            }
            
            /* ------------------------------------------------------------- */
            /* turn: */

            add_result = NS( MemPool_append_aligned )(
                pool, min_int64_member_length, alignment );
            
            assert( ( NS( AllocResult_valid )( &add_result ) ) &&
                    ( NS(AllocResult_is_aligned)( &add_result, alignment ) ) );
            
            ptr_attr = NS( AllocResult_get_pointer )( &add_result );
            NS( Particles_assign_ptr_to_lost_at_turn )( 
                particles, ( SIXTRL_INT64_T* )ptr_attr );
            
            if( make_packed )
            {
                ptrdiff_t const dist = ( ptr_attr - particles_begin );
                
                if( dist > 0 )
                {
                    SIXTRL_UINT64_T const attr_offset = ( SIXTRL_UINT64_T )dist;                    
                    memcpy( pack_info_it, &attr_offset, u64_stride );
                    pack_info_it = pack_info_it + u64_stride;
                }
                else
                {
                    break;
                }
            }
            
            /* ------------------------------------------------------------- */
            /* state: */

            add_result = NS( MemPool_append_aligned )(
                pool, min_int64_member_length, alignment );
            
            if( !NS( AllocResult_valid )( &add_result ) ) break;
            
            assert( ( NS( AllocResult_valid )( &add_result ) ) &&
                    ( NS(AllocResult_is_aligned)( &add_result, alignment ) ) );
            
            ptr_attr = NS( AllocResult_get_pointer )( &add_result );
            NS( Particles_assign_ptr_to_state )( 
                particles, ( SIXTRL_INT64_T* )ptr_attr );
            
            if( make_packed )
            {
                ptrdiff_t const dist = ( ptr_attr - particles_begin );
                
                if( dist > 0 )
                {
                    SIXTRL_UINT64_T const attr_offset = ( SIXTRL_UINT64_T )dist;                    
                    memcpy( pack_info_it, &attr_offset, u64_stride );
                    pack_info_it = pack_info_it + u64_stride;
                }
                else
                {
                    break;
                }
            }
            
            /* ------------------------------------------------------------- */
            /* s: */

            add_result = NS( MemPool_append_aligned )(
                pool, min_double_member_length, alignment );
            
            if( !NS( AllocResult_valid )( &add_result ) ) break; 
            
            assert( ( NS( AllocResult_valid )( &add_result ) ) &&
                    ( NS(AllocResult_is_aligned)( &add_result, alignment ) ) );
            
            ptr_attr = NS( AllocResult_get_pointer )( &add_result );
            NS( Particles_assign_ptr_to_s )( particles, ( double*)ptr_attr );
            
            if( make_packed )
            {
                ptrdiff_t const dist = ( ptr_attr - particles_begin );
                
                if( dist > 0 )
                {
                    SIXTRL_UINT64_T const attr_offset = ( SIXTRL_UINT64_T )dist;                    
                    memcpy( pack_info_it, &attr_offset, u64_stride );
                    pack_info_it = pack_info_it + u64_stride;
                }
                else
                {
                    break;
                }
            }
            
            /* ------------------------------------------------------------- */
            /* x: */

            add_result = NS( MemPool_append_aligned )(
                pool, min_double_member_length, alignment );
            
            if( !NS( AllocResult_valid )( &add_result ) ) break; 
            
            assert( ( NS( AllocResult_valid )( &add_result ) ) &&
                    ( NS(AllocResult_is_aligned)( &add_result, alignment ) ) );
            
            ptr_attr = NS( AllocResult_get_pointer )( &add_result );
            NS( Particles_assign_ptr_to_x )( particles, ( double*)ptr_attr );
            
            if( make_packed )
            {
                ptrdiff_t const dist = ( ptr_attr - particles_begin );
                
                if( dist > 0 )
                {
                    SIXTRL_UINT64_T const attr_offset = ( SIXTRL_UINT64_T )dist;                    
                    memcpy( pack_info_it, &attr_offset, u64_stride );
                    pack_info_it = pack_info_it + u64_stride;
                }
                else
                {
                    break;
                }
            }
            

            /* ------------------------------------------------------------- */
            /* y: */

            add_result = NS( MemPool_append_aligned )(
                pool, min_double_member_length, alignment );
            
            if( !NS( AllocResult_valid )( &add_result ) ) break;  
            
            assert( ( NS( AllocResult_valid )( &add_result ) ) &&
                    ( NS(AllocResult_is_aligned)( &add_result, alignment ) ) );
            
            ptr_attr = NS( AllocResult_get_pointer )( &add_result );
            NS( Particles_assign_ptr_to_y )( particles, ( double*)ptr_attr );
            
            if( make_packed )
            {
                ptrdiff_t const dist = ( ptr_attr - particles_begin );
                
                if( dist > 0 )
                {
                    SIXTRL_UINT64_T const attr_offset = ( SIXTRL_UINT64_T )dist;                    
                    memcpy( pack_info_it, &attr_offset, u64_stride );
                    pack_info_it = pack_info_it + u64_stride;
                }
                else
                {
                    break;
                }
            }
            
            /* ------------------------------------------------------------- */
            /* px: */

            add_result = NS( MemPool_append_aligned )(
                pool, min_double_member_length, alignment );
            
            if( !NS( AllocResult_valid )( &add_result ) ) break; 
            
            assert( ( NS( AllocResult_valid )( &add_result ) ) &&
                    ( NS(AllocResult_is_aligned)( &add_result, alignment ) ) );
            
            ptr_attr = NS( AllocResult_get_pointer )( &add_result );
            NS( Particles_assign_ptr_to_px )( particles, ( double*)ptr_attr );
            
            if( make_packed )
            {
                ptrdiff_t const dist = ( ptr_attr - particles_begin );
                
                if( dist > 0 )
                {
                    SIXTRL_UINT64_T const attr_offset = ( SIXTRL_UINT64_T )dist;                    
                    memcpy( pack_info_it, &attr_offset, u64_stride );
                    pack_info_it = pack_info_it + u64_stride;
                }
                else
                {
                    break;
                }
            }
            
            /* ------------------------------------------------------------- */
            /* py: */

            add_result = NS( MemPool_append_aligned )(
                pool, min_double_member_length, alignment );
            
            if( !NS( AllocResult_valid )( &add_result ) ) break;  
            
            assert( ( NS( AllocResult_valid )( &add_result ) ) &&
                    ( NS(AllocResult_is_aligned)( &add_result, alignment ) ) );
            
            ptr_attr = NS( AllocResult_get_pointer )( &add_result );
            NS( Particles_assign_ptr_to_py )( particles, ( double*)ptr_attr );
            
            if( make_packed )
            {
                ptrdiff_t const dist = ( ptr_attr - particles_begin );
                
                if( dist > 0 )
                {
                    SIXTRL_UINT64_T const attr_offset = ( SIXTRL_UINT64_T )dist;                    
                    memcpy( pack_info_it, &attr_offset, u64_stride );
                    pack_info_it = pack_info_it + u64_stride;
                }
                else
                {
                    break;
                }
            }
            
            /* ------------------------------------------------------------- */
            /* sigma: */

            add_result = NS( MemPool_append_aligned )(
                pool, min_double_member_length, alignment );
            
            if( !NS( AllocResult_valid )( &add_result ) ) break;
            
            assert( ( NS( AllocResult_valid )( &add_result ) ) &&
                    ( NS(AllocResult_is_aligned)( &add_result, alignment ) ) );
            
            ptr_attr = NS( AllocResult_get_pointer )( &add_result );
            NS( Particles_assign_ptr_to_sigma )( particles, ( double*)ptr_attr );
            
            if( make_packed )
            {
                ptrdiff_t const dist = ( ptr_attr - particles_begin );
                
                if( dist > 0 )
                {
                    SIXTRL_UINT64_T const attr_offset = ( SIXTRL_UINT64_T )dist;                    
                    memcpy( pack_info_it, &attr_offset, u64_stride );
                    pack_info_it = pack_info_it + u64_stride;
                }
                else
                {
                    break;
                }
            }
            
            /* ------------------------------------------------------------- */
            /* psigma: */

            add_result = NS( MemPool_append_aligned )(
                pool, min_double_member_length, alignment );
            
            assert( ( NS( AllocResult_valid )( &add_result ) ) &&
                    ( NS(AllocResult_is_aligned)( &add_result, alignment ) ) );
            
            ptr_attr = NS( AllocResult_get_pointer )( &add_result );
            NS( Particles_assign_ptr_to_psigma )( particles, ( double*)ptr_attr );
            
            if( make_packed )
            {
                ptrdiff_t const dist = ( ptr_attr - particles_begin );
                
                if( dist > 0 )
                {
                    SIXTRL_UINT64_T const attr_offset = ( SIXTRL_UINT64_T )dist;                    
                    memcpy( pack_info_it, &attr_offset, u64_stride );
                    pack_info_it = pack_info_it + u64_stride;
                }
                else
                {
                    break;
                }
            }
            

            /* ------------------------------------------------------------- */
            /* delta: */

            add_result = NS( MemPool_append_aligned )(
                pool, min_double_member_length, alignment );
            
            assert( ( NS( AllocResult_valid )( &add_result ) ) &&
                    ( NS(AllocResult_is_aligned)( &add_result, alignment ) ) );
            
            ptr_attr = NS( AllocResult_get_pointer )( &add_result );
            NS( Particles_assign_ptr_to_delta )( particles, ( double*)ptr_attr );
            
            if( make_packed )
            {
                ptrdiff_t const dist = ( ptr_attr - particles_begin );
                
                if( dist > 0 )
                {
                    SIXTRL_UINT64_T const attr_offset = ( SIXTRL_UINT64_T )dist;                    
                    memcpy( pack_info_it, &attr_offset, u64_stride );
                    pack_info_it = pack_info_it + u64_stride;
                }
                else
                {
                    break;
                }
            }
            

            /* ------------------------------------------------------------- */
            /* rpp: */

            add_result = NS( MemPool_append_aligned )(
                pool, min_double_member_length, alignment );
            
            if( !NS( AllocResult_valid )( &add_result ) ) break; 
            
            assert( ( NS( AllocResult_valid )( &add_result ) ) &&
                    ( NS(AllocResult_is_aligned)( &add_result, alignment ) ) );
            
            ptr_attr = NS( AllocResult_get_pointer )( &add_result );
            NS( Particles_assign_ptr_to_rpp )( particles, ( double*)ptr_attr );
            
            if( make_packed )
            {
                ptrdiff_t const dist = ( ptr_attr - particles_begin );
                
                if( dist > 0 )
                {
                    SIXTRL_UINT64_T const attr_offset = ( SIXTRL_UINT64_T )dist;                    
                    memcpy( pack_info_it, &attr_offset, u64_stride );
                    pack_info_it = pack_info_it + u64_stride;
                }
                else
                {
                    break;
                }
            }
            
            /* ------------------------------------------------------------- */
            /* rvv: */

            add_result = NS( MemPool_append_aligned )(
                pool, min_double_member_length, alignment );
            
            if( !NS( AllocResult_valid )( &add_result ) ) break;
            
            assert( ( NS( AllocResult_valid )( &add_result ) ) &&
                    ( NS(AllocResult_is_aligned)( &add_result, alignment ) ) );
            
            ptr_attr = NS( AllocResult_get_pointer )( &add_result );
            NS( Particles_assign_ptr_to_rvv )( particles, ( double*)ptr_attr );
            
            if( make_packed )
            {
                ptrdiff_t const dist = ( ptr_attr - particles_begin );
                
                if( dist > 0 )
                {
                    SIXTRL_UINT64_T const attr_offset = ( SIXTRL_UINT64_T )dist;                    
                    memcpy( pack_info_it, &attr_offset, u64_stride );
                    pack_info_it = pack_info_it + u64_stride;
                }
                else
                {
                    break;
                }
            }
            
            /* ------------------------------------------------------------- */
            /* chi: */

            add_result = NS( MemPool_append_aligned )(
                pool, min_double_member_length, alignment );
            
            if( !NS( AllocResult_valid )( &add_result ) ) break; 
            
            assert( ( NS( AllocResult_valid )( &add_result ) ) &&
                    ( NS(AllocResult_is_aligned)( &add_result, alignment ) ) );
            
            ptr_attr = NS( AllocResult_get_pointer )( &add_result );
            NS( Particles_assign_ptr_to_chi )( particles, ( double*)ptr_attr );
            
            if( make_packed )
            {
                ptrdiff_t const dist = ( ptr_attr - particles_begin );
                
                if( dist > 0 )
                {
                    SIXTRL_UINT64_T const attr_offset = ( SIXTRL_UINT64_T )dist;                    
                    memcpy( pack_info_it, &attr_offset, u64_stride );
                    pack_info_it = pack_info_it + u64_stride;
                }
                else
                {
                    break;
                }
            }

            /* ============================================================= */
            
            if( make_packed )
            {
                ptrdiff_t temp_length;
                SIXTRL_UINT64_T  length = ( SIXTRL_UINT64_T )0u;
                
                unsigned char* end_ptr = 
                    NS(MemPool_get_next_begin_pointer)(pool, alignment );
                                    
                assert( ( ptr_length_info != 0 ) && ( end_ptr != 0 ) &&
                        ( particles_begin != 0 ) );
                
                temp_length = ( end_ptr - particles_begin );
                
                assert( temp_length > 0 );
                
                length = ( SIXTRL_UINT64_T )temp_length;
                memcpy( ptr_length_info, &length, u64_stride );                
            }
            
            success = true;
            
        } while( false );

        if( success )
        {
            assert( particles_begin != 0 );
            NS(Particles_set_ptr_mem_begin)( particles, particles_begin );
        }
        else
        {
            *particles = rollback_particles;
            *pool = rollback_mem_pool;
        }
    }

    return success;
}

/* ------------------------------------------------------------------------- */

NS( Particles ) * NS( Particles_new )( SIXTRL_SIZE_T npart )
{
    NS( Particles )* particles = 0;
    NS( MemPool )* ptr_mem_pool = 0;

    SIXTRL_SIZE_T chunk_size = NS( PARTICLES_DEFAULT_MEMPOOL_CHUNK_SIZE );
    SIXTRL_SIZE_T alignment = NS( PARTICLES_DEFAULT_MEMPOOL_ALIGNMENT );

    SIXTRL_SIZE_T const required_capacity =
        NS( Particles_predict_required_capacity )( 
            npart, &chunk_size, &alignment, true );

    if( ( required_capacity > (SIXTRL_SIZE_T)0u ) && ( (SIXTRL_UINT64_T)npart < UINT64_MAX ) )
    {
        bool success = false;

        ptr_mem_pool = (NS( MemPool )*)malloc( sizeof( NS( MemPool ) ) );

        if( ptr_mem_pool != 0 )
        {
            NS( MemPool_init )( ptr_mem_pool, required_capacity, chunk_size );

            if( ( NS( MemPool_get_buffer )( ptr_mem_pool ) == 0 ) ||
                ( NS( MemPool_get_chunk_size )( ptr_mem_pool ) !=
                  chunk_size ) ||
                ( NS( MemPool_get_capacity )( ptr_mem_pool ) <
                  required_capacity ) )
            {
                NS( MemPool_free )( ptr_mem_pool );
                free( ptr_mem_pool );
                ptr_mem_pool = 0;
            }
        }

        particles = NS( Particles_preset )(
            (NS( Particles )*)malloc( sizeof( NS( Particles ) ) ) );

        if( ( ptr_mem_pool != 0 ) && ( particles != 0 ) &&
            ( NS( Particles_map_to_mempool )(
                particles, ptr_mem_pool, npart, alignment, true ) ) )
        {
            SIXTRL_UINT64_T flags = NS( PARTICLES_FLAGS_PACKED ) |
                             NS( PARTICLES_FLAGS_OWNS_MEMORY ) |
                             NS( PARTICLES_FLAGS_MEM_CTX_MEMPOOL );

            SIXTRL_UINT64_T temp_alignment = (SIXTRL_UINT64_T)alignment;

            if( temp_alignment <= NS( PARTICLES_MAX_ALIGNMENT ) )
            {
                flags |= ( temp_alignment
                           << NS( PARTICLES_FLAGS_ALIGN_MASK_OFFSET_BITS ) );
            } else
            {
                /* this should not be necessary, but for paranoia's sake: */
                flags &= ~( NS( PARTICLES_FLAGS_ALIGN_MASK ) );
            }

            assert( NS(Particles_get_mem_begin )( particles ) != 0 );
            
            NS( Particles_set_flags )( particles, flags );
            NS( Particles_set_size )( particles, (SIXTRL_UINT64_T)npart );
            NS( Particles_set_ptr_mem_context )( particles, ptr_mem_pool );

            success = true;
        }

        if( !success )
        {
            if( ptr_mem_pool != 0 )
            {
                NS( MemPool_free )( ptr_mem_pool );
                free( ptr_mem_pool );
                ptr_mem_pool = 0;
            }

            if( particles != 0 )
            {
                NS( Particles_free )( particles );
                free( particles );
                particles = 0;
            }
        }

        assert(
            ( ( success ) && ( particles != 0 ) && ( ptr_mem_pool != 0 ) ) ||
            ( ( !success ) && ( particles == 0 ) && ( ptr_mem_pool == 0 ) ) );
    }

    return particles;
}

NS( Particles ) * NS( Particles_new_aligned )( 
    SIXTRL_SIZE_T const npart, SIXTRL_SIZE_T alignment )
{
    NS( Particles )* particles = 0;
    NS( MemPool )* ptr_mem_pool = 0;

    SIXTRL_SIZE_T chunk_size = NS( PARTICLES_DEFAULT_MEMPOOL_CHUNK_SIZE );

    SIXTRL_SIZE_T const required_capacity =
        NS( Particles_predict_required_capacity )( 
            npart, &chunk_size, &alignment, true );

    if( ( required_capacity > (SIXTRL_SIZE_T)0u ) && ( (SIXTRL_UINT64_T)npart < UINT64_MAX ) )
    {
        bool success = false;

        ptr_mem_pool = (NS( MemPool )*)malloc( sizeof( NS( MemPool ) ) );

        if( ptr_mem_pool != 0 )
        {
            NS( MemPool_init )( ptr_mem_pool, required_capacity, chunk_size );

            if( ( NS( MemPool_get_buffer )( ptr_mem_pool ) == 0 ) ||
                ( NS( MemPool_get_chunk_size )( ptr_mem_pool ) !=
                  chunk_size ) ||
                ( NS( MemPool_get_capacity )( ptr_mem_pool ) <
                  required_capacity ) )
            {
                NS( MemPool_free )( ptr_mem_pool );
                free( ptr_mem_pool );
                ptr_mem_pool = 0;
            }
        }

        particles = NS( Particles_preset )(
            (NS( Particles )*)malloc( sizeof( NS( Particles ) ) ) );

        if( ( ptr_mem_pool != 0 ) && ( particles != 0 ) &&
            ( NS( Particles_map_to_mempool )(
                particles, ptr_mem_pool, npart, alignment, true ) ) )
        {
            SIXTRL_UINT64_T flags = NS( PARTICLES_FLAGS_PACKED ) |
                             NS( PARTICLES_FLAGS_OWNS_MEMORY ) |
                             NS( PARTICLES_FLAGS_MEM_CTX_MEMPOOL );

            SIXTRL_UINT64_T temp_alignment = (SIXTRL_UINT64_T)alignment;

            if( temp_alignment <= NS( PARTICLES_MAX_ALIGNMENT ) )
            {
                flags |= ( temp_alignment
                           << NS( PARTICLES_FLAGS_ALIGN_MASK_OFFSET_BITS ) );
            } else
            {
                /* this should not be necessary, but for paranoia's sake: */
                flags &= ~( NS( PARTICLES_FLAGS_ALIGN_MASK ) );
            }

            assert( NS(Particles_get_mem_begin )( particles ) != 0 );
            
            NS( Particles_set_flags )( particles, flags );
            NS( Particles_set_size )( particles, (SIXTRL_UINT64_T)npart );
            NS( Particles_set_ptr_mem_context )( particles, ptr_mem_pool );

            success = true;
        }

        if( !success )
        {
            if( ptr_mem_pool != 0 )
            {
                NS( MemPool_free )( ptr_mem_pool );
                free( ptr_mem_pool );
                ptr_mem_pool = 0;
            }

            if( particles != 0 )
            {
                NS( Particles_free )( particles );
                free( particles );
                particles = 0;
            }
        }

        assert(
            ( ( success ) && ( particles != 0 ) && ( ptr_mem_pool != 0 ) ) ||
            ( ( !success ) && ( particles == 0 ) && ( ptr_mem_pool == 0 ) ) );
    }

    return particles;
}

/* ------------------------------------------------------------------------- */

NS( Particles ) *
    NS( Particles_new_on_mempool )( SIXTRL_SIZE_T npart,
                                    NS( MemPool ) * SIXTRL_RESTRICT pool )
{
    NS( Particles )* particles = 0;

    SIXTRL_SIZE_T chunk_size = NS( MemPool_get_chunk_size )( pool );
    SIXTRL_SIZE_T alignment = NS( PARTICLES_DEFAULT_MEMPOOL_ALIGNMENT );

    SIXTRL_SIZE_T const required_capacity =
        NS( Particles_predict_required_capacity )( 
            npart, &chunk_size, &alignment, true );

    if( ( required_capacity > (SIXTRL_SIZE_T)0u ) &&
        ( chunk_size == NS( MemPool_get_chunk_size )( pool ) ) &&
        ( chunk_size > (SIXTRL_SIZE_T)0u ) && ( alignment >= chunk_size ) &&
        ( (SIXTRL_UINT64_T)npart < UINT64_MAX ) )
    {
        bool success = false;

        particles = NS( Particles_preset )(
            (NS( Particles )*)malloc( sizeof( NS( Particles ) ) ) );

        if( ( particles != 0 ) && 
            ( NS( Particles_map_to_mempool )( 
                particles, pool, npart, alignment, true ) ) )
        {
            SIXTRL_UINT64_T flags = NS( PARTICLES_FLAGS_PACKED ) |
                             NS( PARTICLES_FLAGS_MEM_CTX_MEMPOOL );

            SIXTRL_UINT64_T temp_alignment = (SIXTRL_UINT64_T)alignment;

            if( temp_alignment <= NS( PARTICLES_MAX_ALIGNMENT ) )
            {
                flags |= ( temp_alignment
                           << NS( PARTICLES_FLAGS_ALIGN_MASK_OFFSET_BITS ) );
            } else
            {
                /* this should not be necessary, but for paranoia's sake: */
                flags &= ~( NS( PARTICLES_FLAGS_ALIGN_MASK ) );
            }

            assert( NS(Particles_get_mem_begin )( particles ) != 0 );
            
            NS( Particles_set_flags )( particles, flags );
            NS( Particles_set_size )( particles, (SIXTRL_UINT64_T)npart );
            NS( Particles_set_ptr_mem_context )( particles, pool );
            
            success = true;
        }

        if( ( !success ) && ( particles != 0 ) )
        {
            NS( Particles_free )( particles );
            free( particles );
            particles = 0;
        }

        assert( ( ( success ) && ( particles != 0 ) ) ||
                ( ( !success ) && ( particles == 0 ) ) );
    }

    return particles;
}

/* ------------------------------------------------------------------------ */

NS( Particles ) * NS( Particles_new_single )()
{
    typedef NS( Particles ) particle_t;
    typedef NS( SingleParticle ) single_t;

    particle_t* ptr_particles =
        NS( Particles_preset )( (particle_t*)malloc( sizeof( particle_t ) ) );

    single_t* ptr_single = (single_t*)malloc( sizeof( single_t ) );

    bool success = false;

    if( ( ptr_particles != 0 ) && ( ptr_single != 0 ) &&
        ( NS( Particles_map_to_single_particle )( ptr_particles,
                                                  ptr_single ) ) )
    {
        SIXTRL_UINT64_T flags = NS( PARTICLES_FLAGS_OWNS_MEMORY ) |
                         NS( PARTICLES_FLAGS_MEM_CTX_SINGLEPARTICLE );

        NS( Particles_set_flags )( ptr_particles, flags );
        NS( Particles_set_size )( ptr_particles, ( SIXTRL_UINT64_T )1u );
        NS( Particles_set_ptr_mem_context )( ptr_particles, ptr_single );

        success = true;
    }

    if( !success )
    {
        if( ptr_single != 0 )
        {
            free( ptr_single );
            ptr_single = 0;
        }

        if( ptr_particles != 0 )
        {
            free( ptr_particles );
            ptr_particles = 0;
        }
    }

    assert( ( ( success ) && ( ptr_particles != 0 ) && ( ptr_single != 0 ) ) ||
            ( ( !success ) && ( ptr_particles == 0 ) && ( ptr_single == 0 ) ) );

    return ptr_particles;
}

/* ------------------------------------------------------------------------ */

NS( Particles ) *
    NS( Particles_new_on_single )( NS( SingleParticle ) * ptr_single )
{
    typedef NS( Particles ) particle_t;

    particle_t* ptr_particles =
        NS( Particles_preset )( (particle_t*)malloc( sizeof( particle_t ) ) );

    bool success = false;

    if( ( ptr_particles != 0 ) && ( ptr_single != 0 ) &&
        ( NS( Particles_map_to_single_particle )( ptr_particles,
                                                  ptr_single ) ) )
    {
        SIXTRL_UINT64_T flags = NS( PARTICLES_FLAGS_MEM_CTX_SINGLEPARTICLE );

        NS( Particles_set_flags )( ptr_particles, flags );
        NS( Particles_set_size )( ptr_particles, ( SIXTRL_UINT64_T )1u );
        NS( Particles_set_ptr_mem_context )( ptr_particles, ptr_single );

        success = true;
    }

    if( !success )
    {
        if( ptr_particles != 0 )
        {
            free( ptr_particles );
            ptr_particles = 0;
        }
    }

    assert( ( ( success ) && ( ptr_particles != 0 ) ) ||
            ( ( !success ) && ( ptr_particles == 0 ) ) );

    return ptr_particles;
}

/* ------------------------------------------------------------------------ */

bool NS(Particles_unpack_read_header)(
    unsigned char const* SIXTRL_RESTRICT ptr_mem_begin, 
    SIXTRL_SIZE_T* SIXTRL_RESTRICT ptr_length, 
    SIXTRL_SIZE_T* SIXTRL_RESTRICT ptr_num_particles, 
    SIXTRL_SIZE_T* SIXTRL_RESTRICT ptr_num_attrs )
{
    bool success = false;
    
    if( ( ptr_mem_begin != 0 ) && ( ptr_length != 0 ) && 
        ( ptr_num_particles != 0 ) && ( ptr_num_attrs ) )
    {
        static SIXTRL_SIZE_T const ZERO_SIZE = ( SIXTRL_SIZE_T )0u;
        SIXTRL_SIZE_T const u64_stride = sizeof( SIXTRL_UINT64_T );
        
        SIXTRL_UINT64_T temp = ( SIXTRL_UINT64_T )0u;
        
        unsigned char const* SIXTRL_RESTRICT it = ptr_mem_begin;
        
        assert( ptr_num_particles != ptr_length );
        assert( ptr_num_attrs     != ptr_length );
        assert( ptr_num_particles != ptr_num_attrs );
        
        *ptr_length = ZERO_SIZE;
        *ptr_num_attrs     = ZERO_SIZE;
        *ptr_num_particles = ZERO_SIZE;
        
        /* ----------------------------------------------------------------- */
        /* length: */
        
        memcpy( &temp, it, u64_stride );
        it = it + u64_stride;
        
        if( temp < NS(PARTICLES_PACK_BLOCK_LENGTH) )
        {
            return success;
        }
        
        *ptr_length = ( SIXTRL_SIZE_T )temp;
        
        /* ----------------------------------------------------------------- */
        /* pack indicator: */
        
        memcpy( &temp, it, u64_stride );
        it = it + u64_stride;
        
        if( temp != NS(PARTICLES_PACK_INDICATOR) )
        {
            return success;
        }
        
        /* ----------------------------------------------------------------- */
        /* num of particles: */
                
        memcpy( &temp, it, u64_stride );
        it = it + u64_stride;
        
        if( temp == ( SIXTRL_UINT64_T )0u )
        {
            return success;
        }
        
        *ptr_num_particles = ( SIXTRL_SIZE_T )temp;
        
        /* ----------------------------------------------------------------- */
        /* num_of_attributes: */
        
        memcpy( &temp, it, u64_stride );
        it = it + u64_stride;
        
        *ptr_num_attrs = ( SIXTRL_SIZE_T )temp;
        
        if( *ptr_num_attrs != NS(PARTICLES_NUM_OF_ATTRIBUTES) )
        {
            return success;
        }
        
        success = true;
    }
    
    return success;
}

bool NS(Particles_unpack_check_consistency)(
    unsigned char const* SIXTRL_RESTRICT ptr_mem_begin,
    SIXTRL_SIZE_T  length, SIXTRL_SIZE_T num_of_particles, SIXTRL_SIZE_T num_of_attributes )
{
    bool is_consistent = false;
    
    static SIXTRL_SIZE_T const ZERO_SIZE = ( SIXTRL_SIZE_T )0u;
    
    if( ( ptr_mem_begin != 0 ) && 
        ( length > NS(PARTICLES_PACK_BLOCK_LENGTH) ) &&
        ( num_of_particles  > ( SIXTRL_SIZE_T )0u ) &&
        ( ( num_of_attributes == NS(PARTICLES_NUM_OF_ATTRIBUTES ) ) ) )
    {
        SIXTRL_SIZE_T const u64_stride = sizeof( SIXTRL_UINT64_T );   
        SIXTRL_SIZE_T ii = ZERO_SIZE;
        SIXTRL_SIZE_T calculated_size = ZERO_SIZE;
        
        SIXTRL_UINT64_T prev_offset;
        SIXTRL_UINT64_T temp = ( SIXTRL_UINT64_T )0u;        
        unsigned char const* SIXTRL_RESTRICT it = ptr_mem_begin;
                
        /* ----------------------------------------------------------------- */
        /* length: */
        
        memcpy( &temp, it, u64_stride );
        it = it + u64_stride;        
        
        if( length != ( SIXTRL_SIZE_T )temp )
        {
            return is_consistent;
        }
        
        /* ----------------------------------------------------------------- */
        /* pack indicator: */
        
        memcpy( &temp, it, u64_stride );
        it = it + u64_stride;
        
        if( temp != NS(PARTICLES_PACK_INDICATOR) )
        {
            return is_consistent;
        }
        
        /* ----------------------------------------------------------------- */
        /* num of particles: */
                
        memcpy( &temp, it, u64_stride );
        it = it + u64_stride;
        
        if( num_of_particles != ( SIXTRL_SIZE_T )temp )
        {
            return is_consistent;
        }
        
        /* ----------------------------------------------------------------- */
        /* num_of_attributes: */
        
        memcpy( &temp, it, u64_stride );
        it = it + u64_stride;
        
        if( num_of_attributes != ( SIXTRL_SIZE_T )temp )
        {
            return is_consistent;
        }
        
        temp = 0;
        
        for( ii = ZERO_SIZE ; ii < num_of_attributes ; ++ii )
        {
            prev_offset = temp;
            memcpy( &temp, it, u64_stride );
            it = it + u64_stride;
            
            if( prev_offset < temp )
            {
                calculated_size += ( temp - prev_offset );
            }
            else
            {
                return is_consistent;
            }
        }
        
        calculated_size += sizeof( double ) * num_of_particles;
        
        is_consistent = ( calculated_size <= length );
    }
    
    return is_consistent;    
}

bool NS(Particles_unpack_map_memory)(
    struct NS(Particles)* SIXTRL_RESTRICT p, unsigned char* ptr_begin, 
    SIXTRL_SIZE_T length, SIXTRL_SIZE_T num_of_particles, SIXTRL_SIZE_T num_of_attributes )
{
    bool success = false;
    
    static SIXTRL_SIZE_T const ZERO_SIZE = ( SIXTRL_SIZE_T )0u;
    
    if( ( p != 0 ) && ( length > NS(PARTICLES_PACK_BLOCK_LENGTH) ) &&
        ( num_of_particles > ZERO_SIZE ) && 
        ( num_of_attributes == NS(PARTICLES_NUM_OF_ATTRIBUTES) ) )
    {
        SIXTRL_SIZE_T const u64_stride = sizeof( SIXTRL_UINT64_T );
        
        unsigned char const*  it = ptr_begin;
        SIXTRL_SIZE_T calculated_size   = ZERO_SIZE;
        
        SIXTRL_UINT64_T prev_off;
        SIXTRL_UINT64_T off = ( SIXTRL_UINT64_T )0u;
        
        /* move past the beginning for the header, i.e. 
         * - the length indicator  (SIXTRL_UINT64_T)
         * - the pack   indicator  (SIXTRL_UINT64_T)
         * - the num_of_particles  (SIXTRL_UINT64_T)
         * - the num_of_attributes (SIXTRL_UINT64_T) 
         * Note that the contents of these fields is NOT verified here! */
        it = it + u64_stride * ( SIXTRL_SIZE_T )4u;
        
        /* Now read num_of_attributes times the offset, apply it on the base 
         * pointr and assign the address to the data member: */
        
        prev_off = off;
        memcpy( &off, it, u64_stride );
        it = it + u64_stride;
        assert( NS(PARTICLES_PACK_BLOCK_LENGTH) <= off );
        calculated_size += ( off - prev_off );
        NS(Particles_assign_ptr_to_q0)( p, ( double* )( ptr_begin + off ) );
        
        prev_off = off;
        memcpy( &off, it, u64_stride );
        assert( prev_off + num_of_particles * sizeof( double ) <= off );
        calculated_size += ( off - prev_off );
        it = it + u64_stride;        
        NS(Particles_assign_ptr_to_mass0)( p, ( double* )( ptr_begin + off ) );
        
        prev_off = off;
        memcpy( &off, it, u64_stride );
        it = it + u64_stride;
        assert( prev_off + num_of_particles * sizeof( double ) <= off );
        calculated_size += ( off - prev_off );        
        NS(Particles_assign_ptr_to_beta0)( p, ( double* )( ptr_begin + off ) );
        
        prev_off = off;
        memcpy( &off, it, u64_stride );
        it = it + u64_stride;
        assert( prev_off + num_of_particles * sizeof( double ) <= off );
        calculated_size += ( off - prev_off );        
        NS(Particles_assign_ptr_to_gamma0)( p, ( double* )( ptr_begin + off ) );
        
        prev_off = off;
        memcpy( &off, it, u64_stride );
        it = it + u64_stride;        
        assert( prev_off + num_of_particles * sizeof( double ) <= off );
        calculated_size += ( off - prev_off );        
        NS(Particles_assign_ptr_to_p0c)( p, ( double* )( ptr_begin + off ) );
        
        prev_off = off;
        memcpy( &off, it, u64_stride );
        it = it + u64_stride;        
        assert( prev_off + num_of_particles * sizeof( SIXTRL_INT64_T ) <= off );
        calculated_size += ( off - prev_off );        
        NS(Particles_assign_ptr_to_particle_id)( p, ( SIXTRL_INT64_T* )( ptr_begin + off ) );
        
        prev_off = off;
        memcpy( &off, it, u64_stride );
        it = it + u64_stride;
        assert( prev_off + num_of_particles * sizeof( SIXTRL_INT64_T ) <= off );
        calculated_size += ( off - prev_off );        
        NS(Particles_assign_ptr_to_lost_at_element_id)( p, ( SIXTRL_INT64_T* )( ptr_begin + off ) );
        
        prev_off = off;
        memcpy( &off, it, u64_stride );
        it = it + u64_stride;        
        assert( prev_off + num_of_particles * sizeof( SIXTRL_INT64_T ) <= off );
        calculated_size += ( off - prev_off );        
        NS(Particles_assign_ptr_to_lost_at_turn)( p, ( SIXTRL_INT64_T* )( ptr_begin + off ) );
        
        prev_off = off;
        memcpy( &off, it, u64_stride );
        it = it + u64_stride;
        assert( prev_off + num_of_particles * sizeof( SIXTRL_INT64_T ) <= off );
        calculated_size += ( off - prev_off );        
        NS(Particles_assign_ptr_to_state)( p, ( SIXTRL_INT64_T* )( ptr_begin + off ) );
        
        prev_off = off;
        memcpy( &off, it, u64_stride );
        it = it + u64_stride;
        assert( prev_off + num_of_particles * sizeof( double ) <= off );
        calculated_size += ( off - prev_off );        
        NS(Particles_assign_ptr_to_s)( p, ( double* )( ptr_begin + off ) );
        
        prev_off = off;
        memcpy( &off, it, u64_stride );
        it = it + u64_stride;
        assert( prev_off + num_of_particles * sizeof( double ) <= off );
        calculated_size += ( off - prev_off );        
        NS(Particles_assign_ptr_to_x)( p, ( double* )( ptr_begin + off ) );
        
        prev_off = off;
        memcpy( &off, it, u64_stride );
        it = it + u64_stride;
        assert( prev_off + num_of_particles * sizeof( double ) <= off );
        calculated_size += ( off - prev_off );        
        NS(Particles_assign_ptr_to_y)( p, ( double* )( ptr_begin + off ) );
        
        prev_off = off;
        memcpy( &off, it, u64_stride );
        it = it + u64_stride;
        assert( prev_off + num_of_particles * sizeof( double ) <= off );
        calculated_size += ( off - prev_off );        
        NS(Particles_assign_ptr_to_px)( p, ( double* )( ptr_begin + off ) );
        
        prev_off = off;
        memcpy( &off, it, u64_stride );
        it = it + u64_stride;
        assert( prev_off + num_of_particles * sizeof( double ) <= off );
        calculated_size += ( off - prev_off );        
        NS(Particles_assign_ptr_to_py)( p, ( double* )( ptr_begin + off ) );
        
        prev_off = off;
        memcpy( &off, it, u64_stride );
        it = it + u64_stride;
        assert( prev_off + num_of_particles * sizeof( double ) <= off );
        calculated_size += ( off - prev_off );        
        NS(Particles_assign_ptr_to_sigma)( p, ( double* )( ptr_begin + off ) );
        
        prev_off = off;
        memcpy( &off, it, u64_stride );
        it = it + u64_stride;
        assert( prev_off + num_of_particles * sizeof( double ) <= off );
        calculated_size += ( off - prev_off );        
        NS(Particles_assign_ptr_to_psigma)( p, ( double* )( ptr_begin + off ) );
        
        prev_off = off;
        memcpy( &off, it, u64_stride );
        it = it + u64_stride;
        assert( prev_off + num_of_particles * sizeof( double ) <= off );
        calculated_size += ( off - prev_off );        
        NS(Particles_assign_ptr_to_delta)( p, ( double* )( ptr_begin + off ) );
        
        prev_off = off;
        memcpy( &off, it, u64_stride );
        it = it + u64_stride;
        assert( prev_off + num_of_particles * sizeof( double ) <= off );
        calculated_size += ( off - prev_off );        
        NS(Particles_assign_ptr_to_rpp)( p, ( double* )( ptr_begin + off ) );
        
        prev_off = off;
        memcpy( &off, it, u64_stride );
        it = it + u64_stride;
        assert( prev_off + num_of_particles * sizeof( double ) <= off );
        calculated_size += ( off - prev_off );        
        NS(Particles_assign_ptr_to_rvv)( p, ( double* )( ptr_begin + off ) );
        
        prev_off = off;
        memcpy( &off, it, u64_stride );
        it = it + u64_stride;
        assert( prev_off + num_of_particles * sizeof( double ) <= off );
        calculated_size += ( off - prev_off );        
        NS(Particles_assign_ptr_to_chi)( p, ( double* )( ptr_begin + off ) );
        
        calculated_size += num_of_particles * sizeof( double );
        
        success = ( calculated_size <= length );
    }
    
    return success;
}

bool NS(Particles_unpack)( NS(Particles)* SIXTRL_RESTRICT particles, 
    unsigned char* SIXTRL_RESTRICT mem, SIXTRL_UINT64_T flags )
{
    bool success = false;
    
    static SIXTRL_SIZE_T const ZERO_SIZE = ( SIXTRL_SIZE_T )0u;
    
    if( ( particles != 0 ) && ( mem != 0 ) )
    {
        bool const map_memory = 
            ( ( flags & NS(PARTICLES_UNPACK_MAP) ) == NS(PARTICLES_UNPACK_MAP) );
        
        bool const copy_memory = 
            ( ( flags & NS(PARTICLES_UNPACK_COPY) ) == NS(PARTICLES_UNPACK_COPY) );
            
        bool const check_consistency =
            ( ( flags & NS(PARTICLES_UNPACK_CHECK_CONSISTENCY ) ) ==
                NS( PARTICLES_UNPACK_CHECK_CONSISTENCY) );
        
        SIXTRL_SIZE_T length = ZERO_SIZE;
        SIXTRL_SIZE_T num_of_attributes = ZERO_SIZE;
        SIXTRL_SIZE_T num_of_particles  = ZERO_SIZE;
        
        if( !NS(Particles_unpack_read_header)( 
                mem, &length, &num_of_particles, &num_of_attributes ) )
        {
            return success;
        }
        
        if( ( check_consistency ) &&
            ( !NS(Particles_unpack_check_consistency)( 
                mem, length, num_of_particles, num_of_attributes ) ) )
        {
            return success;
        }
        
        /* ------------------------------------------------------------------ */
        
        if( copy_memory )
        {
            /* TODO: Implement copy_memory operation!!! */
            return success;
        }
        else if( map_memory )
        {
            if( NS(Particles_get_const_ptr_mem_context )( particles ) != 0 )
            {
                NS(Particles_free)( particles );
            }
            
            assert( NS(Particles_get_const_ptr_mem_context )( particles ) == 0 );
            
            if( NS(Particles_unpack_map_memory)( particles, mem, length, 
                num_of_particles, num_of_attributes ) )
            {
                SIXTRL_UINT64_T const flags = NS(PARTICLES_FLAGS_PACKED) |
                                       NS(PARTICLES_FLAGS_MEM_CTX_FLAT_MEMORY);
                                       
                NS(Particles_set_flags)( particles, flags );
                
                NS(Particles_set_ptr_mem_begin)( particles, mem );
                NS(Particles_set_ptr_mem_context)( particles, mem );
                NS(Particles_set_size)( particles, num_of_particles );
                
                success = true;
            }
        }         
    }
    
    return success;
}

/* ------------------------------------------------------------------------ */

void NS( Particles_free )( struct NS( Particles ) * SIXTRL_RESTRICT particles )
{
    if( NS( Particles_manages_own_memory )( particles ) )
    {
        if( NS( Particles_uses_mempool )( particles ) )
        {
            NS( MemPool )* ptr_mem_pool =
                (NS( MemPool )*)NS( Particles_get_const_mem_pool )( particles );

            if( ptr_mem_pool != 0 )
            {
                NS( MemPool_free )( ptr_mem_pool );
                free( ptr_mem_pool );
                ptr_mem_pool = 0;
            }
        }

        if( NS( Particles_uses_single_particle )( particles ) )
        {
            typedef NS( SingleParticle ) single_t;

            single_t* ptr_single_particle = (single_t*)NS(
                Particles_get_const_base_single_particle )( particles );

            if( ptr_single_particle != 0 )
            {
                free( ptr_single_particle );
                ptr_single_particle = 0;
            }
        }
    }

    NS( Particles_preset )( particles );

    return;
}

/* -------------------------------------------------------------------------- */

bool NS( Particles_has_defined_alignment )( const struct NS( Particles ) *
                                            const SIXTRL_RESTRICT p )
{
    return ( NS( Particles_alignment )( p ) != ( SIXTRL_UINT64_T )0u );
}

bool NS( Particles_is_aligned )( const struct NS( Particles ) *
                                     const SIXTRL_RESTRICT p,
                                 SIXTRL_SIZE_T alignment )
{
    bool is_aligned = false;

    if( ( alignment != ( SIXTRL_UINT64_T )0u ) &&
        ( alignment <= NS( PARTICLES_MAX_ALIGNMENT ) ) )
    {
        SIXTRL_UINT64_T const align_flags = NS( Particles_alignment )( p );

        if( align_flags != ( SIXTRL_UINT64_T )0u )
        {
            /* Has defined alignment == true */

            SIXTRL_UINT64_T const asked_alignment = (SIXTRL_UINT64_T)alignment;

            is_aligned =
                ( ( asked_alignment == align_flags ) ||
                  ( ( asked_alignment < align_flags ) &&
                    ( ( align_flags % asked_alignment ) == ( SIXTRL_UINT64_T )0u ) ) );

            assert( ( !is_aligned ) ||
                    ( NS( Particles_check_alignment )( p, alignment ) ) );
        } else if( ( p != 0 ) && ( NS( Particles_get_x )( p ) != 0 ) )
        {
            /* No defined alignment but at least one member is not 0 ->
             * verify that the alignment is true  by checking each member */

            is_aligned = NS( Particles_check_alignment )( p, alignment );
        }
    }

    return is_aligned;
}

bool NS( Particles_check_alignment )( const struct NS( Particles ) *
                                          const SIXTRL_RESTRICT p,
                                      SIXTRL_SIZE_T n )
{
    bool is_aligned = false;

    static SIXTRL_SIZE_T const Z0 = (SIXTRL_SIZE_T)0u;

    if( ( p != 0 ) && ( n != Z0 ) && ( NS( Particles_get_q0 )( p ) != 0 ) &&
        ( NS( Particles_get_mass0 )( p ) != 0 ) &&
        ( NS( Particles_get_beta0 )( p ) != 0 ) &&
        ( NS( Particles_get_gamma0 )( p ) != 0 ) &&
        ( NS( Particles_get_p0c )( p ) != 0 ) &&
        ( NS( Particles_get_particle_id )( p ) != 0 ) &&
        ( NS( Particles_get_lost_at_element_id )( p ) != 0 ) &&
        ( NS( Particles_get_lost_at_turn )( p ) != 0 ) &&
        ( NS( Particles_get_state )( p ) != 0 ) &&
        ( NS( Particles_get_s )( p ) != 0 ) &&
        ( NS( Particles_get_x )( p ) != 0 ) &&
        ( NS( Particles_get_y )( p ) != 0 ) &&
        ( NS( Particles_get_px )( p ) != 0 ) &&
        ( NS( Particles_get_py )( p ) != 0 ) &&
        ( NS( Particles_get_sigma )( p ) != 0 ) &&
        ( NS( Particles_get_psigma )( p ) != 0 ) &&
        ( NS( Particles_get_delta )( p ) != 0 ) &&
        ( NS( Particles_get_rpp )( p ) != 0 ) &&
        ( NS( Particles_get_rvv )( p ) != 0 ) &&
        ( NS( Particles_get_chi )( p ) != 0 ) )
    {
        is_aligned =
            ( ( ( ( (uintptr_t)NS( Particles_get_q0 )( p ) ) % n ) == Z0 ) &&
              ( ( ( (uintptr_t)NS( Particles_get_mass0 )( p ) ) % n ) == Z0 ) &&
              ( ( ( (uintptr_t)NS( Particles_get_beta0 )( p ) ) % n ) == Z0 ) &&
              ( ( ( (uintptr_t)NS( Particles_get_gamma0 )( p ) ) % n ) ==
                Z0 ) &&
              ( ( ( (uintptr_t)NS( Particles_get_p0c )( p ) ) % n ) == Z0 ) &&
              ( ( ( (uintptr_t)NS( Particles_get_particle_id )( p ) ) % n ) ==
                Z0 ) &&
              ( ( ( (uintptr_t)NS( Particles_get_lost_at_element_id )( p ) ) %
                  n ) == Z0 ) &&
              ( ( ( (uintptr_t)NS( Particles_get_lost_at_turn )( p ) ) % n ) ==
                Z0 ) &&
              ( ( ( (uintptr_t)NS( Particles_get_state )( p ) ) % n ) == Z0 ) &&
              ( ( ( (uintptr_t)NS( Particles_get_s )( p ) ) % n ) == Z0 ) &&
              ( ( ( (uintptr_t)NS( Particles_get_x )( p ) ) % n ) == Z0 ) &&
              ( ( ( (uintptr_t)NS( Particles_get_y )( p ) ) % n ) == Z0 ) &&
              ( ( ( (uintptr_t)NS( Particles_get_px )( p ) ) % n ) == Z0 ) &&
              ( ( ( (uintptr_t)NS( Particles_get_py )( p ) ) % n ) == Z0 ) &&
              ( ( ( (uintptr_t)NS( Particles_get_sigma )( p ) ) % n ) == Z0 ) &&
              ( ( ( (uintptr_t)NS( Particles_get_psigma )( p ) ) % n ) ==
                Z0 ) &&
              ( ( ( (uintptr_t)NS( Particles_get_delta )( p ) ) % n ) == Z0 ) &&
              ( ( ( (uintptr_t)NS( Particles_get_rpp )( p ) ) % n ) == Z0 ) &&
              ( ( ( (uintptr_t)NS( Particles_get_rvv )( p ) ) % n ) == Z0 ) &&
              ( ( ( (uintptr_t)NS( Particles_get_chi )( p ) ) % n ) == Z0 ) );
    }

    return is_aligned;
}

SIXTRL_UINT64_T NS( Particles_alignment )( const struct NS( Particles ) *
                                    const SIXTRL_RESTRICT p )
{
    SIXTRL_UINT64_T alignment = ( SIXTRL_UINT64_T )0u;

    if( p != 0 )
    {
        SIXTRL_UINT64_T const flags = NS( Particles_get_flags )( p );
        alignment = ( ( flags & NS( PARTICLES_FLAGS_ALIGN_MASK ) ) >>
                      NS( PARTICLES_FLAGS_ALIGN_MASK_OFFSET_BITS ) );
    }

    return alignment;
}

/* ------------------------------------------------------------------------ */

bool NS( Particles_is_consistent )( const NS( Particles ) *
                                    const SIXTRL_RESTRICT p )
{
    bool is_consistent = false;

    static SIXTRL_SIZE_T const ZERO = (SIXTRL_SIZE_T)0u;
    SIXTRL_SIZE_T const num_particles = NS( Particles_get_size )( p );

    if( ( p != 0 ) && ( num_particles > ZERO ) )
    {
        is_consistent  = ( NS( Particles_get_q0 )( p ) != 0 );
        is_consistent &= ( NS( Particles_get_mass0 )( p ) != 0 );
        is_consistent &= ( NS( Particles_get_beta0 )( p ) != 0 );
        is_consistent &= ( NS( Particles_get_gamma0 )( p ) != 0 );
        is_consistent &= ( NS( Particles_get_p0c )( p ) != 0 );
        is_consistent &= ( NS( Particles_get_particle_id )( p ) != 0 );
        is_consistent &= ( NS( Particles_get_lost_at_element_id )( p ) != 0 );
        is_consistent &= ( NS( Particles_get_lost_at_turn )( p ) != 0 );
        is_consistent &= ( NS( Particles_get_state )( p ) != 0 );
        is_consistent &= ( NS( Particles_get_s )( p ) != 0 );
        is_consistent &= ( NS( Particles_get_x )( p ) != 0 );
        is_consistent &= ( NS( Particles_get_y )( p ) != 0 );
        is_consistent &= ( NS( Particles_get_px )( p ) != 0 );
        is_consistent &= ( NS( Particles_get_py )( p ) != 0 );
        is_consistent &= ( NS( Particles_get_sigma )( p ) != 0 );
        is_consistent &= ( NS( Particles_get_psigma )( p ) != 0 );
        is_consistent &= ( NS( Particles_get_delta )( p ) != 0 );
        is_consistent &= ( NS( Particles_get_rpp )( p ) != 0 );
        is_consistent &= ( NS( Particles_get_rvv )( p ) != 0 );
        is_consistent &= ( NS( Particles_get_chi )( p ) != 0 );

        if( ( is_consistent ) && ( NS( Particles_manages_own_memory )( p ) ) )
        {
            is_consistent =
                ( ( NS( Particles_get_const_ptr_mem_context )( p ) != 0 ) &&
                  ( ( NS( Particles_uses_mempool )( p ) ) ||
                    ( NS( Particles_uses_single_particle )( p ) ) ) );
        }

        if( ( is_consistent ) &&
            ( NS( Particles_has_defined_alignment )( p ) ) )
        {
            SIXTRL_UINT64_T const def_alignment = NS( Particles_alignment )( p );
            is_consistent = NS( Particles_check_alignment )( p, def_alignment );
        }

        if( ( is_consistent ) && ( NS( Particles_is_packed( p ) ) ) )
        {

            ptrdiff_t const min_double_len = sizeof( double ) * num_particles;
            ptrdiff_t const min_int64_len = sizeof( SIXTRL_INT64_T ) * num_particles;

            /* ------------------------------------------------------------- */
            /* distance q0 -> mass0 */
            
            unsigned char const* prev_elem =
                (unsigned char*)NS( Particles_get_q0 )( p );

            unsigned char const* ptr_elem =
                (unsigned char*)NS( Particles_get_mass0 )( p );

            is_consistent = ( ( ptr_elem - prev_elem ) >= min_double_len );

            /* ------------------------------------------------------------- */
            /* distance mass0 -> beta0 */

            prev_elem = ptr_elem;
            ptr_elem = (unsigned char*)NS( Particles_get_beta0 )( p );
            is_consistent &= ( ( ptr_elem - prev_elem ) >= min_double_len );

            /* ------------------------------------------------------------- */
            /* distance beta0 -> gamma0 */

            prev_elem = ptr_elem;
            ptr_elem = (unsigned char*)NS( Particles_get_gamma0 )( p );
            is_consistent &= ( ( ptr_elem - prev_elem ) >= min_double_len );

            /* ------------------------------------------------------------- */
            /* distance gamma0 -> p0c */

            prev_elem = ptr_elem;
            ptr_elem = (unsigned char*)NS( Particles_get_p0c )( p );
            is_consistent &= ( ( ptr_elem - prev_elem ) >= min_double_len );

            /* ------------------------------------------------------------- */
            /* distance p0c -> partid */

            prev_elem = ptr_elem;
            ptr_elem = (unsigned char*)NS( Particles_get_particle_id )( p );
            is_consistent &= ( ( ptr_elem - prev_elem ) >= min_int64_len );

            /* ------------------------------------------------------------- */
            /* distance partid -> elemid */

            prev_elem = ptr_elem;
            ptr_elem =
                (unsigned char*)NS( Particles_get_lost_at_element_id )( p );

            is_consistent &= ( ( ptr_elem - prev_elem ) >= min_int64_len );

            /* ------------------------------------------------------------- */
            /* distance elemid -> turn */

            prev_elem = ptr_elem;
            ptr_elem = (unsigned char*)NS( Particles_get_lost_at_turn )( p );
            is_consistent &= ( ( ptr_elem - prev_elem ) >= min_int64_len );

            /* ------------------------------------------------------------- */
            /* distance turn -> state */

            prev_elem = ptr_elem;
            ptr_elem = (unsigned char*)NS( Particles_get_state )( p );
            is_consistent &= ( ( ptr_elem - prev_elem ) >= min_int64_len );

            /* ------------------------------------------------------------- */
            /* distance state -> s */

            prev_elem = ptr_elem;
            ptr_elem = (unsigned char*)NS( Particles_get_s )( p );
            is_consistent &= ( ( ptr_elem - prev_elem ) >= min_double_len );

            /* ------------------------------------------------------------- */
            /* distance s -> x */

            prev_elem = ptr_elem;
            ptr_elem = (unsigned char*)NS( Particles_get_x )( p );
            is_consistent &= ( ( ptr_elem - prev_elem ) >= min_double_len );

            /* ------------------------------------------------------------- */
            /* distance x -> y */

            prev_elem = ptr_elem;
            ptr_elem = (unsigned char*)NS( Particles_get_y )( p );
            is_consistent &= ( ( ptr_elem - prev_elem ) >= min_double_len );

            /* ------------------------------------------------------------- */
            /* distance y -> px */

            prev_elem = ptr_elem;
            ptr_elem = (unsigned char*)NS( Particles_get_px )( p );
            is_consistent &= ( ( ptr_elem - prev_elem ) >= min_double_len );

            /* ------------------------------------------------------------- */
            /* distance px -> py */

            prev_elem = ptr_elem;
            ptr_elem = (unsigned char*)NS( Particles_get_py )( p );
            is_consistent &= ( ( ptr_elem - prev_elem ) >= min_double_len );

            /* ------------------------------------------------------------- */
            /* distance py -> sigma */

            prev_elem = ptr_elem;
            ptr_elem = (unsigned char*)NS( Particles_get_sigma )( p );
            is_consistent &= ( ( ptr_elem - prev_elem ) >= min_double_len );

            /* ------------------------------------------------------------- */
            /* distance sigma -> psigma */

            prev_elem = ptr_elem;
            ptr_elem = (unsigned char*)NS( Particles_get_psigma )( p );
            is_consistent &= ( ( ptr_elem - prev_elem ) >= min_double_len );

            /* ------------------------------------------------------------- */
            /* distance psigma -> delta */

            prev_elem = ptr_elem;
            ptr_elem = (unsigned char*)NS( Particles_get_delta )( p );
            is_consistent &= ( ( ptr_elem - prev_elem ) >= min_double_len );

            /* ------------------------------------------------------------- */
            /* distance delta -> rpp */

            prev_elem = ptr_elem;
            ptr_elem = (unsigned char*)NS( Particles_get_rpp )( p );
            is_consistent &= ( ( ptr_elem - prev_elem ) >= min_double_len );

            /* ------------------------------------------------------------- */
            /* distance rpp -> rvv */

            prev_elem = ptr_elem;
            ptr_elem = (unsigned char*)NS( Particles_get_rvv )( p );
            is_consistent &= ( ( ptr_elem - prev_elem ) >= min_double_len );

            /* ------------------------------------------------------------- */
            /* distance rvv -> chi */

            prev_elem = ptr_elem;
            ptr_elem = (unsigned char*)NS( Particles_get_chi )( p );
            is_consistent &= ( ( ptr_elem - prev_elem ) >= min_double_len );
        }
    }

    return is_consistent;
}

/* ------------------------------------------------------------------------ */

bool NS( Particles_deep_copy_one )( struct NS( Particles ) *
                                        SIXTRL_RESTRICT dest,
                                    SIXTRL_UINT64_T dest_id,
                                    struct NS( Particles )
                                        const* SIXTRL_RESTRICT source,
                                    SIXTRL_UINT64_T source_id )
{
    bool success = false;

    if( ( dest != 0 ) && ( NS( Particles_get_size )( dest ) > dest_id ) &&
        ( source != 0 ) && ( NS( Particles_get_size )( source ) > source_id ) )
    {
        NS( Particles_set_q0_value )
        ( dest, dest_id, NS( Particles_get_q0_value )( source, source_id ) );

        NS( Particles_set_mass0_value )
        ( dest, dest_id, NS( Particles_get_mass0_value )( source, source_id ) );

        NS( Particles_set_beta0_value )
        ( dest, dest_id, NS( Particles_get_beta0_value )( source, source_id ) );

        NS( Particles_set_gamma0_value )
        ( dest,
          dest_id,
          NS( Particles_get_gamma0_value )( source, source_id ) );

        NS( Particles_set_p0c_value )
        ( dest, dest_id, NS( Particles_get_p0c_value )( source, source_id ) );

        NS( Particles_set_particle_id_value )
        ( dest,
          dest_id,
          NS( Particles_get_particle_id_value )( source, source_id ) );

        NS( Particles_set_lost_at_element_id_value )
        ( dest,
          dest_id,
          NS( Particles_get_lost_at_element_id_value )( source, source_id ) );

        NS( Particles_set_lost_at_turn_value )
        ( dest,
          dest_id,
          NS( Particles_get_lost_at_turn_value )( source, source_id ) );

        NS( Particles_set_state_value )
        ( dest, dest_id, NS( Particles_get_state_value )( source, source_id ) );

        NS( Particles_set_s_value )
        ( dest, dest_id, NS( Particles_get_s_value )( source, source_id ) );

        NS( Particles_set_x_value )
        ( dest, dest_id, NS( Particles_get_x_value )( source, source_id ) );

        NS( Particles_set_y_value )
        ( dest, dest_id, NS( Particles_get_y_value )( source, source_id ) );

        NS( Particles_set_px_value )
        ( dest, dest_id, NS( Particles_get_px_value )( source, source_id ) );

        NS( Particles_set_py_value )
        ( dest, dest_id, NS( Particles_get_py_value )( source, source_id ) );

        NS( Particles_set_sigma_value )
        ( dest, dest_id, NS( Particles_get_sigma_value )( source, source_id ) );

        NS( Particles_set_psigma_value )
        ( dest,
          dest_id,
          NS( Particles_get_psigma_value )( source, source_id ) );

        NS( Particles_set_delta_value )
        ( dest, dest_id, NS( Particles_get_delta_value )( source, source_id ) );

        NS( Particles_set_rpp_value )
        ( dest, dest_id, NS( Particles_get_rpp_value )( source, source_id ) );

        NS( Particles_set_rvv_value )
        ( dest, dest_id, NS( Particles_get_rvv_value )( source, source_id ) );

        NS( Particles_set_chi_value )
        ( dest, dest_id, NS( Particles_get_chi_value )( source, source_id ) );

        success = true;
    }

    return success;
}

/* ------------------------------------------------------------------------ */

bool NS( Particles_deep_copy_all )( struct NS( Particles ) *
                                        SIXTRL_RESTRICT dest,
                                    struct NS( Particles )
                                        const* SIXTRL_RESTRICT source )
{
    bool success = false;

    if( ( dest != 0 ) && ( source != 0 ) &&
        ( NS( Particles_get_size )( dest ) ==
          NS( Particles_get_size )( source ) ) )
    {
        assert( NS( Particles_is_consistent )( dest ) );
        assert( NS( Particles_is_consistent )( source ) );

        NS( Particles_set_q0 )( dest, NS( Particles_get_q0 )( source ) );
        NS( Particles_set_mass0 )( dest, NS( Particles_get_mass0 )( source ) );
        NS( Particles_set_beta0 )( dest, NS( Particles_get_beta0 )( source ) );
        NS( Particles_set_gamma0 )
        ( dest, NS( Particles_get_gamma0 )( source ) );
        NS( Particles_set_p0c )( dest, NS( Particles_get_p0c )( source ) );

        NS( Particles_set_particle_id )
        ( dest, NS( Particles_get_particle_id )( source ) );

        NS( Particles_set_lost_at_element_id )
        ( dest, NS( Particles_get_lost_at_element_id )( source ) );

        NS( Particles_set_lost_at_turn )
        ( dest, NS( Particles_get_lost_at_turn )( source ) );

        NS( Particles_set_state )( dest, NS( Particles_get_state )( source ) );

        NS( Particles_set_s )( dest, NS( Particles_get_s )( source ) );
        NS( Particles_set_x )( dest, NS( Particles_get_x )( source ) );
        NS( Particles_set_y )( dest, NS( Particles_get_y )( source ) );
        NS( Particles_set_px )( dest, NS( Particles_get_px )( source ) );
        NS( Particles_set_py )( dest, NS( Particles_get_py )( source ) );
        NS( Particles_set_sigma )( dest, NS( Particles_get_sigma )( source ) );

        NS( Particles_set_psigma )
        ( dest, NS( Particles_get_psigma )( source ) );
        NS( Particles_set_delta )( dest, NS( Particles_get_delta )( source ) );
        NS( Particles_set_rpp )( dest, NS( Particles_get_rpp )( source ) );
        NS( Particles_set_rvv )( dest, NS( Particles_get_rvv )( source ) );
        NS( Particles_set_chi )( dest, NS( Particles_get_chi )( source ) );

        success = true;
    }

    return success;
}

/* end: sixtracklib/common/details/particles.c */
