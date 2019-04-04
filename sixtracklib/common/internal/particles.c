#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/internal/buffer_main_defines.h"
#include "sixtracklib/common/buffer/buffer_object.h"
#include "sixtracklib/common/particles.h"

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "sixtracklib/common/internal/buffer_main_defines.h"
#include "sixtracklib/common/internal/particles_defines.h"

static int NS(Particles_particle_id_recursive_merge_sort)(
    NS(particle_index_t)* SIXTRL_RESTRICT particle_id_array,
    NS(particle_index_t)* SIXTRL_RESTRICT lhs_temp_array,
    NS(particle_index_t)* SIXTRL_RESTRICT rhs_temp_array,
    NS(particle_num_elements_t) const left_index,
    NS(particle_num_elements_t) const right_index );

static int NS(Particles_particle_id_merge_and_check_for_duplicate)(
    NS(particle_index_t)* SIXTRL_RESTRICT particle_id_array,
    NS(particle_index_t)* SIXTRL_RESTRICT lhs_temp_array,
    NS(particle_index_t)* SIXTRL_RESTRICT rhs_temp_array,
    NS(particle_num_elements_t) const lhs_begin_index,
    NS(particle_num_elements_t) const lhs_end_index,
    NS(particle_num_elements_t) const rhs_end_index );

/* ------------------------------------------------------------------------- */

int NS(Particles_particle_id_merge_and_check_for_duplicate)(
    NS(particle_index_t)* SIXTRL_RESTRICT particle_id_array,
    NS(particle_index_t)* SIXTRL_RESTRICT lhs_temp_array,
    NS(particle_index_t)* SIXTRL_RESTRICT rhs_temp_array,
    NS(particle_num_elements_t) const lhs_begin_index,
    NS(particle_num_elements_t) const lhs_end_index,
    NS(particle_num_elements_t) const rhs_end_index )
{
    int success = 0;

    typedef NS(particle_num_elements_t) num_elem_t;

    num_elem_t ii = ( num_elem_t )0u;
    num_elem_t jj = lhs_begin_index;
    num_elem_t kk = lhs_begin_index;

    num_elem_t const lhs_num = lhs_end_index - lhs_begin_index;
    num_elem_t const rhs_num = rhs_end_index - lhs_end_index;

    SIXTRL_ASSERT( lhs_temp_array    != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( rhs_temp_array    != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( particle_id_array != SIXTRL_NULLPTR );

    SIXTRL_ASSERT( lhs_begin_index >= ( num_elem_t )0u );
    SIXTRL_ASSERT( lhs_end_index   >= lhs_begin_index );
    SIXTRL_ASSERT( rhs_end_index   >= lhs_end_index   );

    for( ; jj < lhs_end_index ; ++jj, ++ii )
    {
        lhs_temp_array[ ii ] = particle_id_array[ jj ];
    }

    jj = lhs_end_index;
    ii = ( num_elem_t )0u;

    for(  ; jj < rhs_end_index ; ++jj, ++ii )
    {
        rhs_temp_array[ ii ] = particle_id_array[ jj ];
    }

    ii = ( num_elem_t )0u;
    jj = ( num_elem_t )0u;

    while( ( ii < lhs_num ) && ( jj < rhs_num ) )
    {
        if( lhs_temp_array[ ii ] < rhs_temp_array[ jj ] )
        {
            particle_id_array[ kk ] = lhs_temp_array[ ii ];
            ++ii;
        }
        else if( lhs_temp_array[ ii ] > rhs_temp_array[ jj ] )
        {
            particle_id_array[ kk ] = rhs_temp_array[ jj ];
            ++jj;
        }
        else
        {
            SIXTRL_ASSERT( lhs_temp_array[ ii ] == rhs_temp_array[ jj ] );
            success = -1;
            break;
        }

        ++kk;
    };

    if( ( success == 0 ) && ( ii < lhs_num ) )
    {
        while( ii < lhs_num )
        {
            particle_id_array[ kk++ ] = lhs_temp_array[ ii++ ];
        };
    }

    if( ( success == 0 ) && ( jj < rhs_num ) )
    {
        while( jj < rhs_num )
        {
            particle_id_array[ kk++ ] = rhs_temp_array[ jj++ ];
        };
    }

    return success;
}

int NS(Particles_particle_id_recursive_merge_sort)(
    NS(particle_index_t)* SIXTRL_RESTRICT particle_id_array,
    NS(particle_index_t)* SIXTRL_RESTRICT lhs_temp_array,
    NS(particle_index_t)* SIXTRL_RESTRICT rhs_temp_array,
    NS(particle_num_elements_t) const left_index,
    NS(particle_num_elements_t) const right_index )
{
    typedef NS(particle_num_elements_t) num_elem_t;

    int success = -1;

    SIXTRL_ASSERT( particle_id_array != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( lhs_temp_array    != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( rhs_temp_array    != SIXTRL_NULLPTR );

    if( left_index < right_index )
    {
        num_elem_t const mid_index =
            left_index + ( right_index - left_index ) / 2;

        num_elem_t const mid_p1_index = mid_index + ( num_elem_t )1u;

        if( ( 0 == NS(Particles_particle_id_recursive_merge_sort)(
                particle_id_array, lhs_temp_array, rhs_temp_array,
                left_index, mid_index ) ) &&
            ( 0 == NS(Particles_particle_id_recursive_merge_sort)(
                particle_id_array, lhs_temp_array, rhs_temp_array,
                mid_p1_index, right_index ) ) )
        {
            success = NS(Particles_particle_id_merge_and_check_for_duplicate)(
                particle_id_array, lhs_temp_array, rhs_temp_array,
                left_index, mid_p1_index, right_index + ( num_elem_t )1u );
        }
    }
    else if( left_index == right_index )
    {
        success = 0;
    }


    return success;
}

int NS(Particles_get_min_max_particle_id)(
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT particles,
    SIXTRL_ARGPTR_DEC NS(particle_index_t)* SIXTRL_RESTRICT ptr_min_id,
    SIXTRL_ARGPTR_DEC NS(particle_index_t)* SIXTRL_RESTRICT ptr_max_id )
{
    typedef NS(particle_num_elements_t) num_elem_t;
    typedef NS(particle_index_t)        index_t;
    typedef NS(buffer_size_t)           buf_size_t;

    int success = -1;

    buf_size_t const num_particles =
        NS(Particles_get_num_of_particles)( particles );

    if( num_particles > ( num_elem_t )0u )
    {
        buf_size_t const buffer_size = sizeof( num_elem_t ) * num_particles;

        num_elem_t* particle_id_array = ( num_elem_t* )malloc( buffer_size );
        num_elem_t* lhs_temp_array    = ( num_elem_t* )malloc( buffer_size );
        num_elem_t* rhs_temp_array    = ( num_elem_t* )malloc( buffer_size );

        if( ( buffer_size > ( buf_size_t )0u ) &&
            ( particle_id_array != SIXTRL_NULLPTR ) &&
            ( lhs_temp_array    != SIXTRL_NULLPTR ) &&
            ( rhs_temp_array    != SIXTRL_NULLPTR ) )
        {
            num_elem_t ii = ( num_elem_t )0u;

            for( ; ii < num_particles ; ++ii )
            {
                index_t const temp_particle_id =
                    NS(Particles_get_particle_id_value)( particles, ii );

                lhs_temp_array[ ii ]    = ( num_elem_t )0u;
                rhs_temp_array[ ii ]    = ( num_elem_t )0u;
                particle_id_array[ ii ] = ( temp_particle_id >= ( index_t )0u )
                        ? ( num_elem_t )(  temp_particle_id )
                        : ( num_elem_t )( -temp_particle_id );
            }

            success = NS(Particles_particle_id_recursive_merge_sort)(
                particle_id_array, lhs_temp_array, rhs_temp_array,
                ( num_elem_t )0u, num_particles - ( num_elem_t )1u );
        }

        #if !defined( NDEBUG )
        if( success == 0 )
        {
            num_elem_t ii = ( num_elem_t )0u;
            num_elem_t jj = ( num_elem_t )1u;

            for( ; jj < num_particles ; ++ii, ++jj )
            {
                if( particle_id_array[ ii ] >= particle_id_array[ jj ] )
                {
                    success = -1;
                    break;
                }
            }
        }
        #endif /* !defined( NDEBUG ) */

        if( success == 0 )
        {
            if( ptr_max_id != SIXTRL_NULLPTR )
            {
                *ptr_max_id = ( index_t )particle_id_array[
                    num_particles - ( num_elem_t )1u ];
            }

            if( ptr_min_id != SIXTRL_NULLPTR )
            {
                *ptr_min_id = ( index_t )particle_id_array[ 0 ];
            }
        }

        free( particle_id_array );
        free( lhs_temp_array );
        free( rhs_temp_array );

        particle_id_array = SIXTRL_NULLPTR;
        lhs_temp_array    = SIXTRL_NULLPTR;
        rhs_temp_array    = SIXTRL_NULLPTR;
    }

    return success;
}

int NS(Particles_get_min_max_attributes)(
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(Particles) *const
        SIXTRL_RESTRICT particles,
    SIXTRL_ARGPTR_DEC NS(particle_index_t)* SIXTRL_RESTRICT ptr_min_part_id,
    SIXTRL_ARGPTR_DEC NS(particle_index_t)* SIXTRL_RESTRICT ptr_max_part_id,
    SIXTRL_ARGPTR_DEC NS(particle_index_t)* SIXTRL_RESTRICT ptr_min_element_id,
    SIXTRL_ARGPTR_DEC NS(particle_index_t)* SIXTRL_RESTRICT ptr_max_element_id,
    SIXTRL_ARGPTR_DEC NS(particle_index_t)* SIXTRL_RESTRICT ptr_min_turn_id,
    SIXTRL_ARGPTR_DEC NS(particle_index_t)* SIXTRL_RESTRICT ptr_max_turn_id )
{
    NS(Particles_init_min_max_attributes_for_find)(
        ptr_min_part_id, ptr_max_part_id,
        ptr_min_element_id, ptr_max_element_id,
        ptr_min_turn_id, ptr_max_turn_id );

    return NS(Particles_find_min_max_attributes)( particles, ptr_min_part_id,
        ptr_max_part_id, ptr_min_element_id, ptr_max_element_id,
            ptr_min_turn_id, ptr_max_turn_id );
}

int NS(Particles_buffer_get_min_max_attributes_of_particles_set)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT pbuffer,
    NS(buffer_size_t) const  num_particle_sets,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t) const* SIXTRL_RESTRICT indices_begin,
    SIXTRL_ARGPTR_DEC NS(particle_index_t)* SIXTRL_RESTRICT ptr_min_part_id,
    SIXTRL_ARGPTR_DEC NS(particle_index_t)* SIXTRL_RESTRICT ptr_max_part_id,
    SIXTRL_ARGPTR_DEC NS(particle_index_t)* SIXTRL_RESTRICT ptr_min_element_id,
    SIXTRL_ARGPTR_DEC NS(particle_index_t)* SIXTRL_RESTRICT ptr_max_element_id,
    SIXTRL_ARGPTR_DEC NS(particle_index_t)* SIXTRL_RESTRICT ptr_min_turn_id,
    SIXTRL_ARGPTR_DEC NS(particle_index_t)* SIXTRL_RESTRICT ptr_max_turn_id )
{
    NS(Particles_init_min_max_attributes_for_find)(
        ptr_min_part_id, ptr_max_part_id,
        ptr_min_element_id, ptr_max_element_id,
        ptr_min_turn_id, ptr_max_turn_id );

    return NS(Particles_buffer_find_min_max_attributes_of_particles_set)(
        pbuffer, num_particle_sets, indices_begin, ptr_min_part_id,
            ptr_max_part_id, ptr_min_element_id, ptr_max_element_id,
                ptr_min_turn_id, ptr_max_turn_id );
}

int NS(Particles_buffer_get_min_max_attributes)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT pbuffer,
    SIXTRL_ARGPTR_DEC NS(particle_index_t)* SIXTRL_RESTRICT ptr_min_part_id,
    SIXTRL_ARGPTR_DEC NS(particle_index_t)* SIXTRL_RESTRICT ptr_max_part_id,
    SIXTRL_ARGPTR_DEC NS(particle_index_t)* SIXTRL_RESTRICT ptr_min_element_id,
    SIXTRL_ARGPTR_DEC NS(particle_index_t)* SIXTRL_RESTRICT ptr_max_element_id,
    SIXTRL_ARGPTR_DEC NS(particle_index_t)* SIXTRL_RESTRICT ptr_min_turn_id,
    SIXTRL_ARGPTR_DEC NS(particle_index_t)* SIXTRL_RESTRICT ptr_max_turn_id )
{
    NS(Particles_init_min_max_attributes_for_find)(
        ptr_min_part_id, ptr_max_part_id,
        ptr_min_element_id, ptr_max_element_id,
        ptr_min_turn_id, ptr_max_turn_id );

    return NS(Particles_buffer_find_min_max_attributes)( pbuffer,
        ptr_min_part_id, ptr_max_part_id, ptr_min_element_id,
        ptr_max_element_id, ptr_min_turn_id, ptr_max_turn_id );
}

NS(buffer_size_t)
NS(Particles_buffer_get_total_num_of_particles_on_particle_sets)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const  num_particle_sets,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t) const* pset_begin )
{
    NS(buffer_size_t) total_num_particles = ( NS(buffer_size_t) )0u;

    if( ( buffer != SIXTRL_NULLPTR ) &&
        ( NS(Buffer_is_particles_buffer)( buffer ) ) &&
        ( num_particle_sets > ( NS(buffer_size_t) )0u ) &&
        ( pset_begin != SIXTRL_NULLPTR ) )
    {
        SIXTRL_ARGPTR_DEC NS(buffer_size_t) const* it = pset_begin;

        SIXTRL_ARGPTR_DEC NS(buffer_size_t) const*
            pset_end = it + num_particle_sets;

        SIXTRL_ARGPTR_DEC NS(buffer_size_t) const* prev = pset_end;

        for( ; it != pset_end ; prev = it++ )
        {
            NS(buffer_size_t) const index = *it;
            SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles) const* particles =
                NS(Particles_buffer_get_const_particles)( buffer, index );

            if( ( ( prev == pset_end ) || ( *prev < index ) ) &&
                ( particles != SIXTRL_NULLPTR ) )
            {
                total_num_particles +=
                    NS(Particles_get_num_of_particles)( particles );
            }
            else
            {
                total_num_particles =  ( NS(buffer_size_t) )0u;
                break;
            }
        }
    }

    return total_num_particles;
}

/* end: sixtracklib/common/internal/particles.c */
