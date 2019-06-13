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

SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* NS(Particles_preset_ext)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles )
{
    return NS(Particles_preset)( particles );
}

NS(particle_num_elements_t) NS(Particles_get_num_of_particles_ext)( const
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles) *const SIXTRL_RESTRICT particles )
{
    return NS(Particles_get_num_of_particles)( particles );
}

/* ------------------------------------------------------------------------- */

NS(particle_num_elements_t)
NS(BufferIndex_get_total_num_of_particles_in_range_ext)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT begin,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT end )
{
    return NS(BufferIndex_get_total_num_of_particles_in_range_ext)(
        begin, end );
}

NS(buffer_size_t)
NS(BufferIndex_get_total_num_of_particle_blocks_in_range_ext)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT begin,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT end )
{
    return NS(BufferIndex_get_total_num_of_particle_blocks_in_range_ext)(
        begin, end );
}

/* ------------------------------------------------------------------------- */

NS(arch_status_t) NS(Particles_copy_single_ext)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT dest,
    NS(particle_num_elements_t) const dest_idx,
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT src,
    NS(particle_num_elements_t) const src_idx )
{
    return NS(Particles_copy_single)( dest, dest_idx, src, src_idx );
}

NS(arch_status_t) NS(Particles_copy_range_ext)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT dest,
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT src,
    NS(particle_num_elements_t) const src_start_index,
    NS(particle_num_elements_t) const src_end_index,
    NS(particle_num_elements_t) dest_start_index )
{
    return NS(Particles_copy_range)(
        dest, src, src_start_index, src_end_index, dest_start_index );
}

NS(arch_status_t) NS(Particles_copy_ext)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT dest,
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT src )
{
    return NS(Particles_copy)( dest, src );
}

void NS(Particles_calculate_difference_ext)(
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT lhs,
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT rhs,
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT diff )
{
    NS(Particles_calculate_difference)( lhs, rhs, diff );
    return;
}

/* ------------------------------------------------------------------------- */

bool NS(Buffer_is_particles_buffer_ext)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const buffer )
{
    return NS(Buffer_is_particles_buffer)( buffer );
}

NS(particle_num_elements_t)
NS(Particles_buffer_get_total_num_of_particles_ext)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer )
{
    return NS(Particles_buffer_get_total_num_of_particles)( buffer );
}

NS(buffer_size_t) NS(Particles_buffer_get_num_of_particle_blocks_ext)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer )
{
    return NS(Particles_buffer_get_num_of_particle_blocks)( buffer );
}

SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)*
NS(Particles_buffer_get_particles_ext)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const part_obj_idx  )
{
    return NS(Particles_buffer_get_particles)( buffer, part_obj_idx );
}

SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles) const*
NS(Particles_buffer_get_const_particles_ext)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const part_obj_idx )
{
    return NS(Particles_buffer_get_const_particles)( buffer, part_obj_idx );
}

bool NS(Particles_buffers_have_same_structure_ext)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT lhs,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT rhs )
{
    return NS(Particles_buffers_have_same_structure)( lhs, rhs );
}

void NS(Particles_buffers_calculate_difference_ext)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT lhs,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT rhs,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT diff )
{
    NS(Particles_buffers_calculate_difference_ext)( lhs, rhs, diff );
    return;
}

void NS(Particles_buffer_clear_particles_ext)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer )
{
    NS(Particles_buffer_clear_particles_ext)( buffer );
}

/* ------------------------------------------------------------------------- */

NS(buffer_size_t) NS(Particles_get_required_num_slots_ext)(
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const num_particles )
{
    return NS(Particles_get_required_num_slots)( buffer, num_particles );
}

NS(buffer_size_t) NS(Particles_get_required_num_dataptrs_ext)(
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const num_particles )
{
    return NS(Particles_get_required_num_dataptrs)( buffer, num_particles );
}

bool NS(Particles_can_be_added_ext)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const num_particles,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_objects,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_slots,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_dataptrs )
{
    return NS(Particles_can_be_added)( buffer, num_particles, requ_objects,
                                       requ_slots, requ_dataptrs );
}

SIXTRL_BUFFER_DATAPTR_DEC NS(Particles)* NS(Particles_new_ext)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const num_particles )
{
    return NS(Particles_new)( buffer, num_particles );
}

SIXTRL_BUFFER_DATAPTR_DEC NS(Particles)* NS(Particles_add_ext)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const  num_particles,
    NS(particle_real_ptr_t)  q0_ptr,        NS(particle_real_ptr_t)  mass0_ptr,
    NS(particle_real_ptr_t)  beta0_ptr,     NS(particle_real_ptr_t)  gamma0_ptr,
    NS(particle_real_ptr_t)  p0c_ptr,       NS(particle_real_ptr_t)  s_ptr,
    NS(particle_real_ptr_t)  x_ptr,         NS(particle_real_ptr_t)  y_ptr,
    NS(particle_real_ptr_t)  px_ptr,        NS(particle_real_ptr_t)  py_ptr,
    NS(particle_real_ptr_t)  zeta_ptr,      NS(particle_real_ptr_t)  psigma_ptr,
    NS(particle_real_ptr_t)  delta_ptr,     NS(particle_real_ptr_t)  rpp_ptr,
    NS(particle_real_ptr_t)  rvv_ptr,       NS(particle_real_ptr_t)  chi_ptr,
    NS(particle_real_ptr_t)  charge_ratio_ptr,
    NS(particle_index_ptr_t) particle_id_ptr,
    NS(particle_index_ptr_t) at_element_id_ptr,
    NS(particle_index_ptr_t) at_turn_ptr,
    NS(particle_index_ptr_t) state_ptr )
{
    return NS(Particles_add)( buffer, num_particles, q0_ptr, mass0_ptr,
        beta0_ptr, gamma0_ptr, p0c_ptr, s_ptr, x_ptr, y_ptr, px_ptr, py_ptr,
            zeta_ptr, psigma_ptr, delta_ptr, rpp_ptr, rvv_ptr, chi_ptr,
                charge_ratio_ptr, particle_id_ptr, at_element_id_ptr,
                    at_turn_ptr, state_ptr );
}

SIXTRL_BUFFER_DATAPTR_DEC NS(Particles)* NS(Particles_add_copy_ext)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT p )
{
    return NS(Particles_add_copy)( buffer, p );
}

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

/* end: sixtracklib/common/particles/particles.c */
