#include "sixtracklib/testlib/common/track/track_particles_cpu.h"

#include <stddef.h>
#include <stdlib.h>

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/particles.h"
#include "sixtracklib/common/buffer.h"
#include "sixtracklib/common/track/definitions.h"
#include "sixtracklib/common/track/track.h"

NS(track_status_t) NS(TestTrackCpu_track_particles_until_turn_cpu)(
    NS(Buffer)* SIXTRL_RESTRICT pbuffer,
    NS(buffer_size_t) const num_psets,
    NS(buffer_size_t) const* SIXTRL_RESTRICT pset_indices_begin,
    const NS(Buffer) *const SIXTRL_RESTRICT belems_buffer,
    NS(particle_index_t) const until_turn )
{
    typedef NS(buffer_size_t) size_t;
    NS(track_status_t) status = NS(TRACK_STATUS_GENERAL_FAILURE);

    if( ( pbuffer != SIXTRL_NULLPTR ) && ( num_psets > ( size_t )0u ) &&
        ( pset_indices_begin != SIXTRL_NULLPTR ) &&
        ( NS(Buffer_is_particles_buffer)( pbuffer ) ) &&
        ( belems_buffer != SIXTRL_NULLPTR ) )
    {
        size_t const* pset_it  = pset_indices_begin;
        size_t const* pset_end = pset_it + num_psets;

        status = NS(TRACK_SUCCESS);

        for( ; pset_it != pset_end ; ++pset_it )
        {
            status |= NS(Track_all_particles_until_turn)(
                NS(Particles_buffer_get_particles)( pbuffer, *pset_it ),
                    belems_buffer, until_turn );
        }
    }

    return status;
}

NS(track_status_t) NS(TestTrackCpu_track_particles_elem_by_elem_until_turn_cpu)(
    NS(Buffer)* SIXTRL_RESTRICT pbuffer,
    NS(buffer_size_t) const num_psets,
    NS(buffer_size_t) const* SIXTRL_RESTRICT pset_indices_begin,
    const NS(Buffer) *const SIXTRL_RESTRICT belems_buffer,
    NS(ElemByElemConfig)* SIXTRL_RESTRICT e_by_e_conf,
    NS(particle_index_t) const until_turn )
{
    typedef NS(buffer_size_t) size_t;
    typedef NS(track_status_t) track_status_t;
    typedef NS(elem_by_elem_out_addr_t) address_t;

    track_status_t status = NS(TRACK_STATUS_GENERAL_FAILURE);

    if( ( pbuffer != SIXTRL_NULLPTR ) && ( num_psets > ( size_t )0u ) &&
        ( pset_indices_begin != SIXTRL_NULLPTR ) &&
        ( NS(Buffer_is_particles_buffer)( pbuffer ) ) &&
        ( belems_buffer != SIXTRL_NULLPTR ) &&
        ( NS(Buffer_get_num_of_objects)( belems_buffer ) > ( size_t )0u ) &&
        ( e_by_e_conf  != SIXTRL_NULLPTR ) &&
        ( NS(ElemByElemConfig_get_output_store_address)( e_by_e_conf ) !=
          ( address_t )0u ) )
    {
        size_t const* pset_it = pset_indices_begin;
        size_t const* pset_end = pset_it + num_psets;

        status = NS(TRACK_SUCCESS);

        for( ; pset_it != pset_end ; ++pset_it )
        {
            status |= NS(Track_all_particles_element_by_element_until_turn)(
                NS(Particles_buffer_get_particles)( pbuffer, *pset_it ),
                    e_by_e_conf, belems_buffer, until_turn );
        }
    }

    return status;
}

NS(track_status_t) NS(TestTrackCpu_track_particles_line_until_turn_cpu)(
    NS(Buffer)* SIXTRL_RESTRICT pbuffer,
    NS(buffer_size_t) const num_psets,
    NS(buffer_size_t) const* SIXTRL_RESTRICT pset_indices_begin,
    const NS(Buffer) *const SIXTRL_RESTRICT belems_buffer,
    NS(buffer_size_t) const num_line_segments,
    NS(buffer_size_t) const* SIXTRL_RESTRICT line_segments_begin_index_begin,
    NS(buffer_size_t) const* SIXTRL_RESTRICT line_segments_end_index_begin,
    NS(particle_index_t) const until_turn, bool const always_finish_line )
{
    typedef NS(buffer_size_t)  size_t;
    typedef NS(track_status_t) track_status_t;

    track_status_t status = NS(TRACK_STATUS_GENERAL_FAILURE);

    size_t const num_beam_elements =
        NS(Buffer_get_num_of_objects)( belems_buffer );

    if( ( pbuffer != SIXTRL_NULLPTR ) && ( num_psets > ( size_t )0u ) &&
        ( pset_indices_begin != SIXTRL_NULLPTR ) &&
        ( NS(Buffer_is_particles_buffer)( pbuffer ) ) &&
        ( belems_buffer != SIXTRL_NULLPTR ) &&
        ( num_beam_elements > ( size_t )0u ) &&
        ( num_line_segments > ( size_t )0u ) &&
        ( line_segments_begin_index_begin != SIXTRL_NULLPTR ) &&
        ( line_segments_end_index_begin   != SIXTRL_NULLPTR ) )
    {
        size_t const* pset_it = pset_indices_begin;
        size_t const* pset_end = pset_it + num_psets;

        size_t const* line_begin_idx_it  = line_segments_begin_index_begin;
        size_t const* line_begin_idx_end =
            line_begin_idx_it + num_line_segments;

        size_t const* line_end_idx_it = line_segments_end_index_begin;
        size_t prev_line_begin_idx = ( size_t )0u;
        size_t prev_line_end_idx = ( size_t )0u;

        bool first_segment = true;
        status = NS(TRACK_SUCCESS);

        for( ; line_begin_idx_it != line_begin_idx_end ;
                ++line_begin_idx_it, ++line_end_idx_it )
        {
            if( ( !first_segment ) &&
                ( ( *line_begin_idx_it <= prev_line_begin_idx ) ||
                  ( *line_end_idx_it   <= prev_line_end_idx   ) ||
                  ( *line_begin_idx_it <= *line_end_idx_it    ) ) )
            {
                status = NS(TRACK_STATUS_GENERAL_FAILURE);
                break;
            }

            first_segment = false;

            if( ( *line_begin_idx_it > num_beam_elements ) ||
                ( *line_end_idx_it > num_beam_elements ) )
            {
                status = NS(TRACK_STATUS_GENERAL_FAILURE);
                break;
            }

            prev_line_begin_idx = *line_begin_idx_it;
            prev_line_end_idx   = *line_end_idx_it;
        }

        if( status != NS(TRACK_SUCCESS) ) return status;

        if( *( line_segments_end_index_begin +
                ( num_line_segments - ( ( size_t )1u ) ) ) !=
            num_beam_elements )
        {
            status = NS(TRACK_STATUS_GENERAL_FAILURE);
        }

        if( status != NS(TRACK_SUCCESS) ) return status;

        for( ; pset_it != pset_end ; ++pset_it )
        {
            bool finish_line = false;
            bool first_particle_in_set = true;

            NS(particle_index_t) at_turn = ( NS(particle_index_t) )-1;
            NS(particle_index_t) at_element_id = ( NS(particle_index_t) )-1;

            size_t particle_idx = ( size_t )0u;

            NS(Particles)* particles = NS(Particles_buffer_get_particles)(
                pbuffer, *pset_it );

            size_t const num_particles =
                NS(Particles_get_num_of_particles)( particles );

            if( particles == SIXTRL_NULLPTR )
            {
                status = NS(TRACK_STATUS_GENERAL_FAILURE);
                break;
            }

            for( ; particle_idx < num_particles ; ++particle_idx )
            {
                if( ( first_particle_in_set ) &&
                    ( status == NS(TRACK_SUCCESS) ) )
                {
                    at_turn = NS(Particles_get_at_element_id_value)(
                        particles, particle_idx );

                    at_element_id = NS(Particles_get_at_turn_value)(
                        particles, particle_idx );

                    first_particle_in_set = false;
                }
                else if( status == NS(TRACK_SUCCESS) )
                {
                    if( ( at_turn != NS(Particles_get_at_turn_value)(
                            particles, particle_idx ) ) ||
                        ( at_element_id !=
                            NS(Particles_get_at_element_id_value)(
                                particles, particle_idx ) ) )
                    {
                        status = NS(TRACK_STATUS_GENERAL_FAILURE);
                        break;
                    }
                }
            }

            if( status != NS(TRACK_SUCCESS) )
            {
                break;
            }

            while( at_turn < until_turn )
            {
                line_begin_idx_it  = line_segments_begin_index_begin;
                line_begin_idx_end = line_begin_idx_it + num_line_segments;
                line_end_idx_it    = line_segments_end_index_begin;

                for( ; line_begin_idx_it != line_begin_idx_end ;
                        ++line_begin_idx_it, ++line_end_idx_it )
                {
                    finish_line = ( *line_end_idx_it == num_beam_elements );

                    status = NS(Track_all_particles_line)( particles,
                        belems_buffer, *line_begin_idx_it,
                            *line_end_idx_it, finish_line );

                    prev_line_begin_idx = *line_begin_idx_it;
                    prev_line_end_idx   = *line_end_idx_it;

                    if( status != NS(TRACK_SUCCESS) )
                    {
                        break;
                    }

                    if( finish_line )
                    {
                        ++at_turn;
                    }
                }

                if( ( status == NS(TRACK_SUCCESS) ) && ( !finish_line ) &&
                    ( always_finish_line ) )
                {
                    SIXTRL_ASSERT( prev_line_end_idx < num_beam_elements );

                    status = NS(Track_all_particles_line)(
                        particles, belems_buffer, prev_line_end_idx,
                            num_beam_elements, true );

                    ++at_turn;
                }

                if( status != NS(TRACK_SUCCESS) )
                {
                    break;
                }
            }

            if( status != NS(TRACK_SUCCESS) )
            {
                break;
            }
        }
    }

    return status;
}

/* end: tests/sixtracklib/testlib/common/track/track_particles_cpu.c */
