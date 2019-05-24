#include "sixtracklib/testlib/common/track/track_particles_cpu.h"

#include <stddef.h>
#include <stdlib.h>

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/particles.h"
#include "sixtracklib/common/buffer.h"
#include "sixtracklib/common/track/definitions.h"
#include "sixtracklib/common/track.h"

NS(track_status_t) NS(TestTrackCpu_track_particles_until_cpu)(
    NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
    NS(buffer_size_t) const num_particle_sets,
    NS(buffer_size_t) const* SIXTRL_RESTRICT particle_set_indices_begin,
    const NS(Buffer) *const SIXTRL_RESTRICT beam_elements_buffer,
    NS(particle_index_t) const until_turn )
{
    typedef NS(buffer_size_t) buf_size_t;

    NS(track_status_t) status = NS(TRACK_STATUS_GENERAL_FAILURE);

    if( ( particles_buffer != SIXTRL_NULLPTR ) &&
        ( num_particle_sets > ( buf_size_t )0u ) &&
        ( particle_set_indices_begin != SIXTRL_NULLPTR ) &&
        ( NS(Buffer_is_particles_buffer)( particles_buffer ) ) &&
        ( beam_elements_buffer != SIXTRL_NULLPTR ) )
    {
        buf_size_t const* pset_it = particle_set_indices_begin;
        buf_size_t const* pset_end = pset_it + num_particle_sets;

        status = NS(TRACK_SUCCESS);

        for( ; pset_it != pset_end ; ++pset_it )
        {
            NS(Particles)* particles = NS(Particles_buffer_get_particles)(
                particles_buffer, *pset_it );

            if( particles == SIXTRL_NULLPTR )
            {
                status = NS(TRACK_STATUS_GENERAL_FAILURE);
                break;
            }

            status = NS(Track_all_particles_until_turn)(
                particles, beam_elements_buffer, until_turn );

            if( status != NS(TRACK_SUCCESS) )
            {
                break;
            }
        }
    }

    return status;
}


NS(track_status_t) NS(TestTrackCpu_track_particles_elem_by_elem_until_turn_cpu)(
    NS(Buffer)* SIXTRL_RESTRICT particles_buffer, 
    NS(buffer_size_t) const num_particle_sets,
    NS(buffer_size_t) const* SIXTRL_RESTRICT particle_set_indices_begin,
    const NS(Buffer) *const SIXTRL_RESTRICT beam_elements_buffer,
    NS(ElemByElemConfig)* SIXTRL_RESTRICT elem_by_elem_config,
    NS(particle_index_t) const until_turn )
{
    typedef NS(buffer_size_t) buf_size_t;
    typedef NS(track_status_t) track_status_t;
    typedef NS(elem_by_elem_out_addr_t) address_t;
    
    track_status_t status = NS(TRACK_STATUS_GENERAL_FAILURE);

    if( ( particles_buffer != SIXTRL_NULLPTR ) &&
        ( num_particle_sets > ( buf_size_t )0u ) &&
        ( particle_set_indices_begin != SIXTRL_NULLPTR ) &&
        ( NS(Buffer_is_particles_buffer)( particles_buffer ) ) &&
        ( beam_elements_buffer != SIXTRL_NULLPTR ) &&
        ( NS(Buffer_get_num_of_objects)( beam_elements_buffer ) >
            ( buf_size_t )0u ) &&
        ( elem_by_elem_config  != SIXTRL_NULLPTR ) &&
        ( NS(ElemByElemConfig_get_output_store_address)( 
            elem_by_elem_config ) != ( address_t )0u ) )
    {
        buf_size_t const* pset_it = particle_set_indices_begin;
        buf_size_t const* pset_end = pset_it + num_particle_sets;
        
        buf_size_t const be_end_idx = 
            NS(Buffer_get_num_of_objects)( beam_elements_buffer );
            
        buf_size_t const be_begin_idx = ( buf_size_t )0u;

        status = NS(TRACK_SUCCESS);

        for( ; pset_it != pset_end ; ++pset_it )
        {
            NS(Particles)* particles = NS(Particles_buffer_get_particles)(
                particles_buffer, *pset_it );
            
            if( particles == SIXTRL_NULLPTR )
            {
                status = NS(TRACK_STATUS_GENERAL_FAILURE);
                break;
            }
            
            track_status = NS(Track_all_particles_element_by_elements_details)(
                particles, elem_by_elem_config, beam_elements_buffer, 
                    be_begin_idx, be_end_idx );
            
            if( track_status != NS(TRACK_SUCCESS) )
            {
                break;
            }
        }        
    }
    
    return track_status;
}


NS(track_status_t) NS(TestTrackCpu_track_particles_line_until_turn_cpu)(
    NS(Buffer)* SIXTRL_RESTRICT particles_buffer, 
    NS(buffer_size_t) const num_particle_sets, 
    NS(buffer_size_t) const* SIXTRL_RESTRICT particle_set_indices_begin,
    const NS(Buffer) *const SIXTRL_RESTRICT beam_elements_buffer, 
    NS(buffer_size_t) const num_line_segments, 
    NS(buffer_size_t) const* SIXTRL_RESTRICT line_segments_begin_index_begin, 
    NS(buffer_size_t) const* SIXTRL_RESTRICT line_segments_end_index_begin,
    NS(particle_index_t) const until_turn, bool const always_finish_line )
{
    typedef NS(buffer_size_t)  buf_size_t;
    typedef NS(track_status_t) track_status_t;
        
    track_status_t status = NS(TRACK_STATUS_GENERAL_FAILURE);
    
    buf_size_t const num_beam_elements = 
        NS(Buffer_get_num_of_objects)( beam_elements_buffer );

    if( ( particles_buffer != SIXTRL_NULLPTR ) &&
        ( num_particle_sets > ( buf_size_t )0u ) &&
        ( particle_set_indices_begin != SIXTRL_NULLPTR ) &&
        ( NS(Buffer_is_particles_buffer)( particles_buffer ) ) &&
        ( beam_elements_buffer != SIXTRL_NULLPTR ) &&
        ( num_beam_elements > ( buf_size_t )0u ) &&
        ( num_line_segments > ( buf_size_t )0u ) &&
        ( line_segments_begin_index_begin != SIXTRL_NULLPTR ) &&
        ( line_segments_end_index_begin   != SIXTRL_NULLPTR ) )
    {
        buf_size_t const* pset_it = particle_set_indices_begin;
        buf_size_t const* pset_end = pset_it + num_particle_sets;
        
        buf_size_t const be_end_idx = 
            NS(Buffer_get_num_of_objects)( beam_elements_buffer );
            
        buf_size_t const be_begin_idx = ( buf_size_t )0u;
        
        buf_size_t const* line_begin_idx_it  = line_segments_begin_index_begin;
        buf_size_t const* line_begin_idx_end = 
            line_begin_idx_it + num_line_segments;
            
        buf_size_t const* line_end_idx_it = line_segments_end_index;
        
        buf_size_t prev_line_begin_idx = ( buf_size_t )0u;
        buf_size_t prev_line_end_idx = ( buf_size_t )0u;
        
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
                ( num_line_segments - ( ( buf_size_t )1u ) ) ) !=
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
            
            buf_size_t particle_idx = ( buf_size_t )0u;
            buf_size_t const num_particles = 
                NS(Particles_get_num_of_particles)( particles );
            
            NS(Particles)* particles = NS(Particles_buffer_get_particles)(
                particles_buffer, *pset_it );
            
            if( particles == SIXTRL_NULLPTR )
            {
                status = NS(TRACK_STATUS_GENERAL_FAILURE);
                break;
            }
            
            for( ; particle_idx < num_particles ; ++particle_idx )
            {
                if( ( first_particle_in_set ) && 
                    ( track_status == NS(TRACK_SUCCESS) ) )
                {
                    at_turn = NS(Particles_get_at_element_id_value)(
                        particles, particle_idx );
                    
                    at_element_id = NS(Particles_get_at_turn_value)(
                        particles, particle_idx );
                    
                    first_particle_in_set = false;
                }
                else if( track_status == NS(TRACK_SUCCESS) )
                {
                    if( ( at_turn != NS(Particles_get_at_turn_value)(
                            particles, particle_idx ) ) ||
                        ( at_element_id != 
                            NS(Particles_get_at_element_id_value)(
                                particles, particle_idx ) ) )
                    {
                        track_status = NS(TRACK_STATUS_GENERAL_FAILURE);
                        break;
                    }
                }   
            }
            
            if( track_status != NS(TRACK_SUCCESS) )
            {
                break;
            }
            
            while( at_turn < until_turn )
            {
                line_begin_idx_it  = line_segments_begin_index;
                line_begin_idx_end = line_begin_idx_it + num_line_segments;
                line_end_idx_it    = line_segments_end_index;
            
                for( ; line_begin_idx_it != line_begin_idx_end ; 
                        ++line_begin_idx_it, ++line_end_idx_it )
                {
                    finish_line = ( *line_end_idx_it == num_beam_elements );
                    
                    track_staus = NS(Track_all_particles_line_ext)( particles, 
                        beam_elements_buffer, *line_begin_idx_it, 
                            *line_end_idx_it, finish_line );
                    
                    prev_line_begin_idx = *line_begin_idx_it;
                    prev_line_end_idx   = *line_end_idx_it;
                    
                    if( track_status != NS(TRACK_SUCCESS) )
                    {
                        break;
                    }
                    
                    if( finish_line )
                    {
                        ++at_turn;
                    }
                }
                
                if( ( track_status == NS(TRACK_SUCCESS) ) && ( !finish_line ) &&
                    ( always_finish_line ) )
                {
                    SIXTRL_ASSERT( prev_line_end_idx < num_beam_elements );
                    
                    track_status = NS(Track_all_particles_line_ext)(
                        particles, beam_elements_buffer, *prev_line_end_idx, 
                            num_beam_elements, true );
                    
                    ++at_turn;
                }
                
                if( track_status != NS(TRACK_SUCCESS) )
                {
                    break;
                }                                
            }
            
            if( track_status != NS(TRACK_SUCCESS) )
            {
                break;
            }
        }
    }
    
    return track_status;
}

/* end: tests/sixtracklib/testlib/common/track/track_particles_cpu.c */
