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
    NS(buffer_size_t) const until_turn )
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

/* end: tests/sixtracklib/testlib/common/track/track_particles_cpu.c */
