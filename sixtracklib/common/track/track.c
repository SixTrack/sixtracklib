#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/track/track.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stddef.h>
    #include <stdint.h>
    #include <stdlib.h>
    #include <math.h>
#endif /* #if !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #if !defined( _GPUCODE )
        #include "sixtracklib/common/buffer.h"
        #include "sixtracklib/common/particles.h"
    #endif /* !defined( _GPUCODE ) */
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( _GPUCODE )

NS(track_status_t) NS(Track_particle_until_turn)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_index,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const
        SIXTRL_RESTRICT belements_buffer,
        NS(particle_index_t) const until_turn )
{
    return NS(Track_particle_until_turn_objs)( particles, particle_index,
        NS(Buffer_get_const_objects_begin)( belements_buffer ),
        NS(Buffer_get_const_objects_end)( belements_buffer ), until_turn );
}

NS(track_status_t) NS(Track_all_particles_until_turn)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT be_buffer,
    NS(particle_index_t) const until_turn )
{
    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* be_iter_t;

    NS(track_status_t) status = SIXTRL_TRACK_SUCCESS;

    NS(particle_num_elements_t) ii = ( NS(particle_num_elements_t) )0u;
    NS(particle_num_elements_t) const nn =
        NS(Particles_get_num_of_particles)( particles );

    be_iter_t line_begin = NS(Buffer_get_const_objects_begin)( be_buffer );
    be_iter_t line_end = NS(Buffer_get_const_objects_end)( be_buffer );

    for( ; ii < nn ; ++ii )
    {
        status |= NS(Track_particle_until_turn_objs)(
            particles, ii, line_begin, line_end, until_turn );
    }

    return status;
}

NS(track_status_t) NS(Track_particle_element_by_element_until_turn)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_index,
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC const
        NS(ElemByElemConfig) *const SIXTRL_RESTRICT elem_by_elem_config,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const
        SIXTRL_RESTRICT belements_buffer,
    NS(particle_index_t) const until_turn )
{
    return NS(Track_particle_element_by_element_until_turn_objs)(
        particles, particle_index, elem_by_elem_config,
        NS(Buffer_get_const_objects_begin)( belements_buffer ),
        NS(Buffer_get_const_objects_end)( belements_buffer ), until_turn );
}

NS(track_status_t) NS(Track_all_particles_element_by_element_until_turn)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC const
        NS(ElemByElemConfig) *const SIXTRL_RESTRICT elem_by_elem_config,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const
        SIXTRL_RESTRICT belements_buffer,
    NS(particle_index_t) const until_turn )
{
    NS(track_status_t) status = SIXTRL_TRACK_SUCCESS;

    NS(particle_num_elements_t) ii = ( NS(particle_num_elements_t) )0u;
    NS(particle_num_elements_t) const nn =
        NS(Particles_get_num_of_particles)( particles );

    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* line_begin =
        NS(Buffer_get_const_objects_begin)( belements_buffer );

    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* line_end =
        NS(Buffer_get_const_objects_end)( belements_buffer );

    for( ; ii < nn ; ++ii )
    {
        status |= NS(Track_particle_element_by_element_until_turn_objs)(
            particles, ii, elem_by_elem_config, line_begin, line_end,
                until_turn );
    }

    return status;
}

NS(track_status_t) NS(Track_particle_line)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_index,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const
        SIXTRL_RESTRICT belements_buffer,
    NS(buffer_size_t) const line_begin_idx,
    NS(buffer_size_t) const line_end_idx, bool const finish_turn )
{
    return NS(Track_particle_line_objs)( particles, particle_index,
        NS(Buffer_get_const_object)( belements_buffer, line_begin_idx ),
        NS(Buffer_get_const_object)( belements_buffer, line_end_idx ),
        finish_turn );
}

NS(track_status_t) NS(Track_all_particles_line)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT be_buffer,
    NS(buffer_size_t) const begin_idx, NS(buffer_size_t) const end_idx,
    bool const finish_turn )
{
    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* be_iter_t;

    NS(track_status_t) status = SIXTRL_TRACK_SUCCESS;

    NS(particle_num_elements_t) ii = ( NS(particle_num_elements_t) )0u;
    NS(particle_num_elements_t) const nn =
        NS(Particles_get_num_of_particles)( particles );

    be_iter_t line_begin = NS(Buffer_get_const_object)( be_buffer, begin_idx );
    be_iter_t line_end = NS(Buffer_get_const_object)( be_buffer, end_idx );

    for( ; ii < nn ; ++ii )
    {
        status |= NS(Track_particle_line_objs)(
            particles, ii, line_begin, line_end, finish_turn );
    }

    return status;
}

#endif /* !defined( _GPUCODE ) */

/* end: /common/track/track.c */
