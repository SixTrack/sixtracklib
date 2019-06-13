#include "sixtracklib/testlib/common/track/track_job_setup.h"

#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/particles.h"
#include "sixtracklib/common/buffer.h"
#include "sixtracklib/common/be_monitor/be_monitor.h"
#include "sixtracklib/common/output/elem_by_elem_config.h"
#include "sixtracklib/common/output/output_buffer.h"
#include "sixtracklib/common/track/track_job_base.h"

int NS(TestTrackJob_compare_particle_set_indices_lists)(
    NS(buffer_size_t) const  lhs_length,
    NS(buffer_size_t) const* SIXTRL_RESTRICT lhs_particles_set_indices_begin,
    NS(buffer_size_t) const  rhs_length,
    NS(buffer_size_t) const* SIXTRL_RESTRICT rhs_particles_set_indices_begin )
{
    int cmp_result = -1;

    if( lhs_length == rhs_length )
    {
        SIXTRL_ASSERT( lhs_particles_set_indices_begin != SIXTRL_NULLPTR );
        SIXTRL_ASSERT( rhs_particles_set_indices_begin != SIXTRL_NULLPTR );

        NS(buffer_size_t) const* lhs_it  = lhs_particles_set_indices_begin;
        NS(buffer_size_t) const* lhs_end = lhs_it + lhs_length;
        NS(buffer_size_t) const* rhs_it = rhs_particles_set_indices_begin;

        cmp_result = 0;

        for( ; lhs_it != lhs_end ; ++lhs_it, ++rhs_it )
        {
            if( *lhs_it != *rhs_it )
            {
                if( *lhs_it < *rhs_it )
                {
                    cmp_result = +1;
                    break;
                }
                else
                {
                    cmp_result = -1;
                    break;
                }
            }
        }
    }
    else if( lhs_length < rhs_length )
    {
        cmp_result = +1;
    }

    return cmp_result;
}

bool NS(TestTrackJob_setup_no_required_output)(
    const NS(TrackJobBaseNew) *const SIXTRL_RESTRICT job,
    const NS(Buffer) *const SIXTRL_RESTRICT particles_buffer,
    NS(buffer_size_t) const num_psets,
    NS(buffer_size_t) const* SIXTRL_RESTRICT pset_indices_begin,
    const NS(Buffer) *const SIXTRL_RESTRICT beam_elements_buffer,
    const NS(Buffer) *const SIXTRL_RESTRICT ext_output_buffer )
{
    typedef NS(buffer_size_t) buf_size_t;

    SIXTRL_STATIC_VAR buf_size_t const ZERO = ( buf_size_t )0u;

    bool success = (
        ( job != SIXTRL_NULLPTR ) &&
        ( ( ( ext_output_buffer == SIXTRL_NULLPTR ) &&
            ( !NS(TrackJobNew_has_output_buffer)(  job ) ) &&
            ( !NS(TrackJobNew_owns_output_buffer)( job ) ) ) ||
          ( ( ext_output_buffer != SIXTRL_NULLPTR ) &&
            (  NS(TrackJobNew_has_output_buffer)( job ) ) &&
            ( !NS(TrackJobNew_owns_output_buffer)( job ) ) ) ) &&
        ( !NS(TrackJobNew_has_beam_monitor_output)( job ) ) &&
        (  NS(TrackJobNew_get_num_beam_monitors)( job ) == ZERO ) );

    if( success )
    {
        success = (
          ( NS(TrackJobNew_get_beam_monitor_indices_begin)( job ) == SIXTRL_NULLPTR ) &&
          ( NS(TrackJobNew_get_beam_monitor_indices_end)( job ) == SIXTRL_NULLPTR ) );
    }

    if( success )
    {
        success = ( 0 == NS(TestTrackJob_compare_particle_set_indices_lists)(
            num_psets, pset_indices_begin,
            NS(TrackJobNew_get_num_particle_sets)( job ),
            NS(TrackJobNew_get_particle_set_indices_begin)( job ) ) );
    }

    if( success )
    {
        success = (
            ( !NS(TrackJobNew_has_elem_by_elem_output)( job ) ) &&
            ( !NS(TrackJobNew_has_elem_by_elem_config)( job ) ) &&
            (  NS(TrackJobNew_get_elem_by_elem_config)(
                job ) == SIXTRL_NULLPTR ) );
    }

    if( success )
    {
        if( ext_output_buffer != SIXTRL_NULLPTR )
        {
            success = (
                (  NS(TrackJobNew_get_const_output_buffer)(
                    job ) == ext_output_buffer ) &&
                (  NS(TrackJobNew_has_output_buffer)( job ) ) &&
                ( !NS(TrackJobNew_owns_output_buffer)( job ) ) );
        }
        else
        {
            success = (
                ( NS(TrackJobNew_get_const_output_buffer)(
                    job ) == SIXTRL_NULLPTR ) &&
                ( !NS(TrackJobNew_has_output_buffer)( job ) ) &&
                ( !NS(TrackJobNew_owns_output_buffer)( job ) ) );
        }
    }

    if( success )
    {
        success = ( ( NS(TrackJobNew_get_const_particles_buffer)(
                        job ) == particles_buffer ) &&
                    ( NS(TrackJobNew_get_const_beam_elements_buffer)(
                        job ) == beam_elements_buffer ) );
    }

    return success;
}

bool NS(TestTrackJob_setup_no_beam_monitors_elem_by_elem)(
    const NS(TrackJobBaseNew) *const SIXTRL_RESTRICT job,
    const NS(Buffer) *const SIXTRL_RESTRICT particles_buffer,
    NS(buffer_size_t) const num_psets,
    NS(buffer_size_t) const* SIXTRL_RESTRICT pset_indices_begin,
    const NS(Buffer) *const SIXTRL_RESTRICT beam_elements_buffer,
    const NS(Buffer) *const SIXTRL_RESTRICT ext_output_buffer,
    NS(buffer_size_t) const until_turn_elem_by_elem )
{
    bool success = false;

    typedef NS(buffer_size_t)       buf_size_t;
    typedef NS(Buffer)              c_buffer_t;
    typedef NS(Particles)           particles_t;
    typedef NS(ElemByElemConfig)    elem_by_elem_conf_t;

    SIXTRL_STATIC_VAR buf_size_t const ZERO = ( buf_size_t )0u;
    SIXTRL_STATIC_VAR buf_size_t const ONE  = ( buf_size_t )1u;

    c_buffer_t const* output_buffer = SIXTRL_NULLPTR;
    elem_by_elem_conf_t const* elem_by_elem_conf = SIXTRL_NULLPTR;

    SIXTRL_ASSERT( until_turn_elem_by_elem > ZERO );
    SIXTRL_ASSERT( particles_buffer != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( beam_elements_buffer != SIXTRL_NULLPTR );

    particles_t const* particles =
    NS(Particles_buffer_get_const_particles)(
            particles_buffer, ZERO );

    SIXTRL_ASSERT( particles != SIXTRL_NULLPTR );

    buf_size_t const NUM_BEAM_ELEMENTS =
        NS(Buffer_get_num_of_objects)( beam_elements_buffer );

    buf_size_t const NUM_PARTICLES =
        NS(Particles_get_num_of_particles)( particles );

    success = ( ( job != SIXTRL_NULLPTR ) &&
        ( NUM_BEAM_ELEMENTS > ZERO ) &&
        ( NUM_PARTICLES > ZERO ) &&
        (  NS(TrackJobNew_has_output_buffer)(  job ) ) &&
        ( !NS(TrackJobNew_has_beam_monitor_output)( job ) ) &&
        (  NS(TrackJobNew_get_num_beam_monitors)( job ) == ZERO ) &&
        (  NS(TrackJobNew_get_beam_monitor_indices_begin)(
            job ) == SIXTRL_NULLPTR ) &&
        (  NS(TrackJobNew_get_beam_monitor_indices_end)(
            job ) == SIXTRL_NULLPTR ) );

    if( success )
    {
        success = ( 0 == NS(TestTrackJob_compare_particle_set_indices_lists)(
            num_psets, pset_indices_begin,
            NS(TrackJobNew_get_num_particle_sets)( job ),
            NS(TrackJobNew_get_particle_set_indices_begin)( job ) ) );
    }

    if( success )
    {
        elem_by_elem_conf = NS(TrackJobNew_get_elem_by_elem_config)( job );

        success = ( ( elem_by_elem_conf != SIXTRL_NULLPTR ) &&
            ( NS(ElemByElemConfig_is_active)( elem_by_elem_conf ) ) &&
            ( ( buf_size_t )(
                NS(ElemByElemConfig_get_out_store_num_particles)(
                    elem_by_elem_conf ) ) >=
                ( NUM_PARTICLES * NUM_BEAM_ELEMENTS
                    * until_turn_elem_by_elem ) ) &&
            ( NS(ElemByElemConfig_is_rolling)( elem_by_elem_conf ) ==
              NS(TrackJobNew_get_default_elem_by_elem_config_rolling_flag)(
                    job ) ) );
    }

    if( success )
    {
        output_buffer = NS(TrackJobNew_get_const_output_buffer)( job );

        success = ( ( output_buffer != SIXTRL_NULLPTR ) &&
            ( NS(Buffer_get_num_of_objects)(
                output_buffer ) == ONE ) &&
            ( NS(Buffer_is_particles_buffer)( output_buffer ) ) &&
            ( NS(Particles_get_num_of_particles)(
                NS(Particles_buffer_get_const_particles)(
                    output_buffer, ZERO ) ) >=
              NS(ElemByElemConfig_get_out_store_num_particles)(
                elem_by_elem_conf ) ) &&
            (  NS(TrackJobNew_get_beam_monitor_output_buffer_offset)(
                job ) == ONE ) &&
            (  NS(TrackJobNew_get_beam_monitor_output_buffer_offset)(
                job ) >= NS(Buffer_get_num_of_objects)( output_buffer )
            ) );
    }

    if( success )
    {
        if( ext_output_buffer != SIXTRL_NULLPTR )
        {
            success = (
                ( !NS(TrackJobNew_owns_output_buffer)( job ) ) &&
                (  NS(TrackJobNew_get_const_output_buffer)(
                    job ) == ext_output_buffer ) &&
                (  NS(TrackJobNew_has_output_buffer)( job ) ) &&
                ( !NS(TrackJobNew_owns_output_buffer)( job ) ) );
        }
        else
        {
            success = (
                ( NS(TrackJobNew_owns_output_buffer)( job ) ) &&
                ( NS(TrackJobNew_get_const_output_buffer)(
                    job ) != SIXTRL_NULLPTR ) &&
                (  NS(TrackJobNew_has_output_buffer)( job ) ) &&
                (  NS(TrackJobNew_owns_output_buffer)( job ) ) );
        }
    }

    if( success )
    {
        success = ( ( NS(TrackJobNew_get_const_particles_buffer)(
                        job ) == particles_buffer ) &&
                    ( NS(TrackJobNew_get_const_beam_elements_buffer)(
                        job ) == beam_elements_buffer ) );
    }

    return success;
}

bool NS(TestTrackJob_setup_beam_monitors_and_elem_by_elem)(
    const NS(TrackJobBaseNew) *const SIXTRL_RESTRICT job,
    const NS(Buffer) *const SIXTRL_RESTRICT particles_buffer,
    NS(buffer_size_t) const num_psets,
    NS(buffer_size_t) const* SIXTRL_RESTRICT pset_indices_begin,
    const NS(Buffer) *const SIXTRL_RESTRICT beam_elements_buffer,
    const NS(Buffer) *const SIXTRL_RESTRICT ext_output_buffer,
    NS(buffer_size_t) const num_beam_monitors,
    NS(buffer_size_t) const until_turn,
    NS(buffer_size_t) const until_turn_elem_by_elem )
{
    bool success = false;

    typedef NS(buffer_size_t)       buf_size_t;
    typedef NS(Buffer)              c_buffer_t;
    typedef NS(Particles)           particles_t;
    typedef NS(ElemByElemConfig)    elem_by_elem_conf_t;

    SIXTRL_STATIC_VAR buf_size_t const ZERO = ( buf_size_t )0u;
    SIXTRL_STATIC_VAR buf_size_t const ONE  = ( buf_size_t )1u;

    c_buffer_t const* output_buffer = SIXTRL_NULLPTR;
    elem_by_elem_conf_t const* elem_by_elem_conf = SIXTRL_NULLPTR;

    SIXTRL_ASSERT( particles_buffer != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( beam_elements_buffer != SIXTRL_NULLPTR );

    particles_t const* particles = NS(Particles_buffer_get_const_particles)(
            particles_buffer, ZERO );

    SIXTRL_ASSERT( particles != SIXTRL_NULLPTR );

    buf_size_t const NUM_BEAM_ELEMENTS =
        NS(Buffer_get_num_of_objects)( beam_elements_buffer );

    buf_size_t const NUM_PARTICLES =
        NS(Particles_get_num_of_particles)( particles );

    success = ( ( job != SIXTRL_NULLPTR ) &&
        ( NUM_BEAM_ELEMENTS > ZERO ) &&
        ( NUM_PARTICLES > ZERO ) &&
        (  NS(TrackJobNew_has_output_buffer)(  job ) ) &&
        (  NS(TrackJobNew_has_beam_monitor_output)( job ) ) &&
        (  NS(TrackJobNew_get_num_beam_monitors)( job ) == num_beam_monitors ) );

    if( success )
    {
        success = ( 0 == NS(TestTrackJob_compare_particle_set_indices_lists)(
            num_psets, pset_indices_begin,
            NS(TrackJobNew_get_num_particle_sets)( job ),
            NS(TrackJobNew_get_particle_set_indices_begin)( job ) ) );
    }

    if( ( success ) && ( num_beam_monitors > ZERO ) )
    {
        buf_size_t const* be_mon_idx_it  =
            NS(TrackJobNew_get_beam_monitor_indices_begin)( job );

        buf_size_t const* be_mon_idx_end =
            NS(TrackJobNew_get_beam_monitor_indices_end)( job );

        success = ( ( job != SIXTRL_NULLPTR ) &&
            ( be_mon_idx_it  != SIXTRL_NULLPTR ) &&
            ( be_mon_idx_end != SIXTRL_NULLPTR ) &&
            ( NS(TrackJobNew_get_beam_monitor_output_buffer_offset)( job ) >=
              NS(TrackJobNew_get_elem_by_elem_output_buffer_offset)( job ) ) &&
            ( ( ( ptrdiff_t )0 ) < ( ( ( intptr_t )be_mon_idx_end ) -
                ( intptr_t )be_mon_idx_it ) ) &&
            ( ( buf_size_t )( be_mon_idx_end - be_mon_idx_it ) ==
                num_beam_monitors ) );

        if( success )
        {
            for( ; be_mon_idx_it != be_mon_idx_end ; ++be_mon_idx_it )
            {
                success &= ( NS(OBJECT_TYPE_BEAM_MONITOR) ==
                    NS(Object_get_type_id)( NS(Buffer_get_const_object)(
                        beam_elements_buffer, *be_mon_idx_it ) ) );
            }
        }
    }

    if( ( success ) && ( until_turn_elem_by_elem > ZERO ) )
    {
        elem_by_elem_conf = NS(TrackJobNew_get_elem_by_elem_config)( job );

        success = (
            (  NS(TrackJobNew_has_elem_by_elem_output)( job ) ) &&
            (  NS(TrackJobNew_has_elem_by_elem_config)( job ) ) &&
            (  NS(TrackJobNew_get_elem_by_elem_output_buffer_offset)( job ) == ZERO ) &&
            ( elem_by_elem_conf != SIXTRL_NULLPTR ) &&
            ( NS(ElemByElemConfig_is_active)( elem_by_elem_conf ) ) &&
            ( ( buf_size_t )(
                NS(ElemByElemConfig_get_out_store_num_particles)( elem_by_elem_conf ) ) >=
                ( NUM_PARTICLES * NUM_BEAM_ELEMENTS * until_turn_elem_by_elem ) ) &&
            ( NS(ElemByElemConfig_is_rolling)( elem_by_elem_conf ) ==
              NS(TrackJobNew_get_default_elem_by_elem_config_rolling_flag)( job ) ) );
    }

    if( ( success ) &&
        ( ( until_turn_elem_by_elem > ZERO ) || ( until_turn > ZERO ) ||
          ( num_beam_monitors > ZERO ) ) )
    {
        output_buffer = NS(TrackJobNew_get_const_output_buffer)( job );

        buf_size_t requ_num_output_elems = num_beam_monitors;

        if( NS(TrackJobNew_has_elem_by_elem_output)( job ) )
        {
            requ_num_output_elems += ONE;
        }

        success = ( ( output_buffer != SIXTRL_NULLPTR ) &&
            ( NS(Buffer_get_num_of_objects)(
                output_buffer ) == requ_num_output_elems ) &&
            ( NS(Buffer_is_particles_buffer)( output_buffer ) ) );
    }

    if( ( success ) && ( until_turn_elem_by_elem > ZERO ) &&
        ( elem_by_elem_conf != SIXTRL_NULLPTR ) )
    {
        success = (
            ( NS(Particles_get_num_of_particles)(
                NS(Particles_buffer_get_const_particles)( output_buffer,
                NS(TrackJobNew_get_elem_by_elem_output_buffer_offset)(
                    job ) ) ) >=
              NS(ElemByElemConfig_get_out_store_num_particles)(
                elem_by_elem_conf ) ) );
    }


    if( ( success ) && ( until_turn > ZERO ) &&
        ( num_beam_monitors > ZERO ) )
    {
        success = (
            (  NS(TrackJobNew_get_beam_monitor_output_buffer_offset)(
                    job ) >=
                NS(TrackJobNew_get_elem_by_elem_output_buffer_offset)(
                    job ) ) &&
            (  NS(TrackJobNew_get_beam_monitor_output_buffer_offset)(
                job ) < NS(Buffer_get_num_of_objects)( output_buffer )
            ) );
    }

    if( success )
    {
        if( ext_output_buffer != SIXTRL_NULLPTR )
        {
            success = (
                ( !NS(TrackJobNew_owns_output_buffer)( job ) ) &&
                (  NS(TrackJobNew_get_const_output_buffer)(
                    job ) == ext_output_buffer ) &&
                (  NS(TrackJobNew_has_output_buffer)( job ) ) &&
                ( !NS(TrackJobNew_owns_output_buffer)( job ) ) );
        }
        else
        {
            success = (
                ( NS(TrackJobNew_owns_output_buffer)( job ) ) &&
                ( NS(TrackJobNew_get_const_output_buffer)(
                    job ) != SIXTRL_NULLPTR ) &&
                (  NS(TrackJobNew_has_output_buffer)( job ) ) &&
                (  NS(TrackJobNew_owns_output_buffer)( job ) ) );
        }
    }

    if( success )
    {
        success = (
            ( NS(TrackJobNew_get_const_particles_buffer)( job ) ==
                particles_buffer ) &&
            ( NS(TrackJobNew_get_const_beam_elements_buffer)( job ) ==
                beam_elements_buffer ) );
    }

    return success;
}

/* end: tests/sixtracklib/testlib/common/track/track_job_setup_c99.cpp */
