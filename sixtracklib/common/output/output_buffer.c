#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/output/output_buffer.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stdbool.h>
    #include <stdint.h>
    #include <stdlib.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/buffer/buffer_type.h"
    #include "sixtracklib/common/buffer/buffer_object.h"
    #include "sixtracklib/common/buffer/buffer_garbage.h"
    #include "sixtracklib/common/buffer.h"
    #include "sixtracklib/common/internal/objects_type_id.h"
    #include "sixtracklib/common/be_monitor/be_monitor.h"
    #include "sixtracklib/common/output/elem_by_elem_config.h"
    #include "sixtracklib/common/particles.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

/* ------------------------------------------------------------------------- */
/* OutputBuffer preparation functions: */
/* ------------------------------------------------------------------------- */

SIXTRL_HOST_FN int NS(OutputBuffer_prepare)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT belements,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT output_buffer,
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT p,
    NS(buffer_size_t) const target_num_output_turns,
    NS(buffer_size_t) const target_num_elem_by_elem_turns,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT
        ptr_elem_by_elem_out_index_offset,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT
        ptr_beam_monitor_out_index_offset,
    SIXTRL_ARGPTR_DEC NS(particle_index_t)* SIXTRL_RESTRICT ptr_min_turn_id )
{
    int success = -1;

    typedef NS(particle_index_t)    index_t;

    SIXTRL_ASSERT( belements     != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( output_buffer != SIXTRL_NULLPTR );

    SIXTRL_ASSERT( !NS(Buffer_needs_remapping)( belements ) );
    SIXTRL_ASSERT( !NS(Buffer_needs_remapping)( output_buffer ) );

    if( p != SIXTRL_NULLPTR )
    {
        index_t min_particle_id = ( index_t )0;
        index_t max_particle_id = ( index_t )-1;

        index_t min_element_id  = ( index_t )0;
        index_t max_element_id  = ( index_t )-1;

        index_t min_turn_id     = ( index_t )0;
        index_t max_turn_id     = ( index_t )-1;

        if( 0 == NS(Particles_get_min_max_attributes)(
            p, &min_particle_id, &max_particle_id, &min_element_id,
               &max_element_id, &min_turn_id, &max_turn_id ) )
        {
            index_t temp_min_element_id = min_element_id;
            index_t temp_max_element_id = max_element_id;

            SIXTRL_ASSERT( min_turn_id     >= ( index_t )0u );
            SIXTRL_ASSERT( min_turn_id     <= max_turn_id );

            SIXTRL_ASSERT( min_element_id  >= ( index_t )0u );
            SIXTRL_ASSERT( min_element_id  <= max_element_id );

            SIXTRL_ASSERT( min_particle_id >= ( index_t )0u );
            SIXTRL_ASSERT( max_particle_id >= min_particle_id );

            success = NS(ElemByElemConfig_get_min_max_element_id_from_buffer)(
                belements, &temp_min_element_id, &temp_max_element_id,
                    SIXTRL_NULLPTR, 0 );

            if( success == 0 )
            {
                if( min_element_id > temp_min_element_id )
                {
                    min_element_id = temp_min_element_id;
                }

                if( max_element_id < temp_max_element_id )
                {
                    max_element_id = temp_max_element_id;
                }
            }

            if( success == 0 )
            {
                success = NS(OutputBuffer_prepare_detailed)(
                    belements, output_buffer, min_particle_id, max_particle_id,
                    min_element_id, max_element_id, min_turn_id, max_turn_id,
                    target_num_output_turns, target_num_elem_by_elem_turns,
                    ptr_elem_by_elem_out_index_offset,
                    ptr_beam_monitor_out_index_offset, SIXTRL_NULLPTR );

                if( ( success == 0 ) && ( ptr_min_turn_id != SIXTRL_NULLPTR ) )
                {
                    *ptr_min_turn_id = min_turn_id;
                }
            }
        }
    }

    return success;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

int NS(OutputBuffer_prepare_for_particle_set)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT belements,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT output_buffer,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT pb,
    NS(buffer_size_t) const num_particle_sets,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t) const* particle_set_indices_begin,
    NS(buffer_size_t) const target_num_output_turns,
    NS(buffer_size_t) const target_num_elem_by_elem_turns,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT
        ptr_elem_by_elem_out_index_offset,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT
        ptr_beam_monitor_out_index_offset,
    SIXTRL_ARGPTR_DEC NS(particle_index_t)* SIXTRL_RESTRICT ptr_min_turn_id )
{
    int success = -1;

    typedef NS(particle_index_t) index_t;

    index_t min_particle_id = ( index_t )0;
    index_t max_particle_id = ( index_t )-1;

    index_t min_element_id  = ( index_t )0;
    index_t max_element_id  = ( index_t )-1;

    index_t min_turn_id     = ( index_t )0;
    index_t max_turn_id     = ( index_t )-1;

    SIXTRL_ASSERT( pb            != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( belements     != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( output_buffer != SIXTRL_NULLPTR );

    SIXTRL_ASSERT( !NS(Buffer_needs_remapping)( pb ) );
    SIXTRL_ASSERT( !NS(Buffer_needs_remapping)( belements ) );
    SIXTRL_ASSERT( !NS(Buffer_needs_remapping)( output_buffer ) );

    if( 0 == NS(Particles_buffer_get_min_max_attributes_of_particles_set)(
        pb, num_particle_sets, particle_set_indices_begin, &min_particle_id,
        &max_particle_id, &min_element_id, &max_element_id,
        &min_turn_id, &max_turn_id ) )
    {
        index_t temp_min_element_id = min_element_id;
        index_t temp_max_element_id = max_element_id;

        SIXTRL_ASSERT( min_turn_id     >= ( index_t )0u );
        SIXTRL_ASSERT( min_turn_id     <= max_turn_id );

        SIXTRL_ASSERT( min_element_id  >= ( index_t )0u );
        SIXTRL_ASSERT( min_element_id  <= max_element_id );

        SIXTRL_ASSERT( min_particle_id >= ( index_t )0u );
        SIXTRL_ASSERT( max_particle_id <= ( index_t )0u );

        success = NS(ElemByElemConfig_get_min_max_element_id_from_buffer)(
            belements, &temp_min_element_id, &temp_max_element_id,
                SIXTRL_NULLPTR, 0 );

        if( success == 0 )
        {
            if( min_element_id > temp_min_element_id )
            {
                min_element_id = temp_min_element_id;
            }

            if( max_element_id < temp_max_element_id )
            {
                max_element_id = temp_max_element_id;
            }
        }

        if( success == 0 )
        {
            success = NS(OutputBuffer_prepare_detailed)(
                belements, output_buffer, min_particle_id, max_particle_id,
                min_element_id, max_element_id, min_turn_id, max_turn_id,
                target_num_output_turns, target_num_elem_by_elem_turns,
                ptr_elem_by_elem_out_index_offset,
                ptr_beam_monitor_out_index_offset, SIXTRL_NULLPTR );

            if( ( success == 0 ) && ( ptr_min_turn_id != SIXTRL_NULLPTR ) )
            {
                *ptr_min_turn_id = min_turn_id;
            }
        }
    }

    return success;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

int NS(OutputBuffer_prepare_detailed)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT belements,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT output_buffer,
    NS(particle_index_t)     const min_particle_id,
    NS(particle_index_t)     const max_particle_id,
    NS(particle_index_t)     const min_element_id,
    NS(particle_index_t)     const max_element_id,
    NS(particle_index_t)     const min_turn_id,
    NS(particle_index_t)     const max_turn_id,
    NS(buffer_size_t)              target_num_output_turns,
    NS(buffer_size_t)        const target_num_elem_by_elem_turns,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT
        ptr_elem_by_elem_out_index_offset,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT
        ptr_beam_monitor_out_index_offset,
    SIXTRL_ARGPTR_DEC NS(particle_index_t)* SIXTRL_RESTRICT
        ptr_max_elem_by_elem_turn_id )
{
    int success = -1;

    typedef NS(buffer_size_t)       buf_size_t;
    typedef NS(particle_index_t)    index_t;

    buf_size_t elem_by_elem_index_offset = ( buf_size_t )0u;
    buf_size_t beam_monitor_index_offset = ( buf_size_t )0u;
    index_t    max_elem_by_elem_turn_id  = min_turn_id;

    SIXTRL_ASSERT( belements     != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( output_buffer != SIXTRL_NULLPTR );

    SIXTRL_ASSERT( !NS(Buffer_needs_remapping)( belements ) );
    SIXTRL_ASSERT( !NS(Buffer_needs_remapping)( output_buffer ) );

    if( target_num_output_turns < target_num_elem_by_elem_turns )
    {
        target_num_output_turns = target_num_elem_by_elem_turns;
    }

    if( target_num_elem_by_elem_turns > ( buf_size_t )0u )
    {
        max_elem_by_elem_turn_id += target_num_elem_by_elem_turns;

        if( max_elem_by_elem_turn_id < max_turn_id )
        {
            max_elem_by_elem_turn_id = max_turn_id;
        }

        success = NS(ElemByElemConfig_prepare_output_buffer_detailed)(
            output_buffer, min_particle_id, max_particle_id,
            min_element_id, max_element_id, min_turn_id,
            max_elem_by_elem_turn_id, &elem_by_elem_index_offset );
    }
    else
    {
        success = 0;
    }

    if( success == 0 )
    {
        success = NS(BeamMonitor_prepare_output_buffer_detailed)(
            belements, output_buffer, min_particle_id, max_particle_id,
            min_turn_id, &beam_monitor_index_offset );
    }

    if( success == 0 )
    {
        if( ptr_elem_by_elem_out_index_offset != SIXTRL_NULLPTR )
        {
            *ptr_elem_by_elem_out_index_offset = elem_by_elem_index_offset;
        }

        if( ptr_beam_monitor_out_index_offset != SIXTRL_NULLPTR )
        {
            *ptr_beam_monitor_out_index_offset = beam_monitor_index_offset;
        }

        if( ptr_max_elem_by_elem_turn_id != SIXTRL_NULLPTR )
        {
            *ptr_max_elem_by_elem_turn_id = max_elem_by_elem_turn_id;
        }
    }

    return success;
}

/* ------------------------------------------------------------------------- */
/* NS(ElemByElemConfig) based prepare output functions: */
/* ------------------------------------------------------------------------- */

int NS(ElemByElemConfig_prepare_output_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT belements,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT output_buffer,
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT p,
    NS(buffer_size_t) const num_elem_by_elem_turns,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_index_offset )
{
    int success = -1;

    typedef NS(particle_index_t) index_t;
    typedef NS(buffer_size_t)    buf_size_t;

    index_t min_particle_id = ( index_t )0;
    index_t max_particle_id = ( index_t )-1;

    index_t min_element_id  = ( index_t )0;
    index_t max_element_id  = ( index_t )-1;

    index_t min_turn_id     = ( index_t )0;
    index_t max_turn_id     = ( index_t )-1;

    SIXTRL_ASSERT( belements     != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( output_buffer != SIXTRL_NULLPTR );

    SIXTRL_ASSERT( !NS(Buffer_needs_remapping)( belements ) );
    SIXTRL_ASSERT( !NS(Buffer_needs_remapping)( output_buffer ) );

    if( ( num_elem_by_elem_turns > ( buf_size_t )0u ) &&
        ( 0 == NS(Particles_get_min_max_attributes)( p,
            &min_particle_id, &max_particle_id, &min_element_id,
            &max_element_id, &min_turn_id, &max_turn_id ) ) )
    {
        index_t temp_min_element_id = ( index_t )0;
        index_t temp_max_element_id = ( index_t )-1;
        buf_size_t num_elem_by_elem_objects = ( buf_size_t )0u;

        success = NS(ElemByElemConfig_get_min_max_element_id_from_buffer)(
            belements, &temp_min_element_id, &temp_max_element_id,
                &num_elem_by_elem_objects, ( index_t )0u );

        SIXTRL_ASSERT( min_turn_id     >= ( index_t )0u );
        SIXTRL_ASSERT( min_turn_id     <= max_turn_id );

        SIXTRL_ASSERT( min_element_id  >= ( index_t )0u );
        SIXTRL_ASSERT( min_element_id  <= max_element_id );

        SIXTRL_ASSERT( min_particle_id >= ( index_t )0u );
        SIXTRL_ASSERT( max_particle_id >= min_particle_id );

        if( ( success == 0 ) &&
            ( num_elem_by_elem_objects > ( buf_size_t )0u ) )
        {
            SIXTRL_ASSERT( temp_min_element_id >= ( index_t )0u );
            SIXTRL_ASSERT( temp_min_element_id <= temp_max_element_id );

            if( min_element_id > temp_min_element_id )
            {
                min_element_id = temp_min_element_id;
            }

            if( max_element_id < temp_max_element_id )
            {
                max_element_id = temp_max_element_id;
            }
        }

        if( success == 0 )
        {
            index_t const max_elem_by_elem_turn_id =
                ( max_turn_id + num_elem_by_elem_turns ) - ( index_t )1;

            SIXTRL_ASSERT( max_elem_by_elem_turn_id >= min_turn_id );

            success = NS(ElemByElemConfig_prepare_output_buffer_detailed)(
                output_buffer, min_particle_id, max_particle_id,
                min_element_id, max_element_id, min_turn_id,
                max_elem_by_elem_turn_id, ptr_index_offset );
        }
    }

    return success;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

int NS(ElemByElemConfig_prepare_output_buffer_for_particle_set)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT belements,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT output_buffer,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT pb,
    NS(buffer_size_t) const num_particle_sets,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t) const* particle_set_indices_begin,
    NS(buffer_size_t) const target_num_elem_by_elem_turns,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_index_offset )
{
    int success = -1;

    typedef NS(particle_index_t) index_t;
    typedef NS(buffer_size_t)    buf_size_t;

    index_t min_particle_id = ( index_t )0;
    index_t max_particle_id = ( index_t )-1;

    index_t min_element_id  = ( index_t )0;
    index_t max_element_id  = ( index_t )-1;

    index_t min_turn_id     = ( index_t )0;
    index_t max_turn_id     = ( index_t )-1;

    SIXTRL_ASSERT( pb            != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( belements     != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( output_buffer != SIXTRL_NULLPTR );

    SIXTRL_ASSERT( !NS(Buffer_needs_remapping)( belements ) );
    SIXTRL_ASSERT( !NS(Buffer_needs_remapping)( output_buffer ) );
    SIXTRL_ASSERT( !NS(Buffer_needs_remapping)( pb ) );

    if( ( target_num_elem_by_elem_turns > ( buf_size_t )0u ) &&
        ( 0 == NS(Particles_buffer_get_min_max_attributes_of_particles_set)(
            pb, num_particle_sets, particle_set_indices_begin,
            &min_particle_id, &max_particle_id, &min_element_id,
            &max_element_id, &min_turn_id, &max_turn_id ) ) )
    {
        index_t temp_min_element_id = ( index_t )0;
        index_t temp_max_element_id = ( index_t )-1;
        buf_size_t num_elem_by_elem_objects = ( buf_size_t )0u;

        success = NS(ElemByElemConfig_get_min_max_element_id_from_buffer)(
            belements, &temp_min_element_id, &temp_max_element_id,
                &num_elem_by_elem_objects, ( index_t )0u );

        SIXTRL_ASSERT( min_turn_id     >= ( index_t )0u );
        SIXTRL_ASSERT( min_turn_id     <= max_turn_id );

        SIXTRL_ASSERT( min_element_id  >= ( index_t )0u );
        SIXTRL_ASSERT( min_element_id  <= max_element_id );

        SIXTRL_ASSERT( min_particle_id >= ( index_t )0u );
        SIXTRL_ASSERT( max_particle_id <= ( index_t )0u );

        if( ( success == 0 ) &&
            ( num_elem_by_elem_objects > ( buf_size_t )0u ) )
        {
            SIXTRL_ASSERT( temp_min_element_id >= ( index_t )0u );
            SIXTRL_ASSERT( temp_min_element_id <= temp_max_element_id );

            if( min_element_id > temp_min_element_id )
            {
                min_element_id = temp_min_element_id;
            }

            if( max_element_id < temp_max_element_id )
            {
                max_element_id = temp_max_element_id;
            }
        }

        if( success == 0 )
        {
            index_t max_elem_by_elem_turn_id =
                min_turn_id + target_num_elem_by_elem_turns;

            if( max_elem_by_elem_turn_id < max_turn_id )
            {
                max_elem_by_elem_turn_id = max_turn_id;
            }

            success = NS(ElemByElemConfig_prepare_output_buffer_detailed)(
                output_buffer, min_particle_id, max_particle_id, min_element_id,
                max_element_id, min_turn_id, max_elem_by_elem_turn_id,
                ptr_index_offset );
        }
    }

    return success;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

int NS(ElemByElemConfig_prepare_output_buffer_detailed)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)*SIXTRL_RESTRICT out_buffer,
    NS(particle_index_t) const min_particle_id,
    NS(particle_index_t) const max_particle_id,
    NS(particle_index_t) const min_element_id,
    NS(particle_index_t) const max_element_id,
    NS(particle_index_t) const min_turn_id,
    NS(particle_index_t) const max_elem_by_elem_turn_id,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_index_offset )
{
    int success = -1;

    typedef NS(buffer_size_t)    buf_size_t;
    typedef NS(particle_index_t) index_t;

    SIXTRL_STATIC_VAR index_t const ZERO = ( index_t )0u;
    SIXTRL_STATIC_VAR index_t const ONE  = ( index_t )1u;

    buf_size_t const initial_out_buffer_index_offset =
        NS(Buffer_get_num_of_objects)( out_buffer );

    SIXTRL_ASSERT( out_buffer != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( !NS(Buffer_needs_remapping)( out_buffer ) );

    if( ( min_particle_id >= ZERO ) &&
        ( min_particle_id <= max_particle_id ) &&
        ( min_element_id  >= ZERO ) && ( min_element_id <= max_element_id ) &&
        ( min_turn_id >= ZERO ) &&
        ( min_turn_id <= max_elem_by_elem_turn_id ) )
    {
        buf_size_t num_objects  = NS(Buffer_get_num_of_objects)( out_buffer );
        buf_size_t num_slots    = NS(Buffer_get_num_of_slots)( out_buffer );
        buf_size_t num_dataptrs = NS(Buffer_get_num_of_dataptrs)( out_buffer );

        buf_size_t const num_garbage =
            NS(Buffer_get_num_of_garbage_ranges)( out_buffer );

        buf_size_t const num_particles_to_store = ( buf_size_t )(
            ONE + max_particle_id - min_particle_id );

        buf_size_t const num_elements_to_store = ( buf_size_t )(
            ONE + max_element_id - min_element_id );

        buf_size_t const num_turns_to_store = ( buf_size_t )(
            ONE + max_elem_by_elem_turn_id - min_turn_id );

        buf_size_t const particles_store_size = num_particles_to_store *
            num_elements_to_store * num_turns_to_store;

        ++num_objects;

        num_slots += NS(Particles_get_required_num_slots)(
            out_buffer, particles_store_size );

        num_dataptrs += NS(Particles_get_required_num_dataptrs)(
            out_buffer, particles_store_size );

        success = NS(Buffer_reserve)( out_buffer, num_objects, num_slots,
                                     num_dataptrs, num_garbage );

        if( success == 0 )
        {
            success = ( SIXTRL_NULLPTR != NS(Particles_new)(
                out_buffer, particles_store_size ) ) ? 0 : -1;
        }

        if( ( ptr_index_offset != SIXTRL_NULLPTR ) && ( success == 0 ) )
        {
            SIXTRL_ASSERT( NS(Buffer_get_num_of_objects)( out_buffer ) ==
                ( initial_out_buffer_index_offset + ( buf_size_t )1u ) );

            *ptr_index_offset = initial_out_buffer_index_offset;
        }
    }

    return success;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

int NS(ElemByElemConfig_prepare_output_buffer_from_conf)(
    SIXTRL_BE_ARGPTR_DEC struct NS(ElemByElemConfig)* SIXTRL_RESTRICT config,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)*SIXTRL_RESTRICT output_buffer,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_index_offset )
{
    int success = -1;

    SIXTRL_ASSERT( output_buffer != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( !NS(Buffer_needs_remapping)( output_buffer ) );

    if( config != SIXTRL_NULLPTR )
    {
        success = NS(ElemByElemConfig_prepare_output_buffer_detailed)(
            output_buffer,
            NS(ElemByElemConfig_get_min_particle_id)( config ),
            NS(ElemByElemConfig_get_max_particle_id)( config ),
            NS(ElemByElemConfig_get_min_element_id)( config ),
            NS(ElemByElemConfig_get_max_element_id)( config ),
            NS(ElemByElemConfig_get_min_turn)( config ),
            NS(ElemByElemConfig_get_max_turn)( config ), ptr_index_offset );
    }

    return success;
}

/* ------------------------------------------------------------------------- */
/* NS(BeamMonitor) based prepare output functions: */
/* ------------------------------------------------------------------------- */

int NS(BeamMonitor_prepare_output_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT belements_buffer,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT output_buffer,
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT p,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_index_offset )
{
    int success = -1;

    typedef NS(particle_index_t) index_t;
    typedef NS(buffer_size_t)    buf_size_t;

    index_t min_particle_id = ( index_t )0;
    index_t max_particle_id = ( index_t )-1;

    index_t min_element_id  = ( index_t )0;
    index_t max_element_id  = ( index_t )-1;

    index_t min_turn_id     = ( index_t )0;
    index_t max_turn_id     = ( index_t )-1;

    buf_size_t const num_beam_elements =
        NS(Buffer_get_num_of_objects)( belements_buffer );

    SIXTRL_ASSERT( belements_buffer != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( !NS(Buffer_needs_remapping)( belements_buffer ) );

    SIXTRL_ASSERT( output_buffer != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( !NS(Buffer_needs_remapping)( output_buffer ) );

    success = NS(Particles_get_min_max_attributes)( p, &min_particle_id,
        &max_particle_id, &min_element_id, &max_element_id,
            &min_turn_id, &max_turn_id );

    if( ( success == 0 ) && ( min_element_id >= ( index_t )0u ) &&
        ( min_element_id <= max_element_id ) && ( num_beam_elements >=
            ( buf_size_t )( max_element_id - min_element_id ) ) )
    {
        success = NS(BeamMonitor_prepare_output_buffer_detailed)(
            belements_buffer, output_buffer, min_particle_id, max_particle_id,
            min_turn_id, ptr_index_offset );
    }

    return success;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

int NS(BeamMonitor_prepare_output_buffer_for_particle_set)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT belements_buffer,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT output_buffer,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT pb,
    NS(buffer_size_t) const num_particle_sets,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)
        const* SIXTRL_RESTRICT particle_set_indices_begin,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_index_offset )
{
    int success = -1;

    typedef NS(particle_index_t) index_t;
    typedef NS(buffer_size_t)    buf_size_t;

    index_t min_particle_id = ( index_t )0;
    index_t max_particle_id = ( index_t )-1;

    index_t min_element_id  = ( index_t )0;
    index_t max_element_id  = ( index_t )-1;

    index_t min_turn_id     = ( index_t )0;
    index_t max_turn_id     = ( index_t )-1;

    buf_size_t const num_beam_elements =
        NS(Buffer_get_num_of_objects)( belements_buffer );

    SIXTRL_ASSERT( pb != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( !NS(Buffer_needs_remapping)( pb ) );

    SIXTRL_ASSERT( belements_buffer != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( !NS(Buffer_needs_remapping)( belements_buffer ) );

    SIXTRL_ASSERT( output_buffer != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( !NS(Buffer_needs_remapping)( output_buffer ) );

    success = NS(Particles_buffer_get_min_max_attributes_of_particles_set)( pb,
        num_particle_sets, particle_set_indices_begin,
        &min_particle_id, &max_particle_id, &min_element_id, &max_element_id,
        &min_turn_id, &max_turn_id );

    if( ( success == 0 ) && ( min_element_id >= ( index_t )0u ) &&
        ( min_element_id <= max_element_id ) && ( num_beam_elements >=
            ( buf_size_t )( max_element_id - min_element_id ) ) )
    {
        success = NS(BeamMonitor_prepare_output_buffer_detailed)(
            belements_buffer, output_buffer, min_particle_id, max_particle_id,
            min_turn_id, ptr_index_offset );
    }

    return success;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

int NS(BeamMonitor_prepare_output_buffer_detailed)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT belements_buffer,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT output_buffer,
    NS(particle_index_t)  const min_particle_id,
    NS(particle_index_t)  const max_particle_id,
    NS(particle_index_t)  const min_turn_id,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_index_offset )
{
    int success = -1;

    typedef NS(buffer_size_t)                         buf_size_t;
    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)*  ptr_obj_t;
    typedef SIXTRL_BE_ARGPTR_DEC NS(BeamMonitor)*     ptr_beam_monitor_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC NS(Particles)*  ptr_particles_t;
    typedef NS(buffer_addr_t)                         address_t;
    typedef NS(be_monitor_turn_t)                     nturn_t;
    typedef NS(particle_index_t)                      index_t;
    typedef NS(object_type_id_t)                      type_id_t;

    SIXTRL_STATIC_VAR buf_size_t const ZERO      = ( buf_size_t )0u;
    SIXTRL_STATIC_VAR nturn_t    const TZERO     = ( nturn_t )0u;
    SIXTRL_STATIC_VAR index_t    const IZERO     = ( index_t )0u;
    SIXTRL_STATIC_VAR address_t  const ADDR_ZERO = ( address_t )0u;

    ptr_obj_t be_begin = NS(Buffer_get_objects_begin)( belements_buffer );
    ptr_obj_t be_end   = NS(Buffer_get_objects_end)( belements_buffer );

    buf_size_t const out_buffer_index_offset =
        NS(Buffer_get_num_of_objects)( output_buffer );

    buf_size_t num_beam_monitors = ZERO;
    buf_size_t num_particles_per_turn = ZERO;

    nturn_t const first_turn_id = ( nturn_t )min_turn_id;

    ptr_obj_t be_it = be_begin;

    SIXTRL_ASSERT( belements_buffer != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( !NS(Buffer_needs_remapping)( belements_buffer ) );

    SIXTRL_ASSERT( output_buffer != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( !NS(Buffer_needs_remapping)( output_buffer ) );

    if( ( min_turn_id < IZERO ) || ( min_particle_id < IZERO ) ||
        ( max_particle_id < min_particle_id ) )
    {
        return success;
    }

    num_particles_per_turn = ( buf_size_t )(
        1u + max_particle_id - min_particle_id );

    SIXTRL_ASSERT( num_particles_per_turn > ZERO );
    success = 0;

    for( ; be_it != be_end ; ++be_it )
    {
        type_id_t const type_id = NS(Object_get_type_id)( be_it );

        if( ( type_id == NS(OBJECT_TYPE_BEAM_MONITOR) ) &&
            ( NS(Object_get_begin_addr)( be_it ) > ADDR_ZERO ) )
        {
            ptr_beam_monitor_t monitor = ( ptr_beam_monitor_t )( uintptr_t
                )NS(Object_get_begin_addr)( be_it );

            nturn_t const start = NS(BeamMonitor_get_start)( monitor );
            nturn_t const nstores = NS(BeamMonitor_get_num_stores)( monitor );

            if( ( start >= first_turn_id ) && ( nstores > TZERO ) )
            {
                buf_size_t const num_particles_to_store =
                    num_particles_per_turn * nstores;

                ptr_particles_t particles =
                    NS(Particles_new)( output_buffer, num_particles_to_store );

                if( particles != SIXTRL_NULLPTR )
                {
                    NS(BeamMonitor_set_min_particle_id)(
                        monitor, min_particle_id );

                    NS(BeamMonitor_set_max_particle_id)(
                        monitor, max_particle_id );

                    NS(BeamMonitor_set_out_address)(
                        monitor, ( address_t )0u );

                    ++num_beam_monitors;
                }
                else
                {
                    success = -1;
                    break;
                }
            }
        }
    }

    SIXTRL_ASSERT( ( success != 0 ) || ( NS(Buffer_get_num_of_objects)(
        output_buffer ) == ( num_beam_monitors + out_buffer_index_offset ) ) );

    if( ( ptr_index_offset != SIXTRL_NULLPTR ) && ( success == 0 ) )
    {
        *ptr_index_offset = out_buffer_index_offset;
    }

    return success;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

int NS(BeamMonitor_assign_output_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT belements_buffer,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT out_buffer,
    NS(particle_index_t) const min_turn_id,
    NS(buffer_size_t) const num_elem_by_elem_turns  )
{
    int success = -1;

    typedef NS(buffer_size_t) buf_size_t;

    SIXTRL_STATIC_VAR buf_size_t const ZERO = ( buf_size_t )0u;
    SIXTRL_STATIC_VAR buf_size_t const ONE  = ( buf_size_t )1u;

    if( ( num_elem_by_elem_turns > ZERO ) &&
        ( NS(Buffer_get_num_of_objects)( belements_buffer ) > ONE ) )
    {
        success = NS(BeamMonitor_assign_output_buffer_from_offset)(
            belements_buffer, out_buffer, min_turn_id, ONE );
    }
    else if( num_elem_by_elem_turns == ZERO )
    {
        success = NS(BeamMonitor_assign_output_buffer_from_offset)(
            belements_buffer, out_buffer, min_turn_id, ZERO );
    }

    return success;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

int NS(BeamMonitor_assign_output_buffer_from_offset)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT belements_buffer,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT out_buffer,
    NS(particle_index_t) const min_turn_id,
    NS(buffer_size_t) const out_buffer_index_offset )
{
    return NS(BeamMonitor_assign_managed_output_buffer)(
        NS(Buffer_get_data_begin)( belements_buffer ),
        NS(Buffer_get_data_begin)( out_buffer ),
        min_turn_id, out_buffer_index_offset,
        NS(Buffer_get_slot_size)( out_buffer ) );
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

int NS(BeamMonitor_setup_for_particles_all)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT belements_buffer,
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT p )
{
    return NS(BeamMonitor_setup_managed_buffer_for_particles_all)(
        NS(Buffer_get_data_begin)( belements_buffer ),
        p, NS(Buffer_get_slot_size)( belements_buffer ) );
}

/* end: sixtracklib/common/output/be_monitor.c */
