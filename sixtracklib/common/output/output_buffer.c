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

int NS(BeamMonitor_prepare_particles_out_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT belements_buffer,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT out_buffer,
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT p,
    NS(buffer_size_t) const num_elem_by_elem_turns )
{
    typedef NS(buffer_size_t)                        buf_size_t;
    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)* ptr_obj_t;
    typedef SIXTRL_BE_ARGPTR_DEC NS(BeamMonitor)*    ptr_beam_monitor_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC NS(Particles)* ptr_particles_t;
    typedef NS(be_monitor_turn_t)                    nturn_t;
    typedef NS(particle_index_t)                     index_t;
    typedef NS(particle_num_elements_t)              num_elem_t;
    typedef NS(object_type_id_t)                     type_id_t;

    int success = -1;

    SIXTRL_STATIC_VAR buf_size_t const ZERO       = ( buf_size_t )0u;
    SIXTRL_STATIC_VAR index_t    const ZERO_INDEX = ( index_t )0u;

    buf_size_t const num_beam_elements =
        NS(Buffer_get_num_of_objects)( belements_buffer );

    buf_size_t out_buffer_index = NS(Buffer_get_num_of_objects)( out_buffer );

    SIXTRL_ASSERT( !NS(Buffer_needs_remapping)( out_buffer ) );
    SIXTRL_ASSERT( !NS(Buffer_needs_remapping)( belements_buffer ) );

    if( num_elem_by_elem_turns > ZERO )
    {
        success = NS(ElemByElemConfig_prepare_particles_out_buffer)(
            belements_buffer, out_buffer, p, num_elem_by_elem_turns );

        if( success == 0 )
        {
            out_buffer_index = NS(Buffer_get_num_of_objects)( out_buffer );
        }
    }
    else
    {
        success = 0;
    }

    ( void )out_buffer_index;

    if( ( success == 0 ) && ( p != SIXTRL_NULLPTR ) &&
        ( NS(Particles_get_num_of_particles)( p ) > ( num_elem_t )0 ) &&
        ( belements_buffer != SIXTRL_NULLPTR ) &&
        ( num_beam_elements > ZERO ) && ( out_buffer != SIXTRL_NULLPTR ) )
    {
        index_t max_particle_id = ZERO_INDEX;
        index_t min_particle_id = ZERO_INDEX;

        success = NS(Particles_get_min_max_particle_id)(
            p, &min_particle_id, &max_particle_id );

        if( ( success == 0 ) && ( min_particle_id > ZERO_INDEX ) )
        {
            min_particle_id = ZERO_INDEX;
        }

        if( ( success == 0 ) &&
            ( min_particle_id >= ZERO_INDEX ) &&
            ( min_particle_id <= max_particle_id ) )
        {
            buf_size_t const stored_num_particles =
                NS(Particles_get_num_of_particles)( p );

            buf_size_t const requ_num_particles =
                ( ( buf_size_t )( max_particle_id - min_particle_id ) +
                    ( buf_size_t )1u );

            buf_size_t const num_particles =
                ( stored_num_particles < requ_num_particles )
                    ? requ_num_particles : stored_num_particles;

            buf_size_t num_slots =
                NS(Buffer_get_num_of_slots)( out_buffer );

            buf_size_t num_objects =
                NS(Buffer_get_num_of_objects)( out_buffer );

            buf_size_t num_dataptrs =
                NS(Buffer_get_num_of_dataptrs)( out_buffer );

            buf_size_t num_garbage  =
                NS(Buffer_get_num_of_garbage_ranges)( out_buffer );

            buf_size_t num_beam_monitors = ZERO;

            ptr_obj_t it = NS(Buffer_get_objects_begin)( belements_buffer );
            ptr_obj_t be_end = NS(Buffer_get_objects_end)( belements_buffer );
            ptr_obj_t first_beam_monitor = SIXTRL_NULLPTR;
            ptr_obj_t last_beam_monitor  = be_end;

            for( ; it != be_end ; ++it )
            {
                type_id_t const type_id = NS(Object_get_type_id)( it );

                if( type_id == NS(OBJECT_TYPE_BEAM_MONITOR) )
                {
                    ptr_beam_monitor_t monitor = ( ptr_beam_monitor_t
                        )( uintptr_t )NS(Object_get_begin_addr)( it );

                    nturn_t const num_stores = ( monitor != SIXTRL_NULLPTR )
                        ? NS(BeamMonitor_get_num_stores)( monitor )
                        : ( nturn_t )0u;

                    buf_size_t const num_particles_to_store = ( num_stores > 0 )
                        ? num_particles * ( ( buf_size_t )num_stores ) : ZERO;

                    if( num_particles_to_store > ZERO )
                    {
                        SIXTRL_ASSERT( monitor != SIXTRL_NULLPTR );

                        ++num_objects;
                        ++num_beam_monitors;

                        num_slots    += NS(Particles_get_required_num_slots)(
                            out_buffer, num_particles_to_store );

                        num_dataptrs += NS(Particles_get_required_num_dataptrs)(
                            out_buffer, num_particles_to_store );

                        if( first_beam_monitor == SIXTRL_NULLPTR )
                        {
                            first_beam_monitor = it;
                        }

                        last_beam_monitor = it;
                    }
                }
            }

            if( ( success == 0 ) &&
                ( 0 == NS(Buffer_reserve)( out_buffer, num_objects, num_slots,
                                           num_dataptrs, num_garbage ) ) &&
                ( num_beam_monitors > ZERO ) )
            {
                buf_size_t num_beam_monitors_prepared = ZERO;

                SIXTRL_ASSERT( num_beam_elements >= num_beam_monitors );
                SIXTRL_ASSERT( first_beam_monitor != SIXTRL_NULLPTR );
                SIXTRL_ASSERT( last_beam_monitor  != be_end );

                it = first_beam_monitor;
                ++last_beam_monitor;

                for( ; it != last_beam_monitor ; ++it )
                {
                    if( NS(Object_get_type_id)( it ) ==
                        NS(OBJECT_TYPE_BEAM_MONITOR) )
                    {
                        ptr_beam_monitor_t monitor = ( ptr_beam_monitor_t
                            )NS(Object_get_const_begin_ptr)( it );

                        nturn_t const nn = ( monitor != SIXTRL_NULLPTR )
                            ? NS(BeamMonitor_get_num_stores)( monitor )
                            : ( nturn_t )0u;

                        buf_size_t const num_particles_to_store = ( nn > 0 )
                            ? num_particles * ( ( buf_size_t )nn ) : ZERO;

                        if( num_particles_to_store > ZERO )
                        {
                            ptr_particles_t particles = NS(Particles_new)(
                                out_buffer, num_particles_to_store );

                            if( particles != SIXTRL_NULLPTR )
                            {
                                 NS(BeamMonitor_set_min_particle_id)(
                                    monitor, min_particle_id );

                                NS(BeamMonitor_set_max_particle_id)(
                                    monitor, max_particle_id );

                                ++num_beam_monitors_prepared;
                            }
                            else
                            {
                                success = -1;
                                break;
                            }
                        }
                    }
                }

                if( ( success == 0 ) &&
                    ( num_beam_monitors_prepared != num_beam_monitors ) )
                {
                    success = -1;
                }

                SIXTRL_ASSERT( ( success != 0 ) ||
                    ( NS(Buffer_get_num_of_objects)( out_buffer )
                        == num_objects ) );
            }
        }

        if( success != 0 )
        {
            NS(Buffer_reset)( out_buffer );
        }
    }

    return success;
}

int NS(BeamMonitor_setup_for_particles_all)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT beam_elements_buffer,
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(Particles)
        *const SIXTRL_RESTRICT particles )
{
    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)* ptr_obj_t;
    typedef SIXTRL_BE_ARGPTR_DEC NS(BeamMonitor)*    ptr_beam_monitor_t;
    typedef NS(particle_index_t)                     index_t;
    typedef NS(be_monitor_addr_t)                    addr_t;

    index_t max_particle_id = ( index_t )0u;
    index_t min_particle_id = ( index_t )0u;

    int success = -1;

    ptr_obj_t be_it  = NS(Buffer_get_objects_begin)( beam_elements_buffer );
    ptr_obj_t be_end = NS(Buffer_get_objects_end)( beam_elements_buffer );

    if( ( 0 == NS(Particles_get_min_max_particle_id)(
            particles, &min_particle_id, &max_particle_id ) ) &&
        ( max_particle_id >= min_particle_id ) &&
        ( min_particle_id >= ( index_t )0u ) &&
        ( be_it  != SIXTRL_NULLPTR ) && ( be_end != SIXTRL_NULLPTR ) )
    {
        SIXTRL_ASSERT( !NS(Buffer_needs_remapping)( beam_elements_buffer ) );
        SIXTRL_ASSERT( ( uintptr_t )be_it <= ( uintptr_t )be_end );

        min_particle_id = ( index_t )0u;

        for( ; be_it != be_end ; ++be_it )
        {
            if( NS(Object_get_type_id)( be_it ) ==
                NS(OBJECT_TYPE_BEAM_MONITOR) )
            {
                ptr_beam_monitor_t monitor = ( ptr_beam_monitor_t
                    )NS(Object_get_begin_ptr)( be_it );

                NS(BeamMonitor_set_min_particle_id)( monitor, min_particle_id );
                NS(BeamMonitor_set_max_particle_id)( monitor, max_particle_id );
                NS(BeamMonitor_set_out_address)( monitor, ( addr_t )0 );
            }
        }

        success = 0;
    }

    return success;
}


/* ------------------------------------------------------------------------- */
/* Element - by - Element Output Buffer: */

SIXTRL_HOST_FN int NS(ElemByElemConfig_prepare_particles_out_buffer_detailed)(
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC
        NS(ElemByElemConfig)* SIXTRL_RESTRICT config,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT out_buffer,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT
        ptr_out_buffer_index_offset )
{
    typedef NS(buffer_size_t) buf_size_t;
    typedef SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* ptr_particles_t;

    int success = -1;
    SIXTRL_STATIC_VAR buf_size_t const ZERO = ( buf_size_t )0u;

    buf_size_t const out_store_num_particles =
        NS(ElemByElemConfig_get_out_store_num_particles)( config );

    if( ( config != SIXTRL_NULLPTR ) && ( out_buffer != SIXTRL_NULLPTR ) &&
        ( out_store_num_particles > ZERO ) )
    {
        buf_size_t const index = NS(Buffer_get_num_of_objects)( out_buffer );
        ptr_particles_t particles = NS(Particles_new)(
            out_buffer, out_store_num_particles );

        if( particles != SIXTRL_NULLPTR );
        {
            SIXTRL_ASSERT( NS(Particles_get_num_of_particles)( particles ) ==
                out_store_num_particles );

            SIXTRL_ASSERT( NS(Buffer_get_num_of_objects)(
                out_buffer ) > index );

            success = 0;
        }

        if( ( success == 0 ) &&
            ( ptr_out_buffer_index_offset != SIXTRL_NULLPTR ) )
        {
            *ptr_out_buffer_index_offset = index;
        }
    }

    return success;
}


SIXTRL_HOST_FN int NS(ElemByElemConfig_prepare_particles_out_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT belements,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT out_buffer,
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(Particles) *const
        SIXTRL_RESTRICT particles,
    NS(buffer_size_t) const num_elem_by_elem_turns )
{
    typedef NS(buffer_size_t)  buf_size_t;
    typedef NS(particle_index_t) index_t;

    int success = -1;

    SIXTRL_STATIC_VAR buf_size_t const ZERO = ( buf_size_t )0u;

    buf_size_t const num_beam_elements =
        NS(Buffer_get_num_of_objects)( belements );

    SIXTRL_ASSERT( ( out_buffer == SIXTRL_NULLPTR ) ||
                   ( !NS(Buffer_needs_remapping)( out_buffer ) ) );

    SIXTRL_ASSERT( ( belements == SIXTRL_NULLPTR ) ||
                   ( !NS(Buffer_needs_remapping)( belements ) ) );

    if( ( num_elem_by_elem_turns > ZERO ) && ( num_beam_elements > ZERO ) &&
        ( particles != SIXTRL_NULLPTR ) && ( out_buffer != SIXTRL_NULLPTR ) )
    {
        SIXTRL_STATIC_VAR index_t const ZERO_INDEX = ( index_t )0u;
        SIXTRL_STATIC_VAR index_t const ONE_INDEX  = ( index_t )1u;

        buf_size_t num_slots    = NS(Buffer_get_num_of_slots)( out_buffer );
        buf_size_t num_objects  = NS(Buffer_get_num_of_objects)( out_buffer );
        buf_size_t num_dataptrs = NS(Buffer_get_num_of_dataptrs)( out_buffer );

        buf_size_t num_elem_by_elem_stored_particles = ZERO;

        buf_size_t num_garbage  =
            NS(Buffer_get_num_of_garbage_ranges)( out_buffer );

        index_t max_particle_id = ZERO_INDEX;
        index_t min_particle_id = ZERO_INDEX;

        index_t min_element_id  = ZERO_INDEX;
        index_t max_element_id  = ZERO_INDEX;

        index_t min_turn_id     = ZERO_INDEX;
        index_t max_turn_id     = ZERO_INDEX;

        success = NS(Particles_get_min_max_particle_id)(
            particles, &min_particle_id, &max_particle_id );

        success |= NS(Particles_get_min_max_at_element_id_value)(
            particles, &min_element_id, &max_element_id );

        success |= NS(Particles_get_min_max_at_turn_value)(
            particles, &min_turn_id, &max_turn_id );

        if( success == 0 )
        {
            index_t const end_turn_id = min_turn_id + num_elem_by_elem_turns;
            index_t const end_element_id = min_element_id + num_beam_elements;

            if( min_particle_id > ZERO_INDEX )
            {
                min_particle_id = ZERO_INDEX;
            }

            if( ( max_turn_id + ONE_INDEX ) < end_turn_id )
            {
                max_turn_id = end_turn_id - ONE_INDEX;
            }

            if( ( max_element_id + ONE_INDEX ) < end_element_id )
            {
                max_element_id = end_element_id - ONE_INDEX;
            }

            SIXTRL_ASSERT( max_element_id >= min_element_id );
            SIXTRL_ASSERT( max_turn_id    >= min_turn_id );
        }

        if( ( success == 0 ) && ( min_particle_id <= max_particle_id ) )
        {
            buf_size_t const num_particles_to_store = ( buf_size_t )(
                ONE_INDEX + max_particle_id - min_particle_id );

            buf_size_t const num_elements_to_store = ( buf_size_t )(
                ONE_INDEX + max_element_id - min_element_id );

            buf_size_t const num_turns_to_store = ( buf_size_t )(
                ONE_INDEX + max_turn_id - min_turn_id );

            num_elem_by_elem_stored_particles = num_particles_to_store *
                num_elements_to_store * num_turns_to_store;

            ++num_objects;

            num_slots += NS(Particles_get_required_num_slots)(
                out_buffer, num_elem_by_elem_stored_particles );

            num_dataptrs += NS(Particles_get_required_num_dataptrs)(
                out_buffer, num_elem_by_elem_stored_particles );
        }

        if( ( success == 0 ) && ( num_elem_by_elem_stored_particles > ZERO ) &&
            ( 0 == NS(Buffer_reserve)( out_buffer, num_objects, num_slots,
                                       num_dataptrs, num_garbage ) ) )
        {
            success = ( SIXTRL_NULLPTR != NS(Particles_new)( out_buffer,
                    num_elem_by_elem_stored_particles ) ) ? 0 : -1;
        }
        else
        {
            success |= -1;
        }
    }

    return success;
}

SIXTRL_HOST_FN int NS(ElemByElemConfig_assign_particles_out_buffer)(
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC
        NS(ElemByElemConfig)* SIXTRL_RESTRICT conf,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT output_buffer,
    NS(buffer_size_t) const out_buffer_index_offset )
{
    int success = -1;

    typedef NS(particle_num_elements_t)                     num_elem_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC NS(Object) const*     ptr_object_t;
    typedef SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)*       ptr_particles_t;
    typedef NS(buffer_addr_t)                               address_t;

    ptr_object_t ptr_object = NS(Buffer_get_const_object)(
        output_buffer, out_buffer_index_offset );

    address_t const out_addr = NS(Object_get_begin_addr)( ptr_object );
    NS(object_type_id_t) const type_id = NS(Object_get_type_id)( ptr_object );

    if( ( conf != SIXTRL_NULLPTR ) &&
        ( type_id == NS(OBJECT_TYPE_PARTICLE) ) &&
        ( ptr_object != SIXTRL_NULLPTR ) && ( out_addr != ( address_t )0u ) )
    {
        ptr_particles_t out_particles = ( ptr_particles_t )( uintptr_t
            )NS(Object_get_begin_addr)( ptr_object );

        num_elem_t const required_num_out_particles =
            NS(ElemByElemConfig_get_out_store_num_particles)( conf );

        num_elem_t const available_num_out_particles =
            NS(Particles_get_num_of_particles)( out_particles );

        if( ( required_num_out_particles >= ( num_elem_t )0u ) &&
            ( required_num_out_particles <= available_num_out_particles ) )
        {
            NS(ElemByElemConfig_set_output_store_address)( conf, out_addr );
            success = 0;
        }
    }

    return success;
}

/* end: sixtracklib/common/output/be_monitor.c */
