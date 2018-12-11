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
    #include "sixtracklib/common/particles.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

extern SIXTRL_HOST_FN int NS(BeamMonitor_setup_for_particles_all)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT beam_elements_buffer,
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT particles );

extern SIXTRL_HOST_FN int NS(BeamMonitor_prepare_particles_out_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT belements,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT out_buffer,
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT particles,
    NS(buffer_size_t) const num_elem_by_elem_turns );

int NS(BeamMonitor_prepare_particles_out_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT belements,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT out_buffer,
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT particles,
    NS(buffer_size_t) const num_elem_by_elem_turns )
{
    typedef NS(buffer_size_t)                        buf_size_t;
    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)* ptr_obj_t;
    typedef SIXTRL_BE_ARGPTR_DEC NS(BeamMonitor)*    ptr_beam_monitor_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC NS(Particles)* ptr_particles_t;
    typedef NS(be_monitor_turn_t)                    nturn_t;
    typedef NS(particle_index_t)                     index_t;

    SIXTRL_STATIC_VAR buf_size_t const ZERO       = ( buf_size_t )0u;
    SIXTRL_STATIC_VAR index_t    const ZERO_INDEX = ( index_t )0u;

    buf_size_t const num_beam_elements =
        NS(Buffer_get_num_of_objects)( belements );

    index_t max_particle_id = ZERO_INDEX;
    index_t min_particle_id = ZERO_INDEX;

    int success = NS(Particles_get_min_max_particle_id)(
        particles, &min_particle_id, &max_particle_id );

    if( ( success == 0 ) && ( min_particle_id > ZERO_INDEX ) )
    {
        min_particle_id = ZERO_INDEX;
    }

    if( ( out_buffer != SIXTRL_NULLPTR ) && ( belements != SIXTRL_NULLPTR ) &&
        ( success == 0 ) && ( max_particle_id >= min_particle_id ) &&
        ( min_particle_id >= ZERO_INDEX ) && ( num_beam_elements > ZERO ) )
    {
        buf_size_t const stored_num_particles =
            NS(Particles_get_num_of_particles)( particles );

        buf_size_t const requ_num_particles =
            ( ( buf_size_t )( max_particle_id - min_particle_id ) +
                ( buf_size_t )1u );

        buf_size_t const num_particles =
            ( stored_num_particles < requ_num_particles )
                ? requ_num_particles : stored_num_particles;

        buf_size_t total_num_elem_by_elem_objects = ZERO;

        buf_size_t num_slots               = ZERO;
        buf_size_t num_objects             = ZERO;
        buf_size_t num_dataptrs            = ZERO;

        buf_size_t num_elem_by_elem_blocks = ZERO;
        buf_size_t num_beam_monitors       = ZERO;

        ptr_obj_t it = NS(Buffer_get_objects_begin)( belements );
        ptr_obj_t be_end = NS(Buffer_get_objects_end)( belements );
        ptr_obj_t last_beam_monitor = be_end;

        SIXTRL_ASSERT( !NS(Buffer_needs_remapping)( out_buffer ) );
        SIXTRL_ASSERT( !NS(Buffer_needs_remapping)( belements ) );

        for( ; it != be_end ; ++it )
        {
            NS(object_type_id_t) const type_id = NS(Object_get_type_id)( it );

            if( type_id == NS(OBJECT_TYPE_BEAM_MONITOR) )
            {
                ptr_beam_monitor_t monitor =
                    ( ptr_beam_monitor_t )NS(Object_get_const_begin_ptr)( it );

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

                    last_beam_monitor = it;
                }

                ++num_elem_by_elem_blocks;
            }
            else if( ( type_id != NS(OBJECT_TYPE_PARTICLE) ) &&
                     ( type_id != NS(OBJECT_TYPE_NONE)     ) &&
                     ( type_id != NS(OBJECT_TYPE_LINE)     ) &&
                     ( type_id != NS(OBJECT_TYPE_INVALID)  ) )
            {
                ++num_elem_by_elem_blocks;
            }
        }

        if( ( num_elem_by_elem_turns > ZERO ) && ( num_elem_by_elem_blocks > ZERO ) )
        {
            buf_size_t num_particles_to_store =
                num_elem_by_elem_blocks * num_elem_by_elem_turns;

            total_num_elem_by_elem_objects = num_particles_to_store;
            num_particles_to_store *= num_particles;

            ++num_objects;

            num_slots += NS(Particles_get_required_num_slots)(
                    out_buffer, num_particles_to_store );

            num_dataptrs += NS(Particles_get_required_num_dataptrs)(
                    out_buffer, num_particles_to_store );
        }

        if( ( 0 == NS(Buffer_reset)( out_buffer ) ) &&
            ( 0 == NS(Buffer_reserve)( out_buffer,
                num_objects, num_slots, num_dataptrs, ZERO ) ) )
        {
            SIXTRL_ASSERT( NS(Buffer_get_num_of_objects)( out_buffer ) == ZERO );
            SIXTRL_ASSERT( NS(Buffer_get_num_of_slots)( out_buffer ) == ZERO );
            SIXTRL_ASSERT( NS(Buffer_get_num_of_dataptrs )( out_buffer ) == ZERO );

            success = 0;

            if( total_num_elem_by_elem_objects > ZERO )
            {
                ptr_particles_t elem_by_elem_particles = NS(Particles_new)(
                    out_buffer, num_particles * total_num_elem_by_elem_objects );

                if( elem_by_elem_particles == SIXTRL_NULLPTR )
                {
                    success = -1;
                }
            }

            if( ( success == 0 ) && ( num_beam_elements > ZERO ) &&
                ( last_beam_monitor != be_end ) )
            {
                buf_size_t num_beam_monitors_prepared = ZERO;

                it = NS(Buffer_get_objects_begin)( belements );
                ++last_beam_monitor;

                for( ; it != last_beam_monitor ; ++it )
                {
                    NS(object_type_id_t) const type_id =
                        NS(Object_get_type_id)( it );

                    if( type_id == NS(OBJECT_TYPE_BEAM_MONITOR) )
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
            }
        }

        SIXTRL_ASSERT( ( success != 0 ) ||
            ( NS(Buffer_get_num_of_objects)( out_buffer ) == num_objects ) );

        if( success != 0 )
        {
            NS(Buffer_reset)( out_buffer );
        }
    }

    return success;
}

int NS(BeamMonitor_setup_for_particles_all)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT beam_elements_buffer,
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT particles )
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
            if( NS(Object_get_type_id)( be_it ) == NS(OBJECT_TYPE_BEAM_MONITOR) )
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

/* end: sixtracklib/common/output/be_monitor.c */
