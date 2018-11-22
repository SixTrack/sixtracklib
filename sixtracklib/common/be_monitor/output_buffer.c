#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/be_monitor/output_buffer.h"
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

extern SIXTRL_HOST_FN int NS(BeamMonitor_prepare_particles_out_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT belements,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT out_buffer,
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT particles,
    NS(buffer_size_t) const num_elem_by_elem_turns );

int NS(BeamMonitor_prepare_particles_out_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT belements,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT out_buffer,
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT particles,
    NS(buffer_size_t) const num_elem_by_elem_turns )
{
    typedef NS(buffer_size_t) buf_size_t;
    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const*  ptr_obj_t;
    typedef SIXTRL_BE_ARGPTR_DEC NS(BeamMonitor) const*     ptr_beam_monitor_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC NS(Particles)*        ptr_particles_t;
    typedef NS(be_monitor_turn_t)                           nturn_t;
    typedef NS(particle_index_t)                            index_t;

    buf_size_t const num_beam_elements =
        NS(Buffer_get_num_of_objects)( belements );

    index_t max_particle_id = ( index_t )0u;
    index_t min_particle_id = ( index_t )0u;

    int success = NS(Particles_get_min_max_particle_id)(
        particles, &min_particle_id, &max_particle_id );

    SIXTRL_STATIC_VAR buf_size_t const ZERO = ( buf_size_t )0u;

    if( ( out_buffer != SIXTRL_NULLPTR ) && ( belements != SIXTRL_NULLPTR ) &&
        ( success == 0 ) && ( max_particle_id >= min_particle_id ) &&
        ( min_particle_id >= ( index_t )0u ) &&
        ( num_beam_elements > ZERO ) )
    {
        buf_size_t const stored_num_particles =
            NS(Particles_get_num_of_particles)( particles );

        buf_size_t const requ_num_particles =
            ( buf_size_t )max_particle_id + ( buf_size_t )1u;

        buf_size_t const num_particles =
            ( stored_num_particles < requ_num_particles )
                ? requ_num_particles : stored_num_particles;

        buf_size_t const req_num_slots_per_obj =
            NS(Particles_get_required_num_slots)( out_buffer, num_particles );

        buf_size_t const req_num_dataptrs_per_obj =
            NS(Particles_get_required_num_dataptrs)( out_buffer, num_particles );

        buf_size_t num_slots               = ZERO;
        buf_size_t num_objects             = ZERO;
        buf_size_t num_dataptrs            = ZERO;

        buf_size_t num_elem_by_elem_blocks = ZERO;
        buf_size_t num_beam_monitors       = ZERO;

        ptr_obj_t it = NS(Buffer_get_const_objects_begin)( belements );
        ptr_obj_t be_end = NS(Buffer_get_const_objects_end)( belements );

        SIXTRL_ASSERT( !NS(Buffer_needs_remapping)( out_buffer ) );
        SIXTRL_ASSERT( !NS(Buffer_needs_remapping)( belements ) );

        SIXTRL_ASSERT( req_num_slots_per_obj    > ZERO );
        SIXTRL_ASSERT( req_num_dataptrs_per_obj > ZERO );

        for( ; it != be_end ; ++it )
        {
            NS(object_type_id_t) const type_id = NS(Object_get_type_id)( it );

            if( type_id == NS(OBJECT_TYPE_BEAM_MONITOR) )
            {
                ptr_beam_monitor_t monitor =
                    ( ptr_beam_monitor_t )NS(Object_get_const_begin_ptr)( it );

                nturn_t const num_stores =
                    NS(BeamMonitor_get_num_stores)( monitor );

                SIXTRL_ASSERT( monitor != SIXTRL_NULLPTR );

                num_objects  += num_stores;
                num_slots    += num_stores * req_num_slots_per_obj;
                num_dataptrs += num_stores * req_num_dataptrs_per_obj;

                ++num_beam_monitors;
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

        if( num_elem_by_elem_turns > ( buf_size_t )0u )
        {
            buf_size_t const elem_by_elem_objects =
                num_elem_by_elem_turns * num_elem_by_elem_blocks;

            num_objects  += elem_by_elem_objects;
            num_slots    += elem_by_elem_objects * req_num_slots_per_obj;
            num_dataptrs += elem_by_elem_objects * req_num_dataptrs_per_obj;
        }

        if( ( 0 == NS(Buffer_reset)( out_buffer ) ) &&
            ( 0 == NS(Buffer_reserve)( out_buffer,
                num_objects, num_slots, num_dataptrs, ZERO ) ) )
        {
            buf_size_t ii = ZERO;

            SIXTRL_ASSERT( NS(Buffer_get_num_of_objects)( out_buffer ) == ZERO );
            SIXTRL_ASSERT( NS(Buffer_get_num_of_slots)(   out_buffer ) == ZERO );
            SIXTRL_ASSERT( NS(Buffer_get_num_of_dataptrs )( out_buffer ) == ZERO );

            success = 0;

            for( ; ii < num_objects ; ++ii )
            {
                ptr_particles_t particles =
                    NS(Particles_new)( out_buffer, num_particles );

                if( particles == SIXTRL_NULLPTR )
                {
                    success = -1;
                    break;
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

/* end: sixtracklib/common/be_monitor/be_monitor.c */
