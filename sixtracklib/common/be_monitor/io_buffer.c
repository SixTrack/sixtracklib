#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/be_monitor/io_buffer.h"
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

extern SIXTRL_HOST_FN int NS(BeamMonitor_prepare_io_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT belements,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT io_buffer,
    NS(buffer_size_t) const num_particles, bool const enable_elem_by_elem_dump );


int NS(BeamMonitor_prepare_io_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT belements,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT io_buffer,
    NS(buffer_size_t) const num_particles, bool const enable_elem_by_elem_dump )
{
    typedef NS(buffer_size_t) buf_size_t;
    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const*  ptr_obj_t;
    typedef SIXTRL_BE_ARGPTR_DEC NS(BeamMonitor) const*     ptr_beam_monitor_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC NS(Particles)*        ptr_particles_t;
    typedef NS(be_monitor_turn_t)                           nturn_t;

    int success = -1;

    buf_size_t const num_beam_elements =
        NS(Buffer_get_num_of_objects)( belements );

    SIXTRL_STATIC_VAR buf_size_t const ZERO = ( buf_size_t )0u;

    if( ( io_buffer != SIXTRL_NULLPTR ) &&
        ( belements != SIXTRL_NULLPTR ) &&
        ( num_particles > ZERO ) && ( num_beam_elements > ZERO ) )
    {
        buf_size_t const req_num_slots_per_obj =
            NS(Particles_get_required_num_slots)( io_buffer, num_particles );

        buf_size_t const req_num_dataptrs_per_obj =
            NS(Particles_get_required_num_dataptrs)( io_buffer, num_particles );

        buf_size_t num_slots               = ZERO;
        buf_size_t num_objects             = ZERO;
        buf_size_t num_dataptrs            = ZERO;

        buf_size_t num_elem_by_elem_blocks = ZERO;
        buf_size_t num_beam_monitors       = ZERO;

        ptr_obj_t it = NS(Buffer_get_const_objects_begin)( belements );
        ptr_obj_t be_end = NS(Buffer_get_const_objects_end)( belements );

        SIXTRL_ASSERT( !NS(Buffer_needs_remapping)( io_buffer ) );
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

                if( num_stores > ZERO )
                {
                    num_objects  += num_stores;
                    num_slots    += num_stores * req_num_slots_per_obj;
                    num_dataptrs += num_stores * req_num_dataptrs_per_obj;

                    ++num_beam_monitors;
                }
            }
            else if( ( type_id != NS(OBJECT_TYPE_PARTICLE) ) &&
                     ( type_id != NS(OBJECT_TYPE_NONE)     ) &&
                     ( type_id != NS(OBJECT_TYPE_LINE)     ) &&
                     ( type_id != NS(OBJECT_TYPE_INVALID)  ) )
            {
                ++num_elem_by_elem_blocks;
            }
        }

        if( enable_elem_by_elem_dump )
        {
            ++num_elem_by_elem_blocks;

            num_objects  += num_elem_by_elem_blocks;
            num_slots    += num_elem_by_elem_blocks * req_num_slots_per_obj;
            num_dataptrs += num_elem_by_elem_blocks * req_num_dataptrs_per_obj;
        }

        if( ( 0 == NS(Buffer_reset)( io_buffer ) ) &&
            ( 0 == NS(Buffer_reserve)( io_buffer,
                num_objects, num_slots, num_dataptrs, ZERO ) ) )
        {
            buf_size_t ii = ZERO;

            SIXTRL_ASSERT( NS(Buffer_get_num_of_objects)( io_buffer ) == ZERO );
            SIXTRL_ASSERT( NS(Buffer_get_num_of_slots)(   io_buffer ) == ZERO );
            SIXTRL_ASSERT( NS(Buffer_get_num_of_dataptrs )( io_buffer ) == ZERO );

            success = 0;

            for( ; ii < num_objects ; ++ii )
            {
                ptr_particles_t particles =
                    NS(Particles_new)( io_buffer, num_particles );

                if( particles == SIXTRL_NULLPTR )
                {
                    success = -1;
                    break;
                }
            }
        }

        SIXTRL_ASSERT( ( success != 0 ) ||
            ( NS(Buffer_get_num_of_objects)( io_buffer ) == num_objects ) );

        if( success != 0 )
        {
            NS(Buffer_reset)( io_buffer );
        }
    }

    return success;
}

/* end: sixtracklib/common/be_monitor/be_monitor.c */
