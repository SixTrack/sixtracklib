#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/buffer.h"
#include "sixtracklib/common/particles.h"
#include "sixtracklib/common/track/definitions.h"
#include "sixtracklib/common/track/track.h"
#include "sixtracklib/common/be_monitor/be_monitor.h"
#include "sixtracklib/common/output/output_buffer.h"
// #include "sixtracklib/common/context/context_base.h"

#include "sixtracklib/opencl/argument.h"
#include "sixtracklib/opencl/context.h"

#include "sixtracklib/opencl/track.h"

NS(Buffer)* NS(TrackCL)(
    NS(ClContext)* context,
    struct NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
    struct NS(Buffer)* SIXTRL_RESTRICT beam_elements_buffer,
    struct NS(Buffer)* SIXTRL_RESTRICT out_buffer,
    int const until_turn,
    int const elem_by_elem_turns )
{
    typedef NS(buffer_size_t)       buf_size_t;
    typedef NS(particle_index_t)    index_t;

    NS(Buffer)* ptr_out_buffer = NULL;

    NS(Particles)* particles =
        NS(Particles_buffer_get_particles)( particles_buffer, 0u );

    if( ( context != SIXTRL_NULLPTR ) && ( particles != SIXTRL_NULLPTR ) &&
        ( beam_elements_buffer != SIXTRL_NULLPTR ) )
    {
        NS(ClArgument)* particles_arg     = SIXTRL_NULLPTR;
        NS(ClArgument)* beam_elements_arg = SIXTRL_NULLPTR;
        NS(ClArgument)* output_arg        = SIXTRL_NULLPTR;

        bool const has_beam_monitors =
            NS(BeamMonitor_are_present_in_buffer)( beam_elements_buffer );

        bool const prepare_out_buffer =
            ( ( elem_by_elem_turns > 0 ) || ( has_beam_monitors ) );

        buf_size_t elem_by_elem_index_offset = ( buf_size_t )0u;
        buf_size_t beam_monitor_index_offset = ( buf_size_t )0u;
        index_t min_turn_id                  = ( index_t )0;

        if( out_buffer != SIXTRL_NULLPTR )
        {
            index_t max_turn_id = ( index_t )-1;
            ptr_out_buffer = out_buffer;

            NS(Particles_get_min_max_at_turn_value)(
                particles, &min_turn_id, &max_turn_id );

            if( prepare_out_buffer )
            {
                beam_monitor_index_offset = ( buf_size_t )1u;
            }
        }
        else
        {
            ptr_out_buffer = NS(Buffer_new)( 0u );

            NS(OutputBuffer_prepare)( beam_elements_buffer, ptr_out_buffer,
                particles, elem_by_elem_turns, &elem_by_elem_index_offset,
                &beam_monitor_index_offset, &min_turn_id );
        }

        particles_arg =
            NS(ClArgument_new_from_buffer)( particles_buffer, context );

        beam_elements_arg =
            NS(ClArgument_new_from_buffer)( beam_elements_buffer, context );

        output_arg =
            NS(ClArgument_new_from_buffer)( ptr_out_buffer, context );

        if( prepare_out_buffer )
        {
            NS(ClContext_assign_beam_monitor_out_buffer)(
                context, beam_elements_arg, out_buffer, min_turn_id,
                beam_monitor_index_offset );
        }

        if( elem_by_elem_turns > 0 )
        {
            NS(ClContext_track_element_by_element)( context, particles_arg,
                beam_elements_arg, output_arg, elem_by_elem_turns,
                    elem_by_elem_index_offset );
        }

        if( until_turn > elem_by_elem_turns )
        {
            NS(ClContext_track_until)(
                context, particles_arg, beam_elements_arg, until_turn );
        }

        NS(ClArgument_read)( particles_arg, particles_buffer );
        NS(ClArgument_read)( output_arg, ptr_out_buffer );

        NS(ManagedBuffer_remap) ( (unsigned char *) particles_buffer->data_addr, 8);
        NS(ManagedBuffer_remap) ( (unsigned char *) ptr_out_buffer->data_addr, 8);

        //NS(Particles_add_copy)( ptr_out_buffer, particles );

        NS(ClArgument_delete)(particles_arg);
        NS(ClArgument_delete)(beam_elements_arg);
        NS(ClArgument_delete)(output_arg);

    }

    return ptr_out_buffer;
}

/* end: sixtracklib/opencl/internal/track.c */

