#include "sixtracklib/opencl/track.h"

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/buffer.h"
#include "sixtracklib/common/particles.h"
#include "sixtracklib/common/track.h"
#include "sixtracklib/common/be_monitor/be_monitor.h"
#include "sixtracklib/common/be_monitor/output_buffer.h"

#include "sixtracklib/opencl/argument.h"
#include "sixtracklib/opencl/context.h"

NS(Buffer)* NS(TrackCL)(
    char const* SIXTRL_RESTRICT device_id_str,
    struct NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
    struct NS(Buffer)* SIXTRL_RESTRICT beam_elements_buffer,
    struct NS(Buffer)* SIXTRL_RESTRICT out_buffer,
    int const until_turn,
    int const elem_by_elem_turns )
{
    typedef NS(buffer_size_t) buf_size_t;

    NS(Buffer)* ptr_out_buffer = SIXTRL_NULLPTR;
    NS(ClContext)* context   = NS(ClContext_create)();
    if (device_id_str==NULL){
        NS(ClContextBase_print_nodes_info)( context );
        NS(ClContextBase_delete)( context );
        return ptr_out_buffer;
    } else {
        NS(ClContextBase_select_node)(context, device_id_str);
    };

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

        buf_size_t const out_buffer_offset = ( elem_by_elem_turns > 0 )
                ? ( buf_size_t )1u : ( buf_size_t )0u;

        if( out_buffer != SIXTRL_NULLPTR )
        {
            ptr_out_buffer = out_buffer;
        }
        else
        {
            ptr_out_buffer = NS(Buffer_new)( 0u );
        }

        if( prepare_out_buffer )
        {
            NS(BeamMonitor_prepare_particles_out_buffer)(
                beam_elements_buffer, ptr_out_buffer,
                    particles, elem_by_elem_turns );
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
                context, beam_elements_arg, out_buffer, out_buffer_offset );
        }

        if( elem_by_elem_turns > 0 )
        {
            NS(ClContext_track_element_by_element)( context, particles_arg,
                beam_elements_arg, output_arg, elem_by_elem_turns,
                    out_buffer_offset );
        }

        if( until_turn > elem_by_elem_turns )
        {
            NS(ClContext_track)(
                context, particles_arg, beam_elements_arg, until_turn );
        }

        NS(ClArgument_read)( particles_arg, particles_buffer );
        NS(ClArgument_read)( output_arg, ptr_out_buffer );

        particles = NS(Particles_buffer_get_particles)( particles_buffer, 0u );
        NS(Particles_add_copy)( ptr_out_buffer, particles );
    }

    NS(ClContextBase_delete)( context );

    return ptr_out_buffer;
}

/* end: sixtracklib/opencl/internal/track.c */

