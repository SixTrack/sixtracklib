#include "sixtracklib/testlib/common/output/assign_be_monitor_ctrl_arg.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/particles/definitions.h"
#include "sixtracklib/common/buffer.h"
#include "sixtracklib/common/control/definitions.h"
#include "sixtracklib/common/control/argument_base.h"
#include "sixtracklib/common/be_monitor/be_monitor.h"
#include "sixtracklib/common/be_monitor/output_buffer.h"

#include "sixtracklib/testlib/common/particles/particles.h"

NS(arch_status_t)
NS(TestBeamMonitorCtrlArg_prepare_assign_beam_monitor_output_buffer)(
    NS(ArgumentBase)* SIXTRL_RESTRICT beam_elements_arg,
    NS(Buffer)* SIXTRL_RESTRICT beam_elements_buffer,
    NS(ArgumentBase)* SIXTRL_RESTRICT output_arg,
    NS(Buffer)* SIXTRL_RESTRICT output_buffer,
    NS(particle_index_t) const min_turn_id,
    NS(buffer_size_t) const output_buffer_index_offset,
    NS(ArgumentBase)* SIXTRL_RESTRICT result_arg )
{
    NS(arch_status_t) status = NS(ARCH_STATUS_GENERAL_FAILURE);

    if( ( beam_elements_arg != SIXTRL_NULLPTR ) &&
        ( beam_elements_buffer != SIXTRL_NULLPTR ) &&
        ( output_arg != SIXTRL_NULLPTR ) &&
        ( output_buffer != SIXTRL_NULLPTR ) &&
        ( NS(Buffer_is_particles_buffer)( output_buffer ) ) &&
        ( NS(Buffer_get_num_of_objects)( output_buffer ) >
            output_buffer_index_offset ) )
    {
        status = NS(BeamMonitor_assign_output_buffer_from_offset)(
            beam_elements_buffer, output_buffer, min_turn_id,
                out_buffer_index_offset );

        if( status == NS(ARCH_STATUS_SUCCESS) )
        {
            status = NS(Argument_send_buffer)(
                beam_elements_arg, beam_elements_buffer );
        }

        if( status == NS(ARCH_STATUS_SUCCESS) )
        {
            status = NS(Argument_send_buffer)( output_arg, output_buffer );
        }

        if( ( status == NS(ARCH_STATUS_SUCCESS) ) &&
            ( result_arg != SIXTRL_NULLPTR ) )
        {
            NS(arch_debugging_t) result_register =
                SIXTRL_ARCH_DEBUGGING_REGISTER_EMPTY;

            status = NS(Argument_send_raw_argument)(
                result_arg, &result_register, sizeof( result_register ) );
        }
    }

    return status;
}

NS(arch_status_t)
NS(TestBeamMonitorCtrlArg_evaluate_assign_beam_monitor_output_buffer)(
    NS(ArgumentBase)* SIXTRL_RESTRICT beam_elements_arg,
    NS(Buffer)* SIXTRL_RESTRICT beam_elements_buffer,
    NS(ArgumentBase)* SIXTRL_RESTRICT output_arg,
    NS(Buffer)* SIXTRL_RESTRICT output_buffer,
    const NS(Buffer) *const SIXTRL_RESTRICT cmp_output_buffer,
    NS(buffer_size_t) const out_buffer_index_offset,
    NS(buffer_size_t) const num_beam_monitors,
    bool const compare_buffer_content,
    NS(particle_real_t) const abs_tolerance,
    NS(ArgumentBase)* SIXTRL_RESTRICT result_arg )
{
    typedef NS(buffer_addr_t) address_t;
    typedef NS(buffer_addr_diff_t) addr_diff_t;
    typedef NS(BeamMonitor) const* ptr_be_monitor_t;
    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* obj_iter_t;

    NS(arch_status_t) status = NS(ARCH_STATUS_GENERAL_FAILURE);
    NS(buffer_size_t) const nn = out_buffer_index_offset + num_beam_monitors;

    if( ( beam_elements_arg != SIXTRL_NULLPTR ) &&
        ( beamelements_buffer != SIXTRL_NULLPTR ) &&
        ( output_arg != SIXTRL_NULLPTR ) &&
        ( output_buffer != SIXTRL_NULLPTR ) &&
        ( cmp_output_buffer != SIXTRL_NULLPTR ) &&
        ( !NS(Buffer_needs_remapping)( cmp_output_buffer ) ) &&
        ( NS(Buffer_get_num_of_objects)( cmp_output_buffer ) >= nn ) &&
        ( NS(Buffer_is_particles_buffer)( cmp_output_buffer ) >
            ( NS(buffer_size_t) )0u ) )
    {
        NS(buffer_size_t) const slot_size =
            NS(Buffer_get_slot_size)( output_buffer );

        address_t host_base_addr = ( address_t )0u;
        address_t remote_base_addr = ( address_t )0u;

        SIXTRL_ASSERT( slot_size > ( NS(buffer_size_t) )0u );

        status = NS(Argument_receive_buffer)(
            bem_elements_arg, beam_elements_buffer );

        if( ( result_arg != SIXTRL_NULLPTR ) &&
            ( status == NS(ARCH_STATUS_SUCCESS) ) )
        {
            NS(arch_debugging_t) result_register =
                SIXTRL_ARCH_DEBUGGING_GENERAL_FAILURE;

            status = NS(Argument_receive_raw_argument)(
                result_arg, &result_register, sizeof( result_register ) );

            if( ( status == NS(ARCH_STATUS_SUCCESS) ) &&
                ( result_register != SIXTRL_ARCH_DEBUGGING_REGISTER_EMPTY ) )
            {
                if( NS(DebugReg_has_status_flags_set)( result_register ) )
                {
                    status = NS(DebugReg_get_stored_arch_status)(
                        result_register );
                }
                else
                {
                    status = NS(ARCH_STATUS_GENERAL_FAILURE);
                }

                SIXTRL_ASSERT( status != NS(ARCH_STATUS_SUCCESS) );
            }

            SIXTRL_ASSERT( ( status != NS(ARCH_STATUS_SUCCESS) ) ||
                ( result_register == SIXTRL_ARCH_DEBUGGING_REGISTER_EMPTY ) );
        }

        if( status == NS(ARCH_STATUS_SUCCESS) )
        {
            NS(Buffer_clear)( output_buffer, true );
            NS(Buffer_reset)( output_buffer );

            status = NS(Argument_receive_buffer_without_remapping)(
                output_arg, output_buffer );

            if( status == NS(ARCH_STATUS_SUCCESS) )
            {
                remote_base_addr = NS(ManagedBuffer_get_stored_begin_addr)(
                    NS(Buffer_get_data_begin)( output_buffer ), slot_size );

                host_base_addr = NS(ManagedBuffer_get_buffer_begin_addr)(
                    NS(Buffer_get_data_begin)( output_buffer ), slot_size );

                NS(Buffer_remap)( output_buffer );

                if( ( NS(Buffer_needs_remapping)( output_buffer ) ) ||
                    ( NS(Buffer_get_num_of_objects)( output_buffer ) <
                      ( output_buffer_index_offset + num_beam_monitors ) ) )
                {
                    status = NS(ARCH_STATUS_GENERAL_FAILURE);
                }
            }
        }

        if( status ==NS(ARCH_STATUS_SUCCESS) )
        {
            status = NS(ARCH_STATUS_GENERAL_FAILURE);

            if( ( remote_base_addr != ( address_t )0u ) &&
                ( host_base_addr   != ( address_t )0u ) )
            {
                NS(buffer_size_t) ii = out_buffer_index_offset;

                obj_iter_t be_it  = NS(Buffer_get_const_objects_begin)(
                    beam_elements_buffer );

                obj_iter_t be_end = NS(Buffer_get_const_objects_end)(
                    beam_elements_buffer );

                /* host = remote + diff_addr => diff_addr = host - remote */
                NS(buffer_addr_diff_t) const diff_addr =
                    ( host_base_addr >= remote_base_addr )
                    ? ( addr_diff_t )( host_base_addr - remote_base_addr )
                    : -( ( addr_diff_t )( remote_base_addr - host_base_addr ) );

                status = NS(ARCH_STATUS_SUCCESS);

                for( ; be_it != be_end ; ++be_it )
                {
                    NS(Particles) const* out_particles = SIXTRL_NULLPTR;
                    ptr_be_monitor_t be_monitor = SIXTRL_NULLPTR;
                    address_t cmp_out_addr = ( address_t )0u;
                    address_t be_out_addr  = ( address_t )0u;

                    if( NS(Object_get_type_id)( be_it ) !=
                        NS(OBJECT_TYPE_BEAM_MONITOR) )
                    {
                        continue;
                    }

                    out_particles = NS(Particles_buffer_get_const_particles)(
                            output_buffer, ii );

                    be_monitor = ( ptr_be_monitor_t )( uintptr_t
                        )NS(Object_get_begin_addr)( be_it );

                    be_out_addr = NS(BeamMonitor_get_out_address)( be_monitor );

                    cmp_out_addr = ( address_t )( uintptr_t
                        )NS(Particles_buffer_get_const_particles)(
                            output_buffer, ii++ );

                    if( ( diff_addr >= ( NS(buffer_addr_diff_t) )0u ) ||
                        ( ( -diff_addr ) >= (
                            NS(buffer_addr_diff_t) )be_out_addr ) )
                    {
                        be_out_addr += diff_addr;

                        if( be_out_addr != cmp_out_addr )
                        {
                            status = NS(ARCH_STATUS_GENERAL_FAILURE);
                            break;
                        }
                    }
                    else
                    {
                        status = NS(ARCH_STATUS_GENERAL_FAILURE);
                        break;
                    }

                    ++ii;

                    if( ii >= nn )
                    {
                        break;
                    }
                }
            }
        }

        if( status == NS(ARCH_STATUS_SUCCESS) )
        {
            NS(buffer_size_t) ii = out_buffer_index_offset;

            for( ; ii < nn ; ++ii )
            {
                NS(Particles) const* cmp_particles =
                    NS(Particles_buffer_get_const_particles)(
                        cmp_output_buffer, ii );

                NS(Particles) const* out_particles =
                    NS(Particles_buffer_get_const_particles)(
                        output_buffer, ii );


                if( ( out_particles == SIXTRL_NULLPTR ) ||
                    ( cmp_particles == SIXTRL_NULLPTR ) ||
                    ( NS(Particles_get_num_of_particles)( out_particles ) !=
                      NS(Particles_get_num_of_particles)( cmp_particles ) ) )
                {
                    status = NS(ARCH_STATUS_GENERAL_FAILURE);
                    break;
                }
            }
        }

        if( ( status == NS(ARCH_STATUS_SUCCESS) ) &&
            ( compare_buffer_content ) &&
            ( abs_tolerance >= ( NS(particle_real_t) )0.0 ) )
        {
            NS(buffer_size_t) ii = out_buffer_index_offset;

            for( ; ii < nn ; ++ii )
            {
                NS(Particles) const* cmp_particles =
                    NS(Particles_buffer_get_const_particles)(
                        cmp_output_buffer, ii );

                NS(Particles) const* out_particles =
                    NS(Particles_buffer_get_const_particles)(
                        output_buffer, ii );

                if( ( 0 != NS(Particles_compare_values)(
                            cmp_particles, out_particles ) ) &&
                    ( 0 != NS(Particles_compare_values_with_treshold)(
                        cmp_particles, out_particles, abs_tolerance ) ) )
                {
                    status = NS(ARCH_STATUS_GENERAL_FAILURE);
                    break;
                }
            }
        }
    }

    return status;
}

/* tests/sixtracklib/testlib/common/output/assign_elem_by_elem_ctrl_arg.c */
