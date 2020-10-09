#include "sixtracklib/testlib/common/output/assign_elem_by_elem_ctrl_arg.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/particles/definitions.h"
#include "sixtracklib/common/buffer.h"
#include "sixtracklib/common/control/definitions.h"
#include "sixtracklib/common/control/argument_base.h"
#include "sixtracklib/common/output/elem_by_elem_config.h"
#include "sixtracklib/common/output/output_buffer.h"

#include "sixtracklib/testlib/common/particles/particles.h"

NS(arch_status_t)
NS(TestElemByElemConfigCtrlArg_prepare_assign_output_buffer)(
    const NS(Buffer) *const SIXTRL_RESTRICT particles_buffer,
    NS(buffer_size_t) const num_particle_sets,
    NS(buffer_size_t) const* SIXTRL_RESTRICT pset_indices_begin,
    NS(Buffer)* SIXTRL_RESTRICT beam_elements_buffer,
    NS(ArgumentBase)* SIXTRL_RESTRICT elem_by_elem_config_buffer_arg,
    NS(Buffer)* SIXTRL_RESTRICT elem_by_elem_config_buffer,
    NS(buffer_size_t) const elem_by_elem_config_index,
    NS(ArgumentBase)* SIXTRL_RESTRICT output_arg,
    NS(Buffer)* SIXTRL_RESTRICT output_buffer,
    NS(buffer_size_t)* SIXTRL_RESTRICT ptr_output_buffer_index_offset,
    NS(buffer_size_t) const until_turn_elem_by_elem,
    NS(ArgumentBase)* SIXTRL_RESTRICT result_arg )
{
    NS(arch_status_t) status = NS(ARCH_STATUS_GENERAL_FAILURE);

    if( ( particles_buffer != SIXTRL_NULLPTR ) &&
        ( num_particle_sets > ( NS(buffer_size_t) )0u ) &&
        ( pset_indices_begin != SIXTRL_NULLPTR ) &&
        ( beam_elements_buffer != SIXTRL_NULLPTR ) &&
        ( elem_by_elem_config_buffer_arg != SIXTRL_NULLPTR ) &&
        ( elem_by_elem_config_buffer != SIXTRL_NULLPTR ) &&
        ( ( NS(Buffer_get_num_of_objects)( elem_by_elem_config_buffer ) ==
            ( NS(buffer_size_t) )0u ) ||
          ( NS(Buffer_get_num_of_objects)( elem_by_elem_config_buffer ) >
            elem_by_elem_config_index ) ) &&
        ( output_arg != SIXTRL_NULLPTR ) &&
        ( output_buffer != SIXTRL_NULLPTR ) &&
        ( ( NS(Buffer_is_particles_buffer)( output_buffer ) ) ||
          ( NS(Buffer_get_num_of_objects)( output_buffer ) ==
              ( NS(buffer_size_t) )0u ) ) )
    {
        typedef NS(particle_index_t) pindex_t;
        typedef NS(buffer_size_t) buf_size_t;

        pindex_t min_particle_id, max_particle_id;
        pindex_t min_at_element_id, max_at_element_id;
        pindex_t min_at_turn_id, max_at_turn_id;

        buf_size_t num_elem_by_elem_elements = ( buf_size_t )0u;
        pindex_t const start_elem_id = ( pindex_t )0u;

        buf_size_t output_buffer_index_offset = ( buf_size_t )0u;
        buf_size_t beam_monitor_output_offset = ( buf_size_t )0u;
        pindex_t   max_elem_by_elem_turn_id   = ( pindex_t )-1;

        NS(ElemByElemConfig)* elem_by_elem_config =
            NS(ElemByElemConfig_from_buffer)(
                elem_by_elem_config_buffer, elem_by_elem_config_index );

        status = NS(OutputBuffer_get_min_max_attributes_on_particle_sets)(
            particles_buffer, num_particle_sets, pset_indices_begin,
            beam_elements_buffer,
            &min_particle_id, &max_particle_id, &min_at_element_id,
            &max_at_element_id, &min_at_turn_id, &max_at_turn_id,
            &num_elem_by_elem_elements, start_elem_id );

        if( status == NS(ARCH_STATUS_SUCCESS) )
        {
            status = NS(OutputBuffer_prepare_detailed)(
                beam_elements_buffer, output_buffer,
                min_particle_id, max_particle_id, min_at_element_id,
                max_at_element_id, min_at_turn_id, max_at_turn_id,
                until_turn_elem_by_elem, &output_buffer_index_offset,
                &beam_monitor_output_offset,
                &max_elem_by_elem_turn_id );
        }

        if( ptr_output_buffer_index_offset != SIXTRL_NULLPTR )
        {
            *ptr_output_buffer_index_offset = output_buffer_index_offset;
        }



        if( status == NS(ARCH_STATUS_SUCCESS) )
        {
            status = NS(ElemByElemConfig_init_detailed)( elem_by_elem_config,
                NS(ELEM_BY_ELEM_ORDER_DEFAULT), min_particle_id,
                    max_particle_id, min_at_element_id, max_at_element_id,
                        min_at_turn_id, max_elem_by_elem_turn_id, true );
        }

        if( status == NS(ARCH_STATUS_SUCCESS) )
        {
            status = NS(Argument_send_buffer)(
                elem_by_elem_config_buffer_arg, elem_by_elem_config_buffer );
        }

        if( status == NS(ARCH_STATUS_SUCCESS) )
        {
            status = NS(Argument_send_buffer)( output_arg, output_buffer );
        }

        if( status == NS(ARCH_STATUS_SUCCESS) )
        {
            status = NS(ElemByElemConfig_assign_output_buffer)(
                elem_by_elem_config, output_buffer,
                    output_buffer_index_offset );
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

NS(arch_status_t) NS(TestElemByElemConfigCtrlArg_evaluate_assign_output_buffer)(
    NS(ArgumentBase)* SIXTRL_RESTRICT elem_by_elem_config_buffer_arg,
    NS(Buffer)* SIXTRL_RESTRICT elem_by_elem_config_buffer,
    NS(buffer_size_t) const elem_by_elem_config_index,
    NS(ArgumentBase)* SIXTRL_RESTRICT output_arg,
    NS(Buffer)* SIXTRL_RESTRICT output_buffer,
    NS(buffer_size_t) const output_buffer_index_offset,
    NS(ArgumentBase)* SIXTRL_RESTRICT result_arg )
{
    typedef NS(buffer_addr_t) address_t;
    typedef NS(buffer_size_t) buf_size_t;
    typedef NS(buffer_addr_diff_t) addr_diff_t;

    NS(arch_status_t) status = NS(ARCH_STATUS_GENERAL_FAILURE);
    buf_size_t const nn = output_buffer_index_offset + ( buf_size_t )1u;

    if( ( elem_by_elem_config_buffer_arg != SIXTRL_NULLPTR ) &&
        ( elem_by_elem_config_buffer != SIXTRL_NULLPTR ) &&
        ( output_arg != SIXTRL_NULLPTR ) &&
        ( output_buffer != SIXTRL_NULLPTR ) &&
        ( NS(Buffer_get_num_of_objects)( elem_by_elem_config_buffer ) >
            elem_by_elem_config_index ) &&
        ( NS(Buffer_get_num_of_objects)( output_buffer ) >
            output_buffer_index_offset ) &&
        ( NS(Buffer_is_particles_buffer)( output_buffer ) ) )
    {
        NS(ElemByElemConfig)* elem_by_elem_config = SIXTRL_NULLPTR;

        NS(Particles) const* cmp_particles =
            NS(Particles_buffer_get_const_particles)( output_buffer,
                output_buffer_index_offset );

        NS(particle_num_elements_t) const CMP_NUM_PARTICLES =
            NS(Particles_get_num_of_particles)( cmp_particles );

        NS(buffer_size_t) const slot_size =
            NS(Buffer_get_slot_size)( output_buffer );

        address_t host_base_addr = ( address_t )0u;
        address_t remote_base_addr = ( address_t )0u;

        SIXTRL_ASSERT( slot_size > ( NS(buffer_size_t) )0u );

        status = NS(Argument_receive_buffer)(
            elem_by_elem_config_buffer_arg, elem_by_elem_config_buffer );

        if( ( result_arg != SIXTRL_NULLPTR ) &&
            ( status == NS(ARCH_STATUS_SUCCESS) ) )
        {
            NS(arch_debugging_t) result_register =
                SIXTRL_ARCH_DEBUGGING_GENERAL_FAILURE;

            elem_by_elem_config = NS(ElemByElemConfig_from_buffer)(
                    elem_by_elem_config_buffer, elem_by_elem_config_index );

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
            cmp_particles = SIXTRL_NULLPTR;
            NS(Buffer_clear)( output_buffer, true );
            NS(Buffer_reset)( output_buffer );

            status = NS(Argument_receive_buffer_without_remap)(
                output_arg, output_buffer );

            if( status == NS(ARCH_STATUS_SUCCESS) )
            {
                remote_base_addr = NS(ManagedBuffer_get_stored_begin_addr)(
                    NS(Buffer_get_data_begin)( output_buffer ), slot_size );

                host_base_addr = NS(ManagedBuffer_get_buffer_begin_addr)(
                    NS(Buffer_get_data_begin)( output_buffer ), slot_size );

                NS(Buffer_remap)( output_buffer );

                if( ( NS(Buffer_needs_remapping)( output_buffer ) ) ||
                    ( NS(Buffer_get_num_of_objects)( output_buffer ) < nn ) )
                {
                    status = NS(ARCH_STATUS_GENERAL_FAILURE);
                }
            }
        }

        if( status == NS(ARCH_STATUS_SUCCESS) )
        {
            status = NS(ARCH_STATUS_GENERAL_FAILURE);

            if( ( remote_base_addr != ( address_t )0u ) &&
                ( host_base_addr   != ( address_t )0u ) )
            {
                address_t const cmp_out_addr = ( address_t )( uintptr_t
                    )NS(Particles_buffer_get_const_particles)(
                            output_buffer, output_buffer_index_offset );

                address_t elem_by_elem_conf_out_addr  = ( address_t )0u;

                /* host = remote + diff_addr => diff_addr = host - remote */
                NS(buffer_addr_diff_t) const diff_addr =
                    ( host_base_addr >= remote_base_addr )
                    ? ( addr_diff_t )( host_base_addr - remote_base_addr )
                    : -( ( addr_diff_t )( remote_base_addr - host_base_addr ) );

                elem_by_elem_conf_out_addr =
                NS(ElemByElemConfig_get_output_store_address)(
                    elem_by_elem_config );

                status = NS(ARCH_STATUS_SUCCESS);

                if( ( diff_addr >= ( NS(buffer_addr_diff_t) )0u ) ||
                    ( ( -diff_addr ) <= (
                        NS(buffer_addr_diff_t) )elem_by_elem_conf_out_addr ) )
                {
                    elem_by_elem_conf_out_addr += diff_addr;

                    if( elem_by_elem_conf_out_addr != cmp_out_addr )
                    {
                        status = NS(ARCH_STATUS_GENERAL_FAILURE);
                    }
                }
                else
                {
                    status = NS(ARCH_STATUS_GENERAL_FAILURE);
                }
            }
        }

        if( status == NS(ARCH_STATUS_SUCCESS) )
        {
            NS(Particles) const* cmp_particles =
                NS(Particles_buffer_get_const_particles)(
                    output_buffer, output_buffer_index_offset );

            if( ( cmp_particles == SIXTRL_NULLPTR ) ||
                ( CMP_NUM_PARTICLES != NS(Particles_get_num_of_particles)(
                    cmp_particles ) ) )
            {
                status = NS(ARCH_STATUS_GENERAL_FAILURE);
            }
        }
    }

    return status;
}

/* tests/sixtracklib/testlib/common/output/assign_elem_by_elem_ctrl_arg.c */
