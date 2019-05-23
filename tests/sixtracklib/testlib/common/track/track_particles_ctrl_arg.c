#include "sixtracklib/testlib/common/track/track_particles_ctrl_arg.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/particles/definitions.h"
#include "sixtracklib/common/buffer.h"
#include "sixtracklib/common/control/definitions.h"
#include "sixtracklib/common/control/argument_base.h"
#include "sixtracklib/common/control/debug_register.h"

#include "sixtracklib/testlib/common/particles/particles.h"

NS(arch_status_t) NS(TestTrackCtrlArg_prepare_ctrl_arg_tracking)(
    NS(ArgumentBase)* SIXTRL_RESTRICT particles_arg,
    NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
    NS(ArgumentBase)* SIXTRL_RESTRICT beam_elements_arg,
    NS(Buffer)* SIXTRL_RESTRICT beam_elements_buffer,
    NS(ArgumentBase)* SIXTRL_RESTRICT result_arg )
{
    NS(arch_status_t) status = NS(ARCH_STATUS_GENERAL_FAILURE);

    if( ( particles_arg != SIXTRL_NULLPTR ) &&
        ( particles_buffer != SIXTRL_NULLPTR ) &&
        ( beam_elements_arg != SIXTRL_NULLPTR ) &&
        ( beam_elements_buffer != SIXTRL_NULLPTR ) )
    {
        status = NS(Argument_send_buffer)( particles_arg, particles_buffer );

        if( status == NS(ARCH_STATUS_SUCCESS) )
        {
            status = NS(Argument_send_buffer)(
                beam_elements_arg, beam_elements_buffer );
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


NS(arch_status_t) NS(TestTrackCtrlArg_evaulate_ctrl_arg_tracking)(
    NS(ArgumentBase)* SIXTRL_RESTRICT particles_arg,
    NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
    NS(buffer_size_t) const num_particle_sets,
    NS(buffer_size_t) const* SIXTRL_RESTRICT particle_set_indices_begin,
    const NS(Buffer) *const SIXTRL_RESTRICT cmp_particles_buffer,
    NS(particle_real_t) const abs_tolerance,
    NS(ArgumentBase)* SIXTRL_RESTRICT result_arg )
{
    NS(arch_status_t) status = NS(ARCH_STATUS_GENERAL_FAILURE);

    if( ( particles_arg != SIXTRL_NULLPTR ) &&
        ( particles_buffer != SIXTRL_NULLPTR ) &&
        ( cmp_particles_buffer != SIXTRL_NULLPTR ) &&
        ( particle_set_indices_begin != SIXTRL_NULLPTR ) &&
        ( num_particle_sets > ( NS(buffer_size_t) )0u ) )
    {
        NS(buffer_size_t) const* pset_it  = particle_set_indices_begin;
        NS(buffer_size_t) const* pset_end = pset_it + num_particle_sets;

        /* Ensure that we are operating on the data returned by the argument
         * and not some remnant state already present in the particle_buffer) */

        NS(Buffer_clear)( particles_buffer, true );
        NS(Buffer_reset)( particles_buffer );

        status = NS(Argument_receive_buffer)(
            particles_arg, particles_buffer );

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

        while( ( status == NS(ARCH_STATUS_SUCCESS) ) &&
               ( pset_it != pset_end ) )
        {
            NS(Particles) const* cmp_particles =
                NS(Particles_buffer_get_const_particles)(
                    cmp_particles_buffer, *pset_it );

            NS(Particles) const* particles =
                NS(Particles_buffer_get_const_particles)(
                    particles_buffer, *pset_it );

            if( ( cmp_particles == SIXTRL_NULLPTR ) ||
                ( particles == SIXTRL_NULLPTR ) )
            {
                status = NS(ARCH_STATUS_GENERAL_FAILURE);
                break;
            }

            if( ( 0 != NS(Particles_compare_values)( cmp_particles, particles ) ) &&
                ( abs_tolerance > ( NS(particle_real_t) )0 ) &&
                ( 0 != NS(Particles_compare_values_with_treshold)(
                    cmp_particles, particles, abs_tolerance ) ) )
            {
                status = NS(ARCH_STATUS_GENERAL_FAILURE);
                break;
            }

            ++pset_it;
        }
    }

    return status;
}

/* end: tests/sixtracklib/testlib/common/track/track_particles_ctrl_arg.c */
