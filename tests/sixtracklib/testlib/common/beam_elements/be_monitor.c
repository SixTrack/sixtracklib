#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/testlib/common/beam_elements/be_monitor.h"
    #include "sixtracklib/common/be_monitor/be_monitor.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stddef.h>
    #include <stdlib.h>

    #if !defined( _GPUCODE )
    #include <stdio.h>
    #endif /* !defined( _GPUCODE ) */

#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

void NS(BeamMonitor_print)(
    SIXTRL_ARGPTR_DEC FILE* SIXTRL_RESTRICT fp,
    SIXTRL_BE_ARGPTR_DEC const NS(BeamMonitor) *const SIXTRL_RESTRICT elem )
{
    if( ( elem != SIXTRL_NULLPTR ) && ( fp != SIXTRL_NULLPTR ) )
    {
        fprintf( fp, "beam-monitor        | num_stores      = %21ld\r\n"
                "                    | start           = %21ld\r\n"
                "                    | skip            = %21ld\r\n"
                "                    | out_address     = %21lx\r\n"
                "                    | min_particle_id = %21ld\r\n"
                "                    | max_particle_id = %21ld\r\n"
                "                    | is_rolling      = %21ld\r\n"
                "                    | is_turn_ordered = %21ld\r\n",
                ( long int )NS(BeamMonitor_num_stores)( elem ),
                ( long int )NS(BeamMonitor_start)( elem ),
                ( long int )NS(BeamMonitor_skip)( elem ),
                ( uintptr_t )NS(BeamMonitor_out_address)( elem ),
                ( long int )NS(BeamMonitor_min_particle_id)( elem ),
                ( long int )NS(BeamMonitor_max_particle_id)( elem ),
                ( long int )NS(BeamMonitor_is_rolling)( elem ),
                ( long int )NS(BeamMonitor_is_turn_ordered)( elem ) );
    }
}
