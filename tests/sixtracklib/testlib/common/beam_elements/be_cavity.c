#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stddef.h>
    #include <stdlib.h>

    #if !defined( _GPUCODE )
    #include <stdio.h>
    #endif /* !defined( _GPUCODE ) */

#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/be_cavity/be_cavity.h"
    #include "sixtracklib/common/buffer/buffer_type.h"
    #include "sixtracklib/common/internal/type_comparison_helpers.hpp"
    #include "sixtracklib/testlib/common/beam_elements/be_cavity.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

void NS(Cavity_print)( SIXTRL_ARGPTR_DEC FILE* SIXTRL_RESTRICT fp,
    SIXTRL_BE_ARGPTR_DEC const NS(Cavity) *const SIXTRL_RESTRICT cavity )
{
    if( ( fp != SIXTRL_NULLPTR ) && ( cavity != SIXTRL_NULLPTR ) )
    {
        fprintf( fp,
            "|cavity           | voltage   = %+16.12f V   \r\n"
            "                  | frequency = %+20.12f Hz; \r\n"
            "                  | lag       = %+15.12f deg;\r\n",
            NS(Cavity_voltage)( cavity ), NS(Cavity_frequency)( cavity ),
            NS(Cavity_lag)( cavity ) );
    }
}
