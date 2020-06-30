#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include "sixtracklib/testlib/common/beam_elements/be_limit_rect.h"
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stddef.h>
    #include <stdlib.h>
    #include <stdio.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/be_limit/definitions.h"
    #include "sixtracklib/common/be_limit/be_limit_rect.h"
    #include "sixtracklib/common/buffer/buffer_type.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( _GPUCODE )

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stdio.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

void NS(LimitRect_print)( SIXTRL_ARGPTR_DEC FILE* SIXTRL_RESTRICT fp,
    SIXTRL_BE_ARGPTR_DEC const NS(LimitRect) *const SIXTRL_RESTRICT limit )
{
    if( ( fp != SIXTRL_NULLPTR ) && ( limit != SIXTRL_NULLPTR ) )
    {
        fprintf( fp, "|limit_rect       | min_x    = %+20.12f m;\r\n"
                "                  | max_x    = %+20.12f m;\r\n"
                "                  | min_y    = %+20.12f m;\r\n"
                "                  | max_y    = %+20.12f m;\r\n",
                NS(LimitRect_min_x)( limit ), NS(LimitRect_max_x)( limit ),
                NS(LimitRect_min_y)( limit ), NS(LimitRect_max_y)( limit ) );
    }

    return;
}

#endif /* !defined( _GPUCODE ) */

/* end: tests/sixtracklib/testlib/common/beam_elements/be_limit_rect.c */
