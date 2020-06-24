#include "sixtracklib/testlib/common/beam_elements/be_drift.h"

void NS(Drift_print)( SIXTRL_ARGPTR_DEC FILE* SIXTRL_RESTRICT fp,
    SIXTRL_BE_ARGPTR_DEC const NS(Drift) *const SIXTRL_RESTRICT elem )
{
    if( ( fp != SIXTRL_NULLPTR ) && ( elem != SIXTRL_NULLPTR ) )
    {
        fprintf( fp, "drift             | length   = %+20.18f m;\r\n",
                 NS(Drift_length)( elem ) );
    }
}

void NS(DriftExact_print)( SIXTRL_ARGPTR_DEC FILE* SIXTRL_RESTRICT fp,
    SIXTRL_BE_ARGPTR_DEC const NS(DriftExact) *const SIXTRL_RESTRICT elem )
{
    if( ( fp != SIXTRL_NULLPTR ) && ( elem != SIXTRL_NULLPTR ) )
    {
        fprintf( fp, "exact drift       | length   = %+20.18f m;\r\n",
                 NS(DriftExact_length)( elem ) );
    }
}

