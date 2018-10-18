#include "sixtracklib/common/buffer/alignment.h"

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include "sixtracklib/common/definitions.h"

extern SIXTRL_UINT64_T NS(Alignment_calculate_commonN)(
    SIXTRL_UINT64_T const* SIXTRL_RESTRICT numbers,
    SIXTRL_UINT64_T const num_of_operands );

/* ------------------------------------------------------------------------- */

SIXTRL_UINT64_T NS(Alignment_calculate_commonN)(
    SIXTRL_UINT64_T const* SIXTRL_RESTRICT numbers,
    SIXTRL_UINT64_T const num_of_operands )
{
    SIXTRL_UINT64_T result = ( SIXTRL_UINT64_T )1u;

    SIXTRL_UINT64_T const NSTATIC = ( SIXTRL_UINT64_T )64u;
    SIXTRL_UINT64_T const LOG2_NUM_OPERANDS = NS(log2_ceil)( num_of_operands );

    SIXTRL_UINT64_T static_temp[ 64 ];
    SIXTRL_UINT64_T* temp = 0;

    if( LOG2_NUM_OPERANDS <= NSTATIC )
    {
        temp = &static_temp[ 0 ];
    }
    else
    {
        temp = ( SIXTRL_UINT64_T* )malloc(
            sizeof( SIXTRL_UINT64_T ) * LOG2_NUM_OPERANDS );
    }

    if( temp != 0 )
    {
        SIXTRL_UINT64_T ii = ( SIXTRL_UINT64_T )0u;
        SIXTRL_UINT64_T step2 = LOG2_NUM_OPERANDS;

        for( ; ii < num_of_operands ; ++ii )
        {
            temp[ ii ] = numbers[ ii ];
        }

        for( ; ii < LOG2_NUM_OPERANDS ; ++ii )
        {
            temp[ ii ] = ( SIXTRL_UINT64_T )1u;
        }

        while( step2 > 2u )
        {
            SIXTRL_UINT64_T step  = ( step2 >> 1u );
            SIXTRL_UINT64_T kk    = 0u;
            SIXTRL_UINT64_T jj    = step;

            for( ii = kk ; jj < LOG2_NUM_OPERANDS ;
                    kk += step, ii += step2, jj += step2 )
            {
                temp[ kk ] = NS(Alignment_calculate_common)(
                    temp[ ii ], temp[ jj ] );
            }

            step2 <<= 1u;
        }

        result = temp[ 0 ];
    }

    if( LOG2_NUM_OPERANDS > NSTATIC )
    {
        free( temp );
        temp = 0;
    }

    return result;
}

/* end: sixtracklib/common/buffer/alignment.c */
