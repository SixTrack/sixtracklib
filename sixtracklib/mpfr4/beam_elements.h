#ifndef SIXTRACKLIB_MPFR4_BEAM_ELEMENTS_H__
#define SIXTRACKLIB_MPFR4_BEAM_ELEMENTS_H__

#include <stdint.h>
#include <stdio.h>
#include <stdarg.h>

#include <mpfr.h>

#include "sixtracklib/mpfr4/track.h"

#include "sixtracklib/common/impl/block_info_impl.h"
#include "sixtracklib/common/impl/be_drift_impl.h"
#include "sixtracklib/common/beam_elements.h"

#if defined( __cplusplus )
extern "C" {
#endif /* if defined( __cplusplus ) */

SIXTRL_STATIC int NS(BeamElements_create_beam_element_mpfr4)(
    void* SIXTRL_RESTRICT ptr_beam_element,
    NS(BeamElements)* SIXTRL_RESTRICT beam_elements,
    NS(BlockType) const type_id, mpfr_prec_t const prec );

SIXTRL_STATIC void NS(BeamElements_free_mpfr4)(
    NS(BeamElements)* SIXTRL_RESTRICT beam_elements );

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE int NS(BeamElements_create_beam_element_mpfr4)(
    void* SIXTRL_RESTRICT ptr_beam_element,
    NS(BeamElements)* SIXTRL_RESTRICT beam_elements,
    NS(BlockType) const type_id, mpfr_prec_t const prec )
{
    int success = -1;

    NS(BlockInfo)* block_info = NS(BeamElements_create_beam_element)(
        ptr_beam_element, beam_elements, type_id );

    if( block_info != 0 )
    {
        NS(BlockType) const type_id = NS(BlockInfo_get_type_id)( block_info );

        SIXTRL_ASSERT( ptr_beam_element != 0 );

        switch( type_id )
        {
            case NS(BLOCK_TYPE_DRIFT):
            case NS(BLOCK_TYPE_DRIFT_EXACT):
            {
                NS(Drift)* drift = ( NS(Drift)* )ptr_beam_element;
                SIXTRL_REAL_T* ptr_length = NS(Drift_get_length)( drift );

                SIXTRL_ASSERT( ptr_length != 0 );
                mpfr_init2( ptr_length->value, prec );

                success = 0;
                break;
            }

            default:
            {
                success = -1;
            }
        };
    }

    return success;
}

SIXTRL_INLINE void NS(BeamElements_free_mpfr4)(
    NS(BeamElements)* SIXTRL_RESTRICT beam_elements )
{
    NS(block_num_elements_t) ii = 0;
    NS(block_num_elements_t) const NUM_ELEMENTS =
        NS(BeamElements_get_num_of_blocks)( beam_elements );

    NS(BlockInfo)* be_block_info_it =
        NS(BeamElements_get_block_infos_begin)( beam_elements );

    for( ii = 0 ; ii < NUM_ELEMENTS ; ++ii, ++be_block_info_it )
    {
        NS(BlockType) const type_id =
            NS(BlockInfo_get_type_id)( be_block_info_it );

        switch( type_id )
        {
            case NS(BLOCK_TYPE_DRIFT):
            case NS(BLOCK_TYPE_DRIFT_EXACT):
            {
                NS(Drift) drift;

                if( 0 == NS(BeamElements_get_beam_element)(
                        &drift, beam_elements, ii ) )
                {
                    SIXTRL_REAL_T* length = NS(Drift_get_length)( &drift );

                    if( length != 0 )
                    {
                        mpfr_clear( length->value );
                    }
                }

                break;
            }

            default:
            {
                printf( "unknown type_id %u -> skipping\r\n",
                        ( unsigned )NS(BlockType_to_number)( type_id ) );
            }
        };

    }

    NS(BeamElements_free)( beam_elements );

    return;
}

#if defined( __cplusplus )
}
#endif /* if defined( __cplusplus ) */

#endif /* SIXTRACKLIB_MPFR4_BEAM_ELEMENTS_H__ */

/* end:  */
