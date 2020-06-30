#ifndef SIXTRACKLIB_TESTLIB_COMMON_BEAM_ELEMENTS_BE_MONITOR_H__
#define SIXTRACKLIB_TESTLIB_COMMON_BEAM_ELEMENTS_BE_MONITOR_H__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stddef.h>
    #include <stdlib.h>

    #if !defined( _GPUCODE )
    #include <stdio.h>
    #endif /* !defined( _GPUCODE ) */

#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/be_monitor/be_monitor.h"
    #include "sixtracklib/common/buffer/buffer_type.h"
    #include "sixtracklib/common/internal/type_comparison_helpers.hpp"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined(  _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

SIXTRL_STATIC SIXTRL_FN int NS(BeamMonitor_compare_values)(
    SIXTRL_BE_ARGPTR_DEC const NS(BeamMonitor) *const SIXTRL_RESTRICT lhs,
    SIXTRL_BE_ARGPTR_DEC const NS(BeamMonitor) *const SIXTRL_RESTRICT rhs
) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN void NS(BeamMonitor_print_out)(
    SIXTRL_BE_ARGPTR_DEC const NS(BeamMonitor) *const SIXTRL_RESTRICT elem );

#if !defined( _GPUCODE )

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(BeamMonitor_print)(
    SIXTRL_ARGPTR_DEC FILE* SIXTRL_RESTRICT fp,
    SIXTRL_BE_ARGPTR_DEC const NS(BeamMonitor) *const SIXTRL_RESTRICT elem );

#endif /* !defined( _GPUCODE ) */

#if !defined(  _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */
/* !!!!!!!!!!  Implementation of inline functions for NS(BeamMonitor)  !!!!!!!!!!! */
/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */

#if !defined(  _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

SIXTRL_INLINE int NS(BeamMonitor_compare_values)(
    SIXTRL_BE_ARGPTR_DEC const NS(BeamMonitor) *const SIXTRL_RESTRICT lhs,
    SIXTRL_BE_ARGPTR_DEC const NS(BeamMonitor) *const SIXTRL_RESTRICT
        rhs ) SIXTRL_NOEXCEPT
{
    int cmp_result = -1;

    if( ( lhs != SIXTRL_NULLPTR ) && ( rhs != SIXTRL_NULLPTR ) )
    {
        cmp_result = 0;

        if( lhs != rhs )
        {
            cmp_result = ( lhs->num_stores == rhs->num_stores )
                ? 0 : ( ( lhs->num_stores > rhs->num_stores ) ? +1 : -1 );

            if( cmp_result == 0 )
            {
                cmp_result = ( lhs->start == rhs->start )
                    ? 0 : ( ( lhs->start > rhs->start ) ? +1 : -1 );
            }

            if( cmp_result == 0 )
            {
                cmp_result = ( lhs->skip == rhs->skip )
                    ? 0 : ( ( lhs->skip > rhs->skip ) ? +1 : -1 );
            }

            if( cmp_result == 0 )
            {
                cmp_result = ( lhs->out_address == rhs->out_address )
                    ? 0 : ( ( lhs->out_address > rhs->out_address ) ? +1 : -1 );
            }

            if( cmp_result == 0 )
            {
                cmp_result = ( lhs->max_particle_id == rhs->max_particle_id )
                    ? 0 : ( ( lhs->max_particle_id > rhs->max_particle_id )
                        ? +1 : -1 );
            }

            if( cmp_result == 0 )
            {
                cmp_result = ( lhs->min_particle_id == rhs->min_particle_id )
                    ? 0 : ( ( lhs->min_particle_id > rhs->min_particle_id )
                        ? +1 : -1 );
            }

            if( cmp_result == 0 )
            {
                cmp_result = ( lhs->is_rolling == rhs->is_rolling )
                    ? 0 : ( ( lhs->is_rolling > rhs->is_rolling ) ? +1 : -1 );
            }

            if( cmp_result == 0 )
            {
                cmp_result = ( lhs->is_turn_ordered == rhs->is_turn_ordered )
                    ? 0 : ( ( lhs->is_turn_ordered > rhs->is_turn_ordered )
                        ? +1 : -1 );
            }
        }
    }
    else if( rhs != SIXTRL_NULLPTR )
    {
        cmp_result = +1;
    }

    return cmp_result;
}

SIXTRL_INLINE void NS(BeamMonitor_print_out)( SIXTRL_BE_ARGPTR_DEC const
    NS(BeamMonitor) *const SIXTRL_RESTRICT elem )
{
    if( elem != SIXTRL_NULLPTR )
    {
        printf( "beam-monitor        | num_stores      = %21ld\r\n"
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

#if !defined(  _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */
#endif /* SIXTRACKLIB_TESTLIB_COMMON_BEAM_ELEMENTS_BE_MONITOR_H__ */
