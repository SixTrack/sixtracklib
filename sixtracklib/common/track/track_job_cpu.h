#ifndef SIXTRACKLIB_COMMMON_TRACK_TRACK_JOB_CPU_C99_H__
#define SIXTRACKLIB_COMMMON_TRACK_TRACK_JOB_CPU_C99_H__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include "sixtracklib/common/track/track_job_cpu.hpp"
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stddef.h>
    #include <stdlib.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/buffer.h"
    #include "sixtracklib/common/control/definitions.h"
    #include "sixtracklib/common/track/track_job_base.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus ) && !defined( _GPUCODE )
extern "C" {
#endif /* C++, Host */

SIXTRL_EXTERN SIXTRL_HOST_FN NS(CpuTrackJob)* NS(CpuTrackJob_create)( void );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(CpuTrackJob)*
NS(CpuTrackJob_new_from_config_str)( char const* SIXTRL_RESTRICT config_str );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(CpuTrackJob)*
NS(CpuTrackJob_new)(
    NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
    NS(Buffer)* SIXTRL_RESTRICT beam_elements_buffer );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(CpuTrackJob)*
NS(CpuTrackJob_new_with_output)(
    NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
    NS(Buffer)* SIXTRL_RESTRICT beam_elements_buffer,
    NS(Buffer)* SIXTRL_RESTRICT output_buffer,
    NS(buffer_size_t) const until_turn_elem_by_elem );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(CpuTrackJob)*
NS(CpuTrackJob_new_detailed)(
    NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
    NS(buffer_size_t) const num_psets,
    NS(buffer_size_t) const* SIXTRL_RESTRICT pset_indices_begin,
    NS(Buffer)* SIXTRL_RESTRICT beam_elements_buffer,
    NS(Buffer)* SIXTRL_RESTRICT output_buffer,
    NS(buffer_size_t) const until_turn_elem_by_elem,
    char const* SIXTRL_RESTRICT config_str );

#if defined( __cplusplus ) && !defined( _GPUCODE )
}
#endif /* C++, Host */

#endif /* #define SIXTRACKLIB_COMMMON_TRACK_TRACK_JOB_CPU_C99_H__ */

/* end: sixtracklib/common/track/track_job_cpu.h */
