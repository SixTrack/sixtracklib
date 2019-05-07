#ifndef SIXTRACKLIB_COMMON_TRACK_DEFINITIONS_H__
#define SIXTRACKLIB_COMMON_TRACK_DEFINITIONS_H__

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/buffer/buffer_type.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

/* ------------------------------------------------------------------------- */

#if !defined( SIXTRL_TRACK_SUCCESS )
    #define SIXTRL_TRACK_SUCCESS 0
#endif /* !defined( SIXTRL_TRACK_SUCCESS ) */

#if defined( __cplusplus ) && !defined( _GPUCODE )
extern "C" {
#endif /* defined( __cplusplus ) && ( !defined( _GPUCODE ) */

typedef SIXTRL_INT32_T      NS(track_status_t);

#if !defined( _GPUCODE )
typedef SIXTRL_UINT16_T     NS(track_job_collect_flag_t);
typedef NS(buffer_size_t)   NS(track_job_size_t);
typedef SIXTRL_INT64_T      NS(track_job_type_t);

/* ------------------------------------------------------------------------- */

SIXTRL_STATIC_VAR NS(track_status_t) const
    NS(TRACK_SUCCESS) = ( NS(track_status_t) )0u;

SIXTRL_STATIC_VAR NS(track_status_t) const
    NS(TRACK_STATUS_GENERAL_FAILURE) = ( NS(track_status_t) )-1;

SIXTRL_STATIC_VAR NS(track_job_collect_flag_t) const
    NS(TRACK_JOB_COLLECT_NONE) = ( NS(track_job_collect_flag_t) )0x00;

SIXTRL_STATIC_VAR NS(track_job_collect_flag_t) const
    NS(TRACK_JOB_COLLECT_PARTICLES) = ( NS(track_job_collect_flag_t) )0x01;

SIXTRL_STATIC_VAR NS(track_job_collect_flag_t) const
    NS(TRACK_JOB_COLLECT_BEAM_ELEMENTS) = ( NS(track_job_collect_flag_t) )0x02;

SIXTRL_STATIC_VAR NS(track_job_collect_flag_t) const
    NS(TRACK_JOB_COLLECT_OUTPUT) = ( NS(track_job_collect_flag_t) )0x04;

SIXTRL_STATIC_VAR NS(track_job_collect_flag_t) const
    NS(TRACK_JOB_COLLECT_SUCCSS_FLAG) = ( NS(track_job_collect_flag_t) )0x08;

SIXTRL_STATIC_VAR NS(track_job_collect_flag_t) const
    NS(TRACK_JOB_COLLECT_ALL) = ( NS(track_job_collect_flag_t) )0x0F;

SIXTRL_STATIC_VAR NS(track_job_collect_flag_t) const
    NS(TRACK_JOB_COLLECT_DEFAULT_FLAGS) = ( NS(track_job_collect_flag_t) )0x05;

SIXTRL_STATIC_VAR NS(track_job_size_t) const
    NS(TRACK_JOB_DEFAULT_NUM_PARTICLE_SETS) = ( NS(track_job_size_t) )1u;

SIXTRL_STATIC_VAR NS(track_job_size_t) const
    NS(TRACK_JOB_DEFAULT_PARTICLE_SET_INDICES)[] = { 0u };

#endif /* !defined( _GPUCODE ) */

#if defined( __cplusplus ) && !defined( _GPUCODE )
}
#endif /* defined( __cplusplus ) && ( !defined( _GPUCODE ) */

#if defined( __cplusplus )
namespace SIXTRL_CXX_NAMESPACE
{
    using track_status_t = ::NS(track_status_t);
    SIXTRL_STATIC_VAR track_status_t const TRACK_SUCCESS = track_status_t{ 0 };

    SIXTRL_STATIC_VAR track_status_t const
        TRACK_STATUS_GENERAL_FAILURE = track_status_t{ -1 };

}
#endif /* defined( __cplusplus )  */


#if defined( __cplusplus ) && !defined( _GPUCODE )
namespace SIXTRL_CXX_NAMESPACE
{
    using track_job_collect_flag_t = ::NS(track_job_collect_flag_t);
    using track_job_size_t         = ::NS(track_job_size_t);
    using track_job_type_t         = ::NS(track_job_type_t);

    SIXTRL_STATIC_VAR track_job_collect_flag_t const
        TRACK_JOB_COLLECT_NONE = track_job_collect_flag_t{ 0x0000 };

    SIXTRL_STATIC_VAR track_job_collect_flag_t const
        TRACK_JOB_COLLECT_PARTICLES = track_job_collect_flag_t{ 0x0001 };

    SIXTRL_STATIC_VAR track_job_collect_flag_t const
        TRACK_JOB_COLLECT_BEAM_ELEMENTS = track_job_collect_flag_t{ 0x0002 };

    SIXTRL_STATIC_VAR track_job_collect_flag_t const
        TRACK_JOB_COLLECT_OUTPUT = track_job_collect_flag_t{ 0x0004 };

    SIXTRL_STATIC_VAR track_job_collect_flag_t const
        TRACK_JOB_COLLECT_SUCCESS_FLAG = track_job_collect_flag_t{ 0x0008 };

    SIXTRL_STATIC_VAR track_job_collect_flag_t const
        TRACK_JOB_COLLECT_ALL = track_job_collect_flag_t{ 0x000F };

    SIXTRL_STATIC_VAR track_job_collect_flag_t const
        TRACK_JOB_COLLECT_DEFAULT_FLAGS = track_job_collect_flag_t{ 0x0005 };

    SIXTRL_STATIC_VAR track_job_size_t const
        TRACK_JOB_DEFAULT_NUM_PARTICLE_SETS = track_job_size_t{ 1 };

    SIXTRL_STATIC_VAR track_job_size_t const
        TRACK_JOB_DEFAULT_PARTICLE_SET_INDICES[] = { track_job_size_t{ 0 } };
}
#endif /* defined( __cplusplus ) && !defined( _GPUCODE ) */

#endif /* SIXTRACKLIB_COMMON_TRACK_DEFINITIONS_H__ */
/* end: sixtracklib/common/track/definitions.h */
