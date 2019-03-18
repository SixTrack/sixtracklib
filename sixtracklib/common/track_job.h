#ifndef SIXTRACKLIB_SIXTRACKLIB_COMMON_TRACK_JOB_H__
#define SIXTRACKLIB_SIXTRACKLIB_COMMON_TRACK_JOB_H__

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/buffer.h"
    #include "sixtracklib/common/internal/track_job_base.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( _GPUCODE )
#if defined( __cplusplus )

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <string>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/buffer.hpp"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

namespace SIXTRL_CXX_NAMESPACE
{
    SIXTRL_HOST_FN TrackJobBase* createTrackJob(
        const char *const SIXTRL_RESTRICT type_str,
        const char *const SIXTRL_RESTRICT device_id_str  = nullptr,
        const char *const SIXTRL_RESTRICT config_str     = nullptr );

    SIXTRL_HOST_FN TrackJobBase* createTrackJob(
        const char *const SIXTRL_RESTRICT type_str,
        ::NS(Buffer)* particles_buffer,
        ::NS(Buffer)* beam_elements_buffer,
        const char *const SIXTRL_RESTRICT device_id_str  = nullptr,
        const char *const SIXTRL_RESTRICT config_str     = nullptr,
        ::NS(buffer_size_t) const num_elem_by_elem_turns
            = ::NS(buffer_size_t){ 0 } );

    SIXTRL_HOST_FN TrackJobBase* createTrackJob(
        const char *const SIXTRL_RESTRICT type_str,
        ::NS(Buffer)* particles_buffer,
        ::NS(Buffer)* beam_elements_buffer,
        ::NS(Buffer)* output_buffer,
        const char *const SIXTRL_RESTRICT device_id_str  = nullptr,
        const char *const SIXTRL_RESTRICT config_str     = nullptr,
        ::NS(buffer_size_t) const num_elem_by_elem_turns
            = ::NS(buffer_size_t){ 0 } );


    SIXTRL_HOST_FN TrackJobBase* createTrackJob(
        std::string const& SIXTRL_RESTRICT_REF type_str,
        std::string const& SIXTRL_RESTRICT_REF device_id_str = std::string{},
        std::string const& SIXTRL_RESTRICT_REF config_str    = std::string{} );

    SIXTRL_HOST_FN TrackJobBase* createTrackJob(
        std::string const& SIXTRL_RESTRICT_REF type_str,
        Buffer& SIXTRL_RESTRICT_REF particles_buffer,
        Buffer& SIXTRL_RESTRICT_REF beam_elements_buffer,
        std::string const& SIXTRL_RESTRICT_REF device_id_str = std::string{},
        std::string const& SIXTRL_RESTRICT_REF config_str  = std::string{},
        Buffer::size_type const num_elem_by_elem_turns =
            Buffer::size_type{ 0 } );

    SIXTRL_HOST_FN TrackJobBase* createTrackJob(
        std::string const& SIXTRL_RESTRICT_REF type_str,
        Buffer& SIXTRL_RESTRICT_REF particles_buffer,
        Buffer& SIXTRL_RESTRICT_REF beam_elements_buffer,
        Buffer& SIXTRL_RESTRICT_REF output_buffer,
        std::string const& SIXTRL_RESTRICT_REF device_id_str = std::string{},
        std::string const& SIXTRL_RESTRICT_REF config_str    = std::string{},
        Buffer::size_type const num_elem_by_elem_turns =
            Buffer::size_type{ 0 } );
}

#endif /* defined( __cplusplus ) */

struct NS(ElemByElemConfig);

SIXTRL_HOST_FN NS(TrackJobBase)* NS(TrackJob_new)(
    const char *const SIXTRL_RESTRICT type_str,
    const char *const SIXTRL_RESTRICT device_id_str,
    NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
    NS(Buffer)* SIXTRL_RESTRICT beam_elements_buffer,
    NS(Buffer)* SIXTRL_RESTRICT output_buffer,
    const char *const SIXTRL_RESTRICT config_str );

SIXTRL_HOST_FN void NS(TrackJob_track_until_turn)(
    NS(TrackJobBase)* SIXTRL_RESTRICT job,
    NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
    NS(Buffer)* SIXTRL_RESTRICT beam_elements_buffer,
    NS(buffer_size_t) const until_turn );

SIXTRL_HOST_FN void NS(TrackJob_track_elem_by_elem)(
    NS(TrackJobBase)* SIXTRL_RESTRICT job,
    NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
    NS(Buffer)* SIXTRL_RESTRICT beam_elements_buffer,
    NS(buffer_size_t) const num_elem_by_elem_turns );

SIXTRL_HOST_FN void NS(TrackJob_delete)(
    NS(TrackJobBase)* SIXTRL_RESTRICT job );

SIXTRL_HOST_FN NS(Buffer)* NS(TrackJob_get_output_buffer)(
    const NS(TrackJobBase) *const SIXTRL_RESTRICT job );

SIXTRL_HOST_FN struct NS(ElemByElemConfig)*
NS(TrackJob_get_elem_by_elem_config)(
    const NS(TrackJobBase) *const SIXTRL_RESTRICT job );

#endif /* !defined( _GPUCODE ) */

#endif /* SIXTRACKLIB_SIXTRACKLIB_COMMON_TRACK_JOB_H__ */

/* end: sixtracklib/common/track_job.h */
