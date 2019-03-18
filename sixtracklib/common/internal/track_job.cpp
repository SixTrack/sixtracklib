#include "sixtracklib/common/track_job.h"

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/buffer.h"
    #include "sixtracklib/common/internal/track_job_base.h"
    #include "sixtracklib/common/internal/track_job_cpu.h"
    #include "sixtracklib/common/output/elem_by_elem_config.h"
    #include "sixtracklib/common/generated/modules.h"

    #if defined( SIXTRACKLIB_ENABLE_MODULE_OPENCL ) && \
        SIXTRACKLIB_ENABLE_MODULE_OPENCL == 1

    #include "sixtracklib/opencl/track_job.h"

    #endif /* OpenCL 1.x */

#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( _GPUCODE )

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <cstddef>
    #include <cstdint>
    #include <cstdlib>
    #include <cstring>
    #include <string>
    #include <algorithm>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/buffer.hpp"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

namespace SIXTRL_CXX_NAMESPACE
{
    SIXTRL_HOST_FN TrackJobBase* createTrackJob(
        const char *const SIXTRL_RESTRICT type_str,
        const char *const SIXTRL_RESTRICT device_id_str,
        const char *const SIXTRL_RESTRICT config_str )
    {
        using size_t = Buffer::size_type;

        std::string const _type_str =
            ( ( type_str != nullptr ) &&
              ( std::strlen( type_str ) > size_t{ 0 } ) )
            ? std::string{ type_str } : std::string{};

        std::string const _device_id =
            ( ( device_id_str != nullptr ) &&
              ( std::strlen( device_id_str ) > size_t{ 0 } ) )
                ? std::string{ device_id_str } : std::string{};

        std::string const _config_str =
            ( ( config_str != nullptr ) &&
              ( std::strlen( config_str ) > size_t{ 0 } ) )
                ? std::string{ config_str } : std::string{};

        return createTrackJob( _type_str, _device_id, _config_str );
    }

    SIXTRL_HOST_FN TrackJobBase* createTrackJob(
        const char *const SIXTRL_RESTRICT type_str,
        ::NS(Buffer)* particles_buffer,
        ::NS(Buffer)* beam_elements_buffer,
        const char *const SIXTRL_RESTRICT device_id_str,
        const char *const SIXTRL_RESTRICT config_str,
        ::NS(buffer_size_t) const num_elem_by_elem_turns )
    {
        TrackJobBase* ptr_job = createTrackJob(
            type_str, device_id_str, config_str );

        if( ptr_job != nullptr )
        {
            ptr_job->reset( particles_buffer, beam_elements_buffer, nullptr,
                            num_elem_by_elem_turns );
        }

        return ptr_job;
    }

    SIXTRL_HOST_FN TrackJobBase* createTrackJob(
        const char *const SIXTRL_RESTRICT type_str,
        ::NS(Buffer)* particles_buffer,
        ::NS(Buffer)* beam_elements_buffer,
        ::NS(Buffer)* output_buffer,
        const char *const SIXTRL_RESTRICT device_id_str,
        const char *const SIXTRL_RESTRICT config_str,
        ::NS(buffer_size_t) const num_elem_by_elem_turns )
    {
        TrackJobBase* ptr_job = createTrackJob(
            type_str, device_id_str, config_str );

        if( ptr_job != nullptr )
        {
            ptr_job->reset( particles_buffer, beam_elements_buffer,
                            output_buffer, num_elem_by_elem_turns );
        }

        return ptr_job;
    }


    SIXTRL_HOST_FN TrackJobBase* createTrackJob(
        std::string const& SIXTRL_RESTRICT_REF type_str,
        std::string const& SIXTRL_RESTRICT_REF device_id_str,
        std::string const& SIXTRL_RESTRICT_REF config_str )
    {
        TrackJobBase* ptr_job = nullptr;

        if( 0 == type_str.compare( SIXTRL_CXX_NAMESPACE::TRACK_JOB_CPU_STR ) )
        {
            ptr_job = new SIXTRL_CXX_NAMESPACE::TrackJobCpu( config_str );
        }
        #if defined( SIXTRACKLIB_ENABLE_MODULE_OPENCL ) && \
            SIXTRACKLIB_ENABLE_MODULE_OPENCL == 1
        else if( 0 == type_str.compare( "opencl" ) )
        {
            ptr_job = new SIXTRL_CXX_NAMESPACE::TrackJobOcl(
                device_id, config_str );
        }
        #endif /* OpenCL 1.x Module */

        return ptr_job;
    }

    SIXTRL_HOST_FN TrackJobBase* createTrackJob(
        std::string const& SIXTRL_RESTRICT type_str,
        Buffer& particles_buffer,
        Buffer& beam_elements_buffer,
        std::string const& SIXTRL_RESTRICT device_id_str,
        std::string const& SIXTRL_RESTRICT config_str,
        Buffer::size_type const num_elem_by_elem_turns )
    {
        TrackJobBase* ptr_job = createTrackJob( type_str,
            device_id_str, config_str );

        if( ptr_job != nullptr )
        {
            ptr_job->reset( particles_buffer, beam_elements_buffer,
                            num_elem_by_elem_turns );
        }

        return ptr_job;
    }

    SIXTRL_HOST_FN TrackJobBase* createTrackJob(
        std::string const& SIXTRL_RESTRICT type_str,
        Buffer& particles_buffer,
        Buffer& beam_elements_buffer,
        Buffer& output_buffer,
        std::string const& SIXTRL_RESTRICT device_id_str,
        std::string const& SIXTRL_RESTRICT config_str,
        Buffer::size_type const num_elem_by_elem_turns )
    {
        TrackJobBase* ptr_job = createTrackJob( type_str,
            device_id_str, config_str );

        if( ptr_job != nullptr )
        {
            ptr_job->reset( particles_buffer, beam_elements_buffer,
                            output_buffer, num_elem_by_elem_turns );
        }

        return ptr_job;
    }
}

SIXTRL_HOST_FN ::NS(TrackJobBase)* NS(TrackJob_new)(
    const char *const SIXTRL_RESTRICT type_str,
    const char *const SIXTRL_RESTRICT device_id_str,
    ::NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
    ::NS(Buffer)* SIXTRL_RESTRICT beam_elements_buffer,
    ::NS(Buffer)* SIXTRL_RESTRICT output_buffer,
    const char *const SIXTRL_RESTRICT config_str )
{
    return SIXTRL_CXX_NAMESPACE::createTrackJob(
        type_str, device_id_str, particles_buffer,
            beam_elements_buffer, output_buffer, config_str );
}

SIXTRL_HOST_FN void NS(TrackJob_track_until_turn)(
    ::NS(TrackJobBase)* SIXTRL_RESTRICT job,
    ::NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
    ::NS(Buffer)* SIXTRL_RESTRICT beam_elements_buffer,
    ::NS(buffer_size_t) const until_turn )
{
    if( job != nullptr )
    {
        job->track( particles_buffer, beam_elements_buffer, until_turn );
    }

    return;
}

SIXTRL_HOST_FN void NS(TrackJob_track_elem_by_elem)(
    ::NS(TrackJobBase)* SIXTRL_RESTRICT job,
    ::NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
    ::NS(Buffer)* SIXTRL_RESTRICT beam_elements_buffer,
    ::NS(buffer_size_t) const num_elem_by_elem_turns )
{
    if( job != nullptr )
    {
        job->trackElemByElem(
            particles_buffer, beam_elements_buffer,
            num_elem_by_elem_turns );
    }

    return;
}

SIXTRL_HOST_FN void NS(TrackJob_delete)(
    ::NS(TrackJobBase)* SIXTRL_RESTRICT job )
{
    if( job != nullptr ) delete job;
    return;
}

SIXTRL_HOST_FN ::NS(Buffer)* NS(TrackJob_get_output_buffer)(
    const ::NS(TrackJobBase) *const SIXTRL_RESTRICT job )
{
    return ( job != nullptr )
        ? job->ptrOutputBuffer() : nullptr;
}

SIXTRL_HOST_FN ::NS(ElemByElemConfig)* ::NS(TrackJob_get_elem_by_elem_config)(
    const ::NS(TrackJobBase) *const SIXTRL_RESTRICT job )
{
    if( job != nullptr )
    {
        return job->elemByElemConfig();
    }
    else
    {
        ::NS(ElemByElemConfig) dummy;
        ::NS(ElemByElemConfig_preset)( &dummy );

        return dummy;
    }
}

/* end: sixtracklib/common/internal/track_job.cpp */
