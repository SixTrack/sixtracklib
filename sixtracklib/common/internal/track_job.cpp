#include "sixtracklib/common/track_job.h"

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #if defined( __cplusplus )
        #include <algorithm>
        #include <cstddef>
        #include <cstdint>
        #include <cstdlib>
        #include <memory>
        #include <string>
        #include <vector>
    #endif /* defined( __cplusplus ) */
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/generated/modules.h"
    #include "sixtracklib/common/buffer.h"
    #include "sixtracklib/common/particles.h"
    #include "sixtracklib/common/beam_elements.h"
    #include "sixtracklib/common/be_monitor/be_monitor.h"
    #include "sixtracklib/common/context/context_abs_base.h"
    #include "sixtracklib/common/output/elem_by_elem_config.h"
    #include "sixtracklib/common/output/output_buffer.h"
    #include "sixtracklib/common/internal/track_job_base.h"
    #include "sixtracklib/common/track_job_cpu.h"

    #if defined( SIXTRACKLIB_ENABLE_MODULE_OPENCL ) && \
        ( SIXTRACKLIB_ENABLE_MODULE_OPENCL == 1 )

    #include "sixtracklib/opencl/track_job_cl.h"

    #endif /* OPENCL */

#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( _GPUCODE )
#if defined( __cplusplus )

namespace SIXTRL_CXX_NAMESPACE
{
    SIXTRL_INLINE SIXTRL_HOST_FN TrackJobBase* TrackJob_create(
        const char *const SIXTRL_RESTRICT arch_str,
        const char *const SIXTRL_RESTRICT config_str )
    {
        TrackJobBase* ptr_job = nullptr;

        std::string const sanitized_str =
            TrackJob_sanitize_arch_str( arch_str );

        if( !sanitized_str.empty() )
        {
            #if defined( SIXTRACKLIB_ENABLE_MODULE_OPENCL ) && \
                ( SIXTRACKLIB_ENABLE_MODULE_OPENCL == 1 )
            if( 0 == sanitized_str.compare( TRACK_JOB_CL_STR ) )
            {
                using track_job_t = SIXTRL_CXX_NAMESPACE::TrackJobCl;

                std::string const device_id_str =
                    TrackJob_extract_device_id_str( config_str );

                char const* device_id_cstr = ( !device_id_str.empty() )
                    ? device_id_str.c_str() : nullptr;

                ptr_job = new track_job_t( device_id_cstr, config_str );
            }
            else
            #endif /* OpenCL 1.x */
            if( 0 == sanitized_str.compare( TRACK_JOB_CPU_STR ) )
            {
                using track_job_t = SIXTRL_CXX_NAMESPACE::TrackJobCpu;
                ptr_job = new track_job_t( config_str );
            }
        }

        return ptr_job;
    }

    SIXTRL_INLINE SIXTRL_HOST_FN TrackJobBase* TrackJob_new(
        const char *const SIXTRL_RESTRICT arch_str,
        ::NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
        ::NS(Buffer)* SIXTRL_RESTRICT beam_elemements_buffer,
        const char *const SIXTRL_RESTRICT config_str )
    {
        using size_t = ::NS(buffer_size_t);
        size_t const pset_indices[] = { size_t{ 0 } };

        return TrackJob_new( arch_str, particles_buffer, size_t{ 1 },
            &pset_indices[ 0 ], beam_elemements_buffer, nullptr, size_t{ 0 },
                config_str );
    }

    SIXTRL_INLINE SIXTRL_HOST_FN TrackJobBase* TrackJob_new(
        const char *const SIXTRL_RESTRICT arch_str,
        ::NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
        ::NS(Buffer)* SIXTRL_RESTRICT beam_elemements_buffer,
        ::NS(Buffer)* SIXTRL_RESTRICT output_buffer,
        ::NS(buffer_size_t) const dump_elem_by_elem_turns,
        const char *const SIXTRL_RESTRICT config_str )
    {
        using size_t = ::NS(buffer_size_t);
        size_t const pset_indices[] = { size_t{ 0 } };

        return TrackJob_new( arch_str, particles_buffer, size_t{ 1 },
            &pset_indices[ 0 ], beam_elemements_buffer, output_buffer,
                dump_elem_by_elem_turns, config_str );
    }

    SIXTRL_INLINE SIXTRL_HOST_FN TrackJobBase* TrackJob_new(
        const char *const SIXTRL_RESTRICT arch_str,
        ::NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
        ::NS(buffer_size_t) const num_particle_sets,
        ::NS(buffer_size_t) const* SIXTRL_RESTRICT pset_indices_begin,
        ::NS(Buffer)* SIXTRL_RESTRICT belemements_buffer,
        ::NS(Buffer)* SIXTRL_RESTRICT output_buffer,
        ::NS(buffer_size_t) const dump_elem_by_elem_turns,
        const char *const SIXTRL_RESTRICT config_str )
    {
        TrackJobBase* ptr_job = nullptr;

        std::string const sanitized_str =
            TrackJob_sanitize_arch_str( arch_str );

        if( !sanitized_str.empty() )
        {
            #if defined( SIXTRACKLIB_ENABLE_MODULE_OPENCL ) && \
                ( SIXTRACKLIB_ENABLE_MODULE_OPENCL == 1 )
            if( sanitized_str.compare( TRACK_JOB_CL_STR ) == 0 )
            {
                std::string const device_id_str =
                    TrackJob_extract_device_id_str( config_str );

                char const* device_id_cstr = ( !device_id_str.empty() )
                    ? device_id_str.c_str() : nullptr;

                ptr_job = new TrackJobCl( device_id_cstr, particles_buffer,
                    num_particle_sets, pset_indices_begin, belemements_buffer,
                        output_buffer, dump_elem_by_elem_turns, config_str );
            }
            else
            #endif /* OpenCL 1.x */
            if( sanitized_str.compare( TRACK_JOB_CPU_STR ) == 0 )
            {
                ptr_job = new TrackJobCpu( particles_buffer,
                    num_particle_sets, pset_indices_begin, belemements_buffer,
                        output_buffer, dump_elem_by_elem_turns, config_str );
            }
        }

        return ptr_job;
    }


    SIXTRL_INLINE SIXTRL_HOST_FN TrackJobBase* TrackJob_create(
        std::string const& SIXTRL_RESTRICT_REF arch_str,
        std::string const& SIXTRL_RESTRICT_REF config_str )
    {
        TrackJobBase* ptr_job = nullptr;

        std::string const sanitized_str =
            TrackJob_sanitize_arch_str( arch_str.c_str() );

        if( !sanitized_str.empty() )
        {
            #if defined( SIXTRACKLIB_ENABLE_MODULE_OPENCL ) && \
                ( SIXTRACKLIB_ENABLE_MODULE_OPENCL == 1 )
            if( sanitized_str.compare( TRACK_JOB_CL_STR ) == 0 )
            {
                std::string const device_id_str =
                    TrackJob_extract_device_id_str( config_str.c_str() );

                ptr_job = new TrackJobCl( device_id_str, config_str );
            }
            else
            #endif /* OpenCL 1.x */
            if( sanitized_str.compare( TRACK_JOB_CPU_STR ) == 0 )
            {
                ptr_job = new TrackJobCpu( config_str );
            }
        }

        return ptr_job;
    }

    SIXTRL_INLINE SIXTRL_HOST_FN TrackJobBase* TrackJob_new(
        std::string const& SIXTRL_RESTRICT_REF arch_str,
        Buffer& SIXTRL_RESTRICT_REF particles_buffer,
        Buffer& SIXTRL_RESTRICT_REF beam_elemements_buffer,
        std::string const& SIXTRL_RESTRICT_REF config_str )
    {
        using size_t = Buffer::size_type;
        size_t const pset_indices[] = { size_t{ 0 } };

        return TrackJob_new( arch_str, particles_buffer, size_t{ 1 },
            &pset_indices[ 0 ], beam_elemements_buffer, nullptr, size_t{ 0 },
                config_str );
    }

    SIXTRL_INLINE SIXTRL_HOST_FN TrackJobBase* TrackJob_new(
        std::string const& SIXTRL_RESTRICT_REF arch_str,
        Buffer& SIXTRL_RESTRICT_REF particles_buffer,
        Buffer& SIXTRL_RESTRICT_REF beam_elemements_buffer,
        Buffer* SIXTRL_RESTRICT output_buffer,
        Buffer::size_type const dump_elem_by_elem_turns,
        std::string const& SIXTRL_RESTRICT_REF config_str )
    {
        using size_t = Buffer::size_type;
        size_t const pset_indices[] = { size_t{ 0 } };

        return TrackJob_new( arch_str, particles_buffer, size_t{ 1 },
            &pset_indices[ 0 ], beam_elemements_buffer, output_buffer,
                dump_elem_by_elem_turns, config_str );
    }

    SIXTRL_HOST_FN TrackJobBase* TrackJob_new(
        std::string const& SIXTRL_RESTRICT_REF arch_str,
        Buffer& SIXTRL_RESTRICT_REF particles_buffer,
        Buffer::size_type const num_particle_sets,
        Buffer::size_type const* SIXTRL_RESTRICT particle_set_indices_begin,
        Buffer& SIXTRL_RESTRICT_REF beam_elemements_buffer,
        Buffer* SIXTRL_RESTRICT output_buffer,
        Buffer::size_type const dump_elem_by_elem_turns,
        std::string const& SIXTRL_RESTRICT_REF config_str )
    {
        TrackJobBase* ptr_job = nullptr;

        std::string const sanitized_str =
            TrackJob_sanitize_arch_str( arch_str.c_str() );

        Buffer::size_type const* particle_set_indices_end =
            particle_set_indices_begin;

        if( ( particle_set_indices_begin != nullptr ) &&
            ( num_particle_sets > Buffer::size_type{ 0 } ) )
        {
            std::advance( particle_set_indices_end, num_particle_sets );
        }

        if( !sanitized_str.empty() )
        {
            #if defined( SIXTRACKLIB_ENABLE_MODULE_OPENCL ) && \
                ( SIXTRACKLIB_ENABLE_MODULE_OPENCL == 1 )
            if( sanitized_str.compare( TRACK_JOB_CL_STR ) == 0 )
            {
                std::string const device_id_str =
                    TrackJob_extract_device_id_str( config_str.c_str() );

                ptr_job = new TrackJobCl( device_id_str, particles_buffer,
                    particle_set_indices_begin, particle_set_indices_end,
                        beam_elemements_buffer, output_buffer,
                            dump_elem_by_elem_turns, config_str );
            }
            else
            #endif /* OpenCL 1.x */
            if( sanitized_str.compare( TRACK_JOB_CPU_STR ) == 0 )
            {
                ptr_job = new TrackJobCpu( particles_buffer,
                    particle_set_indices_begin, particle_set_indices_end,
                        beam_elemements_buffer, output_buffer,
                            dump_elem_by_elem_turns, config_str );
            }
        }

        return ptr_job;
    }
}


SIXTRL_HOST_FN NS(TrackJobBase)* NS(TrackJob_create)(
    const char *const SIXTRL_RESTRICT arch,
    const char *const SIXTRL_RESTRICT config_str )
{
    return SIXTRL_CXX_NAMESPACE::TrackJob_create( arch, config_str );
}

SIXTRL_HOST_FN NS(TrackJobBase)* NS(TrackJob_new)(
    const char *const SIXTRL_RESTRICT arch,
    NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
    NS(Buffer)* SIXTRL_RESTRICT beam_elem_buffer,
    const char *const SIXTRL_RESTRICT config_str )
{
    return SIXTRL_CXX_NAMESPACE::TrackJob_new(
        arch, particles_buffer, beam_elem_buffer, config_str );
}

SIXTRL_HOST_FN NS(TrackJobBase)* NS(TrackJob_new_with_output)(
    const char *const SIXTRL_RESTRICT arch,
    NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
    NS(Buffer)* SIXTRL_RESTRICT beam_elem_buffer,
    NS(Buffer)* SIXTRL_RESTRICT output_buffer,
    NS(buffer_size_t) const dump_elem_by_elem_turns,
    const char *const SIXTRL_RESTRICT config_str )
{
    return SIXTRL_CXX_NAMESPACE::TrackJob_new(
        arch, particles_buffer, beam_elem_buffer, output_buffer,
            dump_elem_by_elem_turns, config_str );
}

SIXTRL_HOST_FN NS(TrackJobBase)* NS(TrackJob_new_detailed)(
    const char *const SIXTRL_RESTRICT arch,
    NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
    NS(buffer_size_t) const num_particle_sets,
    NS(buffer_size_t) const* SIXTRL_RESTRICT pset_indices_begin,
    NS(Buffer)* SIXTRL_RESTRICT beam_elem_buffer,
    NS(Buffer)* SIXTRL_RESTRICT output_buffer,
    NS(buffer_size_t) const dump_elem_by_elem_turns,
    const char *const SIXTRL_RESTRICT config_str )
{
    return SIXTRL_CXX_NAMESPACE::TrackJob_new( arch, particles_buffer,
        num_particle_sets, pset_indices_begin, beam_elem_buffer, output_buffer,
            dump_elem_by_elem_turns, config_str );
}

/* ------------------------------------------------------------------------- */

SIXTRL_HOST_FN void NS(TrackJob_delete)(
    NS(TrackJobBase)* SIXTRL_RESTRICT job )
{
    if( job != nullptr ) delete job;
}

SIXTRL_HOST_FN NS(track_status_t) NS(TrackJob_track_until)(
    NS(TrackJobBase)* SIXTRL_RESTRICT job,
    NS(buffer_size_t) const until_turn )
{
    return ( job != nullptr )
        ? job->track( until_turn )
        : ::NS(track_status_t){ -1 };
}

SIXTRL_HOST_FN NS(track_status_t) NS(TrackJob_track_elem_by_elem)(
    NS(TrackJobBase)* SIXTRL_RESTRICT job, NS(buffer_size_t) const until_turn )
{
    return ( job != nullptr )
        ? job->trackElemByElem( until_turn )
        : ::NS(track_status_t){ -1 };
}

SIXTRL_HOST_FN void NS(TrackJob_collect)(
    NS(TrackJobBase)* SIXTRL_RESTRICT job )
{
    if( job != nullptr ) job->collect();
    return;
}

SIXTRL_HOST_FN void NS(TrackJob_clear)( NS(TrackJobBase)* SIXTRL_RESTRICT job )
{
    if( job != nullptr ) job->clear();
}

SIXTRL_HOST_FN bool NS(TrackJob_reset)( NS(TrackJobBase)* SIXTRL_RESTRICT job,
    NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
    NS(Buffer)* SIXTRL_RESTRICT beam_elem_buffer,
    NS(Buffer)* SIXTRL_RESTRICT ptr_output_buffer )
{
    return ( job != nullptr )
        ? job->reset( particles_buffer, beam_elem_buffer, ptr_output_buffer )
        : false;
}

SIXTRL_HOST_FN bool NS(TrackJob_reset_with_output)(
    NS(TrackJobBase)* SIXTRL_RESTRICT job,
    NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
    NS(Buffer)* SIXTRL_RESTRICT beam_elem_buffer,
    NS(Buffer)* SIXTRL_RESTRICT output_buffer,
    NS(buffer_size_t) const dump_elem_by_elem_turns )
{
    return ( job != nullptr )
        ? job->reset( particles_buffer, beam_elem_buffer, output_buffer,
                      dump_elem_by_elem_turns )
        : false;
}

SIXTRL_HOST_FN bool NS(TrackJob_reset_detailed)(
    NS(TrackJobBase)* SIXTRL_RESTRICT job,
    NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
    NS(buffer_size_t) const num_particle_sets,
    NS(buffer_size_t) const* SIXTRL_RESTRICT particle_set_indices_begin,
    NS(Buffer)* SIXTRL_RESTRICT beam_elem_buffer,
    NS(Buffer)* SIXTRL_RESTRICT output_buffer,
    NS(buffer_size_t) const dump_elem_by_elem_turns )
{
    return ( job != nullptr ) ? job->reset( particles_buffer, num_particle_sets,
            particle_set_indices_begin, beam_elem_buffer, output_buffer,
                dump_elem_by_elem_turns )
        : false;
}

SIXTRL_HOST_FN bool NS(TrackJob_assign_output_buffer)(
    NS(TrackJobBase)* SIXTRL_RESTRICT job,
    NS(Buffer)* SIXTRL_RESTRICT ptr_output_buffer )
{
    return ( job != nullptr )
        ? job->assignOutputBuffer( ptr_output_buffer ) : false;
}

/* ------------------------------------------------------------------------- */

SIXTRL_HOST_FN NS(track_job_type_t) NS(TrackJob_get_type_id)(
    const NS(TrackJobBase) *const SIXTRL_RESTRICT job )
{
    return ( job != nullptr ) ? job->type() : ::NS(track_job_type_t){ -1 };
}

SIXTRL_HOST_FN char const* NS(TrackJob_get_type_str)(
    const NS(TrackJobBase) *const SIXTRL_RESTRICT job )
{
    return ( job != nullptr ) ? job->typeStr().c_str() : nullptr;
}

SIXTRL_HOST_FN bool NS(TrackJob_has_device_id_str)(
    const NS(TrackJobBase) *const SIXTRL_RESTRICT job )
{
    return ( job != nullptr ) ? job->hasDeviceIdStr() : false;
}

SIXTRL_HOST_FN char const* NS(TrackJob_get_device_id_str)(
    const NS(TrackJobBase) *const SIXTRL_RESTRICT job )
{
    return ( job != nullptr ) ? job->deviceIdStr().c_str() : nullptr;
}

SIXTRL_HOST_FN bool NS(TrackJob_has_config_str)(
    const NS(TrackJobBase) *const SIXTRL_RESTRICT job )
{
    return ( job != nullptr ) ? job->hasConfigStr() : false;
}

SIXTRL_HOST_FN char const* NS(TrackJob_get_config_str)(
    const NS(TrackJobBase) *const SIXTRL_RESTRICT job )
{
    return ( job != nullptr ) ? job->configStr().c_str() : nullptr;
}

/* ------------------------------------------------------------------------- */

SIXTRL_HOST_FN NS(buffer_size_t) NS(TrackJob_get_num_particle_sets)(
    const NS(TrackJobBase) *const SIXTRL_RESTRICT job )
{
    return ( job != nullptr )
        ? job->numParticleSets() : ::NS(buffer_size_t){ 0 };
}

SIXTRL_HOST_FN NS(buffer_size_t) const*
NS(TrackJob_get_particle_set_indices_begin)(
    const NS(TrackJobBase) *const SIXTRL_RESTRICT job )
{
    return ( job != nullptr ) ? job->particleSetIndicesBegin() : nullptr;
}

SIXTRL_HOST_FN NS(buffer_size_t) const*
NS(TrackJob_get_particle_set_indices_end)(
    const NS(TrackJobBase) *const SIXTRL_RESTRICT job )
{
    return ( job != nullptr ) ? job->particleSetIndicesEnd() : nullptr;
}

SIXTRL_HOST_FN NS(buffer_size_t) NS(TrackJob_get_particle_set_index)(
    const NS(TrackJobBase) *const SIXTRL_RESTRICT job,
    NS(buffer_size_t) const n )
{
    SIXTRL_ASSERT( job != nullptr );
    return job->particleSetIndex( n );
}

/* ------------------------------------------------------------------------- */

SIXTRL_HOST_FN NS(particle_index_t) NS(TrackJob_get_min_particle_id)(
    const NS(TrackJobBase) *const SIXTRL_RESTRICT job )
{
    using index_t = ::NS(particle_index_t);
    return ( job != nullptr ) ? job->minParticleId() : index_t{ -1 };
}

SIXTRL_HOST_FN NS(particle_index_t) NS(TrackJob_get_max_particle_id)(
    const NS(TrackJobBase) *const SIXTRL_RESTRICT job )
{
    using index_t = ::NS(particle_index_t);
    return ( job != nullptr ) ? job->maxParticleId() : index_t{ -1 };
}

SIXTRL_HOST_FN NS(particle_index_t) NS(TrackJob_get_min_element_id)(
    const NS(TrackJobBase) *const SIXTRL_RESTRICT job )
{
    using index_t = ::NS(particle_index_t);
    return ( job != nullptr ) ? job->minElementId() : index_t{ -1 };
}

SIXTRL_HOST_FN NS(particle_index_t) NS(TrackJob_get_max_element_id)(
    const NS(TrackJobBase) *const SIXTRL_RESTRICT job )
{
    using index_t = ::NS(particle_index_t);
    return ( job != nullptr ) ? job->maxElementId() : index_t{ -1 };
}

SIXTRL_HOST_FN NS(particle_index_t) NS(TrackJob_get_min_initial_turn_id)(
    const NS(TrackJobBase) *const SIXTRL_RESTRICT job )
{
    using index_t = ::NS(particle_index_t);
    return ( job != nullptr ) ? job->minInitialTurnId() : index_t{ -1 };
}

SIXTRL_HOST_FN NS(particle_index_t) NS(TrackJob_get_max_initial_turn_id)(
    const NS(TrackJobBase) *const SIXTRL_RESTRICT job )
{
    using index_t = ::NS(particle_index_t);
    return ( job != nullptr ) ? job->maxInitialTurnId() : index_t{ -1 };
}

/* ------------------------------------------------------------------------- */

SIXTRL_HOST_FN NS(Buffer)* NS(TrackJob_get_particles_buffer)(
    NS(TrackJobBase)* SIXTRL_RESTRICT job )
{
    return ( job != nullptr ) ? job->ptrCParticlesBuffer() : nullptr;
}

SIXTRL_HOST_FN NS(Buffer) const* NS(TrackJob_get_const_particles_buffer)(
    const NS(TrackJobBase) *const SIXTRL_RESTRICT job )
{
    return ( job != nullptr ) ? job->ptrCParticlesBuffer() : nullptr;
}

SIXTRL_HOST_FN NS(Buffer)*
NS(TrackJob_get_beam_elements_buffer)( NS(TrackJobBase)* SIXTRL_RESTRICT job )
{
    return ( job != nullptr ) ? job->ptrCBeamElementsBuffer() : nullptr;
}

SIXTRL_HOST_FN NS(Buffer) const*
NS(TrackJob_get_const_beam_elements_buffer)(
    const NS(TrackJobBase) *const SIXTRL_RESTRICT job )
{
    return ( job != nullptr ) ? job->ptrCBeamElementsBuffer() : nullptr;
}

/* ------------------------------------------------------------------------- */

SIXTRL_HOST_FN bool NS(TrackJob_has_output_buffer)(
    const NS(TrackJobBase) *const SIXTRL_RESTRICT job )
{
    return ( job != nullptr ) ? job->hasOutputBuffer() : false;
}

SIXTRL_HOST_FN bool NS(TrackJob_owns_output_buffer)(
    const NS(TrackJobBase) *const SIXTRL_RESTRICT job )
{
    return ( job != nullptr ) ? job->ownsOutputBuffer() : false;
}

SIXTRL_HOST_FN bool NS(TrackJob_has_elem_by_elem_output)(
    const NS(TrackJobBase) *const SIXTRL_RESTRICT job )
{
    return ( job != nullptr ) ? job->hasElemByElemOutput() : false;
}

SIXTRL_HOST_FN bool NS(TrackJob_has_beam_monitor_output)(
    const NS(TrackJobBase) *const SIXTRL_RESTRICT job )
{
    return ( job != nullptr ) ? job->hasBeamMonitorOutput() : false;
}

SIXTRL_HOST_FN NS(buffer_size_t)
NS(TrackJob_get_beam_monitor_output_buffer_offset)(
    const NS(TrackJobBase) *const SIXTRL_RESTRICT job )
{
    SIXTRL_ASSERT( job != nullptr );
    return job->beamMonitorsOutputBufferOffset();
}

SIXTRL_HOST_FN NS(buffer_size_t)
NS(TrackJob_get_elem_by_elem_output_buffer_offset)(
    const NS(TrackJobBase) *const SIXTRL_RESTRICT job )
{
    SIXTRL_ASSERT( job != nullptr );
    return job->elemByElemOutputBufferOffset();
}

SIXTRL_HOST_FN NS(buffer_size_t)
NS(TrackJob_get_num_elem_by_elem_turns)(
    const NS(TrackJobBase) *const SIXTRL_RESTRICT job )
{
    return ( job != nullptr )
        ? job->numElemByElemTurns() : ::NS(buffer_size_t){ 0 };
}

SIXTRL_HOST_FN NS(Buffer)* NS(TrackJob_get_output_buffer)(
    NS(TrackJobBase)* SIXTRL_RESTRICT job )
{
    return ( job != nullptr ) ? job->ptrCOutputBuffer() : nullptr;
}

SIXTRL_HOST_FN NS(Buffer) const*
NS(TrackJob_get_const_output_buffer)(
    const NS(TrackJobBase) *const SIXTRL_RESTRICT job )
{
    return ( job != nullptr ) ? job->ptrCOutputBuffer() : nullptr;
}

/* ------------------------------------------------------------------------- */

SIXTRL_HOST_FN bool NS(TrackJob_has_beam_monitors)(
    const NS(TrackJobBase) *const SIXTRL_RESTRICT job )
{
    return ( job != nullptr ) ? job->hasBeamMonitors() : false;
}

SIXTRL_HOST_FN NS(buffer_size_t) NS(TrackJob_get_num_beam_monitors)(
    const NS(TrackJobBase) *const SIXTRL_RESTRICT job )
{
    return ( job != nullptr )
        ? job->numBeamMonitors() : ::NS(buffer_size_t){ 0 };
}

SIXTRL_HOST_FN NS(buffer_size_t) const*
NS(TrackJob_get_beam_monitor_indices_begin)(
    const NS(TrackJobBase) *const SIXTRL_RESTRICT job )
{
    return ( job != nullptr ) ? job->beamMonitorIndicesBegin() : nullptr;
}

SIXTRL_HOST_FN NS(buffer_size_t) const*
NS(TrackJob_get_beam_monitor_indices_end)(
    const NS(TrackJobBase) *const SIXTRL_RESTRICT job )
{
    return ( job != nullptr ) ? job->beamMonitorIndicesEnd() : nullptr;
}

SIXTRL_HOST_FN NS(buffer_size_t) NS(TrackJob_get_beam_monitor_index)(
    const NS(TrackJobBase) *const SIXTRL_RESTRICT job,
    NS(buffer_size_t) const n )
{
    return ( job != nullptr )
        ? job->beamMonitorIndex( n ) : ::NS(buffer_size_t){ 0 };
}

/* ------------------------------------------------------------------------- */

SIXTRL_HOST_FN bool NS(TrackJob_has_elem_by_elem_config)(
    const NS(TrackJobBase) *const SIXTRL_RESTRICT job )
{
    return ( job != nullptr ) ? job->hasElemByElemConfig() : false;
}

SIXTRL_HOST_FN NS(ElemByElemConfig) const*
NS(TrackJob_get_elem_by_elem_config)(
    const NS(TrackJobBase) *const SIXTRL_RESTRICT job )
{
    return ( job != nullptr ) ? job->ptrElemByElemConfig() : nullptr;
}

SIXTRL_HOST_FN bool NS(TrackJob_is_elem_by_elem_config_rolling)(
    const NS(TrackJobBase) *const SIXTRL_RESTRICT job )
{
    return ( job != nullptr ) ? job->elemByElemRolling() : false;
}

SIXTRL_HOST_FN bool
NS(TrackJob_get_default_elem_by_elem_config_rolling_flag)(
    const NS(TrackJobBase) *const SIXTRL_RESTRICT job )
{
    return ( job != nullptr ) ? job->defaultElemByElemRolling() : false;
}

SIXTRL_HOST_FN void
NS(TrackJob_set_default_elem_by_elem_config_rolling_flag)(
    NS(TrackJobBase)* SIXTRL_RESTRICT job, bool const is_rolling_flag )
{
    if( job != nullptr )
    {
        job->setDefaultElemByElemRolling( is_rolling_flag );
    }

    return;
}

SIXTRL_HOST_FN NS(elem_by_elem_order_t)
NS(TrackJob_get_elem_by_elem_config_order)(
    const NS(TrackJobBase) *const SIXTRL_RESTRICT job )
{
    return ( job != nullptr )
        ? job->elemByElemOrder() : ::NS(ELEM_BY_ELEM_ORDER_INVALID);
}

SIXTRL_HOST_FN NS(elem_by_elem_order_t)
NS(TrackJob_get_default_elem_by_elem_config_order)(
    const NS(TrackJobBase) *const SIXTRL_RESTRICT job )
{
    return ( job != nullptr )
        ? job->defaultElemByElemOrder()
        : ::NS(ELEM_BY_ELEM_ORDER_INVALID);
}

SIXTRL_HOST_FN void
NS(TrackJob_set_default_elem_by_elem_config_order)(
    NS(TrackJobBase)* SIXTRL_RESTRICT job,
    NS(elem_by_elem_order_t) const order )
{
    if( job != nullptr )
    {
        job->setDefaultElemByElemOrder( order );
    }

    return;
}

#endif /* defined( __cplusplus ) */
#endif /* !defined( _GPUCODE ) */

/* end: sixtracklib/common/internal/track_job.cpp */
