#ifndef SIXTRACKLIB_COMMMON_TRACK_TRACK_JOB_CPU_CXX_HPP__
#define SIXTRACKLIB_COMMMON_TRACK_TRACK_JOB_CPU_CXX_HPP__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #if defined( __cplusplus )
        #include <algorithm>
        #include <cstddef>
        #include <cstdint>
        #include <cstdlib>
        #include <memory>
        #include <string>
        #include <vector>
    #else /* !defined( __cplusplus ) */
        #include <stdbool.h>
        #include <stddef.h>
        #include <stdint.h>
        #include <stdlib.h>
    #endif /* !defined( __cplusplus ) */
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/control/definitions.h"
    #include "sixtracklib/common/track/definitions.h"

    #if defined( __cplusplus )
        #include "sixtracklib/common/buffer.hpp"
        #include "sixtracklib/common/track/track_job_base.hpp"
    #endif /* defined( __cplusplus ) */

#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus ) && !defined( _GPUCODE )

namespace SIXTRL_CXX_NAMESPACE
{
    class CpuTrackJob : public SIXTRL_CXX_NAMESPACE::TrackJobBaseNew
    {
        public:

        SIXTRL_HOST_FN explicit CpuTrackJob(
            const char *const SIXTRL_RESTRICT config_str = nullptr );

        SIXTRL_HOST_FN explicit CpuTrackJob(
            std::string const& SIXTRL_RESTRICT_REF config_str );

        SIXTRL_HOST_FN CpuTrackJob(
            c_buffer_t* SIXTRL_RESTRICT particles_buffer,
            c_buffer_t* SIXTRL_RESTRICT beam_elements_buffer,
            c_buffer_t* SIXTRL_RESTRICT ptr_output_buffer = nullptr,
            size_type const until_turn_elem_by_elem = size_type{ 0 },
            const char *const SIXTRL_RESTRICT config_str = nullptr );

        SIXTRL_HOST_FN CpuTrackJob(
            buffer_t& SIXTRL_RESTRICT_REF particles_buffer,
            buffer_t& SIXTRL_RESTRICT_REF beam_elements_buffer,
            buffer_t* SIXTRL_RESTRICT ptr_output_buffer = nullptr,
            size_type const until_turn_elem_by_elem  = size_type{ 0 },
            std::string const& config_str = std::string{} );

        SIXTRL_HOST_FN CpuTrackJob(
            buffer_t& SIXTRL_RESTRICT_REF particles_buffer,
            size_type const particle_set_index,
            buffer_t& SIXTRL_RESTRICT_REF beam_elements_buffer,
            buffer_t* SIXTRL_RESTRICT ptr_output_buffer = nullptr,
            size_type const until_turn_elem_by_elem  = size_type{ 0 },
            std::string const& config_str = std::string{} );

        SIXTRL_HOST_FN CpuTrackJob(
            c_buffer_t* SIXTRL_RESTRICT particles_buffer,
            size_type const particle_set_index,
            c_buffer_t* SIXTRL_RESTRICT beam_elements_buffer,
            c_buffer_t* SIXTRL_RESTRICT ptr_output_buffer = nullptr,
            size_type const until_turn_elem_by_elem  = size_type{ 0 },
            char const* SIXTRL_RESTRICT config_str = nullptr );

        SIXTRL_HOST_FN CpuTrackJob(
            buffer_t& SIXTRL_RESTRICT_REF particles_buffer,
            size_type const num_particle_sets,
            size_type const* SIXTRL_RESTRICT particle_set_indices_begin,
            buffer_t& SIXTRL_RESTRICT_REF beam_elements_buffer,
            buffer_t* SIXTRL_RESTRICT ptr_output_buffer = nullptr,
            size_type const until_turn_elem_by_elem  = size_type{ 0 },
            std::string const& config_str = std::string{} );

        SIXTRL_HOST_FN CpuTrackJob(
            c_buffer_t* SIXTRL_RESTRICT particles_buffer,
            size_type const num_particle_sets,
            size_type const* SIXTRL_RESTRICT particle_set_indices_begin,
            c_buffer_t* SIXTRL_RESTRICT beam_elements_buffer,
            c_buffer_t* SIXTRL_RESTRICT ptr_output_buffer = nullptr,
            size_type const until_turn_elem_by_elem  = size_type{ 0 },
            char const* SIXTRL_RESTRICT config_str = nullptr );

        template< typename PartSetIndexIter >
        SIXTRL_HOST_FN CpuTrackJob(
            buffer_t& SIXTRL_RESTRICT_REF particles_buffer,
            PartSetIndexIter particle_set_indices_begin,
            PartSetIndexIter particle_set_indices_end,
            buffer_t& SIXTRL_RESTRICT_REF beam_elements_buffer,
            buffer_t* SIXTRL_RESTRICT ptr_output_buffer = nullptr,
            size_type const until_turn_elem_by_elem  = size_type{ 0 },
            std::string const& config_str = std::string{} );

        template< typename PartSetIndexIter >
        SIXTRL_HOST_FN CpuTrackJob(
            c_buffer_t* SIXTRL_RESTRICT particles_buffer,
            PartSetIndexIter particle_set_indices_begin,
            PartSetIndexIter particle_set_indices_end,
            c_buffer_t* SIXTRL_RESTRICT beam_elements_buffer,
            c_buffer_t* SIXTRL_RESTRICT ptr_output_buffer = nullptr,
            size_type const until_turn_elem_by_elem  = size_type{ 0 },
            char const* SIXTRL_RESTRICT config_str = nullptr );

        SIXTRL_HOST_FN CpuTrackJob( CpuTrackJob const& other ) = default;

        SIXTRL_HOST_FN CpuTrackJob&
        operator=( CpuTrackJob const& rhs ) = default;

        SIXTRL_HOST_FN CpuTrackJob( CpuTrackJob&& other ) = default;
        SIXTRL_HOST_FN CpuTrackJob& operator=( CpuTrackJob&& rhs ) = default;

        SIXTRL_HOST_FN virtual ~CpuTrackJob() = default;

        protected:

        SIXTRL_HOST_FN virtual status_t doFetchParticleAddresses() override;

        SIXTRL_HOST_FN virtual track_status_t doTrackUntilTurn(
            size_type const until_turn ) override;

        SIXTRL_HOST_FN virtual track_status_t doTrackElemByElem(
            size_type const until_turn ) override;

        SIXTRL_HOST_FN virtual track_status_t doTrackLine(
            size_type const line_begin_idx, size_type const line_end_idx,
            bool const finish_turn ) override;
    };

    SIXTRL_STATIC CpuTrackJob::collect_flag_t collect(
        CpuTrackJob& SIXTRL_RESTRICT_REF track_job );

    CpuTrackJob::track_status_t trackUntil(
        CpuTrackJob& SIXTRL_RESTRICT_REF track_job,
        CpuTrackJob::size_type const until_turn );

    CpuTrackJob::track_status_t trackElemByElem(
        CpuTrackJob& SIXTRL_RESTRICT_REF track_job,
        CpuTrackJob::size_type const until_turn );

    CpuTrackJob::track_status_t trackLine(
        CpuTrackJob& SIXTRL_RESTRICT_REF track_job,
        CpuTrackJob::size_type const belem_begin_id,
        CpuTrackJob::size_type const belem_end_id,
        bool const finish_turn = false );
}

#endif /* #if defined( __cplusplus ) && !defined( _GPUCODE ) && \
             !defined( __CUDACC__  ) && !defined( __CUDA_ARCH__ ) */

#if defined( __cplusplus ) && !defined( _GPUCODE )
extern "C" {
#endif /* C++ */

#if defined( __cplusplus ) && !defined( _GPUCODE ) && \
   !defined( __CUDACC__  ) && !defined( __CUDACC__ )

typedef SIXTRL_CXX_NAMESPACE::CpuTrackJob NS(CpuTrackJob);

#else /* C++, Host */

typedef void NS(CpuTrackJob);

#endif /* C++, Host */

#if defined( __cplusplus ) && !defined( _GPUCODE )
}
#endif /* C++ */

/* ************************************************************************* */
/* *******   Implementation of inline and template member functions  ******* */
/* ************************************************************************* */

#if defined( __cplusplus   ) && !defined( _GPUCODE ) && \
   !defined( __CUDA_ARCH__ ) && !defined( __CUDACC__ )

namespace SIXTRL_CXX_NAMESPACE
{
    SIXTRL_INLINE CpuTrackJob::collect_flag_t
    collect( CpuTrackJob& SIXTRL_RESTRICT_REF track_job )
    {
        return track_job.collect();
    }

    template< typename PartSetIndexIter >
    CpuTrackJob::CpuTrackJob(
        CpuTrackJob::buffer_t& SIXTRL_RESTRICT_REF particles_buffer,
        PartSetIndexIter pset_indices_begin, PartSetIndexIter pset_indices_end,
        CpuTrackJob::buffer_t& SIXTRL_RESTRICT_REF beam_elements_buffer,
        CpuTrackJob::buffer_t* SIXTRL_RESTRICT ptr_output_buffer,
        CpuTrackJob::size_type const until_turn_elem_by_elem,
        std::string const& config_str ) : SIXTRL_CXX_NAMESPACE::TrackJobBaseNew(
            SIXTRL_CXX_NAMESPACE::ARCHITECTURE_CPU,
            SIXTRL_ARCHITECTURE_CPU_STR, config_str )
    {
        using _base_track_job_t = SIXTRL_CXX_NAMESPACE::TrackJobBaseNew;

        CpuTrackJob::status_t status = this->doSetParticleSetIndices(
            pset_indices_begin, pset_indices_end,
                particles_buffer.getCApiPtr() );

        SIXTRL_ASSERT( status == SIXTRL_CXX_NAMESPACE::ARCH_STATUS_SUCCESS );

        status = _base_track_job_t::doReset( particles_buffer.getCApiPtr(),
            beam_elements_buffer.getCApiPtr(), ( ptr_output_buffer != nullptr )
                ? ptr_output_buffer->getCApiPtr() : nullptr,
            until_turn_elem_by_elem );

        if( status == SIXTRL_CXX_NAMESPACE::ARCH_STATUS_SUCCESS )
        {
            this->doSetCxxBufferPointers(
                particles_buffer, beam_elements_buffer, ptr_output_buffer );
        }
    }

    template< typename PartSetIndexIter >
    CpuTrackJob::CpuTrackJob(
        c_buffer_t* SIXTRL_RESTRICT particles_buffer,
        PartSetIndexIter pset_indices_begin, PartSetIndexIter pset_indices_end,
        c_buffer_t* SIXTRL_RESTRICT beam_elements_buffer,
        c_buffer_t* SIXTRL_RESTRICT ptr_output_buffer,
        size_type const until_turn_elem_by_elem,
        char const* SIXTRL_RESTRICT config_str ) :
            SIXTRL_CXX_NAMESPACE::TrackJobBaseNew(
                SIXTRL_CXX_NAMESPACE::ARCHITECTURE_CPU,
                SIXTRL_ARCHITECTURE_CPU_STR, config_str )
    {
        using _base_track_job_t = SIXTRL_CXX_NAMESPACE::TrackJobBaseNew;

        CpuTrackJob::status_t status = this->doSetParticleSetIndices(
            pset_indices_begin, pset_indices_end, particles_buffer );

        SIXTRL_ASSERT( status == SIXTRL_CXX_NAMESPACE::ARCH_STATUS_SUCCESS );

        status = _base_track_job_t::doReset( particles_buffer,
            beam_elements_buffer, ptr_output_buffer, until_turn_elem_by_elem );

        SIXTRL_ASSERT( status == SIXTRL_CXX_NAMESPACE::ARCH_STATUS_SUCCESS );
        ( void )status;
    }
}

#endif /* #if defined( __cplusplus ) && !defined( _GPUCODE ) && \
             !defined( __CUDACC__  ) && !defined( __CUDA_ARCH__ ) */

#endif /* SIXTRACKLIB_COMMMON_TRACK_TRACK_JOB_CPU_CXX_HPP__ */
/* end: sixtracklib/sixtracklib/common/track/track_job_cpu.hpp */
