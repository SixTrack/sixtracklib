#ifndef SIXTRACKLIB_SIXTRACKLIB_COMMON_INTERNAL_TRACK_JOB_CPU_H__
#define SIXTRACKLIB_SIXTRACKLIB_COMMON_INTERNAL_TRACK_JOB_CPU_H__

#if !defined( SIXTRL_NO_INCLUDES )
    #if defined( __cplusplus )
        #include "sixtracklib/common/buffer.hpp"
        #include "sixtracklib/common/particles.hpp"
    #endif /* defined( __cplusplus ) */

    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/buffer.h"
    #include "sixtracklib/common/particles.h"
    #include "sixtracklib/common/internal/track_job_base.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus )

namespace SIXTRL_CXX_NAMESPACE
{
    class TrackJobCpu : public TrackJobBase
    {
        private:

        using _base_t               = TrackJobBase;

        public:

        using context_t             = _base_t::context_t;
        using c_buffer_t            = _base_t::c_buffer_t;
        using buffer_t              = _base_t::buffer_t;
        using size_type             = _base_t::size_type;
        using elem_by_elem_config_t = _base_t::elem_by_elem_config_t;

        TrackJobCpu( buffer_t& SIXTRL_RESTRICT_REF particles_buffer,
                     buffer_t& SIXTRL_RESTRICT_REF beam_elements_buffer,
                     size_type const until_turn              = size_type{ 0 },
                     size_type const num_elem_by_elem_turns  = size_type{ 0 },
                     size_type const out_buffer_index_offset = size_type{ 0 } );

        TrackJobCpu( buffer_t& SIXTRL_RESTRICT_REF particles_buffer,
                     buffer_t& SIXTRL_RESTRICT_REF beam_elements_buffer,
                     buffer_t& SIXTRL_RESTRICT_REF output_buffer,
                     size_type const until_turn              = size_type{ 0 },
                     size_type const num_elem_by_elem_turns  = size_type{ 0 },
                     size_type const out_buffer_index_offset = size_type{ 0 } );

        TrackJobCpu( c_buffer_t* SIXTRL_RESTRICT particles_buffer,
                     c_buffer_t* SIXTRL_RESTRICT beam_elements_buffer,
                     size_type const until_turn              = size_type{ 0 },
                     size_type const num_elem_by_elem_turns  = size_type{ 0 },
                     size_type const out_buffer_index_offset = size_type{ 0 } );

        TrackJobCpu( c_buffer_t* SIXTRL_RESTRICT particles_buffer,
                     c_buffer_t* SIXTRL_RESTRICT beam_elements_buffer,
                     c_buffer_t* SIXTRL_RESTRICT output_buffer,
                     size_type const until_turn              = size_type{ 0 },
                     size_type const num_elem_by_elem_turns  = size_type{ 0 },
                     size_type const out_buffer_index_offset = size_type{ 0 } );

        TrackJobCpu( TrackJobCpu const& other ) = default;
        TrackJobCpu( TrackJobCpu&& other ) = default;

        TrackJobCpu& operator=( TrackJobCpu const& rhs ) = default;
        TrackJobCpu& operator=( TrackJobCpu&& rhs ) = default;

        virtual ~TrackJobCpu() = default;

        protected:

        virtual track_status_t doTrack( size_type const until_turn,
            c_buffer_t* SIXTRL_RESTRICT output_buffer ) override;

        virtual track_status_t doTrackElemByElem(
            size_type const elem_by_elem_turns,
            elem_by_elem_config_t const* SIXTRL_RESTRICT elem_by_elem_config,
            c_buffer_t* SIXTRL_RESTRICT output_buffer ) override;

        private:

        track_status_t doTrackPrivImpl( size_type const until_turn,
            c_buffer_t* SIXTRL_RESTRICT output_buffer );

        track_status_t doTrackElemByElemPrivImpl(
            size_type const elem_by_elem_turns,
            elem_by_elem_config_t const* SIXTRL_RESTRICT elem_by_elem_config,
            c_buffer_t* SIXTRL_RESTRICT output_buffer );

    };
}

typedef SIXTRL_CXX_NAMESPACE::TrackJobCpu   NS(TrackJobCpu);

#else /* defined( __cplusplus ) */

typedef void NS(TrackJobCpu);

#endif /* defined( __cplusplus ) */

#if defined( __cplusplus ) && !defined( _GPUCODE )
extern "C" {
#endif /* defined( __cplusplus ) && !defined( _GPUCODE ) */

SIXTRL_EXTERN SIXTRL_HOST_FN NS(TrackJobCpu)* NS(TrackJobCpu_new)(
    NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
    NS(Buffer)* SIXTRL_RESTRICT beam_elements_buffer,
    NS(Buffer)* SIXTRL_RESTRICT output_buffer );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(TrackJobCpu)* NS(TrackJobCpu_new_elem_by_elem)(
    NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
    NS(Buffer)* SIXTRL_RESTRICT beam_elements_buffer,
    NS(Buffer)* SIXTRL_RESTRICT output_buffer,
    NS(particle_index_t) const num_elem_by_elem_turns );

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(TrackJobCpu_delete)(
    NS(TrackJobCpu)* SIXTRL_RESTRICT track_job );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(Buffer)* NS(TrackJobCpu_track_until_turn)(
    NS(TrackJobCpu)* SIXTRL_RESTRICT track_job,
    NS(particle_index_t) const until_turn,
    NS(Buffer)* SIXTRL_RESTRICT output_buffer );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(Buffer)* NS(TrackJobCpu_track_elem_by_elem)(
    NS(TrackJobCpu)* SIXTRL_RESTRICT track_job,
    NS(particle_index_t) const num_elem_by_elem_turns,
    NS(Buffer)* SIXTRL_RESTRICT output_buffer,
    NS(buffer_size_t) const out_buffer_index_offset );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(Buffer)* NS(TrackJobCpu_track)(
    NS(TrackJobCpu)* SIXTRL_RESTRICT track_job,
    NS(particle_index_t) const until_turn,
    NS(particle_index_t) const num_elem_by_elem_turns,
    NS(Buffer)* SIXTRL_RESTRICT output_buffer,
    NS(buffer_size_t) const out_buffer_index_offset );

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(TrackJobCpu_collect)(
    NS(TrackJobCpu)* SIXTRL_RESTRICT track_job );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(Buffer)*
NS(TrackJobCpu_get_particle_buffer)(
    NS(TrackJobCpu)* SIXTRL_RESTRICT track_job );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(Buffer)*
NS(TrackJobCpu_get_beam_elemenets_buffer)(
    NS(TrackJobCpu)* SIXTRL_RESTRICT track_job );

#if defined( __cplusplus ) && !defined( _GPUCODE )
}
#endif /* defined( __cplusplus ) && !defined( _GPUCODE ) */

#endif /* SIXTRACKLIB_SIXTRACKLIB_COMMON_INTERNAL_TRACK_JOB_CPU_H__ */

/* end: sixtracklib/common/internal/track_job_cpu.h */
