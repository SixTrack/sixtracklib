#ifndef SIXTRL_SIXTRACKLIB_OPENCL_TRACK_CL_JOB_H__
#define SIXTRL_SIXTRACKLIB_OPENCL_TRACK_CL_JOB_H__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #if defined( __cplusplus )
        #include <memory>
    #endif /* defined( __cplusplus ) */
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #if defined( __cplusplus )
        #include "sixtracklib/common/buffer.hpp"
    #endif /* defined( __cplusplus ) */

    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/buffer.h"
    #include "sixtracklib/common/particles.h"
    #include "sixtracklib/common/internal/track_job_base.h"

    #include "sixtracklib/opencl/context.h"
    #include "sixtracklib/opencl/argument.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus )

namespace SIXTRL_CXX_NAMESPACE
{
    class TrackJobCl : public TrackJobBase
    {
        private:

        using _base_t = TrackJobBase;

        public:

        using context_t             = _base_t::context_t;
        using c_buffer_t            = _base_t::c_buffer_t;
        using buffer_t              = _base_t::buffer_t;
        using size_type             = _base_t::size_type;
        using elem_by_elem_config_t = _base_t::elem_by_elem_config_t;

        using cl_arg_t              = SIXTRL_CXX_NAMESPACE::ClArgument;
        using cl_context_t          = SIXTRL_CXX_NAMESPACE::ClContext;

        TrackJobCl( char const* SIXTRL_RESTRICT device_id_str,
                     c_buffer_t* SIXTRL_RESTRICT particles_buffer,
                     c_buffer_t* SIXTRL_RESTRICT beam_elements_buffer,
                     size_type const until_turn              = size_type{ 0 },
                     size_type const num_elem_by_elem_turns  = size_type{ 0 } );

        TrackJobCl( char const* SIXTRL_RESTRICT device_id_str,
                     c_buffer_t* SIXTRL_RESTRICT particles_buffer,
                     c_buffer_t* SIXTRL_RESTRICT beam_elements_buffer,
                     c_buffer_t* SIXTRL_RESTRICT output_buffer,
                     size_type const until_turn              = size_type{ 0 },
                     size_type const num_elem_by_elem_turns  = size_type{ 0 } );

        TrackJobCl( TrackJobCl const& other ) = delete;
        TrackJobCl( TrackJobCl&& other ) = delete;

        TrackJobCl& operator=( TrackJobCl const& rhs ) = delete;
        TrackJobCl& operator=( TrackJobCl&& rhs ) = delete;

        virtual ~TrackJobCl();

        c_buffer_t* track( size_type const until_turn );

        void collect();

        cl_context_t&         context()          SIXTRL_RESTRICT;
        cl_context_t const&   context()    const SIXTRL_RESTRICT;

        NS(ClContext)*        ptrContext()       SIXTRL_RESTRICT;
        NS(ClContext) const*  ptrContext() const SIXTRL_RESTRICT;

        protected:

        virtual bool doInitBuffers(
            c_buffer_t* SIXTRL_RESTRICT particles_buffer,
            c_buffer_t* SIXTRL_RESTRICT belements_buffer,
            c_buffer_t* SIXTRL_RESTRICT output_buffer,
            size_type const num_elem_by_elem_turns,
            size_type const until_turn,
            size_type const* SIXTRL_RESTRICT particle_blkidx_begin,
            size_type const  particle_blk_idx_length,
            size_type* SIXTRL_RESTRICT ptr_elem_by_elem_index_offset,
            size_type* SIXTRL_RESTRICT ptr_beam_monitor_index_offset,
            particle_index_t* SIXTRL_RESTRICT ptr_min_turn_id ) override;

        private:

        using ptr_cl_context_t = std::unique_ptr< cl_context_t >;
        using ptr_cl_arg_t     = std::unique_ptr< cl_arg_t >;

        bool doInitBuffersClImpl(
            c_buffer_t* SIXTRL_RESTRICT particles_buffer,
            c_buffer_t* SIXTRL_RESTRICT belements_buffer,
            c_buffer_t* SIXTRL_RESTRICT output_buffer,
            size_type const num_elem_by_elem_turns,
            size_type const until_turn,
            size_type const* SIXTRL_RESTRICT particle_blkidx_begin,
            size_type const  particle_blk_idx_length,
            size_type* SIXTRL_RESTRICT ptr_elem_by_elem_index_offset,
            size_type* SIXTRL_RESTRICT ptr_beam_monitor_index_offset,
            particle_index_t* SIXTRL_RESTRICT ptr_min_turn_id );

        ptr_cl_context_t m_ptr_context;
        ptr_cl_arg_t     m_ptr_particles_buffer_arg;
        ptr_cl_arg_t     m_ptr_beam_elements_buffer_arg;
        ptr_cl_arg_t     m_ptr_output_buffer_arg;

        size_type        m_elem_by_elem_index_offset;
        size_type        m_beam_monitor_index_offset;
        size_type        m_particle_block_idx;

        bool             m_owns_output_buffer;
    };
}

typedef SIXTRL_CXX_NAMESPACE::TrackJobCl    NS(TrackJobCl);

#else /* defined( __cplusplus ) */

typedef void NS(TrackJobCl);

#endif /* defined( __cplusplus ) */

#if defined( __cplusplus ) && !defined( _GPUCODE )
extern "C" {
#endif /* defined( __cplusplus ) && !defined( _GPUCODE ) */

SIXTRL_EXTERN SIXTRL_HOST_FN NS(TrackJobCl)* NS(TrackJobCl_new)(
    char const* SIXTRL_RESTRICT device_id_str,
    NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
    NS(Buffer)* SIXTRL_RESTRICT beam_elements_buffer,
    NS(buffer_size_t) const num_elem_by_elem_turns,
    NS(buffer_size_t) const until_turn );


SIXTRL_EXTERN SIXTRL_HOST_FN NS(TrackJobCl)*
NS(TrackJobCl_new_using_output_buffer)(
    char const* SIXTRL_RESTRICT device_id_str,
    NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
    NS(Buffer)* SIXTRL_RESTRICT beam_elements_buffer,
    NS(Buffer)* SIXTRL_RESTRICT output_buffer,
    NS(buffer_size_t) const num_elem_by_elem_turns,
    NS(buffer_size_t) const until_turn );

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(TrackJobCl_delete)(
    NS(TrackJobCl)* SIXTRL_RESTRICT track_job );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(Buffer)* NS(TrackJobCl_track)(
    NS(TrackJobCl)* SIXTRL_RESTRICT track_job,
    NS(buffer_size_t) const until_turn );

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(TrackJobCl_collect)(
    NS(TrackJobCl)* SIXTRL_RESTRICT track_job );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(Buffer)* NS(TrackJobCl_get_particle_buffer)(
    NS(TrackJobCl)* SIXTRL_RESTRICT track_job );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(Buffer)* NS(TrackJobCl_get_output_buffer)(
    NS(TrackJobCl)* SIXTRL_RESTRICT track_job );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(Buffer)*
NS(TrackJobCl_get_beam_elements_buffer)(
    NS(TrackJobCl)* SIXTRL_RESTRICT track_job );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(TrackJobCl)* NS(TrackJobCl_new)(
    char const* SIXTRL_RESTRICT device_id_str,
    NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
    NS(Buffer)* SIXTRL_RESTRICT beam_elements_buffer,
    NS(buffer_size_t) const num_elem_by_elem_turns,
    NS(buffer_size_t) const until_turn );


SIXTRL_EXTERN SIXTRL_HOST_FN NS(TrackJobCl)*
NS(TrackJobCl_new_using_output_buffer)(
    char const* SIXTRL_RESTRICT device_id_str,
    NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
    NS(Buffer)* SIXTRL_RESTRICT beam_elements_buffer,
    NS(Buffer)* SIXTRL_RESTRICT output_buffer,
    NS(buffer_size_t) const num_elem_by_elem_turns,
    NS(buffer_size_t) const until_turn );

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(TrackJobCl_delete)(
    NS(TrackJobCl)* SIXTRL_RESTRICT track_job );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(Buffer)* NS(TrackJobCl_track)(
    NS(TrackJobCl)* SIXTRL_RESTRICT track_job,
    NS(buffer_size_t) const until_turn );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(ClContext)*
NS(TrackJobCl_get_context)( NS(TrackJobCl)* SIXTRL_RESTRICT track_job );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(ClContext) const*
NS(TrackJobCl_get_const_context)(
    NS(TrackJobCl)* SIXTRL_RESTRICT track_job );

#if defined( __cplusplus ) && !defined( _GPUCODE )
}
#endif /* defined( __cplusplus ) && !defined( _GPUCODE ) */

#endif /* SIXTRL_SIXTRACKLIB_OPENCL_TRACK_CL_JOB_H__ */

/* end: sixtracklib/opencl/track_job.h */
