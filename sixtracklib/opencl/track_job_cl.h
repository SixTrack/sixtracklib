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
        #include "sixtracklib/common/particles.hpp"
    #endif /* defined( __cplusplus ) */

    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/buffer.h"
    #include "sixtracklib/common/particles.h"
    #include "sixtracklib/common/internal/track_job_base.h"
    #include "sixtracklib/common/track_job.h"

    #include "sixtracklib/opencl/context.h"
    #include "sixtracklib/opencl/argument.h"
    #include "sixtracklib/opencl/make_track_job.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( _GPUCODE )

#if defined( __cplusplus )

namespace SIXTRL_CXX_NAMESPACE
{
    SIXTRL_STATIC_VAR track_job_type_t const
        TRACK_JOB_CL_ID = track_job_type_t{SIXTRL_TRACK_JOB_CL_ID};

    SIXTRL_STATIC_VAR char const
        TRACK_JOB_CL_STR[] = SIXTRL_TRACK_JOB_CL_STR;

    class TrackJobCl : public TrackJobBase
    {
        private:

        using _base_t = TrackJobBase;

        public:

        using context_t             = SIXTRL_CXX_NAMESPACE::ClContext;
        using c_buffer_t            = _base_t::c_buffer_t;
        using buffer_t              = _base_t::buffer_t;
        using size_type             = _base_t::size_type;
        using elem_by_elem_config_t = _base_t::elem_by_elem_config_t;
        using track_status_t        = _base_t::track_status_t;
        using type_t                = _base_t::type_t;
        using output_buffer_flag_t  = _base_t::output_buffer_flag_t;
        using collect_flag_t        = _base_t::collect_flag_t;
        using push_flag_t           = _base_t::push_flag_t;
        using status_t              = _base_t::status_t;

        using cl_arg_t              = SIXTRL_CXX_NAMESPACE::ClArgument;
        using cl_context_t          = SIXTRL_CXX_NAMESPACE::ClContext;
        using cl_buffer_t           = cl_context_t::cl_buffer_t;

        SIXTRL_HOST_FN explicit TrackJobCl(
            std::string const& SIXTRL_RESTRICT_REF dev_id_str = std::string{},
            std::string const& SIXTRL_RESTRICT_REF config_str = std::string{} );

        SIXTRL_HOST_FN explicit TrackJobCl(
            const char *const SIXTRL_RESTRICT device_id_str,
            const char *const SIXTRL_RESTRICT config_str = nullptr );

        SIXTRL_HOST_FN TrackJobCl(
            const char *const SIXTRL_RESTRICT device_id_str,
            c_buffer_t* SIXTRL_RESTRICT particles_buffer,
            c_buffer_t* SIXTRL_RESTRICT beam_elements_buffer,
            c_buffer_t* SIXTRL_RESTRICT ptr_output_buffer = nullptr,
            size_type const until_turn_elem_by_elem = size_type{ 0 },
            const char *const SIXTRL_RESTRICT config_str = nullptr );

        SIXTRL_HOST_FN TrackJobCl(
            const char *const SIXTRL_RESTRICT device_id_str,
            c_buffer_t* SIXTRL_RESTRICT particles_buffer,
            size_type const num_particle_sets,
            size_type const* SIXTRL_RESTRICT particle_set_indices_begin,
            c_buffer_t* SIXTRL_RESTRICT beam_elements_buffer,
            c_buffer_t* SIXTRL_RESTRICT ptr_output_buffer = nullptr,
            size_type const until_turn_elem_by_elem = size_type{ 0 },
            const char *const SIXTRL_RESTRICT config_str = nullptr );

        template< typename PartSetIndexIter >
        SIXTRL_HOST_FN TrackJobCl(
            const char *const SIXTRL_RESTRICT device_id_str,
            c_buffer_t* SIXTRL_RESTRICT particles_buffer,
            PartSetIndexIter particle_set_indices_begin,
            PartSetIndexIter particle_set_indices_end,
            c_buffer_t* SIXTRL_RESTRICT beam_elements_buffer,
            c_buffer_t* SIXTRL_RESTRICT ptr_output_buffer = nullptr,
            size_type const until_turn_elem_by_elem = size_type{ 0 },
            const char *const SIXTRL_RESTRICT config_str = nullptr );

        SIXTRL_HOST_FN TrackJobCl(
            std::string const& SIXTRL_RESTRICT_REF device_id_str,
            buffer_t& SIXTRL_RESTRICT_REF particles_buffer,
            buffer_t& SIXTRL_RESTRICT_REF beam_elements_buffer,
            buffer_t* SIXTRL_RESTRICT ptr_output_buffer = nullptr,
            size_type const until_turn_elem_by_elem  = size_type{ 0 },
            std::string const& config_str = std::string{} );

        template< typename PartSetIndexIter >
        SIXTRL_HOST_FN TrackJobCl(
            std::string const& SIXTRL_RESTRICT_REF device_id_str,
            buffer_t& SIXTRL_RESTRICT_REF particles_buffer,
            PartSetIndexIter particle_set_indices_begin,
            PartSetIndexIter particle_set_indices_end,
            buffer_t& SIXTRL_RESTRICT_REF beam_elements_buffer,
            buffer_t* SIXTRL_RESTRICT ptr_output_buffer = nullptr,
            size_type const until_turn_elem_by_elem = size_type{ 0 },
            std::string SIXTRL_RESTRICT_REF config_str = std::string{} );

        SIXTRL_HOST_FN TrackJobCl( TrackJobCl const& other ) = default;
        SIXTRL_HOST_FN TrackJobCl( TrackJobCl&& other ) = default;

        SIXTRL_HOST_FN TrackJobCl& operator=( TrackJobCl const& rhs ) = default;
        SIXTRL_HOST_FN TrackJobCl& operator=( TrackJobCl&& rhs ) = default;

        SIXTRL_HOST_FN virtual ~TrackJobCl() SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN size_type totalNumParticles() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN cl_context_t& context() SIXTRL_RESTRICT;
        SIXTRL_HOST_FN cl_context_t const& context() const SIXTRL_RESTRICT;
        SIXTRL_HOST_FN NS(ClContext)* ptrContext() SIXTRL_RESTRICT;
        SIXTRL_HOST_FN NS(ClContext) const* ptrContext() const SIXTRL_RESTRICT;

        SIXTRL_HOST_FN cl_arg_t& particlesArg() SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN cl_arg_t const& particlesArg() const SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN cl_arg_t* ptrParticlesArg() SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN cl_arg_t const* ptrParticlesArg() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN cl_arg_t& beamElementsArg() SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN cl_arg_t const& beamElementsArg() const SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN cl_arg_t* ptrBeamElementsArg() SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN cl_arg_t const*
        ptrBeamElementsArg() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN cl_arg_t& outputBufferArg() SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN cl_arg_t const& outputBufferArg() const SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN cl_arg_t* ptrOutputBufferArg() SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN cl_arg_t const* ptrOutputBufferArg() const SIXTRL_NOEXCEPT;


        SIXTRL_HOST_FN cl_buffer_t const&
        clElemByElemConfigBuffer() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN cl_buffer_t& clElemByElemConfigBuffer() SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN cl_buffer_t const*
        ptrClElemByElemConfigBuffer() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN cl_buffer_t*
        ptrClElemByElemConfigBuffer() SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN status_t updateBeamElementsRegion(
            size_type const offset, size_type const length,
            void const* SIXTRL_RESTRICT new_value );

        SIXTRL_HOST_FN status_t updateBeamElementsRegions(
            size_type const num_regions_to_update,
            size_type const* SIXTRL_RESTRICT offsets,
            size_type const* SIXTRL_RESTRICT lengths,
            void const* SIXTRL_RESTRICT const* SIXTRL_RESTRICT new_values );

        protected:

        using ptr_cl_context_t = std::unique_ptr< cl_context_t >;
        using ptr_cl_arg_t     = std::unique_ptr< cl_arg_t >;
        using ptr_cl_buffer_t  = std::unique_ptr< cl_buffer_t >;

        SIXTRL_HOST_FN void doSetTotalNumParticles(
            size_type const total_num_particles ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN virtual bool doPrepareContext(
            char const* SIXTRL_RESTRICT device_id_str,
            const char *const SIXTRL_RESTRICT ptr_config_str );

        SIXTRL_HOST_FN virtual bool doPrepareParticlesStructures(
            c_buffer_t* SIXTRL_RESTRICT ptr_particles_buffer ) override;

        SIXTRL_HOST_FN virtual bool doPrepareBeamElementsStructures(
            c_buffer_t* SIXTRL_RESTRICT ptr_beam_elem_buffer ) override;

        SIXTRL_HOST_FN virtual bool doPrepareOutputStructures(
            c_buffer_t* SIXTRL_RESTRICT particles_buffer,
            c_buffer_t* SIXTRL_RESTRICT beam_elem_buffer,
            c_buffer_t* SIXTRL_RESTRICT ptr_output_buffer,
            size_type const until_turn_elem_by_elem ) override;

        SIXTRL_HOST_FN virtual bool doAssignOutputBufferToBeamMonitors(
            c_buffer_t* SIXTRL_RESTRICT beam_elem_buffer,
            c_buffer_t* SIXTRL_RESTRICT output_buffer,
            particle_index_t const min_turn_id,
            size_type const output_buffer_index_offset ) override;

        SIXTRL_HOST_FN virtual bool doAssignOutputBufferToElemByElemConfig(
            elem_by_elem_config_t* SIXTRL_RESTRICT elem_by_elem_config,
            c_buffer_t* SIXTRL_RESTRICT output_buffer,
            size_type const output_buffer_offset_index ) override;

        SIXTRL_HOST_FN virtual bool doReset(
            c_buffer_t* SIXTRL_RESTRICT particles_buffer,
            c_buffer_t* SIXTRL_RESTRICT beam_elem_buffer,
            c_buffer_t* SIXTRL_RESTRICT ptr_output_buffer,
            size_type const until_turn_elem_by_elem ) override;

        SIXTRL_HOST_FN virtual track_status_t doTrackUntilTurn(
            size_type const until_turn ) override;

        SIXTRL_HOST_FN virtual track_status_t doTrackElemByElem(
            size_type const until_turn ) override;

        SIXTRL_HOST_FN virtual track_status_t doTrackLine(
            size_type const line_begin_idx, size_type const line_end_idx,
            bool const finish_turn ) override;

        SIXTRL_HOST_FN virtual void doCollect(
            collect_flag_t const flags ) override;

        SIXTRL_HOST_FN virtual void doPush( push_flag_t const flags ) override;

        SIXTRL_HOST_FN virtual void doParseConfigStr(
            const char *const SIXTRL_RESTRICT config_str ) override;

        SIXTRL_HOST_FN void doUpdateStoredContext(
            ptr_cl_context_t&& context );

        SIXTRL_HOST_FN void doUpdateStoredParticlesArg(
            ptr_cl_arg_t&& particle_arg );

        SIXTRL_HOST_FN void doUpdateStoredBeamElementsArg(
            ptr_cl_arg_t&& beam_elements_arg );

        SIXTRL_HOST_FN void doUpdateStoredOutputArg(
            ptr_cl_arg_t&& output_arg );

        SIXTRL_HOST_FN void doUpdateStoredClElemByElemConfigBuffer(
            ptr_cl_buffer_t&& cl_elem_by_elem_config_buffer );

        private:

        template< typename PartSetIndexIter >
        SIXTRL_HOST_FN bool doInitTrackJobCl(
            const char *const SIXTRL_RESTRICT device_id_str,
            c_buffer_t* SIXTRL_RESTRICT particles_buffer,
            PartSetIndexIter particle_set_indices_begin,
            PartSetIndexIter particle_set_indices_end,
            c_buffer_t* SIXTRL_RESTRICT beam_elements_buffer,
            c_buffer_t* SIXTRL_RESTRICT ptr_output_buffer,
            size_type const until_turn_elem_by_elem,
            const char *const SIXTRL_RESTRICT config_str );

        SIXTRL_HOST_FN bool doPrepareContextOclImpl(
            const char *const SIXTRL_RESTRICT device_id_str,
            const char *const SIXTRL_RESTRICT ptr_config_str );

        SIXTRL_HOST_FN void doParseConfigStrOclImpl(
            const char *const SIXTRL_RESTRICT config_str );

        SIXTRL_HOST_FN bool doPrepareParticlesStructuresOclImp(
            c_buffer_t* SIXTRL_RESTRICT ptr_particles_buffer );

        SIXTRL_HOST_FN bool doPrepareBeamElementsStructuresOclImp(
            c_buffer_t* SIXTRL_RESTRICT ptr_beam_elem_buffer );

        SIXTRL_HOST_FN bool doPrepareOutputStructuresOclImpl(
            c_buffer_t* SIXTRL_RESTRICT particles_buffer,
            c_buffer_t* SIXTRL_RESTRICT beam_elem_buffer,
            c_buffer_t* SIXTRL_RESTRICT ptr_output_buffer,
            size_type const until_turn_elem_by_elem );

        SIXTRL_HOST_FN bool doAssignOutputBufferToBeamMonitorsOclImp(
            c_buffer_t* SIXTRL_RESTRICT beam_elem_buffer,
            c_buffer_t* SIXTRL_RESTRICT output_buffer,
            particle_index_t const min_turn_id,
            size_type const output_buffer_index_offset );

        SIXTRL_HOST_FN bool doAssignOutputBufferToElemByElemConfigOclImpl(
            elem_by_elem_config_t* SIXTRL_RESTRICT elem_by_elem_config,
            c_buffer_t* SIXTRL_RESTRICT output_buffer,
            size_type const output_buffer_offset_index );

        SIXTRL_HOST_FN bool doResetOclImp(
            c_buffer_t* SIXTRL_RESTRICT particles_buffer,
            c_buffer_t* SIXTRL_RESTRICT beam_elem_buffer,
            c_buffer_t* SIXTRL_RESTRICT ptr_output_buffer,
            size_type const until_turn_elem_by_elem );

        std::vector< size_type > m_num_particles_in_pset;

        ptr_cl_context_t m_ptr_context;
        ptr_cl_arg_t     m_ptr_particles_buffer_arg;
        ptr_cl_arg_t     m_ptr_beam_elements_buffer_arg;
        ptr_cl_arg_t     m_ptr_output_buffer_arg;
        ptr_cl_buffer_t  m_ptr_cl_elem_by_elem_config_buffer;

        size_type        m_total_num_particles;
    };

    SIXTRL_HOST_FN void collect(
        TrackJobCl& SIXTRL_RESTRICT_REF track_job ) SIXTRL_NOEXCEPT;

    SIXTRL_HOST_FN void collect( TrackJobCl& SIXTRL_RESTRICT_REF track_job,
        track_job_collect_flag_t const flags ) SIXTRL_NOEXCEPT;

    SIXTRL_HOST_FN void push( TrackJobCl& SIXTRL_RESTRICT_REF track_job,
        track_job_collect_flag_t const flags ) SIXTRL_NOEXCEPT;

    SIXTRL_HOST_FN TrackJobCl::track_status_t track(
        TrackJobCl& SIXTRL_RESTRICT_REF job,
        TrackJobCl::size_type const until_turn ) SIXTRL_NOEXCEPT;

    SIXTRL_HOST_FN TrackJobCl::track_status_t trackElemByElem(
        TrackJobCl& SIXTRL_RESTRICT_REF job,
        TrackJobCl::size_type const until_turn_elem_by_elem ) SIXTRL_NOEXCEPT;

    SIXTRL_HOST_FN TrackJobCl::track_status_t trackLine(
        TrackJobCl& SIXTRL_RESTRICT_REF job,
        TrackJobCl::size_type const line_begin_idx,
        TrackJobCl::size_type const line_end_idx,
        bool const finish_turn = false ) SIXTRL_NOEXCEPT;
}

typedef SIXTRL_CXX_NAMESPACE::TrackJobCl    NS(TrackJobCl);

SIXTRL_STATIC_VAR NS(track_job_type_t) const NS(TRACK_JOB_CL_ID) =
        SIXTRL_TRACK_JOB_CL_ID;

SIXTRL_STATIC_VAR char const NS(TRACK_JOB_CL_STR)[] =
        SIXTRL_TRACK_JOB_CL_STR;

#else /* defined( __cplusplus ) */

typedef void NS(TrackJobCl);

SIXTRL_STATIC_VAR NS(track_job_type_t) const NS(TRACK_JOB_CL_ID) =
    SIXTRL_TRACK_JOB_CL_ID;

SIXTRL_STATIC_VAR char const NS(TRACK_JOB_CL_STR)[] =
    SIXTRL_TRACK_JOB_CL_STR;

#endif /* defined( __cplusplus ) */
#endif /* defined( _GPUCODE ) */

#if defined( __cplusplus ) && !defined( _GPUCODE )
extern "C" {
#endif /* defined( __cplusplus ) && !defined( _GPUCODE ) */

SIXTRL_EXTERN SIXTRL_HOST_FN NS(TrackJobCl)* NS(TrackJobCl_create)(
    const char *const SIXTRL_RESTRICT device_id_str );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(TrackJobCl)*
NS(TrackJobCl_create_from_config_str)(
    const char *const SIXTRL_RESTRICT device_id_str,
    const char *const SIXTRL_RESTRICT config_str );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(TrackJobCl)* NS(TrackJobCl_new)(
    const char *const SIXTRL_RESTRICT device_id_str,
    NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
    NS(Buffer)* SIXTRL_RESTRICT beam_elements_buffer );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(TrackJobCl)* NS(TrackJobCl_new_with_output)(
    const char *const SIXTRL_RESTRICT device_id_str,
    NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
    NS(Buffer)* SIXTRL_RESTRICT beam_elements_buffer,
    NS(Buffer)* SIXTRL_RESTRICT output_buffer,
    NS(buffer_size_t) const until_turn_elem_by_elem );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(TrackJobCl)* NS(TrackJobCl_new_detailed)(
    const char *const SIXTRL_RESTRICT device_id_str,
    NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
    NS(buffer_size_t) const num_particle_sets,
    NS(buffer_size_t) const* SIXTRL_RESTRICT particle_set_indices_begin,
    NS(Buffer)* SIXTRL_RESTRICT beam_elements_buffer,
    NS(Buffer)* SIXTRL_RESTRICT output_buffer,
    NS(buffer_size_t) const until_turn_elem_by_elem,
    const char *const SIXTRL_RESTRICT config_str );

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(TrackJobCl_reset)(
    NS(TrackJobCl)* SIXTRL_RESTRICT job,
    NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
    NS(Buffer)* SIXTRL_RESTRICT beam_elements_buffer,
    NS(Buffer)* SIXTRL_RESTRICT output_buffer );

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(TrackJobCl_reset_with_output)(
    NS(TrackJobCl)* SIXTRL_RESTRICT job,
    NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
    NS(Buffer)* SIXTRL_RESTRICT beam_elements_buffer,
    NS(Buffer)* SIXTRL_RESTRICT output_buffer,
    NS(buffer_size_t) const until_turn_elem_by_elem );

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(TrackJobCl_reset_detailed)(
    NS(TrackJobCl)* SIXTRL_RESTRICT job,
    NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
    NS(buffer_size_t) const num_particle_sets,
    NS(buffer_size_t) const* SIXTRL_RESTRICT particle_set_indices_begin,
    NS(Buffer)* SIXTRL_RESTRICT beam_elements_buffer,
    NS(Buffer)* SIXTRL_RESTRICT output_buffer,
    NS(buffer_size_t) const until_turn_elem_by_elem );

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(TrackJobCl_delete)(
    NS(TrackJobCl)* SIXTRL_RESTRICT track_job );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(track_status_t)
NS(TrackJobCl_track_until_turn)(
    NS(TrackJobCl)* SIXTRL_RESTRICT track_job,
    NS(buffer_size_t) const until_turn );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(track_status_t)
NS(TrackJobCl_track_elem_by_elem)(
    NS(TrackJobCl)* SIXTRL_RESTRICT track_job,
    NS(buffer_size_t) const until_turn );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(track_status_t)
NS(TrackJobCl_track_line)(
    NS(TrackJobCl)* SIXTRL_RESTRICT track_job,
    NS(buffer_size_t) const line_begin_idx,
    NS(buffer_size_t) const line_end_idx,
    bool const finish_turn );

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(TrackJobCl_collect)(
    NS(TrackJobCl)* SIXTRL_RESTRICT track_job );

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(TrackJobCl_collect_detailed)(
    NS(TrackJobCl)* SIXTRL_RESTRICT track_job,
    NS(track_job_collect_flag_t) const flags );


SIXTRL_EXTERN SIXTRL_HOST_FN NS(ClContext)*
NS(TrackJobCl_get_context)( NS(TrackJobCl)* SIXTRL_RESTRICT track_job );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(ClContext) const*
NS(TrackJobCl_get_const_context)(
    const NS(TrackJobCl) *const SIXTRL_RESTRICT track_job );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(ClArgument)*
NS(TrackJobCl_get_particles_buffer_arg)(
    NS(TrackJobCl)* SIXTRL_RESTRICT track_job );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(ClArgument) const*
NS(TrackJobCl_get_const_particles_buffer_arg)(
    const NS(TrackJobCl) *const SIXTRL_RESTRICT track_job );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(ClArgument)*
NS(TrackJobCl_get_beam_elements_buffer_arg)(
    NS(TrackJobCl)* SIXTRL_RESTRICT track_job );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(ClArgument) const*
NS(TrackJobCl_get_const_beam_elements_buffer_arg)(
    const NS(TrackJobCl) *const SIXTRL_RESTRICT track_job );

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(TrackJobCl_has_output_buffer_arg)(
    const NS(TrackJobCl) *const SIXTRL_RESTRICT track_job );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(ClArgument)*
NS(TrackJobCl_get_output_buffer_arg)(
    NS(TrackJobCl)* SIXTRL_RESTRICT track_job );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(ClArgument) const*
NS(TrackJobCl_get_const_output_buffer_arg)(
    const NS(TrackJobCl) *const SIXTRL_RESTRICT track_job );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t)
NS(TrackJobCl_update_beam_elements_region)(
    NS(TrackJobCl)* SIXTRL_RESTRICT track_job,
    NS(context_size_t) const offset, NS(context_size_t) const length,
    void const* SIXTRL_RESTRICT new_value );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t)
NS(TrackJobCl_update_beam_elements_regions)(
    NS(TrackJobCl)* SIXTRL_RESTRICT track_job,
    NS(context_size_t) const num_regions_to_update,
    NS(context_size_t) const* offsets, NS(context_size_t) const* lengths,
    void const* SIXTRL_RESTRICT const* SIXTRL_RESTRICT new_value );

#if defined( __cplusplus ) && !defined( _GPUCODE )
}
#endif /* defined( __cplusplus ) && !defined( _GPUCODE ) */

/* ************************************************************************** */
/* ******   Implementation of inline and template methods / functions  ****** */
/* ************************************************************************** */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/output/output_buffer.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus ) && !defined( _GPUCODE )

namespace SIXTRL_CXX_NAMESPACE
{
    template< typename Iter >
    SIXTRL_HOST_FN bool TrackJobCl::doInitTrackJobCl(
        const char *const SIXTRL_RESTRICT device_id_str,
        TrackJobCl::c_buffer_t* SIXTRL_RESTRICT particles_buffer,
        Iter pset_begin, Iter pset_end,
        TrackJobCl::c_buffer_t* SIXTRL_RESTRICT belements_buffer,
        TrackJobCl::c_buffer_t* SIXTRL_RESTRICT output_buffer,
        TrackJobCl::size_type const until_turn_elem_by_elem,
        const char *const SIXTRL_RESTRICT config_str )
    {
        using _base_t = SIXTRL_CXX_NAMESPACE::TrackJobBase;
        using _size_t = _base_t::size_type;
        using flags_t = ::NS(output_buffer_flag_t);

        bool success  = false;
        this->doSetRequiresCollectFlag( true );

        if( config_str != nullptr )
        {
            this->doSetConfigStr( config_str );
            _base_t::doParseConfigStr( this->ptrConfigStr() );
            this->doParseConfigStrOclImpl( this->ptrConfigStr() );
        }

        success = this->doPrepareContextOclImpl(
            device_id_str, this->ptrConfigStr() );

        if( this->ptrContext() == nullptr ) success = false;
        if( particles_buffer == nullptr ) success = false;

        if( ( success ) && ( pset_begin != pset_end ) &&
            ( std::distance( pset_begin, pset_end ) > std::ptrdiff_t{ 0 } ) )
        {
            this->doSetParticleSetIndices(
                pset_begin, pset_end, particles_buffer );
        }
        else if( success )
        {
            _size_t const fallback_pset_indices[] =
            {
                _size_t{ 0 }, _size_t{ 0 }
            };

            this->doSetParticleSetIndices( &fallback_pset_indices[ 0 ],
                    &fallback_pset_indices[ 1 ], particles_buffer );
        }

        if( success )
        {
            success = _base_t::doPrepareParticlesStructures( particles_buffer );
        }

        if( success )
        {
            success = this->doPrepareParticlesStructuresOclImp(
                particles_buffer );
        }

        if( success ) this->doSetPtrCParticleBuffer( particles_buffer );

        if( success )
        {
            success = _base_t::doPrepareBeamElementsStructures(
                belements_buffer );
        }

        if( success )
        {
            success = this->doPrepareBeamElementsStructuresOclImp(
                belements_buffer );
        }

        if( success ) this->doSetPtrCBeamElementsBuffer( belements_buffer );

        flags_t const out_buffer_flags =
            ::NS(OutputBuffer_required_for_tracking_of_particle_sets)(
                particles_buffer, this->numParticleSets(),
                    this->particleSetIndicesBegin(), belements_buffer,
                        until_turn_elem_by_elem );

        bool const requires_output_buffer =
            ::NS(OutputBuffer_requires_output_buffer)( out_buffer_flags );

        if( success )
        {
            if( ( requires_output_buffer ) || ( output_buffer != nullptr ) )
            {
                success = _base_t::doPrepareOutputStructures( particles_buffer,
                    belements_buffer, output_buffer, until_turn_elem_by_elem );

                if( ( success ) && ( output_buffer != nullptr ) &&
                    ( !this->ownsOutputBuffer() ) )
                {
                    this->doSetPtrCOutputBuffer( output_buffer );
                }

                if( ( success ) && ( this->hasOutputBuffer() ) )
                {
                    success = this->doPrepareOutputStructuresOclImpl(
                        particles_buffer, belements_buffer,
                            this->ptrCOutputBuffer(), until_turn_elem_by_elem );
                }
            }
        }

        if( ( success ) && ( this->hasOutputBuffer() ) &&
            ( requires_output_buffer ) )
        {
            if( ::NS(OutputBuffer_requires_elem_by_elem_output)(
                    out_buffer_flags ) )
            {
                success = _base_t::doAssignOutputBufferToElemByElemConfig(
                    this->ptrElemByElemConfig(), this->ptrCOutputBuffer(),
                        this->elemByElemOutputBufferOffset() );

                if( success )
                {
                    success = this->doAssignOutputBufferToElemByElemConfigOclImpl(
                        this->ptrElemByElemConfig(), this->ptrCOutputBuffer(),
                            this->elemByElemOutputBufferOffset() );
                }
            }

            if( ( success ) &&
                ( ::NS(OutputBuffer_requires_beam_monitor_output)(
                    out_buffer_flags ) ) )
            {
                success = _base_t::doAssignOutputBufferToBeamMonitors(
                    belements_buffer, this->ptrCOutputBuffer(),
                        this->minInitialTurnId(),
                            this->beamMonitorsOutputBufferOffset() );

                if( success )
                {
                    success = this->doAssignOutputBufferToBeamMonitorsOclImp(
                        belements_buffer, this->ptrCOutputBuffer(),
                            this->minInitialTurnId(),
                                this->beamMonitorsOutputBufferOffset() );
                }
            }
        }

        return success;
    }

    template< typename Iter >
    SIXTRL_HOST_FN TrackJobCl::TrackJobCl(
        const char *const SIXTRL_RESTRICT device_id_str,
        TrackJobCl::c_buffer_t* SIXTRL_RESTRICT particles_buffer,
        Iter pset_begin, Iter pset_end,
        TrackJobCl::c_buffer_t* SIXTRL_RESTRICT belements_buffer,
        TrackJobCl::c_buffer_t* SIXTRL_RESTRICT output_buffer,
        TrackJobCl::size_type const until_turn_elem_by_elem,
        const char *const SIXTRL_RESTRICT config_str ) :
        TrackJobBase( SIXTRL_CXX_NAMESPACE::TRACK_JOB_CL_STR,
                      SIXTRL_CXX_NAMESPACE::TRACK_JOB_CL_ID ),
        m_ptr_context( nullptr ),
        m_ptr_particles_buffer_arg( nullptr ),
        m_ptr_beam_elements_buffer_arg( nullptr ),
        m_ptr_output_buffer_arg( nullptr ),
        m_ptr_cl_elem_by_elem_config_buffer( nullptr),
        m_total_num_particles( TrackJobBase::size_type{ 0 } )
    {
        this->doInitTrackJobCl( device_id_str, particles_buffer, pset_begin,
            pset_end, belements_buffer, output_buffer, until_turn_elem_by_elem,
                config_str );
    }

    template< typename Iter >
    SIXTRL_HOST_FN TrackJobCl::TrackJobCl(
        std::string const& SIXTRL_RESTRICT_REF device_id_str,
        TrackJobCl::buffer_t& SIXTRL_RESTRICT_REF particles_buffer,
        Iter pset_begin, Iter pset_end,
        TrackJobCl::buffer_t& SIXTRL_RESTRICT_REF belements_buffer,
        TrackJobCl::buffer_t* SIXTRL_RESTRICT ptr_output_buffer,
        TrackJobCl::size_type const until_turn_elem_by_elem,
        std::string SIXTRL_RESTRICT_REF config_str ) :
        TrackJobBase( SIXTRL_CXX_NAMESPACE::TRACK_JOB_CL_STR,
                      SIXTRL_CXX_NAMESPACE::TRACK_JOB_CL_ID ),
        m_ptr_context( nullptr ),
        m_ptr_particles_buffer_arg( nullptr ),
        m_ptr_beam_elements_buffer_arg( nullptr ),
        m_ptr_output_buffer_arg( nullptr ),
        m_ptr_cl_elem_by_elem_config_buffer( nullptr),
        m_total_num_particles( TrackJobBase::size_type{ 0 } )
    {
        TrackJobCl::c_buffer_t* out_buffer = ( ptr_output_buffer != nullptr )
            ? ptr_output_buffer->getCApiPtr() : nullptr;

        bool const success = this->doInitTrackJobCl( device_id_str.c_str(),
            particles_buffer.getCApiPtr(), pset_begin, pset_end,
            belements_buffer.getCApiPtr(), out_buffer, until_turn_elem_by_elem,
            config_str.c_str() );

        if( success )
        {
            this->doSetPtrParticleBuffer( &particles_buffer );
            this->doSetPtrBeamElementsBuffer( &belements_buffer );

            if( ( out_buffer != nullptr ) && ( this->hasOutputBuffer() ) )
            {
                this->doSetPtrOutputBuffer( ptr_output_buffer );
            }
        }
    }
}

#endif /* defined( __cplusplus ) && !defined( _GPUCODE ) */

#endif /* SIXTRL_SIXTRACKLIB_OPENCL_TRACK_CL_JOB_H__ */

/* end: sixtracklib/opencl/track_job.h */
