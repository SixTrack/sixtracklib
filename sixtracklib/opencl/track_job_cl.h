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

#if !defined( _GPUCODE )

#if !defined( SIXTRL_TRACK_JOB_OPENCL_ID )
    #define   SIXTRL_TRACK_JOB_OPENCL_ID 2
#endif /* !defined( SIXTRL_TRACK_JOB_OPENCL_ID ) */

#if !defined( SIXTRL_TRACK_JOB_OPENCL_STR )
    #define   SIXTRL_TRACK_JOB_OPENCL_STR "opencl"
#endif /* !defined( SIXTRL_TRACK_JOB_OPENCL_STR ) */

#if defined( __cplusplus )

namespace SIXTRL_CXX_NAMESPACE
{
    SIXTRL_STATIC_VAR track_job_type_t const
        TRACK_JOB_OPENCL_ID = track_job_type_t{SIXTRL_TRACK_JOB_OPENCL_ID};

    SIXTRL_STATIC_VAR char const
        TRACK_JOB_OPENCL_STR[] = SIXTRL_TRACK_JOB_OPENCL_STR;

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

        using cl_arg_t              = SIXTRL_CXX_NAMESPACE::ClArgument;
        using cl_context_t          = SIXTRL_CXX_NAMESPACE::ClContext;

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
            size_type const target_num_output_turns       = size_type{ 0 },
            size_type const num_elem_by_elem_turns        = size_type{ 0 },
            c_buffer_t* SIXTRL_RESTRICT ptr_output_buffer = nullptr,
            const char *const SIXTRL_RESTRICT config_str  = nullptr );

        SIXTRL_HOST_FN TrackJobCl(
            const char *const SIXTRL_RESTRICT device_id_str,
            c_buffer_t* SIXTRL_RESTRICT particles_buffer,
            size_type const num_particle_sets,
            size_type const* SIXTRL_RESTRICT particle_set_indices_begin,
            c_buffer_t* SIXTRL_RESTRICT beam_elements_buffer,
            size_type const target_num_output_turns       = size_type{ 0 },
            size_type const num_elem_by_elem_turns        = size_type{ 0 },
            c_buffer_t* SIXTRL_RESTRICT ptr_output_buffer = nullptr,
            const char *const SIXTRL_RESTRICT config_str  = nullptr );

        template< typename PartSetIndexIter >
        SIXTRL_HOST_FN TrackJobCl(
            const char *const SIXTRL_RESTRICT device_id_str,
            c_buffer_t* SIXTRL_RESTRICT particles_buffer,
            PartSetIndexIter particle_set_indices_begin,
            PartSetIndexIter particle_set_indices_end,
            c_buffer_t* SIXTRL_RESTRICT beam_elements_buffer,
            size_type const target_num_output_turns       = size_type{ 0 },
            size_type const num_elem_by_elem_turns        = size_type{ 0 },
            c_buffer_t* SIXTRL_RESTRICT ptr_output_buffer = nullptr,
            const char *const SIXTRL_RESTRICT config_str  = nullptr );

        SIXTRL_HOST_FN TrackJobCl(
            std::string const& SIXTRL_RESTRICT_REF device_id_str,
            buffer_t& SIXTRL_RESTRICT_REF particles_buffer,
            buffer_t& SIXTRL_RESTRICT_REF beam_elements_buffer,
            size_type const target_num_output_turns = size_type{ 0 },
            size_type const num_elem_by_elem_turns  = size_type{ 0 },
            buffer_t* SIXTRL_RESTRICT ptr_output_buffer = nullptr,
            std::string const& config_str = std::string{} );

        template< typename PartSetIndexIter >
        SIXTRL_HOST_FN TrackJobCl(
            std::string const& SIXTRL_RESTRICT_REF device_id_str,
            buffer_t& SIXTRL_RESTRICT_REF particles_buffer,
            PartSetIndexIter particle_set_indices_begin,
            PartSetIndexIter particle_set_indices_end,
            buffer_t& SIXTRL_RESTRICT_REF beam_elements_buffer,
            size_type const target_num_output_turns       = size_type{ 0 },
            size_type const num_elem_by_elem_turns        = size_type{ 0 },
            buffer_t* SIXTRL_RESTRICT ptr_output_buffer   = nullptr,
            std::string SIXTRL_RESTRICT_REF config_str    = std::string{} );

        SIXTRL_HOST_FN TrackJobCl( TrackJobCl const& other ) = default;
        SIXTRL_HOST_FN TrackJobCl( TrackJobCl&& other ) = default;

        SIXTRL_HOST_FN TrackJobCl& operator=( TrackJobCl const& rhs ) = default;
        SIXTRL_HOST_FN TrackJobCl& operator=( TrackJobCl&& rhs ) = default;

        SIXTRL_HOST_FN virtual ~TrackJobCl() SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN std::string deviceIdStr() const SIXTRL_NOEXCEPT;

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

        protected:

        using ptr_cl_context_t = std::unique_ptr< cl_context_t >;
        using ptr_cl_arg_t     = std::unique_ptr< cl_arg_t >;

        SIXTRL_HOST_FN virtual bool doPrepareContext(
            char const* SIXTRL_RESTRICT device_id_str,
            const char *const SIXTRL_RESTRICT ptr_config_str );

        SIXTRL_HOST_FN virtual bool doPrepareParticlesStructures(
            c_buffer_t* SIXTRL_RESTRICT ptr_particles_buffer ) override;

        SIXTRL_HOST_FN virtual bool doPrepareBeamElementsStructures(
            c_buffer_t* SIXTRL_RESTRICT ptr_beam_elem_buffer ) override;

        SIXTRL_HOST_FN virtual bool doPrepareOutputStructures(
            c_buffer_t const* SIXTRL_RESTRICT particles_buffer,
            c_buffer_t const* SIXTRL_RESTRICT beam_elem_buffer,
            c_buffer_t* SIXTRL_RESTRICT ptr_output_buffer,
            size_type const target_num_output_turns,
            size_type const num_elem_by_elem_turns ) override;

        SIXTRL_HOST_FN virtual bool doAssignOutputBufferToBeamMonitors(
            c_buffer_t* SIXTRL_RESTRICT beam_elem_buffer,
            c_buffer_t* SIXTRL_RESTRICT output_buffer ) override;

        SIXTRL_HOST_FN virtual bool doReset(
            c_buffer_t const* SIXTRL_RESTRICT particles_buffer,
            c_buffer_t const* SIXTRL_RESTRICT beam_elem_buffer,
            c_buffer_t* SIXTRL_RESTRICT ptr_output_buffer,
            size_type const target_num_output_turns,
            size_type const num_elem_by_elem_turns ) override;

        SIXTRL_HOST_FN virtual track_status_t doTrackUntilTurn(
            size_type const until_turn ) override;

        SIXTRL_HOST_FN virtual track_status_t doTrackElemByElem(
            size_type const until_turn ) override;

        SIXTRL_HOST_FN virtual void doCollect() override;

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

        private:

        template< typename PartSetIndexIter >
        SIXTRL_HOST_FN bool doInitTrackJobCl(
            const char *const SIXTRL_RESTRICT device_id_str,
            c_buffer_t* SIXTRL_RESTRICT particles_buffer,
            PartSetIndexIter particle_set_indices_begin,
            PartSetIndexIter particle_set_indices_end,
            c_buffer_t* SIXTRL_RESTRICT beam_elements_buffer,
            size_type const target_num_output_turns,
            size_type const num_elem_by_elem_turns,
            c_buffer_t* SIXTRL_RESTRICT ptr_output_buffer,
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

        SIXTRL_HOST_FN bool doPrepareOutputStructuresOclImp(
            c_buffer_t const* SIXTRL_RESTRICT particles_buffer,
            c_buffer_t const* SIXTRL_RESTRICT beam_elem_buffer,
            c_buffer_t* SIXTRL_RESTRICT ptr_output_buffer,
            size_type const target_num_output_turns,
            size_type const num_elem_by_elem_turns );

        SIXTRL_HOST_FN bool doAssignOutputBufferToBeamMonitorsOclImp(
            c_buffer_t* SIXTRL_RESTRICT beam_elem_buffer,
            c_buffer_t* SIXTRL_RESTRICT output_buffer );

        SIXTRL_HOST_FN bool doResetOclImp(
            c_buffer_t const* SIXTRL_RESTRICT particles_buffer,
            c_buffer_t const* SIXTRL_RESTRICT beam_elem_buffer,
            c_buffer_t* SIXTRL_RESTRICT ptr_output_buffer,
            size_type const target_num_output_turns,
            size_type const num_elem_by_elem_turns );

        ptr_cl_context_t m_ptr_context;
        ptr_cl_arg_t     m_ptr_particles_buffer_arg;
        ptr_cl_arg_t     m_ptr_beam_elements_buffer_arg;
        ptr_cl_arg_t     m_ptr_output_buffer_arg;
    };

    template< typename PartSetIndexIter >
    SIXTRL_HOST_FN bool TrackJobCl::doInitTrackJobCl(
        const char *const SIXTRL_RESTRICT device_id_str,
        TrackJobCl::c_buffer_t* SIXTRL_RESTRICT particles_buffer,
        PartSetIndexIter particle_set_indices_begin,
        PartSetIndexIter particle_set_indices_end,
        TrackJobCl::c_buffer_t* SIXTRL_RESTRICT beam_elements_buffer,
        TrackJobCl::size_type const target_num_output_turns,
        TrackJobCl::size_type const num_elem_by_elem_turns,
        TrackJobCl::c_buffer_t* SIXTRL_RESTRICT ptr_output_buffer,
        const char *const SIXTRL_RESTRICT config_str  )
    {
        using _base_t = TrackJobBase;
        bool success = false;

        if( config_str != nullptr )
        {
            this->doSetConfigStr( config_str );
            _base_t::doParseConfigStr( this->ptrConfigStr() );
            this->doParseConfigStrOclImpl( this->ptrConfigStr() );
        }

        this->doPrepareContextOclImpl( device_id_str, this->ptrConfigStr() );

        if( ( this->ptrContext() != nullptr ) &&
            ( this->ptrContext()->hasSelectedNode() ) )
        {
            success = true;

            _base_t::doPrepareParticlesStructures( particles_buffer );
            this->doPrepareParticlesStructuresOclImp( particles_buffer );

            if( ( particle_set_indices_begin != nullptr ) &&
                ( particle_set_indices_end   != nullptr ) &&
                ( particle_set_indices_begin != particle_set_indices_end ) )
            {
                if( std::distance( particle_set_indices_begin,
                    particle_set_indices_end ) > std::ptrdiff_t{ 0 } )
                {
                    this->doSetParticleSetIndices(
                        particle_set_indices_begin, particle_set_indices_end );
                }
            }

            _base_t::doPrepareBeamElementsStructures( beam_elements_buffer );
            this->doPrepareBeamElementsStructuresOclImp( beam_elements_buffer );

            _base_t::doPrepareOutputStructures( this->ptrCParticlesBuffer(),
                this->ptrCBeamElementsBuffer(), ptr_output_buffer,
                target_num_output_turns, num_elem_by_elem_turns );

            if( this->ptrOutputBuffer() != nullptr )
            {
                this->doPrepareOutputStructuresOclImp(
                    this->ptrCParticlesBuffer(), this->ptrCBeamElementsBuffer(),
                    this->ptrCOutputBuffer(), target_num_output_turns,
                    num_elem_by_elem_turns );

                _base_t::doAssignOutputBufferToBeamMonitors(
                    this->ptrCBeamElementsBuffer(), this->ptrCOutputBuffer() );

                this->doAssignOutputBufferToBeamMonitorsOclImp(
                    this->ptrCBeamElementsBuffer(), this->ptrCOutputBuffer() );
            }
        }

        return success;
    }

    template< typename PartSetIndexIter >
    SIXTRL_HOST_FN TrackJobCl::TrackJobCl(
        const char *const SIXTRL_RESTRICT device_id_str,
        TrackJobCl::c_buffer_t* SIXTRL_RESTRICT particles_buffer,
        PartSetIndexIter particle_set_indices_begin,
        PartSetIndexIter particle_set_indices_end,
        TrackJobCl::c_buffer_t* SIXTRL_RESTRICT beam_elements_buffer,
        TrackJobCl::size_type const target_num_output_turns,
        TrackJobCl::size_type const num_elem_by_elem_turns,
        TrackJobCl::c_buffer_t* SIXTRL_RESTRICT ptr_output_buffer,
        const char *const SIXTRL_RESTRICT config_str ) :
        TrackJobBase( SIXTRL_CXX_NAMESPACE::TRACK_JOB_OPENCL_STR,
                      SIXTRL_CXX_NAMESPACE::TRACK_JOB_OPENCL_ID ),
        m_ptr_context( nullptr ),
        m_ptr_particles_buffer_arg( nullptr ),
        m_ptr_beam_elements_buffer_arg( nullptr ),
        m_ptr_output_buffer_arg( nullptr )
    {
        this->doInitTrackJobCl( device_id_str, particles_buffer,
            particle_set_indices_begin, particle_set_indices_end,
            beam_elements_buffer, target_num_output_turns,
            num_elem_by_elem_turns, ptr_output_buffer, config_str );
    }

    template< typename PartSetIndexIter >
    SIXTRL_HOST_FN TrackJobCl::TrackJobCl(
        std::string const& SIXTRL_RESTRICT_REF device_id_str,
        TrackJobCl::buffer_t& SIXTRL_RESTRICT_REF particles_buffer,
        PartSetIndexIter particle_set_indices_begin,
        PartSetIndexIter particle_set_indices_end,
        TrackJobCl::buffer_t& SIXTRL_RESTRICT_REF beam_elements_buffer,
        TrackJobCl::size_type const target_num_output_turns,
        TrackJobCl::size_type const num_elem_by_elem_turns,
        TrackJobCl::buffer_t* SIXTRL_RESTRICT ptr_output_buffer,
        std::string SIXTRL_RESTRICT_REF config_str ) :
        TrackJobBase( SIXTRL_CXX_NAMESPACE::TRACK_JOB_OPENCL_STR,
                      SIXTRL_CXX_NAMESPACE::TRACK_JOB_OPENCL_ID ),
        m_ptr_context( nullptr ),
        m_ptr_particles_buffer_arg( nullptr ),
        m_ptr_beam_elements_buffer_arg( nullptr ),
        m_ptr_output_buffer_arg( nullptr )
    {
        this->doInitTrackJobCl( device_id_str.c_str(),
            particles_buffer.getCApiPtr(),
            particle_set_indices_begin, particle_set_indices_end,
            beam_elements_buffer.getCApiPtr(),
            target_num_output_turns, num_elem_by_elem_turns,
            ( ptr_output_buffer != nullptr )
                ? ptr_output_buffer->getCApiPtr() : nullptr,
            config_str.c_str() );
    }

    SIXTRL_HOST_FN void collect(
        TrackJobCl& SIXTRL_RESTRICT_REF track_job ) SIXTRL_NOEXCEPT;

    SIXTRL_HOST_FN TrackJobCl::track_status_t track(
        TrackJobCl& SIXTRL_RESTRICT_REF job,
        TrackJobCl::size_type const until_turn ) SIXTRL_NOEXCEPT;

    SIXTRL_HOST_FN TrackJobCl::track_status_t trackElemByElem(
        TrackJobCl& SIXTRL_RESTRICT_REF job,
        TrackJobCl::size_type const num_elem_by_elem_turns ) SIXTRL_NOEXCEPT;
}

typedef SIXTRL_CXX_NAMESPACE::TrackJobCl    NS(TrackJobCl);

SIXTRL_STATIC_VAR NS(track_job_type_t) const NS(TRACK_JOB_OPENCL_ID) =
        SIXTRL_TRACK_JOB_OPENCL_ID;

SIXTRL_STATIC_VAR char const NS(TRACK_JOB_OPENCL_STR)[] =
        SIXTRL_TRACK_JOB_OPENCL_STR;

#else /* defined( __cplusplus ) */

typedef void NS(TrackJobCl);

SIXTRL_STATIC_VAR NS(track_job_type_t) const NS(TRACK_JOB_OPENCL_ID) =
    SIXTRL_TRACK_JOB_OPENCL_ID;

SIXTRL_STATIC_VAR char const NS(TRACK_JOB_OPENCL_STR)[] =
    SIXTRL_TRACK_JOB_OPENCL_STR;

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
    NS(Buffer)* SIXTRL_RESTRICT beam_elements_buffer,
    NS(buffer_size_t) const num_elem_by_elem_turns,
    NS(buffer_size_t) const until_turn );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(TrackJobCl)* NS(TrackJobCl_new_detailed)(
    const char *const SIXTRL_RESTRICT device_id_str,
    NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
    NS(buffer_size_t) const num_particle_sets,
    NS(buffer_size_t) const* SIXTRL_RESTRICT particle_set_indices_begin,
    NS(Buffer)* SIXTRL_RESTRICT beam_elements_buffer,
    NS(Buffer)* SIXTRL_RESTRICT output_buffer,
    NS(buffer_size_t) const max_output_turns,
    NS(buffer_size_t) const num_elem_by_elem_turns,
    const char *const SIXTRL_RESTRICT config_str );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(TrackJobCl)*
NS(TrackJobCl_new_using_output_buffer)(
    const char *const SIXTRL_RESTRICT device_id_str,
    NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
    NS(Buffer)* SIXTRL_RESTRICT beam_elements_buffer,
    NS(Buffer)* SIXTRL_RESTRICT output_buffer,
    NS(buffer_size_t) const num_elem_by_elem_turns,
    NS(buffer_size_t) const until_turn );

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(TrackJobCl_delete)(
    NS(TrackJobCl)* SIXTRL_RESTRICT track_job );

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(TrackJobCl_track_until_turn)(
    NS(TrackJobCl)* SIXTRL_RESTRICT track_job,
    NS(buffer_size_t) const until_turn );

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(TrackJobCl_track_elem_by_elem)(
    NS(TrackJobCl)* SIXTRL_RESTRICT track_job,
    NS(buffer_size_t) const until_turn );

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(TrackJobCl_collect)(
    NS(TrackJobCl)* SIXTRL_RESTRICT track_job );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(ClContext)*
NS(TrackJobCl_get_context)( NS(TrackJobCl)* SIXTRL_RESTRICT track_job );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(ClContext) const*
NS(TrackJobCl_get_const_context)(
    const NS(TrackJobCl) *const SIXTRL_RESTRICT track_job );

#if defined( __cplusplus ) && !defined( _GPUCODE )
}
#endif /* defined( __cplusplus ) && !defined( _GPUCODE ) */

#endif /* SIXTRL_SIXTRACKLIB_OPENCL_TRACK_CL_JOB_H__ */

/* end: sixtracklib/opencl/track_job.h */
