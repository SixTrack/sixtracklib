#include "sixtracklib/opencl/track_job_cl.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>

#if !defined( SIXTRL_NO_INCLUDES )
    #if defined( __cplusplus )
        #include "sixtracklib/common/buffer.hpp"
    #endif /* defined( __cplusplus ) */

    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/buffer.h"
    #include "sixtracklib/common/particles.h"
    #include "sixtracklib/common/internal/track_job_base.h"
    #include "sixtracklib/common/output/output_buffer.h"

    #include "sixtracklib/opencl/context.h"
    #include "sixtracklib/opencl/argument.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

namespace SIXTRL_CXX_NAMESPACE
{
    SIXTRL_HOST_FN TrackJobCl::TrackJobCl(
        std::string const& SIXTRL_RESTRICT_REF device_id_str,
        std::string const& SIXTRL_RESTRICT_REF config_str ) :
        TrackJobBase( SIXTRL_CXX_NAMESPACE::TRACK_JOB_OPENCL_STR,
                      SIXTRL_CXX_NAMESPACE::TRACK_JOB_OPENCL_ID ),
        m_ptr_context( nullptr ),
        m_ptr_particles_buffer_arg( nullptr ),
        m_ptr_beam_elements_buffer_arg( nullptr ),
        m_ptr_output_buffer_arg( nullptr )
    {
        if( !config_str.empty() )
        {
            this->doSetConfigStr( config_str.c_str() );
            TrackJobBase::doParseConfigStr( this->ptrConfigStr() );
            this->doParseConfigStrOclImpl(  this->ptrConfigStr() );
        }

        this->doPrepareContextOclImpl(
            device_id_str.c_str(), this->ptrConfigStr() );
    }

    SIXTRL_HOST_FN TrackJobCl::TrackJobCl(
        const char *const SIXTRL_RESTRICT device_id_str,
        const char *const SIXTRL_RESTRICT config_str ) :
        TrackJobBase( SIXTRL_CXX_NAMESPACE::TRACK_JOB_OPENCL_STR,
                      SIXTRL_CXX_NAMESPACE::TRACK_JOB_OPENCL_ID ),
        m_ptr_context( nullptr ),
        m_ptr_particles_buffer_arg( nullptr ),
        m_ptr_beam_elements_buffer_arg( nullptr ),
        m_ptr_output_buffer_arg( nullptr )
    {
        if( config_str != nullptr )
        {
            this->doSetConfigStr( config_str );
            TrackJobBase::doParseConfigStr( this->ptrConfigStr() );
            this->doParseConfigStrOclImpl(  this->ptrConfigStr() );
        }

        this->doPrepareContextOclImpl( device_id_str, this->ptrConfigStr() );
    }

    SIXTRL_HOST_FN TrackJobCl::TrackJobCl(
        const char *const SIXTRL_RESTRICT device_id_str,
        TrackJobCl::c_buffer_t* SIXTRL_RESTRICT particles_buffer,
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
        using size_t = TrackJobCl::size_type;
        size_t const particle_set_indices[] = { size_t{ 0 }, size_t{ 0 } };

        this->doInitTrackJobCl( device_id_str, particles_buffer,
            &particle_set_indices[ 0 ], &particle_set_indices[ 1 ],
            beam_elements_buffer, target_num_output_turns,
            num_elem_by_elem_turns, ptr_output_buffer, config_str );
    }

    SIXTRL_HOST_FN TrackJobCl::TrackJobCl(
        const char *const SIXTRL_RESTRICT device_id_str,
        TrackJobCl::c_buffer_t* SIXTRL_RESTRICT particles_buffer,
        TrackJobCl::size_type const num_particle_sets,
        TrackJobCl::size_type const* SIXTRL_RESTRICT particle_set_indices_begin,
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
        using size_t  = TrackJobCl::size_type;

        size_t const* particle_set_indices_end = particle_set_indices_begin;

        if( ( particle_set_indices_end != nullptr ) &&
            ( num_particle_sets > size_t{ 0 } ) )
        {
            std::advance( particle_set_indices_end, num_particle_sets );
        }

        this->doInitTrackJobCl( device_id_str, particles_buffer,
            particle_set_indices_begin, particle_set_indices_end,
            beam_elements_buffer, target_num_output_turns,
            num_elem_by_elem_turns, ptr_output_buffer, config_str );
    }

    SIXTRL_HOST_FN TrackJobCl::TrackJobCl(
        std::string const& SIXTRL_RESTRICT_REF device_id_str,
        TrackJobCl::buffer_t& SIXTRL_RESTRICT_REF particles_buffer,
        TrackJobCl::buffer_t& SIXTRL_RESTRICT_REF beam_elements_buffer,
        TrackJobCl::size_type const target_num_output_turns,
        TrackJobCl::size_type const num_elem_by_elem_turns,
        TrackJobCl::buffer_t* SIXTRL_RESTRICT ptr_output_buffer,
        std::string const& config_str ) :
        TrackJobBase( SIXTRL_CXX_NAMESPACE::TRACK_JOB_OPENCL_STR,
                      SIXTRL_CXX_NAMESPACE::TRACK_JOB_OPENCL_ID ),
        m_ptr_context( nullptr ),
        m_ptr_particles_buffer_arg( nullptr ),
        m_ptr_beam_elements_buffer_arg( nullptr ),
        m_ptr_output_buffer_arg( nullptr )

    {
        using size_t = TrackJobCl::size_type;
        size_t const particle_set_indices[] = { size_t{ 0 }, size_t{ 0 } };

        this->doInitTrackJobCl( device_id_str.c_str(),
            particles_buffer.getCApiPtr(),
            &particle_set_indices[ 0 ], &particle_set_indices[ 1 ],
            beam_elements_buffer.getCApiPtr(),
            target_num_output_turns, num_elem_by_elem_turns,
            ( ptr_output_buffer != nullptr )
                ? ptr_output_buffer->getCApiPtr() : nullptr,
            config_str.c_str() );
    }

    SIXTRL_HOST_FN TrackJobCl::~TrackJobCl() SIXTRL_NOEXCEPT
    {

    }

    SIXTRL_HOST_FN std::string TrackJobCl::deviceIdStr() const SIXTRL_NOEXCEPT
    {
        return ( this->ptrContext() != nullptr )
            ? this->ptrContext()->selectedNodeIdStr()
            : std::string{};
    }

    SIXTRL_HOST_FN TrackJobCl::cl_context_t&
    TrackJobCl::context() SIXTRL_RESTRICT
    {
        return const_cast< TrackJobCl::cl_context_t& >(
            static_cast< TrackJobCl const& >( *this ).context() );
    }


    SIXTRL_HOST_FN TrackJobCl::cl_context_t const&
    TrackJobCl::context() const SIXTRL_RESTRICT
    {
        SIXTRL_ASSERT( this->ptrContext() != nullptr );
        return *( this->ptrContext() );
    }

    SIXTRL_HOST_FN ::NS(ClContext)* TrackJobCl::ptrContext() SIXTRL_RESTRICT
    {
        return this->m_ptr_context.get();
    }

    SIXTRL_HOST_FN ::NS(ClContext) const*
    TrackJobCl::ptrContext() const SIXTRL_RESTRICT
    {
        return this->m_ptr_context.get();
    }

    SIXTRL_HOST_FN TrackJobCl::cl_arg_t&
    TrackJobCl::particlesArg() SIXTRL_NOEXCEPT
    {
        return const_cast< TrackJobCl::cl_arg_t& >(
            static_cast< TrackJobCl const& >( *this ).particlesArg() );
    }

    SIXTRL_HOST_FN TrackJobCl::cl_arg_t const&
    TrackJobCl::particlesArg() const SIXTRL_NOEXCEPT
    {
        SIXTRL_ASSERT( this->ptrParticlesArg() != nullptr );
        return *(this->ptrParticlesArg() );
    }

    SIXTRL_HOST_FN TrackJobCl::cl_arg_t*
    TrackJobCl::ptrParticlesArg() SIXTRL_NOEXCEPT
    {
        return const_cast< TrackJobCl::cl_arg_t* >( static_cast<
            TrackJobCl const& >( *this ).ptrParticlesArg() );
    }

    SIXTRL_HOST_FN TrackJobCl::cl_arg_t const*
    TrackJobCl::ptrParticlesArg() const SIXTRL_NOEXCEPT
    {
        return this->m_ptr_particles_buffer_arg.get();
    }

    SIXTRL_HOST_FN TrackJobCl::cl_arg_t&
    TrackJobCl::beamElementsArg() SIXTRL_NOEXCEPT
    {
        return const_cast< TrackJobCl::cl_arg_t& >( static_cast<
            TrackJobCl const& >( *this ).beamElementsArg() );
    }

    SIXTRL_HOST_FN TrackJobCl::cl_arg_t const&
    TrackJobCl::beamElementsArg() const SIXTRL_NOEXCEPT
    {
        SIXTRL_ASSERT( this->ptrBeamElementsArg() != nullptr );
        return *( this->ptrBeamElementsArg() );
    }

    SIXTRL_HOST_FN TrackJobCl::cl_arg_t*
    TrackJobCl::ptrBeamElementsArg() SIXTRL_NOEXCEPT
    {
        return const_cast< TrackJobCl::cl_arg_t* >( static_cast<
            TrackJobCl const& >( *this ).ptrBeamElementsArg() );
    }

    SIXTRL_HOST_FN TrackJobCl::cl_arg_t const*
    TrackJobCl::ptrBeamElementsArg() const SIXTRL_NOEXCEPT
    {
        return this->m_ptr_beam_elements_buffer_arg.get();
    }

    SIXTRL_HOST_FN TrackJobCl::cl_arg_t&
    TrackJobCl::outputBufferArg() SIXTRL_NOEXCEPT
    {
        return const_cast< TrackJobCl::cl_arg_t& >( static_cast<
            TrackJobCl const& >( *this ).outputBufferArg() );
    }

    SIXTRL_HOST_FN TrackJobCl::cl_arg_t const&
    TrackJobCl::outputBufferArg() const SIXTRL_NOEXCEPT
    {
        SIXTRL_ASSERT( this->ptrOutputBufferArg() != nullptr );
        return *( this->m_ptr_output_buffer_arg );
    }

    SIXTRL_HOST_FN TrackJobCl::cl_arg_t*
    TrackJobCl::ptrOutputBufferArg() SIXTRL_NOEXCEPT
    {
        return const_cast< TrackJobCl::cl_arg_t* >( static_cast<
            TrackJobCl const& >( *this ).ptrOutputBufferArg() );
    }

    SIXTRL_HOST_FN TrackJobCl::cl_arg_t const*
    TrackJobCl::ptrOutputBufferArg() const SIXTRL_NOEXCEPT
    {
        return this->m_ptr_output_buffer_arg.get();
    }

    SIXTRL_HOST_FN bool TrackJobCl::doPrepareParticlesStructures(
        TrackJobCl::c_buffer_t* SIXTRL_RESTRICT pb )
    {
        bool success = TrackJobBase::doPrepareParticlesStructures( pb );

        if( success )
        {
            success = this->doPrepareParticlesStructuresOclImp(
                this->ptrParticlesBuffer() );
        }

        return success;
    }

    SIXTRL_HOST_FN bool TrackJobCl::doPrepareBeamElementsStructures(
        TrackJobCl::c_buffer_t* SIXTRL_RESTRICT belems )
    {
        bool success = this->doPrepareBeamElementsStructures( belems );

        if( success )
        {
            success = this->doPrepareBeamElementsStructures(
                this->ptrCBeamElementsBuffer() );
        }

        return success;
    }

    SIXTRL_HOST_FN bool TrackJobCl::doPrepareOutputStructures(
        TrackJobCl::c_buffer_t const* SIXTRL_RESTRICT particles_buffer,
        TrackJobCl::c_buffer_t const* SIXTRL_RESTRICT beam_elem_buffer,
        TrackJobCl::c_buffer_t* SIXTRL_RESTRICT ptr_output_buffer,
        TrackJobCl::size_type const target_num_output_turns,
        TrackJobCl::size_type const num_elem_by_elem_turns )
    {
        bool success = TrackJobBase::doPrepareOutputStructures( particles_buffer,
                beam_elem_buffer, ptr_output_buffer, target_num_output_turns,
                num_elem_by_elem_turns );

        if( success )
        {
            success = this->doPrepareOutputStructuresOclImp( particles_buffer,
                beam_elem_buffer, ptr_output_buffer, target_num_output_turns,
                num_elem_by_elem_turns );
        }

        return success;
    }

    SIXTRL_HOST_FN bool TrackJobCl::doAssignOutputBufferToBeamMonitors(
        TrackJobCl::c_buffer_t* SIXTRL_RESTRICT beam_elem_buffer,
        TrackJobCl::c_buffer_t* SIXTRL_RESTRICT output_buffer )
    {
        return false;
    }

    SIXTRL_HOST_FN bool TrackJobCl::doReset(
        TrackJobCl::c_buffer_t* SIXTRL_RESTRICT particles_buffer,
        TrackJobCl::c_buffer_t* SIXTRL_RESTRICT beam_elem_buffer,
        TrackJobCl::c_buffer_t* SIXTRL_RESTRICT ptr_output_buffer,
        TrackJobCl::size_type const target_num_output_turns,
        TrackJobCl::size_type const num_elem_by_elem_turns )
    {
        bool success = false;

        if( TrackJobBase::doReset( particles_buffer, beam_elem_buffer,
            ptr_output_buffer, target_num_output_turns,
                num_elem_by_elem_turns ) )
        {
            success = this->doResetOclImp(
                particles_buffer, beam_elem_buffer, ptr_output_buffer,
                    target_num_output_turns, num_elem_by_elem_turns );

        }

        return success;
    }

    SIXTRL_HOST_FN TrackJobCl::track_status_t TrackJobCl::doTrackUntilTurn(
        TrackJobCl::size_type const until_turn )
    {
        return SIXTRL_CXX_NAMESPACE::track( *this, until_turn );
    }

    SIXTRL_HOST_FN TrackJobCl::track_status_t TrackJobCl::doTrackElemByElem(
        TrackJobCl::size_type const until_turn )
    {
        return SIXTRL_CXX_NAMESPACE::trackElemByElem( *this, until_turn );
    }


    SIXTRL_HOST_FN void TrackJobCl::doCollect()
    {
        SIXTRL_CXX_NAMESPACE::collect( *this );
        return;
    }

    SIXTRL_HOST_FN void TrackJobCl::doParseConfigStr(
        const char *const SIXTRL_RESTRICT  )
    {
        return;
    }

    SIXTRL_HOST_FN void TrackJobCl::doUpdateStoredContext(
        TrackJobCl::ptr_cl_context_t&& context )
    {
        this->m_ptr_context = std::move( context );
        return;
    }

    SIXTRL_HOST_FN void TrackJobCl::doUpdateStoredParticlesArg(
        TrackJobCl::ptr_cl_arg_t&& particle_arg )
    {
        this->m_ptr_particles_buffer_arg = std::move( particle_arg );
        return;
    }

    SIXTRL_HOST_FN void TrackJobCl::doUpdateStoredBeamElementsArg(
        TrackJobCl::ptr_cl_arg_t&& beam_elements_arg )
    {
        this->m_ptr_beam_elements_buffer_arg = std::move( beam_elements_arg );
        return;
    }

    SIXTRL_HOST_FN void TrackJobCl::doUpdateStoredOutputArg(
        TrackJobCl::ptr_cl_arg_t&& output_arg )
    {
        this->m_ptr_output_buffer_arg = std::move( output_arg );
        return;
    }

    SIXTRL_HOST_FN bool TrackJobCl::doPrepareContext(
        char const* SIXTRL_RESTRICT device_id_str,
        char const* SIXTRL_RESTRICT ptr_config_str )
    {
        return this->doPrepareContextOclImpl( device_id_str, ptr_config_str );
    }

    SIXTRL_HOST_FN bool TrackJobCl::doPrepareContextOclImpl(
        const char *const SIXTRL_RESTRICT device_id_str,
        const char *const SIXTRL_RESTRICT ptr_config_str )
    {
        using _this_t       = TrackJobCl;
        using context_t     = _this_t::cl_context_t;
        using ptr_context_t = _this_t::ptr_cl_context_t;

        bool success = false;

        ptr_context_t ptr_ctx( new context_t(
            device_id_str, ptr_config_str ) );

        if( ptr_ctx.get() != nullptr )
        {
            if( device_id_str != nullptr )
            {
                success = ptr_ctx->hasSelectedNode();
            }
            else
            {
                success = true;
            }
        }

        if( success )
        {
            this->doUpdateStoredContext( std::move( ptr_ctx ) );
        }

        return success;
    }

    SIXTRL_HOST_FN void TrackJobCl::doParseConfigStrOclImpl(
        const char *const SIXTRL_RESTRICT )
    {
        return;
    }

    SIXTRL_HOST_FN bool TrackJobCl::doPrepareParticlesStructuresOclImp(
        TrackJobCl::c_buffer_t* SIXTRL_RESTRICT pb )
    {
        using _this_t = TrackJobCl;

        using arg_t     = _this_t::cl_arg_t;
        using ptr_arg_t = _this_t::ptr_cl_arg_t;

        bool success = false;

        if( ( this->ptrContext() != nullptr ) && ( pb != nullptr ) )
        {
            ptr_arg_t particles_arg( new arg_t( pb, this->ptrContext() ) );
            this->doUpdateStoredParticlesArg( std::move( particles_arg ) );

            success = (
                ( this->ptrParticlesArg() != nullptr ) &&
                ( this->particlesArg().usesCObjectBuffer() ) &&
                ( this->particlesArg().context() == this->ptrContext() ) &&
                ( this->particlesArg().ptrCObjectBuffer() == pb ) );
        }

        return success;
    }

    SIXTRL_HOST_FN bool TrackJobCl::doPrepareBeamElementsStructuresOclImp(
        TrackJobCl::c_buffer_t* SIXTRL_RESTRICT belems )
    {
        using _this_t = TrackJobCl;

        using arg_t     = _this_t::cl_arg_t;
        using ptr_arg_t = _this_t::ptr_cl_arg_t;

        bool success = false;

        if( ( this->ptrContext() != nullptr ) && ( belems != nullptr ) )
        {
            ptr_arg_t belems_arg( new arg_t( belems, this->ptrContext() ) );
            this->doUpdateStoredBeamElementsArg( std::move( belems_arg ) );

            success = (
                ( this->ptrBeamElementsArg() != nullptr ) &&
                ( this->beamElementsArg().usesCObjectBuffer() ) &&
                ( this->beamElementsArg().context() == this->ptrContext() ) &&
                ( this->beamElementsArg().ptrCObjectBuffer() == belems ) );
        }

        return success;
    }

    SIXTRL_HOST_FN bool TrackJobCl::doPrepareOutputStructuresOclImp(
        TrackJobCl::c_buffer_t const* SIXTRL_RESTRICT pb,
        TrackJobCl::c_buffer_t const* SIXTRL_RESTRICT belems,
        TrackJobCl::c_buffer_t* SIXTRL_RESTRICT ptr_out_buffer,
        TrackJobCl::size_type const target_num_output_turns,
        TrackJobCl::size_type const num_elem_by_elem_turns )
    {
        using _this_t = TrackJobCl;

        using arg_t     = _this_t::cl_arg_t;
        using ptr_arg_t = _this_t::ptr_cl_arg_t;

        bool success = false;

        if( ( this->ptrContext() != nullptr ) &&
            ( ptr_out_buffer != nullptr ) && ( pb != nullptr ) )
        {
            ptr_arg_t ptr( new arg_t( ptr_out_buffer, this->ptrContext() ) );

            this->doUpdateStoredOutputArg( std::move( ptr ) );

            success = (
                ( this->ptrOutputBufferArg() != nullptr ) &&
                ( this->outputBufferArg().usesCObjectBuffer() ) &&
                ( this->outputBufferArg().ptrCObjectBuffer() ==
                  ptr_out_buffer ) );
        }

        return success;
    }

    SIXTRL_HOST_FN bool TrackJobCl::doAssignOutputBufferToBeamMonitorsOclImp(
        TrackJobCl::c_buffer_t* SIXTRL_RESTRICT beam_elem_buffer,
        TrackJobCl::c_buffer_t* SIXTRL_RESTRICT output_buffer )
    {
        bool success = false;

        using size_t = TrackJobCl::size_type;

        ( void )beam_elem_buffer;
        ( void )output_buffer;

        if( ( this->ptrContext() != nullptr ) &&
            ( this->ptrBeamElementsArg() != nullptr ) &&
            ( this->ptrOutputBufferArg() != nullptr ) &&
            ( ( this->hasBeamMonitorOutput() ) ||
              ( this->hasElemByElemOutput()  ) ) )
        {
            size_t const output_buffer_index_offset = std::min(
                this->elemByElemOutputBufferOffset(),
                this->beamMonitorsOutputBufferOffset() );

            success = ( 0 == this->ptrContext()->assignBeamMonitorIoBuffer(
                this->beamElementsArg(), this->outputBufferArg(),
                this->minInitialTurnId(), output_buffer_index_offset ) );
        }

        return success;
    }

    SIXTRL_HOST_FN bool TrackJobCl::doResetOclImp(
        TrackJobCl::c_buffer_t* SIXTRL_RESTRICT particles_buffer,
        TrackJobCl::c_buffer_t* SIXTRL_RESTRICT beam_elem_buffer,
        TrackJobCl::c_buffer_t* SIXTRL_RESTRICT ptr_output_buffer,
        TrackJobCl::size_type const target_num_output_turns,
        TrackJobCl::size_type const num_elem_by_elem_turns )
    {
        bool success = false;

        using c_buffer_t = TrackJobCl::c_buffer_t;
        using size_t     = TrackJobCl::size_type;

        if( ( this->ptrContext() != nullptr ) &&
            ( this->ptrContext()->hasSelectedNode() ) )
        {
            size_t const particle_set_indices[] = { size_t{ 0 }, size_t{ 0 } };

            success = true;

            _base_t::doPrepareParticlesStructures(
                const_cast< c_buffer_t* >( particles_buffer ) );

            this->doPrepareParticlesStructuresOclImp(
                const_cast< c_buffer_t* >( particles_buffer ) );

            this->doSetParticleSetIndices(
                &particle_set_indices[ 0 ],
                &particle_set_indices[ 1 ] );

            _base_t::doPrepareBeamElementsStructures(
                const_cast< c_buffer_t* >( beam_elem_buffer ) );

            this->doPrepareBeamElementsStructuresOclImp(
                const_cast< c_buffer_t* >( beam_elem_buffer ) );

            _base_t::doPrepareOutputStructures( this->ptrCParticlesBuffer(),
                this->ptrCBeamElementsBuffer(), ptr_output_buffer,
                target_num_output_turns, num_elem_by_elem_turns );

            if( this->ptrCOutputBuffer() != nullptr )
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

    SIXTRL_HOST_FN void collect(
        TrackJobCl& SIXTRL_RESTRICT_REF job ) SIXTRL_NOEXCEPT
    {
        if( ( job.ptrParticlesArg()       != nullptr ) &&
            ( job.ptrCParticlesBuffer()   != nullptr ) )
        {
            job.ptrParticlesArg()->read( job.ptrCParticlesBuffer() );
        }

        if( ( job.ptrOutputBufferArg()    != nullptr ) &&
            ( job.ptrCOutputBuffer()      != nullptr ) )
        {
            job.ptrOutputBufferArg()->read( job.ptrCOutputBuffer() );
        }

        if( ( job.ptrBeamElementsArg()     != nullptr ) &&
            ( job.ptrCBeamElementsBuffer() != nullptr ) )
        {
            job.ptrBeamElementsArg()->read( job.ptrCBeamElementsBuffer() );
        }

        return;
    }

    SIXTRL_HOST_FN TrackJobCl::track_status_t track(
        TrackJobCl& SIXTRL_RESTRICT_REF job,
        TrackJobCl::size_type const until_turn ) SIXTRL_NOEXCEPT
    {
        using status_t = TrackJobCl::track_status_t;
        SIXTRL_ASSERT( job.ptrContext() != nullptr );
        SIXTRL_ASSERT( job.ptrContext()->hasSelectedNode() );

        return static_cast< status_t >( job.ptrContext()->track( until_turn ) );
    }

    SIXTRL_HOST_FN TrackJobCl::track_status_t trackElemByElem(
        TrackJobCl& SIXTRL_RESTRICT_REF job,
        TrackJobCl::size_type const num_elem_by_elem_turns ) SIXTRL_NOEXCEPT
    {
        using status_t = TrackJobCl::track_status_t;
        status_t status = status_t{ -1 };

        SIXTRL_ASSERT( job.ptrContext() != nullptr );
        SIXTRL_ASSERT( job.ptrContext()->hasSelectedNode() );

        if( job.hasElemByElemOutput() )
        {
            status = static_cast< status_t >(
                job.ptrContext()->trackElementByElement(
                    num_elem_by_elem_turns,
                    job.elemByElemOutputBufferOffset() ) );
        }

        return status;
    }
}

SIXTRL_HOST_FN NS(TrackJobCl)* NS(TrackJobCl_create)(
    const char *const SIXTRL_RESTRICT device_id_str )
{
    return new SIXTRL_CXX_NAMESPACE::TrackJobCl( device_id_str, nullptr );
}

SIXTRL_HOST_FN NS(TrackJobCl)*
NS(TrackJobCl_create_from_config_str)(
    const char *const SIXTRL_RESTRICT device_id_str,
    const char *const SIXTRL_RESTRICT config_str )
{
    return new SIXTRL_CXX_NAMESPACE::TrackJobCl( device_id_str, config_str );
}

SIXTRL_HOST_FN NS(TrackJobCl)* NS(TrackJobCl_new)(
    const char *const SIXTRL_RESTRICT device_id_str,
    NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
    NS(Buffer)* SIXTRL_RESTRICT beam_elements_buffer,
    NS(buffer_size_t) const num_elem_by_elem_turns,
    NS(buffer_size_t) const until_turn )
{
    return new SIXTRL_CXX_NAMESPACE::TrackJobCl( device_id_str,
        particles_buffer, beam_elements_buffer, until_turn,
        num_elem_by_elem_turns, nullptr );
}

SIXTRL_HOST_FN NS(TrackJobCl)* NS(TrackJobCl_new_using_output_buffer)(
    const char *const SIXTRL_RESTRICT device_id_str,
    NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
    NS(Buffer)* SIXTRL_RESTRICT beam_elements_buffer,
    NS(Buffer)* SIXTRL_RESTRICT output_buffer,
    NS(buffer_size_t) const target_num_output_turns,
    NS(buffer_size_t) const target_num_elem_by_elem_turns )
{
    using size_t = SIXTRL_CXX_NAMESPACE::TrackJobCl::size_type;

    size_t const particle_set_indices[] = { size_t{ 0 }, size_t{ 0 } };

    return new SIXTRL_CXX_NAMESPACE::TrackJobCl( device_id_str,
        particles_buffer, size_t{ 1 }, &particle_set_indices[ 0 ],
        beam_elements_buffer, target_num_output_turns,
        target_num_elem_by_elem_turns, output_buffer, nullptr );
}

SIXTRL_HOST_FN NS(TrackJobCl)* NS(TrackJobCl_new_detailed)(
    const char *const SIXTRL_RESTRICT device_id_str,
    NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
    NS(buffer_size_t) const num_particle_sets,
    NS(buffer_size_t) const* SIXTRL_RESTRICT particle_set_indices_begin,
    NS(Buffer)* SIXTRL_RESTRICT beam_elements_buffer,
    NS(Buffer)* SIXTRL_RESTRICT output_buffer,
    NS(buffer_size_t) const max_output_turns,
    NS(buffer_size_t) const num_elem_by_elem_turns,
    const char *const SIXTRL_RESTRICT config_str )
{
    return new SIXTRL_CXX_NAMESPACE::TrackJobCl( device_id_str,
        particles_buffer, num_particle_sets, particle_set_indices_begin,
        beam_elements_buffer, max_output_turns, num_elem_by_elem_turns,
        output_buffer, config_str );
}


SIXTRL_HOST_FN void NS(TrackJobCl_delete)(
    NS(TrackJobCl)* SIXTRL_RESTRICT track_job )
{
    delete track_job;
}

SIXTRL_HOST_FN bool NS(TrackJobCl_track_until_turn)(
    NS(TrackJobCl)* SIXTRL_RESTRICT track_job,
    NS(buffer_size_t) const until_turn )
{
    SIXTRL_ASSERT( track_job != nullptr );
    return ( 0 == SIXTRL_CXX_NAMESPACE::track( *track_job, until_turn ) );
}

SIXTRL_HOST_FN bool NS(TrackJobCl_track_elem_by_elem)(
    NS(TrackJobCl)* SIXTRL_RESTRICT track_job,
    NS(buffer_size_t) const until_turn )
{
    SIXTRL_ASSERT( track_job != nullptr );
    return ( 0 == SIXTRL_CXX_NAMESPACE::trackElemByElem(
        *track_job, until_turn ) );
}

SIXTRL_HOST_FN void NS(TrackJobCl_collect)(
    NS(TrackJobCl)* SIXTRL_RESTRICT track_job )
{
    SIXTRL_ASSERT( track_job != nullptr );
    SIXTRL_CXX_NAMESPACE::collect( *track_job );
    return;
}

SIXTRL_HOST_FN NS(ClContext)* NS(TrackJobCl_get_context)(
    NS(TrackJobCl)* SIXTRL_RESTRICT track_job )
{
    return ( track_job != nullptr ) ? track_job->ptrContext() : nullptr;
}

SIXTRL_HOST_FN NS(ClContext) const* NS(TrackJobCl_get_const_context)(
    const NS(TrackJobCl) *const SIXTRL_RESTRICT track_job )
{
    return ( track_job != nullptr ) ? track_job->ptrContext() : nullptr;
}

/* end: /opencl/internal/track_job_cl.cpp */
