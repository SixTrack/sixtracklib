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
        TrackJobBase( SIXTRL_CXX_NAMESPACE::TRACK_JOB_CL_STR,
                      SIXTRL_CXX_NAMESPACE::TRACK_JOB_CL_ID ),
        m_ptr_context( nullptr ),
        m_ptr_particles_buffer_arg( nullptr ),
        m_ptr_beam_elements_buffer_arg( nullptr ),
        m_ptr_output_buffer_arg( nullptr ),
        m_ptr_cl_elem_by_elem_config_buffer( nullptr ),
        m_total_num_particles( TrackJobBase::size_type{ 0 } )
    {
        if( !config_str.empty() )
        {
            this->doSetConfigStr( config_str.c_str() );
            TrackJobBase::doParseConfigStr( this->ptrConfigStr() );
            this->doParseConfigStrOclImpl(  this->ptrConfigStr() );
        }

        this->doSetDeviceIdStr( device_id_str.c_str() );

        this->doPrepareContextOclImpl(
            device_id_str.c_str(), this->ptrConfigStr() );
    }

    SIXTRL_HOST_FN TrackJobCl::TrackJobCl(
        const char *const SIXTRL_RESTRICT device_id_str,
        const char *const SIXTRL_RESTRICT config_str ) :
        TrackJobBase( SIXTRL_CXX_NAMESPACE::TRACK_JOB_CL_STR,
                      SIXTRL_CXX_NAMESPACE::TRACK_JOB_CL_ID ),
        m_ptr_context( nullptr ),
        m_ptr_particles_buffer_arg( nullptr ),
        m_ptr_beam_elements_buffer_arg( nullptr ),
        m_ptr_output_buffer_arg( nullptr ),
        m_ptr_cl_elem_by_elem_config_buffer( nullptr ),
        m_total_num_particles( TrackJobBase::size_type{ 0 } )
    {
        if( config_str != nullptr )
        {
            this->doSetConfigStr( config_str );
            TrackJobBase::doParseConfigStr( this->ptrConfigStr() );
            this->doParseConfigStrOclImpl(  this->ptrConfigStr() );
        }

        this->doSetDeviceIdStr( device_id_str );
        this->doPrepareContextOclImpl( device_id_str, this->ptrConfigStr() );
    }

    SIXTRL_HOST_FN TrackJobCl::TrackJobCl(
        const char *const SIXTRL_RESTRICT device_id_str,
        TrackJobCl::c_buffer_t* SIXTRL_RESTRICT particles_buffer,
        TrackJobCl::c_buffer_t* SIXTRL_RESTRICT beam_elements_buffer,
        TrackJobCl::c_buffer_t* SIXTRL_RESTRICT ptr_output_buffer,
        TrackJobCl::size_type const dump_elem_by_elem_turns,
        const char *const SIXTRL_RESTRICT config_str ) :
        TrackJobBase( SIXTRL_CXX_NAMESPACE::TRACK_JOB_CL_STR,
                      SIXTRL_CXX_NAMESPACE::TRACK_JOB_CL_ID ),
        m_ptr_context( nullptr ),
        m_ptr_particles_buffer_arg( nullptr ),
        m_ptr_beam_elements_buffer_arg( nullptr ),
        m_ptr_output_buffer_arg( nullptr ),
        m_ptr_cl_elem_by_elem_config_buffer( nullptr ),
        m_total_num_particles( TrackJobBase::size_type{ 0 } )
    {
        this->doInitTrackJobCl( device_id_str, particles_buffer,
            this->particleSetIndicesBegin(), this->particleSetIndicesEnd(),
            beam_elements_buffer, ptr_output_buffer, dump_elem_by_elem_turns,
            config_str );
    }

    SIXTRL_HOST_FN TrackJobCl::TrackJobCl(
        const char *const SIXTRL_RESTRICT device_id_str,
        TrackJobCl::c_buffer_t* SIXTRL_RESTRICT particles_buffer,
        TrackJobCl::size_type const num_particle_sets,
        TrackJobCl::size_type const* SIXTRL_RESTRICT pset_begin,
        TrackJobCl::c_buffer_t* SIXTRL_RESTRICT belements_buffer,
        TrackJobCl::c_buffer_t* SIXTRL_RESTRICT ptr_output_buffer,
        TrackJobCl::size_type const dump_elem_by_elem_turns,
        const char *const SIXTRL_RESTRICT config_str ) :
        TrackJobBase( SIXTRL_CXX_NAMESPACE::TRACK_JOB_CL_STR,
                      SIXTRL_CXX_NAMESPACE::TRACK_JOB_CL_ID ),
        m_ptr_context( nullptr ),
        m_ptr_particles_buffer_arg( nullptr ),
        m_ptr_beam_elements_buffer_arg( nullptr ),
        m_ptr_output_buffer_arg( nullptr ),
        m_ptr_cl_elem_by_elem_config_buffer( nullptr ),
        m_total_num_particles( TrackJobBase::size_type{ 0 } )
    {
        using size_t  = TrackJobCl::size_type;
        size_t const* pset_end = pset_begin;

        if( ( pset_end != nullptr ) && ( num_particle_sets > size_t{ 0 } ) )
        {
            std::advance( pset_end, num_particle_sets );
        }

        this->doInitTrackJobCl( device_id_str, particles_buffer,
            pset_begin, pset_end, belements_buffer, ptr_output_buffer,
            dump_elem_by_elem_turns, config_str );
    }

    SIXTRL_HOST_FN TrackJobCl::TrackJobCl(
        std::string const& SIXTRL_RESTRICT_REF device_id_str,
        TrackJobCl::buffer_t& SIXTRL_RESTRICT_REF particles_buffer,
        TrackJobCl::buffer_t& SIXTRL_RESTRICT_REF belements_buffer,
        TrackJobCl::buffer_t* SIXTRL_RESTRICT ptr_output_buffer,
        TrackJobCl::size_type const dump_elem_by_elem_turns,
        std::string const& config_str ) :
        TrackJobBase( SIXTRL_CXX_NAMESPACE::TRACK_JOB_CL_STR,
                      SIXTRL_CXX_NAMESPACE::TRACK_JOB_CL_ID ),
        m_ptr_context( nullptr ),
        m_ptr_particles_buffer_arg( nullptr ),
        m_ptr_beam_elements_buffer_arg( nullptr ),
        m_ptr_output_buffer_arg( nullptr ),
        m_ptr_cl_elem_by_elem_config_buffer( nullptr ),
        m_total_num_particles( TrackJobBase::size_type{ 0 } )
    {
        using c_buffer_t = TrackJobCl::c_buffer_t;

        c_buffer_t* ptr_part_buffer  = particles_buffer.getCApiPtr();
        c_buffer_t* ptr_belem_buffer = belements_buffer.getCApiPtr();
        c_buffer_t* ptr_out_buffer   = ( ptr_output_buffer != nullptr )
            ? ptr_output_buffer->getCApiPtr() : nullptr;

        this->doInitTrackJobCl( device_id_str.c_str(), ptr_part_buffer,
            this->particleSetIndicesBegin(), this->particleSetIndicesEnd(),
                ptr_belem_buffer, ptr_out_buffer, dump_elem_by_elem_turns,
                    config_str.c_str() );
    }

    SIXTRL_HOST_FN TrackJobCl::~TrackJobCl() SIXTRL_NOEXCEPT {}

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


    SIXTRL_HOST_FN TrackJobCl::cl_buffer_t const&
    TrackJobCl::clElemByElemConfigBuffer() const SIXTRL_NOEXCEPT
    {
        SIXTRL_ASSERT( this->ptrClElemByElemConfigBuffer() != nullptr );
        return *( this->ptrClElemByElemConfigBuffer() );
    }

    SIXTRL_HOST_FN TrackJobCl::cl_buffer_t&
    TrackJobCl::clElemByElemConfigBuffer() SIXTRL_NOEXCEPT
    {
        using _this_t = TrackJobCl;
        using ref_t   = _this_t::cl_buffer_t&;

        return const_cast< ref_t >( static_cast< _this_t const& >(
                *this ).clElemByElemConfigBuffer() );
    }

    SIXTRL_HOST_FN TrackJobCl::cl_buffer_t const*
    TrackJobCl::ptrClElemByElemConfigBuffer() const SIXTRL_NOEXCEPT
    {
        return this->m_ptr_cl_elem_by_elem_config_buffer.get();
    }

    SIXTRL_HOST_FN TrackJobCl::cl_buffer_t*
    TrackJobCl::ptrClElemByElemConfigBuffer() SIXTRL_NOEXCEPT
    {
        return this->m_ptr_cl_elem_by_elem_config_buffer.get();
    }


    SIXTRL_HOST_FN bool TrackJobCl::doPrepareParticlesStructures(
        TrackJobCl::c_buffer_t* SIXTRL_RESTRICT pb )
    {
        return ( ( TrackJobBase::doPrepareParticlesStructures( pb ) ) &&
                 ( this->doPrepareParticlesStructuresOclImp( pb ) ) );
    }

    SIXTRL_HOST_FN bool TrackJobCl::doPrepareBeamElementsStructures(
        TrackJobCl::c_buffer_t* SIXTRL_RESTRICT belems )
    {
        return ( ( TrackJobBase::doPrepareBeamElementsStructures( belems ) ) &&
            ( this->doPrepareBeamElementsStructuresOclImp( belems ) ) );
    }

    SIXTRL_HOST_FN bool TrackJobCl::doPrepareOutputStructures(
        TrackJobCl::c_buffer_t* SIXTRL_RESTRICT part_buffer,
        TrackJobCl::c_buffer_t* SIXTRL_RESTRICT belem_buffer,
        TrackJobCl::c_buffer_t* SIXTRL_RESTRICT output_buffer,
        TrackJobCl::size_type const dump_elem_by_elem_turns )
    {
        bool success = TrackJobBase::doPrepareOutputStructures( part_buffer,
                belem_buffer, output_buffer, dump_elem_by_elem_turns );

        if( ( success ) && ( this->hasOutputBuffer() ) )
        {
            success = this->doPrepareOutputStructuresOclImpl(
                part_buffer, belem_buffer, this->ptrCOutputBuffer(),
                dump_elem_by_elem_turns );
        }

        return success;
    }

    SIXTRL_HOST_FN bool TrackJobCl::doAssignOutputBufferToBeamMonitors(
        TrackJobCl::c_buffer_t* SIXTRL_RESTRICT belem_buffer,
        TrackJobCl::c_buffer_t* SIXTRL_RESTRICT out_buffer )
    {
        bool success = TrackJobBase::doAssignOutputBufferToBeamMonitors(
                belem_buffer, out_buffer );

        if( ( success ) &&
            ( ( this->hasElemByElemOutput() ) ||
              ( this->hasBeamMonitorOutput() ) ) )
        {
            success = this->doAssignOutputBufferToBeamMonitorsOclImp(
                belem_buffer, out_buffer );
        }

        return success;
    }

    SIXTRL_HOST_FN bool TrackJobCl::doReset(
        TrackJobCl::c_buffer_t* SIXTRL_RESTRICT particles_buffer,
        TrackJobCl::c_buffer_t* SIXTRL_RESTRICT beam_elem_buffer,
        TrackJobCl::c_buffer_t* SIXTRL_RESTRICT ptr_output_buffer,
        TrackJobCl::size_type const dump_elem_by_elem_turns )
    {
        return this->doResetOclImp( particles_buffer, beam_elem_buffer,
                ptr_output_buffer, dump_elem_by_elem_turns );
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

    SIXTRL_HOST_FN void TrackJobCl::doUpdateStoredClElemByElemConfigBuffer(
            TrackJobCl::ptr_cl_buffer_t&& cl_elem_by_elem_config_buffer )
    {
        this->m_ptr_cl_elem_by_elem_config_buffer =
            std::move( cl_elem_by_elem_config_buffer );

        return;
    }

    SIXTRL_HOST_FN TrackJobCl::size_type
    TrackJobCl::totalNumParticles() const SIXTRL_NOEXCEPT
    {
        return this->m_total_num_particles;
    }

    SIXTRL_HOST_FN void TrackJobCl::doSetTotalNumParticles(
        TrackJobCl::size_type const total_num_particles ) SIXTRL_NOEXCEPT
    {
        this->m_total_num_particles = total_num_particles;
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

        using arg_t        = _this_t::cl_arg_t;
        using ptr_arg_t    = _this_t::ptr_cl_arg_t;
        using size_t       = _this_t::size_type;

        bool success = false;

        if( ( this->ptrContext() != nullptr ) && ( pb != nullptr ) )
        {
            size_t const total_num_particles =
            ::NS(Particles_buffer_get_total_num_of_particles_on_particle_sets)(
                pb, this->numParticleSets(), this->particleSetIndicesBegin() );

            this->doSetTotalNumParticles( total_num_particles );

            ptr_arg_t particles_arg( new arg_t( pb, this->ptrContext() ) );
            this->doUpdateStoredParticlesArg( std::move( particles_arg ) );

            if( ( total_num_particles > size_t{ 0 } ) &&
                ( this->ptrParticlesArg() != nullptr ) &&
                ( this->particlesArg().usesCObjectBuffer() ) &&
                ( this->particlesArg().context() == this->ptrContext() ) &&
                ( this->particlesArg().ptrCObjectBuffer() == pb ) &&
                ( ( !this->context().hasSelectedNode() ) ||
                  (  this->context().assignParticleArg(
                        this->particlesArg() ) ) ) )
            {
                success = true;
            }
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
                ( this->beamElementsArg().ptrCObjectBuffer() == belems ) &&
                ( ( !this->context().hasSelectedNode() ) ||
                  (  this->context().assignBeamElementsArg(
                         this->beamElementsArg() ) ) ) );
        }

        return success;
    }

    SIXTRL_HOST_FN bool TrackJobCl::doPrepareOutputStructuresOclImpl(
        TrackJobCl::c_buffer_t* SIXTRL_RESTRICT pb,
        TrackJobCl::c_buffer_t* SIXTRL_RESTRICT belems,
        TrackJobCl::c_buffer_t* SIXTRL_RESTRICT ptr_out_buffer,
        TrackJobCl::size_type const dump_elem_by_elem_turns )
    {
        using _this_t = TrackJobCl;

        using arg_t         = _this_t::cl_arg_t;
        using size_t        = _this_t::size_type;
        using cl_buffer_t   = _this_t::cl_buffer_t;
        using ptr_arg_t     = _this_t::ptr_cl_arg_t;
        using ptr_buffer_t  = _this_t::ptr_cl_buffer_t;
        using elem_config_t = _this_t::elem_by_elem_config_t;

        bool success = true;
        ( void )dump_elem_by_elem_turns;

        if( ( ptr_out_buffer != nullptr ) && ( pb != nullptr ) )
        {
            cl::CommandQueue* ptr_queue = ( this->ptrContext() != nullptr )
                ? this->context().openClQueue() : nullptr;

            ptr_arg_t ptr( new arg_t( ptr_out_buffer, this->ptrContext() ) );
            this->doUpdateStoredOutputArg( std::move( ptr ) );

            success &= ( this->ptrOutputBufferArg() != nullptr );

            if( ( success ) && ( this->ptrContext() != nullptr ) &&
                ( this->context().hasSelectedNode() ) )
            {
                success &= this->context().assignOutputBufferArg(
                    this->outputBufferArg() );
            }

            if( ( success ) && ( this->ptrContext()   != nullptr ) &&
                ( this->ptrContext()->openClContext() != nullptr ) &&
                ( ptr_queue != nullptr ) &&
                ( this->ptrElemByElemConfig() != nullptr ) )
            {
                ptr_buffer_t ptr_buffer( new cl_buffer_t(
                    *( this->ptrContext()->openClContext() ), CL_MEM_READ_WRITE,
                    sizeof( elem_config_t ), nullptr ) );

                this->doUpdateStoredClElemByElemConfigBuffer(
                    std::move( ptr_buffer ) );

                success = ( this->ptrClElemByElemConfigBuffer() != nullptr );

                if( success )
                {
                    cl_buffer_t& conf = this->clElemByElemConfigBuffer();

                    cl_int const ret = ptr_queue->enqueueWriteBuffer(
                        conf, CL_TRUE, size_t{ 0 }, sizeof( elem_config_t ),
                            this->ptrElemByElemConfig() );

                    success = ( ( ret == CL_SUCCESS ) &&
                        ( ( !this->context().hasSelectedNode() ) ||
                          ( ( this->context().assignElemByElemConfigBuffer(
                                conf ) ) &&
                            ( this->context().assignElemByElemBufferOffset(
                                this->elemByElemOutputBufferOffset() ) ) ) ) );
                }
            }
        }

        return success;
    }

    SIXTRL_HOST_FN bool TrackJobCl::doAssignOutputBufferToBeamMonitorsOclImp(
        TrackJobCl::c_buffer_t* SIXTRL_RESTRICT belem_buffer,
        TrackJobCl::c_buffer_t* SIXTRL_RESTRICT output_buffer )
    {
        if( ( this->ptrContext() != nullptr ) &&
            ( this->context().hasSelectedNode() ) &&
            ( this->context().hasAssignBeamMonitorIoBufferKernel() ) &&
            ( this->hasBeamMonitorOutput() ) &&
            ( this->ptrBeamElementsArg() != nullptr ) &&
            ( this->beamElementsArg().context() == this->ptrContext() ) &&
            ( this->beamElementsArg().usesCObjectBuffer() ) &&
            ( this->beamElementsArg().ptrCObjectBuffer() == belem_buffer ) &&
            ( this->ptrOutputBufferArg() != nullptr ) &&
            ( this->outputBufferArg().context() == this->ptrContext() ) &&
            ( this->outputBufferArg().usesCObjectBuffer() ) &&
            ( this->outputBufferArg().ptrCObjectBuffer() == output_buffer ) )
        {
            using size_t = TrackJobCl::size_type;

            size_t const beam_monitor_buffer_offset =
                this->beamMonitorsOutputBufferOffset();

            return ( 0 == this->context().assignBeamMonitorIoBuffer(
                this->beamElementsArg(), this->outputBufferArg(),
                this->minInitialTurnId(), beam_monitor_buffer_offset ) );
        }

        return false;
    }

    SIXTRL_HOST_FN bool TrackJobCl::doResetOclImp(
        TrackJobCl::c_buffer_t* SIXTRL_RESTRICT part_buffer,
        TrackJobCl::c_buffer_t* SIXTRL_RESTRICT belem_buffer,
        TrackJobCl::c_buffer_t* SIXTRL_RESTRICT out_buffer,
        TrackJobCl::size_type const dump_elem_by_elem_turns )
    {
        bool success = false;
        using flag_t = TrackJobBase::track_job_output_flag_t;

        if( ( _base_t::doPrepareParticlesStructures( part_buffer ) ) &&
            ( _base_t::doPrepareBeamElementsStructures( belem_buffer ) ) &&
            ( this->doPrepareParticlesStructuresOclImp( part_buffer ) ) &&
            ( this->doPrepareBeamElementsStructuresOclImp( belem_buffer ) ) )
        {
            flag_t const needs_output = ::NS(TrackJob_needs_output_buffer)(
                part_buffer, belem_buffer, dump_elem_by_elem_turns );

            this->doSetPtrCParticleBuffer( part_buffer );
            this->doSetPtrCBeamElementsBuffer( belem_buffer );

            if( ( needs_output != ::NS(TRACK_JOB_OUTPUT_NONE) ) ||
                ( out_buffer   != nullptr ) )
            {
                success = ( this->doPrepareOutputStructures( part_buffer,
                    belem_buffer, out_buffer, dump_elem_by_elem_turns ) );

                if( ( success ) && ( this->hasOutputBuffer() ) &&
                    ( needs_output != ::NS(TRACK_JOB_OUTPUT_NONE) ) )
                {
                    success = _base_t::doAssignOutputBufferToBeamMonitors(
                            belem_buffer, this->ptrCOutputBuffer() );
                }
                else if( ( success ) &&
                    ( out_buffer != nullptr ) && ( !this->ownsOutputBuffer() ) )
                {
                    this->doSetPtrCOutputBuffer( out_buffer );
                }
            }
            else
            {
                success = true;
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
            job.particlesArg().read( job.ptrCParticlesBuffer() );
        }

        if( ( job.ptrOutputBufferArg() != nullptr ) &&
            ( job.ptrCOutputBuffer()   != nullptr ) )
        {
            job.outputBufferArg().read( job.ptrCOutputBuffer() );
        }

        if( ( job.ptrBeamElementsArg() != nullptr ) &&
            ( job.ptrCBeamElementsBuffer() != nullptr ) )
        {
            job.beamElementsArg().read( job.ptrCBeamElementsBuffer() );
        }

        return;
    }

    SIXTRL_HOST_FN TrackJobCl::track_status_t track(
        TrackJobCl& SIXTRL_RESTRICT_REF job,
        TrackJobCl::size_type const until_turn ) SIXTRL_NOEXCEPT
    {
        using status_t    = TrackJobCl::track_status_t;
        using kernel_id_t = TrackJobCl::cl_context_t::kernel_id_t;

        SIXTRL_ASSERT( job.ptrContext() != nullptr );
        SIXTRL_ASSERT( job.ptrParticlesArg() != nullptr );
        SIXTRL_ASSERT( job.ptrBeamElementsArg() != nullptr );
        SIXTRL_ASSERT( job.ptrContext()->hasSelectedNode() );
        SIXTRL_ASSERT( job.context().hasTrackingKernel() );
        SIXTRL_ASSERT( job.totalNumParticles() > TrackJobCl::size_type{ 0 } );

        kernel_id_t const  kid = job.context().trackingKernelId();
        int64_t const until_turn_arg = until_turn;

        job.context().assignKernelArgumentValue( kid, 2u, until_turn_arg );

        return ( job.context().runKernel( kid, job.totalNumParticles() ) )
            ? status_t{ 0 } : status_t{ -1 };
    }

    SIXTRL_HOST_FN TrackJobCl::track_status_t trackElemByElem(
        TrackJobCl& SIXTRL_RESTRICT_REF job,
        TrackJobCl::size_type const until_turn ) SIXTRL_NOEXCEPT
    {
        using status_t    = TrackJobCl::track_status_t;
        using kernel_id_t = TrackJobCl::cl_context_t::kernel_id_t;

        SIXTRL_ASSERT( job.ptrContext() != nullptr );
        SIXTRL_ASSERT( job.ptrParticlesArg() != nullptr );
        SIXTRL_ASSERT( job.ptrBeamElementsArg() != nullptr );
        SIXTRL_ASSERT( job.hasOutputBuffer() );
        SIXTRL_ASSERT( job.ptrOutputBufferArg() != nullptr );
        SIXTRL_ASSERT( job.hasElemByElemOutput() );
        SIXTRL_ASSERT( job.ptrElemByElemConfig() != nullptr );
        SIXTRL_ASSERT( job.ptrClElemByElemConfigBuffer() != nullptr );
        SIXTRL_ASSERT( job.ptrContext()->hasSelectedNode() );
        SIXTRL_ASSERT( job.context().hasElementByElementTrackingKernel() );
        SIXTRL_ASSERT( job.totalNumParticles() > TrackJobCl::size_type{ 0 } );

        kernel_id_t const  kid =
            job.context().elementByElementTrackingKernelId();

        int64_t const until_turn_arg = until_turn;

        job.context().assignKernelArgumentValue( kid, 4u, until_turn_arg );

        return ( job.context().runKernel( kid, job.totalNumParticles() ) )
            ? status_t{ 0 } : status_t{ -1 };
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
    NS(Buffer)* SIXTRL_RESTRICT beam_elements_buffer )
{
    return new SIXTRL_CXX_NAMESPACE::TrackJobCl( device_id_str,
        particles_buffer, beam_elements_buffer );
}

SIXTRL_HOST_FN NS(TrackJobCl)* NS(TrackJobCl_new_with_output)(
    const char *const SIXTRL_RESTRICT device_id_str,
    NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
    NS(Buffer)* SIXTRL_RESTRICT beam_elements_buffer,
    NS(Buffer)* SIXTRL_RESTRICT output_buffer,
    NS(buffer_size_t) const dump_elem_by_elem_turns )
{
    using size_t = SIXTRL_CXX_NAMESPACE::TrackJobCl::size_type;

    size_t const particle_set_indices[] = { size_t{ 0 }, size_t{ 0 } };

    return new SIXTRL_CXX_NAMESPACE::TrackJobCl( device_id_str,
        particles_buffer, size_t{ 1 }, &particle_set_indices[ 0 ],
        beam_elements_buffer, output_buffer, dump_elem_by_elem_turns );
}

SIXTRL_HOST_FN NS(TrackJobCl)* NS(TrackJobCl_new_detailed)(
    const char *const SIXTRL_RESTRICT device_id_str,
    NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
    NS(buffer_size_t) const num_particle_sets,
    NS(buffer_size_t) const* SIXTRL_RESTRICT particle_set_indices_begin,
    NS(Buffer)* SIXTRL_RESTRICT beam_elements_buffer,
    NS(Buffer)* SIXTRL_RESTRICT output_buffer,
    NS(buffer_size_t) const dump_elem_by_elem_turns,
    const char *const SIXTRL_RESTRICT config_str )
{
    return new SIXTRL_CXX_NAMESPACE::TrackJobCl( device_id_str,
        particles_buffer, num_particle_sets, particle_set_indices_begin,
        beam_elements_buffer, output_buffer, dump_elem_by_elem_turns,
        config_str );
}

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(TrackJobCl_reset)(
    NS(TrackJobCl)* SIXTRL_RESTRICT job,
    NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
    NS(Buffer)* SIXTRL_RESTRICT beam_elements_buffer,
    NS(Buffer)* SIXTRL_RESTRICT output_buffer )
{
    if( job != nullptr )
    {
        SIXTRL_CXX_NAMESPACE::TrackJobCl::size_type const
            dump_elem_by_elem_turns = job->numElemByElemTurns();

        return job->reset( particles_buffer, beam_elements_buffer,
                           output_buffer, dump_elem_by_elem_turns );
    }

    return false;
}

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(TrackJobCl_reset_with_output)(
    NS(TrackJobCl)* SIXTRL_RESTRICT job,
    NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
    NS(Buffer)* SIXTRL_RESTRICT beam_elements_buffer,
    NS(Buffer)* SIXTRL_RESTRICT output_buffer,
    NS(buffer_size_t) const dump_elem_by_elem_turns )
{
    return ( job != nullptr )
        ? job->reset( particles_buffer, beam_elements_buffer, output_buffer,
                      dump_elem_by_elem_turns )
        : false;
}

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(TrackJobCl_reset_detailed)(
    NS(TrackJobCl)* SIXTRL_RESTRICT job,
    NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
    NS(buffer_size_t) const num_particle_sets,
    NS(buffer_size_t) const* SIXTRL_RESTRICT pset_begin,
    NS(Buffer)* SIXTRL_RESTRICT beam_elements_buffer,
    NS(Buffer)* SIXTRL_RESTRICT output_buffer,
    NS(buffer_size_t) const dump_elem_by_elem_turns )
{
    return ( job != nullptr )
        ? job->reset( particles_buffer, num_particle_sets, pset_begin,
                      beam_elements_buffer, output_buffer,
                      dump_elem_by_elem_turns )
        : false;
}

SIXTRL_HOST_FN void NS(TrackJobCl_delete)(
    NS(TrackJobCl)* SIXTRL_RESTRICT track_job )
{
    delete track_job;
}

SIXTRL_HOST_FN NS(track_status_t) NS(TrackJobCl_track_until_turn)(
    NS(TrackJobCl)* SIXTRL_RESTRICT track_job,
    NS(buffer_size_t) const until_turn )
{
    SIXTRL_ASSERT( track_job != nullptr );
    return SIXTRL_CXX_NAMESPACE::track( *track_job, until_turn );
}

SIXTRL_HOST_FN NS(track_status_t) NS(TrackJobCl_track_elem_by_elem)(
    NS(TrackJobCl)* SIXTRL_RESTRICT track_job,
    NS(buffer_size_t) const until_turn )
{
    SIXTRL_ASSERT( track_job != nullptr );
    return SIXTRL_CXX_NAMESPACE::trackElemByElem( *track_job, until_turn );
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

SIXTRL_EXTERN SIXTRL_HOST_FN NS(ClArgument)*
NS(TrackJobCl_get_particles_buffer_arg)(
    NS(TrackJobCl)* SIXTRL_RESTRICT track_job )
{
    return ( track_job != nullptr ) ? track_job->ptrParticlesArg() : nullptr;
}

SIXTRL_EXTERN SIXTRL_HOST_FN NS(ClArgument) const*
NS(TrackJobCl_get_const_particles_buffer_arg)(
    const NS(TrackJobCl) *const SIXTRL_RESTRICT track_job )
{
    return ( track_job != nullptr ) ? track_job->ptrParticlesArg() : nullptr;
}

SIXTRL_EXTERN SIXTRL_HOST_FN NS(ClArgument)*
NS(TrackJobCl_get_beam_elements_buffer_arg)(
    NS(TrackJobCl)* SIXTRL_RESTRICT job )
{
    return ( job != nullptr ) ? job->ptrBeamElementsArg() : nullptr;
}

SIXTRL_EXTERN SIXTRL_HOST_FN NS(ClArgument) const*
NS(TrackJobCl_get_const_beam_elements_buffer_arg)(
    const NS(TrackJobCl) *const SIXTRL_RESTRICT job )
{
    return ( job != nullptr ) ? job->ptrBeamElementsArg() : nullptr;
}

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(TrackJobCl_has_output_buffer_arg)(
    const NS(TrackJobCl) *const SIXTRL_RESTRICT job )
{
    return ( ( job != nullptr ) && ( job->ptrOutputBufferArg() != nullptr ) );
}

SIXTRL_EXTERN SIXTRL_HOST_FN NS(ClArgument)*
NS(TrackJobCl_get_output_buffer_arg)( NS(TrackJobCl)* SIXTRL_RESTRICT job )
{
    return ( job != nullptr ) ? job->ptrOutputBufferArg() : nullptr;
}

SIXTRL_EXTERN SIXTRL_HOST_FN NS(ClArgument) const*
NS(TrackJobCl_get_const_output_buffer_arg)(
    const NS(TrackJobCl) *const SIXTRL_RESTRICT job )
{
    return ( job != nullptr ) ? job->ptrOutputBufferArg() : nullptr;
}

/* end: /opencl/internal/track_job_cl.cpp */
