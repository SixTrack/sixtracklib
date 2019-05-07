#include "sixtracklib/cuda/track_job.hpp"


#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #if defined( __cplusplus ) && !defined( _GPUCODE ) && \
       !defined( __CUDA_ARCH__ )
        #include <cstddef>
        #include <cstdlib>

        #include <cuda_runtime_api.h>

    #endif /* C++, Host */
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/control/definitions.h"
    #include "sixtracklib/common/control/arch_base.hpp"
    #include "sixtracklib/common/control/kernel_config_base.hpp"
    #include "sixtracklib/common/internal/track_job_base.h"
    #if defined( __cplusplus ) && !defined( _GPUCODE ) && \
       !defined( __CUDA_ARCH__ )
        #include "sixtracklib/common/buffer.hpp"
    #endif /* C++, Host */
    #include "sixtracklib/common/buffer.h"

    #include "sixtracklib/cuda/definitions.h"
    #include "sixtracklib/cuda/controller.hpp"
    #include "sixtracklib/cuda/argument.hpp"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus ) && !defined( _GPUCODE ) && !defined( __CUDA_ARCH__ )

namespace SIXTRL_CXX_NAMESPACE
{
    TrackJobCuda::TrackJobCuda(
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

    TrackJobCuda::TrackJobCuda(
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

    TrackJobCuda::TrackJobCuda(
        const char *const SIXTRL_RESTRICT device_id_str,
        TrackJobCuda::c_buffer_t* SIXTRL_RESTRICT particles_buffer,
        TrackJobCuda::c_buffer_t* SIXTRL_RESTRICT beam_elements_buffer,
        TrackJobCuda::c_buffer_t* SIXTRL_RESTRICT ptr_output_buffer,
        TrackJobCuda::size_type const until_turn_elem_by_elem,
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
        this->doInitTrackJobCuda( device_id_str, particles_buffer,
            this->particleSetIndicesBegin(), this->particleSetIndicesEnd(),
            beam_elements_buffer, ptr_output_buffer, until_turn_elem_by_elem,
            config_str );
    }

    TrackJobCuda::TrackJobCuda(
        const char *const SIXTRL_RESTRICT device_id_str,
        TrackJobCuda::c_buffer_t* SIXTRL_RESTRICT particles_buffer,
        TrackJobCuda::size_type const num_particle_sets,
        TrackJobCuda::size_type const* SIXTRL_RESTRICT pset_begin,
        TrackJobCuda::c_buffer_t* SIXTRL_RESTRICT belements_buffer,
        TrackJobCuda::c_buffer_t* SIXTRL_RESTRICT ptr_output_buffer,
        TrackJobCuda::size_type const until_turn_elem_by_elem,
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
        using size_t  = TrackJobCuda::size_type;
        size_t const* pset_end = pset_begin;

        if( ( pset_end != nullptr ) && ( num_particle_sets > size_t{ 0 } ) )
        {
            std::advance( pset_end, num_particle_sets );
        }

        this->doInitTrackJobCuda( device_id_str, particles_buffer,
            pset_begin, pset_end, belements_buffer, ptr_output_buffer,
            until_turn_elem_by_elem, config_str );
    }

    TrackJobCuda::TrackJobCuda(
        std::string const& SIXTRL_RESTRICT_REF device_id_str,
        TrackJobCuda::buffer_t& SIXTRL_RESTRICT_REF particles_buffer,
        TrackJobCuda::buffer_t& SIXTRL_RESTRICT_REF belements_buffer,
        TrackJobCuda::buffer_t* SIXTRL_RESTRICT ptr_output_buffer,
        TrackJobCuda::size_type const until_turn_elem_by_elem,
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
        using c_buffer_t = TrackJobCuda::c_buffer_t;

        c_buffer_t* ptr_part_buffer  = particles_buffer.getCApiPtr();
        c_buffer_t* ptr_belem_buffer = belements_buffer.getCApiPtr();
        c_buffer_t* ptr_out_buffer   = ( ptr_output_buffer != nullptr )
            ? ptr_output_buffer->getCApiPtr() : nullptr;

        bool const success = this->doInitTrackJobCuda(
            device_id_str.c_str(), ptr_part_buffer,
            this->particleSetIndicesBegin(), this->particleSetIndicesEnd(),
            ptr_belem_buffer, ptr_out_buffer, until_turn_elem_by_elem,
                config_str.c_str() );

        if( success )
        {
            this->doSetPtrParticleBuffer( &particles_buffer );
            this->doSetPtrBeamElementsBuffer( &belements_buffer );

            if( ( ptr_out_buffer != nullptr ) && ( this->hasOutputBuffer() ) )
            {
                this->doSetPtrOutputBuffer( ptr_output_buffer );
            }
        }
    }

    TrackJobCuda::~TrackJobCuda() SIXTRL_NOEXCEPT {}

    TrackJobCuda::cl_context_t&
    TrackJobCuda::context() SIXTRL_RESTRICT
    {
        return const_cast< TrackJobCuda::cl_context_t& >(
            static_cast< TrackJobCuda const& >( *this ).context() );
    }


    TrackJobCuda::cl_context_t const&
    TrackJobCuda::context() const SIXTRL_RESTRICT
    {
        SIXTRL_ASSERT( this->ptrContext() != nullptr );
        return *( this->ptrContext() );
    }

    ::NS(ClContext)* TrackJobCuda::ptrContext() SIXTRL_RESTRICT
    {
        return this->m_ptr_context.get();
    }

    ::NS(ClContext) const*
    TrackJobCuda::ptrContext() const SIXTRL_RESTRICT
    {
        return this->m_ptr_context.get();
    }

    TrackJobCuda::cl_arg_t&
    TrackJobCuda::particlesArg() SIXTRL_NOEXCEPT
    {
        return const_cast< TrackJobCuda::cl_arg_t& >(
            static_cast< TrackJobCuda const& >( *this ).particlesArg() );
    }

    TrackJobCuda::cl_arg_t const&
    TrackJobCuda::particlesArg() const SIXTRL_NOEXCEPT
    {
        SIXTRL_ASSERT( this->ptrParticlesArg() != nullptr );
        return *(this->ptrParticlesArg() );
    }

    TrackJobCuda::cl_arg_t*
    TrackJobCuda::ptrParticlesArg() SIXTRL_NOEXCEPT
    {
        return const_cast< TrackJobCuda::cl_arg_t* >( static_cast<
            TrackJobCuda const& >( *this ).ptrParticlesArg() );
    }

    TrackJobCuda::cl_arg_t const*
    TrackJobCuda::ptrParticlesArg() const SIXTRL_NOEXCEPT
    {
        return this->m_ptr_particles_buffer_arg.get();
    }

    TrackJobCuda::cl_arg_t&
    TrackJobCuda::beamElementsArg() SIXTRL_NOEXCEPT
    {
        return const_cast< TrackJobCuda::cl_arg_t& >( static_cast<
            TrackJobCuda const& >( *this ).beamElementsArg() );
    }

    TrackJobCuda::cl_arg_t const&
    TrackJobCuda::beamElementsArg() const SIXTRL_NOEXCEPT
    {
        SIXTRL_ASSERT( this->ptrBeamElementsArg() != nullptr );
        return *( this->ptrBeamElementsArg() );
    }

    TrackJobCuda::cl_arg_t*
    TrackJobCuda::ptrBeamElementsArg() SIXTRL_NOEXCEPT
    {
        return const_cast< TrackJobCuda::cl_arg_t* >( static_cast<
            TrackJobCuda const& >( *this ).ptrBeamElementsArg() );
    }

    TrackJobCuda::cl_arg_t const*
    TrackJobCuda::ptrBeamElementsArg() const SIXTRL_NOEXCEPT
    {
        return this->m_ptr_beam_elements_buffer_arg.get();
    }

    TrackJobCuda::cl_arg_t&
    TrackJobCuda::outputBufferArg() SIXTRL_NOEXCEPT
    {
        return const_cast< TrackJobCuda::cl_arg_t& >( static_cast<
            TrackJobCuda const& >( *this ).outputBufferArg() );
    }

    TrackJobCuda::cl_arg_t const&
    TrackJobCuda::outputBufferArg() const SIXTRL_NOEXCEPT
    {
        SIXTRL_ASSERT( this->ptrOutputBufferArg() != nullptr );
        return *( this->m_ptr_output_buffer_arg );
    }

    TrackJobCuda::cl_arg_t*
    TrackJobCuda::ptrOutputBufferArg() SIXTRL_NOEXCEPT
    {
        return const_cast< TrackJobCuda::cl_arg_t* >( static_cast<
            TrackJobCuda const& >( *this ).ptrOutputBufferArg() );
    }

    TrackJobCuda::cl_arg_t const*
    TrackJobCuda::ptrOutputBufferArg() const SIXTRL_NOEXCEPT
    {
        return this->m_ptr_output_buffer_arg.get();
    }


    TrackJobCuda::cl_buffer_t const&
    TrackJobCuda::clElemByElemConfigBuffer() const SIXTRL_NOEXCEPT
    {
        SIXTRL_ASSERT( this->ptrClElemByElemConfigBuffer() != nullptr );
        return *( this->ptrClElemByElemConfigBuffer() );
    }

    TrackJobCuda::cl_buffer_t&
    TrackJobCuda::clElemByElemConfigBuffer() SIXTRL_NOEXCEPT
    {
        using _this_t = TrackJobCuda;
        using ref_t   = _this_t::cl_buffer_t&;

        return const_cast< ref_t >( static_cast< _this_t const& >(
                *this ).clElemByElemConfigBuffer() );
    }

    TrackJobCuda::cl_buffer_t const*
    TrackJobCuda::ptrClElemByElemConfigBuffer() const SIXTRL_NOEXCEPT
    {
        return this->m_ptr_cl_elem_by_elem_config_buffer.get();
    }

    TrackJobCuda::cl_buffer_t*
    TrackJobCuda::ptrClElemByElemConfigBuffer() SIXTRL_NOEXCEPT
    {
        return this->m_ptr_cl_elem_by_elem_config_buffer.get();
    }


    bool TrackJobCuda::doPrepareParticlesStructures(
        TrackJobCuda::c_buffer_t* SIXTRL_RESTRICT pb )
    {
        return ( ( TrackJobBase::doPrepareParticlesStructures( pb ) ) &&
                 ( this->doPrepareParticlesStructuresOclImp( pb ) ) );
    }

    bool TrackJobCuda::doPrepareBeamElementsStructures(
        TrackJobCuda::c_buffer_t* SIXTRL_RESTRICT belems )
    {
        return ( ( TrackJobBase::doPrepareBeamElementsStructures( belems ) ) &&
            ( this->doPrepareBeamElementsStructuresOclImp( belems ) ) );
    }

    bool TrackJobCuda::doPrepareOutputStructures(
        TrackJobCuda::c_buffer_t* SIXTRL_RESTRICT part_buffer,
        TrackJobCuda::c_buffer_t* SIXTRL_RESTRICT belem_buffer,
        TrackJobCuda::c_buffer_t* SIXTRL_RESTRICT output_buffer,
        TrackJobCuda::size_type const until_turn_elem_by_elem )
    {
        bool success = TrackJobBase::doPrepareOutputStructures( part_buffer,
                belem_buffer, output_buffer, until_turn_elem_by_elem );

        if( ( success ) && ( this->hasOutputBuffer() ) )
        {
            success = this->doPrepareOutputStructuresOclImpl(
                part_buffer, belem_buffer, this->ptrCOutputBuffer(),
                until_turn_elem_by_elem );
        }

        return success;
    }

    bool TrackJobCuda::doAssignOutputBufferToBeamMonitors(
        TrackJobCuda::c_buffer_t* SIXTRL_RESTRICT belem_buffer,
        TrackJobCuda::c_buffer_t* SIXTRL_RESTRICT out_buffer )
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

    bool TrackJobCuda::doReset(
        TrackJobCuda::c_buffer_t* SIXTRL_RESTRICT particles_buffer,
        TrackJobCuda::c_buffer_t* SIXTRL_RESTRICT beam_elem_buffer,
        TrackJobCuda::c_buffer_t* SIXTRL_RESTRICT ptr_output_buffer,
        TrackJobCuda::size_type const until_turn_elem_by_elem )
    {
        return this->doResetOclImp( particles_buffer, beam_elem_buffer,
                ptr_output_buffer, until_turn_elem_by_elem );
    }

    TrackJobCuda::track_status_t TrackJobCuda::doTrackUntilTurn(
        TrackJobCuda::size_type const until_turn )
    {
        return SIXTRL_CXX_NAMESPACE::track( *this, until_turn );
    }

    TrackJobCuda::track_status_t TrackJobCuda::doTrackElemByElem(
        TrackJobCuda::size_type const until_turn )
    {
        return SIXTRL_CXX_NAMESPACE::trackElemByElem( *this, until_turn );
    }

    TrackJobCuda::track_status_t TrackJobCuda::doTrackLine(
        TrackJobCuda::size_type const line_begin_idx,
        TrackJobCuda::size_type const line_end_idx,
        bool const finish_turn )
    {
        return SIXTRL_CXX_NAMESPACE::trackLine(
            *this, line_begin_idx, line_end_idx, finish_turn );
    }


    TrackJobCuda::collect_flag_t
    TrackJobCuda::doCollect( TrackJobCuda::collect_flag_t const flags )
    {
        return SIXTRL_CXX_NAMESPACE::collect( *this, flags );
    }

    bool TrackJobCuda::doParseConfigStr( const char *const SIXTRL_RESTRICT  )
    {
        return true;
    }

    void TrackJobCuda::doUpdateStoredContext(
        TrackJobCuda::ptr_cl_context_t&& context )
    {
        this->m_ptr_context = std::move( context );
        return;
    }

    void TrackJobCuda::doUpdateStoredParticlesArg(
        TrackJobCuda::ptr_cl_arg_t&& particle_arg )
    {
        this->m_ptr_particles_buffer_arg = std::move( particle_arg );
        return;
    }

    void TrackJobCuda::doUpdateStoredBeamElementsArg(
        TrackJobCuda::ptr_cl_arg_t&& beam_elements_arg )
    {
        this->m_ptr_beam_elements_buffer_arg = std::move( beam_elements_arg );
        return;
    }

    void TrackJobCuda::doUpdateStoredOutputArg(
        TrackJobCuda::ptr_cl_arg_t&& output_arg )
    {
        this->m_ptr_output_buffer_arg = std::move( output_arg );
        return;
    }

    void TrackJobCuda::doUpdateStoredClElemByElemConfigBuffer(
            TrackJobCuda::ptr_cl_buffer_t&& cl_elem_by_elem_config_buffer )
    {
        this->m_ptr_cl_elem_by_elem_config_buffer =
            std::move( cl_elem_by_elem_config_buffer );

        return;
    }

    TrackJobCuda::size_type TrackJobCuda::totalNumParticles() const SIXTRL_NOEXCEPT
    {
        return this->m_total_num_particles;
    }

    void TrackJobCuda::doSetTotalNumParticles(
        TrackJobCuda::size_type const total_num_particles ) SIXTRL_NOEXCEPT
    {
        this->m_total_num_particles = total_num_particles;
        return;
    }

    bool TrackJobCuda::doPrepareContext(
        char const* SIXTRL_RESTRICT device_id_str,
        char const* SIXTRL_RESTRICT ptr_config_str )
    {
        return this->doPrepareContextOclImpl( device_id_str, ptr_config_str );
    }

    bool TrackJobCuda::doPrepareContextOclImpl(
        const char *const SIXTRL_RESTRICT device_id_str,
        const char *const SIXTRL_RESTRICT ptr_config_str )
    {
        using _this_t       = TrackJobCuda;
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

    void TrackJobCuda::doParseConfigStrOclImpl(
        const char *const SIXTRL_RESTRICT )
    {
        return;
    }

    bool TrackJobCuda::doPrepareParticlesStructuresOclImp(
        TrackJobCuda::c_buffer_t* SIXTRL_RESTRICT pb )
    {
        using _this_t = TrackJobCuda;

        using arg_t        = _this_t::cl_arg_t;
        using ptr_arg_t    = _this_t::ptr_cl_arg_t;
        using size_t       = _this_t::size_type;

        bool success = false;

        if( ( this->ptrContext() != nullptr ) && ( pb != nullptr ) )
        {
            SIXTRL_ASSERT( this->particleSetIndicesBegin() != nullptr );
            SIXTRL_ASSERT( this->numParticleSets() == size_t{ 1 } );

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
                        this->particlesArg(),
                        this->particleSetIndicesBegin(),
                        this->particleSetIndicesEnd() ) ) ) )
            {
                success = true;
            }
        }

        return success;
    }

    bool TrackJobCuda::doPrepareBeamElementsStructuresOclImp(
        TrackJobCuda::c_buffer_t* SIXTRL_RESTRICT belems )
    {
        using _this_t = TrackJobCuda;

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

    bool TrackJobCuda::doPrepareOutputStructuresOclImpl(
        TrackJobCuda::c_buffer_t* SIXTRL_RESTRICT pb,
        TrackJobCuda::c_buffer_t* SIXTRL_RESTRICT belems,
        TrackJobCuda::c_buffer_t* SIXTRL_RESTRICT ptr_out_buffer,
        TrackJobCuda::size_type const until_turn_elem_by_elem )
    {
        using _this_t = TrackJobCuda;

        using arg_t         = _this_t::cl_arg_t;
        using size_t        = _this_t::size_type;
        using cl_buffer_t   = _this_t::cl_buffer_t;
        using ptr_arg_t     = _this_t::ptr_cl_arg_t;
        using ptr_buffer_t  = _this_t::ptr_cl_buffer_t;
        using elem_config_t = _this_t::elem_by_elem_config_t;

        bool success = true;
        ( void )until_turn_elem_by_elem;

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

    bool TrackJobCuda::doAssignOutputBufferToBeamMonitorsOclImp(
        TrackJobCuda::c_buffer_t* SIXTRL_RESTRICT belem_buffer,
        TrackJobCuda::c_buffer_t* SIXTRL_RESTRICT output_buffer )
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
            using size_t = TrackJobCuda::size_type;

            size_t const beam_monitor_buffer_offset =
                this->beamMonitorsOutputBufferOffset();

            return ( 0 == this->context().assignBeamMonitorIoBuffer(
                this->beamElementsArg(), this->outputBufferArg(),
                this->minInitialTurnId(), beam_monitor_buffer_offset ) );
        }

        return false;
    }

    bool TrackJobCuda::doResetOclImp(
        TrackJobCuda::c_buffer_t* SIXTRL_RESTRICT pb,
        TrackJobCuda::c_buffer_t* SIXTRL_RESTRICT belem_buffer,
        TrackJobCuda::c_buffer_t* SIXTRL_RESTRICT out_buffer,
        TrackJobCuda::size_type const until_turn_elem_by_elem )
    {
        bool success = false;
        using flags_t = ::NS(output_buffer_flag_t);

        if( ( _base_t::doPrepareParticlesStructures( pb ) ) &&
            ( _base_t::doPrepareBeamElementsStructures( belem_buffer ) ) &&
            ( this->doPrepareParticlesStructuresOclImp( pb ) ) &&
            ( this->doPrepareBeamElementsStructuresOclImp( belem_buffer ) ) )
        {
            flags_t const flags =
            ::NS(OutputBuffer_required_for_tracking_of_particle_sets)( pb,
                this->numParticleSets(), this->particleSetIndicesBegin(),
                    belem_buffer, until_turn_elem_by_elem );

            bool const requires_output_buffer =
                ::NS(OutputBuffer_requires_output_buffer)( flags );

            this->doSetPtrCParticleBuffer( pb );
            this->doSetPtrCBeamElementsBuffer( belem_buffer );

            if( ( requires_output_buffer ) || ( out_buffer != nullptr ) )
            {
                success = ( this->doPrepareOutputStructures( pb,
                    belem_buffer, out_buffer, until_turn_elem_by_elem ) );

                if( ( success ) && ( this->hasOutputBuffer() ) &&
                    ( requires_output_buffer ) )
                {
                    success = _base_t::doAssignOutputBufferToBeamMonitors(
                            belem_buffer, this->ptrCOutputBuffer() );
                }
                else if( ( success ) && ( out_buffer != nullptr ) &&
                         ( !this->ownsOutputBuffer() ) )
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

    void collect(
        TrackJobCuda& SIXTRL_RESTRICT_REF job ) SIXTRL_NOEXCEPT
    {
        SIXTRL_CXX_NAMESPACE::collect( job, job.collectFlags() );
        return;
    }

    void collect( TrackJobCuda& SIXTRL_RESTRICT_REF job,
            SIXTRL_CXX_NAMESPACE::track_job_collect_flag_t const flags
        ) SIXTRL_NOEXCEPT
    {
        if( ( TrackJobCuda::IsCollectFlagSet(
                flags, TRACK_JOB_COLLECT_PARTICLES ) ) &&
            ( job.ptrParticlesArg() != nullptr ) &&
            ( job.ptrCParticlesBuffer() != nullptr ) )
        {
            job.particlesArg().read( job.ptrCParticlesBuffer() );
        }

        if( ( TrackJobCuda::IsCollectFlagSet(
                flags, TRACK_JOB_COLLECT_OUTPUT ) ) &&
            ( job.ptrOutputBufferArg() != nullptr ) &&
            ( job.ptrCOutputBuffer() != nullptr ) )
        {
            job.outputBufferArg().read( job.ptrCOutputBuffer() );
        }

        if( ( TrackJobCuda::IsCollectFlagSet(
                flags, TRACK_JOB_COLLECT_BEAM_ELEMENTS ) ) &&
            ( job.ptrBeamElementsArg() != nullptr ) &&
            ( job.ptrCBeamElementsBuffer() != nullptr ) )
        {
            job.beamElementsArg().read( job.ptrCBeamElementsBuffer() );
        }

        return;
    }

    TrackJobCuda::track_status_t track(
        TrackJobCuda& SIXTRL_RESTRICT_REF job,
        TrackJobCuda::size_type const until_turn ) SIXTRL_NOEXCEPT
    {
        using status_t    = TrackJobCuda::track_status_t;
        using kernel_id_t = TrackJobCuda::cl_context_t::kernel_id_t;

        SIXTRL_ASSERT( job.ptrContext() != nullptr );
        SIXTRL_ASSERT( job.ptrParticlesArg() != nullptr );
        SIXTRL_ASSERT( job.ptrBeamElementsArg() != nullptr );
        SIXTRL_ASSERT( job.ptrContext()->hasSelectedNode() );
        SIXTRL_ASSERT( job.context().hasTrackingKernel() );
        SIXTRL_ASSERT( job.totalNumParticles() > TrackJobCuda::size_type{ 0 } );

        kernel_id_t const  kid = job.context().trackingKernelId();
        int64_t const until_turn_arg = until_turn;

        job.context().assignKernelArgumentValue( kid, 2u, until_turn_arg );

        return ( job.context().runKernel( kid, job.totalNumParticles() ) )
            ? status_t{ 0 } : status_t{ -1 };
    }

    TrackJobCuda::track_status_t trackElemByElem(
        TrackJobCuda& SIXTRL_RESTRICT_REF job,
        TrackJobCuda::size_type const until_turn ) SIXTRL_NOEXCEPT
    {
        using status_t    = TrackJobCuda::track_status_t;
        using kernel_id_t = TrackJobCuda::cl_context_t::kernel_id_t;

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
        SIXTRL_ASSERT( job.totalNumParticles() > TrackJobCuda::size_type{ 0 } );

        kernel_id_t const  kid =
            job.context().elementByElementTrackingKernelId();

        int64_t const until_turn_arg = until_turn;

        job.context().assignKernelArgumentValue( kid, 4u, until_turn_arg );

        return ( job.context().runKernel( kid, job.totalNumParticles() ) )
            ? status_t{ 0 } : status_t{ -1 };
    }

    TrackJobCuda::track_status_t trackLine(
        TrackJobCuda& SIXTRL_RESTRICT_REF job,
        TrackJobCuda::size_type const line_begin_idx,
        TrackJobCuda::size_type const line_end_idx,
        bool const finish_turn ) SIXTRL_NOEXCEPT
    {
        SIXTRL_ASSERT( job.ptrContext() != nullptr );
        SIXTRL_ASSERT( job.ptrParticlesArg() != nullptr );
        SIXTRL_ASSERT( job.ptrBeamElementsArg() != nullptr );
        SIXTRL_ASSERT( job.totalNumParticles() > TrackJobCuda::size_type{ 0 } );

        ClContext& ctx = job.context();
        SIXTRL_ASSERT( ctx.hasSelectedNode() );
        SIXTRL_ASSERT( ctx.hasLineTrackingKernel() );

        return ctx.trackLine( line_begin_idx, line_end_idx, finish_turn );
    }

}

#endif /* C++, Host */

/* end: sixtracklib/cuda/track/track_job.cpp */
