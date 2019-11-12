#include "sixtracklib/opencl/track_job_cl.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>

#if !defined( SIXTRL_NO_INCLUDES )
    #if defined( __cplusplus )
        #include "sixtracklib/common/buffer.hpp"
        #include "sixtracklib/common/output/output_buffer.hpp"
    #endif /* defined( __cplusplus ) */

    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/buffer.h"
    #include "sixtracklib/common/particles.h"
    #include "sixtracklib/common/internal/track_job_base.h"

    #include "sixtracklib/opencl/context.h"
    #include "sixtracklib/opencl/argument.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

namespace st = SIXTRL_CXX_NAMESPACE;

namespace SIXTRL_CXX_NAMESPACE
{
    namespace
    {
        using _this_t   = st::TrackJobCl;
        using _size_t   = _this_t::size_type;
        using _status_t = _this_t::status_t;
    }

    TrackJobCl::TrackJobCl(
        std::string const& SIXTRL_RESTRICT_REF device_id_str,
        std::string const& SIXTRL_RESTRICT_REF config_str ) :
        TrackJobBase( st::TRACK_JOB_CL_STR, st::TRACK_JOB_CL_ID ),
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

        this->doSetRequiresCollectFlag( true );
        this->doSetDeviceIdStr( device_id_str.c_str() );

        this->doPrepareContextOclImpl(
            device_id_str.c_str(), this->ptrConfigStr() );
    }

    TrackJobCl::TrackJobCl(
        const char *const SIXTRL_RESTRICT device_id_str,
        const char *const SIXTRL_RESTRICT config_str ) :
        TrackJobBase( st::TRACK_JOB_CL_STR, st::TRACK_JOB_CL_ID ),
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

        this->doSetRequiresCollectFlag( true );
        this->doSetDeviceIdStr( device_id_str );
        this->doPrepareContextOclImpl( device_id_str, this->ptrConfigStr() );
    }

    TrackJobCl::TrackJobCl(
        const char *const SIXTRL_RESTRICT device_id_str,
        _this_t::c_buffer_t* SIXTRL_RESTRICT particles_buffer,
        _this_t::c_buffer_t* SIXTRL_RESTRICT beam_elements_buffer,
        _this_t::c_buffer_t* SIXTRL_RESTRICT ptr_output_buffer,
        _size_t const until_turn_elem_by_elem,
        const char *const SIXTRL_RESTRICT config_str ) :
        TrackJobBase( st::TRACK_JOB_CL_STR, st::TRACK_JOB_CL_ID ),
        m_ptr_context( nullptr ),
        m_ptr_particles_buffer_arg( nullptr ),
        m_ptr_beam_elements_buffer_arg( nullptr ),
        m_ptr_output_buffer_arg( nullptr ),
        m_ptr_cl_elem_by_elem_config_buffer( nullptr ),
        m_total_num_particles( TrackJobBase::size_type{ 0 } )
    {
        this->doInitTrackJobCl( device_id_str, particles_buffer,
            this->particleSetIndicesBegin(), this->particleSetIndicesEnd(),
            beam_elements_buffer, ptr_output_buffer, until_turn_elem_by_elem,
            config_str );
    }

    TrackJobCl::TrackJobCl(
        const char *const SIXTRL_RESTRICT device_id_str,
        _this_t::c_buffer_t* SIXTRL_RESTRICT particles_buffer,
        _size_t const num_particle_sets,
        _size_t const* SIXTRL_RESTRICT pset_begin,
        _this_t::c_buffer_t* SIXTRL_RESTRICT belements_buffer,
        _this_t::c_buffer_t* SIXTRL_RESTRICT ptr_output_buffer,
        _size_t const until_turn_elem_by_elem,
        const char *const SIXTRL_RESTRICT config_str ) :
        TrackJobBase( st::TRACK_JOB_CL_STR, st::TRACK_JOB_CL_ID ),
        m_ptr_context( nullptr ),
        m_ptr_particles_buffer_arg( nullptr ),
        m_ptr_beam_elements_buffer_arg( nullptr ),
        m_ptr_output_buffer_arg( nullptr ),
        m_ptr_cl_elem_by_elem_config_buffer( nullptr ),
        m_total_num_particles( TrackJobBase::size_type{ 0 } )
    {
        using size_t  = _size_t;
        size_t const* pset_end = pset_begin;

        if( ( pset_end != nullptr ) && ( num_particle_sets > size_t{ 0 } ) )
        {
            std::advance( pset_end, num_particle_sets );
        }

        this->doInitTrackJobCl( device_id_str, particles_buffer,
            pset_begin, pset_end, belements_buffer, ptr_output_buffer,
            until_turn_elem_by_elem, config_str );
    }

    TrackJobCl::TrackJobCl(
        std::string const& SIXTRL_RESTRICT_REF device_id_str,
        TrackJobCl::buffer_t& SIXTRL_RESTRICT_REF particles_buffer,
        TrackJobCl::buffer_t& SIXTRL_RESTRICT_REF belements_buffer,
        TrackJobCl::buffer_t* SIXTRL_RESTRICT ptr_output_buffer,
        _size_t const until_turn_elem_by_elem,
        std::string const& config_str ) :
        TrackJobBase( st::TRACK_JOB_CL_STR, st::TRACK_JOB_CL_ID ),
        m_ptr_context( nullptr ),
        m_ptr_particles_buffer_arg( nullptr ),
        m_ptr_beam_elements_buffer_arg( nullptr ),
        m_ptr_output_buffer_arg( nullptr ),
        m_ptr_cl_elem_by_elem_config_buffer( nullptr ),
        m_total_num_particles( TrackJobBase::size_type{ 0 } )
    {
        using c_buffer_t = _this_t::c_buffer_t;

        c_buffer_t* ptr_part_buffer  = particles_buffer.getCApiPtr();
        c_buffer_t* ptr_belem_buffer = belements_buffer.getCApiPtr();
        c_buffer_t* ptr_out_buffer   = ( ptr_output_buffer != nullptr )
            ? ptr_output_buffer->getCApiPtr() : nullptr;

        bool const success = this->doInitTrackJobCl(
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

    TrackJobCl::~TrackJobCl() SIXTRL_NOEXCEPT {}

    _this_t::cl_context_t& TrackJobCl::context() SIXTRL_RESTRICT
    {
        return const_cast< _this_t::cl_context_t& >(
            static_cast< _this_t const& >( *this ).context() );
    }


    _this_t::cl_context_t const& TrackJobCl::context() const SIXTRL_RESTRICT
    {
        SIXTRL_ASSERT( this->ptrContext() != nullptr );
        return *( this->ptrContext() );
    }

    ::NS(ClContext)* TrackJobCl::ptrContext() SIXTRL_RESTRICT
    {
        return this->m_ptr_context.get();
    }

    ::NS(ClContext) const* TrackJobCl::ptrContext() const SIXTRL_RESTRICT
    {
        return this->m_ptr_context.get();
    }

    _this_t::cl_arg_t& TrackJobCl::particlesArg() SIXTRL_NOEXCEPT
    {
        return const_cast< _this_t::cl_arg_t& >(
            static_cast< _this_t const& >( *this ).particlesArg() );
    }

    _this_t::cl_arg_t const& TrackJobCl::particlesArg() const SIXTRL_NOEXCEPT
    {
        SIXTRL_ASSERT( this->ptrParticlesArg() != nullptr );
        return *(this->ptrParticlesArg() );
    }

    _this_t::cl_arg_t* TrackJobCl::ptrParticlesArg() SIXTRL_NOEXCEPT
    {
        return const_cast< _this_t::cl_arg_t* >( static_cast< _this_t const& >(
            *this ).ptrParticlesArg() );
    }

    _this_t::cl_arg_t const* TrackJobCl::ptrParticlesArg() const SIXTRL_NOEXCEPT
    {
        return this->m_ptr_particles_buffer_arg.get();
    }

    _this_t::cl_arg_t& TrackJobCl::beamElementsArg() SIXTRL_NOEXCEPT
    {
        return const_cast< _this_t::cl_arg_t& >( static_cast<
            TrackJobCl const& >( *this ).beamElementsArg() );
    }

    _this_t::cl_arg_t const& TrackJobCl::beamElementsArg() const SIXTRL_NOEXCEPT
    {
        SIXTRL_ASSERT( this->ptrBeamElementsArg() != nullptr );
        return *( this->ptrBeamElementsArg() );
    }

    _this_t::cl_arg_t* TrackJobCl::ptrBeamElementsArg() SIXTRL_NOEXCEPT
    {
        return const_cast< _this_t::cl_arg_t* >( static_cast< _this_t const& >(
            *this ).ptrBeamElementsArg() );
    }

    _this_t::cl_arg_t const*
    TrackJobCl::ptrBeamElementsArg() const SIXTRL_NOEXCEPT
    {
        return this->m_ptr_beam_elements_buffer_arg.get();
    }

    _this_t::cl_arg_t& TrackJobCl::outputBufferArg() SIXTRL_NOEXCEPT
    {
        return const_cast< _this_t::cl_arg_t& >( static_cast< _this_t const& >(
            *this ).outputBufferArg() );
    }

    _this_t::cl_arg_t const& TrackJobCl::outputBufferArg() const SIXTRL_NOEXCEPT
    {
        SIXTRL_ASSERT( this->ptrOutputBufferArg() != nullptr );
        return *( this->m_ptr_output_buffer_arg );
    }

    _this_t::cl_arg_t* TrackJobCl::ptrOutputBufferArg() SIXTRL_NOEXCEPT
    {
        return const_cast< _this_t::cl_arg_t* >( static_cast< _this_t const& >(
            *this ).ptrOutputBufferArg() );
    }

    _this_t::cl_arg_t const*
    TrackJobCl::ptrOutputBufferArg() const SIXTRL_NOEXCEPT
    {
        return this->m_ptr_output_buffer_arg.get();
    }


    _this_t::cl_buffer_t const&
    TrackJobCl::clElemByElemConfigBuffer() const SIXTRL_NOEXCEPT
    {
        SIXTRL_ASSERT( this->ptrClElemByElemConfigBuffer() != nullptr );
        return *( this->ptrClElemByElemConfigBuffer() );
    }

    _this_t::cl_buffer_t& TrackJobCl::clElemByElemConfigBuffer() SIXTRL_NOEXCEPT
    {
        using ref_t = _this_t::cl_buffer_t&;
        return const_cast< ref_t >( static_cast< _this_t const& >(
                *this ).clElemByElemConfigBuffer() );
    }

    _this_t::cl_buffer_t const*
    TrackJobCl::ptrClElemByElemConfigBuffer() const SIXTRL_NOEXCEPT
    {
        return this->m_ptr_cl_elem_by_elem_config_buffer.get();
    }

    _this_t::cl_buffer_t*
    TrackJobCl::ptrClElemByElemConfigBuffer() SIXTRL_NOEXCEPT
    {
        return this->m_ptr_cl_elem_by_elem_config_buffer.get();
    }

    _this_t::status_t TrackJobCl::updateBeamElementsRegion(
        _size_t const offset, _size_t const length,
        void const* SIXTRL_RESTRICT new_value )
    {
        return ( this->m_ptr_beam_elements_buffer_arg != nullptr )
            ? this->m_ptr_beam_elements_buffer_arg->updateRegion(
                offset, length, new_value )
            : ::NS(ARCH_STATUS_GENERAL_FAILURE);
    }

    _this_t::status_t TrackJobCl::updateBeamElementsRegions(
        _size_t const num_regions_to_update,
        _size_t const* SIXTRL_RESTRICT offsets,
        _size_t const* SIXTRL_RESTRICT lengths,
        void const* SIXTRL_RESTRICT const* SIXTRL_RESTRICT new_values )
    {
        return ( this->m_ptr_beam_elements_buffer_arg != nullptr )
            ? this->m_ptr_beam_elements_buffer_arg->updateRegions(
                num_regions_to_update, offsets, lengths, new_values )
            : ::NS(ARCH_STATUS_GENERAL_FAILURE);
    }


    bool TrackJobCl::doPrepareParticlesStructures(
        _this_t::c_buffer_t* SIXTRL_RESTRICT pb )
    {
        return ( ( TrackJobBase::doPrepareParticlesStructures( pb ) ) &&
                 ( this->doPrepareParticlesStructuresOclImp( pb ) ) );
    }

    bool TrackJobCl::doPrepareBeamElementsStructures(
        _this_t::c_buffer_t* SIXTRL_RESTRICT belems )
    {
        return ( ( TrackJobBase::doPrepareBeamElementsStructures( belems ) ) &&
            ( this->doPrepareBeamElementsStructuresOclImp( belems ) ) );
    }

    bool TrackJobCl::doPrepareOutputStructures(
        _this_t::c_buffer_t* SIXTRL_RESTRICT part_buffer,
        _this_t::c_buffer_t* SIXTRL_RESTRICT belem_buffer,
        _this_t::c_buffer_t* SIXTRL_RESTRICT output_buffer,
        _size_t const until_turn_elem_by_elem )
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

    bool TrackJobCl::doAssignOutputBufferToBeamMonitors(
        _this_t::c_buffer_t* SIXTRL_RESTRICT belem_buffer,
        _this_t::c_buffer_t* SIXTRL_RESTRICT out_buffer,
        _this_t::particle_index_t const min_turn_id,
        _this_t::size_type const output_buffer_offset_index )
    {
        bool success = TrackJobBase::doAssignOutputBufferToBeamMonitors(
                belem_buffer, out_buffer, min_turn_id,
                    output_buffer_offset_index );

        if( ( success ) && ( this->hasBeamMonitorOutput() ) )
        {
            success = this->doAssignOutputBufferToBeamMonitorsOclImp(
                belem_buffer, out_buffer, min_turn_id,
                    output_buffer_offset_index );
        }

        return success;
    }

    bool TrackJobCl::doAssignOutputBufferToElemByElemConfig(
        _this_t::elem_by_elem_config_t* SIXTRL_RESTRICT elem_by_elem_conf,
        _this_t::c_buffer_t* SIXTRL_RESTRICT out_buffer,
        _this_t::size_type const output_buffer_offset_index )
    {
        bool success = TrackJobBase::doAssignOutputBufferToElemByElemConfig(
                elem_by_elem_conf, out_buffer, output_buffer_offset_index );

        if( ( success ) && ( this->hasElemByElemOutput() ) )
        {
            success = this->doAssignOutputBufferToElemByElemConfigOclImpl(
                elem_by_elem_conf, out_buffer, output_buffer_offset_index );
        }

        return success;
    }

    bool TrackJobCl::doReset(
        _this_t::c_buffer_t* SIXTRL_RESTRICT particles_buffer,
        _this_t::c_buffer_t* SIXTRL_RESTRICT beam_elem_buffer,
        _this_t::c_buffer_t* SIXTRL_RESTRICT ptr_output_buffer,
        _size_t const until_turn_elem_by_elem )
    {
        return this->doResetOclImp( particles_buffer, beam_elem_buffer,
                ptr_output_buffer, until_turn_elem_by_elem );
    }

    _this_t::track_status_t TrackJobCl::doTrackUntilTurn(
        _size_t const until_turn )
    {
        return st::track( *this, until_turn );
    }

    _this_t::track_status_t TrackJobCl::doTrackElemByElem(
        _size_t const until_turn )
    {
        return st::trackElemByElem( *this, until_turn );
    }

    _this_t::track_status_t TrackJobCl::doTrackLine(
        _size_t const line_begin_idx, _size_t const line_end_idx,
        bool const finish_turn )
    {
        return st::trackLine(
            *this, line_begin_idx, line_end_idx, finish_turn );
    }


    void TrackJobCl::doCollect( TrackJobCl::collect_flag_t const flags )
    {
        st::collect( *this, flags );
    }

    void TrackJobCl::doPush( TrackJobCl::push_flag_t const flags )
    {
        st::push( *this, flags );
    }

    void TrackJobCl::doParseConfigStr( const char *const SIXTRL_RESTRICT  ) {}

    void TrackJobCl::doUpdateStoredContext(
        TrackJobCl::ptr_cl_context_t&& context )
    {
        this->m_ptr_context = std::move( context );
    }

    void TrackJobCl::doUpdateStoredParticlesArg(
        TrackJobCl::ptr_cl_arg_t&& particle_arg )
    {
        this->m_ptr_particles_buffer_arg = std::move( particle_arg );
    }

    void TrackJobCl::doUpdateStoredBeamElementsArg(
        TrackJobCl::ptr_cl_arg_t&& beam_elements_arg )
    {
        this->m_ptr_beam_elements_buffer_arg = std::move( beam_elements_arg );
    }

    void TrackJobCl::doUpdateStoredOutputArg(
        TrackJobCl::ptr_cl_arg_t&& output_arg )
    {
        this->m_ptr_output_buffer_arg = std::move( output_arg );
    }

    void TrackJobCl::doUpdateStoredClElemByElemConfigBuffer(
            TrackJobCl::ptr_cl_buffer_t&& cl_elem_by_elem_config_buffer )
    {
        this->m_ptr_cl_elem_by_elem_config_buffer =
            std::move( cl_elem_by_elem_config_buffer );
    }

    _size_t TrackJobCl::totalNumParticles() const SIXTRL_NOEXCEPT
    {
        return this->m_total_num_particles;
    }

    void TrackJobCl::doSetTotalNumParticles(
        _size_t const total_num_particles ) SIXTRL_NOEXCEPT
    {
        this->m_total_num_particles = total_num_particles;
    }

    bool TrackJobCl::doPrepareContext(
        char const* SIXTRL_RESTRICT device_id_str,
        char const* SIXTRL_RESTRICT ptr_config_str )
    {
        return this->doPrepareContextOclImpl( device_id_str, ptr_config_str );
    }

    bool TrackJobCl::doPrepareContextOclImpl(
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

    void TrackJobCl::doParseConfigStrOclImpl(
        const char *const SIXTRL_RESTRICT ) {}

    bool TrackJobCl::doPrepareParticlesStructuresOclImp(
        _this_t::c_buffer_t* SIXTRL_RESTRICT pb )
    {
        using _this_t = TrackJobCl;

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
                  ( ( this->context().assign_particles_arg(
                        this->particlesArg() ) == st::ARCH_STATUS_SUCCESS ) &&
                    ( this->context().assign_particle_set_arg(
                        *this->particleSetIndicesBegin(),
                        *this->numParticlesInSetsBegin() ) ==
                        st::ARCH_STATUS_SUCCESS ) ) ) )
            {
                success = true;
            }
        }

        return success;
    }

    bool TrackJobCl::doPrepareBeamElementsStructuresOclImp(
        _this_t::c_buffer_t* SIXTRL_RESTRICT belems )
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
                  (  this->context().assign_beam_elements_arg(
                         this->beamElementsArg() ) == st::ARCH_STATUS_SUCCESS )
                ) );
        }

        return success;
    }

    bool TrackJobCl::doPrepareOutputStructuresOclImpl(
        _this_t::c_buffer_t* SIXTRL_RESTRICT pb,
        _this_t::c_buffer_t* SIXTRL_RESTRICT belems,
        _this_t::c_buffer_t* SIXTRL_RESTRICT ptr_out_buffer,
        _size_t const until_turn_elem_by_elem )
    {
        using _this_t = TrackJobCl;

        using arg_t         = _this_t::cl_arg_t;
        using ptr_arg_t     = _this_t::ptr_cl_arg_t;
        using ptr_buffer_t  = _this_t::ptr_cl_buffer_t;

        bool success = true;
        ( void )until_turn_elem_by_elem;

        if( ptr_out_buffer != nullptr )
        {
            ptr_arg_t ptr( new arg_t( ptr_out_buffer, this->ptrContext() ) );
            this->doUpdateStoredOutputArg( std::move( ptr ) );

            success &= ( this->ptrOutputBufferArg() != nullptr );

            if( ( success ) && ( this->ptrContext() != nullptr ) &&
                ( this->context().hasSelectedNode() ) )
            {
                success &= ( this->context().assign_output_buffer_arg(
                    this->outputBufferArg() ) == st::ARCH_STATUS_SUCCESS );
            }
        }

        if( ( success ) && ( ptr_out_buffer != nullptr ) && ( pb != nullptr ) )
        {
            /* TODO: Update this region as soon as the OpenCL argument
             *       can handle raw data rather than only CObject buffers */

            if( ( success ) && ( this->ptrContext()   != nullptr ) &&
                ( this->ptrElemByElemConfig() != nullptr ) )
            {
                ptr_buffer_t ptr_buffer =
                    this->ptrContext( )->create_elem_by_elem_config_arg();

                this->doUpdateStoredClElemByElemConfigBuffer(
                    std::move( ptr_buffer ) );

                success = ( this->ptrClElemByElemConfigBuffer() != nullptr );

                if( success )
                {
                    success = ( st::ARCH_STATUS_SUCCESS ==
                        this->ptrContext()->push_elem_by_elem_config_arg(
                            *this->ptrClElemByElemConfigBuffer(),
                            *this->ptrElemByElemConfig() ) );
                }

                if( success )
                {
                    success = ( st::ARCH_STATUS_SUCCESS ==
                        this->ptrContext()->assign_elem_by_elem_config_arg(
                            *this->ptrClElemByElemConfigBuffer() ) );
                }
            }
        }

        return success;
    }

    bool TrackJobCl::doAssignOutputBufferToBeamMonitorsOclImp(
        _this_t::c_buffer_t* SIXTRL_RESTRICT belem_buffer,
        _this_t::c_buffer_t* SIXTRL_RESTRICT output_buffer,
        _this_t::particle_index_t const min_turn_id,
        _this_t::size_type const output_buffer_index_offset )
    {
        bool success = false;

        if( ( this->ptrContext() != nullptr ) &&
            ( this->context().hasSelectedNode() ) &&
            ( this->context().has_assign_beam_monitor_output_kernel() ) &&
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
            success = ( st::ARCH_STATUS_SUCCESS ==
                this->context().assign_output_buffer_arg(
                    this->outputBufferArg() ) );

            if( success )
            {
                success = ( st::ARCH_STATUS_SUCCESS ==
                    this->context().assign_beam_monitor_output(
                        min_turn_id, output_buffer_index_offset ) );
            }

            if( success )
            {
                success = this->beamElementsArg().read( belem_buffer );
            }
        }

        return success;
    }

    bool TrackJobCl::doAssignOutputBufferToElemByElemConfigOclImpl(
        _this_t::elem_by_elem_config_t* SIXTRL_RESTRICT elem_by_elem_conf,
        _this_t::c_buffer_t* SIXTRL_RESTRICT out_buffer,
        _this_t::size_type const output_buffer_index_offset )
    {
        bool success = false;

        if( ( this->ptrContext() != nullptr ) &&
            ( this->context().hasSelectedNode() ) &&
            ( this->context().has_assign_elem_by_elem_output_kernel() ) &&
            ( this->hasElemByElemOutput() ) &&
            ( this->ptrClElemByElemConfigBuffer() != nullptr ) &&
            ( this->ptrElemByElemConfig() != nullptr ) &&
            ( this->ptrElemByElemConfig() == elem_by_elem_conf ) &&
            ( this->ptrBeamElementsArg() != nullptr ) &&
            ( this->ptrOutputBufferArg() != nullptr ) &&
            ( this->outputBufferArg().context() == this->ptrContext() ) &&
            ( this->outputBufferArg().usesCObjectBuffer() ) &&
            ( this->outputBufferArg().ptrCObjectBuffer() == out_buffer ) )
        {
            success = ( st::ARCH_STATUS_SUCCESS ==
                this->context().assign_output_buffer_arg(
                    this->outputBufferArg() ) );

            if( success )
            {
                success = ( st::ARCH_STATUS_SUCCESS ==
                    this->context().assign_elem_by_elem_output(
                        output_buffer_index_offset ) );
            }

            if( success )
            {
                success = ( this->context().collect_elem_by_elem_config_arg(
                    *this->ptrClElemByElemConfigBuffer(), *elem_by_elem_conf )
                        == st::ARCH_STATUS_SUCCESS );
            }
        }

        return success;
    }

    bool TrackJobCl::doResetOclImp(
        _this_t::c_buffer_t* SIXTRL_RESTRICT pbuffer,
        _this_t::c_buffer_t* SIXTRL_RESTRICT belem_buffer,
        _this_t::c_buffer_t* SIXTRL_RESTRICT out_buffer,
        _size_t const until_turn_elem_by_elem )
    {
        using _this_t = st::TrackJobCl;
        using _base_t = st::TrackJobBase;
        using output_buffer_flag_t = _this_t::output_buffer_flag_t;

        bool success = _base_t::doPrepareParticlesStructures( pbuffer );

        if( success )
        {
            success = _base_t::doPrepareBeamElementsStructures( belem_buffer );
        }

        if( success )
        {
            success = this->doPrepareParticlesStructuresOclImp( pbuffer );
        }

        if( success )
        {
            success = this->doPrepareBeamElementsStructuresOclImp( belem_buffer );
        }

        output_buffer_flag_t const out_buffer_flags =
        ::NS(OutputBuffer_required_for_tracking_of_particle_sets)( pbuffer,
            this->numParticleSets(), this->particleSetIndicesBegin(),
                belem_buffer, until_turn_elem_by_elem );

        bool const requires_output_buffer =
            ::NS(OutputBuffer_requires_output_buffer)( out_buffer_flags );

        if( success )
        {
            this->doSetPtrCParticleBuffer( pbuffer );
            this->doSetPtrCBeamElementsBuffer( belem_buffer );
        }

        if( success )
        {
            if( ( requires_output_buffer ) || ( out_buffer != nullptr ) )
            {
                success = _base_t::doPrepareOutputStructures( pbuffer,
                    belem_buffer, out_buffer, until_turn_elem_by_elem );

                if( ( success ) && ( out_buffer != nullptr ) &&
                    ( !this->ownsOutputBuffer() ) )
                {
                    this->doSetPtrCOutputBuffer( out_buffer );
                }

                if( ( success ) && ( this->hasOutputBuffer() ) )
                {
                    success = this->doPrepareOutputStructuresOclImpl( pbuffer,
                        belem_buffer, this->ptrCOutputBuffer(),
                            until_turn_elem_by_elem );
                }
            }
        }

        if( ( success ) && ( this->hasOutputBuffer() ) &&
            ( requires_output_buffer ) )
        {
            if( ::NS(OutputBuffer_requires_elem_by_elem_output)( out_buffer_flags ) )
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

            if( ::NS(OutputBuffer_requires_beam_monitor_output)(
                    out_buffer_flags ) )
            {
                if( success )
                {
                    success = _base_t::doAssignOutputBufferToBeamMonitors(
                        belem_buffer, this->ptrCOutputBuffer(),
                            this->minInitialTurnId(),
                                this->beamMonitorsOutputBufferOffset() );
                }

                if( success )
                {
                    success = this->doAssignOutputBufferToBeamMonitorsOclImp(
                        belem_buffer, this->ptrCOutputBuffer(),
                            this->minInitialTurnId(),
                                this->beamMonitorsOutputBufferOffset() );
                }
            }
        }
        else if( ( success) && ( requires_output_buffer ) )
        {
            if( !this->hasOutputBuffer() ) success = false;
        }

        return success;
    }

    void collect( TrackJobCl& SIXTRL_RESTRICT_REF job ) SIXTRL_NOEXCEPT
    {
        st::collect( job, job.collectFlags() );
    }

    void collect( TrackJobCl& SIXTRL_RESTRICT_REF job,
                  st::track_job_collect_flag_t const flags ) SIXTRL_NOEXCEPT
    {
        if( ( TrackJobCl::IsCollectFlagSet(
                flags, TRACK_JOB_IO_PARTICLES ) ) &&
            ( job.ptrParticlesArg() != nullptr ) &&
            ( job.ptrCParticlesBuffer() != nullptr ) )
        {
            job.particlesArg().read( job.ptrCParticlesBuffer() );
        }

        if( ( TrackJobCl::IsCollectFlagSet(
                flags, TRACK_JOB_IO_OUTPUT ) ) &&
            ( job.ptrOutputBufferArg() != nullptr ) &&
            ( job.ptrCOutputBuffer() != nullptr ) )
        {
            job.outputBufferArg().read( job.ptrCOutputBuffer() );
        }

        if( ( TrackJobCl::IsCollectFlagSet(
                flags, TRACK_JOB_IO_BEAM_ELEMENTS ) ) &&
            ( job.ptrBeamElementsArg() != nullptr ) &&
            ( job.ptrCBeamElementsBuffer() != nullptr ) )
        {
            job.beamElementsArg().read( job.ptrCBeamElementsBuffer() );
        }
    }

    void push( TrackJobCl& SIXTRL_RESTRICT_REF job,
        st::track_job_collect_flag_t const flags ) SIXTRL_NOEXCEPT
    {
        namespace st = SIXTRL_CXX_NAMESPACE;
        bool success = true;

         if( ( TrackJobCl::IsCollectFlagSet(
                flags, st::TRACK_JOB_IO_PARTICLES ) ) &&
            ( job.ptrParticlesArg() != nullptr ) &&
            ( job.ptrCParticlesBuffer() != nullptr ) )
        {
            success &= job.particlesArg().write( job.ptrCParticlesBuffer() );
        }

        if( ( TrackJobCl::IsCollectFlagSet(
                flags, st::TRACK_JOB_IO_OUTPUT ) ) &&
            ( job.ptrOutputBufferArg() != nullptr ) &&
            ( job.ptrCOutputBuffer() != nullptr ) )
        {
            success &= job.outputBufferArg().write( job.ptrCOutputBuffer() );
        }

        if( ( TrackJobCl::IsCollectFlagSet(
                flags, st::TRACK_JOB_IO_BEAM_ELEMENTS ) ) &&
            ( job.ptrBeamElementsArg() != nullptr ) &&
            ( job.ptrCBeamElementsBuffer() != nullptr ) )
        {
            success &= job.beamElementsArg().write(
                job.ptrCBeamElementsBuffer() );
        }

        ( void )success;
    }

    _this_t::track_status_t track(
        TrackJobCl& SIXTRL_RESTRICT_REF job,
        _size_t const until_turn ) SIXTRL_NOEXCEPT
    {
        _this_t::track_status_t status = st::TRACK_STATUS_GENERAL_FAILURE;

        SIXTRL_ASSERT( job.ptrContext() != nullptr );
        SIXTRL_ASSERT( job.ptrParticlesArg() != nullptr );
        SIXTRL_ASSERT( job.ptrBeamElementsArg() != nullptr );
        SIXTRL_ASSERT( job.ptrContext()->hasSelectedNode() );
        SIXTRL_ASSERT( job.context().has_track_until_kernel() );
        SIXTRL_ASSERT( job.particleSetIndicesBegin() != nullptr );
        SIXTRL_ASSERT( job.numParticlesInSetsBegin() != nullptr );

        _size_t const num_psets = job.numParticleSets();

        if( num_psets == _size_t{ 1 } )
        {
            SIXTRL_ASSERT( job.context().selected_particle_set() ==
                *job.particleSetIndicesBegin() );

            SIXTRL_ASSERT( job.context().num_particles_in_selected_set() ==
                *job.numParticlesInSetsBegin() );

            status = job.context().track_until( until_turn );
        }
        else if( num_psets > _size_t{ 1 } )
        {
            auto pset_it  = job.particleSetIndicesBegin();
            auto pset_end = job.particleSetIndicesEnd();
            auto npart_it = job.numParticlesInSetsBegin();

            uint64_t const saved_pset_idx_arg = static_cast< uint64_t >(
                job.context().selected_particle_set() );

            SIXTRL_ASSERT( std::distance( pset_it, pset_end ) ==
                std::distance( npart_it, job.numParticlesInSetsEnd() ) );

            status = st::TRACK_SUCCESS;

            for( ; pset_it != pset_end ; ++pset_it, ++npart_it )
            {
                SIXTRL_ASSERT( ( *pset_it != saved_pset_idx_arg ) ||
                    ( job.context().num_particles_in_selected_set() ==
                      *npart_it ) );

                status |= job.context().track_until(
                    until_turn, *pset_it, *npart_it, false );
            }

            SIXTRL_ASSERT( job.context().track_until_kernel_id() !=
                st::ARCH_ILLEGAL_KERNEL_ID );

            job.context().assignKernelArgumentValue(
                job.context().track_until_kernel_id(), _size_t{ 1 },
                    saved_pset_idx_arg );
        }

        return status;
    }

    _this_t::track_status_t trackElemByElem(
        TrackJobCl& SIXTRL_RESTRICT_REF job,
        _size_t const until_turn ) SIXTRL_NOEXCEPT
    {
        _this_t::track_status_t status = st::TRACK_STATUS_GENERAL_FAILURE;

        SIXTRL_ASSERT( job.ptrContext() != nullptr );
        SIXTRL_ASSERT( job.ptrParticlesArg() != nullptr );
        SIXTRL_ASSERT( job.ptrBeamElementsArg() != nullptr );
        SIXTRL_ASSERT( job.hasOutputBuffer() );
        SIXTRL_ASSERT( job.ptrOutputBufferArg() != nullptr );
        SIXTRL_ASSERT( job.hasElemByElemOutput() );
        SIXTRL_ASSERT( job.ptrElemByElemConfig() != nullptr );
        SIXTRL_ASSERT( job.ptrClElemByElemConfigBuffer() != nullptr );
        SIXTRL_ASSERT( job.ptrContext()->hasSelectedNode() );
        SIXTRL_ASSERT( job.context().has_track_elem_by_elem_kernel() );
        SIXTRL_ASSERT( job.particleSetIndicesBegin() != nullptr );
        SIXTRL_ASSERT( job.numParticlesInSetsBegin() != nullptr );

        _size_t const num_psets = job.numParticleSets();

        if( num_psets == _size_t{ 1 } )
        {
            SIXTRL_ASSERT( job.context().selected_particle_set() ==
                *job.particleSetIndicesBegin() );

            SIXTRL_ASSERT( job.context().num_particles_in_selected_set() ==
                *job.numParticlesInSetsBegin() );

            status = job.context().track_elem_by_elem( until_turn );
        }
        else if( num_psets > _size_t{ 1 } )
        {
            auto pset_it  = job.particleSetIndicesBegin();
            auto pset_end = job.particleSetIndicesEnd();
            auto npart_it = job.numParticlesInSetsBegin();

            uint64_t const saved_pset_idx_arg = static_cast< uint64_t >(
                job.context().selected_particle_set() );

            SIXTRL_ASSERT( std::distance( pset_it, pset_end ) ==
                std::distance( npart_it, job.numParticlesInSetsEnd() ) );

            status = st::TRACK_SUCCESS;

            for( ; pset_it != pset_end ; ++pset_it, ++npart_it )
            {
                SIXTRL_ASSERT( ( *pset_it != saved_pset_idx_arg ) ||
                    ( job.context().num_particles_in_selected_set() ==
                      *npart_it ) );

                status |= job.context().track_elem_by_elem(
                    until_turn, *pset_it, *npart_it, false );
            }

            SIXTRL_ASSERT( job.context().track_elem_by_elem_kernel_id() !=
                st::ARCH_ILLEGAL_KERNEL_ID );

            job.context().assignKernelArgumentValue(
                job.context().track_elem_by_elem_kernel_id(), _size_t{ 1 },
                    saved_pset_idx_arg );
        }

        return status;
    }

    _this_t::track_status_t trackLine(
        TrackJobCl& SIXTRL_RESTRICT_REF job,
        _size_t const line_begin_idx,
        _size_t const line_end_idx,
        bool const finish_turn ) SIXTRL_NOEXCEPT
    {
        _this_t::track_status_t status = st::TRACK_STATUS_GENERAL_FAILURE;

        SIXTRL_ASSERT( job.ptrContext() != nullptr );
        SIXTRL_ASSERT( job.ptrParticlesArg() != nullptr );
        SIXTRL_ASSERT( job.ptrBeamElementsArg() != nullptr );
        SIXTRL_ASSERT( job.ptrContext()->hasSelectedNode() );
        SIXTRL_ASSERT( job.context().has_track_elem_by_elem_kernel() );
        SIXTRL_ASSERT( job.particleSetIndicesBegin() != nullptr );
        SIXTRL_ASSERT( job.numParticlesInSetsBegin() != nullptr );

        _size_t const num_psets = job.numParticleSets();

        if( num_psets == _size_t{ 1 } )
        {
            SIXTRL_ASSERT( job.context().selected_particle_set() ==
                *job.particleSetIndicesBegin() );

            SIXTRL_ASSERT( job.context().num_particles_in_selected_set() ==
                *job.numParticlesInSetsBegin() );

            status = job.context().track_line(
                line_begin_idx, line_end_idx, finish_turn );
        }
        else if( num_psets > _size_t{ 1 } )
        {
            auto pset_it  = job.particleSetIndicesBegin();
            auto pset_end = job.particleSetIndicesEnd();
            auto npart_it = job.numParticlesInSetsBegin();

            uint64_t const saved_pset_idx_arg = static_cast< uint64_t >(
                job.context().selected_particle_set() );

            SIXTRL_ASSERT( std::distance( pset_it, pset_end ) ==
                std::distance( npart_it, job.numParticlesInSetsEnd() ) );

            status = st::TRACK_SUCCESS;

            for( ; pset_it != pset_end ; ++pset_it, ++npart_it )
            {
                SIXTRL_ASSERT( ( *pset_it != saved_pset_idx_arg ) ||
                    ( job.context().num_particles_in_selected_set() ==
                      *npart_it ) );

                status |= job.context().track_line( line_begin_idx,
                    line_end_idx, finish_turn, *pset_it, *npart_it, false );
            }

            SIXTRL_ASSERT( job.context().track_line_kernel_id() !=
                st::ARCH_ILLEGAL_KERNEL_ID );

            job.context().assignKernelArgumentValue(
                job.context().track_line_kernel_id(), _size_t{ 1 },
                    saved_pset_idx_arg );
        }

        return status;
    }
}

NS(TrackJobCl)* NS(TrackJobCl_create)(
    const char *const SIXTRL_RESTRICT device_id_str )
{
    return new st::TrackJobCl( device_id_str, nullptr );
}

NS(TrackJobCl)*
NS(TrackJobCl_create_from_config_str)(
    const char *const SIXTRL_RESTRICT device_id_str,
    const char *const SIXTRL_RESTRICT config_str )
{
    return new st::TrackJobCl( device_id_str, config_str );
}

NS(TrackJobCl)* NS(TrackJobCl_new)(
    const char *const SIXTRL_RESTRICT device_id_str,
    NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
    NS(Buffer)* SIXTRL_RESTRICT beam_elements_buffer )
{
    return new st::TrackJobCl( device_id_str,
        particles_buffer, beam_elements_buffer );
}

NS(TrackJobCl)* NS(TrackJobCl_new_with_output)(
    const char *const SIXTRL_RESTRICT device_id_str,
    NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
    NS(Buffer)* SIXTRL_RESTRICT beam_elements_buffer,
    NS(Buffer)* SIXTRL_RESTRICT output_buffer,
    NS(buffer_size_t) const until_turn_elem_by_elem )
{
    using _this_t = st::TrackJobCl;
    return new _this_t( device_id_str, particles_buffer,
        _this_t::DefaultNumParticleSetIndices(),
        _this_t::DefaultParticleSetIndicesBegin(),
        beam_elements_buffer, output_buffer, until_turn_elem_by_elem );
}

NS(TrackJobCl)* NS(TrackJobCl_new_detailed)(
    const char *const SIXTRL_RESTRICT device_id_str,
    NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
    NS(buffer_size_t) const num_particle_sets,
    NS(buffer_size_t) const* SIXTRL_RESTRICT particle_set_indices_begin,
    NS(Buffer)* SIXTRL_RESTRICT beam_elements_buffer,
    NS(Buffer)* SIXTRL_RESTRICT output_buffer,
    NS(buffer_size_t) const until_turn_elem_by_elem,
    const char *const SIXTRL_RESTRICT config_str )
{
    return new st::TrackJobCl( device_id_str,
        particles_buffer, num_particle_sets, particle_set_indices_begin,
        beam_elements_buffer, output_buffer, until_turn_elem_by_elem,
        config_str );
}

SIXTRL_EXTERN bool NS(TrackJobCl_reset)(
    NS(TrackJobCl)* SIXTRL_RESTRICT job,
    NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
    NS(Buffer)* SIXTRL_RESTRICT beam_elements_buffer,
    NS(Buffer)* SIXTRL_RESTRICT output_buffer )
{
    if( job != nullptr )
    {
        st::_size_t const
            until_turn_elem_by_elem = job->numElemByElemTurns();

        return job->reset( particles_buffer, beam_elements_buffer,
                           output_buffer, until_turn_elem_by_elem );
    }

    return false;
}

SIXTRL_EXTERN bool NS(TrackJobCl_reset_with_output)(
    NS(TrackJobCl)* SIXTRL_RESTRICT job,
    NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
    NS(Buffer)* SIXTRL_RESTRICT beam_elements_buffer,
    NS(Buffer)* SIXTRL_RESTRICT output_buffer,
    NS(buffer_size_t) const until_turn_elem_by_elem )
{
    return ( job != nullptr )
        ? job->reset( particles_buffer, beam_elements_buffer, output_buffer,
                      until_turn_elem_by_elem )
        : false;
}

SIXTRL_EXTERN bool NS(TrackJobCl_reset_detailed)(
    NS(TrackJobCl)* SIXTRL_RESTRICT job,
    NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
    NS(buffer_size_t) const num_particle_sets,
    NS(buffer_size_t) const* SIXTRL_RESTRICT pset_begin,
    NS(Buffer)* SIXTRL_RESTRICT beam_elements_buffer,
    NS(Buffer)* SIXTRL_RESTRICT output_buffer,
    NS(buffer_size_t) const until_turn_elem_by_elem )
{
    return ( job != nullptr )
        ? job->reset( particles_buffer, num_particle_sets, pset_begin,
                      beam_elements_buffer, output_buffer,
                      until_turn_elem_by_elem )
        : false;
}

void NS(TrackJobCl_delete)(
    NS(TrackJobCl)* SIXTRL_RESTRICT track_job )
{
    delete track_job;
}

NS(track_status_t) NS(TrackJobCl_track_until_turn)(
    NS(TrackJobCl)* SIXTRL_RESTRICT track_job,
    NS(buffer_size_t) const until_turn )
{
    SIXTRL_ASSERT( track_job != nullptr );
    return st::track( *track_job, until_turn );
}

NS(track_status_t) NS(TrackJobCl_track_elem_by_elem)(
    NS(TrackJobCl)* SIXTRL_RESTRICT track_job,
    NS(buffer_size_t) const until_turn )
{
    SIXTRL_ASSERT( track_job != nullptr );
    return st::trackElemByElem( *track_job, until_turn );
}

NS(track_status_t) NS(TrackJobCl_track_line)(
    NS(TrackJobCl)* SIXTRL_RESTRICT track_job,
    NS(buffer_size_t) const line_begin_idx,
    NS(buffer_size_t) const line_end_idx,
    bool const finish_turn )
{
    SIXTRL_ASSERT( track_job != nullptr );
    return st::trackLine(
        *track_job, line_begin_idx, line_end_idx, finish_turn );
}

void NS(TrackJobCl_collect)(
    NS(TrackJobCl)* SIXTRL_RESTRICT track_job )
{
    if( track_job != nullptr ) track_job->collect();
}

void NS(TrackJobCl_collect_detailed)(
    ::NS(TrackJobCl)* SIXTRL_RESTRICT track_job,
    ::NS(track_job_collect_flag_t) const flags )
{
    if( track_job != nullptr ) track_job->collect( flags );
}

NS(ClContext)* NS(TrackJobCl_get_context)(
    NS(TrackJobCl)* SIXTRL_RESTRICT track_job )
{
    return ( track_job != nullptr ) ? track_job->ptrContext() : nullptr;
}

NS(ClContext) const* NS(TrackJobCl_get_const_context)(
    const NS(TrackJobCl) *const SIXTRL_RESTRICT track_job )
{
    return ( track_job != nullptr ) ? track_job->ptrContext() : nullptr;
}

SIXTRL_EXTERN NS(ClArgument)*
NS(TrackJobCl_get_particles_buffer_arg)(
    NS(TrackJobCl)* SIXTRL_RESTRICT track_job )
{
    return ( track_job != nullptr ) ? track_job->ptrParticlesArg() : nullptr;
}

SIXTRL_EXTERN NS(ClArgument) const*
NS(TrackJobCl_get_const_particles_buffer_arg)(
    const NS(TrackJobCl) *const SIXTRL_RESTRICT track_job )
{
    return ( track_job != nullptr ) ? track_job->ptrParticlesArg() : nullptr;
}

SIXTRL_EXTERN NS(ClArgument)*
NS(TrackJobCl_get_beam_elements_buffer_arg)(
    NS(TrackJobCl)* SIXTRL_RESTRICT job )
{
    return ( job != nullptr ) ? job->ptrBeamElementsArg() : nullptr;
}

SIXTRL_EXTERN NS(ClArgument) const*
NS(TrackJobCl_get_const_beam_elements_buffer_arg)(
    const NS(TrackJobCl) *const SIXTRL_RESTRICT job )
{
    return ( job != nullptr ) ? job->ptrBeamElementsArg() : nullptr;
}

SIXTRL_EXTERN bool NS(TrackJobCl_has_output_buffer_arg)(
    const NS(TrackJobCl) *const SIXTRL_RESTRICT job )
{
    return ( ( job != nullptr ) && ( job->ptrOutputBufferArg() != nullptr ) );
}

SIXTRL_EXTERN NS(ClArgument)*
NS(TrackJobCl_get_output_buffer_arg)( NS(TrackJobCl)* SIXTRL_RESTRICT job )
{
    return ( job != nullptr ) ? job->ptrOutputBufferArg() : nullptr;
}

SIXTRL_EXTERN NS(ClArgument) const*
NS(TrackJobCl_get_const_output_buffer_arg)(
    const NS(TrackJobCl) *const SIXTRL_RESTRICT job )
{
    return ( job != nullptr ) ? job->ptrOutputBufferArg() : nullptr;
}

::NS(arch_status_t) NS(TrackJobCl_update_beam_elements_region)(
    ::NS(TrackJobCl)* SIXTRL_RESTRICT track_job,
    ::NS(arch_size_t) const offset, NS(arch_size_t) const length,
    void const* SIXTRL_RESTRICT new_value )
{
    return ( track_job != nullptr )
        ? track_job->updateBeamElementsRegion( offset, length, new_value )
        : ::NS(ARCH_STATUS_GENERAL_FAILURE);
}

::NS(arch_status_t) NS(TrackJobCl_update_beam_elements_regions)(
    ::NS(TrackJobCl)* SIXTRL_RESTRICT track_job,
    ::NS(arch_size_t) const num_regions_to_update,
    ::NS(arch_size_t) const* offsets, NS(arch_size_t) const* lengths,
    void const* SIXTRL_RESTRICT const* SIXTRL_RESTRICT new_value )
{
    return ( track_job != nullptr )
        ? track_job->updateBeamElementsRegions(
            num_regions_to_update, offsets, lengths, new_value )
        : ::NS(ARCH_STATUS_GENERAL_FAILURE);
}

/* end: /opencl/internal/track_job_cl.cpp */
