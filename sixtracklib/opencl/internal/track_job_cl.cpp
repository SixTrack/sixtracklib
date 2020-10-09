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

namespace SIXTRL_CXX_NAMESPACE
{
    namespace
    {
        namespace st = SIXTRL_CXX_NAMESPACE;
        using tjob_t      = st::TrackJobCl;
        using st_size_t   = tjob_t::size_type;
        using st_status_t = tjob_t::status_t;
    }

    tjob_t::TrackJobCl(
        std::string const& SIXTRL_RESTRICT_REF device_id_str,
        std::string const& SIXTRL_RESTRICT_REF config_str ) :
        TrackJobBase( st::TRACK_JOB_CL_STR, st::TRACK_JOB_CL_ID ),
        m_ptr_context( nullptr ),
        m_ptr_particles_buffer_arg( nullptr ),
        m_ptr_beam_elements_buffer_arg( nullptr ),
        m_ptr_output_buffer_arg( nullptr ),
        m_ptr_particles_addr_buffer_arg( nullptr ),
        m_ptr_elem_by_elem_config_arg( nullptr ),
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

    tjob_t::TrackJobCl(
        const char *const SIXTRL_RESTRICT device_id_str,
        const char *const SIXTRL_RESTRICT config_str ) :
        TrackJobBase( st::TRACK_JOB_CL_STR, st::TRACK_JOB_CL_ID ),
        m_ptr_context( nullptr ),
        m_ptr_particles_buffer_arg( nullptr ),
        m_ptr_beam_elements_buffer_arg( nullptr ),
        m_ptr_output_buffer_arg( nullptr ),
        m_ptr_particles_addr_buffer_arg( nullptr ),
        m_ptr_elem_by_elem_config_arg( nullptr ),
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

    tjob_t::TrackJobCl(
        const char *const SIXTRL_RESTRICT device_id_str,
        tjob_t::c_buffer_t* SIXTRL_RESTRICT particles_buffer,
        tjob_t::c_buffer_t* SIXTRL_RESTRICT beam_elements_buffer,
        tjob_t::c_buffer_t* SIXTRL_RESTRICT ptr_output_buffer,
        st_size_t const until_turn_elem_by_elem,
        const char *const SIXTRL_RESTRICT config_str ) :
        TrackJobBase( st::TRACK_JOB_CL_STR, st::TRACK_JOB_CL_ID ),
        m_ptr_context( nullptr ),
        m_ptr_particles_buffer_arg( nullptr ),
        m_ptr_beam_elements_buffer_arg( nullptr ),
        m_ptr_output_buffer_arg( nullptr ),
        m_ptr_particles_addr_buffer_arg( nullptr ),
        m_ptr_elem_by_elem_config_arg( nullptr ),
        m_total_num_particles( TrackJobBase::size_type{ 0 } )
    {
        this->doInitTrackJobCl( device_id_str, particles_buffer,
            this->particleSetIndicesBegin(), this->particleSetIndicesEnd(),
            beam_elements_buffer, ptr_output_buffer, until_turn_elem_by_elem,
            config_str );
    }

    tjob_t::TrackJobCl(
        const char *const SIXTRL_RESTRICT device_id_str,
        tjob_t::c_buffer_t* SIXTRL_RESTRICT particles_buffer,
        st_size_t const num_particle_sets,
        st_size_t const* SIXTRL_RESTRICT pset_begin,
        tjob_t::c_buffer_t* SIXTRL_RESTRICT belements_buffer,
        tjob_t::c_buffer_t* SIXTRL_RESTRICT ptr_output_buffer,
        st_size_t const until_turn_elem_by_elem,
        const char *const SIXTRL_RESTRICT config_str ) :
        TrackJobBase( st::TRACK_JOB_CL_STR, st::TRACK_JOB_CL_ID ),
        m_ptr_context( nullptr ),
        m_ptr_particles_buffer_arg( nullptr ),
        m_ptr_beam_elements_buffer_arg( nullptr ),
        m_ptr_output_buffer_arg( nullptr ),
        m_ptr_particles_addr_buffer_arg( nullptr ),
        m_ptr_elem_by_elem_config_arg( nullptr ),
        m_total_num_particles( TrackJobBase::size_type{ 0 } )
    {
        using size_t  = st_size_t;
        size_t const* pset_end = pset_begin;

        if( ( pset_end != nullptr ) && ( num_particle_sets > size_t{ 0 } ) )
        {
            std::advance( pset_end, num_particle_sets );
        }

        this->doInitTrackJobCl( device_id_str, particles_buffer,
            pset_begin, pset_end, belements_buffer, ptr_output_buffer,
            until_turn_elem_by_elem, config_str );
    }

    tjob_t::TrackJobCl(
        std::string const& SIXTRL_RESTRICT_REF device_id_str,
        tjob_t::buffer_t& SIXTRL_RESTRICT_REF particles_buffer,
        tjob_t::buffer_t& SIXTRL_RESTRICT_REF belements_buffer,
        tjob_t::buffer_t* SIXTRL_RESTRICT ptr_output_buffer,
        st_size_t const until_turn_elem_by_elem,
        std::string const& config_str ) :
        TrackJobBase( st::TRACK_JOB_CL_STR, st::TRACK_JOB_CL_ID ),
        m_ptr_context( nullptr ),
        m_ptr_particles_buffer_arg( nullptr ),
        m_ptr_beam_elements_buffer_arg( nullptr ),
        m_ptr_output_buffer_arg( nullptr ),
        m_ptr_particles_addr_buffer_arg( nullptr ),
        m_ptr_elem_by_elem_config_arg( nullptr ),
        m_total_num_particles( TrackJobBase::size_type{ 0 } )
    {
        using c_buffer_t = tjob_t::c_buffer_t;

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

    tjob_t::~TrackJobCl() SIXTRL_NOEXCEPT {}

    tjob_t::cl_context_t& tjob_t::context() SIXTRL_NOEXCEPT
    {
        return const_cast< tjob_t::cl_context_t& >(
            static_cast< tjob_t const& >( *this ).context() );
    }


    tjob_t::cl_context_t const& tjob_t::context() const SIXTRL_NOEXCEPT
    {
        SIXTRL_ASSERT( this->ptrContext() != nullptr );
        return *( this->ptrContext() );
    }

    ::NS(ClContext)* tjob_t::ptrContext() SIXTRL_NOEXCEPT
    {
        return this->m_ptr_context.get();
    }

    ::NS(ClContext) const* tjob_t::ptrContext() const SIXTRL_NOEXCEPT
    {
        return this->m_ptr_context.get();
    }

    tjob_t::cl_arg_t& tjob_t::particlesArg() SIXTRL_NOEXCEPT
    {
        return const_cast< tjob_t::cl_arg_t& >(
            static_cast< tjob_t const& >( *this ).particlesArg() );
    }

    tjob_t::cl_arg_t const& tjob_t::particlesArg() const SIXTRL_NOEXCEPT
    {
        SIXTRL_ASSERT( this->ptrParticlesArg() != nullptr );
        return *(this->ptrParticlesArg() );
    }

    tjob_t::cl_arg_t* tjob_t::ptrParticlesArg() SIXTRL_NOEXCEPT
    {
        return const_cast< tjob_t::cl_arg_t* >( static_cast< tjob_t const& >(
            *this ).ptrParticlesArg() );
    }

    tjob_t::cl_arg_t const* tjob_t::ptrParticlesArg() const SIXTRL_NOEXCEPT
    {
        return this->m_ptr_particles_buffer_arg.get();
    }

    tjob_t::cl_arg_t& tjob_t::beamElementsArg() SIXTRL_NOEXCEPT
    {
        return const_cast< tjob_t::cl_arg_t& >( static_cast<
            TrackJobCl const& >( *this ).beamElementsArg() );
    }

    tjob_t::cl_arg_t const& tjob_t::beamElementsArg() const SIXTRL_NOEXCEPT
    {
        SIXTRL_ASSERT( this->ptrBeamElementsArg() != nullptr );
        return *( this->ptrBeamElementsArg() );
    }

    tjob_t::cl_arg_t* tjob_t::ptrBeamElementsArg() SIXTRL_NOEXCEPT
    {
        return const_cast< tjob_t::cl_arg_t* >( static_cast< tjob_t const& >(
            *this ).ptrBeamElementsArg() );
    }

    tjob_t::cl_arg_t const*
    tjob_t::ptrBeamElementsArg() const SIXTRL_NOEXCEPT
    {
        return this->m_ptr_beam_elements_buffer_arg.get();
    }

    tjob_t::cl_arg_t& tjob_t::outputBufferArg() SIXTRL_NOEXCEPT
    {
        return const_cast< tjob_t::cl_arg_t& >( static_cast< tjob_t const& >(
            *this ).outputBufferArg() );
    }

    tjob_t::cl_arg_t const& tjob_t::outputBufferArg() const SIXTRL_NOEXCEPT
    {
        SIXTRL_ASSERT( this->ptrOutputBufferArg() != nullptr );
        return *( this->m_ptr_output_buffer_arg );
    }

    tjob_t::cl_arg_t* tjob_t::ptrOutputBufferArg() SIXTRL_NOEXCEPT
    {
        return const_cast< tjob_t::cl_arg_t* >( static_cast< tjob_t const& >(
            *this ).ptrOutputBufferArg() );
    }

    tjob_t::cl_arg_t const*
    tjob_t::ptrOutputBufferArg() const SIXTRL_NOEXCEPT
    {
        return this->m_ptr_output_buffer_arg.get();
    }

    tjob_t::cl_arg_t const&
    tjob_t::particlesAddrBufferArg() const SIXTRL_NOEXCEPT
    {
        SIXTRL_ASSERT( this->ptrParticlesAddrBufferArg() != nullptr );
        return *this->m_ptr_particles_addr_buffer_arg;
    }

    tjob_t::cl_arg_t& tjob_t::particlesAddrBufferArg() SIXTRL_NOEXCEPT
    {
        SIXTRL_ASSERT( this->ptrParticlesAddrBufferArg() != nullptr );
        return *this->m_ptr_particles_addr_buffer_arg;
    }

    tjob_t::cl_arg_t const*
    tjob_t::ptrParticlesAddrBufferArg() const SIXTRL_NOEXCEPT
    {
        return this->m_ptr_particles_addr_buffer_arg.get();
    }

    tjob_t::cl_arg_t* tjob_t::ptrParticlesAddrBufferArg() SIXTRL_NOEXCEPT
    {
        return this->m_ptr_particles_addr_buffer_arg.get();
    }

    /* --------------------------------------------------------------------- */

    tjob_t::cl_arg_t const&
    tjob_t::elemByelemConfigBufferArg() const SIXTRL_NOEXCEPT
    {
        SIXTRL_ASSERT( this->ptrElemByElemConfigBufferArg() != nullptr );
        return *( this->ptrElemByElemConfigBufferArg() );
    }

    tjob_t::cl_arg_t& tjob_t::elemByelemConfigBufferArg() SIXTRL_NOEXCEPT
    {
        using ref_t = tjob_t::cl_arg_t&;
        return const_cast< ref_t >( static_cast< tjob_t const& >(
                *this ).elemByelemConfigBufferArg() );
    }

    tjob_t::cl_arg_t const*
    tjob_t::ptrElemByElemConfigBufferArg() const SIXTRL_NOEXCEPT
    {
        return this->m_ptr_elem_by_elem_config_arg.get();
    }

    tjob_t::cl_arg_t& tjob_t::particles_addr_arg() SIXTRL_NOEXCEPT
    {
        SIXTRL_ASSERT( this->m_ptr_particles_addr_buffer_arg.get() != nullptr );
        return *( this->m_ptr_particles_addr_buffer_arg.get() );
    }

    tjob_t::cl_arg_t const& tjob_t::particles_addr_arg() const SIXTRL_NOEXCEPT
    {
        SIXTRL_ASSERT( this->m_ptr_particles_addr_buffer_arg.get() != nullptr );
        return *( this->m_ptr_particles_addr_buffer_arg.get() );
    }

    tjob_t::cl_arg_t* tjob_t::ptr_particles_addr_arg() SIXTRL_NOEXCEPT
    {
        return this->m_ptr_particles_addr_buffer_arg.get();
    }

    tjob_t::cl_arg_t const*
        tjob_t::ptr_particles_addr_arg() const SIXTRL_NOEXCEPT
    {
        return this->m_ptr_particles_addr_buffer_arg.get();
    }

    tjob_t::cl_arg_t* tjob_t::ptrElemByElemConfigBufferArg() SIXTRL_NOEXCEPT
    {
        return this->m_ptr_elem_by_elem_config_arg.get();
    }

    tjob_t::status_t tjob_t::updateBeamElementsRegion(
        st_size_t const offset, st_size_t const length,
        void const* SIXTRL_RESTRICT new_value )
    {
        return ( this->m_ptr_beam_elements_buffer_arg != nullptr )
            ? this->m_ptr_beam_elements_buffer_arg->updateRegion(
                offset, length, new_value )
            : ::NS(ARCH_STATUS_GENERAL_FAILURE);
    }

    tjob_t::status_t tjob_t::updateBeamElementsRegions(
        st_size_t const num_regions_to_update,
        st_size_t const* SIXTRL_RESTRICT offsets,
        st_size_t const* SIXTRL_RESTRICT lengths,
        void const* SIXTRL_RESTRICT const* SIXTRL_RESTRICT new_values )
    {
        return ( this->m_ptr_beam_elements_buffer_arg != nullptr )
            ? this->m_ptr_beam_elements_buffer_arg->updateRegions(
                num_regions_to_update, offsets, lengths, new_values )
            : ::NS(ARCH_STATUS_GENERAL_FAILURE);
    }

    std::uintptr_t tjob_t::opencl_context_addr() const SIXTRL_NOEXCEPT
    {
        return ( this->ptrContext() != nullptr )
            ? this->ptrContext()->openClContextAddr() : std::uintptr_t{ 0 };
    }

    std::uintptr_t tjob_t::opencl_queue_addr() const SIXTRL_NOEXCEPT
    {
        return ( this->ptrContext() != nullptr )
            ? this->ptrContext()->openClQueueAddr() : std::uintptr_t{ 0 };
    }

    st_status_t tjob_t::doFetchParticleAddresses()
    {
        st_status_t status = st::ARCH_STATUS_GENERAL_FAILURE;

        if( ( this->ptrContext() != nullptr ) &&
            ( this->ptrContext()->hasSelectedNode() ) &&
            ( this->ptrContext()->has_fetch_particles_addr_kernel() ) &&
            ( this->ptr_particles_addr_arg() != nullptr ) )
        {
            status = this->ptrContext()->fetch_particles_addr();
        }

        if( status == st::ARCH_STATUS_SUCCESS )
        {
            status = ( this->ptr_particles_addr_arg()->read(
                this->doGetPtrParticlesAddrBuffer() ) )
                    ? st::ARCH_STATUS_SUCCESS : st::ARCH_STATUS_GENERAL_FAILURE;
        }

        return status;
    }
=======
    /* --------------------------------------------------------------------- */

    tjob_t::cl_arg_t const* tjob_t::ptr_const_argument_by_buffer_id(
        tjob_t::size_type const buffer_id ) const SIXTRL_NOEXCEPT
    {
         tjob_t::cl_arg_t const* ptr_arg = nullptr;

        switch( buffer_id )
        {
            case st::ARCH_PARTICLES_BUFFER_ID:
            {
                ptr_arg = this->ptrParticlesArg();
                break;
            }

            case st::ARCH_BEAM_ELEMENTS_BUFFER_ID:
            {
                ptr_arg = this->ptrBeamElementsArg();
                break;
            }

            case st::ARCH_OUTPUT_BUFFER_ID:
            {
                ptr_arg = this->ptrOutputBufferArg();
                break;
            }

            case st::ARCH_ELEM_BY_ELEM_CONFIG_BUFFER_ID:
            {
                ptr_arg = this->ptrElemByElemConfigBufferArg();
                break;
            }

            case st::ARCH_PARTICLE_ADDR_BUFFER_ID:
            {
                ptr_arg = this->ptrParticlesAddrBufferArg();
                break;
            }

            default:
            {
                ptr_arg = this->ptr_const_stored_buffer_argument( buffer_id );
            }
        };

        return ptr_arg;
    }

    tjob_t::cl_arg_t* tjob_t::ptr_argument_by_buffer_id(
        tjob_t::size_type const buffer_id ) SIXTRL_NOEXCEPT
    {
        return const_cast< tjob_t::cl_arg_t* >(
            this->ptr_const_argument_by_buffer_id( buffer_id ) );
    }

    tjob_t::cl_arg_t const& tjob_t::argument_by_buffer_id(
        tjob_t::size_type const buffer_id ) const
    {
        tjob_t::cl_arg_t const* ptr_arg =
            this->ptr_const_argument_by_buffer_id( buffer_id );

        if( ptr_arg == nullptr )
        {
            std::ostringstream a2str;
            a2str << "unable to get buffer argument for buffer_id="
                  << buffer_id;

            throw std::runtime_error( a2str.str() );
        }

        return *ptr_arg;
    }

    tjob_t::cl_arg_t& tjob_t::argument_by_buffer_id(
        tjob_t::size_type const buffer_id )
    {
        return const_cast< tjob_t::cl_arg_t& >( static_cast< tjob_t const& >(
            *this ).argument_by_buffer_id( buffer_id ) );
    }

    /* -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- */

    tjob_t::cl_arg_t const* tjob_t::ptr_const_stored_buffer_argument(
        tjob_t::size_type const buffer_id ) const SIXTRL_NOEXCEPT
    {
        tjob_t::cl_arg_t const* ptr_arg = nullptr;

        st_size_t const min_buffer_id = this->min_stored_buffer_id();
        st_size_t const max_buffer_id_plus_one =
            this->max_stored_buffer_id() + st_size_t{ 1 };

        if( ( min_buffer_id != st::ARCH_ILLEGAL_BUFFER_ID ) &&
            ( buffer_id != st::ARCH_ILLEGAL_BUFFER_ID ) &&
            ( buffer_id >= min_buffer_id ) &&
            ( buffer_id <  max_buffer_id_plus_one ) )
        {
            st_size_t const stored_buffer_id = buffer_id - min_buffer_id;

            if( stored_buffer_id < this->m_stored_buffers_args.size() )
            {
                ptr_arg = this->m_stored_buffers_args[ stored_buffer_id ].get();
            }
        }

        return ptr_arg;
    }

    tjob_t::cl_arg_t* tjob_t::ptr_stored_buffer_argument(
        tjob_t::size_type const buffer_id ) SIXTRL_NOEXCEPT
    {
        return const_cast< tjob_t::cl_arg_t* >(
            this->ptr_const_stored_buffer_argument( buffer_id ) );
    }

    tjob_t::cl_arg_t const& tjob_t::stored_buffer_argument(
        tjob_t::size_type const buffer_id ) const
    {
        tjob_t::cl_arg_t const* ptr_arg =
            this->ptr_const_stored_buffer_argument( buffer_id );

        if( ptr_arg == nullptr )
        {
            std::ostringstream a2str;
            a2str << "unable to get stored buffer argument for buffer_id="
                  << buffer_id;

            throw std::runtime_error( a2str.str() );
        }

        return *ptr_arg;
    }

    tjob_t::cl_arg_t& tjob_t::stored_buffer_argument(
        tjob_t::size_type const buffer_id )
    {
        return const_cast< tjob_t::cl_arg_t& >( static_cast<
            tjob_t const& >( *this ).stored_buffer_argument( buffer_id ) );
    }

    /* --------------------------------------------------------------------- */

    bool tjob_t::doPrepareParticlesStructures(
        tjob_t::c_buffer_t* SIXTRL_RESTRICT pb )
    {
        return ( ( TrackJobBase::doPrepareParticlesStructures( pb ) ) &&
                 ( this->doPrepareParticlesStructuresOclImp( pb ) ) );
    }

    bool tjob_t::doPrepareBeamElementsStructures(
        tjob_t::c_buffer_t* SIXTRL_RESTRICT belems )
    {
        return ( ( TrackJobBase::doPrepareBeamElementsStructures( belems ) ) &&
            ( this->doPrepareBeamElementsStructuresOclImp( belems ) ) );
    }

    bool tjob_t::doPrepareOutputStructures(
        tjob_t::c_buffer_t* SIXTRL_RESTRICT part_buffer,
        tjob_t::c_buffer_t* SIXTRL_RESTRICT belem_buffer,
        tjob_t::c_buffer_t* SIXTRL_RESTRICT output_buffer,
        st_size_t const until_turn_elem_by_elem )
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

    bool tjob_t::doAssignOutputBufferToBeamMonitors(
        tjob_t::c_buffer_t* SIXTRL_RESTRICT belem_buffer,
        tjob_t::c_buffer_t* SIXTRL_RESTRICT out_buffer,
        tjob_t::particle_index_t const min_turn_id,
        tjob_t::size_type const output_buffer_offset_index )
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

    bool tjob_t::doAssignOutputBufferToElemByElemConfig(
        tjob_t::elem_by_elem_config_t* SIXTRL_RESTRICT elem_by_elem_conf,
        tjob_t::c_buffer_t* SIXTRL_RESTRICT out_buffer,
        tjob_t::size_type const output_buffer_offset_index )
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

    bool tjob_t::doReset(
        tjob_t::c_buffer_t* SIXTRL_RESTRICT particles_buffer,
        tjob_t::c_buffer_t* SIXTRL_RESTRICT beam_elem_buffer,
        tjob_t::c_buffer_t* SIXTRL_RESTRICT ptr_output_buffer,
        st_size_t const until_turn_elem_by_elem )
    {
        return this->doResetOclImp( particles_buffer, beam_elem_buffer,
                ptr_output_buffer, until_turn_elem_by_elem );
    }

    /* --------------------------------------------------------------------- */

    tjob_t::size_type tjob_t::doAddStoredBuffer(
        tjob_t::buffer_store_t&& assigned_buffer_handle )
    {
        tjob_t::size_type buffer_id = _base_t::doAddStoredBuffer(
            std::move( assigned_buffer_handle ) );

        if( buffer_id != st::ARCH_ILLEGAL_BUFFER_ID )
        {
            if( st::ARCH_STATUS_SUCCESS != this->doAddStoredBufferOclImpl(
                    buffer_id ) )
            {
                this->doRemoveStoredBufferOclImpl( buffer_id );
                buffer_id = st::ARCH_ILLEGAL_BUFFER_ID;
            }
        }

        return buffer_id;
    }

    tjob_t::status_t tjob_t::doRemoveStoredBuffer(
        tjob_t::size_type const buffer_id )
    {
        tjob_t::status_t status = _base_t::doRemoveStoredBuffer( buffer_id );
        status |= this->doRemoveStoredBufferOclImpl( buffer_id );

        return status;
    }

    tjob_t::status_t tjob_t::doPushStoredBuffer(
        tjob_t::size_type const buffer_id )
    {
        return this->doPushStoredBufferOclImpl( buffer_id );
    }

    tjob_t::status_t tjob_t::doCollectStoredBuffer(
        tjob_t::size_type const buffer_id )
    {
        return this->doCollectStoredBufferOclImpl( buffer_id );
    }

    tjob_t::status_t tjob_t::doPerformAddressAssignments(
        tjob_t::assign_item_key_t const& SIXTRL_RESTRICT_REF assign_item_key )
    {
        return this->doPerformAddressAssignmentsOclImpl( assign_item_key );
    }

    /* --------------------------------------------------------------------- */

    tjob_t::track_status_t tjob_t::doTrackUntilTurn(
        st_size_t const until_turn )
    {
        return st::track( *this, until_turn );
    }

    tjob_t::track_status_t tjob_t::doTrackElemByElem(
        st_size_t const until_turn )
    {
        return st::trackElemByElem( *this, until_turn );
    }

    tjob_t::track_status_t tjob_t::doTrackLine(
        st_size_t const line_begin_idx, st_size_t const line_end_idx,
        bool const finish_turn )
    {
        return st::trackLine(
            *this, line_begin_idx, line_end_idx, finish_turn );
    }


    void tjob_t::doCollect( tjob_t::collect_flag_t const flags )
    {
        st::collect( *this, flags );
    }

    void tjob_t::doPush( tjob_t::push_flag_t const flags )
    {
        st::push( *this, flags );
    }

    void tjob_t::doParseConfigStr( const char *const SIXTRL_RESTRICT  ) {}

    void tjob_t::doUpdateStoredContext(
        tjob_t::ptr_cl_context_t&& context )
    {
        this->m_ptr_context = std::move( context );
    }

    void tjob_t::doUpdateStoredClParticlesArg(
        tjob_t::ptr_cl_arg_t&& particle_arg )
    {
        this->m_ptr_particles_buffer_arg = std::move( particle_arg );
    }

    void tjob_t::doUpdateStoredClBeamElementsArg(
        tjob_t::ptr_cl_arg_t&& beam_elements_arg )
    {
        this->m_ptr_beam_elements_buffer_arg = std::move( beam_elements_arg );
    }

    void tjob_t::doUpdateStoredClOutputArg(
        tjob_t::ptr_cl_arg_t&& output_arg )
    {
        this->m_ptr_output_buffer_arg = std::move( output_arg );
    }

    void tjob_t::doUpdateStoredClElemByElemConfigArg(
            tjob_t::ptr_cl_arg_t&& cl_elem_by_elem_config_buffer )
    {
        this->m_ptr_elem_by_elem_config_arg =
            std::move( cl_elem_by_elem_config_buffer );
    }

    tjob_t::status_t tjob_t::do_update_stored_particles_addr_arg(
        tjob_t::ptr_cl_arg_t&& particles_addr_arg )
    {
        st_status_t status = st::ARCH_STATUS_GENERAL_FAILURE;

        this->m_ptr_particles_addr_buffer_arg =
            std::move( particles_addr_arg );

        if( ( this->ptr_particles_addr_arg() != nullptr ) &&
            ( this->ptr_particles_addr_arg()->usesCObjectBuffer() ) &&
            ( this->ptr_particles_addr_arg()->context() ==
              this->ptrContext() ) &&
            ( ( !this->context().hasSelectedNode() ) ||
              ( ( this->context().assign_particles_addr_buffer_arg(
                  this->particles_addr_arg() ) ==
                  st::ARCH_STATUS_SUCCESS ) ) ) )
        {
            status = st::ARCH_STATUS_SUCCESS;
        }

        return status;
    }

    st_size_t tjob_t::totalNumParticles() const SIXTRL_NOEXCEPT
    {
        return this->m_total_num_particles;
    }

    void tjob_t::doSetTotalNumParticles(
        st_size_t const total_num_particles ) SIXTRL_NOEXCEPT
    {
        this->m_total_num_particles = total_num_particles;
    }

    bool tjob_t::doPrepareContext(
        char const* SIXTRL_RESTRICT device_id_str,
        char const* SIXTRL_RESTRICT ptr_config_str )
    {
        return this->doPrepareContextOclImpl( device_id_str, ptr_config_str );
    }

    /* --------------------------------------------------------------------- */

    bool tjob_t::doPrepareContextOclImpl(
        const char *const SIXTRL_RESTRICT device_id_str,
        const char *const SIXTRL_RESTRICT ptr_config_str )
    {
        using tjob_t       = TrackJobCl;
        using context_t     = tjob_t::cl_context_t;
        using ptr_context_t = tjob_t::ptr_cl_context_t;

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

    void tjob_t::doParseConfigStrOclImpl(
        const char *const SIXTRL_RESTRICT ) {}

    bool tjob_t::doPrepareParticlesStructuresOclImp(
        tjob_t::c_buffer_t* SIXTRL_RESTRICT pb )
    {
        using tjob_t = TrackJobCl;

        using arg_t        = tjob_t::cl_arg_t;
        using ptr_arg_t    = tjob_t::ptr_cl_arg_t;
        using size_t       = tjob_t::size_type;

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
            this->doUpdateStoredClParticlesArg( std::move( particles_arg ) );

            ptr_arg_t particles_addr_arg( new arg_t(
                this->doGetPtrParticlesAddrBuffer(), this->ptrContext() ) );

            success = ( st::ARCH_STATUS_SUCCESS ==
                this->do_update_stored_particles_addr_arg(
                    std::move( particles_addr_arg ) ) );

            if( ( success ) &&
                ( total_num_particles > size_t{ 0 } ) &&
                ( this->ptrParticlesArg() != nullptr ) &&
                ( this->ptr_particles_addr_arg() != nullptr ) &&
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

    bool tjob_t::doPrepareBeamElementsStructuresOclImp(
        tjob_t::c_buffer_t* SIXTRL_RESTRICT belems )
    {
        using tjob_t = TrackJobCl;

        using arg_t     = tjob_t::cl_arg_t;
        using ptr_arg_t = tjob_t::ptr_cl_arg_t;

        bool success = false;

        if( ( this->ptrContext() != nullptr ) && ( belems != nullptr ) )
        {
            ptr_arg_t belems_arg( new arg_t( belems, this->ptrContext() ) );
            this->doUpdateStoredClBeamElementsArg( std::move( belems_arg ) );

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

    bool tjob_t::doPrepareOutputStructuresOclImpl(
        tjob_t::c_buffer_t* SIXTRL_RESTRICT pb,
        tjob_t::c_buffer_t* SIXTRL_RESTRICT belems,
        tjob_t::c_buffer_t* SIXTRL_RESTRICT ptr_out_buffer,
        st_size_t const until_turn_elem_by_elem )
    {
        using tjob_t = TrackJobCl;

        using arg_t         = tjob_t::cl_arg_t;
        using ptr_arg_t     = tjob_t::ptr_cl_arg_t;

        bool success = true;

        ( void )until_turn_elem_by_elem;
        ( void )belems;

        if( ptr_out_buffer != nullptr )
        {
            ptr_arg_t ptr( new arg_t( ptr_out_buffer, this->ptrContext() ) );
            this->doUpdateStoredClOutputArg( std::move( ptr ) );

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
                ptr_arg_t ptr( new arg_t( this->ptrElemByElemConfigCBuffer(),
                                          this->ptrContext() ) );

                this->doUpdateStoredClElemByElemConfigArg( std::move( ptr ) );

                success = ( ( this->ptrElemByElemConfigCBuffer() != nullptr ) &&
                    ( this->ptrElemByElemConfigBufferArg() != nullptr ) );

                if( success )
                {
                    success = this->ptrElemByElemConfigBufferArg()->write(
                        this->ptrElemByElemConfigCBuffer() );
                }

                if( success )
                {
                    success = ( st::ARCH_STATUS_SUCCESS ==
                        this->ptrContext()->assign_elem_by_elem_config_buffer_arg(
                            *this->ptrElemByElemConfigBufferArg() ) );
                }

                if( success )
                {
                    success = ( st::ARCH_STATUS_SUCCESS ==
                        this->ptrContext()->assign_elem_by_elem_config_index_arg(
                            st_size_t{ 0 } ) );
                }
            }
        }

        return success;
    }

    bool tjob_t::doAssignOutputBufferToBeamMonitorsOclImp(
        tjob_t::c_buffer_t* SIXTRL_RESTRICT belem_buffer,
        tjob_t::c_buffer_t* SIXTRL_RESTRICT output_buffer,
        tjob_t::particle_index_t const min_turn_id,
        tjob_t::size_type const output_buffer_index_offset )
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

    bool tjob_t::doAssignOutputBufferToElemByElemConfigOclImpl(
        tjob_t::elem_by_elem_config_t* SIXTRL_RESTRICT elem_by_elem_conf,
        tjob_t::c_buffer_t* SIXTRL_RESTRICT out_buffer,
        tjob_t::size_type const output_buffer_index_offset )
    {
        bool success = false;

        if( ( this->ptrContext() != nullptr ) &&
            ( this->context().hasSelectedNode() ) &&
            ( this->context().has_assign_elem_by_elem_output_kernel() ) &&
            ( this->hasElemByElemOutput() ) &&
            ( this->ptrElemByElemConfigBufferArg() != nullptr ) &&
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
                tjob_t::c_buffer_t* elem_by_elem_buffer =
                    this->buffer_by_buffer_id(
                        st::ARCH_ELEM_BY_ELEM_CONFIG_BUFFER_ID );

                success = ( ( elem_by_elem_buffer != nullptr ) &&
                            ( this->ptrElemByElemConfigBufferArg()->read(
                                elem_by_elem_buffer ) ) );
            }

            success = true;
        }

        return success;
    }

    bool tjob_t::doResetOclImp(
        tjob_t::c_buffer_t* SIXTRL_RESTRICT pbuffer,
        tjob_t::c_buffer_t* SIXTRL_RESTRICT belem_buffer,
        tjob_t::c_buffer_t* SIXTRL_RESTRICT out_buffer,
        st_size_t const until_turn_elem_by_elem )
    {
        using tjob_t = st::TrackJobCl;
        using _base_t = st::TrackJobBase;
        using output_buffer_flag_t = tjob_t::output_buffer_flag_t;

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

    /* --------------------------------------------------------------------- */

    tjob_t::status_t tjob_t::doAddStoredBufferOclImpl(
        tjob_t::size_type const buffer_id )
    {
        tjob_t::status_t status = st::ARCH_STATUS_GENERAL_FAILURE;

        tjob_t::buffer_store_t* ptr_stored_buffer =
                this->doGetPtrBufferStore( buffer_id );

        tjob_t::context_t* ptr_context = this->ptrContext();

        if( ( ptr_context != nullptr ) && ( ptr_stored_buffer != nullptr ) &&
            ( buffer_id >= st::ARCH_MIN_USER_DEFINED_BUFFER_ID ) )
        {
            tjob_t::size_type const stored_buffer_id =
                buffer_id - st::ARCH_MIN_USER_DEFINED_BUFFER_ID;

            tjob_t::size_type const nn = this->doGetStoredBufferSize();
            tjob_t::size_type ii = this->m_stored_buffers_args.size();

            for( ; ii < nn ; ++ii )
            {
                this->m_stored_buffers_args.emplace_back( nullptr );
            }

            SIXTRL_ASSERT( this->m_stored_buffers_args.size() >= nn );
            SIXTRL_ASSERT( this->m_stored_buffers_args.size() >
                           stored_buffer_id );

            if( ptr_stored_buffer->ptr_cxx_buffer() != nullptr )
            {
                this->m_stored_buffers_args[ stored_buffer_id ].reset(
                    new tjob_t::cl_arg_t( *ptr_stored_buffer->ptr_cxx_buffer(),
                            ptr_context ) );

                status = st::ARCH_STATUS_SUCCESS;

            }
            else if( ptr_stored_buffer->ptr_buffer() != nullptr )
            {
                this->m_stored_buffers_args[ stored_buffer_id ].reset(
                    new tjob_t::cl_arg_t( ptr_stored_buffer->ptr_buffer(),
                            ptr_context ) );

                status = st::ARCH_STATUS_SUCCESS;
            }
        }

        return status;
    }

    tjob_t::status_t tjob_t::doRemoveStoredBufferOclImpl(
        tjob_t::size_type const buffer_id )
    {
        tjob_t::status_t status = st::ARCH_STATUS_GENERAL_FAILURE;

        if( buffer_id >= st::ARCH_MIN_USER_DEFINED_BUFFER_ID )
        {
            tjob_t::size_type const stored_buffer_id =
                buffer_id - st::ARCH_MIN_USER_DEFINED_BUFFER_ID;

            SIXTRL_ASSERT( this->m_stored_buffers_args.size() >
                           stored_buffer_id );

            this->m_stored_buffers_args[ stored_buffer_id ].reset( nullptr );
            status = st::ARCH_STATUS_SUCCESS;
        }

        return status;
    }

    tjob_t::status_t tjob_t::doPushStoredBufferOclImpl(
        tjob_t::size_type const buffer_id )
    {
        tjob_t::status_t status = st::ARCH_STATUS_GENERAL_FAILURE;

        tjob_t::buffer_store_t* ptr_stored_buffer =
                this->doGetPtrBufferStore( buffer_id );

        tjob_t::cl_arg_t* ptr_arg =
            this->ptr_stored_buffer_argument( buffer_id );

        if( ( ptr_stored_buffer != nullptr ) && ( ptr_arg != nullptr ) )
        {
            SIXTRL_ASSERT( ptr_arg->context() == this->ptrContext() );

            if( ptr_stored_buffer->ptr_cxx_buffer() != nullptr )
            {
                status = ( ptr_arg->write(
                    *ptr_stored_buffer->ptr_cxx_buffer() ) )
                    ? st::ARCH_STATUS_SUCCESS
                    : st::ARCH_STATUS_GENERAL_FAILURE;
            }
            else if( ptr_stored_buffer->ptr_buffer() != nullptr )
            {
                status = ( ptr_arg->write(
                    ptr_stored_buffer->ptr_buffer() ) )
                    ? st::ARCH_STATUS_SUCCESS
                    : st::ARCH_STATUS_GENERAL_FAILURE;
            }
        }

        return status;
    }

    tjob_t::status_t tjob_t::doCollectStoredBufferOclImpl(
        tjob_t::size_type const buffer_id )
    {
        tjob_t::status_t status = st::ARCH_STATUS_GENERAL_FAILURE;

        tjob_t::buffer_store_t* ptr_stored_buffer =
                this->doGetPtrBufferStore( buffer_id );

        tjob_t::cl_arg_t* ptr_arg =
            this->ptr_stored_buffer_argument( buffer_id );

        if( ( ptr_stored_buffer != nullptr ) && ( ptr_arg != nullptr ) )
        {
            SIXTRL_ASSERT( ptr_arg->context() == this->ptrContext() );

            if( ptr_stored_buffer->ptr_cxx_buffer() != nullptr )
            {
                status = ( ptr_arg->read( *ptr_stored_buffer->ptr_cxx_buffer() ) )
                    ? st::ARCH_STATUS_SUCCESS
                    : st::ARCH_STATUS_GENERAL_FAILURE;
            }
            else if( ptr_stored_buffer->ptr_buffer() != nullptr )
            {
                status = ( ptr_arg->read( ptr_stored_buffer->ptr_buffer() ) )
                    ? st::ARCH_STATUS_SUCCESS
                    : st::ARCH_STATUS_GENERAL_FAILURE;
            }
        }

        return status;
    }

    tjob_t::status_t tjob_t::doAddAssignAddressItemOclImpl(
        tjob_t::assign_item_t const& SIXTRL_RESTRICT_REF assign_item,
        tjob_t::size_type* SIXTRL_RESTRICT ptr_item_index )
    {
        ( void )assign_item;
        ( void )ptr_item_index;

        return st::ARCH_STATUS_SUCCESS;
    }

    tjob_t::status_t tjob_t::doPerformAddressAssignmentsOclImpl(
        tjob_t::assign_item_key_t const& SIXTRL_RESTRICT_REF key )
    {
        using size_t = tjob_t::size_type;
        using c_buffer_t = tjob_t::c_buffer_t;
        using cl_arg_t = tjob_t::cl_arg_t;

        tjob_t::status_t status = st::ARCH_STATUS_GENERAL_FAILURE;

        size_t const dest_buffer_id = key.dest_buffer_id;
        c_buffer_t* dest_buffer = this->buffer_by_buffer_id( dest_buffer_id );
        cl_arg_t* dest_arg = this->ptr_argument_by_buffer_id( dest_buffer_id );

        size_t const src_buffer_id = key.src_buffer_id;
        c_buffer_t const* src_buffer =
            this->buffer_by_buffer_id( src_buffer_id );
        cl_arg_t* src_arg = this->ptr_argument_by_buffer_id( src_buffer_id );

        c_buffer_t* assign_buffer =
            this->doGetPtrAssignAddressItemsBuffer( key );

        if( ( this->has_assign_items( dest_buffer_id, src_buffer_id ) ) &&
            ( this->ptrContext() != nullptr ) && ( assign_buffer != nullptr ) &&
            ( src_buffer != nullptr ) && ( dest_buffer != nullptr ) &&
            ( src_arg != nullptr ) && ( src_arg->usesCObjectBuffer() ) &&
            ( src_arg->ptrCObjectBuffer() == src_buffer ) &&
            ( src_arg->context() == this->ptrContext() ) &&
            ( dest_arg != nullptr ) && ( dest_arg->usesCObjectBuffer() ) &&
            ( dest_arg->ptrCObjectBuffer() == dest_buffer ) &&
            ( dest_arg->context() == this->ptrContext() ) )
        {
            status = st::ARCH_STATUS_SUCCESS;

            if( ::NS(Buffer_get_num_of_objects)( assign_buffer ) > size_t{ 0 } )
            {
                cl_arg_t assign_items_arg( assign_buffer, this->ptrContext() );
                status = this->ptrContext()->assign_addresses( assign_items_arg,
                    *dest_arg, dest_buffer_id, *src_arg, src_buffer_id );
            }
        }

        return status;
    }

    /* ********************************************************************* */

    void collect( TrackJobCl& SIXTRL_RESTRICT_REF job ) SIXTRL_NOEXCEPT
    {
        st::collect( job, job.collectFlags() );
    }

    void collect( TrackJobCl& SIXTRL_RESTRICT_REF job,
                  st::track_job_collect_flag_t const flags ) SIXTRL_NOEXCEPT
    {
        if( ( tjob_t::IsCollectFlagSet(
                flags, TRACK_JOB_IO_PARTICLES ) ) &&
            ( job.ptrParticlesArg() != nullptr ) &&
            ( job.ptrCParticlesBuffer() != nullptr ) )
        {
            job.particlesArg().read( job.ptrCParticlesBuffer() );
        }

        if( ( tjob_t::IsCollectFlagSet(
                flags, TRACK_JOB_IO_OUTPUT ) ) &&
            ( job.ptrOutputBufferArg() != nullptr ) &&
            ( job.ptrCOutputBuffer() != nullptr ) )
        {
            job.outputBufferArg().read( job.ptrCOutputBuffer() );
        }

        if( ( tjob_t::IsCollectFlagSet(
                flags, TRACK_JOB_IO_BEAM_ELEMENTS ) ) &&
            ( job.ptrBeamElementsArg() != nullptr ) &&
            ( job.ptrCBeamElementsBuffer() != nullptr ) )
        {
            job.beamElementsArg().read( job.ptrCBeamElementsBuffer() );
        }

        if( ( tjob_t::IsCollectFlagSet(
                flags, st::TRACK_JOB_IO_PARTICLES_ADDR ) ) &&
            ( job.ptr_particles_addr_arg() != nullptr ) )
        {
            job.particles_addr_arg().read( job.ptrParticleAddressesBuffer() );
        }
    }

    void push( TrackJobCl& SIXTRL_RESTRICT_REF job,
        st::track_job_collect_flag_t const flags ) SIXTRL_NOEXCEPT
    {
        namespace st = SIXTRL_CXX_NAMESPACE;
        bool success = true;

         if( ( tjob_t::IsCollectFlagSet(
                flags, st::TRACK_JOB_IO_PARTICLES ) ) &&
            ( job.ptrParticlesArg() != nullptr ) &&
            ( job.ptrCParticlesBuffer() != nullptr ) )
        {
            success &= job.particlesArg().write( job.ptrCParticlesBuffer() );
        }

        if( ( tjob_t::IsCollectFlagSet(
                flags, st::TRACK_JOB_IO_OUTPUT ) ) &&
            ( job.ptrOutputBufferArg() != nullptr ) &&
            ( job.ptrCOutputBuffer() != nullptr ) )
        {
            success &= job.outputBufferArg().write( job.ptrCOutputBuffer() );
        }

        if( ( tjob_t::IsCollectFlagSet(
                flags, st::TRACK_JOB_IO_BEAM_ELEMENTS ) ) &&
            ( job.ptrBeamElementsArg() != nullptr ) &&
            ( job.ptrCBeamElementsBuffer() != nullptr ) )
        {
            success &= job.beamElementsArg().write(
                job.ptrCBeamElementsBuffer() );
        }

        if( ( tjob_t::IsCollectFlagSet(
                flags, st::TRACK_JOB_IO_PARTICLES_ADDR ) ) &&
            ( job.ptr_particles_addr_arg() != nullptr ) )
        {
            success &= job.particles_addr_arg().write(
                job.ptrParticleAddressesBuffer() );
        }

        ( void )success;
    }

    tjob_t::track_status_t track(
        TrackJobCl& SIXTRL_RESTRICT_REF job,
        st_size_t const until_turn ) SIXTRL_NOEXCEPT
    {
        tjob_t::track_status_t status = st::TRACK_STATUS_GENERAL_FAILURE;

        SIXTRL_ASSERT( job.ptrContext() != nullptr );
        SIXTRL_ASSERT( job.ptrParticlesArg() != nullptr );
        SIXTRL_ASSERT( job.ptrBeamElementsArg() != nullptr );
        SIXTRL_ASSERT( job.ptrContext()->hasSelectedNode() );
        SIXTRL_ASSERT( job.context().has_track_until_kernel() );
        SIXTRL_ASSERT( job.particleSetIndicesBegin() != nullptr );
        SIXTRL_ASSERT( job.numParticlesInSetsBegin() != nullptr );

        st_size_t const num_psets = job.numParticleSets();

        if( num_psets == st_size_t{ 1 } )
        {
            SIXTRL_ASSERT( job.context().selected_particle_set() ==
                *job.particleSetIndicesBegin() );

            SIXTRL_ASSERT( job.context().num_particles_in_selected_set() ==
                *job.numParticlesInSetsBegin() );

            status = job.context().track_until( until_turn );
        }
        else if( num_psets > st_size_t{ 1 } )
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
                job.context().track_until_kernel_id(), st_size_t{ 1 },
                    saved_pset_idx_arg );
        }

        return status;
    }

    tjob_t::track_status_t trackElemByElem(
        TrackJobCl& SIXTRL_RESTRICT_REF job,
        st_size_t const until_turn ) SIXTRL_NOEXCEPT
    {
        tjob_t::track_status_t status = st::TRACK_STATUS_GENERAL_FAILURE;

        SIXTRL_ASSERT( job.ptrContext() != nullptr );
        SIXTRL_ASSERT( job.ptrParticlesArg() != nullptr );
        SIXTRL_ASSERT( job.ptrBeamElementsArg() != nullptr );
        SIXTRL_ASSERT( job.hasOutputBuffer() );
        SIXTRL_ASSERT( job.ptrOutputBufferArg() != nullptr );
        SIXTRL_ASSERT( job.hasElemByElemOutput() );
        SIXTRL_ASSERT( job.ptrElemByElemConfig() != nullptr );
        SIXTRL_ASSERT( job.ptrElemByElemConfigBufferArg() != nullptr );
        SIXTRL_ASSERT( job.ptrContext()->hasSelectedNode() );
        SIXTRL_ASSERT( job.context().has_track_elem_by_elem_kernel() );
        SIXTRL_ASSERT( job.particleSetIndicesBegin() != nullptr );
        SIXTRL_ASSERT( job.numParticlesInSetsBegin() != nullptr );

        st_size_t const num_psets = job.numParticleSets();

        if( num_psets == st_size_t{ 1 } )
        {
            SIXTRL_ASSERT( job.context().selected_particle_set() ==
                *job.particleSetIndicesBegin() );

            SIXTRL_ASSERT( job.context().num_particles_in_selected_set() ==
                *job.numParticlesInSetsBegin() );

            status = job.context().track_elem_by_elem( until_turn );
        }
        else if( num_psets > st_size_t{ 1 } )
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
                job.context().track_elem_by_elem_kernel_id(), st_size_t{ 1 },
                    saved_pset_idx_arg );
        }

        return status;
    }

    tjob_t::track_status_t trackLine(
        TrackJobCl& SIXTRL_RESTRICT_REF job,
        st_size_t const line_begin_idx,
        st_size_t const line_end_idx,
        bool const finish_turn ) SIXTRL_NOEXCEPT
    {
        tjob_t::track_status_t status = st::TRACK_STATUS_GENERAL_FAILURE;

        SIXTRL_ASSERT( job.ptrContext() != nullptr );
        SIXTRL_ASSERT( job.ptrParticlesArg() != nullptr );
        SIXTRL_ASSERT( job.ptrBeamElementsArg() != nullptr );
        SIXTRL_ASSERT( job.ptrContext()->hasSelectedNode() );
        SIXTRL_ASSERT( job.context().has_track_elem_by_elem_kernel() );
        SIXTRL_ASSERT( job.particleSetIndicesBegin() != nullptr );
        SIXTRL_ASSERT( job.numParticlesInSetsBegin() != nullptr );

        st_size_t const num_psets = job.numParticleSets();

        if( num_psets == st_size_t{ 1 } )
        {
            SIXTRL_ASSERT( job.context().selected_particle_set() ==
                *job.particleSetIndicesBegin() );

            SIXTRL_ASSERT( job.context().num_particles_in_selected_set() ==
                *job.numParticlesInSetsBegin() );

            status = job.context().track_line(
                line_begin_idx, line_end_idx, finish_turn );
        }
        else if( num_psets > st_size_t{ 1 } )
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
                job.context().track_line_kernel_id(), st_size_t{ 1 },
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
    using tjob_t = st::TrackJobCl;
    return new tjob_t( device_id_str, particles_buffer,
        tjob_t::DefaultNumParticleSetIndices(),
        tjob_t::DefaultParticleSetIndicesBegin(),
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
        st::st_size_t const
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

bool NS(TrackJobCl_has_particles_addr_buffer_arg)(
    const ::NS(TrackJobCl) *const SIXTRL_RESTRICT job )
{
    return ( ( job != nullptr ) &&
             ( job->ptrParticlesAddrBufferArg() != nullptr ) );
}

::NS(ClArgument)* NS(TrackJobCl_get_particles_addr_buffer_arg)(
    ::NS(TrackJobCl)* SIXTRL_RESTRICT job )
{
    return ( job != nullptr ) ? job->ptrParticlesAddrBufferArg() : nullptr;
}

::NS(ClArgument) const* NS(TrackJobCl_get_const_particles_addr_arg)(
    const ::NS(TrackJobCl) *const SIXTRL_RESTRICT job )
{
    return ( job != nullptr ) ? job->ptrParticlesAddrBufferArg() : nullptr;
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

std::uintptr_t NS(TrackJobCl_get_opencl_context_addr)(
    const NS(TrackJobCl) *const SIXTRL_RESTRICT track_job ) SIXTRL_NOEXCEPT
{
    return ( track_job != nullptr )
        ? track_job->opencl_context_addr() : std::uintptr_t{ 0 };
}

std::uintptr_t NS(TrackJobCl_get_opencl_queue_addr)(
    const NS(TrackJobCl) *const SIXTRL_RESTRICT track_job ) SIXTRL_NOEXCEPT
{
    return ( track_job != nullptr )
        ? track_job->opencl_queue_addr() : std::uintptr_t{ 0 };

/* ------------------------------------------------------------------------- */

::NS(ClArgument) const* NS(TrackJobCl_const_argument_by_buffer_id)(
    const ::NS(TrackJobCl) *const SIXTRL_RESTRICT job,
    ::NS(buffer_size_t) const buffer_id )
{
    return ( job != nullptr )
        ? job->ptr_const_argument_by_buffer_id( buffer_id ) : nullptr;
}

::NS(ClArgument)* NS(TrackJobCl_argument_by_buffer_id)(
    ::NS(TrackJobCl)* SIXTRL_RESTRICT job, ::NS(buffer_size_t) const buffer_id )
{
    return ( job != nullptr )
        ? job->ptr_argument_by_buffer_id( buffer_id ) : nullptr;
}

::NS(ClArgument) const* NS(TrackJobCl_const_stored_buffer_argument)(
    const ::NS(TrackJobCl) *const SIXTRL_RESTRICT job,
    ::NS(buffer_size_t) const buffer_id )
{
    return ( job != nullptr )
        ? job->ptr_const_stored_buffer_argument( buffer_id ) : nullptr;
}

::NS(ClArgument)* NS(TrackJobCl_stored_buffer_argument)(
    ::NS(TrackJobCl)* SIXTRL_RESTRICT job, ::NS(buffer_size_t) const buffer_id )
{
    return ( job != nullptr )
        ? job->ptr_stored_buffer_argument( buffer_id ) : nullptr;
}
