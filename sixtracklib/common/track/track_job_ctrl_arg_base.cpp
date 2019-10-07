#include "sixtracklib/common/track/track_job_ctrl_arg_base.hpp"

#if defined( __cplusplus ) && !defined( _GPUCODE ) && !defined( __CUDA_ARCH__ )
    #if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
        #include <algorithm>
        #include <cstddef>
        #include <cstdint>
        #include <cstdlib>
        #include <memory>
        #include <string>
        #include <vector>
    #endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

    #if !defined( SIXTRL_NO_INCLUDES )
        #include "sixtracklib/common/definitions.h"
        #include "sixtracklib/common/control/definitions.h"
        #include "sixtracklib/common/buffer.hpp"
        #include "sixtracklib/common/buffer.h"
        #include "sixtracklib/common/particles.h"
        #include "sixtracklib/common/output/output_buffer.h"
        #include "sixtracklib/common/output/elem_by_elem_config.h"
        #include "sixtracklib/common/control/controller_base.hpp"
        #include "sixtracklib/common/control/node_controller_base.hpp"
        #include "sixtracklib/common/control/argument_base.hpp"
        #include "sixtracklib/common/track/definitions.h"
        #include "sixtracklib/common/track/track_job_base.hpp"
    #endif /* defined( __cplusplus ) */

#endif /* C++, Host */

#if defined( __cplusplus ) && !defined( _GPUCODE )

namespace st = SIXTRL_CXX_NAMESPACE;

namespace SIXTRL_CXX_NAMESPACE
{
    namespace
    {
        SIXTRL_STATIC st::TrackJobCtrlArgBase::status_t
        TrackJobCtrlArgBase_do_set_kernel_id(
            st::TrackJobCtrlArgBase  const& SIXTRL_RESTRICT,
            st::TrackJobCtrlArgBase::kernel_id_t& SIXTRL_RESTRICT_REF,
            st::TrackJobCtrlArgBase::kernel_id_t const ) SIXTRL_NOEXCEPT;

        SIXTRL_STATIC bool TrackJobCtrlArgBase_has_kernel_id(
            st::TrackJobCtrlArgBase const& SIXTRL_RESTRICT,
            st::TrackJobCtrlArgBase::kernel_id_t const ) SIXTRL_NOEXCEPT;
    }

    bool TrackJobCtrlArgBase::hasAssignOutputToBeamMonitorsKernel()
        const SIXTRL_NOEXCEPT
    {
        return st::TrackJobCtrlArgBase_has_kernel_id( *this,
            this->assignOutputToBeamMonitorsKernelId() );
    }

    TrackJobCtrlArgBase::kernel_id_t
    TrackJobCtrlArgBase::assignOutputToBeamMonitorsKernelId(
        ) const SIXTRL_NOEXCEPT
    {
        return this->m_assign_output_bemon_kernel_id;
    }

    TrackJobCtrlArgBase::status_t
    TrackJobCtrlArgBase::setAssignOutputToBeamMonitorsKernelId(
        TrackJobCtrlArgBase::kernel_id_t const id )
    {
        return this->doSetAssignOutputToBeamMonitorsKernelId( id );
    }

    /* --------------------------------------------------------------------- */

    bool TrackJobCtrlArgBase::hasAssignOutputToElemByElemConfigKernel(
        ) const SIXTRL_NOEXCEPT
    {
        return st::TrackJobCtrlArgBase_has_kernel_id( *this,
            this->assignOutputToElemByElemConfigKernelId() );
    }

    TrackJobCtrlArgBase::kernel_id_t
    TrackJobCtrlArgBase::assignOutputToElemByElemConfigKernelId()
        const SIXTRL_NOEXCEPT
    {
        return this->m_assign_output_elem_by_elem_kernel_id;
    }

    TrackJobCtrlArgBase::status_t
    TrackJobCtrlArgBase::setAssignOutputToElemByElemConfigKernelId(
        TrackJobCtrlArgBase::kernel_id_t const id )
    {
        return this->doSetAssignOutputToElemByElemConfigKernelId( id );
    }

    /* --------------------------------------------------------------------- */

    bool TrackJobCtrlArgBase::hasTrackUntilKernel() const SIXTRL_NOEXCEPT
    {
        return st::TrackJobCtrlArgBase_has_kernel_id(
            *this, this->trackUntilKernelId() );
    }

    TrackJobCtrlArgBase::kernel_id_t
    TrackJobCtrlArgBase::trackUntilKernelId() const SIXTRL_NOEXCEPT
    {
        return this->m_track_until_kernel_id;
    }

    TrackJobCtrlArgBase::status_t
    TrackJobCtrlArgBase::setTrackUntilKernelId(
        TrackJobCtrlArgBase::kernel_id_t const id )
    {
        return this->doSetTrackUntilKernelId( id );
    }

    /* --------------------------------------------------------------------- */

    bool TrackJobCtrlArgBase::hasTrackLineKernel() const SIXTRL_NOEXCEPT
    {
        return st::TrackJobCtrlArgBase_has_kernel_id(
            *this, this->trackLineKernelId() );
    }

    TrackJobCtrlArgBase::kernel_id_t
    TrackJobCtrlArgBase::trackLineKernelId() const SIXTRL_NOEXCEPT
    {
        return this->m_track_line_kernel_id;
    }

    TrackJobCtrlArgBase::status_t TrackJobCtrlArgBase::setTrackLineKernelId(
        TrackJobCtrlArgBase::kernel_id_t const id )
    {
        return this->doSetTrackLineKernelId( id );
    }

    /* --------------------------------------------------------------------- */

    bool TrackJobCtrlArgBase::hasTrackElemByElemKernel() const SIXTRL_NOEXCEPT
    {
        return st::TrackJobCtrlArgBase_has_kernel_id(
            *this, this->trackElemByElemKernelId() );
    }

    TrackJobCtrlArgBase::kernel_id_t
    TrackJobCtrlArgBase::trackElemByElemKernelId() const SIXTRL_NOEXCEPT
    {
        return this->m_track_elem_by_elem_kernel_id;
    }

    TrackJobCtrlArgBase::status_t
    TrackJobCtrlArgBase::setTrackElemByElemKernelId(
        TrackJobCtrlArgBase::kernel_id_t const id )
    {
        return this->doSetTrackElemByElemKernelId( id );
    }

    /* --------------------------------------------------------------------- */

    bool TrackJobCtrlArgBase::hasFetchParticlesAddressesKernel()
        const SIXTRL_NOEXCEPT
    {
        return st::TrackJobCtrlArgBase_has_kernel_id( *this,
            this->fetchParticlesAddressesKernelId() );
    }

    TrackJobCtrlArgBase::kernel_id_t
    TrackJobCtrlArgBase::fetchParticlesAddressesKernelId() const SIXTRL_NOEXCEPT
    {
        return this->m_fetch_particles_addr_kernel_id;
    }

    TrackJobCtrlArgBase::status_t
    TrackJobCtrlArgBase::setFetchParticlesAddressesKernelId(
        TrackJobCtrlArgBase::kernel_id_t const id )
    {
        return this->doSetFetchParticlesAddressesKernelId( id );
    }

    /* --------------------------------------------------------------------- */

    TrackJobCtrlArgBase::TrackJobCtrlArgBase(
        TrackJobCtrlArgBase::arch_id_t const arch_id,
        char const* SIXTRL_RESTRICT arch_str,
        char const* SIXTRL_RESTRICT config_str ) :
        st::TrackJobBaseNew( arch_id, arch_str, config_str ),
        m_stored_controller( nullptr ),
        m_stored_particles_arg( nullptr ),
        m_stored_beam_elements_arg( nullptr ),
        m_stored_output_arg( nullptr ),
        m_stored_elem_by_elem_conf_arg( nullptr ),
        m_stored_particles_addr_arg( nullptr ),
        m_stored_debug_flag_arg( nullptr ),
        m_assign_output_bemon_kernel_id(
            st::ControllerBase::ILLEGAL_KERNEL_ID ),
        m_assign_output_elem_by_elem_kernel_id(
            st::ControllerBase::ILLEGAL_KERNEL_ID ),
        m_track_until_kernel_id( st::ControllerBase::ILLEGAL_KERNEL_ID ),
        m_track_line_kernel_id( st::ControllerBase::ILLEGAL_KERNEL_ID ),
        m_track_elem_by_elem_kernel_id(
            st::ControllerBase::ILLEGAL_KERNEL_ID ),
        m_fetch_particles_addr_kernel_id(
            st::ControllerBase::ILLEGAL_KERNEL_ID )
    {
        this->doSetDefaultPrepareResetClearFlags(
            st::TRACK_JOB_CLEAR_PARTICLE_STRUCTURES |
            st::TRACK_JOB_CLEAR_BEAM_ELEMENT_STRUCTURES |
            st::TRACK_JOB_CLEAR_OUTPUT_STRUCTURES );

        this->doSetDefaultAllClearFlags(
            st::TRACK_JOB_CLEAR_PARTICLE_STRUCTURES |
            st::TRACK_JOB_CLEAR_BEAM_ELEMENT_STRUCTURES |
            st::TRACK_JOB_CLEAR_OUTPUT_STRUCTURES |
            st::TRACK_JOB_CLEAR_CONTROLLER |
            st::TRACK_JOB_CLEAR_DEFAULT_KERNELS );
    }

    TrackJobCtrlArgBase::TrackJobCtrlArgBase(
        TrackJobCtrlArgBase const& other ) :
        st::TrackJobBaseNew( other ),
        m_stored_controller( nullptr ),
        m_stored_particles_arg( nullptr ),
        m_stored_beam_elements_arg( nullptr ),
        m_stored_output_arg( nullptr ),
        m_stored_elem_by_elem_conf_arg( nullptr ),
        m_stored_particles_addr_arg( nullptr ),
        m_stored_debug_flag_arg( nullptr ),
        m_assign_output_bemon_kernel_id(
            other.m_assign_output_bemon_kernel_id ),
        m_assign_output_elem_by_elem_kernel_id(
            other.m_assign_output_elem_by_elem_kernel_id ),
        m_track_until_kernel_id( other.m_track_until_kernel_id ),
        m_track_line_kernel_id( other.m_track_line_kernel_id ),
        m_track_elem_by_elem_kernel_id( other.m_track_elem_by_elem_kernel_id ),
        m_fetch_particles_addr_kernel_id(
            other.m_fetch_particles_addr_kernel_id )
    {
//         using base_ctrl_t = TrackJobCtrlArgBase::controller_base_t;
//         using base_arg_t  = TrackJobCtrlArgBase::argument_base_t;

//         using stored_ctrl_base_t = TrackJobCtrlArgBase::stored_ctrl_base_t;
//         using stored_arg_base_t  = TrackJobCtrlArgBase::stored_arg_base_t;

        /* TODO: Implement copying of arguments and controller !!!! */
    }

    TrackJobCtrlArgBase::TrackJobCtrlArgBase(
        TrackJobCtrlArgBase&& other ) SIXTRL_NOEXCEPT :
        st::TrackJobBaseNew( std::move( other ) ),
        m_stored_controller( std::move( other.m_stored_controller ) ),
        m_stored_particles_arg( std::move( other.m_stored_particles_arg ) ),
        m_stored_beam_elements_arg(
            std::move( other.m_stored_beam_elements_arg ) ),
        m_stored_output_arg( std::move( other.m_stored_output_arg ) ),
        m_stored_elem_by_elem_conf_arg(
            std::move( other.m_stored_elem_by_elem_conf_arg ) ),
        m_stored_particles_addr_arg(
            std::move( other.m_stored_particles_addr_arg ) ),
        m_stored_debug_flag_arg(
            std::move( other.m_stored_debug_flag_arg ) ),
        m_assign_output_bemon_kernel_id( std::move(
            other.m_assign_output_bemon_kernel_id ) ),
        m_assign_output_elem_by_elem_kernel_id( std::move(
            other.m_assign_output_elem_by_elem_kernel_id ) ),
        m_track_until_kernel_id( std::move( other.m_track_until_kernel_id ) ),
        m_track_line_kernel_id( std::move( other.m_track_line_kernel_id ) ),
        m_track_elem_by_elem_kernel_id( std::move(
            other.m_track_elem_by_elem_kernel_id ) ),
        m_fetch_particles_addr_kernel_id(
            std::move( other.m_fetch_particles_addr_kernel_id ) )
    {
        other.doClearAllCtrlArgBaseImpl();
    }

    TrackJobCtrlArgBase& TrackJobCtrlArgBase::operator=(
        TrackJobCtrlArgBase const& rhs )
    {
        if( ( this != &rhs ) && ( this->isArchCompatibleWith( rhs ) ) )
        {
            this->m_stored_controller.reset( nullptr );

            this->m_stored_particles_arg.reset( nullptr );
            this->m_stored_beam_elements_arg.reset( nullptr );
            this->m_stored_output_arg.reset( nullptr );
            this->m_stored_elem_by_elem_conf_arg.reset( nullptr );
            this->m_stored_particles_addr_arg.reset( nullptr );
            this->m_stored_debug_flag_arg.reset( nullptr );

            this->m_assign_output_bemon_kernel_id =
                rhs.m_assign_output_bemon_kernel_id;

            this->m_assign_output_elem_by_elem_kernel_id =
                rhs.m_assign_output_elem_by_elem_kernel_id;

            this->m_track_until_kernel_id = rhs.m_track_until_kernel_id;
            this->m_track_line_kernel_id  = rhs.m_track_line_kernel_id;

            this->m_track_elem_by_elem_kernel_id =
                rhs.m_track_elem_by_elem_kernel_id;

            this->m_fetch_particles_addr_kernel_id =
                rhs.m_fetch_particles_addr_kernel_id;

            /* TODO: Implement copying of arguments and controller */
        }

        return *this;
    }

    TrackJobCtrlArgBase& TrackJobCtrlArgBase::operator=(
        TrackJobCtrlArgBase&& rhs ) SIXTRL_NOEXCEPT
    {
        if( ( this != &rhs ) && ( this->isArchCompatibleWith( rhs ) ) )
        {
            this->m_stored_controller = std::move( rhs.m_stored_controller );

            this->m_stored_particles_arg =
                std::move( rhs.m_stored_particles_arg );

            this->m_stored_beam_elements_arg =
                std::move( rhs.m_stored_beam_elements_arg );

            this->m_stored_output_arg = std::move( rhs.m_stored_output_arg );

            this->m_stored_elem_by_elem_conf_arg =
                std::move( rhs.m_stored_elem_by_elem_conf_arg );

            this->m_stored_particles_addr_arg =
                std::move( rhs.m_stored_particles_addr_arg );

            this->m_stored_debug_flag_arg =
                std::move( rhs.m_stored_debug_flag_arg );

            this->m_assign_output_bemon_kernel_id =
                std::move( rhs.m_assign_output_bemon_kernel_id );

            this->m_assign_output_elem_by_elem_kernel_id =
                std::move( rhs.m_assign_output_elem_by_elem_kernel_id );

            this->m_track_until_kernel_id =
                std::move( rhs.m_track_until_kernel_id );

            this->m_track_line_kernel_id =
                std::move( rhs.m_track_line_kernel_id );

            this->m_track_elem_by_elem_kernel_id =
                std::move( rhs.m_track_elem_by_elem_kernel_id );

            this->m_fetch_particles_addr_kernel_id =
                std::move( rhs.m_fetch_particles_addr_kernel_id );

            rhs.doClearAllCtrlArgBaseImpl();
        }

        return *this;
    }

    /* --------------------------------------------------------------------- */

    TrackJobCtrlArgBase::status_t
    TrackJobCtrlArgBase::doClearParticleAddresses(
            TrackJobCtrlArgBase::size_type const index )
    {
        TrackJobCtrlArgBase::status_t status = st::ARCH_STATUS_GENERAL_FAILURE;

        if( ( this->ptrParticlesAddrArgBase() != nullptr ) &&
            ( this->ptrParticlesAddrArgBase()->usesCObjectsCxxBuffer() ) &&
            ( this->doGetPtrParticlesAddrBuffer() != nullptr ) &&
            ( this->doGetPtrParticlesAddrBuffer() ==
              this->ptrParticlesAddrArgBase()->ptrCObjectsCxxBuffer() ) )
        {
            using _this_t = TrackJobCtrlArgBase;
            using _base_t = _this_t::_base_track_job_t;

            /* NOTE: If the architecture we are working on requires collecting,
             *       we collect first *all* particle addresses before
             *       attempting to clear the single particle address that is
             *       pertinent to this call. Then, we (always) send the
             *       result via the argument.
             *
             *       This is obviously not optimal but should work in most
             *       circumstances. If not suitable for a deriving implementation,
             *       onus is on the corresponding implementation to specialize
             *       this function accordingly */

            status = st::ARCH_STATUS_SUCCESS;

            if( this->requiresCollecting() )
            {
                status = this->collectParticlesAddresses();
            }

            if( status == st::ARCH_STATUS_SUCCESS )
            {
                status = _base_t::doClearParticleAddresses( index );
            }

            if( status == st::ARCH_STATUS_SUCCESS )
            {
                status = this->ptrParticlesAddrArgBase()->send(
                    *this->doGetPtrParticlesAddrBuffer() );
            }
        }

        return status;
    }

    TrackJobCtrlArgBase::status_t
    TrackJobCtrlArgBase::doClearAllParticleAddresses()
    {
        TrackJobCtrlArgBase::status_t status = st::ARCH_STATUS_GENERAL_FAILURE;

        if( ( this->ptrParticlesAddrArgBase() != nullptr ) &&
            ( this->ptrParticlesAddrArgBase()->usesCObjectsCxxBuffer() ) &&
            ( this->doGetPtrParticlesAddrBuffer() != nullptr ) &&
            ( this->doGetPtrParticlesAddrBuffer() ==
              this->ptrParticlesAddrArgBase()->ptrCObjectsCxxBuffer() ) )
        {
            using _this_t = TrackJobCtrlArgBase;
            using _base_t = _this_t::_base_track_job_t;

            status = _base_t::doClearAllParticleAddresses();

            if( status == st::ARCH_STATUS_SUCCESS )
            {
                status = this->ptrParticlesAddrArgBase()->send(
                    *this->doGetPtrParticlesAddrBuffer() );
            }
        }

        return status;
    }

    /* --------------------------------------------------------------------- */

    void TrackJobCtrlArgBase::doClear(
        TrackJobCtrlArgBase::clear_flag_t const clear_flags )
    {
        using _this_t = TrackJobCtrlArgBase;

        if( _this_t::IsClearFlagSet(
                clear_flags, st::TRACK_JOB_CLEAR_DEFAULT_KERNELS ) )
        {
            this->doClearDefaultKernels();
        }

        if( _this_t::IsClearFlagSet(
                clear_flags, st::TRACK_JOB_CLEAR_CONTROLLER ) )
        {
            this->doClearController();
        }

        st::TrackJobBaseNew::doClear( clear_flags );

        return;
    }

    void TrackJobCtrlArgBase::doClearParticlesStructures()
    {
        this->doClearParticlesStructuresCtrlArgBaseImpl();
        st::TrackJobBaseNew::doClearParticlesStructures();
    }

    void TrackJobCtrlArgBase::doClearBeamElementsStructures()
    {
        this->doClearBeamElementsStructuresCtrlArgBaseImpl();
        st::TrackJobBaseNew::doClearBeamElementsStructures();
    }

    void TrackJobCtrlArgBase::doClearOutputStructures()
    {
        this->doClearOutputStructuresCtrlArgBaseImpl();
        st::TrackJobBaseNew::doClearOutputStructures();
    }

    void TrackJobCtrlArgBase::doClearController()
    {
        this->doClearControllerCtrlArgBaseImpl();
    }

    void TrackJobCtrlArgBase::doClearDefaultKernels()
    {
        this->doClearDefaultKernelsCtrlArgBaseImpl();
    }

    /* --------------------------------------------------------------------- */

    TrackJobCtrlArgBase::collect_flag_t TrackJobCtrlArgBase::doCollect(
        TrackJobCtrlArgBase::collect_flag_t const flags )
    {
        using _this_t = st::TrackJobCtrlArgBase;
        using _base_t = st::TrackJobBaseNew;
        using collect_flag_t = _this_t::collect_flag_t;
        using status_t = _this_t::status_t;

        collect_flag_t result = st::TRACK_JOB_IO_NONE;

        if( ( _base_t::IsCollectFlagSet(
                flags, st::TRACK_JOB_IO_PARTICLES ) ) &&
            ( this->ptrParticlesArgBase() != nullptr ) &&
            ( this->ptrParticlesArgBase()->usesCObjectsBuffer() ) &&
            ( this->ptrCParticlesBuffer() != nullptr ) )
        {
            status_t const status = this->ptrParticlesArgBase()->receive(
                this->ptrCParticlesBuffer() );

            if( status == st::ARCH_STATUS_SUCCESS )
            {
                result |= st::TRACK_JOB_IO_PARTICLES;
            }
        }

        if( ( _base_t::IsCollectFlagSet(
                flags, st::TRACK_JOB_IO_BEAM_ELEMENTS ) ) &&
            ( this->ptrBeamElementsArgBase() != nullptr ) &&
            ( this->ptrBeamElementsArgBase()->usesCObjectsBuffer() ) &&
            ( this->ptrCBeamElementsBuffer() != nullptr ) )
        {
            status_t const status = this->ptrBeamElementsArgBase()->receive(
                this->ptrCBeamElementsBuffer() );

            if( status == st::ARCH_STATUS_SUCCESS )
            {
                result |= st::TRACK_JOB_IO_BEAM_ELEMENTS;
            }
        }

        if( ( _base_t::IsCollectFlagSet(
                flags, st::TRACK_JOB_IO_OUTPUT ) ) &&
            ( this->ptrOutputArgBase() != nullptr ) &&
            ( this->ptrOutputArgBase()->usesCObjectsBuffer() ) &&
            ( this->ptrCOutputBuffer() != nullptr ) )
        {
            status_t const status = this->ptrOutputArgBase()->receive(
                this->ptrCOutputBuffer() );

            if( status == st::ARCH_STATUS_SUCCESS )
            {
                result |= st::TRACK_JOB_IO_OUTPUT;
            }
        }

        if( ( _base_t::IsCollectFlagSet(
                flags, st::TRACK_JOB_IO_DEBUG_REGISTER ) ) &&
            ( this->ptrDebugRegisterArgBase() != nullptr ) &&
            ( this->ptrDebugRegisterArgBase()->usesRawArgument() ) &&
            ( this->doGetPtrLocalDebugRegister() != nullptr ) )
        {
            using debug_register_t = _base_t::debug_register_t;

            status_t const status = this->ptrDebugRegisterArgBase()->receive(
                this->doGetPtrLocalDebugRegister(),
                    sizeof( debug_register_t ) );

            if( status == st::ARCH_STATUS_SUCCESS )
            {
                result |= st::TRACK_JOB_IO_DEBUG_REGISTER;
            }
        }

        if( ( _base_t::IsCollectFlagSet(
                flags, st::TRACK_JOB_IO_PARTICLES_ADDR ) ) &&
            ( this->ptrParticlesAddrArgBase() != nullptr ) &&
            ( this->ptrParticlesAddrArgBase()->usesCObjectsCxxBuffer() ) &&
            ( this->doGetPtrParticlesAddrBuffer() != nullptr ) &&
            ( this->doGetPtrParticlesAddrBuffer() ==
              this->ptrParticlesAddrArgBase()->ptrCObjectsCxxBuffer() ) )
        {
            _base_t::buffer_t& particles_addr_buffer =
                *this->doGetPtrParticlesAddrBuffer();

            status_t const status = this->ptrParticlesAddrArgBase()->receive(
                particles_addr_buffer );

            if( status == st::ARCH_STATUS_SUCCESS )
            {
                result |= st::TRACK_JOB_IO_PARTICLES_ADDR;
            }
        }

        return result;
    }

    TrackJobCtrlArgBase::push_flag_t TrackJobCtrlArgBase::doPush(
        TrackJobCtrlArgBase::push_flag_t const flags )
    {
        using _this_t = st::TrackJobCtrlArgBase;
        using _base_t = st::TrackJobBaseNew;
        using push_flag_t = _this_t::push_flag_t;
        using status_t = _this_t::status_t;

        push_flag_t result = st::TRACK_JOB_IO_NONE;

        if( ( _base_t::IsCollectFlagSet(
                flags, st::TRACK_JOB_IO_PARTICLES ) ) &&
            ( this->ptrParticlesArgBase() != nullptr ) &&
            ( this->ptrParticlesArgBase()->usesCObjectsBuffer() ) &&
            ( this->ptrCParticlesBuffer() != nullptr ) )
        {
            status_t const status = this->ptrParticlesArgBase()->send(
                this->ptrCParticlesBuffer() );

            if( status == st::ARCH_STATUS_SUCCESS )
            {
                result |= st::TRACK_JOB_IO_PARTICLES;
            }
        }

        if( ( _base_t::IsCollectFlagSet(
                flags, st::TRACK_JOB_IO_BEAM_ELEMENTS ) ) &&
            ( this->ptrBeamElementsArgBase() != nullptr ) &&
            ( this->ptrBeamElementsArgBase()->usesCObjectsBuffer() ) &&
            ( this->ptrCBeamElementsBuffer() != nullptr ) )
        {
            status_t const status = this->ptrBeamElementsArgBase()->send(
                this->ptrCBeamElementsBuffer() );

            if( status == st::ARCH_STATUS_SUCCESS )
            {
                result |= st::TRACK_JOB_IO_BEAM_ELEMENTS;
            }
        }

        if( ( _base_t::IsCollectFlagSet(
                flags, st::TRACK_JOB_IO_OUTPUT ) ) &&
            ( this->ptrOutputArgBase() != nullptr ) &&
            ( this->ptrOutputArgBase()->usesCObjectsBuffer() ) &&
            ( this->ptrCOutputBuffer() != nullptr ) )
        {
            status_t const status = this->ptrOutputArgBase()->send(
                this->ptrCOutputBuffer() );

            if( status == st::ARCH_STATUS_SUCCESS )
            {
                result |= st::TRACK_JOB_IO_OUTPUT;
            }
        }

        if( ( _base_t::IsCollectFlagSet(
                flags, st::TRACK_JOB_IO_DEBUG_REGISTER ) ) &&
            ( this->ptrDebugRegisterArgBase() != nullptr ) &&
            ( this->ptrDebugRegisterArgBase()->usesRawArgument() ) &&
            ( this->doGetPtrLocalDebugRegister() != nullptr ) )
        {
            using debug_register_t = _base_t::debug_register_t;

            status_t const status = this->ptrDebugRegisterArgBase()->send(
                this->doGetPtrLocalDebugRegister(),
                    sizeof( debug_register_t ) );

            if( status == st::ARCH_STATUS_SUCCESS )
            {
                result |= st::TRACK_JOB_IO_DEBUG_REGISTER;
            }
        }

        return result;
    }

    /* --------------------------------------------------------------------- */

    TrackJobCtrlArgBase::status_t
    TrackJobCtrlArgBase::doSwitchDebugMode( bool const is_in_debug_mode )
    {
        using _base_t = st::TrackJobBaseNew;

        TrackJobCtrlArgBase::status_t status = st::ARCH_STATUS_SUCCESS;

        if( this->ptrControllerBase() != nullptr )
        {
            if( is_in_debug_mode )
            {
                status = this->ptrControllerBase()->enableDebugMode();
            }
            else
            {
                status = this->ptrControllerBase()->disableDebugMode();
            }
        }

        if( status == st::ARCH_STATUS_SUCCESS )
        {
            status = _base_t::doSwitchDebugMode( is_in_debug_mode );
        }

        return status;
    }

    TrackJobCtrlArgBase::status_t TrackJobCtrlArgBase::doSetDebugRegister(
        TrackJobCtrlArgBase::debug_register_t const dbg_reg )
    {
        using _base_t      = st::TrackJobBaseNew;
        using _this_t      = TrackJobCtrlArgBase;
        using size_t       = _this_t::size_type;
        using status_t     = _this_t::status_t;
        using ptr_arg_t    = _this_t::argument_base_t*;
        using ptr_ctrl_t   = _this_t::controller_base_t*;

        status_t status = st::ARCH_STATUS_GENERAL_FAILURE;
        size_t const dbg_reg_size = sizeof( _this_t::debug_register_t );

        ptr_arg_t ptr_debug_register_arg = this->ptrDebugRegisterArgBase();
        ptr_ctrl_t ptr_ctrl_base = this->ptrControllerBase();

        if( ( ptr_debug_register_arg != nullptr ) &&
            ( ptr_debug_register_arg->ptrControllerBase() == ptr_ctrl_base ) &&
            ( ptr_debug_register_arg->usesRawArgument() ) &&
            ( this->doGetPtrLocalDebugRegister() != nullptr ) &&
            ( ptr_debug_register_arg->ptrRawArgument() ==
              this->doGetPtrLocalDebugRegister() ) &&
            ( ptr_debug_register_arg->size() == dbg_reg_size ) )
        {
            status = ptr_debug_register_arg->send( &dbg_reg, dbg_reg_size );

            if( status == st::ARCH_STATUS_SUCCESS )
            {
                status = _base_t::doSetDebugRegister( dbg_reg );
            }
        }

        return status;
    }

    TrackJobCtrlArgBase::status_t TrackJobCtrlArgBase::doFetchDebugRegister(
        TrackJobCtrlArgBase::debug_register_t* ptr_dbg_reg )
    {
        using _base_t      = st::TrackJobBaseNew;
        using _this_t      = TrackJobCtrlArgBase;
        using size_t       = _this_t::size_type;
        using status_t     = _this_t::status_t;
        using ptr_arg_t    = _this_t::argument_base_t*;
        using ptr_ctrl_t   = _this_t::controller_base_t*;

        status_t status = st::ARCH_STATUS_GENERAL_FAILURE;
        size_t const dbg_reg_size = sizeof( _this_t::debug_register_t );

        ptr_arg_t ptr_debug_register_arg = this->ptrDebugRegisterArgBase();
        ptr_ctrl_t ptr_ctrl_base = this->ptrControllerBase();

        if( ( ptr_debug_register_arg != nullptr ) &&
            ( ptr_debug_register_arg->ptrControllerBase() == ptr_ctrl_base ) &&
            ( ptr_debug_register_arg->usesRawArgument() ) &&
            ( this->doGetPtrLocalDebugRegister() != nullptr ) &&
            ( ptr_debug_register_arg->ptrRawArgument() ==
              this->doGetPtrLocalDebugRegister() ) &&
            ( ptr_debug_register_arg->size() == dbg_reg_size ) )
        {
            status = ptr_debug_register_arg->receive(
                ptr_dbg_reg, dbg_reg_size );

            if( status == st::ARCH_STATUS_SUCCESS )
            {
                status = _base_t::doFetchDebugRegister( ptr_dbg_reg );
            }
        }

        return status;
    }

    /* ----------------------------------------------------------------- */

    TrackJobCtrlArgBase::status_t
    TrackJobCtrlArgBase::doSetAssignOutputToBeamMonitorsKernelId(
        TrackJobCtrlArgBase::kernel_id_t const kernel_id )
    {
        return this->doSetAssignOutputToBeamMonitorsKernelIdCtrlArgBaseImpl(
            kernel_id );
    }

    TrackJobCtrlArgBase::status_t
    TrackJobCtrlArgBase::doSetAssignOutputToElemByElemConfigKernelId(
        TrackJobCtrlArgBase::kernel_id_t const kernel_id )
    {
        return this->doSetAssignOutputToElemByElemConfigKernelIdCtrlArgBaseImpl(
            kernel_id );
    }

    TrackJobCtrlArgBase::status_t TrackJobCtrlArgBase::doSetTrackUntilKernelId(
        TrackJobCtrlArgBase::kernel_id_t const kernel_id )
    {
        return this->doSetTrackUntilKernelIdCtrlArgBaseImpl( kernel_id );
    }

    TrackJobCtrlArgBase::status_t TrackJobCtrlArgBase::doSetTrackLineKernelId(
        TrackJobCtrlArgBase::kernel_id_t const kernel_id )
    {
        return this->doSetTrackLineKernelIdCtrlArgBaseImpl( kernel_id );
    }

    TrackJobCtrlArgBase::status_t
    TrackJobCtrlArgBase::doSetTrackElemByElemKernelId(
        TrackJobCtrlArgBase::kernel_id_t const kernel_id )
    {
        return this->doSetTrackElemByElemKernelIdCtrlArgBaseImpl( kernel_id );
    }

    TrackJobCtrlArgBase::status_t
    TrackJobCtrlArgBase::doSetFetchParticlesAddressesKernelId(
        TrackJobCtrlArgBase::kernel_id_t const kernel_id )
    {
        return this->doSetFetchParticlesAddressesKernelIdCtrlArgBaseImpl(
            kernel_id );
    }

    /* --------------------------------------------------------------------- */

    TrackJobCtrlArgBase::status_t TrackJobCtrlArgBase::doPrepareController(
        char const* SIXTRL_RESTRICT )
    {
        return st::ARCH_STATUS_SUCCESS;
    }

    TrackJobCtrlArgBase::status_t TrackJobCtrlArgBase::doPrepareDefaultKernels(
        char const* SIXTRL_RESTRICT )
    {
        return st::ARCH_STATUS_SUCCESS;
    }

    /* --------------------------------------------------------------------- */

    TrackJobCtrlArgBase::controller_base_t const*
    TrackJobCtrlArgBase::ptrControllerBase() const SIXTRL_NOEXCEPT
    {
        return this->m_stored_controller.get();
    }

    TrackJobCtrlArgBase::controller_base_t*
    TrackJobCtrlArgBase::ptrControllerBase() SIXTRL_NOEXCEPT
    {
        return this->m_stored_controller.get();
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    TrackJobCtrlArgBase::argument_base_t const*
    TrackJobCtrlArgBase::ptrParticlesArgBase() const SIXTRL_NOEXCEPT
    {
        return this->m_stored_particles_arg.get();
    }

    TrackJobCtrlArgBase::argument_base_t*
    TrackJobCtrlArgBase::ptrParticlesArgBase() SIXTRL_NOEXCEPT
    {
        return this->m_stored_particles_arg.get();
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    TrackJobCtrlArgBase::argument_base_t const*
    TrackJobCtrlArgBase::ptrBeamElementsArgBase() const SIXTRL_NOEXCEPT
    {
        return this->m_stored_beam_elements_arg.get();
    }

    TrackJobCtrlArgBase::argument_base_t*
    TrackJobCtrlArgBase::ptrBeamElementsArgBase() SIXTRL_NOEXCEPT
    {
        return this->m_stored_beam_elements_arg.get();
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    TrackJobCtrlArgBase::argument_base_t const*
    TrackJobCtrlArgBase::ptrOutputArgBase() const SIXTRL_NOEXCEPT
    {
        return this->m_stored_output_arg.get();
    }

    TrackJobCtrlArgBase::argument_base_t*
    TrackJobCtrlArgBase::ptrOutputArgBase() SIXTRL_NOEXCEPT
    {
        return this->m_stored_output_arg.get();
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    TrackJobCtrlArgBase::argument_base_t const*
    TrackJobCtrlArgBase::ptrElemByElemConfigArgBase() const SIXTRL_NOEXCEPT
    {
        return this->m_stored_elem_by_elem_conf_arg.get();
    }

    TrackJobCtrlArgBase::argument_base_t*
    TrackJobCtrlArgBase::ptrElemByElemConfigArgBase() SIXTRL_NOEXCEPT
    {
        return this->m_stored_elem_by_elem_conf_arg.get();
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    TrackJobCtrlArgBase::argument_base_t const*
    TrackJobCtrlArgBase::ptrParticlesAddrArgBase() const SIXTRL_NOEXCEPT
    {
        return this->m_stored_particles_addr_arg.get();
    }

    TrackJobCtrlArgBase::argument_base_t*
    TrackJobCtrlArgBase::ptrParticlesAddrArgBase() SIXTRL_NOEXCEPT
    {
        return this->m_stored_particles_addr_arg.get();
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    TrackJobCtrlArgBase::argument_base_t const*
    TrackJobCtrlArgBase::ptrDebugRegisterArgBase() const SIXTRL_NOEXCEPT
    {
        return this->m_stored_debug_flag_arg.get();
    }

    TrackJobCtrlArgBase::argument_base_t*
    TrackJobCtrlArgBase::ptrDebugRegisterArgBase() SIXTRL_NOEXCEPT
    {
        return this->m_stored_debug_flag_arg.get();
    }

    /* --------------------------------------------------------------------- */

    void TrackJobCtrlArgBase::doUpdateStoredController(
        TrackJobCtrlArgBase::stored_ctrl_base_t&&
            ptr_controller_base ) SIXTRL_NOEXCEPT
    {
        this->m_stored_controller = std::move( ptr_controller_base );
    }

    void TrackJobCtrlArgBase::doUpdateStoredParticlesArg(
        TrackJobCtrlArgBase::stored_arg_base_t&&
            ptr_particles_arg ) SIXTRL_NOEXCEPT
    {
        this->m_stored_particles_arg = std::move( ptr_particles_arg );
    }

    void TrackJobCtrlArgBase::doUpdateStoredBeamElementsArg(
        TrackJobCtrlArgBase::stored_arg_base_t&&
            ptr_beam_elem_arg ) SIXTRL_NOEXCEPT
    {
        this->m_stored_beam_elements_arg = std::move( ptr_beam_elem_arg );
    }

    void TrackJobCtrlArgBase::doUpdateStoredOutputArg(
        TrackJobCtrlArgBase::stored_arg_base_t&&
            ptr_output_arg ) SIXTRL_NOEXCEPT
    {
        this->m_stored_output_arg = std::move( ptr_output_arg );
    }

    void TrackJobCtrlArgBase::doUpdateStoredElemByElemConfigArg(
        TrackJobCtrlArgBase::stored_arg_base_t&&
            ptr_elem_by_elem_conf_arg ) SIXTRL_NOEXCEPT
    {
        this->m_stored_elem_by_elem_conf_arg =
            std::move( ptr_elem_by_elem_conf_arg );
    }

    void TrackJobCtrlArgBase::doUpdateStoredParticlesAddrArg(
        TrackJobCtrlArgBase::stored_arg_base_t&&
            ptr_particles_addr_arg ) SIXTRL_NOEXCEPT
    {
        this->m_stored_particles_addr_arg =
            std::move( ptr_particles_addr_arg );
    }

    void TrackJobCtrlArgBase::doUpdateStoredDebugRegisterArg(
        TrackJobCtrlArgBase::stored_arg_base_t&&
            ptr_debug_flag_arg ) SIXTRL_NOEXCEPT
    {
        this->m_stored_debug_flag_arg = std::move( ptr_debug_flag_arg );
    }

    /* --------------------------------------------------------------------- */

    void TrackJobCtrlArgBase::doClearAllCtrlArgBaseImpl() SIXTRL_NOEXCEPT
    {
        this->doClearDefaultKernelsCtrlArgBaseImpl();
        this->doClearControllerCtrlArgBaseImpl();

        this->doClearParticlesStructuresCtrlArgBaseImpl();
        this->doClearBeamElementsStructuresCtrlArgBaseImpl();
        this->doClearOutputStructuresCtrlArgBaseImpl();
    }

    void TrackJobCtrlArgBase::doClearParticlesStructuresCtrlArgBaseImpl(
        ) SIXTRL_NOEXCEPT
    {
        this->m_stored_particles_arg.reset( nullptr );
        this->m_stored_particles_addr_arg.reset( nullptr );
    }

    void TrackJobCtrlArgBase::doClearBeamElementsStructuresCtrlArgBaseImpl(
        ) SIXTRL_NOEXCEPT
    {
        this->m_stored_beam_elements_arg.reset( nullptr );
    }

    void TrackJobCtrlArgBase::doClearOutputStructuresCtrlArgBaseImpl(
        ) SIXTRL_NOEXCEPT
    {
        this->m_stored_elem_by_elem_conf_arg.reset( nullptr );
        this->m_stored_output_arg.reset( nullptr );
    }

    void TrackJobCtrlArgBase::doClearDefaultKernelsCtrlArgBaseImpl(
        ) SIXTRL_NOEXCEPT
    {
        using base_ctrl_t = st::TrackJobCtrlArgBase::controller_base_t;

        this->m_assign_output_bemon_kernel_id = base_ctrl_t::ILLEGAL_KERNEL_ID;
        this->m_assign_output_elem_by_elem_kernel_id =
            base_ctrl_t::ILLEGAL_KERNEL_ID;

        this->m_track_until_kernel_id = base_ctrl_t::ILLEGAL_KERNEL_ID;
        this->m_track_line_kernel_id = base_ctrl_t::ILLEGAL_KERNEL_ID;
        this->m_track_elem_by_elem_kernel_id = base_ctrl_t::ILLEGAL_KERNEL_ID;
        this->m_fetch_particles_addr_kernel_id =
            base_ctrl_t::ILLEGAL_KERNEL_ID;
    }

    void TrackJobCtrlArgBase::doClearControllerCtrlArgBaseImpl(
        ) SIXTRL_NOEXCEPT
    {
        this->m_stored_controller.reset( nullptr );
    }

    /* --------------------------------------------------------------------- */

    TrackJobCtrlArgBase::status_t
    TrackJobCtrlArgBase::doSetAssignOutputToBeamMonitorsKernelIdCtrlArgBaseImpl(
        TrackJobCtrlArgBase::kernel_id_t const kernel_id )
    {
        return st::TrackJobCtrlArgBase_do_set_kernel_id(
            *this, this->m_assign_output_bemon_kernel_id, kernel_id );
    }

    TrackJobCtrlArgBase::status_t
    TrackJobCtrlArgBase::doSetAssignOutputToElemByElemConfigKernelIdCtrlArgBaseImpl(
        TrackJobCtrlArgBase::kernel_id_t const kernel_id )
    {
        return st::TrackJobCtrlArgBase_do_set_kernel_id( *this,
            this->m_assign_output_elem_by_elem_kernel_id, kernel_id );
    }

    TrackJobCtrlArgBase::status_t
    TrackJobCtrlArgBase::doSetTrackUntilKernelIdCtrlArgBaseImpl(
        TrackJobCtrlArgBase::kernel_id_t const kernel_id )
    {
        return st::TrackJobCtrlArgBase_do_set_kernel_id(
            *this, this->m_track_until_kernel_id, kernel_id );
    }

    TrackJobCtrlArgBase::status_t
    TrackJobCtrlArgBase::doSetTrackLineKernelIdCtrlArgBaseImpl(
        TrackJobCtrlArgBase::kernel_id_t const kernel_id )
    {
        return st::TrackJobCtrlArgBase_do_set_kernel_id(
            *this, this->m_track_line_kernel_id, kernel_id );
    }

    TrackJobCtrlArgBase::status_t
    TrackJobCtrlArgBase::doSetTrackElemByElemKernelIdCtrlArgBaseImpl(
        TrackJobCtrlArgBase::kernel_id_t const kernel_id )
    {
        return st::TrackJobCtrlArgBase_do_set_kernel_id(
            *this, this->m_track_elem_by_elem_kernel_id, kernel_id );
    }

    TrackJobCtrlArgBase::status_t
    TrackJobCtrlArgBase::doSetFetchParticlesAddressesKernelIdCtrlArgBaseImpl(
        TrackJobCtrlArgBase::kernel_id_t const kernel_id )
    {
        return st::TrackJobCtrlArgBase_do_set_kernel_id(
            *this, this->m_fetch_particles_addr_kernel_id, kernel_id );
    }

    namespace
    {
        SIXTRL_INLINE st::TrackJobCtrlArgBase::status_t
        TrackJobCtrlArgBase_do_set_kernel_id(
            st::TrackJobCtrlArgBase const& SIXTRL_RESTRICT track_job,
            st::TrackJobCtrlArgBase::kernel_id_t&
                SIXTRL_RESTRICT_REF track_job_kernel_id,
            st::TrackJobCtrlArgBase::kernel_id_t const kernel_id
        ) SIXTRL_NOEXCEPT
        {
            TrackJobCtrlArgBase::status_t status =
                st::ARCH_STATUS_GENERAL_FAILURE;

            if( ( track_job.ptrControllerBase() != nullptr ) &&
                ( track_job.ptrControllerBase()->hasKernel( kernel_id ) ) )
            {
                track_job_kernel_id = kernel_id;
                status = st::ARCH_STATUS_SUCCESS;
            }

            return status;
        }

        SIXTRL_INLINE bool TrackJobCtrlArgBase_has_kernel_id(
            st::TrackJobCtrlArgBase const& SIXTRL_RESTRICT track_job,
            st::TrackJobCtrlArgBase::kernel_id_t
                const kernel_id ) SIXTRL_NOEXCEPT
        {
            return ( ( kernel_id != st::ControllerBase::ILLEGAL_KERNEL_ID ) &&
                     ( track_job.ptrControllerBase() != nullptr ) &&
                     ( track_job.ptrControllerBase()->hasKernel(
                         kernel_id ) ) );
        }
    }
}

#endif /* C++, Host */

/* end: sixtracklib/common/track/track_job_ctrl_arg_base.cpp */
