#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/cuda/track_job.hpp"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus ) && !defined( _GPUCODE ) && \
   !defined( __CUDACC__  ) && !defined( __CUDA_ARCH__ )

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <cstddef>
    #include <cstdlib>
    #include <stdexcept>

    #include <cuda_runtime_api.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */
#endif /* C++, Host */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/control/definitions.h"
    #include "sixtracklib/common/control/arch_base.hpp"
    #include "sixtracklib/common/control/kernel_config_base.hpp"
    #include "sixtracklib/common/track/definitions.h"
    #include "sixtracklib/common/track/track_job_ctrl_arg_base.hpp"
    #include "sixtracklib/common/track/track_job_nodectrl_arg_base.hpp"
    #include "sixtracklib/common/buffer.hpp"
    #include "sixtracklib/common/buffer.h"

    #include "sixtracklib/cuda/definitions.h"
    #include "sixtracklib/cuda/controller.hpp"
    #include "sixtracklib/cuda/argument.hpp"
    #include "sixtracklib/cuda/control/default_kernel_config.h"
    #include "sixtracklib/cuda/wrappers/track_job_wrappers.h"

#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus ) && !defined( _GPUCODE ) && \
   !defined( __CUDACC__  ) && !defined( __CUDA_ARCH__ )

namespace st = SIXTRL_CXX_NAMESPACE;

namespace SIXTRL_CXX_NAMESPACE
{
    CudaTrackJob::CudaTrackJob(
        const char *const SIXTRL_RESTRICT config_str ) :
            st::TrackJobNodeCtrlArgBase( st::ARCHITECTURE_CUDA,
                SIXTRL_ARCHITECTURE_CUDA_STR, config_str )
    {
        bool success = this->doPrepareControllerCudaImpl( config_str );
        success &= this->doPrepareDefaultKernelsCudaImpl( config_str );

        SIXTRL_ASSERT( success );
        ( void )success;
    }

    CudaTrackJob::CudaTrackJob(
        std::string const& SIXTRL_RESTRICT_REF config_str ) :
            st::TrackJobNodeCtrlArgBase( st::ARCHITECTURE_CUDA,
                SIXTRL_ARCHITECTURE_CUDA_STR, config_str.c_str() )
    {
        bool success = this->doPrepareControllerCudaImpl( config_str.c_str() );
        success &= this->doPrepareDefaultKernelsCudaImpl( config_str.c_str() );

        SIXTRL_ASSERT( success );
        ( void )success;
    }

    CudaTrackJob::CudaTrackJob(
        const char *const SIXTRL_RESTRICT node_id_str,
        c_buffer_t* SIXTRL_RESTRICT particles_buffer,
        c_buffer_t* SIXTRL_RESTRICT beam_elements_buffer,
        c_buffer_t* SIXTRL_RESTRICT output_buffer,
        size_type const until_turn_elem_by_elem,
        const char *const SIXTRL_RESTRICT config_str ) :
            st::TrackJobNodeCtrlArgBase( st::ARCHITECTURE_CUDA,
                SIXTRL_ARCHITECTURE_CUDA_STR, config_str )
    {
        using _this_t = st::CudaTrackJob;
        bool const success = this->doInitCudaTrackJob( config_str,
            particles_buffer, _this_t::DefaultParticleSetIndicesBegin(),
                _this_t::DefaultParticleSetIndicesEnd(), beam_elements_buffer,
                    output_buffer, until_turn_elem_by_elem );

        SIXTRL_ASSERT( success );
        ( void )success;
    }

    CudaTrackJob::CudaTrackJob(
        std::string const& SIXTRL_RESTRICT_REF node_id_str,
        buffer_t& SIXTRL_RESTRICT_REF particles_buffer,
        buffer_t& SIXTRL_RESTRICT_REF beam_elements_buffer,
        buffer_t* SIXTRL_RESTRICT ptr_output_buffer,
        size_type const until_turn_elem_by_elem,
        std::string const& config_str ) :
            st::TrackJobNodeCtrlArgBase( st::ARCHITECTURE_CUDA,
                SIXTRL_ARCHITECTURE_CUDA_STR, config_str.c_str() )
    {
        using _this_t = CudaTrackJob;

        bool const success = this->doInitCudaTrackJob( config_str,
            particles_buffer, _this_t::DefaultParticleSetIndicesBegin(),
                _this_t::DefaultParticleSetIndicesEnd(), beam_elements_buffer,
                    ptr_output_buffer, until_turn_elem_by_elem );

        if( success )
        {
            this->doSetPtrParticleBuffer( &particles_buffer );
            this->doSetPtrBeamElementsBuffer( &beam_elements_buffer );

            if( ( ptr_output_buffer != nullptr ) &&
                ( this->hasOutputBuffer() ) && ( !this->ownsOutputBuffer() ) )
            {
                this->doSetPtrOutputBuffer( ptr_output_buffer );
            }
        }
    }

    CudaTrackJob::~CudaTrackJob()
    {

    }

    /* ===================================================================== */

    bool CudaTrackJob::hasCudaController() const SIXTRL_NOEXCEPT
    {
        return ( this->ptrCudaController() != nullptr );
    }

    CudaTrackJob::cuda_controller_t& CudaTrackJob::cudaController()
    {
        return const_cast< CudaTrackJob::cuda_controller_t& >( static_cast<
            CudaTrackJob const& >( *this ).cudaController() );
    }

    CudaTrackJob::cuda_controller_t const& CudaTrackJob::cudaController() const
    {
        if( !this->hasCudaController() )
        {
            throw std::runtime_error( "no cuda controller stored" );
        }

        return *this->ptrCudaController();
    }

    CudaTrackJob::cuda_controller_t*
    CudaTrackJob::ptrCudaController() SIXTRL_NOEXCEPT
    {
        return const_cast< CudaTrackJob::cuda_controller_t* >( static_cast<
            CudaTrackJob const& >( *this ).ptrCudaController() );
    }

    CudaTrackJob::cuda_controller_t const*
    CudaTrackJob::ptrCudaController() const SIXTRL_NOEXCEPT
    {
        using ctrl_t = CudaTrackJob::cuda_controller_t;
        using ptr_base_ctrl_t = CudaTrackJob::controller_base_t const*;

        ptr_base_ctrl_t ptr_base_ctrl = this->ptrControllerBase();

        return ( ptr_base_ctrl != nullptr )
            ? ptr_base_ctrl->asDerivedController< ctrl_t >( this->archId() )
            : nullptr;
    }

    /* --------------------------------------------------------------------- */

    bool CudaTrackJob::hasCudaParticlesArg() const SIXTRL_NOEXCEPT
    {
        return ( this->ptrCudaParticlesArg() != nullptr );
    }

    CudaTrackJob::cuda_argument_t& CudaTrackJob::cudaParticlesArg()
    {
        return this->doGetRefCudaArgument( this->ptrParticlesArgBase(),
            "ptrParticlesArgBase()" );
    }

    CudaTrackJob::cuda_argument_t const& CudaTrackJob::cudaParticlesArg() const
    {
        return this->doGetRefCudaArgument( this->ptrParticlesArgBase(),
            "ptrParticlesArgBase()" );
    }

    CudaTrackJob::cuda_argument_t*
    CudaTrackJob::ptrCudaParticlesArg() SIXTRL_NOEXCEPT
    {
        return this->doGetPtrCudaArgument( this->ptrParticlesArgBase() );
    }

    CudaTrackJob::cuda_argument_t const*
    CudaTrackJob::ptrCudaParticlesArg() const SIXTRL_NOEXCEPT
    {
        return this->doGetPtrCudaArgument( this->ptrParticlesArgBase() );
    }

    /* --------------------------------------------------------------------- */

    bool CudaTrackJob::hasCudaBeamElementsArg() const SIXTRL_NOEXCEPT
    {
        return ( this->ptrCudaBeamElementsArg() != nullptr );
    }

    CudaTrackJob::cuda_argument_t& CudaTrackJob::cudaBeamElementsArg()
    {
        return this->doGetRefCudaArgument( this->ptrBeamElementsArgBase(),
            "ptrBeamElementsArgBase()" );
    }

    CudaTrackJob::cuda_argument_t const&
    CudaTrackJob::cudaBeamElementsArg() const
    {
        return this->doGetRefCudaArgument( this->ptrBeamElementsArgBase(),
            "ptrBeamElementsArgBase()" );
    }

    CudaTrackJob::cuda_argument_t const*
    CudaTrackJob::ptrCudaBeamElementsArg() const SIXTRL_NOEXCEPT
    {
        return this->doGetPtrCudaArgument( this->ptrBeamElementsArgBase() );
    }

    CudaTrackJob::cuda_argument_t*
    CudaTrackJob::ptrCudaBeamElementsArg() SIXTRL_NOEXCEPT
    {
        return this->doGetPtrCudaArgument( this->ptrBeamElementsArgBase() );
    }

    /* --------------------------------------------------------------------- */

    bool CudaTrackJob::hasCudaOutputArg() const SIXTRL_NOEXCEPT
    {
        return ( this->ptrCudaOutputArg() != nullptr );
    }

    CudaTrackJob::cuda_argument_t& CudaTrackJob::cudaOutputArg()
    {
        return this->doGetRefCudaArgument( this->ptrOutputArgBase(),
            "ptrOutputArgBase()" );
    }

    CudaTrackJob::cuda_argument_t const& CudaTrackJob::cudaOutputArg() const
    {
        return this->doGetRefCudaArgument( this->ptrOutputArgBase(),
            "ptrOutputArgBase()" );
    }

    CudaTrackJob::cuda_argument_t*
    CudaTrackJob::ptrCudaOutputArg() SIXTRL_NOEXCEPT
    {
        return this->doGetPtrCudaArgument( this->ptrOutputArgBase() );
    }

    CudaTrackJob::cuda_argument_t const*
    CudaTrackJob::ptrCudaOutputArg() const SIXTRL_NOEXCEPT
    {
        return this->doGetPtrCudaArgument( this->ptrOutputArgBase() );
    }

    /* --------------------------------------------------------------------- */

    bool CudaTrackJob::hasCudaElemByElemConfigArg() const SIXTRL_NOEXCEPT
    {
        return ( this->ptrCudaElemByElemConfigArg() != nullptr );
    }

    CudaTrackJob::cuda_argument_t& CudaTrackJob::cudaElemByElemConfigArg()
    {
        return this->doGetRefCudaArgument( this->ptrElemByElemConfigArgBase(),
            "ptrElemByElemConfigArgBase()" );
    }

    CudaTrackJob::cuda_argument_t const&
    CudaTrackJob::cudaElemByElemConfigArg() const
    {
        return this->doGetRefCudaArgument( this->ptrElemByElemConfigArgBase(),
            "ptrElemByElemConfigArgBase()" );
    }

    CudaTrackJob::cuda_argument_t*
    CudaTrackJob::ptrCudaElemByElemConfigArg() SIXTRL_NOEXCEPT
    {
        return this->doGetPtrCudaArgument(
            this->ptrElemByElemConfigArgBase() );
    }

    CudaTrackJob::cuda_argument_t const*
    CudaTrackJob::ptrCudaElemByElemConfigArg() const SIXTRL_NOEXCEPT
    {
        return this->doGetPtrCudaArgument(
            this->ptrElemByElemConfigArgBase() );
    }

    /* --------------------------------------------------------------------- */

    bool CudaTrackJob::hasCudaDebugRegisterArg() const SIXTRL_NOEXCEPT
    {
        return ( this->ptrCudaDebugRegisterArg() != nullptr );
    }

    CudaTrackJob::cuda_argument_t& CudaTrackJob::cudaDebugRegisterArg()
    {
        return this->doGetRefCudaArgument(
            this->ptrDebugRegisterArgBase(), "ptrDebugRegisterArgBase()" );
    }

    CudaTrackJob::cuda_argument_t const&
    CudaTrackJob::cudaDebugRegisterArg() const
    {
        return this->doGetRefCudaArgument(
            this->ptrDebugRegisterArgBase(), "ptrDebugRegisterArgBase()" );
    }

    CudaTrackJob::cuda_argument_t*
    CudaTrackJob::ptrCudaDebugRegisterArg() SIXTRL_NOEXCEPT
    {
        return this->doGetPtrCudaArgument( this->ptrDebugRegisterArgBase() );
    }

    CudaTrackJob::cuda_argument_t const*
    CudaTrackJob::ptrCudaDebugRegisterArg() const SIXTRL_NOEXCEPT
    {
        return this->doGetPtrCudaArgument( this->ptrDebugRegisterArgBase() );
    }

    /* --------------------------------------------------------------------- */

    bool CudaTrackJob::hasCudaParticlesAddrArg() const SIXTRL_NOEXCEPT
    {
        return ( this->ptrCudaParticlesAddrArg() != nullptr );
    }

    CudaTrackJob::cuda_argument_t const&
    CudaTrackJob::cudaParticlesAddrArg() const
    {
        return this->doGetRefCudaArgument(
            this->ptrParticlesAddrArgBase(), "ptrParticlesAddrArgBase()" );
    }

    CudaTrackJob::cuda_argument_t& CudaTrackJob::cudaParticlesAddrArg()
    {
        return this->doGetRefCudaArgument(
            this->ptrParticlesAddrArgBase(), "ptrParticlesAddrArgBase()" );
    }

    CudaTrackJob::cuda_argument_t*
    CudaTrackJob::ptrCudaParticlesAddrArg() SIXTRL_NOEXCEPT
    {
        return this->doGetPtrCudaArgument( this->ptrParticlesAddrArgBase() );
    }

    CudaTrackJob::cuda_argument_t const*
    CudaTrackJob::ptrCudaParticlesAddrArg() const SIXTRL_NOEXCEPT
    {
        return this->doGetPtrCudaArgument( this->ptrParticlesAddrArgBase() );
    }

    /* ===================================================================== */

    bool CudaTrackJob::doPrepareController(
        char const* SIXTRL_RESTRICT config_str )
    {
        return this->doPrepareControllerCudaImpl( config_str );
    }

    bool CudaTrackJob::doPrepareDefaultKernels(
        char const* SIXTRL_RESTRICT config_str )
    {
        return this->doPrepareDefaultKernelsCudaImpl( config_str );
    }

    bool CudaTrackJob::doPrepareParticlesStructures(
        CudaTrackJob::c_buffer_t* SIXTRL_RESTRICT ptr_pbuffer )
    {
        using _base_t = st::CudaTrackJob::_base_track_job_t;
        return ( ( _base_t::doPrepareParticlesStructures( ptr_pbuffer ) ) &&
                 ( this->doPrepareParticlesStructuresCudaImpl( ptr_pbuffer ) ) );
    }

    bool CudaTrackJob::doPrepareBeamElementsStructures(
        CudaTrackJob::c_buffer_t* SIXTRL_RESTRICT belems )
    {
        using _base_t = st::CudaTrackJob::_base_track_job_t;
        return ( ( _base_t::doPrepareBeamElementsStructures( belems ) ) &&
                 ( this->doPrepareBeamElementsStructuresCudaImpl( belems ) ) );
    }

    bool CudaTrackJob::doPrepareOutputStructures(
        CudaTrackJob::c_buffer_t* SIXTRL_RESTRICT pbuffer,
        CudaTrackJob::c_buffer_t* SIXTRL_RESTRICT belems,
        CudaTrackJob::c_buffer_t* SIXTRL_RESTRICT output,
        CudaTrackJob::size_type const until_turn_elem_by_elem )
    {
        using _base_t = st::CudaTrackJob::_base_track_job_t;

        bool success = _base_t::doPrepareOutputStructures( pbuffer, belems,
            output, until_turn_elem_by_elem );

        if( ( success ) && ( this->hasOutputBuffer() ) )
        {
            success = this->doPrepareOutputStructuresCudaImpl( pbuffer, belems,
                this->ptrCOutputBuffer(), until_turn_elem_by_elem );
        }

        return success;
    }

    bool CudaTrackJob::doAssignOutputBufferToBeamMonitors(
        CudaTrackJob::c_buffer_t* SIXTRL_RESTRICT belems,
        CudaTrackJob::c_buffer_t* SIXTRL_RESTRICT output,
        CudaTrackJob::particle_index_t const min_turn_id,
        CudaTrackJob::size_type const output_buffer_offset_index )
    {
        using _base_t = st::CudaTrackJob::_base_track_job_t;

        bool success = _base_t::doAssignOutputBufferToBeamMonitors(
            belems, output, min_turn_id, output_buffer_offset_index );

        if( ( success ) && ( this->hasBeamMonitorOutput() ) )
        {
            success = this->doAssignOutputBufferToBeamMonitorsCudaImpl(
                belems, output, min_turn_id, output_buffer_offset_index );
        }

        return success;
    }

    bool CudaTrackJob::doAssignOutputBufferToElemByElemConfig(
        CudaTrackJob::elem_by_elem_config_t* SIXTRL_RESTRICT elem_by_elem_conf,
        CudaTrackJob::c_buffer_t* SIXTRL_RESTRICT output,
        CudaTrackJob::size_type const output_buffer_offset_index )
    {
        using _base_t = st::CudaTrackJob::_base_track_job_t;

        bool success = _base_t::doAssignOutputBufferToElemByElemConfig(
            elem_by_elem_conf, output, output_buffer_offset_index );

        if( ( success ) && ( this->hasElemByElemOutput() ) )
        {
            success = this->doAssignOutputBufferToElemByElemConfigCudaImpl(
                elem_by_elem_conf, output, output_buffer_offset_index );
        }

        return success;
    }


    bool CudaTrackJob::doReset(
        CudaTrackJob::c_buffer_t* SIXTRL_RESTRICT pbuffer,
        CudaTrackJob::c_buffer_t* SIXTRL_RESTRICT belems,
        CudaTrackJob::c_buffer_t* SIXTRL_RESTRICT output,
        CudaTrackJob::size_type const until_turn_elem_by_elem )
    {
        return this->doResetCudaImpl(
            pbuffer, belems, output, until_turn_elem_by_elem );
    }

    /* --------------------------------------------------------------------- */

    CudaTrackJob::cuda_argument_t const& CudaTrackJob::doGetRefCudaArgument(
        CudaTrackJob::argument_base_t const* ptr_base_arg,
        char const* SIXTRL_RESTRICT arg_name,
        bool const requires_exact_match ) const
    {
        using arg_t  = CudaTrackJob::cuda_argument_t;
        using size_t = CudaTrackJob::size_type;

        arg_t const* cuda_arg = ( ptr_base_arg != nullptr )
            ? ptr_base_arg->asDerivedArgument< arg_t >( this->archId() )
            : nullptr;

        if( cuda_arg == nullptr )
        {
            size_t const msg_capacity = size_t{ 101 };
            size_t const max_msg_length = msg_capacity - size_t{ 1 };

            char msg[ 101 ];
            std::memset( &msg[ 0 ], int{ '\0' }, msg_capacity );

            if( ( arg_name != nullptr ) && ( std::strlen( arg_name ) > 0u ) )
            {
                std::strncpy( msg, arg_name, max_msg_length );
                std::strncat( msg, " ", max_msg_length - std::strlen( msg ) );
            }

            std::strncat( msg, "argument not available, can't dereference",
                          max_msg_length - std::strlen( msg ) );

            throw std::runtime_error( &msg[ 0 ] );
        }

        return *cuda_arg;
    }

    CudaTrackJob::cuda_argument_t& CudaTrackJob::doGetRefCudaArgument(
        CudaTrackJob::argument_base_t const* ptr_base_arg,
        char const* SIXTRL_RESTRICT arg_name,
        bool const requires_exact_match )
    {
        return const_cast< CudaTrackJob::cuda_argument_t& >( static_cast<
            CudaTrackJob const& >( *this ).doGetRefCudaArgument( ptr_base_arg,
                arg_name, requires_exact_match ) );
    }

    CudaTrackJob::cuda_argument_t const* CudaTrackJob::doGetPtrCudaArgument(
        CudaTrackJob::argument_base_t const* ptr_base_arg,
        bool const requires_exact_match ) const SIXTRL_NOEXCEPT
    {
        using arg_t = CudaTrackJob::cuda_argument_t;

        return ( ptr_base_arg != nullptr )
            ? ptr_base_arg->asDerivedArgument< arg_t >(
                this->archId(), requires_exact_match ) : nullptr;
    }

    CudaTrackJob::cuda_argument_t* CudaTrackJob::doGetPtrCudaArgument(
        CudaTrackJob::argument_base_t* ptr_base_arg,
        bool const requires_exact_match ) SIXTRL_NOEXCEPT
    {
        return const_cast< CudaTrackJob::cuda_argument_t* >( static_cast<
            CudaTrackJob const& >( *this ).doGetPtrCudaArgument(
                ptr_base_arg, requires_exact_match ) );
    }

    /* ===================================================================== */

    bool CudaTrackJob::doPrepareControllerCudaImpl(
        char const* SIXTRL_RESTRICT config_str )
    {
        using _this_t           = st::CudaTrackJob;
        using cuda_ctrl_t       = _this_t::cuda_controller_t;
        using cuda_ctrl_store_t = _this_t::cuda_ctrl_store_t;

        using cuda_arg_t        = _this_t::cuda_argument_t;
        using cuda_arg_store_t  = _this_t::cuda_arg_store_t;
        using debug_register_t  = _this_t::debug_register_t;

        SIXTRL_ASSERT( this->ptrControllerBase() == nullptr );

        cuda_ctrl_store_t ptr_store_cuda_ctrl( new cuda_ctrl_t( config_str ) );
        this->doUpdateStoredController( std::move( ptr_store_cuda_ctrl ) );

        bool success = this->hasCudaController();

        if( ( success ) && ( nullptr != this->doGetPtrLocalDebugRegister() ) )
        {
            *( this->doGetPtrLocalDebugRegister() ) =
                st::ARCH_DEBUGGING_REGISTER_EMPTY;

            cuda_arg_store_t ptr_store_debug_flag_arg( new cuda_arg_t(
                this->doGetPtrLocalDebugRegister(), sizeof( debug_register_t ),
                this->ptrCudaController() ) );

            this->doUpdateStoredDebugRegisterArg(
                std::move( ptr_store_debug_flag_arg ) );

            success = this->hasCudaDebugRegisterArg();
        }

        return success;
    }

    bool CudaTrackJob::doPrepareDefaultKernelsCudaImpl(
        char const* SIXTRL_RESTRICT config_str )
    {
        using _this_t = st::CudaTrackJob;
        using size_t = _this_t::size_type;
        using cuda_ctrl_t = _this_t::cuda_controller_t;
        using cuda_kernel_config_t = _this_t::cuda_kernel_config_t;
        using kernel_id_t = _this_t::kernel_id_t;
        using cuda_node_info_t = _this_t::cuda_node_info_t;
        using node_index_t = _this_t::node_index_t;

        bool success = false;

        cuda_ctrl_t* ptr_cuda_ctrl = this->ptrCudaController();

        if( ptr_cuda_ctrl != nullptr )
        {
            node_index_t const selected_node_index =
                ptr_cuda_ctrl->selectedNodeIndex();

            cuda_node_info_t const* ptr_node_info = ptr_cuda_ctrl->ptrNodeInfo(
                selected_node_index );

            std::string kernel_name( 256, '\0' );
            std::string const kernel_prefix( SIXTRL_C99_NAMESPACE_PREFIX_STR );

            bool const use_debug = this->isInDebugMode();

            size_t num_kernel_args = size_t{ 0 };
            kernel_id_t kernel_id = cuda_controller_t::ILLEGAL_KERNEL_ID;

            /* trackUntilKernelId() */

            num_kernel_args = size_t{ 5 };
            kernel_name.clear();
            kernel_name += kernel_prefix;
            kernel_name += "Track_particles_until_turn_cuda_wrapper";

            kernel_id = ptr_cuda_ctrl->addCudaKernelConfig(
                kernel_name, num_kernel_args );

            cuda_kernel_config_t* ptr_kernel_config =
                ptr_cuda_ctrl->ptrKernelConfig( kernel_id );

            if( st::ARCH_STATUS_SUCCESS !=
                ::NS(CudaKernelConfig_configure_track_until_turn_kernel)(
                    ptr_kernel_config, ptr_node_info,
                        this->totalNumParticlesInParticleSets() ) )
            {
                kernel_id = cuda_ctrl_t::ILLEGAL_KERNEL_ID;
            }

            this->setTrackUntilKernelId( kernel_id );

            /* trackElemByElemKernelId() */

            num_kernel_args = size_t{ 8 };
            kernel_name.clear();
            kernel_name += kernel_prefix;
            kernel_name +=
                "Track_particles_elem_by_elem_until_turn_cuda_wrapper";

            kernel_id = ptr_cuda_ctrl->addCudaKernelConfig(
                kernel_name, num_kernel_args );

            ptr_kernel_config = ptr_cuda_ctrl->ptrKernelConfig( kernel_id );

            if( st::ARCH_STATUS_SUCCESS !=
                ::NS(CudaKernelConfig_configure_track_elem_by_elem_until_turn_kernel)(
                    ptr_kernel_config, ptr_node_info,
                        this->totalNumParticlesInParticleSets() ) )
            {
                kernel_id = cuda_ctrl_t::ILLEGAL_KERNEL_ID;
            }

            this->setTrackElemByElemKernelId( kernel_id );

            /* trackLineKernelId() */

            num_kernel_args = size_t{ 7 };
            kernel_name.clear();
            kernel_name += kernel_prefix;
            kernel_name += "Track_particles_line_cuda_wrapper";

            kernel_id = ptr_cuda_ctrl->addCudaKernelConfig(
                kernel_name, num_kernel_args );

            ptr_kernel_config = ptr_cuda_ctrl->ptrKernelConfig( kernel_id );

            if( st::ARCH_STATUS_SUCCESS !=
                ::NS(CudaKernelConfig_configure_track_line_kernel)(
                    ptr_kernel_config, ptr_node_info,
                        this->totalNumParticlesInParticleSets() ) )
            {
                kernel_id = cuda_ctrl_t::ILLEGAL_KERNEL_ID;
            }

            this->setTrackLineKernelId( kernel_id );

            /* assignOutputToBeamMonitorsKernelId() */

            num_kernel_args = size_t{ 5 };
            kernel_name.clear();
            kernel_name += kernel_prefix;
            kernel_name +=
                "BeamMonitor_assign_out_buffer_from_offset_cuda_wrapper";

            kernel_id = ptr_cuda_ctrl->addCudaKernelConfig(
                kernel_name, num_kernel_args );

            ptr_kernel_config = ptr_cuda_ctrl->ptrKernelConfig( kernel_id );

            if( st::ARCH_STATUS_SUCCESS !=
                ::NS(CudaKernelConfig_configure_assign_output_to_beam_monitors_kernel)(
                    ptr_kernel_config, ptr_node_info,
                        this->numBeamMonitors() ) )
            {
                kernel_id = cuda_ctrl_t::ILLEGAL_KERNEL_ID;
            }

            this->setAssignOutputToBeamMonitorsKernelId( kernel_id );

            /* assignOutputToElemByElemConfigKernelId() */

            num_kernel_args = size_t{ 4 };
            kernel_name.clear();
            kernel_name += kernel_prefix;
            kernel_name +=
                "ElemByElemConfig_assign_out_buffer_from_offset_cuda_wrapper";

            kernel_id = ptr_cuda_ctrl->addCudaKernelConfig(
                kernel_name, num_kernel_args );

            ptr_kernel_config = ptr_cuda_ctrl->ptrKernelConfig( kernel_id );

            if( st::ARCH_STATUS_SUCCESS !=
                ::NS(CudaKernelConfig_configure_assign_output_to_elem_by_elem_config_kernel)(
                    ptr_kernel_config, ptr_node_info ) )
            {
                kernel_id = cuda_ctrl_t::ILLEGAL_KERNEL_ID;
            }

            this->setAssignOutputToElemByElemConfigKernelId( kernel_id );


            /* fetchParticlesAddressesKernelId() */

            num_kernel_args = size_t{ 3 };
            kernel_name.clear();
            kernel_name += kernel_prefix;
            kernel_name += "Particles_buffer_store_all_addresses_cuda_wrapper";

            kernel_id = ptr_cuda_ctrl->addCudaKernelConfig(
                kernel_name, num_kernel_args );

            ptr_kernel_config = ptr_cuda_ctrl->ptrKernelConfig( kernel_id );

            if( st::ARCH_STATUS_SUCCESS !=
                ::NS(CudaKernelConfig_configure_fetch_particles_addresses_kernel)(
                    ptr_kernel_config, ptr_node_info, this->numParticleSets() ) )
            {
                kernel_id = cuda_ctrl_t::ILLEGAL_KERNEL_ID;
            }

            this->setFetchParticlesAddressesKernelId( kernel_id );
        }

        return success;
    }

    bool CudaTrackJob::doPrepareParticlesStructuresCudaImpl(
        CudaTrackJob::c_buffer_t* SIXTRL_RESTRICT particles_buffer )
    {
        using _this_t = st::CudaTrackJob;

        using cuda_ctrl_t      = _this_t::cuda_controller_t;
        using cuda_arg_t       = _this_t::cuda_argument_t;
        using store_base_arg_t = _this_t::stored_arg_base_t;

        bool success = false;

        cuda_ctrl_t* ptr_cuda_ctrl = this->ptrCudaController();

        if( ( ptr_cuda_ctrl != nullptr ) && ( particles_buffer != nullptr ) )
        {
            if( this->doGetPtrParticlesAddrBuffer() != nullptr )
            {
                store_base_arg_t cuda_particles_addr_arg(
                    new cuda_arg_t( *this->doGetPtrParticlesAddrBuffer(),
                        ptr_cuda_ctrl ) );

                success = ( ( cuda_particles_addr_arg.get() != nullptr ) &&
                    ( cuda_particles_addr_arg->ptrControllerBase() ==
                      ptr_cuda_ctrl ) &&
                    ( cuda_particles_addr_arg->usesCObjectsCxxBuffer() ) &&
                    ( cuda_particles_addr_arg->ptrCObjectsCxxBuffer() ==
                      this->doGetPtrParticlesAddrBuffer() ) );

                this->doUpdateStoredParticlesAddrArg(
                    std::move( cuda_particles_addr_arg ) );

                success &= (
                    ( this->ptrParticlesAddrArgBase() != nullptr ) &&
                    ( this->ptrParticlesAddrArgBase()->usesCObjectsCxxBuffer() ) &&
                    ( this->ptrParticlesAddrArgBase()->ptrCObjectsCxxBuffer()
                      == this->doGetPtrParticlesAddrBuffer() ) );
            }

            store_base_arg_t cuda_particles_arg(
                new cuda_arg_t( particles_buffer, ptr_cuda_ctrl ) );

            if( ( cuda_particles_arg.get() != nullptr ) &&
                ( cuda_particles_arg->usesCObjectsBuffer() ) &&
                ( cuda_particles_arg->ptrCObjectsBuffer() ==
                  particles_buffer  ) )
            {
                this->doUpdateStoredParticlesArg(
                    std::move( cuda_particles_arg ) );

                if( ( this->ptrParticlesArgBase() != nullptr ) &&
                    ( this->ptrParticlesArgBase()->ptrControllerBase() ==
                      ptr_cuda_ctrl ) &&
                    ( this->ptrParticlesArgBase()->usesCObjectsBuffer() ) &&
                    ( this->ptrParticlesArgBase()->ptrCObjectsBuffer() ==
                      particles_buffer ) )
                {
                    success = true;
                }
            }
        }

        return success;
    }

    bool CudaTrackJob::doPrepareBeamElementsStructuresCudaImpl(
         CudaTrackJob::c_buffer_t* SIXTRL_RESTRICT belems_buffer )
    {
        using _this_t          = st::CudaTrackJob;
        using cuda_ctrl_t      = _this_t::cuda_controller_t;
        using cuda_arg_t       = _this_t::cuda_argument_t;
        using store_base_arg_t = _this_t::stored_arg_base_t;

        bool success = false;

        cuda_ctrl_t* ptr_cuda_ctrl = this->ptrCudaController();

        if( ( ptr_cuda_ctrl != nullptr ) && ( belems_buffer != nullptr ) )
        {
            store_base_arg_t cuda_belems_arg(
                new cuda_arg_t( belems_buffer, ptr_cuda_ctrl ) );

            if( ( cuda_belems_arg.get() != nullptr ) &&
                ( cuda_belems_arg->usesCObjectsBuffer() ) &&
                ( cuda_belems_arg->ptrCObjectsBuffer() == belems_buffer ) )
            {
                this->doUpdateStoredBeamElementsArg(
                    std::move( cuda_belems_arg ) );

                if( ( this->ptrBeamElementsArgBase() != nullptr ) &&
                    ( this->ptrBeamElementsArgBase()->ptrControllerBase() ==
                      ptr_cuda_ctrl ) &&
                    ( this->ptrBeamElementsArgBase()->usesCObjectsBuffer() ) &&
                    ( this->ptrBeamElementsArgBase()->ptrCObjectsBuffer() ==
                      belems_buffer ) )
                {
                    success = true;
                }
            }
        }

        return success;
    }

    bool CudaTrackJob::doPrepareOutputStructuresCudaImpl(
        CudaTrackJob::c_buffer_t* SIXTRL_RESTRICT particles_buffer,
        CudaTrackJob::c_buffer_t* SIXTRL_RESTRICT belems_buffer,
        CudaTrackJob::c_buffer_t* SIXTRL_RESTRICT output_buffer,
        CudaTrackJob::size_type const )
    {
        using _this_t          = st::CudaTrackJob;
        using cuda_ctrl_t      = _this_t::cuda_controller_t;
        using cuda_arg_t       = _this_t::cuda_argument_t;
        using store_base_arg_t = _this_t::stored_arg_base_t;
        using elem_config_t    = _this_t::elem_by_elem_config_t;
        using ctrl_status_t    = _this_t::status_t;

        bool success = false;

        cuda_ctrl_t* ptr_cuda_ctrl = this->ptrCudaController();

        if( ( ptr_cuda_ctrl != nullptr ) && ( output_buffer != nullptr ) &&
            ( particles_buffer != nullptr ) )
        {
            store_base_arg_t cuda_elem_by_elem_conf_arg( nullptr );

            if( this->ptrElemByElemConfig() == nullptr )
            {
                success = true;
            }
            else
            {
                cuda_elem_by_elem_conf_arg.reset( new cuda_arg_t(
                    this->ptrElemByElemConfig(), sizeof( elem_config_t ),
                        ptr_cuda_ctrl ) );

                success = ( ( cuda_elem_by_elem_conf_arg.get() != nullptr ) &&
                    ( cuda_elem_by_elem_conf_arg->ptrControllerBase() ==
                      ptr_cuda_ctrl ) );

                this->doUpdateStoredElemByElemConfigArg( std::move(
                    cuda_elem_by_elem_conf_arg ) );

                success &= (
                    ( this->ptrElemByElemConfigArgBase() != nullptr ) &&
                    ( this->ptrElemByElemConfigArgBase()->ptrControllerBase()
                      == ptr_cuda_ctrl ) );

                if( success )
                {
                    ctrl_status_t const status =
                    this->ptrElemByElemConfigArgBase()->send(
                        this->ptrElemByElemConfig(), sizeof( elem_config_t ) );

                    success = (
                        ( status == st::ARCH_STATUS_SUCCESS ) &&
                        ( this->ptrElemByElemConfigArgBase(
                            )->usesRawArgument() )&&
                        ( this->ptrElemByElemConfigArgBase()->ptrRawArgument()
                            == this->ptrElemByElemConfig() ) );
                }
            }

            if( success )
            {
                store_base_arg_t cuda_output_arg(
                    new cuda_arg_t( output_buffer, ptr_cuda_ctrl ) );

                success = ( ( cuda_output_arg.get() != nullptr ) &&
                    ( cuda_output_arg->ptrControllerBase() == ptr_cuda_ctrl ) &&
                    ( cuda_output_arg->usesCObjectsBuffer() ) &&
                    ( cuda_output_arg->ptrCObjectsBuffer() != nullptr ) &&
                    ( cuda_output_arg->ptrCObjectsBuffer() == output_buffer ) );

                this->doUpdateStoredOutputArg( std::move( cuda_output_arg ) );

                success &= ( ( this->ptrOutputArgBase() != nullptr ) &&
                    ( this->ptrOutputArgBase()->ptrControllerBase() ==
                      ptr_cuda_ctrl ) &&
                    ( this->ptrOutputArgBase()->usesCObjectsBuffer() ) &&
                    ( this->ptrOutputArgBase()->ptrCObjectsBuffer() ==
                      output_buffer ) );
            }
        }

        return success;
    }

    bool CudaTrackJob::doAssignOutputBufferToBeamMonitorsCudaImpl(
        CudaTrackJob::c_buffer_t* SIXTRL_RESTRICT belems_buffer,
        CudaTrackJob::c_buffer_t* SIXTRL_RESTRICT output_buffer,
        CudaTrackJob::particle_index_t const min_turn_id,
        CudaTrackJob::size_type const output_buffer_offset_index )
    {
        using _this_t            = st::CudaTrackJob;
        using cuda_ctrl_t        = _this_t::cuda_controller_t;
        using cuda_arg_t         = _this_t::cuda_argument_t;
        using cuda_kernel_conf_t = _this_t::cuda_kernel_config_t;
        using kernel_id_t        = _this_t::kernel_id_t;
        using ctrl_status_t      = _this_t::status_t;
        using size_t             = _this_t::size_type;
        using particle_index_t   = _this_t::particle_index_t;

        ctrl_status_t status = st::ARCH_STATUS_GENERAL_FAILURE;
        cuda_ctrl_t* ptr_cuda_ctrl = this->ptrCudaController();
        kernel_id_t const kid = this->assignOutputToBeamMonitorsKernelId();

        bool const controller_ready = (
            ( ptr_cuda_ctrl != nullptr ) &&
            ( ptr_cuda_ctrl->readyForRunningKernel() ) &&
            ( ptr_cuda_ctrl->readyForSend() ) &&
            ( ptr_cuda_ctrl->readyForReceive() ) &&
            ( ptr_cuda_ctrl->hasKernel( kid ) ) &&
            ( ptr_cuda_ctrl->ptrKernelConfigBase( kid ) != nullptr ) );

        cuda_arg_t* output_arg = this->ptrCudaOutputArg();

        bool const output_ready = ( ( this->hasBeamMonitorOutput() ) &&
            ( output_buffer != nullptr ) && ( output_arg != nullptr ) &&
            ( ::NS(Buffer_get_num_of_objects)( output_buffer ) >
              output_buffer_offset_index ) &&
            ( output_arg->ptrControllerBase() == ptr_cuda_ctrl ) &&
            ( output_arg->usesCObjectsBuffer() ) &&
            ( output_arg->ptrCObjectsBuffer() == output_buffer ) );

        cuda_arg_t* belem_arg = this->ptrCudaBeamElementsArg();

        bool const belems_ready = (
            ( belem_arg != nullptr ) && ( belems_buffer != nullptr ) &&
            ( belem_arg->ptrControllerBase() == ptr_cuda_ctrl ) &&
            ( belem_arg->usesCObjectsBuffer() ) &&
            ( belem_arg->ptrCObjectsBuffer() == belems_buffer ) );

        cuda_kernel_conf_t const* kernel_conf = ( controller_ready )
            ? ptr_cuda_ctrl->ptrKernelConfig( kid ) : nullptr;

        if( ( controller_ready ) && ( output_ready ) &&
            ( belems_ready ) && ( kernel_conf != nullptr ) )
        {
            size_t const offset = this->beamMonitorsOutputBufferOffset();
            particle_index_t const min_turn_id = this->minInitialTurnId();

            if( !this->isInDebugMode() )
            {
                ::NS(BeamMonitor_assign_out_buffer_from_offset_cuda_wrapper)(
                    kernel_conf, belem_arg, output_arg, min_turn_id,
                    output_buffer_offset_index, nullptr );

                status = st::ARCH_STATUS_SUCCESS;
            }
            else if( this->hasCudaDebugRegisterArg() )
            {
                if( this->prepareDebugRegisterForUse() ==
                    st::ARCH_STATUS_SUCCESS )
                {
                    ::NS(BeamMonitor_assign_out_buffer_from_offset_cuda_wrapper)(
                        kernel_conf, belem_arg, output_arg, min_turn_id,
                        output_buffer_offset_index,
                        this->ptrCudaDebugRegisterArg() );

                    status = this->evaluateDebugRegisterAfterUse();
                }
            }
        }

        return ( status == st::ARCH_STATUS_SUCCESS );
    }

    bool CudaTrackJob::doAssignOutputBufferToElemByElemConfigCudaImpl(
        CudaTrackJob::elem_by_elem_config_t* SIXTRL_RESTRICT elem_by_elem_config,
        CudaTrackJob::c_buffer_t* SIXTRL_RESTRICT output_buffer,
        CudaTrackJob::size_type const output_buffer_offset_index )
    {
        using _this_t            = st::CudaTrackJob;
        using cuda_ctrl_t        = _this_t::cuda_controller_t;
        using cuda_arg_t         = _this_t::cuda_argument_t;
        using cuda_kernel_conf_t = _this_t::cuda_kernel_config_t;
        using kernel_id_t        = _this_t::kernel_id_t;
        using ctrl_status_t      = _this_t::status_t;
        using size_t             = _this_t::size_type;

        ctrl_status_t status = st::ARCH_STATUS_GENERAL_FAILURE;
        cuda_ctrl_t* ptr_cuda_ctrl = this->ptrCudaController();
        kernel_id_t const kid = this->assignOutputToBeamMonitorsKernelId();

        bool const controller_ready = (
            ( ptr_cuda_ctrl != nullptr ) &&
            ( ptr_cuda_ctrl->readyForRunningKernel() ) &&
            ( ptr_cuda_ctrl->readyForSend() ) &&
            ( ptr_cuda_ctrl->readyForReceive() ) &&
            ( ptr_cuda_ctrl->hasKernel( kid ) ) &&
            ( ptr_cuda_ctrl->ptrKernelConfigBase( kid ) != nullptr ) );

        cuda_arg_t* output_arg = this->ptrCudaOutputArg();

        bool const output_ready = ( ( this->hasBeamMonitorOutput() ) &&
            ( output_buffer != nullptr ) && ( output_arg != nullptr ) &&
            ( ::NS(Buffer_get_num_of_objects)( output_buffer ) >
              output_buffer_offset_index ) &&
            ( output_arg->ptrControllerBase() == ptr_cuda_ctrl ) &&
            ( output_arg->usesCObjectsBuffer() ) &&
            ( output_arg->ptrCObjectsBuffer() == output_buffer ) );

        cuda_arg_t* elem_by_elem_conf_arg = this->ptrCudaElemByElemConfigArg();

        bool const elem_by_elem_conf_ready = (
            ( elem_by_elem_conf_arg != nullptr ) &&
            ( elem_by_elem_config != nullptr ) &&
            ( elem_by_elem_conf_arg->ptrControllerBase() == ptr_cuda_ctrl ) &&
            ( elem_by_elem_conf_arg->usesRawArgument() ) &&
            ( elem_by_elem_conf_arg->ptrRawArgument() ==
                elem_by_elem_config ) );

        cuda_kernel_conf_t const* kernel_conf = ( controller_ready )
            ? ptr_cuda_ctrl->ptrKernelConfig( kid ) : nullptr;

        if( ( controller_ready ) && ( output_ready ) &&
            ( elem_by_elem_conf_ready ) && ( kernel_conf != nullptr ) )
        {
            if( !this->isInDebugMode() )
            {
                ::NS(ElemByElemConfig_assign_out_buffer_from_offset_cuda_wrapper)(
                    kernel_conf, elem_by_elem_conf_arg, output_arg,
                        output_buffer_offset_index, nullptr );

                status = st::ARCH_STATUS_SUCCESS;
            }
            else if( this->hasCudaDebugRegisterArg() )
            {
                if( this->prepareDebugRegisterForUse() ==
                    st::ARCH_STATUS_SUCCESS )
                {
                    ::NS(ElemByElemConfig_assign_out_buffer_from_offset_cuda_wrapper)(
                        kernel_conf, elem_by_elem_conf_arg, output_arg,
                        output_buffer_offset_index,
                        this->ptrCudaDebugRegisterArg() );

                    status = this->evaluateDebugRegisterAfterUse();
                }
            }
        }

        return ( status == st::ARCH_STATUS_SUCCESS );
    }

    /* --------------------------------------------------------------------- */

    CudaTrackJob::status_t CudaTrackJob::doFetchParticleAddresses()
    {
        using _this_t            = st::CudaTrackJob;
        using cuda_ctrl_t        = _this_t::cuda_controller_t;
        using cuda_arg_t         = _this_t::cuda_argument_t;
        using cuda_kernel_conf_t = _this_t::cuda_kernel_config_t;
        using kernel_id_t        = _this_t::kernel_id_t;
        using ctrl_status_t      = _this_t::status_t;
        using size_t             = _this_t::size_type;
        using particle_index_t   = _this_t::particle_index_t;
        using debug_register_t     = _this_t::debug_register_t;
        using buffer_t           = _this_t::buffer_t;
        using c_buffer_t         = _this_t::c_buffer_t;

        status_t status = st::ARCH_STATUS_GENERAL_FAILURE;

        kernel_id_t const kid = this->fetchParticlesAddressesKernelId();
        cuda_ctrl_t* ptr_ctrl = this->ptrCudaController();

        c_buffer_t* pbuffer = this->ptrCParticlesBuffer();
        buffer_t* paddr_buffer = this->doGetPtrParticlesAddrBuffer();

        bool const controller_ready = (
            ( ptr_ctrl != nullptr ) &&
            ( ptr_ctrl->readyForRunningKernel() ) &&
            ( ptr_ctrl->readyForSend() ) &&
            ( ptr_ctrl->readyForReceive() ) &&
            ( ptr_ctrl->hasKernel( kid ) ) &&
            ( ptr_ctrl->ptrKernelConfigBase( kid ) != nullptr ) );

        cuda_arg_t* particles_arg = this->ptrCudaParticlesArg();

        bool const particles_ready = (
            ( pbuffer != nullptr ) && ( particles_arg != nullptr ) &&
            ( particles_arg->ptrControllerBase() == ptr_ctrl ) &&
            ( particles_arg->usesCObjectsBuffer() ) &&
            ( particles_arg->ptrCObjectsBuffer() == pbuffer ) );

        cuda_arg_t* particles_addr_arg = this->ptrCudaParticlesAddrArg();

        bool const particles_addr_ready = (
            ( particles_addr_arg != nullptr ) && ( paddr_buffer != nullptr ) &&
            ( particles_addr_arg->ptrControllerBase() == ptr_ctrl ) &&
            ( particles_addr_arg->usesCObjectsCxxBuffer() ) &&
            ( particles_addr_arg->ptrCObjectsCxxBuffer() == paddr_buffer ) );

        cuda_kernel_conf_t const* kernel_conf = ( controller_ready )
            ? ptr_ctrl->ptrKernelConfig( kid ) : nullptr;

        cuda_arg_t* debug_flag_arg = nullptr;

        if( ( controller_ready ) && ( particles_ready ) &&
            ( particles_addr_ready ) && ( kernel_conf != nullptr ) )
        {
            if( !this->isInDebugMode() )
            {
                ::NS(Particles_buffer_store_all_addresses_cuda_wrapper)(
                    kernel_conf, particles_addr_arg, particles_arg, nullptr );

                status = st::ARCH_STATUS_SUCCESS;
            }
            else if( this->hasCudaDebugRegisterArg() )
            {
                if( this->prepareDebugRegisterForUse() ==
                    st::ARCH_STATUS_SUCCESS )
                {
                    ::NS(Particles_buffer_store_all_addresses_cuda_wrapper)(
                        kernel_conf, particles_addr_arg, particles_arg,
                        this->ptrCudaDebugRegisterArg() );

                    status = this->evaluateDebugRegisterAfterUse();
                }
            }
        }

        return status;
    }

    CudaTrackJob::track_status_t CudaTrackJob::doTrackUntilTurn(
        CudaTrackJob::size_type const until_turn )
    {
        return st::trackUntilTurn( *this, until_turn );
    }

    CudaTrackJob::track_status_t CudaTrackJob::doTrackElemByElem(
        CudaTrackJob::size_type const until_turn_elem_by_elem )
    {
        return st::trackElemByElemUntilTurn( *this, until_turn_elem_by_elem );
    }

    CudaTrackJob::track_status_t CudaTrackJob::doTrackLine(
            CudaTrackJob::size_type const be_begin_index,
            CudaTrackJob::size_type const be_end_index,
            bool const finish_turn )
    {
        return st::trackLine(
            *this, be_begin_index, be_end_index, finish_turn );
    }

    /* --------------------------------------------------------------------- */

    bool CudaTrackJob::doResetCudaImpl(
        CudaTrackJob::c_buffer_t* SIXTRL_RESTRICT pbuffer,
        CudaTrackJob::c_buffer_t* SIXTRL_RESTRICT belem_buffer,
        CudaTrackJob::c_buffer_t* SIXTRL_RESTRICT ptr_output_buffer,
        CudaTrackJob::size_type const until_turn_elem_by_elem )
    {
        bool success = false;
        using _this_t = st::CudaTrackJob;
        using _base_t = _this_t::_base_track_job_t;
        using output_buffer_flag_t = _this_t::output_buffer_flag_t;

        if( ( _base_t::doPrepareParticlesStructures( pbuffer ) ) &&
            ( _base_t::doPrepareBeamElementsStructures( belem_buffer ) ) &&
            ( this->doPrepareParticlesStructuresCudaImpl( pbuffer ) ) &&
            ( this->doPrepareBeamElementsStructuresCudaImpl( belem_buffer ) ) )
        {
            output_buffer_flag_t const out_buffer_flags =
            ::NS(OutputBuffer_required_for_tracking_of_particle_sets)( pbuffer,
                this->numParticleSets(), this->particleSetIndicesBegin(),
                    belem_buffer, until_turn_elem_by_elem );

            bool const requires_output_buffer =
                ::NS(OutputBuffer_requires_output_buffer)( out_buffer_flags );

            this->doSetPtrCParticleBuffer( pbuffer );
            this->doSetPtrCBeamElementsBuffer( belem_buffer );

            if( ( requires_output_buffer ) || ( ptr_output_buffer != nullptr ) )
            {
                success = this->doPrepareOutputStructures( pbuffer,
                    belem_buffer, ptr_output_buffer, until_turn_elem_by_elem );

                success &= this->doPrepareOutputStructuresCudaImpl( pbuffer,
                    belem_buffer, ptr_output_buffer, until_turn_elem_by_elem );
            }

            if( ( success ) && ( this->hasOutputBuffer() ) &&
                ( requires_output_buffer ) )
            {
                if( ::NS(OutputBuffer_requires_elem_by_elem_output)(
                        out_buffer_flags ) )
                {
                    success &= _base_t::doAssignOutputBufferToElemByElemConfig(
                        this->ptrElemByElemConfig(), this->ptrCOutputBuffer(),
                            this->elemByElemOutputBufferOffset() );

                    success &= this->doAssignOutputBufferToElemByElemConfigCudaImpl(
                        this->ptrElemByElemConfig(), this->ptrCOutputBuffer(),
                            this->elemByElemOutputBufferOffset() );
                }

                if( ::NS(OutputBuffer_requires_beam_monitor_output)(
                        out_buffer_flags ) )
                {
                    success &= _base_t::doAssignOutputBufferToBeamMonitors(
                        belem_buffer, this->ptrCOutputBuffer(),
                        this->minInitialTurnId(),
                        this->beamMonitorsOutputBufferOffset() );

                    success &= this->doAssignOutputBufferToBeamMonitorsCudaImpl(
                        belem_buffer, this->ptrCOutputBuffer(),
                        this->minInitialTurnId(),
                        this->beamMonitorsOutputBufferOffset() );
                }
            }
        }

        return success;
    }

    /* ********************************************************************* */
    /* ********    Implement CudaTrackJob stand-alone functions   ********** */
    /* ********************************************************************* */

    CudaTrackJob::collect_flag_t collect(
        CudaTrackJob& SIXTRL_RESTRICT_REF track_job )
    {
        return track_job.collect();
    }

    CudaTrackJob::track_status_t trackUntilTurn(
        CudaTrackJob& SIXTRL_RESTRICT_REF trackjob,
        CudaTrackJob::size_type const until_turn )
    {
        using track_job_t     = CudaTrackJob;
        using controller_t    = track_job_t::cuda_controller_t;
        using argument_t      = track_job_t::cuda_argument_t;
        using kernel_config_t = track_job_t::cuda_kernel_config_t;
        using kernel_id_t     = track_job_t::kernel_id_t;
        using size_t          = track_job_t::size_type;

        CudaTrackJob::track_status_t status = st::TRACK_STATUS_GENERAL_FAILURE;

        kernel_id_t const kid = trackjob.trackUntilKernelId();
        SIXTRL_ASSERT( kid != controller_t::ILLEGAL_KERNEL_ID );

        controller_t const* ptr_ctrl = trackjob.ptrCudaController();
        SIXTRL_ASSERT( ptr_ctrl != nullptr );
        SIXTRL_ASSERT( ptr_ctrl->hasSelectedNode() );
        SIXTRL_ASSERT( ptr_ctrl->readyForRunningKernel() );

        kernel_config_t const* kernel_conf = ptr_ctrl->ptrKernelConfig( kid );
        SIXTRL_ASSERT( kernel_conf != nullptr );
        SIXTRL_ASSERT( trackjob.hasCudaParticlesArg() );
        SIXTRL_ASSERT( trackjob.hasCudaBeamElementsArg() );

        SIXTRL_ASSERT( trackjob.numParticleSets() == size_t{ 1 } );
        SIXTRL_ASSERT( trackjob.particleSetIndicesBegin() != nullptr );

        size_t const pset_index = *trackjob.particleSetIndicesBegin();

        if( !trackjob.isInDebugMode() )
        {
            ::NS(Track_particles_until_turn_cuda_wrapper)( kernel_conf,
                trackjob.ptrCudaParticlesArg(), pset_index,
                trackjob.ptrCudaBeamElementsArg(), until_turn, nullptr );

            status = st::TRACK_SUCCESS;
        }
        else if( trackjob.hasCudaDebugRegisterArg() )
        {
            using ctrl_status_t = track_job_t::status_t;

            ctrl_status_t ctrl_status = trackjob.prepareDebugRegisterForUse();

            if( ctrl_status == st::ARCH_STATUS_SUCCESS )
            {
                ::NS(Track_particles_until_turn_cuda_wrapper)( kernel_conf,
                    trackjob.ptrCudaParticlesArg(), pset_index,
                    trackjob.ptrCudaBeamElementsArg(), until_turn,
                    trackjob.ptrCudaDebugRegisterArg() );

                ctrl_status = trackjob.evaluateDebugRegisterAfterUse();
            }

            if( ctrl_status == st::ARCH_STATUS_SUCCESS )
            {
                status = st::TRACK_SUCCESS;
            }
            else
            {
                status = static_cast< track_status_t >( ctrl_status );
            }
        }

        return status;
    }

    CudaTrackJob::track_status_t trackElemByElemUntilTurn(
        CudaTrackJob& SIXTRL_RESTRICT_REF trackjob,
        CudaTrackJob::size_type const until_turn )
    {
        using track_job_t     = CudaTrackJob;
        using controller_t    = track_job_t::cuda_controller_t;
        using argument_t      = track_job_t::cuda_argument_t;
        using kernel_config_t = track_job_t::cuda_kernel_config_t;
        using kernel_id_t     = track_job_t::kernel_id_t;

        CudaTrackJob::track_status_t status = st::TRACK_STATUS_GENERAL_FAILURE;

        kernel_id_t const kid = trackjob.trackElemByElemKernelId();
        SIXTRL_ASSERT( kid != controller_t::ILLEGAL_KERNEL_ID );

        controller_t const* ptr_ctrl = trackjob.ptrCudaController();
        SIXTRL_ASSERT( ptr_ctrl != nullptr );
        SIXTRL_ASSERT( ptr_ctrl->hasSelectedNode() );
        SIXTRL_ASSERT( ptr_ctrl->readyForRunningKernel() );

        kernel_config_t const* kernel_conf = ptr_ctrl->ptrKernelConfig( kid );
        SIXTRL_ASSERT( kernel_conf != nullptr );
        SIXTRL_ASSERT( trackjob.hasCudaParticlesArg() );
        SIXTRL_ASSERT( trackjob.hasCudaBeamElementsArg() );
        SIXTRL_ASSERT( trackjob.hasCudaElemByElemConfigArg() );
        SIXTRL_ASSERT( trackjob.hasCudaOutputArg() );
        SIXTRL_ASSERT( trackjob.ptrElemByElemConfig() != nullptr );
        SIXTRL_ASSERT( trackjob.hasElemByElemOutput() );

        SIXTRL_ASSERT( trackjob.numParticleSets() == size_t{ 1 } );
        SIXTRL_ASSERT( trackjob.particleSetIndicesBegin() != nullptr );

        size_t const pset_index = *trackjob.particleSetIndicesBegin();

        if( !trackjob.isInDebugMode() )
        {
            ::NS(Track_particles_elem_by_elem_until_turn_cuda_wrapper)(
                kernel_conf, trackjob.ptrCudaParticlesArg(), pset_index,
                trackjob.ptrCudaBeamElementsArg(), trackjob.ptrCudaOutputArg(),
                trackjob.ptrCudaElemByElemConfigArg(),
                trackjob.beamMonitorsOutputBufferOffset(), until_turn,
                nullptr );

            status = st::TRACK_SUCCESS;
        }
        else if( trackjob.hasCudaDebugRegisterArg() )
        {
            using ctrl_status_t = track_job_t::status_t;

            ctrl_status_t ctrl_status = trackjob.prepareDebugRegisterForUse();

            if( ctrl_status == st::ARCH_STATUS_SUCCESS )
            {
                ::NS(Track_particles_elem_by_elem_until_turn_cuda_wrapper)(
                    kernel_conf, trackjob.ptrCudaParticlesArg(), pset_index,
                    trackjob.ptrCudaBeamElementsArg(),
                    trackjob.ptrCudaOutputArg(),
                    trackjob.ptrCudaElemByElemConfigArg(),
                    trackjob.beamMonitorsOutputBufferOffset(), until_turn,
                    trackjob.ptrCudaDebugRegisterArg() );

                ctrl_status = trackjob.evaluateDebugRegisterAfterUse();
            }

            if( ctrl_status == st::ARCH_STATUS_SUCCESS )
            {
                status = st::TRACK_SUCCESS;
            }
            else
            {
                status = static_cast< track_status_t >( ctrl_status );
            }
        }

        return status;
    }

    CudaTrackJob::track_status_t trackLine(
        CudaTrackJob& SIXTRL_RESTRICT_REF trackjob,
        CudaTrackJob::size_type const belem_begin_id,
        CudaTrackJob::size_type const belem_end_id,
        bool const finish_turn )
    {
        using track_job_t     = CudaTrackJob;
        using controller_t    = track_job_t::cuda_controller_t;
        using argument_t      = track_job_t::cuda_argument_t;
        using kernel_config_t = track_job_t::cuda_kernel_config_t;
        using kernel_id_t     = track_job_t::kernel_id_t;

        CudaTrackJob::track_status_t status = st::TRACK_STATUS_GENERAL_FAILURE;

        kernel_id_t const kid = trackjob.trackElemByElemKernelId();
        SIXTRL_ASSERT( kid != controller_t::ILLEGAL_KERNEL_ID );

        controller_t const* ptr_ctrl = trackjob.ptrCudaController();
        SIXTRL_ASSERT( ptr_ctrl != nullptr );
        SIXTRL_ASSERT( ptr_ctrl->hasSelectedNode() );
        SIXTRL_ASSERT( ptr_ctrl->readyForRunningKernel() );

        kernel_config_t const* kernel_conf = ptr_ctrl->ptrKernelConfig( kid );
        SIXTRL_ASSERT( kernel_conf != nullptr );
        SIXTRL_ASSERT( trackjob.hasCudaParticlesArg() );
        SIXTRL_ASSERT( trackjob.hasCudaBeamElementsArg() );

        SIXTRL_ASSERT( trackjob.numParticleSets() == size_t{ 1 } );
        SIXTRL_ASSERT( trackjob.particleSetIndicesBegin() != nullptr );

        size_t const pset_index = *trackjob.particleSetIndicesBegin();

        if( !trackjob.isInDebugMode() )
        {
            ::NS(Track_particles_line_cuda_wrapper)(
                kernel_conf, trackjob.ptrCudaParticlesArg(), pset_index,
                trackjob.ptrCudaBeamElementsArg(),
                belem_begin_id, belem_end_id, finish_turn, nullptr );

            status = st::TRACK_SUCCESS;
        }
        else if( trackjob.hasCudaDebugRegisterArg() )
        {
            using ctrl_status_t = track_job_t::status_t;

            ctrl_status_t ctrl_status = trackjob.prepareDebugRegisterForUse();

            if( ctrl_status == st::ARCH_STATUS_SUCCESS )
            {
                ::NS(Track_particles_line_cuda_wrapper)( kernel_conf,
                    trackjob.ptrCudaParticlesArg(), pset_index,
                    trackjob.ptrCudaBeamElementsArg(),
                    belem_begin_id, belem_end_id, finish_turn,
                    trackjob.ptrCudaDebugRegisterArg() );

                ctrl_status = trackjob.evaluateDebugRegisterAfterUse();
            }

            if( ctrl_status == st::ARCH_STATUS_SUCCESS )
            {
                status = st::TRACK_SUCCESS;
            }
            else
            {
                status = static_cast< track_status_t >( ctrl_status );
            }
        }

        return status;
    }
}

#endif /* C++, Host */

/* end: sixtracklib/cuda/track/track_job.cpp */
