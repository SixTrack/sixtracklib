#include "sixtracklib/cuda/controller.hpp"

#if defined( __cplusplus ) && !defined( _GPUCODE ) && !defined( __CUDA_ARCH__ )

#include <cstddef>
#include <cstdlib>

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/control/definitions.h"
#include "sixtracklib/common/control/node_id.hpp"
#include "sixtracklib/common/control/controller_base.hpp"
#include "sixtracklib/common/control/node_controller_base.hpp"

#include "sixtracklib/cuda/definitions.h"
#include "sixtracklib/cuda/argument.hpp"
#include "sixtracklib/cuda/wrappers/controller_wrappers.h"

namespace st = SIXTRL_CXX_NAMESPACE;

namespace SIXTRL_CXX_NAMESPACE
{
    CudaController::CudaController( char const* config_str ) :
        st::NodeControllerBase( st::ARCHITECTURE_CUDA,
            SIXTRL_ARCHITECTURE_CUDA_STR, config_str ),
        m_cuda_debug_register( nullptr )
    {
        this->doInitCudaController();

        CudaController::status_t status = this->doInitCudaDebugRegister();

        if( status == st::ARCH_STATUS_SUCCESS )
        {
            status = this->doInitAllCudaNodes();
        }

        if( status == st::ARCH_STATUS_SUCCESS )
        {
            CudaController::node_index_t node_index_to_select =
                this->doGetNodeIndexFromConfigString( config_str );

            if( ( node_index_to_select == CudaController::UNDEFINED_INDEX ) &&
                ( this->usesAutoSelect() ) && ( this->hasDefaultNode() ) )
            {
                node_index_to_select = this->defaultNodeIndex();
            }

            status = this->doSelectNodeCudaImpl( node_index_to_select );

            if( ( status == st::ARCH_STATUS_SUCCESS ) &&
                ( this->hasSelectedNode() ) )
            {
                this->doEnableCudaController();
            }
        }
    }

    CudaController::CudaController(
        CudaController::cuda_device_index_t const cuda_node_index,
        char const* SIXTRL_RESTRICT config_str ) :
        st::NodeControllerBase( st::ARCHITECTURE_CUDA,
            SIXTRL_ARCHITECTURE_CUDA_STR, config_str ),
        m_cuda_debug_register( nullptr )
    {
        this->doInitCudaController();
        CudaController::status_t status = this->doInitCudaDebugRegister();

        if( status == st::ARCH_STATUS_SUCCESS )
        {
            status = this->doInitAllCudaNodes();
        }

        if( status == st::ARCH_STATUS_SUCCESS )
        {
            CudaController::node_index_t node_index_to_select =
                this->doFindAvailableNodesByCudaDeviceIndex( cuda_node_index );

            if( node_index_to_select == CudaController::UNDEFINED_INDEX )
            {
                node_index_to_select = this->doGetNodeIndexFromConfigString(
                    config_str );
            }

            if( ( node_index_to_select == CudaController::UNDEFINED_INDEX ) &&
                ( this->usesAutoSelect() ) && ( this->hasDefaultNode() ) )
            {
                node_index_to_select = this->defaultNodeIndex();
            }

            status = this->doSelectNodeCudaImpl( node_index_to_select );

            if( ( status == st::ARCH_STATUS_SUCCESS ) &&
                ( this->hasSelectedNode() ) )
            {
                this->doEnableCudaController();
            }
        }
    }

    CudaController::CudaController(
        CudaController::node_id_t const& SIXTRL_RESTRICT_REF node_id,
        char const* SIXTRL_RESTRICT config_str ) :
        st::NodeControllerBase( st::ARCHITECTURE_CUDA,
            SIXTRL_ARCHITECTURE_CUDA_STR, config_str ),
        m_cuda_debug_register( nullptr )
    {
        this->doInitCudaController();

        CudaController::status_t status = this->doInitCudaDebugRegister();

        if( status == st::ARCH_STATUS_SUCCESS )
        {
            status = this->doInitAllCudaNodes();
        }

        if( status == st::ARCH_STATUS_SUCCESS )
        {
            CudaController::node_index_t node_index_to_select =
                this->doFindAvailableNodesIndex(
                    node_id.platformId(), node_id.deviceId() );

            if( node_index_to_select == CudaController::UNDEFINED_INDEX )
            {
                node_index_to_select = this->doGetNodeIndexFromConfigString(
                    config_str );
            }

            if( ( node_index_to_select == CudaController::UNDEFINED_INDEX ) &&
                ( this->usesAutoSelect() ) && ( this->hasDefaultNode() ) )
            {
                node_index_to_select = this->defaultNodeIndex();
            }

           status = this->doSelectNodeCudaImpl( node_index_to_select );

            if( ( status == st::ARCH_STATUS_SUCCESS ) &&
                ( this->hasSelectedNode() ) )
            {
                this->doEnableCudaController();
            }
        }
    }

    CudaController::CudaController(
        CudaController::platform_id_t const platform_id,
        CudaController::device_id_t const device_id,
        char const* SIXTRL_RESTRICT config_str ) :
        st::NodeControllerBase( st::ARCHITECTURE_CUDA,
            SIXTRL_ARCHITECTURE_CUDA_STR, config_str ),
        m_cuda_debug_register( nullptr )
    {
        this->doInitCudaController();

        CudaController::status_t status = this->doInitCudaDebugRegister();

        if( status == st::ARCH_STATUS_SUCCESS )
        {
            status = this->doInitAllCudaNodes();
        }

        if( status == st::ARCH_STATUS_SUCCESS )
        {
            CudaController::node_index_t node_index_to_select =
                this->doFindAvailableNodesIndex( platform_id, device_id );

            if( node_index_to_select == CudaController::UNDEFINED_INDEX )
            {
                node_index_to_select = this->doGetNodeIndexFromConfigString(
                    config_str );
            }

            if( ( node_index_to_select == CudaController::UNDEFINED_INDEX ) &&
                ( this->usesAutoSelect() ) && ( this->hasDefaultNode() ) )
            {
                node_index_to_select = this->defaultNodeIndex();
            }

            status = this->doSelectNodeCudaImpl( node_index_to_select );

            if( ( status == st::ARCH_STATUS_SUCCESS ) &&
                ( this->hasSelectedNode() ) )
            {
                this->doEnableCudaController();
            }
        }
    }

    /* --------------------------------------------------------------------- */

    CudaController::CudaController( std::string const& config_str  ) :
        st::NodeControllerBase( st::ARCHITECTURE_CUDA,
            SIXTRL_ARCHITECTURE_CUDA_STR, config_str.c_str() ),
        m_cuda_debug_register( nullptr )
    {
        this->doInitCudaController();

        CudaController::status_t status = this->doInitCudaDebugRegister();

        if( status == st::ARCH_STATUS_SUCCESS )
        {
            status = this->doInitAllCudaNodes();
        }

        if( status == st::ARCH_STATUS_SUCCESS )
        {
            CudaController::node_index_t node_index_to_select =
                this->doGetNodeIndexFromConfigString( config_str.c_str() );

            if( ( node_index_to_select == CudaController::UNDEFINED_INDEX ) &&
                ( this->usesAutoSelect() ) && ( this->hasDefaultNode() ) )
            {
                node_index_to_select = this->defaultNodeIndex();
            }

            status = this->doSelectNodeCudaImpl( node_index_to_select );

            if( ( status == st::ARCH_STATUS_SUCCESS ) &&
                ( this->hasSelectedNode() ) )
            {
                this->doEnableCudaController();
            }
        }
    }

    CudaController::CudaController(
        CudaController::cuda_device_index_t const cuda_node_index,
        std::string const& SIXTRL_RESTRICT_REF config_str ) :
        st::NodeControllerBase( st::ARCHITECTURE_CUDA,
            SIXTRL_ARCHITECTURE_CUDA_STR, config_str.c_str() ),
        m_cuda_debug_register( nullptr )
    {
        this->doInitCudaController();

        CudaController::status_t status = this->doInitCudaDebugRegister();

        if( status == st::ARCH_STATUS_SUCCESS )
        {
            status = this->doInitAllCudaNodes();
        }

        if( status == st::ARCH_STATUS_SUCCESS )
        {
            CudaController::node_index_t node_index_to_select =
                this->doFindAvailableNodesByCudaDeviceIndex( cuda_node_index );

            if( node_index_to_select == CudaController::UNDEFINED_INDEX )
            {
                node_index_to_select = this->doGetNodeIndexFromConfigString(
                    config_str.c_str() );
            }

            if( ( node_index_to_select == CudaController::UNDEFINED_INDEX ) &&
                ( this->usesAutoSelect() ) && ( this->hasDefaultNode() ) )
            {
                node_index_to_select = this->defaultNodeIndex();
            }

            status = this->doSelectNodeCudaImpl( node_index_to_select );

            if( ( status == st::ARCH_STATUS_SUCCESS ) &&
                ( this->hasSelectedNode() ) )
            {
                this->doEnableCudaController();
            }
        }
    }

    CudaController::CudaController(
        CudaController::node_id_t const& SIXTRL_RESTRICT_REF node_id,
        std::string const& SIXTRL_RESTRICT_REF config_str ) :
        st::NodeControllerBase( st::ARCHITECTURE_CUDA,
            SIXTRL_ARCHITECTURE_CUDA_STR, config_str.c_str() ),
        m_cuda_debug_register( nullptr )
    {
        this->doInitCudaController();

        CudaController::status_t status = this->doInitCudaDebugRegister();

        if( status == st::ARCH_STATUS_SUCCESS )
        {
            status = this->doInitAllCudaNodes();
        }

        if( status == st::ARCH_STATUS_SUCCESS )
        {
            CudaController::node_index_t node_index_to_select =
                this->doFindAvailableNodesIndex(
                    node_id.platformId(), node_id.deviceId() );

            if( node_index_to_select == CudaController::UNDEFINED_INDEX )
            {
                node_index_to_select = this->doGetNodeIndexFromConfigString(
                    config_str.c_str() );
            }

            if( ( node_index_to_select == CudaController::UNDEFINED_INDEX ) &&
                ( this->usesAutoSelect() ) && ( this->hasDefaultNode() ) )
            {
                node_index_to_select = this->defaultNodeIndex();
            }

            status = this->doSelectNodeCudaImpl( node_index_to_select );

            if( ( status == st::ARCH_STATUS_SUCCESS ) &&
                ( this->hasSelectedNode() ) )
            {
                this->doEnableCudaController();
            }
        }
    }

    CudaController::CudaController(
        CudaController::platform_id_t const platform_id,
        CudaController::device_id_t const device_id,
        std::string const& SIXTRL_RESTRICT_REF config_str ) :
        st::NodeControllerBase( st::ARCHITECTURE_CUDA,
            SIXTRL_ARCHITECTURE_CUDA_STR, config_str.c_str() ),
        m_cuda_debug_register( nullptr )
    {
        this->doInitCudaController();

        CudaController::status_t status = this->doInitCudaDebugRegister();

        if( status == st::ARCH_STATUS_SUCCESS )
        {
            status = this->doInitAllCudaNodes();
        }

        if( status == st::ARCH_STATUS_SUCCESS )
        {
            CudaController::node_index_t node_index_to_select =
                this->doFindAvailableNodesIndex( platform_id, device_id );

            if( node_index_to_select == CudaController::UNDEFINED_INDEX )
            {
                node_index_to_select = this->doGetNodeIndexFromConfigString(
                    config_str.c_str() );
            }

            if( ( node_index_to_select == CudaController::UNDEFINED_INDEX ) &&
                ( this->usesAutoSelect() ) && ( this->hasDefaultNode() ) )
            {
                node_index_to_select = this->defaultNodeIndex();
            }

            status = this->doSelectNodeCudaImpl( node_index_to_select );

            if( ( status == st::ARCH_STATUS_SUCCESS ) &&
                ( this->hasSelectedNode() ) )
            {
                this->doEnableCudaController();
            }
        }
    }

    /* --------------------------------------------------------------------- */

    CudaController::~CudaController() SIXTRL_NOEXCEPT
    {
        if( this->m_cuda_debug_register != nullptr )
        {
            cudaError_t const err = ::cudaFree( this->m_cuda_debug_register );
            SIXTRL_ASSERT( err == ::cudaSuccess );
            this->m_cuda_debug_register = nullptr;
        }
    }

    /* --------------------------------------------------------------------- */

    CudaController::node_info_t const* CudaController::ptrNodeInfo(
        CudaController::size_type const index ) const SIXTRL_NOEXCEPT
    {
        using node_info_t = CudaController::node_info_t;
        auto node_info_base = this->ptrNodeInfoBase( index );

        return ( node_info_base != nullptr )
            ? node_info_base->asDerivedNodeInfo< node_info_t >(
                st::ARCHITECTURE_CUDA ) : nullptr;
    }

    CudaController::node_info_t const* CudaController::ptrNodeInfo(
        CudaController::platform_id_t const platform_idx,
        CudaController::device_id_t const device_idx ) const SIXTRL_NOEXCEPT
    {
        return this->ptrNodeInfo( this->doFindAvailableNodesIndex(
            platform_idx, device_idx ) );
    }

    CudaController::node_info_t const* CudaController::ptrNodeInfo(
        CudaController::node_id_t const& node_id ) const SIXTRL_NOEXCEPT
    {
        return this->ptrNodeInfo( this->doFindAvailableNodesIndex(
            node_id.platformId(), node_id.deviceId() ) );
    }

    CudaController::node_info_t const* CudaController::ptrNodeInfo(
        char const* SIXTRL_RESTRICT node_id_str ) const SIXTRL_NOEXCEPT
    {
        return this->ptrNodeInfo( this->doFindAvailableNodesIndex(
            node_id_str ) );
    }

    CudaController::node_info_t const* CudaController::ptrNodeInfo(
        std::string const& SIXTRL_RESTRICT_REF
            node_id_str ) const SIXTRL_NOEXCEPT
    {
        return this->ptrNodeInfo( this->doFindAvailableNodesIndex(
            node_id_str.c_str() ) );
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    CudaController::status_t CudaController::selectNodeByCudaIndex(
        CudaController::cuda_device_index_t const cuda_device_index )
    {
        return this->selectNode(
            this->doFindAvailableNodesByCudaDeviceIndex( cuda_device_index ) );
    }

    CudaController::status_t CudaController::selectNodeByPciBusId(
        std::string const& SIXTRL_RESTRICT_REF pci_bus_id )
    {
        return this->selectNode(
            this->doFindAvailableNodesByPciBusId( pci_bus_id.c_str() ) );
    }

    CudaController::status_t CudaController::selectNodeByPciBusId(
        char const* SIXTRL_RESTRICT pci_bus_id )
    {
        return this->selectNode(
            this->doFindAvailableNodesByPciBusId( pci_bus_id ) );
    }

    /* ===================================================================== */

    CudaController::status_t CudaController::sendMemory(
        CudaController::cuda_arg_buffer_t SIXTRL_RESTRICT destination,
        void const* SIXTRL_RESTRICT source,
        CudaController::size_type const source_length )
    {
        return CudaController::PerformSendOperation(
            destination, source, source_length );
    }

    CudaController::status_t CudaController::receiveMemory(
        void* SIXTRL_RESTRICT destination,
        CudaController::cuda_const_arg_buffer_t SIXTRL_RESTRICT source,
        CudaController::size_type const source_length )
    {
        return CudaController::PerformReceiveOperation(
            destination, source_length, source, source_length );
    }

    CudaController::status_t
    CudaController::remap(
        cuda_arg_buffer_t SIXTRL_RESTRICT managed_buffer_begin,
        size_type const slot_size )
    {
        return this->doRemapCObjectsBufferDirectly(
            managed_buffer_begin, slot_size );
    }

    bool CudaController::isRemapped(
        CudaController::cuda_arg_buffer_t SIXTRL_RESTRICT managed_buffer_begin,
        CudaController::size_type const slot_size )
    {
        CudaController::status_t status = st::ARCH_STATUS_GENERAL_FAILURE;
        bool const is_remapped = this->doCheckIsCobjectsBufferRemappedDirectly(
            &status, managed_buffer_begin, slot_size );

        if( status != st::ARCH_STATUS_SUCCESS )
        {
            throw std::runtime_error(
                "unable to perform isRemapped() check on managd cuda buffer" );
        }

        return is_remapped;
    }

    /* ===================================================================== */

    CudaController::kernel_config_t const*
    CudaController::ptrKernelConfig(
        kernel_id_t const kernel_id ) const SIXTRL_NOEXCEPT
    {
        auto ptr_config = this->ptrKernelConfigBase( kernel_id );

        return ( ptr_config != nullptr )
            ? ptr_config->asDerivedKernelConfig<
                CudaController::kernel_config_t >( this->archId() )
            : nullptr;
    }

    CudaController::kernel_config_t const*
    CudaController::ptrKernelConfig( std::string const&
        SIXTRL_RESTRICT_REF kernel_name ) const SIXTRL_NOEXCEPT
    {
        return this->ptrKernelConfig(
            this->doFindKernelConfigByName( kernel_name.c_str() ) );
    }

    CudaController::kernel_config_t const*
    CudaController::ptrKernelConfig(
        char const* SIXTRL_RESTRICT kernel_name ) const SIXTRL_NOEXCEPT
    {
        return this->ptrKernelConfig(
            this->doFindKernelConfigByName( kernel_name ) );
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    CudaController::kernel_config_t* CudaController::ptrKernelConfig(
        kernel_id_t const kernel_id ) SIXTRL_NOEXCEPT
    {
        return const_cast< CudaController::kernel_config_t* >(
            static_cast< CudaController const& >(
                *this ).ptrKernelConfig( kernel_id ) );
    }

    CudaController::kernel_config_t* CudaController::ptrKernelConfig(
        std::string const& SIXTRL_RESTRICT_REF kernel_name ) SIXTRL_NOEXCEPT
    {
        return this->ptrKernelConfig(
            this->doFindKernelConfigByName( kernel_name.c_str() ) );
    }

    CudaController::kernel_config_t* CudaController::ptrKernelConfig(
        char const* SIXTRL_RESTRICT kernel_name ) SIXTRL_NOEXCEPT
    {
        return this->ptrKernelConfig(
            this->doFindKernelConfigByName( kernel_name ) );
    }

    /* --------------------------------------------------------------------- */

    CudaController::kernel_id_t CudaController::addCudaKernelConfig(
        CudaController::kernel_config_t const& kernel_config )
    {
        CudaController::ptr_cuda_kernel_config_t ptr_kernel_conf(
            new kernel_config_t( kernel_config ) );

        return this->doAppendCudaKernelConfig( std::move( ptr_kernel_conf ) );
    }

    CudaController::kernel_id_t CudaController::addCudaKernelConfig(
        std::string const& kernel_name,
        CudaController::size_type const num_arguments,
        CudaController::size_type const grid_dim ,
        CudaController::size_type const shared_mem_per_block,
        CudaController::size_type const max_blocks_limit,
        char const* SIXTRL_RESTRICT config_str )
    {
        return this->addCudaKernelConfig( kernel_name.c_str(), num_arguments,
            grid_dim, shared_mem_per_block, max_blocks_limit, config_str );
    }

    CudaController::kernel_id_t CudaController::addCudaKernelConfig(
        char const* SIXTRL_RESTRICT kernel_name,
        CudaController::size_type const num_arguments,
        CudaController::size_type const grid_dim,
        CudaController::size_type const shared_mem_per_block,
        CudaController::size_type const max_blocks_limit,
        char const* SIXTRL_RESTRICT config_str )
    {
        using kernel_config_t = CudaController::kernel_config_t;

        CudaController::ptr_cuda_kernel_config_t ptr_kernel_conf(
            new kernel_config_t( grid_dim, grid_dim, shared_mem_per_block,
                max_blocks_limit, kernel_config_t::DEFAULT_WARP_SIZE,
                    config_str ) );

        ptr_kernel_conf->setName( kernel_name );

        if( ptr_kernel_conf.get() != nullptr )
        {
            ptr_kernel_conf->setName( kernel_name );
        }

        return this->doAppendCudaKernelConfig( std::move( ptr_kernel_conf ) );
    }

    CudaController::kernel_id_t CudaController::doAppendCudaKernelConfig(
        CudaController::ptr_cuda_kernel_config_t&&
            ptr_kernel_conf ) SIXTRL_NOEXCEPT
    {
        using _this_t = CudaController;
        using size_t = _this_t::size_type;
        using kernel_id_t = _this_t::kernel_id_t;
        using node_info_t = _this_t::node_info_t;
        using kernel_config_t = _this_t::kernel_config_t;

        kernel_id_t kernel_id = _this_t::ILLEGAL_KERNEL_ID;

        if( ( this->hasSelectedNode() ) &&
            ( ptr_kernel_conf.get() != nullptr ) )
        {
            node_info_t const* ptr_node_info = this->ptrNodeInfo(
                this->selectedNodeIndex() );

            if( ptr_node_info == nullptr )
            {
                return kernel_id;
            }

            size_t const warp_size =
                ptr_node_info->cudaDeviceProperties().warpSize;

            if( warp_size != ptr_kernel_conf->warpSize() )
            {
                ptr_kernel_conf->setWarpSize( warp_size );
            }

            if( ptr_kernel_conf->needsUpdate() )
            {
                ptr_kernel_conf->update();
            }

            kernel_id = this->doAppendKernelConfig(
                std::move( ptr_kernel_conf ) );
        }

        return kernel_id;
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    CudaController::status_t CudaController::doSend(
        CudaController::ptr_arg_base_t SIXTRL_RESTRICT dest_arg,
        const void *const SIXTRL_RESTRICT source,
        CudaController::size_type const source_length )
    {
        using _this_t    = st::CudaController;
        using status_t   = _this_t::status_t;
        using size_t     = _this_t::size_type;
        using cuda_arg_t = st::CudaArgument;
        using cuda_arg_buffer_t = cuda_arg_t::cuda_arg_buffer_t;

        status_t status = status_t{ -1 };

        SIXTRL_ASSERT( this->readyForSend() );
        SIXTRL_ASSERT( dest_arg != nullptr );
        SIXTRL_ASSERT( source   != nullptr );
        SIXTRL_ASSERT( source_length > size_t{ 0 } );
        SIXTRL_ASSERT( dest_arg->capacity() >= source_length );

        if( dest_arg->hasArgumentBuffer() )
        {
            cuda_arg_t* cuda_arg = dest_arg->asDerivedArgument< cuda_arg_t >(
                st::ARCHITECTURE_CUDA );

            if( ( cuda_arg != nullptr ) && ( cuda_arg->hasCudaArgBuffer() ) )
            {
                status = _this_t::PerformSendOperation(
                    cuda_arg->cudaArgBuffer(), source, source_length );
            }
        }

        return status;
    }

    CudaController::status_t CudaController::doReceive(
        void* SIXTRL_RESTRICT destination,
        CudaController::size_type const dest_capacity,
        CudaController::ptr_arg_base_t SIXTRL_RESTRICT src_arg )
    {
        using _this_t    = CudaController;
        using status_t   = _this_t::status_t;
        using size_t     = _this_t::size_type;
        using cuda_arg_t = st::CudaArgument;
        using cuda_arg_buffer_t = cuda_arg_t::cuda_arg_buffer_t;

        status_t status = status_t{ -1 };

        SIXTRL_ASSERT( this->readyForReceive() );
        SIXTRL_ASSERT( destination != nullptr );
        SIXTRL_ASSERT( src_arg != nullptr );
        SIXTRL_ASSERT( src_arg->size() > size_t{ 0 } );
        SIXTRL_ASSERT( dest_capacity >= src_arg->size() );

        if( src_arg->hasArgumentBuffer() )
        {
            cuda_arg_t* cuda_arg = src_arg->asDerivedArgument< cuda_arg_t >(
                st::ARCHITECTURE_CUDA );

            if( ( cuda_arg != nullptr ) && ( cuda_arg->hasCudaArgBuffer() ) )
            {
                status = _this_t::PerformReceiveOperation( destination,
                    dest_capacity, cuda_arg->cudaArgBuffer(), src_arg->size() );
            }
        }

        return status;
    }

    CudaController::status_t CudaController::doSetDebugRegister(
        CudaController::debug_register_t const debug_register )
    {
        using _base_t = st::CudaController::_base_controller_t;
        using _this_t = st::CudaController;
        using status_t = _this_t::status_t;

        status_t status =
            this->doSetDebugRegisterCudaBaseImpl( debug_register );

        if( status == st::ARCH_STATUS_SUCCESS )
        {
            status = _base_t::doSetDebugRegister( debug_register );
        }

        return status;
    }

    CudaController::status_t CudaController::doFetchDebugRegister(
        CudaController::debug_register_t* SIXTRL_RESTRICT ptr_debug_register )
    {
        using _base_t = st::CudaController::_base_controller_t;
        using _this_t = st::CudaController;
        using status_t = _this_t::status_t;

        status_t status =
            this->doFetchDebugRegisterCudaBaseImpl( ptr_debug_register );

        if( status == st::ARCH_STATUS_SUCCESS )
        {
            status = _base_t::doFetchDebugRegister( ptr_debug_register );
        }

        return status;
    }

    CudaController::status_t CudaController::doRemapCObjectsBufferArg(
        CudaController::ptr_arg_base_t SIXTRL_RESTRICT buffer_arg )
    {
        using   status_t = st::CudaController::status_t;
        using     size_t = st::CudaController::size_type;
        using cuda_arg_t = st::CudaArgument;

        status_t status = st::ARCH_STATUS_GENERAL_FAILURE;

        if( ( buffer_arg == nullptr ) || ( !this->readyForRunningKernel() ) )
        {
            return status;
        }

        cuda_arg_t* buffer_cuda_arg = buffer_arg->asDerivedArgument<
            st::CudaArgument >( this->archId() );

        if( ( buffer_cuda_arg != nullptr ) &&
            ( buffer_cuda_arg->usesCObjectsBuffer() ) &&
            ( buffer_cuda_arg->hasCudaArgBuffer() ) &&
            ( buffer_cuda_arg->capacity() > size_t{ 0 } ) )
        {
            status = this->doRemapCObjectsBufferDirectly(
                buffer_cuda_arg->cudaArgBuffer(),
                buffer_cuda_arg->cobjectsBufferSlotSize() );
        }

        return status;
    }

    bool CudaController::doIsCObjectsBufferArgRemapped(
        CudaController::ptr_arg_base_t SIXTRL_RESTRICT buffer_arg,
        CudaController::status_t* SIXTRL_RESTRICT ptr_status )
    {
        using   status_t = st::CudaController::status_t;
        using     size_t = st::CudaController::size_type;
        using cuda_arg_t = st::CudaArgument;

        bool is_remapped = false;

        if( ( buffer_arg == nullptr ) || ( !this->readyForRunningKernel() ) )
        {
            if( ptr_status != nullptr )
            {
                *ptr_status = st::ARCH_STATUS_GENERAL_FAILURE;
            }

            return is_remapped;
        }

        cuda_arg_t* buffer_cuda_arg = buffer_arg->asDerivedArgument<
            st::CudaArgument >( this->archId() );

        if( ( buffer_cuda_arg != nullptr ) &&
            ( buffer_cuda_arg->usesCObjectsBuffer() ) &&
            ( buffer_cuda_arg->hasCudaArgBuffer() ) &&
            ( buffer_cuda_arg->capacity() > size_t{ 0 } ) )
        {
            is_remapped = this->doCheckIsCobjectsBufferRemappedDirectly(
                ptr_status, buffer_cuda_arg->cudaArgBuffer(),
                buffer_cuda_arg->cobjectsBufferSlotSize() );
        }

        return is_remapped;
    }

    CudaController::status_t CudaController::doRemapCObjectsBufferDirectly(
        CudaController::cuda_arg_buffer_t SIXTRL_RESTRICT managed_buffer_begin,
        CudaController::size_type const slot_size )
    {
        using    _this_t        = CudaController;
        using kernel_config_t   = _this_t::kernel_config_t;
        using   status_t        = _this_t::status_t;
        using     size_t        = _this_t::size_type;
        using cuda_arg_buffer_t = _this_t::cuda_arg_buffer_t;

        status_t status = st::ARCH_STATUS_GENERAL_FAILURE;

        if( ( this->readyForRunningKernel() ) && ( slot_size > size_t{ 0 } ) &&
            ( managed_buffer_begin != nullptr ) )
        {
            bool const in_debug_mode = this->isInDebugMode();

            kernel_config_t const* kernel_config = this->ptrKernelConfig(
                ( in_debug_mode )
                    ? this->remapCObjectBufferDebugKernelId()
                    : this->remapCObjectBufferKernelId() );

            if( kernel_config != nullptr )
            {
                if( !in_debug_mode )
                {
                    ::NS(Buffer_remap_cuda_wrapper)(
                        kernel_config, managed_buffer_begin, slot_size );

                    status = st::ARCH_STATUS_SUCCESS;
                }
                else if( this->doGetPtrCudaDebugRegister() != nullptr )
                {
                    status = this->prepareDebugRegisterForUse();

                    if( status == st::ARCH_STATUS_SUCCESS )
                    {
                        ::NS(Buffer_remap_cuda_debug_wrapper)( kernel_config,
                            managed_buffer_begin, slot_size,
                                this->doGetPtrCudaDebugRegister() );

                        status = this->evaluateDebugRegisterAfterUse();
                    }
                }
            }
        }

        return status;
    }

    bool CudaController::doCheckIsCobjectsBufferRemappedDirectly(
        CudaController::status_t* SIXTRL_RESTRICT ptr_status,
        CudaController::cuda_arg_buffer_t SIXTRL_RESTRICT managed_buffer_begin,
        CudaController::size_type const slot_size )
    {
        using    _this_t        = CudaController;
        using   status_t        = _this_t::status_t;
        using     size_t        = _this_t::size_type;
        using cuda_arg_buffer_t = _this_t::cuda_arg_buffer_t;

        bool is_remapped = false;

        if( ( this->readyForRunningKernel() ) && ( slot_size > size_t{ 0 } ) &&
            ( managed_buffer_begin != nullptr ) )
        {
            is_remapped = ::NS(Buffer_is_remapped_cuda_wrapper)(
                managed_buffer_begin, slot_size,
                    this->doGetPtrCudaDebugRegister(), ptr_status );
        }

        return is_remapped;
    }

    CudaController::status_t CudaController::doSelectNode( CudaController::node_index_t const idx )
    {
        return this->doSelectNodeCudaImpl( idx );
    }

    CudaController::status_t CudaController::doChangeSelectedNode(
        CudaController::node_index_t const current_selected_node_idx,
        CudaController::node_index_t const new_selected_node_index )
    {
        CudaController::status_t status = st::ARCH_STATUS_GENERAL_FAILURE;

        auto ptr_node_info = this->ptrNodeInfo( new_selected_node_index );

        if( ( ptr_node_info != nullptr ) &&
            ( ptr_node_info->hasCudaDeviceIndex() ) &&
            ( this->isNodeAvailable( new_selected_node_index ) ) &&
            ( st::NodeControllerBase::doChangeSelectedNode(
                current_selected_node_idx, new_selected_node_index ) ) )
        {
            ::cudaError_t err = ::cudaSetDevice(
                ptr_node_info->cudaDeviceIndex() );

            if( err == ::cudaSuccess )
            {
                status = st::ARCH_STATUS_SUCCESS;
            }

            if( ( !this->canUnselectNode() ) &&
                ( ( status != st::ARCH_STATUS_SUCCESS ) ||
                  ( !this->hasSelectedNode() ) ) )
            {
                status = this->doSelectNode( current_selected_node_idx );
            }
        }

        return status;
    }

    CudaController::status_t CudaController::doInitCudaDebugRegister()
    {
        CudaController::status_t status = st::ARCH_STATUS_GENERAL_FAILURE;

        using debug_register_t = CudaController::debug_register_t;

        if( this->m_cuda_debug_register == nullptr )
        {
            ::cudaError_t err = ::cudaMalloc(
                &this->m_cuda_debug_register, sizeof( debug_register_t ) );

            if( err == ::cudaSuccess )
            {
                status = st::ARCH_STATUS_SUCCESS;
            }
            else if( this->m_cuda_debug_register != nullptr )
            {
                err = ::cudaFree( this->m_cuda_debug_register );
                SIXTRL_ASSERT( err == ::cudaSuccess );
                ( void )err;

                this->m_cuda_debug_register = nullptr;
            }
        }

        return status;
    }

    CudaController::node_index_t
    CudaController::doFindAvailableNodesByCudaDeviceIndex(
        CudaController::cuda_device_index_t const idx ) const SIXTRL_NOEXCEPT
    {
        using _this_t          = st::CudaController;
        using node_id_t        = _this_t::node_id_t;
        using node_info_base_t = _this_t::node_info_base_t;
        using node_info_t      = _this_t::node_info_t;
        using node_index_t     = _this_t::node_index_t;

        node_index_t found_node_index = node_id_t::UNDEFINED_INDEX;
        node_index_t const num_avail_nodes = this->numAvailableNodes();

        if( ( num_avail_nodes > node_index_t{ 0 } ) &&
            ( idx != node_info_t::ILLEGAL_DEV_INDEX ) )
        {
            node_index_t ii = node_index_t{ 0 };

            for( ; ii < num_avail_nodes ; ++ii )
            {
                node_info_base_t const* ptr_base = this->ptrNodeInfoBase( ii );

                node_info_t const* ptr_node_info = ( ptr_base != nullptr )
                    ? ptr_base->asDerivedNodeInfo< node_info_t >(
                        st::ARCHITECTURE_CUDA ) : nullptr;

                if( ( ptr_node_info != nullptr ) &&
                    ( ptr_node_info->hasCudaDeviceIndex() ) &&
                    ( ptr_node_info->cudaDeviceIndex() == idx ) )
                {
                    SIXTRL_ASSERT( ( !ptr_node_info->hasNodeIndex() ) ||
                        ( ptr_node_info->nodeIndex() == ii ) );

                    found_node_index = ii;
                    break;
                }
            }
        }

        return found_node_index;
    }

    CudaController::node_index_t
    CudaController::doFindAvailableNodesByPciBusId(
        char const* SIXTRL_RESTRICT pci_bus_id_str ) const SIXTRL_NOEXCEPT
    {
        using _this_t          = st::CudaController;
        using node_id_t        = _this_t::node_id_t;
        using node_info_base_t = _this_t::node_info_base_t;
        using node_info_t      = _this_t::node_info_t;
        using node_index_t     = _this_t::node_index_t;

        node_index_t found_node_index = node_id_t::UNDEFINED_INDEX;
        node_index_t const num_avail_nodes = this->numAvailableNodes();

        if( ( num_avail_nodes > node_index_t{ 0 } ) &&
            ( pci_bus_id_str != nullptr ) &&
            ( std::strlen( pci_bus_id_str ) > std::size_t{ 0 } ) )
        {
            node_index_t ii = node_index_t{ 0 };

            for( ; ii < num_avail_nodes ; ++ii )
            {
                node_info_base_t const* ptr_base = this->ptrNodeInfoBase( ii );

                node_info_t const* nodeinfo = ( ptr_base != nullptr )
                    ? ptr_base->asDerivedNodeInfo< node_info_t >(
                        st::ARCHITECTURE_CUDA ) : nullptr;

                if( ( nodeinfo != nullptr ) && ( nodeinfo->hasPciBusId() ) &&
                    ( nodeinfo->pciBusId().compare( pci_bus_id_str ) == 0 ) )
                {
                    SIXTRL_ASSERT( ( !nodeinfo->hasNodeIndex() ) ||
                                   (  nodeinfo->nodeIndex() == ii ) );

                    found_node_index = ii;
                    break;
                }
            }
        }

        return found_node_index;
    }

    CudaController::node_index_t
    CudaController::doGetNodeIndexFromConfigString(
            char const* SIXTRL_RESTRICT select_str )
    {
        CudaController::node_index_t found_node_index =
            CudaController::UNDEFINED_INDEX;

        if( select_str != nullptr )
        {
            found_node_index = this->doFindAvailableNodesIndex( select_str );

            if( found_node_index == CudaController::UNDEFINED_INDEX )
            {
                found_node_index =
                    this->doFindAvailableNodesByPciBusId( select_str );
            }
        }

        return found_node_index;
    }

    /* -------------------------------------------------------------------- */

    void CudaController::doInitCudaController()
    {
        this->doSetReadyForRunningKernelsFlag( false );
        this->doSetReadyForSendFlag( false );
        this->doSetReadyForReceiveFlag( false );

        this->doSetCanDirectlyChangeSelectedNodeFlag( true );
        this->doSetCanUnselectNodeFlag( false );
        this->doSetUseAutoSelectFlag( true );

        return;
    }

    CudaController::status_t CudaController::doInitAllCudaNodes()
    {
        CudaController::status_t status = st::ARCH_STATUS_GENERAL_FAILURE;

        using _this_t = CudaController;
        using node_index_t     = _this_t::node_index_t;
        using node_info_base_t = _this_t::node_info_base_t;
        using node_info_t      = _this_t::node_info_t;

        node_index_t const initial_num_avail_nodes = this->numAvailableNodes();

        bool first = true;

        int num_devices = int{ -1 };
        ::cudaError_t err = ::cudaGetDeviceCount( &num_devices );

        if( ( err == ::cudaSuccess ) && ( num_devices > int{ 0 } ) )
        {
            for( int cu_idx = int{ 0 } ; cu_idx < num_devices ; ++cu_idx )
            {
                ::cudaDeviceProp cu_properties;
                err = ::cudaGetDeviceProperties( &cu_properties, cu_idx );

                if( err != ::cudaSuccess )
                {
                    continue;
                }

                std::unique_ptr< node_info_t > ptr_node_info(
                    new node_info_t( cu_idx, cu_properties ) );

                if( ptr_node_info.get() == nullptr )
                {
                    continue;
                }

                SIXTRL_ASSERT( ptr_node_info->platformId() ==
                    _this_t::node_id_t::ILLEGAL_PLATFORM_ID );

                SIXTRL_ASSERT( ptr_node_info->deviceId() ==
                    _this_t::node_id_t::ILLEGAL_DEVICE_ID );

                /* Check if this node is already present -> use first the
                 * cuda device index and then the the PCI Bus ID for
                 * eliminating duplicates
                 *
                 * WARNING: This searches linearily (!) over all existing
                 * nodes. If the number of nodes gets high, this can impose
                 * some performance problems -> replace with a hash-table
                 * or something O(log(N)) / O(1) in such a case */

                if( _this_t::node_id_t::UNDEFINED_INDEX !=
                    this->doFindAvailableNodesByCudaDeviceIndex( cu_idx ) )
                {
                    continue;
                }

                if( ( ptr_node_info->hasPciBusId() ) &&
                    ( _this_t::node_id_t::UNDEFINED_INDEX !=
                        this->doFindAvailableNodesByPciBusId(
                          ptr_node_info->pciBusId().c_str() ) ) )
                {
                    continue;
                }

                ptr_node_info->setCudaDeviceIndex( cu_idx );

                node_index_t node_index = this->doAppendAvailableNodeInfoBase(
                    std::move( ptr_node_info ) );

                if( ( node_index != _this_t::UNDEFINED_INDEX ) && ( first ) )
                {
                    this->doSetDefaultNodeIndex( node_index );
                    first = false;
                }
            }

            if( this->numAvailableNodes() > initial_num_avail_nodes )
            {
                status = st::ARCH_STATUS_SUCCESS;
            }
        }

        return status;
    }

    void CudaController::doEnableCudaController()
    {
        SIXTRL_ASSERT( this->hasSelectedNode() );

        if( this->hasRemapCObjectBufferKernel() )
        {
            this->doSetReadyForSendFlag( true );
        }

        this->doSetReadyForReceiveFlag( true );
        this->doSetReadyForRunningKernelsFlag( true );
    }

    CudaController::cuda_arg_buffer_t
    CudaController::doGetPtrCudaDebugRegister() SIXTRL_NOEXCEPT
    {
        return this->m_cuda_debug_register;
    }

    CudaController::cuda_const_arg_buffer_t
    CudaController::doGetPtrCudaDebugRegister() const SIXTRL_NOEXCEPT
    {
        return this->m_cuda_debug_register;
    }

    CudaController::status_t CudaController::PerformSendOperation(
        ::NS(cuda_arg_buffer_t) SIXTRL_RESTRICT destination,
        void const* SIXTRL_RESTRICT src_begin,
        CudaController::size_type const src_length )
    {
        using status_t = CudaController::status_t;
        using size_t = CudaController::size_type;

        status_t status = st::ARCH_STATUS_GENERAL_FAILURE;

        if( ( destination != nullptr ) && ( src_begin != nullptr ) &&
            ( reinterpret_cast< std::uintptr_t >( destination ) !=
              reinterpret_cast< std::uintptr_t >( src_begin ) ) &&
            ( src_length > size_t{ 0 } ) )
        {
            ::cudaError_t const ret = ::cudaMemcpy(
                destination, src_begin, src_length, cudaMemcpyHostToDevice );

            if( ret == ::cudaSuccess )
            {
                status = st::ARCH_STATUS_SUCCESS;
            }
        }

        return status;
    }

    CudaController::status_t CudaController::PerformReceiveOperation(
        void* SIXTRL_RESTRICT destination,
        CudaController::size_type const destination_capacity,
        ::NS(cuda_const_arg_buffer_t) SIXTRL_RESTRICT src_begin,
        CudaController::size_type const src_length )
    {
        using status_t = CudaController::status_t;
        using size_t   = CudaController::size_type;

        status_t status = st::ARCH_STATUS_GENERAL_FAILURE;

        if( ( destination != nullptr ) && ( src_begin != nullptr ) &&
            ( reinterpret_cast< std::uintptr_t >( destination ) !=
              reinterpret_cast< std::uintptr_t >( src_begin ) ) &&
            ( destination_capacity >= src_length ) &&
            ( src_length > size_t{ 0 } ) )
        {
            ::cudaError_t const err = ::cudaMemcpy(
                destination, src_begin, src_length, cudaMemcpyDeviceToHost );

            if( err == ::cudaSuccess )
            {
                status = st::ARCH_STATUS_SUCCESS;
            }
        }

        return status;
    }

    CudaController::status_t CudaController::doSetDebugRegisterCudaBaseImpl(
        CudaController::debug_register_t const debug_register )
    {
        return CudaController::PerformSendOperation(
            this->doGetPtrCudaDebugRegister(),
                &debug_register, sizeof( debug_register ) );
    }

    CudaController::status_t CudaController::doFetchDebugRegisterCudaBaseImpl(
        CudaController::debug_register_t* SIXTRL_RESTRICT ptr_debug_register )
    {
        using debug_register_t = CudaController::debug_register_t;

        return CudaController::PerformReceiveOperation(
            ptr_debug_register, sizeof( debug_register_t ),
            this->doGetPtrCudaDebugRegister(),
            sizeof( debug_register_t ) );
    }

    CudaController::status_t CudaController::doSelectNodeCudaImpl(
            CudaController::node_index_t const node_index )
    {
        using _base_ctrl_t = st::NodeControllerBase;
        using _this_t = st::CudaController;
        using node_info_base_t = _this_t::node_info_base_t;
        using node_info_t = _this_t::node_info_t;
        using cuda_dev_index_t = node_info_t::cuda_dev_index_t;
        using size_t = _this_t::size_type;
        using kernel_id_t = _this_t::kernel_id_t;
        using kernel_config_t = _this_t::kernel_config_t;

        CudaController::status_t status = st::ARCH_STATUS_GENERAL_FAILURE;

        node_info_base_t* node_base = this->doGetPtrNodeInfoBase( node_index );

        node_info_t* ptr_node_info = ( node_base != nullptr )
            ? node_base->asDerivedNodeInfo< node_info_t >(
                st::ARCHITECTURE_CUDA ) : nullptr;

        if( ( ptr_node_info != nullptr ) &&
            ( ptr_node_info->hasCudaDeviceIndex() ) )
        {
            cuda_dev_index_t cuda_dev_index = ptr_node_info->cudaDeviceIndex();

            if( cuda_dev_index != node_info_t::ILLEGAL_DEV_INDEX )
            {
                ::cudaError_t const err = ::cudaSetDevice( cuda_dev_index );

                if( err == ::cudaSuccess )
                {
                    status = _base_ctrl_t::doSelectNode( node_index );
                }
            }
        }

        if( ( status == st::ARCH_STATUS_SUCCESS ) &&
            ( this->hasSelectedNode() ) )
        {
            std::string kernel_name( size_t{ 64 }, '\0' );

            kernel_id_t kernel_id = _this_t::ILLEGAL_KERNEL_ID;

            kernel_name.clear();
            kernel_name = SIXTRL_C99_NAMESPACE_PREFIX_STR;
            kernel_name += "Buffer_remap_cuda_wrapper";

            ptr_cuda_kernel_config_t ptr_remap_kernel( new kernel_config_t(
                kernel_name, size_t{ 1 } ) );

            SIXTRL_ASSERT( ptr_remap_kernel.get() != nullptr );
            bool success = ptr_remap_kernel->setNumWorkItems( size_t{ 1 } );
            success &= ptr_remap_kernel->setWorkGroupSizes( size_t{ 1 } );
            success &= ptr_remap_kernel->update();
            success &= !ptr_remap_kernel->needsUpdate();

            if( success )
            {
                kernel_id = this->doAppendCudaKernelConfig(
                    std::move( ptr_remap_kernel ) );
            }
            else
            {
                status = st::ARCH_STATUS_GENERAL_FAILURE;
            }

            this->setRemapCObjectBufferKernelId( kernel_id );

            kernel_id = _this_t::ILLEGAL_KERNEL_ID;
            kernel_name.clear();
            kernel_name  = SIXTRL_C99_NAMESPACE_PREFIX_STR;
            kernel_name += "Buffer_remap_cuda_debug_wrapper";

            ptr_cuda_kernel_config_t ptr_debug_remap_kernel(
                new kernel_config_t( kernel_name, size_t{ 2 } ) );

            SIXTRL_ASSERT( ptr_debug_remap_kernel.get() != nullptr );

            success = ptr_debug_remap_kernel->setNumWorkItems( size_t{ 1 } );
            success &= ptr_debug_remap_kernel->setWorkGroupSizes( size_t{ 1 } );
            success &= ptr_debug_remap_kernel->update();
            success &= !ptr_debug_remap_kernel->needsUpdate();

            if( success )
            {
                kernel_id = this->doAppendCudaKernelConfig(
                    std::move( ptr_debug_remap_kernel ) );
            }
            else
            {
                status = st::ARCH_STATUS_GENERAL_FAILURE;
            }

            this->setRemapCObjectBufferDebugKernelId( kernel_id );
        }

        return status;
    }
}

#endif /* C++, Host */

/* end: sixtracklib/cuda/internal/controller.cpp */
