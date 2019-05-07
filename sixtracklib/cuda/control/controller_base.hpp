#ifndef SIXTRACKLIB_CUDA_CONTROL_CONTROLLER_BASE_HPP__
#define SIXTRACKLIB_CUDA_CONTROL_CONTROLLER_BASE_HPP__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #if defined( __cplusplus ) && !defined( _GPUCODE ) && \
       !defined( __CUDA_ARCH__ )
        #include <cstddef>
        #include <cstdlib>
    #endif /* C++, Host */
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/control/definitions.h"
    #include "sixtracklib/common/control/controller_base.hpp"
    #include "sixtracklib/common/control/controller_on_nodes_base.hpp"
    #include "sixtracklib/common/buffer.h"

    #if defined( __cplusplus ) && !defined( _GPUCODE ) && \
       !defined( __CUDA_ARCH__ )
        #include "sixtracklib/common/buffer.hpp"
    #endif /* C++, Host */

    #include "sixtracklib/cuda/definitions.h"
    #include "sixtracklib/cuda/control/node_info.hpp"
    #include "sixtracklib/cuda/control/kernel_config.hpp"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus ) && !defined( _GPUCODE ) && !defined( __CUDA_ARCH__ )

namespace SIXTRL_CXX_NAMESPACE
{
    class CudaControllerBase :
        public SIXTRL_CXX_NAMESPACE::ControllerOnNodesBase
    {
        private:

        using _base_controller_t = SIXTRL_CXX_NAMESPACE::ControllerOnNodesBase;

        public:

        using node_info_t         = SIXTRL_CXX_NAMESPACE::CudaNodeInfo;
        using kernel_config_t     = SIXTRL_CXX_NAMESPACE::CudaKernelConfig;
        using cuda_device_index_t = node_info_t::cuda_dev_index_t;

        using arch_id_t           = _base_controller_t::arch_id_t;
        using node_id_t           = _base_controller_t::node_id_t;
        using node_info_base_t    = _base_controller_t::node_info_base_t;
        using size_type           = _base_controller_t::size_type;
        using platform_id_t       = _base_controller_t::platform_id_t;
        using device_id_t         = _base_controller_t::device_id_t;
        using node_index_t        = _base_controller_t::node_index_t;
        using ptr_arg_base_t      = _base_controller_t::ptr_arg_base_t;
        using status_t            = _base_controller_t::status_t;
        using kernel_id_t         = _base_controller_t::kernel_id_t;
        using buffer_t            = _base_controller_t::buffer_t;
        using c_buffer_t          = _base_controller_t::c_buffer_t;

        SIXTRL_HOST_FN virtual ~CudaControllerBase() = default;

        /* ---------------------------------------------------------------- */

        SIXTRL_HOST_FN node_info_t const* ptrNodeInfo(
            size_type const index ) const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN node_info_t const* ptrNodeInfo(
            platform_id_t const platform_idx,
            device_id_t const device_idx ) const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN node_info_t const* ptrNodeInfo(
            node_id_t const& node_id ) const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN node_info_t const* ptrNodeInfo(
            char const* SIXTRL_RESTRICT node_id_str ) const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN node_info_t const* ptrNodeInfo(
            std::string const& SIXTRL_RESTRICT_REF node_id_str
            ) const SIXTRL_NOEXCEPT;

        /* ---------------------------------------------------------------- */

        SIXTRL_HOST_FN bool selectNodeByCudaIndex(
            cuda_device_index_t const cuda_device_index );

        SIXTRL_HOST_FN bool selectNodeByPciBusId(
            std::string const& SIXTRL_RESTRICT_REF pci_bus_id );

        SIXTRL_HOST_FN bool selectNodeByPciBusId(
            char const* SIXTRL_RESTRICT pci_bus_id );

        /* ================================================================ */

        SIXTRL_HOST_FN kernel_config_t const*
        ptrKernelConfig( kernel_id_t const kernel_id ) const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN kernel_config_t const* ptrKernelConfig(
            std::string const& SIXTRL_RESTRICT_REF kernel_name
        ) const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN kernel_config_t const* ptrKernelConfig(
            char const* SIXTRL_RESTRICT kernel_name ) const SIXTRL_NOEXCEPT;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        SIXTRL_HOST_FN kernel_config_t*
        ptrKernelConfig( kernel_id_t const kernel_id ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN kernel_config_t* ptrKernelConfig(
            std::string const& SIXTRL_RESTRICT_REF kname ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN kernel_config_t* ptrKernelConfig(
            char const* SIXTRL_RESTRICT kernel_name ) SIXTRL_NOEXCEPT;

        /* ----------------------------------------------------------------- */

        SIXTRL_HOST_FN kernel_id_t addCudaKernelConfig(
            kernel_config_t const& kernel_config );

        SIXTRL_HOST_FN kernel_id_t addCudaKernelConfig(
            std::string const& kernel_name,
            size_type const num_arguments,
            size_type const grid_dim = size_type{ 1 },
            size_type const shared_mem_per_block = size_type{ 0 },
            size_type const max_blocks_limit = size_type{ 0 },
            char const* SIXTRL_RESTRICT config_str = nullptr );

        SIXTRL_HOST_FN kernel_id_t addCudaKernelConfig(
            char const* SIXTRL_RESTRICT kernel_name,
            size_type const num_arguments,
            size_type const grid_dim = size_type{ 1 },
            size_type const shared_mem_per_block = size_type{ 0 },
            size_type const max_blocks_limit = size_type{ 0 },
            char const* SIXTRL_RESTRICT config_str = nullptr );

        protected:

        using ptr_cuda_kernel_config_t = std::unique_ptr< kernel_config_t >;

        SIXTRL_HOST_FN explicit CudaControllerBase(
            char const* config_str = nullptr );

        SIXTRL_HOST_FN CudaControllerBase(
            CudaControllerBase const& other ) = default;

        SIXTRL_HOST_FN CudaControllerBase(
            CudaControllerBase&& other ) = default;

        SIXTRL_HOST_FN CudaControllerBase& operator=(
            CudaControllerBase const& rhs ) = default;

        SIXTRL_HOST_FN CudaControllerBase& operator=(
            CudaControllerBase&& rhs) = default;

        SIXTRL_HOST_FN virtual status_t doSend(
            ptr_arg_base_t SIXTRL_RESTRICT destination,
            const void *const SIXTRL_RESTRICT source,
            size_type const source_length ) override;

        SIXTRL_HOST_FN virtual status_t doReceive(
            void* SIXTRL_RESTRICT destination,
            size_type const dest_capacity,
            ptr_arg_base_t SIXTRL_RESTRICT source ) override;

        SIXTRL_HOST_FN virtual status_t doRemapSentCObjectsBuffer(
            ptr_arg_base_t SIXTRL_RESTRICT arg,
            size_type const arg_size ) override;

        SIXTRL_HOST_FN virtual bool
            doSelectNode( node_index_t const node_index ) override;

        SIXTRL_HOST_FN node_index_t doFindAvailableNodesByCudaDeviceIndex(
            cuda_device_index_t const cuda_device_index ) const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN node_index_t doFindAvailableNodesByPciBusId(
            char const* SIXTRL_RESTRICT pci_bus_id_str ) const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN bool doInitAllCudaNodes();

        SIXTRL_HOST_FN kernel_id_t doAppendCudaKernelConfig(
            ptr_cuda_kernel_config_t&& ptr_kernel_conf ) SIXTRL_NOEXCEPT;
    };
}

extern "C" {

typedef SIXTRL_CXX_NAMESPACE::CudaControllerBase   NS(CudaControllerBase);

}

#else /* C++, Host */

typedef void NS(CudaControllerBase);

#endif /* C++, Host */

#endif /* SIXTRACKLIB_CUDA_CONTROL_CONTROLLER_BASE_HPP__ */

/* end: sixtracklib/cuda/control/controller_base.hpp */
