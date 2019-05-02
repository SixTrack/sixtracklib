#ifndef SIXTRACKLIB_CUDA_INTERNAL_CONTROL_BASE_HPP__
#define SIXTRACKLIB_CUDA_INTERNAL_CONTROL_BASE_HPP__

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
    #include "sixtracklib/common/control/context_base.hpp"
    #include "sixtracklib/common/control/context_base_with_nodes.hpp"
    #include "sixtracklib/common/buffer.h"
    #include "sixtracklib/cuda/node_info.hpp"

    #if defined( __cplusplus ) && !defined( _GPUCODE ) && \
       !defined( __CUDA_ARCH__ )
        #include "sixtracklib/common/buffer.hpp"
    #endif /* C++, Host */
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
        using cuda_device_index_t = node_info_t::cuda_device_index_t;

        using arch_id_t           = _base_controller_t::arch_id_t;
        using node_id_t           = _base_controller_t::node_id_t;
        using node_info_base_t    = _base_controller_t::node_info_base_t;
        using size_type           = _base_controller_t::size_type;
        using platform_id_t       = _base_controller_t::platform_id_t;
        using device_id_t         = _base_controller_t::device_id_t;
        using ptr_arg_base_t      = _base_controller_t::ptr_arg_base_t;
        using status_t            = _base_controller_t::status_t;
        using buffer_t            = _base_controller_t::buffer_t;
        using c_buffer_t          = _base_controller_t::c_buffer_t;

        SIXTRL_HOST_FN virtual ~CudaControllerBase() = default;

        /* ---------------------------------------------------------------- */

        SIXTRL_HOST_FN node_info_t const* ptrNodesInfo(
            size_type const index ) const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN node_info_t const* ptrNodesInfo(
            platform_id_t const platform_idx,
            device_id_t const device_idx ) const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN node_info_t const* ptrNodesInfo(
            node_id_t const& node_id ) const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN node_info_t const* ptrNodesInfo(
            char const* SIXTRL_RESTRICT node_id_str ) const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN node_info_t const* ptrNodesInfo(
            std::string const& SIXTRL_RESTRICT_REF node_id_str
            ) const SIXTRL_NOEXCEPT;

        /* ---------------------------------------------------------------- */

        SIXTRL_HOST_FN bool selectNodeByCudaIndex(
            cuda_device_index_t const cuda_device_index );

        SIXTRL_HOST_FN bool selectNodeByPciBusId(
            std::string const& SIXTRL_RESTRICT_REF pci_bus_id );

        SIXTRL_HOST_FN bool selectNodeByPciBusId(
            char const* SIXTRL_RESTRICT pci_bus_id );

        /* ---------------------------------------------------------------- */

        protected:

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
            doSelectNode( size_type node_index ) override;

        SIXTRL_HOST_FN bool doInitAllCudaNodes();
    };
}

extern "C" {

typedef SIXTRL_CXX_NAMESPACE::CudaControllerBase   NS(CudaControllerBase);

}

#else /* C++, Host */

typedef void NS(CudaControllerBase);

#endif /* C++, Host */

#endif /* SIXTRACKLIB_CUDA_INTERNAL_CONTROL_BASE_HPP__ */

/* end: sixtracklib/cuda/internal/context_base.hpp */
