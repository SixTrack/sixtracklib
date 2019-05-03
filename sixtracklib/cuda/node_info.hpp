#ifndef SIXTRACKLIB_CUDA_NODE_INFO_HPP__
#define SIXTRACKLIB_CUDA_NODE_INFO_HPP__

#if !defined( SIXTRKL_NO_SYSTEM_INCLUDES )
    #if defined( __cplusplus ) && !defined( _GPUCODE ) && \
       !defined( __CUDA_ARCH__ )
        #include <cstddef>
        #include <cstdlib>
        #include <cstring>
        #include <cstdio>

        #include <cuda.h>
        #include <cuda_runtime.h>
    #endif /* C++, Host */
#endif /* !defined( SIXTRKL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/control/definitions.h"
    #include "sixtracklib/common/control/node_info.hpp"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus ) && !defined( _GPUCODE ) && !defined( __CUDA_ARCH__ )

namespace SIXTRL_CXX_NAMESPACE
{
    class CudaNodeInfo : public SIXTRL_CXX_NAMESPACE::NodeInfoBase
    {
        private:

        using _base_node_info_t = SIXTRL_CXX_NAMESPACE::NodeInfoBase;

        public:

        using node_id_t     = _base_node_info_t::node_id_t;
        using platform_id_t = _base_node_info_t::platform_id_t;
        using device_id_t   = _base_node_info_t::device_id_t;
        using node_index_t  = _base_node_info_t::node_index_t;
        using size_type     = _base_node_info_t::size_type;
        using arch_info_t   = _base_node_info_t::arch_info_t;
        using arch_id_t     = _base_node_info_t::arch_id_t;

        using cuda_dev_index_t = int;

        static SIXTRL_CONSTEXPR_OR_CONST cuda_dev_index_t
            ILLEGAL_DEV_INDEX = cuda_dev_index_t{ -1 };

        SIXTRL_HOST_FN explicit CudaNodeInfo(
            cuda_dev_index_t const cuda_dev_index = ILLEGAL_DEV_INDEX );

        SIXTRL_HOST_FN CudaNodeInfo(
            cuda_dev_index_t const cuda_dev_index,
            ::cudaDeviceProp const& cuda_device_properties );

        SIXTRL_HOST_FN CudaNodeInfo( CudaNodeInfo const& other ) = default;
        SIXTRL_HOST_FN CudaNodeInfo( CudaNodeInfo&& other ) = default;

        SIXTRL_HOST_FN CudaNodeInfo& operator=( CudaNodeInfo&& rhs ) = default;

        SIXTRL_HOST_FN CudaNodeInfo& operator=(
            CudaNodeInfo const& rhs ) = default;

        SIXTRL_HOST_FN virtual ~CudaNodeInfo() = default;

        SIXTRL_HOST_FN bool hasPciBusId() const SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN std::string const& pciBusId() const SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN char const* ptrPciBusIdStr() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN ::cudaDeviceProp const*
        ptrCudaDeviceProperties() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN ::cudaDeviceProp* ptrCudaDeviceProperties() SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN bool hasCudaDeviceIndex() const SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN cuda_dev_index_t cudaDeviceIndex() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN void setCudaDeviceIndex(
            cuda_dev_index_t const index ) SIXTRL_NOEXCEPT;

        protected:

        SIXTRL_HOST_FN virtual void doPrintToOutputStream(
            std::ostream& SIXTRL_RESTRICT_REF output ) const override;

        private:
        ::cudaDeviceProp    m_cu_device_properties;
        cuda_dev_index_t    m_cu_device_index;
        std::string         m_cu_device_pci_bus_id;
    };
}

#else /* C++, Host */

typedef void NS(CudaNodeInfo);

#endif /* C++, Host */

#endif /* SIXTRACKLIB_CUDA_NODE_INFO_HPP__ */

/* end : sixtracklib/cuda/node_info.hpp */
