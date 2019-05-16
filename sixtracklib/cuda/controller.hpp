#ifndef SIXTRACKLIB_CUDA_CONTROLLER_HPP__
#define SIXTRACKLIB_CUDA_CONTROLLER_HPP__

#if defined( __cplusplus   ) && !defined( _GPUCODE ) && \
   !defined( __CUDA_ARCH__ ) && !defined( __CUDACC__ )

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <cstddef>
    #include <cstdlib>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#endif /* C++, Host */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/control/definitions.h"
    #include "sixtracklib/common/control/node_controller_base.h"
    #include "sixtracklib/cuda/definitions.h"
    #include "sixtracklib/cuda/control/argument_base.h"
//     #include "sixtracklib/cuda/control/node_info.h"
    #include "sixtracklib/cuda/control/kernel_config.h"
    #include "sixtracklib/common/buffer.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus   ) && !defined( _GPUCODE ) && \
   !defined( __CUDA_ARCH__ ) && !defined( __CUDACC__ )

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/cuda/control/argument_base.hpp"
    #include "sixtracklib/cuda/control/node_info.hpp"
    #include "sixtracklib/cuda/control/kernel_config.hpp"
    #include "sixtracklib/common/buffer.hpp"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

namespace SIXTRL_CXX_NAMESPACE
{
    class CudaController : public SIXTRL_CXX_NAMESPACE::NodeControllerBase
    {
        private:

        using _base_controller_t = SIXTRL_CXX_NAMESPACE::NodeControllerBase;

        public:

        using node_info_t = SIXTRL_CXX_NAMESPACE::CudaNodeInfo;
        using kernel_config_t = SIXTRL_CXX_NAMESPACE::CudaKernelConfig;
        using cuda_device_index_t = node_info_t::cuda_dev_index_t;

        using cuda_arg_buffer_t = ::NS(cuda_arg_buffer_t);
        using cuda_const_arg_buffer_t = ::NS(cuda_const_arg_buffer_t);

        static SIXTRL_CONSTEXPR_OR_CONST size_type DEFAULT_WARP_SIZE =
                SIXTRL_CXX_NAMESPACE::ARCH_CUDA_DEFAULT_WARP_SIZE;

        SIXTRL_HOST_FN explicit CudaController(
            char const* config_str = nullptr );

        SIXTRL_HOST_FN explicit CudaController(
            cuda_device_index_t const node_index,
            char const* SIXTRL_RESTRICT config_str = nullptr );

        SIXTRL_HOST_FN explicit CudaController(
            node_id_t const& SIXTRL_RESTRICT_REF node_id,
            char const* SIXTRL_RESTRICT config_str = nullptr );

        SIXTRL_HOST_FN CudaController(
            platform_id_t const platform_id, device_id_t const device_id,
            char const* SIXTRL_RESTRICT config_str = nullptr );

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        SIXTRL_HOST_FN explicit CudaController(
            std::string const& SIXTRL_RESTRICT_REF config_str );

        SIXTRL_HOST_FN CudaController(
            cuda_device_index_t const node_index,
            std::string const& SIXTRL_RESTRICT_REF config_str );

        SIXTRL_HOST_FN CudaController(
            node_id_t const& SIXTRL_RESTRICT_REF node_id,
            std::string const& SIXTRL_RESTRICT_REF config_str );

        SIXTRL_HOST_FN CudaController(
            platform_id_t const platform_id, device_id_t const device_id,
            std::string const& SIXTRL_RESTRICT_REF config_str );

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        SIXTRL_HOST_FN CudaController( CudaController const& other) = default;
        SIXTRL_HOST_FN CudaController( CudaController&& other) = default;

        SIXTRL_HOST_FN CudaController&
        operator=( CudaController const& other) = default;

        SIXTRL_HOST_FN CudaController& operator=(
            CudaController&& other) = default;

        SIXTRL_HOST_FN virtual ~CudaController() SIXTRL_NOEXCEPT;

        /* ================================================================= */

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

        status_t sendMemory(
            cuda_arg_buffer_t SIXTRL_RESTRICT destination,
            void const* SIXTRL_RESTRICT source,
            size_type const source_length );

        status_t receiveMemory(
            void* SIXTRL_RESTRICT destination,
            cuda_const_arg_buffer_t SIXTRL_RESTRICT source,
            size_type const source_length );

        /* ================================================================ */

        using _base_controller_t::remap;
        using _base_controller_t::isRemapped;

        status_t remap( cuda_arg_buffer_t SIXTRL_RESTRICT managed_buffer_begin,
            size_type const slot_size );

        bool isRemapped( cuda_arg_buffer_t SIXTRL_RESTRICT managed_buffer_begin,
            size_type const slot_size );

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

         SIXTRL_HOST_FN virtual status_t doSend(
            ptr_arg_base_t SIXTRL_RESTRICT destination,
            const void *const SIXTRL_RESTRICT source,
            size_type const source_length ) override;

        SIXTRL_HOST_FN virtual status_t doReceive(
            void* SIXTRL_RESTRICT destination,
            size_type const dest_capacity,
            ptr_arg_base_t SIXTRL_RESTRICT source ) override;

        SIXTRL_HOST_FN virtual status_t doRemapCObjectsBufferArg(
            ptr_arg_base_t SIXTRL_RESTRICT arg ) override;

        SIXTRL_HOST_FN virtual bool doIsCObjectsBufferArgRemapped(
            ptr_arg_base_t SIXTRL_RESTRICT arg,
            status_t* SIXTRL_RESTRICT ptr_status ) override;

        SIXTRL_HOST_FN virtual status_t doSetDebugRegister(
            debug_register_t const debug_register ) override;

        SIXTRL_HOST_FN virtual status_t doFetchDebugRegister(
            debug_register_t* SIXTRL_RESTRICT ptr_debug_register ) override;

        SIXTRL_HOST_FN virtual bool
            doSelectNode( node_index_t const node_index ) override;

        SIXTRL_HOST_FN virtual bool doChangeSelectedNode(
            node_index_t const current_selected_node_idx,
            node_index_t const new_selected_node_index ) override;

        /* ----------------------------------------------------------------- */

        SIXTRL_HOST_FN status_t doRemapCObjectsBufferDirectly(
            cuda_arg_buffer_t SIXTRL_RESTRICT managed_buffer_begin,
            size_type const slot_size );

        SIXTRL_HOST_FN bool doCheckIsCobjectsBufferRemappedDirectly(
            status_t* SIXTRL_RESTRICT ptr_status,
            cuda_arg_buffer_t SIXTRL_RESTRICT managed_buffer_begin,
            size_type const slot_size );

        /* ----------------------------------------------------------------- */

        SIXTRL_HOST_FN  bool doInitCudaDebugRegister();

        SIXTRL_HOST_FN node_index_t doFindAvailableNodesByCudaDeviceIndex(
            cuda_device_index_t const cuda_device_index ) const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN node_index_t doFindAvailableNodesByPciBusId(
            char const* SIXTRL_RESTRICT pci_bus_id_str ) const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN node_index_t doGetNodeIndexFromConfigString(
            char const* SIXTRL_RESTRICT select_str );

        SIXTRL_HOST_FN void doInitCudaController();
        SIXTRL_HOST_FN bool doInitAllCudaNodes();
        SIXTRL_HOST_FN void doEnableCudaController();

        SIXTRL_HOST_FN kernel_id_t doAppendCudaKernelConfig(
            ptr_cuda_kernel_config_t&& ptr_kernel_conf ) SIXTRL_NOEXCEPT;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        SIXTRL_HOST_FN cuda_arg_buffer_t
        doGetPtrCudaDebugRegister() SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN cuda_const_arg_buffer_t
        doGetPtrCudaDebugRegister() const SIXTRL_NOEXCEPT;

        private:

        SIXTRL_HOST_FN static status_t PerformSendOperation(
            cuda_arg_buffer_t SIXTRL_RESTRICT destination,
            void const* SIXTRL_RESTRICT src_begin, size_type const src_length );

        SIXTRL_HOST_FN static status_t PerformReceiveOperation(
            void* SIXTRL_RESTRICT destination,
            size_type const destination_capacity,
            cuda_const_arg_buffer_t SIXTRL_RESTRICT source_begin,
            size_type const source_length );

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        SIXTRL_HOST_FN status_t doSetDebugRegisterCudaBaseImpl(
            debug_register_t const debug_register );

        SIXTRL_HOST_FN status_t doFetchDebugRegisterCudaBaseImpl(
            debug_register_t* SIXTRL_RESTRICT ptr_debug_register );

        SIXTRL_HOST_FN bool doSelectNodeCudaImpl( node_index_t const idx );

        SIXTRL_HOST_FN status_t doRemapCObjectsBufferCudaBaseImpl(
            ptr_arg_base_t SIXTRL_RESTRICT arg, size_type const arg_size );

        cuda_arg_buffer_t   m_cuda_debug_register;
    };
}
#endif /* C++, Host */

#if defined( __cplusplus ) && defined( _GPUCODE )
extern "C" {
#endif /* C++ */

#if defined( __cplusplus   ) && !defined( _GPUCODE ) && \
   !defined( __CUDA_ARCH__ ) && !defined( __CUDACC__ )

typedef SIXTRL_CXX_NAMESPACE::CudaController NS(CudaController);

#else /* C++, Host */

typedef void NS(CudaController);

#endif /* C++, Host */

#if defined( __cplusplus ) && defined( _GPUCODE )
}
#endif /* C++ */

#endif /* SIXTRACKLIB_CUDA_CONTROLLER_HPP__ */

/* end: sixtracklib/cuda/controller.hpp */
