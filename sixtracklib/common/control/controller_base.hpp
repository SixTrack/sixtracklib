#ifndef SIXTRACKLIB_COMMON_CONTROL_CONTROLLER_BASE_HPP__
#define SIXTRACKLIB_COMMON_CONTROL_CONTROLLER_BASE_HPP__

#if defined( __cplusplus   ) && !defined( _GPUCODE ) && \
   !defined( __CUDA_ARCH__ ) && !defined( __CUDACC__ )

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <cstddef>
    #include <cstdlib>
    #include <stdexcept>
    #include <string>
    #include <memory>
    #include <vector>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#endif /* C++, Host */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/control/definitions.h"
    #include "sixtracklib/common/control/arch_base.h"
    #include "sixtracklib/common/buffer.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus   ) && !defined( _GPUCODE ) && \
   !defined( __CUDA_ARCH__ ) && !defined( __CUDACC__ )

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/buffer.hpp"
    #include "sixtracklib/common/control/arch_base.hpp"
    #include "sixtracklib/common/control/kernel_config_base.hpp"
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

namespace SIXTRL_CXX_NAMESPACE
{
    class ArgumentBase;

    class ControllerBase : public SIXTRL_CXX_NAMESPACE::ArchDebugBase
    {
        private:

        using _base_arch_obj_t = SIXTRL_CXX_NAMESPACE::ArchDebugBase;

        public:

        using arch_id_t            = _base_arch_obj_t::arch_id_t;
        using status_t             = _base_arch_obj_t::status_t;
        using debug_register_t     = _base_arch_obj_t::debug_register_t;
        using buffer_t             = SIXTRL_CXX_NAMESPACE::Buffer;

        using c_buffer_t           = buffer_t::c_api_t;
        using size_type            = buffer_t::size_type;
        using kernel_config_base_t = SIXTRL_CXX_NAMESPACE::KernelConfigBase;
        using kernel_id_t          = kernel_config_base_t::kernel_id_t;

        using arg_base_ref_t       = ArgumentBase&;
        using ptr_arg_base_t       = ArgumentBase*;
        using ptr_const_arg_base_t = ArgumentBase const*;

        using perform_remap_flag_t =
            SIXTRL_CXX_NAMESPACE::ctrl_perform_remap_flag_t;


        static SIXTRL_CONSTEXPR_OR_CONST arch_id_t ILLEGAL_ARCH_ID =
            SIXTRL_CXX_NAMESPACE::ARCHITECTURE_ILLEGAL;

        static SIXTRL_CONSTEXPR_OR_CONST arch_id_t NO_ARCH_ID =
            SIXTRL_CXX_NAMESPACE::ARCHITECTURE_NONE;

        static SIXTRL_CONSTEXPR_OR_CONST kernel_id_t ILLEGAL_KERNEL_ID =
            SIXTRL_CXX_NAMESPACE::ARCH_ILLEGAL_KERNEL_ID;


        SIXTRL_HOST_FN bool usesNodes() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN void clear();

        /* ----------------------------------------------------------------- */

        SIXTRL_HOST_FN bool readyForRunningKernel() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN bool readyForSend()          const SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN bool readyForReceive()       const SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN bool readyForRemap()         const SIXTRL_NOEXCEPT;

        /* ----------------------------------------------------------------- */

        SIXTRL_HOST_FN status_t send( ptr_arg_base_t SIXTRL_RESTRICT dest,
            void const* SIXTRL_RESTRICT source, size_type const src_length );

        SIXTRL_HOST_FN status_t send( ptr_arg_base_t SIXTRL_RESTRICT dest,
            const c_buffer_t *const SIXTRL_RESTRICT source,
            perform_remap_flag_t const perform_remap_flag =
                SIXTRL_CXX_NAMESPACE::CTRL_PERFORM_REMAP );

        SIXTRL_HOST_FN status_t send( ptr_arg_base_t SIXTRL_RESTRICT dest,
            buffer_t const& SIXTRL_RESTRICT_REF source,
            perform_remap_flag_t const perform_remap_flag =
                SIXTRL_CXX_NAMESPACE::CTRL_PERFORM_REMAP );

        /* ----------------------------------------------------------------- */

        SIXTRL_HOST_FN status_t receive( void* SIXTRL_RESTRICT destination,
            size_type const destination_capacity,
            ptr_arg_base_t SIXTRL_RESTRICT source );

        SIXTRL_HOST_FN status_t receive( c_buffer_t* SIXTRL_RESTRICT dest,
            ptr_arg_base_t SIXTRL_RESTRICT source,
            perform_remap_flag_t const perform_remap_flag =
                SIXTRL_CXX_NAMESPACE::CTRL_PERFORM_REMAP );

        SIXTRL_HOST_FN status_t receive( buffer_t& SIXTRL_RESTRICT_REF dest,
            ptr_arg_base_t SIXTRL_RESTRICT source,
            perform_remap_flag_t const perform_remap_flag =
                SIXTRL_CXX_NAMESPACE::CTRL_PERFORM_REMAP );

        /* ================================================================= */

        SIXTRL_HOST_FN size_type numKernels() const SIXTRL_NOEXCEPT;

        /* ----------------------------------------------------------------- */

        SIXTRL_HOST_FN bool
        hasRemapCObjectBufferKernel() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN kernel_id_t
        remapCObjectBufferKernelId() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN void setRemapCObjectBufferKernelId(
            kernel_id_t const kernel_id ) SIXTRL_NOEXCEPT;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        SIXTRL_HOST_FN bool
        hasRemapCObjectBufferDebugKernel() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN kernel_id_t
        remapCObjectBufferDebugKernelId() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN void setRemapCObjectBufferDebugKernelId(
            kernel_id_t const kernel_id ) SIXTRL_NOEXCEPT;

        /* ----------------------------------------------------------------- */

        SIXTRL_HOST_FN size_type kernelWorkItemsDim(
            kernel_id_t const kernel_id ) const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN size_type kernelWorkGroupsDim(
            kernel_id_t const kernel_id ) const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN size_type kernelNumArguments(
            kernel_id_t const kernel_id ) const SIXTRL_NOEXCEPT;

        /* ----------------------------------------------------------------- */

        SIXTRL_HOST_FN bool kernelHasName(
            kernel_id_t const id ) const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN std::string const& kernelName(
            kernel_id_t const kernel_id ) const;

        SIXTRL_HOST_FN char const* ptrKernelNameStr(
            kernel_id_t const kernel_id ) const SIXTRL_NOEXCEPT;

        /* ----------------------------------------------------------------- */

        SIXTRL_HOST_FN bool hasKernel(
            kernel_id_t const kernel_id ) const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN bool hasKernel( std::string const& SIXTRL_RESTRICT_REF
                kernel_name ) const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN bool hasKernel(
            char const* SIXTRL_RESTRICT kernel_name ) const SIXTRL_NOEXCEPT;

        /* ----------------------------------------------------------------- */

        SIXTRL_HOST_FN kernel_config_base_t const* ptrKernelConfigBase(
            kernel_id_t const kernel_id ) const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN kernel_config_base_t const* ptrKernelConfigBase(
            std::string const& SIXTRL_RESTRICT_REF
                kernel_name ) const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN kernel_config_base_t const* ptrKernelConfigBase(
            char const* SIXTRL_RESTRICT kernel_name ) const SIXTRL_NOEXCEPT;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        SIXTRL_HOST_FN kernel_config_base_t* ptrKernelConfigBase(
            kernel_id_t const kernel_id ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN kernel_config_base_t* ptrKernelConfigBase(
            std::string const& SIXTRL_RESTRICT_REF kern_name ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN kernel_config_base_t* ptrKernelConfigBase(
            char const* SIXTRL_RESTRICT kernel_name ) SIXTRL_NOEXCEPT;

        /* ----------------------------------------------------------------- */

        template< typename... Args >
        SIXTRL_HOST_FN bool setKernelNumWorkItems(
            kernel_id_t kernel_id, Args&&... params ) SIXTRL_NOEXCEPT;

        template< typename... Args >
        SIXTRL_HOST_FN bool setKernelWorkGroupSizes(
            kernel_id_t kernel_id, Args&&... params ) SIXTRL_NOEXCEPT;

        template< typename... Args >
        SIXTRL_HOST_FN bool setKernelWorkItemOffsets(
            kernel_id_t kernel_id, Args&&... params ) SIXTRL_NOEXCEPT;

        template< typename... Args >
        SIXTRL_HOST_FN bool setKernelPreferredWorkGroupMultiple(
            kernel_id_t kernel_id, Args&&... params ) SIXTRL_NOEXCEPT;

        /* ================================================================= */

        SIXTRL_HOST_FN status_t remap(
            ptr_arg_base_t SIXTRL_RESTRICT ptr_arg );

        SIXTRL_HOST_FN status_t remap(
            arg_base_ref_t SIXTRL_RESTRICT_REF arg );


        SIXTRL_HOST_FN bool isRemapped(
            ptr_arg_base_t SIXTRL_RESTRICT ptr_arg );

        SIXTRL_HOST_FN bool isRemapped(
            arg_base_ref_t SIXTRL_RESTRICT_REF arg );

        /* ================================================================= */

        SIXTRL_HOST_FN virtual ~ControllerBase() SIXTRL_NOEXCEPT;

        template< class Derived > Derived const* asDerivedController(
            arch_id_t const requ_arch_id,
            bool const exact_match_required = false ) const SIXTRL_NOEXCEPT;

        template< class Derived > Derived* asDerivedController(
            arch_id_t const requ_arch_id,
            bool const exact_match_required = false ) SIXTRL_NOEXCEPT;

        protected:

        using ptr_kernel_conf_base_t = std::unique_ptr< kernel_config_base_t >;

        SIXTRL_HOST_FN explicit ControllerBase(
            arch_id_t const arch_id,
            const char *const SIXTRL_RESTRICT arch_str,
            const char *const SIXTRL_RESTRICT config_str = nullptr );

        SIXTRL_HOST_FN ControllerBase( ControllerBase const& other ) = default;
        SIXTRL_HOST_FN ControllerBase( ControllerBase&& other ) = default;

        SIXTRL_HOST_FN ControllerBase&
        operator=( ControllerBase const& rhs ) = default;

        SIXTRL_HOST_FN ControllerBase&
        operator=( ControllerBase&& rhs ) = default;

        /* ----------------------------------------------------------------- */

        SIXTRL_HOST_FN virtual void doClear();

        SIXTRL_HOST_FN virtual status_t doSend(
            ptr_arg_base_t SIXTRL_RESTRICT destination,
            const void *const SIXTRL_RESTRICT source,
            size_type const source_length );

        SIXTRL_HOST_FN virtual status_t doReceive(
            void* SIXTRL_RESTRICT destination, size_type const dest_capacity,
            ptr_arg_base_t SIXTRL_RESTRICT source );

        SIXTRL_HOST_FN virtual status_t doRemapCObjectsBufferArg(
            ptr_arg_base_t SIXTRL_RESTRICT arg );

        SIXTRL_HOST_FN virtual bool doIsCObjectsBufferArgRemapped(
            ptr_arg_base_t SIXTRL_RESTRICT arg,
            status_t* SIXTRL_RESTRICT ptr_status );

        SIXTRL_HOST_FN virtual bool doSwitchDebugMode(
            bool const is_in_debug_mode ) override;

        /* ----------------------------------------------------------------- */

        SIXTRL_HOST_FN void doSetUsesNodesFlag(
            bool const flag ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN void doSetReadyForRunningKernelsFlag(
            bool const flag ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN void doSetReadyForSendFlag(
            bool const flag ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN void doSetReadyForReceiveFlag(
            bool const flag ) SIXTRL_NOEXCEPT;

        /* ----------------------------------------------------------------- */

        SIXTRL_HOST_FN void doReserveForNumKernelConfigs(
            size_type const kernel_configs_capacity );

        SIXTRL_HOST_FN kernel_id_t doFindKernelConfigByName(
            char const* SIXTRL_RESTRICT kernel_name ) const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN kernel_id_t doAppendKernelConfig(
            ptr_kernel_conf_base_t&& ptr_kernel_conf_base ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN void doRemoveKernelConfig(
            kernel_id_t const kernel_id ) SIXTRL_NOEXCEPT;

        private:

        using kernel_config_list_t = std::vector< ptr_kernel_conf_base_t >;

        kernel_config_list_t        m_kernel_configs;
        size_type                   m_num_kernels;

        kernel_id_t                 m_remap_kernel_id;
        kernel_id_t                 m_remap_debug_kernel_id;

        bool                        m_uses_nodes;
        bool                        m_ready_for_running_kernels;
        bool                        m_ready_for_send;
        bool                        m_ready_for_receive;
    };
}
#endif /* C++, host */

#if defined( __cplusplus ) && !defined( _GPUCODE )
extern "C" {
#endif /* C++, Host */

#if defined( __cplusplus   ) && !defined( _GPUCODE ) && \
   !defined( __CUDA_ARCH__ ) && !defined( __CUDACC__ )

typedef SIXTRL_CXX_NAMESPACE::ControllerBase NS(ControllerBase);

#else /* C++, host */

typedef void NS(ControllerBase);

#endif /* C++, host */

#if defined( __cplusplus ) && !defined( _GPUCODE )
}
#endif /* C++, Host */

#if defined( __cplusplus   ) && !defined( _GPUCODE ) && \
   !defined( __CUDA_ARCH__ ) && !defined( __CUDACC__ )

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <type_traits>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

namespace SIXTRL_CXX_NAMESPACE
{
    template< typename... Args >
    bool ControllerBase::setKernelNumWorkItems(
        ControllerBase::kernel_id_t kernel_id,
        Args&&... params ) SIXTRL_NOEXCEPT
    {
        ControllerBase::kernel_config_base_t* ptr_kernel_conf =
            this->ptrKernelConfigBase( kernel_id );

        return ( ptr_kernel_conf != nullptr )
            ? ptr_kernel_conf->setNumWorkItems(
                std::forward< Args >( params )... ) : false;
    }

    template< typename... Args >
    bool ControllerBase::setKernelWorkGroupSizes(
        ControllerBase::kernel_id_t kernel_id,
        Args&&... params ) SIXTRL_NOEXCEPT
    {
        ControllerBase::kernel_config_base_t* ptr_kernel_conf =
            this->ptrKernelConfigBase( kernel_id );

        return ( ptr_kernel_conf != nullptr )
            ? ptr_kernel_conf->setWorkGroupSizes(
                std::forward< Args >( params )... ) : false;
    }

    template< typename... Args >
    bool ControllerBase::setKernelWorkItemOffsets(
        ControllerBase::kernel_id_t kernel_id,
        Args&&... params ) SIXTRL_NOEXCEPT
    {
        ControllerBase::kernel_config_base_t* ptr_kernel_conf =
            this->ptrKernelConfigBase( kernel_id );

        return ( ptr_kernel_conf != nullptr )
            ? ptr_kernel_conf->setWorkItemOffset(
                std::forward< Args >( params )... ) : false;
    }

    template< typename... Args >
    bool ControllerBase::setKernelPreferredWorkGroupMultiple(
        ControllerBase::kernel_id_t kernel_id,
        Args&&... params ) SIXTRL_NOEXCEPT
    {
        ControllerBase::kernel_config_base_t* ptr_kernel_conf =
            this->ptrKernelConfigBase( kernel_id );

        return ( ptr_kernel_conf != nullptr )
            ? ptr_kernel_conf->setPreferredWorkGroupMultiple(
                std::forward< Args >( params )... ) : false;
    }

    /* --------------------------------------------------------------------- */

    template< class Derived > Derived const*
    ControllerBase::asDerivedController(
        ControllerBase::arch_id_t const requ_arch_id,
        bool const exact_match_required ) const SIXTRL_NOEXCEPT
    {
        Derived const* ptr_derived = nullptr;

        using _this_t = SIXTRL_CXX_NAMESPACE::ControllerBase;

        static_assert( std::is_base_of< _this_t, Derived >::value,
                       "asDerivedController< Derived > requires Dervied to be "
                       "derived from SIXTRL_CXX_NAMESPACE::ControllerBase" );

        if( ( ( !exact_match_required ) &&
              ( this->isArchCompatibleWith( requ_arch_id ) ) ) ||
            ( this->isArchIdenticalTo( requ_arch_id ) ) )
        {
            ptr_derived = static_cast< Derived const* >( this );
        }

        return ptr_derived;
    }

    template< class Derived > Derived* ControllerBase::asDerivedController(
        ControllerBase::arch_id_t const requ_arch_id,
        bool const exact_match_required ) SIXTRL_NOEXCEPT
    {
        using _this_t = SIXTRL_CXX_NAMESPACE::ControllerBase;

        return const_cast< Derived* >( static_cast< _this_t const& >(
            *this ).asDerivedController< Derived >(
                requ_arch_id, exact_match_required ) );
    }
}

#endif /* C++, Host */

#endif /* SIXTRACKLIB_COMMON_CONTROL_CONTROLLER_BASE_HPP__ */

/* end: sixtracklib/common/control/controller_base.hpp */
