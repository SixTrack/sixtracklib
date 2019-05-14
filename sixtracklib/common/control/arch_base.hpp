#ifndef SIXTRACKLIB_COMMON_CONTROL_ARCH_BASE_OBJECT_HPP__
#define SIXTRACKLIB_COMMON_CONTROL_ARCH_BASE_OBJECT_HPP__

#if defined( __cplusplus ) && !defined( _GPUCODE ) && \
   !defined( __CUDACC__  ) && !defined( __CUDA_ARCH__ )

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/control/arch_info.hpp"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <cstddef>
    #include <cstdlib>
    #include <string>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#endif /* C++, Host */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/control/definitions.h"
    #include "sixtracklib/common/buffer/buffer_type.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */


#if defined( __cplusplus ) && !defined( _GPUCODE ) && \
   !defined( __CUDACC__  ) && !defined( __CUDA_ARCH__ )

namespace SIXTRL_CXX_NAMESPACE
{
    class ArchBase : public SIXTRL_CXX_NAMESPACE::ArchInfo
    {
        public:

        using arch_info_t = SIXTRL_CXX_NAMESPACE::ArchInfo;
        using status_t    = SIXTRL_CXX_NAMESPACE::arch_status_t;

        SIXTRL_HOST_FN explicit ArchBase( arch_id_t const arch_id,
            const char *const SIXTRL_RESTRICT arch_str = nullptr,
            const char *const SIXTRL_RESTRICT config_str = nullptr );

        SIXTRL_HOST_FN explicit ArchBase(
            arch_id_t const arch_id,
            std::string const& SIXTRL_RESTRICT arch_str,
            std::string const& SIXTRL_RESTRICT config_str = std::string{} );

        SIXTRL_HOST_FN ArchBase( ArchBase const& other ) = default;
        SIXTRL_HOST_FN ArchBase( ArchBase&& other ) = default;

        SIXTRL_HOST_FN ArchBase& operator=( ArchBase const& rhs ) = default;
        SIXTRL_HOST_FN ArchBase& operator=( ArchBase&& rhs ) = default;

        SIXTRL_HOST_FN virtual ~ArchBase() = default;

        template< class Derived > SIXTRL_HOST_FN Derived const* asDerived(
            arch_id_t const required_arch_id,
            bool requires_exact_match = false ) const SIXTRL_NOEXCEPT;

        template< class Derived > SIXTRL_HOST_FN Derived* asDerived(
            arch_id_t const required_arch_id,
            bool requires_exact_match = false ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN bool hasConfigStr() const SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN std::string const& configStr() const SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN char const* ptrConfigStr() const SIXTRL_NOEXCEPT;

        protected:

        SIXTRL_HOST_FN virtual bool doParseConfigStr(
            const char *const SIXTRL_RESTRICT config_str );

        SIXTRL_HOST_FN void doUpdateStoredConfigStr(
            const char *const SIXTRL_RESTRICT config_str );

        private:

        SIXTRL_HOST_FN bool doParseConfigStrArchBase(
            const char *const SIXTRL_RESTRICT config_str ) SIXTRL_NOEXCEPT;

        std::string m_config_str;
    };

    class ArchDebugBase : public SIXTRL_CXX_NAMESPACE::ArchBase
    {
        private:

        using _arch_base_t = SIXTRL_CXX_NAMESPACE::ArchBase;

        public:

        using debug_register_t = SIXTRL_CXX_NAMESPACE::arch_debugging_t;

        static SIXTRL_CONSTEXPR_OR_CONST debug_register_t DEBUG_REGISTER_OK =
            SIXTRL_CXX_NAMESPACE::ARCH_DEBUGGING_REGISTER_EMPTY;

        SIXTRL_HOST_FN explicit ArchDebugBase( arch_id_t const arch_id,
            const char *const SIXTRL_RESTRICT arch_str = nullptr,
            const char *const SIXTRL_RESTRICT config_str = nullptr );

        SIXTRL_HOST_FN ArchDebugBase( ArchDebugBase const& other ) = default;
        SIXTRL_HOST_FN ArchDebugBase( ArchDebugBase&& other ) = default;

        SIXTRL_HOST_FN ArchDebugBase&
        operator=( ArchDebugBase const& rhs ) = default;

        SIXTRL_HOST_FN ArchDebugBase&
        operator=( ArchDebugBase&& rhs ) = default;

        SIXTRL_HOST_FN virtual ~ArchDebugBase() = default;

        SIXTRL_HOST_FN bool isInDebugMode() const SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN bool enableDebugMode();
        SIXTRL_HOST_FN bool disableDebugMode();

        /* ----------------------------------------------------------------- */

        SIXTRL_HOST_FN debug_register_t debugRegister();

        SIXTRL_HOST_FN status_t setDebugRegister(
            debug_register_t const debug_register );

        SIXTRL_HOST_FN status_t prepareDebugRegisterForUse();
        SIXTRL_HOST_FN status_t evaluateDebugRegisterAfterUse();

        protected:

        static SIXTRL_CONSTEXPR_OR_CONST size_type
            DEBUG_REGISTER_SIZE = sizeof( debug_register_t );

        SIXTRL_HOST_FN virtual bool doSwitchDebugMode(
            bool const is_in_debug_mode );

        SIXTRL_HOST_FN virtual status_t doSetDebugRegister(
            debug_register_t const debug_register );

        SIXTRL_HOST_FN virtual status_t doFetchDebugRegister(
            debug_register_t* SIXTRL_RESTRICT ptr_debug_register );

        /* ----------------------------------------------------------------- */

        SIXTRL_HOST_FN void doSetDebugModeFlag(
            bool const debug_mode ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN debug_register_t const*
        doGetPtrLocalDebugRegister() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN debug_register_t*
        doGetPtrLocalDebugRegister() SIXTRL_NOEXCEPT;

        private:

        bool m_debug_mode;
        debug_register_t m_local_debug_register;
    };
}

#endif /* C++, Host */

#if defined( __cplusplus ) && !defined( _GPUCODE )
extern "C" {
#endif /* C++ */

#if defined( __cplusplus ) && !defined( _GPUCODE ) && \
   !defined( __CUDACC__  ) && !defined( __CUDA_ARCH__ )

typedef SIXTRL_CXX_NAMESPACE::ArchBase      NS(ArchBase);
typedef SIXTRL_CXX_NAMESPACE::ArchDebugBase NS(ArchDebugBase);

#else /* C++, Host */

typedef void NS(ArchBase);
typedef void NS(ArchDebugBase);

#endif /* C++, Host */

#if defined( __cplusplus ) && !defined( _GPUCODE )
}
#endif /* C++ */

#endif /* SIXTRACKLIB_COMMON_CONTROL_ARCH_BASE_OBJECT_HPP__ */

/* end: sixtracklib/common/control/arch_base.hpp */
