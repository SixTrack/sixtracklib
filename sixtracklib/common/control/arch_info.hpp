#ifndef SIXTRACKLIB_COMMON_CONTROL_ARCH_INFO_HPP__
#define SIXTRACKLIB_COMMON_CONTROL_ARCH_INFO_HPP__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #if defined( __cplusplus ) && !defined( _GPUCODE ) && \
       !defined( __CUDA_ARCH__ )
        #include <cstddef>
        #include <cstdlib>
        #include <string>
    #endif
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/control/definitions.h"
    #include "sixtracklib/common/buffer/buffer_type.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */


#if defined( __cplusplus ) && !defined( _GPUCODE ) && !defined( __CUDA_ARCH__ )

namespace SIXTRL_CXX_NAMESPACE
{
    class ArchInfo
    {
        public:

        using arch_id_t = SIXTRL_CXX_NAMESPACE::arch_id_t;
        using size_type = SIXTRL_CXX_NAMESPACE::arch_size_t;

        SIXTRL_HOST_FN explicit ArchInfo( arch_id_t const arch_id =
                SIXTRL_CXX_NAMESPACE::ARCHITECTURE_ILLEGAL,
            const char *const SIXTRL_RESTRICT arch_str = nullptr );

        SIXTRL_HOST_FN explicit ArchInfo( arch_id_t const arch_id,
            std::string const& arch_str );

        SIXTRL_HOST_FN ArchInfo( ArchInfo const& other ) = default;
        SIXTRL_HOST_FN ArchInfo( ArchInfo&& other ) = default;

        SIXTRL_HOST_FN ArchInfo& operator=( ArchInfo const& rhs ) = default;
        SIXTRL_HOST_FN ArchInfo& operator=( ArchInfo&& rhs ) = default;
        SIXTRL_HOST_FN virtual ~ArchInfo() = default;

        SIXTRL_HOST_FN arch_id_t archId() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN bool hasArchStr() const SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN std::string const& archStr() const SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN char const* ptrArchStr() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN bool isArchCompatibleWith(
            arch_id_t const arch_id ) const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN bool isArchCompatibleWith(
            ArchInfo const& SIXTRL_RESTRICT other ) const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN bool isArchIdenticalTo(
            arch_id_t const arch_id ) const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN bool isArchIdenticalTo(
            ArchInfo const& SIXTRL_RESTRICT rhs ) const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN void reset( arch_id_t const arch_id,
            const char *const SIXTRL_RESTRICT arch_str = nullptr );

        SIXTRL_HOST_FN void reset() SIXTRL_NOEXCEPT;

        protected:

        SIXTRL_HOST_FN void doSetArchId(
            arch_id_t const arch_id ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN void doSetArchStr(
            const char *const SIXTRL_RESTRICT arch_str );

        private:

        std::string m_arch_str;
        arch_id_t   m_arch_id;
    };
}

typedef SIXTRL_CXX_NAMESPACE::ArchInfo NS(ArchInfo);

#else /* C++, Host */

typedef void NS(ArchInfo);

#endif /* C++, Host */

#endif /* SIXTRACKLIB_COMMON_CONTROL_ARCH_INFO_HPP__ */

/* end: sixtracklib/common/control/arch_info.hpp */
