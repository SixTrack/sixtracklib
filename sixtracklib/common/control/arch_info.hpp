#ifndef SIXTRACKLIB_COMMON_CONTROL_ARCH_INFO_HPP__
#define SIXTRACKLIB_COMMON_CONTROL_ARCH_INFO_HPP__

#if defined( __cplusplus   ) && !defined( _GPUCODE ) && \
   !defined( __CUDA_ARCH__ ) && !defined( __CUDACC__ )

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


#if  defined( __cplusplus ) && !defined( _GPUCODE ) && \
    !defined( __CUDACC__  ) && !defined( __CUDA_ARCH__ )

namespace SIXTRL_CXX_NAMESPACE
{
    class ArchInfo
    {
        public:

        using arch_id_t = SIXTRL_CXX_NAMESPACE::arch_id_t;
        using size_type = SIXTRL_CXX_NAMESPACE::arch_size_t;

        static constexpr arch_id_t ILLEGAL_ARCH =
            SIXTRL_CXX_NAMESPACE::ARCHITECTURE_ILLEGAL;

        SIXTRL_HOST_FN explicit ArchInfo(
            arch_id_t const arch_id = ILLEGAL_ARCH,
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

    SIXTRL_HOST_FN SIXTRL_STATIC std::string ArchInfo_sanitize_arch_str(
        std::string const& SIXTRL_RESTRICT_REF arch_str );

    SIXTRL_HOST_FN SIXTRL_STATIC std::string ArchInfo_sanitize_arch_str(
        const char *const SIXTRL_RESTRICT arch_str );

    SIXTRL_HOST_FN SIXTRL_STATIC SIXTRL_CXX_NAMESPACE::arch_id_t
    ArchInfo_arch_string_to_arch_id(
        std::string const& SIXTRL_RESTRICT_REF arch_str ) SIXTRL_NOEXCEPT;

    SIXTRL_HOST_FN SIXTRL_STATIC SIXTRL_CXX_NAMESPACE::arch_id_t
    ArchInfo_arch_string_to_arch_id(
        char const* SIXTRL_RESTRICT arch_str ) SIXTRL_NOEXCEPT;
}
#endif /* C++, Host */

#if defined( __cplusplus ) && !defined( _GPUCODE )
extern "C" {
#endif /* C++ */

#if defined( __cplusplus   ) && !defined( _GPUCODE ) && \
   !defined( __CUDA_ARCH__ ) && !defined( __CUDACC__ )

typedef SIXTRL_CXX_NAMESPACE::ArchInfo NS(ArchInfo);

#else /* C++, Host */

typedef void NS(ArchInfo);

#endif /* C++, Host */

#if defined( __cplusplus ) && !defined( _GPUCODE )
}
#endif /* C++ */

/* ************************************************************************* */
/* ********     Inline and template function implementation     ************ */
/* ************************************************************************* */

#if  defined( __cplusplus ) && !defined( _GPUCODE ) && \
    !defined( __CUDACC__  ) && !defined( __CUDA_ARCH__ )

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <algorithm>
    #include <cstring>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

namespace SIXTRL_CXX_NAMESPACE
{
    SIXTRL_INLINE std::string ArchInfo_sanitize_arch_str(
        std::string const& SIXTRL_RESTRICT_REF arch_str )
    {
        return SIXTRL_CXX_NAMESPACE::ArchInfo_sanitize_arch_str(
            arch_str.c_str() );
    }

    SIXTRL_INLINE std::string ArchInfo_sanitize_arch_str(
        const char *const SIXTRL_RESTRICT arch_str )
    {
        std::size_t const arch_str_len = ( arch_str != nullptr )
            ? std::strlen( arch_str ) : std::size_t{ 0 };

        std::string result_str;

        if( arch_str_len > std::size_t{ 0 } )
        {
            char const* it  = arch_str;
            char const* end = arch_str;
            std::advance( end, arch_str_len );

            result_str.resize( arch_str_len );
            std::transform( it, end, result_str.begin(), ::tolower );
        }

        return result_str;
    }

    SIXTRL_INLINE SIXTRL_CXX_NAMESPACE::arch_id_t
    ArchInfo_arch_string_to_arch_id(
        std::string const& SIXTRL_RESTRICT_REF arch_str ) SIXTRL_NOEXCEPT
    {
        return ArchInfo_arch_string_to_arch_id( arch_str.c_str() );
    }

    SIXTRL_INLINE SIXTRL_CXX_NAMESPACE::arch_id_t
    ArchInfo_arch_string_to_arch_id(
        char const* SIXTRL_RESTRICT arch_str ) SIXTRL_NOEXCEPT
    {
        SIXTRL_CXX_NAMESPACE::arch_id_t arch_id =
            SIXTRL_CXX_NAMESPACE::ARCHITECTURE_NONE;

        if( ( arch_str != nullptr ) &&
            ( std::strlen( arch_str ) > std::size_t{ 0 } ) )
        {
            if( 0 == std::strcmp( arch_str, SIXTRL_ARCHITECTURE_CPU_STR ) )
            {
                arch_id = SIXTRL_CXX_NAMESPACE::ARCHITECTURE_CPU;
            }
            else if( 0 == std::strcmp(
                    arch_str, SIXTRL_ARCHITECTURE_OPENCL_STR ) )
            {
                arch_id = SIXTRL_CXX_NAMESPACE::ARCHITECTURE_OPENCL;
            }
            else if( 0 == std::strcmp(
                    arch_str, SIXTRL_ARCHITECTURE_CUDA_STR ) )
            {
                arch_id = SIXTRL_CXX_NAMESPACE::ARCHITECTURE_CUDA;
            }
            else
            {
                arch_id = SIXTRL_CXX_NAMESPACE::ARCHITECTURE_ILLEGAL;
            }
        }

        return arch_id;
    }
}

#endif /* C++, Host */

#endif /* SIXTRACKLIB_COMMON_CONTROL_ARCH_INFO_HPP__ */

/* end: sixtracklib/common/control/arch_info.hpp */
