#ifndef SIXTRACKLIB_COMMON_CONTEXT_CONTEXT_BASE_H__
#define SIXTRACKLIB_COMMON_CONTEXT_CONTEXT_BASE_H__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #if defined( __cplusplus )
        #include <cstddef>
        #include <cstdint>
        #include <cstdlib>
        #include <string>
    #else /* !defined( __cplusplus ) */
        #include <stdbool.h>
        #include <stddef.h>
        #include <stdint.h>
    #endif /* !defined( __cplusplus ) */
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus ) && !defined( _GPUCODE )
extern "C" {
#endif /* defined( __cplusplus ) && !defined( _GPUCODE ) */

typedef SIXTRL_INT64_T  NS(context_type_id_t);

#if defined( __cplusplus ) && !defined( _GPUCODE )
}
#endif /* defined( __cplusplus ) && !defined( _GPUCODE ) */

#if defined( __cplusplus )

namespace SIXTRL_CXX_NAMESPACE
{
    using context_type_t = ::NS(context_type_id_t);

    class ContextBase
    {
        public:

        using type_id_t = context_type_t;
        using size_type = uint64_t;

        SIXTRL_HOST_FN type_id_t type() const SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN std::string const& typeStr() const SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN char const* ptrTypeStr() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN bool hasConfigStr()            const SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN std::string const& configStr() const SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN char const* ptrConfigStr()     const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN bool usesNodes() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN void clear();

        SIXTRL_HOST_FN virtual ~ContextBase() = default;

        protected:

        using type_id_t = context_type_int_t;

        SIXTRL_HOST_FN explicit ContextBase(
            const char *const SIXTRL_RESTRICT type_str,
            type_id_t const type_id );

        SIXTRL_HOST_FN ContextBase( ContextBase const& other ) = default;
        SIXTRL_HOST_FN ContextBase( ContextBase&& other ) = default;

        SIXTRL_HOST_FN ContextBase& operator=( ContextBase const& rhs ) = default;
        SIXTRL_HOST_FN ContextBase& operator=( ContextBase&& rhs ) = default;

        SIXTRL_HOST_FN virtual void doClear();

        SIXTRL_HOST_FN virtual void doParseConfigStr(
            const char *const SIXTRL_RESTRICT config_str );

        SIXTRL_HOST_FN void doSetUsesNodesFlag(
            bool const flag ) SIXTRL_NOEXCEPT;

        private:

        std::string m_config_str;
        std::string m_type_id_str;
        type_id_t   m_type_id;

        bool        m_uses_nodes;
    };
}

#if !defined( _GPUCODE )
extern "C" {
#endif /* !defined( _GPUCODE ) */

typedef SIXTRL_CXX_NAMESPACE::ContextBase            NS(ContextBase);
typedef SIXTRL_CXX_NAMESPACE::ContextBase::size_type NS(context_size_t);

#if !defined( _GPUCODE )
}
#endif /* !defined( _GPUCODE ) */

#else /* !defined( __cplusplus ) */

typedef void            NS(ContextBase);
typedef SIXTRL_UINT64_T NS(context_size_t);

#endif /* defined( __cplusplus ) */

#endif /* SIXTRACKLIB_COMMON_CONTEXT_CONTEXT_BASE_H__ */

/* end: sixtracklib/common/context/context_abs_base.h */
