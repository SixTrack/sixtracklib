#ifndef SIXTRACKLIB_COMMON_CONTEXT_CONTEXT_BASE_H__
#define SIXTRACKLIB_COMMON_CONTEXT_CONTEXT_BASE_H__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #if defined( __cplusplus )
        #include <cstddef>
        #include <cstdlib>
        #include <string>
    #else /* !defined( __cplusplus ) */
        #include <stdbool.h>
        #include <stddef.h>
    #endif /* !defined( __cplusplus ) */

    #include <stdint.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"

    #if defined( __cplusplus )
        #include "sixtracklib/common/buffer.hpp"
    #endif /* defined( __cplusplus ) */

    #include "sixtracklib/common/buffer.h"
    #include "sixtracklib/common/context/definitions.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus )

namespace SIXTRL_CXX_NAMESPACE
{
    class ArgumentBase;

    class ContextBase
    {
        public:

        using type_id_t      = SIXTRL_CXX_NAMESPACE::context_type_id_t;
        using status_t       = SIXTRL_CXX_NAMESPACE::context_status_t;
        using buffer_t       = SIXTRL_CXX_NAMESPACE::Buffer;
        using c_buffer_t     = ::NS(Buffer);
        using size_type      = ::NS(context_size_t);
        using ptr_arg_base_t = ArgumentBase*;

        SIXTRL_HOST_FN type_id_t type() const SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN std::string const& typeStr() const SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN char const* ptrTypeStr() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN bool hasConfigStr()            const SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN std::string const& configStr() const SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN char const* ptrConfigStr()     const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN bool usesNodes() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN void clear();

        SIXTRL_HOST_FN bool readyForSend()    const SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN bool readyForReceive() const SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN bool readyForRemap()   const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN status_t send( ptr_arg_base_t SIXTRL_RESTRICT dest,
            void const* SIXTRL_RESTRICT source, size_type const src_length );

        SIXTRL_HOST_FN status_t send( ptr_arg_base_t SIXTRL_RESTRICT dest,
            const c_buffer_t *const SIXTRL_RESTRICT source );

        SIXTRL_HOST_FN status_t send( ptr_arg_base_t SIXTRL_RESTRICT dest,
            buffer_t const& SIXTRL_RESTRICT_REF source );

        SIXTRL_HOST_FN status_t receive( void* SIXTRL_RESTRICT destination,
            size_type const destination_capacity,
            ptr_arg_base_t SIXTRL_RESTRICT source );

        SIXTRL_HOST_FN status_t receive( c_buffer_t* SIXTRL_RESTRICT dest,
            ptr_arg_base_t SIXTRL_RESTRICT source );

        SIXTRL_HOST_FN status_t receive( buffer_t& SIXTRL_RESTRICT_REF dest,
            ptr_arg_base_t SIXTRL_RESTRICT source );

        SIXTRL_HOST_FN status_t remapSentCObjectsBuffer(
            ptr_arg_base_t SIXTRL_RESTRICT arg,
            size_type const arg_size = size_type{ 0 } );

        SIXTRL_HOST_FN bool isInDebugMode() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN virtual ~ContextBase() = default;

        protected:

        SIXTRL_HOST_FN explicit ContextBase( type_id_t const type_id,
            const char *const SIXTRL_RESTRICT type_str,
            const char *const SIXTRL_RESTRICT config_str = nullptr );

        SIXTRL_HOST_FN ContextBase( ContextBase const& other ) = default;
        SIXTRL_HOST_FN ContextBase( ContextBase&& other ) = default;

        SIXTRL_HOST_FN ContextBase&
        operator=( ContextBase const& rhs ) = default;

        SIXTRL_HOST_FN ContextBase&
        operator=( ContextBase&& rhs ) = default;

        SIXTRL_HOST_FN virtual void doClear();

        SIXTRL_HOST_FN virtual void doParseConfigStr(
            const char *const SIXTRL_RESTRICT config_str );

        SIXTRL_HOST_FN virtual status_t doSend(
            ptr_arg_base_t SIXTRL_RESTRICT destination,
            const void *const SIXTRL_RESTRICT source,
            size_type const source_length );

        SIXTRL_HOST_FN virtual status_t doReceive(
            void* SIXTRL_RESTRICT destination, size_type const dest_capacity,
            ptr_arg_base_t SIXTRL_RESTRICT source );

        SIXTRL_HOST_FN virtual status_t doRemapSentCObjectsBuffer(
            ptr_arg_base_t SIXTRL_RESTRICT arg,
            size_type arg_size );

        SIXTRL_HOST_FN void doSetTypeId(
            type_id_t const type_id ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN void doSetTypeIdStr(
            const char *const SIXTRL_RESTRICT type_id_str ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN void doSetUsesNodesFlag(
            bool const flag ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN void doSetReadyForSendFlag(
            bool const flag ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN void doSetReadyForReceiveFlag(
            bool const flag ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN void doSetReadyForRemapFlag(
            bool const flag ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN void doSetDebugModeFlag(
            bool const flag ) SIXTRL_NOEXCEPT;

        private:

        SIXTRL_HOST_FN void doParseConfigStrBaseImpl(
            const char *const SIXTRL_RESTRICT config_str );

        std::string m_config_str;
        std::string m_type_id_str;
        type_id_t   m_type_id;

        bool        m_uses_nodes;
        bool        m_ready_for_remap;
        bool        m_ready_for_send;
        bool        m_ready_for_receive;
        bool        m_debug_mode;
    };
}

#if !defined( _GPUCODE )
extern "C" {
#endif /* !defined( _GPUCODE ) */

typedef SIXTRL_CXX_NAMESPACE::ContextBase  NS(ContextBase);

#if !defined( _GPUCODE )
}
#endif /* !defined( _GPUCODE ) */

#else /* !defined( __cplusplus ) */

typedef void NS(ContextBase);

#endif /* defined( __cplusplus ) */

#endif /* SIXTRACKLIB_COMMON_CONTEXT_CONTEXT_BASE_H__ */

/* end: sixtracklib/common/context/context_abs_base.h */
