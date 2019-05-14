#ifndef SIXTRACKLIB_COMMON_CONTROL_NODE_ID_HPP__
#define SIXTRACKLIB_COMMON_CONTROL_NODE_ID_HPP__

#if defined( __cplusplus ) && !defined( _GPUCODE ) && \
   !defined( __CUDACC__  ) && !defined( __CUDA_ARCH__ )

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <cstddef>
    #include <cstdlib>
    #include <cstdio>
    #include <cstring>
    #include <string>
    #include <ostream>
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
    class NodeId
    {
        public:

        using size_type     = SIXTRL_CXX_NAMESPACE::ctrl_size_t;
        using platform_id_t = SIXTRL_CXX_NAMESPACE::node_platform_id_t;
        using device_id_t   = SIXTRL_CXX_NAMESPACE::node_device_id_t;
        using index_t       = SIXTRL_CXX_NAMESPACE::node_index_t;

        static SIXTRL_CONSTEXPR_OR_CONST platform_id_t ILLEGAL_PLATFORM_ID =
            SIXTRL_CXX_NAMESPACE::NODE_ILLEGAL_PATFORM_ID;

        static SIXTRL_CONSTEXPR_OR_CONST device_id_t ILLEGAL_DEVICE_ID =
            SIXTRL_CXX_NAMESPACE::NODE_ILLEGAL_DEVICE_ID;

        static SIXTRL_CONSTEXPR_OR_CONST node_index_t UNDEFINED_INDEX =
            SIXTRL_CXX_NAMESPACE::NODE_UNDEFINED_INDEX;

        SIXTRL_FN explicit NodeId(
            platform_id_t const platform_id = ILLEGAL_PLATFORM_ID,
            device_id_t const device_id = ILLEGAL_DEVICE_ID,
            node_index_t const node_index = UNDEFINED_INDEX ) SIXTRL_NOEXCEPT;

        SIXTRL_FN explicit NodeId(
            std::string const& SIXTRL_RESTRICT_REF id_str );

        SIXTRL_FN explicit NodeId( const char *const SIXTRL_RESTRICT id_str );

        SIXTRL_FN NodeId( NodeId const& other ) = default;
        SIXTRL_FN NodeId( NodeId&& other ) = default;

        SIXTRL_FN NodeId& operator=( NodeId const& rhs ) = default;
        SIXTRL_FN NodeId& operator=( NodeId&& rhs ) = default;

        SIXTRL_FN ~NodeId() = default;

        SIXTRL_FN bool valid() const SIXTRL_NOEXCEPT;
        SIXTRL_FN platform_id_t platformId() const SIXTRL_NOEXCEPT;
        SIXTRL_FN device_id_t   deviceId() const SIXTRL_NOEXCEPT;

        SIXTRL_FN bool hasIndex() const SIXTRL_NOEXCEPT;
        SIXTRL_FN node_index_t index() const SIXTRL_NOEXCEPT;

        SIXTRL_FN void setPlatformId(
            platform_id_t const id ) SIXTRL_NOEXCEPT;

        SIXTRL_FN void setDeviceId(
            device_id_t const id ) SIXTRL_NOEXCEPT;

        SIXTRL_FN void setIndex( node_index_t const index ) SIXTRL_NOEXCEPT;

        SIXTRL_FN bool fromString(
            std::string const& SIXTRL_RESTRICT_REF id_str ) SIXTRL_NOEXCEPT;

        SIXTRL_FN bool fromString(
            const char *const SIXTRL_RESTRICT id_str ) SIXTRL_NOEXCEPT;

        SIXTRL_FN std::string toString() const;

        SIXTRL_FN bool toString( char* SIXTRL_RESTRICT node_id_str,
            size_type const node_id_str_capacity ) const SIXTRL_NOEXCEPT;

        SIXTRL_FN bool operator<(
            NodeId const& SIXTRL_RESTRICT_REF rhs ) const SIXTRL_NOEXCEPT;

        SIXTRL_FN void clear() SIXTRL_NOEXCEPT;

        SIXTRL_FN void reset( platform_id_t const platform_id,
            device_id_t const device_id, index_t const node_index =
                SIXTRL_CXX_NAMESPACE::NODE_UNDEFINED_INDEX ) SIXTRL_NOEXCEPT;

        private:

        platform_id_t m_platform_id;
        device_id_t   m_device_id;
        node_index_t  m_node_index;
    };

    SIXTRL_HOST_FN void printNodeId( ::FILE* SIXTRL_RESTRICT fp,
        SIXTRL_CXX_NAMESPACE::NodeId const& SIXTRL_RESTRICT_REF node_id );

    SIXTRL_HOST_FN std::ostream& operator<<(
        std::ostream& SIXTRL_RESTRICT_REF output,
        SIXTRL_CXX_NAMESPACE::NodeId const& SIXTRL_RESTRICT_REF node_id );

    SIXTRL_HOST_FN int compareNodeIds(
        SIXTRL_CXX_NAMESPACE::NodeId const& SIXTRL_RESTRICT_REF lhs,
        SIXTRL_CXX_NAMESPACE::NodeId const& SIXTRL_RESTRICT_REF rhs );

    SIXTRL_STATIC SIXTRL_HOST_FN std::string
    NodeId_extract_node_id_str_from_config_str(
        std::string const& SIXTRL_RESTRICT_REF config_str );

    SIXTRL_STATIC SIXTRL_HOST_FN std::string
    NodeId_extract_node_id_str_from_config_str(
        char const* SIXTRL_RESTRICT config_str );
}
#endif /* C++, Host */

#if defined( __cplusplus ) && !defined( _GPUCODE )
extern "C" {
#endif /* c++ */

#if defined( __cplusplus ) && !defined( _GPUCODE ) && \
   !defined( __CUDACC__  ) && !defined( __CUDA_ARCH__ )

typedef SIXTRL_CXX_NAMESPACE::NodeId  NS(NodeId);

#else  /* !C++, Host */

typedef void NS(NodeId);

#endif /* C++, Host */

#if defined( __cplusplus ) && !defined( _GPUCODE )
}
#endif /* c++ */

/* ************************************************************************* */
/* ******  Implementation of inline and template member functions    ******* */
/* ************************************************************************* */

#if defined( __cplusplus ) && !defined( _GPUCODE ) && \
   !defined( __CUDACC__  ) && !defined( __CUDA_ARCH__ )

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <algorithm>
    #include <cstring>
    #include <regex>
    #include <utility>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

namespace SIXTRL_CXX_NAMESPACE
{
    SIXTRL_INLINE std::string NodeId_extract_node_id_str_from_config_str(
        std::string const& SIXTRL_RESTRICT_REF config_str )
    {
        using SIXTRL_CXX_NAMESPACE::NodeId_extract_node_id_str_from_config_str;
        return NodeId_extract_node_id_str_from_config_str( config_str.c_str());
    }

    SIXTRL_INLINE std::string NodeId_extract_node_id_str_from_config_str(
        char const* SIXTRL_RESTRICT config_str )
    {
        if( ( config_str != nullptr ) &&
            ( std::strlen( config_str ) > std::size_t{ 0 } ) )
        {
            /*
            std::regex re(
                        "device_id_str[:blank:]*=[:blank:]*"
                              "([:digit:]+.[:digit:]+)[A-Za-z0-9_\\-#=:;., \t]*|"
                        "^[A-Za-z0-9_\\-#=;.:, \t]*([:digit:]+.[:digit:]+);|"
                        "([:digit:]+.[:digit:]+)" );*/

            std::regex re( "\\s*([0-9]+\\.[0-9]+)[\\sA-Za-z0-9#\\;]*" );
            std::cmatch matches;

            std::regex_match( config_str, matches, re );

            if( ( matches.ready() ) && ( !matches.empty() ) )
            {
                return std::string{ matches[ matches.size() - 1 ] };
            }
        }

        return std::string{ "" };
    }
}

#endif /* C++, Host */

#endif /* SIXTRACKLIB_COMMON_CONTROL_NODE_ID_HPP__ */

/* end: sixtracklib/common/control/node_id.hpp */
