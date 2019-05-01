#ifndef SIXTRACKLIB_COMMON_CONTROL_NODE_ID_HPP__
#define SIXTRACKLIB_COMMON_CONTROL_NODE_ID_HPP__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #if defined( __cplusplus ) && !defined( _GPUCODE ) && \
       !defined( __CUDA_ARCH__ )
        #include <cstddef>
        #include <cstdlib>
        #include <cstdio>
        #include <cstring>
        #include <string>
        #include <ostream>
    #endif
#endif !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/context/definitions.h"
    #include "sixtracklib/common/buffer/buffer_type.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus ) && !defined( _GPUCODE ) && !defined( __CUDA_ARCH__ )

namespace SIXTRL_CXX_NAMESPACE
{
    class NodeId
    {
        using platform_id_t = SIXTRL_CXX_NAMESPACE::node_platform_id_t;
        using device_id_t   = SIXTRL_CXX_NAMESPACE::node_device_id_t;
        using index_t       = SIXTRL_CXX_NAMESPACE::node_index_t;
        using size_type     = ::NS(buffer_size_t);

        SIXTRL_FN explicit NodeId( platform_id_t const platform_id =
                SIXTRL_CXX_NAMESPACE::ENODE_ILLEGAL_PATFORM_ID,
            device_id_t const device_id =
                SIXTRL_CXX_NAMESPACE::NODE_ILLEGAL_DEVICE_ID,
            index_t const node_index =
                SIXTRL_CXX_NAMESPACE::NODE_UNDEFINED_INDEX
             ) SIXTRL_NOEXCEPT;

        SIXTRL_FN explicit NodeId(
            std::string const& SIXTRL_RESTRICT_REF id_str ) SIXTRL_NOEXCEPT;

        SIXTRL_FN explicit NodeId(
            const char *const SIXTRL_RESTRICT id_str ) SIXTRL_NOEXCEPT;

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

        SIXTRL_FN void setNodeIndex(
            node_index_t const index ) SIXTRL_NOEXCEPT;

        SIXTRL_FN bool fromString(
            std::string const& SIXTRL_RESTRICT_REF id_str ) SIXTRL_NOEXCEPT;

        SIXTRL_FN bool fromString(
            const char *const SIXTRL_RESTRICT id_str ) SIXTRL_NOEXCEPT;

        SIXTRL_FN std::string toString() const;

        SIXTRL_FN bool toString( char* SIXTRL_RESTRICT node_id_str,
            size_type const max_str_length ) const SIXTRL_NOEXCEPT;

        SIXTRL_FN bool operator<(
            NodeId const& SIXTRL_RESTRIT_REF rhs ) const SIXTRL_NOEXCEPT;

        SIXTRL_FN void clear() SIXTRL_NOEXCEPT;

        SIXTRL_FN void reset( platform_id_t const platform_id,
            device_id_t const device_id, index_t const node_index =
                SIXTRL_CXX_NAMESPACE::NODE_UNDEFINED_INDEX ) SIXTRL_NOEXCEPT;

        private:

        platform_id_t m_platform_id;
        device_id_t   m_device_id;
        node_index_t  m_node_index;
    };

    void printNodeId( ::FILE* SIXTRL_RESTRICT fp,
        SIXTRL_CXX_NAMESPACE::NodeId const& SIXTRL_RESTRICT_REF node_id );

    std::ostream& operator<<( std::ostream& SIXTRL_RESTRICT_REF output,
        SIXTRL_CXX_NAMESPACE::NodeId const& SIXTRL_RESTRICT_REF node_id );

    int compareNodeIds(
        SIXTRL_CXX_NAMESPACE::NodeId const& SIXTRL_RESTRICT_REF lhs,
        SIXTRL_CXX_NAMESPACE::NodeId const& SIXTRL_RESTRICT_REF rhs );
}

typedef SIXTRL_CXX_NAMESPACE::NodeId    NS(NodeId);

#else  /* !C++, Host */

typedef void NS(NodeId);

#endif /* C++, Host */

#endif /* SIXTRACKLIB_COMMON_CONTROL_NODE_BASE_HPP__ */
/* end: sixtracklib/common/context/node_base.hpp */
