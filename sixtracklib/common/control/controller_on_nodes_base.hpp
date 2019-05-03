#ifndef SIXTRACKLIB_COMMON_CONTROLLER_ON_NODES_BASE_HPP__
#define SIXTRACKLIB_COMMON_CONTROLLER_ON_NODES_BASE_HPP__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #if defined( __cplusplus ) && !defined( _GPUCODE ) && \
       !defined( __CUDA_ARCH__ )
        #include <cstddef>
        #include <cstdlib>
        #include <string>
        #include <iostream>
        #include <vector>
    #endif /* C++, Host */
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/control/definitions.h"
    #include "sixtracklib/common/control/node_id.h"
    #include "sixtracklib/common/control/node_info.h"
    #include "sixtracklib/common/control/controller_base.hpp"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus ) && !defined( _GPUCODE ) && !defined( __CUDA_ARCH__ )

namespace SIXTRL_CXX_NAMESPACE
{
    class ControllerOnNodesBase : public SIXTRL_CXX_NAMESPACE::ControllerBase
    {
        private:

        using _controller_base_t   = ControllerBase;

        public:

        using size_type         = _controller_base_t::size_type;
        using arch_id_t         = _controller_base_t::arch_id_t;

        using node_id_t         = SIXTRL_CXX_NAMESPACE::NodeId;
        using node_info_base_t  = SIXTRL_CXX_NAMESPACE::NodeInfoBase;
        using platform_id_t     = node_id_t::platform_id_t;
        using device_id_t       = node_id_t::device_id_t;
        using node_index_t      = node_id_t::index_t;

        static SIXTRL_CONSTEXPR_OR_CONST size_type UNDEFINED_INDEX =
            SIXTRL_CXX_NAMESPACE::NODE_UNDEFINED_INDEX;

        SIXTRL_HOST_FN node_index_t numAvailableNodes() const SIXTRL_NOEXCEPT;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        SIXTRL_HOST_FN bool hasDefaultNode() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN node_id_t const*
        ptrDefaultNodeId() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN node_info_base_t const*
        ptrDefaultNodeInfoBase() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN node_index_t defaultNodeIndex() const SIXTRL_NOEXCEPT;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        SIXTRL_HOST_FN bool isNodeAvailable(
            node_index_t const node_index ) const SIXTRL_RESTRICT;

        SIXTRL_HOST_FN bool isNodeAvailable(
            node_id_t const& node_id ) const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN bool isNodeAvailable(
            platform_id_t const platform_index,
            device_id_t const device_index ) const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN bool isNodeAvailable(
            char const* node_id_str ) const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN bool isNodeAvailable(
            std::string const& node_id_str ) const SIXTRL_NOEXCEPT;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        SIXTRL_HOST_FN bool isDefaultNode(
            char const* node_id_str ) const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN bool isDefaultNode( std::string const&
            SIXTRL_RESTRICT_REF node_id_str ) const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN bool isDefaultNode(
            node_id_t const& node_id ) const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN bool isDefaultNode( platform_id_t const platform_index,
            device_id_t const device_index ) const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN bool isDefaultNode(
            node_index_t const node_index ) const SIXTRL_NOEXCEPT;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        SIXTRL_HOST_FN node_id_t const* ptrNodeId(
            char const* SIXTRL_RESTRICT node_id_str ) const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN node_id_t const* ptrNodeId( std::string const&
            SIXTRL_RESTRICT_REF node_id_str ) const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN node_id_t const* ptrNodeId(
            platform_id_t const platform_index,
            device_id_t const device_index ) const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN node_id_t const* ptrNodeId(
            node_index_t const index ) const SIXTRL_NOEXCEPT;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        SIXTRL_HOST_FN node_info_base_t const* ptrNodeInfoBase(
            node_index_t const index ) const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN node_info_base_t const* ptrNodeInfoBase(
            platform_id_t const platform_idx,
            device_id_t const device_idx ) const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN node_info_base_t const* ptrNodeInfoBase(
            node_id_t const& node_id ) const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN node_info_base_t const* ptrNodeInfoBase(
            char const* SIXTRL_RESTRICT node_id_str ) const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN node_info_base_t const* ptrNodeInfoBase(
            std::string const& SIXTRL_RESTRICT_REF node_id_str
            ) const SIXTRL_NOEXCEPT;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        SIXTRL_HOST_FN bool hasSelectedNode() const SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN node_index_t selectedNodeIndex() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN node_id_t const*
        ptrSelectedNodeId() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN node_info_base_t const*
        ptrSelectedNodeInfoBase() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN std::string
        selectedNodeIdStr() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN char const*
        ptrSelectedNodeIdStr() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN bool selectedNodeIdStr( char* SIXTRL_RESTRICT node_str,
            size_type const max_str_length ) const SIXTRL_NOEXCEPT;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        SIXTRL_HOST_FN bool selectNode( node_index_t const index );
        SIXTRL_HOST_FN bool selectNode( node_id_t const& node_id );
        SIXTRL_HOST_FN bool selectNode(
            platform_id_t const platform_idx, device_id_t const device_idx );

        SIXTRL_HOST_FN bool selectNode( char const* node_id_str );
        SIXTRL_HOST_FN bool selectNode( std::string const& node_id_str );

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        SIXTRL_HOST_FN void printAvailableNodesInfo() const;

        SIXTRL_HOST_FN void printAvailableNodesInfo(
            std::ostream& SIXTRL_RESTRICT_REF os ) const;

        SIXTRL_HOST_FN void printAvailableNodesInfo(
            ::FILE* SIXTRL_RESTRICT output ) const;

        SIXTRL_HOST_FN std::string availableNodesInfoToString() const;
        SIXTRL_HOST_FN void storeAvailableNodesInfoToCString(
            char* SIXTRL_RESTRICT nodes_info_str,
            size_type const nodes_info_str_capacity,
            size_type* SIXTRL_RESTRICT ptr_required_max_nodes_info_str_length
        ) const;

        SIXTRL_HOST_FN virtual ~ControllerOnNodesBase() SIXTRL_NOEXCEPT;

        protected:

        using ptr_node_info_base_t = std::unique_ptr< node_info_base_t >;

        static SIXTRL_CONSTEXPR_OR_CONST size_type NODE_ID_STR_CAPACITY =
            size_type{ 18 };

        SIXTRL_HOST_FN ControllerOnNodesBase(
            arch_id_t const arch_id,
            const char *const SIXTRL_RESTRICT arch_str,
            const char *const SIXTRL_RESTRICT config_str = nullptr );

        SIXTRL_HOST_FN ControllerOnNodesBase(
            ControllerOnNodesBase const& other ) = default;

        SIXTRL_HOST_FN ControllerOnNodesBase(
            ControllerOnNodesBase&& other ) = default;

        SIXTRL_HOST_FN ControllerOnNodesBase& operator=(
            ControllerOnNodesBase const& rhs ) = default;

        SIXTRL_HOST_FN ControllerOnNodesBase& operator=(
            ControllerOnNodesBase&& rhs ) = default;

        SIXTRL_HOST_FN virtual void doClear() override;
        SIXTRL_HOST_FN virtual bool doSelectNode(
            node_index_t const node_index );

        SIXTRL_HOST_FN node_index_t doFindAvailableNodesIndex(
            platform_id_t const platform_idx,
            device_id_t const device_idx ) const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN node_index_t doFindAvailableNodesIndex(
            char const* SIXTRL_RESTRICT node_id_str ) const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN void doClearAvailableNodes() SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN node_index_t doAppendAvailableNodeInfoBase(
            ptr_node_info_base_t&& SIXTRL_RESTRICT_REF ptr_node_info_base );

        SIXTRL_HOST_FN node_info_base_t* doGetPtrNodeInfoBase(
            node_index_t const node_index ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN void doSetDefaultNodeIndex(
            node_index_t const node_index ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN platform_id_t doGetNextPlatformId(
            ) const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN device_id_t doGetNextDeviceIdForPlatform(
            platform_id_t const platform_id ) const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN platform_id_t doGetPlatformIdByPlatformName(
            char const* SIXTRL_RESTRICT platform_name ) const SIXTRL_NOEXCEPT;

        private:

        SIXTRL_HOST_FN void doClearOnNodesBaseImpl() SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN bool doSelectNodeControllerOnNodesBaseImpl(
            node_index_t const node_index ) SIXTRL_NOEXCEPT;

        std::vector< ptr_node_info_base_t > m_available_nodes;
        std::vector< char > m_selected_node_id_str;

        node_id_t const* m_ptr_default_node_id;
        node_id_t const* m_ptr_selected_node_id;
    };
}

#endif /* C++, Host */

#if defined( __cplusplus ) && !defined( _GPUCODE ) && !defined( __CUDA_ARCH__ )

extern "C" {

typedef SIXTRL_CXX_NAMESPACE::ControllerOnNodesBase NS(ControllerOnNodesBase);

typedef SIXTRL_CXX_NAMESPACE::ControllerOnNodesBase::node_id_t NS(node_id_t);

typedef SIXTRL_CXX_NAMESPACE::ControllerOnNodesBase::node_info_base_t
        NS(node_info_base_t);

}

#else /* C++, Host */

typedef void NS(ContextNodeBase);

typedef NS(NodeId)              NS(node_id_t);
typedef NS(ComputeNodeInfo)     NS(node_info_base_t);

#endif /* C++, Host */

#endif /* SIXTRACKLIB_COMMON_CONTROLLER_ON_NODES_BASE_HPP__ */

/* end: sixtracklib/common/control/controller_on_nodes_base.hpp */
