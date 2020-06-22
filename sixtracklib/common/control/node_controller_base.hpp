#ifndef SIXTRACKLIB_COMMON_CONTROL_NODE_CONTROLLER_BASE_HPP__
#define SIXTRACKLIB_COMMON_CONTROL_NODE_CONTROLLER_BASE_HPP__

#if defined( __cplusplus   ) && !defined( _GPUCODE ) && \
   !defined( __CUDA_ARCH__ ) && !defined( __CUDACC__ )

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
        #include <cstddef>
        #include <cstdlib>
        #include <string>
        #include <iostream>
        #include <vector>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#endif /* C++, Host */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/control/definitions.h"
    #include "sixtracklib/common/control/node_id.h"
    #include "sixtracklib/common/control/node_info.h"
    #include "sixtracklib/common/control/controller_base.hpp"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus   ) && !defined( _GPUCODE ) && \
   !defined( __CUDA_ARCH__ ) && !defined( __CUDACC__ )

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/control/controller_base.hpp"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

namespace SIXTRL_CXX_NAMESPACE
{
    class NodeControllerBase : public SIXTRL_CXX_NAMESPACE::ControllerBase
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
            node_index_t const node_index ) const SIXTRL_NOEXCEPT;

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

        SIXTRL_HOST_FN bool isSelectedNode(
            char const* node_id_str ) const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN bool isSelectedNode( std::string const&
            SIXTRL_RESTRICT_REF node_id_str ) const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN bool isSelectedNode(
            node_id_t const& node_id ) const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN bool isSelectedNode( platform_id_t const platform_index,
            device_id_t const device_index ) const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN bool isSelectedNode(
            node_index_t const node_index ) const SIXTRL_NOEXCEPT;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        SIXTRL_HOST_FN node_index_t nodeIndex(
            node_id_t const& SIXTRL_RESTRICT_REF node_id
            ) const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN node_index_t nodeIndex(
            platform_id_t const platform_id,
            device_id_t const device_id ) const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN node_index_t nodeIndex(
            node_info_base_t const* SIXTRL_RESTRICT
                ptr_node_info ) const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN node_index_t nodeIndex(
            char const* SIXTRL_RESTRICT node_id_str ) const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN node_index_t nodeIndex(
            std::string const& SIXTRL_RESTRICT_REF
                node_id_str ) const SIXTRL_NOEXCEPT;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        SIXTRL_HOST_FN node_index_t
        minAvailableNodeIndex() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN node_index_t
        maxAvailableNodeIndex() const SIXTRL_NOEXCEPT;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        SIXTRL_HOST_FN size_type availableNodeIndices(
            size_type const max_num_node_indices,
            node_index_t* SIXTRL_RESTRICT node_indices_begin
        ) const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN size_type availableNodeIds(
            size_type const max_num_node_ids,
            node_id_t* SIXTRL_RESTRICT node_ids_begin ) const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN size_type availableBaseNodeInfos(
            size_type const max_num_node_infos,
            node_info_base_t const** SIXTRL_RESTRICT ptr_node_infos_begin
        ) const SIXTRL_NOEXCEPT;

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

        SIXTRL_HOST_FN status_t selectedNodeIdStr(
            char* SIXTRL_RESTRICT node_str,
            size_type const max_str_length ) const SIXTRL_NOEXCEPT;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        SIXTRL_HOST_FN bool usesAutoSelect() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN status_t selectNode( node_index_t const index );
        SIXTRL_HOST_FN status_t selectNode( node_id_t const& node_id );
        SIXTRL_HOST_FN status_t selectNode(
            platform_id_t const platform_idx, device_id_t const device_idx );

        SIXTRL_HOST_FN status_t selectNode( char const* node_id_str );
        SIXTRL_HOST_FN status_t selectNode( std::string const& node_id_str );

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        SIXTRL_HOST_FN bool canChangeSelectedNode() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN bool
        canDirectlyChangeSelectedNode() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN status_t changeSelectedNode(
            node_index_t const new_selected_node_index );

        SIXTRL_HOST_FN status_t changeSelectedNode(
            node_index_t const current_selected_node_index,
            node_index_t const new_selected_node_index );

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        SIXTRL_HOST_FN bool canUnselectNode() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN status_t unselectNode();
        SIXTRL_HOST_FN status_t unselectNode( node_index_t const index );
        SIXTRL_HOST_FN status_t unselectNode( node_id_t const& node_id );
        SIXTRL_HOST_FN status_t unselectNode(
            platform_id_t const platform_idx, device_id_t const device_idx );

        SIXTRL_HOST_FN status_t unselectNode( char const* node_id_str );
        SIXTRL_HOST_FN status_t unselectNode( std::string const& node_id_str );

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

        SIXTRL_HOST_FN virtual ~NodeControllerBase() SIXTRL_NOEXCEPT;

        protected:

        using ptr_node_info_base_t = std::unique_ptr< node_info_base_t >;

        static SIXTRL_CONSTEXPR_OR_CONST size_type NODE_ID_STR_CAPACITY =
            size_type{ 18 };

        SIXTRL_HOST_FN NodeControllerBase(
            arch_id_t const arch_id,
            const char *const SIXTRL_RESTRICT arch_str,
            const char *const SIXTRL_RESTRICT config_str = nullptr );

        SIXTRL_HOST_FN NodeControllerBase(
            NodeControllerBase const& other ) = default;

        SIXTRL_HOST_FN NodeControllerBase(
            NodeControllerBase&& other ) = default;

        SIXTRL_HOST_FN NodeControllerBase& operator=(
            NodeControllerBase const& rhs ) = default;

        SIXTRL_HOST_FN NodeControllerBase& operator=(
            NodeControllerBase&& rhs ) = default;

        SIXTRL_HOST_FN virtual void doClear() override;

        SIXTRL_HOST_FN virtual status_t doSelectNode(
            node_index_t const node_index );

        SIXTRL_HOST_FN virtual status_t doChangeSelectedNode(
            node_index_t const current_selected_node_index,
            node_index_t const new_selected_node_index );

        SIXTRL_HOST_FN virtual status_t doUnselectNode(
            node_index_t const selected_node_index );

        /* ----------------------------------------------------------------- */

        SIXTRL_HOST_FN node_index_t doFindAvailableNodesIndex(
            platform_id_t const platform_idx,
            device_id_t const device_idx ) const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN node_index_t doFindAvailableNodesIndex(
            char const* SIXTRL_RESTRICT node_id_str ) const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN void doClearAvailableNodes() SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN void doRemoveNodeFromSelection(
            node_index_t const selected_node_index ) SIXTRL_NOEXCEPT;

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

        /* ----------------------------------------------------------------- */

        SIXTRL_HOST_FN void doSetCanDirectlyChangeSelectedNodeFlag(
            bool const can_directly_change_selected_node ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN void doSetCanUnselectNodeFlag(
            bool const can_unselect_node ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN void doSetUseAutoSelectFlag(
            bool const use_autoselect ) SIXTRL_NOEXCEPT;

        private:

        SIXTRL_HOST_FN void doClearNodeControllerBaseImpl() SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN status_t doSelectNodeNodeControllerBaseImpl(
            node_index_t const node_index ) SIXTRL_NOEXCEPT;

        std::vector< ptr_node_info_base_t > m_available_nodes;
        std::vector< char > m_selected_node_id_str;

        node_id_t const* m_ptr_default_node_id;
        node_id_t const* m_ptr_selected_node_id;

        node_index_t m_min_available_node_index;
        node_index_t m_max_available_node_index;

        bool m_can_directly_change_selected_node;
        //bool m_node_change_requires_kernels;
        bool m_can_unselect_node;
        bool m_use_autoselect;
    };

    SIXTRL_STATIC SIXTRL_HOST_FN NodeControllerBase const* asNodeController(
        ControllerBase const* SIXTRL_RESTRICT base_controller );

    SIXTRL_STATIC SIXTRL_HOST_FN NodeControllerBase* asNodeController(
        ControllerBase* SIXTRL_RESTRICT base_controller );
}

#endif /* C++, Host */

#if defined( __cplusplus ) && !defined( _GPUCODE )
extern "C" {
#endif /* C++, Host */

#if defined( __cplusplus   ) && !defined( _GPUCODE ) && \
   !defined( __CUDA_ARCH__ ) && !defined( __CUDACC__ )

typedef SIXTRL_CXX_NAMESPACE::NodeControllerBase NS(NodeControllerBase);

#else /* C++, Host */

typedef void NS(NodeControllerBase);

#endif /* C++, Host */

#if defined( __cplusplus ) && !defined( _GPUCODE )
}
#endif /* C++, Host */

#if defined( __cplusplus   ) && !defined( _GPUCODE ) && \
   !defined( __CUDA_ARCH__ ) && !defined( __CUDACC__ )

namespace SIXTRL_CXX_NAMESPACE
{
    SIXTRL_INLINE NodeControllerBase const* asNodeController(
        ControllerBase const* SIXTRL_RESTRICT base_controller )
    {
        return ( ( base_controller != nullptr ) &&
                 ( base_controller->usesNodes() ) )
            ? static_cast< NodeControllerBase const* >( base_controller )
            : nullptr;
    }

    SIXTRL_INLINE NodeControllerBase* asNodeController(
        ControllerBase* SIXTRL_RESTRICT base_controller )
    {
        ControllerBase const* cptr_base_ctrl = base_controller;

        return const_cast< NodeControllerBase* >(
            asNodeController( cptr_base_ctrl ) );
    }
}

#endif /* C++, Host */

#endif /* SIXTRACKLIB_COMMON_CONTROL_NODE_CONTROLLER_BASE_HPP__ */

/* end: sixtracklib/common/control/node_controller_base.hpp */
