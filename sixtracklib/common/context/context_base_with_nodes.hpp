#ifndef SIXTRACKLIB_COMMON_CONTEXT_NODE_BASE_HPP__
#define SIXTRACKLIB_COMMON_CONTEXT_NODE_BASE_HPP__

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
    #include "sixtracklib/common/context/definitions.h"
    #include "sixtracklib/common/context/node_id.h"
    #include "sixtracklib/common/context/node_info.h"
    #include "sixtracklib/common/context/compute_arch.h"
    #include "sixtracklib/common/context/context_base.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus ) && !defined( _GPUCODE ) && !defined( __CUDA_ARCH__ )

namespace SIXTRL_CXX_NAMESPACE
{
    class ContextOnNodesBase : public SIXTRL_CXX_NAMESPACE::ContextBase
    {
        private:

        using _context_base_t   = ContextBase;

        public:

        using size_type         = _context_base_t::size_type;
        using arch_id_t         = _context_base_t::arch_id_t;

        using node_id_t         = SIXTRL_CXX_NAMESPACE::NodeId;
        using node_info_base_t  = SIXTRL_CXX_NAMESPACE::NodeInfoBase;
        using platform_id_t     = node_id_t::platform_id_t;
        using device_id_t       = node_id_t::device_id_t;

        SIXTRL_HOST_FN size_type numAvailableNodes() const SIXTRL_NOEXCEPT;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        SIXTRL_HOST_FN bool hasDefaultNode() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN node_id_t const*
        ptrDefaultNodeId() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN node_info_base_t const*
        defaultNodeInfoBase() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN size_type defaultNodeIndex() const SIXTRL_NOEXCEPT;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        SIXTRL_HOST_FN bool isNodeAvailable(
            size_type const node_index ) const SIXTRL_RESTRICT;

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
            size_type const node_index ) const SIXTRL_NOEXCEPT;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        SIXTRL_HOST_FN node_id_t const* ptrNodeId(
            char const* SIXTRL_RESTRICT node_id_str ) const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN node_id_t const* ptrNodeId( std::string const&
            SIXTRL_RESTRICT_REF node_id_str ) const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN node_id_t const* ptrNodeId(
            platform_id_t const platform_index,
            device_id_t const device_index ) const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN node_id_t const* ptrNodeId(
            size_type const index ) const SIXTRL_NOEXCEPT;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        SIXTRL_HOST_FN node_info_base_t const* ptrNodesInfoBase(
            size_type const index ) const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN node_info_base_t const* ptrNodesInfoBase(
            platform_id_t const platform_idx,
            device_id_t const device_idx ) const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN node_info_base_t const* ptrNodesInfoBase(
            node_id_t const& node_id ) const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN node_info_base_t const* ptrNodesInfoBase(
            char const* SIXTRL_RESTRICT node_id_str ) const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN node_info_base_t const* ptrNodesInfoBase(
            std::string const& SIXTRL_RESTRICT_REF node_id_str
            ) const SIXTRL_NOEXCEPT;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        SIXTRL_HOST_FN bool hasSelectedNode() const SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN size_type selectedNodeIndex() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN node_id_t const* ptrSelectedNodeId()
            const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN node_info_t const*
        ptrSelectedNodeInfoBase() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN std::string selectedNodeIdStr() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN char const*
        ptrSelectedNodeIdStr() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN bool selectedNodeIdStr(
            char* SIXTRL_RESTRICT node_id_str,
            size_type const max_str_length ) const SIXTRL_NOEXCEPT;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        SIXTRL_HOST_FN bool selectNode( node_id_t const& node_id );
        SIXTRL_HOST_FN bool selectNode( platform_id_t const platform_idx,
                         device_id_t const device_idx );

        SIXTRL_HOST_FN bool selectNode( char const* node_id_str );
        SIXTRL_HOST_FN bool selectNode( std::string const& node_id_str );
        SIXTRL_HOST_FN bool selectNode( size_type const index );

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        SIXTRL_HOST_FN void printAvailableNodesInfo() const;

        SIXTRL_HOST_FN void printAvailableNodesInfo(
            std::ostream& SIXTRL_RESTRICT_REF os ) const;

        SIXTRL_HOST_FN void printAvailableNodesInfo(
            ::FILE* SIXTRL_RESTRICT output ) const;

        SIXTRL_HOST_FN std::string availableNodesInfoToString() const;

        SIXTRL_HOST_FN virtual ~ContextOnNodesBase() SIXTRL_NOEXCEPT;

        protected:

        using std::unique_ptr< node_info_base_t > ptr_node_info_base_t;

        SIXTRL_HOST_FN ContextOnNodesBase(
            arch_id_t const arch_id, const char *const arch_str,
            const char *const SIXTRL_RESTRICT config_str = nullptr );

        SIXTRL_HOST_FN ContextOnNodesBase(
            ContextOnNodesBase const& other ) = default;

        SIXTRL_HOST_FN ContextOnNodesBase(
            ContextOnNodesBase&& other ) = default;

        SIXTRL_HOST_FN ContextOnNodesBase& operator=(
            ContextOnNodesBase const& rhs ) = default;

        SIXTRL_HOST_FN ContextOnNodesBase& operator=(
            ContextOnNodesBase&& rhs ) = default;

        SIXTRL_HOST_FN virtual void doClear() override;
        SIXTRL_HOST_FN virtual bool doSelectNode( size_type node_index );

        SIXTRL_HOST_FN virtual size_type doGetDefaultNodeIndex() const;

        SIXTRL_HOST_FN size_type doFindAvailableNodesIndex(
            platform_id_t const platform_idx,
            device_id_t const device_idx ) const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN size_type doFindAvailableNodesIndex(
            char const* SIXTRL_RESTRICT node_id_str ) const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN void doClearAvailableNodes() SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN size_type doAppendAvailableNodeInfoBase(
            ptr_node_info_base_t&& SIXTRL_RESTRICT_REF ptr_node_info_base );

        private:

        SIXTRL_HOST_FN void doClearOnNodesBaseImpl() SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN size_type
        doGetDefaultNodeIndexOnNodesBaseImpl() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN bool doSelectNodeOnNodesBaseImpl(
            size_type const node_index ) SIXTRL_NOEXCEPT;

        std::vector< ptr_node_info_base_t > m_available_nodes;
        std::vector< char > m_selected_node_id_str;

        node_id_t const* m_ptr_default_node_id;
        node_id_t const* m_ptr_selected_node_id;

        int64_t m_selected_node_index;
    };
}

#endif /* C++, Host */

#if defined( __cplusplus ) && !defined( _GPUCODE ) && !defined( __CUDA_ARCH__ )

extern "C" {

typedef SIXTRL_CXX_NAMESPACE::ContextOnNodesBase NS(ContextOnNodesBase);

typedef SIXTRL_CXX_NAMESPACE::ContextOnNodesBase::node_id_t NS(node_id_t);

typedef SIXTRL_CXX_NAMESPACE::ContextOnNodesBase::node_info_base_t
        NS(node_info_base_t);

#else /* C++, Host */

typedef void NS(ContextNodeBase);

typedef NS(NodeId)              NS(node_id_t);
typedef NS(ComputeNodeInfo)     NS(node_info_base_t);

#endif /* C++, Host */

/* ************************************************************************* */
/* ********      Implementtion of inline and template methods       ******** */
/* ************************************************************************* */

#if defined( __cplusplus ) && !defined( _GPUCODE ) && !defined( __CUDA_ARCH__ )

namespace SIXTRL_CXX_NAMESPACE
{
    template< typename IdIter, typename InfoIter >
    ContextOnNodesBase::size_type
    ContextOnNodesBase::doAppendAvailableNodeRange(
        IdIter id_begin, IdIter id_end, InfoIter info_it )
    {
        using _this_t = ContextOnNodesBase;
        using  size_t = _this_t::size_type;

        size_t num_additional_nodes = size_t{ 0 };
        bool can_select_node = false;

        std::ptrdiff_t const temp_num_in = std::distance( id_begin, id_end );

        if( temp_num_in > std::ptrdiff_t{ 0 } )
        {
            size_t next_node_idx = this->numAvailableNodes();
            IdIter it = id_begin;

            for(  ; it != id_end ; ++it, ++info_it, ++next_node_idx )
            {
                size_t const next_node_index = this->numAvailableNodes();

                if( next_node_idx ==
                    this->doAppendAvailableNode( *it, *info_it ) )
                {
                    ++num_additional_nodes;
                }
            }
        }

        return num_additional_nodes;
    }
}

#endif /* C++, Host */

#endif /* SIXTRACKLIB_COMMON_CONTEXT_NODE_BASE_HPP__ */

/* end: sixtracklib/common/context/context_node_base.hpp */
