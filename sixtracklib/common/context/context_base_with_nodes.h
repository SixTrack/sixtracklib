#ifndef SIXTRACKLIB_COMMON_CONTEXT_NODE_BASE_H__
#define SIXTRACKLIB_COMMON_CONTEXT_NODE_BASE_H__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #if defined( __cplusplus )
        #include <cstddef>
        #include <cstdlib>
        #include <string>
        #include <iostream>
        #include <vector>
    #else  /* defined( __cplusplus ) */
        #include <stdbool.h>
        #include <stddef.h>
        #include <stdlib.h>
        #include <string.h>
    #endif /* defined( __cplusplus ) */

    #include <stdint.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/context/compute_arch.h"
    #include "sixtracklib/common/context/context_base.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus )

namespace SIXTRL_CXX_NAMESPACE
{
    class ContextOnNodesBase : public ContextBase
    {
        private:

        using _context_base_t   = ContextBase;

        public:

        using size_type         = _context_base_t::size_type;
        using type_id_t         = _context_base_t::type_id_t;

        using node_id_t         = ::NS(ComputeNodeId);
        using node_info_t       = ::NS(ComputeNodeInfo);
        using platform_id_t     = ::NS(comp_node_id_num_t);
        using device_id_t       = ::NS(comp_node_id_num_t);

        SIXTRL_HOST_FN size_type numAvailableNodes() const SIXTRL_NOEXCEPT;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        SIXTRL_HOST_FN node_info_t const*
        availableNodesInfoBegin() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN node_info_t const*
        availableNodesInfoEnd() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN node_info_t const*
        defaultNodeInfo() const SIXTRL_NOEXCEPT;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        SIXTRL_HOST_FN node_id_t defaultNodeId() const SIXTRL_NOEXCEPT;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        SIXTRL_HOST_FN bool isNodeAvailable(
            size_type const node_index ) const SIXTRL_RESTRICT;

        SIXTRL_HOST_FN bool isNodeAvailable(
            node_id_t const node_id ) const SIXTRL_NOEXCEPT;

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
            node_id_t const node_id ) const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN bool isDefaultNode( platform_id_t const platform_index,
            device_id_t const device_index ) const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN bool isDefaultNode(
            size_type const node_index ) const SIXTRL_NOEXCEPT;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        SIXTRL_HOST_FN node_id_t const* ptrAvailableNodesId(
            char const* SIXTRL_RESTRICT node_id_str ) const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN node_id_t const* ptrAvailableNodesId(
            std::string const& SIXTRL_RESTRICT_REF
                node_id_str ) const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN node_id_t const* ptrAvailableNodesId(
            platform_id_t const platform_index,
            device_id_t const device_index ) const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN node_id_t const* ptrAvailableNodesId(
            size_type const index ) const SIXTRL_NOEXCEPT;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        SIXTRL_HOST_FN node_info_t const* ptrAvailableNodesInfo(
            size_type const index ) const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN node_info_t const* ptrAvailableNodesInfo(
            platform_id_t const platform_idx,
            device_id_t const device_idx ) const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN node_info_t const* ptrAvailableNodesInfo(
            node_id_t const node_id ) const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN node_info_t const* ptrAvailableNodesInfo(
            char const* SIXTRL_RESTRICT node_id_str ) const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN node_info_t const* ptrAvailableNodesInfo(
            std::string const& SIXTRL_RESTRICT_REF node_id_str
            ) const SIXTRL_NOEXCEPT;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        SIXTRL_HOST_FN bool hasSelectedNode() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN node_id_t const*
            ptrSelectedNodeId() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN node_info_t const*
            ptrSelectedNodeInfo() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN std::string selectedNodeIdStr() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN char const*
        ptrSelectedNodeIdStr() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN bool selectedNodeIdStr(
            char* SIXTRL_RESTRICT node_id_str,
            size_type const max_str_length ) const SIXTRL_NOEXCEPT;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        SIXTRL_HOST_FN bool selectNode( node_id_t const node_id );
        SIXTRL_HOST_FN bool selectNode( platform_id_t const platform_idx,
                         device_id_t const device_idx );

        SIXTRL_HOST_FN bool selectNode( char const* node_id_str );
        SIXTRL_HOST_FN bool selectNode( std::string const& node_id_str );
        SIXTRL_HOST_FN bool selectNode( size_type const index );

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        SIXTRL_HOST_FN void printNodesInfo() const;

        SIXTRL_HOST_FN void printNodesInfo(
            std::ostream& SIXTRL_RESTRICT_REF os ) const;

        SIXTRL_HOST_FN void printNodesInfo(
            size_type const index ) const;

        SIXTRL_HOST_FN void printNodesInfo(
            std::ostream& SIXTRL_RESTRICT_REF os, size_type const idx ) const;

        SIXTRL_HOST_FN void printNodesInfo(
            node_id_t const node_id ) const;

        SIXTRL_HOST_FN void printNodesInfo(
            std::ostream& SIXTRL_RESTRICT_REF os,
            node_id_t const node_id ) const;

        SIXTRL_HOST_FN void printNodesInfo(
            platform_id_t const platform_idx,
            device_id_t const device_idx ) const;

        SIXTRL_HOST_FN void printNodesInfo(
            std::ostream& SIXTRL_RESTRICT_REF os,
            platform_id_t const platform_idx,
            device_id_t const device_idx ) const;

        SIXTRL_HOST_FN void printNodesInfo(
            char const* SIXTRL_RESTRICT node_id_str ) const;

        SIXTRL_HOST_FN void printNodesInfo(
            std::ostream& SIXTRL_RESTRICT_REF os,
            char const* SIXTRL_RESTRICT node_id_str ) const;

        SIXTRL_HOST_FN void printNodesInfo(
            std::string const& SIXTRL_RESTRICT_REF node_id_str ) const;

        SIXTRL_HOST_FN void printNodesInfo(
            std::ostream& SIXTRL_RESTRICT_REF os,
            std::string const& SIXTRL_RESTRICT_REF node_id_str ) const;

        SIXTRL_HOST_FN void printSelectedNodesInfo() const;
        SIXTRL_HOST_FN void printSelectedNodesInfo( std::ostream& os ) const;

        SIXTRL_HOST_FN friend std::ostream& operator<<(
            std::ostream& SIXTRL_RESTRICT ostream,
            ContextOnNodesBase const& SIXTRL_RESTRICT_REF context );

        virtual ~ContextOnNodesBase() SIXTRL_NOEXCEPT;

        protected:

        ContextOnNodesBase(
            type_id_t const type_id, const char *const type_id_str,
            const char *const SIXTRL_RESTRICT config_str = nullptr );

        ContextOnNodesBase( ContextOnNodesBase const& other ) = default;
        ContextOnNodesBase( ContextOnNodesBase&& other ) = default;

        ContextOnNodesBase& operator=( ContextOnNodesBase const& rhs ) = default;
        ContextOnNodesBase& operator=( ContextOnNodesBase&& rhs ) = default;

        virtual void doClear() override;
        virtual bool doSelectNode( size_type node_index );

        virtual size_type doGetDefaultNodeIndex() const;

        virtual void doPrintNodesInfo( std::ostream& ostream,
            node_info_t const& SIXTRL_RESTRICT_REF node_index ) const;

        size_type doFindAvailableNodesIndex(
            platform_id_t const platform_idx,
            device_id_t const device_idx ) const SIXTRL_NOEXCEPT;

        size_type doFindAvailableNodesIndex(
            char const* SIXTRL_RESTRICT node_id_str ) const SIXTRL_NOEXCEPT;

        void doClearAvailableNodes() SIXTRL_NOEXCEPT;

        size_type doAppendAvailableNode(
            node_id_t const& SIXTRL_RESTRICT_REF node_id,
            node_info_t const& SIXTRL_RESTRICT_REF node_info );

        template< typename IdIter, typename InfoIter >
        size_type doAppendAvailableNodeRange(
            IdIter id_begin, IdIter id_end, InfoIter info_begin );

        private:

        void doClearOnNodesBaseImpl() SIXTRL_NOEXCEPT;

        size_type doGetDefaultNodeIndexOnNodesBaseImpl() const SIXTRL_NOEXCEPT;

        bool doSelectNodeOnNodesBaseImpl(
            size_type const node_index ) SIXTRL_NOEXCEPT;

        std::vector< node_id_t >    m_available_nodes_id;
        std::vector< node_info_t >  m_available_nodes_info;

        std::vector< char >         m_selected_node_id_str;
        int64_t                     m_selected_node_index;
    };
}

#endif /* defined( __cplusplus ) */

#if defined( __cplusplus ) && !defined( _GPUCODE )
extern "C" {
#endif /* defined( __cplusplus ) && !defined( _GPUCODE ) */

#if defined( __cplusplus )

typedef SIXTRL_CXX_NAMESPACE::ContextOnNodesBase NS(ContextOnNodesBase);

typedef SIXTRL_CXX_NAMESPACE::ContextOnNodesBase::node_id_t
        NS(context_node_id_t);

typedef SIXTRL_CXX_NAMESPACE::ContextOnNodesBase::node_info_t
        NS(context_node_info_t);

typedef SIXTRL_CXX_NAMESPACE::ContextOnNodesBase::platform_id_t
        NS(context_platform_id_t);

typedef SIXTRL_CXX_NAMESPACE::ContextOnNodesBase::device_id_t
        NS(context_device_id_t);

#else /* !defined( __cplusplus ) */

typedef void NS(ContextNodeBase);

typedef NS(ComputeNodeId)       NS(context_node_id_t);
typedef NS(ComputeNodeInfo)     NS(context_node_info_t);
typedef NS(comp_node_id_num_t)  NS(context_platform_id_t);
typedef NS(comp_node_id_num_t)  NS(context_device_id_t);

#endif /* defined( __cplusplus ) */

#if defined( __cplusplus ) && !defined( _GPUCODE )
}
#endif /* defined( __cplusplus ) && !defined( _GPUCODE ) */

/* ************************************************************************* */
/* ********      Implementtion of inline and template methods       ******** */
/* ************************************************************************* */

#if defined( __cplusplus )

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

#endif /* defined( __cplusplus ) */

#endif /* SIXTRACKLIB_COMMON_CONTEXT_NODE_BASE_H__ */

/* end: sixtracklib/common/internal/context_node_base.h */
