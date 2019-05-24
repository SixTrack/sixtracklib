#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/track/track_job_nodectrl_arg_base.hpp"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus ) && !defined( _GPUCODE )
    #if !defined( SIXTRL_NO_INCLUDES )
        #include "sixtracklib/common/definitions.h"
        #include "sixtracklib/common/control/definitions.h"
        #include "sixtracklib/common/control/node_id.hpp"
        #include "sixtracklib/common/control/controller_base.hpp"
        #include "sixtracklib/common/control/node_controller_base.hpp"
        #include "sixtracklib/common/track/definitions.h"
        #include "sixtracklib/common/track/track_job_base.hpp"
    #endif /* defined( __cplusplus ) */

#endif /* C++, Host */

#if defined( __cplusplus ) && !defined( _GPUCODE )

namespace st = SIXTRL_CXX_NAMESPACE;

namespace SIXTRL_CXX_NAMESPACE
{
    using _this_t = TrackJobNodeCtrlArgBase;
    using _base_t = st::TrackJobCtrlArgBase;

    TrackJobNodeCtrlArgBase::TrackJobNodeCtrlArgBase(
        TrackJobNodeCtrlArgBase::arch_id_t const arch_id,
        char const* SIXTRL_RESTRICT arch_str,
        char const* SIXTRL_RESTRICT config_str ) :
        st::TrackJobCtrlArgBase( arch_id, arch_str, config_str )
    {

    }

    TrackJobNodeCtrlArgBase::node_controller_base_t const*
        TrackJobNodeCtrlArgBase::ptrNodeControllerBase() const SIXTRL_NOEXCEPT
    {
        return st::asNodeController( this->ptrControllerBase() );
    }

    TrackJobNodeCtrlArgBase::node_controller_base_t*
        TrackJobNodeCtrlArgBase::ptrNodeControllerBase() SIXTRL_NOEXCEPT
    {
        return st::asNodeController( this->ptrControllerBase() );
    }

    /* --------------------------------------------------------------------- */

    _this_t::status_t _this_t::doSelectNodeOnController(
        _this_t::node_index_t const node_index )
    {
        using node_ctrl_t = _this_t::node_controller_base_t;

        node_ctrl_t* ptr_node_ctrl = this->ptrNodeControllerBase();

        return ( ptr_node_ctrl != nullptr )
            ? ptr_node_ctrl->selectNode( node_index )
            : st::ARCH_STATUS_GENERAL_FAILURE;
    }

    _this_t::status_t  _this_t::doUnselectNodeOnController(
        _this_t::node_index_t const selected_node_index )
    {
        using node_ctrl_t = _this_t::node_controller_base_t;

        node_ctrl_t* ptr_node_ctrl = this->ptrNodeControllerBase();

        return ( ptr_node_ctrl != nullptr )
            ? ptr_node_ctrl->unselectNode( selected_node_index )
            : st::ARCH_STATUS_GENERAL_FAILURE;
    }

    _this_t::status_t  _this_t::doChangeSelectedNodeOnController(
        _this_t::node_index_t const currently_selected_index,
        _this_t::node_index_t const new_selected_node_index )
    {
        using node_ctrl_t = _this_t::node_controller_base_t;

        node_ctrl_t* ptr_node_ctrl = this->ptrNodeControllerBase();

        return ( ptr_node_ctrl != nullptr )
            ? ptr_node_ctrl->changeSelectedNode( currently_selected_index,
                    new_selected_node_index )
            : st::ARCH_STATUS_GENERAL_FAILURE;
    }
}

#endif /* C++, Host */

/* end: sixtracklib/common/track/track_job_nodectrl_arg_base.cpp */
