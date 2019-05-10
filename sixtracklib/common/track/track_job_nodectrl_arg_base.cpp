#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/track/track_job_nodectrl_arg_base.hpp"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus ) && !defined( _GPUCODE ) && !defined( __CUDA_ARCH__ )
    #if !defined( SIXTRL_NO_INCLUDES )
        #include "sixtracklib/common/definitions.h"
        #include "sixtracklib/common/control/definitions.h"
        #include "sixtracklib/common/control/node_id.hpp"
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

    _this_t::node_controller_base_t const*
    _this_t::ptrNodeControllerBase() const SIXTRL_NOEXCEPT
    {
        return st::asControllerOnNodes( this->ptrControllerBase() );
    }

    _this_t::node_controller_base_t*
    _this_t::ptrNodeControllerBase() SIXTRL_NOEXCEPT
    {
        return st::asControllerOnNodes( this->ptrControllerBase() );
    }

    /* --------------------------------------------------------------------- */

    _this_t::TrackJobNodeCtrlArgBase( _this_t::arch_id_t const arch_id,
        char const* SIXTRL_RESTRICT arch_str,
        char const* SIXTRL_RESTRICT config_str ) :
        _base_t( arch_id, arch_str, config_str )
    {

    }

    /* --------------------------------------------------------------------- */

    bool _this_t::doSelectNodeOnController(
        _this_t::node_index_t const node_index )
    {
        using node_ctrl_t = _this_t::node_controller_base_t;

        node_ctrl_t* ptr_node_ctrl =
            st::asControllerOnNodes( this->ptrControllerBase() );

        return ( ( ptr_node_ctrl != nullptr ) &&
                 ( ptr_node_ctrl->selectNode( node_index ) ) );
    }

    bool _this_t::doUnselectNodeOnController(
        _this_t::node_index_t const selected_node_index )
    {
        using node_ctrl_t = _this_t::node_controller_base_t;

        node_ctrl_t* ptr_node_ctrl =
            st::asControllerOnNodes( this->ptrControllerBase() );

        return ( ( ptr_node_ctrl != nullptr ) &&
                 ( ptr_node_ctrl->unselectNode( node_index ) ) );
    }

    bool _this_t::doChangeSelectedNodeOnController(
        _this_t::node_index_t const currently_selected_index,
        _this_t::node_index_t const new_selected_node_index )
    {
        using node_ctrl_t = _this_t::node_controller_base_t;

        node_ctrl_t* ptr_node_ctrl =
            st::asControllerOnNodes( this->ptrControllerBase() );

        return ( ( ptr_node_ctrl != nullptr ) &&
                 ( ptr_node_ctrl->changeSelectedNode( currently_selected_index,
                    new_selected_node_index ) ) );
    }
}

#endif /* C++, Host */

/* end: sixtracklib/common/track/track_job_nodectrl_arg_base.cpp */
