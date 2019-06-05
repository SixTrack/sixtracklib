#ifndef SIXTRACKLIB_COMMON_TACK_JOB_NODECTRL_ARG_BASE_HPP__
#define SIXTRACKLIB_COMMON_TACK_JOB_NODECTRL_ARG_BASE_HPP__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #if defined( __cplusplus ) && !defined( _GPUCODE ) && \
       !defined( __CUDA_ARCH__ )
        #include <cstddef>
        #include <cstdlib>
        #include <cstring>
        #include <string>
    #endif /* C++, Host */
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/control/definitions.h"
    #include "sixtracklib/common/control/controller_base.hpp"
    #include "sixtracklib/common/control/node_controller_base.hpp"
    #include "sixtracklib/common/control/argument_base.hpp"
    #include "sixtracklib/common/track/definitions.h"
    #include "sixtracklib/common/track/track_job_base.hpp"
    #include "sixtracklib/common/track/track_job_ctrl_arg_base.hpp"

    #if defined( __cplusplus )
        #include "sixtracklib/common/buffer.hpp"
    #endif /* defined( __cplusplus ) */

    #include "sixtracklib/common/buffer.h"
    #include "sixtracklib/common/particles.h"
    #include "sixtracklib/common/output/output_buffer.h"
    #include "sixtracklib/common/output/elem_by_elem_config.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus ) && !defined( _GPUCODE )

namespace SIXTRL_CXX_NAMESPACE
{
    class TrackJobNodeCtrlArgBase :
        public SIXTRL_CXX_NAMESPACE::TrackJobCtrlArgBase
    {
        private:

        using _base_track_job_t = SIXTRL_CXX_NAMESPACE::TrackJobCtrlArgBase;

        public:

        using node_controller_base_t =
            SIXTRL_CXX_NAMESPACE::NodeControllerBase;

        using kernel_id_t           = controller_base_t::kernel_id_t;
        using kernel_config_base_t  = controller_base_t::kernel_config_base_t;

        SIXTRL_HOST_FN virtual ~TrackJobNodeCtrlArgBase() = default;

        SIXTRL_HOST_FN node_controller_base_t const*
        ptrNodeControllerBase() const SIXTRL_NOEXCEPT;

        /* ================================================================ */

        SIXTRL_HOST_FN bool hasSelectedNode() const SIXTRL_NOEXCEPT;

        protected:

        using stored_node_ctrl_base_t =
            std::unique_ptr< node_controller_base_t >;

        using node_index_t = node_controller_base_t::node_index_t;

        SIXTRL_HOST_FN TrackJobNodeCtrlArgBase(
            arch_id_t const arch_id, char const* SIXTRL_RESTRICT arch_str,
            char const* SIXTRL_RESTRICT config_str = nullptr );

        SIXTRL_HOST_FN TrackJobNodeCtrlArgBase(
            TrackJobNodeCtrlArgBase const& other ) = default;

        SIXTRL_HOST_FN TrackJobNodeCtrlArgBase(
            TrackJobNodeCtrlArgBase&& other ) = default;

        SIXTRL_HOST_FN TrackJobNodeCtrlArgBase& operator=(
            TrackJobNodeCtrlArgBase const& rhs ) = default;

        SIXTRL_HOST_FN TrackJobNodeCtrlArgBase& operator=(
            TrackJobNodeCtrlArgBase&& rhs )  = default;

        /* ----------------------------------------------------------------- */

        SIXTRL_HOST_FN node_controller_base_t*
            ptrNodeControllerBase() SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN status_t doSelectNodeOnController(
            node_index_t const node_index );

        SIXTRL_HOST_FN status_t doUnselectNodeOnController(
            node_index_t const selected_node_index );

        SIXTRL_HOST_FN status_t doChangeSelectedNodeOnController(
            node_index_t const currently_selected_node_index,
            node_index_t const new_selected_node_index );
    };
}

#else /* C++, Host */

#endif /* C++, Host */

#endif /* SIXTRACKLIB_COMMON_TACK_JOB_NODECTRL_ARG_BASE_HPP__ */

/* end: sixtracklib/common/track/track_job_nodectrl_arg_base.hpp */
