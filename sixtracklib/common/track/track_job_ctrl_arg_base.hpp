#ifndef SIXTRACKLIB_COMMON_TRACK_TRACK_JOB_CONTROLLER_ARGUMENT_BASE_HPP__
#define SIXTRACKLIB_COMMON_TRACK_TRACK_JOB_CONTROLLER_ARGUMENT_BASE_HPP__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #if defined( __cplusplus )
        #include <algorithm>
        #include <cstddef>
        #include <cstdint>
        #include <cstdlib>
        #include <memory>
        #include <string>
        #include <vector>
    #else /* !defined( __cplusplus ) */
        #include <stdbool.h>
        #include <stddef.h>
        #include <stdint.h>
        #include <stdlib.h>
    #endif /* !defined( __cplusplus ) */
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/control/definitions.h"
    #include "sixtracklib/common/control/controller_base.hpp"
    #include "sixtracklib/common/control/argument_base.hpp"
    #include "sixtracklib/common/track/definitions.h"
    #include "sixtracklib/common/track/track_job_base.hpp"

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
    class TrackJobCtrlArgBase : public SIXTRL_CXX_NAMESPACE::TrackJobBaseNew
    {
        private:

        using _base_track_job_t     = SIXTRL_CXX_NAMESPACE::TrackJobBaseNew;

        public:

        using controller_base_t     = SIXTRL_CXX_NAMESPACE::ControllerBase;
        using argument_base_t       = SIXTRL_CXX_NAMESPACE::ArgumentBase;

        using kernel_id_t           = controller_base_t::kernel_id_t;
        using kernel_config_base_t  = controller_base_t::kernel_config_base_t;

        SIXTRL_HOST_FN virtual ~TrackJobCtrlArgBase() = default;

        /* ================================================================ */

        SIXTRL_HOST_FN bool hasRemapKernel() const SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN kernel_id_t remapKernelId() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN bool
        hasAssignOutputToBeamMonitorsKernel() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN kernel_id_t
        assignOutputToBeamMonitorsKernelId() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN bool hasTrackUntilKernel() const SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN kernel_id_t trackUntilKernelId() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN bool hasTrackLineKernel() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN kernel_id_t
        trackLineKernelId() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN bool hasTrackElemByElemKernel() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN kernel_id_t
        trackElemByElemKernelId() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN bool
        hasExtractParticlesAddressesKernel() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN kernel_id_t
        extractParticlesAddressesKernelId() const SIXTRL_NOEXCEPT;

        /* ================================================================ */

        protected:

        using stored_ctrl_base_t = std::unique_ptr< controller_base_t >;
        using stored_arg_base_t  = std::unique_ptr< argument_base_t >;

        SIXTRL_HOST_FN TrackJobCtrlArgBase(
            arch_id_t const arch_id, char const* SIXTRL_RESTRICT arch_str,
            char const* SIXTRL_RESTRICT config_str = nullptr );

        SIXTRL_HOST_FN TrackJobCtrlArgBase( TrackJobCtrlArgBase const& other );

        SIXTRL_HOST_FN TrackJobCtrlArgBase(
            TrackJobCtrlArgBase&& other ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN TrackJobCtrlArgBase& operator=(
            TrackJobCtrlArgBase const& rhs );

        SIXTRL_HOST_FN TrackJobCtrlArgBase& operator=(
            TrackJobCtrlArgBase&& rhs ) SIXTRL_NOEXCEPT;

        /* ----------------------------------------------------------------- */

        SIXTRL_HOST_FN virtual void doClear() override;

        SIXTRL_HOST_FN virtual collect_flag_t doCollect(
            collect_flag_t const flags ) override;

        /* ----------------------------------------------------------------- */

        SIXTRL_HOST_FN controller_base_t const*
        ptrControllerBase() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN controller_base_t* ptrControllerBase() SIXTRL_NOEXCEPT;

        /* ----------------------------------------------------------------- */

        SIXTRL_HOST_FN argument_base_t const*
        ptrParticlesArg() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN argument_base_t* ptrParticlesArgBase() SIXTRL_NOEXCEPT;


        SIXTRL_HOST_FN argument_base_t const*
        ptrBeamElementsArg() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN argument_base_t*
        ptrBeamElementsArgBase() SIXTRL_NOEXCEPT;


        SIXTRL_HOST_FN argument_base_t const*
        ptrOutputArg() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN argument_base_t*
        ptrOutputArgBase() SIXTRL_NOEXCEPT;


        SIXTRL_HOST_FN argument_base_t const*
        ptrElemByElemConfigArgBase() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN argument_base_t*
        ptrElemByElemConfigArgBase() SIXTRL_NOEXCEPT;


        SIXTRL_HOST_FN argument_base_t const*
        ptrParticlesAddrArgBase() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN argument_base_t*
        ptrParticlesAddrArgBase() SIXTRL_NOEXCEPT;


        SIXTRL_HOST_FN argument_base_t const*
        ptrSuccessFlagArgBase() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN argument_base_t*
        ptrSuccessFlagArgBase() SIXTRL_NOEXCEPT;

        /* ----------------------------------------------------------------- */

        SIXTRL_HOST_FN void doUpdateStoredController(
            stored_ctrl_base_t&& ptr_controller_base ) SIXTRL_NOEXCEPT;

        /* ----------------------------------------------------------------- */

        SIXTRL_HOST_FN void doUpdateStoredParticlesArg(
            stored_arg_base_t&& ptr_particles_arg ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN void doUpdateStoredBeamElementsArg(
            stored_arg_base_t&& ptr_beam_elem_arg ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN void doUpdateStoredOutputArg(
            stored_arg_base_t&& ptr_output_arg ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN void doUpdateStoredElemByElemConfigArg(
            stored_arg_base_t&& ptr_elem_by_elem_conf_arg ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN void doUpdateStoredParticlesAddrArg(
            stored_arg_base_t&& ptr_particles_addr_arg ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN void doUpdateStoredSuccessFlagArg(
            stored_arg_base_t&& ptr_success_flag_arg ) SIXTRL_NOEXCEPT;

        /* ----------------------------------------------------------------- */

        SIXTRL_HOST_FN void doSetRemapKernelId(
            kernel_id_t const id ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN void doSetAssignOutputToBeamMonitorsKernelId(
            kernel_id_t const id ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN void doSetTrackUntilKernelId(
            kernel_id_t const id ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN void doSetTrackLineKernelId(
            kernel_id_t const id ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN void doSetTrackElemByElemKernelId(
            kernel_id_t const id ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN void doSetExtractParticlesAddressesKernelId(
            kernel_id_t const id ) SIXTRL_NOEXCEPT;

        private:

        SIXTRL_HOST_FN void doClearCtrlArgBaseImpl() SIXTRL_NOEXCEPT;

        stored_ctrl_base_t      m_stored_controller;

        stored_arg_base_t       m_stored_particles_arg;
        stored_arg_base_t       m_stored_beam_elements_arg;
        stored_arg_base_t       m_stored_output_arg;
        stored_arg_base_t       m_stored_elem_by_elem_conf_arg;
        stored_arg_base_t       m_stored_particles_addr_arg;
        stored_arg_base_t       m_stored_success_flag_arg;

        kernel_id_t             m_remap_kernel_id;
        kernel_id_t             m_assign_output_bemon_kernel_id;
        kernel_id_t             m_track_until_kernel_id;
        kernel_id_t             m_track_line_kernel_id;
        kernel_id_t             m_track_elem_by_elem_kernel_id;
        kernel_id_t             m_extract_particles_addr_kernel_id;
    };
}

typedef SIXTRL_CXX_NAMESPACE::TrackJobCtrlArgBase NS(TrackJobCtrlArgBase);

#else /* defined( __cplusplus ) && !defined( _GPUCODE ) */

typedef void NS(TrackJobCtrlArgBase);

#endif /* defined( __cplusplus ) && !defined( _GPUCODE ) */

#endif /* SIXTRACKLIB_COMMON_TRACK_TRACK_JOB_CONTROLLER_ARGUMENT_BASE_HPP__ */

/*end: sixtracklib/common/track/track_job_base_ctrl_arg_base.hpp */
