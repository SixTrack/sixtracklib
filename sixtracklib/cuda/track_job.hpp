#ifndef SIXTRACKLIB_CUDA_TRACK_JOB_HPP__
#define SIXTRACKLIB_CUDA_TRACK_JOB_HPP__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #if defined( __cplusplus ) && !defined( _GPUCODE ) && \
       !defined( __CUDA_ARCH__ )
        #include <cstddef>
        #include <cstdlib>
    #endif /* C++, Host */
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/control/definitions.h"
    #include "sixtracklib/common/control/arch_base.hpp"
    #include "sixtracklib/common/control/kernel_config_base.hpp"
    #include "sixtracklib/common/track/definitions.h"
    #include "sixtracklib/common/track/track_job_ctrl_arg_base.hpp"

    #if defined( __cplusplus ) && !defined( _GPUCODE ) && \
       !defined( __CUDA_ARCH__ )
        #include "sixtracklib/common/buffer.hpp"
    #endif /* C++, Host */
    #include "sixtracklib/common/buffer.h"

    #include "sixtracklib/cuda/definitions.h"
    #include "sixtracklib/cuda/controller.hpp"
    #include "sixtracklib/cuda/argument.hpp"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus ) && !defined( _GPUCODE ) && !defined( __CUDA_ARCH__ )
namespace SIXTRL_CXX_NAMESPACE
{
    class CudaTrackJob : public SIXTRL_CXX_NAMESPACE::TrackJobCtrlArgBase
    {
        private:

        using _base_track_job_t = SIXTRL_CXX_NAMESPACE::TrackJobCtrlArgBase;

        public:

        using cuda_controller_t     = SIXTRL_CXX_NAMESPACE::CudaController;
        using cuda_argument_t       = SIXTRL_CXX_NAMESPACE::CudaArgument;
        using cuda_kernel_config_t  = SIXTRL_CXX_NAMESPACE::CudaKernelConfig;

        using arch_id_t             = _base_track_job_t::arch_id_t;
        using buffer_t              = _base_track_job_t::buffer_t;
        using c_buffer_t            = _base_track_job_t::c_buffer_t;
        using size_type             = _base_track_job_t::size_type;
        using kernel_id_t           = _base_track_job_t::kernel_id_t;
        using track_status_t        = _base_track_job_t::track_status_t;
        using status_t              = _base_track_job_t::status_t;
        using elem_by_elem_config_t = _base_track_job_t::elem_by_elem_config_t;
        using output_buffer_flag_t  = _base_track_job_t::output_buffer_flag_t;
        using collect_flag_t        = _base_track_job_t::collect_flag_t;

        using cuda_node_info_t      = cuda_controller_t::node_info_t;
        using node_id_t             = cuda_controller_t::node_id_t;
        using platform_id_t         = cuda_controller_t::platform_id_t;
        using device_id_t           = cuda_controller_t::device_id_t;

        SIXTRL_HOST_FN explicit CudaTrackJob(
            const char *const SIXTRL_RESTRICT node_id_str = nullptr,
            const char *const SIXTRL_RESTRICT config_str = nullptr );

        SIXTRL_HOST_FN explicit CudaTrackJob(
            std::string const& SIXTRL_RESTRICT_REF node_id_str,
            std::string const& SIXTRL_RESTRICT_REF config_str = std::string{} );

        SIXTRL_HOST_FN CudaTrackJob(
            const char *const SIXTRL_RESTRICT node_id_str,
            c_buffer_t* SIXTRL_RESTRICT particles_buffer,
            c_buffer_t* SIXTRL_RESTRICT beam_elements_buffer,
            c_buffer_t* SIXTRL_RESTRICT ptr_output_buffer = nullptr,
            size_type const until_turn_elem_by_elem = size_type{ 0 },
            const char *const SIXTRL_RESTRICT config_str = nullptr );

        SIXTRL_HOST_FN CudaTrackJob(
            std::string const& SIXTRL_RESTRICT_REF node_id_str,
            buffer_t& SIXTRL_RESTRICT_REF particles_buffer,
            buffer_t& SIXTRL_RESTRICT_REF beam_elements_buffer,
            buffer_t* SIXTRL_RESTRICT ptr_output_buffer = nullptr,
            size_type const until_turn_elem_by_elem  = size_type{ 0 },
            std::string const& config_str = std::string{} );

        SIXTRL_HOST_FN CudaTrackJob( CudaTrackJob const& other ) = default;
        SIXTRL_HOST_FN CudaTrackJob( CudaTrackJob&& other ) = default;

        SIXTRL_HOST_FN CudaTrackJob& operator=( CudaTrackJob const& ) =default;
        SIXTRL_HOST_FN CudaTrackJob& operator=( CudaTrackJob&& rhs ) = default;

        SIXTRL_HOST_FN virtual ~CudaTrackJob() SIXTRL_NOEXCEPT;

        /* ----------------------------------------------------------------- */

        SIXTRL_HOST_FN cuda_controller_t& cudaController() SIXTRL_RESTRICT;
        SIXTRL_HOST_FN cuda_controller_t const&
        cudaController() const SIXTRL_RESTRICT;

        SIXTRL_HOST_FN cuda_controller_t* ptrCudaController() SIXTRL_RESTRICT;

        SIXTRL_HOST_FN cuda_controller_t const*
        ptrCudaController() const SIXTRL_RESTRICT;

        /* ----------------------------------------------------------------- */

        SIXTRL_HOST_FN cuda_argument_t& particlesArg() SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN cuda_argument_t const&
        particlesArg() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN cuda_argument_t*
        ptrParticlesArg() SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN cuda_argument_t const*
        ptrParticlesArg() const SIXTRL_NOEXCEPT;

        /* ----------------------------------------------------------------- */

        SIXTRL_HOST_FN cuda_argument_t& beamElementsArg() SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN cuda_argument_t const&
        beamElementsArg() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN cuda_argument_t* ptrBeamElementsArg() SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN cuda_argument_t const*
        ptrBeamElementsArg() const SIXTRL_NOEXCEPT;

        /* ----------------------------------------------------------------- */

        SIXTRL_HOST_FN cuda_argument_t& outputArg() SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN cuda_argument_t const&
        outputArg() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN cuda_argument_t* ptrOutputArg() SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN cuda_argument_t const*
        ptrOutputArg() const SIXTRL_NOEXCEPT;

        /* ----------------------------------------------------------------- */

        SIXTRL_HOST_FN cuda_argument_t& elemByElemConfigArg() SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN cuda_argument_t const&
        elemByElemConfigArg() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN cuda_argument_t const*
        ptrElemByElemConfigArg() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN cuda_argument_t*
        ptrElemByElemConfigArg() SIXTRL_NOEXCEPT;

        /* ----------------------------------------------------------------- */

        protected:

        using cuda_ctrl_store_t     = std::unique_ptr< cuda_controller_t >;
        using cuda_arg_store_t      = std::unique_ptr< cuda_argument_t >;

        SIXTRL_HOST_FN void doSetTotalNumParticles(
            size_type const total_num_particles ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN virtual bool doPrepareController(
            char const* SIXTRL_RESTRICT device_id_str,
            const char *const SIXTRL_RESTRICT ptr_config_str );

        SIXTRL_HOST_FN virtual bool doPrepareParticlesStructures(
            c_buffer_t* SIXTRL_RESTRICT ptr_particles_buffer ) override;

        SIXTRL_HOST_FN virtual bool doPrepareBeamElementsStructures(
            c_buffer_t* SIXTRL_RESTRICT ptr_beam_elem_buffer ) override;

        SIXTRL_HOST_FN virtual bool doPrepareOutputStructures(
            c_buffer_t* SIXTRL_RESTRICT particles_buffer,
            c_buffer_t* SIXTRL_RESTRICT beam_elem_buffer,
            c_buffer_t* SIXTRL_RESTRICT ptr_output_buffer,
            size_type const until_turn_elem_by_elem ) override;

        SIXTRL_HOST_FN virtual bool doAssignOutputBufferToBeamMonitors(
            c_buffer_t* SIXTRL_RESTRICT beam_elem_buffer,
            c_buffer_t* SIXTRL_RESTRICT output_buffer ) override;

        SIXTRL_HOST_FN virtual bool doReset(
            c_buffer_t* SIXTRL_RESTRICT particles_buffer,
            c_buffer_t* SIXTRL_RESTRICT beam_elem_buffer,
            c_buffer_t* SIXTRL_RESTRICT ptr_output_buffer,
            size_type const until_turn_elem_by_elem ) override;

        SIXTRL_HOST_FN virtual track_status_t doTrackUntilTurn(
            size_type const until_turn ) override;

        SIXTRL_HOST_FN virtual track_status_t doTrackElemByElem(
            size_type const until_turn ) override;

        SIXTRL_HOST_FN virtual track_status_t doTrackLine(
            size_type const line_begin_idx, size_type const line_end_idx,
            bool const finish_turn ) override;

        SIXTRL_HOST_FN virtual collect_flag_t doCollect(
            collect_flag_t const flags ) override;

        SIXTRL_HOST_FN virtual bool doParseConfigStr(
            const char *const SIXTRL_RESTRICT config_str ) override;

        SIXTRL_HOST_FN void doUpdateStoredController(
            cuda_ctrl_store_t&& ptr_stored_controller );

        SIXTRL_HOST_FN void doUpdateStoredParticlesArg(
            cuda_arg_store_t&& ptr_stored_particle_arg );

        SIXTRL_HOST_FN void doUpdateStoredBeamElementsArg(
            cuda_arg_store_t&& ptr_stored_beam_elements_arg );

        SIXTRL_HOST_FN void doUpdateStoredOutputArg(
            cuda_arg_store_t&& ptr_stored_output_arg );

        SIXTRL_HOST_FN void doUpdateStoredClElemByElemConfigBuffer(
            cuda_arg_store_t&& ptr_stored_elem_by_elem_conf_arg );

        private:

        template< typename PartSetIndexIter >
        SIXTRL_HOST_FN bool doInitCudaTrackJob(
            const char *const SIXTRL_RESTRICT node_id_str,
            c_buffer_t* SIXTRL_RESTRICT particles_buffer,
            PartSetIndexIter particle_set_indices_begin,
            PartSetIndexIter particle_set_indices_end,
            c_buffer_t* SIXTRL_RESTRICT beam_elements_buffer,
            c_buffer_t* SIXTRL_RESTRICT ptr_output_buffer,
            size_type const until_turn_elem_by_elem,
            const char *const SIXTRL_RESTRICT config_str );

        SIXTRL_HOST_FN bool doPrepareControllerCudaImpl(
            const char *const SIXTRL_RESTRICT device_id_str,
            const char *const SIXTRL_RESTRICT ptr_config_str );

        SIXTRL_HOST_FN void doParseConfigStrCudaImpl(
            const char *const SIXTRL_RESTRICT config_str );

        SIXTRL_HOST_FN bool doPrepareParticlesStructuresCudaImpl(
            c_buffer_t* SIXTRL_RESTRICT ptr_particles_buffer );

        SIXTRL_HOST_FN bool doPrepareBeamElementsStructuresCudaImp(
            c_buffer_t* SIXTRL_RESTRICT ptr_beam_elem_buffer );

        SIXTRL_HOST_FN bool doPrepareOutputStructuresCudaImpl(
            c_buffer_t* SIXTRL_RESTRICT particles_buffer,
            c_buffer_t* SIXTRL_RESTRICT beam_elem_buffer,
            c_buffer_t* SIXTRL_RESTRICT ptr_output_buffer,
            size_type const until_turn_elem_by_elem );

        SIXTRL_HOST_FN bool doAssignOutputBufferToBeamMonitorsCudaImp(
            c_buffer_t* SIXTRL_RESTRICT beam_elem_buffer,
            c_buffer_t* SIXTRL_RESTRICT output_buffer );

        SIXTRL_HOST_FN bool doResetCudaImp(
            c_buffer_t* SIXTRL_RESTRICT particles_buffer,
            c_buffer_t* SIXTRL_RESTRICT beam_elem_buffer,
            c_buffer_t* SIXTRL_RESTRICT ptr_output_buffer,
            size_type const until_turn_elem_by_elem );

        cuda_ctrl_store_t       m_stored_controller;
        cuda_arg_store_t        m_stored_particles_arg;
        cuda_arg_store_t        m_stored_beam_elements_arg;
        cuda_arg_store_t        m_stored_output_arg;
        cuda_arg_store_t        m_stored_elem_by_elem_conf_arg;

        kernel_id_t             m_remap_kernel_id;
        kernel_id_t             m_assign_outbuffer_kernel_id;
        kernel_id_t             m_track_until_kernel_id;
        kernel_id_t             m_track_line_kernel_id;
        kernel_id_t             m_track_elem_by_elem_kernel_id;
    };
}

typedef SIXTRL_CXX_NAMESPACE::CudaTrackJob NS(CudaTrackJob);

#else /* C++, Host */

typedef void NS(CudaTrackJob);

#endif /* C++, Host */

#endif /* SIXTRACKLIB_CUDA_TRACK_JOB_HPP__ */