#ifndef SIXTRACKLIB_CUDA_TRACK_JOB_HPP__
#define SIXTRACKLIB_CUDA_TRACK_JOB_HPP__

#if defined( __cplusplus   ) && !defined( _GPUCODE ) && \
   !defined( __CUDA_ARCH__ ) && !defined( __CUDACC__ )

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <cstddef>
    #include <cstdlib>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#endif /* C++, Host */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/control/definitions.h"
    #include "sixtracklib/common/control/arch_base.hpp"
    #include "sixtracklib/common/control/kernel_config_base.h"
    #include "sixtracklib/common/track/definitions.h"
    #include "sixtracklib/common/buffer.h"
    #include "sixtracklib/cuda/definitions.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus   ) && !defined( _GPUCODE ) && \
   !defined( __CUDA_ARCH__ ) && !defined( __CUDACC__ )

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/control/arch_base.hpp"
    #include "sixtracklib/common/control/kernel_config_base.hpp"
    #include "sixtracklib/common/track/track_job_nodectrl_arg_base.hpp"
    #include "sixtracklib/common/buffer.hpp"

    #include "sixtracklib/cuda/definitions.h"
    #include "sixtracklib/cuda/controller.hpp"
    #include "sixtracklib/cuda/argument.hpp"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

namespace SIXTRL_CXX_NAMESPACE
{
    class CudaTrackJob : public SIXTRL_CXX_NAMESPACE::TrackJobNodeCtrlArgBase
    {
        private:

        using _base_track_job_t =
            SIXTRL_CXX_NAMESPACE::TrackJobNodeCtrlArgBase;

        public:

        using cuda_controller_t     = SIXTRL_CXX_NAMESPACE::CudaController;
        using cuda_node_info_t      = cuda_controller_t::node_info_t;
        using cuda_argument_t       = SIXTRL_CXX_NAMESPACE::CudaArgument;
        using cuda_kernel_config_t  = SIXTRL_CXX_NAMESPACE::CudaKernelConfig;

        using arch_id_t             = _base_track_job_t::arch_id_t;
        using buffer_t              = _base_track_job_t::buffer_t;
        using c_buffer_t            = _base_track_job_t::c_buffer_t;
        using size_type             = _base_track_job_t::size_type;
        using kernel_id_t           = _base_track_job_t::kernel_id_t;
        using kernel_config_base_t  = _base_track_job_t::kernel_config_base_t;
        using track_status_t        = _base_track_job_t::track_status_t;
        using status_t              = _base_track_job_t::status_t;
        using elem_by_elem_config_t = _base_track_job_t::elem_by_elem_config_t;
        using output_buffer_flag_t  = _base_track_job_t::output_buffer_flag_t;
        using collect_flag_t        = _base_track_job_t::collect_flag_t;
        using push_flag_t           = _base_track_job_t::push_flag_t;
        using particles_addr_t      = _base_track_job_t::particles_addr_t;

        using node_index_t          = cuda_controller_t::node_index_t;
        using node_id_t             = cuda_controller_t::node_id_t;
        using node_info_base_t      = cuda_controller_t::node_info_base_t;
        using platform_id_t         = cuda_controller_t::platform_id_t;
        using device_id_t           = cuda_controller_t::device_id_t;

        /* ----------------------------------------------------------------- */

        SIXTRL_STATIC SIXTRL_HOST_FN size_type NumAvailableNodes();

        SIXTRL_STATIC SIXTRL_HOST_FN size_type
        GetAvailableNodeIdsList(
            size_type const max_num_node_ids,
            node_id_t* SIXTRL_RESTRICT node_ids_begin );

        SIXTRL_STATIC SIXTRL_HOST_FN size_type
        GetAvailableNodeIndicesList(
            size_type const max_num_node_indices,
            node_index_t* SIXTRL_RESTRICT node_indices_begin );

        /* ----------------------------------------------------------------- */


        SIXTRL_HOST_FN explicit CudaTrackJob(
            const char *const SIXTRL_RESTRICT config_str = nullptr );

        SIXTRL_HOST_FN explicit CudaTrackJob(
            std::string const& SIXTRL_RESTRICT_REF config_str );

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

        SIXTRL_HOST_FN CudaTrackJob(
            std::string const& SIXTRL_RESTRICT_REF node_id_str,
            buffer_t& SIXTRL_RESTRICT_REF particles_buffer,
            size_type const particle_set_index,
            buffer_t& SIXTRL_RESTRICT_REF beam_elements_buffer,
            buffer_t* SIXTRL_RESTRICT ptr_output_buffer = nullptr,
            size_type const until_turn_elem_by_elem  = size_type{ 0 },
            std::string const& config_str = std::string{} );

        SIXTRL_HOST_FN CudaTrackJob(
            char const* SIXTRL_RESTRICT node_id_str,
            c_buffer_t* SIXTRL_RESTRICT particles_buffer,
            size_type const particle_set_index,
            c_buffer_t* SIXTRL_RESTRICT beam_elements_buffer,
            c_buffer_t* SIXTRL_RESTRICT ptr_output_buffer = nullptr,
            size_type const until_turn_elem_by_elem  = size_type{ 0 },
            char const* SIXTRL_RESTRICT config_str = nullptr );

        SIXTRL_HOST_FN CudaTrackJob(
            std::string const& SIXTRL_RESTRICT_REF node_id_str,
            buffer_t& SIXTRL_RESTRICT_REF particles_buffer,
            size_type const num_particle_sets,
            size_type const* SIXTRL_RESTRICT particle_set_indices_begin,
            buffer_t& SIXTRL_RESTRICT_REF beam_elements_buffer,
            buffer_t* SIXTRL_RESTRICT ptr_output_buffer = nullptr,
            size_type const until_turn_elem_by_elem  = size_type{ 0 },
            std::string const& config_str = std::string{} );

        SIXTRL_HOST_FN CudaTrackJob(
            char const* SIXTRL_RESTRICT node_id_str,
            c_buffer_t* SIXTRL_RESTRICT particles_buffer,
            size_type const num_particle_sets,
            size_type const* SIXTRL_RESTRICT particle_set_indices_begin,
            c_buffer_t* SIXTRL_RESTRICT beam_elements_buffer,
            c_buffer_t* SIXTRL_RESTRICT ptr_output_buffer = nullptr,
            size_type const until_turn_elem_by_elem  = size_type{ 0 },
            char const* SIXTRL_RESTRICT config_str = nullptr );

        template< typename PartSetIndexIter >
        SIXTRL_HOST_FN CudaTrackJob(
            std::string const& SIXTRL_RESTRICT_REF node_id_str,
            buffer_t& SIXTRL_RESTRICT_REF particles_buffer,
            PartSetIndexIter particle_set_indices_begin,
            PartSetIndexIter particle_set_indices_end,
            buffer_t& SIXTRL_RESTRICT_REF beam_elements_buffer,
            buffer_t* SIXTRL_RESTRICT ptr_output_buffer = nullptr,
            size_type const until_turn_elem_by_elem  = size_type{ 0 },
            std::string const& config_str = std::string{} );

        template< typename PartSetIndexIter >
        SIXTRL_HOST_FN CudaTrackJob(
            char const* SIXTRL_RESTRICT node_id_str,
            c_buffer_t* SIXTRL_RESTRICT particles_buffer,
            PartSetIndexIter particle_set_indices_begin,
            PartSetIndexIter particle_set_indices_end,
            c_buffer_t* SIXTRL_RESTRICT beam_elements_buffer,
            c_buffer_t* SIXTRL_RESTRICT ptr_output_buffer = nullptr,
            size_type const until_turn_elem_by_elem  = size_type{ 0 },
            char const* SIXTRL_RESTRICT config_str = nullptr );

        SIXTRL_HOST_FN CudaTrackJob( CudaTrackJob const& other ) = default;
        SIXTRL_HOST_FN CudaTrackJob( CudaTrackJob&& other ) = default;

        SIXTRL_HOST_FN CudaTrackJob& operator=( CudaTrackJob const& ) =default;
        SIXTRL_HOST_FN CudaTrackJob& operator=( CudaTrackJob&& rhs ) = default;

        SIXTRL_HOST_FN virtual ~CudaTrackJob() SIXTRL_NOEXCEPT;

        /* ================================================================= */

        SIXTRL_HOST_FN bool hasCudaController() const SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN cuda_controller_t& cudaController();
        SIXTRL_HOST_FN cuda_controller_t const& cudaController() const;

        SIXTRL_HOST_FN cuda_controller_t* ptrCudaController() SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN cuda_controller_t const*
        ptrCudaController() const SIXTRL_NOEXCEPT;

        /* ----------------------------------------------------------------- */

        SIXTRL_HOST_FN bool hasCudaParticlesArg() const SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN cuda_argument_t& cudaParticlesArg();
        SIXTRL_HOST_FN cuda_argument_t const& cudaParticlesArg() const;

        SIXTRL_HOST_FN cuda_argument_t*
        ptrCudaParticlesArg() SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN cuda_argument_t const*
        ptrCudaParticlesArg() const SIXTRL_NOEXCEPT;

        /* ----------------------------------------------------------------- */

        SIXTRL_HOST_FN bool hasCudaBeamElementsArg() const SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN cuda_argument_t& cudaBeamElementsArg();
        SIXTRL_HOST_FN cuda_argument_t const& cudaBeamElementsArg() const;

        SIXTRL_HOST_FN cuda_argument_t const*
        ptrCudaBeamElementsArg() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN cuda_argument_t*
        ptrCudaBeamElementsArg() SIXTRL_NOEXCEPT;

        /* ----------------------------------------------------------------- */

        SIXTRL_HOST_FN bool hasCudaOutputArg() const SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN cuda_argument_t& cudaOutputArg();
        SIXTRL_HOST_FN cuda_argument_t const& cudaOutputArg() const;

        SIXTRL_HOST_FN cuda_argument_t* ptrCudaOutputArg() SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN cuda_argument_t const*
        ptrCudaOutputArg() const SIXTRL_NOEXCEPT;

        /* ----------------------------------------------------------------- */

        SIXTRL_HOST_FN bool hasCudaElemByElemConfigArg() const SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN cuda_argument_t& cudaElemByElemConfigArg();
        SIXTRL_HOST_FN cuda_argument_t const& cudaElemByElemConfigArg() const;

        SIXTRL_HOST_FN cuda_argument_t const*
        ptrCudaElemByElemConfigArg() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN cuda_argument_t*
        ptrCudaElemByElemConfigArg() SIXTRL_NOEXCEPT;

        /* ----------------------------------------------------------------- */

        SIXTRL_HOST_FN bool hasCudaDebugRegisterArg() const SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN cuda_argument_t& cudaDebugRegisterArg();
        SIXTRL_HOST_FN cuda_argument_t const& cudaDebugRegisterArg() const;

        SIXTRL_HOST_FN cuda_argument_t const*
        ptrCudaDebugRegisterArg() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN cuda_argument_t*
        ptrCudaDebugRegisterArg() SIXTRL_NOEXCEPT;

        /* ----------------------------------------------------------------- */

        SIXTRL_HOST_FN bool hasCudaParticlesAddrArg() const SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN cuda_argument_t const& cudaParticlesAddrArg() const;
        SIXTRL_HOST_FN cuda_argument_t& cudaParticlesAddrArg();

        SIXTRL_HOST_FN cuda_argument_t const*
        ptrCudaParticlesAddrArg() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN cuda_argument_t*
        ptrCudaParticlesAddrArg() SIXTRL_NOEXCEPT;

        /* ================================================================= */

        protected:

        using argument_base_t   = _base_track_job_t::argument_base_t;
        using cuda_ctrl_store_t = std::unique_ptr< cuda_controller_t >;
        using cuda_arg_store_t  = std::unique_ptr< cuda_argument_t >;

        using cuda_kernel_conf_store_t =
            std::unique_ptr< cuda_kernel_config_t >;

        SIXTRL_HOST_FN virtual status_t doPrepareController(
            char const* SIXTRL_RESTRICT config_str ) override;

        SIXTRL_HOST_FN virtual status_t doPrepareDefaultKernels(
            char const* SIXTRL_RESTRICT config_str ) override;

        /* ----------------------------------------------------------------- */

        SIXTRL_HOST_FN virtual status_t doPrepareParticlesStructures(
            c_buffer_t* SIXTRL_RESTRICT ptr_particles_buffer ) override;

        SIXTRL_HOST_FN virtual status_t doPrepareBeamElementsStructures(
            c_buffer_t* SIXTRL_RESTRICT ptr_beam_elem_buffer ) override;

        SIXTRL_HOST_FN virtual status_t doPrepareOutputStructures(
            c_buffer_t* SIXTRL_RESTRICT particles_buffer,
            c_buffer_t* SIXTRL_RESTRICT beam_elem_buffer,
            c_buffer_t* SIXTRL_RESTRICT ptr_output_buffer,
            size_type const until_turn_elem_by_elem ) override;

        SIXTRL_HOST_FN virtual status_t doAssignOutputBufferToBeamMonitors(
            c_buffer_t* SIXTRL_RESTRICT beam_elem_buffer,
            c_buffer_t* SIXTRL_RESTRICT output_buffer,
            particle_index_t const min_turn_id,
            size_type const output_buffer_offset_index ) override;

        SIXTRL_HOST_FN virtual status_t doAssignOutputBufferToElemByElemConfig(
            elem_by_elem_config_t* SIXTRL_RESTRICT elem_by_elem_config,
            c_buffer_t* SIXTRL_RESTRICT output_buffer,
            size_type const output_buffer_offset_index ) override;

        SIXTRL_HOST_FN virtual status_t doReset(
            c_buffer_t* SIXTRL_RESTRICT particles_buffer,
            c_buffer_t* SIXTRL_RESTRICT beam_elem_buffer,
            c_buffer_t* SIXTRL_RESTRICT ptr_output_buffer,
            size_type const until_turn_elem_by_elem ) override;

        /* ----------------------------------------------------------------- */

        SIXTRL_HOST_FN virtual status_t
        doSetAssignOutputToBeamMonitorsKernelId(
            kernel_id_t const id ) override;

        SIXTRL_HOST_FN virtual status_t
        doSetAssignOutputToElemByElemConfigKernelId(
            kernel_id_t const id ) override;

        SIXTRL_HOST_FN virtual status_t
        doSetTrackUntilKernelId( kernel_id_t const id ) override;

        SIXTRL_HOST_FN virtual status_t
        doSetTrackLineKernelId( kernel_id_t const id ) override;

        SIXTRL_HOST_FN virtual status_t
        doSetTrackElemByElemKernelId( kernel_id_t const id )  override;

        SIXTRL_HOST_FN virtual status_t
        doSetFetchParticlesAddressesKernelId( kernel_id_t const id ) override;

        /* ----------------------------------------------------------------- */

        SIXTRL_HOST_FN virtual status_t doFetchParticleAddresses() override;

        SIXTRL_HOST_FN virtual track_status_t doTrackUntilTurn(
            size_type const until_turn ) override;

        SIXTRL_HOST_FN virtual track_status_t doTrackElemByElem(
            size_type const until_turn ) override;

        SIXTRL_HOST_FN virtual track_status_t doTrackLine(
            size_type const line_begin_idx, size_type const line_end_idx,
            bool const finish_turn ) override;

        /* ----------------------------------------------------------------- */

        SIXTRL_HOST_FN cuda_argument_t const& doGetRefCudaArgument(
            argument_base_t const* ptr_base_arg,
            char const* SIXTRL_RESTRICT arg_name = "",
            bool const requires_exact_match = false ) const;

        SIXTRL_HOST_FN cuda_argument_t& doGetRefCudaArgument(
            argument_base_t const* ptr_base_arg,
            char const* SIXTRL_RESTRICT arg_name = "",
            bool const requires_exact_match = false );

        SIXTRL_HOST_FN cuda_argument_t const* doGetPtrCudaArgument(
            argument_base_t const* ptr_base_arg,
            bool const requires_exact_match = false ) const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN cuda_argument_t* doGetPtrCudaArgument(
            argument_base_t* ptr_base_arg,
            bool const requires_exact_match = false ) SIXTRL_NOEXCEPT;


        private:

        template< typename PartSetIndexIter >
        SIXTRL_HOST_FN status_t doInitCudaTrackJob(
            const char *const SIXTRL_RESTRICT config_str,
            c_buffer_t* SIXTRL_RESTRICT particles_buffer,
            PartSetIndexIter particle_set_indices_begin,
            PartSetIndexIter particle_set_indices_end,
            c_buffer_t* SIXTRL_RESTRICT beam_elements_buffer,
            c_buffer_t* SIXTRL_RESTRICT ptr_output_buffer,
            size_type const until_turn_elem_by_elem );

        template< typename PartSetIndexIter >
        SIXTRL_HOST_FN status_t doInitCudaTrackJob(
            std::string const& SIXTRL_RESTRICT_REF config_str,
            buffer_t& SIXTRL_RESTRICT_REF particles_buffer,
            PartSetIndexIter particle_set_indices_begin,
            PartSetIndexIter particle_set_indices_end,
            buffer_t& SIXTRL_RESTRICT_REF beam_elements_buffer,
            buffer_t* SIXTRL_RESTRICT ptr_output_buffer,
            size_type const until_turn_elem_by_elem );

        SIXTRL_HOST_FN status_t doPrepareControllerCudaImpl(
            const char *const SIXTRL_RESTRICT ptr_config_str );

        SIXTRL_HOST_FN status_t doPrepareDefaultKernelsCudaImpl(
            const char *const SIXTRL_RESTRICT ptr_config_str );

        SIXTRL_HOST_FN status_t doPrepareParticlesStructuresCudaImpl(
            c_buffer_t* SIXTRL_RESTRICT ptr_particles_buffer );

        SIXTRL_HOST_FN status_t doPrepareBeamElementsStructuresCudaImpl(
            c_buffer_t* SIXTRL_RESTRICT ptr_beam_elem_buffer );

        SIXTRL_HOST_FN status_t doPrepareOutputStructuresCudaImpl(
            c_buffer_t* SIXTRL_RESTRICT particles_buffer,
            c_buffer_t* SIXTRL_RESTRICT beam_elem_buffer,
            c_buffer_t* SIXTRL_RESTRICT ptr_output_buffer,
            size_type const until_turn_elem_by_elem );

        SIXTRL_HOST_FN status_t doAssignOutputBufferToBeamMonitorsCudaImpl(
            c_buffer_t* SIXTRL_RESTRICT beam_elem_buffer,
            c_buffer_t* SIXTRL_RESTRICT output_buffer,
            particle_index_t const min_turn_id,
            size_type const output_buffer_offset_index );

        SIXTRL_HOST_FN status_t doAssignOutputBufferToElemByElemConfigCudaImpl(
            elem_by_elem_config_t* SIXTRL_RESTRICT elem_by_elem_config,
            c_buffer_t* SIXTRL_RESTRICT output_buffer,
            size_type const output_buffer_offset_index );

        SIXTRL_HOST_FN status_t doResetCudaImpl(
            c_buffer_t* SIXTRL_RESTRICT particles_buffer,
            c_buffer_t* SIXTRL_RESTRICT beam_elem_buffer,
            c_buffer_t* SIXTRL_RESTRICT ptr_output_buffer,
            size_type const until_turn_elem_by_elem );

        /* ---------------------------------------------------------------- */

        SIXTRL_HOST_FN status_t
        doSetAssignOutputToBeamMonitorsKernelIdCudaImpl(
            kernel_id_t const id ) SIXTRL_HOST_FN;

        SIXTRL_HOST_FN status_t
        doSetAssignOutputToElemByElemConfigKernelIdCudaImpl(
            kernel_id_t const id ) SIXTRL_HOST_FN;

        SIXTRL_HOST_FN status_t
        doSetTrackUntilKernelIdCudaImpl(
            kernel_id_t const id ) SIXTRL_HOST_FN;

        SIXTRL_HOST_FN status_t
        doSetTrackLineKernelIdCudaImpl(
            kernel_id_t const id ) SIXTRL_HOST_FN;

        SIXTRL_HOST_FN status_t
        doSetTrackElemByElemKernelIdCudaImpl(
            kernel_id_t const id ) SIXTRL_HOST_FN;

        SIXTRL_HOST_FN status_t
        doSetFetchParticlesAddressesKernelIdCudaImpl(
            kernel_id_t const id ) SIXTRL_HOST_FN;
    };

    CudaTrackJob::push_flag_t push(
        CudaTrackJob& SIXTRL_RESTRICT_REF track_job,
        CudaTrackJob::push_flag_t const flag =
            SIXTRL_CXX_NAMESPACE::TRACK_JOB_IO_BEAM_ELEMENTS );

    CudaTrackJob::collect_flag_t collect(
        CudaTrackJob& SIXTRL_RESTRICT_REF track_job );

    CudaTrackJob::track_status_t trackUntilTurn(
        CudaTrackJob& SIXTRL_RESTRICT_REF track_job,
        CudaTrackJob::size_type const until_turn );

    CudaTrackJob::track_status_t trackElemByElemUntilTurn(
        CudaTrackJob& SIXTRL_RESTRICT_REF track_job,
        CudaTrackJob::size_type const until_turn );

    CudaTrackJob::track_status_t trackLine(
        CudaTrackJob& SIXTRL_RESTRICT_REF track_job,
        CudaTrackJob::size_type const belem_begin_id,
        CudaTrackJob::size_type const belem_end_id,
        bool const finish_turn = false );
}

#endif /* #if defined( __cplusplus ) && !defined( _GPUCODE ) && \
   !defined( __CUDACC__  ) && !defined( __CUDA_ARCH__ ) */

#if defined( __cplusplus ) && !defined( _GPUCODE )
extern "C" {
#endif /* C++ */

#if defined( __cplusplus ) && !defined( _GPUCODE ) && \
   !defined( __CUDACC__  ) && !defined( __CUDACC__ )

typedef SIXTRL_CXX_NAMESPACE::CudaTrackJob NS(CudaTrackJob);

#else /* C++, Host */

typedef void NS(CudaTrackJob);

#endif /* C++, Host */

#if defined( __cplusplus ) && !defined( _GPUCODE )
}
#endif /* C++ */

/* ************************************************************************* */
/* *******   Implementation of inline and template member functions  ******* */
/* ************************************************************************* */

#if defined( __cplusplus   ) && !defined( _GPUCODE ) && \
   !defined( __CUDA_ARCH__ ) && !defined( __CUDACC__ )

namespace SIXTRL_CXX_NAMESPACE
{
    template< typename PartSetIndexIter >
    CudaTrackJob::CudaTrackJob(
        std::string const& SIXTRL_RESTRICT_REF node_id_str,
        CudaTrackJob::buffer_t& SIXTRL_RESTRICT_REF particles_buffer,
        PartSetIndexIter pset_indices_begin, PartSetIndexIter pset_indices_end,
        CudaTrackJob::buffer_t& SIXTRL_RESTRICT_REF belems_buffer,
        CudaTrackJob::buffer_t* SIXTRL_RESTRICT ptr_output_buffer,
        CudaTrackJob::size_type const until_turn_elem_by_elem,
        std::string const& config_str ) :
        SIXTRL_CXX_NAMESPACE::TrackJobNodeCtrlArgBase(
                SIXTRL_CXX_NAMESPACE::ARCHITECTURE_CUDA,
                SIXTRL_ARCHITECTURE_CUDA_STR, config_str.c_str() )
    {
        CudaTrackJob::status_t const status = this->doInitCudaTrackJob(
            config_str, particles_buffer, pset_indices_begin, pset_indices_end,
                belems_buffer, ptr_output_buffer, until_turn_elem_by_elem );

        SIXTRL_ASSERT( status == SIXTRL_CXX_NAMESPACE::ARCH_STATUS_SUCCESS );
        ( void )status;
    }

    template< typename PartSetIndexIter >
    CudaTrackJob::CudaTrackJob(
        char const* SIXTRL_RESTRICT node_id_str,
        CudaTrackJob::c_buffer_t* SIXTRL_RESTRICT particles_buffer,
        PartSetIndexIter pset_indices_begin, PartSetIndexIter pset_indices_end,
        CudaTrackJob::c_buffer_t* SIXTRL_RESTRICT belems_buffer,
        CudaTrackJob::c_buffer_t* SIXTRL_RESTRICT ptr_output_buffer,
        CudaTrackJob::size_type const until_turn_elem_by_elem,
        char const* SIXTRL_RESTRICT config_str ) :
        SIXTRL_CXX_NAMESPACE::TrackJobNodeCtrlArgBase(
            SIXTRL_CXX_NAMESPACE::ARCHITECTURE_CUDA,
            SIXTRL_ARCHITECTURE_CUDA_STR, config_str )
    {
        CudaTrackJob::status_t const status = this->doInitCudaTrackJob(
            config_str, particles_buffer, pset_indices_begin, pset_indices_end,
                belems_buffer, ptr_output_buffer, until_turn_elem_by_elem );

        SIXTRL_ASSERT( status == SIXTRL_CXX_NAMESPACE::ARCH_STATUS_SUCCESS );
        ( void )status;
    }


    template< typename PartSetIndexIter >
    CudaTrackJob::status_t CudaTrackJob::doInitCudaTrackJob(
        std::string const& SIXTRL_RESTRICT_REF config_str,
        CudaTrackJob::buffer_t& SIXTRL_RESTRICT_REF particles_buffer,
        PartSetIndexIter pset_begin,
        PartSetIndexIter pset_end,
        CudaTrackJob::buffer_t& SIXTRL_RESTRICT_REF beam_elements_buffer,
        CudaTrackJob::buffer_t* SIXTRL_RESTRICT ptr_output_buffer,
        CudaTrackJob::size_type const until_turn_elem_by_elem )
    {
        using c_buffer_t = SIXTRL_CXX_NAMESPACE::CudaTrackJob::c_buffer_t;

        c_buffer_t* ptr_c_out_buffer = ( ptr_output_buffer != nullptr )
            ? ptr_output_buffer->getCApiPtr() : nullptr;

        CudaTrackJob::status_t status =
            this->doInitCudaTrackJob( config_str.c_str(),
                particles_buffer.getCApiPtr(), pset_begin, pset_end,
                    beam_elements_buffer.getCApiPtr(), ptr_c_out_buffer,
                        until_turn_elem_by_elem );

        if( status == SIXTRL_CXX_NAMESPACE::ARCH_STATUS_SUCCESS )
        {
            this->doSetCxxBufferPointers(
                particles_buffer, beam_elements_buffer, ptr_output_buffer );
        }

        return status;
    }

    template< typename PartSetIndexIter >
    CudaTrackJob::status_t CudaTrackJob::doInitCudaTrackJob(
        const char *const SIXTRL_RESTRICT config_str,
        CudaTrackJob::c_buffer_t* SIXTRL_RESTRICT pbuffer,
        PartSetIndexIter pset_begin,
        PartSetIndexIter pset_end,
        CudaTrackJob::c_buffer_t* SIXTRL_RESTRICT belem_buffer,
        CudaTrackJob::c_buffer_t* SIXTRL_RESTRICT output_buffer,
        CudaTrackJob::size_type const until_turn_elem_by_elem )
    {
        using _this_t     = SIXTRL_CXX_NAMESPACE::CudaTrackJob;
        using _base_t     = SIXTRL_CXX_NAMESPACE::TrackJobNodeCtrlArgBase;
        using cuda_ctrl_t = _this_t::cuda_controller_t;
        using size_t      = _this_t::size_type;
        using diff_t      = std::ptrdiff_t;
        using c_buffer_t  = _this_t::c_buffer_t;
        using pindex_t    = _this_t::particle_index_t;

        using output_buffer_flag_t = _this_t::output_buffer_flag_t;

        CudaTrackJob::status_t status =
            this->doPrepareControllerCudaImpl( config_str );

        cuda_ctrl_t* ptr_ctrl = this->ptrCudaController();

        if( ptr_ctrl == nullptr ) status = ::NS(ARCH_STATUS_GENERAL_FAILURE);

        if( status == ::NS(ARCH_STATUS_SUCCESS) )
        {
            status = _base_t::doPrepareParticlesStructures( pbuffer );
        }

        if( status == ::NS(ARCH_STATUS_SUCCESS) )
        {
            status = this->doPrepareParticlesStructuresCudaImpl( pbuffer );
        }

        if( status == ::NS(ARCH_STATUS_SUCCESS) )
        {
            this->doSetPtrCParticlesBuffer( pbuffer );
        }

        if( ( status == ::NS(ARCH_STATUS_SUCCESS) ) &&
            ( pset_begin != pset_end ) &&
            ( std::distance( pset_begin, pset_end ) > diff_t{ 0 } ) )
        {
            this->doSetParticleSetIndices( pset_begin, pset_end, pbuffer );
        }
        else if( status == ::NS(ARCH_STATUS_SUCCESS) )
        {
            status = ::NS(ARCH_STATUS_GENERAL_FAILURE);
        }

        size_t const num_psets = this->numParticleSets();
        size_t const* pset_id_begin = this->particleSetIndicesBegin();

        if( status == ::NS(ARCH_STATUS_SUCCESS) )
        {
            status = _base_t::doPrepareBeamElementsStructures( belem_buffer );
        }

        if( status == ::NS(ARCH_STATUS_SUCCESS) )
        {
            status = this->doPrepareBeamElementsStructuresCudaImpl(belem_buffer);
        }

        if( status == ::NS(ARCH_STATUS_SUCCESS) )
        {
            this->doSetPtrCBeamElementsBuffer( belem_buffer );
        }

        if( status == ::NS(ARCH_STATUS_SUCCESS) )
        {
            status = this->doPrepareDefaultKernelsCudaImpl( config_str );
        }

        output_buffer_flag_t const out_buffer_flags =
            ::NS(OutputBuffer_required_for_tracking_of_particle_sets)( pbuffer,
                this->numParticleSets(), this->particleSetIndicesBegin(),
                    belem_buffer, until_turn_elem_by_elem );

        bool const requires_output_buffer =
            ::NS(OutputBuffer_requires_output_buffer)( out_buffer_flags );

        if( ( requires_output_buffer ) || ( output_buffer != nullptr ) )
        {
            if( status == ::NS(ARCH_STATUS_SUCCESS) )
            {
                status = _base_t::doPrepareOutputStructures( pbuffer,
                    belem_buffer, output_buffer, until_turn_elem_by_elem );
            }

            if( status == ::NS(ARCH_STATUS_SUCCESS) )
            {
                status = this->doPrepareOutputStructuresCudaImpl(
                    pbuffer, belem_buffer, this->ptrCOutputBuffer(),
                        until_turn_elem_by_elem );
            }
        }

        if( ( status == ::NS(ARCH_STATUS_SUCCESS) ) &&
            ( this->hasOutputBuffer() ) && ( requires_output_buffer ) )
        {
            if( ::NS(OutputBuffer_requires_elem_by_elem_output)(
                    out_buffer_flags ) )
            {
                size_t const out_offset = this->elemByElemOutputBufferOffset();
                c_buffer_t* out_buffer = this->ptrCOutputBuffer();

                if( status == ::NS(ARCH_STATUS_SUCCESS) )
                {
                    status = _base_t::doAssignOutputBufferToElemByElemConfig(
                    this->ptrElemByElemConfig(), out_buffer, out_offset );
                }

                if( status == ::NS(ARCH_STATUS_SUCCESS) )
                {
                    status = this->doAssignOutputBufferToElemByElemConfigCudaImpl(
                    this->ptrElemByElemConfig(), out_buffer, out_offset );
                }
            }

            if( ::NS(OutputBuffer_requires_beam_monitor_output)(
                    out_buffer_flags ) )
            {
                pindex_t const min_turn_id = this->minInitialTurnId();
                size_t const offset = this->beamMonitorsOutputBufferOffset();
                c_buffer_t* ptr_output_buffer = this->ptrCOutputBuffer();

                if( status == ::NS(ARCH_STATUS_SUCCESS) )
                {
                    status = _base_t::doAssignOutputBufferToBeamMonitors(
                        belem_buffer, ptr_output_buffer, min_turn_id, offset );
                }

                if( status == ::NS(ARCH_STATUS_SUCCESS) )
                {
                    status = this->doAssignOutputBufferToBeamMonitorsCudaImpl(
                    belem_buffer, ptr_output_buffer, min_turn_id, offset );
                }
            }
        }

        return status;
    }
}

#endif /* C++, Host */

#endif /* SIXTRACKLIB_CUDA_TRACK_JOB_HPP__ */
