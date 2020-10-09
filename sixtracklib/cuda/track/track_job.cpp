#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/cuda/track_job.hpp"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus ) && !defined( _GPUCODE ) && \
   !defined( __CUDACC__  ) && !defined( __CUDA_ARCH__ )

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <cstddef>
    #include <cstdlib>
    #include <stdexcept>
    #include <sstream>

    #include <cuda_runtime_api.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */
#endif /* C++, Host */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/control/definitions.h"
    #include "sixtracklib/common/control/arch_base.hpp"
    #include "sixtracklib/common/control/kernel_config_base.hpp"
    #include "sixtracklib/common/internal/compiler_attributes.h"
    #include "sixtracklib/common/track/definitions.h"
    #include "sixtracklib/common/track/track_job_ctrl_arg_base.hpp"
    #include "sixtracklib/common/track/track_job_nodectrl_arg_base.hpp"
    #include "sixtracklib/common/buffer.hpp"
    #include "sixtracklib/common/buffer.h"

    #include "sixtracklib/cuda/definitions.h"
    #include "sixtracklib/cuda/controller.hpp"
    #include "sixtracklib/cuda/argument.hpp"
    #include "sixtracklib/cuda/control/default_kernel_config.h"
    #include "sixtracklib/cuda/wrappers/track_job_wrappers.h"

    #include "external/toml11/toml.hpp"

#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus ) && !defined( _GPUCODE ) && \
   !defined( __CUDACC__  ) && !defined( __CUDA_ARCH__ )

namespace SIXTRL_CXX_NAMESPACE
{
    namespace
    {
        namespace st = SIXTRL_CXX_NAMESPACE;
        using tjob_t = st::CudaTrackJob;
        using st_size_t = tjob_t::size_type;
        using st_status_t = st_status_t;
        using st_track_status_t = tjob_t::track_status_t;
    }

    constexpr st_size_t tjob_t::DEFAULT_TRACK_THREADS_PER_BLOCK;
    constexpr st_size_t tjob_t::DEFAULT_THREADS_PER_BLOCK;

    /* --------------------------------------------------------------------- */

    st_size_t tjob_t::NumAvailableNodes()
    {
        st_size_t num_available_nodes = st_size_t{ 0 };

        int temp_num_devices = int{ -1 };
        ::cudaError_t err = ::cudaGetDeviceCount( &temp_num_devices );

        if( ( err == ::cudaSuccess ) && ( temp_num_devices > int{ 0 } ) )
        {
            num_available_nodes = static_cast< st_size_t >( temp_num_devices );
        }

        return num_available_nodes;
    }

    st_size_t tjob_t::GetAvailableNodeIdsList( st_size_t const max_num_node_ids,
        tjob_t::node_id_t* SIXTRL_RESTRICT node_ids_begin )
    {
        st_size_t num_retrieved_nodes = st_size_t{ 0 };
        if( ( max_num_node_ids > size_t{ 0 } ) && ( node_ids_begin != nullptr ) )
        {
            std::unique_ptr< st::CudaController > ptr_ctrl(
                new st::CudaController );

            if( ( ptr_ctrl.get() != nullptr ) &&
                ( ptr_ctrl->numAvailableNodes() > size_t{ 0 } ) )
            {
                num_retrieved_nodes = ptr_ctrl->availableNodeIds(
                    max_num_node_ids, node_ids_begin );

                SIXTRL_ASSERT( num_retrieved_nodes <= max_num_node_ids );
                SIXTRL_ASSERT( num_retrieved_nodes <=
                    ptr_ctrl->numAvailableNodes() );
            }
        }

        return num_retrieved_nodes;
    }

    st_size_t tjob_t::GetAvailableNodeIndicesList(
        st_size_t const max_num_node_indices,
        tjob_t::node_index_t* SIXTRL_RESTRICT node_indices_begin )
    {
        st_size_t num_retrieved_nodes = st_size_t{ 0 };
        if( ( max_num_node_indices > size_t{ 0 } ) &&
            ( node_indices_begin != nullptr ) )
        {
            std::unique_ptr< st::CudaController > ptr_ctrl(
                new st::CudaController );

            if( ( ptr_ctrl.get() != nullptr ) &&
                ( ptr_ctrl->numAvailableNodes() > st_size_t{ 0 } ) )
            {
                num_retrieved_nodes = ptr_ctrl->availableNodeIndices(
                    max_num_node_indices, node_indices_begin );

                SIXTRL_ASSERT( num_retrieved_nodes <= max_num_node_indices );
                SIXTRL_ASSERT( num_retrieved_nodes <=
                    ptr_ctrl->numAvailableNodes() );
            }
        }

        return num_retrieved_nodes;
    }

    /* --------------------------------------------------------------------- */


    tjob_t::CudaTrackJob( const char *const SIXTRL_RESTRICT config_str ) :
            st::TrackJobNodeCtrlArgBase( st::ARCHITECTURE_CUDA,
                SIXTRL_ARCHITECTURE_CUDA_STR, config_str ),
        m_track_threads_per_block( tjob_t::DEFAULT_TRACK_THREADS_PER_BLOCK ),
        m_default_threads_per_block( tjob_t::DEFAULT_THREADS_PER_BLOCK )
    {
        st_status_t status = st::ARCH_STATUS_SUCCESS;

        if( config_str != nullptr )
        {
            status = this->doParseConfigStrCudaImpl( config_str );
        }

        if( status == st::ARCH_STATUS_SUCCESS )
        {
            status = this->doPrepareControllerCudaImpl( config_str );
        }

        if( status == st::ARCH_STATUS_SUCCESS )
        {
            status = this->doPrepareDefaultKernelsCudaImpl( config_str );
        }

        SIXTRL_ASSERT( status == st::ARCH_STATUS_SUCCESS );
        ( void )status;
    }

    tjob_t::CudaTrackJob( std::string const& SIXTRL_RESTRICT_REF config_str ) :
            st::TrackJobNodeCtrlArgBase( st::ARCHITECTURE_CUDA,
                SIXTRL_ARCHITECTURE_CUDA_STR, config_str.c_str() ),
        m_track_threads_per_block( tjob_t::DEFAULT_TRACK_THREADS_PER_BLOCK ),
        m_default_threads_per_block( tjob_t::DEFAULT_THREADS_PER_BLOCK )
    {
        st_status_t status = st::ARCH_STATUS_SUCCESS;

        if( !config_str.empty() )
        {
            status = this->doParseConfigStrCudaImpl( config_str.c_str() );
        }

        if( status == st::ARCH_STATUS_SUCCESS )
        {
            status = this->doPrepareControllerCudaImpl( config_str.c_str() );
        }

        if( status == st::ARCH_STATUS_SUCCESS )
        {
            status = this->doPrepareDefaultKernelsCudaImpl( config_str.c_str() );
        }

        SIXTRL_ASSERT( status == st::ARCH_STATUS_SUCCESS );
        ( void )status;
    }

    tjob_t::CudaTrackJob(
        const char *const SIXTRL_RESTRICT node_id_str,
        c_buffer_t* SIXTRL_RESTRICT particles_buffer,
        c_buffer_t* SIXTRL_RESTRICT beam_elements_buffer,
        c_buffer_t* SIXTRL_RESTRICT output_buffer,
        size_type const until_turn_elem_by_elem,
        const char *const SIXTRL_RESTRICT config_str ) :
            st::TrackJobNodeCtrlArgBase( st::ARCHITECTURE_CUDA,
                SIXTRL_ARCHITECTURE_CUDA_STR, config_str ),
        m_track_threads_per_block( tjob_t::DEFAULT_TRACK_THREADS_PER_BLOCK ),
        m_default_threads_per_block( tjob_t::DEFAULT_THREADS_PER_BLOCK )
    {
        st_status_t const status = this->doInitCudaTrackJob( node_id_str,
            particles_buffer, tjob_t::DefaultParticleSetIndicesBegin(),
                tjob_t::DefaultParticleSetIndicesEnd(), beam_elements_buffer,
                    output_buffer, until_turn_elem_by_elem, config_str );

        SIXTRL_ASSERT( status == st::ARCH_STATUS_SUCCESS );
        ( void )status;
    }

    tjob_t::CudaTrackJob(
        std::string const& SIXTRL_RESTRICT_REF node_id_str,
        buffer_t& SIXTRL_RESTRICT_REF particles_buffer,
        buffer_t& SIXTRL_RESTRICT_REF beam_elements_buffer,
        buffer_t* SIXTRL_RESTRICT ptr_output_buffer,
        size_type const until_turn_elem_by_elem,
        std::string const& config_str ) :
            st::TrackJobNodeCtrlArgBase( st::ARCHITECTURE_CUDA,
                SIXTRL_ARCHITECTURE_CUDA_STR, config_str.c_str() ),
        m_track_threads_per_block( tjob_t::DEFAULT_TRACK_THREADS_PER_BLOCK ),
        m_default_threads_per_block( tjob_t::DEFAULT_THREADS_PER_BLOCK )
    {
        st_status_t const status = this->doInitCudaTrackJob( node_id_str,
            particles_buffer, tjob_t::DefaultParticleSetIndicesBegin(),
                tjob_t::DefaultParticleSetIndicesEnd(), beam_elements_buffer,
                    ptr_output_buffer, until_turn_elem_by_elem, config_str );

        if( status == st::ARCH_STATUS_SUCCESS )
        {
            this->doSetPtrParticlesBuffer( &particles_buffer );
            this->doSetPtrBeamElementsBuffer( &beam_elements_buffer );

            if( ( ptr_output_buffer != nullptr ) &&
                ( this->hasOutputBuffer() ) && ( !this->ownsOutputBuffer() ) )
            {
                this->doSetPtrOutputBuffer( ptr_output_buffer );
            }
        }
    }

    tjob_t::CudaTrackJob(
        std::string const& SIXTRL_RESTRICT_REF node_id_str,
        tjob_t::buffer_t& SIXTRL_RESTRICT_REF particles_buffer,
        st_size_t const particle_set_index,
        tjob_t::buffer_t& SIXTRL_RESTRICT_REF belems_buffer,
        tjob_t::buffer_t* SIXTRL_RESTRICT ptr_output_buffer,
        st_size_t const until_turn_elem_by_elem,
        std::string const& config_str ) :
            st::TrackJobNodeCtrlArgBase( st::ARCHITECTURE_CUDA,
                SIXTRL_ARCHITECTURE_CUDA_STR, config_str.c_str() ),
        m_track_threads_per_block( tjob_t::DEFAULT_TRACK_THREADS_PER_BLOCK ),
        m_default_threads_per_block( tjob_t::DEFAULT_THREADS_PER_BLOCK )
    {
        st_size_t const* psets_begin = &particle_set_index;
        st_size_t const* psets_end = psets_begin;
        std::advance( psets_end, st_size_t{ 1 } );

        st_status_t const status = this->doInitCudaTrackJob( node_id_str,
            particles_buffer, psets_begin, psets_end, belems_buffer,
                ptr_output_buffer, until_turn_elem_by_elem, config_str );

        if( status == st::ARCH_STATUS_SUCCESS )
        {
            this->doSetPtrParticlesBuffer( &particles_buffer );
            this->doSetPtrBeamElementsBuffer( &belems_buffer );

            if( ( ptr_output_buffer != nullptr ) &&
                ( this->hasOutputBuffer() ) && ( !this->ownsOutputBuffer() ) )
            {
                this->doSetPtrOutputBuffer( ptr_output_buffer );
            }
        }
    }

    tjob_t::CudaTrackJob( char const* SIXTRL_RESTRICT node_id_str,
        tjob_t::c_buffer_t* SIXTRL_RESTRICT particles_buffer,
        st_size_t const particle_set_index,
        tjob_t::c_buffer_t* SIXTRL_RESTRICT belems_buffer,
        tjob_t::c_buffer_t* SIXTRL_RESTRICT ptr_output_buffer,
        st_size_t const until_turn_elem_by_elem,
        char const* SIXTRL_RESTRICT config_str ) :
            st::TrackJobNodeCtrlArgBase( st::ARCHITECTURE_CUDA,
                SIXTRL_ARCHITECTURE_CUDA_STR, config_str ),
        m_track_threads_per_block( tjob_t::DEFAULT_TRACK_THREADS_PER_BLOCK ),
        m_default_threads_per_block( tjob_t::DEFAULT_THREADS_PER_BLOCK )
    {
        st_size_t const* psets_begin = &particle_set_index;
        st_size_t const* psets_end = psets_begin;
        std::advance( psets_end, st_size_t{ 1 } );

        st_status_t const status = this->doInitCudaTrackJob( node_id_str,
            particles_buffer, psets_begin, psets_end, belems_buffer,
                ptr_output_buffer, until_turn_elem_by_elem, config_str );

        SIXTRL_ASSERT( status == st::ARCH_STATUS_SUCCESS );
        ( void )status;
    }

    tjob_t::CudaTrackJob(
        std::string const& SIXTRL_RESTRICT_REF node_id_str,
        tjob_t::buffer_t& SIXTRL_RESTRICT_REF particles_buffer,
        st_size_t const num_particle_sets,
        st_size_t const* SIXTRL_RESTRICT pset_indices_begin,
        tjob_t::buffer_t& SIXTRL_RESTRICT_REF belems_buffer,
        tjob_t::buffer_t* SIXTRL_RESTRICT ptr_output_buffer,
        st_size_t const until_turn_elem_by_elem,
        std::string const& config_str ) :
            st::TrackJobNodeCtrlArgBase( st::ARCHITECTURE_CUDA,
                SIXTRL_ARCHITECTURE_CUDA_STR, config_str.c_str() ),
        m_track_threads_per_block( tjob_t::DEFAULT_TRACK_THREADS_PER_BLOCK ),
        m_default_threads_per_block( tjob_t::DEFAULT_THREADS_PER_BLOCK )
    {
        st_size_t const* pset_indices_end = pset_indices_begin;

        if( pset_indices_end != nullptr )
        {
            std::advance( pset_indices_end, num_particle_sets );
        }

        st_status_t const status = this->doInitCudaTrackJob(
            node_id_str, particles_buffer, pset_indices_begin, pset_indices_end,
                belems_buffer, ptr_output_buffer, until_turn_elem_by_elem,
                    config_str );

        SIXTRL_ASSERT( status == st::ARCH_STATUS_SUCCESS );
        ( void )status;
    }

    tjob_t::CudaTrackJob(
        char const* SIXTRL_RESTRICT node_id_str,
        tjob_t::c_buffer_t* SIXTRL_RESTRICT particles_buffer,
        st_size_t const num_particle_sets,
        st_size_t const* SIXTRL_RESTRICT pset_indices_begin,
        tjob_t::c_buffer_t* SIXTRL_RESTRICT belems_buffer,
        tjob_t::c_buffer_t* SIXTRL_RESTRICT ptr_output_buffer,
        st_size_t const until_turn_elem_by_elem,
        char const* SIXTRL_RESTRICT config_str ) :
            st::TrackJobNodeCtrlArgBase( st::ARCHITECTURE_CUDA,
                SIXTRL_ARCHITECTURE_CUDA_STR, config_str ),
        m_track_threads_per_block( tjob_t::DEFAULT_TRACK_THREADS_PER_BLOCK ),
        m_default_threads_per_block( tjob_t::DEFAULT_THREADS_PER_BLOCK )
    {
        st_status_t status = st::ARCH_STATUS_GENERAL_FAILURE;
        st_size_t const* pset_indices_end = pset_indices_begin;

        if( ( pset_indices_end != nullptr ) &&
            ( num_particle_sets > st_size_t{ 0 } ) )
        {
            std::advance( pset_indices_end, num_particle_sets );
            status = this->doInitCudaTrackJob( node_id_str, particles_buffer,
                pset_indices_begin, pset_indices_end, belems_buffer,
                    ptr_output_buffer, until_turn_elem_by_elem, config_str );
        }
        else
        {
            status = this->doInitCudaTrackJob( node_id_str, particles_buffer,
                tjob_t::DefaultParticleSetIndicesBegin(),
                tjob_t::DefaultParticleSetIndicesEnd(),
                    belems_buffer, ptr_output_buffer, until_turn_elem_by_elem,
                        config_str );
        }

        SIXTRL_ASSERT( status == st::ARCH_STATUS_SUCCESS );
        ( void )status;
    }

    tjob_t::~CudaTrackJob() SIXTRL_NOEXCEPT {}

    /* ===================================================================== */

    bool tjob_t::hasCudaController() const SIXTRL_NOEXCEPT
    {
        return ( this->ptrCudaController() != nullptr );
    }

    tjob_t::cuda_controller_t& tjob_t::cudaController()
    {
        return const_cast< tjob_t::cuda_controller_t& >( static_cast<
            tjob_t const& >( *this ).cudaController() );
    }

    tjob_t::cuda_controller_t const& tjob_t::cudaController() const
    {
        if( !this->hasCudaController() )
        {
            throw std::runtime_error( "no cuda controller stored" );
        }

        return *this->ptrCudaController();
    }

    tjob_t::cuda_controller_t* tjob_t::ptrCudaController() SIXTRL_NOEXCEPT
    {
        return const_cast< tjob_t::cuda_controller_t* >( static_cast<
            tjob_t const& >( *this ).ptrCudaController() );
    }

    tjob_t::cuda_controller_t const*
    tjob_t::ptrCudaController() const SIXTRL_NOEXCEPT
    {
        using ctrl_t = tjob_t::cuda_controller_t;
        using ptr_base_ctrl_t = tjob_t::controller_base_t const*;

        ptr_base_ctrl_t ptr_base_ctrl = this->ptrControllerBase();

        return ( ptr_base_ctrl != nullptr )
            ? ptr_base_ctrl->asDerivedController< ctrl_t >( this->archId() )
            : nullptr;
    }

    /* --------------------------------------------------------------------- */

    bool tjob_t::hasCudaParticlesArg() const SIXTRL_NOEXCEPT
    {
        return ( this->ptrCudaParticlesArg() != nullptr );
    }

    tjob_t::cuda_argument_t& tjob_t::cudaParticlesArg()
    {
        return this->doGetRefCudaArgument( this->ptrParticlesArgBase(),
            "ptrParticlesArgBase()" );
    }

    tjob_t::cuda_argument_t const& tjob_t::cudaParticlesArg() const
    {
        return this->doGetRefCudaArgument( this->ptrParticlesArgBase(),
            "ptrParticlesArgBase()" );
    }

    tjob_t::cuda_argument_t* tjob_t::ptrCudaParticlesArg() SIXTRL_NOEXCEPT
    {
        return this->doGetPtrCudaArgument( this->ptrParticlesArgBase() );
    }

    tjob_t::cuda_argument_t const*
    tjob_t::ptrCudaParticlesArg() const SIXTRL_NOEXCEPT
    {
        return this->doGetPtrCudaArgument( this->ptrParticlesArgBase() );
    }

    /* --------------------------------------------------------------------- */

    bool tjob_t::hasCudaBeamElementsArg() const SIXTRL_NOEXCEPT
    {
        return ( this->ptrCudaBeamElementsArg() != nullptr );
    }

    tjob_t::cuda_argument_t& tjob_t::cudaBeamElementsArg()
    {
        return this->doGetRefCudaArgument( this->ptrBeamElementsArgBase(),
            "ptrBeamElementsArgBase()" );
    }

    tjob_t::cuda_argument_t const& tjob_t::cudaBeamElementsArg() const
    {
        return this->doGetRefCudaArgument( this->ptrBeamElementsArgBase(),
            "ptrBeamElementsArgBase()" );
    }

    tjob_t::cuda_argument_t const*
    tjob_t::ptrCudaBeamElementsArg() const SIXTRL_NOEXCEPT
    {
        return this->doGetPtrCudaArgument( this->ptrBeamElementsArgBase() );
    }

    tjob_t::cuda_argument_t*
    tjob_t::ptrCudaBeamElementsArg() SIXTRL_NOEXCEPT
    {
        return this->doGetPtrCudaArgument( this->ptrBeamElementsArgBase() );
    }

    /* --------------------------------------------------------------------- */

    bool tjob_t::hasCudaOutputArg() const SIXTRL_NOEXCEPT
    {
        return ( this->ptrCudaOutputArg() != nullptr );
    }

    tjob_t::cuda_argument_t& tjob_t::cudaOutputArg()
    {
        return this->doGetRefCudaArgument( this->ptrOutputArgBase(),
            "ptrOutputArgBase()" );
    }

    tjob_t::cuda_argument_t const& tjob_t::cudaOutputArg() const
    {
        return this->doGetRefCudaArgument( this->ptrOutputArgBase(),
            "ptrOutputArgBase()" );
    }

    tjob_t::cuda_argument_t* tjob_t::ptrCudaOutputArg() SIXTRL_NOEXCEPT
    {
        return this->doGetPtrCudaArgument( this->ptrOutputArgBase() );
    }

    tjob_t::cuda_argument_t const*
    tjob_t::ptrCudaOutputArg() const SIXTRL_NOEXCEPT
    {
        return this->doGetPtrCudaArgument( this->ptrOutputArgBase() );
    }

    /* --------------------------------------------------------------------- */

    bool tjob_t::hasCudaElemByElemConfigArg() const SIXTRL_NOEXCEPT
    {
        return ( this->ptrCudaElemByElemConfigArg() != nullptr );
    }

    tjob_t::cuda_argument_t& tjob_t::cudaElemByElemConfigArg()
    {
        return this->doGetRefCudaArgument( this->ptrElemByElemConfigArgBase(),
            "ptrElemByElemConfigArgBase()" );
    }

    tjob_t::cuda_argument_t const& tjob_t::cudaElemByElemConfigArg() const
    {
        return this->doGetRefCudaArgument( this->ptrElemByElemConfigArgBase(),
            "ptrElemByElemConfigArgBase()" );
    }

    tjob_t::cuda_argument_t* tjob_t::ptrCudaElemByElemConfigArg() SIXTRL_NOEXCEPT
    {
        return this->doGetPtrCudaArgument(
            this->ptrElemByElemConfigArgBase() );
    }

    tjob_t::cuda_argument_t const*
    tjob_t::ptrCudaElemByElemConfigArg() const SIXTRL_NOEXCEPT
    {
        return this->doGetPtrCudaArgument(
            this->ptrElemByElemConfigArgBase() );
    }

    /* --------------------------------------------------------------------- */

    bool tjob_t::hasCudaDebugRegisterArg() const SIXTRL_NOEXCEPT
    {
        return ( this->ptrCudaDebugRegisterArg() != nullptr );
    }

    tjob_t::cuda_argument_t& tjob_t::cudaDebugRegisterArg()
    {
        return this->doGetRefCudaArgument(
            this->ptrDebugRegisterArgBase(), "ptrDebugRegisterArgBase()" );
    }

    tjob_t::cuda_argument_t const& tjob_t::cudaDebugRegisterArg() const
    {
        return this->doGetRefCudaArgument(
            this->ptrDebugRegisterArgBase(), "ptrDebugRegisterArgBase()" );
    }

    tjob_t::cuda_argument_t* tjob_t::ptrCudaDebugRegisterArg() SIXTRL_NOEXCEPT
    {
        return this->doGetPtrCudaArgument( this->ptrDebugRegisterArgBase() );
    }

    tjob_t::cuda_argument_t const*
    tjob_t::ptrCudaDebugRegisterArg() const SIXTRL_NOEXCEPT
    {
        return this->doGetPtrCudaArgument( this->ptrDebugRegisterArgBase() );
    }

    /* --------------------------------------------------------------------- */

    bool tjob_t::hasCudaParticlesAddrArg() const SIXTRL_NOEXCEPT
    {
        return ( this->ptrCudaParticlesAddrArg() != nullptr );
    }

    tjob_t::cuda_argument_t const& tjob_t::cudaParticlesAddrArg() const
    {
        return this->doGetRefCudaArgument(
            this->ptrParticlesAddrArgBase(), "ptrParticlesAddrArgBase()" );
    }

    tjob_t::cuda_argument_t& tjob_t::cudaParticlesAddrArg()
    {
        return this->doGetRefCudaArgument(
            this->ptrParticlesAddrArgBase(), "ptrParticlesAddrArgBase()" );
    }

    tjob_t::cuda_argument_t* tjob_t::ptrCudaParticlesAddrArg() SIXTRL_NOEXCEPT
    {
        return this->doGetPtrCudaArgument( this->ptrParticlesAddrArgBase() );
    }

    tjob_t::cuda_argument_t const*
    tjob_t::ptrCudaParticlesAddrArg() const SIXTRL_NOEXCEPT
    {
        return this->doGetPtrCudaArgument( this->ptrParticlesAddrArgBase() );
    }

    st_size_t tjob_t::default_threads_per_block() const SIXTRL_NOEXCEPT
    {
        return this->m_default_threads_per_block;
    }

    st_size_t tjob_t::default_track_threads_per_block() const SIXTRL_NOEXCEPT
    {
        return this->m_track_threads_per_block;
    }

    /* --------------------------------------------------------------------- */

    tjob_t::cuda_argument_t const* tjob_t::ptr_const_argument_by_buffer_id(
        st_size_t const buffer_id ) const SIXTRL_NOEXCEPT
    {
        tjob_t::cuda_argument_t const* ptr_arg = nullptr;

        switch( buffer_id )
        {
            case st::ARCH_PARTICLES_BUFFER_ID:
            {
                ptr_arg = this->ptrCudaParticlesArg();
                break;
            }

            case st::ARCH_BEAM_ELEMENTS_BUFFER_ID:
            {
                ptr_arg = this->ptrCudaBeamElementsArg();
                break;
            }

            case st::ARCH_OUTPUT_BUFFER_ID:
            {
                ptr_arg = this->ptrCudaOutputArg();
                break;
            }

            case st::ARCH_ELEM_BY_ELEM_CONFIG_BUFFER_ID:
            {
                ptr_arg = this->ptrCudaElemByElemConfigArg();
                break;
            }

            case st::ARCH_PARTICLE_ADDR_BUFFER_ID:
            {
                ptr_arg = this->ptrCudaParticlesAddrArg();
                break;
            }

            default:
            {
                ptr_arg = this->ptr_const_stored_buffer_argument( buffer_id );
            }
        };

        return ptr_arg;
    }

    tjob_t::cuda_argument_t* tjob_t::ptr_argument_by_buffer_id(
        st_size_t const buffer_id ) SIXTRL_NOEXCEPT
    {
        return const_cast< tjob_t::cuda_argument_t* >(
            this->ptr_const_argument_by_buffer_id( buffer_id ) );
    }

    tjob_t::cuda_argument_t const& tjob_t::argument_by_buffer_id(
        st_size_t const buffer_id ) const
    {
        tjob_t::cuda_argument_t const* ptr_arg =
            this->ptr_const_argument_by_buffer_id( buffer_id );

        if( ptr_arg == nullptr )
        {
            std::ostringstream a2str;
            a2str << "unable to get buffer argument for buffer_id="
                  << buffer_id;

            throw std::runtime_error( a2str.str() );
        }

        return *ptr_arg;
    }

    tjob_t::cuda_argument_t& tjob_t::argument_by_buffer_id(
        st_size_t const buffer_id )
    {
        return const_cast< tjob_t::cuda_argument_t& >( static_cast<
            tjob_t const& >( *this ).argument_by_buffer_id( buffer_id ) );
    }

    /* -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --  */

    tjob_t::cuda_argument_t const* tjob_t::ptr_const_stored_buffer_argument(
        st_size_t const buffer_id ) const SIXTRL_NOEXCEPT
    {
        tjob_t::cuda_argument_t const* ptr_arg = nullptr;

        st_size_t const min_buffer_id = this->min_stored_buffer_id();
        st_size_t const max_buffer_id_plus_one =
            this->max_stored_buffer_id() + st_size_t{ 1 };

        if( ( min_buffer_id != st::ARCH_ILLEGAL_BUFFER_ID ) &&
            ( buffer_id != st::ARCH_ILLEGAL_BUFFER_ID ) &&
            ( buffer_id >= min_buffer_id ) &&
            ( buffer_id <  max_buffer_id_plus_one ) )
        {
            st_size_t const stored_buffer_id = buffer_id - min_buffer_id;

            if( stored_buffer_id < this->m_stored_buffer_args.size() )
            {
                ptr_arg = this->m_stored_buffer_args[ stored_buffer_id ].get();
            }
        }

        return ptr_arg;
    }

    tjob_t::cuda_argument_t* tjob_t::ptr_stored_buffer_argument(
        st_size_t const buffer_id ) SIXTRL_NOEXCEPT
    {
        return const_cast< tjob_t::cuda_argument_t* >(
            this->ptr_const_stored_buffer_argument( buffer_id ) );
    }

    tjob_t::cuda_argument_t const& tjob_t::stored_buffer_argument(
        st_size_t const buffer_id ) const
    {
        tjob_t::cuda_argument_t const* ptr_arg =
            this->ptr_const_stored_buffer_argument( buffer_id );

        if( ptr_arg == nullptr )
        {
            std::ostringstream a2str;
            a2str << "unable to get stored buffer argument for buffer_id="
                  << buffer_id;

            throw std::runtime_error( a2str.str() );
        }

        return *ptr_arg;
    }

    tjob_t::cuda_argument_t& tjob_t::stored_buffer_argument(
        st_size_t const buffer_id )
    {
        return const_cast< tjob_t::cuda_argument_t& >( static_cast<
            tjob_t const& >( *this ).stored_buffer_argument( buffer_id ) );
    }

    /* ===================================================================== */

    st_status_t tjob_t::doPrepareController(
        char const* SIXTRL_RESTRICT config_str )
    {
        return this->doPrepareControllerCudaImpl( config_str );
    }

    st_status_t tjob_t::doPrepareDefaultKernels(
        char const* SIXTRL_RESTRICT config_str )
    {
        return this->doPrepareDefaultKernelsCudaImpl( config_str );
    }

    st_status_t tjob_t::doPrepareParticlesStructures(
        tjob_t::c_buffer_t* SIXTRL_RESTRICT ptr_pbuffer )
    {
        using _base_t = tjob_t::_base_track_job_t;
        return ( ( _base_t::doPrepareParticlesStructures( ptr_pbuffer ) ) &&
                 ( this->doPrepareParticlesStructuresCudaImpl( ptr_pbuffer ) ) );
    }

    st_status_t tjob_t::doPrepareBeamElementsStructures(
        tjob_t::c_buffer_t* SIXTRL_RESTRICT belems )
    {
        using _base_t = tjob_t::_base_track_job_t;
        return ( ( _base_t::doPrepareBeamElementsStructures( belems ) ) &&
                 ( this->doPrepareBeamElementsStructuresCudaImpl( belems ) ) );
    }

    st_status_t tjob_t::doPrepareOutputStructures(
        tjob_t::c_buffer_t* SIXTRL_RESTRICT pbuffer,
        tjob_t::c_buffer_t* SIXTRL_RESTRICT belems,
        tjob_t::c_buffer_t* SIXTRL_RESTRICT output,
        st_size_t const until_turn_elem_by_elem )
    {
        using _base_t = tjob_t::_base_track_job_t;

        st_status_t status = _base_t::doPrepareOutputStructures(
                pbuffer, belems, output, until_turn_elem_by_elem );

        if( ( status == st::ARCH_STATUS_SUCCESS ) &&
            ( this->hasOutputBuffer() ) )
        {
            status = this->doPrepareOutputStructuresCudaImpl( pbuffer, belems,
                this->ptrCOutputBuffer(), until_turn_elem_by_elem );
        }

        return status;
    }

    st_status_t tjob_t::doAssignOutputBufferToBeamMonitors(
        tjob_t::c_buffer_t* SIXTRL_RESTRICT belems,
        tjob_t::c_buffer_t* SIXTRL_RESTRICT output,
        tjob_t::particle_index_t const min_turn_id,
        st_size_t const output_buffer_offset_index )
    {
        using _base_t = tjob_t::_base_track_job_t;

        st_status_t status = _base_t::doAssignOutputBufferToBeamMonitors(
            belems, output, min_turn_id, output_buffer_offset_index );

        if( ( status == st::ARCH_STATUS_SUCCESS ) && ( this->hasBeamMonitorOutput() ) )
        {
            status = this->doAssignOutputBufferToBeamMonitorsCudaImpl(
                belems, output, min_turn_id, output_buffer_offset_index );
        }

        return status;
    }

    st_status_t tjob_t::doAssignOutputBufferToElemByElemConfig(
        tjob_t::elem_by_elem_config_t* SIXTRL_RESTRICT elem_by_elem_conf,
        tjob_t::c_buffer_t* SIXTRL_RESTRICT output,
        st_size_t const output_buffer_offset_index )
    {
        using _base_t = tjob_t::_base_track_job_t;
        st_status_t status = _base_t::doAssignOutputBufferToElemByElemConfig(
            elem_by_elem_conf, output, output_buffer_offset_index );

        if( ( status == st::ARCH_STATUS_SUCCESS ) &&
            ( this->hasElemByElemOutput() ) )
        {
            status = this->doAssignOutputBufferToElemByElemConfigCudaImpl(
                elem_by_elem_conf, output, output_buffer_offset_index );
        }

        return status;
    }

    st_status_t tjob_t::doReset(
        tjob_t::c_buffer_t* SIXTRL_RESTRICT pbuffer,
        tjob_t::c_buffer_t* SIXTRL_RESTRICT belems,
        tjob_t::c_buffer_t* SIXTRL_RESTRICT output,
        st_size_t const until_turn_elem_by_elem )
    {
        return this->doResetCudaImpl(
            pbuffer, belems, output, until_turn_elem_by_elem );
    }

    bool tjob_t::doParseConfigStr(
        const char *const SIXTRL_RESTRICT config_str )
    {
        return ( ( _base_t::doParseConfigStr( config_str ) ) &&
                 (  st::ARCH_STATUS_SUCCESS ==
                    this->doParseConfigStrCudaImpl( config_str ) ) );
    }

    /* ----------------------------------------------------------------- */

    st_size_t tjob_t::do_add_stored_buffer(
        tjob_t::buffer_store_t&& assigned_buffer_handle )
    {
        st_size_t buffer_id = tjob_t::_base_track_job_t::do_add_stored_buffer(
            std::move( assigned_buffer_handle ) );

        if( buffer_id != st::ARCH_ILLEGAL_BUFFER_ID )
        {
            if( st::ARCH_STATUS_SUCCESS != this->do_add_stored_buffer_cuda_impl(
                    buffer_id ) )
            {
                this->do_remove_stored_buffer_cuda_impl( buffer_id );
                buffer_id = st::ARCH_ILLEGAL_BUFFER_ID;
            }
        }

        return buffer_id;
    }

    st_status_t tjob_t::do_remove_stored_buffer( st_size_t const buffer_id )
    {
        return tjob_t::_base_track_job_t::do_remove_stored_buffer( buffer_id ) |
               this->do_remove_stored_buffer_cuda_impl( buffer_id );
    }

    st_status_t tjob_t::do_push_stored_buffer( st_size_t const buffer_id )
    {
        return this->do_push_stored_buffer_cuda_impl( buffer_id );
    }

    st_status_t tjob_t::do_collect_stored_buffer( st_size_t const buffer_id )
    {
        return this->do_collect_stored_buffer_cuda_impl( buffer_id );
    }

    st_status_t tjob_t::do_perform_address_assignments(
        tjob_t::assign_item_key_t const& SIXTRL_RESTRICT_REF key )
    {
        return this->do_perform_address_assignments_cuda_impl( key );
    }

    /* --------------------------------------------------------------------- */

    st_status_t tjob_t::doSetAssignOutputToBeamMonitorsKernelId(
        tjob_t::kernel_id_t const id )
    {
        return this->doSetAssignOutputToBeamMonitorsKernelIdCudaImpl( id );
    }

    st_status_t tjob_t::doSetAssignOutputToElemByElemConfigKernelId(
        tjob_t::kernel_id_t const id )
    {
        return this->doSetAssignOutputToElemByElemConfigKernelIdCudaImpl( id );
    }

    st_status_t tjob_t::doSetTrackUntilKernelId( tjob_t::kernel_id_t const id )
    {
        return this->doSetTrackUntilKernelIdCudaImpl( id );
    }

    st_status_t tjob_t::doSetTrackLineKernelId( tjob_t::kernel_id_t const id )
    {
        return this->doSetTrackLineKernelIdCudaImpl( id );
    }

    st_status_t tjob_t::doSetTrackElemByElemKernelId(
        tjob_t::kernel_id_t const id )
    {
        return this->doSetTrackElemByElemKernelIdCudaImpl( id );
    }

    st_status_t tjob_t::doSetFetchParticlesAddressesKernelId(
        tjob_t::kernel_id_t const id )
    {
        return this->doSetFetchParticlesAddressesKernelIdCudaImpl( id );
    }

    st_status_t tjob_t::do_set_assign_addresses_kernel_id(
        tjob_t::kernel_id_t const id )
    {
        return this->do_set_assign_addresses_kernel_id_cuda_impl( id );
    }

    /* --------------------------------------------------------------------- */

    tjob_t::cuda_argument_t const& tjob_t::doGetRefCudaArgument(
        tjob_t::argument_base_t const* ptr_base_arg,
        char const* SIXTRL_RESTRICT arg_name,
        bool const SIXTRL_UNUSED( requires_exact_match ) ) const
    {
        using arg_t  = tjob_t::cuda_argument_t;
        using size_t = st_size_t;

        arg_t const* cuda_arg = ( ptr_base_arg != nullptr )
            ? ptr_base_arg->asDerivedArgument< arg_t >( this->archId() )
            : nullptr;

        if( cuda_arg == nullptr )
        {
            size_t const msg_capacity = size_t{ 101 };
            size_t const max_msg_length = msg_capacity - size_t{ 1 };

            char msg[ 101 ];
            std::memset( &msg[ 0 ], int{ '\0' }, msg_capacity );

            if( ( arg_name != nullptr ) && ( std::strlen( arg_name ) > 0u ) )
            {
                std::strncpy( msg, arg_name, max_msg_length );
                std::strncat( msg, " ", max_msg_length - std::strlen( msg ) );
            }

            std::strncat( msg, "argument not available, can't dereference",
                          max_msg_length - std::strlen( msg ) );

            throw std::runtime_error( &msg[ 0 ] );
        }

        return *cuda_arg;
    }

    tjob_t::cuda_argument_t& tjob_t::doGetRefCudaArgument(
        tjob_t::argument_base_t const* ptr_base_arg,
        char const* SIXTRL_RESTRICT arg_name,
        bool const requires_exact_match )
    {
        return const_cast< tjob_t::cuda_argument_t& >( static_cast<
            tjob_t const& >( *this ).doGetRefCudaArgument( ptr_base_arg,
                arg_name, requires_exact_match ) );
    }

    tjob_t::cuda_argument_t const* tjob_t::doGetPtrCudaArgument(
        tjob_t::argument_base_t const* ptr_base_arg,
        bool const requires_exact_match ) const SIXTRL_NOEXCEPT
    {
        using arg_t = tjob_t::cuda_argument_t;

        return ( ptr_base_arg != nullptr )
            ? ptr_base_arg->asDerivedArgument< arg_t >(
                this->archId(), requires_exact_match ) : nullptr;
    }

    tjob_t::cuda_argument_t* tjob_t::doGetPtrCudaArgument(
        tjob_t::argument_base_t* ptr_base_arg,
        bool const requires_exact_match ) SIXTRL_NOEXCEPT
    {
        return const_cast< tjob_t::cuda_argument_t* >( static_cast<
            tjob_t const& >( *this ).doGetPtrCudaArgument(
                ptr_base_arg, requires_exact_match ) );
    }

    /* ===================================================================== */

    st_status_t tjob_t::doPrepareControllerCudaImpl(
        char const* SIXTRL_RESTRICT config_str )
    {
        st_status_t status = st::ARCH_STATUS_GENERAL_FAILURE;
        using cuda_ctrl_t       = tjob_t::cuda_controller_t;
        using cuda_ctrl_store_t = tjob_t::cuda_ctrl_store_t;

        using cuda_arg_t        = tjob_t::cuda_argument_t;
        using cuda_arg_store_t  = tjob_t::cuda_arg_store_t;
        using debug_register_t  = tjob_t::debug_register_t;

        SIXTRL_ASSERT( this->ptrControllerBase() == nullptr );

        cuda_ctrl_store_t ptr_store_cuda_ctrl( new cuda_ctrl_t( config_str ) );
        this->doUpdateStoredController( std::move( ptr_store_cuda_ctrl ) );

        if( ( this->hasCudaController() ) &&
            ( nullptr != this->doGetPtrLocalDebugRegister() ) )
        {
            *( this->doGetPtrLocalDebugRegister() ) =
                st::ARCH_DEBUGGING_REGISTER_EMPTY;

            cuda_arg_store_t ptr_store_debug_flag_arg( new cuda_arg_t(
                this->doGetPtrLocalDebugRegister(), sizeof( debug_register_t ),
                this->ptrCudaController() ) );

            this->doUpdateStoredDebugRegisterArg(
                std::move( ptr_store_debug_flag_arg ) );

            if( this->hasCudaDebugRegisterArg() )
            {
                status = st::ARCH_STATUS_SUCCESS;
            }
        }

        return status;
    }

    st_status_t tjob_t::doPrepareDefaultKernelsCudaImpl(
        char const* SIXTRL_RESTRICT SIXTRL_UNUSED( config_str ) )
    {
        using tjob_t = st::CudaTrackJob;
//         using _base_t = tjob_t::_base_track_job_t;
        using size_t  = st_size_t;
        using cuda_ctrl_t = tjob_t::cuda_controller_t;
//         using cuda_kernel_config_t = tjob_t::cuda_kernel_config_t;
        using kernel_id_t = tjob_t::kernel_id_t;
        using cuda_node_info_t = tjob_t::cuda_node_info_t;
        using node_index_t = tjob_t::node_index_t;

        st_status_t status = st::ARCH_STATUS_GENERAL_FAILURE;
        cuda_ctrl_t* ptr_cuda_ctrl = this->ptrCudaController();

        if( ( ptr_cuda_ctrl != nullptr ) &&
            ( ptr_cuda_ctrl->numAvailableNodes() > node_index_t{ 0 } ) &&
            ( ptr_cuda_ctrl->hasSelectedNode() ) )
        {
            node_index_t const selected_node_index =
                ptr_cuda_ctrl->selectedNodeIndex();

            cuda_node_info_t const* ptr_node_info = ptr_cuda_ctrl->ptrNodeInfo(
                selected_node_index );

            bool success = ( ( ptr_node_info != nullptr ) &&
                ( selected_node_index != st::NODE_UNDEFINED_INDEX ) );

            std::string kernel_name( 256, '\0' );
            std::string const kernel_prefix( SIXTRL_C99_NAMESPACE_PREFIX_STR );

            size_t num_kernel_args = size_t{ 0 };
            kernel_id_t kernel_id = cuda_controller_t::ILLEGAL_KERNEL_ID;

            /* trackUntilKernelId() */

            num_kernel_args = size_t{ 5 };
            kernel_name.clear();
            kernel_name += kernel_prefix;
            kernel_name += "Track_particles_until_turn_cuda_wrapper";

            kernel_id = ptr_cuda_ctrl->addCudaKernelConfig(
                kernel_name, num_kernel_args );

            success &= ( this->doSetTrackUntilKernelIdCudaImpl( kernel_id ) ==
                st::ARCH_STATUS_SUCCESS );

            /* trackElemByElemKernelId() */

            num_kernel_args = size_t{ 7 };
            kernel_name.clear();
            kernel_name += kernel_prefix;
            kernel_name +=
                "Track_particles_elem_by_elem_until_turn_cuda_wrapper";

            kernel_id = ptr_cuda_ctrl->addCudaKernelConfig(
                kernel_name, num_kernel_args );

            success &= ( this->doSetTrackElemByElemKernelIdCudaImpl(
                kernel_id ) == st::ARCH_STATUS_SUCCESS );

            /* trackLineKernelId() */

            num_kernel_args = size_t{ 7 };
            kernel_name.clear();
            kernel_name += kernel_prefix;
            kernel_name += "Track_particles_line_cuda_wrapper";

            kernel_id = ptr_cuda_ctrl->addCudaKernelConfig(
                kernel_name, num_kernel_args );

            success &= ( this->doSetTrackLineKernelIdCudaImpl(
                kernel_id ) == st::ARCH_STATUS_SUCCESS );

            /* assignOutputToBeamMonitorsKernelId() */

            num_kernel_args = size_t{ 5 };
            kernel_name.clear();
            kernel_name += kernel_prefix;
            kernel_name +=
                "BeamMonitor_assign_out_buffer_from_offset_cuda_wrapper";

            kernel_id = ptr_cuda_ctrl->addCudaKernelConfig(
                kernel_name, num_kernel_args );

            success &= ( this->doSetAssignOutputToBeamMonitorsKernelIdCudaImpl(
                kernel_id ) == st::ARCH_STATUS_SUCCESS );

            /* assignOutputToElemByElemConfigKernelId() */

            num_kernel_args = size_t{ 5 };
            kernel_name.clear();
            kernel_name += kernel_prefix;
            kernel_name +=
                "ElemByElemConfig_assign_out_buffer_from_offset_cuda_wrapper";

            kernel_id = ptr_cuda_ctrl->addCudaKernelConfig(
                kernel_name, num_kernel_args );

            success &= ( this->doSetAssignOutputToElemByElemConfigKernelIdCudaImpl(
                    kernel_id ) == st::ARCH_STATUS_SUCCESS );

            /* fetchParticlesAddressesKernelId() */

            num_kernel_args = size_t{ 3 };
            kernel_name.clear();
            kernel_name += kernel_prefix;
            kernel_name += "Particles_buffer_store_all_addresses_cuda_wrapper";

            kernel_id = ptr_cuda_ctrl->addCudaKernelConfig(
                kernel_name, num_kernel_args );

            success &= ( this->doSetFetchParticlesAddressesKernelId(
                    kernel_id ) == st::ARCH_STATUS_SUCCESS );

            if( success )
            {
                status = st::ARCH_STATUS_SUCCESS;
            }

            /* assign_addresses_kernel_id() */

            size_t nargs = size_t{ 8 };
            kernel_name.clear();
            kernel_name += kernel_prefix;
            kernel_name += "AssignAddressItem_process_managed_buffer_cuda_wrapper";
            kernel_id = ptr_cuda_ctrl->addCudaKernelConfig( kernel_name, nargs );

            success &= ( this->do_set_assign_addresses_kernel_id(
                    kernel_id ) == st::ARCH_STATUS_SUCCESS );
            if( success ) status = st::ARCH_STATUS_SUCCESS;
        }
        else if( ( ptr_cuda_ctrl != nullptr ) &&
            ( ptr_cuda_ctrl->numAvailableNodes() > node_index_t{ 0 } ) )
        {
            status = st::ARCH_STATUS_SUCCESS;
        }

        return status;
    }

    st_status_t tjob_t::doPrepareParticlesStructuresCudaImpl(
        tjob_t::c_buffer_t* SIXTRL_RESTRICT particles_buffer )
    {
        using cuda_ctrl_t      = tjob_t::cuda_controller_t;
        using cuda_arg_t       = tjob_t::cuda_argument_t;
        using store_base_arg_t = tjob_t::stored_arg_base_t;

        st_status_t status = st::ARCH_STATUS_GENERAL_FAILURE;
        cuda_ctrl_t* ptr_cuda_ctrl = this->ptrCudaController();

        if( ( ptr_cuda_ctrl != nullptr ) && ( particles_buffer != nullptr ) )
        {
            if( this->doGetPtrParticlesAddrBuffer() != nullptr )
            {
                store_base_arg_t cuda_particles_addr_arg(
                    new cuda_arg_t( *this->doGetPtrParticlesAddrBuffer(),
                        ptr_cuda_ctrl ) );

                if( ( cuda_particles_addr_arg.get() != nullptr ) &&
                    ( cuda_particles_addr_arg->ptrControllerBase() ==
                      ptr_cuda_ctrl ) &&
                    ( cuda_particles_addr_arg->usesCObjectsCxxBuffer() ) &&
                    ( cuda_particles_addr_arg->ptrCObjectsCxxBuffer() ==
                      this->doGetPtrParticlesAddrBuffer() ) )
                {
                    status = st::ARCH_STATUS_SUCCESS;
                }

                this->doUpdateStoredParticlesAddrArg(
                    std::move( cuda_particles_addr_arg ) );

                if( status == st::ARCH_STATUS_SUCCESS )
                {
                    if( ( this->ptrParticlesAddrArgBase() != nullptr ) &&
                        ( this->ptrParticlesAddrArgBase(
                            )->usesCObjectsCxxBuffer() ) &&
                        ( this->ptrParticlesAddrArgBase()->ptrCObjectsCxxBuffer()
                           == this->doGetPtrParticlesAddrBuffer() ) )
                    {
                        status = st::ARCH_STATUS_SUCCESS;
                    }
                    else
                    {
                        status = st::ARCH_STATUS_GENERAL_FAILURE;
                    }
                }
            }

            store_base_arg_t cuda_particles_arg(
                new cuda_arg_t( particles_buffer, ptr_cuda_ctrl ) );

            if( ( status == st::ARCH_STATUS_SUCCESS ) &&
                ( cuda_particles_arg.get() != nullptr ) &&
                ( cuda_particles_arg->usesCObjectsBuffer() ) &&
                ( cuda_particles_arg->ptrCObjectsBuffer() ==
                  particles_buffer  ) )
            {
                this->doUpdateStoredParticlesArg(
                    std::move( cuda_particles_arg ) );

                if( ( this->ptrParticlesArgBase() != nullptr ) &&
                    ( this->ptrParticlesArgBase()->ptrControllerBase() ==
                      ptr_cuda_ctrl ) &&
                    ( this->ptrParticlesArgBase()->usesCObjectsBuffer() ) &&
                    ( this->ptrParticlesArgBase()->ptrCObjectsBuffer() ==
                      particles_buffer ) )
                {
                    status = st::ARCH_STATUS_SUCCESS;
                }
                else
                {
                    status = st::ARCH_STATUS_GENERAL_FAILURE;
                }
            }
        }

        return status;
    }

    st_status_t tjob_t::doPrepareBeamElementsStructuresCudaImpl(
         tjob_t::c_buffer_t* SIXTRL_RESTRICT belems_buffer )
    {
        using cuda_ctrl_t      = tjob_t::cuda_controller_t;
        using cuda_arg_t       = tjob_t::cuda_argument_t;
        using store_base_arg_t = tjob_t::stored_arg_base_t;

        st_status_t status = st::ARCH_STATUS_GENERAL_FAILURE;
        cuda_ctrl_t* ptr_cuda_ctrl = this->ptrCudaController();

        if( ( ptr_cuda_ctrl != nullptr ) && ( belems_buffer != nullptr ) )
        {
            store_base_arg_t cuda_belems_arg(
                new cuda_arg_t( belems_buffer, ptr_cuda_ctrl ) );

            if( ( cuda_belems_arg.get() != nullptr ) &&
                ( cuda_belems_arg->usesCObjectsBuffer() ) &&
                ( cuda_belems_arg->ptrCObjectsBuffer() == belems_buffer ) )
            {
                this->doUpdateStoredBeamElementsArg(
                    std::move( cuda_belems_arg ) );

                if( ( this->ptrBeamElementsArgBase() != nullptr ) &&
                    ( this->ptrBeamElementsArgBase()->ptrControllerBase() ==
                      ptr_cuda_ctrl ) &&
                    ( this->ptrBeamElementsArgBase()->usesCObjectsBuffer() ) &&
                    ( this->ptrBeamElementsArgBase()->ptrCObjectsBuffer() ==
                      belems_buffer ) )
                {
                    status = st::ARCH_STATUS_SUCCESS;
                }
            }
        }

        return status;
    }

    st_status_t tjob_t::doPrepareOutputStructuresCudaImpl(
        tjob_t::c_buffer_t* SIXTRL_RESTRICT particles_buffer,
        tjob_t::c_buffer_t* SIXTRL_RESTRICT belems_buffer,
        tjob_t::c_buffer_t* SIXTRL_RESTRICT output_buffer,
        st_size_t const )
    {
        using cuda_ctrl_t      = tjob_t::cuda_controller_t;
        using cuda_arg_t       = tjob_t::cuda_argument_t;
        using store_base_arg_t = tjob_t::stored_arg_base_t;
//         using elem_config_t    = tjob_t::elem_by_elem_config_t;
//         using ctrl_status_t    = st_status_t;

        st_status_t status = st::ARCH_STATUS_GENERAL_FAILURE;
        cuda_ctrl_t* ptr_cuda_ctrl = this->ptrCudaController();

        if( ( ptr_cuda_ctrl != nullptr ) && ( output_buffer != nullptr ) &&
            ( particles_buffer != nullptr ) )
        {
            store_base_arg_t cuda_elem_by_elem_conf_arg( nullptr );

            if( this->ptrElemByElemConfig() == nullptr )
            {
                status = st::ARCH_STATUS_SUCCESS;
            }
            else
            {
                cuda_elem_by_elem_conf_arg.reset( new cuda_arg_t(
                    this->elem_by_elem_config_cxx_buffer(), ptr_cuda_ctrl ) );

                status = ( ( cuda_elem_by_elem_conf_arg.get() != nullptr ) &&
                    ( cuda_elem_by_elem_conf_arg->ptrControllerBase() ==
                      ptr_cuda_ctrl ) )
                    ? st::ARCH_STATUS_SUCCESS
                    : st::ARCH_STATUS_GENERAL_FAILURE;

                this->doUpdateStoredElemByElemConfigArg( std::move(
                    cuda_elem_by_elem_conf_arg ) );

                if( status == st::ARCH_STATUS_SUCCESS )
                {
                    if( ( this->ptrElemByElemConfigArgBase() != nullptr ) &&
                        ( this->ptrElemByElemConfigArgBase(
                            )->ptrControllerBase() == ptr_cuda_ctrl ) )
                    {
                        status = st::ARCH_STATUS_SUCCESS;
                    }
                    else
                    {
                        status = st::ARCH_STATUS_GENERAL_FAILURE;
                    }
                }

                if( status == st::ARCH_STATUS_SUCCESS )
                {
                    status = this->ptrElemByElemConfigArgBase()->send(
                        this->elem_by_elem_config_cxx_buffer() );

                    if( status == st::ARCH_STATUS_SUCCESS )
                    {
                        if( ( this->ptrElemByElemConfigArgBase(
                            )->usesCObjectsCxxBuffer() ) &&
                            ( this->ptrElemByElemConfigArgBase(
                                )->ptrCObjectsCxxBuffer() == std::addressof(
                                this->elem_by_elem_config_cxx_buffer() ) ) )
                        {
                            status = st::ARCH_STATUS_SUCCESS;
                        }
                        else
                        {
                            status = st::ARCH_STATUS_GENERAL_FAILURE;
                        }
                    }
                }
            }

            if( status == st::ARCH_STATUS_SUCCESS )
            {
                store_base_arg_t cuda_output_arg(
                    new cuda_arg_t( output_buffer, ptr_cuda_ctrl ) );

                if( ( cuda_output_arg.get() != nullptr ) &&
                    ( cuda_output_arg->ptrControllerBase() == ptr_cuda_ctrl ) &&
                    ( cuda_output_arg->usesCObjectsBuffer() ) &&
                    ( cuda_output_arg->ptrCObjectsBuffer() != nullptr ) &&
                    ( cuda_output_arg->ptrCObjectsBuffer() == output_buffer ) )
                {
                    status = st::ARCH_STATUS_SUCCESS;
                }
                else
                {
                    status = st::ARCH_STATUS_GENERAL_FAILURE;
                }

                this->doUpdateStoredOutputArg( std::move( cuda_output_arg ) );

                if( status == st::ARCH_STATUS_SUCCESS )
                {
                    if( ( this->ptrOutputArgBase() != nullptr ) &&
                        ( this->ptrOutputArgBase()->ptrControllerBase() ==
                          ptr_cuda_ctrl ) &&
                        ( this->ptrOutputArgBase()->usesCObjectsBuffer() ) &&
                        ( this->ptrOutputArgBase()->ptrCObjectsBuffer() ==
                          output_buffer ) )
                    {
                        status = st::ARCH_STATUS_SUCCESS;
                    }
                    else
                    {
                        status = st::ARCH_STATUS_GENERAL_FAILURE;
                    }
                }
            }
        }

        return status;
    }

    st_status_t tjob_t::doAssignOutputBufferToBeamMonitorsCudaImpl(
        tjob_t::c_buffer_t* SIXTRL_RESTRICT belems_buffer,
        tjob_t::c_buffer_t* SIXTRL_RESTRICT output_buffer,
        tjob_t::particle_index_t const min_turn_id,
        st_size_t const output_buffer_offset_index )
    {
        using cuda_ctrl_t        = tjob_t::cuda_controller_t;
        using cuda_arg_t         = tjob_t::cuda_argument_t;
        using cuda_kernel_conf_t = tjob_t::cuda_kernel_config_t;
        using kernel_id_t        = tjob_t::kernel_id_t;
        using ctrl_status_t      = st_status_t;
//         using size_t             = st_size_t;
        using particle_index_t   = tjob_t::particle_index_t;

        ctrl_status_t status = st::ARCH_STATUS_GENERAL_FAILURE;
        cuda_ctrl_t* ptr_cuda_ctrl = this->ptrCudaController();
        kernel_id_t const kid = this->assignOutputToBeamMonitorsKernelId();

        bool const controller_ready = (
            ( ptr_cuda_ctrl != nullptr ) &&
            ( ptr_cuda_ctrl->readyForRunningKernel() ) &&
            ( ptr_cuda_ctrl->readyForSend() ) &&
            ( ptr_cuda_ctrl->readyForReceive() ) &&
            ( ptr_cuda_ctrl->hasKernel( kid ) ) &&
            ( ptr_cuda_ctrl->ptrKernelConfigBase( kid ) != nullptr ) );

        cuda_arg_t* output_arg = this->ptrCudaOutputArg();

        bool const output_ready = ( ( this->hasBeamMonitorOutput() ) &&
            ( output_buffer != nullptr ) && ( output_arg != nullptr ) &&
            ( ::NS(Buffer_get_num_of_objects)( output_buffer ) >
              output_buffer_offset_index ) &&
            ( output_arg->ptrControllerBase() == ptr_cuda_ctrl ) &&
            ( output_arg->usesCObjectsBuffer() ) &&
            ( output_arg->ptrCObjectsBuffer() == output_buffer ) );

        cuda_arg_t* belem_arg = this->ptrCudaBeamElementsArg();

        bool const belems_ready = (
            ( belem_arg != nullptr ) && ( belems_buffer != nullptr ) &&
            ( belem_arg->ptrControllerBase() == ptr_cuda_ctrl ) &&
            ( belem_arg->usesCObjectsBuffer() ) &&
            ( belem_arg->ptrCObjectsBuffer() == belems_buffer ) );

        cuda_kernel_conf_t const* kernel_conf = ( controller_ready )
            ? ptr_cuda_ctrl->ptrKernelConfig( kid ) : nullptr;

        if( ( controller_ready ) && ( output_ready ) &&
            ( belems_ready ) && ( kernel_conf != nullptr ) )
        {
            //size_t const offset = this->beamMonitorsOutputBufferOffset();
            particle_index_t const min_turn_id = this->minInitialTurnId();

            if( !this->isInDebugMode() )
            {
                ::NS(BeamMonitor_assign_out_buffer_from_offset_cuda_wrapper)(
                    kernel_conf, belem_arg, output_arg, min_turn_id,
                    output_buffer_offset_index, nullptr );

                status = st::ARCH_STATUS_SUCCESS;
            }
            else if( this->hasCudaDebugRegisterArg() )
            {
                if( this->prepareDebugRegisterForUse() ==
                    st::ARCH_STATUS_SUCCESS )
                {
                    ::NS(BeamMonitor_assign_out_buffer_from_offset_cuda_wrapper)(
                        kernel_conf, belem_arg, output_arg, min_turn_id,
                        output_buffer_offset_index,
                        this->ptrCudaDebugRegisterArg() );

                    status = this->evaluateDebugRegisterAfterUse();
                }
            }
        }

        return status;
    }

    st_status_t tjob_t::doAssignOutputBufferToElemByElemConfigCudaImpl(
        tjob_t::elem_by_elem_config_t* SIXTRL_RESTRICT elem_by_elem_config,
        tjob_t::c_buffer_t* SIXTRL_RESTRICT output_buffer,
        st_size_t const output_buffer_offset_index )
    {
        using tjob_t            = st::CudaTrackJob;
        using cuda_ctrl_t        = tjob_t::cuda_controller_t;
        using cuda_arg_t         = tjob_t::cuda_argument_t;
        using cuda_kernel_conf_t = tjob_t::cuda_kernel_config_t;
        using kernel_id_t        = tjob_t::kernel_id_t;
        using ctrl_status_t      = st_status_t;
//         using size_t             = st_size_t;

        ctrl_status_t status = st::ARCH_STATUS_GENERAL_FAILURE;
        cuda_ctrl_t* ptr_cuda_ctrl = this->ptrCudaController();
        kernel_id_t const kid = this->assignOutputToBeamMonitorsKernelId();

        bool const controller_ready = (
            ( ptr_cuda_ctrl != nullptr ) &&
            ( ptr_cuda_ctrl->readyForRunningKernel() ) &&
            ( ptr_cuda_ctrl->readyForSend() ) &&
            ( ptr_cuda_ctrl->readyForReceive() ) &&
            ( ptr_cuda_ctrl->hasKernel( kid ) ) &&
            ( ptr_cuda_ctrl->ptrKernelConfigBase( kid ) != nullptr ) );

        cuda_arg_t* output_arg = this->ptrCudaOutputArg();

        bool const output_ready = ( ( this->hasElemByElemOutput() ) &&
            ( output_buffer != nullptr ) && ( output_arg != nullptr ) &&
            ( ::NS(Buffer_get_num_of_objects)( output_buffer ) >
              output_buffer_offset_index ) &&
            ( output_arg->ptrControllerBase() == ptr_cuda_ctrl ) &&
            ( output_arg->usesCObjectsBuffer() ) &&
            ( output_arg->ptrCObjectsBuffer() == output_buffer ) );

        cuda_arg_t* elem_by_elem_conf_arg = this->ptrCudaElemByElemConfigArg();

        bool const elem_by_elem_conf_ready = (
            ( elem_by_elem_conf_arg != nullptr ) &&
            ( elem_by_elem_config != nullptr ) &&
            ( elem_by_elem_conf_arg->ptrControllerBase() == ptr_cuda_ctrl ) &&
            ( elem_by_elem_conf_arg->usesCObjectsCxxBuffer() ) &&
            ( elem_by_elem_conf_arg->ptrCObjectsCxxBuffer() ==
                std::addressof( this->elem_by_elem_config_cxx_buffer() ) ) );

        cuda_kernel_conf_t const* kernel_conf = ( controller_ready )
            ? ptr_cuda_ctrl->ptrKernelConfig( kid ) : nullptr;

        if( ( controller_ready ) && ( output_ready ) &&
            ( elem_by_elem_conf_ready ) && ( kernel_conf != nullptr ) )
        {
            if( !this->isInDebugMode() )
            {
                ::NS(ElemByElemConfig_assign_out_buffer_from_offset_cuda_wrapper)(
                    kernel_conf, elem_by_elem_conf_arg,
                        this->elem_by_elem_config_index(), output_arg,
                            output_buffer_offset_index, nullptr );

                status = st::ARCH_STATUS_SUCCESS;
            }
            else if( this->hasCudaDebugRegisterArg() )
            {
                if( this->prepareDebugRegisterForUse() ==
                    st::ARCH_STATUS_SUCCESS )
                {
                    ::NS(ElemByElemConfig_assign_out_buffer_from_offset_cuda_wrapper)(
                        kernel_conf, elem_by_elem_conf_arg,
                            this->elem_by_elem_config_index(), output_arg,
                                output_buffer_offset_index,
                                    this->ptrCudaDebugRegisterArg() );

                    status = this->evaluateDebugRegisterAfterUse();
                }
            }
        }

        return status;
    }

    /* --------------------------------------------------------------------- */

    st_status_t tjob_t::doFetchParticleAddresses()
    {
        using tjob_t            = st::CudaTrackJob;
        using cuda_ctrl_t        = tjob_t::cuda_controller_t;
        using cuda_arg_t         = tjob_t::cuda_argument_t;
        using cuda_kernel_conf_t = tjob_t::cuda_kernel_config_t;
        using kernel_id_t        = tjob_t::kernel_id_t;
//         using ctrl_status_t      = st_status_t;
//         using size_t             = st_size_t;
//         using particle_index_t   = tjob_t::particle_index_t;
//         using debug_register_t     = tjob_t::debug_register_t;
        using buffer_t           = tjob_t::buffer_t;
        using c_buffer_t         = tjob_t::c_buffer_t;

        status_t status = st::ARCH_STATUS_GENERAL_FAILURE;

        kernel_id_t const kid = this->fetchParticlesAddressesKernelId();
        cuda_ctrl_t* ptr_ctrl = this->ptrCudaController();

        c_buffer_t* pbuffer = this->ptrCParticlesBuffer();
        buffer_t* paddr_buffer = this->doGetPtrParticlesAddrBuffer();

        bool const controller_ready = (
            ( ptr_ctrl != nullptr ) &&
            ( ptr_ctrl->readyForRunningKernel() ) &&
            ( ptr_ctrl->readyForSend() ) &&
            ( ptr_ctrl->readyForReceive() ) &&
            ( ptr_ctrl->hasKernel( kid ) ) &&
            ( ptr_ctrl->ptrKernelConfigBase( kid ) != nullptr ) );

        cuda_arg_t* particles_arg = this->ptrCudaParticlesArg();

        bool const particles_ready = (
            ( pbuffer != nullptr ) && ( particles_arg != nullptr ) &&
            ( particles_arg->ptrControllerBase() == ptr_ctrl ) &&
            ( particles_arg->usesCObjectsBuffer() ) &&
            ( particles_arg->ptrCObjectsBuffer() == pbuffer ) );

        cuda_arg_t* particles_addr_arg = this->ptrCudaParticlesAddrArg();

        bool const particles_addr_ready = (
            ( particles_addr_arg != nullptr ) && ( paddr_buffer != nullptr ) &&
            ( particles_addr_arg->ptrControllerBase() == ptr_ctrl ) &&
            ( particles_addr_arg->usesCObjectsCxxBuffer() ) &&
            ( particles_addr_arg->ptrCObjectsCxxBuffer() == paddr_buffer ) );

        cuda_kernel_conf_t const* kernel_conf = ( controller_ready )
            ? ptr_ctrl->ptrKernelConfig( kid ) : nullptr;

        //cuda_arg_t* debug_flag_arg = nullptr;

        if( ( controller_ready ) && ( particles_ready ) &&
            ( particles_addr_ready ) && ( kernel_conf != nullptr ) )
        {
            if( !this->isInDebugMode() )
            {
                ::NS(Particles_buffer_store_all_addresses_cuda_wrapper)(
                    kernel_conf, particles_addr_arg, particles_arg, nullptr );

                status = st::ARCH_STATUS_SUCCESS;
            }
            else if( this->hasCudaDebugRegisterArg() )
            {
                if( this->prepareDebugRegisterForUse() ==
                    st::ARCH_STATUS_SUCCESS )
                {
                    ::NS(Particles_buffer_store_all_addresses_cuda_wrapper)(
                        kernel_conf, particles_addr_arg, particles_arg,
                        this->ptrCudaDebugRegisterArg() );

                    status = this->evaluateDebugRegisterAfterUse();
                }
            }
        }

        return status;
    }

    st_track_status_t tjob_t::doTrackUntilTurn(
        st_size_t const until_turn )
    {
        return st::trackUntilTurn( *this, until_turn );
    }

    st_track_status_t tjob_t::doTrackElemByElem(
        st_size_t const until_turn_elem_by_elem )
    {
        return st::trackElemByElemUntilTurn( *this, until_turn_elem_by_elem );
    }

    st_track_status_t tjob_t::doTrackLine(
            st_size_t const be_begin_index,
            st_size_t const be_end_index, bool const finish_turn )
    {
        return st::trackLine(
            *this, be_begin_index, be_end_index, finish_turn );
    }

    /* --------------------------------------------------------------------- */

    st_status_t tjob_t::doResetCudaImpl(
        tjob_t::c_buffer_t* SIXTRL_RESTRICT pbuffer,
        tjob_t::c_buffer_t* SIXTRL_RESTRICT belem_buffer,
        tjob_t::c_buffer_t* SIXTRL_RESTRICT ptr_output_buffer,
        st_size_t const until_turn_elem_by_elem )
    {
s        using _base_t = tjob_t::_base_track_job_t;
        using output_buffer_flag_t = tjob_t::output_buffer_flag_t;

        st_status_t status = _base_t::doPrepareParticlesStructures( pbuffer );

        if( status == st::ARCH_STATUS_SUCCESS )
        {
            status = _base_t::doPrepareBeamElementsStructures( belem_buffer );
        }

        if( status == st::ARCH_STATUS_SUCCESS )
        {
            status = this->doPrepareParticlesStructuresCudaImpl( pbuffer );
        }

        if( status == st::ARCH_STATUS_SUCCESS )
        {
            status =
                this->doPrepareBeamElementsStructuresCudaImpl( belem_buffer );
        }

        if( status == st::ARCH_STATUS_SUCCESS )
        {
            status = this->doPrepareDefaultKernelsCudaImpl( this->ptrConfigStr() );
        }

        output_buffer_flag_t const out_buffer_flags =
        ::NS(OutputBuffer_required_for_tracking_of_particle_sets)( pbuffer,
            this->numParticleSets(), this->particleSetIndicesBegin(),
                belem_buffer, until_turn_elem_by_elem );

        bool const requires_output_buffer =
            ::NS(OutputBuffer_requires_output_buffer)( out_buffer_flags );

        if( status == st::ARCH_STATUS_SUCCESS )
        {
            this->doSetPtrCParticlesBuffer( pbuffer );
            this->doSetPtrCBeamElementsBuffer( belem_buffer );
        }

        if( ( requires_output_buffer ) || ( ptr_output_buffer != nullptr ) )
        {
            status = _base_t::doPrepareOutputStructures( pbuffer,
                belem_buffer, ptr_output_buffer, until_turn_elem_by_elem );

            if( ( status == st::ARCH_STATUS_SUCCESS ) &&
                ( this->hasOutputBuffer() ) )
            {
                status = this->doPrepareOutputStructuresCudaImpl( pbuffer,
                    belem_buffer, this->ptrCOutputBuffer(),
                        until_turn_elem_by_elem );
            }
            else if( !this->hasOutputBuffer() )
            {
                status = st::ARCH_STATUS_GENERAL_FAILURE;
            }
        }

        if( ( status == st::ARCH_STATUS_SUCCESS ) &&
            ( this->hasOutputBuffer() ) && ( requires_output_buffer ) )
        {
            if( ::NS(OutputBuffer_requires_elem_by_elem_output)( out_buffer_flags ) )
            {
                status = _base_t::doAssignOutputBufferToElemByElemConfig(
                    this->ptrElemByElemConfig(), this->ptrCOutputBuffer(),
                        this->elemByElemOutputBufferOffset() );

                if( status == st::ARCH_STATUS_SUCCESS )
                {
                    status = this->doAssignOutputBufferToElemByElemConfigCudaImpl(
                    this->ptrElemByElemConfig(), this->ptrCOutputBuffer(),
                        this->elemByElemOutputBufferOffset() );
                }
            }

            if( ::NS(OutputBuffer_requires_beam_monitor_output)(
                    out_buffer_flags ) )
            {
                if( status == st::ARCH_STATUS_SUCCESS )
                {
                    status = _base_t::doAssignOutputBufferToBeamMonitors(
                    belem_buffer, this->ptrCOutputBuffer(),
                    this->minInitialTurnId(),
                    this->beamMonitorsOutputBufferOffset() );
                }

                if( status == st::ARCH_STATUS_SUCCESS )
                {
                    status = this->doAssignOutputBufferToBeamMonitorsCudaImpl(
                    belem_buffer, this->ptrCOutputBuffer(),
                    this->minInitialTurnId(),
                    this->beamMonitorsOutputBufferOffset() );
                }
            }
        }
        else if( ( status == st::ARCH_STATUS_SUCCESS ) &&
                 ( requires_output_buffer ) )
        {
            if( !this->hasOutputBuffer() )
            {
                status = st::ARCH_STATUS_GENERAL_FAILURE;
            }
        }

        return status;
    }

    st_status_t tjob_t::doParseConfigStrCudaImpl(
        const char *const SIXTRL_RESTRICT config_str )
    {
        st_status_t status = st::ARCH_STATUS_SUCCESS;

        if( ( config_str != nullptr ) &&
            ( std::strlen( config_str ) > st_size_t{ 0 } ) )
        {
            status = st::ARCH_STATUS_SUCCESS;

            std::stringstream a2str;
            a2str << config_str;

            bool has_cuda_table = false;
            toml::value data;

            try
            {
                data = toml::parse( a2str );
            }
            catch( toml::exception& exp )
            {
                status = st::ARCH_STATUS_GENERAL_FAILURE;
            }

            if( status != st::ARCH_STATUS_SUCCESS ) return status;

            try
            {
                auto const& cuda = toml::find( data, "cuda" );

                this->m_default_threads_per_block = toml::find_or(
                    cuda, "threads_per_block",
                        tjob_t::DEFAULT_THREADS_PER_BLOCK );

                this->m_track_threads_per_block = toml::find_or(
                    cuda, "track_threads_per_block",
                        this->m_default_threads_per_block );

                if( ( this->m_default_threads_per_block ==
                        st_size_t{ 0 } ) ||
                    ( ( this->m_default_threads_per_block %
                        tjob_t::cuda_kernel_config_t::DEFAULT_WARP_SIZE ) !=
                        st_size_t{ 0 } ) )
                {
                    this->m_default_threads_per_block =
                        tjob_t::DEFAULT_THREADS_PER_BLOCK;
                }

                if( ( this->m_track_threads_per_block ==
                        st_size_t{ 0 } ) ||
                    ( ( this->m_track_threads_per_block %
                        tjob_t::cuda_kernel_config_t::DEFAULT_WARP_SIZE ) !=
                        st_size_t{ 0 } ) )
                {
                    this->m_track_threads_per_block =
                        tjob_t::DEFAULT_TRACK_THREADS_PER_BLOCK;
                }
            }
            catch( toml::exception& exp )
            {
                status = st::ARCH_STATUS_GENERAL_FAILURE;
            }
        }

        return status;
    }

    /* --------------------------------------------------------------------- */

    st_status_t tjob_t::doSetAssignOutputToBeamMonitorsKernelIdCudaImpl(
        tjob_t::kernel_id_t const kernel_id ) SIXTRL_HOST_FN
        tjob_t::kernel_id_t const kernel_id ) SIXTRL_NOEXCEPT
    {
        using _base_t         = tjob_t::_base_track_job_t;
        using cuda_ctrl_t     = tjob_t::cuda_controller_t;
        using kernel_config_t = tjob_t::cuda_kernel_config_t;
        using size_t          = st_size_t;
        using status_t        = st_status_t;
        using node_index_t    = cuda_ctrl_t::node_index_t;
        using node_info_t     = cuda_ctrl_t::node_info_t;

        tjob_t::cuda_controller_t* ptr_cuda_ctrl = this->ptrCudaController();

        node_index_t const selected_node_index = ( ptr_cuda_ctrl != nullptr )
            ? ptr_cuda_ctrl->selectedNodeIndex()
            : tjob_t::cuda_controller_t::UNDEFINED_INDEX;

        node_info_t const* node_info = ( ptr_cuda_ctrl != nullptr )
            ? ptr_cuda_ctrl->ptrNodeInfo( selected_node_index ) : nullptr;

        tjob_t::cuda_kernel_config_t* kernel_conf = ( ptr_cuda_ctrl != nullptr )
            ? ptr_cuda_ctrl->ptrKernelConfig( kernel_id ) : nullptr;

        SIXTRL_ASSERT( this->m_default_threads_per_block > size_t{ 0 } );
        SIXTRL_ASSERT( ( this->m_default_threads_per_block %
                         kernel_config_t::DEFAULT_WARP_SIZE ) == size_t{ 0 } );

        size_t num_beam_monitors = size_t{ 0 };

        if( this->hasBeamMonitorOutput() )
        {
            num_beam_monitors = this->numBeamMonitors();
        }
        else
        {
            num_beam_monitors = size_t{ 1 };
        }

        st_status_t status =
        ::NS(CudaKernelConfig_configure_assign_output_to_beam_monitors_kernel)(
            kernel_conf, node_info, num_beam_monitors,
                this->m_default_threads_per_block );

        if( status == st::ARCH_STATUS_SUCCESS )
        {
            status = _base_t::doSetAssignOutputToBeamMonitorsKernelId(
                kernel_id );
        }

        return status;
    }

    st_status_t tjob_t::doSetAssignOutputToElemByElemConfigKernelIdCudaImpl(
        tjob_t::kernel_id_t const kernel_id ) SIXTRL_NOEXCEPT
    {
        using cuda_ctrl_t     = tjob_t::cuda_controller_t;
        using kernel_config_t = tjob_t::cuda_kernel_config_t;
        using node_index_t    = cuda_ctrl_t::node_index_t;
        using node_info_t     = cuda_ctrl_t::node_info_t;

        cuda_ctrl_t* ptr_cuda_ctrl = this->ptrCudaController();

        node_index_t const selected_node_index = ( ptr_cuda_ctrl != nullptr )
            ? ptr_cuda_ctrl->selectedNodeIndex()
            : cuda_ctrl_t::UNDEFINED_INDEX;

        node_info_t const* node_info = ( ptr_cuda_ctrl != nullptr )
            ? ptr_cuda_ctrl->ptrNodeInfo( selected_node_index ) : nullptr;

        kernel_config_t* kernel_conf = ( ptr_cuda_ctrl != nullptr )
            ? ptr_cuda_ctrl->ptrKernelConfig( kernel_id ) : nullptr;

        SIXTRL_ASSERT( this->m_default_threads_per_block > size_t{ 0 } );
        SIXTRL_ASSERT( ( this->m_default_threads_per_block %
            kernel_config_t::DEFAULT_WARP_SIZE ) == size_t{ 0 } );

        st_status_t status =
        ::NS(CudaKernelConfig_configure_assign_output_to_elem_by_elem_config_kernel)(
            kernel_conf, node_info, this->m_default_threads_per_block );

        if( status == st::ARCH_STATUS_SUCCESS )
        {
            status =
            _base_t::doSetAssignOutputToElemByElemConfigKernelId( kernel_id );
        }

        return status;
    }

    st_status_t tjob_t::doSetTrackUntilKernelIdCudaImpl(
        tjob_t::kernel_id_t const kernel_id ) SIXTRL_NOEXCEPT
    {
        using _base_t         = tjob_t::_base_track_job_t;
        using cuda_ctrl_t     = tjob_t::cuda_controller_t;
        using kernel_config_t = tjob_t::cuda_kernel_config_t;
        using node_index_t    = cuda_ctrl_t::node_index_t;
        using node_info_t     = cuda_ctrl_t::node_info_t;

        cuda_ctrl_t* ptr_cuda_ctrl = this->ptrCudaController();

        node_index_t const selected_node_index = ( ptr_cuda_ctrl != nullptr )
            ? ptr_cuda_ctrl->selectedNodeIndex()
            : cuda_ctrl_t::UNDEFINED_INDEX;

        node_info_t const* node_info = ( ptr_cuda_ctrl != nullptr )
            ? ptr_cuda_ctrl->ptrNodeInfo( selected_node_index ) : nullptr;

        kernel_config_t* kernel_conf = ( ptr_cuda_ctrl != nullptr )
            ? ptr_cuda_ctrl->ptrKernelConfig( kernel_id ) : nullptr;

        SIXTRL_ASSERT( this->m_track_threads_per_block > st_size_t{ 0 } );
        SIXTRL_ASSERT( st_size_t{ 0 } == this->m_track_threads_per_block %
                         kernel_config_t::DEFAULT_WARP_SIZE );

        st_status_t status =
        ::NS(CudaKernelConfig_configure_track_until_turn_kernel)(
            kernel_conf, node_info, this->totalNumParticlesInParticleSets(),
                this->m_track_threads_per_block );

        if( status == st::ARCH_STATUS_SUCCESS )
        {
            status = _base_t::doSetTrackUntilKernelId( kernel_id );
        }

        return status;
    }

    st_status_t tjob_t::doSetTrackLineKernelIdCudaImpl(
        tjob_t::kernel_id_t const kernel_id ) SIXTRL_NOEXCEPT
    {
        using _base_t         = tjob_t::_base_track_job_t;
        using cuda_ctrl_t     = tjob_t::cuda_controller_t;
        using kernel_config_t = tjob_t::cuda_kernel_config_t;
        using node_index_t    = cuda_ctrl_t::node_index_t;
        using node_info_t     = cuda_ctrl_t::node_info_t;

        cuda_ctrl_t* ptr_cuda_ctrl = this->ptrCudaController();

        node_index_t const selected_node_index = ( ptr_cuda_ctrl != nullptr )
            ? ptr_cuda_ctrl->selectedNodeIndex()
            : cuda_ctrl_t::UNDEFINED_INDEX;

        node_info_t const* node_info = ( ptr_cuda_ctrl != nullptr )
            ? ptr_cuda_ctrl->ptrNodeInfo( selected_node_index ) : nullptr;

        kernel_config_t* kernel_conf = ( ptr_cuda_ctrl != nullptr )
            ? ptr_cuda_ctrl->ptrKernelConfig( kernel_id ) : nullptr;

        SIXTRL_ASSERT( this->m_track_threads_per_block > st_size_t{ 0 } );
        SIXTRL_ASSERT( st_size_t{ 0 } == this->m_track_threads_per_block %
                         kernel_config_t::DEFAULT_WARP_SIZE );

        st_status_t status = ::NS(CudaKernelConfig_configure_track_line_kernel)(
            kernel_conf, node_info, this->totalNumParticlesInParticleSets(),
                this->m_track_threads_per_block );

        if( status == st::ARCH_STATUS_SUCCESS )
        {
            status = _base_t::doSetTrackLineKernelId( kernel_id );
        }

        return status;
    }

    st_status_t tjob_t::doSetTrackElemByElemKernelIdCudaImpl(
        tjob_t::kernel_id_t const kernel_id ) SIXTRL_NOEXCEPT
    {
        using _base_t         = tjob_t::_base_track_job_t;
        using cuda_ctrl_t     = tjob_t::cuda_controller_t;
        using kernel_config_t = tjob_t::cuda_kernel_config_t;
        using node_index_t    = cuda_ctrl_t::node_index_t;
        using node_info_t     = cuda_ctrl_t::node_info_t;

        cuda_ctrl_t* ptr_cuda_ctrl = this->ptrCudaController();

        node_index_t const selected_node_index = ( ptr_cuda_ctrl != nullptr )
            ? ptr_cuda_ctrl->selectedNodeIndex()
            : cuda_ctrl_t::UNDEFINED_INDEX;

        node_info_t const* node_info = ( ptr_cuda_ctrl != nullptr )
            ? ptr_cuda_ctrl->ptrNodeInfo( selected_node_index ) : nullptr;

        kernel_config_t* kernel_conf = ( ptr_cuda_ctrl != nullptr )
            ? ptr_cuda_ctrl->ptrKernelConfig( kernel_id ) : nullptr;

        SIXTRL_ASSERT( this->m_track_threads_per_block > st_size_t{ 0 } );
        SIXTRL_ASSERT( st_size_t{ 0 } == this->m_track_threads_per_block %
                       kernel_config_t::DEFAULT_WARP_SIZE );

        st_status_t status =
        ::NS(CudaKernelConfig_configure_track_elem_by_elem_until_turn_kernel)(
            kernel_conf, node_info, this->totalNumParticlesInParticleSets(),
                this->m_track_threads_per_block );

        if( status == st::ARCH_STATUS_SUCCESS )
        {
            status = _base_t::doSetTrackElemByElemKernelId( kernel_id );
        }

        return status;
    }

    st_status_t tjob_t::doSetFetchParticlesAddressesKernelIdCudaImpl(
            tjob_t::kernel_id_t const kernel_id ) SIXTRL_NOEXCEPT
    {
        using _base_t         = tjob_t::_base_track_job_t;
        using cuda_ctrl_t     = tjob_t::cuda_controller_t;
        using kernel_config_t = tjob_t::cuda_kernel_config_t;
        using node_index_t    = cuda_ctrl_t::node_index_t;
        using node_info_t     = cuda_ctrl_t::node_info_t;

        cuda_ctrl_t* ptr_cuda_ctrl = this->ptrCudaController();

        node_index_t const selected_node_index = ( ptr_cuda_ctrl != nullptr )
            ? ptr_cuda_ctrl->selectedNodeIndex()
            : cuda_ctrl_t::UNDEFINED_INDEX;

        node_info_t const* node_info = ( ptr_cuda_ctrl != nullptr )
            ? ptr_cuda_ctrl->ptrNodeInfo( selected_node_index ) : nullptr;

        kernel_config_t* kernel_conf = ( ptr_cuda_ctrl != nullptr )
            ? ptr_cuda_ctrl->ptrKernelConfig( kernel_id ) : nullptr;

        SIXTRL_ASSERT( this->m_default_threads_per_block > st_size_t{ 0 } );
        SIXTRL_ASSERT( st_size_t{ 0 } == this->m_default_threads_per_block %
                         kernel_config_t::DEFAULT_WARP_SIZE );

        size_t num_particle_sets = size_t{ 0 };

        if( this->doGetPtrParticlesAddrBuffer() != nullptr )
        {
            num_particle_sets = this->doGetPtrParticlesAddrBuffer()->getSize();
        }
        else if( this->ptrCParticlesBuffer() != nullptr )
        {
            num_particle_sets = ::NS(Buffer_get_num_of_objects)(
                this->ptrCParticlesBuffer() );
        }

        st_status_t status =
        ::NS(CudaKernelConfig_configure_track_line_kernel)( kernel_conf,
            node_info, num_particle_sets, this->m_default_threads_per_block );

        if( status == st::ARCH_STATUS_SUCCESS )
        {
            status = _base_t::doSetFetchParticlesAddressesKernelId( kernel_id );
        }

        return status;
    }

    st_status_t tjob_t::do_set_assign_addresses_kernel_id_cuda_impl(
        tjob_t::kernel_id_t const kernel_id ) SIXTRL_NOEXCEPT
    {
        using _base_t         = tjob_t::_base_track_job_t;
        using cuda_ctrl_t     = tjob_t::cuda_controller_t;
        using kernel_config_t = tjob_t::cuda_kernel_config_t;
        using status_t        = st_status_t;
        using node_index_t    = cuda_ctrl_t::node_index_t;
        using node_info_t     = cuda_ctrl_t::node_info_t;

        cuda_ctrl_t* ptr_cuda_ctrl = this->ptrCudaController();

        node_index_t const selected_node_index = ( ptr_cuda_ctrl != nullptr )
            ? ptr_cuda_ctrl->selectedNodeIndex()
            : cuda_ctrl_t::UNDEFINED_INDEX;

        node_info_t const* node_info = ( ptr_cuda_ctrl != nullptr )
            ? ptr_cuda_ctrl->ptrNodeInfo( selected_node_index ) : nullptr;

        kernel_config_t* kernel_conf = ( ptr_cuda_ctrl != nullptr )
            ? ptr_cuda_ctrl->ptrKernelConfig( kernel_id ) : nullptr;

        status_t status = ::NS(CudaKernelConfig_configure_assign_address_kernel)(
            kernel_conf, node_info );

        if( status == st::ARCH_STATUS_SUCCESS )
        {
            status = _base_t::do_set_assign_addresses_kernel_id( kernel_id );
        }

        return status;
    }

    /* --------------------------------------------------------------------- */

    st_status_t tjob_t::do_add_stored_buffer_cuda_impl(
        st_size_t const buffer_id )
    {
        st_status_t status = st::ARCH_STATUS_GENERAL_FAILURE;
        tjob_t::buffer_store_t* ptr_stored_buffer =
                this->do_get_ptr_buffer_store( buffer_id );

        tjob_t::cuda_controller_t* controller = this->ptrCudaController();

        if( ( controller != nullptr ) && ( ptr_stored_buffer != nullptr ) &&
            ( buffer_id >= st::ARCH_MIN_USER_DEFINED_BUFFER_ID ) )
        {
            st_size_t const stored_buffer_id =
                buffer_id - st::ARCH_MIN_USER_DEFINED_BUFFER_ID;

            st_size_t const nn = this->do_get_stored_buffer_size();
            st_size_t ii = this->m_stored_buffer_args.size();

            for( ; ii < nn ; ++ii )
            {
                this->m_stored_buffer_args.emplace_back( nullptr );
            }

            SIXTRL_ASSERT( this->m_stored_buffer_args.size() >= nn );
            SIXTRL_ASSERT( this->m_stored_buffer_args.size() >
                           stored_buffer_id );

            if( ptr_stored_buffer->ptr_cxx_buffer() != nullptr )
            {
                tjob_t::buffer_t& stored_buffer =
                    *ptr_stored_buffer->ptr_cxx_buffer();

                this->m_stored_buffer_args[ stored_buffer_id ].reset(
                    new tjob_t::cuda_argument_t( stored_buffer, controller ) );
                status = st::ARCH_STATUS_SUCCESS;
            }
            else if( ptr_stored_buffer->ptr_buffer() != nullptr )
            {
                tjob_t::c_buffer_t* stored_buffer =
                    ptr_stored_buffer->ptr_buffer();

                this->m_stored_buffer_args[ stored_buffer_id ].reset( new
                    tjob_t::cuda_argument_t( stored_buffer, controller ) );
                status = st::ARCH_STATUS_SUCCESS;
            }
        }

        return status;
    }

    st_status_t tjob_t::do_remove_stored_buffer_cuda_impl(
        st_size_t const buffer_id )
    {
        st_status_t status = st::ARCH_STATUS_GENERAL_FAILURE;
        if( buffer_id >= st::ARCH_MIN_USER_DEFINED_BUFFER_ID )
        {
            st_size_t const stored_buffer_id =
                buffer_id - st::ARCH_MIN_USER_DEFINED_BUFFER_ID;

            SIXTRL_ASSERT( this->m_stored_buffer_args.size() > stored_buffer_id );
            this->m_stored_buffer_args[ stored_buffer_id ].reset( nullptr );
            status = st::ARCH_STATUS_SUCCESS;
        }

        return status;
    }

    st_status_t tjob_t::do_push_stored_buffer_cuda_impl(
        st_size_t const buffer_id )
    {
        st_status_t status = st::ARCH_STATUS_GENERAL_FAILURE;

        tjob_t::buffer_store_t* ptr_stored_buffer =
                this->do_get_ptr_buffer_store( buffer_id );

        tjob_t::cuda_argument_t* ptr_arg =
            this->ptr_stored_buffer_argument( buffer_id );

        if( ( ptr_stored_buffer != nullptr ) && ( ptr_arg != nullptr ) )
        {
            SIXTRL_ASSERT( ptr_arg->ptrControllerBase() ==
                           this->ptrControllerBase() );

            if( ptr_stored_buffer->ptr_cxx_buffer() != nullptr )
            {
                status = ptr_arg->send( *ptr_stored_buffer->ptr_cxx_buffer() );
            }
            else if( ptr_stored_buffer->ptr_buffer() != nullptr )
            {
                status = ptr_arg->send( ptr_stored_buffer->ptr_buffer() );
            }
        }

        return status;
    }

    st_status_t tjob_t::do_collect_stored_buffer_cuda_impl(
        st_size_t const buffer_id )
    {
        st_status_t status = st::ARCH_STATUS_GENERAL_FAILURE;

        tjob_t::buffer_store_t* stored_buffer =
                this->do_get_ptr_buffer_store( buffer_id );

        tjob_t::cuda_argument_t* ptr_arg =
            this->ptr_stored_buffer_argument( buffer_id );

        if( ( stored_buffer != nullptr ) && ( ptr_arg != nullptr ) )
        {
            SIXTRL_ASSERT( ptr_arg->ptrControllerBase() ==
                          this->ptrControllerBase() );

            if( stored_buffer->ptr_cxx_buffer() != nullptr )
            {
                status = ptr_arg->receive( *stored_buffer->ptr_cxx_buffer() );
            }
            else if( stored_buffer->ptr_buffer() != nullptr )
            {
                status = ptr_arg->receive( stored_buffer->ptr_buffer() );
            }
        }

        return status;
    }

    st_status_t tjob_t::do_add_assign_address_cuda_impl(
        assign_item_t const& SIXTRL_RESTRICT_REF assign_item,
        st_size_t* SIXTRL_RESTRICT ptr_item_index )
    {
        ( void )assign_item;
        ( void )ptr_item_index;

        return st::ARCH_STATUS_SUCCESS;
    }

    st_status_t tjob_t::do_perform_address_assignments_cuda_impl(
        tjob_t::assign_item_key_t const& SIXTRL_RESTRICT_REF key )
    {
        st_status_t status = st::ARCH_STATUS_GENERAL_FAILURE;

        st_size_t const dest_buffer_id = key.dest_buffer_id;
        tjob_t::c_buffer_t* dest_buffer =
            this->buffer_by_buffer_id( dest_buffer_id );
        tjob_t::cuda_argument_t* dest_arg =
            this->ptr_argument_by_buffer_id( dest_buffer_id );

        size_t const src_buffer_id = key.src_buffer_id;
        c_buffer_t const* src_buffer =
            this->buffer_by_buffer_id( src_buffer_id );
        tjob_t::cuda_argument_t* src_arg =
            this->ptr_argument_by_buffer_id( src_buffer_id );

        c_buffer_t* assign_buffer =
            this->do_get_ptr_assign_address_items_buffer( key );

        if( ( this->has_assign_items( dest_buffer_id, src_buffer_id ) ) &&
            ( this->ptrCudaController() != nullptr ) &&
            ( assign_buffer != nullptr ) &&
            ( src_buffer != nullptr ) && ( dest_buffer != nullptr ) &&
            ( src_arg != nullptr ) && ( src_arg->usesCObjectsBuffer() ) &&
            ( src_arg->ptrCObjectsBuffer() == src_buffer ) &&
            ( src_arg->cudaController() == this->ptrCudaController() ) &&
            ( dest_arg != nullptr ) && ( dest_arg->usesCObjectsBuffer() ) &&
            ( dest_arg->ptrCObjectsBuffer() == dest_buffer ) &&
            ( dest_arg->cudaController() == this->ptrCudaController() ) &&
            ( this->has_assign_addresses_kernel() ) )
        {
            tjob_t::kernel_id_t const kid = this->assign_addresses_kernel_id();
            tjob_t::cuda_kernel_config_t* kernel_conf =
                this->ptrCudaController()->ptrKernelConfig( kid );

            if( ( kid != st::ARCH_ILLEGAL_KERNEL_ID ) &&
                ( kernel_conf != nullptr ) )
            {
                tjob_t::cuda_argument_t assign_items_arg(
                    assign_buffer, this->ptrCudaController() );

                NS(AssignAddressItem_process_managed_buffer_cuda_wrapper)(
                    kernel_conf, &assign_items_arg, dest_arg, dest_buffer_id,
                        src_arg, src_buffer_id );

                status = st::ARCH_STATUS_SUCCESS;
            }
        }

        return status;
    }

    /* ********************************************************************* */
    /* ********    Implement CudaTrackJob stand-alone functions   ********** */
    /* ********************************************************************* */

    tjob_t::collect_flag_t collect(
        CudaTrackJob& SIXTRL_RESTRICT_REF track_job )
    {
        return track_job.collect();
    }

    tjob_t::push_flag_t push(
        CudaTrackJob& SIXTRL_RESTRICT_REF track_job,
        tjob_t::push_flag_t const flag )
    {
        return track_job.push( flag );
    }

    st_track_status_t trackUntilTurn(
        CudaTrackJob& SIXTRL_RESTRICT_REF trackjob,
        st_size_t const until_turn )
    {
        using track_job_t     = CudaTrackJob;
        using controller_t    = track_job_t::cuda_controller_t;
//         using argument_t      = track_job_t::cuda_argument_t;
        using kernel_config_t = track_job_t::cuda_kernel_config_t;
        using kernel_id_t     = track_job_t::kernel_id_t;
        using size_t          = track_job_t::size_type;

        st_track_status_t status = st::TRACK_STATUS_GENERAL_FAILURE;

        kernel_id_t const kid = trackjob.trackUntilKernelId();
        SIXTRL_ASSERT( kid != controller_t::ILLEGAL_KERNEL_ID );

        controller_t const* ptr_ctrl = trackjob.ptrCudaController();
        SIXTRL_ASSERT( ptr_ctrl != nullptr );
        SIXTRL_ASSERT( ptr_ctrl->hasSelectedNode() );
        SIXTRL_ASSERT( ptr_ctrl->readyForRunningKernel() );

        kernel_config_t const* kernel_conf = ptr_ctrl->ptrKernelConfig( kid );
        SIXTRL_ASSERT( kernel_conf != nullptr );
        SIXTRL_ASSERT( trackjob.hasCudaParticlesArg() );
        SIXTRL_ASSERT( trackjob.hasCudaBeamElementsArg() );

        SIXTRL_ASSERT( trackjob.numParticleSets() == size_t{ 1 } );
        SIXTRL_ASSERT( trackjob.particleSetIndicesBegin() != nullptr );

        size_t const pset_index = *trackjob.particleSetIndicesBegin();

        if( !trackjob.isInDebugMode() )
        {
            ::NS(Track_particles_until_turn_cuda_wrapper)( kernel_conf,
                trackjob.ptrCudaParticlesArg(), pset_index,
                trackjob.ptrCudaBeamElementsArg(), until_turn, nullptr );

            status = st::TRACK_SUCCESS;
        }
        else if( trackjob.hasCudaDebugRegisterArg() )
        {
            using ctrl_status_t = track_job_t::status_t;

            ctrl_status_t ctrl_status = trackjob.prepareDebugRegisterForUse();

            if( ctrl_status == st::ARCH_STATUS_SUCCESS )
            {
                ::NS(Track_particles_until_turn_cuda_wrapper)( kernel_conf,
                    trackjob.ptrCudaParticlesArg(), pset_index,
                    trackjob.ptrCudaBeamElementsArg(), until_turn,
                    trackjob.ptrCudaDebugRegisterArg() );

                ctrl_status = trackjob.evaluateDebugRegisterAfterUse();
            }

            if( ctrl_status == st::ARCH_STATUS_SUCCESS )
            {
                status = st::TRACK_SUCCESS;
            }
            else
            {
                status = static_cast< track_status_t >( ctrl_status );
            }
        }

        return status;
    }

    st_track_status_t trackElemByElemUntilTurn(
        CudaTrackJob& SIXTRL_RESTRICT_REF trackjob,
        st_size_t const until_turn_elem_by_elem )
        st_size_t const until_turn_elem_by_elem )
    {
        using track_job_t     = CudaTrackJob;
        using controller_t    = track_job_t::cuda_controller_t;
//         using argument_t      = track_job_t::cuda_argument_t;
        using kernel_config_t = track_job_t::cuda_kernel_config_t;
        using kernel_id_t     = track_job_t::kernel_id_t;

        st_track_status_t status = st::TRACK_STATUS_GENERAL_FAILURE;

        kernel_id_t const kid = trackjob.trackElemByElemKernelId();
        SIXTRL_ASSERT( kid != controller_t::ILLEGAL_KERNEL_ID );

        controller_t const* ptr_ctrl = trackjob.ptrCudaController();
        SIXTRL_ASSERT( ptr_ctrl != nullptr );
        SIXTRL_ASSERT( ptr_ctrl->hasSelectedNode() );
        SIXTRL_ASSERT( ptr_ctrl->readyForRunningKernel() );

        kernel_config_t const* kernel_conf = ptr_ctrl->ptrKernelConfig( kid );
        SIXTRL_ASSERT( kernel_conf != nullptr );
        SIXTRL_ASSERT( trackjob.hasCudaParticlesArg() );
        SIXTRL_ASSERT( trackjob.hasCudaBeamElementsArg() );
        SIXTRL_ASSERT( trackjob.hasCudaElemByElemConfigArg() );
        SIXTRL_ASSERT( trackjob.hasCudaOutputArg() );
        SIXTRL_ASSERT( trackjob.ptrElemByElemConfig() != nullptr );
        SIXTRL_ASSERT( trackjob.hasElemByElemOutput() );

        SIXTRL_ASSERT( trackjob.numParticleSets() == size_t{ 1 } );
        SIXTRL_ASSERT( trackjob.particleSetIndicesBegin() != nullptr );

        size_t const pset_index = *trackjob.particleSetIndicesBegin();

        if( !trackjob.isInDebugMode() )
        {
            ::NS(Track_particles_elem_by_elem_until_turn_cuda_wrapper)(
                kernel_conf, trackjob.ptrCudaParticlesArg(), pset_index,
                trackjob.ptrCudaBeamElementsArg(),
                trackjob.ptrCudaElemByElemConfigArg(),
                trackjob.elem_by_elem_config_index(), until_turn_elem_by_elem,
                nullptr );

            status = st::TRACK_SUCCESS;
        }
        else if( trackjob.hasCudaDebugRegisterArg() )
        {
            using ctrl_status_t = track_job_t::status_t;

            ctrl_status_t ctrl_status = trackjob.prepareDebugRegisterForUse();

            if( ctrl_status == st::ARCH_STATUS_SUCCESS )
            {
                ::NS(Track_particles_elem_by_elem_until_turn_cuda_wrapper)(
                    kernel_conf, trackjob.ptrCudaParticlesArg(), pset_index,
                    trackjob.ptrCudaBeamElementsArg(),
                    trackjob.ptrCudaElemByElemConfigArg(),
                    trackjob.elem_by_elem_config_index(),
                    until_turn_elem_by_elem,
                    trackjob.ptrCudaDebugRegisterArg() );

                ctrl_status = trackjob.evaluateDebugRegisterAfterUse();
            }

            if( ctrl_status == st::ARCH_STATUS_SUCCESS )
            {
                status = st::TRACK_SUCCESS;
            }
            else
            {
                status = static_cast< track_status_t >( ctrl_status );
            }
        }

        return status;
    }

    st_track_status_t trackLine(
        CudaTrackJob& SIXTRL_RESTRICT_REF trackjob,
        st_size_t const belem_begin_id,
        st_size_t const belem_end_id, bool const finish_turn )
    {
        using track_job_t     = CudaTrackJob;
        using controller_t    = track_job_t::cuda_controller_t;
//         using argument_t      = track_job_t::cuda_argument_t;
        using kernel_config_t = track_job_t::cuda_kernel_config_t;
        using kernel_id_t     = track_job_t::kernel_id_t;

        st_track_status_t status = st::TRACK_STATUS_GENERAL_FAILURE;

        kernel_id_t const kid = trackjob.trackElemByElemKernelId();
        SIXTRL_ASSERT( kid != controller_t::ILLEGAL_KERNEL_ID );

        controller_t const* ptr_ctrl = trackjob.ptrCudaController();
        SIXTRL_ASSERT( ptr_ctrl != nullptr );
        SIXTRL_ASSERT( ptr_ctrl->hasSelectedNode() );
        SIXTRL_ASSERT( ptr_ctrl->readyForRunningKernel() );

        kernel_config_t const* kernel_conf = ptr_ctrl->ptrKernelConfig( kid );
        SIXTRL_ASSERT( kernel_conf != nullptr );
        SIXTRL_ASSERT( trackjob.hasCudaParticlesArg() );
        SIXTRL_ASSERT( trackjob.hasCudaBeamElementsArg() );

        SIXTRL_ASSERT( trackjob.numParticleSets() == size_t{ 1 } );
        SIXTRL_ASSERT( trackjob.particleSetIndicesBegin() != nullptr );

        size_t const pset_index = *trackjob.particleSetIndicesBegin();

        if( !trackjob.isInDebugMode() )
        {
            ::NS(Track_particles_line_cuda_wrapper)(
                kernel_conf, trackjob.ptrCudaParticlesArg(), pset_index,
                trackjob.ptrCudaBeamElementsArg(),
                belem_begin_id, belem_end_id, finish_turn, nullptr );

            status = st::TRACK_SUCCESS;
        }
        else if( trackjob.hasCudaDebugRegisterArg() )
        {
            using ctrl_status_t = track_job_t::status_t;

            ctrl_status_t ctrl_status = trackjob.prepareDebugRegisterForUse();

            if( ctrl_status == st::ARCH_STATUS_SUCCESS )
            {
                ::NS(Track_particles_line_cuda_wrapper)( kernel_conf,
                    trackjob.ptrCudaParticlesArg(), pset_index,
                    trackjob.ptrCudaBeamElementsArg(),
                    belem_begin_id, belem_end_id, finish_turn,
                    trackjob.ptrCudaDebugRegisterArg() );

                ctrl_status = trackjob.evaluateDebugRegisterAfterUse();
            }

            if( ctrl_status == st::ARCH_STATUS_SUCCESS )
            {
                status = st::TRACK_SUCCESS;
            }
            else
            {
                status = static_cast< track_status_t >( ctrl_status );
            }
        }

        return status;
    }
}
#endif /* C++, Host */
