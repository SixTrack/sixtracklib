#include "sixtracklib/cuda/control/kernel_config.hpp"

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #if defined( __cplusplus ) && !defined( _GPUCODE ) && \
       !defined( __CUDA_ARCH__ )
        #include <cstddef>
        #include <cstdlib>
        #include <iomanip>
        #include <ostream>
    #endif /* C++, Host */

    #include <cuda_runtime_api.h>

#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/control/definitions.h"
    #include "sixtracklib/common/control/kernel_config_base.hpp"
    #include "sixtracklib/cuda/definitions.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus ) && !defined( _GPUCODE ) && !defined( __CUDA_ARCH__ )

namespace st = SIXTRL_CXX_NAMESPACE;

namespace SIXTRL_CXX_NAMESPACE
{
    CudaKernelConfig::CudaKernelConfig(
        CudaKernelConfig::size_type const block_dimensions,
        CudaKernelConfig::size_type const threads_per_block_dimensions,
        CudaKernelConfig::size_type const shared_mem_per_block,
        CudaKernelConfig::size_type const max_block_size_limit,
        CudaKernelConfig::size_type const warp_size,
        char const* SIXTRL_RESTRICT config_str ) :
        st::KernelConfigBase( st::ARCHITECTURE_CUDA,
            SIXTRL_ARCHITECTURE_CUDA_STR, config_str, block_dimensions,
            threads_per_block_dimensions ),
            m_blocks(), m_threads_per_block(),
            m_warp_size( CudaKernelConfig::DEFAULT_WARP_SIZE ),
            m_shared_mem_per_block( shared_mem_per_block ),
            m_max_block_size_limit( max_block_size_limit )
    {
        using _this_t = st::CudaKernelConfig;
        using size_t = _this_t::size_type;

        if( warp_size > size_t{ 0 } )
        {
            this->setWarpSize( warp_size );
        }
    }

    CudaKernelConfig::CudaKernelConfig(
        std::string const& SIXTRL_RESTRICT_REF name_str,
        CudaKernelConfig::size_type const num_kernel_arguments,
        CudaKernelConfig::size_type const block_dimensions,
        CudaKernelConfig::size_type const threads_per_block_dimensions,
        CudaKernelConfig::size_type const shared_mem_per_block,
        CudaKernelConfig::size_type const max_block_size_limit,
        CudaKernelConfig::size_type const warp_size,
        char const* SIXTRL_RESTRICT config_str ) :
        st::KernelConfigBase( st::ARCHITECTURE_CUDA,
            SIXTRL_ARCHITECTURE_CUDA_STR, config_str, block_dimensions,
            threads_per_block_dimensions ),
            m_blocks(), m_threads_per_block(),
            m_warp_size( CudaKernelConfig::DEFAULT_WARP_SIZE ),
            m_shared_mem_per_block( shared_mem_per_block ),
            m_max_block_size_limit( max_block_size_limit )
    {
        using _this_t = st::CudaKernelConfig;
        using size_t = _this_t::size_type;

        if( warp_size > size_t{ 0 } )
        {
            this->setWarpSize( warp_size );
        }

        this->setNumArguments( num_kernel_arguments );
        this->setName( name_str );
    }

    CudaKernelConfig::CudaKernelConfig(
        char const* SIXTRL_RESTRICT name_str,
        CudaKernelConfig::size_type const num_kernel_arguments,
        CudaKernelConfig::size_type const block_dimensions,
        CudaKernelConfig::size_type const threads_per_block_dimensions,
        CudaKernelConfig::size_type const shared_mem_per_block,
        CudaKernelConfig::size_type const max_block_size_limit,
        CudaKernelConfig::size_type const warp_size,
        char const* SIXTRL_RESTRICT config_str ) :
        st::KernelConfigBase( st::ARCHITECTURE_CUDA,
            SIXTRL_ARCHITECTURE_CUDA_STR, config_str, block_dimensions,
            threads_per_block_dimensions ),
            m_blocks(), m_threads_per_block(),
            m_warp_size( CudaKernelConfig::DEFAULT_WARP_SIZE ),
            m_shared_mem_per_block( shared_mem_per_block ),
            m_max_block_size_limit( max_block_size_limit )
    {
        using _this_t = st::CudaKernelConfig;
        using size_t = _this_t::size_type;

        if( warp_size > size_t{ 0 } )
        {
            this->setWarpSize( warp_size );
        }

        this->setNumArguments( num_kernel_arguments );
        this->setName( name_str );
    }

    CudaKernelConfig::size_type
    CudaKernelConfig::sharedMemPerBlock() const SIXTRL_NOEXCEPT
    {
        return this->m_shared_mem_per_block;
    }

    void CudaKernelConfig::setSharedMemPerBlock(
        CudaKernelConfig::size_type const shared_mem_per_block ) SIXTRL_NOEXCEPT
    {
        this->m_shared_mem_per_block = shared_mem_per_block;
    }

    CudaKernelConfig::size_type
    CudaKernelConfig::maxBlockSizeLimit() const SIXTRL_NOEXCEPT
    {
        return this->m_max_block_size_limit;
    }

    void CudaKernelConfig::setMaxBlockSizeLimit( CudaKernelConfig::size_type
        const max_block_size_limit ) SIXTRL_NOEXCEPT
    {
        this->m_max_block_size_limit = max_block_size_limit;
    }

    ::dim3 const& CudaKernelConfig::blocks() const SIXTRL_NOEXCEPT
    {
        return this->m_blocks;
    }

    ::dim3 const& CudaKernelConfig::threadsPerBlock() const SIXTRL_NOEXCEPT
    {
        return this->m_threads_per_block;
    }

    ::dim3 const* CudaKernelConfig::ptrBlocks() const SIXTRL_NOEXCEPT
    {
        return ( !this->needsUpdate() ) ? &this->m_blocks : nullptr;
    }

    ::dim3 const* CudaKernelConfig::ptrThreadsPerBlock() const SIXTRL_NOEXCEPT
    {
        return ( !this->needsUpdate() ) ? &this->m_threads_per_block : nullptr;
    }

    CudaKernelConfig::size_type
    CudaKernelConfig::warpSize() const SIXTRL_NOEXCEPT
    {
        return this->m_warp_size;
    }

    void CudaKernelConfig::setWarpSize(
        CudaKernelConfig::size_type const warp_size ) SIXTRL_NOEXCEPT
    {
        if( warp_size > CudaKernelConfig::size_type{ 0 } )
        {
            this->m_warp_size = warp_size;
        }

        return;
    }

    bool CudaKernelConfig::doUpdate()
    {
        bool success = false;

        using size_t = CudaKernelConfig::size_type;

        size_t const items_dim = this->workItemsDim();
        size_t const wgroups_dim = this->workGroupsDim();

        if( ( items_dim > size_t{ 0 } ) && ( wgroups_dim == items_dim ) )
        {
            size_t ii = size_t{ 0 };
            success = true;

            size_t num_blocks[]  = { size_t{ 1 }, size_t{ 1 }, size_t{ 1 } };
            size_t num_threads[] = { size_t{ 1 }, size_t{ 1 }, size_t{ 1 } };

            for( ; ii < items_dim ; ++ii )
            {
                size_t const wgsize = this->workGroupSize( ii );
                size_t const num_items = this->numWorkItems( ii );

                if( ( wgsize == size_t{ 0 } ) ||
                    ( num_items == size_t{ 0 } ) )
                {
                    success = false;
                    break;
                }

                num_threads[ ii ] = wgsize;
                num_blocks[ ii ]  = num_items / wgsize;

                if( size_t{ 0 } != ( num_items % wgsize ) )
                {
                    ++num_blocks[ ii ];
                }
            }

            if( success )
            {
                size_t const total_num_blocks =
                    num_blocks[ 0 ] * num_blocks[ 1 ] * num_blocks[ 2 ];

                this->m_blocks.x = num_blocks[ 0 ];
                this->m_blocks.y = num_blocks[ 1 ];
                this->m_blocks.z = num_blocks[ 2 ];

                size_t const threads_per_block =
                    num_threads[ 0 ] * num_threads[ 1 ] * num_threads[ 2 ];

                this->m_threads_per_block.x = num_threads[ 0 ];
                this->m_threads_per_block.y = num_threads[ 1 ];
                this->m_threads_per_block.z = num_threads[ 2 ];

                success = ( ( total_num_blocks  > size_t{ 0 } ) &&
                            ( threads_per_block > size_t{ 0 } ) );
            }
        }

        return success;
    }

    CudaKernelConfig::size_type
    CudaKernelConfig::totalNumBlocks() const SIXTRL_NOEXCEPT
    {
        return ( !this->needsUpdate() )
            ? ( this->m_blocks.x * this->m_blocks.y * this->m_blocks.z )
            : CudaKernelConfig::size_type{ 0 };
    }

    CudaKernelConfig::size_type
    CudaKernelConfig::totalNumThreadsPerBlock() const SIXTRL_NOEXCEPT
    {
        return ( !this->needsUpdate() )
            ? ( this->m_threads_per_block.x * this->m_threads_per_block.y *
                this->m_threads_per_block.z )
            : CudaKernelConfig::size_type{ 0 };
    }

    CudaKernelConfig::size_type
    CudaKernelConfig::totalNumThreads() const SIXTRL_NOEXCEPT
    {
        return this->totalNumBlocks() * this->totalNumThreadsPerBlock();
    }

    void CudaKernelConfig::doPrintToOutputStream(
        std::ostream& SIXTRL_RESTRICT_REF output ) const
    {
        using size_t = KernelConfigBase::size_type;

        if( this->needsUpdate() )
        {
            output << "!!! WARNING: Preliminary values, "
                   << "call update() before using !!!\r\n\r\n";
        }

        if( this->hasName() )
        {
            output << "kernel name          : " << this->name() << "\r\n";
        }

        output << "num kernel arguments : "
               << std::setw( 6 ) << this->numArguments() << "\r\n";

        if( this->workItemsDim() > size_t{ 0 } )
        {
            output << "block dim            : " << this->workItemsDim()
                   << "blocks               : [ "
                   << std::setw( 5 ) << this->blocks().x << " / "
                   << std::setw( 5 ) << this->blocks().y << " / "
                   << std::setw( 5 ) << this->blocks().z << " ]\r\n";
        }

        if( this->workGroupsDim() > size_t{ 0 } )
        {
            output << "threads grid dim     : " << this->workItemsDim()
                   << "threads per block    : [ "
                   << std::setw( 5 ) << this->threadsPerBlock().x << " / "
                   << std::setw( 5 ) << this->threadsPerBlock().y << " / "
                   << std::setw( 5 ) << this->threadsPerBlock().z
                   << " ]\r\n";
        }

        output << "shared mem per block : "
               << std::setw( 6 ) << this->sharedMemPerBlock()
               << " bytes\r\n"
               << "warp size            : "
               << std::setw( 6 ) << this->warpSize() << " threads\r\n";

        if( this->maxBlockSizeLimit() > size_t{ 0 } )
        {
            output << "max block size limit : " << std::setw( 6 )
                   << this->maxBlockSizeLimit() << "\r\n";
        }

        return;
    }
}

#endif /* C++, Host */

/* end: sixtracklib/cuda/control/kernel_config.cpp */
