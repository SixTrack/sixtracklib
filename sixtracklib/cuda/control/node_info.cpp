#include "sixtracklib/cuda/control/node_info.hpp"

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sstream>

#include <cuda.h>
#include <cuda_runtime.h>

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/control/definitions.h"
#include "sixtracklib/common/control/node_id.hpp"
#include "sixtracklib/common/control/node_info.hpp"
#include "sixtracklib/common/control/arch_info.hpp"

namespace SIXTRL_CXX_NAMESPACE
{
    CudaNodeInfo::CudaNodeInfo(
        CudaNodeInfo::cuda_dev_index_t const cuda_dev_index ) :
        SIXTRL_CXX_NAMESPACE::NodeInfoBase(
            SIXTRL_CXX_NAMESPACE::ARCHITECTURE_CUDA,
            SIXTRL_ARCHITECTURE_CUDA_STR ),
        m_cu_device_properties(),
        m_cu_device_index( cuda_dev_index ),
        m_cu_device_pci_bus_id( "0000:00:00.0" )
    {

    }

    CudaNodeInfo::CudaNodeInfo(
        CudaNodeInfo::cuda_dev_index_t const cuda_dev_index,
        ::cudaDeviceProp const& cuda_device_properties ) :
        SIXTRL_CXX_NAMESPACE::NodeInfoBase(
            SIXTRL_CXX_NAMESPACE::ARCHITECTURE_CUDA,
            SIXTRL_ARCHITECTURE_CUDA_STR ),
        m_cu_device_properties( cuda_device_properties ),
        m_cu_device_index( cuda_dev_index ),
        m_cu_device_pci_bus_id( "0000:00:00.0" )
    {
        using size_t = CudaNodeInfo::size_type;

        ::cudaError_t err = ::cudaSuccess;
        int max_num_devices = int{ 0 };

        if( cuda_device_properties.major >= 2 )
        {
            err = ::cudaGetDeviceCount( &max_num_devices );

            if( err != ::cudaSuccess )
            {
                max_num_devices = int{ 0 };
            }
        }
        else if( cuda_dev_index != CudaNodeInfo::ILLEGAL_DEV_INDEX )
        {
            max_num_devices = cuda_dev_index + 1;
        }

        if( ( err == ::cudaSuccess ) &&
            ( cuda_dev_index != CudaNodeInfo::ILLEGAL_DEV_INDEX ) &&
            ( max_num_devices > cuda_dev_index ) )
        {
            std::ostringstream a2str;

            int cuda_driver_version  = int{ 0 };
            int cuda_runtime_version = int{ 0 };

            a2str << "Cuda";

            err = ::cudaDriverGetVersion( &cuda_driver_version );

            if( err == ::cudaSuccess )
            {
                a2str << " :: Driver v" << cuda_driver_version / 1000
                      << "." << ( cuda_driver_version % 1000 ) / 10;
            }

            err = ::cudaRuntimeGetVersion( &cuda_runtime_version );

            if( err == ::cudaSuccess )
            {
                a2str << " :: Runtime v" << cuda_runtime_version / 1000
                      << "." << ( cuda_runtime_version % 1000 ) / 10;
            }

            this->setPlatformName( a2str.str() );

            a2str.str( "" );
            a2str << "compute_capability="
                  << cuda_device_properties.major << "."
                  << cuda_device_properties.minor << "; "
                  << "multiprocessor_count="
                  << this->m_cu_device_properties.multiProcessorCount
                  << ";";

            this->setDescription( a2str.str() );

            if( std::strlen( cuda_device_properties.name ) > size_t{ 0 } )
            {
                a2str.str( "" );
                a2str << cuda_device_properties.name;

                char pci_bus_id[] =
                {
                    '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0',
                    '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0'
                };

                err = ::cudaDeviceGetPCIBusId( pci_bus_id, 16, cuda_dev_index );

                if( err == ::cudaSuccess )
                {
                    a2str << " [" << pci_bus_id << "]";
                    this->m_cu_device_pci_bus_id = &pci_bus_id[ 0 ];
                }

                this->setDeviceName( a2str.str() );
            }
        }
    }

    bool CudaNodeInfo::hasPciBusId() const SIXTRL_NOEXCEPT
    {
        return ( ( !this->m_cu_device_pci_bus_id.empty() ) &&
                 (  this->m_cu_device_pci_bus_id.compare(
                     "0000:00:00.0" ) != 0 ) );
    }

    std::string const& CudaNodeInfo::pciBusId() const SIXTRL_NOEXCEPT
    {
        return this->m_cu_device_pci_bus_id;
    }

    char const* CudaNodeInfo::ptrPciBusIdStr() const SIXTRL_NOEXCEPT
    {
        return this->m_cu_device_pci_bus_id.c_str();
    }

    ::cudaDeviceProp const&
    CudaNodeInfo::cudaDeviceProperties() const SIXTRL_NOEXCEPT
    {
        return this->m_cu_device_properties;
    }

    ::cudaDeviceProp const*
    CudaNodeInfo::ptrCudaDeviceProperties() const SIXTRL_NOEXCEPT
    {
        return &this->m_cu_device_properties;
    }

    bool CudaNodeInfo::hasCudaDeviceIndex() const SIXTRL_NOEXCEPT
    {
        return ( this->m_cu_device_index != CudaNodeInfo::ILLEGAL_DEV_INDEX );
    }

    CudaNodeInfo::cuda_dev_index_t
    CudaNodeInfo::cudaDeviceIndex() const SIXTRL_NOEXCEPT
    {
        return this->m_cu_device_index;
    }

    void CudaNodeInfo::setCudaDeviceIndex(
        CudaNodeInfo::cuda_dev_index_t const index ) SIXTRL_NOEXCEPT
    {
        this->m_cu_device_index = index;
    }

    void CudaNodeInfo::doPrintToOutputStream(
        std::ostream& SIXTRL_RESTRICT_REF output ) const
    {
        SIXTRL_CXX_NAMESPACE::NodeInfoBase::doPrintToOutputStream( output );

        output << "cuda device index     : "
               << this->m_cu_device_index << "\r\n";
    }
}

/* end: sixtracklib/cuda/control/node_info.cpp */
