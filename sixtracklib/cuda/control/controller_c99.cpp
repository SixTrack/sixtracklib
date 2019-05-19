#include "sixtracklib/cuda/controller.h"

#if defined( __cplusplus ) && !defined( _GPUCODE ) && !defined( __CUDA_ARCH__ )

#include <cstddef>
#include <cstdlib>

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/cuda/definitions.h"
#include "sixtracklib/cuda/controller.hpp"
#include "sixtracklib/cuda/argument.hpp"

NS(CudaController)* NS(CudaController_create)( void )
{
    return new SIXTRL_CXX_NAMESPACE::CudaController( "" );
}

bool NS(CudaController_select_node_by_cuda_device_index)(
    NS(CudaController)* SIXTRL_RESTRICT ctrl, int cuda_device_index )
{
    return ( ( ctrl != nullptr ) &&
             ( ctrl->selectNodeByCudaIndex( cuda_device_index ) ) );
}

bool NS(CudaController_select_node_by_cuda_pci_bus_id)(
    NS(CudaController)* SIXTRL_RESTRICT ctrl,
    char const* SIXTRL_RESTRICT cuda_pci_bus_id )
{
    return ( ( ctrl != nullptr ) &&
             ( ctrl->selectNodeByPciBusId( cuda_pci_bus_id ) ) );
}

void NS(CudaController_delete)( NS(CudaController)* SIXTRL_RESTRICT ctrl )
{
    delete ctrl;
    return;
}

/* ------------------------------------------------------------------------- */

::NS(arch_status_t) NS(CudaController_remap_managed_cobject_buffer)(
    ::NS(CudaController)* SIXTRL_RESTRICT ctrl,
    ::NS(cuda_arg_buffer_t) SIXTRL_RESTRICT managed_buffer_begin,
    ::NS(arch_size_t) const slot_size )
{
    return ( ctrl != nullptr )
        ? ctrl->remap( managed_buffer_begin, slot_size )
        : ::NS(ARCH_STATUS_GENERAL_FAILURE);
}

bool NS(CudaController_is_managed_cobject_buffer_remapped)(
    ::NS(CudaController)* SIXTRL_RESTRICT ctrl,
    ::NS(cuda_arg_buffer_t) SIXTRL_RESTRICT managed_buffer_begin,
    ::NS(arch_size_t) const slot_size )
{
    return ( ( ctrl != nullptr ) &&
             ( ctrl->isRemapped( managed_buffer_begin, slot_size ) ) );
}

::NS(arch_status_t) NS(CudaController_send_memory)(
    ::NS(CudaController)* SIXTRL_RESTRICT ctrl,
    ::NS(cuda_arg_buffer_t) SIXTRL_RESTRICT destination,
    void const* SIXTRL_RESTRICT source, ::NS(arch_size_t) const source_length )
{
    return ( ctrl != nullptr )
        ? ctrl->sendMemory( destination, source, source_length )
        : ::NS(ARCH_STATUS_GENERAL_FAILURE);
}

::NS(arch_status_t) NS(CudaController_receive_memory)(
    ::NS(CudaController)* SIXTRL_RESTRICT ctrl,
    void* SIXTRL_RESTRICT destination,
    ::NS(cuda_const_arg_buffer_t) SIXTRL_RESTRICT source,
    ::NS(arch_size_t) const source_length )
{
    return ( ctrl != nullptr )
        ? ctrl->receiveMemory( destination, source, source_length )
        : ::NS(ARCH_STATUS_GENERAL_FAILURE);
}

#endif /* C++, Host */

/* end: sixtracklib/cuda/internal/controller_c99.cpp */