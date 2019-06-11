#include "sixtracklib/common/control/controller_base.h"

#if defined( __cplusplus ) && !defined( _GPUCODE ) && !defined( __CUDA_ARCH__ )

#include <cstddef>
#include <cstdlib>
#include <cstdint>

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/control/definitions.h"
#include "sixtracklib/common/control/controller_base.hpp"
#include "sixtracklib/common/control/argument_base.hpp"

namespace st = SIXTRL_CXX_NAMESPACE;

void NS(Controller_delete)( ::NS(ControllerBase)* SIXTRL_RESTRICT ctrl )
{
    if( ctrl != nullptr ) delete ctrl;
    return;
}

void NS(Controller_clear)( ::NS(ControllerBase)* SIXTRL_RESTRICT ctrl )
{
    if( ctrl != nullptr ) ctrl->clear();
    return;
}

::NS(arch_id_t) NS(Controller_get_arch_id)(
    const ::NS(ControllerBase) *const SIXTRL_RESTRICT ctrl )
{
    return ( ctrl != nullptr )
        ? ctrl->archId() : ::NS(ARCHITECTURE_ILLEGAL);
}

bool NS(Controller_has_arch_string)(
    const ::NS(ControllerBase) *const SIXTRL_RESTRICT ctrl )
{
    return ( ( ctrl != nullptr ) && ( ctrl->hasArchStr() ) );
}

char const* NS(Controller_get_arch_string)(
    const ::NS(ControllerBase) *const SIXTRL_RESTRICT ctrl )
{
    return ( ctrl != nullptr ) ? ctrl->ptrArchStr() : nullptr;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

bool NS(Controller_has_config_string)(
    const ::NS(ControllerBase) *const SIXTRL_RESTRICT ctrl )
{
    return ( ctrl != nullptr ) ? ctrl->hasConfigStr() : false;
}

char const* NS(Controller_get_config_string)(
    const ::NS(ControllerBase) *const SIXTRL_RESTRICT ctrl )
{
    return ( ctrl != nullptr ) ? ctrl->ptrConfigStr() : nullptr;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

bool NS(Controller_uses_nodes)(
    const ::NS(ControllerBase) *const SIXTRL_RESTRICT ctrl )
{
    return ( ( ctrl != nullptr ) && ( ctrl->usesNodes() ) );
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

::NS(arch_status_t) NS(Controller_send_detailed)(
    ::NS(ControllerBase)* SIXTRL_RESTRICT ctrl,
    ::NS(ArgumentBase)* SIXTRL_RESTRICT destination,
    void const* SIXTRL_RESTRICT source, NS(arch_size_t) const src_len )
{
    return ( ctrl != nullptr )
        ? ctrl->send( destination, source, src_len )
        : ::NS(ARCH_STATUS_GENERAL_FAILURE);
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

::NS(arch_status_t) NS(Controller_send_buffer)(
    ::NS(ControllerBase)* SIXTRL_RESTRICT ctrl,
    ::NS(ArgumentBase)* SIXTRL_RESTRICT destination,
    ::NS(Buffer) const* SIXTRL_RESTRICT source )
{
    return ( ctrl != nullptr )
        ? ctrl->send( destination, source )
        : ::NS(ARCH_STATUS_GENERAL_FAILURE);
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

::NS(arch_status_t) NS(Controller_receive_detailed)(
    ::NS(ControllerBase)* SIXTRL_RESTRICT ctrl,
    void* SIXTRL_RESTRICT destination,
    ::NS(arch_size_t) const destination_capacity,
    ::NS(ArgumentBase)* SIXTRL_RESTRICT source )
{
    return ( ctrl != nullptr )
        ? ctrl->receive( destination, destination_capacity, source )
        : ::NS(ARCH_STATUS_GENERAL_FAILURE);
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

::NS(arch_status_t) NS(Controller_receive_buffer)(
    ::NS(ControllerBase)* SIXTRL_RESTRICT ctrl,
    ::NS(Buffer)* SIXTRL_RESTRICT destination,
    ::NS(ArgumentBase)* SIXTRL_RESTRICT source )
{
    return ( ctrl != nullptr )
        ? ctrl->receive( destination, source )
        : ::NS(ARCH_STATUS_GENERAL_FAILURE);
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

::NS(arch_status_t) NS(Controller_remap_cobjects_buffer_arg)(
    ::NS(ControllerBase)* SIXTRL_RESTRICT ctrl,
    ::NS(ArgumentBase)* SIXTRL_RESTRICT arg )
{
    return ( ctrl != nullptr )
        ? ctrl->remap( arg ) : ::NS(ARCH_STATUS_GENERAL_FAILURE);
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

bool NS(Controller_is_ready_to_run_kernel)(
    const NS(ControllerBase) *const SIXTRL_RESTRICT ctrl )
{
    return ( ( ctrl != nullptr ) && ( ctrl->readyForRunningKernel() ) );
}

bool NS(Controller_is_ready_to_remap)(
    const ::NS(ControllerBase) *const SIXTRL_RESTRICT ctrl )
{
    return ( ( ctrl != nullptr ) && ( ctrl->readyForRemap() ) );
}

bool NS(Controller_is_ready_to_send)(
    const ::NS(ControllerBase) *const SIXTRL_RESTRICT ctrl )
{
    return ( ( ctrl != nullptr ) && ( ctrl->readyForSend() ) );
}

bool NS(Controller_is_ready_to_receive)(
    const ::NS(ControllerBase) *const SIXTRL_RESTRICT ctrl )
{
    return ( ( ctrl != nullptr ) && ( ctrl->readyForReceive() ) );
}

bool NS(Controller_is_in_debug_mode)(
    const ::NS(ControllerBase) *const SIXTRL_RESTRICT ctrl )
{
    return ( ( ctrl != nullptr ) && ( ctrl->isInDebugMode() ) );
}

::NS(arch_status_t) NS(Controller_enable_debug_mode)(
    ::NS(ControllerBase)* SIXTRL_RESTRICT ctrl )
{
    return ( ctrl != nullptr )
        ? ctrl->enableDebugMode() : st::ARCH_STATUS_GENERAL_FAILURE;
}

::NS(arch_status_t) NS(Controller_disable_debug_mode)(
    ::NS(ControllerBase)* SIXTRL_RESTRICT ctrl )
{
    return ( ctrl != nullptr )
        ? ctrl->disableDebugMode() : st::ARCH_STATUS_GENERAL_FAILURE;
}

/* ========================================================================= */

::NS(arch_size_t) NS(Controller_get_num_of_kernels)(
    const ::NS(ControllerBase) *const SIXTRL_RESTRICT controller )
{
    return ( controller != nullptr )
        ? controller->numKernels() : ::NS(arch_size_t){ 0 };
}

::NS(arch_size_t) NS(Controller_get_kernel_work_items_dim)(
    const ::NS(ControllerBase) *const SIXTRL_RESTRICT controller,
    ::NS(arch_kernel_id_t) const kernel_id )
{
    return ( controller != nullptr )
        ? controller->kernelWorkItemsDim( kernel_id )
        : ::NS(arch_size_t){ 0 };
}

::NS(arch_size_t) NS(Controller_get_kernel_work_groups_dim)(
    const ::NS(ControllerBase) *const  SIXTRL_RESTRICT controller,
    ::NS(arch_kernel_id_t) const kernel_id )
{
    return ( controller != nullptr )
        ? controller->kernelWorkGroupsDim( kernel_id )
        : ::NS(arch_size_t){ 0 };
}

::NS(arch_size_t) NS(Controller_get_num_of_kernel_arguments)(
    const ::NS(ControllerBase) *const SIXTRL_RESTRICT controller,
    ::NS(arch_kernel_id_t) const kernel_id )
{
    return ( controller != nullptr )
        ? controller->kernelNumArguments( kernel_id )
        : ::NS(arch_size_t){ 0 };
}

/* ------------------------------------------------------------------------- */

bool NS(Controller_kernel_has_name)(
    const ::NS(ControllerBase) *const SIXTRL_RESTRICT controller,
    NS(arch_kernel_id_t) const kernel_id )
{
    return ( ( controller != nullptr ) &&
             ( controller->kernelHasName( kernel_id ) ) );
}

char const* NS(Controller_get_kernel_name_string)(
    const ::NS(ControllerBase) *const SIXTRL_RESTRICT controller,
    ::NS(arch_kernel_id_t) const kernel_id )
{
    return ( controller != nullptr ) ?
        controller->ptrKernelNameStr( kernel_id ) : nullptr;
}

/* ------------------------------------------------------------------------- */

bool NS(Controller_has_kernel_id)(
    const ::NS(ControllerBase) *const SIXTRL_RESTRICT controller,
    ::NS(arch_kernel_id_t) const kernel_id )
{
    return ( ( controller != nullptr ) &&
             ( controller->hasKernel( kernel_id ) ) );
}

bool NS(Controller_has_kernel_by_name)(
    const ::NS(ControllerBase) *const SIXTRL_RESTRICT controller,
    char const* SIXTRL_RESTRICT kernel_name )
{
    return ( ( controller != nullptr ) &&
             ( controller->hasKernel( kernel_name ) ) );
}

/* ------------------------------------------------------------------------- */

::NS(KernelConfigBase) const* NS(Controller_get_ptr_const_kernel_config_base)(
    const ::NS(ControllerBase) *const SIXTRL_RESTRICT controller,
    ::NS(arch_kernel_id_t) const kernel_id )
{
    return ( controller != nullptr )
        ? controller->ptrKernelConfigBase( kernel_id ) : nullptr;
}

::NS(KernelConfigBase) const*
NS(Controller_get_ptr_const_kernel_config_base_by_name)(
    const ::NS(ControllerBase) *const SIXTRL_RESTRICT controller,
    char const* SIXTRL_RESTRICT kernel_name )
{
    return ( controller != nullptr )
        ? controller->ptrKernelConfigBase( kernel_name ) : nullptr;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

::NS(KernelConfigBase)* NS(Controller_get_ptr_kernel_config_base)(
    ::NS(ControllerBase)* SIXTRL_RESTRICT controller,
    ::NS(arch_kernel_id_t) const kernel_id )
{
    return ( controller != nullptr )
        ? controller->ptrKernelConfigBase( kernel_id ) : nullptr;
}

::NS(KernelConfigBase)* NS(Controller_get_ptr_kernel_config_base_by_name)(
    ::NS(ControllerBase)* SIXTRL_RESTRICT controller,
    char const* SIXTRL_RESTRICT kernel_name )
{
    return ( controller != nullptr )
        ? controller->ptrKernelConfigBase( kernel_name ) : nullptr;
}

/* ================================================================= */

bool NS(Controller_has_remap_cobject_buffer_kernel)(
    const ::NS(ControllerBase) *const SIXTRL_RESTRICT controller )
{
    return ( ( controller != nullptr ) &&
             ( controller->hasRemapCObjectBufferKernel() ) );
}

::NS(ctrl_kernel_id_t) NS(Controller_get_remap_cobject_buffer_kernel_id)(
    const ::NS(ControllerBase) *const SIXTRL_RESTRICT controller )
{
    return ( controller != nullptr )
        ? controller->remapCObjectBufferKernelId()
        : st::ControllerBase::ILLEGAL_KERNEL_ID;
}

void NS(Controller_set_remap_cobject_buffer_kernel_id)(
    ::NS(ControllerBase)* SIXTRL_RESTRICT controller,
    ::NS(ctrl_kernel_id_t) const kernel_id )
{
    if( controller != nullptr )
    {
        controller->setRemapCObjectBufferKernelId( kernel_id );
    }
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

bool NS(Controller_has_remap_cobject_buffer_debug_kernel)(
    const ::NS(ControllerBase) *const SIXTRL_RESTRICT controller )
{
    return ( ( controller != nullptr ) &&
             ( controller->hasRemapCObjectBufferDebugKernel() ) );
}

::NS(ctrl_kernel_id_t) NS(Controller_get_remap_cobject_buffer_debug_kernel_id)(
    const ::NS(ControllerBase) *const SIXTRL_RESTRICT controller )
{
    return ( controller != nullptr )
        ? controller->remapCObjectBufferDebugKernelId()
        : st::ControllerBase::ILLEGAL_KERNEL_ID;
}

void NS(Controller_set_remap_cobject_buffer_debug_kernel_id)(
    ::NS(ControllerBase)* SIXTRL_RESTRICT controller,
    ::NS(ctrl_kernel_id_t) const kernel_id )
{
    if( controller != nullptr )
    {
        controller->setRemapCObjectBufferDebugKernelId( kernel_id );
    }
}

/* ================================================================= */

#endif /* C++, Host */

/* end: sixtracklib/common/control/controller_base_c99.cpp */
