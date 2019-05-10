#if !defined( SIXTRL_NO_INCLUDE )
    #include "sixtracklib/cuda/wrappers/controller_wrappers.h"
#endif /* !defined( SIXTRL_NO_INCLUDE ) */

#include !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <cuda_runtime_api.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDE )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/buffer/buffer_type.h"
    #include "sixtracklib/common/control/definitions.h"

    #include "sixtracklib/cuda/definitions.h"
    #include "sixtracklib/cuda/argument.h"
    #include "sixtracklib/cuda/control/kernel_config.h"
#endif /* !defined( SIXTRL_NO_INCLUDE ) */

NS(ctrl_status_t) NS(CudaController_remap_cobjects_buffer_wrapper)(
    const NS(CudaKernelConfig) *const SIXTRL_RESTRICT kernel_config,
    NS(CudaArgument)* SIXTRL_RESTRICT buffer_arg,
    NS(buffer_size_t) const slot_size,
    NS(CudaArgument)* SIXTRL_RESTRICT debug_flag_arg )
{
    typedef NS(ctrl_status_t) status_t;

    status_t status = NS(CONTROLLER_STATUS_GENERAL_FAILURE);


    return status;
}

/* end: sixtracklib/cuda/wrappers/controller_wrappers.cu */