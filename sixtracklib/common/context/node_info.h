#ifndef SIXTRACKLIB_COMMON_CONTROL_NODE_INFO_H__
#define SIXTRACKLIB_COMMON_CONTROL_NODE_INFO_H__

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/context/node_info.hpp"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stdbool.h>
    #include <stdef.h>
    #include <stdlib.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defiend( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/context/definitions.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */


#if defined( __cplusplus ) && !defined( _GPUCODE ) && !defined( __CUDA_ARCH__ )
extern "C" {
#endif /* C++, Host */

#if !defined( _GPUCODE )

/* ------------------------------------------------------------------------- */

SIXTRL_EXTERN SIXTRL_HOST_FN NS(NodeId) const* NS(NodeInfo_get_ptr_node_id)(
    SIXTR_ARGPPTR_DEC const NS(NodeInfoBase) *const SIXTRL_RESTRICT info );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(node_platform_id_t)
NS(NodeInfo_get_platform_id)(
    SIXTR_ARGPPTR_DEC const NS(NodeInfoBase) *const SIXTRL_RESTRICT info );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(node_device_id_t)
NS(NodeInfo_get_device_id)(
    SIXTR_ARGPPTR_DEC const NS(NodeInfoBase) *const SIXTRL_RESTRICT info );

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(NodeInfo_is_default_node)(
    SIXTR_ARGPPTR_DEC const NS(NodeInfoBase) *const SIXTRL_RESTRICT info );

/* ------------------------------------------------------------------------- */

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_id_t) NS(NodeInfo_get_arch_id)(
    SIXTR_ARGPPTR_DEC const NS(NodeInfoBase) *const SIXTRL_RESTRICT info );

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(NodeInfo_has_arch_string)(
    SIXTR_ARGPPTR_DEC const NS(NodeInfoBase) *const SIXTRL_RESTRICT info );

SIXTRL_EXTERN SIXTRL_HOST_FN char const* NS(NodeInfo_get_arch_string)(
    SIXTR_ARGPPTR_DEC const NS(NodeInfoBase) *const SIXTRL_RESTRICT info );

/* ------------------------------------------------------------------------- */

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(NodeInfo_has_platform_name)(
    SIXTR_ARGPPTR_DEC const NS(NodeInfoBase) *const SIXTRL_RESTRICT info );

SIXTRL_EXTERN SIXTRL_HOST_FN char const* NS(NodeInfo_get_platform_name)(
    SIXTR_ARGPPTR_DEC const NS(NodeInfoBase) *const SIXTRL_RESTRICT info );

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(NodeInfo_has_device_name)(
    SIXTR_ARGPPTR_DEC const NS(NodeInfoBase) *const SIXTRL_RESTRICT info );

SIXTRL_EXTERN SIXTRL_HOST_FN char const* NS(NodeInfo_get_device_name)(
    SIXTR_ARGPPTR_DEC const NS(NodeInfoBase) *const SIXTRL_RESTRICT info );

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(NodeInfo_has_description)(
    SIXTR_ARGPPTR_DEC const NS(NodeInfoBase) *const SIXTRL_RESTRICT info );

SIXTRL_EXTERN SIXTRL_HOST_FN char const* NS(NodeInfo_get_description)(
    SIXTR_ARGPPTR_DEC const NS(NodeInfoBase) *const SIXTRL_RESTRICT info );

/* ------------------------------------------------------------------------- */

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(NodeInfo_reset)(
    SIXTR_ARGPPTR_DEC NS(NodeInfoBase)* SIXTRL_RESTRICT info,
    NS(arch_id_t) const arch_id, const char *const SIXTRL_RESTRICT arch_str,
    NS(platform_id_t) const platform_id, NS(device_id_t) const device_id,
    const char *const SIXTRL_RESTRICT platform_name,
    const char *const SIXTRL_RESTRICT device_name,
    const char *const SIXTRL_RESTRICT description );

/* ------------------------------------------------------------------------- */

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(NodeInfo_print)(
    SIXTR_ARGPPTR_DEC const NS(NodeInfoBase) *const SIXTRL_RESTRICT info,
    FILE* SIXTRL_RESTRICT output );

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(NodeInfo_print_out)(
    SIXTR_ARGPPTR_DEC const NS(NodeInfoBase) *const SIXTRL_RESTRICT info );

/* ------------------------------------------------------------------------- */

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(NodeInfo_mark_as_default)(
    SIXTR_ARGPPTR_DEC NS(NodeInfoBase)* SIXTRL_RESTRICT info,
    bool const is_default );

#endif /* !defined( _GPUCODE ) */

#if defined( __cplusplus ) && !defined( _GPUCODE ) && !defined( __CUDA_ARCH__ )
}
#endif /* C++, Host */

#endif /* SIXTRACKLIB_COMMON_CONTROL_NODE_INFO_H__ */
/* end: sixtracklib/common/context/node_info.h */
