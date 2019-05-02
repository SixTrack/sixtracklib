#ifndef SIXTRACKLIB_COMMON_CONTROL_ARCH_INFO_H__
#define SIXTRACKLIB_COMMON_CONTROL_ARCH_INFO_H__

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/control/arch_info.hpp"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stdbool.h>
    #include <stddef.h>
    #include <stdlib.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/control/definitions.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus ) && !defined( _GPUCODE ) && !defined( __CUDA_ARCH__ )
extern "C" {
#endif /* C++, Host */

#if !defined( _GPUCODE )

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_id_t) NS(ArchInfo_get_arch_id)(
    SIXTRL_ARGPTR_DEC const NS(ArchInfo) *const SIXTRL_RESTRICT arch_info );

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(ArchInfo_has_arch_str)(
    SIXTRL_ARGPTR_DEC const NS(ArchInfo) *const SIXTRL_RESTRICT arch_info );

SIXTRL_EXTERN SIXTRL_HOST_FN char const* NS(ArchInfo_get_arch_string)(
    SIXTRL_ARGPTR_DEC const NS(ArchInfo) *const SIXTRL_RESTRICT arch_info );

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(ArchInfo_is_compatible_with)(
    SIXTRL_ARGPTR_DEC const NS(ArchInfo) *const SIXTRL_RESTRICT arch_info,
    SIXTRL_ARGPTR_DEC const NS(ArchInfo) *const SIXTRL_RESTRICT other );

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(ArchInfo_is_compatible_with_arch_id)(
    SIXTRL_ARGPTR_DEC const NS(ArchInfo) *const SIXTRL_RESTRICT arch_info,
    NS(arch_id_t) const other_arch_id );

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(ArchInfo_is_identical_to)(
    SIXTRL_ARGPTR_DEC const NS(ArchInfo) *const SIXTRL_RESTRICT arch_info,
    SIXTRL_ARGPTR_DEC const NS(ArchInfo) *const SIXTRL_RESTRICT other );

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(ArchInfo_is_identical_to_arch_id)(
    SIXTRL_ARGPTR_DEC const NS(ArchInfo) *const SIXTRL_RESTRICT arch_info,
    NS(arch_id_t) const other_arch_id );

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(ArchInfo_reset)(
    SIXTRL_ARGPTR_DEC NS(ArchInfo)* SIXTRL_RESTRICT arch_info,
    NS(arch_id_t) const arch_id, const char *const SIXTRL_RESTRICT arch_str );

#endif /* !defined( _GPUCODE ) */

#if defined( __cplusplus ) && !defined( _GPUCODE ) && !defined( __CUDA_ARCH__ )
}
#endif /* C++, Host */

#endif /* SIXTRACKLIB_COMMON_CONTROL_ARCH_INFO_H__ */

/* end: sixtracklib/common/control/arch_info.h */
