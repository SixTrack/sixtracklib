#ifndef SIXTRACKLIB_COMMON_CONTROL_ARCH_BASE_H__
#define SIXTRACKLIB_COMMON_CONTROL_ARCH_BASE_H__

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/context/definitions.h"
    #include "sixtracklib/common/context/arch_base.hpp"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus ) && !defined( _GPUCODE ) && !defined( __CUDA_ARCH__ )
extern "C" {
#endif /* C++, Host */

#if !defined( _GPUCODE )

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(ArchBase_has_config_str)(
    SIXTRL_ARGPTR_DEC const NS(ArchBase) *const SIXTRL_RESTRICT arch_base );

SIXTRL_EXTERN SIXTRL_HOST_FN char const* NS(ArchBase_get_config_string)(
    SIXTRL_ARGPTR_DEC const NS(ArchBase) *const SIXTRL_RESTRICT arch_base );

#endif /* !defined( _GPUCODE ) */

#if defined( __cplusplus ) && !defined( _GPUCODE ) && !defined( __CUDA_ARCH__ )
}
#endif /* C++, Host */

#endif /* SIXTRACKLIB_COMMON_CONTROL_ARCH_BASE_H__ */
/* end: sixtracklib/common/context/arch_base.h */
