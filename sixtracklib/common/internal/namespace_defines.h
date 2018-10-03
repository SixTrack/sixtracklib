#ifndef SIXTRACKLIB_COMMON_INTERNAL_NAMESPACE_DEFINES_HEADER_H__
#define SIXTRACKLIB_COMMON_INTERNAL_NAMESPACE_DEFINES_HEADER_H__

#include "sixtracklib/common/generated/namespace.h"

#if !defined( SIXTRL_C99_NAMESPACE )
    #define   SIXTRL_C99_NAMESPACE  st_
#endif /* !defined( SIXTRL_C99_NAMESPACE ) */

#if !defined( NS_CONCAT_ )
    #define NS_CONCAT_( A, B ) A##B
#endif /* !defined( NS_CONCAT ) */

#if !defined( NS_CONCAT )
    #define NS_CONCAT( A, B ) NS_CONCAT_( A, B )
#endif /* !defined( NS_CONCAT ) */

#if !defined( NS )
    #define NS(name) NS_CONCAT( SIXTRL_C99_NAMESPACE, name )
#endif /* !defined( NS ) */

#if !defined( _GPUCODE )
#if !defined( NSVAR )
    #define NSVAR(name, ...) \
        NS_CONCAT( SIXTRL_C99_NAMESPACE, NS_CONCAT( #name, ##__VA_ARGS__ ) )
#endif /* !defined( NSVAR ) */
#endif /* !defined( _GPUCODE ) */

#if !defined( NSEXT )
    #define NSEXT( ns, name ) NS_CONCAT( #ns, #name )
#endif /* !defined( NS_EXT ) */

#if !defined( _GPUCODE )
#if !defined( NSEXTVAR )
    #define NSEXTVAR( ns, name, ... ) \
        NS_CONCAT( NS_CONCAT( #ns, "_" ), NS_CONCAT( #name, ##__VA_ARGS__ ) )
#endif /* !defined( NSEXTVAR ) */
#endif /* !defined( _GPUCODE ) */

#endif /* SIXTRACKLIB_COMMON_INTERNAL_NAMESPACE_DEFINES_HEADER_H__ */

/* end: sixtracklib/common/internal/namespace_defines.h */
