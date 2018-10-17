#ifndef SIXTRACKLIB_COMMON_INTERNAL_BUFFER_MAIN_DEFINES_H__
#define SIXTRACKLIB_COMMON_INTERNAL_BUFFER_MAIN_DEFINES_H__

#if !defined( SIXTRL_BUFFER_ARGPTR_DEC )
    #define   SIXTRL_BUFFER_ARGPTR_DEC_UNDEF
    #if defined( SIXTRL_ARGPTR_DEC )
        #define  SIXTRL_BUFFER_ARGPTR_DEC SIXTRL_ARGPTR_DEC
    #else
        #define SIXTRL_BUFFER_ARGPTR_DEC
    #endif /* defined( SIXTRL_ARGPTR_DEC ) */
#endif /* !defined( SIXTRL_BUFFER_GARBAGE_ARGPTR_DEC ) */

#if !defined( SIXTRL_BUFFER_DATAPTR_DEC )
    #define SIXTRL_BUFFER_DATAPTR_DEC_UNDEF
    #if defined( SIXTRL_DATAPTR_DEC )
        #define  SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_DATAPTR_DEC
    #else /* !defined( SIXTRL_DATAPTR_DEC ) */
        #define  SIXTRL_BUFFER_DATAPTR_DEC
    #endif /* defined( SIXTRL_DATAPTR_DEC ) */
#endif /* !defined( SIXTRL_BUFFER_DATAPTR_DEC ) */

#endif /* SIXTRACKLIB_COMMON_INTERNAL_BUFFER_MAIN_DEFINES_H__ */

/* end: sixtracklib/common/internal/buffer_main_defines.h */
