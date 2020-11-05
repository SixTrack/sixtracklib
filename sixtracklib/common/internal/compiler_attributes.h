#ifndef SIXTRACKLIB_COMMON_INTERNAL_COMPILER_ATTRIBUTES_H__
#define SIXTRACKLIB_COMMON_INTERNAL_COMPILER_ATTRIBUTES_H__

#if !defined SIXTRL_UNUSED
    #if defined( __cplusplus )
        #define SIXTRL_UNUSED( arg ) /* can be omitted -> nothing to do here! */
    #elif ( ( defined( __GNUC__ ) && ( __GNUC__ >= 3 ) ) ) || \
          ( defined( __clang__ ) )
        #define SIXTRL_UNUSED( arg ) arg __attribute__((unused))
    #else
        #define SIXTRL_UNUSED( arg ) arg
    #endif /* C++ / compiler */
#endif /* !defined( SIXTRL_UNUSED ) */

#endif /* SIXTRACKLIB_COMMON_INTERNAL_COMPILER_ATTRIBUTES_H__ */
