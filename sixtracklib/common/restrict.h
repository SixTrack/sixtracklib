#ifndef SIXTRACKLIB_SIXTRACKLIB_COMMON_RESTRICT_H__
#define SIXTRACKLIB_SIXTRACKLIB_COMMON_RESTRICT_H__

#if defined( SIXTRL_RESTRICT )
    #undef  SIXTRL_RESTRICT
#endif /* SIXTRL_RESTRICT */

#if defined( SIXTRL_RESTRICT_REF )
    #undef  SIXTRL_RESTRICT_REF
#endif /* SIXTRL_RESTRICT_REF */

#if defined( SIXTRL_ENABLE_RESTRICT__ )
    
    #ifdef __cplusplus

        #if ( ( defined( __clang__ ) ) || ( ( ( defined( __GNUC__ ) ) && ( __GNUC__ >= 3 ) ) ) )
        
            #define SIXTRL_RESTRICT     __restrict__
            #define SIXTRL_RESTRICT_REF __restrict__
        
        #elif ( defined( _MSC_VER ) && _MSC_VER >= 1600 )
        
            #define SIXTRL_RESTRICT __restrict
        
        #endif /* ( defined( _MSC_VER ) && _MSC_VER >= 1600 ) */
        
    #else /* __cplusplus */
        
        #if ( ( defined( __clang__ ) ) || ( ( ( defined( __GNUC__ ) ) && ( __GNUC__ >= 3 ) ) ) )
        
            #if ( __STDC_VERSION__ >= 199901L )
                #define SIXTRL_RESTRICT restrict /* TODO: Check if clang supports this! */
            #else
                #define SIXTRL_RESTRICT __restrict__
            #endif /* C99 support */
        
        #endif /* gcc/mingw or clang */
        
    #endif /*defined( __cplusplus ) */
    
#endif /* defined( SIXTRL_ENABLE_RESTRICT__ ) */



#ifndef SIXTRL_RESTRICT 
    #define SIXTRL_RESTRICT
#endif /* SIXTRL_RESTRICT */

#ifndef SIXTRL_RESTRICT_REF
    #define SIXTRL_RESTRICT_REF
#endif /* SIXTRL_RESTRICT_REF */

#endif /* SIXTRACKLIB_SIXTRACKLIB_COMMON_RESTRICT_H__ */

/* end: sixtracklib/sixtracklib/compat/restrict.h */
