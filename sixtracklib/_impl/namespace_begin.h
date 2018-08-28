#ifndef SIXTRACKLIB__IMPL_NAMESPACE_BEGIN_H__
#define SIXTRACKLIB__IMPL_NAMESPACE_BEGIN_H__

#if !defined( SIXTRL_NAMESPACE )
    #define SIXTRL_NAMESPACE

    #if !defined( __NAMESPACE )
        #define __NAMESPACE
    #endif /* __NAMESPACE */

#else /* defined( SIXTRL_NAMESPACE ) */

    #if !defined( __NAMESPACE )
        #define __NAMESPACE #SIXTRL_NAMESPACE ## "_"
    #endif /* !defined( __NAMESPACE ) */

#endif /* defined( SIXTRL_NAMESPACE ) */

#if !defined( NS_CONCAT_ )
    #define NS_CONCAT_( A, B ) A##B
#endif /* !defined( NS_CONCAT ) */

#if !defined( NS_CONCAT )
    #define NS_CONCAT( A, B ) NS_CONCAT_( A, B )
#endif /* !defined( NS_CONCAT ) */

#if !defined( NS )
    #define NS(name) NS_CONCAT( __NAMESPACE, name )
#endif /* !defined( NS ) */

#if !defined( NSVAR )
    #define NSVAR(name, ...) \
        NS_CONCAT( __NAMESPACE, NS_CONCAT( #name, ##__VA_ARGS__ ) )
#endif /* !defined( NSVAR ) */

#if !defined( NSEXT )
    #define NSEXT( ns, name ) NS_CONCAT( #ns, #name )
#endif /* !defined( NS_EXT ) */

#if !defined( NSEXTVAR )
    #define NSEXTVAR( ns, name, ... ) \
        NS_CONCAT( NS_CONCAT( #ns, "_" ), NS_CONCAT( #name, ##__VA_ARGS__ ) )
#endif /* !defined( NSEXTVAR ) */

#endif /* SIXTRACKLIB__IMPL_NAMESPACE_BEGIN_H__ */

/* end: sixtracklib/_impl/namespace_begin.h */
