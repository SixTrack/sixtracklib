#ifndef __SIXTRACKLIB__IMPL_NAMESPACE_BEGIN_H__
#define __SIXTRACKLIB__IMPL_NAMESPACE_BEGIN_H__

#if !defined( NS_CONCAT_ )
    #define NS_CONCAT_( A, B ) A##B
#endif /* !defined( NS_CONCAT ) */

#if !defined( NS_CONCAT )
    #define NS_CONCAT( A, B ) NS_CONCAT_( A, B )
#endif /* !defined( NS_CONCAT ) */

#if !defined( __NAMESPACE )
    #define __NAMESPACE 
#endif /* !defined( __NAMESPACE ) */

#if !defined( NS )
    #define NS(name) NS_CONCAT( __NAMESPACE, name )
#endif /* !defined( NS ) */

#endif /* __SIXTRACKLIB__IMPL_NAMESPACE_BEGIN_H__ */

/* end: sixtracklib/_impl/namespace_begin.h */
