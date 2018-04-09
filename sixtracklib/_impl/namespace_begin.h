#pragma once

#if !defined( NS_CONCAT_ )
    #define NS_CONCAT_( A, B ) A##B
#endif /* !defined( NS_CONCAT ) */

#if !defined( NS_CONCAT )
    #define NS_CONCAT( A, B ) NS_CONCAT_( A, B )
#endif /* !defined( NS_CONCAT ) */

#if !defined( NAMESPACE )
    #define NAMESPACE 
#endif /* !defined( NAMESPACE ) */

#if !defined( NS )
    #define NS(name) NS_CONCAT( NAMESPACE, name )
#endif /* !defined( NS ) */

/* end: sixtracklib/_impl/namespace_begin.h */
