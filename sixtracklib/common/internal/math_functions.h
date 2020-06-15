#ifndef SIXTRACKLIB_COMMON_INTERNAL_MATH_FUNCTIONS_H__
#define SIXTRACKLIB_COMMON_INTERNAL_MATH_FUNCTIONS_H__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #if defined( __cplusplus )
        #include <type_traits>
        #include <cmath>
    #endif /* __cplusplus */
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/internal/type_store_traits.hpp"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus )
namespace SIXTRL_CXX_NAMESPACE
{
    template< typename T >
    SIXTRL_STATIC SIXTRL_INLINE SIXTRL_FN typename std::enable_if<
        SIXTRL_CXX_NAMESPACE::Type_is_scalar< T >(),
        typename TypeMethodParamTraits< T >::value_type >::type
    sign( typename TypeMethodParamTraits< T >::const_argument_type
        arg ) SIXTRL_NOEXCEPT
    {
        typedef typename TypeMethodParamTraits< T >::value_type value_t;
        return ( arg >= value_t{ 0 } ) ? value_t{ 1 } : value_t{ -1 };
    }

    template< typename T >
    SIXTRL_STATIC SIXTRL_INLINE SIXTRL_FN typename std::enable_if<
        !SIXTRL_CXX_NAMESPACE::Type_is_scalar< T >(),
        typename TypeMethodParamTraits< T >::value_type >::type
    sign( typename TypeMethodParamTraits< T >::const_argument_type
        arg ) SIXTRL_NOEXCEPT
    {
        typedef typename TypeMethodParamTraits< T >::value_type value_t;

        return static_cast< value_t >( value_t{ 0 } < arg ) -
               static_cast< value_t >( arg < value_t{ 0 } );
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    template< typename T >
    SIXTRL_STATIC SIXTRL_INLINE SIXTRL_FN typename std::enable_if<
        SIXTRL_CXX_NAMESPACE::Type_is_scalar< T >(),
        typename TypeMethodParamTraits< T >::value_type >::type
    abs( typename TypeMethodParamTraits< T >::const_argument_type
        arg ) SIXTRL_NOEXCEPT
    {
        typedef typename TypeMethodParamTraits< T >::value_type value_t;
        return ( arg >= value_t{ 0 } ) ? arg : -arg;
    }


    template< typename T >
    SIXTRL_STATIC SIXTRL_INLINE SIXTRL_FN typename std::enable_if<
        !SIXTRL_CXX_NAMESPACE::Type_is_scalar< T >(),
        typename TypeMethodParamTraits< T >::value_type >::type
    abs( typename TypeMethodParamTraits< T >::const_argument_type
        arg ) SIXTRL_NOEXCEPT
    {
        return arg * SIXTRL_CXX_NAMESPACE::sign< T >( arg );
    }
}
#endif /* C++ */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/internal/type_comparison_helpers.hpp"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus )
namespace SIXTRL_CXX_NAMESPACE
{
    template< typename T >
    SIXTRL_STATIC SIXTRL_INLINE SIXTRL_FN
    typename TypeMethodParamTraits< T >::value_type sin( typename
        TypeMethodParamTraits< T >::const_argument_type arg ) SIXTRL_NOEXCEPT
    {
        using std::sin;
        return sin( arg );
    }

    template< typename T >
    SIXTRL_STATIC SIXTRL_INLINE SIXTRL_FN
    typename TypeMethodParamTraits< T >::value_type asin( typename
        TypeMethodParamTraits< T >::const_argument_type arg ) SIXTRL_NOEXCEPT
    {
        namespace st = SIXTRL_CXX_NAMESPACE;
        using std::asin;

        SIXTRL_ASSERT( st::Type_comp_all_more_or_equal< typename
            TypeMethodParamTraits< T >::value_type >( arg,
                typename TypeMethodParamTraits< T >::value_type{ -1 } ) );
        SIXTRL_ASSERT( st::Type_comp_all_less_or_equal< typename
            TypeMethodParamTraits< T >::value_type >( arg,
                typename TypeMethodParamTraits< T >::value_type{ +1 } ) );

        asin( arg );
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    template< typename T >
    SIXTRL_STATIC SIXTRL_INLINE SIXTRL_FN
    typename TypeMethodParamTraits< T >::value_type cos( typename
        TypeMethodParamTraits< T >::const_argument_type arg ) SIXTRL_NOEXCEPT
    {
        using std::cos;
        return cos( arg );
    }

    template< typename T >
    SIXTRL_STATIC SIXTRL_INLINE SIXTRL_FN
    typename TypeMethodParamTraits< T >::value_type acos( typename
        TypeMethodParamTraits< T >::const_argument_type arg ) SIXTRL_NOEXCEPT
    {
        using std::acos;

        SIXTRL_ASSERT( SIXTRL_CXX_NAMESPACE::Type_comp_all_more_or_equal<
            typename TypeMethodParamTraits< T >::value_type >( arg,
                typename TypeMethodParamTraits< T >::value_type{ -1 } ) );
        SIXTRL_ASSERT( SIXTRL_CXX_NAMESPACE::Type_comp_all_less_or_equal<
            typename TypeMethodParamTraits< T >::value_type >( arg,
                typename TypeMethodParamTraits< T >::value_type{ +1 } ) );

        return acos( arg );
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    template< typename T >
    SIXTRL_STATIC SIXTRL_INLINE SIXTRL_FN
    typename TypeMethodParamTraits< T >::value_type tan( typename
        TypeMethodParamTraits< T >::const_argument_type arg ) SIXTRL_NOEXCEPT
    {
        using std::tan;
        return tan( arg );
    }

    template< typename T >
    SIXTRL_STATIC SIXTRL_INLINE SIXTRL_FN
    typename TypeMethodParamTraits< T >::value_type atan2(
        typename TypeMethodParamTraits< T >::const_argument_type x,
        typename TypeMethodParamTraits< T >::const_argument_type y
    ) SIXTRL_NOEXCEPT
    {
        using std::atan2;
        return atan2( x, y );
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    template< typename T >
    SIXTRL_STATIC SIXTRL_INLINE SIXTRL_FN
    typename TypeMethodParamTraits< T >::value_type sqrt( typename
        TypeMethodParamTraits< T >::const_argument_type arg ) SIXTRL_NOEXCEPT
    {
        using std::sqrt;
        SIXTRL_ASSERT( SIXTRL_CXX_NAMESPACE::Type_comp_all_more_or_equal<
            typename TypeMethodParamTraits< T >::value_type >( arg,
                typename TypeMethodParamTraits< T >::value_type{ 0 } ) );

        return sqrt( arg );
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    template< typename T >
    SIXTRL_STATIC SIXTRL_INLINE SIXTRL_FN
    typename TypeMethodParamTraits< T >::value_type exp( typename
        TypeMethodParamTraits< T >::const_argument_type arg ) SIXTRL_NOEXCEPT
    {
        using std::exp;
        return exp( arg );
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    template< typename T >
    SIXTRL_STATIC SIXTRL_INLINE SIXTRL_FN
    typename TypeMethodParamTraits< T >::value_type log( typename
        TypeMethodParamTraits< T >::const_argument_type arg ) SIXTRL_NOEXCEPT
    {
        using std::log;
        SIXTRL_ASSERT( SIXTRL_CXX_NAMESPACE::Type_comp_all_more< typename
            TypeMethodParamTraits< T >::value_type >( arg, typename
                TypeMethodParamTraits< T >::value_type{ 0 } ) );

        return log( arg );
    }
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

template< typename T >
SIXTRL_STATIC SIXTRL_INLINE SIXTRL_FN
typename SIXTRL_CXX_NAMESPACE::TypeMethodParamTraits< T >::value_type
NS(sin)( typename SIXTRL_CXX_NAMESPACE::TypeMethodParamTraits<
    T >::const_argument_type arg ) SIXTRL_NOEXCEPT
{
    using std::sin;
    return sin( arg );
}

template< typename T >
SIXTRL_STATIC SIXTRL_INLINE SIXTRL_FN
typename SIXTRL_CXX_NAMESPACE::TypeMethodParamTraits< T >::value_type
NS(asin)( typename SIXTRL_CXX_NAMESPACE::TypeMethodParamTraits<
    T >::const_argument_type arg ) SIXTRL_NOEXCEPT
{
    using std::asin;
    SIXTRL_ASSERT( SIXTRL_CXX_NAMESPACE::Type_comp_all_more_or_equal< typename
        SIXTRL_CXX_NAMESPACE::TypeMethodParamTraits< T >::value_type >( arg,
            typename SIXTRL_CXX_NAMESPACE::TypeMethodParamTraits<
                T >::value_type{ -1 } ) );
    SIXTRL_ASSERT( SIXTRL_CXX_NAMESPACE::Type_comp_all_more_or_equal< typename
        SIXTRL_CXX_NAMESPACE::TypeMethodParamTraits< T >::value_type >( arg,
            typename SIXTRL_CXX_NAMESPACE::TypeMethodParamTraits<
                T >::value_type{ +1 } ) );

    return asin( arg );
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

template< typename T >
SIXTRL_STATIC SIXTRL_INLINE SIXTRL_FN
typename SIXTRL_CXX_NAMESPACE::TypeMethodParamTraits< T >::value_type
NS(cos)( typename SIXTRL_CXX_NAMESPACE::TypeMethodParamTraits<
    T >::const_argument_type arg ) SIXTRL_NOEXCEPT
{
    using std::cos;
    return cos( arg );
}

template< typename T >
SIXTRL_STATIC SIXTRL_INLINE SIXTRL_FN
typename SIXTRL_CXX_NAMESPACE::TypeMethodParamTraits< T >::value_type
NS(acos)( typename SIXTRL_CXX_NAMESPACE::TypeMethodParamTraits<
    T >::const_argument_type arg ) SIXTRL_NOEXCEPT
{
    using std::acos;
    SIXTRL_ASSERT( SIXTRL_CXX_NAMESPACE::Type_comp_all_more_or_equal< typename
        SIXTRL_CXX_NAMESPACE::TypeMethodParamTraits< T >::value_type >( arg,
            typename SIXTRL_CXX_NAMESPACE::TypeMethodParamTraits<
                T >::value_type{ -1 } ) );
    SIXTRL_ASSERT( SIXTRL_CXX_NAMESPACE::Type_comp_all_more_or_equal< typename
        SIXTRL_CXX_NAMESPACE::TypeMethodParamTraits< T >::value_type >( arg,
            typename SIXTRL_CXX_NAMESPACE::TypeMethodParamTraits<
                T >::value_type{ +1 } ) );

    return acos( arg );
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

template< typename T >
SIXTRL_STATIC SIXTRL_INLINE SIXTRL_FN
typename SIXTRL_CXX_NAMESPACE::TypeMethodParamTraits< T >::value_type
NS(tan)( typename SIXTRL_CXX_NAMESPACE::TypeMethodParamTraits<
    T >::const_argument_type arg ) SIXTRL_NOEXCEPT
{
    using std::tan;
    return tan( arg );
}


template< typename T >
SIXTRL_STATIC SIXTRL_INLINE SIXTRL_FN
typename SIXTRL_CXX_NAMESPACE::TypeMethodParamTraits< T >::value_type
NS(atan2)( typename SIXTRL_CXX_NAMESPACE::TypeMethodParamTraits<
                T >::const_argument_type x,
           typename SIXTRL_CXX_NAMESPACE::TypeMethodParamTraits<
                T >::const_argument_type y ) SIXTRL_NOEXCEPT
{
    using std::atan2;
    return atan2( x, y );
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

template< typename T >
SIXTRL_STATIC SIXTRL_INLINE SIXTRL_FN
typename SIXTRL_CXX_NAMESPACE::TypeMethodParamTraits< T >::value_type
NS(sqrt)( typename SIXTRL_CXX_NAMESPACE::TypeMethodParamTraits<
    T >::const_argument_type arg ) SIXTRL_NOEXCEPT
{
    using std::sqrt;
    SIXTRL_ASSERT( SIXTRL_CXX_NAMESPACE::Type_comp_all_more_or_equal< typename
        SIXTRL_CXX_NAMESPACE::TypeMethodParamTraits< T >::value_type >( arg,
            typename SIXTRL_CXX_NAMESPACE::TypeMethodParamTraits<
                T >::value_type{ 0 } ) );

    return sqrt( arg );
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

template< typename T >
SIXTRL_STATIC SIXTRL_INLINE SIXTRL_FN
typename SIXTRL_CXX_NAMESPACE::TypeMethodParamTraits< T >::value_type
NS(exp)( typename SIXTRL_CXX_NAMESPACE::TypeMethodParamTraits<
    T >::const_argument_type arg ) SIXTRL_NOEXCEPT
{
    using std::exp;
    return exp( arg );
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

template< typename T >
SIXTRL_STATIC SIXTRL_INLINE SIXTRL_FN
typename SIXTRL_CXX_NAMESPACE::TypeMethodParamTraits< T >::value_type
NS(log)( typename SIXTRL_CXX_NAMESPACE::TypeMethodParamTraits<
    T >::const_argument_type arg ) SIXTRL_NOEXCEPT
{
    using std::log;
    SIXTRL_ASSERT( SIXTRL_CXX_NAMESPACE::Type_comp_all_more< typename
        SIXTRL_CXX_NAMESPACE::TypeMethodParamTraits< T >::value_type >( arg,
            typename SIXTRL_CXX_NAMESPACE::TypeMethodParamTraits<
                T >::value_type{ 0 } ) );

    return log( arg );
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

template< typename T >
SIXTRL_STATIC SIXTRL_INLINE SIXTRL_FN typename std::enable_if<
    SIXTRL_CXX_NAMESPACE::Type_is_scalar< T >(),
    typename SIXTRL_CXX_NAMESPACE::TypeMethodParamTraits<
        T >::value_type >::type
NS(sign)( typename SIXTRL_CXX_NAMESPACE::TypeMethodParamTraits<
    T >::const_argument_type arg ) SIXTRL_NOEXCEPT
{
    namespace st = SIXTRL_CXX_NAMESPACE;
    typedef typename st::TypeMethodParamTraits< T >::value_type value_t;
    return ( arg >= value_t{ 0 } ) ? value_t{  1  } : value_t{ -1  };
}

template< typename T >
SIXTRL_STATIC SIXTRL_INLINE SIXTRL_FN typename std::enable_if<
    !SIXTRL_CXX_NAMESPACE::Type_is_scalar< T >(),
    typename SIXTRL_CXX_NAMESPACE::TypeMethodParamTraits<
        T >::value_type >::type
NS(sign)( typename SIXTRL_CXX_NAMESPACE::TypeMethodParamTraits<
    T >::const_argument_type arg ) SIXTRL_NOEXCEPT
{
    namespace st = SIXTRL_CXX_NAMESPACE;
    typedef typename st::TypeMethodParamTraits< T >::value_type value_t;
    return static_cast< value_t >( value_t{ 0 } < arg ) -
           static_cast< value_t >( arg < value_t{ 0 } );
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

template< typename T >
SIXTRL_STATIC SIXTRL_INLINE SIXTRL_FN typename std::enable_if<
    SIXTRL_CXX_NAMESPACE::Type_is_scalar< T >(),
    typename SIXTRL_CXX_NAMESPACE::TypeMethodParamTraits<
        T >::value_type >::type
NS(abs)( typename SIXTRL_CXX_NAMESPACE::TypeMethodParamTraits<
    T >::const_argument_type arg ) SIXTRL_NOEXCEPT
{
    namespace st = SIXTRL_CXX_NAMESPACE;
    typedef typename st::TypeMethodParamTraits< T >::value_type value_t;
    return ( arg >= value_t{ 0 } ) ? arg : -arg;
}

template< typename T >
SIXTRL_STATIC SIXTRL_INLINE SIXTRL_FN typename std::enable_if<
    !SIXTRL_CXX_NAMESPACE::Type_is_scalar< T >(),
    typename SIXTRL_CXX_NAMESPACE::TypeMethodParamTraits<
        T >::value_type >::type
NS(abs)( typename SIXTRL_CXX_NAMESPACE::TypeMethodParamTraits<
    T >::const_argument_type arg ) SIXTRL_NOEXCEPT
{
    return arg * SIXTRL_CXX_NAMESPACE::sign< T >( arg );
}

#endif /* defined( __cplusplus ) */

#if defined( __cplusplus ) && !defined( _GPUCODE )
extern "C" {
#endif /* defined( __cplusplus ) && !defined( _GPUCODE ) */

SIXTRL_STATIC SIXTRL_FN SIXTRL_REAL_T
NS(sin)( SIXTRL_REAL_T const arg ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_REAL_T
NS(asin)( SIXTRL_REAL_T const arg ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_REAL_T
NS(cos)( SIXTRL_REAL_T const arg ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_REAL_T
NS(acos)( SIXTRL_REAL_T const arg ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_REAL_T
NS(tan)( SIXTRL_REAL_T const arg ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_REAL_T
NS(atan2)( SIXTRL_REAL_T const x, SIXTRL_REAL_T const y ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_REAL_T
NS(sqrt)( SIXTRL_REAL_T const arg ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_REAL_T
NS(exp)( SIXTRL_REAL_T const arg ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_REAL_T
NS(log)( SIXTRL_REAL_T const arg ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_REAL_T
NS(sign)( SIXTRL_REAL_T const arg ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_REAL_T
NS(abs)( SIXTRL_REAL_T const arg ) SIXTRL_NOEXCEPT;

#if defined( __cplusplus ) && !defined( _GPUCODE )
}
#endif /* defined( __cplusplus ) && !defined( _GPUCODE ) */

/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */
/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #if !defined( __cplusplus )
        #include <math.h>
    #endif /* __cplusplus */
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if defined( __cplusplus ) && !defined( _GPUCODE )
extern "C" {
#endif /* defined( __cplusplus ) && !defined( _GPUCODE ) */

SIXTRL_INLINE SIXTRL_REAL_T NS(sin)( SIXTRL_REAL_T const arg ) SIXTRL_NOEXCEPT
{
    #if defined( __cplusplus )
    using std::sin;
    #endif /* defined( __cplusplus ) */

    return sin( arg );
}

SIXTRL_INLINE SIXTRL_REAL_T NS(asin)( SIXTRL_REAL_T const arg ) SIXTRL_NOEXCEPT
{
    #if defined( __cplusplus )
    using std::asin;
    #endif /* defined( __cplusplus ) */

    SIXTRL_ASSERT( ( arg >= ( SIXTRL_REAL_T )-1 ) &&
                   ( arg <= ( SIXTRL_REAL_T )+1 ) );
    return asin( arg );
}

SIXTRL_INLINE SIXTRL_REAL_T NS(cos)( SIXTRL_REAL_T const arg ) SIXTRL_NOEXCEPT
{
    #if defined( __cplusplus )
    using std::cos;
    #endif /* defined( __cplusplus ) */

    return cos( arg );
}

SIXTRL_INLINE SIXTRL_REAL_T NS(acos)( SIXTRL_REAL_T const arg ) SIXTRL_NOEXCEPT
{
    #if defined( __cplusplus )
    using std::acos;
    #endif /* defined( __cplusplus ) */

    SIXTRL_ASSERT( ( arg >= ( SIXTRL_REAL_T )-1 ) &&
                   ( arg <= ( SIXTRL_REAL_T )+1 ) );
    return acos( arg );
}

SIXTRL_INLINE SIXTRL_REAL_T NS(tan)( SIXTRL_REAL_T const x ) SIXTRL_NOEXCEPT
{
    #if defined( __cplusplus )
    using std::tan;
    #endif /* defined( __cplusplus ) */
    return tan( x );
}

SIXTRL_INLINE SIXTRL_REAL_T NS(atan2)(
    SIXTRL_REAL_T const x, SIXTRL_REAL_T const y ) SIXTRL_NOEXCEPT
{
    #if defined( __cplusplus )
    using std::atan2;
    #endif /* defined( __cplusplus ) */

    return atan2( x, y );
}

SIXTRL_INLINE SIXTRL_REAL_T NS(sqrt)( SIXTRL_REAL_T const arg ) SIXTRL_NOEXCEPT
{
    #if defined( __cplusplus )
    using std::sqrt;
    #endif /* defined( __cplusplus ) */

    SIXTRL_ASSERT( arg >= ( SIXTRL_REAL_T )0 );
    return sqrt( arg );
}

SIXTRL_INLINE SIXTRL_REAL_T NS(exp)( SIXTRL_REAL_T const arg ) SIXTRL_NOEXCEPT
{
    #if defined( __cplusplus )
    using std::exp;
    #endif /* defined( __cplusplus ) */

    return exp( arg );
}

SIXTRL_INLINE SIXTRL_REAL_T NS(log)( SIXTRL_REAL_T const arg ) SIXTRL_NOEXCEPT
{
    #if defined( __cplusplus )
    using std::log;
    #endif /* defined( __cplusplus ) */

    SIXTRL_ASSERT( arg > ( SIXTRL_REAL_T )0 );
    return log( arg );
}

SIXTRL_INLINE SIXTRL_REAL_T NS(sign)( SIXTRL_REAL_T const arg ) SIXTRL_NOEXCEPT
{
    typedef SIXTRL_REAL_T real_t;
    return ( arg >= ( real_t )0 ) ? ( ( real_t )1 ) : ( ( real_t )-1 );
}

SIXTRL_INLINE SIXTRL_REAL_T NS(abs)( SIXTRL_REAL_T const arg ) SIXTRL_NOEXCEPT
{
    typedef SIXTRL_REAL_T real_t;
    return ( arg >= ( real_t )0 ) ? arg : -arg;
}

#if defined( __cplusplus ) && !defined( _GPUCODE )
}
#endif /* defined( __cplusplus ) && !defined( _GPUCODE ) */
#endif /* SIXTRACKLIB_COMMON_INTERNAL_MATH_FUNCTIONS_H__ */
