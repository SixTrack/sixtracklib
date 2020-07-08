#ifndef SIXTRACKLIB_COMMON_INTERNAL_MATH_FUNCTIONS_H__
#define SIXTRACKLIB_COMMON_INTERNAL_MATH_FUNCTIONS_H__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #if defined( __cplusplus )
        #include <algorithm>
        #include <type_traits>
        #include <cfloat>
        #include <cmath>
    #else
        #include <math.h>
        #include <float.h>
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
        #if !defined( _GPUCODE ) /* ADL! */
        using std::sin;
        #endif /* ADL / Host */
        return sin( arg );
    }

    template< typename T >
    SIXTRL_STATIC SIXTRL_INLINE SIXTRL_FN
    typename TypeMethodParamTraits< T >::value_type asin( typename
        TypeMethodParamTraits< T >::const_argument_type arg ) SIXTRL_NOEXCEPT
    {
        namespace st = SIXTRL_CXX_NAMESPACE;
        #if !defined( _GPUCODE ) /* ADL! */
        using std::asin;
        #endif /* ADL / Host */

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
        #if !defined( _GPUCODE ) /* ADL! */
        using std::cos;
        #endif /* ADL / Host */
        return cos( arg );
    }

    template< typename T >
    SIXTRL_STATIC SIXTRL_INLINE SIXTRL_FN
    typename TypeMethodParamTraits< T >::value_type acos( typename
        TypeMethodParamTraits< T >::const_argument_type arg ) SIXTRL_NOEXCEPT
    {
        #if !defined( _GPUCODE ) /* ADL! */
        using std::acos;
        #endif /* ADL / Host */
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
        #if !defined( _GPUCODE ) /* ADL! */
        using std::tan;
        #endif /* ADL / Host */
        return tan( arg );
    }

    template< typename T >
    SIXTRL_STATIC SIXTRL_INLINE SIXTRL_FN
    typename TypeMethodParamTraits< T >::value_type atan2(
        typename TypeMethodParamTraits< T >::const_argument_type y,
        typename TypeMethodParamTraits< T >::const_argument_type x
    ) SIXTRL_NOEXCEPT
    {
        #if !defined( _GPUCODE ) /* ADL! */
        using std::atan2;
        #endif /* ADL / Host */
        return atan2( y, x );
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    template< typename T >
    SIXTRL_STATIC SIXTRL_INLINE SIXTRL_FN
    typename TypeMethodParamTraits< T >::value_type sqrt( typename
        TypeMethodParamTraits< T >::const_argument_type arg ) SIXTRL_NOEXCEPT
    {
        #if !defined( _GPUCODE ) /* ADL! */
        using std::sqrt;
        #endif /* ADL / Host */
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
        #if !defined( _GPUCODE ) /* ADL! */
        using std::exp;
        #endif /* ADL / Host */
        return exp( arg );
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    template< typename T >
    SIXTRL_STATIC SIXTRL_INLINE SIXTRL_FN
    typename TypeMethodParamTraits< T >::value_type log( typename
        TypeMethodParamTraits< T >::const_argument_type arg ) SIXTRL_NOEXCEPT
    {
        #if !defined( _GPUCODE ) /* ADL! */
        using std::log;
        #endif /* ADL / Host */
        SIXTRL_ASSERT( SIXTRL_CXX_NAMESPACE::Type_comp_all_more< typename
            TypeMethodParamTraits< T >::value_type >( arg, typename
                TypeMethodParamTraits< T >::value_type{ 0 } ) );

        return log( arg );
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    template< typename T >
    SIXTRL_STATIC SIXTRL_INLINE SIXTRL_FN
    typename TypeMethodParamTraits< T >::value_type gamma( typename
        TypeMethodParamTraits< T >::const_argument_type arg ) SIXTRL_NOEXCEPT
    {
        #if !defined( _GPUCODE ) /* ADL! */
        using std::tgamma;
        #endif /* ADL / Host */

        #if !defined( _GPUCODE )
        SIXTRL_ASSERT( SIXTRL_CXX_NAMESPACE::Type_comp_all_more_or_equal< T >(
                arg, typename TypeMethodParamTraits< T >::value_type{ 1 } /
                    std::numeric_limits< typename TypeMethodParamTraits<
                        T >::value_type >::max() ) &&
            SIXTRL_CXX_NAMESPACE::Type_comp_all_less_or_equal< typename
            TypeMethodParamTraits< T >::value_type >( arg, typename
                TypeMethodParamTraits< T >::value_type{ 171.7 } ) );
        #endif /* Host */

        return tgamma( arg );
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    template< typename T >
    SIXTRL_STATIC SIXTRL_INLINE SIXTRL_FN typename std::enable_if<
        SIXTRL_CXX_NAMESPACE::Type_is_scalar< T >(),
        typename TypeMethodParamTraits< T >::value_type >::type
    pow( typename TypeMethodParamTraits< T >::const_argument_type arg,
         typename TypeMethodParamTraits< T >::const_argument_type n
       ) SIXTRL_NOEXCEPT
    {
        #if !defined( _GPUCODE ) /* ADL! */
        using std::pow;
        #endif /* !defined( _GPUCODE ) */
        return pow( arg, n );
    }

    template< typename T, typename I = int64_t >
    SIXTRL_STATIC SIXTRL_INLINE SIXTRL_FN typename std::enable_if<
        SIXTRL_CXX_NAMESPACE::Type_is_scalar< T >() &&
        SIXTRL_CXX_NAMESPACE::Type_is_scalar< I >() &&
        std::is_integral< I >(),
        typename TypeMethodParamTraits< T >::value_type >::type
    pow_int_exp(
        typename TypeMethodParamTraits< T >::const_argument_type base,
        typename TypeMethodParamTraits< I >::const_argument_type n
    ) SIXTRL_NOEXCEPT
    {
        #if defined( _GPUCODE ) && defined( __OPENCL_VERSION__ )
        return pown( base, n );
        #elif ( __cplusplus >= 201103L )
        #if !defined( _GPUCODE ) /* ADL! */
        using std::pow;
        #endif /* ADL / Host */
        return pow( base, n );
        #else
        #if !defined( _GPUCODE ) /* ADL! */
        using std::abs;
        #endif /* ADL / Host */
        namespace st = SIXTRL_CXX_NAMESPACE;
        typedef typename st::TypeMethodParamTraits< I >::value_type int_t;
        typedef typename st::TypeMethodParamTraits< T >::value_type real_t;

        real_t result;
        int_t const pos_exp = abs( n );

        SIXTRL_ASSERT( ( st::Type_comp_all_more< T >(
            st::abs< T >( base ), real_t{ 0 } ) ) || ( n > int_t{ 0 } ) );

        switch( pos_exp )
        {
            case 0:
            {
                result = real_t{ 1 };
                break;
            }

            case 1:
            {
                result = base;
                break;
            }

            case 2:
            {
                result = base * base;
                break;
            }

            case 3:
            {
                result = base * base * base;
                break;
            }

            case 4:
            {
                real_t const base_squ = base * base;
                result = base_squ * base_squ;
                break;
            }

            case 5:
            {
                real_t const base_squ = base * base;
                result = base_squ * base_squ * base;
                break;
            }

            case 6:
            {
                real_t const base_cub = base * base * base;
                result = base_cub * base_cub;
                break;
            }

            case 7:
            {
                real_t const base_cub = base * base * base;
                result = base_cub * base_cub * base;
                break;
            }

            case 8:
            {
                real_t const base_squ = base * base;
                real_t const base_quad = base_squ * base_squ;
                result = base_quad * base_quad;
                break;
            }

            default:
            {
                real_t const base_pow_8 =
                    st::pow_int_exp< T, I >( base, int_t{ 8 } );

                result  = st::pow_int_exp< T, I >( base_pow_8, pos_exp >> 3 );
                result *= st::pow_int_exp< T, I >( base_pow_8,
                    pos_exp - ( ( pos_exp >> 3 ) << 3 ) );
            }
        };

        return ( n >= int_t{ 0 } ) ? result : real_t{ 1 }  / result;
        #endif
    }

    template< typename T >
    SIXTRL_STATIC SIXTRL_INLINE SIXTRL_FN typename std::enable_if<
        SIXTRL_CXX_NAMESPACE::Type_is_scalar< T >(),
        typename TypeMethodParamTraits< T >::value_type >::type
    pow_positive_base(
        typename TypeMethodParamTraits< T >::const_argument_type base,
        typename TypeMethodParamTraits< T >::const_argument_type n
    ) SIXTRL_NOEXCEPT
    {
        #if defined( _GPUCODE ) && defined( __OPENCL_VERSION__ )
        return powr( base, n );
        #else
        #if !defined( _GPUCODE ) /* ADL! */
        using std::pow;
        #endif /* ADL / Host */
        return pow( base, n );
        #endif
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    template< typename T >
    SIXTRL_STATIC SIXTRL_INLINE SIXTRL_FN
    typename std::enable_if< SIXTRL_CXX_NAMESPACE::Type_is_scalar< T >(),
    typename TypeMethodParamTraits< T >::value_type >::type
    max( typename TypeMethodParamTraits< T >::const_argument_type lhs,
         typename TypeMethodParamTraits< T >::const_argument_type rhs
    ) SIXTRL_NOEXCEPT
    {
        #if !defined( _GPUCODE ) /* ADL! */
        using std::max;
        #endif /* ADL / Host */
        return max( lhs, rhs );
    }

    template< typename T >
    SIXTRL_STATIC SIXTRL_INLINE SIXTRL_FN
    typename std::enable_if< SIXTRL_CXX_NAMESPACE::Type_is_scalar< T >(),
    typename TypeMethodParamTraits< T >::value_type >::type
    min( typename TypeMethodParamTraits< T >::const_argument_type lhs,
         typename TypeMethodParamTraits< T >::const_argument_type rhs
    ) SIXTRL_NOEXCEPT
    {
        #if !defined( _GPUCODE ) /* ADL! */
        using std::min;
        #endif /* ADL / Host */
        return min( lhs, rhs );
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
                T >::const_argument_type y,
           typename SIXTRL_CXX_NAMESPACE::TypeMethodParamTraits<
                T >::const_argument_type x ) SIXTRL_NOEXCEPT
{
    using std::atan2;
    return atan2( y, x );
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

template< typename T >
SIXTRL_STATIC SIXTRL_INLINE SIXTRL_FN
typename SIXTRL_CXX_NAMESPACE::TypeMethodParamTraits< T >::value_type
NS(gamma)( typename SIXTRL_CXX_NAMESPACE::TypeMethodParamTraits<
    T >::const_argument_type arg ) SIXTRL_NOEXCEPT
{
    return SIXTRL_CXX_NAMESPACE::gamma< T >( arg );
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

 /* - - -- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

template< typename T >
SIXTRL_STATIC SIXTRL_INLINE SIXTRL_FN
typename SIXTRL_CXX_NAMESPACE::TypeMethodParamTraits< T >::value_type NS(pow)(
    typename SIXTRL_CXX_NAMESPACE::TypeMethodParamTraits<
        T >::const_argument_type arg,
    typename SIXTRL_CXX_NAMESPACE::TypeMethodParamTraits<
        T >::const_argument_type n ) SIXTRL_NOEXCEPT
{
    return SIXTRL_CXX_NAMESPACE::pow< T >( arg, n );
}

template< typename T, typename I = int64_t >
SIXTRL_STATIC SIXTRL_INLINE SIXTRL_FN
typename SIXTRL_CXX_NAMESPACE::TypeMethodParamTraits< T >::value_type
NS(pow_int_exp)(
    typename SIXTRL_CXX_NAMESPACE::TypeMethodParamTraits<
        T >::const_argument_type base,
    typename SIXTRL_CXX_NAMESPACE::TypeMethodParamTraits<
        I >::const_argument_type n ) SIXTRL_NOEXCEPT
{
    return SIXTRL_CXX_NAMESPACE::pow_int_exp< T, I >( base, n );
}

template< typename T >
SIXTRL_STATIC SIXTRL_INLINE SIXTRL_FN
typename SIXTRL_CXX_NAMESPACE::TypeMethodParamTraits< T >::value_type
NS(pow_positive_base)(
    typename SIXTRL_CXX_NAMESPACE::TypeMethodParamTraits<
        T >::const_argument_type base,
    typename SIXTRL_CXX_NAMESPACE::TypeMethodParamTraits<
        T >::const_argument_type n ) SIXTRL_NOEXCEPT
{
    return SIXTRL_CXX_NAMESPACE::pow_positive_base< T >( base, n );
}

/* - - -- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

template< typename T >
SIXTRL_STATIC SIXTRL_INLINE SIXTRL_FN
typename SIXTRL_CXX_NAMESPACE::TypeMethodParamTraits< T >::value_type NS(max)(
    typename SIXTRL_CXX_NAMESPACE::TypeMethodParamTraits<
        T >::const_argument_type lhs,
    typename SIXTRL_CXX_NAMESPACE::TypeMethodParamTraits<
        T >::const_argument_type rhs ) SIXTRL_NOEXCEPT
{
    return SIXTRL_CXX_NAMESPACE::max( lhs, rhs );
}

template< typename T >
SIXTRL_STATIC SIXTRL_INLINE SIXTRL_FN
typename SIXTRL_CXX_NAMESPACE::TypeMethodParamTraits< T >::value_type NS(min)(
    typename SIXTRL_CXX_NAMESPACE::TypeMethodParamTraits<
        T >::const_argument_type lhs,
    typename SIXTRL_CXX_NAMESPACE::TypeMethodParamTraits<
        T >::const_argument_type rhs ) SIXTRL_NOEXCEPT
{
    return SIXTRL_CXX_NAMESPACE::min( lhs, rhs );
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
NS(atan2)( SIXTRL_REAL_T const y, SIXTRL_REAL_T const x ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_REAL_T
NS(sqrt)( SIXTRL_REAL_T const arg ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_REAL_T
NS(exp)( SIXTRL_REAL_T const arg ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_REAL_T
NS(log)( SIXTRL_REAL_T const arg ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_REAL_T NS(gamma)(
    SIXTRL_REAL_T const arg ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_REAL_T
NS(sign)( SIXTRL_REAL_T const arg ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_REAL_T
NS(abs)( SIXTRL_REAL_T const arg ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_REAL_T
NS(pow)( SIXTRL_REAL_T const base, SIXTRL_REAL_T const n ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_REAL_T
NS(pow_positive_base)( SIXTRL_REAL_T const base,
                       SIXTRL_REAL_T const n ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_REAL_T
NS(pow_int_exp)( SIXTRL_REAL_T const base,
                 SIXTRL_INT64_T const n ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_REAL_T
NS(min)( SIXTRL_REAL_T const lhs, SIXTRL_REAL_T const rhs ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_REAL_T
NS(max)( SIXTRL_REAL_T const lhs, SIXTRL_REAL_T const rhs ) SIXTRL_NOEXCEPT;

#if defined( __cplusplus ) && !defined( _GPUCODE )
}
#endif /* defined( __cplusplus ) && !defined( _GPUCODE ) */

/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */
/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */



#if defined( __cplusplus ) && !defined( _GPUCODE )
extern "C" {
#endif /* defined( __cplusplus ) && !defined( _GPUCODE ) */

SIXTRL_INLINE SIXTRL_REAL_T NS(sin)( SIXTRL_REAL_T const arg ) SIXTRL_NOEXCEPT
{
    #if defined( __cplusplus ) && !defined( _GPUCODE ) /* ADL */
    using std::sin;
    #endif /* Host, ADL */

    return sin( arg );
}

SIXTRL_INLINE SIXTRL_REAL_T NS(asin)( SIXTRL_REAL_T const arg ) SIXTRL_NOEXCEPT
{
    #if defined( __cplusplus ) && !defined( _GPUCODE ) /* ADL */
    using std::asin;
    #endif /* Host, ADL */

    SIXTRL_ASSERT( ( arg >= ( SIXTRL_REAL_T )-1 ) &&
                   ( arg <= ( SIXTRL_REAL_T )+1 ) );
    return asin( arg );
}

SIXTRL_INLINE SIXTRL_REAL_T NS(cos)( SIXTRL_REAL_T const arg ) SIXTRL_NOEXCEPT
{
    #if defined( __cplusplus ) && !defined( _GPUCODE ) /* ADL */
    using std::cos;
    #endif /* Host, ADL */

    return cos( arg );
}

SIXTRL_INLINE SIXTRL_REAL_T NS(acos)( SIXTRL_REAL_T const arg ) SIXTRL_NOEXCEPT
{
    #if defined( __cplusplus ) && !defined( _GPUCODE ) /* ADL */
    using std::acos;
    #endif /* Host, ADL */

    SIXTRL_ASSERT( ( arg >= ( SIXTRL_REAL_T )-1 ) &&
                   ( arg <= ( SIXTRL_REAL_T )+1 ) );
    return acos( arg );
}

SIXTRL_INLINE SIXTRL_REAL_T NS(tan)( SIXTRL_REAL_T const x ) SIXTRL_NOEXCEPT
{
    #if defined( __cplusplus ) && !defined( _GPUCODE ) /* ADL */
    using std::tan;
    #endif /* Host, ADL */
    return tan( x );
}

SIXTRL_INLINE SIXTRL_REAL_T NS(atan2)(
    SIXTRL_REAL_T const y, SIXTRL_REAL_T const x ) SIXTRL_NOEXCEPT
{
    #if defined( __cplusplus ) && !defined( _GPUCODE ) /* ADL */
    using std::atan2;
    #endif /* Host, ADL */
    return atan2( y, x );
}

SIXTRL_INLINE SIXTRL_REAL_T NS(sqrt)( SIXTRL_REAL_T const arg ) SIXTRL_NOEXCEPT
{
    #if defined( __cplusplus ) && !defined( _GPUCODE ) /* ADL */
    using std::sqrt;
    #endif /* Host, ADL */
    SIXTRL_ASSERT( arg >= ( SIXTRL_REAL_T )0 );
    return sqrt( arg );
}

SIXTRL_INLINE SIXTRL_REAL_T NS(exp)( SIXTRL_REAL_T const arg ) SIXTRL_NOEXCEPT
{
    #if defined( __cplusplus ) && !defined( _GPUCODE ) /* ADL */
    using std::exp;
    #endif /* ADL / Host */
    return exp( arg );
}

SIXTRL_INLINE SIXTRL_REAL_T NS(log)( SIXTRL_REAL_T const arg ) SIXTRL_NOEXCEPT
{
    #if defined( __cplusplus ) && !defined( _GPUCODE ) /* ADL */
    using std::log;
    #endif /* ADL / Host */

    SIXTRL_ASSERT( arg > ( SIXTRL_REAL_T )0 );
    return log( arg );
}

SIXTRL_INLINE SIXTRL_REAL_T NS(gamma)( SIXTRL_REAL_T const arg ) SIXTRL_NOEXCEPT
{
    #if defined( __cplusplus ) && !defined( _GPUCODE ) /* ADL */
    using std::tgamma;
    #endif /* ADL / Host */

    #if !defined( _GPUCODE )
    SIXTRL_ASSERT( ( SIXTRL_REAL_T )171.7 >= arg );
    SIXTRL_ASSERT( ( SIXTRL_REAL_T )1 / DBL_MAX <= arg );
    #endif /* _GPUCODE */

    return tgamma( arg );
}

SIXTRL_INLINE SIXTRL_REAL_T NS(sign)( SIXTRL_REAL_T const arg ) SIXTRL_NOEXCEPT
{
    typedef SIXTRL_REAL_T real_t;
    return ( arg >= ( real_t )0 ) ? ( ( real_t )1 ) : ( ( real_t )-1 );
}

SIXTRL_INLINE SIXTRL_REAL_T NS(abs)( SIXTRL_REAL_T const arg ) SIXTRL_NOEXCEPT
{
    #if defined( _GPUCODE )
    return fabs( arg );
    #elif defined( __cplusplus )
    using std::fabs; /* ADL */
    return fabs( arg );
    #else
    typedef SIXTRL_REAL_T real_t;
    return ( arg >= ( real_t )0 ) ? arg : -arg;
    #endif
}

SIXTRL_INLINE SIXTRL_REAL_T NS(pow)(
    SIXTRL_REAL_T const base, SIXTRL_REAL_T const n ) SIXTRL_NOEXCEPT
{
    #if defined( __cplusplus ) && !defined( _GPUCODE ) /* ADL */
    using std::pow;
    #endif /* ADL / Host */

    return pow( base, n );
}

SIXTRL_INLINE SIXTRL_REAL_T NS(pow_positive_base)(
    SIXTRL_REAL_T const base, SIXTRL_REAL_T const n ) SIXTRL_NOEXCEPT
{
    #if defined( _GPUCODE ) && defined( __OPENCL_VERSION__ )
    return powr( base, n );
    #else
    #if defined( __cplusplus ) && !defined( _GPUCODE ) /* ADL */
    using std::pow;
    #endif /* ADL / Host */
    return pow( base, n );
    #endif
}

SIXTRL_INLINE SIXTRL_REAL_T NS(pow_int_exp)( SIXTRL_REAL_T const base,
                 SIXTRL_INT64_T const n ) SIXTRL_NOEXCEPT
{
    #if defined( _GPUCODE ) && defined( __OPENCL_VERSION__ )
        return pown( base, n );
        #elif defined( __cplusplus ) && ( __cplusplus >= 201103L )
        #if !defined( _GPUCODE ) /* ADL */
        using std::pow;
        #endif /* ADL / Host */
        return pow( base, n );
        #else
        #if !defined( _GPUCODE ) && defined( __cplusplus ) /* ADL */
        using std::llabs;
        #endif /* ADL / Host */
        typedef SIXTRL_REAL_T real_t;
        typedef SIXTRL_INT64_T int_t;

        real_t result;
        int_t const pos_exp = llabs( n );

        switch( pos_exp )
        {
            case 0:
            {
                result = ( real_t )1;
                break;
            }

            case 1:
            {
                result = base;
                break;
            }

            case 2:
            {
                result = base * base;
                break;
            }

            case 3:
            {
                result = base * base * base;
                break;
            }

            case 4:
            {
                real_t const base_squ = base * base;
                result = base_squ * base_squ;
                break;
            }

            case 5:
            {
                real_t const base_squ = base * base;
                result = base_squ * base_squ * base;
                break;
            }

            case 6:
            {
                real_t const base_cub = base * base * base;
                result = base_cub * base_cub;
                break;
            }

            case 7:
            {
                real_t const base_cub = base * base * base;
                result = base_cub * base_cub * base;
                break;
            }

            case 8:
            {
                real_t const base_squ = base * base;
                real_t const base_quad = base_squ * base_squ;
                result = base_quad * base_quad;
                break;
            }

            default:
            {
                real_t const base_pow_8 = NS(pow_int_exp)( base, ( int_t )8 );
                result  = NS(pow_int_exp)( base_pow_8, pos_exp >> 3 );
                result *= NS(pow_int_exp)( base_pow_8,
                    pos_exp - ( ( pos_exp >> 3 ) << 3 ) );
            }
        };

        return ( n >= ( int_t )0 ) ? result : ( real_t )1  / result;
        #endif
}

SIXTRL_INLINE SIXTRL_REAL_T NS(min)(
    SIXTRL_REAL_T const lhs, SIXTRL_REAL_T const rhs ) SIXTRL_NOEXCEPT
{
    #if defined( _GPUCODE )
    return min( lhs, rhs );
    #elif defined( __cplusplus ) /* ADL */
    using std::min;
    return min( lhs, rhs );
    #else
    return ( lhs <= rhs ) ? lhs : rhs;
    #endif /* _GPUCODE */
}

SIXTRL_INLINE SIXTRL_REAL_T NS(max)(
    SIXTRL_REAL_T const lhs, SIXTRL_REAL_T const rhs ) SIXTRL_NOEXCEPT
{
    #if defined( _GPUCODE )
    return max( lhs, rhs );
    #elif defined( __cplusplus ) /* ADL */
    using std::max;
    return max( lhs, rhs );
    #else
    return ( lhs >= rhs ) ? lhs : rhs;
    #endif /* _GPUCODE */
}

#if defined( __cplusplus ) && !defined( _GPUCODE )
}
#endif /* defined( __cplusplus ) && !defined( _GPUCODE ) */
#endif /* SIXTRACKLIB_COMMON_INTERNAL_MATH_FUNCTIONS_H__ */
