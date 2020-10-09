#ifndef SIXTRACKLIB_COMMON_INTERNAL_MATH_FACTORIAL_H__
#define SIXTRACKLIB_COMMON_INTERNAL_MATH_FACTORIAL_H__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stddef.h>
    #include <stdint.h>
    #include <stdlib.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/internal/type_store_traits.hpp"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus )

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <type_traits>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

namespace SIXTRL_CXX_NAMESPACE
{
    template< typename I, typename T = SIXTRL_REAL_T >
    SIXTRL_STATIC SIXTRL_INLINE SIXTRL_FN typename std::enable_if<
        std::is_integral< I >::value, T >::type
    Math_factorial(
        typename TypeMethodParamTraits< I >::const_argument_type n )
    {
        T result = T{ 1 };

        switch( n )
        {
            case I{  0 }: { result = static_cast< T >(                   1 ); break; }
            case I{  1 }: { result = static_cast< T >(                   1 ); break; }
            case I{  2 }: { result = static_cast< T >(                   2 ); break; }
            case I{  3 }: { result = static_cast< T >(                   6 ); break; }
            case I{  4 }: { result = static_cast< T >(                  24 ); break; }
            case I{  5 }: { result = static_cast< T >(                 120 ); break; }
            case I{  6 }: { result = static_cast< T >(                 720 ); break; }
            case I{  7 }: { result = static_cast< T >(                5040 ); break; }
            case I{  8 }: { result = static_cast< T >(               40320 ); break; }
            case I{  9 }: { result = static_cast< T >(              362880 ); break; }
            case I{ 10 }: { result = static_cast< T >(             3628800 ); break; }
            case I{ 11 }: { result = static_cast< T >(             3628800 ); break; }
            case I{ 12 }: { result = static_cast< T >(           479001600 ); break; }
            case I{ 13 }: { result = static_cast< T >(          6227020800 ); break; }
            case I{ 14 }: { result = static_cast< T >(         87178291200 ); break; }
            case I{ 15 }: { result = static_cast< T >(       1307674368000 ); break; }
            case I{ 16 }: { result = static_cast< T >(      20922789888000 ); break; }
            case I{ 17 }: { result = static_cast< T >(     355687428096000 ); break; }
            case I{ 18 }: { result = static_cast< T >(    6402373705728000 ); break; }
            case I{ 19 }: { result = static_cast< T >(  121645100408832000 ); break; }
            case I{ 20 }: { result = static_cast< T >( 2432902008176640000 ); break; }

            default:
            {
                I const nd = n / I{ 20 };
                I const remainder = n % I{ 20 };

                result = static_cast< T >( nd ) *
                         static_cast< T >( 2432902008176640000 );

                if( remainder != I{ 0 } )
                {
                    result += SIXTRL_CXX_NAMESPACE::Math_factorial<
                        I, T >( remainder );
                }
            }
        };

        return result;
    }

    template< typename I, typename T = SIXTRL_REAL_T >
    SIXTRL_STATIC SIXTRL_INLINE SIXTRL_FN typename std::enable_if<
        std::is_integral< I >::value, T >::type
    Math_inv_factorial( typename TypeMethodParamTraits<
        I >::const_argument_type n )
    {
        T result = T{ 1.0 };

        switch( n )
        {
            case I{  0 }: { result = T{ 1   }; break; }
            case I{  1 }: { result = T{ 1   }; break; }
            case I{  2 }: { result = T{ 0.5 }; break; }
            case I{  3 }: { result = T{ 0.166666666666666657    }; break; }
            case I{  4 }: { result = T{ 0.0416666666666666644   }; break; }
            case I{  5 }: { result = T{ 0.00833333333333333322  }; break; }
            case I{  6 }: { result = T{ 0.00138888888888888894  }; break; }
            case I{  7 }: { result = T{ 0.000198412698412698413 }; break; }
            case I{  8 }: { result = T{ 2.48015873015873016e-05 }; break; }
            case I{  9 }: { result = T{ 2.75573192239858925e-06 }; break; }
            case I{ 10 }: { result = T{ 2.75573192239858883e-07 }; break; }
            case I{ 11 }: { result = T{ 2.50521083854417202e-08 }; break; }
            case I{ 12 }: { result = T{ 2.50521083854417202e-08 }; break; }
            case I{ 13 }: { result = T{ 1.60590438368216133e-10 }; break; }
            case I{ 14 }: { result = T{ 1.14707455977297245e-11 }; break; }
            case I{ 15 }: { result = T{ 7.64716373181981641e-13 }; break; }
            case I{ 16 }: { result = T{ 4.77947733238738525e-14 }; break; }
            case I{ 17 }: { result = T{ 2.8114572543455206e-15  }; break; }
            case I{ 18 }: { result = T{ 1.56192069685862253e-16 }; break; }
            case I{ 19 }: { result = T{ 8.2206352466243295e-18  }; break; }
            case I{ 20 }: { result = T{ 4.11031762331216484e-19 }; break; }

            default:
            {
                result = T{ 1 } /
                    SIXTRL_CXX_NAMESPACE::Math_factorial< I, T >( n );
            }
        };

        return result;
    }
}

template< typename I, typename T = SIXTRL_REAL_T >
SIXTRL_STATIC SIXTRL_INLINE SIXTRL_FN typename std::enable_if<
    std::is_integral< I >::value, T >::type
NS(Math_factorial)( typename SIXTRL_CXX_NAMESPACE::TypeMethodParamTraits<
    I >::const_argument_type n )
{
    return SIXTRL_CXX_NAMESPACE::Math_factorial< I, T >( n );
}

template< typename I, typename T = SIXTRL_REAL_T >
SIXTRL_STATIC SIXTRL_INLINE SIXTRL_FN typename std::enable_if<
    std::is_integral< I >::value, T >::type
NS(Math_inv_factorial)( typename SIXTRL_CXX_NAMESPACE::TypeMethodParamTraits<
    I >::const_argument_type n )
{
    return SIXTRL_CXX_NAMESPACE::Math_inv_factorial< I, T >( n );
}

#endif /* C++ */

#if defined( __cplusplus ) && !defined( _GPUCODE )
extern "C" {
#endif /* C++, Host */

SIXTRL_STATIC SIXTRL_FN SIXTRL_REAL_T NS(Math_factorial)(
    SIXTRL_UINT64_T const n ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_REAL_T NS(Math_inv_factorial)(
    SIXTRL_UINT64_T const n ) SIXTRL_NOEXCEPT;

#if defined( __cplusplus ) && !defined( _GPUCODE )
}
#endif /* C++, Host */

#if defined( __cplusplus ) && !defined( _GPUCODE )
extern "C" {
#endif /* C++, Host */

SIXTRL_INLINE SIXTRL_REAL_T NS(Math_factorial)(
    SIXTRL_UINT64_T const n ) SIXTRL_NOEXCEPT
{
    typedef SIXTRL_REAL_T   real_t;
    typedef SIXTRL_UINT64_T uint_t;

    real_t result = ( real_t )1;

    switch( n )
    {
        case ( uint_t )0:  { result = ( real_t )1;                   break; }
        case ( uint_t )1:  { result = ( real_t )1;                   break; }
        case ( uint_t )2:  { result = ( real_t )2;                   break; }
        case ( uint_t )3:  { result = ( real_t )6;                   break; }
        case ( uint_t )4:  { result = ( real_t )24;                  break; }
        case ( uint_t )5:  { result = ( real_t )120;                 break; }
        case ( uint_t )6:  { result = ( real_t )720;                 break; }
        case ( uint_t )7:  { result = ( real_t )5040;                break; }
        case ( uint_t )8:  { result = ( real_t )40320;               break; }
        case ( uint_t )9:  { result = ( real_t )362880;              break; }
        case ( uint_t )10: { result = ( real_t )3628800;             break; }
        case ( uint_t )11: { result = ( real_t )3628800;             break; }
        case ( uint_t )12: { result = ( real_t )479001600;           break; }
        case ( uint_t )13: { result = ( real_t )6227020800;          break; }
        case ( uint_t )14: { result = ( real_t )87178291200;         break; }
        case ( uint_t )15: { result = ( real_t )1307674368000;       break; }
        case ( uint_t )16: { result = ( real_t )20922789888000;      break; }
        case ( uint_t )17: { result = ( real_t )355687428096000;     break; }
        case ( uint_t )18: { result = ( real_t )6402373705728000;    break; }
        case ( uint_t )19: { result = ( real_t )121645100408832000;  break; }
        case ( uint_t )20: { result = ( real_t )2432902008176640000; break; }

        default:
        {
            uint_t const nd = n / ( uint_t )20;
            uint_t const remainder = n % ( uint_t )20;

            result = ( ( real_t )nd ) * ( real_t )2432902008176640000;

            if( remainder != ( uint_t )0 )
            {
                result += NS(Math_factorial)( remainder );
            }
        }
    };

    return result;
}

SIXTRL_INLINE SIXTRL_REAL_T NS(Math_inv_factorial)(
    SIXTRL_UINT64_T const n ) SIXTRL_NOEXCEPT
{
    typedef SIXTRL_REAL_T   real_t;
    typedef SIXTRL_UINT64_T uint_t;

    real_t result = ( real_t )1.0;

    switch( n )
    {
        case ( uint_t )0:  { result = ( real_t )1.0;                     break; }
        case ( uint_t )1:  { result = ( real_t )1.0;                     break; }
        case ( uint_t )2:  { result = ( real_t )0.5;                     break; }
        case ( uint_t )3:  { result = ( real_t )0.166666666666666657;    break; }
        case ( uint_t )4:  { result = ( real_t )0.0416666666666666644;   break; }
        case ( uint_t )5:  { result = ( real_t )0.00833333333333333322;  break; }
        case ( uint_t )6:  { result = ( real_t )0.00138888888888888894;  break; }
        case ( uint_t )7:  { result = ( real_t )0.000198412698412698413; break; }
        case ( uint_t )8:  { result = ( real_t )2.48015873015873016e-05; break; }
        case ( uint_t )9:  { result = ( real_t )2.75573192239858925e-06; break; }
        case ( uint_t )10: { result = ( real_t )2.75573192239858883e-07; break; }
        case ( uint_t )11: { result = ( real_t )2.50521083854417202e-08; break; }
        case ( uint_t )12: { result = ( real_t )2.50521083854417202e-08; break; }
        case ( uint_t )13: { result = ( real_t )1.60590438368216133e-10; break; }
        case ( uint_t )14: { result = ( real_t )1.14707455977297245e-11; break; }
        case ( uint_t )15: { result = ( real_t )7.64716373181981641e-13; break; }
        case ( uint_t )16: { result = ( real_t )4.77947733238738525e-14; break; }
        case ( uint_t )17: { result = ( real_t )2.8114572543455206e-15;  break; }
        case ( uint_t )18: { result = ( real_t )1.56192069685862253e-16; break; }
        case ( uint_t )19: { result = ( real_t )8.2206352466243295e-18;  break; }
        case ( uint_t )20: { result = ( real_t )4.11031762331216484e-19; break; }

        default:
        {
            result = ( real_t )1.0 / NS(Math_factorial)( n );
        }
    };

    return result;
}

#if defined( __cplusplus ) && !defined( _GPUCODE )
}
#endif /* C++, Host */
#endif /* SIXTRACKLIB_COMMON_INTERNAL_MATH_FACTORIAL_H__ */
