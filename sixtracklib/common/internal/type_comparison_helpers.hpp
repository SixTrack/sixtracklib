#ifndef SIXTRACKLIB_COMMON_INTERNAL_TYPE_COMPARISON_HELPERS_CXX_HPP__
#define SIXTRACKLIB_COMMON_INTERNAL_TYPE_COMPARISON_HELPERS_CXX_HPP__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #if defined( __cplusplus )
        #include <cstdlib>
        #include <iterator>
        #include <limits>
        #include <type_traits>
    #else /* C++ */
        #include <stdint.h>
        #include <limits.h>
        #include <stdbool.h>
    #endif /* C++ */
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/internal/buffer_main_defines.h"
    #include "sixtracklib/common/internal/compiler_attributes.h"
    #include "sixtracklib/common/control/definitions.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus )

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/internal/type_store_traits.hpp"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

namespace SIXTRL_CXX_NAMESPACE
{
    template< typename T >
    struct TypeCompLogicTypeTraits
    {
        typedef bool logic_t;
        typedef int  cmp_result_t;
    };

    /* ********************************************************************** */

    template< class E >
    struct ObjDataLogicTypeTraits
    {
        typedef int cmp_result_t;
        typedef bool logic_t;
    };

    template< typename CmpResultT >
    struct TypeCompResultTraits
    {
        static_assert(
            std::numeric_limits< CmpResultT >::min() < CmpResultT{ 0 } &&
            std::numeric_limits< CmpResultT >::max() > CmpResultT{ 0 },
            "CmpResultT unsuitable for use as a value comparison result type" );

        static constexpr CmpResultT CMP_LHS_EQUAL_RHS =
            CmpResultT{ 0 };

        static constexpr CmpResultT CMP_LHS_LARGER_RHS =
            std::numeric_limits< CmpResultT >::max();

        static constexpr CmpResultT CMP_LHS_SMALLER_RHS =
            std::numeric_limits< CmpResultT >::min();
    };

    template< typename CmpResultT >
    constexpr CmpResultT TypeCompResultTraits< CmpResultT >::CMP_LHS_EQUAL_RHS;

    template< typename CmpResultT >
    constexpr CmpResultT TypeCompResultTraits< CmpResultT >::CMP_LHS_LARGER_RHS;

    template< typename CmpResultT >
    constexpr CmpResultT TypeCompResultTraits< CmpResultT >::CMP_LHS_SMALLER_RHS;

    /* ********************************************************************** */

    template< class Logic >
    struct TypeCompLogicOps
    {
        typedef typename TypeMethodParamTraits< Logic >::argument_type
                logic_arg_t;

        static SIXTRL_FN bool any( logic_arg_t expr ) SIXTRL_NOEXCEPT
        {
            return ( expr ) ? true : false;
        }

        static SIXTRL_FN bool all( logic_arg_t expr ) SIXTRL_NOEXCEPT
        {
            return ( expr ) ? true : false;
        }
    };

    /* Specialization */

    template<> struct TypeCompLogicOps< bool >
    {
        typedef bool const logic_arg_t;

        static SIXTRL_INLINE SIXTRL_FN bool any(
            logic_arg_t expr ) SIXTRL_NOEXCEPT
        {
            return expr;
        }

        static SIXTRL_INLINE SIXTRL_FN bool all(
            logic_arg_t expr ) SIXTRL_NOEXCEPT
        {
            return expr;
        }
    };

    template< typename T >
    SIXTRL_STATIC SIXTRL_INLINE bool Type_comp_all(
        typename TypeMethodParamTraits< typename TypeCompLogicTypeTraits<
            T >::logic_t >::argument_type expression ) SIXTRL_NOEXCEPT
    {
        typedef typename TypeCompLogicTypeTraits< T >::logic_t logic_t;
        return TypeCompLogicOps< logic_t >::all( expression );
    }

    template< typename T >
    static SIXTRL_FN bool Type_comp_any(
        typename TypeMethodParamTraits< typename TypeCompLogicTypeTraits<
            T >::logic_t >::argument_type expression ) SIXTRL_NOEXCEPT
    {
        typedef typename TypeCompLogicTypeTraits< T >::logic_t logic_t;
        return TypeCompLogicOps< logic_t >::any( expression );
    }

    /* ********************************************************************** */

    template< typename T >
    static SIXTRL_FN constexpr bool
    TypeComp_consitent_logic_type() SIXTRL_NOEXCEPT
    {
        /* The type T and its associated logic_t type have to share the same
         * behaviour with respect to being a scalar
         *
         * TODO: Expand this to enforce cardinality -> i.e. a vector valued
         *       type neeeds a vector valued logic_t with the same dimensions,
         *       etc. */

        return
            ( ( SIXTRL_CXX_NAMESPACE::Type_is_scalar< T >() ) &&
              ( SIXTRL_CXX_NAMESPACE::Type_is_scalar<
                    typename TypeCompLogicTypeTraits< T >::logic_t >() )
            ) ||
            ( ( !SIXTRL_CXX_NAMESPACE::Type_is_scalar< T >() ) &&
              ( !SIXTRL_CXX_NAMESPACE::Type_is_scalar<
                    typename TypeCompLogicTypeTraits< T >::logic_t >() ) );
    }

    template< typename LhsT, typename RhsT = LhsT >
    SIXTRL_STATIC SIXTRL_INLINE SIXTRL_FN
    typename TypeCompLogicTypeTraits< LhsT >::logic_t Type_comp_less(
        typename TypeMethodParamTraits< LhsT >::const_argument_type lhs,
        typename TypeMethodParamTraits< RhsT >::const_argument_type rhs
    ) SIXTRL_NOEXCEPT
    {
        typedef typename TypeCompLogicTypeTraits< LhsT >::logic_t logic_t;
        static_assert(
            SIXTRL_CXX_NAMESPACE::TypeComp_consitent_logic_type< LhsT >(),
            "base type LhsT and logic type are inconsistent" );

        return static_cast< logic_t >( lhs < rhs );
    }

    template< typename LhsT, typename RhsT = LhsT >
    SIXTRL_STATIC SIXTRL_INLINE SIXTRL_FN
    typename TypeCompLogicTypeTraits< LhsT >::logic_t Type_comp_equal(
        typename TypeMethodParamTraits< LhsT >::const_argument_type lhs,
        typename TypeMethodParamTraits< RhsT >::const_argument_type rhs
    ) SIXTRL_NOEXCEPT
    {
        typedef typename TypeCompLogicTypeTraits< LhsT >::logic_t logic_t;
        static_assert(
            SIXTRL_CXX_NAMESPACE::TypeComp_consitent_logic_type< LhsT >(),
            "base type LhsT and logic type are inconsistent" );

        return static_cast< logic_t >( lhs == rhs );
    }

    template< typename LhsT, typename RhsT = LhsT >
    SIXTRL_STATIC SIXTRL_INLINE SIXTRL_FN
    typename TypeCompLogicTypeTraits< LhsT >::logic_t Type_comp_less_or_equal(
        typename TypeMethodParamTraits< LhsT >::const_argument_type lhs,
        typename TypeMethodParamTraits< RhsT >::const_argument_type rhs
    ) SIXTRL_NOEXCEPT
    {
        typedef typename TypeCompLogicTypeTraits< LhsT >::logic_t logic_t;
        static_assert(
            SIXTRL_CXX_NAMESPACE::TypeComp_consitent_logic_type< LhsT >(),
            "base type LhsT and logic type are inconsistent" );

        return static_cast< logic_t >( ( lhs < rhs ) || ( lhs == rhs ) );
    }

    template< typename LhsT, typename RhsT = LhsT >
    SIXTRL_STATIC SIXTRL_INLINE SIXTRL_FN
    typename TypeCompLogicTypeTraits< LhsT >::logic_t
    Type_comp_more(
        typename TypeMethodParamTraits< LhsT >::const_argument_type lhs,
        typename TypeMethodParamTraits< RhsT >::const_argument_type rhs
    ) SIXTRL_NOEXCEPT
    {
        namespace st = SIXTRL_CXX_NAMESPACE;
        return !st::Type_comp_less_or_equal< LhsT, RhsT >( lhs, rhs );
    }

    template< typename LhsT, typename RhsT = LhsT >
    SIXTRL_STATIC SIXTRL_INLINE SIXTRL_FN
    typename TypeCompLogicTypeTraits< LhsT >::logic_t
    Type_comp_more_or_equal(
        typename TypeMethodParamTraits< LhsT >::const_argument_type lhs,
        typename TypeMethodParamTraits< RhsT >::const_argument_type rhs
    ) SIXTRL_NOEXCEPT
    {
        return !SIXTRL_CXX_NAMESPACE::Type_comp_less< LhsT, RhsT >( lhs, rhs );
    }

    template< typename LhsT, typename RhsT = LhsT >
    SIXTRL_STATIC SIXTRL_INLINE SIXTRL_FN
    typename TypeCompLogicTypeTraits< LhsT >::logic_t
    Type_comp_not_equal(
        typename TypeMethodParamTraits< LhsT >::const_argument_type lhs,
        typename TypeMethodParamTraits< RhsT >::const_argument_type rhs
    ) SIXTRL_NOEXCEPT
    {
        namespace st = SIXTRL_CXX_NAMESPACE;
        return !st::Type_comp_equal< LhsT, RhsT >( lhs, rhs );
    }

    /* ====================================================================== */

    template< typename LhsT, typename RhsT = LhsT >
    static SIXTRL_FN bool Type_comp_all_less(
        typename TypeMethodParamTraits< LhsT >::const_argument_type lhs,
        typename TypeMethodParamTraits< RhsT >::const_argument_type rhs
    ) SIXTRL_NOEXCEPT
    {
        return SIXTRL_CXX_NAMESPACE::Type_comp_all< LhsT >(
            SIXTRL_CXX_NAMESPACE::Type_comp_less< LhsT, RhsT >( lhs, rhs ) );
    }

    template< typename LhsT, typename RhsT = LhsT >
    SIXTRL_STATIC SIXTRL_INLINE SIXTRL_FN bool Type_comp_any_less(
        typename TypeMethodParamTraits< LhsT >::const_argument_type lhs,
        typename TypeMethodParamTraits< RhsT >::const_argument_type rhs
    ) SIXTRL_NOEXCEPT
    {
        return SIXTRL_CXX_NAMESPACE::Type_comp_any< LhsT >(
            SIXTRL_CXX_NAMESPACE::Type_comp_less< LhsT, RhsT >( lhs, rhs ) );
    }

    /* -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- - */

    template< typename LhsT, typename RhsT = LhsT >
    SIXTRL_STATIC SIXTRL_INLINE SIXTRL_FN bool Type_comp_all_equal(
        typename TypeMethodParamTraits< LhsT >::const_argument_type lhs,
        typename TypeMethodParamTraits< RhsT >::const_argument_type rhs
    ) SIXTRL_NOEXCEPT
    {
        return SIXTRL_CXX_NAMESPACE::Type_comp_all< LhsT >(
            SIXTRL_CXX_NAMESPACE::Type_comp_equal< LhsT, RhsT >( lhs, rhs ) );
    }

    template< typename LhsT, typename RhsT = LhsT >
    SIXTRL_STATIC SIXTRL_INLINE SIXTRL_FN bool Type_comp_any_equal(
        typename TypeMethodParamTraits< LhsT >::const_argument_type lhs,
        typename TypeMethodParamTraits< RhsT >::const_argument_type rhs
    ) SIXTRL_NOEXCEPT
    {
        return SIXTRL_CXX_NAMESPACE::Type_comp_any< LhsT >(
            SIXTRL_CXX_NAMESPACE::Type_comp_equal< LhsT, RhsT >( lhs, rhs ) );
    }

    /* -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- - */

    template< typename LhsT, typename RhsT = LhsT >
    SIXTRL_STATIC SIXTRL_INLINE SIXTRL_FN bool Type_comp_all_less_or_equal(
        typename TypeMethodParamTraits< LhsT >::const_argument_type lhs,
        typename TypeMethodParamTraits< RhsT >::const_argument_type rhs
    ) SIXTRL_NOEXCEPT
    {
        namespace st = SIXTRL_CXX_NAMESPACE;
        return st::Type_comp_all< LhsT >( st::Type_comp_less_or_equal<
            LhsT, RhsT >( lhs, rhs ) );
    }

    template< typename LhsT, typename RhsT = LhsT >
    SIXTRL_STATIC SIXTRL_INLINE SIXTRL_FN bool Type_comp_any_less_or_equal(
        typename TypeMethodParamTraits< LhsT >::const_argument_type lhs,
        typename TypeMethodParamTraits< RhsT >::const_argument_type rhs
    ) SIXTRL_NOEXCEPT
    {
        namespace st = SIXTRL_CXX_NAMESPACE;
        return st::Type_comp_any< LhsT >( st::Type_comp_less_or_equal<
                LhsT, RhsT >( lhs, rhs ) );
    }

    /* -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- - */

    template< typename LhsT, typename RhsT = LhsT >
    SIXTRL_STATIC SIXTRL_INLINE SIXTRL_FN bool Type_comp_all_more(
        typename TypeMethodParamTraits< LhsT >::const_argument_type lhs,
        typename TypeMethodParamTraits< RhsT >::const_argument_type rhs
    ) SIXTRL_NOEXCEPT
    {
        namespace st = SIXTRL_CXX_NAMESPACE;
        return st::Type_comp_all< LhsT >( st::Type_comp_more<
            LhsT, RhsT >( lhs, rhs ) );
    }

    template< typename LhsT, typename RhsT = LhsT >
    SIXTRL_STATIC SIXTRL_INLINE SIXTRL_FN bool Type_comp_any_more(
        typename TypeMethodParamTraits< LhsT >::const_argument_type lhs,
        typename TypeMethodParamTraits< RhsT >::const_argument_type rhs
    ) SIXTRL_NOEXCEPT
    {
        namespace st = SIXTRL_CXX_NAMESPACE;
        return st::Type_comp_any< LhsT >( st::Type_comp_more<
            LhsT, RhsT >( lhs, rhs ) );
    }

    /* -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- - */

    template< typename LhsT, typename RhsT = LhsT >
    SIXTRL_STATIC SIXTRL_INLINE SIXTRL_FN bool Type_comp_all_more_or_equal(
        typename TypeMethodParamTraits< LhsT >::const_argument_type lhs,
        typename TypeMethodParamTraits< RhsT >::const_argument_type rhs
    ) SIXTRL_NOEXCEPT
    {
        return SIXTRL_CXX_NAMESPACE::Type_comp_all< LhsT >(
            SIXTRL_CXX_NAMESPACE::Type_comp_more_or_equal<
                LhsT, RhsT >( lhs, rhs ) );
    }

    template< typename LhsT, typename RhsT = LhsT >
    SIXTRL_STATIC SIXTRL_INLINE SIXTRL_FN bool Type_comp_any_more_or_equal(
        typename TypeMethodParamTraits< LhsT >::const_argument_type lhs,
        typename TypeMethodParamTraits< RhsT >::const_argument_type rhs
    ) SIXTRL_NOEXCEPT
    {
        return SIXTRL_CXX_NAMESPACE::Type_comp_any< LhsT >(
            SIXTRL_CXX_NAMESPACE::Type_comp_more_or_equal<
                LhsT, RhsT >( lhs, rhs ) );
    }

    /* -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- - */

    template< typename LhsT, typename RhsT = LhsT >
    SIXTRL_STATIC SIXTRL_INLINE SIXTRL_FN bool Type_comp_all_not_equal(
        typename TypeMethodParamTraits< LhsT >::const_argument_type lhs,
        typename TypeMethodParamTraits< RhsT >::const_argument_type rhs
    ) SIXTRL_NOEXCEPT
    {
        return SIXTRL_CXX_NAMESPACE::Type_comp_all< LhsT >(
            SIXTRL_CXX_NAMESPACE::Type_comp_not_equal<
                LhsT, RhsT >( lhs, rhs ) );
    }

    template< typename LhsT, typename RhsT = LhsT >
    SIXTRL_STATIC SIXTRL_INLINE SIXTRL_FN bool Type_comp_any_not_equal(
        typename TypeMethodParamTraits< LhsT >::const_argument_type lhs,
        typename TypeMethodParamTraits< RhsT >::const_argument_type rhs
    ) SIXTRL_NOEXCEPT
    {
        return SIXTRL_CXX_NAMESPACE::Type_comp_any< LhsT >(
            SIXTRL_CXX_NAMESPACE::Type_comp_not_equal< LhsT, RhsT >(
                lhs, rhs ) );
    }
}
#endif /* defined( __cplusplus ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/internal/math_functions.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus )
namespace SIXTRL_CXX_NAMESPACE
{
    template< typename LhsT, typename RhsT = LhsT >
    SIXTRL_STATIC SIXTRL_INLINE SIXTRL_FN
    typename TypeCompLogicTypeTraits< LhsT >::logic_t
    Type_comp_is_close(
        typename TypeMethodParamTraits< LhsT >::const_argument_type lhs,
        typename TypeMethodParamTraits< RhsT >::const_argument_type rhs,
        typename TypeMethodParamTraits< RhsT >::const_argument_type rel_tol,
        typename TypeMethodParamTraits< RhsT >::const_argument_type abs_tol )
    {
        namespace st = SIXTRL_CXX_NAMESPACE;
        typedef typename TypeMethodParamTraits< LhsT >::value_type lhs_t;
        typedef typename TypeMethodParamTraits< RhsT >::value_type rhs_t;

        lhs_t diff = lhs;
        diff -= rhs;
        diff  = st::abs< lhs_t >( diff );

        rhs_t cmp_value = st::abs< rhs_t >( rhs );
        cmp_value *= rel_tol;
        cmp_value += abs_tol;

        return st::Type_comp_less_or_equal< lhs_t >( diff, cmp_value );
    }

    /* -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- - */

    template< typename LhsT, typename RhsT = LhsT >
    SIXTRL_STATIC SIXTRL_INLINE SIXTRL_FN bool Type_comp_all_are_close(
        typename TypeMethodParamTraits< LhsT >::const_argument_type lhs,
        typename TypeMethodParamTraits< RhsT >::const_argument_type rhs,
        typename TypeMethodParamTraits< RhsT >::const_argument_type rel_tol,
        typename TypeMethodParamTraits< RhsT >::const_argument_type abs_tol
    ) SIXTRL_NOEXCEPT
    {
        namespace st = SIXTRL_CXX_NAMESPACE;
        return st::Type_comp_all< LhsT >( st::Type_comp_is_close< LhsT, RhsT >(
                lhs, rhs, rel_tol, abs_tol ) );
    }

    template< typename LhsT, typename RhsT = LhsT >
    SIXTRL_STATIC SIXTRL_INLINE SIXTRL_FN bool Type_comp_any_are_close(
        typename TypeMethodParamTraits< LhsT >::const_argument_type lhs,
        typename TypeMethodParamTraits< RhsT >::const_argument_type rhs,
        typename TypeMethodParamTraits< RhsT >::const_argument_type rel_tol,
        typename TypeMethodParamTraits< RhsT >::const_argument_type abs_tol
    ) SIXTRL_NOEXCEPT
    {
        namespace st = SIXTRL_CXX_NAMESPACE;
        return st::Type_comp_any< LhsT >( st::Type_comp_is_close< LhsT, RhsT >(
                lhs, rhs, rel_tol, abs_tol ) );
    }

    /* --------------------------------------------------------------------- */

    template< typename LhsT, typename CmpResultT = typename
                    TypeCompLogicTypeTraits< LhsT >::cmp_result_t,
              typename RhsT = LhsT >
    SIXTRL_STATIC SIXTRL_INLINE SIXTRL_FN CmpResultT Type_value_comp_result(
        typename TypeMethodParamTraits< LhsT >::const_argument_type lhs,
        typename TypeMethodParamTraits< RhsT >::const_argument_type rhs )
    {
        namespace st = SIXTRL_CXX_NAMESPACE;
        typedef CmpResultT result_t;
        typedef TypeCompResultTraits< result_t > cmp_traits_t;

        result_t cmp_result = cmp_traits_t::CMP_LHS_SMALLER_RHS;

        if( st::Type_comp_all_equal< LhsT, RhsT >( rhs, lhs ) )
        {
            cmp_result = cmp_traits_t::CMP_LHS_EQUAL_RHS;
        }
        else if( st::Type_comp_any_more< LhsT, RhsT >( rhs, lhs ) )
        {
            cmp_result = cmp_traits_t::CMP_LHS_LARGER_RHS;
        }

        return cmp_result;
    }

    template< typename LhsT, typename CmpResultT = typename
                TypeCompLogicTypeTraits< LhsT >::cmp_result_t,
              typename RhsT = LhsT >
    SIXTRL_STATIC SIXTRL_INLINE SIXTRL_FN CmpResultT Type_value_comp_result(
        typename TypeMethodParamTraits< LhsT >::const_argument_type lhs,
        typename TypeMethodParamTraits< RhsT >::const_argument_type rhs,
        typename TypeMethodParamTraits< RhsT >::const_argument_type rel_tol,
        typename TypeMethodParamTraits< RhsT >::const_argument_type abs_tol )
    {
        namespace st = SIXTRL_CXX_NAMESPACE;
        typedef CmpResultT result_t;
        typedef TypeCompResultTraits< result_t > cmp_traits_t;

        result_t cmp_result = cmp_traits_t::CMP_LHS_SMALLER_RHS;

        if( st::Type_comp_all_are_close< LhsT, RhsT >(
                rhs, lhs, rel_tol, abs_tol ) )
        {
            cmp_result = cmp_traits_t::CMP_LHS_EQUAL_RHS;
        }
        else if( st::Type_comp_any_more< LhsT, RhsT >( rhs, lhs ) )
        {
            cmp_result = cmp_traits_t::CMP_LHS_LARGER_RHS;
        }

        return cmp_result;
    }

    /* --------------------------------------------------------------------- */

    template< typename LhsIter, typename CmpResultT = typename
                TypeCompLogicTypeTraits< typename std::iterator_traits< LhsIter
                    >::value_type >::cmp_result_t,
              typename RhsIter = LhsIter >
    SIXTRL_STATIC SIXTRL_INLINE SIXTRL_FN CmpResultT
    Type_value_comp_result_for_range( LhsIter lhs_it, LhsIter lhs_end,
                                      RhsIter rhs_it )
    {
        typedef CmpResultT result_t;
        typedef TypeCompResultTraits< result_t > cmp_traits_t;
        typedef typename std::iterator_traits< LhsIter >::value_type lhs_value_t;
        typedef typename std::iterator_traits< RhsIter >::value_type rhs_value_t;

        result_t cmp_result = cmp_traits_t::CMP_LHS_EQUAL_RHS;

        while( ( cmp_result == cmp_traits_t::CMP_LHS_EQUAL_RHS ) &&
               ( lhs_it != lhs_end ) )
        {
            cmp_result = SIXTRL_CXX_NAMESPACE::Type_value_comp_result<
                lhs_value_t, result_t, rhs_value_t >( *lhs_it, *rhs_it );
            ++lhs_it;
            ++rhs_it;
        }

        return cmp_result;
    }

    template< typename LhsIter, typename CmpResultT = typename
                TypeCompLogicTypeTraits< typename std::iterator_traits< LhsIter
                    >::value_type >::cmp_result_t,
              typename RhsIter = LhsIter >
    SIXTRL_STATIC SIXTRL_INLINE SIXTRL_FN CmpResultT
    Type_value_comp_result_for_range(
        LhsIter lhs_it, LhsIter lhs_end, RhsIter rhs_it,
        typename TypeMethodParamTraits< typename std::iterator_traits<
            RhsIter >::value_type >::const_argument_type rel_tol,
        typename TypeMethodParamTraits< typename std::iterator_traits<
            RhsIter >::value_type >::const_argument_type abs_tol )
    {
        typedef CmpResultT result_t;
        typedef TypeCompResultTraits< result_t > cmp_traits_t;
        typedef typename std::iterator_traits< LhsIter >::value_type lhs_value_t;
        typedef typename std::iterator_traits< RhsIter >::value_type rhs_value_t;

        result_t cmp_result = cmp_traits_t::CMP_LHS_EQUAL_RHS;

        while( ( cmp_result == cmp_traits_t::CMP_LHS_EQUAL_RHS ) &&
               ( lhs_it != lhs_end ) )
        {
            cmp_result = SIXTRL_CXX_NAMESPACE::Type_value_comp_result<
                lhs_value_t, rhs_value_t >( *lhs_it, *rhs_it, rel_tol, abs_tol );
            ++lhs_it;
            ++rhs_it;
        }

        return cmp_result;
    }

    /* ===================================================================== */

    template< class LhsT, class RhsT = LhsT, typename CmpResultT =
                    typename ObjDataLogicTypeTraits< LhsT >::cmp_result_t,
              typename SFINAE_Enabled = void >
    struct ObjDataComparisonHelper
    {
        SIXTRL_STATIC SIXTRL_INLINE SIXTRL_FN CmpResultT compare(
            SIXTRL_ARGPTR_DEC const LhsT *const
                SIXTRL_RESTRICT SIXTRL_UNUSED( lhs ),
            SIXTRL_ARGPTR_DEC const RhsT *const
                SIXTRL_RESTRICT SIXTRL_UNUSED( rhs ) ) SIXTRL_NOEXCEPT
        {
            return TypeCompResultTraits< CmpResultT >::CMP_LHS_SMALLER_RHS;
        }

        template< typename... Args >
        SIXTRL_STATIC SIXTRL_INLINE SIXTRL_FN
        CmpResultT compare_with_tolerances(
            SIXTRL_ARGPTR_DEC const LhsT *const
                SIXTRL_RESTRICT SIXTRL_UNUSED( lhs ),
            SIXTRL_ARGPTR_DEC const RhsT *const
                SIXTRL_RESTRICT SIXTRL_UNUSED( rhs ),
            Args&&... SIXTRL_UNUSED( tolerances ) ) SIXTRL_NOEXCEPT
        {
            return TypeCompResultTraits< CmpResultT >::CMP_LHS_SMALLER_RHS;
        }
    };

    template< class LhsT, class RhsT, class DiffT,
              typename SFINAE_Enable = void >
    struct ObjDataDiffHelper
    {
        SIXTRL_STATIC SIXTRL_INLINE SIXTRL_FN arch_status_t calculate(
            SIXTRL_ARGPTR_DEC DiffT* SIXTRL_RESTRICT SIXTRL_UNUSED( diff ),
            SIXTRL_ARGPTR_DEC const LhsT *const SIXTRL_RESTRICT
                SIXTRL_UNUSED( lhs ),
            SIXTRL_ARGPTR_DEC const RhsT *const SIXTRL_RESTRICT
                SIXTRL_UNUSED( rhs ) ) SIXTRL_NOEXCEPT
        {
            return SIXTRL_CXX_NAMESPACE::ARCH_STATUS_GENERAL_FAILURE;
        }
    };

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    template< class LhsElemT, class RhsElemT >
    SIXTRL_STATIC SIXTRL_INLINE SIXTRL_FN
    typename ObjDataComparisonHelper< LhsElemT, RhsElemT >::cmp_result_t
    ObjData_compare( const LhsElemT *const SIXTRL_RESTRICT lhs,
                     const RhsElemT *const SIXTRL_RESTRICT rhs ) SIXTRL_NOEXCEPT
    {
        typedef ObjDataComparisonHelper< LhsElemT, RhsElemT > helper_t;
        return helper_t::compare( lhs, rhs );
    }

    template< class LhsElemT, class RhsElemT, typename... Args >
    SIXTRL_STATIC SIXTRL_INLINE SIXTRL_FN
    typename ObjDataComparisonHelper< LhsElemT, RhsElemT >::cmp_result_t
    ObjData_compare_with_tolerances(
        const LhsElemT *const SIXTRL_RESTRICT lhs,
        const RhsElemT *const SIXTRL_RESTRICT rhs,
        Args&&... tolerances ) SIXTRL_NOEXCEPT
    {
        typedef ObjDataComparisonHelper< LhsElemT, RhsElemT > helper_t;
        return helper_t::compare_with_tolerances(
            lhs, rhs, std::forward< Args >( tolerances )... );
    }
}
#endif /* C++ */

#if defined( __cplusplus ) && !defined( _GPUCODE )
extern "C" {
#endif /* C++, Host */

SIXTRL_STATIC SIXTRL_FN bool NS(Type_comp_less)(
    SIXTRL_REAL_T const lhs, SIXTRL_REAL_T const rhs ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN bool NS(Type_comp_less_or_equal)(
    SIXTRL_REAL_T const lhs, SIXTRL_REAL_T const rhs ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN bool NS(Type_comp_equal)(
    SIXTRL_REAL_T const lhs, SIXTRL_REAL_T const rhs ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN bool NS(Type_comp_not_equal)(
    SIXTRL_REAL_T const lhs, SIXTRL_REAL_T const rhs ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN bool NS(Type_comp_more_or_equal)(
    SIXTRL_REAL_T const lhs, SIXTRL_REAL_T const rhs ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN bool NS(Type_comp_more)(
    SIXTRL_REAL_T const lhs, SIXTRL_REAL_T const rhs ) SIXTRL_NOEXCEPT;

/* ------------------------------------------------------------------------- */

SIXTRL_STATIC SIXTRL_FN bool NS(Type_comp_all_less)(
    SIXTRL_REAL_T const lhs, SIXTRL_REAL_T const rhs ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN bool NS(Type_comp_all_less_or_equal)(
    SIXTRL_REAL_T const lhs, SIXTRL_REAL_T const rhs ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN bool NS(Type_comp_all_equal)(
    SIXTRL_REAL_T const lhs, SIXTRL_REAL_T const rhs ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN bool NS(Type_comp_all_not_equal)(
    SIXTRL_REAL_T const lhs, SIXTRL_REAL_T const rhs ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN bool NS(Type_comp_all_more_or_equal)(
    SIXTRL_REAL_T const lhs, SIXTRL_REAL_T const rhs ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN bool NS(Type_comp_all_more)(
    SIXTRL_REAL_T const lhs, SIXTRL_REAL_T const rhs ) SIXTRL_NOEXCEPT;

/* -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- - */

SIXTRL_STATIC SIXTRL_FN bool NS(Type_comp_any_less)(
    SIXTRL_REAL_T const lhs, SIXTRL_REAL_T const rhs ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN bool NS(Type_comp_any_less_or_equal)(
    SIXTRL_REAL_T const lhs, SIXTRL_REAL_T const rhs ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN bool NS(Type_comp_any_equal)(
    SIXTRL_REAL_T const lhs, SIXTRL_REAL_T const rhs ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN bool NS(Type_comp_any_not_equal)(
    SIXTRL_REAL_T const lhs, SIXTRL_REAL_T const rhs ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN bool NS(Type_comp_any_more_or_equal)(
    SIXTRL_REAL_T const lhs, SIXTRL_REAL_T const rhs ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN bool NS(Type_comp_any_more)(
    SIXTRL_REAL_T const lhs, SIXTRL_REAL_T const rhs ) SIXTRL_NOEXCEPT;

/* ------------------------------------------------------------------------- */

SIXTRL_STATIC SIXTRL_FN bool NS(Type_comp_is_close)(
    SIXTRL_REAL_T const lhs, SIXTRL_REAL_T const rhs,
    SIXTRL_REAL_T const rel_tol, SIXTRL_REAL_T const abs_tol ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN bool NS(Type_comp_all_are_close)(
    SIXTRL_REAL_T const lhs, SIXTRL_REAL_T const rhs,
    SIXTRL_REAL_T const rel_tol, SIXTRL_REAL_T const abs_tol ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN bool NS(Type_comp_any_are_close)(
    SIXTRL_REAL_T const lhs, SIXTRL_REAL_T const rhs,
    SIXTRL_REAL_T const rel_tol, SIXTRL_REAL_T const abs_tol ) SIXTRL_NOEXCEPT;

/* --------------------------------------------------------------------- */

SIXTRL_STATIC SIXTRL_FN int NS(Type_value_comp_result)(
    SIXTRL_REAL_T const lhs, SIXTRL_REAL_T const rhs ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN int NS(Type_value_comp_result_with_tolerances)(
    SIXTRL_REAL_T const lhs, SIXTRL_REAL_T const rhs,
    SIXTRL_REAL_T const rel_tol, SIXTRL_REAL_T const abs_tol ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN int NS(Type_value_comp_result_for_range)(
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT lhs_it,
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT lhs_end,
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT rhs_it
) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN int
NS(Type_value_comp_result_with_tolerances_for_range)(
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT lhs_it,
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT lhs_end,
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT rhs_it,
    SIXTRL_REAL_T const rel_tol, SIXTRL_REAL_T const abs_tol ) SIXTRL_NOEXCEPT;

/* --------------------------------------------------------------------- */

SIXTRL_STATIC SIXTRL_FN int NS(Type_value_comp_result_int64)(
    SIXTRL_INT64_T const lhs, SIXTRL_INT64_T const rhs ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN int NS(Type_value_comp_result_int64_for_range)(
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_INT64_T const* SIXTRL_RESTRICT lhs_it,
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_INT64_T const* SIXTRL_RESTRICT lhs_end,
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_INT64_T const* SIXTRL_RESTRICT rhs_it
) SIXTRL_NOEXCEPT;

#if defined( __cplusplus ) && !defined( _GPUCODE )
}
#endif /* C++, Host */

/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */
/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */

#if defined( __cplusplus ) && !defined( _GPUCODE )
extern "C" {
#endif /* C++, Host */

SIXTRL_INLINE bool NS(Type_comp_less)(
    SIXTRL_REAL_T const lhs, SIXTRL_REAL_T const rhs ) SIXTRL_NOEXCEPT
{
    return ( lhs < rhs );
}

SIXTRL_INLINE bool NS(Type_comp_less_or_equal)(
    SIXTRL_REAL_T const lhs, SIXTRL_REAL_T const rhs ) SIXTRL_NOEXCEPT
{
    return ( lhs <= rhs );
}

SIXTRL_INLINE bool NS(Type_comp_equal)(
    SIXTRL_REAL_T const lhs, SIXTRL_REAL_T const rhs ) SIXTRL_NOEXCEPT
{
    return ( lhs == rhs );
}

SIXTRL_INLINE bool NS(Type_comp_not_equal)(
    SIXTRL_REAL_T const lhs, SIXTRL_REAL_T const rhs ) SIXTRL_NOEXCEPT
{
    return ( lhs != rhs );
}

SIXTRL_INLINE bool NS(Type_comp_more_or_equal)(
    SIXTRL_REAL_T const lhs, SIXTRL_REAL_T const rhs ) SIXTRL_NOEXCEPT
{
    return ( lhs >= rhs );
}

SIXTRL_INLINE bool NS(Type_comp_more)(
    SIXTRL_REAL_T const lhs, SIXTRL_REAL_T const rhs ) SIXTRL_NOEXCEPT
{
    return ( lhs > rhs );
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE bool NS(Type_comp_all_less)(
    SIXTRL_REAL_T const lhs, SIXTRL_REAL_T const rhs ) SIXTRL_NOEXCEPT
{
    return ( lhs < rhs );
}

SIXTRL_INLINE bool NS(Type_comp_all_less_or_equal)(
    SIXTRL_REAL_T const lhs, SIXTRL_REAL_T const rhs ) SIXTRL_NOEXCEPT
{
    return ( lhs <= rhs );
}

SIXTRL_INLINE bool NS(Type_comp_all_equal)(
    SIXTRL_REAL_T const lhs, SIXTRL_REAL_T const rhs ) SIXTRL_NOEXCEPT
{
    return ( lhs == rhs );
}

SIXTRL_INLINE bool NS(Type_comp_all_not_equal)(
    SIXTRL_REAL_T const lhs, SIXTRL_REAL_T const rhs ) SIXTRL_NOEXCEPT
{
    return ( lhs != rhs );
}

SIXTRL_INLINE bool NS(Type_comp_all_more_or_equal)(
    SIXTRL_REAL_T const lhs, SIXTRL_REAL_T const rhs ) SIXTRL_NOEXCEPT
{
    return ( lhs >= rhs );
}

SIXTRL_INLINE bool NS(Type_comp_all_more)(
    SIXTRL_REAL_T const lhs, SIXTRL_REAL_T const rhs ) SIXTRL_NOEXCEPT
{
    return ( lhs > rhs );
}

/* -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- - */

SIXTRL_INLINE bool NS(Type_comp_any_less)(
    SIXTRL_REAL_T const lhs, SIXTRL_REAL_T const rhs ) SIXTRL_NOEXCEPT
{
    return ( lhs < rhs );
}

SIXTRL_INLINE bool NS(Type_comp_any_less_or_equal)(
    SIXTRL_REAL_T const lhs, SIXTRL_REAL_T const rhs ) SIXTRL_NOEXCEPT
{
    return ( lhs <= rhs );
}

SIXTRL_INLINE bool NS(Type_comp_any_equal)(
    SIXTRL_REAL_T const lhs, SIXTRL_REAL_T const rhs ) SIXTRL_NOEXCEPT
{
    return ( lhs == rhs );
}

SIXTRL_INLINE bool NS(Type_comp_any_not_equal)(
    SIXTRL_REAL_T const lhs, SIXTRL_REAL_T const rhs ) SIXTRL_NOEXCEPT
{
    return ( lhs != rhs );
}

SIXTRL_INLINE bool NS(Type_comp_any_more_or_equal)(
    SIXTRL_REAL_T const lhs, SIXTRL_REAL_T const rhs ) SIXTRL_NOEXCEPT
{
    return ( lhs >= rhs );
}

SIXTRL_INLINE bool NS(Type_comp_any_more)(
    SIXTRL_REAL_T const lhs, SIXTRL_REAL_T const rhs ) SIXTRL_NOEXCEPT
{
    return ( lhs > rhs );
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE bool NS(Type_comp_is_close)(
    SIXTRL_REAL_T const lhs, SIXTRL_REAL_T const rhs,
    SIXTRL_REAL_T const rel_tol, SIXTRL_REAL_T const abs_tol ) SIXTRL_NOEXCEPT
{
    SIXTRL_REAL_T const diff = ( lhs >= rhs ) ? lhs - rhs : rhs - lhs;
    SIXTRL_REAL_T cmp_value = ( rhs >= ( SIXTRL_REAL_T )0 ) ? rhs : -rhs;
    cmp_value *= rel_tol;
    cmp_value += abs_tol;

    return ( diff <= cmp_value );
}

SIXTRL_INLINE bool NS(Type_comp_all_are_close)(
    SIXTRL_REAL_T const lhs, SIXTRL_REAL_T const rhs,
    SIXTRL_REAL_T const rel_tol, SIXTRL_REAL_T const abs_tol ) SIXTRL_NOEXCEPT
{
    return NS(Type_comp_is_close)( lhs, rhs, rel_tol, abs_tol );
}

SIXTRL_INLINE bool NS(Type_comp_any_are_close)(
    SIXTRL_REAL_T const lhs, SIXTRL_REAL_T const rhs,
    SIXTRL_REAL_T const rel_tol, SIXTRL_REAL_T const abs_tol ) SIXTRL_NOEXCEPT
{
    return NS(Type_comp_is_close)( lhs, rhs, rel_tol, abs_tol );
}

/* --------------------------------------------------------------------- */

SIXTRL_INLINE int NS(Type_value_comp_result)(
    SIXTRL_REAL_T const lhs, SIXTRL_REAL_T const rhs ) SIXTRL_NOEXCEPT
{
    return ( NS(Type_comp_equal)( lhs, rhs ) )
        ? 0 : ( ( NS(Type_comp_more)( lhs, rhs ) ) ? +1 : -1 );
}

SIXTRL_INLINE int NS(Type_value_comp_result_with_tolerances)(
    SIXTRL_REAL_T const lhs, SIXTRL_REAL_T const rhs,
    SIXTRL_REAL_T const rel_tol, SIXTRL_REAL_T const abs_tol ) SIXTRL_NOEXCEPT
{
    return ( NS(Type_comp_is_close)( lhs, rhs, rel_tol, abs_tol ) )
        ? 0 : ( ( NS(Type_comp_more)( lhs, rhs ) ) ? +1 : -1 );
}

SIXTRL_INLINE int NS(Type_value_comp_result_for_range)(
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT lhs_it,
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT lhs_end,
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT rhs_it
) SIXTRL_NOEXCEPT
{
    int cmp_result = 0;

    SIXTRL_ASSERT( lhs_it != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( rhs_it != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( rhs_it != SIXTRL_NULLPTR );

    while( ( cmp_result == 0 ) && ( lhs_it != lhs_end ) )
    {
        cmp_result = NS(Type_value_comp_result)( *lhs_it++, *rhs_it++ );
    }

    return cmp_result;
}

SIXTRL_INLINE int NS(Type_value_comp_result_with_tolerances_for_range)(
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT lhs_it,
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT lhs_end,
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT rhs_it,
    SIXTRL_REAL_T const rel_tol, SIXTRL_REAL_T const abs_tol ) SIXTRL_NOEXCEPT
{
    int cmp_result = 0;

    SIXTRL_ASSERT( lhs_it != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( rhs_it != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( rhs_it != SIXTRL_NULLPTR );

    while( ( cmp_result == 0 ) && ( lhs_it != lhs_end ) )
    {
        cmp_result = NS(Type_value_comp_result_with_tolerances)(
            *lhs_it++, *rhs_it++, rel_tol, abs_tol );
    }

    return cmp_result;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE int NS(Type_value_comp_result_int64)(
    SIXTRL_INT64_T const lhs, SIXTRL_INT64_T const rhs ) SIXTRL_NOEXCEPT
{
    return ( lhs == rhs ) ? 0 : ( lhs > rhs ) ? +1 : -1;
}

SIXTRL_INLINE int NS(Type_value_comp_result_int64_for_range)(
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_INT64_T const* SIXTRL_RESTRICT lhs_it,
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_INT64_T const* SIXTRL_RESTRICT lhs_end,
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_INT64_T const* SIXTRL_RESTRICT rhs_it
) SIXTRL_NOEXCEPT
{
    int cmp_result = 0;

    SIXTRL_ASSERT( lhs_it != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( rhs_it != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( rhs_it != SIXTRL_NULLPTR );

    while( ( cmp_result == 0 ) && ( lhs_it != lhs_end ) )
    {
        cmp_result = NS(Type_value_comp_result_int64)( *lhs_it++, *rhs_it++ );
    }

    return cmp_result;
}

#if defined( __cplusplus ) && !defined( _GPUCODE )
}
#endif /* C++, Host */

#endif /* SIXTRACKLIB_COMMON_INTERNAL_TYPE_COMPARISON_HELPERS_CXX_HPP__ */
