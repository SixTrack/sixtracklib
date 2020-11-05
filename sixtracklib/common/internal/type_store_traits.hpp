#ifndef SIXTRACKLIB_COMMON_INTERNAL_TYPE_STORE_TRAITS_CXX_HPP__
#define SIXTRACKLIB_COMMON_INTERNAL_TYPE_STORE_TRAITS_CXX_HPP__

#if defined( __cplusplus )

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <cstdlib>
    #include <limits>
    #include <stdint.h>
    #include <type_traits>

    #if !defined( _GPUCODE )
        #include <iterator>
        #include <string>
    #endif /* !defined( _GPUCODE ) */
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/control/definitions.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

namespace SIXTRL_CXX_NAMESPACE
{
    template< typename T >
    struct TypeStoreTraits
    {
        typedef T           value_type;
        typedef T const&    const_reference;
        typedef T const*    const_pointer;
        typedef T&          reference;
        typedef T*          pointer;

        static constexpr bool is_signed = std::numeric_limits< T >::is_signed;
    };

    template< typename T > constexpr bool TypeStoreTraits< T >::is_signed;

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    template< typename T >
    struct TypeMethodParamTraits
    {
        typedef T           value_type;
        typedef T const&    argument_type;
        typedef T const&    const_argument_type;
        typedef T&          ref_argument_type;
        typedef T const&    const_existing_type;
        typedef T&          existing_type;
        typedef T&          ref_existing_type;
        typedef T const*    const_pointer;
        typedef T*          pointer;
        typedef T&&         rvalue_ref;
    };

    /* ********************************************************************* */

    template< typename T >
    struct TypeDimensionalityTraits
    {
        static constexpr arch_size_t min_dimension = arch_size_t{ 1 };
        static constexpr arch_size_t max_dimension = arch_size_t{ 1 };
    };

    template< typename T >
    static SIXTRL_FN constexpr bool
    Type_consistent_dimensionality() SIXTRL_NOEXCEPT
    {
        return ( ( TypeDimensionalityTraits< T >::min_dimension >
                   arch_size_t{ 0 } ) &&
                 ( TypeDimensionalityTraits< T >::min_dimension <=
                   TypeDimensionalityTraits< T >::max_dimension ) );
    }

    template< typename T >
    static SIXTRL_FN constexpr bool
    Type_has_fixed_dimensionality() SIXTRL_NOEXCEPT
    {
        return (
            ( SIXTRL_CXX_NAMESPACE::Type_consistent_dimensionality< T >() ) &&
            ( TypeDimensionalityTraits< T >::min_dimension ==
              TypeDimensionalityTraits< T >::max_dimension ) );
    }

    template< typename T >
    static SIXTRL_FN constexpr bool
    Type_has_variable_dimensionality() SIXTRL_NOEXCEPT
    {
        return (
            ( SIXTRL_CXX_NAMESPACE::Type_consistent_dimensionality< T >() ) &&
            ( TypeDimensionalityTraits< T >::min_dimension !=
              TypeDimensionalityTraits< T >::max_dimension ) );
    }

    template< typename T >
    static SIXTRL_FN constexpr arch_size_t Type_min_dimension() SIXTRL_NOEXCEPT
    {
        return TypeDimensionalityTraits< T >::min_dimension;
    }

    template< typename T >
    static SIXTRL_FN constexpr arch_size_t Type_max_dimension() SIXTRL_NOEXCEPT
    {
        return TypeDimensionalityTraits< T >::max_dimension;
    }

    template< typename T >
    static SIXTRL_FN constexpr bool Type_is_scalar() SIXTRL_NOEXCEPT
    {
        return (
            ( SIXTRL_CXX_NAMESPACE::Type_consistent_dimensionality< T >() ) &&
            ( SIXTRL_CXX_NAMESPACE::Type_has_fixed_dimensionality< T >() ) &&
            ( TypeDimensionalityTraits< T >::min_dimension ==
              arch_size_t{ 1 } ) );
    }

    template< typename T >
    static SIXTRL_FN constexpr bool Type_is_vector_valued() SIXTRL_NOEXCEPT
    {
        return (
            ( SIXTRL_CXX_NAMESPACE::Type_consistent_dimensionality<T>() ) &&
            ( ( SIXTRL_CXX_NAMESPACE::Type_has_variable_dimensionality<T>() ) ||
              ( TypeDimensionalityTraits< T >::max_dimension >
                arch_size_t{ 1 } ) ) );
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    template< typename T >
    struct TypeDimensionalityHelper
    {
        static arch_size_t dimension( typename TypeMethodParamTraits< T
            >::const_pointer SIXTRL_RESTRICT /* ptr */ ) SIXTRL_NOEXCEPT
        {
            return
                ( SIXTRL_CXX_NAMESPACE::Type_consistent_dimensionality< T >() )
                    ? SIXTRL_CXX_NAMESPACE::Type_max_dimension< T >()
                    : arch_size_t{ 0 };
        }
    };

    template< typename T >
    static SIXTRL_FN typename std::enable_if<
        !SIXTRL_CXX_NAMESPACE::Type_consistent_dimensionality< T >(),
        arch_size_t >::type
    Type_dimensionality( typename TypeMethodParamTraits< T >::const_pointer
            /* ptr */ = nullptr ) SIXTRL_NOEXCEPT
    {
        static_assert(
            SIXTRL_CXX_NAMESPACE::Type_consistent_dimensionality< T >(),
            "Inconsistent dimensionality traits for type T" );

        return arch_size_t{ 0 };
    }

    template< typename T >
    static SIXTRL_FN constexpr typename std::enable_if<
        SIXTRL_CXX_NAMESPACE::Type_consistent_dimensionality< T >() &&
        SIXTRL_CXX_NAMESPACE::Type_has_fixed_dimensionality< T >(),
        arch_size_t >::type
    Type_dimensionality( typename TypeMethodParamTraits< T >::const_pointer
            /* ptr */ = nullptr ) SIXTRL_NOEXCEPT
    {
        return TypeDimensionalityTraits< T >::min_dimension;
    }

    template< typename T >
    static SIXTRL_FN constexpr typename std::enable_if<
        SIXTRL_CXX_NAMESPACE::Type_consistent_dimensionality< T >() &&
        SIXTRL_CXX_NAMESPACE::Type_has_variable_dimensionality< T >(),
        arch_size_t >::type
    Type_dimensionality( typename TypeMethodParamTraits< T >::const_pointer
            ptr = nullptr ) SIXTRL_NOEXCEPT
    {
        return TypeDimensionalityHelper< T >::dimension( ptr );
    }

    /* ********************************************************************* */

    template< typename ArgT, typename BaseType >
    static SIXTRL_FN constexpr bool Type_arg_type_is_base_type() SIXTRL_NOEXCEPT
    {
        return std::is_same< typename std::decay< ArgT >::type, BaseType >();
    }

    /* ********************************************************************* */

    template< typename T >
    struct TypeStorageAlignTraits
    {
        static SIXTRL_FN constexpr SIXTRL_CXX_NAMESPACE::arch_size_t
        Alignment() SIXTRL_NOEXCEPT
        {
            #if !defined( SIXTRL_ALIGN_NUM )
                #define SIXTRL_ALIGN_NUM_UNDEF
                #define SIXTRL_ALIGN_NUM 8
            #endif /* SIXTRL_ALIGN_NUM  */

            return ( alignof( T ) <= SIXTRL_ALIGN_NUM )
                ? static_cast< SIXTRL_CXX_NAMESPACE::arch_size_t >(
                    SIXTRL_ALIGN_NUM )
                : static_cast< SIXTRL_CXX_NAMESPACE::arch_size_t >(
                    ( ( alignof( T ) / SIXTRL_ALIGN_NUM ) +
                      ( ( ( alignof( T ) % SIXTRL_ALIGN_NUM ) == 0u ) ? 0 : 1 )
                    ) * SIXTRL_ALIGN_NUM );

            #if defined( SIXTRL_ALIGN_NUM_UNDEF )
                #undef SIXTRL_ALIGN_NUM
                #undef SIXTRL_ALIGN_NUM_UNDEF
            #endif  /* defined( SIXTRL_ALIGN_NUM_UNDEF ) */
        }
    };

    /* --------------------------------------------------------------------- */
    /* Convenience functions: */

    template< typename T >
    static SIXTRL_FN constexpr
    SIXTRL_CXX_NAMESPACE::arch_size_t Type_storage_align() SIXTRL_NOEXCEPT
    {
        return SIXTRL_CXX_NAMESPACE::TypeStorageAlignTraits< T >::Alignment();
    }

    /* ====================================================================== */

    template< class Source, class Dest >
    static SIXTRL_FN constexpr bool Types_are_assignable() SIXTRL_NOEXCEPT
    {
         return std::is_assignable<
                 typename std::add_lvalue_reference<
                     typename std::decay< Dest >::type >::type,
                 typename std::decay< Source >::type >::value;
    }

    template< class Source, class Dest >
    static SIXTRL_FN constexpr bool Types_are_convertible() SIXTRL_NOEXCEPT
    {
         return std::is_convertible<
             typename std::decay< Source >::type,
             typename std::decay< Dest >::type >::value;
    }

    template< class Source, class Dest, typename _Enabled = void >
    struct TypePerformAssignmentHelper
    {
        static SIXTRL_FN constexpr bool can_assign() SIXTRL_NOEXCEPT
        {
            return false;
        }

        static SIXTRL_FN constexpr arch_status_t assign(
            SIXTRL_ARGPTR_DEC typename TypeMethodParamTraits<
                Dest >::ref_argument_type /*destination*/,
            SIXTRL_ARGPTR_DEC typename TypeMethodParamTraits<
                Source >::const_argument_type /*source*/ ) SIXTRL_NOEXCEPT
        {
            return SIXTRL_CXX_NAMESPACE::ARCH_STATUS_GENERAL_FAILURE;
        }
    };

    template< class Source, class Dest >
    struct TypePerformAssignmentHelper< Source, Dest, typename std::enable_if<
       SIXTRL_CXX_NAMESPACE::Types_are_assignable< Source, Dest >() >::type >
    {
        static SIXTRL_FN constexpr bool can_assign() SIXTRL_NOEXCEPT
        {
            return true;
        }

        static SIXTRL_FN arch_status_t assign(
            SIXTRL_ARGPTR_DEC typename TypeMethodParamTraits<
                Dest >::ref_argument_type destination,
            SIXTRL_ARGPTR_DEC typename TypeMethodParamTraits<
                Source >::const_argument_type source ) SIXTRL_NOEXCEPT
        {
            destination = source;
            return SIXTRL_CXX_NAMESPACE::ARCH_STATUS_SUCCESS;
        }
    };

    template< class Source, class Dest >
    struct TypePerformAssignmentHelper< Source, Dest, typename std::enable_if<
       !SIXTRL_CXX_NAMESPACE::Types_are_assignable< Source, Dest >() &&
        SIXTRL_CXX_NAMESPACE::Types_are_convertible< Source, Dest >() >::type >
    {
        static SIXTRL_FN constexpr bool can_assign() SIXTRL_NOEXCEPT
        {
            return true;
        }

        static SIXTRL_FN arch_status_t assign(
            SIXTRL_ARGPTR_DEC typename TypeMethodParamTraits<
                Dest >::ref_argument_type destination,
            SIXTRL_ARGPTR_DEC typename TypeMethodParamTraits<
                Source >::const_argument_type source ) SIXTRL_NOEXCEPT
        {
            destination = source;
            return SIXTRL_CXX_NAMESPACE::ARCH_STATUS_SUCCESS;
        }
    };

    /* ---------------------------------------------------------------------- */

    template< class Source, class Dest >
    static SIXTRL_FN constexpr bool
        Types_can_perform_assignment() SIXTRL_NOEXCEPT
    {
        return TypePerformAssignmentHelper< Source, Dest >::can_assign();
    }

    template< class Source, class Dest >
    static SIXTRL_INLINE SIXTRL_FN arch_status_t Types_perform_assignment(
        SIXTRL_ARGPTR_DEC typename TypeMethodParamTraits< Dest
            >::ref_argument_type destination,
        SIXTRL_ARGPTR_DEC typename TypeMethodParamTraits< Source
            >::const_argument_type source )
    {
        return TypePerformAssignmentHelper< Source, Dest >::assign(
            destination, source );
    }

    #if !defined( __CUDACC__ )
    template< class SrcIter, class DstIter >
    static SIXTRL_FN arch_status_t Types_perform_assignment_for_range(
        SrcIter src_it, SrcIter src_end, DstIter dest_it )
    {
        typedef typename std::iterator_traits<
            SrcIter >::value_type src_value_t;

        typedef typename std::iterator_traits<
            DstIter >::value_type dest_value_t;
    #else /* Cuda */
    template< class T >
    static SIXTRL_FN arch_status_t Types_perform_assignment_for_range(
        T const* SIXTRL_RESTRICT src_it, T const* SIXTRL_RESTRICT src_end,
        T* SIXTRL_RESTRICT dest_it )
    {
        typedef T src_value_t;
        typedef T dest_value_t;
    #endif /* Cuda */

        arch_status_t success = ( src_it != src_end )
            ? SIXTRL_CXX_NAMESPACE::ARCH_STATUS_SUCCESS
            : SIXTRL_CXX_NAMESPACE::ARCH_STATUS_GENERAL_FAILURE;

        for( ; src_it != src_end ; ++src_it, ++dest_it )
        {
            success = TypePerformAssignmentHelper<
                src_value_t, dest_value_t >::assign( *dest_it, *src_it );

            if( success != SIXTRL_CXX_NAMESPACE::ARCH_STATUS_SUCCESS )
            {
                break;
            }
        }

        return success;
    }

    template< class SrcIter, class DstIter >
    static SIXTRL_FN arch_status_t Types_perform_assignment_for_range(
        SrcIter src_begin, DstIter dest_begin,
        arch_size_t const num_items_to_assign )
    {
        SrcIter src_end = src_begin;
        #if !defined( __CUDACC__ )
        std::advance( src_end, num_items_to_assign );
        #else /* !Cuda */
        src_end = src_end + num_items_to_assign;
        #endif /* Cuda */

        return SIXTRL_CXX_NAMESPACE::Types_perform_assignment_for_range(
            src_begin, src_end, dest_begin );
    }

    /* ********************************************************************* */

    template< class SrcT, typename DstT >
    SIXTRL_STATIC SIXTRL_FN constexpr bool
    Types_same_and_direct_assignable() SIXTRL_NOEXCEPT
    {
        return  std::is_same< SrcT, typename TypeMethodParamTraits< typename
                    std::decay< DstT >::type >::value_type >::value ||
                std::is_same< SrcT, typename TypeMethodParamTraits< typename
                    std::decay< DstT >::type >::const_argument_type >::value ||
                std::is_same< SrcT, typename TypeMethodParamTraits< typename
                    std::decay< DstT >::type >::ref_argument_type >::value;
    }

    template< class SrcT, typename DstT >
    SIXTRL_STATIC SIXTRL_FN constexpr bool
    Types_same_and_move_assignable() SIXTRL_NOEXCEPT
    {
        return !SIXTRL_CXX_NAMESPACE::Types_same_and_direct_assignable<
                    SrcT, DstT >() &&
               std::is_same< SrcT, typename TypeMethodParamTraits< typename
                   std::decay< DstT >::type >::rvalue_ref >::value;
    }

    template< class SrcT, typename DstT >
    SIXTRL_STATIC SIXTRL_FN constexpr bool
    Types_same_allow_arithmetic() SIXTRL_NOEXCEPT
    {
        return  SIXTRL_CXX_NAMESPACE::Types_same_and_direct_assignable<
                    SrcT, DstT >() ||
                std::is_same< SrcT, typename TypeMethodParamTraits< typename
                   std::decay< DstT >::type >::rvalue_ref >::value;
    }

    /* ********************************************************************* */
    /* Specializations: */
    /* ********************************************************************* */

    /* --------------------------------------------------------------------- */
    /* double: */

    template<> struct TypeMethodParamTraits< double >
    {
        typedef double                                  value_type;
        typedef value_type const                        const_argument_type;
        typedef value_type                              argument_type;
        typedef value_type& SIXTRL_RESTRICT_REF         ref_argument_type;
        typedef value_type                              const_existing_type;
        typedef value_type                              existing_type;
        typedef value_type& SIXTRL_RESTRICT_REF         ref_existing_type;
        typedef value_type const*                       const_pointer;
        typedef value_type*                             pointer;
        typedef value_type&&                            rvalue_ref;
    };

    /* --------------------------------------------------------------------- */
    /* float: */

    template<> struct TypeMethodParamTraits< float >
    {
        typedef float                                   value_type;
        typedef value_type const                        const_argument_type;
        typedef value_type                              argument_type;
        typedef value_type& SIXTRL_RESTRICT_REF         ref_argument_type;
        typedef value_type                              const_existing_type;
        typedef value_type                              existing_type;
        typedef value_type& SIXTRL_RESTRICT_REF         ref_existing_type;
        typedef value_type const*                       const_pointer;
        typedef value_type*                             pointer;
        typedef value_type&&                            rvalue_ref;
    };

    /* --------------------------------------------------------------------- */
    /* uint64: */

    template<> struct TypeMethodParamTraits< uint64_t >
    {
        typedef uint64_t                                value_type;
        typedef value_type const                        const_argument_type;
        typedef value_type                              argument_type;
        typedef value_type& SIXTRL_RESTRICT_REF         ref_argument_type;
        typedef value_type                              const_existing_type;
        typedef value_type                              existing_type;
        typedef value_type& SIXTRL_RESTRICT_REF         ref_existing_type;
        typedef value_type const*                       const_pointer;
        typedef value_type*                             pointer;
        typedef value_type&&                            rvalue_ref;
    };

    /* --------------------------------------------------------------------- */
    /* int64: */

    template<> struct TypeMethodParamTraits< int64_t >
    {
        typedef int64_t                                 value_type;
        typedef value_type const                        const_argument_type;
        typedef value_type                              argument_type;
        typedef value_type& SIXTRL_RESTRICT_REF         ref_argument_type;
        typedef value_type                              const_existing_type;
        typedef value_type                              existing_type;
        typedef value_type& SIXTRL_RESTRICT_REF         ref_existing_type;
        typedef value_type const*                       const_pointer;
        typedef value_type*                             pointer;
        typedef value_type&&                            rvalue_ref;
    };

    /* --------------------------------------------------------------------- */
    /* uint32: */

    template<> struct TypeMethodParamTraits< uint32_t >
    {
        typedef uint32_t                                value_type;
        typedef value_type const                        const_argument_type;
        typedef value_type                              argument_type;
        typedef value_type& SIXTRL_RESTRICT_REF         ref_argument_type;
        typedef value_type                              const_existing_type;
        typedef value_type                              existing_type;
        typedef value_type& SIXTRL_RESTRICT_REF         ref_existing_type;
        typedef value_type const*                       const_pointer;
        typedef value_type*                             pointer;
        typedef value_type&&                            rvalue_ref;
    };

    /* --------------------------------------------------------------------- */
    /* int32: */

    template<> struct TypeMethodParamTraits< int32_t >
    {
        typedef int32_t                                 value_type;
        typedef value_type const                        const_argument_type;
        typedef value_type                              argument_type;
        typedef value_type& SIXTRL_RESTRICT_REF         ref_argument_type;
        typedef value_type                              const_existing_type;
        typedef value_type                              existing_type;
        typedef value_type& SIXTRL_RESTRICT_REF         ref_existing_type;
        typedef value_type const*                       const_pointer;
        typedef value_type*                             pointer;
        typedef value_type&&                            rvalue_ref;
    };

    /* --------------------------------------------------------------------- */
    /* uint16_t: */

    template<> struct TypeMethodParamTraits< uint16_t >
    {
        typedef uint16_t                                value_type;
        typedef value_type const                        const_argument_type;
        typedef value_type                              argument_type;
        typedef value_type& SIXTRL_RESTRICT_REF         ref_argument_type;
        typedef value_type                              const_existing_type;
        typedef value_type                              existing_type;
        typedef value_type& SIXTRL_RESTRICT_REF         ref_existing_type;
        typedef value_type const*                       const_pointer;
        typedef value_type*                             pointer;
        typedef value_type&&                            rvalue_ref;
    };

    /* --------------------------------------------------------------------- */
    /* int16_t: */

    template<> struct TypeMethodParamTraits< int16_t >
    {
        typedef int16_t                                 value_type;
        typedef value_type const                        const_argument_type;
        typedef value_type                              argument_type;
        typedef value_type& SIXTRL_RESTRICT_REF         ref_argument_type;
        typedef value_type                              const_existing_type;
        typedef value_type                              existing_type;
        typedef value_type& SIXTRL_RESTRICT_REF         ref_existing_type;
        typedef value_type const*                       const_pointer;
        typedef value_type*                             pointer;
        typedef value_type&&                            rvalue_ref;
    };

    /* --------------------------------------------------------------------- */
    /* uint8_t: */

    template<> struct TypeMethodParamTraits< uint8_t >
    {
        typedef uint8_t                                 value_type;
        typedef value_type const                        const_argument_type;
        typedef value_type                              argument_type;
        typedef value_type&                             ref_argument_type;
        typedef value_type                              const_existing_type;
        typedef value_type                              existing_type;
        typedef value_type& SIXTRL_RESTRICT_REF         ref_existing_type;
        typedef value_type const*                       const_pointer;
        typedef value_type*                             pointer;
        typedef value_type&&                            rvalue_ref;
    };

    /* --------------------------------------------------------------------- */
    /* int8_t: */

    template<> struct TypeMethodParamTraits< int8_t >
    {
        typedef int8_t                                  value_type;
        typedef value_type const                        const_argument_type;
        typedef value_type                              argument_type;
        typedef value_type& SIXTRL_RESTRICT_REF         ref_argument_type;
        typedef value_type                              const_existing_type;
        typedef value_type                              existing_type;
        typedef value_type& SIXTRL_RESTRICT_REF         ref_existing_type;
        typedef value_type const*                       const_pointer;
        typedef value_type*                             pointer;
        typedef value_type&&                            rvalue_ref;
    };

    /* --------------------------------------------------------------------- */
    /* bool: */

    template<> struct TypeMethodParamTraits< bool >
    {
        typedef bool                                    value_type;
        typedef value_type const                        const_argument_type;
        typedef value_type                              argument_type;
        typedef value_type& SIXTRL_RESTRICT_REF         ref_argument_type;
        typedef value_type                              const_existing_type;
        typedef value_type                              existing_type;
        typedef value_type& SIXTRL_RESTRICT_REF         ref_existing_type;
        typedef value_type const*                       const_pointer;
        typedef value_type*                             pointer;
        typedef value_type&&                            rvalue_ref;
    };

    #if !defined( _GPUCODE )

    /* --------------------------------------------------------------------- */
    /* std::string: */

    template<> struct TypeMethodParamTraits< std::string >
    {
        typedef std::string                             value_type;
        typedef value_type  const& SIXTRL_RESTRICT_REF  const_argument_type;
        typedef value_type& SIXTRL_RESTRICT_REF         argument_type;
        typedef value_type& SIXTRL_RESTRICT_REF         ref_argument_type;
        typedef value_type const& SIXTRL_RESTRICT_REF   const_existing_type;
        typedef value_type& SIXTRL_RESTRICT_REF         existing_type;
        typedef value_type& SIXTRL_RESTRICT_REF         ref_existing_type;
        typedef value_type const*                       const_pointer;
        typedef value_type*                             pointer;
        typedef value_type&&                            rvalue_ref;
    };

    #endif /* !defined( _GPUCODE ) */
}

#endif /* defined( __cplusplus ) */

#endif /* SIXTRACKLIB_COMMON_INTERNAL_TYPE_STORE_TRAITS_CXX_HPP__ */
