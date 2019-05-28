#ifndef SIXTRACKLIB_COMMON_BE_LIMIT_CXX_HPP__
#define SIXTRACKLIB_COMMON_BE_LIMIT_CXX_HPP__


#if defined( __cplusplus )

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <cstddef>
    #include <cstdlib>
    #include <type_traits>
    #include <utility>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/buffer.hpp"
    #include "sixtracklib/common/be_limit/be_limit.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

namespace SIXTRL_CXX_NAMESPACE
{
    template< typename T >
    struct TLimit
    {
        using value_type = T;
        using type_id_t  = ::NS(object_type_id_t);
        using size_type  = ::NS(buffer_size_t);
        using buffer_t   = Buffer;
        using c_buffer_t = buffer_t::c_api_t;

        SIXTRL_FN explicit TLimit(
            T const& SIXTRL_RESTRICT_REF x_limit =
                T{ SIXTRL_BE_LIMIT_DEFAULT_X_LIMIT },
            T const& SIXTRL_RESTRICT_REF y_limit =
                T{ SIXTRL_BE_LIMIT_DEFAULT_Y_LIMIT } );

        SIXTRL_FN TLimit( TLimit< T > const& other ) = default;
        TLimit( TLimit< T >&& other ) = default;

        SIXTRL_FN TLimit< T >& operator=( TLimit< T > const& other ) = default;
        SIXTRL_FN TLimit< T >& operator=( TLimit< T >&& other ) = default;

        SIXTRL_FN ~TLimit() = default;

        /* ---------------------------------------------------------------- */

        SIXTRL_STATIC SIXTRL_FN bool CanAddToBuffer(
            buffer_t& SIXTRL_RESTRICT_REF buffer,
            SIXTRL_ARGPTR_DEC size_type* SIXTRL_RESTRICT
                ptr_requ_objects  = nullptr,
            SIXTRL_ARGPTR_DEC size_type* SIXTRL_RESTRICT
                ptr_requ_slots    = nullptr,
            SIXTRL_ARGPTR_DEC size_type* SIXTRL_RESTRICT
                ptr_requ_dataptrs = nullptr ) SIXTRL_NOEXCEPT;

        SIXTRL_STATIC SIXTRL_FN SIXTRL_ARGPTR_DEC TXYShift< T >*
        CreateNewOnBuffer( buffer_t& SIXTRL_RESTRICT_REF buffer );

        SIXTRL_STATIC SIXTRL_FN SIXTRL_ARGPTR_DEC TXYShift< T >* AddToBuffer(
            buffer_t& SIXTRL_RESTRICT_REF buffer,
            value_type const& dx, value_type const& dy );

        /* ---------------------------------------------------------------- */

        SIXTRL_FN type_id_t getTypeId() const SIXTRL_NOEXCEPT;

        SIXTRL_STATIC SIXTRL_FN size_type getNumDataPtrs(
            buffer_t const& SIXTRL_RESTRICT_REF buffer ) const SIXTRL_NOEXCEPT;

        SIXTRL_FN value_type const& getXLimit() const SIXTRL_NOEXCEPT;
        SIXTRL_FN value_type const& getYLimit() const SIXTRL_NOEXCEPT;

        SIXTRL_FN void preset() SIXTRL_NOEXCEPT;

        SIXTRL_FN void setXLimit(
            value_type const& SIXTRL_RESTRICT_REF x_limit ) SIXTRL_NOEXCEPT;

        SIXTRL_FN void setYLimit(
            value_type const& SIXTRL_RESTRICT_REF y_limit ) SIXTRL_NOEXCEPT;

        /* ----------------------------------------------------------------- */

        value_type x_limit SIXTRL_ALIGN( 8 );
        value_type y_limit SIXTRL_ALIGN( 8 );
    };

    template< typename T > struct ObjectTypeTraits<
        SIXTRL_CXX_NAMESPACE::TLimit< T > >
    {
        SIXTRL_STATIC SIXTRL_INLINE object_type_id_t Type() SIXTRL_NOEXCEPT
        {
            return ::NS(OBJECT_TYPE_LIMIT);
        }
    };

    template< typename T >
    SIXTRL_ARGPTR_DEC TLimit< T >* TLimit_new( Buffer& buffer );

    template< typename T >
    SIXTRL_ARGPTR_DEC TLimit< T >* TLimit_new(
        SIXTRL_ARGPTR_DEC ::NS(Buffer)* SIXTRL_RESTRICT ptr_buffer );

    template< typename T >
    SIXTRL_ARGPTR_DEC TLimit< T >* TLimit_add( Buffer& buffer,
        typename TLimit< T >::value_type const& x_limits,
        typename TLimit< T >::value_type const& y_limits );

    template< typename T >
    SIXTRL_ARGPTR_DEC TLimit< T >* TLimit_add(
        SIXTRL_ARGPTR_DEC ::NS(Buffer)* SIXTRL_RESTRICT ptr_buffer,
        typename TLimit< T >::value_type const& SIXTRL_RESTRICT_REF x_limits,
        typename TLimit< T >::value_type const& SIXTRL_RESTRICT_REF y_limits );

    /* ===================================================================== *
     * ====  Specialization TLimit< NS(particle_real_t) > :
     * ===================================================================== */

    template<> struct TLimit< NS(particle_real_t) > : public ::NS(Limit)
    {
        using value_type = ::NS(particle_real_t);
        using type_id_t  = ::NS(object_type_id_t);
        using size_type  = ::NS(buffer_size_t);
        using buffer_t   = Buffer;
        using c_buffer_t = buffer_t::c_api_t;
        using c_api_t    = ::NS(Limit);

        SIXTRL_FN explicit TLimit(
            value_type const SIXTRL_RESTRICT_REF x_limit =
                T{ SIXTRL_BE_LIMIT_DEFAULT_X_LIMIT },
            value_type const SIXTRL_RESTRICT_REF y_limit =
                T{ SIXTRL_BE_LIMIT_DEFAULT_Y_LIMIT } );

        SIXTRL_FN TLimit( TLimit< value_type > const& other ) = default;
        TLimit( TLimit< value_type >&& other ) = default;

        SIXTRL_FN TLimit< value_type >& operator=(
            TLimit< value_type > const& other ) = default;

        SIXTRL_FN TLimit< value_type >& operator=(
            TLimit< value_type >&& other ) = default;s

        SIXTRL_FN ~TLimit() = default;

        /* ---------------------------------------------------------------- */

        SIXTRL_STATIC SIXTRL_FN bool CanAddToBuffer(
            buffer_t& SIXTRL_RESTRICT_REF buffer,
            SIXTRL_ARGPTR_DEC size_type* SIXTRL_RESTRICT
                ptr_requ_objects  = nullptr,
            SIXTRL_ARGPTR_DEC size_type* SIXTRL_RESTRICT
                ptr_requ_slots    = nullptr,
            SIXTRL_ARGPTR_DEC size_type* SIXTRL_RESTRICT
                ptr_requ_dataptrs = nullptr );

        SIXTRL_STATIC SIXTRL_FN
        SIXTRL_ARGPTR_DEC TLimit< NS(particle_real_t) >*
        CreateNewOnBuffer( buffer_t& SIXTRL_RESTRICT_REF buffer );

        SIXTRL_STATIC SIXTRL_FN
        SIXTRL_ARGPTR_DEC TLimit< NS(particle_real_t) >* AddToBuffer(
            buffer_t& SIXTRL_RESTRICT_REF buffer,
            value_type const x_limit, value_type const y_limit )
        {
            return reinterpret_cast< RetPtr >(
                ::NS(Limit_add)( buffer.getCApiPtr(), x_limit, y_limit ) );
        }

        /* ----------------------------------------------------------------- */

        SIXTRL_ARGPTR_DEC c_api_t const* getCApiPtr() const SIXTRL_NOEXCEPT;
        SIXTRL_ARGPTR_DEC c_api_t* getCApiPtr() SIXTRL_NOEXCEPT;

        /* ----------------------------------------------------------------- */

        template< typename T > struct ObjectTypeTraits<
        SIXTRL_CXX_NAMESPACE::TLimit< T > >
        {
            SIXTRL_STATIC SIXTRL_INLINE object_type_id_t Type() SIXTRL_NOEXCEPT
            {
                return ::NS(OBJECT_TYPE_LIMIT);
            }
        };

        SIXTRL_FN type_id_t getTypeId() const SIXTRL_NOEXCEPT;
        SIXTRL_FN value_type getDx() const SIXTRL_NOEXCEPT;
        SIXTRL_FN value_type getDx() const SIXTRL_NOEXCEPT;

        SIXTRL_FN void preset() SIXTRL_NOEXCEPT;
        SIXTRL_FN void setXLimit( value_type const x_limit ) SIXTRL_NOEXCEPT;
        SIXTRL_FN void setyLimit( value_type const y_limit ) SIXTRL_NOEXCEPT;

    };

    using Limit = TLimit< NS(particle_real_t) >;

    SIXTRL_ARGPTR_DEC Limit* Limit_new( Buffer& buffer );

    SIXTRL_ARGPTR_DEC Limit* Limit_new(
        SIXTRL_ARGPTR_DEC ::NS(Buffer)* SIXTRL_RESTRICT ptr_buffer );

    SIXTRL_ARGPTR_DEC Limit* Limit_add( Buffer& buffer,
        NS(particle_real_t) const dx, NS(particle_real_t) const dy );

    SIXTRL_ARGPTR_DEC Limit* Limit_add(
        SIXTRL_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT ptr_buffer,
        NS(particle_real_t) const x_limit, NS(particle_real_t) const y_limit );
}



#endif /* SIXTRACKLIB_COMMON_BE_LIMIT_CXX_HPP__ */

/* end: sixtracklib/common/be_limit/be_limit.hpp */
