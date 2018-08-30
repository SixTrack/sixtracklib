#ifndef CXX_SIXTRACKLIB_COMMON_IMPL_BE_DRIFT_HPP__
#define CXX_SIXTRACKLIB_COMMON_IMPL_BE_DRIFT_HPP__

#if defined( __cplusplus )

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <cstddef>
    #include <cstdint>
    #include <cstdlib>
    #include <type_traits>
    #include <utility>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/buffer.hpp"
    #include "sixtracklib/common/impl/be_drift.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */


namespace SIXTRL_NAMESPACE
{
    /* ===================================================================== *
     * ====  TDrift< T > :
     * ===================================================================== */

    template< typename T >
    struct TDrift
    {
        using value_type = T;
        using type_id_t  = NS(object_type_id_t);
        using size_type  = NS(buffer_size_t);
        using buffer_t   = ::NS(Buffer);

        SIXTRL_FN TDrift() = default;
        SIXTRL_FN TDrift( TDrift< T > const& other ) = default;
        SIXTRL_FN TDrift( TDrift< T >&& other ) = default;

        SIXTRL_FN TDrift< T >& operator=( TDrift< T > const& rhs ) = default;
        SIXTRL_FN TDrift< T >& operator=( TDrift< T >&& rhs ) = default;
        SIXTRL_FN ~TDrift() = default;

        /* ----------------------------------------------------------------- */

        SIXTRL_FN static bool CanAddToBuffer(
            buffer_t& SIXTRL_RESTRICT_REF buffer,
            SIXTRL_ARGPTR_DEC size_type* SIXTRL_RESTRICT
                ptr_requ_objects  = nullptr,
            SIXTRL_ARGPTR_DEC size_type* SIXTRL_RESTRICT
                ptr_requ_slots    = nullptr,
            SIXTRL_ARGPTR_DEC size_type* SIXTRL_RESTRICT
                ptr_requ_dataptrs = nullptr
        ) SIXTRL_NOEXCEPT;

        SIXTRL_FN SIXTRL_STATIC
        SIXTRL_ARGPTR_DEC TDrift< T >* CreateNewOnBuffer(
            buffer_t& SIXTRL_RESTRICT_REF buffer );

        SIXTRL_FN SIXTRL_STATIC
        SIXTRL_ARGPTR_DEC TDrift< T >* AddToBuffer(
            buffer_t& SIXTRL_RESTRICT_REF buffer, value_type const length );

        /* ----------------------------------------------------------------- */

        SIXTRL_FN type_id_t getTypeId() const SIXTRL_NOEXCEPT;
        SIXTRL_FN void preset() SIXTRL_NOEXCEPT;
        SIXTRL_FN value_type getLength() const SIXTRL_NOEXCEPT;
        SIXTRL_FN void setLength( value_type const length ) SIXTRL_NOEXCEPT;

        /* ----------------------------------------------------------------- */

        value_type length SIXTRL_ALIGN( 8 );
    };

    template< typename T >
    SIXTRL_ARGPTR_DEC TDrift< T >* TDrift_new( Buffer& buffer );

    template< typename T >
    SIXTRL_ARGPTR_DEC TDrift< T >* TDrift_new(
        SIXTRL_ARGPTR_DEC ::NS(Buffer)* SIXTRL_RESTRICT ptr_buffer );

    template< typename T >
    SIXTRL_ARGPTR_DEC TDrift< T >* TDrift_add(
        Buffer& buffer, T const& length );

    template< typename T >
    SIXTRL_ARGPTR_DEC TDrift< T >* TDrift_add(
        SIXTRL_ARGPTR_DEC ::NS(Buffer)* SIXTRL_RESTRICT ptr_buffer,
        T const& length );

    /* ===================================================================== *
     * ====  TDriftExact< T > :
     * ===================================================================== */

    template< typename T >
    struct TDriftExact
    {
        using value_type = T;
        using type_id_t  = NS(object_type_id_t);
        using size_type  = NS(buffer_size_t);
        using buffer_t   = ::NS(Buffer);

        SIXTRL_FN TDriftExact() = default;
        SIXTRL_FN TDriftExact( TDriftExact< T > const& other ) = default;
        SIXTRL_FN TDriftExact( TDriftExact< T >&& other ) = default;

        SIXTRL_FN TDriftExact< T >& operator=(
            TDriftExact< T > const& rhs ) = default;

        SIXTRL_FN TDriftExact< T >& operator=(
            TDriftExact< T >&& rhs ) = default;

        SIXTRL_FN ~TDriftExact() = default;

        /* ----------------------------------------------------------------- */

        SIXTRL_FN SIXTRL_STATIC bool CanAddToBuffer(
                buffer_t& SIXTRL_RESTRICT_REF buffer,
                SIXTRL_ARGPTR_DEC size_type* SIXTRL_RESTRICT
                    ptr_requ_objects = nullptr,
                SIXTRL_ARGPTR_DEC size_type* SIXTRL_RESTRICT
                    ptr_requ_slots = nullptr,
                SIXTRL_ARGPTR_DEC size_type* SIXTRL_RESTRICT
                    ptr_requ_dataptrs = nullptr ) SIXTRL_NOEXCEPT;

        SIXTRL_FN SIXTRL_STATIC
        SIXTRL_ARGPTR_DEC TDriftExact< T >* CreateNewOnBuffer(
            buffer_t& SIXTRL_RESTRICT_REF buffer );

        SIXTRL_FN SIXTRL_STATIC
        SIXTRL_ARGPTR_DEC TDriftExact< T >* AddToBuffer(
            buffer_t& SIXTRL_RESTRICT_REF buffer,
            value_type const length );

        /* ----------------------------------------------------------------- */

        SIXTRL_FN type_id_t getTypeId() const SIXTRL_NOEXCEPT;
        SIXTRL_FN void preset() SIXTRL_NOEXCEPT;

        SIXTRL_FN value_type getLength() const SIXTRL_NOEXCEPT;
        SIXTRL_FN void setLength( value_type const length ) SIXTRL_NOEXCEPT;

        /* ----------------------------------------------------------------- */

        value_type length SIXTRL_ALIGN( 8 );
    };

    template< typename T >
    SIXTRL_ARGPTR_DEC TDriftExact< T >*
    TDriftExact_new( Buffer& buffer );

    template< typename T >
    SIXTRL_ARGPTR_DEC TDriftExact< T >*
    TDriftExact_new( SIXTRL_ARGPTR_DEC ::NS(Buffer)* SIXTRL_RESTRICT ptr_buf );

    template< typename T >
    SIXTRL_ARGPTR_DEC TDriftExact< T >*
    TDriftExact_add( Buffer& buffer, T const& length );

    template< typename T >
    SIXTRL_ARGPTR_DEC TDriftExact< T >*
    TDriftExact_add( SIXTRL_ARGPTR_DEC ::NS(Buffer)* SIXTRL_RESTRICT ptr_buf,
                     T const& length );

    /* ===================================================================== *
     * Specialization TDrift< NS(drift_real_t) >
     * ===================================================================== */

    template<> struct TDrift< NS(drift_real_t) > : public ::NS(Drift)
    {
        using value_type = NS(drift_real_t);
        using type_id_t  = NS(object_type_id_t);
        using c_api_t    = ::NS(Drift);
        using size_type  = NS(buffer_size_t);
        using buffer_t   = ::NS(Buffer);

        SIXTRL_FN TDrift() = default;
        SIXTRL_FN TDrift( TDrift< NS(drift_real_t) > const& other ) = default;
        SIXTRL_FN TDrift( TDrift< NS(drift_real_t) >&& other ) = default;

        SIXTRL_FN TDrift< NS(drift_real_t) >& operator=(
            TDrift< NS(drift_real_t) > const& rhs ) = default;

        SIXTRL_FN TDrift< NS(drift_real_t) >& operator=(
            TDrift< NS(drift_real_t) >&& rhs ) = default;

        SIXTRL_FN ~TDrift() = default;

        /* ----------------------------------------------------------------- */

        SIXTRL_FN SIXTRL_STATIC bool CanAddToBuffer(
            buffer_t& SIXTRL_RESTRICT_REF buffer,
            SIXTRL_ARGPTR_DEC size_type* SIXTRL_RESTRICT
                ptr_requ_objects  = nullptr,
            SIXTRL_ARGPTR_DEC size_type* SIXTRL_RESTRICT
                ptr_requ_slots    = nullptr,
            SIXTRL_ARGPTR_DEC size_type* SIXTRL_RESTRICT
                ptr_requ_dataptrs = nullptr ) SIXTRL_NOEXCEPT;

        SIXTRL_FN SIXTRL_STATIC
        SIXTRL_ARGPTR_DEC TDrift< NS(drift_real_t) >* CreateNewOnBuffer(
            buffer_t& SIXTRL_RESTRICT_REF buffer ) SIXTRL_NOEXCEPT;

        SIXTRL_FN SIXTRL_STATIC
        SIXTRL_ARGPTR_DEC TDrift< NS(drift_real_t) >* AddToBuffer(
            buffer_t& SIXTRL_RESTRICT_REF buffer, value_type const length );

        /* ----------------------------------------------------------------- */

        SIXTRL_FN type_id_t getTypeId() const SIXTRL_NOEXCEPT;

        SIXTRL_FN c_api_t const* getCApiPtr() const SIXTRL_NOEXCEPT;
        SIXTRL_FN c_api_t* getCApiPtr() SIXTRL_NOEXCEPT;

        SIXTRL_FN void preset() SIXTRL_NOEXCEPT;
        SIXTRL_FN value_type getLength() const SIXTRL_NOEXCEPT;
        SIXTRL_FN void setLength( value_type const length ) SIXTRL_NOEXCEPT;
    };

    using Drift = TDrift< NS(drift_real_t) >;

    SIXTRL_ARGPTR_DEC Drift* Drift_new(
        Buffer& SIXTRL_RESTRICT_REF buffer );

    SIXTRL_ARGPTR_DEC Drift* Drift_new(
        SIXTRL_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT ptr_buffer );

    SIXTRL_ARGPTR_DEC Drift*
    Drift_add( Buffer& SIXTRL_RESTRICT_REF buffer,
               Drift::value_type const length );

    SIXTRL_ARGPTR_DEC Drift* Drift_add(
        SIXTRL_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT ptr_buffer,
        Drift::value_type const length );

    /* ===================================================================== *
     * Specialization TDriftExact< NS(drift_real_t) >
     * ===================================================================== */

    template<> struct TDriftExact< NS(drift_real_t) > : public ::NS(DriftExact)
    {
        using value_type = NS(drift_real_t);
        using type_id_t  = NS(object_type_id_t);
        using c_api_t    = ::NS(DriftExact);
        using size_type  = NS(buffer_size_t);
        using buffer_t   = ::NS(Buffer);

        SIXTRL_FN TDriftExact() = default;

        SIXTRL_FN TDriftExact(
            TDriftExact< NS(drift_real_t) > const& other ) = default;

        SIXTRL_FN TDriftExact(
            TDriftExact< NS(drift_real_t) >&& other ) = default;

        SIXTRL_FN TDriftExact< NS(drift_real_t) >& operator=(
            TDriftExact< NS(drift_real_t) > const& rhs ) = default;

        SIXTRL_FN TDriftExact< NS(drift_real_t) >& operator=(
            TDriftExact< NS(drift_real_t) >&& rhs ) = default;

        SIXTRL_FN ~TDriftExact() = default;

        /* ----------------------------------------------------------------- */

        SIXTRL_FN SIXTRL_STATIC bool CanAddToBuffer(
                buffer_t& SIXTRL_RESTRICT_REF buffer,
                SIXTRL_ARGPTR_DEC size_type* SIXTRL_RESTRICT
                    ptr_requ_objects  = nullptr,
                SIXTRL_ARGPTR_DEC size_type* SIXTRL_RESTRICT
                    ptr_requ_slots    = nullptr,
                SIXTRL_ARGPTR_DEC size_type* SIXTRL_RESTRICT
                    ptr_requ_dataptrs = nullptr
            ) SIXTRL_NOEXCEPT;

        SIXTRL_FN SIXTRL_STATIC
        SIXTRL_ARGPTR_DEC TDriftExact< NS(drift_real_t) >*
        CreateNewOnBuffer( buffer_t& SIXTRL_RESTRICT_REF buffer );

        SIXTRL_FN SIXTRL_STATIC
        SIXTRL_ARGPTR_DEC TDriftExact< NS(drift_real_t) >*
        AddToBuffer( buffer_t& SIXTRL_RESTRICT_REF buffer,
            value_type const length );

        /* ----------------------------------------------------------------- */

        SIXTRL_FN type_id_t getTypeId() const SIXTRL_NOEXCEPT;

        SIXTRL_FN c_api_t const* getCApiPtr() const SIXTRL_NOEXCEPT;
        SIXTRL_FN c_api_t* getCApiPtr() SIXTRL_NOEXCEPT;


        SIXTRL_FN void       preset() SIXTRL_NOEXCEPT;
        SIXTRL_FN value_type getLength() const SIXTRL_NOEXCEPT;
        SIXTRL_FN void setLength( value_type const length ) SIXTRL_NOEXCEPT;

    };

    /* --------------------------------------------------------------------- */

    using DriftExact = TDriftExact< NS(drift_real_t) >;

    SIXTRL_ARGPTR_DEC DriftExact*
    DriftExact_new( Buffer& SIXTRL_RESTRICT_REF buffer );

    SIXTRL_ARGPTR_DEC DriftExact*
    DriftExact_new( SIXTRL_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT ptr_buffer );

    SIXTRL_ARGPTR_DEC DriftExact*
    DriftExact_add( Buffer& SIXTRL_RESTRICT_REF buffer,
                    DriftExact::value_type const length );

    SIXTRL_ARGPTR_DEC DriftExact*
    DriftExact_add( SIXTRL_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT ptr_buffer,
                    DriftExact::value_type const length );
}

/* ************************************************************************* *
 * *** Implementation of inline and template member functions          ***** *
 * ************************************************************************* */

namespace SIXTRL_NAMESPACE
{
    /* ===================================================================== *
     * ====  TDrift< T >:
     * ===================================================================== */

    template< typename T >
    bool TDrift< T >::CanAddToBuffer(
            typename TDrift< T >::buffer_t& SIXTRL_RESTRICT_REF buffer,
            typename TDrift< T >::size_type* SIXTRL_RESTRICT ptr_requ_objects,
            typename TDrift< T >::size_type* SIXTRL_RESTRICT ptr_requ_slots,
            typename TDrift< T >::size_type* SIXTRL_RESTRICT ptr_requ_dataptrs
        ) SIXTRL_NOEXCEPT
    {
        using _this_t = TDrift< T >;

        static_assert( std::is_trivial< _this_t >::value, "" );
        static_assert( std::is_standard_layout< _this_t >::value, "" );

        return ::NS(Buffer_can_add_object)( &buffer, sizeof( _this_t ), 0u,
            nullptr, nullptr, ptr_requ_objects, ptr_requ_slots,
                ptr_requ_dataptrs );
    }

    template< typename T >
    SIXTRL_INLINE SIXTRL_ARGPTR_DEC TDrift< T >*
    TDrift< T >::CreateNewOnBuffer(
        typename TDrift< T >::buffer_t& SIXTRL_RESTRICT_REF buffer )
    {
        using _this_t = TDrift< T >;
        using  ptr_t  = SIXTRL_ARGPTR_DEC _this_t*;

        static_assert( std::is_trivial< _this_t >::value, "" );
        static_assert( std::is_standard_layout< _this_t >::value, "" );

        _this_t temp;
        temp.preset();
        type_id_t const type_id = temp.getTypeId();

        return reinterpret_cast< ptr_t >( static_cast< uintptr_t >(
            ::NS(Object_get_begin_addr)( ::NS(Buffer_add_object)(
                &buffer, &temp, sizeof( _this_t ), type_id, 0u,
                    nullptr, nullptr, nullptr ) ) ) );
    }

    template< typename T >
    SIXTRL_INLINE SIXTRL_ARGPTR_DEC TDrift< T >*
    TDrift< T >::AddToBuffer(
        typename TDrift< T >::buffer_t& SIXTRL_RESTRICT_REF buffer,
        typename TDrift< T >::value_type const length )
    {
        using _this_t = TDrift< T >;
        using  ptr_t  = SIXTRL_ARGPTR_DEC _this_t*;

        static_assert( std::is_trivial< _this_t >::value, "" );
        static_assert( std::is_standard_layout< _this_t >::value, "" );

        _this_t temp;
        temp.setLength( length );
        type_id_t const type_id = temp.getTypeId();

        return reinterpret_cast< ptr_t >( static_cast< uintptr_t >(
            ::NS(Object_get_begin_addr)(
                ::NS(Buffer_add_object)( &buffer, &temp, sizeof( _this_t ),
                    type_id, 0u, nullptr, nullptr, nullptr ) ) ) );
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    template< typename T >
    SIXTRL_INLINE typename TDrift< T >::type_id_t
    TDrift< T >::getTypeId() const SIXTRL_NOEXCEPT
    {
        return ::NS(OBJECT_TYPE_DRIFT);
    }

    template< typename T >
    SIXTRL_INLINE void TDrift< T >::preset() SIXTRL_NOEXCEPT
    {
        this->setLength( value_type{} );
    }

    template< typename T >
    SIXTRL_INLINE typename TDrift< T >::value_type
    TDrift< T >::getLength() const SIXTRL_NOEXCEPT
    {
        return this->length;
    }

    template< typename T >
    SIXTRL_INLINE void TDrift< T >::setLength(
        typename TDrift< T >::value_type const length ) SIXTRL_NOEXCEPT
    {
        this->length = length;
        return;
    }

    /* --------------------------------------------------------------------- */

    template< typename T >
    SIXTRL_INLINE TDrift< T >* TDrift_new( Buffer& buffer )
    {
        return TDrift< T >::CreateNewOnBuffer( *( buffer.getCApiPtr() ) );
    }

    template< typename T >
    SIXTRL_INLINE TDrift< T >* TDrift_new(
        ::NS(Buffer)* SIXTRL_RESTRICT ptr_buffer )
    {
        return ( ptr_buffer != nullptr )
            ? ( TDrift< T >::CreateNewOnBuffer( *ptr_buffer ) )
            : nullptr;
    }

    template< typename T >
    SIXTRL_INLINE TDrift< T >* TDrift_add( Buffer& buffer, T const& length )
    {
        return TDrift< T >::AddToBuffer( *( buffer.getCApiPtr() ), length );
    }

    template< typename T >
    SIXTRL_INLINE TDrift< T >* TDrift_add(
        ::NS(Buffer)* SIXTRL_RESTRICT ptr_buffer, T const& length )
    {
        return ( ptr_buffer != nullptr )
            ? ( TDrift< T >::AddToBuffer( *ptr_buffer, length ) )
            : nullptr;
    }

    /* ===================================================================== *
     * ====  TDriftExact< T >:
     * ===================================================================== */

    template< typename T >
    SIXTRL_INLINE bool TDriftExact< T >::CanAddToBuffer(
            typename TDriftExact< T >::buffer_t& SIXTRL_RESTRICT_REF buffer,
            typename TDriftExact< T >::size_type*
                SIXTRL_RESTRICT ptr_requ_objects,
            typename TDriftExact< T >::size_type*
                SIXTRL_RESTRICT ptr_requ_slots,
            typename TDriftExact< T >::size_type*
                SIXTRL_RESTRICT ptr_requ_dataptrs ) SIXTRL_NOEXCEPT
    {
        using _this_t = TDriftExact< T >;

        static_assert( std::is_trivial< _this_t >::value, "" );
        static_assert( std::is_standard_layout< _this_t >::value, "" );

        return ::NS(Buffer_can_add_object)( &buffer, sizeof( _this_t ), 0u,
            nullptr, nullptr, ptr_requ_objects, ptr_requ_slots,
                ptr_requ_dataptrs );
    }

    template< typename T >
    SIXTRL_INLINE SIXTRL_ARGPTR_DEC TDriftExact< T >*
    TDriftExact< T >::CreateNewOnBuffer(
        typename TDriftExact< T >::buffer_t& SIXTRL_RESTRICT_REF buffer )
    {
        using _this_t = TDriftExact< T >;
        using  ptr_t  = SIXTRL_ARGPTR_DEC _this_t*;

        static_assert( std::is_trivial< _this_t >::value, "" );
        static_assert( std::is_standard_layout< _this_t >::value, "" );

        _this_t temp;
        temp.preset();

        return reinterpret_cast< ptr_t >( static_cast< uintptr_t >(
            ::NS(Object_get_begin_addr)( ::NS(Buffer_add_object)(
                &buffer, &temp, sizeof( _this_t ), temp.getTypeId(), 0u,
                    nullptr, nullptr, nullptr ) ) ) );
    }

    template< typename T >
    SIXTRL_INLINE SIXTRL_ARGPTR_DEC TDriftExact< T >*
    TDriftExact< T >::AddToBuffer(
        typename TDriftExact< T >::buffer_t& SIXTRL_RESTRICT_REF buffer,
        typename TDriftExact< T >::value_type const length )
    {
        using _this_t = TDriftExact< T >;
        using  ptr_t  = SIXTRL_ARGPTR_DEC _this_t*;

        static_assert( std::is_trivial< _this_t >::value, "" );
        static_assert( std::is_standard_layout< _this_t >::value, "" );

        _this_t temp;
        temp.setLength( length );

        return reinterpret_cast< ptr_t >( static_cast< uintptr_t >(
            ::NS(Object_get_begin_addr)( ::NS(Buffer_add_object)(
                &buffer, &temp, sizeof( _this_t ), temp.getTypeId(),
                    0u, nullptr, nullptr, nullptr ) ) ) );
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    template< typename T >
    SIXTRL_INLINE typename TDriftExact< T >::type_id_t
    TDriftExact< T >::getTypeId() const SIXTRL_NOEXCEPT
    {
        return ::NS(OBJECT_TYPE_DRIFT_EXACT);
    }

    template< typename T >
    SIXTRL_INLINE void TDriftExact< T >::preset() SIXTRL_NOEXCEPT
    {
        this->setLength( value_type{} );
        return;
    }

    template< typename T >
    SIXTRL_INLINE typename TDriftExact< T >::value_type
    TDriftExact< T >::getLength() const SIXTRL_NOEXCEPT
    {
        return this->length;
    }

    template< typename T >
    SIXTRL_INLINE void TDriftExact< T >::setLength(
        typename TDriftExact< T >::value_type const length ) SIXTRL_NOEXCEPT
    {
        this->length = length;
        return;
    }

    /* --------------------------------------------------------------------- */

    template< typename T >
    SIXTRL_ARGPTR_DEC TDriftExact< T >* TDriftExact_new( Buffer& buffer )
    {
        return TDriftExact< T >::CreateNewOnBuffer( *buffer.getCApiPtr() );
    }

    template< typename T >
    SIXTRL_ARGPTR_DEC TDriftExact< T >* TDriftExact_new(
        SIXTRL_ARGPTR_DEC ::NS(Buffer)* SIXTRL_RESTRICT ptr_buffer )
    {
        return ( ptr_buffer != nullptr )
            ? ( TDriftExact< T >::CreateNewOnBuffer( *ptr_buffer ) )
            : nullptr;
    }

    template< typename T >
    SIXTRL_ARGPTR_DEC TDriftExact< T >* TDriftExact_add(
        Buffer& buffer, T const& length )
    {
        return TDriftExact< T >::AddToBuffer(
            *( buffer.getCApiPtr() ), length );
    }

    template< typename T >
    SIXTRL_ARGPTR_DEC TDriftExact< T >* TDriftExact_add(
        SIXTRL_ARGPTR_DEC ::NS(Buffer)* SIXTRL_RESTRICT ptr_buffer,
        T const& length )
    {
        return ( ptr_buffer != nullptr )
            ? ( TDriftExact< T >::AddToBuffer( *ptr_buffer, length ) )
            : nullptr;
    }

    /* ===================================================================== *
     * ====  TDrift< NS(drift_real_t) >:
     * ===================================================================== */

    SIXTRL_INLINE bool TDrift< NS(drift_real_t) >::CanAddToBuffer(
            TDrift< NS(drift_real_t) >::buffer_t& SIXTRL_RESTRICT_REF buffer,
            TDrift< NS(drift_real_t) >::size_type*
                SIXTRL_RESTRICT ptr_requ_objects,
            TDrift< NS(drift_real_t) >::size_type*
                SIXTRL_RESTRICT ptr_requ_slots,
            TDrift< NS(drift_real_t) >::size_type*
                SIXTRL_RESTRICT ptr_requ_dataptrs ) SIXTRL_NOEXCEPT
    {
        return ::NS(Drift_can_be_added)(
            &buffer, ptr_requ_objects, ptr_requ_slots, ptr_requ_dataptrs );
    }

    SIXTRL_INLINE SIXTRL_ARGPTR_DEC TDrift< NS(drift_real_t) >*
    TDrift< NS(drift_real_t) >::CreateNewOnBuffer(
        TDrift< NS(drift_real_t) >::buffer_t&
            SIXTRL_RESTRICT_REF buffer ) SIXTRL_NOEXCEPT
    {
        using ptr_t = SIXTRL_ARGPTR_DEC TDrift< NS(drift_real_t) >*;
        return static_cast< ptr_t >( ::NS(Drift_new)( &buffer ) );
    }


    SIXTRL_INLINE SIXTRL_ARGPTR_DEC TDrift< NS(drift_real_t) >*
    TDrift< NS(drift_real_t) >::AddToBuffer(
        TDrift< NS(drift_real_t) >::buffer_t& SIXTRL_RESTRICT_REF buffer,
        TDrift< NS(drift_real_t) >::value_type const length )
    {
        using ptr_t = SIXTRL_ARGPTR_DEC TDrift< NS(drift_real_t) >*;
        return static_cast< ptr_t >( ::NS(Drift_add)( &buffer, length ) );
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    SIXTRL_INLINE SIXTRL_ARGPTR_DEC TDrift< NS(drift_real_t) >::c_api_t const*
    TDrift< NS(drift_real_t) >::getCApiPtr() const SIXTRL_NOEXCEPT
    {
        return reinterpret_cast< SIXTRL_ARGPTR_DEC c_api_t const* >( this );
    }

    SIXTRL_INLINE SIXTRL_ARGPTR_DEC TDrift< NS(drift_real_t) >::c_api_t*
    TDrift< NS(drift_real_t) >::getCApiPtr() SIXTRL_NOEXCEPT
    {
        return const_cast< SIXTRL_ARGPTR_DEC c_api_t* >(
            static_cast< TDrift< NS(drift_real_t) > const& >(
                *this ).getCApiPtr() );
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    SIXTRL_INLINE TDrift< NS(drift_real_t) >::type_id_t
    TDrift< NS(drift_real_t) >::getTypeId() const SIXTRL_NOEXCEPT
    {
        return ::NS(OBJECT_TYPE_DRIFT);
    }

    SIXTRL_INLINE void TDrift< NS(drift_real_t) >::preset() SIXTRL_NOEXCEPT
    {
        ::NS(Drift_preset)( this->getCApiPtr() );
        return;
    }

    SIXTRL_INLINE TDrift< NS(drift_real_t) >::value_type
    TDrift< NS(drift_real_t) >::getLength() const SIXTRL_NOEXCEPT
    {
        return ::NS(Drift_get_length)( this->getCApiPtr() );
    }

    SIXTRL_INLINE void TDrift< NS(drift_real_t) >::setLength(
        TDrift< NS(drift_real_t) >::value_type const length ) SIXTRL_NOEXCEPT
    {
        ::NS(Drift_set_length)( this->getCApiPtr(), length );
        return;
    }

    /* --------------------------------------------------------------------- */

    SIXTRL_INLINE SIXTRL_ARGPTR_DEC Drift*
    Drift_new( Buffer& SIXTRL_RESTRICT_REF buffer )
    {
        using ptr_t = SIXTRL_ARGPTR_DEC Drift*;
        return static_cast< ptr_t >( ::NS(Drift_new)( buffer.getCApiPtr() ) );
    }

    SIXTRL_INLINE SIXTRL_ARGPTR_DEC Drift* Drift_new(
        SIXTRL_ARGPTR_DEC ::NS(Buffer)* SIXTRL_RESTRICT ptr_buffer )
    {
        using ptr_t = SIXTRL_ARGPTR_DEC Drift*;
        return static_cast< ptr_t >( ::NS(Drift_new)( ptr_buffer ) );
    }

    SIXTRL_INLINE SIXTRL_ARGPTR_DEC Drift*
    Drift_add( Buffer& SIXTRL_RESTRICT_REF buffer,
                      Drift::value_type const length )
    {
        using ptr_t = SIXTRL_ARGPTR_DEC Drift*;
        return static_cast< ptr_t >(
            ::NS(Drift_add)( buffer.getCApiPtr(), length ) );
    }

    SIXTRL_INLINE SIXTRL_ARGPTR_DEC Drift*
    Drift_add( NS(Buffer)* SIXTRL_RESTRICT ptr_buffer,
                      Drift::value_type const length )
    {
        using ptr_t = SIXTRL_ARGPTR_DEC Drift*;
        return static_cast< ptr_t >( ::NS(Drift_add)( ptr_buffer, length ) );
    }

    /* ===================================================================== *
     * ====  TDriftExact< NS(drift_real_t) >:
     * ===================================================================== */

    SIXTRL_INLINE bool TDriftExact< NS(drift_real_t) >::CanAddToBuffer(
            TDriftExact< NS(drift_real_t) >::buffer_t& SIXTRL_RESTRICT_REF buffer,
            TDriftExact< NS(drift_real_t) >::size_type*
                SIXTRL_RESTRICT ptr_requ_objects,
            TDriftExact< NS(drift_real_t) >::size_type*
                SIXTRL_RESTRICT ptr_requ_slots,
            TDriftExact< NS(drift_real_t) >::size_type*
                SIXTRL_RESTRICT ptr_requ_dataptrs ) SIXTRL_NOEXCEPT
    {
        return ::NS(DriftExact_can_be_added)(
            &buffer, ptr_requ_objects, ptr_requ_slots, ptr_requ_dataptrs );
    }

    SIXTRL_INLINE SIXTRL_ARGPTR_DEC TDriftExact< NS(drift_real_t) >*
    TDriftExact< NS(drift_real_t) >::CreateNewOnBuffer(
            TDriftExact< NS(drift_real_t) >::buffer_t&
                SIXTRL_RESTRICT_REF buffer )
    {
        using ptr_t = SIXTRL_ARGPTR_DEC TDriftExact< NS(drift_real_t) >*;
        return static_cast< ptr_t >( ::NS(DriftExact_new)( &buffer ) );
    }

    SIXTRL_INLINE SIXTRL_ARGPTR_DEC TDriftExact< NS(drift_real_t) >*
    TDriftExact< NS(drift_real_t) >::AddToBuffer(
        TDriftExact< NS(drift_real_t) >::buffer_t& SIXTRL_RESTRICT_REF buffer,
        TDriftExact< NS(drift_real_t) >::value_type const length )
    {
        using ptr_t = SIXTRL_ARGPTR_DEC TDriftExact< NS(drift_real_t) >*;
        return static_cast< ptr_t >( ::NS(DriftExact_add)( &buffer, length ) );
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    SIXTRL_INLINE SIXTRL_ARGPTR_DEC TDriftExact< NS(drift_real_t) >::c_api_t const*
    TDriftExact< NS(drift_real_t) >::getCApiPtr() const SIXTRL_NOEXCEPT
    {
        return static_cast< SIXTRL_ARGPTR_DEC c_api_t const* >( this );
    }

    SIXTRL_INLINE SIXTRL_ARGPTR_DEC TDriftExact< NS(drift_real_t) >::c_api_t*
    TDriftExact< NS(drift_real_t) >::getCApiPtr() SIXTRL_NOEXCEPT
    {
        return const_cast< SIXTRL_ARGPTR_DEC c_api_t* >(
            static_cast< TDriftExact< NS(drift_real_t) > const& >(
                *this ).getCApiPtr() );
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    SIXTRL_INLINE TDriftExact< NS(drift_real_t) >::type_id_t
    TDriftExact< NS(drift_real_t) >::getTypeId() const SIXTRL_NOEXCEPT
    {
        return ::NS(OBJECT_TYPE_DRIFT_EXACT);
    }

    SIXTRL_INLINE void
    TDriftExact< NS(drift_real_t) >::preset() SIXTRL_NOEXCEPT
    {
        ::NS(DriftExact_preset)( this->getCApiPtr() );
        return;
    }

    SIXTRL_INLINE TDriftExact< NS(drift_real_t) >::value_type
    TDriftExact< NS(drift_real_t) >::getLength() const SIXTRL_NOEXCEPT
    {
        return ::NS(DriftExact_get_length)( this->getCApiPtr() );
    }

    SIXTRL_INLINE void TDriftExact< NS(drift_real_t) >::setLength(
        TDriftExact< NS(drift_real_t) >::value_type const length ) SIXTRL_NOEXCEPT
    {
        ::NS(DriftExact_set_length)( this->getCApiPtr(), length );
        return;
    }

    /* --------------------------------------------------------------------- */

    SIXTRL_INLINE SIXTRL_ARGPTR_DEC DriftExact* DriftExact_new(
        Buffer& SIXTRL_RESTRICT_REF buffer )
    {
        using ptr_t = SIXTRL_ARGPTR_DEC DriftExact*;
        return static_cast< ptr_t >(
            ::NS(DriftExact_new)( buffer.getCApiPtr() ) );
    }

    SIXTRL_INLINE SIXTRL_ARGPTR_DEC DriftExact* DriftExact_new(
        ::NS(Buffer)* SIXTRL_RESTRICT ptr_buffer )
    {
        using ptr_t = SIXTRL_ARGPTR_DEC DriftExact*;
        return static_cast< ptr_t >( ::NS(DriftExact_new)( ptr_buffer ) );
    }

    SIXTRL_INLINE SIXTRL_ARGPTR_DEC DriftExact* DriftExact_add(
        Buffer& SIXTRL_RESTRICT_REF buffer,
        DriftExact::value_type const length )
    {
        using ptr_t = SIXTRL_ARGPTR_DEC DriftExact*;
        return static_cast< ptr_t >(
                ::NS(DriftExact_add)( buffer.getCApiPtr(), length ) );
    }

    SIXTRL_INLINE SIXTRL_ARGPTR_DEC DriftExact* DriftExact_add(
        NS(Buffer)* SIXTRL_RESTRICT ptr_buffer,
        DriftExact::value_type const length )
    {
        using ptr_t = SIXTRL_ARGPTR_DEC DriftExact*;
        return static_cast< ptr_t >(
            ::NS(DriftExact_add)( ptr_buffer, length ) );
    }
}

#endif /* defined( __cplusplus ) */

#endif /* CXX_SIXTRACKLIB_COMMON_IMPL_BE_DRIFT_HPP__ */

/* end: sixtracklib/common/impl/be_drift.hpp */
