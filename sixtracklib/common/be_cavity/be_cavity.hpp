#ifndef CXX_SIXTRACKLIB_COMMON_BE_CAVITY_BEAM_ELEMENT_BE_CAVITY_HPP__
#define CXX_SIXTRACKLIB_COMMON_BE_CAVITY_BEAM_ELEMENT_BE_CAVITY_HPP__

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
    #include "sixtracklib/common/be_cavity/be_cavity.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

/* ===================================================================== *
 * ====  TCavity< T > :
 * ===================================================================== */

namespace SIXTRL_CXX_NAMESPACE
{
    template< typename T >
    struct TCavity
    {
        using value_type = T;
        using type_id_t  = NS(object_type_id_t);
        using size_type  = NS(buffer_size_t);
        using buffer_t   = SIXTRL_CXX_NAMESPACE::Buffer;
        using c_buffer_t = buffer_t::c_api_t;

        SIXTRL_FN TCavity() = default;
        SIXTRL_FN TCavity( TCavity< T > const& other ) = default;
        SIXTRL_FN TCavity( TCavity< T >&& other ) = default;

        SIXTRL_FN TCavity< T >& operator=( TCavity< T > const& rhs ) = default;
        SIXTRL_FN TCavity< T >& operator=( TCavity< T >&& rhs ) = default;
        SIXTRL_FN ~TCavity() = default;

        /* ----------------------------------------------------------------- */

        SIXTRL_FN SIXTRL_STATIC bool CanAddToBuffer(
            buffer_t& SIXTRL_RESTRICT_REF buffer,
            SIXTRL_ARGPTR_DEC size_type* SIXTRL_RESTRICT
                ptr_requ_objects  = nullptr,
            SIXTRL_ARGPTR_DEC size_type* SIXTRL_RESTRICT
                ptr_requ_slots    = nullptr,
            SIXTRL_ARGPTR_DEC size_type* SIXTRL_RESTRICT
                ptr_requ_dataptrs = nullptr ) SIXTRL_NOEXCEPT;

        SIXTRL_FN SIXTRL_STATIC SIXTRL_ARGPTR_DEC TCavity< T >*
        CreateNewOnBuffer( buffer_t& SIXTRL_RESTRICT_REF buffer );

        SIXTRL_FN SIXTRL_STATIC SIXTRL_ARGPTR_DEC TCavity< T >* AddToBuffer(
            c_buffer_t& SIXTRL_RESTRICT_REF buffer,
            value_type const voltage   = value_type{},
            value_type const frequency = value_type{},
            value_type const lag       = value_type{} );

        /* ----------------------------------------------------------------- */

        SIXTRL_FN type_id_t getTypeId()      const SIXTRL_NOEXCEPT;
        SIXTRL_FN size_type getNumDataPtrs() const SIXTRL_NOEXCEPT;

        SIXTRL_FN value_type getVoltage() const SIXTRL_NOEXCEPT;
        SIXTRL_FN value_type getFrequency() const SIXTRL_NOEXCEPT;
        SIXTRL_FN value_type getLag() const SIXTRL_NOEXCEPT;

        SIXTRL_FN void preset() SIXTRL_NOEXCEPT;

        SIXTRL_FN void setVoltage( value_type const& voltage ) SIXTRL_NOEXCEPT;
        SIXTRL_FN void setFrequency( value_type const& frequency ) SIXTRL_NOEXCEPT;
        SIXTRL_FN void setLag( value_type const& lag ) SIXTRL_NOEXCEPT;

        /* ----------------------------------------------------------------- */

        value_type voltage   SIXTRL_ALIGN( 8 );
        value_type frequency SIXTRL_ALIGN( 8 );
        value_type lag       SIXTRL_ALIGN( 8 );
    };

    /* ----------------------------------------------------------------- */

    template< typename T >
    SIXTRL_ARGPTR_DEC TCavity< T >* TCavity_new(
        Buffer& SIXTRL_RESTRICT_REF buffer );

    template< typename T >
    SIXTRL_ARGPTR_DEC TCavity< T >* TCavity_new(
        SIXTRL_ARGPTR_DEC ::NS(Buffer)* SIXTRL_RESTRICT ptr_buffer );

    template< typename T >
    SIXTRL_ARGPTR_DEC TCavity< T >* TCavity_add(
        Buffer& SIXTRL_RESTRICT_REF buffer,
        typename TCavity< T >::value_type const voltage =
            typename TCavity< T >::value_type{},
        typename TCavity< T >::value_type const frequency =
            typename TCavity< T >::value_type{},
        typename TCavity< T >::value_type const lag =
            typename TCavity< T >::value_type{} );

    template< typename T >
    SIXTRL_ARGPTR_DEC TCavity< T >* TCavity_add(
        SIXTRL_ARGPTR_DEC ::NS(Buffer)* SIXTRL_RESTRICT ptr_buffer,
        typename TCavity< T >::value_type const voltage =
            typename TCavity< T >::value_type{},
        typename TCavity< T >::value_type const frequency =
            typename TCavity< T >::value_type{},
        typename TCavity< T >::value_type const lag =
            typename TCavity< T >::value_type{} );

    template< typename T >
    SIXTRL_ARGPTR_DEC TCavity< T >* TCavity_add_copy(
        Buffer& SIXTRL_RESTRICT_REF buffer,
        TCavity< T > const& SIXTRL_RESTRICT_REF orig );

    template< typename T >
    SIXTRL_ARGPTR_DEC TCavity< T >* TCavity_add_copy(
        SIXTRL_ARGPTR_DEC ::NS(Buffer)* SIXTRL_RESTRICT ptr_buffer,
        TCavity< T > const& SIXTRL_RESTRICT_REF orig );

    /* ===================================================================== *
     * ====  Specialization TXYShift<  > :
     * ===================================================================== */

    template<> struct TCavity< SIXTRL_REAL_T > : public ::NS(Cavity)
    {
        using value_type = SIXTRL_REAL_T;
        using type_id_t  = NS(object_type_id_t);
        using size_type  = NS(buffer_size_t);
        using buffer_t   = SIXTRL_CXX_NAMESPACE::Buffer;
        using c_buffer_t = buffer_t::c_api_t;
        using c_api_t    = ::NS(Cavity);

        SIXTRL_FN TCavity() = default;
        SIXTRL_FN TCavity( TCavity< SIXTRL_REAL_T > const& other ) = default;
        SIXTRL_FN TCavity( TCavity< SIXTRL_REAL_T >&& other ) = default;

        SIXTRL_FN TCavity< SIXTRL_REAL_T >& operator=(
            TCavity< SIXTRL_REAL_T > const& rhs ) = default;

        SIXTRL_FN TCavity< SIXTRL_REAL_T >& operator=(
            TCavity< SIXTRL_REAL_T >&& rhs ) = default;

        SIXTRL_FN ~TCavity() = default;

        /* ----------------------------------------------------------------- */

        SIXTRL_FN SIXTRL_STATIC bool CanAddToBuffer(
            buffer_t& SIXTRL_RESTRICT_REF buffer,
            SIXTRL_ARGPTR_DEC size_type* SIXTRL_RESTRICT
                ptr_requ_objects = nullptr,
            SIXTRL_ARGPTR_DEC size_type* SIXTRL_RESTRICT
                ptr_requ_slots = nullptr,
            SIXTRL_ARGPTR_DEC size_type* SIXTRL_RESTRICT
                ptr_requ_dataptrs = nullptr
            ) SIXTRL_NOEXCEPT;

        SIXTRL_FN SIXTRL_STATIC SIXTRL_ARGPTR_DEC TCavity< SIXTRL_REAL_T >*
        CreateNewOnBuffer( buffer_t& SIXTRL_RESTRICT_REF buffer );

        SIXTRL_FN SIXTRL_STATIC SIXTRL_ARGPTR_DEC TCavity< SIXTRL_REAL_T >*
        AddToBuffer(
            c_buffer_t& SIXTRL_RESTRICT_REF buffer,
            value_type const voltage, value_type const frequency,
            value_type const lag );

        /* ----------------------------------------------------------------- */

        SIXTRL_ARGPTR_DEC c_api_t const* getCApiPtr() const SIXTRL_NOEXCEPT;
        SIXTRL_ARGPTR_DEC c_api_t* getCApiPtr() SIXTRL_NOEXCEPT;

        /* ----------------------------------------------------------------- */

        SIXTRL_FN type_id_t getTypeId()      const SIXTRL_NOEXCEPT;
        SIXTRL_FN size_type getNumDataPtrs() const SIXTRL_NOEXCEPT;

        SIXTRL_FN value_type getVoltage() const SIXTRL_NOEXCEPT;
        SIXTRL_FN value_type getFrequency() const SIXTRL_NOEXCEPT;
        SIXTRL_FN value_type getLag() const SIXTRL_NOEXCEPT;

        SIXTRL_FN void preset() SIXTRL_NOEXCEPT;
        SIXTRL_FN void setVoltage( value_type const voltage ) SIXTRL_NOEXCEPT;
        SIXTRL_FN void setFrequency( value_type const frequ ) SIXTRL_NOEXCEPT;
        SIXTRL_FN void setLag( value_type const lag ) SIXTRL_NOEXCEPT;
    };

    /* --------------------------------------------------------------------- */

    using Cavity = TCavity< SIXTRL_REAL_T >;

    SIXTRL_ARGPTR_DEC Cavity* Cavity_new( Buffer& SIXTRL_RESTRICT_REF buffer );

    SIXTRL_ARGPTR_DEC Cavity*
    Cavity_new( SIXTRL_ARGPTR_DEC ::NS(Buffer)* SIXTRL_RESTRICT ptr_buffer );

    SIXTRL_ARGPTR_DEC Cavity* Cavity_add(
        Buffer& SIXTRL_RESTRICT_REF buffer,
        Cavity::value_type const voltage   = Cavity::value_type{},
        Cavity::value_type const frequency = Cavity::value_type{},
        Cavity::value_type const lag       = Cavity::value_type{} );

    SIXTRL_ARGPTR_DEC Cavity* Cavity_add(
        SIXTRL_ARGPTR_DEC ::NS(Buffer)* SIXTRL_RESTRICT ptr_buffer,
        Cavity::value_type const voltage   = Cavity::value_type{},
        Cavity::value_type const frequency = Cavity::value_type{},
        Cavity::value_type const lag       = Cavity::value_type{} );

    SIXTRL_ARGPTR_DEC Cavity* Cavity_add_copy(
        Buffer& SIXTRL_RESTRICT_REF buffer,
        Cavity const& SIXTRL_RESTRICT_REF cavity );

    SIXTRL_ARGPTR_DEC Cavity* Cavity_add_copy(
        SIXTRL_ARGPTR_DEC ::NS(Buffer)* SIXTRL_RESTRICT ptr_buffer,
        Cavity const& SIXTRL_RESTRICT_REF cavity );
}


/* ************************************************************************* *
 * *** Implementation of inline and template member functions          ***** *
 * ************************************************************************* */

namespace SIXTRL_CXX_NAMESPACE
{
    /* ===================================================================== *
     * ====  TXYShift< T > :
     * ===================================================================== */

    template< typename T >
    SIXTRL_INLINE bool TCavity< T >::CanAddToBuffer(
        typename TCavity< T >::buffer_t& SIXTRL_RESTRICT_REF buffer,
        SIXTRL_ARGPTR_DEC typename TCavity< T >::size_type*
            SIXTRL_RESTRICT ptr_requ_objects,
        SIXTRL_ARGPTR_DEC typename TCavity< T >::size_type*
            SIXTRL_RESTRICT ptr_requ_slots,
        SIXTRL_ARGPTR_DEC typename TCavity< T >::size_type*
            SIXTRL_RESTRICT ptr_requ_dataptrs ) SIXTRL_NOEXCEPT
    {
        using _this_t = TCavity< T >;
        using  size_t = typename _this_t::size_type;

        static_assert( std::is_trivial< _this_t >::value, "" );
        static_assert( std::is_standard_layout< _this_t >::value, "" );

        SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST
            size_t num_dataptrs = size_t{ 0 };

        SIXTRL_ARGPTR_DEC size_t const* sizes   = nullptr;
        SIXTRL_ARGPTR_DEC size_t const* counts  = nullptr;

        return ::NS(Buffer_can_add_object)( &buffer, sizeof( _this_t ),
            num_dataptrs, sizes, counts, ptr_requ_objects, ptr_requ_slots,
                ptr_requ_dataptrs );
    }

    template< typename T >
    SIXTRL_INLINE SIXTRL_ARGPTR_DEC TCavity< T >*
    TCavity< T >::CreateNewOnBuffer(
        typename TCavity< T >::buffer_t& SIXTRL_RESTRICT_REF buffer )
    {
        using _this_t = TCavity< T >;
        using  size_t = typename _this_t::size_type;
        using  ptr_t  = SIXTRL_ARGPTR_DEC _this_t*;

        static_assert( std::is_trivial< _this_t >::value, "" );
        static_assert( std::is_standard_layout< _this_t >::value, "" );

        SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST
            size_t num_dataptrs = size_t{ 0 };

        SIXTRL_ARGPTR_DEC size_t const* offsets = nullptr;
        SIXTRL_ARGPTR_DEC size_t const* sizes   = nullptr;
        SIXTRL_ARGPTR_DEC size_t const* counts  = nullptr;

        _this_t temp;
        temp.preset();

        return static_cast< ptr_t >( static_cast< uintptr_t >(
            ::NS(Object_get_begin_addr)( ::NS(Buffer_add_object)(
                &buffer, &temp, sizeof( _this_t ), temp.getTypeId(),
                    num_dataptrs, offsets, sizes, counts ) ) ) );
    }

    template< typename T >
    SIXTRL_INLINE SIXTRL_ARGPTR_DEC TCavity< T >* TCavity< T >::AddToBuffer(
        typename TCavity< T >::c_buffer_t& SIXTRL_RESTRICT_REF buffer,
        typename TCavity< T >::value_type const voltage,
        typename TCavity< T >::value_type const frequency,
        typename TCavity< T >::value_type const lag )
    {
        using _this_t = TCavity< T >;
        using  size_t = typename _this_t::size_type;
        using  ptr_t  = SIXTRL_ARGPTR_DEC _this_t*;

        static_assert( std::is_trivial< _this_t >::value, "" );
        static_assert( std::is_standard_layout< _this_t >::value, "" );

        SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST
            size_t num_dataptrs = size_t{ 0 };

        SIXTRL_ARGPTR_DEC size_t const* offsets = nullptr;
        SIXTRL_ARGPTR_DEC size_t const* sizes   = nullptr;
        SIXTRL_ARGPTR_DEC size_t const* counts  = nullptr;

        _this_t temp;
        temp.setVoltage( voltage );
        temp.setFrequency( frequency );
        temp.setLag( lag );

        return static_cast< ptr_t >( static_cast< uintptr_t >(
            ::NS(Object_get_begin_addr)( ::NS(Buffer_add_object)(
                &buffer, &temp, sizeof( _this_t ), temp.getTypeId(),
                    num_dataptrs, offsets, sizes, counts ) ) ) );
    }

    /* ----------------------------------------------------------------- */

    template< typename T >
    SIXTRL_INLINE typename TCavity< T >::type_id_t
    TCavity< T >::getTypeId() const SIXTRL_NOEXCEPT
    {
        return ::NS(OBJECT_TYPE_CAVITY);
    }

    template< typename T >
    SIXTRL_INLINE typename TCavity< T >::value_type
    TCavity< T >::getVoltage() const SIXTRL_NOEXCEPT
    {
        return this->voltage;
    }

    template< typename T >
    SIXTRL_INLINE typename TCavity< T >::value_type
    TCavity< T >::getFrequency() const SIXTRL_NOEXCEPT
    {
        return this->frequency;
    }

    template< typename T >
    SIXTRL_INLINE typename TCavity< T >::value_type
    TCavity< T >::getLag() const SIXTRL_NOEXCEPT
    {
        return this->lag;
    }

    template< typename T >
    SIXTRL_INLINE void TCavity< T >::preset() SIXTRL_NOEXCEPT
    {
        using value_t = typename TCavity< T >::value_type;

        this->setVoltage( value_t{} );
        this->setFrequency( value_t{} );
        this->setLag( value_t{} );
    }

    template< typename T >
    SIXTRL_INLINE void TCavity< T >::setVoltage(
        typename TCavity< T >::value_type const& v ) SIXTRL_NOEXCEPT
    {
        this->voltage = v;
    }

    template< typename T >
    SIXTRL_INLINE void TCavity< T >::setFrequency(
        typename TCavity< T >::value_type const& frequ ) SIXTRL_NOEXCEPT
    {
        this->frequency = frequ;
    }

    template< typename T >
    SIXTRL_INLINE void TCavity< T >::setLag(
        typename TCavity< T >::value_type const& lag ) SIXTRL_NOEXCEPT
    {
        this->lag = lag;
    }

    /* ----------------------------------------------------------------- */

    template< typename T >
    SIXTRL_ARGPTR_DEC TCavity< T >* TCavity_new(
        Buffer& SIXTRL_RESTRICT_REF buffer )
    {
        return T::CreateNewOnBuffer( *( buffer.getCApiPtr() ) );
    }

    template< typename T >
    SIXTRL_ARGPTR_DEC TCavity< T >* TCavity_new(
        SIXTRL_ARGPTR_DEC ::NS(Buffer)* SIXTRL_RESTRICT ptr_buffer )
    {
        return ( ptr_buffer != nullptr )
            ?  ( T::CreateNewOnBuffer( *ptr_buffer ) )
            :  ( nullptr );
    }

    template< typename T >
    SIXTRL_ARGPTR_DEC TCavity< T >* TCavity_add(
        Buffer& SIXTRL_RESTRICT_REF buffer,
        typename TCavity< T >::value_type const voltage,
        typename TCavity< T >::value_type const frequ,
        typename TCavity< T >::value_type const lag )
    {
        return T::AddToBuffer( *( buffer.getCApiPtr() ), voltage, frequ, lag );
    }

    template< typename T >
    SIXTRL_ARGPTR_DEC TCavity< T >* TCavity_add(
        SIXTRL_ARGPTR_DEC ::NS(Buffer)* SIXTRL_RESTRICT ptr_buffer,
        typename TCavity< T >::value_type const voltage,
        typename TCavity< T >::value_type const frequ,
        typename TCavity< T >::value_type const lag )
    {
        return ( ptr_buffer != nullptr )
            ?  ( T::AddToBuffer( *ptr_buffer, voltage, frequ, lag ) )
            :  ( nullptr );
    }

    template< typename T >
    SIXTRL_ARGPTR_DEC TCavity< T >* TCavity_add_copy(
        Buffer& SIXTRL_RESTRICT_REF buffer,
        TCavity< T > const& SIXTRL_RESTRICT_REF orig )
    {
        return TCavity_add( buffer,
            orig.getVoltage(), orig.getFrequency(), orig.getLag() );
    }

    template< typename T >
    SIXTRL_ARGPTR_DEC TCavity< T >* TCavity_add_copy(
        SIXTRL_ARGPTR_DEC ::NS(Buffer)* SIXTRL_RESTRICT ptr_buffer,
        TCavity< T > const& SIXTRL_RESTRICT_REF orig )
    {
        return TCavity_add( ptr_buffer,
            orig.getVoltage(), orig.getFrequency(), orig.getLag() );
    }

    template< typename T > struct ObjectTypeTraits< TCavity< T > >
    {
        SIXTRL_STATIC SIXTRL_INLINE object_type_id_t Type() SIXTRL_NOEXCEPT
        {
            return NS(OBJECT_TYPE_CAVITY);
        }
    };

    /* ===================================================================== *
     * ====  Specialization TCavity< SIXTRL_REAL_T > :
     * ===================================================================== */

    SIXTRL_INLINE bool TCavity< SIXTRL_REAL_T >::CanAddToBuffer(
        TCavity< SIXTRL_REAL_T >::buffer_t& SIXTRL_RESTRICT_REF buffer,
        SIXTRL_ARGPTR_DEC TCavity< SIXTRL_REAL_T >::size_type*
            SIXTRL_RESTRICT ptr_requ_objects,
        SIXTRL_ARGPTR_DEC TCavity< SIXTRL_REAL_T >::size_type*
            SIXTRL_RESTRICT ptr_requ_slots,
        SIXTRL_ARGPTR_DEC TCavity< SIXTRL_REAL_T >::size_type*
            SIXTRL_RESTRICT ptr_requ_dataptrs ) SIXTRL_NOEXCEPT
    {
        return ::NS(Cavity_can_be_added)( &buffer, ptr_requ_objects,
            ptr_requ_slots, ptr_requ_dataptrs );
    }

    SIXTRL_INLINE SIXTRL_ARGPTR_DEC TCavity< SIXTRL_REAL_T >*
    TCavity< SIXTRL_REAL_T >::CreateNewOnBuffer(
        TCavity< SIXTRL_REAL_T >::buffer_t& SIXTRL_RESTRICT_REF buffer )
    {
        using ptr_t   = SIXTRL_ARGPTR_DEC TCavity< SIXTRL_REAL_T >*;
        return static_cast< ptr_t >( ::NS(Cavity_new)( buffer.getCApiPtr() ) );
    }

    SIXTRL_INLINE SIXTRL_ARGPTR_DEC TCavity< SIXTRL_REAL_T >*
    TCavity< SIXTRL_REAL_T >::AddToBuffer(
        TCavity< SIXTRL_REAL_T >::c_buffer_t& SIXTRL_RESTRICT_REF buffer,
        TCavity< SIXTRL_REAL_T >::value_type const voltage,
        TCavity< SIXTRL_REAL_T >::value_type const frequency,
        TCavity< SIXTRL_REAL_T >::value_type const lag )
    {
        using ptr_t = SIXTRL_ARGPTR_DEC TCavity< SIXTRL_REAL_T >*;
        return static_cast< ptr_t >( ::NS(Cavity_add)(
            &buffer, voltage, frequency, lag ) );
    }

    SIXTRL_ARGPTR_DEC TCavity< SIXTRL_REAL_T >::c_api_t const*
    TCavity< SIXTRL_REAL_T >::getCApiPtr() const SIXTRL_NOEXCEPT
    {
        using ptr_t = TCavity< SIXTRL_REAL_T >::c_api_t const*;
        return reinterpret_cast< ptr_t >( this );
    }

    SIXTRL_ARGPTR_DEC TCavity< SIXTRL_REAL_T >::c_api_t*
    TCavity< SIXTRL_REAL_T >::getCApiPtr() SIXTRL_NOEXCEPT
    {
        using _this_t = TCavity< SIXTRL_REAL_T >;
        using   ptr_t = SIXTRL_ARGPTR_DEC _this_t::c_api_t*;

        return const_cast< ptr_t >( static_cast< _this_t const& >(
            *this ).getCApiPtr() );
    }

    SIXTRL_INLINE TCavity< SIXTRL_REAL_T >::type_id_t
    TCavity< SIXTRL_REAL_T >::getTypeId() const SIXTRL_NOEXCEPT
    {
        return ::NS(OBJECT_TYPE_CAVITY);
    }

    SIXTRL_INLINE TCavity< SIXTRL_REAL_T >::value_type
    TCavity< SIXTRL_REAL_T >::getVoltage() const SIXTRL_NOEXCEPT
    {
        return ::NS(Cavity_get_voltage)( this->getCApiPtr() );
    }

    SIXTRL_INLINE TCavity< SIXTRL_REAL_T >::value_type
    TCavity< SIXTRL_REAL_T >::getFrequency() const SIXTRL_NOEXCEPT
    {
        return ::NS(Cavity_get_frequency)( this->getCApiPtr() );
    }

    SIXTRL_INLINE TCavity< SIXTRL_REAL_T >::value_type
    TCavity< SIXTRL_REAL_T >::getLag() const SIXTRL_NOEXCEPT
    {
        return ::NS(Cavity_get_lag)( this->getCApiPtr() );
    }

    SIXTRL_INLINE void TCavity< SIXTRL_REAL_T >::preset() SIXTRL_NOEXCEPT
    {
        ::NS(Cavity_preset)( this->getCApiPtr() );
        return;
    }

    SIXTRL_INLINE void TCavity< SIXTRL_REAL_T >::setVoltage(
        TCavity< SIXTRL_REAL_T >::value_type const v ) SIXTRL_NOEXCEPT
    {
        ::NS(Cavity_set_voltage)( this->getCApiPtr(), v );
        return;
    }

    SIXTRL_INLINE void TCavity< SIXTRL_REAL_T >::setFrequency(
        TCavity< SIXTRL_REAL_T >::value_type const freq ) SIXTRL_NOEXCEPT
    {
        ::NS(Cavity_set_frequency)( this->getCApiPtr(), freq );
        return;
    }

    SIXTRL_INLINE void TCavity< SIXTRL_REAL_T >::setLag(
        TCavity< SIXTRL_REAL_T >::value_type const lag ) SIXTRL_NOEXCEPT
    {
        ::NS(Cavity_set_lag)( this->getCApiPtr(), lag );
        return;
    }

    /* --------------------------------------------------------------------- */

    SIXTRL_ARGPTR_DEC Cavity* Cavity_new( Buffer& SIXTRL_RESTRICT_REF buffer )
    {
        using ptr_t = SIXTRL_ARGPTR_DEC TCavity< SIXTRL_REAL_T >*;
        return static_cast< ptr_t >( ::NS(Cavity_new)( buffer.getCApiPtr() ) );
    }

    SIXTRL_ARGPTR_DEC Cavity*
    Cavity_new( SIXTRL_ARGPTR_DEC ::NS(Buffer)* SIXTRL_RESTRICT ptr_buffer )
    {
        using ptr_t = SIXTRL_ARGPTR_DEC TCavity< SIXTRL_REAL_T >*;
        return static_cast< ptr_t >( ::NS(Cavity_new)( ptr_buffer ) );
    }

    SIXTRL_ARGPTR_DEC Cavity* Cavity_add(
        Buffer& SIXTRL_RESTRICT_REF buffer, Cavity::value_type const voltage,
        Cavity::value_type const frequency, Cavity::value_type const lag )
    {
        using ptr_t = SIXTRL_ARGPTR_DEC TCavity< SIXTRL_REAL_T >*;
        return static_cast< ptr_t >( ::NS(Cavity_add)(
            buffer.getCApiPtr(), voltage, frequency, lag ) );
    }

    SIXTRL_ARGPTR_DEC Cavity* Cavity_add(
        SIXTRL_ARGPTR_DEC ::NS(Buffer)* SIXTRL_RESTRICT ptr_buffer,
        Cavity::value_type const voltage, Cavity::value_type const frequency,
        Cavity::value_type const lag )
    {
        using ptr_t = SIXTRL_ARGPTR_DEC TCavity< SIXTRL_REAL_T >*;
        return static_cast< ptr_t >( ::NS(Cavity_add)(
            ptr_buffer, voltage, frequency, lag ) );
    }

    SIXTRL_ARGPTR_DEC Cavity* Cavity_add_copy(
        Buffer& SIXTRL_RESTRICT_REF buffer,
        Cavity const& SIXTRL_RESTRICT_REF orig )
    {
        return Cavity_add( buffer,
           orig.getVoltage(), orig.getFrequency(), orig.getLag() );
    }

    SIXTRL_ARGPTR_DEC Cavity* Cavity_add_copy(
        SIXTRL_ARGPTR_DEC ::NS(Buffer)* SIXTRL_RESTRICT ptr_buffer,
        Cavity const& SIXTRL_RESTRICT_REF orig )
    {
        return Cavity_add( ptr_buffer,
           orig.getVoltage(), orig.getFrequency(), orig.getLag() );
    }

    template<> struct ObjectTypeTraits< Cavity >
    {
        SIXTRL_STATIC SIXTRL_INLINE object_type_id_t Type() SIXTRL_NOEXCEPT
        {
            return NS(OBJECT_TYPE_CAVITY);
        }
    };

    template<> struct ObjectTypeTraits< ::NS(Cavity) >
    {
        SIXTRL_STATIC SIXTRL_INLINE object_type_id_t Type() SIXTRL_NOEXCEPT
        {
            return NS(OBJECT_TYPE_CAVITY);
        }
    };
}

#endif /* defined( __cplusplus ) */

#endif /* CXX_SIXTRACKLIB_COMMON_BE_CAVITY_BEAM_ELEMENT_BE_CAVITY_HPP__ */

/* end: sixtracklib/common/be_cavity/be_cavity.hpp */
