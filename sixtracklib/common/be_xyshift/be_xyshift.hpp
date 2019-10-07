#ifndef CXX_SIXTRACKLIB_COMMON_BE_XYSHIFT_BE__HPP__
#define CXX_SIXTRACKLIB_COMMON_BE_XYSHIFT_BE__HPP__

#if defined( __cplusplus )

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <cstddef>
    #include <cstdint>
    #include <cstdlib>
    #include <type_traits>
    #include <utility>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/buffer.hpp"
    #include "sixtracklib/common/be_xyshift/be_xyshift.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */


namespace SIXTRL_CXX_NAMESPACE
{
    /* ===================================================================== *
     * ====  TXYShift< T > :
     * ===================================================================== */

    template< typename T >
    struct TXYShift
    {
        using value_type = T;
        using type_id_t  = NS(object_type_id_t);
        using size_type  = NS(buffer_size_t);
        using buffer_t   = ::NS(Buffer);

        SIXTRL_FN TXYShift() = default;
        SIXTRL_FN TXYShift( TXYShift< T > const& other ) = default;
        SIXTRL_FN TXYShift( TXYShift< T >&& other ) = default;

        SIXTRL_FN TXYShift< T >& operator=(
            TXYShift< T > const& rhs ) = default;

        SIXTRL_FN TXYShift< T >& operator=( TXYShift< T >&& rhs ) = default;

        SIXTRL_FN ~TXYShift() = default;

        /* ---------------------------------------------------------------- */

        SIXTRL_FN SIXTRL_STATIC bool CanAddToBuffer(
            buffer_t& SIXTRL_RESTRICT_REF buffer,
            SIXTRL_ARGPTR_DEC size_type* SIXTRL_RESTRICT
                ptr_requ_objects  = nullptr,
            SIXTRL_ARGPTR_DEC size_type* SIXTRL_RESTRICT
                ptr_requ_slots    = nullptr,
            SIXTRL_ARGPTR_DEC size_type* SIXTRL_RESTRICT
                ptr_requ_dataptrs = nullptr ) SIXTRL_NOEXCEPT;

        SIXTRL_FN SIXTRL_STATIC SIXTRL_ARGPTR_DEC TXYShift< T >*
        CreateNewOnBuffer( buffer_t& SIXTRL_RESTRICT_REF buffer );


        SIXTRL_FN SIXTRL_STATIC SIXTRL_ARGPTR_DEC TXYShift< T >* AddToBuffer(
            buffer_t& SIXTRL_RESTRICT_REF buffer,
            value_type const& dx, value_type const& dy );

        /* ---------------------------------------------------------------- */

        SIXTRL_FN type_id_t getTypeId()      const SIXTRL_NOEXCEPT;
        SIXTRL_FN size_type getNumDataPtrs() const SIXTRL_NOEXCEPT;

        SIXTRL_FN value_type const& getDx() const SIXTRL_NOEXCEPT;
        SIXTRL_FN value_type const& getDy() const SIXTRL_NOEXCEPT;

        SIXTRL_FN void preset() SIXTRL_NOEXCEPT;

        SIXTRL_FN void setDx( value_type const& dx ) SIXTRL_NOEXCEPT;
        SIXTRL_FN void setDy( value_type const& dy ) SIXTRL_NOEXCEPT;

        /* ----------------------------------------------------------------- */

        value_type dx SIXTRL_ALIGN( 8 );
        value_type dy SIXTRL_ALIGN( 8 );
    };

    template< typename T >
    SIXTRL_ARGPTR_DEC TXYShift< T >* TXYShift_new( Buffer& buffer );

    template< typename T >
    SIXTRL_ARGPTR_DEC TXYShift< T >* TXYShift_new(
        SIXTRL_ARGPTR_DEC ::NS(Buffer)* SIXTRL_RESTRICT ptr_buffer );

    template< typename T >
    SIXTRL_ARGPTR_DEC TXYShift< T >* TXYShift_add( Buffer& buffer,
        typename TXYShift< T >::value_type const& dx,
        typename TXYShift< T >::value_type const& dy );

    template< typename T >
    SIXTRL_ARGPTR_DEC TXYShift< T >* TXYShift_add(
        SIXTRL_ARGPTR_DEC ::NS(Buffer)* SIXTRL_RESTRICT ptr_buffer,
        typename TXYShift< T >::value_type const& dx,
        typename TXYShift< T >::value_type const& dy );

    /* ===================================================================== *
     * ====  Specialization TXYShift< NS(xyshift_real_t) > :
     * ===================================================================== */


    template<> struct TXYShift< NS(xyshift_real_t) > : public ::NS(XYShift)
    {
        using value_type = NS(xyshift_real_t);
        using type_id_t  = NS(object_type_id_t);
        using size_type  = NS(buffer_size_t);
        using buffer_t   = ::NS(Buffer);
        using c_api_t    = ::NS(XYShift);

        SIXTRL_FN TXYShift() = default;
        SIXTRL_FN TXYShift( TXYShift< NS(xyshift_real_t) > const& other ) = default;
        SIXTRL_FN TXYShift( TXYShift< NS(xyshift_real_t) >&& other ) = default;

        SIXTRL_FN TXYShift< NS(xyshift_real_t) >& operator=(
            TXYShift< NS(xyshift_real_t) > const& rhs ) = default;

        SIXTRL_FN TXYShift< NS(xyshift_real_t) >& operator=(
            TXYShift< NS(xyshift_real_t) >&& rhs ) = default;

        SIXTRL_FN ~TXYShift() = default;

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
        SIXTRL_ARGPTR_DEC TXYShift< NS(xyshift_real_t) >*
        CreateNewOnBuffer( buffer_t& SIXTRL_RESTRICT_REF buffer );

        SIXTRL_FN SIXTRL_STATIC
        SIXTRL_ARGPTR_DEC TXYShift< NS(xyshift_real_t) >* AddToBuffer(
            buffer_t& SIXTRL_RESTRICT_REF buffer,
            value_type const dx, value_type const dy )
        {
            using ptr_t = SIXTRL_ARGPTR_DEC TXYShift< ::NS(xyshift_real_t) >*;
            return static_cast< ptr_t >( ::NS(XYShift_add)( &buffer, dx, dy ) );
        }

        /* ----------------------------------------------------------------- */

        SIXTRL_ARGPTR_DEC c_api_t const* getCApiPtr() const SIXTRL_NOEXCEPT;
        SIXTRL_ARGPTR_DEC c_api_t* getCApiPtr() SIXTRL_NOEXCEPT;

        /* ----------------------------------------------------------------- */

        SIXTRL_FN type_id_t getTypeId() const SIXTRL_NOEXCEPT;
        SIXTRL_FN value_type getDx() const SIXTRL_NOEXCEPT;
        SIXTRL_FN value_type getDy() const SIXTRL_NOEXCEPT;

        SIXTRL_FN void preset() SIXTRL_NOEXCEPT;
        SIXTRL_FN void setDx( value_type const dx ) SIXTRL_NOEXCEPT;
        SIXTRL_FN void setDy( value_type const dy ) SIXTRL_NOEXCEPT;
    };

    using XYShift = TXYShift< NS(xyshift_real_t) >;

    SIXTRL_ARGPTR_DEC XYShift* XYShift_new( Buffer& buffer );

    SIXTRL_ARGPTR_DEC XYShift* XYShift_new(
        SIXTRL_ARGPTR_DEC ::NS(Buffer)* SIXTRL_RESTRICT ptr_buffer );

    SIXTRL_ARGPTR_DEC XYShift* XYShift_add( Buffer& buffer,
        NS(xyshift_real_t) const dx, NS(xyshift_real_t) const dy );

    SIXTRL_ARGPTR_DEC XYShift* XYShift_add(
        SIXTRL_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT ptr_buffer,
        NS(xyshift_real_t) const dx, NS(xyshift_real_t) const dy );
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
    SIXTRL_INLINE bool TXYShift< T >::CanAddToBuffer(
        buffer_t& SIXTRL_RESTRICT_REF buffer,
        SIXTRL_ARGPTR_DEC typename TXYShift< T >::size_type*
            SIXTRL_RESTRICT ptr_req_objects,
        SIXTRL_ARGPTR_DEC typename TXYShift< T >::size_type*
            SIXTRL_RESTRICT ptr_req_slots,
        SIXTRL_ARGPTR_DEC typename TXYShift< T >::size_type*
            SIXTRL_RESTRICT ptr_req_dataptrs ) SIXTRL_NOEXCEPT
    {
        using _this_t = TXYShift< T >;
        using  size_t = typename _this_t::size_type;

        static_assert( std::is_trivial< _this_t >::value, "" );
        static_assert( std::is_standard_layout< _this_t >::value, "" );

        SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST
            size_t num_dataptrs = size_t{ 0 };

        SIXTRL_ARGPTR_DEC size_t const* sizes  = nullptr;
        SIXTRL_ARGPTR_DEC size_t const* counts = nullptr;

        return ::NS(Buffer_can_add_object)( &buffer, sizeof( _this_t ),
            num_dataptrs, sizes, counts, ptr_req_objects,
                ptr_req_slots, ptr_req_dataptrs );
    }

    template< typename T >
    SIXTRL_INLINE SIXTRL_ARGPTR_DEC TXYShift< T >*
    TXYShift< T >::CreateNewOnBuffer(
        typename TXYShift< T >::buffer_t& SIXTRL_RESTRICT_REF buffer )
    {
        using _this_t = TXYShift< T >;
        using size_t  = typename _this_t::size_type;
        using ptr_t   = SIXTRL_ARGPTR_DEC _this_t*;

        static_assert( std::is_trivial< _this_t >::value, "" );
        static_assert( std::is_standard_layout< _this_t >::value, "" );

        SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST
            size_t num_dataptrs = size_t{ 0 };

        SIXTRL_ARGPTR_DEC size_t const* offsets = nullptr;
        SIXTRL_ARGPTR_DEC size_t const* sizes   = nullptr;
        SIXTRL_ARGPTR_DEC size_t const* counts  = nullptr;

        _this_t temp;
        temp.preset();

        return reinterpret_cast< ptr_t >( ::NS(Object_get_begin_addr)(
            ::NS(Buffer_add_object)( &buffer, &temp, sizeof( _this_t ),
                temp.getTypeId(), num_dataptrs, offsets, sizes, counts ) ) );
    }

    template< typename T >
    SIXTRL_INLINE SIXTRL_ARGPTR_DEC TXYShift< T >* TXYShift< T >::AddToBuffer(
        typename TXYShift< T >::buffer_t& SIXTRL_RESTRICT_REF buffer,
        typename TXYShift< T >::value_type const& SIXTRL_RESTRICT_REF dx,
        typename TXYShift< T >::value_type const& SIXTRL_RESTRICT_REF dy )
    {
        using _this_t = TXYShift< T >;
        using size_t  = typename _this_t::size_type;
        using ptr_t   = SIXTRL_ARGPTR_DEC _this_t*;

        static_assert( std::is_trivial< _this_t >::value, "" );
        static_assert( std::is_standard_layout< _this_t >::value, "" );

        SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST
            size_t num_dataptrs = size_t{ 0 };

        SIXTRL_ARGPTR_DEC size_t const* offsets = nullptr;
        SIXTRL_ARGPTR_DEC size_t const* sizes   = nullptr;
        SIXTRL_ARGPTR_DEC size_t const* counts  = nullptr;

        _this_t temp;
        temp.setDx( dx);
        temp.setDy( dy );

        return reinterpret_cast< ptr_t >( ::NS(Object_get_begin_addr)(
            ::NS(Buffer_add_object)( &buffer, &temp, sizeof( _this_t ),
                temp.getTypeId(), num_dataptrs, offsets, sizes, counts ) ) );
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    template< typename T >
    SIXTRL_INLINE typename TXYShift< T >::type_id_t
    TXYShift< T >::getTypeId() const SIXTRL_NOEXCEPT
    {
        return ::NS(OBJECT_TYPE_XYSHIFT);
    }

    template< typename T >
    SIXTRL_INLINE typename TXYShift< T >::value_type const&
    TXYShift< T >::getDx() const SIXTRL_NOEXCEPT
    {
        return this->dx;
    }

    template< typename T >
    SIXTRL_INLINE typename TXYShift< T >::value_type const&
    TXYShift< T >::getDy() const SIXTRL_NOEXCEPT
    {
        return this->dy;
    }

    template< typename T >
    SIXTRL_INLINE void TXYShift< T >::preset() SIXTRL_NOEXCEPT
    {
        using value_t = typename TXYShift< T >::value_type;

        this->setDx( value_t{} );
        this->setDy( value_t{} );

        return;
    }

    template< typename T >
    SIXTRL_INLINE void TXYShift< T >::setDx(
        typename TXYShift< T >::value_type const& dx ) SIXTRL_NOEXCEPT
    {
        this->dx = dx;
        return;
    }

    template< typename T >
    SIXTRL_INLINE void TXYShift< T >::setDy(
        typename TXYShift< T >::value_type const& dy ) SIXTRL_NOEXCEPT
    {
        this->dy = dy;
        return;
    }

    /* --------------------------------------------------------------------- */

    template< typename T >
    SIXTRL_ARGPTR_DEC TXYShift< T >* TXYShift_new( Buffer& buffer )
    {
        return T::CreateNewOnBuffer( *( buffer.getCApiPtr() ) );
    }

    template< typename T >
    SIXTRL_ARGPTR_DEC TXYShift< T >* TXYShift_new(
        SIXTRL_ARGPTR_DEC ::NS(Buffer)* SIXTRL_RESTRICT ptr_buffer )
    {
        return ( ptr_buffer != nullptr )
            ? ( T::CreateNewOnBuffer( *ptr_buffer ) )
            : ( nullptr );
    }

    template< typename T >
    SIXTRL_ARGPTR_DEC TXYShift< T >* TXYShift_add( Buffer& buffer,
        typename TXYShift< T >::value_type const& dx,
        typename TXYShift< T >::value_type const& dy )
    {
        return T::AddToBuffer( *( buffer.getCApiPtr() ), dx, dy );
    }

    template< typename T >
    SIXTRL_ARGPTR_DEC TXYShift< T >* TXYShift_add(
        SIXTRL_ARGPTR_DEC ::NS(Buffer)* SIXTRL_RESTRICT ptr_buffer,
        typename TXYShift< T >::value_type const& dx,
        typename TXYShift< T >::value_type const& dy )
    {
        return ( ptr_buffer != nullptr )
            ? ( T::AddToBuffer( *ptr_buffer, dx, dy ) )
            : ( nullptr );
    }

    /* ===================================================================== *
     * ====  Specialization TXYShift<  > :
     * ===================================================================== */

    SIXTRL_INLINE bool TXYShift< ::NS(xyshift_real_t) >::CanAddToBuffer(
        TXYShift< ::NS(xyshift_real_t) >::buffer_t& SIXTRL_RESTRICT_REF buffer,
        SIXTRL_ARGPTR_DEC TXYShift< NS(xyshift_real_t) >::size_type*
            SIXTRL_RESTRICT ptr_requ_objects,
        SIXTRL_ARGPTR_DEC TXYShift< NS(xyshift_real_t) >::size_type*
            SIXTRL_RESTRICT ptr_requ_slots,
        SIXTRL_ARGPTR_DEC TXYShift< NS(xyshift_real_t) >::size_type*
            SIXTRL_RESTRICT ptr_requ_dataptrs ) SIXTRL_NOEXCEPT
    {
        return ::NS(XYShift_can_be_added)(
            &buffer, ptr_requ_objects, ptr_requ_slots, ptr_requ_dataptrs );
    }

    SIXTRL_INLINE SIXTRL_ARGPTR_DEC TXYShift< NS(xyshift_real_t) >*
    TXYShift< NS(xyshift_real_t) >::CreateNewOnBuffer(
        TXYShift< NS(xyshift_real_t) >::buffer_t& SIXTRL_RESTRICT_REF buffer )
    {
        using ptr_t = SIXTRL_ARGPTR_DEC TXYShift< NS(xyshift_real_t) >*;
        return static_cast< ptr_t >( ::NS(XYShift_new)( &buffer ) );
    }

    /* ----------------------------------------------------------------- */

    SIXTRL_ARGPTR_DEC TXYShift< NS(xyshift_real_t) >::c_api_t const*
    TXYShift< NS(xyshift_real_t) >::getCApiPtr() const SIXTRL_NOEXCEPT
    {
        using ptr_t = TXYShift< NS(xyshift_real_t) >::c_api_t const*;
        return reinterpret_cast< ptr_t >( this );
    }

    SIXTRL_ARGPTR_DEC TXYShift< NS(xyshift_real_t) >::c_api_t*
    TXYShift< NS(xyshift_real_t) >::getCApiPtr() SIXTRL_NOEXCEPT
    {
        using _this_t = TXYShift< NS(xyshift_real_t) >;
        using   ptr_t = _this_t::c_api_t*;

        return const_cast< ptr_t >( static_cast< _this_t const& >(
            *this ).getCApiPtr() );
    }

    /* ----------------------------------------------------------------- */

    SIXTRL_INLINE TXYShift< NS(xyshift_real_t) >::type_id_t
    TXYShift< NS(xyshift_real_t) >::getTypeId() const SIXTRL_NOEXCEPT
    {
        return ::NS(OBJECT_TYPE_XYSHIFT);
    }

    SIXTRL_INLINE TXYShift< NS(xyshift_real_t) >::value_type
    TXYShift< NS(xyshift_real_t) >::getDx() const SIXTRL_NOEXCEPT
    {
        return ::NS(XYShift_get_dx)( this->getCApiPtr() );
    }

    SIXTRL_INLINE TXYShift< NS(xyshift_real_t) >::value_type
    TXYShift< NS(xyshift_real_t) >::getDy() const SIXTRL_NOEXCEPT
    {
        return ::NS(XYShift_get_dy)( this->getCApiPtr() );
    }

    SIXTRL_INLINE void TXYShift< NS(xyshift_real_t) >::preset() SIXTRL_NOEXCEPT
    {
        ::NS(XYShift_preset)( this->getCApiPtr() );
    }

    SIXTRL_INLINE void TXYShift< NS(xyshift_real_t) >::setDx(
        TXYShift< NS(xyshift_real_t) >::value_type const dx ) SIXTRL_NOEXCEPT
    {
        ::NS(XYShift_set_dx)( this->getCApiPtr(), dx );
    }

    SIXTRL_INLINE void TXYShift< NS(xyshift_real_t) >::setDy(
        TXYShift< NS(xyshift_real_t) >::value_type const dy ) SIXTRL_NOEXCEPT
    {
        ::NS(XYShift_set_dy)( this->getCApiPtr(), dy );
    }

    /* --------------------------------------------------------------------- */

    SIXTRL_INLINE SIXTRL_ARGPTR_DEC XYShift* XYShift_new( Buffer& buffer )
    {
        using ptr_t = SIXTRL_ARGPTR_DEC XYShift*;
        return static_cast< ptr_t >(
            ::NS(XYShift_new)( buffer.getCApiPtr() ) );
    }

    SIXTRL_INLINE SIXTRL_ARGPTR_DEC XYShift* XYShift_new(
        SIXTRL_ARGPTR_DEC ::NS(Buffer)* SIXTRL_RESTRICT ptr_buffer )
    {
        using ptr_t = SIXTRL_ARGPTR_DEC XYShift*;
        return static_cast< ptr_t >( ::NS(XYShift_new)( ptr_buffer ) );
    }

    SIXTRL_INLINE SIXTRL_ARGPTR_DEC XYShift* XYShift_add( Buffer& buffer,
        NS(xyshift_real_t) const dx, NS(xyshift_real_t) const dy )
    {
        using ptr_t = SIXTRL_ARGPTR_DEC XYShift*;
        return static_cast< ptr_t >(
            ::NS(XYShift_add)( buffer.getCApiPtr(), dx, dy ) );
    }

    SIXTRL_INLINE SIXTRL_ARGPTR_DEC XYShift* XYShift_add(
        SIXTRL_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT ptr_buffer,
        NS(xyshift_real_t) const dx, NS(xyshift_real_t) const dy )
    {
        using ptr_t = SIXTRL_ARGPTR_DEC XYShift*;
        return static_cast< ptr_t >( ::NS(XYShift_add)( ptr_buffer, dx, dy ) );
    }
}

#endif /* defined( __cplusplus ) */

#endif /* CXX_SIXTRACKLIB_COMMON_BE_XYSHIFT_BE__HPP__ */

/* end:  sixtracklib/common/be_xyshift/be_xyshift.hpp */
