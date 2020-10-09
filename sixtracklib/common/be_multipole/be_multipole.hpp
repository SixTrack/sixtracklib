#ifndef CXX_SIXTRACKLIB_COMMON_BE_MULTIPOLE_BEAM_ELEM_BE_MULTIPOLE_HPP__
#define CXX_SIXTRACKLIB_COMMON_BE_MULTIPOLE_BEAM_ELEM_BE_MULTIPOLE_HPP__

#if defined( __cplusplus )

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <algorithm>
    #include <cstddef>
    #include <cstdint>
    #include <cstdlib>
    #include <iterator>
    #include <type_traits>
    #include <utility>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/buffer.hpp"
    #include "sixtracklib/common/be_multipole/be_multipole.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

namespace SIXTRL_CXX_NAMESPACE
{
    /* ===================================================================== *
     * ====  TMultipole< T >:
     * ===================================================================== */

    template< typename T >
    struct TMultipole
    {
        using value_type = T;
        using type_id_t  = NS(object_type_id_t);
        using order_type = NS(multipole_order_t);
        using size_type  = NS(buffer_size_t);
        using buffer_t   = ::NS(Buffer);

        SIXTRL_FN TMultipole() = default;
        SIXTRL_FN TMultipole( TMultipole< T > const& other ) = default;
        SIXTRL_FN TMultipole( TMultipole< T >&& other ) = default;

        SIXTRL_FN TMultipole< T >& operator=(
            TMultipole< T > const& rhs ) = default;

        SIXTRL_FN TMultipole< T >& operator=(
            TMultipole< T >&& rhs ) = default;

        SIXTRL_FN ~TMultipole() = default;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        SIXTRL_FN SIXTRL_STATIC bool CanAddToBuffer(
            buffer_t& SIXTRL_RESTRICT_REF buffer,
            order_type const order,
            SIXTRL_ARGPTR_DEC size_type* SIXTRL_RESTRICT ptr_requ_objects,
            SIXTRL_ARGPTR_DEC size_type* SIXTRL_RESTRICT ptr_requ_slots,
            SIXTRL_ARGPTR_DEC size_type* SIXTRL_RESTRICT ptr_requ_dataptrs
            ) SIXTRL_NOEXCEPT;

        SIXTRL_FN SIXTRL_STATIC SIXTRL_ARGPTR_DEC TMultipole< T >*
        CreateNewOnBuffer( buffer_t& SIXTRL_RESTRICT_REF buffer,
                           order_type const order );

        SIXTRL_FN SIXTRL_STATIC SIXTRL_ARGPTR_DEC TMultipole< T >*
        AddToBuffer( buffer_t& SIXTRL_RESTRICT_REF buffer,
            order_type const order  = order_type{ 0 },
            value_type const length = value_type{ 0.0 },
            value_type const hxl    = value_type{ 0.0 },
            value_type const hyl    = value_type{ 0.0 },
            SIXTRL_DATAPTR_DEC value_type const* SIXTRL_RESTRICT bal = nullptr );

        SIXTRL_FN SIXTRL_STATIC value_type Factorial(
            size_type n ) SIXTRL_NOEXCEPT;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        SIXTRL_FN type_id_t getTypeId()      const SIXTRL_NOEXCEPT;
        SIXTRL_FN size_type getNumDataPtrs() const SIXTRL_NOEXCEPT;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        SIXTRL_FN void preset() SIXTRL_NOEXCEPT;


        SIXTRL_FN value_type getLength() const SIXTRL_NOEXCEPT;
        SIXTRL_FN value_type getHxl() const SIXTRL_NOEXCEPT;
        SIXTRL_FN value_type getHyl() const SIXTRL_NOEXCEPT;
        SIXTRL_FN order_type getOrder() const SIXTRL_NOEXCEPT;
        SIXTRL_FN size_type  getBalSize() const SIXTRL_NOEXCEPT;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        SIXTRL_FN value_type getKnlValue(
            size_type const index ) const SIXTRL_NOEXCEPT;

        SIXTRL_FN value_type getKslValue(
            size_type const index ) const SIXTRL_NOEXCEPT;

        SIXTRL_FN value_type getBalValue(
            size_type const index ) const SIXTRL_NOEXCEPT;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        template< typename Iter >
        SIXTRL_FN void setKnl( Iter knl_begin ) SIXTRL_NOEXCEPT;

        template< typename Iter >
        SIXTRL_FN void setKsl( Iter ksl_begin ) SIXTRL_NOEXCEPT;

        template< typename Iter >
        SIXTRL_FN void setBal( Iter bal_begin ) SIXTRL_NOEXCEPT;

        SIXTRL_FN void setKnlValue(
            value_type knl_value, size_type const index ) SIXTRL_NOEXCEPT;

        SIXTRL_FN void setKslValue(
            value_type ksl_value, size_type const index ) SIXTRL_NOEXCEPT;

        SIXTRL_FN void setBalValue(
            value_type bal_value, size_type const index ) SIXTRL_NOEXCEPT;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        SIXTRL_FN SIXTRL_DATAPTR_DEC value_type const*
        begin() const SIXTRL_NOEXCEPT;

        SIXTRL_FN SIXTRL_DATAPTR_DEC value_type const*
        end() const SIXTRL_NOEXCEPT;

        SIXTRL_FN SIXTRL_DATAPTR_DEC value_type*
        begin() SIXTRL_NOEXCEPT;

        SIXTRL_FN SIXTRL_DATAPTR_DEC value_type*
        end() SIXTRL_NOEXCEPT;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        template< typename CPtr = SIXTRL_DATAPTR_DEC value_type const* >
        SIXTRL_FN CPtr getBalBegin() const SIXTRL_NOEXCEPT;

        template< typename CPtr = SIXTRL_DATAPTR_DEC value_type const* >
        SIXTRL_FN CPtr getBalEnd()   const SIXTRL_NOEXCEPT;

        template< typename Ptr = SIXTRL_DATAPTR_DEC value_type* >
        SIXTRL_FN Ptr getBalBegin() SIXTRL_NOEXCEPT;

        template< typename Ptr = SIXTRL_DATAPTR_DEC value_type* >
        SIXTRL_FN Ptr getBalEnd() SIXTRL_NOEXCEPT;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        SIXTRL_FN void setOrder( order_type const order ) SIXTRL_NOEXCEPT;
        SIXTRL_FN void setLength( value_type const length ) SIXTRL_NOEXCEPT;
        SIXTRL_FN void setHxl( value_type const hxl ) SIXTRL_NOEXCEPT;
        SIXTRL_FN void setHyl( value_type const hyl ) SIXTRL_NOEXCEPT;

        template< typename Ptr >
        SIXTRL_FN void assignBal( Ptr ptr_to_bal ) SIXTRL_NOEXCEPT;

        order_type order   SIXTRL_ALIGN( 8 );
        value_type length  SIXTRL_ALIGN( 8 );
        value_type hxl     SIXTRL_ALIGN( 8 );
        value_type hyl     SIXTRL_ALIGN( 8 );

        SIXTRL_DATAPTR_DEC value_type* SIXTRL_RESTRICT bal SIXTRL_ALIGN( 8 );
    };

    template< typename T >
    SIXTRL_ARGPTR_DEC TMultipole< T >*
    TMultipole_new( Buffer& SIXTRL_RESTRICT_REF buffer,
                    typename TMultipole< T >::order_type const order );

    template< typename T >
    SIXTRL_ARGPTR_DEC TMultipole< T >*
    TMultipole_new( SIXTRL_ARGPTR_DEC ::NS(Buffer)* SIXTRL_RESTRICT ptr_buffer,
                    typename TMultipole< T >::order_type const order );

    template< typename T >
    SIXTRL_ARGPTR_DEC TMultipole< T >*
    TMultipole_add( Buffer& SIXTRL_RESTRICT_REF buffer,
        typename TMultipole< T >::order_type const order,
        typename TMultipole< T >::value_type const& length =
            typename TMultipole< T >::value_type{},
        typename TMultipole< T >::value_type const& hxl =
            typename TMultipole< T >::value_type{},
        typename TMultipole< T >::value_type const& hyl =
            typename TMultipole< T >::value_type{},
        SIXTRL_DATAPTR_DEC typename TMultipole< T >::value_type const*
            SIXTRL_RESTRICT bal = nullptr );

    template< typename T >
    SIXTRL_ARGPTR_DEC TMultipole< T >*
    TMultipole_add(
        SIXTRL_ARGPTR_DEC ::NS(Buffer)* SIXTRL_RESTRICT ptr_buffer,
        typename TMultipole< T >::order_type const order,
        typename TMultipole< T >::value_type const& length =
            typename TMultipole< T >::value_type{},
        typename TMultipole< T >::value_type const& hxl =
            typename TMultipole< T >::value_type{},
        typename TMultipole< T >::value_type const& hyl =
            typename TMultipole< T >::value_type{},
        SIXTRL_DATAPTR_DEC typename TMultipole< T >::value_type const*
            SIXTRL_RESTRICT bal = nullptr );

    /* ===================================================================== *
     * ====  TMultipole< SIXTRL_REAL_T >:
     * ===================================================================== */

    template<>
    struct TMultipole< SIXTRL_REAL_T > : public ::NS(Multipole)
    {
        using value_type = SIXTRL_REAL_T;
        using type_id_t  = NS(object_type_id_t);
        using order_type = SIXTRL_INT64_T;
        using order_t    = order_type;
        using size_type  = NS(buffer_size_t);
        using buffer_t   = ::NS(Buffer);
        using c_api_t    = ::NS(Multipole);

        SIXTRL_FN TMultipole() = default;

        SIXTRL_FN TMultipole(
            TMultipole< SIXTRL_REAL_T > const& other ) = default;

        SIXTRL_FN TMultipole(
            TMultipole< SIXTRL_REAL_T >&& other ) = default;

        SIXTRL_FN TMultipole< SIXTRL_REAL_T >& operator=(
            TMultipole< SIXTRL_REAL_T > const& rhs ) = default;

        SIXTRL_FN TMultipole< SIXTRL_REAL_T >& operator=(
            TMultipole< SIXTRL_REAL_T >&& rhs ) = default;

        SIXTRL_FN ~TMultipole() = default;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        SIXTRL_FN SIXTRL_STATIC bool CanAddToBuffer(
            buffer_t& SIXTRL_RESTRICT_REF buffer, order_t const order,
            SIXTRL_ARGPTR_DEC size_type* SIXTRL_RESTRICT
                ptr_requ_objects  = nullptr,
            SIXTRL_ARGPTR_DEC size_type* SIXTRL_RESTRICT
                ptr_requ_slots    = nullptr,
            SIXTRL_ARGPTR_DEC size_type* SIXTRL_RESTRICT
                ptr_requ_dataptrs = nullptr ) SIXTRL_NOEXCEPT;

        SIXTRL_FN SIXTRL_STATIC SIXTRL_ARGPTR_DEC TMultipole< SIXTRL_REAL_T >*
        CreateNewOnBuffer( buffer_t& SIXTRL_RESTRICT_REF buffer,
                           order_type const order );

        SIXTRL_FN SIXTRL_STATIC SIXTRL_ARGPTR_DEC TMultipole< SIXTRL_REAL_T >*
        AddToBuffer( buffer_t& SIXTRL_RESTRICT_REF buffer,
            order_type const order,
            value_type const length,
            value_type const hxl,
            value_type const hyl,
            SIXTRL_DATAPTR_DEC value_type const* SIXTRL_RESTRICT bal );

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        SIXTRL_FN void preset() SIXTRL_NOEXCEPT;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        SIXTRL_ARGPTR_DEC c_api_t const* getCApiPtr() const SIXTRL_NOEXCEPT;
        SIXTRL_ARGPTR_DEC c_api_t* getCApiPtr() SIXTRL_NOEXCEPT;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        SIXTRL_FN type_id_t getTypeId()      const SIXTRL_NOEXCEPT;
        SIXTRL_FN size_type getNumDataPtrs() const SIXTRL_NOEXCEPT;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        SIXTRL_FN value_type getLength()  const SIXTRL_NOEXCEPT;
        SIXTRL_FN value_type getHxl()     const SIXTRL_NOEXCEPT;
        SIXTRL_FN value_type getHyl()     const SIXTRL_NOEXCEPT;
        SIXTRL_FN order_type getOrder()   const SIXTRL_NOEXCEPT;
        SIXTRL_FN size_type  getBalSize() const SIXTRL_NOEXCEPT;

        SIXTRL_FN value_type getKnlValue(
            size_type const index ) const SIXTRL_NOEXCEPT;

        SIXTRL_FN value_type getKslValue(
            size_type const index ) const SIXTRL_NOEXCEPT;

        SIXTRL_FN value_type getBalValue(
            size_type const index ) const SIXTRL_NOEXCEPT;

        template< typename Iter >
        SIXTRL_FN void setKnl( Iter knl_begin ) SIXTRL_NOEXCEPT;

        template< typename Iter >
        SIXTRL_FN void setKsl( Iter ksl_begin ) SIXTRL_NOEXCEPT;

        template< typename Iter >
        SIXTRL_FN void setBal( Iter bal_begin ) SIXTRL_NOEXCEPT;

        SIXTRL_FN void setKnlValue(
            value_type knl_value, size_type const index ) SIXTRL_NOEXCEPT;

        SIXTRL_FN void setKslValue(
            value_type ksl_value, size_type const index ) SIXTRL_NOEXCEPT;

        SIXTRL_FN void setBalValue(
            value_type bal_value, size_type const index ) SIXTRL_NOEXCEPT;

        SIXTRL_FN SIXTRL_DATAPTR_DEC value_type const*
        begin() const SIXTRL_NOEXCEPT;

        SIXTRL_FN SIXTRL_DATAPTR_DEC value_type const*
        end() const SIXTRL_NOEXCEPT;

        SIXTRL_FN SIXTRL_DATAPTR_DEC value_type* begin() SIXTRL_NOEXCEPT;
        SIXTRL_FN SIXTRL_DATAPTR_DEC value_type* end() SIXTRL_NOEXCEPT;

        template< typename CPtr = SIXTRL_DATAPTR_DEC value_type const* >
        SIXTRL_FN CPtr getBalBegin() const SIXTRL_NOEXCEPT;

        template< typename CPtr = SIXTRL_DATAPTR_DEC value_type const* >
        SIXTRL_FN CPtr getBalEnd()   const SIXTRL_NOEXCEPT;

        template< typename Ptr = SIXTRL_DATAPTR_DEC value_type* >
        SIXTRL_FN Ptr getBalBegin() SIXTRL_NOEXCEPT;

        template< typename Ptr = SIXTRL_DATAPTR_DEC value_type* >
        SIXTRL_FN Ptr getBalEnd() SIXTRL_NOEXCEPT;

        SIXTRL_FN void setOrder( order_type const order ) SIXTRL_NOEXCEPT;
        SIXTRL_FN void setLength( value_type const length ) SIXTRL_NOEXCEPT;
        SIXTRL_FN void setHxl( value_type const hxl ) SIXTRL_NOEXCEPT;
        SIXTRL_FN void setHyl( value_type const hyl ) SIXTRL_NOEXCEPT;

        template< typename Ptr >
        SIXTRL_FN void assignBal( Ptr ptr_to_bal ) SIXTRL_NOEXCEPT;

        SIXTRL_FN SIXTRL_STATIC value_type Factorial(
            size_type n ) SIXTRL_NOEXCEPT;
    };

    using Multipole = TMultipole< SIXTRL_REAL_T >;

    SIXTRL_ARGPTR_DEC Multipole* Multipole_new(
        Buffer& SIXTRL_RESTRICT_REF buffer,
        ::NS(multipole_order_t) const order );

    SIXTRL_ARGPTR_DEC Multipole* Multipole_new(
        SIXTRL_ARGPTR_DEC ::NS(Buffer)* SIXTRL_RESTRICT ptr_buffer,
        ::NS(multipole_order_t) const order );

    SIXTRL_ARGPTR_DEC Multipole* Multipole_add(
        Buffer& SIXTRL_RESTRICT_REF buffer,
        ::NS(multipole_order_t) const order,
        SIXTRL_DATAPTR_DEC ::NS(multipole_real_t) const*
            SIXTRL_RESTRICT bal = nullptr,
        ::NS(multipole_real_t) const length = ::NS(multipole_real_t){ 0.0 },
        ::NS(multipole_real_t) const hxl    = ::NS(multipole_real_t){ 0.0 },
        ::NS(multipole_real_t) const hyl    = ::NS(multipole_real_t){ 0.0 } );

    SIXTRL_ARGPTR_DEC Multipole* Multipole_add(
        SIXTRL_ARGPTR_DEC ::NS(Buffer)* SIXTRL_RESTRICT ptr_buffer,
        ::NS(multipole_order_t) const order,
        SIXTRL_DATAPTR_DEC ::NS(multipole_real_t) const*
            SIXTRL_RESTRICT bal = nullptr,
        ::NS(multipole_real_t) const length = ::NS(multipole_real_t){ 0.0 },
        ::NS(multipole_real_t) const hxl    = ::NS(multipole_real_t){ 0.0 },
        ::NS(multipole_real_t) const hyl    = ::NS(multipole_real_t){ 0.0 } );

    template<> struct ObjectTypeTraits< ::NS(Multipole) >
    {
        SIXTRL_STATIC SIXTRL_INLINE object_type_id_t Type() SIXTRL_NOEXCEPT
        {
            return NS(OBJECT_TYPE_MULTIPOLE);
        }
    };

    template<> struct ObjectTypeTraits< Multipole >
    {
        SIXTRL_STATIC SIXTRL_INLINE object_type_id_t Type() SIXTRL_NOEXCEPT
        {
            return NS(OBJECT_TYPE_MULTIPOLE);
        }
    };
}

/* ************************************************************************* *
 * *** Implementation of inline and template member functions          ***** *
 * ************************************************************************* */

namespace SIXTRL_CXX_NAMESPACE
{
    /* ===================================================================== *
     * ====  TMultipole< T >:
     * ===================================================================== */

    template< typename T >
    SIXTRL_INLINE bool TMultipole< T >::CanAddToBuffer(
        typename TMultipole< T >::buffer_t& SIXTRL_RESTRICT_REF buffer,
        typename TMultipole< T >::order_type const order,
        SIXTRL_ARGPTR_DEC typename TMultipole< T >::size_type*
            SIXTRL_RESTRICT ptr_requ_objects,
        SIXTRL_ARGPTR_DEC typename TMultipole< T >::size_type*
            SIXTRL_RESTRICT ptr_requ_slots,
        SIXTRL_ARGPTR_DEC typename TMultipole< T >::size_type*
            SIXTRL_RESTRICT ptr_requ_dataptrs ) SIXTRL_NOEXCEPT
    {
        using _this_t = TMultipole< T >;
        using  size_t = typename _this_t::size_type;

        static_assert( std::is_trivial< _this_t >::value, "" );
        static_assert( std::is_standard_layout< _this_t >::value, "" );

        size_t const sizes[]  = { sizeof( value_type ) };
        size_t const counts[] = { static_cast< size_t >( 2 * order + 2 ) };

        return ::NS(Buffer_can_add_object)( &buffer, sizeof( _this_t ), 1u,
            sizes, counts, ptr_requ_objects, ptr_requ_slots,
                ptr_requ_dataptrs );
    }

    template< typename T >
    SIXTRL_INLINE SIXTRL_ARGPTR_DEC TMultipole< T >*
    TMultipole< T >::CreateNewOnBuffer(
        typename TMultipole< T >::buffer_t& SIXTRL_RESTRICT_REF buffer,
        typename TMultipole< T >::order_type const order )
    {
        using _this_t = TMultipole< T >;
        using size_t  = typename _this_t::size_type;
        using value_t = typename _this_t::value_type;

        static_assert( std::is_trivial< _this_t >::value, "" );
        static_assert( std::is_standard_layout< _this_t >::value, "" );

        type_id_t const type_id = NS(OBJECT_TYPE_MULTIPOLE);

        _this_t temp;
        temp.preset();
        temp.setOrder( order );

        size_t const num_dataptrs = size_t{ 1u };
        size_t const offsets[]    = { offsetof( _this_t, bal ) };
        size_t const sizes[]      = { sizeof( value_t ) };
        size_t const counts[]     = { static_cast< size_t >( 2 * order + 2 ) };

        return static_cast< SIXTRL_ARGPTR_DEC TMultipole< T >* >(
            static_cast< uintptr_t >( ::NS(Object_get_begin_addr)(
                ::NS(Buffer_add_object)( &buffer, &temp, sizeof( _this_t ),
                    type_id, num_dataptrs, offsets, sizes, counts ) ) ) );
    }

    template< typename T >
    SIXTRL_INLINE SIXTRL_ARGPTR_DEC TMultipole< T >*
    TMultipole< T >::AddToBuffer(
        typename TMultipole< T >::buffer_t& SIXTRL_RESTRICT_REF buffer,
        typename TMultipole< T >::order_type const order,
        typename TMultipole< T >::value_type const length,
        typename TMultipole< T >::value_type const hxl,
        typename TMultipole< T >::value_type const hyl,
        SIXTRL_DATAPTR_DEC typename TMultipole< T >::value_type const*
            SIXTRL_RESTRICT bal )
    {
        using _this_t = TMultipole< T >;
        using size_t  = typename _this_t::size_type;
        using value_t = typename _this_t::value_type;

        static_assert( std::is_trivial< _this_t >::value, "" );
        static_assert( std::is_standard_layout< _this_t >::value, "" );

        type_id_t const type_id = NS(OBJECT_TYPE_MULTIPOLE);

        _this_t temp;
        temp.setOrder( order );
        temp.setLength( length );
        temp.setHxl( hxl );
        temp.setHyl( hyl );
        temp.assignBal( bal );

        size_t const num_dataptrs = size_t{ 1 };
        size_t const offsets[]    = { offsetof( _this_t, bal ) };
        size_t const sizes[]      = { sizeof( value_t ) };
        size_t const counts[]     = { static_cast< size_t >( 2 * order + 2 ) };

        return static_cast< SIXTRL_ARGPTR_DEC TMultipole< T >* >(
            static_cast< uintptr_t >( ::NS(Object_get_begin_addr)(
                ::NS(Buffer_add_object)( &buffer, &temp, sizeof( _this_t ),
                    type_id, num_dataptrs, offsets, sizes, counts ) ) ) );
    }

    template< typename T >
    SIXTRL_INLINE typename TMultipole< T >::value_type
    TMultipole< T >::Factorial(
        typename TMultipole< T >::size_type n ) SIXTRL_NOEXCEPT
    {
        using size_t  = typename TMultipole< T >::size_type;
        using value_t = typename TMultipole< T >::value_type;

        return ( n > size_t{ 1 } )
                ? ( static_cast< value_t >( n ) *
                    TMultipole< T >::Factorial( n - size_t{ 1 } ) )
                : ( value_t{ 1.0 } );
    }

    /* ----------------------------------------------------------------- */

    template< typename T >
    SIXTRL_INLINE typename TMultipole< T >::type_id_t
    TMultipole< T >::getTypeId() const SIXTRL_NOEXCEPT
    {
        return NS(OBJECT_TYPE_MULTIPOLE);
    }

    template< typename T >
    SIXTRL_INLINE void TMultipole< T >::preset() SIXTRL_NOEXCEPT
    {
        using _this_t = TMultipole< T >;
        using order_t = typename _this_t::order_type;
        using value_t = typename _this_t::value_type;

        this->setOrder( order_t{ -1 } );
        this->setLength( value_t{} );
        this->setHxl( value_t{} );
        this->setHyl( value_t{} );
        this->assignBal( nullptr );

        return;
    }

    template< typename T >
    SIXTRL_INLINE typename TMultipole< T >::value_type
    TMultipole< T >::getLength() const SIXTRL_NOEXCEPT
    {
        return this->length;
    }

    template< typename T >
    SIXTRL_INLINE typename TMultipole< T >::value_type
    TMultipole< T >::getHxl() const SIXTRL_NOEXCEPT
    {
        return this->hxl;
    }

    template< typename T >
    SIXTRL_INLINE typename TMultipole< T >::value_type
    TMultipole< T >::getHyl() const SIXTRL_NOEXCEPT
    {
        return this->hyl;
    }

    template< typename T >
    SIXTRL_INLINE typename TMultipole< T >::order_type
    TMultipole< T >::getOrder() const SIXTRL_NOEXCEPT
    {
        return this->order;
    }

    template< typename T >
    SIXTRL_INLINE typename TMultipole< T >::size_type
    TMultipole< T >::getBalSize() const SIXTRL_NOEXCEPT
    {
        using _this_t = TMultipole< T >;
        using  size_t = typename _this_t::size_type;

        return ( this->order >= 0 )
            ? static_cast< size_t >( 2 * order + 1 )
            : size_t{ 0 };
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  */

    template< typename T >
    SIXTRL_INLINE typename TMultipole< T >::value_type
    TMultipole< T >::getKnlValue(
        typename TMultipole< T >::size_type const index ) const SIXTRL_NOEXCEPT
    {
        return this->getBalValue( 2 * index ) *
                TMultipole< T >::Factorial( index );
    }

    template< typename T >
    SIXTRL_INLINE typename TMultipole< T >::value_type
    TMultipole< T >::getKslValue(
        typename TMultipole< T >::size_type const index ) const SIXTRL_NOEXCEPT
    {
        return this->getBalValue( 2u * index + 1u ) *
                TMultipole< T >::Factorial( index );
    }

    template< typename T >
    SIXTRL_INLINE typename TMultipole< T >::value_type
    TMultipole< T >::getBalValue(
        typename TMultipole< T >::size_type const index ) const SIXTRL_NOEXCEPT
    {
        using _this_t        = TMultipole< T >;
        using value_t        = typename _this_t::value_type;
        using ptr_to_value_t = SIXTRL_DATAPTR_DEC value_t*;

        ptr_to_value_t bal   = this->begin();

        return ( ( bal != nullptr ) && ( index < this->getBalSize() ) )
            ? bal[ index ] : value_t{ 0 };
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  */

    template< typename T >
    template< typename Iter >
    SIXTRL_INLINE void TMultipole< T >::setKnl( Iter knl_begin ) SIXTRL_NOEXCEPT
    {
        using _this_t        = TMultipole< T >;
        using size_t         = typename _this_t::size_type;
        using order_t        = typename _this_t::order_type;

        order_t const order = this->getOrder();
        size_t  const    nn = ( order >= order_t{ 0 } )
            ? ( static_cast< size_t >( order ) + size_t{ 1 } )
            : size_t{ 0 };

        Iter knl_it  = knl_begin;
        Iter knl_end = knl_it;

        if( knl_it != nullptr )
        {
            std::advance( knl_it, nn );
        }

        size_t ii = size_t{ 0 };

        for( ; knl_it != knl_end ; ++knl_it, ++ii )
        {
            this->setKslValue( *knl_it, ii );
        }

        return;
    }

    template< typename T >
    template< typename Iter >
    SIXTRL_INLINE void TMultipole< T >::setKsl( Iter ksl_begin ) SIXTRL_NOEXCEPT
    {
        using _this_t        = TMultipole< T >;
        using size_t         = typename _this_t::size_type;
        using order_t        = typename _this_t::order_type;

        order_t const order = this->getOrder();
        size_t  const    nn = ( order >= order_t{ 0 } )
            ? ( static_cast< size_t >( order ) + size_t{ 1 } )
            : size_t{ 0 };

        Iter ksl_it  = ksl_begin;
        Iter ksl_end = ksl_it;

        if( ksl_it != nullptr )
        {
            std::advance( ksl_it, nn );
        }

        size_t ii = size_t{ 0 };

        for( ; ksl_it != ksl_end ; ++ksl_it, ++ii )
        {
            this->setKnlValue( *ksl_it, ii );
        }

        return;
    }

    template< typename T >
    template< typename Iter >
    SIXTRL_INLINE void TMultipole< T >::setBal( Iter bal_begin ) SIXTRL_NOEXCEPT
    {
        using size_t = typename TMultipole< T >::size_type;

        size_t const bal_length = this->getBalSize();
        size_t ii = size_t{ 0 };

        Iter bal_it  = bal_begin;
        Iter bal_end = bal_it;

        if( ( bal_it ) && ( bal_length > size_t{ 0 } ) )
        {
            std::advance( bal_it, bal_length );
        }

        for( ; bal_it != bal_end ; ++bal_it, ++ii )
        {
            this->setBalValue( *bal_it, ii );
        }

        return;
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  */

    template< typename T >
    SIXTRL_INLINE void TMultipole< T >::setKnlValue(
        typename TMultipole< T >::value_type knl_value,
        typename TMultipole< T >::size_type const index ) SIXTRL_NOEXCEPT
    {
        using _this_t = TMultipole< T >;
        using size_t  = typename _this_t::size_type;

        this->setBalValue( knl_value / _this_t::Factorial( index ),
                            size_t{ 2 } * index );

        return;
    }

    template< typename T >
    SIXTRL_INLINE void TMultipole< T >::setKslValue(
        typename TMultipole< T >::value_type ksl_value,
        typename TMultipole< T >::size_type const index ) SIXTRL_NOEXCEPT
    {
        using _this_t = TMultipole< T >;
        using size_t  = typename _this_t::size_type;

        this->setBalValue( ksl_value / _this_t::Factorial( index ),
                            size_t{ 2 } * index + size_t{ 1 } );

        return;
    }

    template< typename T >
    SIXTRL_INLINE void TMultipole< T >::setBalValue(
        typename TMultipole< T >::value_type bal_value,
        typename TMultipole< T >::size_type const index ) SIXTRL_NOEXCEPT
    {
        using _this_t        = TMultipole< T >;
        using value_t        = typename _this_t::value_type;
        using ptr_to_value_t = SIXTRL_DATAPTR_DEC value_t*;

        ptr_to_value_t bal = this->begin();

        if( ( this->getBalSize() > index ) && ( bal != nullptr ) )
        {
            bal[ index ] = bal_value / _this_t::Factorial( index );
        }

        return;
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    template< typename T >
    SIXTRL_INLINE typename TMultipole< T >::value_type const*
    TMultipole< T >::begin() const SIXTRL_NOEXCEPT
    {
        return this->getBalBegin<>();
    }

    template< typename T >
    SIXTRL_INLINE typename TMultipole< T >::value_type const*
    TMultipole< T >::end() const SIXTRL_NOEXCEPT
    {
        return this->getBalEnd<>();
    }

    template< typename T >
    SIXTRL_INLINE typename TMultipole< T >::value_type*
    TMultipole< T >::begin() SIXTRL_NOEXCEPT
    {
        using _this_t = TMultipole< T >;
        using   ptr_t = typename _this_t::value_type*;

        return const_cast< ptr_t >( static_cast< _this_t const& >(
            *this ).begin() );
    }

    template< typename T >
    SIXTRL_INLINE typename TMultipole< T >::value_type*
    TMultipole< T >::end() SIXTRL_NOEXCEPT
    {
        using _this_t = TMultipole< T >;
        using   ptr_t = typename _this_t::value_type*;

        return const_cast< ptr_t >( static_cast< _this_t const& >(
            *this ).begin() );
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    template< typename T > template< typename CPtr >
    SIXTRL_INLINE CPtr TMultipole< T >::getBalBegin() const SIXTRL_NOEXCEPT
    {
        using size_t  = typename TMultipole< T >::size_type;
        using value_t = typename TMultipole< T >::value_type;

        static_assert( sizeof( typename std::iterator_traits<
            CPtr >::value_type ) == sizeof( value_type ), "" );

        static_assert( std::is_trivially_assignable<
            typename std::iterator_traits< CPtr >::value_type,
            value_t >::value, "" );

        size_t const bal_size = this->getBalSize();

        SIXTRL_ASSERT(
            ( ( bal_size == 0u ) && ( this->bal == nullptr ) ) ||
            ( ( bal_size >  0u ) && ( this->bal != nullptr ) ) );
        ( void )bal_size;

        return reinterpret_cast< CPtr >( this->bal );
    }

    template< typename T > template< typename CPtr >
    SIXTRL_INLINE CPtr TMultipole< T >::getBalEnd() const SIXTRL_NOEXCEPT
    {
        CPtr end_ptr = this->getBalBegin< CPtr >();

        if( end_ptr != nullptr )
        {
            std::advance( end_ptr, this->getBalSize() );
        }

        return end_ptr;
    }

    template< typename T > template< typename Ptr >
    SIXTRL_INLINE Ptr TMultipole< T >::getBalBegin() SIXTRL_NOEXCEPT
    {
        using out_const_ptr_t = typename std::add_const< Ptr >::type;
        using _this_t         = TMultipole< T >;

        return const_cast< Ptr >( static_cast<
            _this_t const& >( *this ).getBalBegin< out_const_ptr_t >() );
    }

    template< typename T > template< typename Ptr >
    SIXTRL_INLINE Ptr TMultipole< T >::getBalEnd() SIXTRL_NOEXCEPT
    {
        using out_const_ptr_t = typename std::add_const< Ptr >::type;
        using _this_t         = TMultipole< T >;

        return const_cast< Ptr >( static_cast<
            _this_t const& >( *this ).getBalEnd< out_const_ptr_t >() );
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    template< typename T >
    SIXTRL_INLINE void TMultipole< T >::setOrder(
        typename TMultipole< T >::order_type const order ) SIXTRL_NOEXCEPT
    {
        this->order = order;
        return;
    }

    template< typename T >
    SIXTRL_INLINE void TMultipole< T >::setLength(
        typename TMultipole< T >::value_type const length ) SIXTRL_NOEXCEPT
    {
        this->length = length;
        return;
    }

    template< typename T >
    SIXTRL_INLINE void TMultipole< T >::setHxl(
        typename TMultipole< T >::value_type const hxl ) SIXTRL_NOEXCEPT
    {
        this->hxl = hxl;
        return;
    }

    template< typename T >
    SIXTRL_INLINE void TMultipole< T >::setHyl(
        typename TMultipole< T >::value_type const hyl ) SIXTRL_NOEXCEPT
    {
        this->hyl = hyl;
        return;
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    template< typename T >
    template< typename Ptr >
    SIXTRL_INLINE void TMultipole< T >::assignBal( Ptr ptr_to_bal ) SIXTRL_NOEXCEPT
    {
        using value_t   = typename TMultipole< T >::value_type;
        using pointer_t = SIXTRL_DATAPTR_DEC value_t*;

        this->bal = reinterpret_cast< pointer_t >( ptr_to_bal );
        return;
    }

    /* --------------------------------------------------------------------- */

    template< typename T >
    SIXTRL_ARGPTR_DEC TMultipole< T >*
    TMultipole_new( Buffer& SIXTRL_RESTRICT_REF buffer,
                    typename TMultipole< T >::order_type const order )
    {
        return TMultipole< T >::CreateNewOnBuffer(
            *buffer.getCApiPtr(), order );
    }

    template< typename T >
    SIXTRL_ARGPTR_DEC TMultipole< T >*
    TMultipole_new( SIXTRL_ARGPTR_DEC ::NS(Buffer)* SIXTRL_RESTRICT ptr_buffer,
                    typename TMultipole< T >::order_type const order )
    {
        return ( ptr_buffer != nullptr )
            ? ( TMultipole< T >::CreateNewOnBuffer( *ptr_buffer, order ) )
            : nullptr;
    }

    template< typename T >
    SIXTRL_ARGPTR_DEC TMultipole< T >*
    TMultipole_add( Buffer& SIXTRL_RESTRICT_REF buffer,
        typename TMultipole< T >::order_type const order,
        typename TMultipole< T >::value_type const& length,
        typename TMultipole< T >::value_type const& hxl,
        typename TMultipole< T >::value_type const& hyl,
        SIXTRL_DATAPTR_DEC typename TMultipole< T >::value_type
            const* SIXTRL_RESTRICT bal )
    {
        return TMultipole< T >::AddToBuffer(
            *buffer.getCApiPtr(), order, length, hxl, hyl,
            reinterpret_cast< uintptr_t >( bal ) );
    }

    template< typename T >
    SIXTRL_ARGPTR_DEC TMultipole< T >*
    TMultipole_add( ::NS(Buffer)* SIXTRL_RESTRICT ptr_buffer,
        typename TMultipole< T >::order_type const order,
        typename TMultipole< T >::value_type const& length,
        typename TMultipole< T >::value_type const& hxl,
        typename TMultipole< T >::value_type const& hyl,
        SIXTRL_DATAPTR_DEC typename TMultipole< T >::value_type
            const* SIXTRL_RESTRICT bal )
    {
        return ( ptr_buffer != nullptr )
            ? ( TMultipole< T >::AddToBuffer(
                    *ptr_buffer, order, length, hxl, hyl,
                    reinterpret_cast< uintptr_t >( bal ) ) )
            : ( nullptr );
    }

    /* ===================================================================== *
     * ====  TMultipole< SIXTRL_REAL_T >:
     * ===================================================================== */


    SIXTRL_INLINE bool TMultipole< SIXTRL_REAL_T >::CanAddToBuffer(
        TMultipole< SIXTRL_REAL_T >::buffer_t& SIXTRL_RESTRICT_REF buffer,
        TMultipole< SIXTRL_REAL_T >::order_type const order,
        SIXTRL_ARGPTR_DEC TMultipole< SIXTRL_REAL_T >::size_type*
            SIXTRL_RESTRICT ptr_requ_objects,
        SIXTRL_ARGPTR_DEC TMultipole< SIXTRL_REAL_T >::size_type*
            SIXTRL_RESTRICT ptr_requ_slots,
        SIXTRL_ARGPTR_DEC TMultipole< SIXTRL_REAL_T >::size_type*
            SIXTRL_RESTRICT ptr_requ_dataptrs ) SIXTRL_NOEXCEPT
    {
        return ::NS(Multipole_can_be_added)( &buffer, order,
                    ptr_requ_objects, ptr_requ_slots, ptr_requ_dataptrs );
    }

    SIXTRL_INLINE SIXTRL_ARGPTR_DEC TMultipole< SIXTRL_REAL_T >*
    TMultipole< SIXTRL_REAL_T >::CreateNewOnBuffer(
        TMultipole< SIXTRL_REAL_T >::buffer_t& SIXTRL_RESTRICT_REF buffer,
        TMultipole< SIXTRL_REAL_T >::order_type const order )
    {
        return static_cast< SIXTRL_ARGPTR_DEC TMultipole< SIXTRL_REAL_T >* >(
            ::NS(Multipole_new)( &buffer, order ) );
    }

    SIXTRL_INLINE TMultipole< SIXTRL_REAL_T >*
    TMultipole< SIXTRL_REAL_T >::AddToBuffer(
        TMultipole< SIXTRL_REAL_T >::buffer_t& SIXTRL_RESTRICT_REF buffer,
        TMultipole< SIXTRL_REAL_T >::order_type const order,
        TMultipole< SIXTRL_REAL_T >::value_type const length,
        TMultipole< SIXTRL_REAL_T >::value_type const hxl,
        TMultipole< SIXTRL_REAL_T >::value_type const hyl,
        SIXTRL_DATAPTR_DEC TMultipole< SIXTRL_REAL_T >::value_type
            const* SIXTRL_RESTRICT bal )
    {
        return static_cast< SIXTRL_ARGPTR_DEC TMultipole< SIXTRL_REAL_T >* >(
            ::NS(Multipole_add)( &buffer, order, length, hxl, hyl,
                 reinterpret_cast< uintptr_t >( bal ) ) );
    }

    /* ----------------------------------------------------------------- */

    SIXTRL_INLINE void TMultipole< SIXTRL_REAL_T >::preset() SIXTRL_NOEXCEPT
    {
        ::NS(Multipole_preset)( this->getCApiPtr() );
        return;
    }

    SIXTRL_INLINE TMultipole< SIXTRL_REAL_T >::value_type
    TMultipole< SIXTRL_REAL_T >::getLength() const SIXTRL_NOEXCEPT
    {
        return ::NS(Multipole_length)( this->getCApiPtr() );
    }

    SIXTRL_INLINE TMultipole< SIXTRL_REAL_T >::value_type
    TMultipole< SIXTRL_REAL_T >::getHxl() const SIXTRL_NOEXCEPT
    {
        return ::NS(Multipole_hxl)( this->getCApiPtr() );
    }

    SIXTRL_INLINE TMultipole< SIXTRL_REAL_T >::value_type
    TMultipole< SIXTRL_REAL_T >::getHyl() const SIXTRL_NOEXCEPT
    {
        return ::NS(Multipole_hyl)( this->getCApiPtr() );
    }

    SIXTRL_INLINE TMultipole< SIXTRL_REAL_T >::order_type
    TMultipole< SIXTRL_REAL_T >::getOrder() const SIXTRL_NOEXCEPT
    {
        return ::NS(Multipole_order)( this->getCApiPtr() );
    }

    SIXTRL_INLINE TMultipole< SIXTRL_REAL_T >::size_type
    TMultipole< SIXTRL_REAL_T >::getBalSize() const SIXTRL_NOEXCEPT
    {
        using _this_t = TMultipole< SIXTRL_REAL_T >;
        using  size_t = typename _this_t::size_type;
        using order_t = typename _this_t::order_type;

        order_t const order = this->getOrder();

        return ( order >= 0 ) ? static_cast< size_t >( 2 * order + 1 )
            : size_t{ 0 };
    }

    /* ----------------------------------------------------------------- */

    SIXTRL_INLINE SIXTRL_DATAPTR_DEC TMultipole< SIXTRL_REAL_T >::c_api_t const*
    TMultipole< SIXTRL_REAL_T >::getCApiPtr() const SIXTRL_NOEXCEPT
    {
        using c_api_t = TMultipole< SIXTRL_REAL_T >::c_api_t;
        return reinterpret_cast< SIXTRL_DATAPTR_DEC c_api_t const* >( this );
    }

    SIXTRL_INLINE SIXTRL_DATAPTR_DEC TMultipole< SIXTRL_REAL_T >::c_api_t*
    TMultipole< SIXTRL_REAL_T >::getCApiPtr() SIXTRL_NOEXCEPT
    {
        using _this_t = TMultipole< SIXTRL_REAL_T >;
        using   ptr_t = SIXTRL_DATAPTR_DEC _this_t::c_api_t*;

        return const_cast< ptr_t >( static_cast< _this_t const& >(
            *this ).getCApiPtr() );
    }

    /* ----------------------------------------------------------------- */

    SIXTRL_INLINE TMultipole< SIXTRL_REAL_T >::value_type
    TMultipole< SIXTRL_REAL_T >::getKnlValue(
        TMultipole< SIXTRL_REAL_T >::size_type const index ) const SIXTRL_NOEXCEPT
    {
        return ::NS(Multipole_knl)( this->getCApiPtr(), index );
    }

    SIXTRL_INLINE TMultipole< SIXTRL_REAL_T >::value_type
    TMultipole< SIXTRL_REAL_T >::getKslValue(
        TMultipole< SIXTRL_REAL_T >::size_type const index ) const SIXTRL_NOEXCEPT
    {
        return ::NS(Multipole_ksl)( this->getCApiPtr(), index );
    }

    SIXTRL_INLINE TMultipole< SIXTRL_REAL_T >::value_type
    TMultipole< SIXTRL_REAL_T >::getBalValue(
        TMultipole< SIXTRL_REAL_T >::size_type const index ) const SIXTRL_NOEXCEPT
    {
        return ::NS(Multipole_bal)( this->getCApiPtr(), index );
    }

    template< typename Iter >
    SIXTRL_INLINE void TMultipole< SIXTRL_REAL_T >::setKnl(
        Iter knl_begin ) SIXTRL_NOEXCEPT
    {
        using size_t         = TMultipole< SIXTRL_REAL_T >::size_type;
        using order_t        = TMultipole< SIXTRL_REAL_T >::order_type;

        order_t const order = this->getOrder();
        size_t  const    nn = ( order >= order_t{ 0 } )
            ? ( static_cast< size_t >( order ) + size_t{ 1 } )
            : size_t{ 0 };

        Iter knl_it  = knl_begin;
        Iter knl_end = knl_it;

        if( knl_it != nullptr )
        {
            std::advance( knl_it, nn );
        }

        size_t ii = size_t{ 0 };

        for( ; knl_it != knl_end ; ++knl_it, ++ii )
        {
            this->setKslValue( *knl_it, ii );
        }

        return;
    }

    template< typename Iter >
    SIXTRL_INLINE void TMultipole< SIXTRL_REAL_T >::setKsl(
        Iter ksl_begin ) SIXTRL_NOEXCEPT
    {
        using size_t         = TMultipole< SIXTRL_REAL_T >::size_type;
        using order_t        = TMultipole< SIXTRL_REAL_T >::order_type;

        order_t const order = this->getOrder();
        size_t  const    nn = ( order >= order_t{ 0 } )
            ? ( static_cast< size_t >( order ) + size_t{ 1 } )
            : size_t{ 0 };

        Iter ksl_it  = ksl_begin;
        Iter ksl_end = ksl_it;

        if( ksl_it != nullptr )
        {
            std::advance( ksl_it, nn );
        }

        size_t ii = size_t{ 0 };

        for( ; ksl_it != ksl_end ; ++ksl_it, ++ii )
        {
            this->setKnlValue( *ksl_it, ii );
        }

        return;
    }

    template< typename Iter >
    SIXTRL_INLINE void TMultipole< SIXTRL_REAL_T >::setBal(
        Iter bal_begin ) SIXTRL_NOEXCEPT
    {
        using size_t = TMultipole< SIXTRL_REAL_T >::size_type;

        size_t const bal_length = this->getBalSize();
        size_t ii = size_t{ 0 };

        Iter bal_it  = bal_begin;
        Iter bal_end = bal_it;

        if( ( bal_it ) && ( bal_length > size_t{ 0 } ) )
        {
            std::advance( bal_it, bal_length );
        }

        for( ; bal_it != bal_end ; ++bal_it, ++ii )
        {
            this->setBalValue( *bal_it, ii );
        }

        return;
    }

    SIXTRL_INLINE void TMultipole< SIXTRL_REAL_T >::setKnlValue(
        TMultipole< SIXTRL_REAL_T >::value_type const knl_value,
        TMultipole< SIXTRL_REAL_T >::size_type  const index ) SIXTRL_NOEXCEPT
    {
        using _this_t = TMultipole< SIXTRL_REAL_T >;
        using size_t  = _this_t::size_type;

        this->setBalValue( knl_value / _this_t::Factorial( index ),
                            size_t{ 2 } * index );

        return;
    }

    SIXTRL_INLINE void TMultipole< SIXTRL_REAL_T >::setKslValue(
        TMultipole< SIXTRL_REAL_T >::value_type const ksl_value,
        TMultipole< SIXTRL_REAL_T >::size_type  const index ) SIXTRL_NOEXCEPT
    {
        using _this_t = TMultipole< SIXTRL_REAL_T >;
        using size_t  = _this_t::size_type;

        this->setBalValue( ksl_value / _this_t::Factorial( index ),
                            size_t{ 2 } * index + size_t{ 1 } );

        return;
    }

    SIXTRL_INLINE void TMultipole< SIXTRL_REAL_T >::setBalValue(
        TMultipole< SIXTRL_REAL_T >::value_type const bal_value,
        TMultipole< SIXTRL_REAL_T >::size_type const index ) SIXTRL_NOEXCEPT
    {
        using _this_t        = TMultipole< SIXTRL_REAL_T >;
        using value_t        = _this_t::value_type;
        using ptr_to_value_t = SIXTRL_DATAPTR_DEC value_t*;

        ptr_to_value_t bal = this->begin();

        if( ( this->getBalSize() > index ) && ( bal != nullptr ) )
        {
            bal[ index ] = bal_value / _this_t::Factorial( index );
        }

        return;
    }

    /* ----------------------------------------------------------------- */

    SIXTRL_INLINE
    SIXTRL_DATAPTR_DEC TMultipole< SIXTRL_REAL_T >::value_type const*
    TMultipole< SIXTRL_REAL_T >::begin() const SIXTRL_NOEXCEPT
    {
        return this->getBalBegin<>();
    }

    SIXTRL_INLINE
    SIXTRL_DATAPTR_DEC TMultipole< SIXTRL_REAL_T >::value_type const*
    TMultipole< SIXTRL_REAL_T >::end() const SIXTRL_NOEXCEPT
    {
        return this->getBalEnd<>();
    }

    SIXTRL_INLINE SIXTRL_DATAPTR_DEC TMultipole< SIXTRL_REAL_T >::value_type*
    TMultipole< SIXTRL_REAL_T >::begin() SIXTRL_NOEXCEPT
    {
        using _this_t = TMultipole< SIXTRL_REAL_T >;
        using   ptr_t = SIXTRL_DATAPTR_DEC _this_t::value_type*;

        return const_cast< ptr_t >( static_cast< _this_t const& >(
            *this ).begin() );
    }

    SIXTRL_INLINE
    SIXTRL_DATAPTR_DEC TMultipole< SIXTRL_REAL_T >::value_type*
    TMultipole< SIXTRL_REAL_T >::end() SIXTRL_NOEXCEPT
    {
        using _this_t = TMultipole< SIXTRL_REAL_T >;
        using   ptr_t = _this_t::value_type*;

        return const_cast< ptr_t >( static_cast< _this_t const& >(
            *this ).begin() );
    }

    template< typename CPtr > SIXTRL_INLINE CPtr
    TMultipole< SIXTRL_REAL_T >::getBalBegin() const SIXTRL_NOEXCEPT
    {
        static_assert( sizeof( typename std::iterator_traits<
            CPtr >::value_type ) == sizeof( value_type ), "" );

        SIXTRL_ASSERT(
            ( ( this->getBalSize() == 0u ) && ( this->bal_addr == 0u ) ) ||
            ( ( this->getBalSize() >  0u ) && ( this->bal_addr != 0u ) ) );

        return reinterpret_cast< CPtr >(
            static_cast< uintptr_t >( this->bal_addr ) );
    }

    template< typename CPtr > SIXTRL_INLINE CPtr
    TMultipole< SIXTRL_REAL_T >::getBalEnd() const SIXTRL_NOEXCEPT
    {
        CPtr end_ptr = this->getBalBegin< CPtr >();

        if( end_ptr != nullptr )
        {
            std::advance( end_ptr, this->getBalSize() );
        }

        return end_ptr;
    }

    template< typename Ptr >
    SIXTRL_INLINE Ptr TMultipole< SIXTRL_REAL_T >::getBalBegin() SIXTRL_NOEXCEPT
    {
        using out_const_ptr_t = typename std::add_const< Ptr >::type;
        using _this_t         = TMultipole< SIXTRL_REAL_T >;

        return const_cast< Ptr >( static_cast<
            _this_t const& >( *this ).getBalBegin< out_const_ptr_t >() );
    }

    template< typename Ptr >
    SIXTRL_INLINE Ptr TMultipole< SIXTRL_REAL_T >::getBalEnd() SIXTRL_NOEXCEPT
    {
        using out_const_ptr_t = typename std::add_const< Ptr >::type;
        using _this_t         = TMultipole< SIXTRL_REAL_T >;

        return const_cast< Ptr >( static_cast<
            _this_t const& >( *this ).getBalEnd< out_const_ptr_t >() );
    }

    SIXTRL_INLINE void TMultipole< SIXTRL_REAL_T >::setOrder(
        TMultipole< SIXTRL_REAL_T >::order_type const order ) SIXTRL_NOEXCEPT
    {
        this->order = order;
        return;
    }

    SIXTRL_INLINE void TMultipole< SIXTRL_REAL_T >::setLength(
        TMultipole< SIXTRL_REAL_T >::value_type const length ) SIXTRL_NOEXCEPT
    {
        this->length = length;
        return;
    }

    SIXTRL_INLINE void TMultipole< SIXTRL_REAL_T >::setHxl(
        TMultipole< SIXTRL_REAL_T >::value_type const hxl ) SIXTRL_NOEXCEPT
    {
        this->hxl = hxl;
        return;
    }

    SIXTRL_INLINE void TMultipole< SIXTRL_REAL_T >::setHyl(
        TMultipole< SIXTRL_REAL_T >::value_type const hyl ) SIXTRL_NOEXCEPT
    {
        this->hyl = hyl;
        return;
    }

    template< typename Ptr >
    SIXTRL_INLINE void TMultipole< SIXTRL_REAL_T >::assignBal(
        Ptr ptr_to_bal ) SIXTRL_NOEXCEPT
    {
        this->bal_addr = reinterpret_cast< uintptr_t >( ptr_to_bal );
    }

    SIXTRL_INLINE TMultipole< SIXTRL_REAL_T >::value_type
    TMultipole< SIXTRL_REAL_T >::Factorial(
        TMultipole< SIXTRL_REAL_T >::size_type n ) SIXTRL_NOEXCEPT
    {
        return SIXTRL_CXX_NAMESPACE::Math_factorial<
            size_type, SIXTRL_REAL_T >( n );
    }

    /* --------------------------------------------------------------------- */

    SIXTRL_ARGPTR_DEC Multipole* Multipole_new(
        Buffer& SIXTRL_RESTRICT_REF buffer,
        ::NS(multipole_order_t) const order )
    {
        using ptr_t = SIXTRL_ARGPTR_DEC Multipole*;

        return static_cast< ptr_t >(
                ::NS(Multipole_new)( buffer.getCApiPtr(), order ) );
    }

    SIXTRL_ARGPTR_DEC Multipole* Multipole_new(
        SIXTRL_ARGPTR_DEC ::NS(Buffer)* SIXTRL_RESTRICT ptr_buffer,
        ::NS(multipole_order_t) const order )
    {
        using ptr_t = SIXTRL_ARGPTR_DEC Multipole*;

        return static_cast< ptr_t >(
            ::NS(Multipole_new)( ptr_buffer, order ) );
    }

    SIXTRL_ARGPTR_DEC Multipole* Multipole_add(
        Buffer& SIXTRL_RESTRICT_REF buffer,
        ::NS(multipole_order_t) const order,
        SIXTRL_DATAPTR_DEC ::NS(multipole_real_t) const* SIXTRL_RESTRICT bal,
        ::NS(multipole_real_t) const length,
        ::NS(multipole_real_t) const hxl,
        ::NS(multipole_real_t) const hyl )
    {
        using ptr_t = SIXTRL_ARGPTR_DEC Multipole*;

        return static_cast< ptr_t >( ::NS(Multipole_add)( buffer.getCApiPtr(),
            order, length, hxl, hyl, reinterpret_cast< uintptr_t >( bal ) ) );
    }

    SIXTRL_ARGPTR_DEC Multipole* Multipole_add(
        SIXTRL_ARGPTR_DEC ::NS(Buffer)* SIXTRL_RESTRICT ptr_buffer,
        ::NS(multipole_order_t) const order,
        ::NS(multipole_real_t) const length,
        ::NS(multipole_real_t) const hxl,
        ::NS(multipole_real_t) const hyl,
        SIXTRL_DATAPTR_DEC ::NS(multipole_real_t) const* SIXTRL_RESTRICT bal )
    {
        using ptr_t = SIXTRL_ARGPTR_DEC Multipole*;

        return static_cast< ptr_t >( ::NS(Multipole_add)(
            ptr_buffer, order, length, hxl, hyl, reinterpret_cast< uintptr_t >(
                bal ) ) );
    }
}

#endif /* define( __cplusplus ) */
#endif /* CXX_SIXTRACKLIB_COMMON_BE_MULTIPOLE_BEAM_ELEM_BE_MULTIPOLE_HPP__ */
