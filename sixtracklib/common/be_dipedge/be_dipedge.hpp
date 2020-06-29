#ifndef SIXTRACKLIB_COMMON_BE_DIPEDGE_BE_DIPEDGE_CXX_HPP__
#define SIXTRACKLIB_COMMON_BE_DIPEDGE_BE_DIPEDGE_CXX_HPP__

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
    #include "sixtracklib/common/be_dipedge/be_dipedge.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

namespace SIXTRL_CXX_NAMESPACE
{
    typedef ::NS(dipedge_real_t) dipedge_real_t;

    template< typename T >
    struct TDipoleEdge
    {
        using value_type      = T;
        using reference       = T&;
        using const_reference = T const&;
        using type_id_t       = ::NS(object_type_id_t);
        using size_type       = ::NS(buffer_size_t);
        using buffer_t        = Buffer;
        using c_buffer_t      = buffer_t::c_api_t;

        SIXTRL_FN TDipoleEdge() = default;

        SIXTRL_FN TDipoleEdge( TDipoleEdge< T > const& other ) = default;
        SIXTRL_FN TDipoleEdge( TDipoleEdge< T >&& other ) = default;

        SIXTRL_FN TDipoleEdge< T >&
        operator=( TDipoleEdge< T > const& other ) = default;

        SIXTRL_FN TDipoleEdge< T >&
        operator=( TDipoleEdge< T >&& other ) = default;

        SIXTRL_FN ~TDipoleEdge() = default;

        /* ---------------------------------------------------------------- */

        SIXTRL_STATIC SIXTRL_FN bool CanAddToBuffer(
            buffer_t& SIXTRL_RESTRICT_REF buffer,
            SIXTRL_ARGPTR_DEC size_type* SIXTRL_RESTRICT
                ptr_requ_objects  = nullptr,
            SIXTRL_ARGPTR_DEC size_type* SIXTRL_RESTRICT
                ptr_requ_slots    = nullptr,
            SIXTRL_ARGPTR_DEC size_type* SIXTRL_RESTRICT
                ptr_requ_dataptrs = nullptr ) SIXTRL_NOEXCEPT;

        SIXTRL_STATIC SIXTRL_FN bool CanAddToBuffer(
            SIXTRL_BUFFER_ARGPTR_DEC c_buffer_t* SIXTRL_RESTRICT ptr_buffer,
            SIXTRL_ARGPTR_DEC size_type* SIXTRL_RESTRICT
                ptr_requ_objects  = nullptr,
            SIXTRL_ARGPTR_DEC size_type* SIXTRL_RESTRICT
                ptr_requ_slots    = nullptr,
            SIXTRL_ARGPTR_DEC size_type* SIXTRL_RESTRICT
                ptr_requ_dataptrs = nullptr ) SIXTRL_NOEXCEPT;

        /* - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - - - -  */

        SIXTRL_STATIC SIXTRL_FN SIXTRL_ARGPTR_DEC TDipoleEdge< T >*
        CreateNewOnBuffer( buffer_t& SIXTRL_RESTRICT_REF buffer );

        SIXTRL_STATIC SIXTRL_FN SIXTRL_ARGPTR_DEC TDipoleEdge< T >*
        CreateNewOnBuffer( c_buffer_t& SIXTRL_RESTRICT_REF buffer );

        /* - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - - - -  */

        SIXTRL_STATIC SIXTRL_FN SIXTRL_ARGPTR_DEC TDipoleEdge< T >* AddToBuffer(
            buffer_t& SIXTRL_RESTRICT_REF buffer,
            const_reference SIXTRL_RESTRICT_REF r21,
            const_reference SIXTRL_RESTRICT_REF r43 );

        SIXTRL_STATIC SIXTRL_FN SIXTRL_ARGPTR_DEC TDipoleEdge< T >* AddToBuffer(
            c_buffer_t& SIXTRL_RESTRICT_REF buffer,
            const_reference SIXTRL_RESTRICT_REF r21,
            const_reference SIXTRL_RESTRICT_REF r43 );

        /* - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - - - -  */

        SIXTRL_STATIC SIXTRL_FN SIXTRL_ARGPTR_DEC TDipoleEdge< T >*
        AddCopyToBuffer( buffer_t& SIXTRL_RESTRICT_REF buffer,
            TDipoleEdge< T > const& SIXTRL_RESTRICT_REF dipedge );

        SIXTRL_STATIC SIXTRL_FN SIXTRL_ARGPTR_DEC TDipoleEdge< T >*
        AddCopyToBuffer( c_buffer_t& SIXTRL_RESTRICT_REF buffer,
            TDipoleEdge< T > const& SIXTRL_RESTRICT_REF dipedge );

        /* ---------------------------------------------------------------- */

        SIXTRL_FN type_id_t getTypeId() const SIXTRL_NOEXCEPT;

        SIXTRL_STATIC SIXTRL_FN size_type RequiredNumDataPtrs(
            buffer_t const& SIXTRL_RESTRICT_REF buffer ) SIXTRL_NOEXCEPT;

        SIXTRL_STATIC SIXTRL_FN size_type RequiredNumDataPtrs(
            SIXTRL_BUFFER_ARGPTR_DEC const c_buffer_t *const
                SIXTRL_RESTRICT ptr_buffer ) SIXTRL_NOEXCEPT;

        /* ----------------------------------------------------------------- */

        SIXTRL_FN void preset() SIXTRL_NOEXCEPT;
        SIXTRL_FN void clear()  SIXTRL_NOEXCEPT;

        SIXTRL_FN const_reference getR21() const SIXTRL_NOEXCEPT;
        SIXTRL_FN const_reference getR43() const SIXTRL_NOEXCEPT;

        /* ----------------------------------------------------------------- */

        SIXTRL_FN void setR21(
            const_reference SIXTRL_RESTRICT_REF r21 ) SIXTRL_NOEXCEPT;

        SIXTRL_FN void setR43(
            const_reference SIXTRL_RESTRICT_REF r43 ) SIXTRL_NOEXCEPT;

        /* ----------------------------------------------------------------- */

        value_type r21  SIXTRL_ALIGN( 8 );
        value_type r43  SIXTRL_ALIGN( 8 );
    };

    template< typename T > struct ObjectTypeTraits<
        SIXTRL_CXX_NAMESPACE::TDipoleEdge< T > >
    {
        SIXTRL_STATIC SIXTRL_INLINE object_type_id_t Type() SIXTRL_NOEXCEPT
        {
            return ::NS(OBJECT_TYPE_DIPEDGE);
        }
    };

    template< typename T >
    SIXTRL_ARGPTR_DEC TDipoleEdge< T >* TDipoleEdge_new(
        typename TDipoleEdge< T >::buffer_t& SIXTRL_RESTRICT_REF buffer );

    template< typename T >
    SIXTRL_ARGPTR_DEC TDipoleEdge< T >* TDipoleEdge_new(
        typename TDipoleEdge< T >::c_buffer_t& SIXTRL_RESTRICT_REF buffer );


    template< typename T >
    SIXTRL_ARGPTR_DEC TDipoleEdge< T >* TDipoleEdge_add(
        typename TDipoleEdge< T >::buffer_t& SIXTRL_RESTRICT_REF buffer,
        typename TDipoleEdge< T >::const_reference SIXTRL_RESTRICT_REF r21,
        typename TDipoleEdge< T >::const_reference SIXTRL_RESTRICT_REF r43 );

    template< typename T >
    SIXTRL_ARGPTR_DEC TDipoleEdge< T >* TDipoleEdge_add(
        typename TDipoleEdge< T >::c_buffer_t& SIXTRL_RESTRICT_REF buffer,
        typename TDipoleEdge< T >::const_reference SIXTRL_RESTRICT_REF r21,
        typename TDipoleEdge< T >::const_reference SIXTRL_RESTRICT_REF r43 );

    template< typename T >
    SIXTRL_ARGPTR_DEC TDipoleEdge< T >* TDipoleEdge_add_copy(
        typename TDipoleEdge< T >::buffer_t& SIXTRL_RESTRICT_REF buffer,
        TDipoleEdge< T > const& SIXTRL_RESTRICT_REF other );

    template< typename T >
    SIXTRL_ARGPTR_DEC TDipoleEdge< T >* TDipoleEdge_add_copy(
        typename TDipoleEdge< T >::c_buffer_t& SIXTRL_RESTRICT_REF buffer,
        TDipoleEdge< T > const& SIXTRL_RESTRICT_REF other );

    /* ===================================================================== *
     * ====  Specialization TDipoleEdge< NS(dipedge_real_t) > :
     * ===================================================================== */

    template<> struct TDipoleEdge< ::NS(dipedge_real_t) > :
        public ::NS(DipoleEdge)
    {
        using value_type = ::NS(dipedge_real_t);
        using type_id_t  = ::NS(object_type_id_t);
        using size_type  = ::NS(buffer_size_t);
        using buffer_t   = Buffer;
        using c_buffer_t = buffer_t::c_api_t;
        using c_api_t    = ::NS(DipoleEdge);

        /* ---------------------------------------------------------------- */

        SIXTRL_FN TDipoleEdge() = default;

        SIXTRL_FN TDipoleEdge(
            TDipoleEdge< value_type > const& other ) = default;

        SIXTRL_FN TDipoleEdge( TDipoleEdge< value_type >&& other ) = default;

        SIXTRL_FN TDipoleEdge< value_type >& operator=(
            TDipoleEdge< value_type > const& other ) = default;

        SIXTRL_FN TDipoleEdge< value_type >& operator=(
            TDipoleEdge< value_type >&& other ) = default;

        SIXTRL_FN ~TDipoleEdge() = default;

        /* ---------------------------------------------------------------- */

        SIXTRL_STATIC SIXTRL_FN bool CanAddToBuffer(
            buffer_t& SIXTRL_RESTRICT_REF buffer,
            SIXTRL_ARGPTR_DEC size_type* SIXTRL_RESTRICT
                ptr_requ_objects  = nullptr,
            SIXTRL_ARGPTR_DEC size_type* SIXTRL_RESTRICT
                ptr_requ_slots    = nullptr,
            SIXTRL_ARGPTR_DEC size_type* SIXTRL_RESTRICT
                ptr_requ_dataptrs = nullptr ) SIXTRL_NOEXCEPT;

        SIXTRL_STATIC SIXTRL_FN bool CanAddToBuffer(
            SIXTRL_BUFFER_ARGPTR_DEC c_buffer_t* SIXTRL_RESTRICT ptr_buffer,
            SIXTRL_ARGPTR_DEC size_type* SIXTRL_RESTRICT
                ptr_requ_objects  = nullptr,
            SIXTRL_ARGPTR_DEC size_type* SIXTRL_RESTRICT
                ptr_requ_slots    = nullptr,
            SIXTRL_ARGPTR_DEC size_type* SIXTRL_RESTRICT
                ptr_requ_dataptrs = nullptr ) SIXTRL_NOEXCEPT;


        SIXTRL_STATIC SIXTRL_FN
        SIXTRL_ARGPTR_DEC TDipoleEdge< ::NS(dipedge_real_t) >*
        CreateNewOnBuffer( buffer_t& SIXTRL_RESTRICT_REF buffer );

        SIXTRL_STATIC SIXTRL_FN
        SIXTRL_ARGPTR_DEC TDipoleEdge< ::NS(dipedge_real_t) >*
        CreateNewOnBuffer( c_buffer_t& SIXTRL_RESTRICT_REF buffer );


        SIXTRL_STATIC SIXTRL_FN
        SIXTRL_ARGPTR_DEC TDipoleEdge< ::NS(dipedge_real_t) >* AddToBuffer(
            buffer_t& SIXTRL_RESTRICT_REF buffer,
            value_type const r21, value_type const r43 );

        SIXTRL_STATIC SIXTRL_FN
        SIXTRL_ARGPTR_DEC TDipoleEdge< ::NS(dipedge_real_t) >* AddToBuffer(
            c_buffer_t& SIXTRL_RESTRICT_REF buffer,
            value_type const r21, value_type const r43 );

        SIXTRL_STATIC SIXTRL_FN
        SIXTRL_ARGPTR_DEC TDipoleEdge< ::NS(dipedge_real_t) >* AddCopyToBuffer(
            buffer_t& SIXTRL_RESTRICT_REF buffer,
            TDipoleEdge< ::NS(dipedge_real_t) > const& SIXTRL_RESTRICT_REF );

        SIXTRL_STATIC SIXTRL_FN
        SIXTRL_ARGPTR_DEC TDipoleEdge< ::NS(dipedge_real_t) >* AddCopyToBuffer(
            c_buffer_t& SIXTRL_RESTRICT_REF buffer,
            TDipoleEdge< ::NS(dipedge_real_t) > const& SIXTRL_RESTRICT_REF );

        /* ----------------------------------------------------------------- */

        SIXTRL_ARGPTR_DEC c_api_t const* getCApiPtr() const SIXTRL_NOEXCEPT;
        SIXTRL_ARGPTR_DEC c_api_t* getCApiPtr() SIXTRL_NOEXCEPT;

        /* ----------------------------------------------------------------- */

        SIXTRL_STATIC SIXTRL_FN size_type RequiredNumDataPtrs(
            buffer_t const& SIXTRL_RESTRICT_REF buffer ) SIXTRL_NOEXCEPT;

        SIXTRL_STATIC SIXTRL_FN size_type RequiredNumDataPtrs(
            SIXTRL_BUFFER_ARGPTR_DEC const c_buffer_t *const
                SIXTRL_RESTRICT ptr_buffer ) SIXTRL_NOEXCEPT;

        SIXTRL_FN type_id_t getTypeId() const SIXTRL_NOEXCEPT;

        /* ----------------------------------------------------------------- */

        SIXTRL_FN void preset() SIXTRL_NOEXCEPT;
        SIXTRL_FN void clear() SIXTRL_NOEXCEPT;

        SIXTRL_FN value_type getR21() const SIXTRL_NOEXCEPT;
        SIXTRL_FN value_type getR43() const SIXTRL_NOEXCEPT;

        /* ----------------------------------------------------------------- */

        SIXTRL_FN void setR21( value_type const r21 ) SIXTRL_NOEXCEPT;
        SIXTRL_FN void setR43( value_type const r43 ) SIXTRL_NOEXCEPT;
    };

    using DipoleEdge = TDipoleEdge< ::NS(dipedge_real_t) >;

    SIXTRL_ARGPTR_DEC DipoleEdge* DipoleEdge_new(
        DipoleEdge::buffer_t& SIXTRL_RESTRICT_REF buffer );

    SIXTRL_ARGPTR_DEC DipoleEdge* DipoleEdge_new(
        DipoleEdge::c_buffer_t& SIXTRL_RESTRICT_REF buffer );

    SIXTRL_ARGPTR_DEC DipoleEdge* DipoleEdge_add(
        DipoleEdge::buffer_t& SIXTRL_RESTRICT_REF buffer,
        DipoleEdge::value_type const r21, DipoleEdge::value_type const r43 );

    SIXTRL_ARGPTR_DEC DipoleEdge* DipoleEdge_add(
        DipoleEdge::c_buffer_t& SIXTRL_RESTRICT_REF buffer,
        DipoleEdge::value_type const r21, DipoleEdge::value_type const r43 );

    SIXTRL_ARGPTR_DEC DipoleEdge* DipoleEdge_add_copy(
        DipoleEdge::buffer_t& SIXTRL_RESTRICT_REF buffer,
        SIXTRL_CXX_NAMESPACE::DipoleEdge const& SIXTRL_RESTRICT_REF dipedge );

    SIXTRL_ARGPTR_DEC DipoleEdge* DipoleEdge_add_copy(
        DipoleEdge::c_buffer_t& SIXTRL_RESTRICT_REF buffer,
        SIXTRL_CXX_NAMESPACE::DipoleEdge const& SIXTRL_RESTRICT_REF dipedge );
}

/* ************************************************************************* *
 * *** Implementation of inline and template member functions          ***** *
 * ************************************************************************* */

namespace SIXTRL_CXX_NAMESPACE
{
    /* ===================================================================== *
     * ====  TDipoleEdge< T > :
     * ===================================================================== */

    template< typename T >
    SIXTRL_INLINE bool TDipoleEdge< T >::CanAddToBuffer(
        typename TDipoleEdge< T >::buffer_t& SIXTRL_RESTRICT_REF buffer,
        SIXTRL_ARGPTR_DEC typename TDipoleEdge< T >::size_type*
            SIXTRL_RESTRICT req_objects,
        SIXTRL_ARGPTR_DEC typename TDipoleEdge< T >::size_type*
            SIXTRL_RESTRICT req_slots,
        SIXTRL_ARGPTR_DEC typename TDipoleEdge< T >::size_type*
            SIXTRL_RESTRICT req_dataptrs ) SIXTRL_NOEXCEPT
    {
        return TDipoleEdge< T >::CanAddToBuffer(
            buffer.getCApiPtr(), req_objects, req_slots, req_dataptrs );
    }

    template< typename T >
    SIXTRL_INLINE bool TDipoleEdge< T >::CanAddToBuffer(
        SIXTRL_BUFFER_ARGPTR_DEC typename TDipoleEdge< T >::c_buffer_t*
            SIXTRL_RESTRICT ptr_buffer,
        SIXTRL_ARGPTR_DEC typename TDipoleEdge< T >::size_type*
            SIXTRL_RESTRICT req_objects,
        SIXTRL_ARGPTR_DEC typename TDipoleEdge< T >::size_type*
            SIXTRL_RESTRICT req_slots,
        SIXTRL_ARGPTR_DEC typename TDipoleEdge< T >::size_type*
            SIXTRL_RESTRICT req_dataptrs ) SIXTRL_NOEXCEPT
    {
        using _this_t = SIXTRL_CXX_NAMESPACE::TDipoleEdge< T >;
        using  size_t = typename _this_t::size_type;

        static_assert( std::is_trivial< _this_t >::value, "" );
        static_assert( std::is_standard_layout< _this_t >::value, "" );

        size_t const num_dataptrs = _this_t::RequiredNumDataPtrs( ptr_buffer );
        SIXTRL_ASSERT( num_dataptrs == size_t{ 0 } );

        SIXTRL_ARGPTR_DEC size_t const* sizes  = nullptr;
        SIXTRL_ARGPTR_DEC size_t const* counts = nullptr;

        return ::NS(Buffer_can_add_object)( ptr_buffer, sizeof( _this_t ),
            num_dataptrs, sizes, counts, req_objects, req_slots, req_dataptrs );
    }

    template< typename T >
    SIXTRL_INLINE SIXTRL_ARGPTR_DEC TDipoleEdge< T >*
    TDipoleEdge< T >::CreateNewOnBuffer(
        typename TDipoleEdge< T >::buffer_t& SIXTRL_RESTRICT_REF buffer )
    {
        return TDipoleEdge< T >::CreateNewOnBuffer( *buffer.getCApiPtr() );
    }

    template< typename T >
    SIXTRL_INLINE SIXTRL_ARGPTR_DEC TDipoleEdge< T >*
    TDipoleEdge< T >::CreateNewOnBuffer(
        typename TDipoleEdge< T >::c_buffer_t& SIXTRL_RESTRICT_REF buffer )
    {
        using _this_t = SIXTRL_CXX_NAMESPACE::TDipoleEdge< T >;
        using size_t  = typename _this_t::size_type;
        using ptr_t   = SIXTRL_ARGPTR_DEC _this_t*;

        static_assert( std::is_trivial< _this_t >::value, "" );
        static_assert( std::is_standard_layout< _this_t >::value, "" );

        size_t const num_dataptrs = _this_t::RequiredNumDataPtrs( &buffer );
        SIXTRL_ASSERT( num_dataptrs == size_t{ 0 } );

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
    SIXTRL_INLINE SIXTRL_ARGPTR_DEC TDipoleEdge< T >*
    TDipoleEdge< T >::AddToBuffer(
        typename TDipoleEdge< T >::buffer_t& SIXTRL_RESTRICT_REF buffer,
        typename TDipoleEdge< T >::const_reference SIXTRL_RESTRICT_REF r21,
        typename TDipoleEdge< T >::const_reference SIXTRL_RESTRICT_REF r43 )
    {
        return SIXTRL_CXX_NAMESPACE::TDipoleEdge< T>::AddToBuffer(
            *buffer.getCApiPtr(), r21, r43 );
    }

    template< typename T > SIXTRL_INLINE SIXTRL_ARGPTR_DEC TDipoleEdge< T >*
    TDipoleEdge< T >::AddToBuffer(
        typename TDipoleEdge< T >::c_buffer_t& SIXTRL_RESTRICT_REF buffer,
        typename TDipoleEdge< T >::const_reference SIXTRL_RESTRICT_REF r21,
        typename TDipoleEdge< T >::const_reference SIXTRL_RESTRICT_REF r43 )
    {
        using _this_t = SIXTRL_CXX_NAMESPACE::TDipoleEdge< T >;
        using size_t  = typename _this_t::size_type;
        using ptr_t   = SIXTRL_ARGPTR_DEC _this_t*;

        static_assert( std::is_trivial< _this_t >::value, "" );
        static_assert( std::is_standard_layout< _this_t >::value, "" );

        size_t const num_dataptrs = _this_t::RequiredNumDataPtrs( &buffer );
        SIXTRL_ASSERT( num_dataptrs == size_t{ 0 } );

        SIXTRL_ARGPTR_DEC size_t const* offsets = nullptr;
        SIXTRL_ARGPTR_DEC size_t const* sizes   = nullptr;
        SIXTRL_ARGPTR_DEC size_t const* counts  = nullptr;

        _this_t temp;
        temp.setR21( r21 );
        temp.setR43( r43 );

        return reinterpret_cast< ptr_t >( ::NS(Object_get_begin_addr)(
            ::NS(Buffer_add_object)( &buffer, &temp, sizeof( _this_t ),
                temp.getTypeId(), num_dataptrs, offsets, sizes, counts ) ) );
    }

    template< typename T >SIXTRL_INLINE SIXTRL_ARGPTR_DEC TDipoleEdge< T >*
    TDipoleEdge< T >::AddCopyToBuffer(
        typename TDipoleEdge< T >::buffer_t& SIXTRL_RESTRICT_REF buffer,
        TDipoleEdge< T > const& SIXTRL_RESTRICT_REF orig )
    {
        return TDipoleEdge< T >::AddCopyToBuffer( *buffer.getCApiPtr(), orig );
    }

    template< typename T >SIXTRL_INLINE SIXTRL_ARGPTR_DEC TDipoleEdge< T >*
    TDipoleEdge< T >::AddCopyToBuffer(
        typename TDipoleEdge< T >::c_buffer_t& SIXTRL_RESTRICT_REF buffer,
        TDipoleEdge< T > const& SIXTRL_RESTRICT_REF orig )
    {
        using _this_t = SIXTRL_CXX_NAMESPACE::TDipoleEdge< T >;
        using size_t  = typename _this_t::size_type;
        using ptr_t   = SIXTRL_ARGPTR_DEC _this_t*;

        static_assert( std::is_trivial< _this_t >::value, "" );
        static_assert( std::is_standard_layout< _this_t >::value, "" );

        size_t const num_dataptrs = _this_t::RequiredNumDataPtrs( &buffer );
        SIXTRL_ASSERT( num_dataptrs == size_t{ 0 } );

        SIXTRL_ARGPTR_DEC size_t const* offsets = nullptr;
        SIXTRL_ARGPTR_DEC size_t const* sizes   = nullptr;
        SIXTRL_ARGPTR_DEC size_t const* counts  = nullptr;

        _this_t temp;
        temp.setR21( orig.getR21() );
        temp.setR43( orig.getR43() );

        return reinterpret_cast< ptr_t >( ::NS(Object_get_begin_addr)(
            ::NS(Buffer_add_object)( &buffer, &temp, sizeof( _this_t ),
                temp.getTypeId(), num_dataptrs, offsets, sizes, counts ) ) );
    }

    /* ---------------------------------------------------------------- */

    template< typename T >
    typename TDipoleEdge< T >::type_id_t
    TDipoleEdge< T >::getTypeId() const SIXTRL_NOEXCEPT
    {
        return SIXTRL_CXX_NAMESPACE::OBJECT_TYPE_DIPEDGE;
    }

    template< typename T >
    SIXTRL_INLINE typename TDipoleEdge< T >::size_type
    TDipoleEdge< T >::RequiredNumDataPtrs( typename TDipoleEdge< T >::buffer_t
        const& SIXTRL_RESTRICT_REF SIXTRL_UNUSED( buffer ) ) SIXTRL_NOEXCEPT
    {
        return ::NS(DipoleEdge_num_dataptrs)( nullptr );
    }

    template< typename T >
    SIXTRL_INLINE typename TDipoleEdge< T >::size_type
    TDipoleEdge< T >::RequiredNumDataPtrs( const
        typename TDipoleEdge< T >::c_buffer_t *const SIXTRL_RESTRICT
            SIXTRL_UNUSED( buffer ) ) SIXTRL_NOEXCEPT
    {
        return ::NS(DipoleEdge_num_dataptrs)( nullptr );
    }

    template< typename T > void TDipoleEdge< T >::preset() SIXTRL_NOEXCEPT
    {
        this->clear();
    }

    template< typename T > void TDipoleEdge< T >::clear() SIXTRL_NOEXCEPT
    {
        this->setR21( T{0} );
        this->setR43( T{1} );
    }

    template< typename T > typename TDipoleEdge< T >::const_reference
    TDipoleEdge< T >::getR21() const SIXTRL_NOEXCEPT
    {
        return this->r21;
    }

    template< typename T > typename TDipoleEdge< T >::const_reference
    TDipoleEdge< T >::getR43() const SIXTRL_NOEXCEPT
    {
        return this->r43;
    }

    /* ----------------------------------------------------------------- */

    template< typename T > void TDipoleEdge< T >::setR21(
        typename TDipoleEdge< T >::const_reference
            SIXTRL_RESTRICT_REF r21 ) SIXTRL_NOEXCEPT
    {
        this->r21 = r21;
    }

    template< typename T > void TDipoleEdge< T >::setR43(
        typename TDipoleEdge< T >::const_reference
            SIXTRL_RESTRICT_REF r43 ) SIXTRL_NOEXCEPT
    {
        this->r43 = r43;
    }

    /* ----------------------------------------------------------------- */

    template< typename T >
    SIXTRL_ARGPTR_DEC TDipoleEdge< T >* TDipoleEdge_new(
        typename TDipoleEdge< T >::buffer_t& SIXTRL_RESTRICT_REF buffer )
    {
        return TDipoleEdge< T >::CreateNewOnBuffer( *buffer.getCApiPtr() );
    }

    template< typename T >
    SIXTRL_ARGPTR_DEC TDipoleEdge< T >* TDipoleEdge_new(
        typename TDipoleEdge< T >::c_buffer_t& SIXTRL_RESTRICT_REF buffer )
    {
        return TDipoleEdge< T >::CreateNewOnBuffer( buffer );
    }

    template< typename T >
    SIXTRL_ARGPTR_DEC TDipoleEdge< T >* TDipoleEdge_add(
        typename TDipoleEdge< T >::buffer_t& SIXTRL_RESTRICT_REF buffer,
        typename TDipoleEdge< T >::const_reference SIXTRL_RESTRICT_REF r21,
        typename TDipoleEdge< T >::const_reference SIXTRL_RESTRICT_REF r43 )
    {
        return TDipoleEdge< T >::AddToBuffer( *buffer.getCApiPtr(), r21, r43 );
    }

    template< typename T >
    SIXTRL_ARGPTR_DEC TDipoleEdge< T >* TDipoleEdge_add(
        typename TDipoleEdge< T >::c_buffer_t& SIXTRL_RESTRICT_REF buffer,
        typename TDipoleEdge< T >::const_reference SIXTRL_RESTRICT_REF r21,
        typename TDipoleEdge< T >::const_reference SIXTRL_RESTRICT_REF r43 )
    {
        return TDipoleEdge< T >::AddToBuffer( buffer, r21, r43);
    }

    template< typename T >
    SIXTRL_ARGPTR_DEC TDipoleEdge< T >* TDipoleEdge_add_copy(
        typename TDipoleEdge< T >::buffer_t& SIXTRL_RESTRICT_REF buffer,
        TDipoleEdge< T > const& SIXTRL_RESTRICT_REF  other )
    {
        return SIXTRL_CXX_NAMESPACE::TDipoleEdge< T >::AddToBuffer(
            *buffer.getCApiPtr(), other.getR21(), other.getR43() );
    }

    template< typename T >
    SIXTRL_ARGPTR_DEC TDipoleEdge< T >* TDipoleEdge_add_copy(
        typename TDipoleEdge< T >::c_buffer_t& SIXTRL_RESTRICT_REF buffer,
        TDipoleEdge< T > const& SIXTRL_RESTRICT_REF other )
    {
        return SIXTRL_CXX_NAMESPACE::TDipoleEdge< T >::AddToBuffer(
            buffer, other.getXDipoleEdge(), other.getYDipoleEdge() );
    }

    /* ===================================================================== *
     * ====  Specialization TDipoleEdge< ::NS(dipedge_real_t) > :
     * ===================================================================== */

    SIXTRL_INLINE bool DipoleEdge::CanAddToBuffer(
        DipoleEdge::buffer_t& SIXTRL_RESTRICT_REF buffer,
        SIXTRL_ARGPTR_DEC DipoleEdge::size_type* SIXTRL_RESTRICT req_objects,
        SIXTRL_ARGPTR_DEC DipoleEdge::size_type* SIXTRL_RESTRICT req_slots,
        SIXTRL_ARGPTR_DEC DipoleEdge::size_type* SIXTRL_RESTRICT req_dataptrs
    ) SIXTRL_NOEXCEPT
    {
        return ::NS(DipoleEdge_can_be_added)(
            buffer.getCApiPtr(), req_objects, req_slots, req_dataptrs );
    }


    SIXTRL_INLINE bool DipoleEdge::CanAddToBuffer(
        SIXTRL_BUFFER_ARGPTR_DEC DipoleEdge::c_buffer_t*
            SIXTRL_RESTRICT ptr_buffer,
        SIXTRL_ARGPTR_DEC DipoleEdge::size_type* SIXTRL_RESTRICT req_objects,
        SIXTRL_ARGPTR_DEC DipoleEdge::size_type* SIXTRL_RESTRICT req_slots,
        SIXTRL_ARGPTR_DEC DipoleEdge::size_type* SIXTRL_RESTRICT req_dataptrs
    ) SIXTRL_NOEXCEPT
    {
        return ::NS(DipoleEdge_can_be_added)(
            ptr_buffer, req_objects, req_slots, req_dataptrs );
    }

    SIXTRL_INLINE SIXTRL_ARGPTR_DEC DipoleEdge* DipoleEdge::CreateNewOnBuffer(
        DipoleEdge::buffer_t& SIXTRL_RESTRICT_REF buffer )
    {
        return static_cast< SIXTRL_ARGPTR_DEC DipoleEdge* >(
            ::NS(DipoleEdge_new)( buffer.getCApiPtr() ) );
    }

    SIXTRL_INLINE SIXTRL_ARGPTR_DEC DipoleEdge* DipoleEdge::CreateNewOnBuffer(
        DipoleEdge::c_buffer_t& SIXTRL_RESTRICT_REF buffer )
    {
        return static_cast< SIXTRL_ARGPTR_DEC DipoleEdge* >(
            ::NS(DipoleEdge_new)( &buffer ) );
    }

    SIXTRL_INLINE SIXTRL_ARGPTR_DEC DipoleEdge* DipoleEdge::AddToBuffer(
        DipoleEdge::buffer_t& SIXTRL_RESTRICT_REF buffer,
        DipoleEdge::value_type const r21, DipoleEdge::value_type const r43 )
    {
        return static_cast< SIXTRL_ARGPTR_DEC DipoleEdge* >(
            ::NS(DipoleEdge_add)( buffer.getCApiPtr(), r21, r43 ) );
    }

    SIXTRL_INLINE SIXTRL_ARGPTR_DEC DipoleEdge* DipoleEdge::AddToBuffer(
        DipoleEdge::c_buffer_t& SIXTRL_RESTRICT_REF buffer,
        DipoleEdge::value_type const r21, DipoleEdge::value_type const r43 )
    {
        return static_cast< SIXTRL_ARGPTR_DEC DipoleEdge* >(
            ::NS(DipoleEdge_add)( &buffer, r21, r43 ) );
    }

    SIXTRL_INLINE SIXTRL_ARGPTR_DEC DipoleEdge* DipoleEdge::AddCopyToBuffer(
        DipoleEdge::buffer_t& SIXTRL_RESTRICT_REF buffer,
        DipoleEdge const& SIXTRL_RESTRICT_REF orig )
    {
        return static_cast< SIXTRL_ARGPTR_DEC DipoleEdge* >(
            ::NS(DipoleEdge_add_copy)(
                buffer.getCApiPtr(), orig.getCApiPtr() ) );
    }

    SIXTRL_INLINE SIXTRL_ARGPTR_DEC DipoleEdge* DipoleEdge::AddCopyToBuffer(
        DipoleEdge::c_buffer_t& SIXTRL_RESTRICT_REF buffer,
        DipoleEdge const& SIXTRL_RESTRICT_REF orig )
    {
        return static_cast< SIXTRL_ARGPTR_DEC DipoleEdge* >(
            ::NS(DipoleEdge_add_copy)( &buffer, orig.getCApiPtr() ) );
    }

    /* ----------------------------------------------------------------- */

    SIXTRL_ARGPTR_DEC DipoleEdge::c_api_t const*
    DipoleEdge::getCApiPtr() const SIXTRL_NOEXCEPT
    {
        return reinterpret_cast<
            SIXTRL_ARGPTR_DEC DipoleEdge::c_api_t const* >( this );
    }

    SIXTRL_ARGPTR_DEC DipoleEdge::c_api_t*
    DipoleEdge::getCApiPtr() SIXTRL_NOEXCEPT
    {
        return const_cast< SIXTRL_ARGPTR_DEC DipoleEdge::c_api_t* >(
            static_cast< TDipoleEdge< ::NS(dipedge_real_t) > const& >( *this
                ).getCApiPtr() );
    }

    /* ----------------------------------------------------------------- */

    DipoleEdge::type_id_t DipoleEdge::getTypeId() const SIXTRL_NOEXCEPT
    {
        return SIXTRL_CXX_NAMESPACE::OBJECT_TYPE_DIPEDGE;
    }


    DipoleEdge::size_type DipoleEdge::RequiredNumDataPtrs( DipoleEdge::buffer_t
        const& SIXTRL_RESTRICT_REF SIXTRL_UNUSED( buffer ) ) SIXTRL_NOEXCEPT
    {
        return ::NS(DipoleEdge_num_dataptrs)( nullptr );
    }


    DipoleEdge::size_type DipoleEdge::RequiredNumDataPtrs(
        SIXTRL_BUFFER_ARGPTR_DEC const DipoleEdge::c_buffer_t *const
                SIXTRL_RESTRICT SIXTRL_UNUSED( buffer ) ) SIXTRL_NOEXCEPT
    {
        return ::NS(DipoleEdge_num_dataptrs)( nullptr );
    }

    void DipoleEdge::preset() SIXTRL_NOEXCEPT
    {
        ::NS(DipoleEdge_preset)( this->getCApiPtr() );
    }

    void DipoleEdge::clear() SIXTRL_NOEXCEPT
    {
        ::NS(DipoleEdge_clear)( this->getCApiPtr() );
    }

    DipoleEdge::value_type DipoleEdge::getR21() const SIXTRL_NOEXCEPT
    {
        return ::NS(DipoleEdge_r21)( this->getCApiPtr() );
    }

    DipoleEdge::value_type DipoleEdge::getR43() const SIXTRL_NOEXCEPT
    {
        return ::NS(DipoleEdge_r43)( this->getCApiPtr() );
    }

    /* ----------------------------------------------------------------- */

     void DipoleEdge::setR21(
         DipoleEdge::value_type const r21 ) SIXTRL_NOEXCEPT
    {
        ::NS(DipoleEdge_set_r21)( this->getCApiPtr(), r21 );
    }

     void DipoleEdge::setR43(
        DipoleEdge::value_type const r43 ) SIXTRL_NOEXCEPT
    {
        ::NS(DipoleEdge_set_r43)( this->getCApiPtr(), r43 );
    }

    /* --------------------------------------------------------------------- */

    SIXTRL_INLINE SIXTRL_ARGPTR_DEC DipoleEdge*
    DipoleEdge_new( DipoleEdge::buffer_t& SIXTRL_RESTRICT_REF buffer )
    {
        return DipoleEdge::CreateNewOnBuffer( buffer );
    }

    SIXTRL_INLINE SIXTRL_ARGPTR_DEC DipoleEdge*
    DipoleEdge_new( DipoleEdge::c_buffer_t& SIXTRL_RESTRICT_REF buffer )
    {
        return DipoleEdge::CreateNewOnBuffer( buffer );
    }

    SIXTRL_INLINE SIXTRL_ARGPTR_DEC DipoleEdge* DipoleEdge_add(
        DipoleEdge::buffer_t& SIXTRL_RESTRICT_REF buffer,
        DipoleEdge::value_type const r21, DipoleEdge::value_type const r43  )
    {
        return DipoleEdge::AddToBuffer( buffer, r21, r43 );
    }

    SIXTRL_INLINE SIXTRL_ARGPTR_DEC DipoleEdge* DipoleEdge_add(
        DipoleEdge::c_buffer_t& SIXTRL_RESTRICT_REF buffer,
        DipoleEdge::value_type const r21, DipoleEdge::value_type const r43 )
    {
        return DipoleEdge::AddToBuffer( buffer, r21, r43 );
    }

    SIXTRL_INLINE SIXTRL_ARGPTR_DEC DipoleEdge* DipoleEdge_add_copy(
        DipoleEdge::buffer_t& SIXTRL_RESTRICT_REF buffer,
        DipoleEdge const& SIXTRL_RESTRICT_REF orig )
    {
        return DipoleEdge::AddCopyToBuffer( buffer, orig );
    }

    SIXTRL_INLINE SIXTRL_ARGPTR_DEC DipoleEdge* DipoleEdge_add_copy(
        DipoleEdge::c_buffer_t& SIXTRL_RESTRICT_REF buffer,
        DipoleEdge const& SIXTRL_RESTRICT_REF orig )
    {
        return DipoleEdge::AddCopyToBuffer( buffer, orig );
    }
}

#endif /* C++ */

#endif /* SIXTRACKLIB_COMMON_BE_DIPEDGE_BE_DIPEDGE_CXX_HPP__ */

/* end: sixtracklib/common/be_limit/be_limit.hpp */
