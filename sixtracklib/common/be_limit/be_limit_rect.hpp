#ifndef SIXTRACKLIB_COMMON_BE_LIMIT_RECT_CXX_HPP__
#define SIXTRACKLIB_COMMON_BE_LIMIT_RECT_CXX_HPP__

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
    #include "sixtracklib/common/be_limit/be_limit_rect.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

namespace SIXTRL_CXX_NAMESPACE
{
    template< typename T >
    struct TLimitRect
    {
        using value_type      = T;
        using reference       = T&;
        using const_reference = T const&;
        using type_id_t       = ::NS(object_type_id_t);
        using buffer_t        = SIXTRL_CXX_NAMESPACE::Buffer;
        using size_type       = buffer_t::size_type;
        using c_buffer_t      = buffer_t::c_api_t;

        /* ----------------------------------------------------------------- */
        
        static SIXTRL_CONSTEXPR_OR_CONST value_type  DEFAULT_MAX_X = 
             value_type{ SIXTRL_LIMIT_DEFAULT_MAX_X };
        
        static SIXTRL_CONSTEXPR_OR_CONST value_type  DEFAULT_MIN_X = 
            value_type{ SIXTRL_LIMIT_DEFAULT_MIN_X };
            
        static SIXTRL_CONSTEXPR_OR_CONST value_type  DEFAULT_MAX_Y = 
            value_type{ SIXTRL_LIMIT_DEFAULT_MAX_Y };
        
        static SIXTRL_CONSTEXPR_OR_CONST value_type  DEFAULT_MIN_Y = 
            value_type{ SIXTRL_LIMIT_DEFAULT_MIN_Y };
            
        /* ----------------------------------------------------------------- */
        

        SIXTRL_FN TLimitRect() = default;

        SIXTRL_FN TLimitRect( TLimitRect< T > const& other ) = default;
        TLimitRect( TLimitRect< T >&& other ) = default;

        SIXTRL_FN TLimitRect< T >& operator=( 
            TLimitRect< T > const& other ) = default;
            
        SIXTRL_FN TLimitRect< T >& operator=( 
            TLimitRect< T >&& other ) = default;

        SIXTRL_FN ~TLimitRect() = default;

        /* ----------------------------------------------------------------- */

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

        SIXTRL_STATIC SIXTRL_FN SIXTRL_ARGPTR_DEC TLimitRect< T >*
        CreateNewOnBuffer( buffer_t& SIXTRL_RESTRICT_REF buffer );

        SIXTRL_STATIC SIXTRL_FN SIXTRL_ARGPTR_DEC TLimitRect< T >*
        CreateNewOnBuffer( c_buffer_t& SIXTRL_RESTRICT_REF buffer );

        
        SIXTRL_STATIC SIXTRL_FN SIXTRL_ARGPTR_DEC TLimitRect< T >* AddToBuffer(
            buffer_t& SIXTRL_RESTRICT_REF buffer,
            const_reference min_x, const_reference max_x,
            const_reference min_y, const_reference max_y );

        SIXTRL_STATIC SIXTRL_FN SIXTRL_ARGPTR_DEC TLimitRect< T >* AddToBuffer(
            c_buffer_t& SIXTRL_RESTRICT_REF buffer,
            const_reference min_x, const_reference max_x,
            const_reference min_y, const_reference max_y );
        
        
        SIXTRL_STATIC SIXTRL_FN SIXTRL_ARGPTR_DEC TLimitRect< T >* 
        AddCopyToBuffer( buffer_t& SIXTRL_RESTRICT_REF buffer,
            TLimitRect< T > const& SIXTRL_RESTRICT_REF other );
        
        SIXTRL_STATIC SIXTRL_FN SIXTRL_ARGPTR_DEC TLimitRect< T >* 
        AddCopyToBuffer( c_buffer_t& SIXTRL_RESTRICT_REF buffer,
            TLimitRect< T > const& SIXTRL_RESTRICT_REF other );

        /* ----------------------------------------------------------------- */

        SIXTRL_FN type_id_t getTypeId() const SIXTRL_NOEXCEPT;

        SIXTRL_STATIC SIXTRL_FN size_type RequiredNumDataPtrs(
            buffer_t const& SIXTRL_RESTRICT_REF buffer ) SIXTRL_NOEXCEPT;

        SIXTRL_STATIC SIXTRL_FN size_type RequiredNumDataPtrs(
            SIXTRL_BUFFER_ARGPTR_DEC const c_buffer_t *const 
                SIXTRL_RESTRICT ptr_buffer ) SIXTRL_NOEXCEPT;
                
        /* ----------------------------------------------------------------- */
            
        SIXTRL_FN void preset() SIXTRL_NOEXCEPT;
        SIXTRL_FN void clear() SIXTRL_NOEXCEPT;
        
        /* ----------------------------------------------------------------- */
                
        SIXTRL_FN const_reference getMinX() const SIXTRL_NOEXCEPT;
        SIXTRL_FN const_reference getMaxX() const SIXTRL_NOEXCEPT;

        SIXTRL_FN const_reference getMinY() const SIXTRL_NOEXCEPT;
        SIXTRL_FN const_reference getMaxY() const SIXTRL_NOEXCEPT;
        
        /* ----------------------------------------------------------------- */
        
        SIXTRL_FN void setXLimit( const_reference x_lim ) SIXTRL_NOEXCEPT;
        SIXTRL_FN void setYLimit( const_reference y_lim ) SIXTRL_NOEXCEPT;
        
        SIXTRL_FN void setMinX( const_reference min_x ) SIXTRL_NOEXCEPT;
        SIXTRL_FN void setMaxX( const_reference max_x ) SIXTRL_NOEXCEPT;
        
        SIXTRL_FN void setMinY( const_reference min_y ) SIXTRL_NOEXCEPT;
        SIXTRL_FN void setMaxY( const_reference max_y ) SIXTRL_NOEXCEPT;
        
        /* ----------------------------------------------------------------- */

        value_type x_limit SIXTRL_ALIGN( 8 );
        value_type y_limit SIXTRL_ALIGN( 8 );
    };

    template< typename T > struct ObjectTypeTraits<
        SIXTRL_CXX_NAMESPACE::TLimitRect< T > >
    {
        SIXTRL_STATIC SIXTRL_INLINE object_type_id_t Type() SIXTRL_NOEXCEPT
        {
            return ::NS(OBJECT_TYPE_LIMIT_RECT);
        }
    };

    template< typename T >
    SIXTRL_ARGPTR_DEC TLimitRect< T >* TLimitRect_new(
        typename TLimitRect< T >::buffer_t& SIXTRL_RESTRICT_REF buffer );

    template< typename T >
    SIXTRL_ARGPTR_DEC TLimitRect< T >* TLimitRect_new(
        typename TLimitRect< T >::c_buffer_t& SIXTRL_RESTRICT buffer );


    template< typename T >
    SIXTRL_ARGPTR_DEC TLimitRect< T >* TLimitRect_add(
        typename TLimitRect< T >::buffer_t& SIXTRL_RESTRICT_REF buffer,
        typename TLimitRect< T >::const_reference SIXTRL_RESTRICT_REF min_x,
        typename TLimitRect< T >::const_reference SIXTRL_RESTRICT_REF max_x,
        typename TLimitRect< T >::const_reference SIXTRL_RESTRICT_REF min_y,
        typename TLimitRect< T >::const_reference SIXTRL_RESTRICT_REF max_y);

    template< typename T >
    SIXTRL_ARGPTR_DEC TLimitRect< T >* TLimitRect_add(
        typename TLimitRect< T >::c_buffer_t& SIXTRL_RESTRICT buffer,
        typename TLimitRect< T >::const_reference SIXTRL_RESTRICT_REF min_x,
        typename TLimitRect< T >::const_reference SIXTRL_RESTRICT_REF max_x,
        typename TLimitRect< T >::const_reference SIXTRL_RESTRICT_REF min_y,
        typename TLimitRect< T >::const_reference SIXTRL_RESTRICT_REF max_y);

    template< typename T >
    SIXTRL_ARGPTR_DEC TLimitRect< T >* TLimitRect_add_copy(
        typename TLimitRect< T >::buffer_t& SIXTRL_RESTRICT_REF buffer,
        TLimitRect< T > const& SIXTRL_RESTRICT_REF other );

    template< typename T >
    SIXTRL_ARGPTR_DEC TLimitRect< T >* TLimitRect_add_copy(
        typename TLimitRect< T >::c_buffer_t& SIXTRL_RESTRICT_REF buffer,
        TLimitRect< T > const& SIXTRL_RESTRICT_REF other );

    /* ===================================================================== *
     * ====  Specialization TLimitRect< NS(particle_real_t) > :
     * ===================================================================== */

    template<> struct TLimitRect< NS(particle_real_t) > : 
        public ::NS(LimitRect)
    {
        using value_type = ::NS(particle_real_t);
        using type_id_t  = ::NS(object_type_id_t);
        using size_type  = ::NS(buffer_size_t);
        using buffer_t   = SIXTRL_CXX_NAMESPACE::Buffer;
        using c_buffer_t = buffer_t::c_api_t;
        using c_api_t    = ::NS(LimitRect);

        static SIXTRL_CONSTEXPR_OR_CONST value_type  DEFAULT_MAX_X = 
             SIXTRL_CXX_NAMESPACE::LIMIT_DEFAULT_MAX_X;
        
        static SIXTRL_CONSTEXPR_OR_CONST value_type  DEFAULT_MIN_X = 
            SIXTRL_CXX_NAMESPACE::LIMIT_DEFAULT_MIN_X;
            
        static SIXTRL_CONSTEXPR_OR_CONST value_type DEFAULT_MAX_Y = 
            SIXTRL_CXX_NAMESPACE::LIMIT_DEFAULT_MAX_Y;
        
        static SIXTRL_CONSTEXPR_OR_CONST value_type DEFAULT_MIN_Y = 
            SIXTRL_CXX_NAMESPACE::LIMIT_DEFAULT_MIN_Y;

        /* ---------------------------------------------------------------- */
            
        SIXTRL_FN TLimitRect() = default;

        SIXTRL_FN TLimitRect( 
            TLimitRect< value_type > const& other ) = default;
            
        SIXTRL_FN TLimitRect( TLimitRect< value_type >&& other ) = default;

        SIXTRL_FN TLimitRect< value_type >& operator=(
            TLimitRect< value_type > const& other ) = default;

        SIXTRL_FN TLimitRect< value_type >& operator=(
            TLimitRect< value_type >&& other ) = default;

        SIXTRL_FN ~TLimitRect() = default;

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
        SIXTRL_ARGPTR_DEC TLimitRect< ::NS(particle_real_t) >*
        CreateNewOnBuffer( buffer_t& SIXTRL_RESTRICT_REF buffer );

        SIXTRL_STATIC SIXTRL_FN
        SIXTRL_ARGPTR_DEC TLimitRect< ::NS(particle_real_t) >*
        CreateNewOnBuffer( c_buffer_t& SIXTRL_RESTRICT_REF buffer );


        SIXTRL_STATIC SIXTRL_FN
        SIXTRL_ARGPTR_DEC TLimitRect< ::NS(particle_real_t) >* AddToBuffer(
            buffer_t& SIXTRL_RESTRICT_REF buffer,
            value_type const min_x, value_type const max_x, 
            value_type const min_y, value_type const max_y );

        SIXTRL_STATIC SIXTRL_FN
        SIXTRL_ARGPTR_DEC TLimitRect< ::NS(particle_real_t) >* AddToBuffer(
            c_buffer_t& SIXTRL_RESTRICT_REF buffer,
            value_type const min_x, value_type const max_x, 
            value_type const min_y, value_type const max_y );
        
        
        SIXTRL_STATIC SIXTRL_FN 
        SIXTRL_ARGPTR_DEC TLimitRect< ::NS(particle_real_t) >* AddCopyToBuffer(
            buffer_t& SIXTRL_RESTRICT_REF buffer, 
            TLimitRect< ::NS(particle_real_t) > const& 
                SIXTRL_RESTRICT_REF other );
        
        SIXTRL_STATIC SIXTRL_FN 
        SIXTRL_ARGPTR_DEC TLimitRect< ::NS(particle_real_t) >* AddCopyToBuffer(
            c_buffer_t& SIXTRL_RESTRICT_REF buffer,
            TLimitRect< ::NS(particle_real_t) > const& 
                SIXTRL_RESTRICT_REF other );

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
        SIXTRL_FN void clear()  SIXTRL_NOEXCEPT;
        
        /* ----------------------------------------------------------------- */
        
        SIXTRL_FN value_type getMinX() const SIXTRL_NOEXCEPT;
        SIXTRL_FN value_type getMaxX() const SIXTRL_NOEXCEPT;

        SIXTRL_FN value_type getMinY() const SIXTRL_NOEXCEPT;
        SIXTRL_FN value_type getMaxY() const SIXTRL_NOEXCEPT;
        
        /* ----------------------------------------------------------------- */
        
        SIXTRL_FN void setXLimit( value_type const x_lim ) SIXTRL_NOEXCEPT;
        SIXTRL_FN void setYLimit( value_type const y_lim ) SIXTRL_NOEXCEPT;
        
        SIXTRL_FN void setMinX( value_type const min_x ) SIXTRL_NOEXCEPT;
        SIXTRL_FN void setMaxX( value_type const max_x ) SIXTRL_NOEXCEPT;
        
        SIXTRL_FN void setMinY( value_type const min_y ) SIXTRL_NOEXCEPT;
        SIXTRL_FN void setMaxY( value_type const max_y ) SIXTRL_NOEXCEPT;
    };

    using LimitRect = TLimitRect< ::NS(particle_real_t) >;

    SIXTRL_ARGPTR_DEC LimitRect* LimitRect_new( 
        SIXTRL_CXX_NAMESPACE::Buffer& buffer );

    SIXTRL_ARGPTR_DEC LimitRect* LimitRect_new(
        SIXTRL_ARGPTR_DEC ::NS(Buffer)* SIXTRL_RESTRICT ptr_buffer );

    SIXTRL_ARGPTR_DEC LimitRect* LimitRect_add( 
        SIXTRL_CXX_NAMESPACE::Buffer& buffer,
        ::NS(particle_real_t) const x_min, ::NS(particle_real_t) const x_max,
        ::NS(particle_real_t) const y_min, ::NS(particle_real_t) const y_max );

    SIXTRL_ARGPTR_DEC LimitRect* LimitRect_add(
        SIXTRL_BUFFER_ARGPTR_DEC ::NS(Buffer)* SIXTRL_RESTRICT ptr_buffer,
        ::NS(particle_real_t) const x_min, ::NS(particle_real_t) const x_max,
        ::NS(particle_real_t) const y_min, ::NS(particle_real_t) const y_max );

    SIXTRL_ARGPTR_DEC LimitRect* LimitRect_add_copy( 
        SIXTRL_CXX_NAMESPACE::Buffer& buffer,
        SIXTRL_CXX_NAMESPACE::LimitRect const& SIXTRL_RESTRICT_REF limit );

    SIXTRL_ARGPTR_DEC LimitRect* LimitRect_add_copy(
        SIXTRL_ARGPTR_DEC ::NS(Buffer)* SIXTRL_RESTRICT ptr_buffer,
        SIXTRL_CXX_NAMESPACE::LimitRect const& SIXTRL_RESTRICT_REF limit );
}

/* ************************************************************************* *
 * *** Implementation of inline and template member functions          ***** *
 * ************************************************************************* */


namespace SIXTRL_CXX_NAMESPACE
{
    /* ===================================================================== *
     * ====  TLimitRect< T > :
     * ===================================================================== */

    template< typename T >
    bool TLimitRect< T >::CanAddToBuffer(
        typename TLimitRect< T >::buffer_t& SIXTRL_RESTRICT_REF buffer,
        SIXTRL_ARGPTR_DEC typename TLimitRect< T >::size_type*
            SIXTRL_RESTRICT req_objects,
        SIXTRL_ARGPTR_DEC typename TLimitRect< T >::size_type*
            SIXTRL_RESTRICT req_slots,
        SIXTRL_ARGPTR_DEC typename TLimitRect< T >::size_type*
            SIXTRL_RESTRICT req_dataptrs ) SIXTRL_NOEXCEPT
    {
        return TLimitRect< T >::CanAddToBuffer(
            buffer.getCApiPtr(), req_objects, req_slots, req_dataptrs );
    }

    template< typename T >
    bool TLimitRect< T >::CanAddToBuffer(
        typename TLimitRect< T >::c_buffer_t* SIXTRL_RESTRICT ptr_buffer,
        SIXTRL_ARGPTR_DEC typename TLimitRect< T >::size_type*
            SIXTRL_RESTRICT req_objects,
        SIXTRL_ARGPTR_DEC typename TLimitRect< T >::size_type*
            SIXTRL_RESTRICT req_slots,
        SIXTRL_ARGPTR_DEC typename TLimitRect< T >::size_type*
            SIXTRL_RESTRICT req_dataptrs ) SIXTRL_NOEXCEPT
    {
        using _this_t = SIXTRL_CXX_NAMESPACE::TLimitRect< T >;
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
    SIXTRL_INLINE SIXTRL_ARGPTR_DEC TLimitRect< T >* 
    TLimitRect< T >::CreateNewOnBuffer(
        typename TLimitRect< T >::buffer_t& SIXTRL_RESTRICT_REF buffer )
    {
        return TLimitRect< T >::CreateNewOnBuffer( *( buffer.getCApiPtr() ) );
    }

    template< typename T >
    SIXTRL_INLINE SIXTRL_ARGPTR_DEC TLimitRect< T >* 
    TLimitRect< T >::CreateNewOnBuffer(
        typename TLimitRect< T >::c_buffer_t& SIXTRL_RESTRICT_REF buffer )
    {
        using _this_t = SIXTRL_CXX_NAMESPACE::TLimitRect< T >;
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
    SIXTRL_INLINE SIXTRL_ARGPTR_DEC TLimitRect< T >* 
    TLimitRect< T >::AddToBuffer(
        typename TLimitRect< T >::buffer_t& SIXTRL_RESTRICT_REF buffer,
        typename TLimitRect< T >::const_reference SIXTRL_RESTRICT_REF min_x,
        typename TLimitRect< T >::const_reference SIXTRL_RESTRICT_REF max_x,
        typename TLimitRect< T >::const_reference SIXTRL_RESTRICT_REF min_y,
        typename TLimitRect< T >::const_reference SIXTRL_RESTRICT_REF max_y )
    {
        return TLimitRect< T>::AddToBuffer( 
            *buffer.getCApiPtr(), min_x, max_x, min_y, max_y );
    }

    template< typename T >
    SIXTRL_INLINE SIXTRL_ARGPTR_DEC TLimitRect< T >* 
    TLimitRect< T >::AddToBuffer(
        typename TLimitRect< T >::c_buffer_t& SIXTRL_RESTRICT_REF buffer,
        typename TLimitRect< T >::const_reference SIXTRL_RESTRICT_REF min_x,
        typename TLimitRect< T >::const_reference SIXTRL_RESTRICT_REF max_x,
        typename TLimitRect< T >::const_reference SIXTRL_RESTRICT_REF min_y,
        typename TLimitRect< T >::const_reference SIXTRL_RESTRICT_REF max_y )
    {
        using _this_t = SIXTRL_CXX_NAMESPACE::TLimitRect< T >;
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
        temp.setMinX( min_x );
        temp.setMaxX( max_x );
        temp.setMinY( min_y );
        temp.setMaxY( max_y );

        return reinterpret_cast< ptr_t >( ::NS(Object_get_begin_addr)(
            ::NS(Buffer_add_object)( &buffer, &temp, sizeof( _this_t ),
                temp.getTypeId(), num_dataptrs, offsets, sizes, counts ) ) );
    }
    
    template< typename T >
    SIXTRL_INLINE SIXTRL_ARGPTR_DEC TLimitRect< T >* 
    TLimitRect< T >::AddCopyToBuffer(
        typename TLimitRect< T >::buffer_t& SIXTRL_RESTRICT_REF buffer,
        TLimitRect< T > const& SIXTRL_RESTRICT_REF orig )
    {
        return TLimitRect< T >::AddToBuffer( *( buffer.getCApiPtr() ), 
             orig.getMinX(), orig.getMaxX(), orig.getMinY(), orig.getMaxY() );
    }
    
    template< typename T >
    SIXTRL_INLINE SIXTRL_ARGPTR_DEC TLimitRect< T >* 
    TLimitRect< T >::AddCopyToBuffer(
        typename TLimitRect< T >::c_buffer_t& SIXTRL_RESTRICT_REF buffer,
        TLimitRect< T > const& SIXTRL_RESTRICT_REF orig )
    {
        return TLimitRect< T >::AddToBuffer( buffer, 
            orig.getMinX(), orig.getMaxX(), orig.getMinY(), orig.getMaxY() );
    }

    /* --------------------------------------------------------------------- */

    template< typename T >
    SIXTRL_INLINE typename TLimitRect< T >::type_id_t
    TLimitRect< T >::getTypeId() const SIXTRL_NOEXCEPT
    {
        return SIXTRL_CXX_NAMESPACE::OBJECT_TYPE_LIMIT_RECT;
    }

    template< typename T >
    SIXTRL_INLINE typename TLimitRect< T >::size_type
    TLimitRect< T >::RequiredNumDataPtrs( 
        typename TLimitRect< T >::buffer_t const& 
            SIXTRL_RESTRICT_REF buffer ) SIXTRL_NOEXCEPT
    {
        return ::NS(LimitRect_get_required_num_dataptrs)( 
            buffer.getCApiPtr(), nullptr );
    }
    
    template< typename T >
    SIXTRL_INLINE typename TLimitRect< T >::size_type 
    TLimitRect< T >::RequiredNumDataPtrs( SIXTRL_BUFFER_ARGPTR_DEC const 
        typename TLimitRect< T >::c_buffer_t *const
            SIXTRL_RESTRICT ptr_buffer ) SIXTRL_NOEXCEPT
    {
        return ::NS(LimitRect_get_required_num_dataptrs)( ptr_buffer, nullptr );
    }
    
    /* --------------------------------------------------------------------- */

    template< typename T >
    void TLimitRect< T >::preset() SIXTRL_NOEXCEPT
    {
        using _this_t = TLimitRect< T >;
        
        this->min_x = _this_t::DEFAULT_MIN_X;
        this->max_x = _this_t::DEFAULT_MAX_X;
        
        this->min_y = _this_t::DEFAULT_MIN_Y;
        this->max_y = _this_t::DEFAULT_MAX_Y;
    }
    
    template< typename T >
    void TLimitRect< T >::clear() SIXTRL_NOEXCEPT
    {
        this->preset();
    }
    
    /* --------------------------------------------------------------------- */

    template< typename T > typename TLimitRect< T >::const_reference 
    TLimitRect< T >::getMinX() const SIXTRL_NOEXCEPT
    {
        return this->min_x;
    }
    
    template< typename T > typename TLimitRect< T >::const_reference 
    TLimitRect< T >::getMaxX() const SIXTRL_NOEXCEPT
    {
        return this->max_x;
    }

    template< typename T > typename TLimitRect< T >::const_reference 
    TLimitRect< T >::getMinY() const SIXTRL_NOEXCEPT
    {
        return this->min_y;
    }
    
    template< typename T > typename TLimitRect< T >::const_reference 
    TLimitRect< T >::getMaxY() const SIXTRL_NOEXCEPT
    {
        return this->max_y;
    }
    
    /* --------------------------------------------------------------------- */
    
    template< typename T >
    void TLimitRect< T >::setXLimit( 
        typename TLimitRect< T >::const_reference x_lim ) SIXTRL_NOEXCEPT
    {
        using value_t = typename TLimitRect< T >::value_type;
        SIXTRL_ASSERT( x_lim >= ( value_t{ 0 } ) );
        
        this->min_x =  x_lim;
        this->max_x = -x_lim;
    }
    
    template< typename T >
    void TLimitRect< T >::setYLimit( 
        typename TLimitRect< T >::const_reference y_lim ) SIXTRL_NOEXCEPT
    {
        using value_t = typename TLimitRect< T >::value_type;
        SIXTRL_ASSERT( y_lim >= ( value_t{ 0 } ) );
        
        this->min_y =  y_lim;
        this->max_y = -y_lim;
    }
    
    template< typename T >
    void TLimitRect< T >::setMinX( 
        typename TLimitRect< T >::const_reference min_x ) SIXTRL_NOEXCEPT
    {
        this->min_x = min_x;
    }
    
    template< typename T >
    void TLimitRect< T >::setMaxX( 
        typename TLimitRect< T >::const_reference max_x ) SIXTRL_NOEXCEPT
    {
        this->max_x = max_x;
    }
    
    template< typename T >
    void TLimitRect< T >::setMinY( 
        typename TLimitRect< T >::const_reference min_y ) SIXTRL_NOEXCEPT
    {
        this->min_y = min_y;
    }
    
    template< typename T >
    void TLimitRect< T >::setMaxY( 
        typename TLimitRect< T >::const_reference max_y ) SIXTRL_NOEXCEPT
    {
        this->max_y = max_y;
    }
    
    /* --------------------------------------------------------------------- */
    
    template< typename T >
    SIXTRL_INLINE SIXTRL_ARGPTR_DEC TLimitRect< T >* TLimitRect_new(
        typename TLimitRect< T >::buffer_t& buffer )
    {
        return TLimitRect< T >::CreateNewOnBuffer( buffer );
    }

    template< typename T >
    SIXTRL_INLINE SIXTRL_ARGPTR_DEC TLimitRect< T >* TLimitRect_new(
        typename TLimitRect< T >::c_buffer_t& SIXTRL_RESTRICT_REF buffer )
    {
        return TLimitRect< T >::CreateNewOnBuffer( buffer );
    }

    template< typename T >
    SIXTRL_INLINE SIXTRL_ARGPTR_DEC TLimitRect< T >* TLimitRect_add(
        typename TLimitRect< T >::buffer_t& SIXTRL_RESTRICT_REF buffer,
        typename TLimitRect< T >::const_reference SIXTRL_RESTRICT_REF min_x,
        typename TLimitRect< T >::const_reference SIXTRL_RESTRICT_REF max_x,
        typename TLimitRect< T >::const_reference SIXTRL_RESTRICT_REF min_y,
        typename TLimitRect< T >::const_reference SIXTRL_RESTRICT_REF max_y )
    {
        return TLimitRect< T >::AddToBuffer( 
            buffer, min_x, max_x, min_y, max_y );
    }

    template< typename T >
    SIXTRL_INLINE SIXTRL_ARGPTR_DEC TLimitRect< T >* TLimitRect_add(
        typename TLimitRect< T >::c_buffer_t& SIXTRL_RESTRICT_REF buffer,
        typename TLimitRect< T >::const_reference SIXTRL_RESTRICT_REF min_x,
        typename TLimitRect< T >::const_reference SIXTRL_RESTRICT_REF max_x,
        typename TLimitRect< T >::const_reference SIXTRL_RESTRICT_REF min_y,
        typename TLimitRect< T >::const_reference SIXTRL_RESTRICT_REF max_y )
    {
        return TLimitRect< T >::AddToBuffer( 
            buffer, min_x, max_x, min_y, max_y );
    }

    template< typename T >
    SIXTRL_INLINE SIXTRL_ARGPTR_DEC TLimitRect< T >* TLimitRect_add_copy(
        typename TLimitRect< T >::buffer_t& SIXTRL_RESTRICT_REF buffer,
        TLimitRect< T > const& SIXTRL_RESTRICT_REF  other )
    {
        return TLimitRect< T >::AddCopyToBuffer( buffer, other );
    }

    template< typename T >
    SIXTRL_INLINE SIXTRL_ARGPTR_DEC TLimitRect< T >* TLimitRect_add_copy(
        typename TLimitRect< T >::c_buffer_t& SIXTRL_RESTRICT_REF buffer,
        TLimitRect< T > const& SIXTRL_RESTRICT_REF  other )
    {
        return TLimitRect< T >::AddCopyToBuffer( buffer, other );
    }

    /* ===================================================================== *
     * ====  Specialization TLimitRect< ::NS(particle_real_t) > :
     * ===================================================================== */

    SIXTRL_INLINE bool LimitRect::CanAddToBuffer( 
        LimitRect::buffer_t& SIXTRL_RESTRICT_REF buffer,
        SIXTRL_ARGPTR_DEC LimitRect::size_type* SIXTRL_RESTRICT requ_objects,
        SIXTRL_ARGPTR_DEC LimitRect::size_type* SIXTRL_RESTRICT requ_slots,
        SIXTRL_ARGPTR_DEC LimitRect::size_type* SIXTRL_RESTRICT requ_dataptrs 
        ) SIXTRL_NOEXCEPT
    {
        return ::NS(LimitRect_can_be_added)( buffer.getCApiPtr(), requ_objects,
            requ_slots, requ_dataptrs );
    }


    SIXTRL_INLINE bool LimitRect::CanAddToBuffer(
        SIXTRL_BE_ARGPTR_DEC LimitRect::c_buffer_t* SIXTRL_RESTRICT ptr_buffer,
        SIXTRL_ARGPTR_DEC LimitRect::size_type* SIXTRL_RESTRICT requ_objects,
        SIXTRL_ARGPTR_DEC LimitRect::size_type* SIXTRL_RESTRICT requ_slots,
        SIXTRL_ARGPTR_DEC LimitRect::size_type* SIXTRL_RESTRICT requ_dataptrs 
        ) SIXTRL_NOEXCEPT
    {
        return ::NS(LimitRect_can_be_added)( 
            ptr_buffer, requ_objects, requ_slots, requ_dataptrs );
    }
    
    SIXTRL_INLINE SIXTRL_ARGPTR_DEC LimitRect* LimitRect::CreateNewOnBuffer( 
        LimitRect::buffer_t& SIXTRL_RESTRICT_REF buffer )
    {
        return static_cast< SIXTRL_ARGPTR_DEC LimitRect* >(
            ::NS(LimitRect_new)( buffer.getCApiPtr() ) );
    }

    SIXTRL_INLINE SIXTRL_ARGPTR_DEC LimitRect* LimitRect::CreateNewOnBuffer(
        LimitRect::c_buffer_t& SIXTRL_RESTRICT_REF buffer )
    {
        return static_cast< SIXTRL_ARGPTR_DEC LimitRect* >(
            ::NS(LimitRect_new)( &buffer ) );
    }

    SIXTRL_INLINE SIXTRL_ARGPTR_DEC LimitRect* LimitRect::AddToBuffer(
        LimitRect::buffer_t& SIXTRL_RESTRICT_REF buffer,
        LimitRect::value_type const min_x, LimitRect::value_type const max_x,
        LimitRect::value_type const min_y, LimitRect::value_type const max_y )
    {
        return static_cast< SIXTRL_ARGPTR_DEC LimitRect* >( ::NS(LimitRect_add)( 
            buffer.getCApiPtr(), min_x, max_x, min_y, max_y ) );
    }

    SIXTRL_INLINE SIXTRL_ARGPTR_DEC LimitRect*
    LimitRect::AddToBuffer( LimitRect::c_buffer_t& SIXTRL_RESTRICT_REF buffer,
        LimitRect::value_type const min_x, LimitRect::value_type const max_x,
        LimitRect::value_type const min_y, LimitRect::value_type const max_y )
    {
        return static_cast< SIXTRL_ARGPTR_DEC LimitRect* >(
            ::NS(LimitRect_add)( &buffer, min_x, max_x, min_y, max_y ) );
    }
    
    SIXTRL_ARGPTR_DEC LimitRect* LimitRect::AddCopyToBuffer(
        LimitRect::buffer_t& SIXTRL_RESTRICT_REF buffer, 
        LimitRect const& other )
    {
        return static_cast< SIXTRL_ARGPTR_DEC LimitRect* >(
            ::NS(LimitRect_add_copy)( 
                buffer.getCApiPtr(), other.getCApiPtr() ) );
    }

    SIXTRL_ARGPTR_DEC LimitRect* LimitRect::AddCopyToBuffer( 
        LimitRect::c_buffer_t& SIXTRL_RESTRICT_REF buffer, 
        LimitRect const& other )
    {
        return static_cast< SIXTRL_ARGPTR_DEC LimitRect* >(
            ::NS(LimitRect_add_copy)( &buffer, other.getCApiPtr() ) );
    }

    /* --------------------------------------------------------------------- */

    SIXTRL_ARGPTR_DEC LimitRect::c_api_t const* 
    LimitRect::getCApiPtr() const SIXTRL_NOEXCEPT
    {
        return reinterpret_cast< LimitRect::c_api_t const* >( this );
    }

    SIXTRL_ARGPTR_DEC LimitRect::c_api_t*
    LimitRect::getCApiPtr() SIXTRL_NOEXCEPT
    {
        return const_cast< LimitRect::c_api_t* >(
            static_cast< LimitRect const& >( *this ).getCApiPtr() );
    }

    /* --------------------------------------------------------------------- */

    LimitRect::type_id_t LimitRect::getTypeId() const SIXTRL_NOEXCEPT
    {
        return SIXTRL_CXX_NAMESPACE::OBJECT_TYPE_LIMIT_RECT;
    }

    LimitRect::size_type LimitRect::RequiredNumDataPtrs( 
        LimitRect::buffer_t const& SIXTRL_RESTRICT_REF buffer ) SIXTRL_NOEXCEPT
    {
        return ::NS(LimitRect_get_required_num_dataptrs)( 
            buffer.getCApiPtr(), nullptr );
    }

    LimitRect::size_type LimitRect::RequiredNumDataPtrs( 
        SIXTRL_BUFFER_ARGPTR_DEC const LimitRect::c_buffer_t *const 
            SIXTRL_RESTRICT ptr_buffer ) SIXTRL_NOEXCEPT
    {
        return ::NS(LimitRect_get_required_num_dataptrs)( 
            ptr_buffer, nullptr );
    }
    
    /* --------------------------------------------------------------------- */
    
    void LimitRect::preset() SIXTRL_NOEXCEPT
    {
        ::NS(LimitRect_preset)( this->getCApiPtr() );
    }
    
    void LimitRect::clear() SIXTRL_NOEXCEPT
    {
        ::NS(LimitRect_clear)( this->getCApiPtr() );
    }
    
    /* --------------------------------------------------------------------- */
    
    LimitRect::value_type LimitRect::getMinX() const SIXTRL_NOEXCEPT
    {
        return ::NS(LimitRect_get_min_x)( this->getCApiPtr() );
    }
    
    LimitRect::value_type LimitRect::getMaxX() const SIXTRL_NOEXCEPT
    {
        return ::NS(LimitRect_get_max_x)( this->getCApiPtr() );
    }
    
    LimitRect::value_type LimitRect::getMinY() const SIXTRL_NOEXCEPT
    {
        return ::NS(LimitRect_get_min_y)( this->getCApiPtr() );
    }
    
    LimitRect::value_type LimitRect::getMaxY() const SIXTRL_NOEXCEPT
    {
        return ::NS(LimitRect_get_max_y)( this->getCApiPtr() );
    }
    
    /* --------------------------------------------------------------------- */
    
    void LimitRect::setXLimit( 
        LimitRect::value_type const x_limit ) SIXTRL_NOEXCEPT
    {
        ::NS(LimitRect_set_x_limit)( this->getCApiPtr(), x_limit );
    }
    
    void LimitRect::setYLimit( 
        LimitRect::value_type const y_limit ) SIXTRL_NOEXCEPT
    {
        ::NS(LimitRect_set_y_limit)( this->getCApiPtr(), y_limit );
    }
    
    void LimitRect::setMinX( 
        LimitRect::value_type const min_x ) SIXTRL_NOEXCEPT
    {
        ::NS(LimitRect_set_min_x)( this->getCApiPtr(), min_x );
    }
    
    void LimitRect::setMaxX( 
        LimitRect::value_type const max_x ) SIXTRL_NOEXCEPT
    {
        ::NS(LimitRect_set_max_x)( this->getCApiPtr(), max_x );
    }
    
    void LimitRect::setMinY( 
        LimitRect::value_type const min_y ) SIXTRL_NOEXCEPT
    {
        ::NS(LimitRect_set_min_y)( this->getCApiPtr(), min_y );
    }
    
    void LimitRect::setMaxY( 
        LimitRect::value_type const max_y ) SIXTRL_NOEXCEPT
    {
        ::NS(LimitRect_set_max_y)( this->getCApiPtr(), max_y );
    }
    
    /* --------------------------------------------------------------------- */

    SIXTRL_ARGPTR_DEC LimitRect* LimitRect_new(
        SIXTRL_CXX_NAMESPACE::Buffer& SIXTRL_RESTRICT_REF buffer )
    {
        return static_cast< SIXTRL_ARGPTR_DEC LimitRect* >(
            ::NS(LimitRect_new)( buffer.getCApiPtr() ) );
    }

    SIXTRL_ARGPTR_DEC LimitRect* LimitRect_new(
        SIXTRL_ARGPTR_DEC ::NS(Buffer)* SIXTRL_RESTRICT ptr_buffer )
    {
        return static_cast< SIXTRL_ARGPTR_DEC LimitRect* >(
            ::NS(LimitRect_new)( ptr_buffer ) );
    }

    SIXTRL_ARGPTR_DEC SIXTRL_CXX_NAMESPACE::LimitRect* LimitRect_add(
        SIXTRL_CXX_NAMESPACE::Buffer& SIXTRL_RESTRICT_REF buffer,
        NS(particle_real_t) const min_x, NS(particle_real_t) const max_x,
        NS(particle_real_t) const min_y, NS(particle_real_t) const max_y )
    {
        return static_cast< SIXTRL_ARGPTR_DEC LimitRect* >(::NS(LimitRect_add)( 
            buffer.getCApiPtr(), min_x, max_x, min_y, max_y ) );
    }

    SIXTRL_ARGPTR_DEC LimitRect* LimitRect_add(
        SIXTRL_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
        NS(particle_real_t) const min_x, NS(particle_real_t) const max_x,
        NS(particle_real_t) const min_y, NS(particle_real_t) const max_y )
    {
        return static_cast< SIXTRL_ARGPTR_DEC LimitRect* >(
            ::NS(LimitRect_add)( buffer, min_x, max_x, min_y, max_y ) );
    }

    SIXTRL_ARGPTR_DEC SIXTRL_CXX_NAMESPACE::LimitRect* LimitRect_add_copy(
        SIXTRL_CXX_NAMESPACE::Buffer& SIXTRL_RESTRICT_REF buffer,
        SIXTRL_CXX_NAMESPACE::LimitRect const& SIXTRL_RESTRICT_REF other )
    {
        return static_cast< SIXTRL_ARGPTR_DEC LimitRect* >(
            ::NS(LimitRect_add_copy)( buffer.getCApiPtr(), 
                                      other.getCApiPtr() ) );
    }

    SIXTRL_ARGPTR_DEC SIXTRL_CXX_NAMESPACE::LimitRect* LimitRect_add_copy(
        SIXTRL_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT ptr_buffer,
        SIXTRL_CXX_NAMESPACE::LimitRect const& SIXTRL_RESTRICT_REF other )
    {
        return static_cast< SIXTRL_ARGPTR_DEC LimitRect* >(
            ::NS(LimitRect_add_copy)( ptr_buffer, other.getCApiPtr() ) );
    }
}

#endif /* __cplusplus */

#endif /* SIXTRACKLIB_COMMON_BE_LIMIT_RECT_CXX_HPP__ */

/* end: sixtracklib/common/be_limit/be_limit.hpp */
