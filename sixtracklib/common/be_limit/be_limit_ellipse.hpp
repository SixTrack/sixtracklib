#ifndef SIXTRACKLIB_COMMON_BE_LIMIT_ELLIPSE_CXX_HPP__
#define SIXTRACKLIB_COMMON_BE_LIMIT_ELLIPSE_CXX_HPP__

#if defined( __cplusplus )

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <cstddef>
    #include <cstdlib>
    #include <cmath>
    #include <type_traits>
    #include <utility>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/buffer.hpp"
    #include "sixtracklib/common/be_limit/be_limit_ellipse.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

namespace SIXTRL_CXX_NAMESPACE
{
    template< typename T >
    struct TLimitEllipse
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
            
        SIXTRL_FN TLimitEllipse() = default;

        SIXTRL_FN TLimitEllipse( 
            TLimitEllipse< value_type > const& other ) = default;
            
        SIXTRL_FN TLimitEllipse( 
            TLimitEllipse< value_type >&& other ) = default;

        SIXTRL_FN TLimitEllipse< value_type >& operator=(
            TLimitEllipse< value_type > const& other ) = default;

        SIXTRL_FN TLimitEllipse< value_type >& operator=(
            TLimitEllipse< value_type >&& other ) = default;

        SIXTRL_FN ~TLimitEllipse() = default;
            
        /* ----------------------------------------------------------------- */
        
        SIXTRL_FN const_reference getXOrigin() const SIXTRL_NOEXCEPT;
        SIXTRL_FN const_reference getYOrigin() const SIXTRL_NOEXCEPT;
        
        SIXTRL_FN value_type getXHalfAxis() const SIXTRL_NOEXCEPT;
        SIXTRL_FN value_type getYHalfAxis() const SIXTRL_NOEXCEPT;
        
        SIXTRL_FN const_reference getXHalfAxisSqu() const SIXTRL_NOEXCEPT;
        SIXTRL_FN const_reference getYHalfAxisSqu() const SIXTRL_NOEXCEPT;
        
        SIXTRL_FN const_reference 
            getHalfAxesProductSqu() const SIXTRL_NOEXCEPT;
        
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
    

        SIXTRL_STATIC SIXTRL_FN SIXTRL_ARGPTR_DEC TLimitEllipse< T >*
        CreateNewOnBuffer( buffer_t& SIXTRL_RESTRICT_REF buffer );

        SIXTRL_STATIC SIXTRL_FN SIXTRL_ARGPTR_DEC TLimitEllipse< T >*
        CreateNewOnBuffer( c_buffer_t& SIXTRL_RESTRICT_REF buffer );

        
        SIXTRL_STATIC SIXTRL_FN SIXTRL_ARGPTR_DEC TLimitEllipse< T >* 
        AddToBuffer( buffer_t& SIXTRL_RESTRICT_REF buffer,
            const_reference x_origin, const_reference y_origin,
            const_reference x_half_axis, const_reference y_half_axis );

        SIXTRL_STATIC SIXTRL_FN SIXTRL_ARGPTR_DEC TLimitEllipse< T >* 
        AddToBuffer( c_buffer_t& SIXTRL_RESTRICT_REF buffer,
            const_reference x_origin, const_reference y_origin,
            const_reference x_half_axis, const_reference y_half_axis );
        
        
        SIXTRL_STATIC SIXTRL_FN SIXTRL_ARGPTR_DEC TLimitEllipse< T >* 
        AddCopyToBuffer( buffer_t& SIXTRL_RESTRICT_REF buffer,
            TLimitEllipse< T > const& SIXTRL_RESTRICT_REF other );
        
        SIXTRL_STATIC SIXTRL_FN SIXTRL_ARGPTR_DEC TLimitEllipse< T >* 
        AddCopyToBuffer( c_buffer_t& SIXTRL_RESTRICT_REF buffer,
            TLimitEllipse< T > const& SIXTRL_RESTRICT_REF other );

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
        
        SIXTRL_FN void setXOrigin( const_reference x_origin ) SIXTRL_NOEXCEPT;
        SIXTRL_FN void setYOrigin( const_reference y_origin ) SIXTRL_NOEXCEPT;
        
        SIXTRL_FN void setHalfAxes( 
            const_reference x_half_axis, 
            const_reference y_half_axis ) SIXTRL_NOEXCEPT;
            
        SIXTRL_FN void setHalfAxesSqu( 
            const_reference x_half_axis_squ,
            const_reference y_half_axis_squ ) SIXTRL_NOEXCEPT;
            
        /* ----------------------------------------------------------------- */
        
        value_type  x_origin;
        value_type  y_origin;
        value_type  a_squ;
        value_type  b_squ;
        value_type  a_b_squ;
    };
    
    template< typename T > struct ObjectTypeTraits<
        SIXTRL_CXX_NAMESPACE::TLimitEllipse< T > >
    {
        SIXTRL_STATIC SIXTRL_INLINE object_type_id_t Type() SIXTRL_NOEXCEPT
        {
            return ::NS(OBJECT_TYPE_LIMIT_ELLIPSE);
        }
    };

    template< typename T >
    SIXTRL_ARGPTR_DEC TLimitEllipse< T >* TLimitEllipse_new(
        typename TLimitEllipse< T >::buffer_t& SIXTRL_RESTRICT_REF buffer );

    template< typename T >
    SIXTRL_ARGPTR_DEC TLimitEllipse< T >* TLimitEllipse_new(
        typename TLimitEllipse< T >::c_buffer_t& SIXTRL_RESTRICT_REF buffer );


    template< typename T >
    SIXTRL_ARGPTR_DEC TLimitEllipse< T >* TLimitEllipse_add(
        typename TLimitEllipse< T >::buffer_t& SIXTRL_RESTRICT_REF buffer,
        typename TLimitEllipse< T >::const_reference SIXTRL_RESTRICT_REF x_origin,
        typename TLimitEllipse< T >::const_reference SIXTRL_RESTRICT_REF y_origin,
        typename TLimitEllipse< T >::const_reference SIXTRL_RESTRICT_REF x_haxis,
        typename TLimitEllipse< T >::const_reference SIXTRL_RESTRICT_REF y_haxis );

    template< typename T >
    SIXTRL_ARGPTR_DEC TLimitEllipse< T >* TLimitEllipse_add(
        typename TLimitEllipse< T >::c_buffer_t& SIXTRL_RESTRICT_REF buffer,
        typename TLimitEllipse< T >::const_reference SIXTRL_RESTRICT_REF x_origin,
        typename TLimitEllipse< T >::const_reference SIXTRL_RESTRICT_REF y_origin,
        typename TLimitEllipse< T >::const_reference SIXTRL_RESTRICT_REF x_haxis,
        typename TLimitEllipse< T >::const_reference SIXTRL_RESTRICT_REF y_haxis );

    template< typename T >
    SIXTRL_ARGPTR_DEC TLimitEllipse< T >* TLimitEllipse_add_copy(
        typename TLimitEllipse< T >::buffer_t& SIXTRL_RESTRICT_REF buffer,
        TLimitEllipse< T > const& SIXTRL_RESTRICT_REF other );

    template< typename T >
    SIXTRL_ARGPTR_DEC TLimitEllipse< T >* TLimitEllipse_add_copy(
        typename TLimitEllipse< T >::c_buffer_t& SIXTRL_RESTRICT_REF buffer,
        TLimitEllipse< T > const& SIXTRL_RESTRICT_REF other );
    
    /* ===================================================================== *
     * ====  Specialization TLimitEllipse< ::NS(particle_real_t) > :
     * ===================================================================== */
    
    template<> 
    struct TLimitEllipse< ::NS(particle_real_t) > : public ::NS(LimitEllipse)
    {
        using value_type      = ::NS(particle_real_t);
        using reference       = ::NS(particle_real_t)&;
        using const_reference = ::NS(particle_real_t) const&;
        using type_id_t       = ::NS(object_type_id_t);
        using buffer_t        = SIXTRL_CXX_NAMESPACE::Buffer;
        using size_type       = buffer_t::size_type;
        using c_buffer_t      = buffer_t::c_api_t;
        using c_api_t         = ::NS(LimitEllipse);
        
        /* ----------------------------------------------------------------- */
        
        static SIXTRL_CONSTEXPR_OR_CONST value_type  DEFAULT_MAX_X = 
            SIXTRL_CXX_NAMESPACE::LIMIT_DEFAULT_MAX_X;
        
        static SIXTRL_CONSTEXPR_OR_CONST value_type  DEFAULT_MIN_X = 
            SIXTRL_CXX_NAMESPACE::LIMIT_DEFAULT_MIN_X;
            
        static SIXTRL_CONSTEXPR_OR_CONST value_type  DEFAULT_MAX_Y = 
            SIXTRL_CXX_NAMESPACE::LIMIT_DEFAULT_MAX_Y;
        
        static SIXTRL_CONSTEXPR_OR_CONST value_type  DEFAULT_MIN_Y = 
            SIXTRL_CXX_NAMESPACE::LIMIT_DEFAULT_MIN_Y;
        
        /* ----------------------------------------------------------------- */
            
        SIXTRL_FN TLimitEllipse() = default;

        SIXTRL_FN TLimitEllipse( 
            TLimitEllipse< ::NS(particle_real_t) > const& other ) = default;
            
        SIXTRL_FN TLimitEllipse( 
            TLimitEllipse< ::NS(particle_real_t) >&& other ) = default;

        SIXTRL_FN TLimitEllipse< value_type >& operator=(
            TLimitEllipse< ::NS(particle_real_t) > const& other ) = default;

        SIXTRL_FN TLimitEllipse< value_type >& operator=(
            TLimitEllipse< ::NS(particle_real_t) >&& other ) = default;

        SIXTRL_FN ~TLimitEllipse() = default;
            
        /* ----------------------------------------------------------------- */
        
        SIXTRL_FN value_type getXOrigin() const SIXTRL_NOEXCEPT;
        SIXTRL_FN value_type getYOrigin() const SIXTRL_NOEXCEPT;
        
        SIXTRL_FN value_type getXHalfAxis() const SIXTRL_NOEXCEPT;
        SIXTRL_FN value_type getYHalfAxis() const SIXTRL_NOEXCEPT;
        
        SIXTRL_FN value_type getXHalfAxisSqu() const SIXTRL_NOEXCEPT;
        SIXTRL_FN value_type getYHalfAxisSqu() const SIXTRL_NOEXCEPT;
        
        SIXTRL_FN value_type getHalfAxesProductSqu() const SIXTRL_NOEXCEPT;
        
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
    

        
        SIXTRL_STATIC SIXTRL_FN SIXTRL_ARGPTR_DEC 
        TLimitEllipse< ::NS(particle_real_t) >*
        CreateNewOnBuffer( buffer_t& SIXTRL_RESTRICT_REF buffer );

        SIXTRL_STATIC SIXTRL_FN SIXTRL_ARGPTR_DEC 
        TLimitEllipse< ::NS(particle_real_t) >*
        CreateNewOnBuffer( c_buffer_t& SIXTRL_RESTRICT_REF buffer );

        
        SIXTRL_STATIC SIXTRL_FN SIXTRL_ARGPTR_DEC 
        TLimitEllipse< ::NS(particle_real_t) >*
        AddToBuffer( buffer_t& SIXTRL_RESTRICT_REF buffer,
            value_type const x_origin, value_type const y_origin,
            value_type const x_half_axis, value_type const y_half_axis );

        SIXTRL_STATIC SIXTRL_FN SIXTRL_ARGPTR_DEC 
        TLimitEllipse< ::NS(particle_real_t) >*
        AddToBuffer( c_buffer_t& SIXTRL_RESTRICT_REF buffer,
            value_type const x_origin, value_type const y_origin,
            value_type const x_half_axis, value_type const y_half_axis );
        
        
        SIXTRL_STATIC SIXTRL_FN SIXTRL_ARGPTR_DEC 
        TLimitEllipse< ::NS(particle_real_t) >*
        AddCopyToBuffer( buffer_t& SIXTRL_RESTRICT_REF buffer,
            TLimitEllipse< ::NS(particle_real_t) > const& 
                SIXTRL_RESTRICT_REF other );
        
        SIXTRL_STATIC SIXTRL_FN SIXTRL_ARGPTR_DEC 
        TLimitEllipse< ::NS(particle_real_t) >*
        AddCopyToBuffer( c_buffer_t& SIXTRL_RESTRICT_REF buffer,
            TLimitEllipse< ::NS(particle_real_t) > const& 
                SIXTRL_RESTRICT_REF other );
        
        /* ----------------------------------------------------------------- */

        SIXTRL_ARGPTR_DEC c_api_t const* getCApiPtr() const SIXTRL_NOEXCEPT;
        SIXTRL_ARGPTR_DEC c_api_t* getCApiPtr() SIXTRL_NOEXCEPT;

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
        
        SIXTRL_FN void setXOrigin( value_type const x_origin ) SIXTRL_NOEXCEPT;
        SIXTRL_FN void setYOrigin( value_type const y_origin ) SIXTRL_NOEXCEPT;
        
        SIXTRL_FN void setHalfAxes( value_type const x_half_axis,
            value_type const y_half_axis ) SIXTRL_NOEXCEPT;
        
        SIXTRL_FN void setHalfAxesSqu( value_type const x_half_axis_squ,
            value_type const y_half_axis_squ ) SIXTRL_NOEXCEPT;            
    };
    
    using LimitEllipse = TLimitEllipse< ::NS(particle_real_t) >;
    
    SIXTRL_ARGPTR_DEC LimitEllipse* LimitEllipse_new(
        LimitEllipse::buffer_t& SIXTRL_RESTRICT_REF buffer );

    SIXTRL_ARGPTR_DEC LimitEllipse* TLimitEllipse_new( 
        LimitEllipse::c_buffer_t& SIXTRL_RESTRICT_REF buffer );


    SIXTRL_ARGPTR_DEC LimitEllipse* LimitEllipse_add(
        LimitEllipse::buffer_t& SIXTRL_RESTRICT_REF buffer,
        LimitEllipse::value_type const x_origin,
        LimitEllipse::value_type const y_origin,
        LimitEllipse::value_type const x_haxis,
        LimitEllipse::value_type const y_haxis );

    SIXTRL_ARGPTR_DEC LimitEllipse* LimitEllipse_add(
        LimitEllipse::c_buffer_t& SIXTRL_RESTRICT_REF buffer,
        LimitEllipse::value_type const x_origin,
        LimitEllipse::value_type const y_origin,
        LimitEllipse::value_type const x_haxis, 
        LimitEllipse::value_type const y_haxis );

    SIXTRL_ARGPTR_DEC LimitEllipse* LimitEllipse_add_copy(
        LimitEllipse::buffer_t& SIXTRL_RESTRICT_REF buffer,
        LimitEllipse const& SIXTRL_RESTRICT_REF other );

    SIXTRL_ARGPTR_DEC LimitEllipse* LimitEllipse_add_copy(
        LimitEllipse::c_buffer_t& SIXTRL_RESTRICT_REF buffer,
        LimitEllipse const& SIXTRL_RESTRICT_REF other );
}

/* ************************************************************************* *
 * *** Implementation of inline and template member functions          ***** *
 * ************************************************************************* */

namespace SIXTRL_CXX_NAMESPACE
{
    template< typename T >
    typename TLimitEllipse< T >::const_reference 
    TLimitEllipse< T >::getXOrigin() const SIXTRL_NOEXCEPT
    {
        return this->x_origin;
    }
    
    template< typename T > typename TLimitEllipse< T >::const_reference 
    TLimitEllipse< T >::getYOrigin() const SIXTRL_NOEXCEPT
    {
        return this->y_origin;
    }
    
    template< typename T > typename TLimitEllipse< T >::value_type 
    TLimitEllipse< T >::getXHalfAxis() const SIXTRL_NOEXCEPT
    {
        return std::sqrt( this->a_squ );
    }
    
    template< typename T > typename TLimitEllipse< T >::value_type 
    TLimitEllipse< T >::getYHalfAxis() const SIXTRL_NOEXCEPT
    {
        return std::sqrt( this->b_squ );
    }
    
    template< typename T > typename TLimitEllipse< T >::const_reference 
    TLimitEllipse< T >::getXHalfAxisSqu() const SIXTRL_NOEXCEPT
    {
        return this->a_squ;
    }
    
    template< typename T > typename TLimitEllipse< T >::const_reference 
    TLimitEllipse< T >::getYHalfAxisSqu() const SIXTRL_NOEXCEPT
    {
        return this->b_squ;
    }
    
    template< typename T > typename TLimitEllipse< T >::const_reference 
    TLimitEllipse< T >::getHalfAxesProductSqu() const SIXTRL_NOEXCEPT
    {
        return this->a_b_squ;
    }
    
    /* --------------------------------------------------------------------- */
    
    template< typename T >
    SIXTRL_INLINE bool TLimitEllipse< T >::CanAddToBuffer(
        typename TLimitEllipse< T >::buffer_t& SIXTRL_RESTRICT_REF buffer,
        SIXTRL_ARGPTR_DEC typename TLimitEllipse< T >::size_type* 
            SIXTRL_RESTRICT req_objects,
        SIXTRL_ARGPTR_DEC typename TLimitEllipse< T >::size_type* 
            SIXTRL_RESTRICT req_slots,
        SIXTRL_ARGPTR_DEC typename TLimitEllipse< T >::size_type* 
            SIXTRL_RESTRICT req_dataptrs ) SIXTRL_NOEXCEPT
    {
        return TLimitEllipse< T >::CanAddToBuffer( buffer.getCApiPtr(),
            req_objects, req_slots, req_dataptrs );
    }

    template< typename T >
    SIXTRL_INLINE bool TLimitEllipse< T >::CanAddToBuffer(
        SIXTRL_BUFFER_ARGPTR_DEC typename TLimitEllipse< T >::c_buffer_t* 
            SIXTRL_RESTRICT ptr_buffer,
        SIXTRL_ARGPTR_DEC typename TLimitEllipse< T >::size_type* 
            SIXTRL_RESTRICT req_objects,
        SIXTRL_ARGPTR_DEC typename TLimitEllipse< T >::size_type* 
            SIXTRL_RESTRICT req_slots,
        SIXTRL_ARGPTR_DEC typename TLimitEllipse< T >::size_type* 
            SIXTRL_RESTRICT req_dataptrs ) SIXTRL_NOEXCEPT
    {
        using _this_t = SIXTRL_CXX_NAMESPACE::TLimitEllipse< T >;
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


    template< typename T > SIXTRL_INLINE SIXTRL_ARGPTR_DEC TLimitEllipse< T >*
    TLimitEllipse< T >::CreateNewOnBuffer( 
        typename TLimitEllipse< T >::buffer_t& SIXTRL_RESTRICT_REF buffer )
    {
        return TLimitEllipse< T >::CreateNewOnBuffer( *buffer.getCApiPtr() );
    }

    template< typename T > SIXTRL_INLINE SIXTRL_ARGPTR_DEC TLimitEllipse< T >*
    TLimitEllipse< T >::CreateNewOnBuffer( 
        typename TLimitEllipse< T >::c_buffer_t& SIXTRL_RESTRICT_REF buffer )
    {
        using _this_t = SIXTRL_CXX_NAMESPACE::TLimitEllipse< T >;
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
    
    template< typename T > SIXTRL_INLINE SIXTRL_ARGPTR_DEC TLimitEllipse< T >* 
    TLimitEllipse< T >::AddToBuffer( 
        typename TLimitEllipse< T >::buffer_t& SIXTRL_RESTRICT_REF buffer,
        typename TLimitEllipse< T >::const_reference x_origin, 
        typename TLimitEllipse< T >::const_reference y_origin,
        typename TLimitEllipse< T >::const_reference x_half_axis, 
        typename TLimitEllipse< T >::const_reference y_half_axis )
    {
        return TLimitEllipse< T >::AddToBuffer( *buffer.getCApiPtr(),
            x_origin, y_origin, x_half_axis, y_half_axis );
    }

    template< typename T > SIXTRL_INLINE SIXTRL_ARGPTR_DEC TLimitEllipse< T >* 
    TLimitEllipse< T >::AddToBuffer( 
        typename TLimitEllipse< T >::c_buffer_t& SIXTRL_RESTRICT_REF buffer,
        typename TLimitEllipse< T >::const_reference x_origin, 
        typename TLimitEllipse< T >::const_reference y_origin,
        typename TLimitEllipse< T >::const_reference x_half_axis, 
        typename TLimitEllipse< T >::const_reference y_half_axis )
    {
        using _this_t = SIXTRL_CXX_NAMESPACE::TLimitEllipse< T >;
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
        temp.setXOrigin( x_origin );
        temp.setYOrigin( y_origin );
        temp.setHalfAxes( x_half_axis, y_half_axis );

        return reinterpret_cast< ptr_t >( ::NS(Object_get_begin_addr)(
            ::NS(Buffer_add_object)( &buffer, &temp, sizeof( _this_t ),
                temp.getTypeId(), num_dataptrs, offsets, sizes, counts ) ) );
    }
    
    
    template< typename T > SIXTRL_INLINE SIXTRL_ARGPTR_DEC TLimitEllipse< T >* 
    TLimitEllipse< T >::AddCopyToBuffer( 
        typename TLimitEllipse< T >::buffer_t& SIXTRL_RESTRICT_REF buffer,
        TLimitEllipse< T > const& SIXTRL_RESTRICT_REF other )
    {
        TLimitEllipse< T >::AddCopyToBuffer( *buffer.getCApiPtr(), other );
    }
    
    template< typename T > SIXTRL_INLINE SIXTRL_ARGPTR_DEC TLimitEllipse< T >* 
    TLimitEllipse< T >::AddCopyToBuffer( 
        typename TLimitEllipse< T >::c_buffer_t& SIXTRL_RESTRICT_REF buffer,
        TLimitEllipse< T > const& SIXTRL_RESTRICT_REF other )
    {
        using _this_t = SIXTRL_CXX_NAMESPACE::TLimitEllipse< T >;
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
        ::NS(arch_status_t) const status = ::NS(LimitEllipse_copy)(
            temp.getCApiPtr(), other.getCApiPtr() );
        
        if( status == ::NS(ARCH_STATUS_SUCCESS) )
        {
            return reinterpret_cast< ptr_t >( ::NS(Object_get_begin_addr)(
                ::NS(Buffer_add_object)( &buffer, &temp, sizeof( _this_t ),
                    temp.getTypeId(), num_dataptrs, offsets, sizes, counts ) ) );
        }
        
        return nullptr;
    }

    /* --------------------------------------------------------------------- */

    template< typename T > typename TLimitEllipse< T >::type_id_t 
    TLimitEllipse< T >::getTypeId() const SIXTRL_NOEXCEPT
    {
        return ::NS(OBJECT_TYPE_LIMIT_ELLIPSE);
    }

    template< typename T >
    SIXTRL_INLINE typename TLimitEllipse< T >::size_type 
    TLimitEllipse< T >::RequiredNumDataPtrs( 
        typename TLimitEllipse< T >::buffer_t const& SIXTRL_RESTRICT_REF 
            buffer ) SIXTRL_NOEXCEPT
    {
        return ::NS(LimitEllipse_get_required_num_dataptrs)(
            buffer.getCApiPtr(), nullptr );
    }

    template< typename T >
    SIXTRL_INLINE typename TLimitEllipse< T >::size_type 
    TLimitEllipse< T >::RequiredNumDataPtrs( 
        SIXTRL_BUFFER_ARGPTR_DEC const c_buffer_t *const SIXTRL_RESTRICT 
            ptr_buffer ) SIXTRL_NOEXCEPT
    {
        return ::NS(LimitEllipse_get_required_num_dataptrs)( 
            ptr_buffer, nullptr );
    }
            
    /* --------------------------------------------------------------------- */
        
    template< typename T >
    void TLimitEllipse< T >::preset() SIXTRL_NOEXCEPT
    {
        using _this_t = TLimitEllipse< T >;
        using  real_t = typename _this_t::value_type;
        
        real_t const x_half_axis = real_t{ 0.5 } * ( 
            _this_t::DEFAULT_MAX_X - _this_t::DEFAULT_MIN_X );
        
        real_t const y_half_axis = real_t{ 0.5 } * ( 
            _this_t::DEFAULT_MAX_Y - _this_t::DEFAULT_MIN_Y );
        
        real_t const x_origin = real_t{ 0.5 } * (
            _this_t::DEFAULT_MAX_X + _this_t::DEFAULT_MIN_X );
        
        real_t const y_origin = real_t{ 0.5 } * (
            _this_t::DEFAULT_MAX_Y + _this_t::DEFAULT_MIN_Y );
        
        this->setXOrigin( x_origin );
        this->setYOrigin( y_origin );        
        this->setHalfAxes( x_half_axis, y_half_axis );
    }
    
    template< typename T >
    void TLimitEllipse< T >::clear() SIXTRL_NOEXCEPT
    {
        this->preset();
    }
    
    /* --------------------------------------------------------------------- */
    
    template< typename T >
    void TLimitEllipse< T >::setXOrigin( 
        typename TLimitEllipse< T >::const_reference x_origin ) SIXTRL_NOEXCEPT
    {
        this->x_origin = x_origin;
    }
    
    template< typename T >
    void TLimitEllipse< T >::setYOrigin( 
        typename TLimitEllipse< T >::const_reference y_origin ) SIXTRL_NOEXCEPT
    {
        this->y_origin = y_origin;
    }
    
    template< typename T >
    void TLimitEllipse< T >::setHalfAxes( 
        typename TLimitEllipse< T >::const_reference x_half_axis, 
        typename TLimitEllipse< T >::const_reference y_half_axis 
    ) SIXTRL_NOEXCEPT
    {
        this->setHalfAxesSqu( x_half_axis * x_half_axis, 
                              y_half_axis * y_half_axis );
    }
        
    template< typename T >
    void TLimitEllipse< T >::setHalfAxesSqu( 
        typename TLimitEllipse< T >::const_reference x_half_axis_squ,
        typename TLimitEllipse< T >::const_reference y_half_axis_squ 
        ) SIXTRL_NOEXCEPT
    {
        using _this_t = TLimitEllipse< T >;
        using  real_t = typename _this_t::real_t;
        
        SIXTRL_ASSERT( x_half_axis_squ >= real_t{ 0 } );
        SIXTRL_ASSERT( y_half_axis_squ >= real_t{ 0 } );
        
        this->a_squ = x_half_axis_squ;
        this->b_squ = y_half_axis_squ;
        this->a_b_squ = x_half_axis_squ * y_half_axis_squ;
    }
        
    /* ===================================================================== *
     * ====  Specialization TLimitEllipse< ::NS(particle_real_t) > :
     * ===================================================================== */
    
    LimitEllipse::value_type LimitEllipse::getXOrigin() const SIXTRL_NOEXCEPT
    {
        return ::NS(LimitEllipse_get_x_origin)( this->getCApiPtr() );
    }
    
    LimitEllipse::value_type LimitEllipse::getYOrigin() const SIXTRL_NOEXCEPT
    {
        return ::NS(LimitEllipse_get_y_origin)( this->getCApiPtr() );
    }
    
    LimitEllipse::value_type LimitEllipse::getXHalfAxis() const SIXTRL_NOEXCEPT
    {
        return ::NS(LimitEllipse_get_x_half_axis)( this->getCApiPtr() );
    }
    
    LimitEllipse::value_type LimitEllipse::getYHalfAxis() const SIXTRL_NOEXCEPT
    {
        return ::NS(LimitEllipse_get_y_half_axis)( this->getCApiPtr() );
    }
    
    LimitEllipse::value_type 
    LimitEllipse::getXHalfAxisSqu() const SIXTRL_NOEXCEPT
    {
        return ::NS(LimitEllipse_get_x_half_axis_squ)( this->getCApiPtr() );
    }
    
    LimitEllipse::value_type 
    LimitEllipse::getYHalfAxisSqu() const SIXTRL_NOEXCEPT
    {
        return ::NS(LimitEllipse_get_y_half_axis_squ)( this->getCApiPtr() );
    }
    
    LimitEllipse::value_type 
    LimitEllipse::getHalfAxesProductSqu() const SIXTRL_NOEXCEPT
    {
        return ::NS(LimitEllipse_get_half_axes_product_squ)( 
            this->getCApiPtr() );
    }
    
    /* --------------------------------------------------------------------- */
    
    SIXTRL_INLINE bool LimitEllipse::CanAddToBuffer(
        LimitEllipse::buffer_t& SIXTRL_RESTRICT_REF buffer,
        SIXTRL_ARGPTR_DEC LimitEllipse::size_type* SIXTRL_RESTRICT req_objects,
        SIXTRL_ARGPTR_DEC LimitEllipse::size_type* SIXTRL_RESTRICT req_slots,
        SIXTRL_ARGPTR_DEC LimitEllipse::size_type* SIXTRL_RESTRICT req_dataptrs 
    ) SIXTRL_NOEXCEPT
    {
        return ::NS(LimitEllipse_can_be_added)( 
            buffer.getCApiPtr(), req_objects, req_slots, req_dataptrs );
    }

    SIXTRL_INLINE bool LimitEllipse::CanAddToBuffer( SIXTRL_BUFFER_ARGPTR_DEC 
        LimitEllipse::c_buffer_t* SIXTRL_RESTRICT ptr_buffer,
        SIXTRL_ARGPTR_DEC LimitEllipse::size_type* SIXTRL_RESTRICT req_objects,
        SIXTRL_ARGPTR_DEC LimitEllipse::size_type* SIXTRL_RESTRICT req_slots,
        SIXTRL_ARGPTR_DEC LimitEllipse::size_type* SIXTRL_RESTRICT req_dataptrs
    ) SIXTRL_NOEXCEPT
    {
        return ::NS(LimitEllipse_can_be_added)( 
            ptr_buffer, req_objects, req_slots, req_dataptrs );
    }


    SIXTRL_INLINE SIXTRL_ARGPTR_DEC LimitEllipse*
    LimitEllipse::CreateNewOnBuffer( 
        LimitEllipse::buffer_t& SIXTRL_RESTRICT_REF buffer )
    {
        return static_cast< SIXTRL_ARGPTR_DEC LimitEllipse* >(
            ::NS(LimitEllipse_new)( buffer.getCApiPtr() ) );
    }

    SIXTRL_INLINE SIXTRL_ARGPTR_DEC LimitEllipse*
    LimitEllipse::CreateNewOnBuffer( 
        LimitEllipse::c_buffer_t& SIXTRL_RESTRICT_REF buffer )
    {
        return static_cast< SIXTRL_ARGPTR_DEC LimitEllipse* >(
            ::NS(LimitEllipse_new)( &buffer ) );
    }

    
    SIXTRL_INLINE SIXTRL_ARGPTR_DEC LimitEllipse* 
    LimitEllipse::AddToBuffer( 
        LimitEllipse::buffer_t& SIXTRL_RESTRICT_REF buffer,
        LimitEllipse::value_type const x_origin, 
        LimitEllipse::value_type const y_origin,
        LimitEllipse::value_type const x_half_axis, 
        LimitEllipse::value_type const y_half_axis )
    {
        return static_cast< SIXTRL_ARGPTR_DEC LimitEllipse* >(
            ::NS(LimitEllipse_add)( buffer.getCApiPtr(), 
                x_origin, y_origin, x_half_axis, y_half_axis ) );
    }

    SIXTRL_INLINE SIXTRL_ARGPTR_DEC LimitEllipse* 
    LimitEllipse::AddToBuffer( 
        LimitEllipse::c_buffer_t& SIXTRL_RESTRICT_REF buffer,
        LimitEllipse::value_type const x_origin, 
        LimitEllipse::value_type const y_origin,
        LimitEllipse::value_type const x_half_axis, 
        LimitEllipse::value_type const y_half_axis )
    {
        return static_cast< SIXTRL_ARGPTR_DEC LimitEllipse* >(
            ::NS(LimitEllipse_add)( &buffer, x_origin, y_origin, 
                                    x_half_axis, y_half_axis ) );
    }
    
    
    SIXTRL_INLINE SIXTRL_ARGPTR_DEC LimitEllipse* 
    LimitEllipse::AddCopyToBuffer( 
        LimitEllipse::buffer_t& SIXTRL_RESTRICT_REF buffer,
        LimitEllipse const& SIXTRL_RESTRICT_REF other )
    {
        return static_cast< SIXTRL_ARGPTR_DEC LimitEllipse* >(
            ::NS(LimitEllipse_add_copy)( 
                buffer.getCApiPtr(), other.getCApiPtr() ) );
    }
    
    SIXTRL_INLINE SIXTRL_ARGPTR_DEC LimitEllipse* 
    LimitEllipse::AddCopyToBuffer( 
        LimitEllipse::c_buffer_t& SIXTRL_RESTRICT_REF buffer,
        LimitEllipse const& SIXTRL_RESTRICT_REF other )
    {
        return static_cast< SIXTRL_ARGPTR_DEC LimitEllipse* >(
            ::NS(LimitEllipse_add_copy)( &buffer, other.getCApiPtr() ) );
    }
    
    /* --------------------------------------------------------------------- */
    
    SIXTRL_ARGPTR_DEC LimitEllipse::c_api_t const* 
    LimitEllipse::getCApiPtr() const SIXTRL_NOEXCEPT
    {
        return reinterpret_cast< LimitEllipse::c_api_t const* >( this );
    }

    SIXTRL_ARGPTR_DEC LimitEllipse::c_api_t*
    LimitEllipse::getCApiPtr() SIXTRL_NOEXCEPT
    {
        return const_cast< LimitEllipse::c_api_t* >(
            static_cast< LimitEllipse const& >( *this ).getCApiPtr() );
    }

    /* --------------------------------------------------------------------- */

    LimitEllipse::type_id_t LimitEllipse::getTypeId() const SIXTRL_NOEXCEPT
    {
        return ::NS(OBJECT_TYPE_LIMIT_ELLIPSE);
    }

    SIXTRL_INLINE LimitEllipse::size_type LimitEllipse::RequiredNumDataPtrs(
        LimitEllipse::buffer_t const& SIXTRL_RESTRICT_REF 
            buffer ) SIXTRL_NOEXCEPT
    {
        return ::NS(LimitEllipse_get_required_num_dataptrs)( 
            buffer.getCApiPtr(), nullptr );
    }

    SIXTRL_INLINE LimitEllipse::size_type LimitEllipse::RequiredNumDataPtrs(
        SIXTRL_BUFFER_ARGPTR_DEC const LimitEllipse::c_buffer_t *const 
            SIXTRL_RESTRICT ptr_buffer ) SIXTRL_NOEXCEPT
    {
        return ::NS(LimitEllipse_get_required_num_dataptrs)( 
            ptr_buffer, nullptr );
    }
            
    /* --------------------------------------------------------------------- */
        
    void LimitEllipse::preset() SIXTRL_NOEXCEPT
    {
        ::NS(LimitEllipse_preset)( this->getCApiPtr() );
    }
    
    void LimitEllipse::clear() SIXTRL_NOEXCEPT
    {
        ::NS(LimitEllipse_clear)( this->getCApiPtr() );
    }
    
    /* --------------------------------------------------------------------- */
    
    void LimitEllipse::setXOrigin( 
        LimitEllipse::value_type const x_origin ) SIXTRL_NOEXCEPT
    {
        ::NS(LimitEllipse_set_x_origin)( this->getCApiPtr(), x_origin );
    }
    
    void LimitEllipse::setYOrigin( 
        LimitEllipse::value_type const y_origin ) SIXTRL_NOEXCEPT
    {
        ::NS(LimitEllipse_set_y_origin)( this->getCApiPtr(), y_origin );
    }
    
    void LimitEllipse::setHalfAxes( 
        LimitEllipse::value_type const x_half_axis,
        LimitEllipse::value_type const y_half_axis ) SIXTRL_NOEXCEPT
    {
        ::NS(LimitEllipse_set_half_axes)( 
            this->getCApiPtr(), x_half_axis, y_half_axis );
    }
    
    void LimitEllipse::setHalfAxesSqu( 
        LimitEllipse::value_type const x_half_axis_squ,
        LimitEllipse::value_type const y_half_axis_squ ) SIXTRL_NOEXCEPT
    {
        ::NS(LimitEllipse_set_half_axes_squ)( this->getCApiPtr(),
            x_half_axis_squ, y_half_axis_squ );
    }
    
    /* --------------------------------------------------------------------- */
        
    SIXTRL_INLINE SIXTRL_ARGPTR_DEC LimitEllipse* LimitEllipse_new(
        LimitEllipse::buffer_t& SIXTRL_RESTRICT_REF buffer )
    {
        return LimitEllipse::CreateNewOnBuffer( buffer );
    }

    SIXTRL_INLINE SIXTRL_ARGPTR_DEC LimitEllipse* TLimitEllipse_new( 
        LimitEllipse::c_buffer_t& SIXTRL_RESTRICT_REF buffer )
    {
        return LimitEllipse::CreateNewOnBuffer( buffer );
    }

    SIXTRL_INLINE SIXTRL_ARGPTR_DEC LimitEllipse* LimitEllipse_add(
        LimitEllipse::buffer_t& SIXTRL_RESTRICT_REF buffer,
        LimitEllipse::value_type const x_origin,
        LimitEllipse::value_type const y_origin,
        LimitEllipse::value_type const x_half_axis,
        LimitEllipse::value_type const y_half_axis )
    {
        return LimitEllipse::AddToBuffer( *buffer.getCApiPtr(), 
            x_origin, y_origin, x_half_axis, y_half_axis );
    }

    SIXTRL_INLINE SIXTRL_ARGPTR_DEC LimitEllipse* LimitEllipse_add(
        LimitEllipse::c_buffer_t& SIXTRL_RESTRICT_REF buffer,
        LimitEllipse::value_type const x_origin,
        LimitEllipse::value_type const y_origin,
        LimitEllipse::value_type const x_half_axis, 
        LimitEllipse::value_type const y_half_axis )
    {
        return LimitEllipse::AddToBuffer( 
            buffer, x_origin, y_origin, x_half_axis, y_half_axis );
    }

    SIXTRL_INLINE SIXTRL_ARGPTR_DEC LimitEllipse* LimitEllipse_add_copy(
        LimitEllipse::buffer_t& SIXTRL_RESTRICT_REF buffer,
        LimitEllipse const& SIXTRL_RESTRICT_REF other )
    {
        return LimitEllipse::AddCopyToBuffer( *buffer.getCApiPtr(), other );
    }

    SIXTRL_INLINE SIXTRL_ARGPTR_DEC LimitEllipse* LimitEllipse_add_copy(
        LimitEllipse::c_buffer_t& SIXTRL_RESTRICT_REF buffer,
        LimitEllipse const& SIXTRL_RESTRICT_REF other )
    {
        return LimitEllipse::AddCopyToBuffer( buffer, other );
    }
}

#endif /* !defined( _GPUCODE ) */

#endif /* defined( SIXTRACKLIB_COMMON_BE_LIMIT_ELLIPSE_CXX_HPP__ ) */

/* end: sixtracklib/common/be_limit/be_limit_ellipse.hpp */
