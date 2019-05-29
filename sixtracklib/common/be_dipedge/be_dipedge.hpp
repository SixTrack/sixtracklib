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
    template< typename T >
    struct TDipoleEdge
    {
        using value_type      = T;
        using reference       = T&;
        using const_reference = T const&;
        using type_id_t       = ::NS(object_type_id_t);
        using size_type       = ::NS(buffer_size_t);
        using buffer_t        = Buffer;

        static SIXTRL_CONSTEXPR_OR_CONST value_type
            DEFAULT_X_LIMIT = static_cast< value_type >(
                SIXTRL_DEFAULT_X_LIMIT );

        static SIXTRL_CONSTEXPR_OR_CONST value_type
            DEFAULT_Y_LIMIT = static_cast< value_type >(
                SIXTRL_DEFAULT_Y_LIMIT );

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


        SIXTRL_STATIC SIXTRL_FN SIXTRL_ARGPTR_DEC TDipoleEdge< T >*
        CreateNewOnBuffer( buffer_t& SIXTRL_RESTRICT_REF buffer );

        SIXTRL_STATIC SIXTRL_FN SIXTRL_ARGPTR_DEC TDipoleEdge< T >*
        CreateNewOnBuffer(
            SIXTRL_BUFFER_ARGPTR_DEC c_buffer_t* SIXTRL_RESTRICT ptr_buffer );



        SIXTRL_STATIC SIXTRL_FN SIXTRL_ARGPTR_DEC TDipoleEdge< T >* AddToBuffer(
            buffer_t& SIXTRL_RESTRICT_REF buffer,
            value_type const& dx, value_type const& dy );

        SIXTRL_STATIC SIXTRL_FN SIXTRL_ARGPTR_DEC TDipoleEdge< T >* AddToBuffer(
            SIXTRL_BUFFER_ARGPTR_DEC c_buffer_t* SIXTRL_RESTRICT ptr_buffer,
            value_type const& dx, value_type const& dy );

        /* ---------------------------------------------------------------- */

        SIXTRL_FN type_id_t getTypeId() const SIXTRL_NOEXCEPT;

        SIXTRL_STATIC SIXTRL_FN size_type RequiredNumDataPtrs(
            buffer_t const& SIXTRL_RESTRICT_REF buffer ) const SIXTRL_NOEXCEPT;

        /* ----------------------------------------------------------------- */
            
        SIXTRL_FN value_type const& getInvRho() const SIXTRL_NOEXCEPT;
        SIXTRL_FN value_type getRho() const SIXTRL_NOEXCEPT;
        
        SIXTRL_FN value_type const& getCosRotAngle() const SIXTRL_NOEXCEPT;
        SIXTRL_FN value_type const& getTanRotAngle() const SIXTRL_NOEXCEPT;
        SIXTRL_FN value_type getSinRotAngle() const SIXTRL_NOEXCEPT;

        SIXTRL_FN value_type getRotAngleDeg() const SIXTRL_NOEXCEPT;
        SIXTRL_FN value_type getRotAngleRad() const SIXTRL_NOEXCEPT;
        
        SIXTRL_FN value_type const& getCosTiltAngle() const SIXTRL_NOEXCEPT;
        SIXTRL_FN value_type const& getSinTiltAngle() const SIXTRL_NOEXCEPT;

        SIXTRL_FN value_type getTiltAngleDeg() const SIXTRL_NOEXCEPT;
        SIXTRL_FN value_type getTiltAngleRad() const SIXTRL_NOEXCEPT;
        
        SIXTRL_FN value_type const& getB() const SIXTRL_NOEXCEPT;
        
        /* ----------------------------------------------------------------- */
        
        SIXTRL_FN void setInvRho( value_type const& SIXTRL_RESTRICT_REF
            inv_rho ) SIXTRL_NOEXCEPT;
            
        SIXTRL_FN void setCosRotAngle( value_type const& SIXTRL_RESTRICT_REF 
            cos_rot_angle ) SIXTRL_NOEXCEPT;
            
        SIXTRL_FN void  setTanRotAngle( value_type const& SIXTRL_RESTRICT_REF 
            tan_rot_angle ) SIXTRL_NOEXCEPT;

        SIXTRL_FN void setRotAngleDeg( value_type const& SIXTRL_RESTRICT_REF 
            rot_angle_deg ) SIXTRL_NOEXCEPT;
            
        SIXTRL_FN void setRotAngleRad( value_type const& SIXTRL_RESTRICT_REF 
            rot_angle_rad ) SIXTRL_NOEXCEPT;
        
        SIXTRL_FN void setCosTiltAngle( value_type const& SIXTRL_RESTRICT_REF 
            cos_tilt_angle ) SIXTRL_NOEXCEPT;
            
        SIXTRL_FN void setSinTiltAngle( value_type const& SIXTRL_RESTRICT_REF 
            sin_tilt_angle ) SIXTRL_NOEXCEPT;

        SIXTRL_FN void setTiltAngleDeg( value_type const& SIXTRL_RESTRICT_REF 
            tilt_angle_deg) SIXTRL_NOEXCEPT;
            
        SIXTRL_FN void setTiltAngleRad( value_type const& SIXTRL_RESTRICT_REF 
            tilt_angle_rad ) SIXTRL_NOEXCEPT;
        
        SIXTRL_FN void setB( 
            value_type const& SIXTRL_RESTRICT_REF b ) SIXTRL_NOEXCEPT;

        /* ----------------------------------------------------------------- */
         
        value_type inv_rho        SIXTRL_ALIGN( 8 );
        value_type cos_rot_angle  SIXTRL_ALIGN( 8 );
        value_type tan_rot_angle  SIXTRL_ALIGN( 8 );
        value_type b              SIXTRL_ALIGN( 8 );
        value_type cos_tilt_angle SIXTRL_ALIGN( 8 );
        value_type sin_tilt_angle SIXTRL_ALIGN( 8 );
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
        SIXTRL_BUFFER_ARGPTR_DEC typename TDipoleEdge< T >::c_buffer_t*
            SIXTRL_RESTRICT ptr_buffer );


    template< typename T >
    SIXTRL_ARGPTR_DEC TDipoleEdge< T >* TDipoleEdge_add(
        typename TDipoleEdge< T >::buffer_t& SIXTRL_RESTRICT_REF buffer,
        typename TDipoleEdge< T >::value_type const& 
            SIXTRL_RESTRICT_REF inv_rho,
        typename TDipoleEdge< T >::value_type const& 
            SIXTRL_RESTRICT_REF rot_angle_deg,
        typename TDipoleEdge< T >::value_type const& SIXTRL_RESTRICT_REF b, 
        typename TDipoleEdge< T >::value_type const& 
            SIXTRL_RESTRICT_REF tilt_angle_deg );
    
    template< typename T >
    SIXTRL_ARGPTR_DEC TDipoleEdge< T >* TDipoleEdge_add(
        SIXTRL_BUFFER_ARGPTR_DEC typename TDipoleEdge< T >::c_buffer_t*
            SIXTRL_RESTRICT ptr_buffer,
        typename TDipoleEdge< T >::value_type const& 
            SIXTRL_RESTRICT_REF inv_rho,
        typename TDipoleEdge< T >::value_type const& 
            SIXTRL_RESTRICT_REF rot_angle_deg,
        typename TDipoleEdge< T >::value_type const& SIXTRL_RESTRICT_REF b, 
        typename TDipoleEdge< T >::value_type const& 
            SIXTRL_RESTRICT_REF tilt_angle_deg );

    template< typename T >
    SIXTRL_ARGPTR_DEC TDipoleEdge< T >* TDipoleEdge_add_copy(
        typename SIXTRL_CXX_NAMESPACE::TDipoleEdge< T >::buffer_t& 
            SIXTRL_RESTRICT_REF buffer,
        SIXTRL_CXX_NAMESPACE::TDipoleEdge< T > const& 
            SIXTRL_RESTRICT_REF other );

    template< typename T >
    SIXTRL_ARGPTR_DEC TDipoleEdge< T >* TDipoleEdge_add_copy(
        SIXTRL_BUFFER_ARGPTR_DEC typename SIXTRL_CXX_NAMESPACE::TDipoleEdge< T 
            >::c_buffer_t* SIXTRL_RESTRICT_REF ptr_buffer,
        SIXTRL_CXX_NAMESPACE::TDipoleEdge< T > const& 
            SIXTRL_RESTRICT_REF other );

    /* ===================================================================== *
     * ====  Specialization TDipoleEdge< NS(dipedge_real_t) > :
     * ===================================================================== */

    template<> struct TDipoleEdge< NS(dipedge_real_t) > 
        : public ::NS(DipoleEdge)
    {
        using value_type = ::NS(dipedge_real_t);
        using type_id_t  = ::NS(object_type_id_t);
        using size_type  = ::NS(buffer_size_t);
        using buffer_t   = Buffer;
        using c_buffer_t = buffer_t::c_api_t;
        using c_api_t    = ::NS(DipoleEdge);

        SIXTRL_FN TDipoleEdge() = default;

        SIXTRL_FN TDipoleEdge( 
            TDipoleEdge< value_type > const& other ) = default;
            
        SIXTRL_FN TDipoleEdge( TDipoleEdge< value_type >&& other ) = default;

        SIXTRL_FN TDipoleEdge< value_type >& operator=(
            TDipoleEdge< value_type > const& other ) = default;

        SIXTRL_FN TDipoleEdge< value_type >& operator=(
            TDipoleEdge< value_type >&& other ) = default;s

        SIXTRL_FN ~TDipoleEdge() = default;

        /* ---------------------------------------------------------------- */

        SIXTRL_STATIC SIXTRL_FN bool CanAddToBuffer(
            buffer_t& SIXTRL_RESTRICT_REF buffer,
            SIXTRL_ARGPTR_DEC size_type* SIXTRL_RESTRICT
                ptr_requ_objects  = nullptr,
            SIXTRL_ARGPTR_DEC size_type* SIXTRL_RESTRICT
                ptr_requ_slots    = nullptr,
            SIXTRL_ARGPTR_DEC size_type* SIXTRL_RESTRICT
                ptr_requ_dataptrs = nullptr );

        SIXTRL_STATIC SIXTRL_FN bool CanAddToBuffer(
            SIXTRL_BUFFER_ARGPTR_DEC const c_buffer_t *const
                SIXTRL_RESTRICT ptr_buffer,
            SIXTRL_ARGPTR_DEC size_type* SIXTRL_RESTRICT
                ptr_requ_objects  = nullptr,
            SIXTRL_ARGPTR_DEC size_type* SIXTRL_RESTRICT
                ptr_requ_slots    = nullptr,
            SIXTRL_ARGPTR_DEC size_type* SIXTRL_RESTRICT
                ptr_requ_dataptrs = nullptr );


        SIXTRL_STATIC SIXTRL_FN
        SIXTRL_ARGPTR_DEC TDipoleEdge< ::NS(dipedge_real_t) >*
        CreateNewOnBuffer( buffer_t& SIXTRL_RESTRICT_REF buffer );

        SIXTRL_STATIC SIXTRL_FN
        SIXTRL_ARGPTR_DEC TDipoleEdge< ::NS(dipedge_real_t) >*
        CreateNewOnBuffer( SIXTRL_BUFFER_ARGPTR_DEC *buffer_t*
            SIXTRL_RESTRICT ptr_buffer );


        SIXTRL_STATIC SIXTRL_FN
        SIXTRL_ARGPTR_DEC TDipoleEdge< ::NS(dipedge_real_t) >* AddToBuffer(
            buffer_t& SIXTRL_RESTRICT_REF buffer,
            value_type const inv_rho, value_type const rot_angle_deg, 
            value_type const b, value_type const tilt_angle_deg );

        SIXTRL_STATIC SIXTRL_FN
        SIXTRL_ARGPTR_DEC TDipoleEdge< ::NS(dipedge_real_t) >* AddToBuffer(
            SIXTRL_BUFFER_ARGPTR_DEC c_buffer_t* SIXTRL_RESTRICT ptr_buffer,
            value_type const inv_rho, value_type const rot_angle_deg, 
            value_type const b, value_type const tilt_angle_deg );

        /* ----------------------------------------------------------------- */

        SIXTRL_ARGPTR_DEC c_api_t const* getCApiPtr() const SIXTRL_NOEXCEPT;
        SIXTRL_ARGPTR_DEC c_api_t* getCApiPtr() SIXTRL_NOEXCEPT;

        /* ----------------------------------------------------------------- */

        SIXTRL_STATIC SIXTRL_FN size_type RequiredNumDataPtrs(
            buffer_t const& SIXTRL_RESTRICT_REF buffer ) const SIXTRL_NOEXCEPT;

        SIXTRL_STATIC SIXTRL_FN size_type RequiredNumDataPtrs(
            SIXTRL_BUFFER_ARGPTR_DEC const c_buffer_t *const
                SIXTRL_RESTRICT ptr_buffer ) const SIXTRL_NOEXCEPT;

        SIXTRL_FN type_id_t getTypeId() const SIXTRL_NOEXCEPT;

        SIXTRL_FN value_type getInvRho() const SIXTRL_NOEXCEPT;
        SIXTRL_FN value_type getRho() const SIXTRL_NOEXCEPT;
        
        SIXTRL_FN value_type getCosRotAngle() const SIXTRL_NOEXCEPT;
        SIXTRL_FN value_type getTanRotAngle() const SIXTRL_NOEXCEPT;
        SIXTRL_FN value_type getSinRotAngle() const SIXTRL_NOEXCEPT;

        SIXTRL_FN value_type getRotAngleDeg() const SIXTRL_NOEXCEPT;
        SIXTRL_FN value_type getRotAngleRad() const SIXTRL_NOEXCEPT;
        
        SIXTRL_FN value_type getCosTiltAngle() const SIXTRL_NOEXCEPT;
        SIXTRL_FN value_type getSinTiltAngle() const SIXTRL_NOEXCEPT;

        SIXTRL_FN value_type getTiltAngleDeg() const SIXTRL_NOEXCEPT;
        SIXTRL_FN value_type getTiltAngleRad() const SIXTRL_NOEXCEPT;
        
        SIXTRL_FN value_type getB() const SIXTRL_NOEXCEPT;
        
        /* ----------------------------------------------------------------- */
        
        SIXTRL_FN void setInvRho( value_type const inv_rho ) SIXTRL_NOEXCEPT;
            
        SIXTRL_FN void setCosRotAngle( value_type const  
            cos_rot_angle ) SIXTRL_NOEXCEPT;
            
        SIXTRL_FN void  setTanRotAngle( value_type const  
            tan_rot_angle ) SIXTRL_NOEXCEPT;

        SIXTRL_FN void setRotAngleDeg( value_type const  
            rot_angle_deg ) SIXTRL_NOEXCEPT;
            
        SIXTRL_FN void setRotAngleRad( value_type const  
            rot_angle_rad ) SIXTRL_NOEXCEPT;
        
        SIXTRL_FN void setCosTiltAngle( value_type const  
            cos_tilt_angle ) SIXTRL_NOEXCEPT;
            
        SIXTRL_FN void setSinTiltAngle( value_type const  
            sin_tilt_angle ) SIXTRL_NOEXCEPT;

        SIXTRL_FN void setTiltAngleDeg( value_type const  
            tilt_angle_deg) SIXTRL_NOEXCEPT;
            
        SIXTRL_FN void setTiltAngleRad( value_type const  
            tilt_angle_rad ) SIXTRL_NOEXCEPT;
        
        SIXTRL_FN void setB( value_type const b ) SIXTRL_NOEXCEPT;
    };

    using DipoleEdge = TDipoleEdge< ::NS(dipedge_real_t) >;

    SIXTRL_ARGPTR_DEC DipoleEdge* DipoleEdge_new( SIXTRL_CXX_NAMESPACE::Buffer& buffer );

    SIXTRL_ARGPTR_DEC DipoleEdge* DipoleEdge_new(
        SIXTRL_ARGPTR_DEC ::NS(Buffer)* SIXTRL_RESTRICT ptr_buffer );

    SIXTRL_ARGPTR_DEC DipoleEdge* DipoleEdge_add( SIXTRL_CXX_NAMESPACE::Buffer& buffer,
        ::NS(dipedge_real_t) const inv_rho,
        ::NS(dipedge_real_t) const rot_angle_deg,
        ::NS(dipedge_real_t) const b, 
        ::NS(dipedge_real_t) const tilt_angle_deg );

    SIXTRL_ARGPTR_DEC DipoleEdge* DipoleEdge_add(
        SIXTRL_BUFFER_ARGPTR_DEC ::NS(Buffer)* SIXTRL_RESTRICT ptr_buffer,
        ::NS(dipedge_real_t) const inv_rho,
        ::NS(dipedge_real_t) const rot_angle_deg,
        ::NS(dipedge_real_t) const b, 
        ::NS(dipedge_real_t) const tilt_angle_deg );

    SIXTRL_ARGPTR_DEC DipoleEdge* DipoleEdge_add_copy( 
        SIXTRL_CXX_NAMESPACE::Buffer& buffer,
        SIXTRL_CXX_NAMESPACE::DipoleEdge const& SIXTRL_RESTRICT_REF dipedge );

    SIXTRL_ARGPTR_DEC DipoleEdge* DipoleEdge_add(
        SIXTRL_ARGPTR_DEC ::NS(Buffer)* SIXTRL_RESTRICT ptr_buffer,
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
    bool TDipoleEdge< T >::CanAddToBuffer(
        typename TDipoleEdge< T >::buffer_t& SIXTRL_RESTRICT_REF buffer,
        SIXTRL_ARGPTR_DEC typename TDipoleEdge< T >::size_type*
            SIXTRL_RESTRICT req_objects,
        SIXTRL_ARGPTR_DEC typename TDipoleEdge< T >::size_type*
            SIXTRL_RESTRICT req_slots,
        SIXTRL_ARGPTR_DEC typename TDipoleEdge< T >::size_type*
            SIXTRL_RESTRICT req_dataptrs )
    {
        return TDipoleEdge< T >::CanAddToBuffer(
            buffer.getCApiPtr(), req_objects, req_slots, req_dataptrs );
    }

    template< typename T >
    bool TDipoleEdge< T >::CanAddToBuffer(
        SIXTRL_BUFFER_ARGPTR_DEC typename TDipoleEdge< T >::c_buffer_t*
            SIXTRL_RESTRICT ptr_buffer,
        SIXTRL_ARGPTR_DEC typename TDipoleEdge< T >::size_type*
            SIXTRL_RESTRICT req_objects,
        SIXTRL_ARGPTR_DEC typename TDipoleEdge< T >::size_type*
            SIXTRL_RESTRICT req_slots,
        SIXTRL_ARGPTR_DEC typename TDipoleEdge< T >::size_type*
            SIXTRL_RESTRICT req_dataptrs )
    {
        using _this_t = SIXTRL_CXX_NAMESPACE::TDipoleEdge< T >;
        using  size_t = typename _this_t::size_type;

        static_assert( std::is_trivial< _this_t >::value, "" );
        static_assert( std::is_standard_layout< _this_t >::value, "" );

        size_t const num_dataptrs = _this_t::RequiredNumDataPtrs();
        SIXTRL_ASSERT( num_dataptrs == size_t{ 0 } );

        SIXTRL_ARGPTR_DEC size_t const* sizes  = nullptr;
        SIXTRL_ARGPTR_DEC size_t const* counts = nullptr;

        return ::NS(Buffer_can_add_object)( ptr_buffer, sizeof( _this_t ),
            num_dataptrs, sizes, counts, req_objects, req_slots, req_dataptrs );
    }

    template< typename T >
    SIXTRL_STATIC SIXTRL_FN SIXTRL_ARGPTR_DEC TDipoleEdge< T >*
    TDipoleEdge< T >::CreateNewOnBuffer(
        typename TDipoleEdge< T >::buffer_t& SIXTRL_RESTRICT_REF buffer )
    {
        return TDipoleEdge< T >::CreateNewOnBuffer( buffer.getCApiPtr() );
    }

    template< typename T >
    SIXTRL_STATIC SIXTRL_FN SIXTRL_ARGPTR_DEC TDipoleEdge< T >*
    TDipoleEdge< T >::CreateNewOnBuffer(
        SIXTRL_BUFFER_ARGPTR_DEC typename TDipoleEdge< T >::c_buffer_t*
            SIXTRL_RESTRICT buffer )
    {
        using _this_t = SIXTRL_CXX_NAMESPACE::TDipoleEdge< T >;
        using size_t  = typename _this_t::size_type;
        using value_t = typename _this_t::value_type;
        using ptr_t   = SIXTRL_ARGPTR_DEC _this_t*;

        static_assert( std::is_trivial< _this_t >::value, "" );
        static_assert( std::is_standard_layout< _this_t >::value, "" );

        size_t const num_dataptrs = _this_t::RequiredNumDataPtrs();
        SIXTRL_ASSERT( num_dataptrs == size_t{ 0 } );

        SIXTRL_ARGPTR_DEC size_t const* offsets = nullptr;
        SIXTRL_ARGPTR_DEC size_t const* sizes   = nullptr;
        SIXTRL_ARGPTR_DEC size_t const* counts  = nullptr;

        _this_t temp;
        temp.preset();

        return reinterpret_cast< ptr_t >( ::NS(Object_get_begin_addr)(
            ::NS(Buffer_add_object)( ptr_buffer, &temp, sizeof( _this_t ),
                temp.getTypeId(), num_dataptrs, offsets, sizes, counts ) ) );
    }

    template< typename T >
    SIXTRL_STATIC SIXTRL_ARGPTR_DEC TDipoleEdge< T >*
    TDipoleEdge< T >::AddToBuffer(
        typename TDipoleEdge< T >::buffer_t& SIXTRL_RESTRICT_REF buffer,
        typename TDipoleEdge< T >::value_type const& 
            SIXTRL_RESTRICT_REF inv_rho,
        typename TDipoleEdge< T >::value_type const& SIXTRL_RESTRICT_REF 
            SIXTRL_RESTRICT_REF rot_angle_deg,
        typename TDipoleEdge< T >::value_type const& SIXTRL_RESTRICT_REF 
            SIXTRL_RESTRICT_REF b,
        typename TDipoleEdge< T >::value_type const& SIXTRL_RESTRICT_REF 
            SIXTRL_RESTRICT_REF tilt_angle_deg )
    {
        return SIXTRL_CXX_NAMESPACE::TDipoleEdge< T>::AddToBuffer(
            buffer.getCApiPtr(), inv_rho, rot_angle_deg, b, tilt_angle_deg );
    }

    template< typename T >
    SIXTRL_STATIC SIXTRL_ARGPTR_DEC TDipoleEdge< T >*
    TDipoleEdge< T >::AddToBuffer(
        SIXTRL_BUFFER_ARGPTR_DEC typename TDipoleEdge< T >::c_buffer_t*
            SIXTRL_RESTRICT ptr_buffer,
        typename TDipoleEdge< T >::value_type const& 
            SIXTRL_RESTRICT_REF inv_rho,
        typename TDipoleEdge< T >::value_type const& SIXTRL_RESTRICT_REF 
            SIXTRL_RESTRICT_REF rot_angle_deg,
        typename TDipoleEdge< T >::value_type const& SIXTRL_RESTRICT_REF 
            SIXTRL_RESTRICT_REF b,
        typename TDipoleEdge< T >::value_type const& SIXTRL_RESTRICT_REF 
            SIXTRL_RESTRICT_REF tilt_angle_deg )
    {
        using _this_t = SIXTRL_CXX_NAMESPACE::TDipoleEdge< T >;
        using size_t  = typename _this_t::size_type;
        using value_t = typename _this_t::value_type;
        using ptr_t   = SIXTRL_ARGPTR_DEC _this_t*;

        static_assert( std::is_trivial< _this_t >::value, "" );
        static_assert( std::is_standard_layout< _this_t >::value, "" );

        size_t const num_dataptrs = _this_t::RequiredNumDataPtrs();
        SIXTRL_ASSERT( num_dataptrs == size_t{ 0 } );

        SIXTRL_ARGPTR_DEC size_t const* offsets = nullptr;
        SIXTRL_ARGPTR_DEC size_t const* sizes   = nullptr;
        SIXTRL_ARGPTR_DEC size_t const* counts  = nullptr;

        _this_t temp;
        temp.setInvRho( inv_rho );
        temp.setRotAngleDeg( rot_angle_deg );
        temp.setB( b );
        temp.setTiltAngleDeg( tilt_angle_deg );

        return reinterpret_cast< ptr_t >( ::NS(Object_get_begin_addr)(
            ::NS(Buffer_add_object)( ptr_buffer, &temp, sizeof( _this_t ),
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
    SIXTRL_STATIC typename TDipoleEdge< T >::size_type
    TDipoleEdge< T >::RequiredNumDataPtrs( 
        typename TDipoleEdge< T >::buffer_t const& 
            SIXTRL_RESTRICT_REF buffer ) const SIXTRL_NOEXCEPT
    {
        ::NS(DipoleEdge_get_required_num_dataptrs)( 
            buffer.getCApiPtr(), nullptr );
    }

    template< typename T > 
    typename TDipoleEdge< T >::value_type const& 
    TDipoleEdge< T >::getInvRho() const SIXTRL_NOEXCEPT
    {
        return this->inv_rho;
    }
    
    template< typename T > 
    typename TDipoleEdge< T >::value_type 
    TDipoleEdge< T >::getRho() const SIXTRL_NOEXCEPT
    {
        typedef typename TDipoleEdge< T >::value_type real_t;
        
        return ( this->inv_rho > real_t{ 0 } )
            ? static_cast< real_t >( 1 ) / this->inv_rho 
            : static_cast< real_t >( 0 );
    }
    
    template< typename T > 
    typename TDipoleEdge< T >::value_type const& 
    TDipoleEdge< T >::getCosRotAngle() const SIXTRL_NOEXCEPT
    {
        return this->cos_rot_angle;
    }
    
    template< typename T > 
    typename TDipoleEdge< T >::value_type const& 
    TDipoleEdge< T >::getTanRotAngle() const SIXTRL_NOEXCEPT
    {
        return this->tan_rot_angle;
    }
    
    template< typename T > 
    typename TDipoleEdge< T >::value_type 
    TDipoleEdge< T >::getSinRotAngle() const SIXTRL_NOEXCEPT
    {
        return this->getCosRotAngle() * this->getTanRotAngle();
    }

    template< typename T > 
    typename TDipoleEdge< T >::value_type 
    TDipoleEdge< T >::getRotAngleDeg() const SIXTRL_NOEXCEPT
    {
        return SIXTRL_CXX_NAMESPACE::RAD2DEG * this->getRotAngleRad();
    }
    
    template< typename T > 
    typename TDipoleEdge< T >::value_type 
    TDipoleEdge< T >::getRotAngleRad() const SIXTRL_NOEXCEPT
    {
        return std::atan2( this->getCosRotAngle(), this->getSinRotAngle() );
    }
    
    template< typename T > 
    typename TDipoleEdge< T >::value_type const& 
    TDipoleEdge< T >::getCosTiltAngle() const SIXTRL_NOEXCEPT
    {
        return this->cos_tilt_angle;
    }
    
    template< typename T > 
    typename TDipoleEdge< T >::value_type const& 
    TDipoleEdge< T >::getSinTiltAngle() const SIXTRL_NOEXCEPT
    {
        return this->sin_tilt_angle;
    }

    template< typename T > 
    typename TDipoleEdge< T >::value_type 
    TDipoleEdge< T >::getTiltAngleDeg() const SIXTRL_NOEXCEPT
    {
        return SIXTRL_CXX_NAMESPACE::RAD2DEG * this->getTiltAngleRad();
    }
    
    template< typename T > 
    typename TDipoleEdge< T >::value_type 
    TDipoleEdge< T >::getTiltAngleRad() const SIXTRL_NOEXCEPT
    {
        return std::atan2( this->cos_tilt_angle, this->sin_tilt_angle );
    }
    
    template< typename T > 
    typename TDipoleEdge< T >::value_type const&
    TDipoleEdge< T >::getB() const SIXTRL_NOEXCEPT
    {
        return this->b;
    }
    
    /* ----------------------------------------------------------------- */
    
    template< typename T > void TDipoleEdge< T >::setInvRho( 
        typename TDipoleEdge< T >::value_type const& 
            SIXTRL_RESTRICT_REF inv_rho ) SIXTRL_NOEXCEPT
    {
        typedef typename TDipoleEdge< T >::value_type real_t;
        
        if( inv_rho > real_t{ 0 } )
        {
            this->inv_rho = inv_rho;
        }
    }
        
    template< typename T > void TDipoleEdge< T >::setCosRotAngle( 
        typename TDipoleEdge< T >::value_type const& SIXTRL_RESTRICT_REF 
            cos_rot_angle ) SIXTRL_NOEXCEPT
    {
        typedef typename TDipoleEdge< T >::value_type real_t;
        
        if( ( cos_rot_angle >= real_t{ 0 } ) &&
            ( cos_rot_angle <= real_t{ 1 } ) )
        {
            this->cos_rot_angle = cos_rot_angle;
        }
    }
        
    template< typename T > void TDipoleEdge< T >::setTanRotAngle( 
        typename TDipoleEdge< T >::value_type const& SIXTRL_RESTRICT_REF 
            tan_rot_angle ) SIXTRL_NOEXCEPT
    {
        typedef typename TDipoleEdge< T >::value_type real_t;
        
        if( ( tan_rot_angle >= real_t{ 0 } ) &&
            ( tan_rot_angle <= real_t{ 1 } ) )
        {
            this->tan_rot_angle = tan_rot_angle;
        }
    }

    template< typename T > void TDipoleEdge< T >::setRotAngleDeg( 
        typename TDipoleEdge< T >::value_type const& SIXTRL_RESTRICT_REF 
            rot_angle_deg ) SIXTRL_NOEXCEPT
    {
        this->setRotAngleRad( SIXTRL_CXX_NAMESPACE::DEG2RAD * rot_angle_deg );
    }
        
    template< typename T > void TDipoleEdge< T >::setRotAngleRad( 
        typename TDipoleEdge< T >::value_type const& SIXTRL_RESTRICT_REF 
        rot_angle_rad ) SIXTRL_NOEXCEPT
    {
        this->cos_rot_angle = std::cos( rot_angle_rad );
        this->tan_rot_angle = std::cos( rot_angle_rad );
    }
    
    template< typename T > void TDipoleEdge< T >::setCosTiltAngle( 
        typename TDipoleEdge< T >::value_type const& SIXTRL_RESTRICT_REF 
        cos_tilt_angle ) SIXTRL_NOEXCEPT
    {
        typedef typename TDipoleEdge< T >::value_type real_t;
        
        if( ( cos_tilt_angle >= real_t{ 0 } ) &&
            ( cos_tilt_angle <= real_t{ 1 } ) )
        {
            this->cos_tilt_angle = cos_tilt_angle;
        }
    }
        
    template< typename T > void TDipoleEdge< T >::setSinTiltAngle( 
        typename TDipoleEdge< T >::value_type const& SIXTRL_RESTRICT_REF 
        sin_tilt_angle ) SIXTRL_NOEXCEPT
    {
        typedef typename TDipoleEdge< T >::value_type real_t;
        
        if( ( sin_tilt_angle >= real_t{ 0 } ) &&
            ( sin_tilt_angle <= real_t{ 1 } ) )
        {
            this->sin_tilt_angle = sin_tilt_angle;
        }
    }

    template< typename T > void TDipoleEdge< T >::setTiltAngleDeg( 
        typename TDipoleEdge< T >::value_type const& SIXTRL_RESTRICT_REF 
        tilt_angle_deg) SIXTRL_NOEXCEPT
    {
        this->setTiltAngleRad( 
            SIXTRL_CXX_NAMESPACE::DEG2RAD * tilt_angle_deg );
    }
        
    template< typename T > void TDipoleEdge< T >::setTiltAngleRad( 
        typename TDipoleEdge< T >::value_type const& SIXTRL_RESTRICT_REF 
        tilt_angle_rad ) SIXTRL_NOEXCEPT
    {
        this->cos_tilt_angle = std::cos( tilt_angle_rad );
        this->sin_tilt_angle = std::sin( tilt_angle_rad );
    }
    
    template< typename T > void TDipoleEdge< T >::setB( 
        typename TDipoleEdge< T >::value_type const& 
            SIXTRL_RESTRICT_REF b ) SIXTRL_NOEXCEPT
    {
        this->b = b;
    }

    /* ----------------------------------------------------------------- */

    template< typename T >
    SIXTRL_ARGPTR_DEC TDipoleEdge< T >* TDipoleEdge_new(
        typename TDipoleEdge< T >::buffer_t& buffer )
    {
        return SIXTRL_CXX_NAMESPACE::TDipoleEdge< T >::CreateNewOnBuffer(
            buffer.getCApiPtr() );
    }

    template< typename T >
    SIXTRL_ARGPTR_DEC TDipoleEdge< T >* TDipoleEdge_new(
        SIXTRL_BUFFER_ARGPTR_DEC typename TDipoleEdge< T >::c_buffer_t* ptr_buffer )
    {
        return SIXTRL_CXX_NAMESPACE::TDipoleEdge< T >::CreateNewOnBuffer(
            ptr_buffer );
    }

    template< typename T >
    SIXTRL_ARGPTR_DEC TDipoleEdge< T >* TDipoleEdge_add(
        typename TDipoleEdge< T >::buffer_t& SIXTRL_RESTRICT_REF buffer,
        typename TDipoleEdge< T >::value_type const& SIXTRL_RESTRICT_REF x_limits,
        typename TDipoleEdge< T >::value_type const& SIXTRL_RESTRICT_REF y_limits )
    {
        return SIXTRL_CXX_NAMESPACE::TDipoleEdge< T >::AddToBuffer(
            buffer.getCApiPtr(), x_limits, y_limits );
    }

    template< typename T >
    SIXTRL_ARGPTR_DEC TDipoleEdge< T >* TDipoleEdge_add(
        SIXTRL_BUFFER_ARGPTR_DEC typename TDipoleEdge< T >::c_buffer_t*
            SIXTRL_RESTRICT ptr_buffer,
        typename TDipoleEdge< T >::value_type const& SIXTRL_RESTRICT_REF x_limits,
        typename TDipoleEdge< T >::value_type const& SIXTRL_RESTRICT_REF y_limits )
    {
        return SIXTRL_CXX_NAMESPACE::TDipoleEdge< T >::AddToBuffer(
            ptr_buffer, x_limits, y_limits );
    }

    template< typename T >
    SIXTRL_ARGPTR_DEC TDipoleEdge< T >* TDipoleEdge_add_copy(
        typename TDipoleEdge< T >::buffer_t& SIXTRL_RESTRICT_REF buffer,
        TDipoleEdge< T > const& SIXTRL_RESTRICT_REF  other )
    {
        return SIXTRL_CXX_NAMESPACE::TDipoleEdge< T >::AddToBuffer(
            buffer.getCApiPtr(), other.getXDipoleEdge(), other.getYDipoleEdge() );
    }

    template< typename T >
    SIXTRL_ARGPTR_DEC TDipoleEdge< T >* TDipoleEdge_add_copy(
        SIXTRL_BUFFER_ARGPTR_DEC typename TDipoleEdge< T >::c_buffer_t*
            SIXTRL_RESTRICT ptr_buffer,
        TDipoleEdge< T > const& SIXTRL_RESTRICT_REF  other )
    {
        return SIXTRL_CXX_NAMESPACE::TDipoleEdge< T >::AddToBuffer(
            ptr_buffer, other.getXDipoleEdge(), other.getYDipoleEdge() );
    }

    /* ===================================================================== *
     * ====  Specialization TDipoleEdge< ::NS(dipedge_real_t) > :
     * ===================================================================== */

    template<>
    bool TDipoleEdge< ::NS(dipedge_real_t) >::CanAddToBuffer(
        TDipoleEdge< ::NS(dipedge_real_t) >::buffer_t& SIXTRL_RESTRICT_REF buffer,
        SIXTRL_ARGPTR_DEC TDipoleEdge< ::NS(dipedge_real_t) >::size_type*
            SIXTRL_RESTRICT ptr_requ_objects,
        SIXTRL_ARGPTR_DEC TDipoleEdge< ::NS(dipedge_real_t) >::size_type*
            SIXTRL_RESTRICT ptr_requ_slots,
        SIXTRL_ARGPTR_DEC TDipoleEdge< ::NS(dipedge_real_t) >::size_type*
            SIXTRL_RESTRICT ptr_requ_dataptrs )
    {
        return ::NS(DipoleEdge_can_be_added)( buffer.getCApiPtr(), ptr_requ_objects,
            ptr_requ_slots, ptr_requ_dataptrs );
    }


    template<>
    bool TDipoleEdge< ::NS(dipedge_real_t) >::CanAddToBuffer(
        SIXTRL_BUFFER_ARGPTR_DEC TDipoleEdge< ::NS(dipedge_real_t) >::c_buffer_t*
            SIXTRL_RESTRICT ptr_buffer,
        SIXTRL_ARGPTR_DEC TDipoleEdge< ::NS(dipedge_real_t) >::size_type*
            SIXTRL_RESTRICT ptr_requ_objects,
        SIXTRL_ARGPTR_DEC TDipoleEdge< ::NS(dipedge_real_t) >::size_type*
            SIXTRL_RESTRICT ptr_requ_slots,
        SIXTRL_ARGPTR_DEC TDipoleEdge< ::NS(dipedge_real_t) >::size_type*
            SIXTRL_RESTRICT ptr_requ_dataptrs )
    {
        return ::NS(DipoleEdge_can_be_added)( buffer.getCApiPtr(), ptr_requ_objects,
            ptr_requ_slots, ptr_requ_dataptrs );
    }

    template<>
    SIXTRL_STATIC SIXTRL_ARGPTR_DEC TDipoleEdge< NS(dipedge_real_t) >*
    TDipoleEdge< ::NS(dipedge_real_t) >::CreateNewOnBuffer(
        TDipoleEdge< ::NS(dipedge_real_t) >::buffer_t& SIXTRL_RESTRICT_REF buffer )
    {
        return ::NS(DipoleEdge_new)( buffer.getCApiPtr() );
    }

    template<>
    SIXTRL_STATIC SIXTRL_ARGPTR_DEC TDipoleEdge< NS(dipedge_real_t) >*
    TDipoleEdge< ::NS(dipedge_real_t) >::CreateNewOnBuffer(
        SIXTRL_BUFFER_ARGPTR_DEC TDipoleEdge< ::NS(dipedge_real_t) >::c_buffer_t*
            SIXTRL_RESTRICT ptr_buffer )
    {
        return ::NS(DipoleEdge_new)( ptr_buffer );
    }

    template<>
    SIXTRL_ARGPTR_DEC TDipoleEdge< NS(dipedge_real_t) >*
    TDipoleEdge< ::NS(dipedge_real_t) >::AddToBuffer(
        TDipoleEdge< ::NS(dipedge_real_t) >::buffer_t& SIXTRL_RESTRICT_REF buffer,
        TDipoleEdge< ::NS(dipedge_real_t) >::value_type const inv_rho, 
        TDipoleEdge< ::NS(dipedge_real_t) >::value_type const rot_angle_deg, 
        TDipoleEdge< ::NS(dipedge_real_t) >::value_type const b, 
        TDipoleEdge< ::NS(dipedge_real_t) >::value_type const tilt_angle_deg  )
    {
        return ::NS(DipoleEdge_add)( 
            buffer.getCApiPtr(), inv_rho, rot_angle_deg, b, tilt_angle_deg );
    }

    template<>
    SIXTRL_ARGPTR_DEC TDipoleEdge< NS(dipedge_real_t) >*
    TDipoleEdge< ::NS(dipedge_real_t) >::AddToBuffer(
        SIXTRL_BUFFER_ARGPTR_DEC TDipoleEdge< ::NS(dipedge_real_t) >::c_buffer_t*
            SIXTRL_RESTRICT ptr_buffer,
        TDipoleEdge< ::NS(dipedge_real_t) >::value_type const inv_rho, 
        TDipoleEdge< ::NS(dipedge_real_t) >::value_type const rot_angle_deg, 
        TDipoleEdge< ::NS(dipedge_real_t) >::value_type const b, 
        TDipoleEdge< ::NS(dipedge_real_t) >::value_type const tilt_angle_deg  )
    {
        return ::NS(DipoleEdge_add)( 
            ptr_buffer, inv_rho, rot_angle_deg, b, tilt_angle_deg );
    }

    /* ----------------------------------------------------------------- */

    template<>
    SIXTRL_ARGPTR_DEC TDipoleEdge< ::NS(dipedge_real_t) >::c_api_t const*
    TDipoleEdge< ::NS(dipedge_real_t) >::getCApiPtr() const SIXTRL_NOEXCEPT
    {
        using ptr_t = TXYShift< ::NS(dipedge_real_t) >::c_api_t const*;
        return reinterpret_cast< ptr_t >( this );
    }

    template<>
    SIXTRL_ARGPTR_DEC TDipoleEdge< ::NS(dipedge_real_t) >::c_api_t*
    TDipoleEdge< ::NS(dipedge_real_t) >::getCApiPtr() SIXTRL_NOEXCEPT
    {
        return const_cast< TDipoleEdge< ::NS(dipedge_real_t) >::c_api_t* >(
            static_cast< TDipoleEdge< ::NS(dipedge_real_t) > const& >( *this
                ).getCApiPtr() );
    }

    /* ----------------------------------------------------------------- */

    template<>
    TDipoleEdge< ::NS(dipedge_real_t) >::type_id_t
    TDipoleEdge< ::NS(dipedge_real_t) >::getTypeId() const SIXTRL_NOEXCEPT
    {
        return SIXTRL_CXX_NAMESPACE::OBJECT_TYPE_DIPEDGE;
    }

    template<>
    TDipoleEdge< ::NS(dipedge_real_t) >::size_type
    TDipoleEdge< ::NS(dipedge_real_t) >::RequiredNumDataPtrs(
        TDipoleEdge< ::NS(dipedge_real_t) >::buffer_t const&
            SIXTRL_RESTRICT_REF buffer ) const SIXTRL_NOEXCEPT
    {
        return ::NS(DipoleEdge_get_required_num_dataptrs)( 
            buffer.getCApiPtr(), nullptr );
    }

    template<>
    TDipoleEdge< ::NS(dipedge_real_t) >::size_type
    TDipoleEdge< ::NS(dipedge_real_t) >::RequiredNumDataPtrs(
        SIXTRL_BUFFER_ARGPTR_DEC const
            TDipoleEdge< ::NS(dipedge_real_t) >::c_buffer_t *const
                SIXTRL_RESTRICT ptr_buffer ) const SIXTRL_NOEXCEPT
    {
        return ::NS(DipoleEdge_get_required_num_dataptrs)( 
            ptr_buffer, nullptr );
    }

    template<> TDipoleEdge< ::NS(dipedge_real_t) >::value_type 
    TDipoleEdge< ::NS(dipedge_real_t) >::getInvRho() const SIXTRL_NOEXCEPT
    {
        return ::NS(DipoleEdge_get_inv_rho)( this->getCApiPtr() );
    }
    
    template<> TDipoleEdge< ::NS(dipedge_real_t) >::value_type 
    TDipoleEdge< ::NS(dipedge_real_t) >::getRho() const SIXTRL_NOEXCEPT
    {
        return ::NS(DipoleEdge_get_rho)( this->getCApiPtr() );
    }
    
    template<> TDipoleEdge< ::NS(dipedge_real_t) >::value_type 
    TDipoleEdge< ::NS(dipedge_real_t) >::getCosRotAngle() const SIXTRL_NOEXCEPT
    {
        return ::NS(DipoleEdge_get_cos_rot_angle)( this->getCApiPtr() );
    }
    
    template<> TDipoleEdge< ::NS(dipedge_real_t) >::value_type 
    TDipoleEdge< ::NS(dipedge_real_t) >::getTanRotAngle() const SIXTRL_NOEXCEPT
    {
        return ::NS(DipoleEdge_get_tan_rot_angle)( this->getCApiPtr() );
    }
    
    template<> TDipoleEdge< ::NS(dipedge_real_t) >::value_type 
    TDipoleEdge< ::NS(dipedge_real_t) >::getSinRotAngle() const SIXTRL_NOEXCEPT
    {
        return ::NS(DipoleEdge_get_sin_rot_angle)( this->getCApiPtr() );
    }

    template<> TDipoleEdge< ::NS(dipedge_real_t) >::value_type 
    TDipoleEdge< ::NS(dipedge_real_t) >::getRotAngleDeg() const SIXTRL_NOEXCEPT
    {
        return ::NS(DipoleEdge_get_rot_angle)( this->getCApiPtr() );
    }
    
    template<> TDipoleEdge< ::NS(dipedge_real_t) >::value_type 
    TDipoleEdge< ::NS(dipedge_real_t) >::getRotAngleRad() const SIXTRL_NOEXCEPT
    {
        return ::NS(DipoleEdge_get_rot_angle_rad)( this->getCApiPtr() );
    }
    
    template<> TDipoleEdge< ::NS(dipedge_real_t) >::value_type 
    TDipoleEdge< ::NS(dipedge_real_t) >::getCosTiltAngle() const SIXTRL_NOEXCEPT
    {
        return ::NS(DipoleEdge_get_cos_tilt_angle)( this->getCApiPtr() );
    }
    
    template<> TDipoleEdge< ::NS(dipedge_real_t) >::value_type 
    TDipoleEdge< ::NS(dipedge_real_t) >::getSinTiltAngle() const SIXTRL_NOEXCEPT
    {
        return ::NS(DipoleEdge_get_sin_tilt_angle)( this->getCApiPtr() );
    }

    template<> TDipoleEdge< ::NS(dipedge_real_t) >::value_type 
    TDipoleEdge< ::NS(dipedge_real_t) >::getTiltAngleDeg() const SIXTRL_NOEXCEPT
    {
        return ::NS(DipoleEdge_get_tilt_angle)( this->getCApiPtr() );
    }
    
    template<> TDipoleEdge< ::NS(dipedge_real_t) >::value_type 
    TDipoleEdge< ::NS(dipedge_real_t) >::getTiltAngleRad() const SIXTRL_NOEXCEPT
    {
        return ::NS(DipoleEdge_get_tilt_angle_rad)( this->getCApiPtr() );
    }
    
    template<> TDipoleEdge< ::NS(dipedge_real_t) >::value_type 
    TDipoleEdge< ::NS(dipedge_real_t) >::getB() const SIXTRL_NOEXCEPT
    {
        return ::NS(DipoleEdge_get_b)( this->getCApiPtr() );
    }
    
    /* ----------------------------------------------------------------- */
    
    template<> void TDipoleEdge< ::NS(dipedge_real_t) >::setInvRho( 
        TDipoleEdge< ::NS(dipedge_real_t) >::value_type const 
            inv_rho ) SIXTRL_NOEXCEPT
    {
        ::NS(DipoleEdge_set_inv_rho)( this->getCApiPtr(), inv_rho );
    }
        
    template<> void TDipoleEdge< ::NS(dipedge_real_t) >::setCosRotAngle( 
        TDipoleEdge< ::NS(dipedge_real_t) >::value_type const  
            cos_rot_angle ) SIXTRL_NOEXCEPT
    {
        ::NS(DipoleEdge_set_cos_rot_angle)( this->getCApiPtr(), cos_rot_angle );
    }
        
    template<> void TDipoleEdge< ::NS(dipedge_real_t) >::setTanRotAngle( 
        TDipoleEdge< ::NS(dipedge_real_t) >::value_type const 
            tan_rot_angle ) SIXTRL_NOEXCEPT
    {
        ::NS(DipoleEdge_set_tan_rot_angle)( this->getCApiPtr(), tan_rot_angle );
    }

    template<> void TDipoleEdge< ::NS(dipedge_real_t) >::setRotAngleDeg( 
        TDipoleEdge< ::NS(dipedge_real_t) >::value_type const  
            rot_angle_deg ) SIXTRL_NOEXCEPT
    {
        ::NS(DipoleEdge_set_rot_angle)( this->getCApiPtr(), rot_angle_deg );
    }
        
    template<> void TDipoleEdge< ::NS(dipedge_real_t) >::setRotAngleRad( 
        TDipoleEdge< ::NS(dipedge_real_t) >::value_type const 
            rot_angle_rad ) SIXTRL_NOEXCEPT
    {
        ::NS(DipoleEdge_set_rot_angle_rad)( this->getCApiPtr(), rot_angle_rad );
    }
    
    template<> void TDipoleEdge< ::NS(dipedge_real_t) >::setCosTiltAngle( 
        TDipoleEdge< ::NS(dipedge_real_t) >::value_type const  
            cos_tilt_angle ) SIXTRL_NOEXCEPT
    {
        ::NS(DipoleEdge_set_cos_tilt_angle)( 
            this->getCApiPtr(), cos_tilt_angle );
    }
        
    template<> void TDipoleEdge< ::NS(dipedge_real_t) >::setSinTiltAngle( 
        TDipoleEdge< ::NS(dipedge_real_t) >::value_type const  
            sin_tilt_angle ) SIXTRL_NOEXCEPT
    {
        ::NS(DipoleEdge_set_sin_tilt_angle)( 
            this->getCApiPtr(), sin_tilt_angle );
    }

    template<> void TDipoleEdge< ::NS(dipedge_real_t) >::setTiltAngleDeg( 
        TDipoleEdge< ::NS(dipedge_real_t) >::value_type const  tilt_angle_deg
        ) SIXTRL_NOEXCEPT
    {
        ::NS(DipoleEdge_set_tilt_angle)( this->getCApiPtr(), tilt_angle_deg );
    }
        
    template<> void TDipoleEdge< ::NS(dipedge_real_t) >::setTiltAngleRad( 
        TDipoleEdge< ::NS(dipedge_real_t) >::value_type const  
            tilt_angle_rad ) SIXTRL_NOEXCEPT
    {
        ::NS(DipoleEdge_set_tilt_angle_rad)( 
            this->getCApiPtr(), tilt_angle_deg );
    }
    
    template<> void TDipoleEdge< ::NS(dipedge_real_t) >::setB( TDipoleEdge< 
        ::NS(dipedge_real_t) >::value_type const b ) SIXTRL_NOEXCEPT
    {
        ::NS(DipoleEdge_set_b)( this->getCApiPtr(), b );
    }
    
    /* --------------------------------------------------------------------- */
    
    SIXTRL_ARGPTR_DEC SIXTRL_CXX_NAMESPACE::DipoleEdge* DipoleEdge_new(
        SIXTRL_CXX_NAMESPACE::Buffer& SIXTRL_RESTRICT_REF buffer )
    {
        return static_cast< SIXTRL_ARGPTR_DEC DipoleEdge* >(
            ::NS(DipoleEdge_new)( buffer.getCApiPtr() ) );
    }

    SIXTRL_ARGPTR_DEC SIXTRL_CXX_NAMESPACE::DipoleEdge* DipoleEdge_new(
        SIXTRL_ARGPTR_DEC ::NS(Buffer)* SIXTRL_RESTRICT ptr_buffer )
    {
        return static_cast< SIXTRL_ARGPTR_DEC DipoleEdge* >(
            ::NS(DipoleEdge_new)( ptr_buffer ) );
    }

    SIXTRL_ARGPTR_DEC SIXTRL_CXX_NAMESPACE::DipoleEdge* DipoleEdge_add(
        SIXTRL_CXX_NAMESPACE::Buffer& SIXTRL_RESTRICT_REF buffer,
        ::NS(dipedge_real_t) const inv_rho,
        ::NS(dipedge_real_t) const rot_angle_deg,
        ::NS(dipedge_real_t) const b, 
        ::NS(dipedge_real_t) const tilt_angle_deg )
    {
        return static_cast< SIXTRL_ARGPTR_DEC DipoleEdge* >(
            ::NS(DipoleEdge_add)( buffer.getCApiPtr(), 
                  inv_rho, rot_angle_deg, b, tilt_angle_deg ) );
    }

    SIXTRL_ARGPTR_DEC DipoleEdge* DipoleEdge_add(
        SIXTRL_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT ptr_buffer,
        ::NS(dipedge_real_t) const inv_rho,
        ::NS(dipedge_real_t) const rot_angle_deg,
        ::NS(dipedge_real_t) const b, 
        ::NS(dipedge_real_t) const tilt_angle_deg )
    {
        return static_cast< SIXTRL_ARGPTR_DEC DipoleEdge* >(
            ::NS(DipoleEdge_add)( ptr_buffer, 
              inv_rho, rot_angle_deg, b, tilt_angle_deg ) );
    }

    SIXTRL_ARGPTR_DEC SIXTRL_CXX_NAMESPACE::DipoleEdge* DipoleEdge_add_copy(
        SIXTRL_CXX_NAMESPACE::Buffer& SIXTRL_RESTRICT_REF buffer,
        SIXTRL_CXX_NAMESPACE::DipoleEdge const& SIXTRL_RESTRICT_REF other )
    {
        return static_cast< SIXTRL_ARGPTR_DEC DipoleEdge* >(
            ::NS(DipoleEdge_add_copy)( buffer.getCApiPtr(), other.getCApiPtr() );
    }

    SIXTRL_ARGPTR_DEC SIXTRL_CXX_NAMESPACE::DipoleEdge* DipoleEdge_add_copy(
        SIXTRL_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT ptr_buffer,
        SIXTRL_CXX_NAMESPACE::DipoleEdge const& SIXTRL_RESTRICT_REF other )
    {
        return static_cast< SIXTRL_ARGPTR_DEC SIXTRL_CXX_NAMESPACE::DipoleEdge* >(
            ::NS(DipoleEdge_add_copy)( ptr_buffer, other.getCApiPtr() );
    }
}

#endif /* SIXTRACKLIB_COMMON_BE_DIPEDGE_BE_DIPEDGE_CXX_HPP__ */

/* end: sixtracklib/common/be_limit/be_limit.hpp */
