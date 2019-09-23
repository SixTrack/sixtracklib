#ifndef CXX_SIXTRACKLIB_COMMON_BE_SROTATION_BEAM_ELEM_BE_SROTATION_HPP__
#define CXX_SIXTRACKLIB_COMMON_BE_SROTATION_BEAM_ELEM_BE_SROTATION_HPP__

#if defined( __cplusplus )

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <cstddef>
    #include <cstdint>
    #include <cstdlib>
    #include <limits>
    #include <type_traits>
    #include <utility>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/buffer.hpp"
    #include "sixtracklib/common/be_srotation/be_srotation.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */


namespace SIXTRL_CXX_NAMESPACE
{
    /* ===================================================================== *
     * ====  TSRotation< T > :
     * ===================================================================== */

    template< typename T >
    struct TSRotation
    {
        using value_type = T;
        using type_id_t  = NS(object_type_id_t);
        using size_type  = NS(buffer_size_t);
        using buffer_t   = ::NS(Buffer);

        SIXTRL_FN TSRotation() = default;
        SIXTRL_FN TSRotation( TSRotation< T > const& other ) = default;
        SIXTRL_FN TSRotation( TSRotation< T >&& other ) = default;

        SIXTRL_FN TSRotation< T >& operator=(
            TSRotation< T > const& rhs ) = default;

        SIXTRL_FN TSRotation< T >& operator=(
            TSRotation< T >&& rhs ) = default;

        SIXTRL_FN ~TSRotation() = default;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        SIXTRL_FN SIXTRL_STATIC bool CanAddToBuffer(
            buffer_t& SIXTRL_RESTRICT_REF buffer,
            SIXTRL_ARGPTR_DEC size_type*
                SIXTRL_RESTRICT ptr_requ_objects  = nullptr,
            SIXTRL_ARGPTR_DEC size_type*
                SIXTRL_RESTRICT ptr_requ_slots    = nullptr,
            SIXTRL_ARGPTR_DEC size_type*
                SIXTRL_RESTRICT ptr_requ_dataptrs = nullptr ) SIXTRL_NOEXCEPT;

        SIXTRL_FN SIXTRL_STATIC SIXTRL_ARGPTR_DEC TSRotation< T >*
        CreateNewOnBuffer( buffer_t& SIXTRL_RESTRICT_REF buffer );

        SIXTRL_FN SIXTRL_STATIC SIXTRL_ARGPTR_DEC TSRotation< T >* AddToBuffer(
            buffer_t& SIXTRL_RESTRICT_REF buffer, value_type const& angle );

        SIXTRL_FN SIXTRL_STATIC SIXTRL_ARGPTR_DEC TSRotation< T >* AddToBuffer(
            buffer_t& SIXTRL_RESTRICT_REF buffer,
            value_type const& SIXTRL_RESTRICT_REF cos_z,
            value_type const& SIXTRL_RESTRICT_REF sin_z );

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        SIXTRL_FN type_id_t  getTypeId()     const SIXTRL_NOEXCEPT;
        SIXTRL_FN size_type getNumDataPtrs() const SIXTRL_NOEXCEPT;

        SIXTRL_FN value_type getCosAngle() const SIXTRL_NOEXCEPT;
        SIXTRL_FN value_type getSinAngle() const SIXTRL_NOEXCEPT;
        SIXTRL_FN value_type getAngle()    const SIXTRL_NOEXCEPT;
        SIXTRL_FN value_type getAngleDeg() const SIXTRL_NOEXCEPT;

        SIXTRL_FN void preset() SIXTRL_NOEXCEPT;
        SIXTRL_FN void setAngle( value_type const& angle ) SIXTRL_NOEXCEPT;
        SIXTRL_FN void setAngleDeg( value_type const& a_deg ) SIXTRL_NOEXCEPT;

        /* ----------------------------------------------------------------- */

        value_type cos_z SIXTRL_ALIGN( 8 );
        value_type sin_z SIXTRL_ALIGN( 8 );
    };

    /* --------------------------------------------------------------------- */

    template< typename T >
    SIXTRL_ARGPTR_DEC TSRotation< T >* TSRotation_new( Buffer& buffer );

    template< typename T >
    SIXTRL_ARGPTR_DEC TSRotation< T >* TSRotation_new(
        SIXTRL_ARGPTR_DEC ::NS(Buffer)* SIXTRL_RESTRICT ptr_buffer );

    template< typename T >
    SIXTRL_ARGPTR_DEC TSRotation< T >*
    TSRotation_add(
        Buffer& SIXTRL_RESTRICT_REF buffer,
        typename TSRotation< T >::value_type const& SIXTRL_RESTRICT_REF angle );

    template< typename T >
    SIXTRL_ARGPTR_DEC TSRotation< T >*
    TSRotation_add(
        Buffer&  SIXTRL_RESTRICT_REF buffer,
        typename TSRotation< T >::value_type const& SIXTRL_RESTRICT_REF cos_z,
        typename TSRotation< T >::value_type const& SIXTRL_RESTRICT_REF sin_z );

    template< typename T >
    SIXTRL_ARGPTR_DEC TSRotation< T >*
    TSRotation_add(
        SIXTRL_ARGPTR_DEC ::NS(Buffer)* SIXTRL_RESTRICT ptr_buffer,
        typename TSRotation< T >::value_type const& SIXTRL_RESTRICT_REF angle );

    template< typename T >
    SIXTRL_ARGPTR_DEC TSRotation< T >*
    TSRotation_add(
        SIXTRL_ARGPTR_DEC ::NS(Buffer)* SIXTRL_RESTRICT ptr_buffer,
        typename TSRotation< T >::value_type const& SIXTRL_RESTRICT_REF cos_z,
        typename TSRotation< T >::value_type const& SIXTRL_RESTRICT_REF sin_z );

    /* ===================================================================== *
     * ====  Specialization TSRotation<  > :
     * ===================================================================== */

    template<> struct TSRotation< SIXTRL_REAL_T > : public ::NS(SRotation)
    {
        using value_type = SIXTRL_REAL_T;
        using type_id_t  = NS(object_type_id_t);
        using size_type  = NS(buffer_size_t);
        using buffer_t   = ::NS(Buffer);
        using c_api_t    = ::NS(SRotation);

        SIXTRL_FN TSRotation() = default;
        SIXTRL_FN TSRotation( TSRotation< SIXTRL_REAL_T > const& other ) = default;
        SIXTRL_FN TSRotation( TSRotation< SIXTRL_REAL_T >&& other ) = default;

        SIXTRL_FN TSRotation< SIXTRL_REAL_T >& operator=(
            TSRotation< SIXTRL_REAL_T > const& rhs ) = default;

        SIXTRL_FN TSRotation< SIXTRL_REAL_T >& operator=(
            TSRotation< SIXTRL_REAL_T >&& rhs ) = default;

        SIXTRL_FN ~TSRotation() = default;

        /* ----------------------------------------------------------------- */

        SIXTRL_FN SIXTRL_STATIC bool CanAddToBuffer(
            buffer_t& SIXTRL_RESTRICT_REF buffer,
            SIXTRL_ARGPTR_DEC size_type*
                SIXTRL_RESTRICT ptr_requ_objects = nullptr,
            SIXTRL_ARGPTR_DEC size_type*
                SIXTRL_RESTRICT ptr_requ_slots = nullptr,
            SIXTRL_ARGPTR_DEC size_type*
                SIXTRL_RESTRICT ptr_requ_dataptrs = nullptr ) SIXTRL_NOEXCEPT;

        SIXTRL_FN SIXTRL_STATIC SIXTRL_ARGPTR_DEC TSRotation< value_type >*
        CreateNewOnBuffer( buffer_t& SIXTRL_RESTRICT_REF buffer );

        SIXTRL_FN SIXTRL_STATIC SIXTRL_ARGPTR_DEC TSRotation< value_type >*
        AddToBuffer( buffer_t& SIXTRL_RESTRICT_REF buffer,
                     value_type const angle );

        SIXTRL_FN SIXTRL_STATIC SIXTRL_ARGPTR_DEC TSRotation< value_type >*
        AddToBuffer( buffer_t& SIXTRL_RESTRICT_REF buffer,
                     value_type const cos_z, value_type const sin_z );

        /* ----------------------------------------------------------------- */

        SIXTRL_FN SIXTRL_ARGPTR_DEC c_api_t const*
        getCApiPtr() const SIXTRL_NOEXCEPT;

        SIXTRL_FN SIXTRL_ARGPTR_DEC c_api_t* getCApiPtr() SIXTRL_NOEXCEPT;

        /* ----------------------------------------------------------------- */

        SIXTRL_FN type_id_t getTypeId()      const SIXTRL_NOEXCEPT;
        SIXTRL_FN size_type getNumDataPtrs() const SIXTRL_NOEXCEPT;

        SIXTRL_FN value_type getCosAngle() const SIXTRL_NOEXCEPT;
        SIXTRL_FN value_type getSinAngle() const SIXTRL_NOEXCEPT;

        SIXTRL_FN value_type getAngle()    const SIXTRL_NOEXCEPT;
        SIXTRL_FN value_type getAngleDeg() const SIXTRL_NOEXCEPT;

        SIXTRL_FN void preset() SIXTRL_NOEXCEPT;
        SIXTRL_FN void setAngle( value_type const angle ) SIXTRL_NOEXCEPT;
        SIXTRL_FN void setAngleDeg( value_type const a_deg ) SIXTRL_NOEXCEPT;
    };

    /* --------------------------------------------------------------------- */

    using SRotation = TSRotation< SIXTRL_REAL_T >;

    SIXTRL_ARGPTR_DEC SRotation* SRotation_new( Buffer& buffer );

    SIXTRL_ARGPTR_DEC SRotation* SRotation_new(
        SIXTRL_ARGPTR_DEC ::NS(Buffer)* SIXTRL_RESTRICT ptr_buf );

    SIXTRL_ARGPTR_DEC SRotation* SRotation_add(
        Buffer& buffer, SRotation::value_type const angle );

    SIXTRL_ARGPTR_DEC SRotation* SRotation_add(
        SIXTRL_ARGPTR_DEC ::NS(Buffer)* SIXTRL_RESTRICT ptr_buf,
        SRotation::value_type const angle );

    SIXTRL_ARGPTR_DEC SRotation* SRotation_add( Buffer& buffer,
        SRotation::value_type const cos_z, SRotation::value_type const sin_z );

    SIXTRL_ARGPTR_DEC SRotation* SRotation_add(
        SIXTRL_ARGPTR_DEC ::NS(Buffer)* SIXTRL_RESTRICT ptr_buffer,
        SRotation::value_type const cos_z,
        SRotation::value_type const sin_z );

}


/* ************************************************************************* *
 * *** Implementation of inline and template member functions          ***** *
 * ************************************************************************* */

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <cmath>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

namespace SIXTRL_CXX_NAMESPACE
{
    /* ===================================================================== *
     * ====  TSRotation< T > :
     * ===================================================================== */

    template< typename T >
    SIXTRL_INLINE bool TSRotation< T >::CanAddToBuffer(
        typename TSRotation< T >::buffer_t& SIXTRL_RESTRICT_REF buffer,
        SIXTRL_ARGPTR_DEC typename TSRotation< T >::size_type*
            SIXTRL_RESTRICT ptr_requ_objects,
        SIXTRL_ARGPTR_DEC typename TSRotation< T >::size_type*
            SIXTRL_RESTRICT ptr_requ_slots,
        SIXTRL_ARGPTR_DEC typename TSRotation< T >::size_type*
            SIXTRL_RESTRICT ptr_requ_dataptrs ) SIXTRL_NOEXCEPT
    {
        using _this_t = TSRotation< T >;
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
    SIXTRL_INLINE SIXTRL_ARGPTR_DEC TSRotation< T >*
    TSRotation< T >::CreateNewOnBuffer(
        typename TSRotation< T >::buffer_t& SIXTRL_RESTRICT_REF buffer )
    {
        using _this_t = TSRotation< T >;
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

        return static_cast< ptr_t >( static_cast< uintptr_t >(
            ::NS(Object_get_begin_addr)( ::NS(Buffer_add_object)(
                &buffer, &temp, sizeof( _this_t ), temp.getTypeId(),
                    num_dataptrs, offsets, sizes, counts ) ) ) );
    }

    template< typename T >
    SIXTRL_INLINE SIXTRL_ARGPTR_DEC TSRotation< T >*
    TSRotation< T >::AddToBuffer(
        typename TSRotation< T >::buffer_t& SIXTRL_RESTRICT_REF buffer,
        typename TSRotation< T >::value_type const& SIXTRL_RESTRICT_REF angle )
    {
        using _this_t = TSRotation< T >;
        using   ptr_t = SIXTRL_ARGPTR_DEC _this_t*;
        using  size_t = typename _this_t::size_type;

        static_assert( std::is_trivial< _this_t >::value, "" );
        static_assert( std::is_standard_layout< _this_t >::value, "" );

        SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST
            size_t num_dataptrs = size_t{ 0 };

        SIXTRL_ARGPTR_DEC size_t const* offsets = nullptr;
        SIXTRL_ARGPTR_DEC size_t const* sizes   = nullptr;
        SIXTRL_ARGPTR_DEC size_t const* counts  = nullptr;

        _this_t temp;
        temp.setAngle( angle );

        return static_cast< ptr_t >( static_cast< uintptr_t >(
            ::NS(Object_get_begin_addr)( ::NS(Buffer_add_object)(
                &buffer, &temp, sizeof( _this_t ), temp.getTypeId(),
                    num_dataptrs, offsets, sizes, counts ) ) ) );
    }

    template< typename T >
    SIXTRL_INLINE SIXTRL_ARGPTR_DEC TSRotation< T >*
    TSRotation< T >::AddToBuffer(
        typename TSRotation< T >::buffer_t& SIXTRL_RESTRICT_REF buffer,
        typename TSRotation< T >::value_type const& SIXTRL_RESTRICT_REF cos_z,
        typename TSRotation< T >::value_type const& SIXTRL_RESTRICT_REF sin_z )
    {
        using  _this_t = TSRotation< T >;
        using    ptr_t = SIXTRL_ARGPTR_DEC _this_t*;
        using   size_t = typename _this_t::size_type;

        static_assert( std::is_trivial< _this_t >::value, "" );
        static_assert( std::is_standard_layout< _this_t >::value, "" );

        SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST
            size_t num_dataptrs = size_t{ 0 };

        SIXTRL_ARGPTR_DEC size_t const* offsets = nullptr;
        SIXTRL_ARGPTR_DEC size_t const* sizes   = nullptr;
        SIXTRL_ARGPTR_DEC size_t const* counts  = nullptr;

        SIXTRL_ASSERT( std::fabs( ( cos_z * cos_z + sin_z * sin_z ) -
            typename _this_t::value_type{ 1 } ) <=
            std::numeric_limits< typename _this_t::value_type >::epsilon() );

        _this_t temp;
        temp.cos_z = cos_z;
        temp.sin_z = sin_z;

        return static_cast< ptr_t >( static_cast< uintptr_t >(
            ::NS(Object_get_begin_addr)( ::NS(Buffer_add_object)(
                &buffer, &temp, sizeof( _this_t ), temp.getTypeId(),
                    num_dataptrs, offsets, sizes, counts ) ) ) );
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    template< typename T >
    SIXTRL_INLINE typename TSRotation< T >::type_id_t
    TSRotation< T >::getTypeId() const SIXTRL_NOEXCEPT
    {
        return ::NS(OBJECT_TYPE_SROTATION);
    }

    template< typename T >
    SIXTRL_INLINE typename TSRotation< T >::value_type
    TSRotation< T >::getCosAngle() const SIXTRL_NOEXCEPT
    {
        return this->cos_z;
    }

    template< typename T >
    SIXTRL_INLINE typename TSRotation< T >::value_type
    TSRotation< T >::getSinAngle() const SIXTRL_NOEXCEPT
    {
        return this->sin_z;
    }

    template< typename T >
    SIXTRL_INLINE typename TSRotation< T >::value_type
    TSRotation< T >::getAngle() const SIXTRL_NOEXCEPT
    {
        using value_t = typename TSRotation< T >::value_type;
        value_t const angle = std::acos( this->cos_z );

        #if !defined( NDEBUG )
        value_t const temp_sin_z = std::sin( angle );
        value_t const EPS = value_t{ 1e-6 };
        SIXTRL_ASSERT( std::fabs( temp_sin_z - this->sin_z ) < EPS );
        #endif /* !defined( NDEBUG ) */

        return angle;
    }

    template< typename T >
    SIXTRL_INLINE typename TSRotation< T >::value_type
    TSRotation< T >::getAngleDeg() const SIXTRL_NOEXCEPT
    {
        using value_t = typename TSRotation< T >::value_type;
        value_t const RAD2DEG = value_t{ 180.0 } / M_PI;

        return RAD2DEG * this->getAngle();
    }

    template< typename T >
    SIXTRL_INLINE void TSRotation< T >::preset() SIXTRL_NOEXCEPT
    {
        using value_t = typename TSRotation< T >::value_type;
        this->setAngle( value_t{ 0 } );

        return;
    }

    template< typename T >
    SIXTRL_INLINE void TSRotation< T >::setAngle(
        typename TSRotation< T >::value_type const&
            SIXTRL_RESTRICT_REF angle ) SIXTRL_NOEXCEPT
    {
        this->cos_z = std::cos( angle );
        this->sin_z = std::sin( angle );

        return;
    }

    template< typename T >
    SIXTRL_INLINE void TSRotation< T >::setAngleDeg(
        typename TSRotation< T >::value_type const& angle_deg ) SIXTRL_NOEXCEPT
    {
        using value_t = typename TSRotation< T >::value_type;
        value_t const DEG2RAD = M_PI / value_t{ 180.0 };

        this->setAngle( DEG2RAD * angle_deg );
        return;
    }

    /* ===================================================================== *
     * ====  Specialization TSRotation< SIXTRL_REAL_T  > :
     * ===================================================================== */

    SIXTRL_INLINE bool TSRotation< SIXTRL_REAL_T >::CanAddToBuffer(
        TSRotation< SIXTRL_REAL_T >::buffer_t& SIXTRL_RESTRICT_REF buffer,
        SIXTRL_ARGPTR_DEC TSRotation< SIXTRL_REAL_T >::size_type*
            SIXTRL_RESTRICT ptr_requ_objects,
        SIXTRL_ARGPTR_DEC TSRotation< SIXTRL_REAL_T >::size_type*
            SIXTRL_RESTRICT ptr_requ_slots,
        SIXTRL_ARGPTR_DEC TSRotation< SIXTRL_REAL_T >::size_type*
            SIXTRL_RESTRICT ptr_requ_dataptrs ) SIXTRL_NOEXCEPT
    {
        return ::NS(SRotation_can_be_added)(
            &buffer, ptr_requ_objects, ptr_requ_slots, ptr_requ_dataptrs );
    }

    SIXTRL_INLINE SIXTRL_ARGPTR_DEC TSRotation< SIXTRL_REAL_T >*
    TSRotation< SIXTRL_REAL_T >::CreateNewOnBuffer(
        TSRotation< SIXTRL_REAL_T >::buffer_t& SIXTRL_RESTRICT_REF buffer )
    {
        using ptr_t = SIXTRL_ARGPTR_DEC TSRotation< SIXTRL_REAL_T >*;
        return static_cast< ptr_t >( ::NS(SRotation_new)( &buffer ) );
    }

    SIXTRL_INLINE SIXTRL_ARGPTR_DEC TSRotation< SIXTRL_REAL_T >*
    TSRotation< SIXTRL_REAL_T >::AddToBuffer(
        TSRotation< SIXTRL_REAL_T >::buffer_t& SIXTRL_RESTRICT_REF buffer,
        TSRotation< SIXTRL_REAL_T >::value_type const angle )
    {
        using ptr_t = SIXTRL_ARGPTR_DEC TSRotation< SIXTRL_REAL_T >*;
        return static_cast< ptr_t >( ::NS(SRotation_add)( &buffer, angle ) );
    }

    SIXTRL_INLINE SIXTRL_ARGPTR_DEC TSRotation< SIXTRL_REAL_T >*
    TSRotation< SIXTRL_REAL_T >::AddToBuffer(
        TSRotation< SIXTRL_REAL_T >::buffer_t& SIXTRL_RESTRICT_REF buffer,
        TSRotation< SIXTRL_REAL_T >::value_type const cos_z,
        TSRotation< SIXTRL_REAL_T >::value_type const sin_z )
    {
        using ptr_t = SIXTRL_ARGPTR_DEC TSRotation< SIXTRL_REAL_T >*;
        ptr_t srot = static_cast< ptr_t >( ::NS(SRotation_new)( &buffer ) );

        if( srot != nullptr )
        {
            srot->cos_z = cos_z;
            srot->sin_z = sin_z;
        }

        return srot;
    }

    /* ----------------------------------------------------------------- */

    SIXTRL_INLINE
    SIXTRL_ARGPTR_DEC TSRotation< SIXTRL_REAL_T >::c_api_t const*
    TSRotation< SIXTRL_REAL_T >::getCApiPtr() const SIXTRL_NOEXCEPT
    {
        using ptr_t = SIXTRL_ARGPTR_DEC
            TSRotation< SIXTRL_REAL_T >::c_api_t const*;

        return static_cast< ptr_t >( this );
    }

    SIXTRL_INLINE SIXTRL_ARGPTR_DEC TSRotation< SIXTRL_REAL_T >::c_api_t*
    TSRotation< SIXTRL_REAL_T >::getCApiPtr() SIXTRL_NOEXCEPT
    {
        using ptr_t = SIXTRL_ARGPTR_DEC
            TSRotation< SIXTRL_REAL_T >::c_api_t*;

        return const_cast< ptr_t >(
            static_cast< TSRotation< SIXTRL_REAL_T > const& >(
                *this ).getCApiPtr() );
    }

    /* ----------------------------------------------------------------- */

    SIXTRL_INLINE TSRotation< SIXTRL_REAL_T >::type_id_t
    TSRotation< SIXTRL_REAL_T >::getTypeId() const SIXTRL_NOEXCEPT
    {
        return ::NS(OBJECT_TYPE_SROTATION);
    }

    SIXTRL_INLINE TSRotation< SIXTRL_REAL_T >::value_type
    TSRotation< SIXTRL_REAL_T >::getCosAngle() const SIXTRL_NOEXCEPT
    {
        return ::NS(SRotation_get_cos_angle)( this->getCApiPtr() );
    }

    SIXTRL_INLINE TSRotation< SIXTRL_REAL_T >::value_type
    TSRotation< SIXTRL_REAL_T >::getSinAngle() const SIXTRL_NOEXCEPT
    {
        return ::NS(SRotation_get_sin_angle)( this->getCApiPtr() );
    }

    SIXTRL_INLINE TSRotation< SIXTRL_REAL_T >::value_type
    TSRotation< SIXTRL_REAL_T >::getAngle() const SIXTRL_NOEXCEPT
    {
        return ::NS(SRotation_get_angle)( this->getCApiPtr() );
    }

    SIXTRL_INLINE TSRotation< SIXTRL_REAL_T >::value_type
    TSRotation< SIXTRL_REAL_T >::getAngleDeg() const SIXTRL_NOEXCEPT
    {
        return ::NS(SRotation_get_angle_deg)( this->getCApiPtr() );
    }

    SIXTRL_INLINE void TSRotation< SIXTRL_REAL_T >::preset() SIXTRL_NOEXCEPT
    {
        ::NS(SRotation_preset)( this->getCApiPtr() );
        return;
    }

    SIXTRL_INLINE void TSRotation< SIXTRL_REAL_T >::setAngle(
        TSRotation< SIXTRL_REAL_T >::value_type const angle ) SIXTRL_NOEXCEPT
    {
        ::NS(SRotation_set_angle)( this->getCApiPtr(), angle );
        return;
    }

    SIXTRL_INLINE void TSRotation< SIXTRL_REAL_T >::setAngleDeg(
        TSRotation< SIXTRL_REAL_T >::value_type const angle_deg ) SIXTRL_NOEXCEPT
    {
        ::NS(SRotation_set_angle_deg)( this->getCApiPtr(), angle_deg );
        return;
    }

    /* --------------------------------------------------------------------- */

    SIXTRL_ARGPTR_DEC SRotation* SRotation_new( Buffer& buffer )
    {
        using ptr_t = SIXTRL_ARGPTR_DEC SRotation*;
        return static_cast< ptr_t >(
            ::NS(SRotation_new)( buffer.getCApiPtr() ) );
    }

    SIXTRL_ARGPTR_DEC SRotation* SRotation_new(
        SIXTRL_ARGPTR_DEC ::NS(Buffer)* SIXTRL_RESTRICT ptr_buf )
    {
        using ptr_t = SIXTRL_ARGPTR_DEC SRotation*;
        return static_cast< ptr_t >( ::NS(SRotation_new)( ptr_buf ) );
    }

    SIXTRL_ARGPTR_DEC SRotation* SRotation_add(
        Buffer& buffer, SRotation::value_type const angle )
    {
        using ptr_t = SIXTRL_ARGPTR_DEC SRotation*;
        return static_cast< ptr_t >(
            ::NS(SRotation_add)( buffer.getCApiPtr(), angle ) );
    }

    SIXTRL_ARGPTR_DEC SRotation* SRotation_add(
        SIXTRL_ARGPTR_DEC ::NS(Buffer)* SIXTRL_RESTRICT ptr_buf,
        SRotation::value_type const angle )
    {
        using ptr_t = SIXTRL_ARGPTR_DEC SRotation*;
        return static_cast< ptr_t >(
            ::NS(SRotation_add)( ptr_buf, angle ) );
    }

    SIXTRL_ARGPTR_DEC SRotation* SRotation_add( Buffer& buffer,
        SRotation::value_type const cos_z, SRotation::value_type const sin_z )
    {
        using ptr_t = SIXTRL_ARGPTR_DEC SRotation*;
        return static_cast< ptr_t >( ::NS(SRotation_add_detailed)(
            buffer.getCApiPtr(), cos_z, sin_z ) );

    }

    SIXTRL_ARGPTR_DEC SRotation* SRotation_add(
        SIXTRL_ARGPTR_DEC ::NS(Buffer)* SIXTRL_RESTRICT ptr_buffer,
        SRotation::value_type const cos_z,
        SRotation::value_type const sin_z )
    {
        using ptr_t = SIXTRL_ARGPTR_DEC SRotation*;
        return static_cast< ptr_t >(
            ::NS(SRotation_add_detailed)( ptr_buffer, cos_z, sin_z ) );
    }
}

#endif /* defined( __cplusplus ) */

#endif /* CXX_SIXTRACKLIB_COMMON_BE_SROTATION_BEAM_ELEM_BE_SROTATION_HPP__ */

/* end: sixtracklib/common/be_srotation/be_srotation.hpp */
