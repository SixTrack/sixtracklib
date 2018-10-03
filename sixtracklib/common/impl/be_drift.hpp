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
    #include "sixtracklib/_impl/definitions.h"
    #include "sixtracklib/common/impl/beam_elements_defines.h"
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
            SIXTRL_BUFFER_ARGPTR_DEC size_type* SIXTRL_RESTRICT
                ptr_requ_objects  = nullptr,
            SIXTRL_BUFFER_ARGPTR_DEC size_type* SIXTRL_RESTRICT
                ptr_requ_slots    = nullptr,
            SIXTRL_BUFFER_ARGPTR_DEC size_type* SIXTRL_RESTRICT
                ptr_requ_dataptrs = nullptr
        ) SIXTRL_NOEXCEPT;

        SIXTRL_FN SIXTRL_STATIC
        SIXTRL_BE_ARGPTR_DEC TDrift< T >* CreateNewOnBuffer(
            buffer_t& SIXTRL_RESTRICT_REF buffer );

        SIXTRL_FN SIXTRL_STATIC
        SIXTRL_BE_ARGPTR_DEC TDrift< T >* AddToBuffer(
            buffer_t& SIXTRL_RESTRICT_REF buffer, value_type const length );

        /* ----------------------------------------------------------------- */

        SIXTRL_FN type_id_t getTypeId()      const SIXTRL_NOEXCEPT;
        SIXTRL_FN size_type getNumDataPtrs() const SIXTRL_NOEXCEPT;

        SIXTRL_FN void preset() SIXTRL_NOEXCEPT;
        SIXTRL_FN value_type getLength() const SIXTRL_NOEXCEPT;
        SIXTRL_FN void setLength( value_type const length ) SIXTRL_NOEXCEPT;

        /* ----------------------------------------------------------------- */

        template< class ParticleT >
        SIXTRL_FN bool track(
            ParticleT& SIXTRL_RESTRICT_REF particles,
            typename ParticleT::num_elements_t const particle_index );

        /* ----------------------------------------------------------------- */

        value_type length SIXTRL_ALIGN( 8 );
    };

    template< typename T >
    SIXTRL_BE_ARGPTR_DEC TDrift< T >* TDrift_new( Buffer& buffer );

    template< typename T >
    SIXTRL_BE_ARGPTR_DEC TDrift< T >* TDrift_new(
        SIXTRL_BUFFER_ARGPTR_DEC ::NS(Buffer)* SIXTRL_RESTRICT ptr_buffer );

    template< typename T >
    SIXTRL_BE_ARGPTR_DEC TDrift< T >* TDrift_add(
        Buffer& buffer, T const& length );

    template< typename T >
    SIXTRL_BE_ARGPTR_DEC TDrift< T >* TDrift_add(
        SIXTRL_BUFFER_ARGPTR_DEC ::NS(Buffer)* SIXTRL_RESTRICT ptr_buffer,
        T const& length );

    template< typename T >
    SIXTRL_BE_ARGPTR_DEC TDrift< T >* TDrift_add_copy( Buffer& buffer,
        TDrift< T > const& SIXTRL_RESTRICT_REF orig );

    template< typename T >
    SIXTRL_BE_ARGPTR_DEC TDrift< T >* TDrift_add_copy(
        SIXTRL_BUFFER_ARGPTR_DEC ::NS(Buffer)* SIXTRL_RESTRICT ptr_buf,
        TDrift< T > const& SIXTRL_RESTRICT_REF orig );

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
                SIXTRL_BUFFER_ARGPTR_DEC size_type* SIXTRL_RESTRICT
                    ptr_requ_objects = nullptr,
                SIXTRL_BUFFER_ARGPTR_DEC size_type* SIXTRL_RESTRICT
                    ptr_requ_slots = nullptr,
                SIXTRL_BUFFER_ARGPTR_DEC size_type* SIXTRL_RESTRICT
                    ptr_requ_dataptrs = nullptr ) SIXTRL_NOEXCEPT;

        SIXTRL_FN SIXTRL_STATIC
        SIXTRL_BE_ARGPTR_DEC TDriftExact< T >* CreateNewOnBuffer(
            buffer_t& SIXTRL_RESTRICT_REF buffer );

        SIXTRL_FN SIXTRL_STATIC
        SIXTRL_BE_ARGPTR_DEC TDriftExact< T >* AddToBuffer(
            buffer_t& SIXTRL_RESTRICT_REF buffer,
            value_type const length );

        /* ----------------------------------------------------------------- */

        SIXTRL_FN type_id_t getTypeId()      const SIXTRL_NOEXCEPT;
        SIXTRL_FN size_type getNumDataPtrs() const SIXTRL_NOEXCEPT;

        SIXTRL_FN void preset() SIXTRL_NOEXCEPT;

        SIXTRL_FN value_type getLength() const SIXTRL_NOEXCEPT;
        SIXTRL_FN void setLength( value_type const length ) SIXTRL_NOEXCEPT;

        /* ----------------------------------------------------------------- */

        template< class ParticleT >
        SIXTRL_FN bool track(
            ParticleT& SIXTRL_RESTRICT_REF particles,
            typename ParticleT::num_elements_t const particle_index );

        /* ----------------------------------------------------------------- */

        value_type length SIXTRL_ALIGN( 8 );
    };

    template< typename T >
    SIXTRL_BE_ARGPTR_DEC TDriftExact< T >*
    TDriftExact_new( Buffer& buffer );

    template< typename T >
    SIXTRL_BE_ARGPTR_DEC TDriftExact< T >*
    TDriftExact_new(
        SIXTRL_BUFFER_ARGPTR_DEC ::NS(Buffer)* SIXTRL_RESTRICT ptr_buf );

    template< typename T >
    SIXTRL_BE_ARGPTR_DEC TDriftExact< T >*
    TDriftExact_add( Buffer& buffer, T const& length );

    template< typename T >
    SIXTRL_BE_ARGPTR_DEC TDriftExact< T >*
    TDriftExact_add(
        SIXTRL_BUFFER_ARGPTR_DEC ::NS(Buffer)* SIXTRL_RESTRICT ptr_buf,
        T const& length );

    template< typename T >
    SIXTRL_BE_ARGPTR_DEC TDriftExact< T >*
    TDriftExact_add_copy( Buffer& buffer,
                          TDriftExact< T > const& SIXTRL_RESTRICT_REF orig );

    template< typename T >
    SIXTRL_BE_ARGPTR_DEC TDriftExact< T >*
    TDriftExact_add_copy(
        SIXTRL_BUFFER_ARGPTR_DEC ::NS(Buffer)* SIXTRL_RESTRICT ptr_buf,
        TDriftExact< T > const& SIXTRL_RESTRICT_REF orig );

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

        SIXTRL_FN SIXTRL_STATIC SIXTRL_BE_ARGPTR_DEC
        TDrift< NS(drift_real_t) > const* FromBuffer(
            Buffer const& SIXTRL_RESTRICT_REF buffer,
            size_type const be_index ) SIXTRL_NOEXCEPT;

        SIXTRL_FN SIXTRL_STATIC SIXTRL_BE_ARGPTR_DEC
        TDrift< NS(drift_real_t) > const* FromBuffer( SIXTRL_BUFFER_ARGPTR_DEC
            const NS(Buffer) *const SIXTRL_RESTRICT buffer,
            size_type const be_index ) SIXTRL_NOEXCEPT;

        SIXTRL_FN SIXTRL_STATIC SIXTRL_BE_ARGPTR_DEC TDrift< NS(drift_real_t) >*
        FromBuffer( Buffer& SIXTRL_RESTRICT_REF buffer,
                    size_type const be_index ) SIXTRL_NOEXCEPT;

        SIXTRL_FN SIXTRL_STATIC SIXTRL_BE_ARGPTR_DEC TDrift< NS(drift_real_t) >*
        FromBuffer( SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
                    size_type const be_index ) SIXTRL_NOEXCEPT;

        SIXTRL_FN SIXTRL_STATIC SIXTRL_BE_ARGPTR_DEC
        TDrift< NS(drift_real_t) > const* FromBufferObject(
            SIXTRL_BUFFER_OBJ_ARGPTR_DEC const NS(Object) *const
                SIXTRL_RESTRICT be_info ) SIXTRL_NOEXCEPT;

        SIXTRL_FN SIXTRL_STATIC SIXTRL_BE_ARGPTR_DEC TDrift< NS(drift_real_t) >*
        FromBufferObject( SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)*
                SIXTRL_RESTRICT be_info ) SIXTRL_NOEXCEPT;

        /* ----------------------------------------------------------------- */

        SIXTRL_FN SIXTRL_STATIC bool CanAddToBuffer(
            buffer_t& SIXTRL_RESTRICT_REF buffer,
            SIXTRL_BUFFER_ARGPTR_DEC size_type* SIXTRL_RESTRICT
                ptr_requ_objects  = nullptr,
            SIXTRL_BUFFER_ARGPTR_DEC size_type* SIXTRL_RESTRICT
                ptr_requ_slots    = nullptr,
            SIXTRL_BUFFER_ARGPTR_DEC size_type* SIXTRL_RESTRICT
                ptr_requ_dataptrs = nullptr ) SIXTRL_NOEXCEPT;

        SIXTRL_FN SIXTRL_STATIC
        SIXTRL_BE_ARGPTR_DEC TDrift< NS(drift_real_t) >* CreateNewOnBuffer(
            buffer_t& SIXTRL_RESTRICT_REF buffer ) SIXTRL_NOEXCEPT;

        SIXTRL_FN SIXTRL_STATIC
        SIXTRL_BE_ARGPTR_DEC TDrift< NS(drift_real_t) >* AddToBuffer(
            buffer_t& SIXTRL_RESTRICT_REF buffer, value_type const length );

        /* ----------------------------------------------------------------- */

        SIXTRL_FN type_id_t getTypeId()      const SIXTRL_NOEXCEPT;
        SIXTRL_FN size_type getNumDataPtrs() const SIXTRL_NOEXCEPT;

        SIXTRL_FN c_api_t const* getCApiPtr() const SIXTRL_NOEXCEPT;
        SIXTRL_FN c_api_t* getCApiPtr() SIXTRL_NOEXCEPT;

        SIXTRL_FN void preset() SIXTRL_NOEXCEPT;
        SIXTRL_FN value_type getLength() const SIXTRL_NOEXCEPT;
        SIXTRL_FN void setLength( value_type const length ) SIXTRL_NOEXCEPT;

        /* ----------------------------------------------------------------- */

        template< class ParticleT >
        SIXTRL_FN bool track(
            ParticleT& SIXTRL_RESTRICT_REF particles,
            typename ParticleT::num_elements_t const particle_index );

        /* ----------------------------------------------------------------- */
    };

    using Drift = TDrift< NS(drift_real_t) >;

    SIXTRL_BE_ARGPTR_DEC Drift* Drift_new(
        Buffer& SIXTRL_RESTRICT_REF buffer );

    SIXTRL_BE_ARGPTR_DEC Drift* Drift_new(
        SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT ptr_buffer );

    SIXTRL_BE_ARGPTR_DEC Drift*
    Drift_add( Buffer& SIXTRL_RESTRICT_REF buffer,
               Drift::value_type const length );

    SIXTRL_BE_ARGPTR_DEC Drift* Drift_add(
        SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT ptr_buffer,
        Drift::value_type const length );

    SIXTRL_BE_ARGPTR_DEC Drift* Drift_add_copy(
        Buffer& SIXTRL_RESTRICT_REF buffer,
        Drift const& SIXTRL_RESTRICT_REF orig );

    SIXTRL_BE_ARGPTR_DEC Drift* Drift_add_copy(
        SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT ptr_buffer,
        Drift const& SIXTRL_RESTRICT_REF orig );

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

        SIXTRL_FN SIXTRL_STATIC SIXTRL_BE_ARGPTR_DEC
        TDriftExact< NS(drift_real_t) > const* FromBuffer(
            Buffer const& SIXTRL_RESTRICT_REF buffer,
            size_type const be_index ) SIXTRL_NOEXCEPT;

        SIXTRL_FN SIXTRL_STATIC SIXTRL_BE_ARGPTR_DEC
        TDriftExact< NS(drift_real_t) > const* FromBuffer(
            SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
            size_type const be_index ) SIXTRL_NOEXCEPT;

        SIXTRL_FN SIXTRL_STATIC SIXTRL_BE_ARGPTR_DEC
        TDriftExact< NS(drift_real_t) >* FromBuffer(
            Buffer& SIXTRL_RESTRICT_REF buffer,
            size_type const be_index ) SIXTRL_NOEXCEPT;

        SIXTRL_FN SIXTRL_STATIC SIXTRL_BE_ARGPTR_DEC
        TDriftExact< NS(drift_real_t) >* FromBuffer(
            SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
            size_type const be_index ) SIXTRL_NOEXCEPT;

        SIXTRL_FN SIXTRL_STATIC SIXTRL_BE_ARGPTR_DEC
        TDriftExact< NS(drift_real_t) > const* FromBufferObject(
            SIXTRL_BUFFER_OBJ_ARGPTR_DEC const NS(Object) *const
                SIXTRL_RESTRICT be_info ) SIXTRL_NOEXCEPT;

        SIXTRL_FN SIXTRL_STATIC SIXTRL_BE_ARGPTR_DEC
        TDriftExact< NS(drift_real_t) >* FromBufferObject(
            SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)* SIXTRL_RESTRICT
                be_info ) SIXTRL_NOEXCEPT;

        /* ----------------------------------------------------------------- */

        SIXTRL_FN SIXTRL_STATIC bool CanAddToBuffer(
                buffer_t& SIXTRL_RESTRICT_REF buffer,
                SIXTRL_BUFFER_ARGPTR_DEC size_type* SIXTRL_RESTRICT
                    ptr_requ_objects  = nullptr,
                SIXTRL_BUFFER_ARGPTR_DEC size_type* SIXTRL_RESTRICT
                    ptr_requ_slots    = nullptr,
                SIXTRL_BUFFER_ARGPTR_DEC size_type* SIXTRL_RESTRICT
                    ptr_requ_dataptrs = nullptr
            ) SIXTRL_NOEXCEPT;

        SIXTRL_FN SIXTRL_STATIC
        SIXTRL_BE_ARGPTR_DEC TDriftExact< NS(drift_real_t) >*
        CreateNewOnBuffer( buffer_t& SIXTRL_RESTRICT_REF buffer );

        SIXTRL_FN SIXTRL_STATIC
        SIXTRL_BE_ARGPTR_DEC TDriftExact< NS(drift_real_t) >*
        AddToBuffer( buffer_t& SIXTRL_RESTRICT_REF buffer,
            value_type const length );

        /* ----------------------------------------------------------------- */

        SIXTRL_FN type_id_t getTypeId()      const SIXTRL_NOEXCEPT;
        SIXTRL_FN size_type getNumDataPtrs() const SIXTRL_NOEXCEPT;

        SIXTRL_FN c_api_t const* getCApiPtr() const SIXTRL_NOEXCEPT;
        SIXTRL_FN c_api_t* getCApiPtr() SIXTRL_NOEXCEPT;


        SIXTRL_FN void       preset() SIXTRL_NOEXCEPT;
        SIXTRL_FN value_type getLength() const SIXTRL_NOEXCEPT;
        SIXTRL_FN void setLength( value_type const length ) SIXTRL_NOEXCEPT;

        /* ----------------------------------------------------------------- */

        template< class ParticleT >
        SIXTRL_FN bool track( ParticleT& SIXTRL_RESTRICT_REF particles,
            typename ParticleT::num_elements_t const particle_index );

        /* ----------------------------------------------------------------- */
    };

    /* --------------------------------------------------------------------- */

    using DriftExact = TDriftExact< NS(drift_real_t) >;

    SIXTRL_BE_ARGPTR_DEC DriftExact*
    DriftExact_new( Buffer& SIXTRL_RESTRICT_REF buffer );

    SIXTRL_BE_ARGPTR_DEC DriftExact*
    DriftExact_new(
        SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT ptr_buffer );

    SIXTRL_BE_ARGPTR_DEC DriftExact*
    DriftExact_add( Buffer& SIXTRL_RESTRICT_REF buffer,
                    DriftExact::value_type const length );

    SIXTRL_BE_ARGPTR_DEC DriftExact*
    DriftExact_add(
        SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT ptr_buffer,
        DriftExact::value_type const length );

    SIXTRL_BE_ARGPTR_DEC DriftExact*
    DriftExact_add_copy( Buffer& SIXTRL_RESTRICT_REF buffer,
                    DriftExact const& SIXTRL_RESTRICT_REF orig );

    SIXTRL_BE_ARGPTR_DEC DriftExact*
    DriftExact_add_copy(
        SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT ptr_buffer,
        DriftExact const& SIXTRL_RESTRICT_REF orig );
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
            SIXTRL_BUFFER_ARGPTR_DEC typename TDrift< T >::size_type*
                SIXTRL_RESTRICT ptr_requ_objects,
            SIXTRL_BUFFER_ARGPTR_DEC typename TDrift< T >::size_type*
                SIXTRL_RESTRICT ptr_requ_slots,
            SIXTRL_BUFFER_ARGPTR_DEC typename TDrift< T >::size_type*
                SIXTRL_RESTRICT ptr_requ_dataptrs ) SIXTRL_NOEXCEPT
    {
        using _this_t = TDrift< T >;
        using  size_t = typename _this_t::size_type;

        static_assert( std::is_trivial< _this_t >::value, "" );
        static_assert( std::is_standard_layout< _this_t >::value, "" );

        SIXTRL_BUFFER_ARGPTR_DEC size_t const* sizes  = nullptr;
        SIXTRL_BUFFER_ARGPTR_DEC size_t const* counts = nullptr;

        _this_t temp;
        temp.preset();

        return ::NS(Buffer_can_add_object)( &buffer, sizeof( temp ),
            temp.getNumDataPtrs(), sizes, counts, ptr_requ_objects,
                ptr_requ_slots, ptr_requ_dataptrs );
    }

    template< typename T >
    SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC TDrift< T >*
    TDrift< T >::CreateNewOnBuffer(
        typename TDrift< T >::buffer_t& SIXTRL_RESTRICT_REF buffer )
    {
        using _this_t = TDrift< T >;
        using  size_t = typename TDrift< T >::size_type;
        using  ptr_t  = SIXTRL_BE_ARGPTR_DEC _this_t*;

        static_assert( std::is_trivial< _this_t >::value, "" );
        static_assert( std::is_standard_layout< _this_t >::value, "" );

        SIXTRL_BUFFER_ARGPTR_DEC size_t const* offsets = nullptr;
        SIXTRL_BUFFER_ARGPTR_DEC size_t const* sizes   = nullptr;
        SIXTRL_BUFFER_ARGPTR_DEC size_t const* counts  = nullptr;

        _this_t temp;
        temp.preset();

        return reinterpret_cast< ptr_t >( static_cast< uintptr_t >(
            ::NS(Object_get_begin_addr)( ::NS(Buffer_add_object)(
                &buffer, &temp, sizeof( temp ), temp.getTypeId(),
                    temp.getNumDataPtrs(), offsets, sizes, counts ) ) ) );
    }

    template< typename T >
    SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC TDrift< T >*
    TDrift< T >::AddToBuffer(
        typename TDrift< T >::buffer_t& SIXTRL_RESTRICT_REF buffer,
        typename TDrift< T >::value_type const length )
    {
        using _this_t = TDrift< T >;
        using  size_t = typename _this_t::size_type;
        using  ptr_t  = SIXTRL_BE_ARGPTR_DEC _this_t*;

        static_assert( std::is_trivial< _this_t >::value, "" );
        static_assert( std::is_standard_layout< _this_t >::value, "" );

        SIXTRL_BUFFER_ARGPTR_DEC size_t const* offsets = nullptr;
        SIXTRL_BUFFER_ARGPTR_DEC size_t const* sizes   = nullptr;
        SIXTRL_BUFFER_ARGPTR_DEC size_t const* counts  = nullptr;

        _this_t temp;
        temp.setLength( length );

        return reinterpret_cast< ptr_t >( static_cast< uintptr_t >(
            ::NS(Object_get_begin_addr)( ::NS(Buffer_add_object)(
                &buffer, &temp, sizeof( temp ), temp.getTypeId(),
                    temp.getNumDataPtrs(), offsets, sizes, counts ) ) ) );
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    template< typename T >
    SIXTRL_INLINE typename TDrift< T >::type_id_t
    TDrift< T >::getTypeId() const SIXTRL_NOEXCEPT
    {
        return ::NS(OBJECT_TYPE_DRIFT);
    }

    template< typename T >
    SIXTRL_INLINE typename TDrift< T >::size_type
    TDrift< T >::getNumDataPtrs() const SIXTRL_NOEXCEPT
    {
        return typename TDrift< T >::size_type{ 0 };
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

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
    SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC TDrift< T >* TDrift_new( Buffer& buffer )
    {
        return TDrift< T >::CreateNewOnBuffer( *( buffer.getCApiPtr() ) );
    }

    template< typename T >
    SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC TDrift< T >* TDrift_new(
        ::NS(Buffer)* SIXTRL_RESTRICT ptr_buffer )
    {
        return ( ptr_buffer != nullptr )
            ? ( TDrift< T >::CreateNewOnBuffer( *ptr_buffer ) )
            : nullptr;
    }

    template< typename T >
    SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC TDrift< T >* TDrift_add(
        Buffer& buffer, T const& length )
    {
        return TDrift< T >::AddToBuffer( *( buffer.getCApiPtr() ), length );
    }

    template< typename T >
    SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC TDrift< T >* TDrift_add(
        ::NS(Buffer)* SIXTRL_RESTRICT ptr_buffer, T const& length )
    {
        return ( ptr_buffer != nullptr )
            ? ( TDrift< T >::AddToBuffer( *ptr_buffer, length ) )
            : nullptr;
    }

    template< typename T >
    SIXTRL_BE_ARGPTR_DEC TDrift< T >* TDrift_add_copy( Buffer& buffer,
        TDrift< T > const& SIXTRL_RESTRICT_REF orig )
    {
        return TDrift_add( buffer, orig.getLength() );
    }

    template< typename T >
    SIXTRL_BE_ARGPTR_DEC TDrift< T >* TDrift_add_copy(
        SIXTRL_BUFFER_ARGPTR_DEC ::NS(Buffer)* SIXTRL_RESTRICT ptr_buf,
        TDrift< T > const& SIXTRL_RESTRICT_REF orig )
    {
        return TDrift_add( ptr_buf, orig.getLength() );
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
        using  size_t = typename _this_t::size_type;

        static_assert( std::is_trivial< _this_t >::value, "" );
        static_assert( std::is_standard_layout< _this_t >::value, "" );

        SIXTRL_BUFFER_ARGPTR_DEC size_t const* sizes  = nullptr;
        SIXTRL_BUFFER_ARGPTR_DEC size_t const* counts = nullptr;

        _this_t temp;
        temp.preset();

        return ::NS(Buffer_can_add_object)( &buffer, sizeof( temp ),
            temp.getNumDataPtrs(), sizes, counts, ptr_requ_objects,
                ptr_requ_slots, ptr_requ_dataptrs );
    }

    template< typename T >
    SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC TDriftExact< T >*
    TDriftExact< T >::CreateNewOnBuffer(
        typename TDriftExact< T >::buffer_t& SIXTRL_RESTRICT_REF buffer )
    {
        using _this_t = TDriftExact< T >;
        using  size_t = typename _this_t::size_type;
        using  ptr_t  = SIXTRL_BE_ARGPTR_DEC _this_t*;

        static_assert( std::is_trivial< _this_t >::value, "" );
        static_assert( std::is_standard_layout< _this_t >::value, "" );

        SIXTRL_BUFFER_ARGPTR_DEC size_t const* offsets = nullptr;
        SIXTRL_BUFFER_ARGPTR_DEC size_t const* sizes   = nullptr;
        SIXTRL_BUFFER_ARGPTR_DEC size_t const* counts  = nullptr;

        _this_t temp;
        temp.preset();

        return reinterpret_cast< ptr_t >( static_cast< uintptr_t >(
            ::NS(Object_get_begin_addr)( ::NS(Buffer_add_object)(
                &buffer, &temp, sizeof( temp ), temp.getTypeId(),
                    temp.getNumDataPtrs(), offsets, sizes, counts ) ) ) );
    }

    template< typename T >
    SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC TDriftExact< T >*
    TDriftExact< T >::AddToBuffer(
        typename TDriftExact< T >::buffer_t& SIXTRL_RESTRICT_REF buffer,
        typename TDriftExact< T >::value_type const length )
    {
        using _this_t = TDriftExact< T >;
        using  size_t = typename _this_t::size_type;
        using  ptr_t  = SIXTRL_BE_ARGPTR_DEC _this_t*;

        static_assert( std::is_trivial< _this_t >::value, "" );
        static_assert( std::is_standard_layout< _this_t >::value, "" );

        SIXTRL_BUFFER_ARGPTR_DEC size_t const* offsets = nullptr;
        SIXTRL_BUFFER_ARGPTR_DEC size_t const* sizes   = nullptr;
        SIXTRL_BUFFER_ARGPTR_DEC size_t const* counts  = nullptr;

        _this_t temp;
        temp.setLength( length );

        return reinterpret_cast< ptr_t >( static_cast< uintptr_t >(
            ::NS(Object_get_begin_addr)( ::NS(Buffer_add_object)(
                &buffer, &temp, sizeof( temp ), temp.getTypeId(),
                    temp.getNumDataPtrs(), offsets, sizes, counts ) ) ) );
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    template< typename T >
    SIXTRL_INLINE typename TDriftExact< T >::type_id_t
    TDriftExact< T >::getTypeId() const SIXTRL_NOEXCEPT
    {
        return ::NS(OBJECT_TYPE_DRIFT_EXACT);
    }

    template< typename T >
    SIXTRL_INLINE typename TDriftExact< T >::size_type
    TDriftExact< T >::getNumDataPtrs() const SIXTRL_NOEXCEPT
    {
        return typename TDriftExact< T >::size_type{ 0 };
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

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
    SIXTRL_BE_ARGPTR_DEC TDriftExact< T >* TDriftExact_new( Buffer& buffer )
    {
        return TDriftExact< T >::CreateNewOnBuffer( *buffer.getCApiPtr() );
    }

    template< typename T >
    SIXTRL_BE_ARGPTR_DEC TDriftExact< T >* TDriftExact_new(
        SIXTRL_BUFFER_ARGPTR_DEC ::NS(Buffer)* SIXTRL_RESTRICT ptr_buffer )
    {
        return ( ptr_buffer != nullptr )
            ? ( TDriftExact< T >::CreateNewOnBuffer( *ptr_buffer ) )
            : nullptr;
    }

    template< typename T >
    SIXTRL_BE_ARGPTR_DEC TDriftExact< T >* TDriftExact_add(
        Buffer& buffer, T const& length )
    {
        return TDriftExact< T >::AddToBuffer(
            *( buffer.getCApiPtr() ), length );
    }

    template< typename T >
    SIXTRL_BE_ARGPTR_DEC TDriftExact< T >* TDriftExact_add(
        SIXTRL_BUFFER_ARGPTR_DEC ::NS(Buffer)* SIXTRL_RESTRICT ptr_buffer,
        T const& length )
    {
        return ( ptr_buffer != nullptr )
            ? ( TDriftExact< T >::AddToBuffer( *ptr_buffer, length ) )
            : nullptr;
    }

    template< typename T >
    SIXTRL_BE_ARGPTR_DEC TDriftExact< T >* TDriftExact_add_copy( Buffer& buffer,
        TDriftExact< T > const& SIXTRL_RESTRICT_REF orig )
    {
        return TDriftExact_add( buffer, orig.getLength() );
    }

    template< typename T >
    SIXTRL_BE_ARGPTR_DEC TDriftExact< T >* TDriftExact_add_copy(
        SIXTRL_BUFFER_ARGPTR_DEC ::NS(Buffer)* SIXTRL_RESTRICT ptr_buf,
        TDriftExact< T > const& SIXTRL_RESTRICT_REF orig )
    {
        return TDriftExact_add( ptr_buf, orig.getLength() );
    }

    /* ===================================================================== *
     * ====  TDrift< NS(drift_real_t) >:
     * ===================================================================== */

    SIXTRL_INLINE
    SIXTRL_BUFFER_OBJ_DATAPTR_DEC TDrift< NS(drift_real_t) > const*
    TDrift< NS(drift_real_t) >::FromBuffer(
        Buffer const& SIXTRL_RESTRICT_REF buffer,
        TDrift< NS(drift_real_t) >::size_type const be_index ) SIXTRL_NOEXCEPT
    {
        using  _this_t = TDrift< NS(drift_real_t) >;
        return _this_t::FromBufferObject( buffer[ be_index ] );
    }

    SIXTRL_INLINE
    SIXTRL_BUFFER_OBJ_DATAPTR_DEC TDrift< NS(drift_real_t) >*
    TDrift< NS(drift_real_t) >::FromBuffer(
        Buffer& SIXTRL_RESTRICT_REF buffer,
        TDrift< NS(drift_real_t) >::size_type const be_index ) SIXTRL_NOEXCEPT
    {
        using  _this_t = TDrift< NS(drift_real_t) >;
        return _this_t::FromBufferObject( buffer[ be_index ] );
    }

    SIXTRL_INLINE
    SIXTRL_BUFFER_OBJ_DATAPTR_DEC TDrift< NS(drift_real_t) > const*
    TDrift< NS(drift_real_t) >::FromBuffer( SIXTRL_BUFFER_ARGPTR_DEC
        const NS(Buffer) *const SIXTRL_RESTRICT buffer,
            size_type const be_index ) SIXTRL_NOEXCEPT
    {
        using  _this_t = TDrift< NS(drift_real_t) >;
        return _this_t::FromBufferObject(
            NS(Buffer_get_const_object)( buffer, be_index ) );
    }

    SIXTRL_INLINE
    SIXTRL_BUFFER_OBJ_DATAPTR_DEC TDrift< NS(drift_real_t) >*
    TDrift< NS(drift_real_t) >::FromBuffer( SIXTRL_BUFFER_ARGPTR_DEC
        NS(Buffer)* SIXTRL_RESTRICT buffer,
            size_type const be_index ) SIXTRL_NOEXCEPT
    {
        using  _this_t = TDrift< NS(drift_real_t) >;

        return _this_t::FromBufferObject(
            NS(Buffer_get_object)( buffer, be_index ) );
    }

    SIXTRL_INLINE
    SIXTRL_BUFFER_OBJ_DATAPTR_DEC TDrift< NS(drift_real_t) > const*
    TDrift< NS(drift_real_t) >::FromBufferObject(
        SIXTRL_BUFFER_OBJ_ARGPTR_DEC const NS(Object) *const
            SIXTRL_RESTRICT be_info ) SIXTRL_NOEXCEPT
    {
        using  _this_t        = TDrift< NS(drift_real_t) >;
        using beam_element_t  = _this_t;
        using ptr_beam_elem_t =
            SIXTRL_BUFFER_OBJ_DATAPTR_DEC beam_element_t const*;

        if( ( be_info != nullptr ) &&
            ( NS(Object_get_type_id)( be_info ) == NS(OBJECT_TYPE_DRIFT) ) &&
            ( NS(Object_get_size)( be_info ) >= sizeof( _this_t ) ) )
        {
            return reinterpret_cast< ptr_beam_elem_t >(
                static_cast< uintptr_t >( NS(Object_get_begin_addr)(
                    be_info ) ) );
        }

        return nullptr;
    }

    SIXTRL_INLINE SIXTRL_BUFFER_OBJ_DATAPTR_DEC TDrift< NS(drift_real_t) >*
    TDrift< NS(drift_real_t) >::FromBufferObject(
        SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)*
            SIXTRL_RESTRICT be_info ) SIXTRL_NOEXCEPT
    {
        using  _this_t        = TDrift< NS(drift_real_t) >;
        using beam_element_t  = _this_t;
        using ptr_beam_elem_t = SIXTRL_BUFFER_OBJ_DATAPTR_DEC beam_element_t*;

        using object_t        = NS(Object);
        using ptr_const_obj_t = SIXTRL_BUFFER_OBJ_ARGPTR_DEC object_t const*;

        ptr_const_obj_t const_be_info = be_info;

        return const_cast< ptr_beam_elem_t >(
            _this_t::FromBufferObject( const_be_info ) );
    }

    /* --------------------------------------------------------------------- */

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

    SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC TDrift< NS(drift_real_t) >*
    TDrift< NS(drift_real_t) >::CreateNewOnBuffer(
        TDrift< NS(drift_real_t) >::buffer_t&
            SIXTRL_RESTRICT_REF buffer ) SIXTRL_NOEXCEPT
    {
        using ptr_t = SIXTRL_BE_ARGPTR_DEC TDrift< NS(drift_real_t) >*;
        return static_cast< ptr_t >( ::NS(Drift_new)( &buffer ) );
    }


    SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC TDrift< NS(drift_real_t) >*
    TDrift< NS(drift_real_t) >::AddToBuffer(
        TDrift< NS(drift_real_t) >::buffer_t& SIXTRL_RESTRICT_REF buffer,
        TDrift< NS(drift_real_t) >::value_type const length )
    {
        using ptr_t = SIXTRL_BE_ARGPTR_DEC TDrift< NS(drift_real_t) >*;
        return static_cast< ptr_t >( ::NS(Drift_add)( &buffer, length ) );
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC
    TDrift< NS(drift_real_t) >::c_api_t const*
    TDrift< NS(drift_real_t) >::getCApiPtr() const SIXTRL_NOEXCEPT
    {
        return reinterpret_cast< SIXTRL_BE_ARGPTR_DEC c_api_t const* >( this );
    }

    SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC TDrift< NS(drift_real_t) >::c_api_t*
    TDrift< NS(drift_real_t) >::getCApiPtr() SIXTRL_NOEXCEPT
    {
        return const_cast< SIXTRL_BE_ARGPTR_DEC c_api_t* >(
            static_cast< TDrift< NS(drift_real_t) > const& >(
                *this ).getCApiPtr() );
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    SIXTRL_INLINE TDrift< NS(drift_real_t) >::type_id_t
    TDrift< NS(drift_real_t) >::getTypeId() const SIXTRL_NOEXCEPT
    {
        return ::NS(OBJECT_TYPE_DRIFT);
    }

    SIXTRL_INLINE TDrift< NS(drift_real_t) >::size_type
    TDrift< NS(drift_real_t) >::getNumDataPtrs() const SIXTRL_NOEXCEPT
    {
        return TDrift< NS(drift_real_t) >::size_type{ 0 };
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

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

    SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC Drift*
    Drift_new( Buffer& SIXTRL_RESTRICT_REF buffer )
    {
        using ptr_t = SIXTRL_BE_ARGPTR_DEC Drift*;
        return static_cast< ptr_t >( ::NS(Drift_new)( buffer.getCApiPtr() ) );
    }

    SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC Drift* Drift_new(
        SIXTRL_BE_ARGPTR_DEC ::NS(Buffer)* SIXTRL_RESTRICT ptr_buffer )
    {
        using ptr_t = SIXTRL_BE_ARGPTR_DEC Drift*;
        return static_cast< ptr_t >( ::NS(Drift_new)( ptr_buffer ) );
    }

    SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC Drift*
    Drift_add( Buffer& SIXTRL_RESTRICT_REF buffer,
                      Drift::value_type const length )
    {
        using ptr_t = SIXTRL_BE_ARGPTR_DEC Drift*;
        return static_cast< ptr_t >(
            ::NS(Drift_add)( buffer.getCApiPtr(), length ) );
    }

    SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC Drift*
    Drift_add( NS(Buffer)* SIXTRL_RESTRICT ptr_buffer,
                      Drift::value_type const length )
    {
        using ptr_t = SIXTRL_BE_ARGPTR_DEC Drift*;
        return static_cast< ptr_t >( ::NS(Drift_add)( ptr_buffer, length ) );
    }

    SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC Drift*
    Drift_add_copy( Buffer& SIXTRL_RESTRICT_REF buffer,
                    Drift const& SIXTRL_RESTRICT_REF orig )
    {
        return Drift_add( buffer.getCApiPtr(), orig.getLength() );
    }

    SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC Drift*
    Drift_add_copy( NS(Buffer)* SIXTRL_RESTRICT ptr_buffer,
                    Drift const& SIXTRL_RESTRICT_REF orig )
    {
        return Drift_add( ptr_buffer, orig.getLength() );
    }

    /* ===================================================================== *
     * ====  TDriftExact< NS(drift_real_t) >:
     * ===================================================================== */

    SIXTRL_INLINE
    SIXTRL_BUFFER_OBJ_DATAPTR_DEC TDriftExact< NS(drift_real_t) > const*
    TDriftExact< NS(drift_real_t) >::FromBuffer(
        Buffer const& SIXTRL_RESTRICT_REF buffer,
        TDriftExact< NS(drift_real_t) >::size_type const be_index ) SIXTRL_NOEXCEPT
    {
        using  _this_t = TDriftExact< NS(drift_real_t) >;
        return _this_t::FromBufferObject( buffer[ be_index ] );
    }

    SIXTRL_INLINE
    SIXTRL_BUFFER_OBJ_DATAPTR_DEC TDriftExact< NS(drift_real_t) >*
    TDriftExact< NS(drift_real_t) >::FromBuffer(
        Buffer& SIXTRL_RESTRICT_REF buffer,
        TDriftExact< NS(drift_real_t) >::size_type const be_index ) SIXTRL_NOEXCEPT
    {
        using  _this_t = TDriftExact< NS(drift_real_t) >;
        return _this_t::FromBufferObject( buffer[ be_index ] );
    }

    SIXTRL_INLINE
    SIXTRL_BUFFER_OBJ_DATAPTR_DEC TDriftExact< NS(drift_real_t) > const*
    TDriftExact< NS(drift_real_t) >::FromBuffer( SIXTRL_BUFFER_ARGPTR_DEC
        const NS(Buffer) *const SIXTRL_RESTRICT buffer,
            size_type const be_index ) SIXTRL_NOEXCEPT
    {
        using  _this_t = TDriftExact< NS(drift_real_t) >;
        return _this_t::FromBufferObject(
            NS(Buffer_get_const_object)( buffer, be_index ) );
    }

    SIXTRL_INLINE
    SIXTRL_BUFFER_OBJ_DATAPTR_DEC TDriftExact< NS(drift_real_t) >*
    TDriftExact< NS(drift_real_t) >::FromBuffer( SIXTRL_BUFFER_ARGPTR_DEC
        NS(Buffer)* SIXTRL_RESTRICT buffer,
            size_type const be_index ) SIXTRL_NOEXCEPT
    {
        using  _this_t = TDriftExact< NS(drift_real_t) >;

        return _this_t::FromBufferObject(
            NS(Buffer_get_object)( buffer, be_index ) );
    }

    SIXTRL_INLINE
    SIXTRL_BUFFER_OBJ_DATAPTR_DEC TDriftExact< NS(drift_real_t) > const*
    TDriftExact< NS(drift_real_t) >::FromBufferObject(
        SIXTRL_BUFFER_OBJ_ARGPTR_DEC const NS(Object) *const
            SIXTRL_RESTRICT be_info ) SIXTRL_NOEXCEPT
    {
        using  _this_t        = TDriftExact< NS(drift_real_t) >;
        using beam_element_t  = _this_t;
        using ptr_beam_elem_t =
            SIXTRL_BUFFER_OBJ_DATAPTR_DEC beam_element_t const*;

        if( ( be_info != nullptr ) &&
            ( NS(Object_get_type_id)( be_info ) == NS(OBJECT_TYPE_DRIFT_EXACT) ) &&
            ( NS(Object_get_size)( be_info ) >= sizeof( _this_t ) ) )
        {
            return reinterpret_cast< ptr_beam_elem_t >(
                static_cast< uintptr_t >( NS(Object_get_begin_addr)(
                    be_info ) ) );
        }

        return nullptr;
    }

    SIXTRL_INLINE SIXTRL_BUFFER_OBJ_DATAPTR_DEC TDriftExact< NS(drift_real_t) >*
    TDriftExact< NS(drift_real_t) >::FromBufferObject(
        SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)*
            SIXTRL_RESTRICT be_info ) SIXTRL_NOEXCEPT
    {
        using  _this_t        = TDriftExact< NS(drift_real_t) >;
        using beam_element_t  = _this_t;
        using ptr_beam_elem_t = SIXTRL_BUFFER_OBJ_DATAPTR_DEC beam_element_t*;

        using object_t        = NS(Object);
        using ptr_const_obj_t = SIXTRL_BUFFER_OBJ_ARGPTR_DEC object_t const*;

        ptr_const_obj_t const_be_info = be_info;

        return const_cast< ptr_beam_elem_t >(
            _this_t::FromBufferObject( const_be_info ) );
    }

    /* --------------------------------------------------------------------- */

    SIXTRL_INLINE bool TDriftExact< NS(drift_real_t) >::CanAddToBuffer(
            TDriftExact< NS(drift_real_t) >::buffer_t&
                SIXTRL_RESTRICT_REF buffer,
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

    SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC TDriftExact< NS(drift_real_t) >*
    TDriftExact< NS(drift_real_t) >::CreateNewOnBuffer(
            TDriftExact< NS(drift_real_t) >::buffer_t&
                SIXTRL_RESTRICT_REF buffer )
    {
        using ptr_t = SIXTRL_BE_ARGPTR_DEC TDriftExact< NS(drift_real_t) >*;
        return static_cast< ptr_t >( ::NS(DriftExact_new)( &buffer ) );
    }

    SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC TDriftExact< NS(drift_real_t) >*
    TDriftExact< NS(drift_real_t) >::AddToBuffer(
        TDriftExact< NS(drift_real_t) >::buffer_t& SIXTRL_RESTRICT_REF buffer,
        TDriftExact< NS(drift_real_t) >::value_type const length )
    {
        using ptr_t = SIXTRL_BE_ARGPTR_DEC TDriftExact< NS(drift_real_t) >*;
        return static_cast< ptr_t >( ::NS(DriftExact_add)( &buffer, length ) );
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    SIXTRL_INLINE
    SIXTRL_BE_ARGPTR_DEC TDriftExact< NS(drift_real_t) >::c_api_t const*
    TDriftExact< NS(drift_real_t) >::getCApiPtr() const SIXTRL_NOEXCEPT
    {
        return static_cast< SIXTRL_BE_ARGPTR_DEC c_api_t const* >( this );
    }

    SIXTRL_INLINE
    SIXTRL_BE_ARGPTR_DEC TDriftExact< NS(drift_real_t) >::c_api_t*
    TDriftExact< NS(drift_real_t) >::getCApiPtr() SIXTRL_NOEXCEPT
    {
        return const_cast< SIXTRL_BE_ARGPTR_DEC c_api_t* >(
            static_cast< TDriftExact< NS(drift_real_t) > const& >(
                *this ).getCApiPtr() );
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    SIXTRL_INLINE TDriftExact< NS(drift_real_t) >::type_id_t
    TDriftExact< NS(drift_real_t) >::getTypeId() const SIXTRL_NOEXCEPT
    {
        return ::NS(OBJECT_TYPE_DRIFT_EXACT);
    }

    SIXTRL_INLINE TDriftExact< NS(drift_real_t) >::size_type
    TDriftExact< NS(drift_real_t) >::getNumDataPtrs() const SIXTRL_NOEXCEPT
    {
        return TDriftExact< NS(drift_real_t) >::size_type{ 0 };
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

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
        TDriftExact< NS(drift_real_t) >::value_type const
            length ) SIXTRL_NOEXCEPT
    {
        ::NS(DriftExact_set_length)( this->getCApiPtr(), length );
        return;
    }

    /* --------------------------------------------------------------------- */

    SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC DriftExact* DriftExact_new(
        Buffer& SIXTRL_RESTRICT_REF buffer )
    {
        using ptr_t = SIXTRL_BE_ARGPTR_DEC DriftExact*;
        return static_cast< ptr_t >(
            ::NS(DriftExact_new)( buffer.getCApiPtr() ) );
    }

    SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC DriftExact* DriftExact_new(
        ::NS(Buffer)* SIXTRL_RESTRICT ptr_buffer )
    {
        using ptr_t = SIXTRL_BE_ARGPTR_DEC DriftExact*;
        return static_cast< ptr_t >( ::NS(DriftExact_new)( ptr_buffer ) );
    }

    SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC DriftExact* DriftExact_add(
        Buffer& SIXTRL_RESTRICT_REF buffer,
        DriftExact::value_type const length )
    {
        using ptr_t = SIXTRL_BE_ARGPTR_DEC DriftExact*;
        return static_cast< ptr_t >(
                ::NS(DriftExact_add)( buffer.getCApiPtr(), length ) );
    }

    SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC DriftExact* DriftExact_add(
        NS(Buffer)* SIXTRL_RESTRICT ptr_buffer,
        DriftExact::value_type const length )
    {
        using ptr_t = SIXTRL_BE_ARGPTR_DEC DriftExact*;
        return static_cast< ptr_t >(
            ::NS(DriftExact_add)( ptr_buffer, length ) );
    }

    SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC DriftExact*
    DriftExact_add_copy( Buffer& SIXTRL_RESTRICT_REF buffer,
                    DriftExact const& SIXTRL_RESTRICT_REF orig )
    {
        return DriftExact_add( buffer.getCApiPtr(), orig.getLength() );
    }

    SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC DriftExact*
    DriftExact_add_copy( NS(Buffer)* SIXTRL_RESTRICT ptr_buffer,
        DriftExact const& SIXTRL_RESTRICT_REF orig )
    {
        return DriftExact_add( ptr_buffer, orig.getLength() );
    }
}

#endif /* defined( __cplusplus ) */

#endif /* CXX_SIXTRACKLIB_COMMON_IMPL_BE_DRIFT_HPP__ */

/* end: sixtracklib/common/impl/be_drift.hpp */
