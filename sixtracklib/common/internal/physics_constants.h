#ifndef SIXTRACKLIB_COMMON_INTERNAL_PHYSICS_CONSTANTS_H__
#define SIXTRACKLIB_COMMON_INTERNAL_PHYSICS_CONSTANTS_H__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #if defined( __cplusplus )
        #include <cstddef>
        #include <cstdint>
        #include <cstdlib>
        #include <cmath>
        #include <type_traits>
    #else /* defined( __cplusplus ) */
        #include <stddef.h>
        #include <stdint.h>
        #include <stdlib.h>
        #include <math.h>
    #endif /* defined( __cplusplus ) */
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/particles/definitions.h"
    #include "sixtracklib/common/internal/compiler_attributes.h"
    #include "sixtracklib/common/internal/type_store_traits.hpp"
    #include "sixtracklib/common/internal/type_comparison_helpers.hpp"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( SIXTRL_PHYS_CONST_CLIGHT )
    #define SIXTRL_PHYS_CONST_CLIGHT 299792458.0
#endif /* SIXTRL_PHYS_CONST_CLIGHT */

#if !defined( SIXTRL_PHYS_CONST_EPSILON0 )
    #define SIXTRL_PHYS_CONST_EPSILON0 8.854187817620e-12
#endif /* SIXTRL_PHYS_CONST_EPSILON0 */

#if !defined( SIXTRL_PHYS_CONST_MU0 )
    #define SIXTRL_PHYS_CONST_MU0 \
        1.25663706143591729538505735331180115367886775975004232839e-6
#endif /* SIXTRL_PHYS_CONST_MU0 */

#if !defined( SIXTRL_PHYS_CONST_MASS_ATOMIC_SI )
    #define SIXTRL_PHYS_CONST_MASS_ATOMIC_SI 1.66053906660e-27
#endif /* SIXTRL_PHYS_CONST_MASS_ATOMIC_SI */

#if !defined( SIXTRL_PHYS_CONST_MASS_ATOMIC_EV )
    #define SIXTRL_PHYS_CONST_MASS_ATOMIC_EV 931494102.42
#endif /* SIXTRL_PHYS_CONST_MASS_ATOMIC_EV */

#if !defined( SIXTRL_PHYS_CONST_MASS_PROTON_SI )
    #define SIXTRL_PHYS_CONST_MASS_PROTON_SI 1.67262192369e-27
#endif /* SIXTRL_PHYS_CONST_MASS_PROTON_SI */

#if !defined( SIXTRL_PHYS_CONST_MASS_PROTON_EV )
    #define SIXTRL_PHYS_CONST_MASS_PROTON_EV 938272088.16
#endif /* SIXTRL_PHYS_CONST_MASS_PROTON_EV */

#if !defined( SIXTRL_PHYS_CONST_MASS_ELECTRON_SI )
    #define SIXTRL_PHYS_CONST_MASS_ELECTRON_SI 9.1093837015e-31
#endif /* SIXTRL_PHYS_CONST_MASS_ELECTRON_SI */

#if !defined( SIXTRL_PHYS_CONST_MASS_ELECTRON_EV )
    #define SIXTRL_PHYS_CONST_MASS_ELECTRON_EV 510998.95
#endif /* SIXTRL_PHYS_CONST_MASS_ELECTRON_EV */

#if !defined( SIXTRL_PHYS_CONST_CHARGE0_SI )
    #define SIXTRL_PHYS_CONST_CHARGE0_SI 1.602176634e-19
#endif /* SIXTRL_PHYS_CONST_CHARGE0_SI */

#if !defined( SIXTRL_PHYS_CONST_CHARGE0 )
    #define SIXTRL_PHYS_CONST_CHARGE0 1
#endif /* SIXTRL_PHYS_CONST_CHARGE0 */

#if defined( __cplusplus )
namespace SIXTRL_CXX_NAMESPACE
{
    template< class R >
    SIXTRL_STATIC SIXTRL_INLINE SIXTRL_FN
    typename TypeMethodParamTraits< R >::value_type
    PhysConst_clight( typename TypeMethodParamTraits< R >::const_pointer
        SIXTRL_RESTRICT SIXTRL_UNUSED( dummy ) = SIXTRL_NULLPTR )
    {
        return static_cast< typename TypeMethodParamTraits< R >::value_type >(
            SIXTRL_PHYS_CONST_CLIGHT );
    }

    template< class R >
    SIXTRL_STATIC SIXTRL_INLINE SIXTRL_FN typename TypeMethodParamTraits<
        R >::value_type
    PhysConst_epsilon0( typename TypeMethodParamTraits< R >::const_pointer
        SIXTRL_RESTRICT SIXTRL_UNUSED( dummy ) = SIXTRL_NULLPTR )
    {
        return static_cast< typename TypeMethodParamTraits< R >::value_type >(
            SIXTRL_PHYS_CONST_EPSILON0 );
    }

    template< class R >
    SIXTRL_STATIC SIXTRL_INLINE SIXTRL_FN typename TypeMethodParamTraits<
        R >::value_type
    PhysConst_mu0( typename TypeMethodParamTraits< R >::const_pointer
        SIXTRL_RESTRICT SIXTRL_UNUSED( dummy ) = SIXTRL_NULLPTR )
    {
        return static_cast< typename TypeMethodParamTraits< R >::value_type >(
            SIXTRL_PHYS_CONST_MU0 );
    }

    template< class R >
    SIXTRL_STATIC SIXTRL_INLINE SIXTRL_FN typename TypeMethodParamTraits<
        R >::value_type
    PhysConst_mass_atomic_si( typename TypeMethodParamTraits< R >::const_pointer
        SIXTRL_RESTRICT SIXTRL_UNUSED( dummy ) = SIXTRL_NULLPTR )
    {
        return static_cast< typename TypeMethodParamTraits< R >::value_type >(
            SIXTRL_PHYS_CONST_MASS_ATOMIC_SI );
    }

    template< class R >
    SIXTRL_STATIC SIXTRL_INLINE SIXTRL_FN typename TypeMethodParamTraits<
        R >::value_type
    PhysConst_mass_atomic_ev( typename TypeMethodParamTraits< R >::const_pointer
        SIXTRL_RESTRICT SIXTRL_UNUSED( dummy ) = SIXTRL_NULLPTR )
    {
        return static_cast< typename TypeMethodParamTraits< R >::value_type >(
            SIXTRL_PHYS_CONST_MASS_ATOMIC_EV );
    }

    template< class R >
    SIXTRL_STATIC SIXTRL_INLINE SIXTRL_FN typename TypeMethodParamTraits<
        R >::value_type
    PhysConst_mass_proton_si( typename TypeMethodParamTraits< R >::const_pointer
        SIXTRL_RESTRICT SIXTRL_UNUSED( dummy ) = SIXTRL_NULLPTR )
    {
        return static_cast< typename TypeMethodParamTraits< R >::value_type >(
            SIXTRL_PHYS_CONST_MASS_PROTON_SI );
    }

    template< class R >
    SIXTRL_STATIC SIXTRL_INLINE SIXTRL_FN typename TypeMethodParamTraits<
        R >::value_type
    PhysConst_mass_proton_ev( typename TypeMethodParamTraits< R >::const_pointer
        SIXTRL_RESTRICT SIXTRL_UNUSED( dummy ) = SIXTRL_NULLPTR )
    {
        return static_cast< typename TypeMethodParamTraits< R >::value_type >(
            SIXTRL_PHYS_CONST_MASS_PROTON_EV );
    }

    template< class R >
    SIXTRL_STATIC SIXTRL_INLINE SIXTRL_FN typename TypeMethodParamTraits<
        R >::value_type
    PhysConst_mass_electron_si( typename TypeMethodParamTraits<
        R >::const_pointer SIXTRL_RESTRICT
            SIXTRL_UNUSED( dummy ) = SIXTRL_NULLPTR )
    {
        return static_cast< typename TypeMethodParamTraits< R >::value_type >(
            SIXTRL_PHYS_CONST_MASS_ELECTRON_SI );
    }

    template< class R >
    SIXTRL_STATIC SIXTRL_INLINE  SIXTRL_FN typename TypeMethodParamTraits<
        R >::value_type
    PhysConst_mass_electron_ev( typename TypeMethodParamTraits<
        R >::const_pointer SIXTRL_RESTRICT
            SIXTRL_UNUSED( dummy ) = SIXTRL_NULLPTR )
    {
        return static_cast< typename TypeMethodParamTraits< R >::value_type >(
            SIXTRL_PHYS_CONST_MASS_ELECTRON_EV );
    }

    template< class R >
    SIXTRL_STATIC SIXTRL_INLINE SIXTRL_FN typename TypeMethodParamTraits<
        R >::value_type
    PhysConst_charge0_si( typename TypeMethodParamTraits< R >::const_pointer
        SIXTRL_RESTRICT SIXTRL_UNUSED( dummy ) = SIXTRL_NULLPTR )
    {
        return static_cast< typename TypeMethodParamTraits< R >::value_type >(
            SIXTRL_PHYS_CONST_CHARGE0_SI );
    }

    template< class R >
    SIXTRL_STATIC SIXTRL_INLINE SIXTRL_FN typename TypeMethodParamTraits<
        R >::value_type
    PhysConst_charge0( typename TypeMethodParamTraits< R >::const_pointer
        SIXTRL_RESTRICT SIXTRL_UNUSED( dummy ) = SIXTRL_NULLPTR )
    {
        return static_cast< typename TypeMethodParamTraits< R >::value_type >(
            SIXTRL_PHYS_CONST_CHARGE0 );
    }
}


template< class R >
SIXTRL_STATIC SIXTRL_INLINE SIXTRL_FN
typename SIXTRL_CXX_NAMESPACE::TypeMethodParamTraits< R >::value_type
NS(PhysConst_clight)( typename SIXTRL_CXX_NAMESPACE::TypeMethodParamTraits<
    R >::const_pointer SIXTRL_RESTRICT dummy = SIXTRL_NULLPTR )
{
    return SIXTRL_CXX_NAMESPACE::PhysConst_clight< R >( dummy );
}

template< class R >
SIXTRL_STATIC SIXTRL_INLINE SIXTRL_FN
typename SIXTRL_CXX_NAMESPACE::TypeMethodParamTraits< R >::value_type
NS(PhysConst_epsilon0)( typename SIXTRL_CXX_NAMESPACE::TypeMethodParamTraits<
    R >::const_pointer SIXTRL_RESTRICT dummy = SIXTRL_NULLPTR )
{
    return SIXTRL_CXX_NAMESPACE::PhysConst_epsilon0< R >( dummy );
}

template< class R >
SIXTRL_STATIC SIXTRL_INLINE SIXTRL_FN
typename SIXTRL_CXX_NAMESPACE::TypeMethodParamTraits< R >::value_type
NS(PhysConst_mu0)( typename SIXTRL_CXX_NAMESPACE::TypeMethodParamTraits<
    R >::const_pointer SIXTRL_RESTRICT dummy = SIXTRL_NULLPTR )
{
    return SIXTRL_CXX_NAMESPACE::PhysConst_mu0< R >( dummy );
}

template< class R >
SIXTRL_STATIC SIXTRL_INLINE SIXTRL_FN
typename SIXTRL_CXX_NAMESPACE::TypeMethodParamTraits< R >::value_type
NS(PhysConst_mass_atomic_si)( typename
    SIXTRL_CXX_NAMESPACE::TypeMethodParamTraits< R >::const_pointer
        SIXTRL_RESTRICT dummy = SIXTRL_NULLPTR )
{
    return SIXTRL_CXX_NAMESPACE::PhysConst_mass_atomic_si< R >( dummy );
}

template< class R >
SIXTRL_STATIC SIXTRL_INLINE SIXTRL_FN
typename SIXTRL_CXX_NAMESPACE::TypeMethodParamTraits< R >::value_type
NS(PhysConst_mass_atomic_ev)( typename
    SIXTRL_CXX_NAMESPACE::TypeMethodParamTraits< R >::const_pointer
        SIXTRL_RESTRICT dummy = SIXTRL_NULLPTR )
{
    return SIXTRL_CXX_NAMESPACE::PhysConst_mass_atomic_ev< R >( dummy );
}

template< class R >
SIXTRL_STATIC SIXTRL_INLINE SIXTRL_FN
typename SIXTRL_CXX_NAMESPACE::TypeMethodParamTraits< R >::value_type
NS(PhysConst_mass_proton_si)( typename
    SIXTRL_CXX_NAMESPACE::TypeMethodParamTraits< R >::const_pointer
        SIXTRL_RESTRICT dummy = SIXTRL_NULLPTR )
{
    return SIXTRL_CXX_NAMESPACE::PhysConst_mass_proton_si< R >( dummy );
}

template< class R >
SIXTRL_STATIC SIXTRL_INLINE SIXTRL_FN
typename SIXTRL_CXX_NAMESPACE::TypeMethodParamTraits< R >::value_type
NS(PhysConst_mass_proton_ev)( typename
    SIXTRL_CXX_NAMESPACE::TypeMethodParamTraits< R >::const_pointer
        SIXTRL_RESTRICT dummy = SIXTRL_NULLPTR )
{
    return SIXTRL_CXX_NAMESPACE::PhysConst_mass_proton_ev< R >( dummy );
}

template< class R >
SIXTRL_STATIC SIXTRL_INLINE SIXTRL_FN
typename SIXTRL_CXX_NAMESPACE::TypeMethodParamTraits< R >::value_type
NS(PhysConst_mass_electron_si)( typename
    SIXTRL_CXX_NAMESPACE::TypeMethodParamTraits< R >::const_pointer
        SIXTRL_RESTRICT dummy = SIXTRL_NULLPTR )
{
    return SIXTRL_CXX_NAMESPACE::PhysConst_mass_electron_si< R >( dummy );
}

template< class R >
SIXTRL_STATIC SIXTRL_INLINE SIXTRL_FN
typename SIXTRL_CXX_NAMESPACE::TypeMethodParamTraits< R >::value_type
NS(PhysConst_mass_electron_ev)( typename
    SIXTRL_CXX_NAMESPACE::TypeMethodParamTraits< R >::const_pointer
        SIXTRL_RESTRICT dummy = SIXTRL_NULLPTR )
{
    return SIXTRL_CXX_NAMESPACE::PhysConst_mass_electron_ev< R >( dummy );
}

template< class R >
SIXTRL_STATIC SIXTRL_INLINE SIXTRL_FN
typename SIXTRL_CXX_NAMESPACE::TypeMethodParamTraits< R >::value_type
NS(PhysConst_charge0_si)( typename SIXTRL_CXX_NAMESPACE::TypeMethodParamTraits<
    R >::const_pointer SIXTRL_RESTRICT dummy = SIXTRL_NULLPTR )
{
    return SIXTRL_CXX_NAMESPACE::PhysConst_charge0_si< R >( dummy );
}

template< class R >
SIXTRL_STATIC SIXTRL_INLINE SIXTRL_FN
typename SIXTRL_CXX_NAMESPACE::TypeMethodParamTraits< R >::value_type
NS(PhysConst_charge0)( typename SIXTRL_CXX_NAMESPACE::TypeMethodParamTraits<
    R >::const_pointer SIXTRL_RESTRICT dummy = SIXTRL_NULLPTR )
{
    return SIXTRL_CXX_NAMESPACE::PhysConst_charge0< R >( dummy );
}

#endif

#if defined( __cplusplus ) && !defined( _GPUCODE )
extern "C" {
#endif /* defined( __cplusplus ) && !defined( _GPUCODE ) */

SIXTRL_STATIC SIXTRL_FN SIXTRL_REAL_T NS(PhysConst_clight)(
    void ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_REAL_T NS(PhysConst_epsilon0)(
    void )  SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_REAL_T NS(PhysConst_mu0)(
    void ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_REAL_T NS(PhysConst_charge0_si)(
    void ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_REAL_T NS(PhysConst_charge0)(
    void ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_REAL_T NS(PhysConst_mass_atomic_si)(
    void ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_REAL_T NS(PhysConst_mass_atomic_ev)(
    void ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_REAL_T NS(PhysConst_mass_proton_si)(
    void ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_REAL_T NS(PhysConst_mass_proton_ev)(
    void ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_REAL_T NS(PhysConst_mass_electron_si)(
    void ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_REAL_T NS(PhysConst_mass_electron_ev)(
    void ) SIXTRL_NOEXCEPT;

#if defined( __cplusplus ) && !defined( _GPUCODE )
}
#endif /* C++, Host */

/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */
/* !!!!!!!        Inline Methods and Functions Implementations       !!!!!!!! */
/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */

#if defined( __cplusplus ) && !defined( _GPUCODE )
extern "C" {
#endif /* defined( __cplusplus ) && !defined( _GPUCODE ) */

SIXTRL_INLINE SIXTRL_REAL_T NS(PhysConst_clight)( void ) SIXTRL_NOEXCEPT
{
    return ( SIXTRL_REAL_T )SIXTRL_PHYS_CONST_CLIGHT;
}

SIXTRL_INLINE SIXTRL_REAL_T NS(PhysConst_epsilon0)( void )  SIXTRL_NOEXCEPT
{
    return ( SIXTRL_REAL_T )SIXTRL_PHYS_CONST_EPSILON0;
}

SIXTRL_INLINE SIXTRL_REAL_T NS(PhysConst_mu0)( void ) SIXTRL_NOEXCEPT
{
    return ( SIXTRL_REAL_T )SIXTRL_PHYS_CONST_MU0;
}

SIXTRL_INLINE SIXTRL_REAL_T NS(PhysConst_charge0_si)( void ) SIXTRL_NOEXCEPT
{
    return ( SIXTRL_REAL_T )SIXTRL_PHYS_CONST_CHARGE0_SI;
}

SIXTRL_INLINE SIXTRL_REAL_T NS(PhysConst_charge0)( void ) SIXTRL_NOEXCEPT
{
    return ( SIXTRL_REAL_T )SIXTRL_PHYS_CONST_CHARGE0;
}

SIXTRL_INLINE SIXTRL_REAL_T NS(PhysConst_mass_atomic_si)( void ) SIXTRL_NOEXCEPT
{
    return ( SIXTRL_REAL_T )SIXTRL_PHYS_CONST_MASS_ATOMIC_SI;
}

SIXTRL_INLINE SIXTRL_REAL_T NS(PhysConst_mass_atomic_ev)( void ) SIXTRL_NOEXCEPT
{
    return ( SIXTRL_REAL_T )SIXTRL_PHYS_CONST_MASS_ATOMIC_EV;
}

SIXTRL_INLINE SIXTRL_REAL_T NS(PhysConst_mass_proton_si)( void ) SIXTRL_NOEXCEPT
{
    return ( SIXTRL_REAL_T )SIXTRL_PHYS_CONST_MASS_PROTON_SI;
}

SIXTRL_INLINE SIXTRL_REAL_T NS(PhysConst_mass_proton_ev)( void ) SIXTRL_NOEXCEPT
{
    return ( SIXTRL_REAL_T )SIXTRL_PHYS_CONST_MASS_PROTON_EV;
}

SIXTRL_INLINE SIXTRL_REAL_T NS(PhysConst_mass_electron_si)( void ) SIXTRL_NOEXCEPT
{
    return ( SIXTRL_REAL_T )SIXTRL_PHYS_CONST_MASS_ELECTRON_SI;
}

SIXTRL_INLINE SIXTRL_REAL_T NS(PhysConst_mass_electron_ev)( void ) SIXTRL_NOEXCEPT
{
    return ( SIXTRL_REAL_T )SIXTRL_PHYS_CONST_MASS_ELECTRON_EV;
}

#if defined( __cplusplus ) && !defined( _GPUCODE )
}
#endif /* C++, Host */

#endif /* SIXTRACKLIB_COMMON_INTERNAL_PHYSICS_CONSTANTS_H__ */
