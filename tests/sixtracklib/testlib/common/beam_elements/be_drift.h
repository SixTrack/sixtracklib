#ifndef SIXTRACKLIB_TESTLIB_COMMON_BEAM_ELEMENTS_BE_DRIFT_C99_H__
#define SIXTRACKLIB_TESTLIB_COMMON_BEAM_ELEMENTS_BE_DRIFT_C99_H__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stddef.h>
    #include <stdlib.h>

    #if !defined( _GPUCODE )
    #include <stdio.h>
    #endif /* !defined( _GPUCODE ) */

#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/be_drift/be_drift.h"
    #include "sixtracklib/common/buffer/buffer_type.h"
    #include "sixtracklib/common/internal/type_comparison_helpers.hpp"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined(  _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

SIXTRL_STATIC SIXTRL_FN int NS(Drift_compare_values)(
    SIXTRL_BE_ARGPTR_DEC const NS(Drift) *const SIXTRL_RESTRICT lhs,
    SIXTRL_BE_ARGPTR_DEC const NS(Drift) *const SIXTRL_RESTRICT rhs
) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN int NS(Drift_compare_values_with_treshold)(
    SIXTRL_BE_ARGPTR_DEC const NS(Drift) *const SIXTRL_RESTRICT lhs,
    SIXTRL_BE_ARGPTR_DEC const NS(Drift) *const SIXTRL_RESTRICT rhs,
    NS(drift_real_t) const treshold ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN void NS(Drift_print_out)(
    SIXTRL_BE_ARGPTR_DEC const NS(Drift) *const SIXTRL_RESTRICT elem );

#if !defined( _GPUCODE )

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(Drift_print)(
    SIXTRL_ARGPTR_DEC FILE* SIXTRL_RESTRICT fp,
    SIXTRL_BE_ARGPTR_DEC const NS(Drift) *const SIXTRL_RESTRICT elem );

#endif /* !defined( _GPUCODE ) */

/* ------------------------------------------------------------------------- */

SIXTRL_STATIC SIXTRL_FN int NS(DriftExact_compare_values)(
    SIXTRL_BE_ARGPTR_DEC const NS(DriftExact) *const SIXTRL_RESTRICT lhs,
    SIXTRL_BE_ARGPTR_DEC const NS(DriftExact) *const SIXTRL_RESTRICT rhs
) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN int NS(DriftExact_compare_values_with_treshold)(
    SIXTRL_BE_ARGPTR_DEC const NS(DriftExact) *const SIXTRL_RESTRICT lhs,
    SIXTRL_BE_ARGPTR_DEC const NS(DriftExact) *const SIXTRL_RESTRICT rhs,
    NS(drift_real_t) const treshold ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN void NS(DriftExact_print_out)(
    SIXTRL_BE_ARGPTR_DEC const NS(DriftExact) *const SIXTRL_RESTRICT elem );

#if !defined( _GPUCODE )

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(DriftExact_print)(
    SIXTRL_ARGPTR_DEC FILE* SIXTRL_RESTRICT fp,
    SIXTRL_BE_ARGPTR_DEC const NS(DriftExact) *const SIXTRL_RESTRICT elem );

#endif /* !defined( _GPUCODE ) */

#if !defined(  _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */
/* !!!!!!!!!!  Implementation of inline functions for NS(Drift)  !!!!!!!!!!! */
/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */

#if !defined(  _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

SIXTRL_INLINE int NS(Drift_compare_values)(
    SIXTRL_BE_ARGPTR_DEC const NS(Drift) *const SIXTRL_RESTRICT lhs,
    SIXTRL_BE_ARGPTR_DEC const NS(Drift) *const SIXTRL_RESTRICT
        rhs ) SIXTRL_NOEXCEPT
{
    int cmp_result = -1;

    if( ( lhs != SIXTRL_NULLPTR ) && ( rhs != SIXTRL_NULLPTR ) )
    {
        cmp_result = ( lhs != rhs )
            ? NS(Type_value_comp_result)( lhs->length, rhs->length ) : 0;
    }
    else if( rhs != SIXTRL_NULLPTR )
    {
        cmp_result = +1;
    }

    return cmp_result;
}

SIXTRL_INLINE int NS(Drift_compare_values_with_treshold)(
    SIXTRL_BE_ARGPTR_DEC const NS(Drift) *const SIXTRL_RESTRICT lhs,
    SIXTRL_BE_ARGPTR_DEC const NS(Drift) *const SIXTRL_RESTRICT rhs,
    NS(drift_real_t) const treshold ) SIXTRL_NOEXCEPT
{
    int cmp_result = -1;

    if( ( lhs != SIXTRL_NULLPTR ) && ( rhs != SIXTRL_NULLPTR ) )
    {
        cmp_result = ( lhs != rhs )
            ? NS(Type_value_comp_result_with_tolerances)(
                lhs->length, rhs->length, ( NS(drift_real_t) )0, treshold )
            : 0;
    }
    else if( rhs != SIXTRL_NULLPTR )
    {
        cmp_result = +1;
    }

    return cmp_result;
}

SIXTRL_INLINE void NS(Drift_print_out)( SIXTRL_BE_ARGPTR_DEC const
    NS(Drift) *const SIXTRL_RESTRICT elem )
{
    if( elem != SIXTRL_NULLPTR )
    {
        printf( "drift             | length   = %+20.18f m;\r\n",
                NS(Drift_length)( elem ) );
    }
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE int NS(DriftExact_compare_values)(
    SIXTRL_BE_ARGPTR_DEC const NS(DriftExact) *const SIXTRL_RESTRICT lhs,
    SIXTRL_BE_ARGPTR_DEC const NS(DriftExact) *const SIXTRL_RESTRICT
        rhs ) SIXTRL_NOEXCEPT
{
    int cmp_result = -1;

    if( ( lhs != SIXTRL_NULLPTR ) && ( rhs != SIXTRL_NULLPTR ) )
    {
        cmp_result = ( lhs != rhs )
            ? NS(Type_value_comp_result)( lhs->length, rhs->length ) : 0;
    }
    else if( rhs != SIXTRL_NULLPTR )
    {
        cmp_result = +1;
    }

    return cmp_result;
}

SIXTRL_INLINE int NS(DriftExact_compare_values_with_treshold)(
    SIXTRL_BE_ARGPTR_DEC const NS(DriftExact) *const SIXTRL_RESTRICT lhs,
    SIXTRL_BE_ARGPTR_DEC const NS(DriftExact) *const SIXTRL_RESTRICT rhs,
    NS(drift_real_t) const treshold ) SIXTRL_NOEXCEPT
{
    int cmp_result = -1;

    if( ( lhs != SIXTRL_NULLPTR ) && ( rhs != SIXTRL_NULLPTR ) )
    {
        cmp_result = ( lhs != rhs )
            ? NS(Type_value_comp_result_with_tolerances)(
                lhs->length, rhs->length, ( NS(drift_real_t) )0, treshold )
            : 0;
    }
    else if( rhs != SIXTRL_NULLPTR )
    {
        cmp_result = +1;
    }

    return cmp_result;
}

SIXTRL_INLINE void NS(DriftExact_print_out)( SIXTRL_BE_ARGPTR_DEC const
    NS(DriftExact) *const SIXTRL_RESTRICT elem )
{
    if( elem != SIXTRL_NULLPTR )
    {
        printf( "exact drift       | length   = %+20.18f m;\r\n",
                NS(DriftExact_length)( elem ) );
    }
}

#if !defined(  _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */
#endif /* SIXTRACKLIB_TESTLIB_COMMON_BEAM_ELEMENTS_BE_DRIFT_C99_H__ */
