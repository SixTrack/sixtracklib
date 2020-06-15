#ifndef SIXTRACKLIB_COMMON_INTERNAL_MATH_INTERPOL_H__
#define SIXTRACKLIB_COMMON_INTERNAL_MATH_INTERPOL_H__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDE )
    #include <stddef.h>
    #include <stdint.h>
    #include <stdlib.h>
    #include <math.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDE ) */

#if !defined( SIXTRL_NO_INCLUDE )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/control/definitions.h"
    #include "sixtracklib/common/buffer/managed_buffer_minimal.h"
    #include "sixtracklib/common/internal/math_functions.h"
#endif /* !defined( SIXTRL_NO_INCLUDE ) */

#if defined( __cplusplus ) && !defined( _GPUCODE )
extern "C" {
#endif /* C++, Host */

typedef SIXTRL_INT64_T NS(math_abscissa_idx_t);

SIXTRL_STATIC SIXTRL_FN NS(math_abscissa_idx_t) NS(Math_abscissa_index_equ)(
    SIXTRL_REAL_T const x_value, SIXTRL_REAL_T const x0,
    SIXTRL_REAL_T const dx,
    NS(math_abscissa_idx_t) const num_x_values ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(math_abscissa_idx_t) NS(Math_abscissa_index_equ_ex)(
    SIXTRL_REAL_T const x_value, SIXTRL_REAL_T const x0,
    SIXTRL_REAL_T const dx,
    NS(math_abscissa_idx_t) const num_x_values ) SIXTRL_NOEXCEPT;

/* ------------------------------------------------------------------------- */

typedef enum
{
    NS(MATH_INTERPOL_LINEAR) = 0,
    NS(MATH_INTERPOL_CUBIC) = 1,
    NS(MATH_INTERPOL_NONE)  = 255
}
NS(math_interpol_t);

typedef enum
{
    NS(MATH_INTERPOL_LINEAR_BOUNDARY_NONE)        = 0,
    NS(MATH_INTERPOL_LINEAR_BOUNDARY_DEFAULT)     = 0,
    NS(MATH_INTERPOL_CUBIC_BOUNDARY_NATURAL)      = 1,
    NS(MATH_INTERPOL_CUBIC_BOUNDARY_CLAMPED)      = 2,
    NS(MATH_INTERPOL_CUBIC_BOUNDARY_PROPORTIONAL) = 3,
    NS(MATH_INTERPOL_CUBIC_BOUNDARY_NOT_A_KNOT)   = 4,
    NS(MATH_INTERPOL_CUBIC_BOUNDARY_DEFAULT)      = 1
}
NS(math_interpol_boundary_t);

SIXTRL_STATIC SIXTRL_FN SIXTRL_REAL_T
NS(Math_interpol_boundary_begin_default_param)(
    NS(math_interpol_boundary_t) const boundary_type ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_REAL_T
NS(Math_interpol_boundary_end_default_param)(
    NS(math_interpol_boundary_t) const boundary_type ) SIXTRL_NOEXCEPT;

/* ------------------------------------------------------------------------- */

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(Math_interpol_linear_prepare_equ)(
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT yp_begin,
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT y_begin,
    SIXTRL_REAL_T const x0, SIXTRL_REAL_T const dx,
    NS(math_abscissa_idx_t) const num_values,
    NS(math_interpol_boundary_t) const begin_boundary_type,
    NS(math_interpol_boundary_t) const end_boundary_type,
    SIXTRL_REAL_T const begin_boundary_param,
    SIXTRL_REAL_T const end_boundary_param ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(Math_interpol_cubic_prepare_equ)(
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT yp_begin,
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT temp_values_begin,
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT y_begin,
    SIXTRL_REAL_T const x0, SIXTRL_REAL_T const dx,
    NS(math_abscissa_idx_t) const num_values,
    NS(math_interpol_boundary_t) const begin_boundary_type,
    NS(math_interpol_boundary_t) const end_boundary_type,
    SIXTRL_REAL_T const begin_boundary_param,
    SIXTRL_REAL_T const end_boundary_param ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(Math_interpol_prepare_equ)(
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT derivatives_begin,
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT temp_values_begin,
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT y_begin,
    SIXTRL_REAL_T const x0, SIXTRL_REAL_T const dx,
    NS(math_abscissa_idx_t) const num_values,
    NS(math_interpol_t) const interpol_type ) SIXTRL_NOEXCEPT;

/* ------------------------------------------------------------------------- */

SIXTRL_STATIC SIXTRL_FN SIXTRL_REAL_T NS(Math_interpol_linear_y_equ)(
    SIXTRL_REAL_T const x,
    SIXTRL_REAL_T const x0, SIXTRL_REAL_T const dx,
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT y_begin,
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT yp_begin,
    NS(math_abscissa_idx_t) const num_values ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_REAL_T NS(Math_interpol_linear_yp_equ)(
    SIXTRL_REAL_T const x,
    SIXTRL_REAL_T const x0, SIXTRL_REAL_T const dx,
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT y_begin,
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT yp_begin,
    NS(math_abscissa_idx_t) const num_values ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_REAL_T NS(Math_interpol_linear_ypp_equ)(
    SIXTRL_REAL_T const x,
    SIXTRL_REAL_T const x0, SIXTRL_REAL_T const dx,
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT y_begin,
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT yp_begin,
    NS(math_abscissa_idx_t) const num_values ) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC SIXTRL_FN SIXTRL_REAL_T NS(Math_interpol_linear_y_equ_ex)(
    SIXTRL_REAL_T const x,
    SIXTRL_REAL_T const x0, SIXTRL_REAL_T const dx,
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT y_begin,
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT yp_begin,
    NS(math_abscissa_idx_t) const num_values ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_REAL_T NS(Math_interpol_linear_yp_equ_ex)(
    SIXTRL_REAL_T const x,
    SIXTRL_REAL_T const x0, SIXTRL_REAL_T const dx,
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT y_begin,
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT yp_begin,
    NS(math_abscissa_idx_t) const num_values ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_REAL_T NS(Math_interpol_linear_ypp_equ_ex)(
    SIXTRL_REAL_T const x,
    SIXTRL_REAL_T const x0, SIXTRL_REAL_T const dx,
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT y_begin,
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT yp_begin,
    NS(math_abscissa_idx_t) const num_values ) SIXTRL_NOEXCEPT;

/* ------------------------------------------------------------------------- */

SIXTRL_STATIC SIXTRL_FN SIXTRL_REAL_T NS(Math_interpol_cubic_y_equ)(
    SIXTRL_REAL_T const x,
    SIXTRL_REAL_T const x0, SIXTRL_REAL_T const dx,
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT y_begin,
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT yp_begin,
    NS(math_abscissa_idx_t) const num_values ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_REAL_T NS(Math_interpol_cubic_yp_equ)(
    SIXTRL_REAL_T const x,
    SIXTRL_REAL_T const x0, SIXTRL_REAL_T const dx,
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT y_begin,
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT yp_begin,
    NS(math_abscissa_idx_t) const num_values ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_REAL_T NS(Math_interpol_cubic_ypp_equ)(
    SIXTRL_REAL_T const x,
    SIXTRL_REAL_T const x0, SIXTRL_REAL_T const dx,
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT y_begin,
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT yp_begin,
    NS(math_abscissa_idx_t) const num_values ) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC SIXTRL_FN SIXTRL_REAL_T NS(Math_interpol_cubic_y_equ_ex)(
    SIXTRL_REAL_T const x,
    SIXTRL_REAL_T const x0, SIXTRL_REAL_T const dx,
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT y_begin,
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT yp_begin,
    NS(math_abscissa_idx_t) const num_values ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_REAL_T NS(Math_interpol_cubic_yp_equ_ex)(
    SIXTRL_REAL_T const x,
    SIXTRL_REAL_T const x0, SIXTRL_REAL_T const dx,
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT y_begin,
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT yp_begin,
    NS(math_abscissa_idx_t) const num_values ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_REAL_T NS(Math_interpol_cubic_ypp_equ_ex)(
    SIXTRL_REAL_T const x,
    SIXTRL_REAL_T const x0, SIXTRL_REAL_T const dx,
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT y_begin,
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT yp_begin,
    NS(math_abscissa_idx_t) const num_values ) SIXTRL_NOEXCEPT;

/* ------------------------------------------------------------------------- */

SIXTRL_STATIC SIXTRL_FN SIXTRL_REAL_T NS(Math_interpol_y_equ)(
    SIXTRL_REAL_T const x,
    SIXTRL_REAL_T const x0, SIXTRL_REAL_T const dx,
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT y,
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT derivatives,
    NS(math_abscissa_idx_t) const num_values,
    NS(math_interpol_t) const interpol_type ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_REAL_T NS(Math_interpol_yp_equ)(
    SIXTRL_REAL_T const x,
    SIXTRL_REAL_T const x0, SIXTRL_REAL_T const dx,
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT y,
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT derivatives,
    NS(math_abscissa_idx_t) const num_values,
    NS(math_interpol_t) const interpol_type ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_REAL_T NS(Math_interpol_ypp_equ)(
    SIXTRL_REAL_T const x,
    SIXTRL_REAL_T const x0, SIXTRL_REAL_T const dx,
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT y,
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT derivatives,
    NS(math_abscissa_idx_t) const num_values,
    NS(math_interpol_t) const interpol_type ) SIXTRL_NOEXCEPT;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC SIXTRL_FN SIXTRL_REAL_T NS(Math_interpol_y_equ_ex)(
    SIXTRL_REAL_T const x,
    SIXTRL_REAL_T const x0, SIXTRL_REAL_T const dx,
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT y,
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT derivatives,
    NS(math_abscissa_idx_t) const num_values,
    NS(math_interpol_t) const interpol_type ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_REAL_T NS(Math_interpol_yp_equ_ex)(
    SIXTRL_REAL_T const x,
    SIXTRL_REAL_T const x0, SIXTRL_REAL_T const dx,
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT y,
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT derivatives,
    NS(math_abscissa_idx_t) const num_values,
    NS(math_interpol_t) const interpol_type ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_REAL_T NS(Math_interpol_ypp_equ_ex)(
    SIXTRL_REAL_T const x,
    SIXTRL_REAL_T const x0, SIXTRL_REAL_T const dx,
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT y,
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT derivatives,
    NS(math_abscissa_idx_t) const num_values,
    NS(math_interpol_t) const interpol_type ) SIXTRL_NOEXCEPT;

#if defined( __cplusplus ) && !defined( _GPUCODE )
}
#endif /* C++, Host */


/* ************************************************************************* */
/* ************************************************************************* */

#if defined( __cplusplus ) && !defined( _GPUCODE )
extern "C" {
#endif /* C++, Host */

SIXTRL_INLINE NS(math_abscissa_idx_t) NS(Math_abscissa_index_equ)(
    SIXTRL_REAL_T const x_value, SIXTRL_REAL_T const x0,
    SIXTRL_REAL_T const dx,
    NS(math_abscissa_idx_t) const num_x_values ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( !isinf( dx ) && !isnan( dx ) );
    SIXTRL_ASSERT( !isinf( x_value ) && !isnan( x_value ) );
    SIXTRL_ASSERT( !isinf( x0 ) && !isnan( x0 ) );
    SIXTRL_ASSERT( num_x_values >= ( NS(math_abscissa_idx_t) )0 );
    SIXTRL_ASSERT( dx > ( SIXTRL_REAL_T )0 );
    SIXTRL_ASSERT( x_value >= x0 );
    SIXTRL_ASSERT( x_value <= ( x0 + dx * ( SIXTRL_REAL_T )num_x_values ) );
    ( void )num_x_values;

    return ( NS(math_abscissa_idx_t) )( ( x_value - x0 ) / dx );
}

SIXTRL_INLINE NS(math_abscissa_idx_t) NS(Math_abscissa_index_equ_ex)(
    SIXTRL_REAL_T const x_value, SIXTRL_REAL_T const x0,
    SIXTRL_REAL_T const dx,
    NS(math_abscissa_idx_t) const num_x_values ) SIXTRL_NOEXCEPT
{
    SIXTRL_REAL_T const max_x = x0 + dx * ( SIXTRL_REAL_T )num_x_values;

    SIXTRL_ASSERT( !isinf( dx ) && !isnan( dx ) );
    SIXTRL_ASSERT( !isinf( x_value ) && !isnan( x_value ) );
    SIXTRL_ASSERT( !isinf( x0 ) && !isnan( x0 ) );
    SIXTRL_ASSERT( num_x_values >= ( NS(math_abscissa_idx_t) )0 );
    SIXTRL_ASSERT( dx > ( SIXTRL_REAL_T )0 );

    return ( x_value >= x0 )
        ? ( ( x_value < max_x )
                ? NS(Math_abscissa_index_equ)( x_value, x0, dx, num_x_values )
                : num_x_values )
        : ( NS(math_abscissa_idx_t) )-1;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE SIXTRL_REAL_T NS(Math_interpol_boundary_begin_default_param)(
    NS(math_interpol_boundary_t) const boundary_type ) SIXTRL_NOEXCEPT
{
    SIXTRL_REAL_T param = ( SIXTRL_REAL_T )0;

    switch( boundary_type )
    {
        case NS(MATH_INTERPOL_CUBIC_BOUNDARY_PROPORTIONAL):
        {
            param = ( SIXTRL_REAL_T )1;
            break;
        }

        default:
        {
            param = ( SIXTRL_REAL_T )0;
        }
    }

    return param;
}

SIXTRL_INLINE SIXTRL_REAL_T NS(Math_interpol_boundary_end_default_param)(
    NS(math_interpol_boundary_t) const boundary_type ) SIXTRL_NOEXCEPT
{
    return NS(Math_interpol_boundary_begin_default_param)( boundary_type );
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE NS(arch_status_t) NS(Math_interpol_linear_prepare_equ)(
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT yp,
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT y,
    SIXTRL_REAL_T const x0, SIXTRL_REAL_T const dx,
    NS(math_abscissa_idx_t) const num_values,
    NS(math_interpol_boundary_t) const begin_boundary_type,
    NS(math_interpol_boundary_t) const end_boundary_type,
    SIXTRL_REAL_T const begin_boundary_param,
    SIXTRL_REAL_T const end_boundary_param ) SIXTRL_NOEXCEPT
{
    NS(arch_status_t) status = ( NS(arch_status_t)
        )SIXTRL_ARCH_STATUS_GENERAL_FAILURE;

    ( void )begin_boundary_type;
    ( void )begin_boundary_param;

    ( void )end_boundary_type;
    ( void )end_boundary_param;

    if( ( yp != SIXTRL_NULLPTR ) && ( y != SIXTRL_NULLPTR ) &&
        ( yp != y ) && ( !isnan( x0 ) ) && ( !isinf( x0 ) ) &&
        ( !isnan( dx ) ) && ( !isinf( dx ) ) && ( dx > ( SIXTRL_REAL_T )0 ) &&
        ( num_values > ( NS(math_abscissa_idx_t) )1 ) )
    {
        NS(math_abscissa_idx_t) const nn = num_values - 1;
        NS(math_abscissa_idx_t) ii = ( NS(math_abscissa_idx_t) )0;

        SIXTRL_REAL_T y_ii_plus_1 = y[ 0 ];
        SIXTRL_ASSERT( !isnan( y_ii_plus_1 ) );
        SIXTRL_ASSERT( !isinf( y_ii_plus_1 ) );

        for( ; ii < nn ; ++ii )
        {
            SIXTRL_REAL_T const y_ii = y_ii_plus_1;
            y_ii_plus_1 = y[ ii + 1 ];
            SIXTRL_ASSERT( !isnan( y_ii_plus_1 ) );
            SIXTRL_ASSERT( !isinf( y_ii_plus_1 ) );

            yp[ ii ] = ( y_ii_plus_1 - y_ii ) / dx;
        }

        status = ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;
    }

    return status;
}

SIXTRL_INLINE NS(arch_status_t) NS(Math_interpol_cubic_prepare_equ)(
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT ypp,
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT temp_values,
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT y,
    SIXTRL_REAL_T const x0, SIXTRL_REAL_T const dx,
    NS(math_abscissa_idx_t) const num_values,
    NS(math_interpol_boundary_t) const begin_boundary_type,
    NS(math_interpol_boundary_t) const end_boundary_type,
    SIXTRL_REAL_T const begin_boundary_param,
    SIXTRL_REAL_T const end_boundary_param ) SIXTRL_NOEXCEPT
{
    NS(arch_status_t) status = ( NS(arch_status_t)
        )SIXTRL_ARCH_STATUS_GENERAL_FAILURE;

    if( ( ypp != SIXTRL_NULLPTR ) && ( y  != SIXTRL_NULLPTR ) &&
        ( temp_values != SIXTRL_NULLPTR ) &&
        ( ypp != y ) && ( ypp != temp_values ) && ( y != temp_values ) &&
        ( !isnan( x0 ) ) && ( !isinf( x0 ) ) &&
        ( !isnan( dx ) ) && ( !isinf( dx ) ) && ( dx > ( SIXTRL_REAL_T )0 ) &&
        ( num_values > ( NS(math_abscissa_idx_t) )3 ) )
    {
        NS(math_abscissa_idx_t) const n_minus_1 = num_values - 1;
        NS(math_abscissa_idx_t) const n_minus_2 = num_values - 2;
        NS(math_abscissa_idx_t) const n_minus_3 = num_values - 3;

        SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T* l  = temp_values;
        SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T* d  = l  + num_values;
        SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T* h  = d  + num_values;
        SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T* g  = h  + num_values;
        SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T* dy = g  + num_values;

        NS(math_abscissa_idx_t) ii = ( NS(math_abscissa_idx_t) )0;
        status = ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;

        h[  0 ] = dx;
        g[  0 ] = ( SIXTRL_REAL_T )0;
        l[  0 ] = ( SIXTRL_REAL_T )0;
        d[  0 ] = ( SIXTRL_REAL_T )0;
        dy[ 0 ] = ( SIXTRL_REAL_T )0;

        for( ii = 1 ; ii < n_minus_1 ; ++ii )
        {
            h[  ii ]  = dx;
            g[  ii ]  = ( h[ ii - 1 ] + h[ ii ] ) * ( SIXTRL_REAL_T )2;

            SIXTRL_ASSERT( !isnan( y[ ii + 1 ] ) && !isinf( y[ ii + 1 ] ) );
            SIXTRL_ASSERT( !isnan( y[ ii     ] ) && !isinf( y[ ii     ] ) );
            SIXTRL_ASSERT( !isnan( y[ ii - 1 ] ) && !isinf( y[ ii - 1 ] ) );

            dy[ ii ]  = ( y[ ii + 1 ] - y[ ii ] ) / h[ ii ];
            dy[ ii ] -= ( y[ ii ] - y[ ii - 1 ] ) / h[ ii - 1 ];
            dy[ ii ] *= ( SIXTRL_REAL_T )6;
        }

        h[  n_minus_1 ] = dx;
        g[  n_minus_1 ] = ( SIXTRL_REAL_T )0;
        l[  n_minus_1 ] = ( SIXTRL_REAL_T )0;
        d[  n_minus_1 ] = ( SIXTRL_REAL_T )0;
        dy[ n_minus_1 ] = ( SIXTRL_REAL_T )0;

        switch( begin_boundary_type )
        {
            case NS(MATH_INTERPOL_CUBIC_BOUNDARY_NATURAL):
            {
                break;
            };

            case NS(MATH_INTERPOL_CUBIC_BOUNDARY_CLAMPED):
            {
                dy[ 1 ] -= begin_boundary_param * h[ 0 ];
                break;
            }

            case NS(MATH_INTERPOL_CUBIC_BOUNDARY_PROPORTIONAL):
            {
                g[ 1 ]  = ( begin_boundary_param + ( SIXTRL_REAL_T )2 );
                g[ 1 ] *= h[ 0 ];
                g[ 1 ] += h[ 1 ] * ( SIXTRL_REAL_T )2;
                break;
            }

            case NS(MATH_INTERPOL_CUBIC_BOUNDARY_NOT_A_KNOT):
            {
                SIXTRL_REAL_T const d_h = ( h[ 0 ] * h[ 0 ] ) / h[ 1 ];

                g[ 1 ]  = h[ 0 ] * ( SIXTRL_REAL_T )3;
                g[ 1 ] += h[ 1 ] * ( SIXTRL_REAL_T )2;
                g[ 1 ] += d_h;

                h[ 1 ] -= d_h;
                break;
            }

            default:
            {
                status = ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_GENERAL_FAILURE;
            }
        };

        switch( end_boundary_type )
        {
            case NS(MATH_INTERPOL_CUBIC_BOUNDARY_NATURAL):
            {
                break;
            };

            case NS(MATH_INTERPOL_CUBIC_BOUNDARY_CLAMPED):
            {
                dy[ n_minus_2 ] -= end_boundary_param * h[ n_minus_2 ];
                break;
            }

            case NS(MATH_INTERPOL_CUBIC_BOUNDARY_PROPORTIONAL):
            {
                g[ n_minus_2 ]  = h[ n_minus_1 ] * ( SIXTRL_REAL_T )2;
                g[ n_minus_2 ] += h[ n_minus_2 ] * (
                    end_boundary_param + ( SIXTRL_REAL_T )2 );
                break;
            }

            case NS(MATH_INTERPOL_CUBIC_BOUNDARY_NOT_A_KNOT):
            {
                SIXTRL_REAL_T const d_h = ( h[ n_minus_2 ] *
                    h[ n_minus_2 ] ) / h[ n_minus_3 ];

                g[ n_minus_2 ]  = h[ n_minus_3 ] * ( SIXTRL_REAL_T )2;
                g[ n_minus_2 ] += h[ n_minus_2 ] * ( SIXTRL_REAL_T )3;
                g[ n_minus_2 ] += d_h;

                h[ n_minus_3 ] -= d_h;
                break;
            }

            default:
            {
                status = ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_GENERAL_FAILURE;
            }
        };

        if( status == ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS )
        {
            d[ 0 ] = g[ 1 ];
            ypp[ 0 ] = dy[ 1 ];

            for( ii = 1 ; ii < n_minus_2 ; ++ii )
            {
                SIXTRL_ASSERT( NS(abs)( d[ ii - 1 ] ) > ( SIXTRL_REAL_T )0 );
                l[ ii ] = h[ ii ] / d[ ii -1 ];
                d[ ii ] = g[ ii + 1 ] - l[ ii ] * h[ ii ];
                ypp[ ii ] = dy[ ii + 1 ] - ypp[ ii -1 ] * l[ ii ];
            }

            ii = n_minus_3;
            SIXTRL_ASSERT( NS(abs)( d[ ii ] ) > ( SIXTRL_REAL_T )0 );
            ypp[ ii + 1 ] = ypp[ ii ] / d[ ii ];

            for( ii = n_minus_3 - 1 ; ii >= 0 ; --ii )
            {
                SIXTRL_ASSERT( NS(abs)( d[ ii ] ) > ( SIXTRL_REAL_T )0 );
                ypp[ ii + 1 ]  = ( ypp[ ii ] - h[ ii + 1 ] * ypp[ ii + 2 ] );
                ypp[ ii + 1 ] /= d[ ii ];
            }
        }

        if( status == ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS )
        {
            switch( begin_boundary_type )
            {
                case NS(MATH_INTERPOL_CUBIC_BOUNDARY_NATURAL):
                {
                    ypp[ 0 ] = ( SIXTRL_REAL_T )0;
                    break;
                };

                case NS(MATH_INTERPOL_CUBIC_BOUNDARY_CLAMPED):
                {
                    ypp[ 0 ] = begin_boundary_param;
                    break;
                }

                case NS(MATH_INTERPOL_CUBIC_BOUNDARY_PROPORTIONAL):
                {
                    ypp[ 0 ] = begin_boundary_param * ypp[ 1 ];
                    break;
                }

                case NS(MATH_INTERPOL_CUBIC_BOUNDARY_NOT_A_KNOT):
                {
                    break;
                }

                default:
                {
                    status = ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_GENERAL_FAILURE;
                }
            };

            switch( end_boundary_type )
            {
                case NS(MATH_INTERPOL_CUBIC_BOUNDARY_NATURAL):
                {
                    ypp[ n_minus_1 ] = ( SIXTRL_REAL_T )0;
                    break;
                };

                case NS(MATH_INTERPOL_CUBIC_BOUNDARY_CLAMPED):
                {
                    ypp[ n_minus_1 ] = end_boundary_param;
                    break;
                }

                case NS(MATH_INTERPOL_CUBIC_BOUNDARY_PROPORTIONAL):
                {
                    ypp[ n_minus_1 ] = end_boundary_param * ypp[ n_minus_2 ];
                    break;
                }

                case NS(MATH_INTERPOL_CUBIC_BOUNDARY_NOT_A_KNOT):
                {
                    break;
                }

                default:
                {
                    status = ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_GENERAL_FAILURE;
                }
            };
        }
    }

    return status;
}

SIXTRL_INLINE NS(arch_status_t) NS(Math_interpol_prepare_equ)(
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT yp,
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT temp_values,
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT y,
    SIXTRL_REAL_T const x0, SIXTRL_REAL_T const dx,
    NS(math_abscissa_idx_t) const num_values,
    NS(math_interpol_t) const interpol_type ) SIXTRL_NOEXCEPT
{
    NS(arch_status_t) status = ( NS(arch_status_t)
        )SIXTRL_ARCH_STATUS_GENERAL_FAILURE;

    if( interpol_type == NS(MATH_INTERPOL_LINEAR) )
    {
        SIXTRL_REAL_T const begin_param =
            NS(Math_interpol_boundary_begin_default_param)(
                NS(MATH_INTERPOL_LINEAR_BOUNDARY_DEFAULT) );

        SIXTRL_REAL_T const end_param =
            NS(Math_interpol_boundary_begin_default_param)(
                NS(MATH_INTERPOL_LINEAR_BOUNDARY_DEFAULT) );

        status = NS(Math_interpol_linear_prepare_equ)( yp, y, x0, dx,
            num_values, NS(MATH_INTERPOL_LINEAR_BOUNDARY_DEFAULT),
            NS(MATH_INTERPOL_LINEAR_BOUNDARY_DEFAULT), begin_param, end_param );
    }
    else if( interpol_type == NS(MATH_INTERPOL_CUBIC) )
    {
        SIXTRL_REAL_T const begin_param =
            NS(Math_interpol_boundary_begin_default_param)(
                NS(MATH_INTERPOL_CUBIC_BOUNDARY_DEFAULT) );

        SIXTRL_REAL_T const end_param =
            NS(Math_interpol_boundary_begin_default_param)(
                NS(MATH_INTERPOL_CUBIC_BOUNDARY_DEFAULT) );

        status = NS(Math_interpol_cubic_prepare_equ)( yp, temp_values, y,
            x0, dx, num_values, NS(MATH_INTERPOL_CUBIC_BOUNDARY_DEFAULT),
            NS(MATH_INTERPOL_CUBIC_BOUNDARY_DEFAULT), begin_param, end_param );
    }

    return status;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE SIXTRL_REAL_T NS(Math_interpol_linear_y_equ)(
    SIXTRL_REAL_T const x,
    SIXTRL_REAL_T const x0, SIXTRL_REAL_T const dx,
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT y,
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT yp,
    NS(math_abscissa_idx_t) const num_values ) SIXTRL_NOEXCEPT
{
    NS(math_abscissa_idx_t) const idx =
        NS(Math_abscissa_index_equ)( x, x0, dx, num_values );

    SIXTRL_ASSERT( idx >= ( NS(math_abscissa_idx_t) )0 );
    SIXTRL_ASSERT( idx < num_values );
    SIXTRL_ASSERT( y  != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( yp != SIXTRL_NULLPTR );

    return y[ idx ] + ( x - ( x0 + dx * idx ) ) * yp[ idx ];
}

SIXTRL_INLINE SIXTRL_REAL_T NS(Math_interpol_linear_yp_equ)(
    SIXTRL_REAL_T const x,
    SIXTRL_REAL_T const x0, SIXTRL_REAL_T const dx,
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT y,
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT yp,
    NS(math_abscissa_idx_t) const num_values ) SIXTRL_NOEXCEPT
{
    NS(math_abscissa_idx_t) const idx =
        NS(Math_abscissa_index_equ)( x, x0, dx, num_values );

    SIXTRL_ASSERT( idx >= ( NS(math_abscissa_idx_t) )0 );
    SIXTRL_ASSERT( idx < num_values );
    SIXTRL_ASSERT( y  != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( yp != SIXTRL_NULLPTR );
    ( void )y;

    return yp[ idx ];
}

SIXTRL_INLINE SIXTRL_REAL_T NS(Math_interpol_linear_ypp_equ)(
    SIXTRL_REAL_T const x,
    SIXTRL_REAL_T const x0, SIXTRL_REAL_T const dx,
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT y,
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT yp,
    NS(math_abscissa_idx_t) const num_values ) SIXTRL_NOEXCEPT
{
    SIXTRL_ASSERT( y  != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( yp != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( NS(Math_abscissa_index_equ)( x, x0, dx, num_values ) >=
        ( NS(math_abscissa_idx_t) )0 );
    SIXTRL_ASSERT( NS(Math_abscissa_index_equ)( x, x0, dx, num_values ) <
        num_values );

    ( void )x;
    ( void )dx;
    ( void )x0;
    ( void )num_values;
    ( void )y;
    ( void )yp;

    return ( SIXTRL_REAL_T )0;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE SIXTRL_REAL_T NS(Math_interpol_linear_y_equ_ex)(
    SIXTRL_REAL_T const x,
    SIXTRL_REAL_T const x0, SIXTRL_REAL_T const dx,
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT y,
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT yp,
    NS(math_abscissa_idx_t) const num_values ) SIXTRL_NOEXCEPT
{
    SIXTRL_REAL_T y_interpolated = ( SIXTRL_REAL_T )0;

    NS(math_abscissa_idx_t) idx =
        NS(Math_abscissa_index_equ)( x, x0, dx, num_values );

    if( ( idx >= num_values ) &&
        ( num_values > ( NS(math_abscissa_idx_t) )0 ) )
    {
        idx = num_values - 1;
    }

    SIXTRL_ASSERT( y  != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( yp != SIXTRL_NULLPTR );

    if( ( idx >= ( NS(math_abscissa_idx_t) )0 ) && ( idx < num_values ) )
    {
        y_interpolated = y[ idx ] + ( x - ( x0 + dx * idx ) ) * yp[ idx ];
    }
    else if( ( idx < ( NS(math_abscissa_idx_t) )0 ) &&
             ( num_values > ( NS(math_abscissa_idx_t) )0 ) )
    {
        y_interpolated = y[ 0 ] - yp[ 0 ] * ( x0 - x );
    }

    return y_interpolated;
}

SIXTRL_INLINE SIXTRL_REAL_T NS(Math_interpol_linear_yp_equ_ex)(
    SIXTRL_REAL_T const x,
    SIXTRL_REAL_T const x0, SIXTRL_REAL_T const dx,
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT y,
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT yp,
    NS(math_abscissa_idx_t) const num_values ) SIXTRL_NOEXCEPT
{
    SIXTRL_REAL_T yp_interpolated = ( SIXTRL_REAL_T )0;

    NS(math_abscissa_idx_t) idx =
        NS(Math_abscissa_index_equ)( x, x0, dx, num_values );

    if( ( idx >= num_values ) &&
        ( num_values > ( NS(math_abscissa_idx_t) )0 ) )
    {
        yp_interpolated = yp[ num_values - 1 ];
    }
    else if( ( idx < ( NS(math_abscissa_idx_t) )0 ) &&
             ( num_values > ( NS(math_abscissa_idx_t) )0 ) )
    {
        yp_interpolated = yp[ 0 ];
    }

    SIXTRL_ASSERT( y  != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( yp != SIXTRL_NULLPTR );

    ( void )y;

    return yp_interpolated;
}

SIXTRL_INLINE SIXTRL_REAL_T NS(Math_interpol_linear_ypp_equ_ex)(
    SIXTRL_REAL_T const x,
    SIXTRL_REAL_T const x0, SIXTRL_REAL_T const dx,
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT y,
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT yp,
    NS(math_abscissa_idx_t) const num_values ) SIXTRL_NOEXCEPT
{
    return NS(Math_interpol_linear_ypp_equ)( x, x0, dx, y, yp, num_values );
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE SIXTRL_REAL_T NS(Math_interpol_cubic_y_equ)(
    SIXTRL_REAL_T const x,
    SIXTRL_REAL_T const x0, SIXTRL_REAL_T const dx,
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT y,
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT ypp,
    NS(math_abscissa_idx_t) const num_values ) SIXTRL_NOEXCEPT
{
    SIXTRL_REAL_T y_interpolated = ( SIXTRL_REAL_T )0;

    NS(math_abscissa_idx_t) const idx =
        NS(Math_abscissa_index_equ)( x, x0, dx, num_values );

    SIXTRL_STATIC_VAR SIXTRL_REAL_T const SIX = ( SIXTRL_REAL_T )6;
    SIXTRL_REAL_T const t2 = ( x0 + dx * ( idx + 1 ) - x ) / dx;
    SIXTRL_REAL_T const t1 = ( x - ( x0 + dx * idx ) ) / dx;

    SIXTRL_ASSERT( idx >= ( NS(math_abscissa_idx_t) )0 );
    SIXTRL_ASSERT( idx < num_values );
    SIXTRL_ASSERT( y != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( ypp != SIXTRL_NULLPTR );

    y_interpolated  = y[   idx + 1 ] * t1 + y[ idx ] * t2;
    y_interpolated -= ypp[ idx     ] * dx * dx * ( t2 - t2 * t2 * t2 ) / SIX;
    y_interpolated -= ypp[ idx + 1 ] * dx * dx * ( t1 - t1 * t1 * t1 ) / SIX;

    return y_interpolated;
}

SIXTRL_INLINE SIXTRL_REAL_T NS(Math_interpol_cubic_yp_equ)(
    SIXTRL_REAL_T const x,
    SIXTRL_REAL_T const x0, SIXTRL_REAL_T const dx,
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT y,
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT ypp,
    NS(math_abscissa_idx_t) const num_values ) SIXTRL_NOEXCEPT
{
    SIXTRL_REAL_T yp_interpolated = ( SIXTRL_REAL_T )0;

    SIXTRL_STATIC_VAR SIXTRL_REAL_T const ONE   = ( SIXTRL_REAL_T )1;
    SIXTRL_STATIC_VAR SIXTRL_REAL_T const THREE = ( SIXTRL_REAL_T )3;
    SIXTRL_STATIC_VAR SIXTRL_REAL_T const SIX   = ( SIXTRL_REAL_T )6;

    NS(math_abscissa_idx_t) const idx =
        NS(Math_abscissa_index_equ)( x, x0, dx, num_values );

    SIXTRL_REAL_T const t2 = ( x0 + dx * ( idx + 1 ) - x ) / dx;
    SIXTRL_REAL_T const t1 = ( x - ( x0 + dx * idx ) ) / dx;

    SIXTRL_ASSERT( idx >= ( NS(math_abscissa_idx_t) )0 );
    SIXTRL_ASSERT( idx <  num_values );
    SIXTRL_ASSERT( y   != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( ypp != SIXTRL_NULLPTR );

    yp_interpolated  = ( y[ idx + 1 ] - y[ idx ] ) / dx;
    yp_interpolated -= ypp[ idx     ] * dx * ( THREE * t2 * t2 - ONE ) / SIX;
    yp_interpolated -= ypp[ idx + 1 ] * dx * ( ONE - THREE * t1 * t1 ) / SIX;

    return yp_interpolated;
}

SIXTRL_INLINE SIXTRL_REAL_T NS(Math_interpol_cubic_ypp_equ)(
    SIXTRL_REAL_T const x,
    SIXTRL_REAL_T const x0, SIXTRL_REAL_T const dx,
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT y,
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT ypp,
    NS(math_abscissa_idx_t) const num_values ) SIXTRL_NOEXCEPT
{
    NS(math_abscissa_idx_t) const idx =
        NS(Math_abscissa_index_equ)( x, x0, dx, num_values );

    SIXTRL_ASSERT( idx >= ( NS(math_abscissa_idx_t) )0 );
    SIXTRL_ASSERT( idx <  num_values );
    SIXTRL_ASSERT( y   != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( ypp != SIXTRL_NULLPTR );

    return ( ypp[ idx     ] * ( x0 + dx * ( idx + 1 ) - x ) +
             ypp[ idx + 1 ] * ( x - ( x0 + dx * idx ) ) ) / dx;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE SIXTRL_REAL_T NS(Math_interpol_cubic_y_equ_ex)(
    SIXTRL_REAL_T const x,
    SIXTRL_REAL_T const x0, SIXTRL_REAL_T const dx,
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT y,
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT ypp,
    NS(math_abscissa_idx_t) const num_values ) SIXTRL_NOEXCEPT
{
    return NS(Math_interpol_cubic_y_equ)( x, x0, dx, y, ypp, num_values );
}

SIXTRL_INLINE SIXTRL_REAL_T NS(Math_interpol_cubic_yp_equ_ex)(
    SIXTRL_REAL_T const x,
    SIXTRL_REAL_T const x0, SIXTRL_REAL_T const dx,
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT y,
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT ypp,
    NS(math_abscissa_idx_t) const num_values ) SIXTRL_NOEXCEPT
{
    return NS(Math_interpol_cubic_yp_equ)( x, x0, dx, y, ypp, num_values );
}

SIXTRL_INLINE SIXTRL_REAL_T NS(Math_interpol_cubic_ypp_equ_ex)(
    SIXTRL_REAL_T const x,
    SIXTRL_REAL_T const x0, SIXTRL_REAL_T const dx,
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT y,
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT ypp,
    NS(math_abscissa_idx_t) const num_values ) SIXTRL_NOEXCEPT
{
    return NS(Math_interpol_cubic_ypp_equ)( x, x0, dx, y, ypp, num_values );
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE SIXTRL_REAL_T NS(Math_interpol_y_equ)( SIXTRL_REAL_T const x,
    SIXTRL_REAL_T const x0, SIXTRL_REAL_T const dx,
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT y,
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT derivatives,
    NS(math_abscissa_idx_t) const num_values,
    NS(math_interpol_t) const interpol_type ) SIXTRL_NOEXCEPT
{
    if( interpol_type == NS(MATH_INTERPOL_LINEAR) )
    {
        return NS(Math_interpol_linear_y_equ)(
            x, x0, dx, y, derivatives, num_values );
    }
    else if( interpol_type == NS(MATH_INTERPOL_CUBIC) )
    {
        return NS(Math_interpol_cubic_y_equ)(
            x, x0, dx, y, derivatives, num_values );
    }

    return ( SIXTRL_REAL_T )0;
}

SIXTRL_INLINE SIXTRL_REAL_T NS(Math_interpol_yp_equ)(
    SIXTRL_REAL_T const x,
    SIXTRL_REAL_T const x0, SIXTRL_REAL_T const dx,
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT y,
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT derivatives,
    NS(math_abscissa_idx_t) const num_values,
    NS(math_interpol_t) const interpol_type ) SIXTRL_NOEXCEPT
{
    if( interpol_type == NS(MATH_INTERPOL_LINEAR) )
    {
        return NS(Math_interpol_linear_yp_equ)(
            x, x0, dx, y, derivatives, num_values );
    }
    else if( interpol_type == NS(MATH_INTERPOL_CUBIC) )
    {
        return NS(Math_interpol_cubic_yp_equ)(
            x, x0, dx, y, derivatives, num_values );
    }

    return ( SIXTRL_REAL_T )0;
}

SIXTRL_INLINE SIXTRL_REAL_T NS(Math_interpol_ypp_equ)(
    SIXTRL_REAL_T const x,
    SIXTRL_REAL_T const x0, SIXTRL_REAL_T const dx,
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT y,
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT derivatives,
    NS(math_abscissa_idx_t) const num_values,
    NS(math_interpol_t) const interpol_type ) SIXTRL_NOEXCEPT
{
    if( interpol_type == NS(MATH_INTERPOL_LINEAR) )
    {
        return NS(Math_interpol_linear_ypp_equ)(
            x, x0, dx, y, derivatives, num_values );
    }
    else if( interpol_type == NS(MATH_INTERPOL_CUBIC) )
    {
        return NS(Math_interpol_cubic_ypp_equ)(
            x, x0, dx, y, derivatives, num_values );
    }

    return ( SIXTRL_REAL_T )0;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE SIXTRL_REAL_T NS(Math_interpol_y_equ_ex)( SIXTRL_REAL_T const x,
    SIXTRL_REAL_T const x0, SIXTRL_REAL_T const dx,
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT y,
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT derivatives,
    NS(math_abscissa_idx_t) const num_values,
    NS(math_interpol_t) const interpol_type ) SIXTRL_NOEXCEPT
{
    if( interpol_type == NS(MATH_INTERPOL_LINEAR) )
    {
        return NS(Math_interpol_linear_y_equ_ex)(
            x, x0, dx, y, derivatives, num_values );
    }
    else if( interpol_type == NS(MATH_INTERPOL_CUBIC) )
    {
        return NS(Math_interpol_cubic_y_equ_ex)(
            x, x0, dx, y, derivatives, num_values );
    }

    return ( SIXTRL_REAL_T )0;
}

SIXTRL_INLINE SIXTRL_REAL_T NS(Math_interpol_yp_equ_ex)(
    SIXTRL_REAL_T const x,
    SIXTRL_REAL_T const x0, SIXTRL_REAL_T const dx,
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT y,
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT derivatives,
    NS(math_abscissa_idx_t) const num_values,
    NS(math_interpol_t) const interpol_type ) SIXTRL_NOEXCEPT
{
    if( interpol_type == NS(MATH_INTERPOL_LINEAR) )
    {
        return NS(Math_interpol_linear_yp_equ_ex)(
            x, x0, dx, y, derivatives, num_values );
    }
    else if( interpol_type == NS(MATH_INTERPOL_CUBIC) )
    {
        return NS(Math_interpol_cubic_yp_equ_ex)(
            x, x0, dx, y, derivatives, num_values );
    }

    return ( SIXTRL_REAL_T )0;
}

SIXTRL_INLINE SIXTRL_REAL_T NS(Math_interpol_ypp_equ_ex)(
    SIXTRL_REAL_T const x,
    SIXTRL_REAL_T const x0, SIXTRL_REAL_T const dx,
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT y,
    SIXTRL_BUFFER_DATAPTR_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT derivatives,
    NS(math_abscissa_idx_t) const num_values,
    NS(math_interpol_t) const interpol_type ) SIXTRL_NOEXCEPT
{
    if( interpol_type == NS(MATH_INTERPOL_LINEAR) )
    {
        return NS(Math_interpol_linear_ypp_equ_ex)(
            x, x0, dx, y, derivatives, num_values );
    }
    else if( interpol_type == NS(MATH_INTERPOL_CUBIC) )
    {
        return NS(Math_interpol_cubic_ypp_equ_ex)(
            x, x0, dx, y, derivatives, num_values );
    }

    return ( SIXTRL_REAL_T )0;
}

#if defined( __cplusplus ) && !defined( _GPUCODE )
}
#endif /* C++, Host */

#endif /* SIXTRACKLIB_COMMON_INTERNAL_MATH_INTERPOL_H__ */
