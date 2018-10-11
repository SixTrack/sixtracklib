#ifndef SIXTRL_COMMON_BE_BEAMBEAM_BE_BEAMBEAM4D_H__
#define SIXTRL_COMMON_BE_BEAMBEAM_BE_BEAMBEAM4D_H__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stdbool.h>
    #include <stdint.h>
    #include <stdlib.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/internal/buffer_main_defines.h"
    #include "sixtracklib/common/internal/beam_elements_defines.h"
    #include "sixtracklib/common/buffer/buffer_type.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined(  _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

typedef SIXTRL_BE_DATAPTR_DEC SIXTRL_REAL_T*        NS(beambeam4d_real_ptr_t);
typedef SIXTRL_BE_DATAPTR_DEC SIXTRL_REAL_T const*  NS(beambeam4d_real_const_ptr_t);

typedef struct NS(BeamBeam4D)
{
    SIXTRL_UINT64_T                           size      SIXTRL_ALIGN( 8 );
    NS(beambeam4d_real_ptr_t) SIXTRL_RESTRICT data      SIXTRL_ALIGN( 8 );
}
NS(BeamBeam4D);

SIXTRL_FN SIXTRL_STATIC  SIXTRL_BE_ARGPTR_DEC NS(BeamBeam4D)*
NS(BeamBeam4D_preset)(
    SIXTRL_BE_ARGPTR_DEC NS(BeamBeam4D)* SIXTRL_RESTRICT beam_beam );

SIXTRL_FN SIXTRL_STATIC NS(buffer_size_t) NS(BeamBeam4D_get_num_dataptrs)(
    SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam4D) *const SIXTRL_RESTRICT beam_beam );

SIXTRL_FN SIXTRL_STATIC NS(buffer_size_t) NS(BeamBeam4D_get_num_slots)(
    SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam4D) *const SIXTRL_RESTRICT beam_beam,
    NS(buffer_size_t) const slot_size );

#if !defined( _GPUCODE )

SIXTRL_HOST_FN SIXTRL_BUFFER_DATAPTR_DEC NS(BeamBeam4D)*
NS(BeamBeam4D_new)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_UINT64_T const data_size );

SIXTRL_HOST_FN SIXTRL_BUFFER_DATAPTR_DEC NS(BeamBeam4D)*
NS(BeamBeam4D_add)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_UINT64_T const  data_size,
    NS(beambeam4d_real_ptr_t) SIXTRL_RESTRICT input_data );

SIXTRL_HOST_FN SIXTRL_BUFFER_DATAPTR_DEC NS(BeamBeam4D)*
NS(BeamBeam4D_add_copy)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam4D) *const SIXTRL_RESTRICT orig );

#endif /* !defined( _GPUCODE ) */

SIXTRL_FN SIXTRL_STATIC NS(beambeam4d_real_const_ptr_t)
NS(BeamBeam4D_get_const_data)(
    SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam4D) *const SIXTRL_RESTRICT beam_beam );

SIXTRL_FN SIXTRL_STATIC NS(beambeam4d_real_ptr_t) NS(BeamBeam4D_get_data)(
    SIXTRL_BE_ARGPTR_DEC NS(BeamBeam4D)* SIXTRL_RESTRICT beam_beam );

SIXTRL_FN SIXTRL_STATIC SIXTRL_UINT64_T NS(BeamBeam4D_get_data_size)(
    SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam4D) *const SIXTRL_RESTRICT beam_beam );

SIXTRL_FN SIXTRL_STATIC void NS(BeamBeam4D_clear)(
    SIXTRL_BE_ARGPTR_DEC NS(BeamBeam4D)* SIXTRL_RESTRICT beam_beam );

SIXTRL_FN SIXTRL_STATIC void NS(BeamBeam4D_set_data)(
    SIXTRL_BE_ARGPTR_DEC NS(BeamBeam4D)* SIXTRL_RESTRICT beam_beam,
    SIXTRL_BE_ARGPTR_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT ptr_data );

SIXTRL_FN SIXTRL_STATIC void NS(BeamBeam4D_set_size)(
    SIXTRL_BE_ARGPTR_DEC NS(BeamBeam4D)* SIXTRL_RESTRICT beam_beam,
    NS(buffer_size_t) const data_size );

SIXTRL_FN SIXTRL_STATIC void NS(BeamBeam4D_assign_data_ptr)(
    SIXTRL_BE_ARGPTR_DEC NS(BeamBeam4D)* SIXTRL_RESTRICT beam_beam,
    SIXTRL_BE_DATAPTR_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT ptr_data );

SIXTRL_FN SIXTRL_STATIC int NS(BeamBeam4D_copy)(
    SIXTRL_BE_ARGPTR_DEC NS(BeamBeam4D)* SIXTRL_RESTRICT destination,
    SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam4D) *const SIXTRL_RESTRICT source );

SIXTRL_FN SIXTRL_STATIC int NS(BeamBeam4D_compare_values)(
    SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam4D) *const SIXTRL_RESTRICT lhs,
    SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam4D) *const SIXTRL_RESTRICT rhs );

SIXTRL_FN SIXTRL_STATIC int NS(BeamBeam4D_compare_values_with_treshold)(
    SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam4D) *const SIXTRL_RESTRICT lhs,
    SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam4D) *const SIXTRL_RESTRICT rhs,
    SIXTRL_REAL_T const treshold );

#if !defined(  _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

/* ========================================================================= */
/* =====             Implementation of inline functions                ===== */
/* ========================================================================= */

#if !defined( SIXTRL_NO_INCLUDES )
    #if !defined( _GPUCODE ) || defined( __CUDACC__ )
    #include "sixtracklib/common/buffer.h"
    #endif /* !defined( _GPUCODE ) || defined( __CUDACC__ ) */
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

SIXTRL_INLINE NS(buffer_size_t) NS(BeamBeam4D_get_num_dataptrs)(
    SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam4D) *const SIXTRL_RESTRICT beam_beam )
{
    SIXTRL_ASSERT( beam_beam != SIXTRL_NULLPTR );
    return ( NS(buffer_size_t) )1u;
}

SIXTRL_INLINE NS(buffer_size_t) NS(BeamBeam4D_get_num_slots)(
    SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam4D) *const SIXTRL_RESTRICT beam_beam,
    NS(buffer_size_t) const slot_size )
{
    typedef NS(buffer_size_t) buf_size_t;

    buf_size_t const real_size = sizeof( SIXTRL_REAL_T );

    SIXTRL_ASSERT(   real_size > ( buf_size_t )0u );
    SIXTRL_ASSERT(   slot_size >= real_size );
    SIXTRL_ASSERT( ( slot_size %  real_size ) == 0u );

    return ( slot_size / real_size ) *
        NS(BeamBeam4D_get_data_size)( beam_beam );
}

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(BeamBeam4D)*
NS(BeamBeam4D_preset)(
    SIXTRL_BE_ARGPTR_DEC NS(BeamBeam4D)* SIXTRL_RESTRICT beam_beam )
{
    if( beam_beam != SIXTRL_NULLPTR )
    {
        beam_beam->size = ( SIXTRL_UINT64_T )0u;
        beam_beam->data   = SIXTRL_NULLPTR;

        NS(BeamBeam4D_clear)( beam_beam );
    }

    return beam_beam;
}

SIXTRL_INLINE NS(beambeam4d_real_const_ptr_t)
NS(BeamBeam4D_get_const_data)(
    SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam4D) *const SIXTRL_RESTRICT beam_beam )
{
    SIXTRL_ASSERT( beam_beam != SIXTRL_NULLPTR );
    return beam_beam->data;
}

SIXTRL_INLINE NS(beambeam4d_real_ptr_t)
NS(BeamBeam4D_get_data)(
    SIXTRL_BE_ARGPTR_DEC NS(BeamBeam4D)* SIXTRL_RESTRICT beam_beam )
{
    return ( NS(beambeam4d_real_ptr_t) )NS(BeamBeam4D_get_const_data)( beam_beam );
}

SIXTRL_INLINE SIXTRL_UINT64_T NS(BeamBeam4D_get_data_size)(
    SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam4D) *const SIXTRL_RESTRICT beam_beam )
{
    SIXTRL_ASSERT( beam_beam != SIXTRL_NULLPTR );
    return beam_beam->size;
}

SIXTRL_INLINE void NS(BeamBeam4D_clear)(
    SIXTRL_BE_ARGPTR_DEC NS(BeamBeam4D)* SIXTRL_RESTRICT beam_beam )
{
    typedef NS(buffer_size_t ) buf_size_t;

    buf_size_t const data_size = NS(BeamBeam4D_get_data_size)( beam_beam );
    NS(beambeam4d_real_ptr_t) ptr_data = NS(BeamBeam4D_get_data)( beam_beam );

    if( ( data_size > ( NS(buffer_size_t) )0u ) &&
        ( ptr_data != SIXTRL_NULLPTR ) )
    {
        SIXTRL_REAL_T const Z = ( SIXTRL_REAL_T )0;
        SIXTRACKLIB_SET_VALUES( SIXTRL_REAL_T, ptr_data, data_size, Z );
    }

    return;
}

SIXTRL_INLINE void NS(BeamBeam4D_set_data)(
    SIXTRL_BE_ARGPTR_DEC NS(BeamBeam4D)* SIXTRL_RESTRICT beam_beam,
    SIXTRL_BE_ARGPTR_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT ptr_data )
{
    typedef SIXTRL_REAL_T real_t;

    NS(buffer_size_t) const size =
        NS(BeamBeam4D_get_data_size)( beam_beam );

    NS(beambeam4d_real_ptr_t) ptr_dest_data =
        NS(BeamBeam4D_get_data)( beam_beam );

    SIXTRL_ASSERT( ptr_dest_data != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( ptr_data      != SIXTRL_NULLPTR );
    SIXTRACKLIB_COPY_VALUES( real_t, ptr_dest_data, ptr_data, size );

    return;
}

SIXTRL_INLINE void NS(BeamBeam4D_set_size)(
    SIXTRL_BE_ARGPTR_DEC NS(BeamBeam4D)* SIXTRL_RESTRICT beam_beam,
    NS(buffer_size_t) const data_size )
{
    SIXTRL_ASSERT( beam_beam != SIXTRL_NULLPTR );
    beam_beam->size = data_size;
    return;
}

SIXTRL_INLINE void NS(BeamBeam4D_assign_data_ptr)(
    SIXTRL_BE_ARGPTR_DEC NS(BeamBeam4D)* SIXTRL_RESTRICT beam_beam,
    SIXTRL_BE_DATAPTR_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT ptr_data )
{
    SIXTRL_ASSERT( beam_beam != SIXTRL_NULLPTR );
    beam_beam->data = ptr_data;
    return;
}

SIXTRL_INLINE int NS(BeamBeam4D_copy)(
    SIXTRL_BE_ARGPTR_DEC NS(BeamBeam4D)* SIXTRL_RESTRICT destination,
    SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam4D) *const SIXTRL_RESTRICT source )
{
    int success = -1;

    if( ( destination != SIXTRL_NULLPTR ) && ( source != SIXTRL_NULLPTR ) &&
        ( destination != source ) &&
        ( NS(BeamBeam4D_get_const_data)( destination ) != SIXTRL_NULLPTR ) &&
        ( NS(BeamBeam4D_get_const_data)( source      ) != SIXTRL_NULLPTR ) &&
        ( NS(BeamBeam4D_get_data_size)( destination ) ==
          NS(BeamBeam4D_get_data_size)( source ) ) )
    {
        SIXTRL_ASSERT( NS(BeamBeam4D_get_const_data)( destination ) !=
                       NS(BeamBeam4D_get_const_data)( source ) );

        SIXTRACKLIB_COPY_VALUES( SIXTRL_REAL_T,
            NS(BeamBeam4D_get_data)( destination ),
            NS(BeamBeam4D_get_const_data)( source ),
            NS(BeamBeam4D_get_data_size)( source ) );

        success = 0;
    }

    return success;
}

SIXTRL_INLINE int NS(BeamBeam4D_compare_values)(
    SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam4D) *const SIXTRL_RESTRICT lhs,
    SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam4D) *const SIXTRL_RESTRICT rhs )
{
    typedef NS(buffer_size_t)   buf_size_t;
    typedef SIXTRL_REAL_T       real_t;
    typedef SIXTRL_BE_DATAPTR_DEC real_t const* ptr_data_t;

    int cmp_value = -1;

    ptr_data_t lhs_data = NS(BeamBeam4D_get_const_data)( lhs );
    ptr_data_t rhs_data = NS(BeamBeam4D_get_const_data)( rhs );

    buf_size_t const lhs_size = NS(BeamBeam4D_get_data_size)( lhs );
    buf_size_t const rhs_size = NS(BeamBeam4D_get_data_size)( rhs );

    if( ( lhs_data != SIXTRL_NULLPTR  ) && ( rhs_data != SIXTRL_NULLPTR ) )
    {
        if( lhs_size == rhs_size )
        {
            cmp_value = 0;

            if( ( lhs_size > ( buf_size_t )0u ) && ( lhs_data != rhs_data ) )
            {
                buf_size_t ii = ( buf_size_t )0u;

                for( ; ii < lhs_size ; ++ii )
                {
                    if( lhs_data[ ii ] > rhs_data[ ii ] )
                    {
                        cmp_value = +1;
                        break;
                    }
                    else if( lhs_data[ ii ] < rhs_data[ ii ] )
                    {
                        cmp_value = -1;
                        break;
                    }
                }
            }
        }
        else if( lhs_size > rhs_size )
        {
            cmp_value = +1;
        }
        else if( rhs_size < rhs_size )
        {
            cmp_value = -1;
        }
    }
    else if( lhs_data != SIXTRL_NULLPTR )
    {
        SIXTRL_ASSERT( ( rhs_data == SIXTRL_NULLPTR ) &&
                       ( rhs_size == ( buf_size_t )0u ) );
        cmp_value = +1;
    }

    return cmp_value;
}

SIXTRL_INLINE int NS(BeamBeam4D_compare_values_with_treshold)(
    SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam4D) *const SIXTRL_RESTRICT lhs,
    SIXTRL_BE_ARGPTR_DEC const NS(BeamBeam4D) *const SIXTRL_RESTRICT rhs,
    SIXTRL_REAL_T const treshold )
{
    typedef NS(buffer_size_t)   buf_size_t;
    typedef SIXTRL_REAL_T       real_t;
    typedef SIXTRL_BE_DATAPTR_DEC real_t const* ptr_data_t;

    int cmp_value = -1;

    ptr_data_t lhs_data = NS(BeamBeam4D_get_const_data)( lhs );
    ptr_data_t rhs_data = NS(BeamBeam4D_get_const_data)( rhs );

    buf_size_t const lhs_size = NS(BeamBeam4D_get_data_size)( lhs );
    buf_size_t const rhs_size = NS(BeamBeam4D_get_data_size)( rhs );

    if( ( lhs_data != SIXTRL_NULLPTR  ) && ( rhs_data != SIXTRL_NULLPTR ) )
    {
        if( lhs_size == rhs_size )
        {
            cmp_value = 0;

            if( ( lhs_size > ( buf_size_t )0u ) && ( lhs_data != rhs_data ) )
            {
                buf_size_t ii = ( buf_size_t )0u;

                for( ; ii < lhs_size ; ++ii )
                {
                    real_t const diff     = lhs_data[ ii ] - rhs_data[ ii ];
                    real_t const abs_diff = ( diff >= ( real_t )0 )
                        ? diff : -diff;

                    if( abs_diff > treshold )
                    {
                        cmp_value = ( diff > 0 ) ? +1 : -1;
                        break;
                    }
                }
            }
        }
        else if( lhs_size > rhs_size )
        {
            cmp_value = +1;
        }
        else if( rhs_size < rhs_size )
        {
            cmp_value = -1;
        }
    }
    else if( lhs_data != SIXTRL_NULLPTR )
    {
        SIXTRL_ASSERT( ( rhs_data == SIXTRL_NULLPTR ) &&
                       ( rhs_size == ( buf_size_t )0u ) );
        cmp_value = +1;
    }

    return cmp_value;
}

#if !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

#endif /* SIXTRL_COMMON_BE_BEAMBEAM_BE_BEAMBEAM4D_H__ */

/* sixtracklib/common/be_beambeam/be_beambeam4d.h */
