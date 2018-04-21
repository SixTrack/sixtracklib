#ifndef SIXTRACKLIB_COMMON_IMPL_PARTICLES_TYPE_H__
#define SIXTRACKLIB_COMMON_IMPL_PARTICLES_TYPE_H__

#if !defined( _GPUCODE )

#include "sixtracklib/_impl/definitions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#endif /* !defined( _GPUCODE ) */

typedef struct NS( Particles )
{
    SIXTRL_REAL_T* __restrict__ q0;     /* C */
    SIXTRL_REAL_T* __restrict__ mass0;  /* eV */
    SIXTRL_REAL_T* __restrict__ beta0;  /* nounit */
    SIXTRL_REAL_T* __restrict__ gamma0; /* nounit */
    SIXTRL_REAL_T* __restrict__ p0c;    /* eV */

    /* coordinate arrays */
    SIXTRL_INT64_T* __restrict__ partid;
    SIXTRL_INT64_T* __restrict__ elemid; /* element at which the particle was lost */
    SIXTRL_INT64_T* __restrict__ turn;   /* turn at which the particle was lost */
    SIXTRL_INT64_T* __restrict__ state;  /* negative means particle lost */

    SIXTRL_REAL_T* __restrict__ s;     /* [m] */
    SIXTRL_REAL_T* __restrict__ x;     /* [m] */
    SIXTRL_REAL_T* __restrict__ px;    /* Px/P0 */
    SIXTRL_REAL_T* __restrict__ y;     /* [m] */
    SIXTRL_REAL_T* __restrict__ py;    /* Py/P0 */
    SIXTRL_REAL_T* __restrict__ sigma; /* s-beta0*c*t  where t is the time
                      since the beginning of the simulation */

    SIXTRL_REAL_T* __restrict__ psigma; /* (E-E0) / (beta0 P0c) conjugate of sigma */
    SIXTRL_REAL_T* __restrict__ delta;  /* P/P0-1 = 1/rpp-1 */
    SIXTRL_REAL_T* __restrict__ rpp;    /* ratio P0 /P */
    SIXTRL_REAL_T* __restrict__ rvv;    /* ratio beta / beta0 */
    SIXTRL_REAL_T* __restrict__ chi;    /* q/q0 * m/m0  */

    SIXTRL_UINT64_T npart;
    SIXTRL_UINT64_T flags; /* particle flags */
    void* __restrict__  ptr_mem_context; /* memory_context -> can contain */
    void* __restrict__  ptr_mem_begin; /* reference memory addr for (un)packing (optional) */
} NS( Particles ) __attribute__((aligned (SIXTRL_ALIGN)));


SIXTRL_STATIC SIXTRL_UINT64_T const NS(PARTICLES_PACK_INDICATOR) = ( SIXTRL_UINT64_T )1u;

SIXTRL_STATIC SIXTRL_INT64_T const NS( PARTICLE_VALID_STATE ) = ( SIXTRL_INT64_T )0;

SIXTRL_STATIC SIXTRL_UINT64_T const NS( PARTICLES_FLAGS_NONE ) = ( SIXTRL_UINT64_T )0x0000;

SIXTRL_STATIC SIXTRL_UINT64_T const NS( PARTICLES_FLAGS_PACKED ) = ( SIXTRL_UINT64_T )0x0001;

SIXTRL_STATIC SIXTRL_UINT64_T const NS( PARTICLES_FLAGS_OWNS_MEMORY ) = ( SIXTRL_UINT64_T )0x0002;

SIXTRL_STATIC SIXTRL_UINT64_T const
    NS( PARTICLES_FLAGS_MEM_CTX_MEMPOOL ) = ( SIXTRL_UINT64_T )0x0010;

SIXTRL_STATIC SIXTRL_UINT64_T const
    NS( PARTICLES_FLAGS_MEM_CTX_SINGLEPARTICLE ) = ( SIXTRL_UINT64_T )0x0020;
    
SIXTRL_STATIC SIXTRL_UINT64_T const 
    NS(PARTICLES_FLAGS_MEM_CTX_FLAT_MEMORY ) = ( SIXTRL_UINT64_T )0x0040;

SIXTRL_STATIC SIXTRL_UINT64_T const NS( PARTICLES_FLAGS_ALIGN_MASK ) = ( SIXTRL_UINT64_T )0xFFFF00;

SIXTRL_STATIC SIXTRL_UINT64_T const NS( PARTICLES_MAX_ALIGNMENT ) = ( SIXTRL_UINT64_T )0xFFFF;

SIXTRL_STATIC SIXTRL_UINT64_T const
    NS( PARTICLES_FLAGS_ALIGN_MASK_OFFSET_BITS ) = ( SIXTRL_UINT64_T )8;

/* ========================================================================= */

SIXTRL_STATIC SIXTRL_SIZE_T const NS( PARTICLES_DEFAULT_MEMPOOL_CHUNK_SIZE ) = (SIXTRL_SIZE_T)8u;
SIXTRL_STATIC SIXTRL_SIZE_T const NS( PARTICLES_DEFAULT_MEMPOOL_ALIGNMENT ) = (SIXTRL_SIZE_T)16u;
SIXTRL_STATIC SIXTRL_SIZE_T const NS( PARTICLES_NUM_OF_DOUBLE_ELEMENTS ) = (SIXTRL_SIZE_T)16u;
SIXTRL_STATIC SIXTRL_SIZE_T const NS( PARTICLES_NUM_OF_INT64_ELEMENTS ) = (SIXTRL_SIZE_T)4u;
SIXTRL_STATIC SIXTRL_SIZE_T const NS( PARTICLES_NUM_OF_ATTRIBUTES ) = ( SIXTRL_SIZE_T )20u;
SIXTRL_STATIC SIXTRL_SIZE_T const NS( PARTICLES_PACK_BLOCK_LENGTH ) = ( SIXTRL_SIZE_T )192u;


SIXTRL_STATIC SIXTRL_UINT64_T const NS( PARTICLES_UNPACK_MAP  ) = ( SIXTRL_UINT64_T )0x0000;
SIXTRL_STATIC SIXTRL_UINT64_T const NS( PARTICLES_UNPACK_COPY ) = ( SIXTRL_UINT64_T )0x0001;
SIXTRL_STATIC SIXTRL_UINT64_T const NS( PARTICLES_UNPACK_CHECK_CONSISTENCY ) = 
    ( SIXTRL_UINT64_T )0x02;

/* ========================================================================= */

SIXTRL_STATIC SIXTRL_UINT64_T NS( Particles_get_size )( const struct NS( Particles ) *
                                          const SIXTRL_RESTRICT p );

SIXTRL_STATIC void NS( Particles_set_size )( struct NS( Particles ) *
                                          SIXTRL_RESTRICT p,
                                      SIXTRL_UINT64_T const npart );

SIXTRL_STATIC SIXTRL_UINT64_T NS( Particles_get_flags )( const struct NS( Particles ) *
                                           const SIXTRL_RESTRICT p );

SIXTRL_STATIC void NS( Particles_set_flags )( struct NS( Particles ) *
                                           SIXTRL_RESTRICT p,
                                       SIXTRL_UINT64_T const flags );

SIXTRL_STATIC void const* NS( Particles_get_const_ptr_mem_context )(
    const struct NS( Particles ) * const SIXTRL_RESTRICT p );

SIXTRL_STATIC void* NS( Particles_get_ptr_mem_context )( struct NS( Particles ) *
                                                  SIXTRL_RESTRICT p );

SIXTRL_STATIC void const* NS( Particles_get_const_mem_begin )(
    const struct NS(Particles) *const SIXTRL_RESTRICT p );

SIXTRL_STATIC void* NS( Particles_get_mem_begin )(
    struct NS(Particles)* SIXTRL_RESTRICT p );

SIXTRL_STATIC void NS( Particles_set_ptr_mem_context )( struct NS( Particles ) *
                                                     SIXTRL_RESTRICT p,
                                                 void* ptr_mem_context );

SIXTRL_STATIC void NS( Particles_set_ptr_mem_begin )( 
    struct NS(Particles)* SIXTRL_RESTRICT p, void* ptr_mem_begin );

SIXTRL_STATIC void NS( Particles_copy_single_unchecked )( 
    struct NS( Particles ) * SIXTRL_RESTRICT dest,
    SIXTRL_SIZE_T const dest_id,
    const struct NS( Particles ) *const SIXTRL_RESTRICT source, 
    SIXTRL_SIZE_T const source_id );

SIXTRL_STATIC void NS( Particles_copy_all_unchecked )(
    struct NS( Particles ) * SIXTRL_RESTRICT dest,
    const struct NS( Particles ) *const SIXTRL_RESTRICT source );                                                 

/* ========================================================================= */

SIXTRL_STATIC SIXTRL_REAL_T NS( Particles_get_q0_value )( const NS( Particles ) *
                                                const SIXTRL_RESTRICT p,
                                            SIXTRL_UINT64_T const id );

SIXTRL_STATIC SIXTRL_REAL_T const* NS( Particles_get_q0 )( const NS( Particles ) *
                                             const SIXTRL_RESTRICT p );

SIXTRL_STATIC SIXTRL_REAL_T NS( Particles_get_mass0_value )( const NS( Particles ) *
                                                   const SIXTRL_RESTRICT p,
                                               SIXTRL_UINT64_T const id );

SIXTRL_STATIC SIXTRL_REAL_T const* NS( Particles_get_mass0 )( const NS( Particles ) *
                                                const SIXTRL_RESTRICT p );

SIXTRL_STATIC SIXTRL_REAL_T NS( Particles_get_beta0_value )( const NS( Particles ) *
                                                   const SIXTRL_RESTRICT p,
                                               SIXTRL_UINT64_T const id );

SIXTRL_STATIC SIXTRL_REAL_T const* NS( Particles_get_beta0 )( const NS( Particles ) *
                                                const SIXTRL_RESTRICT p );

SIXTRL_STATIC SIXTRL_REAL_T NS( Particles_get_gamma0_value )( const NS( Particles ) *
                                                    const SIXTRL_RESTRICT p,
                                                SIXTRL_UINT64_T const id );

SIXTRL_STATIC SIXTRL_REAL_T const* NS( Particles_get_gamma0 )( const NS( Particles ) *
                                                 const SIXTRL_RESTRICT p );

SIXTRL_STATIC SIXTRL_REAL_T NS( Particles_get_p0c_value )( const NS( Particles ) *
                                                 const SIXTRL_RESTRICT p,
                                             SIXTRL_UINT64_T const id );

SIXTRL_STATIC SIXTRL_REAL_T const* NS( Particles_get_p0c )( const NS( Particles ) *
                                              const SIXTRL_RESTRICT p );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC int NS( Particles_is_particle_lost )( const NS( Particles ) *
                                                 const SIXTRL_RESTRICT p,
                                             SIXTRL_UINT64_T const id );

SIXTRL_STATIC SIXTRL_INT64_T NS( Particles_get_particle_id_value )(
    const NS( Particles ) * const SIXTRL_RESTRICT p, SIXTRL_UINT64_T const id );

SIXTRL_STATIC SIXTRL_INT64_T const* NS( Particles_get_particle_id )(
    const NS( Particles ) * const SIXTRL_RESTRICT p );

SIXTRL_STATIC SIXTRL_INT64_T NS( Particles_get_lost_at_element_id_value )(
    const NS( Particles ) * const SIXTRL_RESTRICT p, SIXTRL_UINT64_T const id );

SIXTRL_STATIC SIXTRL_INT64_T const* NS( Particles_get_lost_at_element_id )(
    const NS( Particles ) * const SIXTRL_RESTRICT p );

SIXTRL_STATIC SIXTRL_INT64_T NS( Particles_get_lost_at_turn_value )(
    const NS( Particles ) * const SIXTRL_RESTRICT p, SIXTRL_UINT64_T const id );

SIXTRL_STATIC SIXTRL_INT64_T const* NS( Particles_get_lost_at_turn )(
    const NS( Particles ) * const SIXTRL_RESTRICT p );

SIXTRL_STATIC SIXTRL_INT64_T NS( Particles_get_state_value )( const NS( Particles ) *
                                                    const SIXTRL_RESTRICT p,
                                                SIXTRL_UINT64_T const id );

SIXTRL_STATIC SIXTRL_INT64_T const* NS( Particles_get_state )( const NS( Particles ) *
                                                 const SIXTRL_RESTRICT p );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC SIXTRL_REAL_T NS( Particles_get_s_value )( const NS( Particles ) *
                                               const SIXTRL_RESTRICT p,
                                           SIXTRL_UINT64_T const id );

SIXTRL_STATIC SIXTRL_REAL_T const* NS( Particles_get_s )( const NS( Particles ) *
                                            const SIXTRL_RESTRICT p );

SIXTRL_STATIC SIXTRL_REAL_T NS( Particles_get_x_value )( const NS( Particles ) *
                                               const SIXTRL_RESTRICT p,
                                           SIXTRL_UINT64_T const id );

SIXTRL_STATIC SIXTRL_REAL_T const* NS( Particles_get_x )( const NS( Particles ) *
                                            const SIXTRL_RESTRICT p );

SIXTRL_STATIC SIXTRL_REAL_T NS( Particles_get_y_value )( const NS( Particles ) *
                                               const SIXTRL_RESTRICT p,
                                           SIXTRL_UINT64_T const id );

SIXTRL_STATIC SIXTRL_REAL_T const* NS( Particles_get_y )( const NS( Particles ) *
                                            const SIXTRL_RESTRICT p );

SIXTRL_STATIC SIXTRL_REAL_T NS( Particles_get_px_value )( const NS( Particles ) *
                                                const SIXTRL_RESTRICT p,
                                            SIXTRL_UINT64_T const id );

SIXTRL_STATIC SIXTRL_REAL_T const* NS( Particles_get_px )( const NS( Particles ) *
                                             const SIXTRL_RESTRICT p );

SIXTRL_STATIC SIXTRL_REAL_T NS( Particles_get_py_value )( const NS( Particles ) *
                                                const SIXTRL_RESTRICT p,
                                            SIXTRL_UINT64_T const id );

SIXTRL_STATIC SIXTRL_REAL_T const* NS( Particles_get_py )( const NS( Particles ) *
                                             const SIXTRL_RESTRICT p );

SIXTRL_STATIC SIXTRL_REAL_T NS( Particles_get_sigma_value )( const NS( Particles ) *
                                                   const SIXTRL_RESTRICT p,
                                               SIXTRL_UINT64_T const id );

SIXTRL_STATIC SIXTRL_REAL_T const* NS( Particles_get_sigma )( const NS( Particles ) *
                                                const SIXTRL_RESTRICT p );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC SIXTRL_REAL_T NS( Particles_get_psigma_value )( const NS( Particles ) *
                                                    const SIXTRL_RESTRICT p,
                                                SIXTRL_UINT64_T const id );

SIXTRL_STATIC SIXTRL_REAL_T const* NS( Particles_get_psigma )( const NS( Particles ) *
                                                 const SIXTRL_RESTRICT p );

SIXTRL_STATIC SIXTRL_REAL_T NS( Particles_get_delta_value )( const NS( Particles ) *
                                                   const SIXTRL_RESTRICT p,
                                               SIXTRL_UINT64_T const id );

SIXTRL_STATIC SIXTRL_REAL_T const* NS( Particles_get_delta )( const NS( Particles ) *
                                                const SIXTRL_RESTRICT p );

SIXTRL_STATIC SIXTRL_REAL_T NS( Particles_get_rpp_value )( const NS( Particles ) *
                                                 const SIXTRL_RESTRICT p,
                                             SIXTRL_UINT64_T const id );

SIXTRL_STATIC SIXTRL_REAL_T const* NS( Particles_get_rpp )( const NS( Particles ) *
                                              const SIXTRL_RESTRICT p );

SIXTRL_STATIC SIXTRL_REAL_T NS( Particles_get_rvv_value )( const NS( Particles ) *
                                                 const SIXTRL_RESTRICT p,
                                             SIXTRL_UINT64_T const id );

SIXTRL_STATIC SIXTRL_REAL_T const* NS( Particles_get_rvv )( const NS( Particles ) *
                                              const SIXTRL_RESTRICT p );

SIXTRL_STATIC SIXTRL_REAL_T NS( Particles_get_chi_value )( const NS( Particles ) *
                                                 const SIXTRL_RESTRICT p,
                                             SIXTRL_UINT64_T const id );

SIXTRL_STATIC SIXTRL_REAL_T const* NS( Particles_get_chi )( const NS( Particles ) *
                                              const SIXTRL_RESTRICT p );

/* ========================================================================= */

SIXTRL_STATIC void NS( Particles_set_q0_value )( NS( Particles ) * SIXTRL_RESTRICT p,
                                          SIXTRL_UINT64_T const id,
                                          SIXTRL_REAL_T q0 );

SIXTRL_STATIC void NS( Particles_set_q0 )( NS( Particles ) * SIXTRL_RESTRICT p,
                                    SIXTRL_REAL_T const* SIXTRL_RESTRICT ptr_q0 );

SIXTRL_STATIC void NS( Particles_assign_ptr_to_q0 )( NS( Particles ) *
                                                  SIXTRL_RESTRICT p,
                                              SIXTRL_REAL_T* ptr_q0 );

SIXTRL_STATIC void NS( Particles_set_mass0_value )(
    NS( Particles ) * SIXTRL_RESTRICT p, SIXTRL_UINT64_T const id, SIXTRL_REAL_T const mass0 );

SIXTRL_STATIC void
    NS( Particles_set_mass0 )( NS( Particles ) * SIXTRL_RESTRICT p,
                               SIXTRL_REAL_T const* SIXTRL_RESTRICT ptr_mass0 );

SIXTRL_STATIC void NS( Particles_assign_ptr_to_mass0 )( NS( Particles ) *
                                                     SIXTRL_RESTRICT p,
                                                 SIXTRL_REAL_T* ptr_mass0 );

SIXTRL_STATIC void NS( Particles_set_beta0_value )(
    NS( Particles ) * SIXTRL_RESTRICT p, SIXTRL_UINT64_T const id, SIXTRL_REAL_T const beta0 );

SIXTRL_STATIC void
    NS( Particles_set_beta0 )( NS( Particles ) * SIXTRL_RESTRICT p,
                               SIXTRL_REAL_T const* SIXTRL_RESTRICT ptr_beta0 );

SIXTRL_STATIC void NS( Particles_assign_ptr_to_beta0 )( NS( Particles ) *
                                                     SIXTRL_RESTRICT p,
                                                 SIXTRL_REAL_T* ptr_beta0 );

SIXTRL_STATIC void NS( Particles_set_gamma0_value )(
    NS( Particles ) * SIXTRL_RESTRICT p, SIXTRL_UINT64_T const id, SIXTRL_REAL_T const gamma0 );

SIXTRL_STATIC void
    NS( Particles_set_gamma0 )( NS( Particles ) * SIXTRL_RESTRICT p,
                                SIXTRL_REAL_T const* SIXTRL_RESTRICT ptr_gamma0 );

SIXTRL_STATIC void NS( Particles_assign_ptr_to_gamma0 )( NS( Particles ) *
                                                      SIXTRL_RESTRICT p,
                                                  SIXTRL_REAL_T* ptr_gamma0 );

SIXTRL_STATIC void NS( Particles_set_p0c_value )( NS( Particles ) * SIXTRL_RESTRICT p,
                                           SIXTRL_UINT64_T const id,
                                           SIXTRL_REAL_T const p0c );

SIXTRL_STATIC void NS( Particles_set_p0c )( NS( Particles ) * SIXTRL_RESTRICT p,
                                     SIXTRL_REAL_T const* SIXTRL_RESTRICT ptr_p0c );

SIXTRL_STATIC void NS( Particles_assign_ptr_to_p0c )( NS( Particles ) *
                                                   SIXTRL_RESTRICT p,
                                               SIXTRL_REAL_T* ptr_p0c );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC void NS( Particles_set_particle_id_value )(
    NS( Particles ) * SIXTRL_RESTRICT p, SIXTRL_UINT64_T const id, SIXTRL_INT64_T const partid );

SIXTRL_STATIC void NS( Particles_set_particle_id )(
    NS( Particles ) * SIXTRL_RESTRICT p,
    SIXTRL_INT64_T const* SIXTRL_RESTRICT ptr_partid );

SIXTRL_STATIC void NS( Particles_assign_ptr_to_particle_id )( NS( Particles ) *
                                                           SIXTRL_RESTRICT p,
                                                       SIXTRL_INT64_T* ptr_partid );

SIXTRL_STATIC void NS( Particles_set_lost_at_element_id_value )(
    NS( Particles ) * SIXTRL_RESTRICT p, SIXTRL_UINT64_T const id, SIXTRL_INT64_T const elemid );

SIXTRL_STATIC void NS( Particles_set_lost_at_element_id )(
    NS( Particles ) * SIXTRL_RESTRICT p,
    SIXTRL_INT64_T const* SIXTRL_RESTRICT ptr_elemid );

SIXTRL_STATIC void NS( Particles_assign_ptr_to_lost_at_element_id )(
    NS( Particles ) * SIXTRL_RESTRICT p, SIXTRL_INT64_T* ptr_elemid );

SIXTRL_STATIC void NS( Particles_set_lost_at_turn_value )(
    NS( Particles ) * SIXTRL_RESTRICT p, SIXTRL_UINT64_T const id, SIXTRL_INT64_T const turn );

SIXTRL_STATIC void
    NS( Particles_set_lost_at_turn )( NS( Particles ) * SIXTRL_RESTRICT p,
                                      SIXTRL_INT64_T const* SIXTRL_RESTRICT ptr_turn );

SIXTRL_STATIC void NS( Particles_assign_ptr_to_lost_at_turn )( NS( Particles ) *
                                                            SIXTRL_RESTRICT p,
                                                        SIXTRL_INT64_T* ptr_turn );

SIXTRL_STATIC void NS( Particles_set_state_value )(
    NS( Particles ) * SIXTRL_RESTRICT p, SIXTRL_UINT64_T const id, SIXTRL_INT64_T const state );

SIXTRL_STATIC void
    NS( Particles_set_state )( NS( Particles ) * SIXTRL_RESTRICT p,
                               SIXTRL_INT64_T const* SIXTRL_RESTRICT ptr_state );

SIXTRL_STATIC void NS( Particles_assign_ptr_to_state )( NS( Particles ) *
                                                     SIXTRL_RESTRICT p,
                                                 SIXTRL_INT64_T* ptr_state );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC void NS( Particles_set_s_value )( NS( Particles ) * SIXTRL_RESTRICT p,
                                         SIXTRL_UINT64_T const id,
                                         SIXTRL_REAL_T s );

SIXTRL_STATIC void NS( Particles_set_s )( NS( Particles ) * SIXTRL_RESTRICT p,
                                   SIXTRL_REAL_T const* SIXTRL_RESTRICT ptr_s );

SIXTRL_STATIC void NS( Particles_assign_ptr_to_s )( NS( Particles ) *
                                                 SIXTRL_RESTRICT p,
                                             SIXTRL_REAL_T* ptr_s );

SIXTRL_STATIC void NS( Particles_set_x_value )( NS( Particles ) * SIXTRL_RESTRICT p,
                                         SIXTRL_UINT64_T const id,
                                         SIXTRL_REAL_T const x );

SIXTRL_STATIC void NS( Particles_set_x )( NS( Particles ) * SIXTRL_RESTRICT p,
                                   SIXTRL_REAL_T const* SIXTRL_RESTRICT ptr_x );

SIXTRL_STATIC void NS( Particles_assign_ptr_to_x )( NS( Particles ) *
                                                 SIXTRL_RESTRICT p,
                                             SIXTRL_REAL_T* ptr_x );

SIXTRL_STATIC void NS( Particles_set_y_value )( NS( Particles ) * SIXTRL_RESTRICT p,
                                         SIXTRL_UINT64_T const id,
                                         SIXTRL_REAL_T const y );

SIXTRL_STATIC void NS( Particles_set_y )( NS( Particles ) * SIXTRL_RESTRICT p,
                                   SIXTRL_REAL_T const* SIXTRL_RESTRICT ptr_y );

SIXTRL_STATIC void NS( Particles_assign_ptr_to_y )( NS( Particles ) *
                                                 SIXTRL_RESTRICT p,
                                             SIXTRL_REAL_T* ptr_y );

SIXTRL_STATIC void NS( Particles_set_px_value )( NS( Particles ) * SIXTRL_RESTRICT p,
                                          SIXTRL_UINT64_T const id,
                                          SIXTRL_REAL_T const  px );

SIXTRL_STATIC void NS( Particles_set_px )( NS( Particles ) * SIXTRL_RESTRICT p,
                                    SIXTRL_REAL_T const* SIXTRL_RESTRICT ptr_px );

SIXTRL_STATIC void NS( Particles_assign_ptr_to_px )( NS( Particles ) *
                                                  SIXTRL_RESTRICT p,
                                              SIXTRL_REAL_T* ptr_px );

SIXTRL_STATIC void NS( Particles_set_py_value )( NS( Particles ) * SIXTRL_RESTRICT p,
                                          SIXTRL_UINT64_T const id,
                                          SIXTRL_REAL_T const  py );

SIXTRL_STATIC void NS( Particles_set_py )( NS( Particles ) * SIXTRL_RESTRICT p,
                                    SIXTRL_REAL_T const* SIXTRL_RESTRICT ptr_py );

SIXTRL_STATIC void NS( Particles_assign_ptr_to_py )( NS( Particles ) *
                                                  SIXTRL_RESTRICT p,
                                              SIXTRL_REAL_T* ptr_py );

SIXTRL_STATIC void NS( Particles_set_sigma_value )(
    NS( Particles ) * SIXTRL_RESTRICT p, SIXTRL_UINT64_T const id, SIXTRL_REAL_T sigma );

SIXTRL_STATIC void
    NS( Particles_set_sigma )( NS( Particles ) * SIXTRL_RESTRICT p,
                               SIXTRL_REAL_T const* SIXTRL_RESTRICT ptr_sigma );

SIXTRL_STATIC void NS( Particles_assign_ptr_to_sigma )( NS( Particles ) *
                                                     SIXTRL_RESTRICT p,
                                                 SIXTRL_REAL_T* ptr_sigma );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC void NS( Particles_set_psigma_value )(
    NS( Particles ) * SIXTRL_RESTRICT p, SIXTRL_UINT64_T const id, SIXTRL_REAL_T psigma );

SIXTRL_STATIC void
    NS( Particles_set_psigma )( NS( Particles ) * SIXTRL_RESTRICT p,
                                SIXTRL_REAL_T const* SIXTRL_RESTRICT ptr_psigma );

SIXTRL_STATIC void NS( Particles_assign_ptr_to_psigma )( NS( Particles ) *
                                                      SIXTRL_RESTRICT p,
                                                  SIXTRL_REAL_T* ptr_psigma );

SIXTRL_STATIC void NS( Particles_set_delta_value )(
    NS( Particles ) * SIXTRL_RESTRICT p, SIXTRL_UINT64_T const id, SIXTRL_REAL_T delta );

SIXTRL_STATIC void
    NS( Particles_set_delta )( NS( Particles ) * SIXTRL_RESTRICT p,
                               SIXTRL_REAL_T const* SIXTRL_RESTRICT ptr_delta );

SIXTRL_STATIC void NS( Particles_assign_ptr_to_delta )( NS( Particles ) *
                                                     SIXTRL_RESTRICT p,
                                                 SIXTRL_REAL_T* ptr_delta );

SIXTRL_STATIC void NS( Particles_set_rpp_value )( NS( Particles ) * SIXTRL_RESTRICT p,
                                           SIXTRL_UINT64_T const id,
                                           SIXTRL_REAL_T const  rpp );

SIXTRL_STATIC void NS( Particles_set_rpp )( NS( Particles ) * SIXTRL_RESTRICT p,
                                     SIXTRL_REAL_T const* SIXTRL_RESTRICT ptr_rpp );

SIXTRL_STATIC void NS( Particles_assign_ptr_to_rpp )( NS( Particles ) *
                                                   SIXTRL_RESTRICT p,
                                               SIXTRL_REAL_T* ptr_rpp );

SIXTRL_STATIC void NS( Particles_set_rvv_value )( NS( Particles ) * SIXTRL_RESTRICT p,
                                           SIXTRL_UINT64_T const id,
                                           SIXTRL_REAL_T const  rvv );

SIXTRL_STATIC void NS( Particles_set_rvv )( NS( Particles ) * SIXTRL_RESTRICT p,
                                     SIXTRL_REAL_T const* SIXTRL_RESTRICT ptr_rvv );

SIXTRL_STATIC void NS( Particles_assign_ptr_to_rvv )( NS( Particles ) *
                                                   SIXTRL_RESTRICT p,
                                               SIXTRL_REAL_T* ptr_rvv );

SIXTRL_STATIC void NS( Particles_set_chi_value )( NS( Particles ) * SIXTRL_RESTRICT p,
                                           SIXTRL_UINT64_T const id,
                                           SIXTRL_REAL_T const  chi );

SIXTRL_STATIC void NS( Particles_set_chi )( NS( Particles ) * SIXTRL_RESTRICT p,
                                     SIXTRL_REAL_T const* SIXTRL_RESTRICT ptr_chi );

SIXTRL_STATIC void NS( Particles_assign_ptr_to_chi )( NS( Particles ) *
                                                   SIXTRL_RESTRICT p,
                                               SIXTRL_REAL_T* ptr_chi );

/* ========================================================================= *
 * ==== IMPLEMENTATION OF INLINE FUNCTIONS
 * ========================================================================= */

/* ========================================================================= */

SIXTRL_INLINE SIXTRL_UINT64_T NS( Particles_get_size )( const struct NS( Particles ) *
                                                 const SIXTRL_RESTRICT p )
{
    return ( p != 0 ) ? p->npart : ( SIXTRL_UINT64_T )0u;
}

SIXTRL_INLINE void NS( Particles_set_size )( struct NS( Particles ) *
                                                 SIXTRL_RESTRICT p,
                                             SIXTRL_UINT64_T npart )
{
    SIXTRL_ASSERT( p != 0 );
    p->npart = npart;
    return;
}

SIXTRL_INLINE SIXTRL_UINT64_T NS( Particles_get_flags )( const struct NS( Particles ) *
                                                  const SIXTRL_RESTRICT p )
{
    return ( p != 0 ) ? p->flags : NS( PARTICLES_FLAGS_NONE );
}

SIXTRL_INLINE void NS( Particles_set_flags )( struct NS( Particles ) *
                                                  SIXTRL_RESTRICT p,
                                              SIXTRL_UINT64_T flags )
{
    SIXTRL_ASSERT( p != 0 );
    p->flags = flags;
    return;
}

SIXTRL_INLINE void const* NS( Particles_get_const_ptr_mem_context )(
    const struct NS( Particles ) * const SIXTRL_RESTRICT p )
{
    return ( p != 0 ) ? p->ptr_mem_context : 0;
}

SIXTRL_INLINE void* NS( Particles_get_ptr_mem_context )(
    struct NS( Particles ) * SIXTRL_RESTRICT p )
{
    /* casting away const-ness of a pointer is ok even in C */
    return (void*)NS( Particles_get_const_ptr_mem_context )( p );
}

SIXTRL_INLINE void const* NS( Particles_get_const_mem_begin )(
    const struct NS( Particles ) *const SIXTRL_RESTRICT p )
{
    return ( p != 0 ) ? p->ptr_mem_begin : 0;
}

SIXTRL_INLINE void* NS( Particles_get_mem_begin )(
    struct NS( Particles )* SIXTRL_RESTRICT p )
{
    /* casting away const-ness of a pointer is ok even in C */
    return ( void* )NS( Particles_get_const_mem_begin )( p );
}

SIXTRL_INLINE void NS( Particles_set_ptr_mem_context )( struct NS( Particles ) *
                                                            SIXTRL_RESTRICT p,
                                                        void* ptr_mem_context )
{
    SIXTRL_ASSERT( p != 0 );
    p->ptr_mem_context = ptr_mem_context;
    return;
}

SIXTRL_INLINE void NS( Particles_set_ptr_mem_begin )( 
    struct NS( Particles ) * SIXTRL_RESTRICT p, void* ptr_mem_begin )
{
    SIXTRL_ASSERT( p != 0 );
    p->ptr_mem_begin = ptr_mem_begin;
    return;
}

SIXTRL_INLINE void NS( Particles_copy_single_unchecked )( 
    struct NS( Particles ) * SIXTRL_RESTRICT des, SIXTRL_SIZE_T const des_id,
    const struct NS( Particles ) *const SIXTRL_RESTRICT src, 
    SIXTRL_SIZE_T const src_id )
{
    SIXTRL_ASSERT( ( des != 0 ) && ( src != 0 ) &&
                   ( NS(Particles_get_size)( des ) > des_id ) &&
                   ( NS(Particles_get_size)( src ) > src_id ) );
    
    NS( Particles_set_q0_value )
    ( des, des_id, NS( Particles_get_q0_value )( src, src_id ) );

    NS( Particles_set_mass0_value )
    ( des, des_id, NS( Particles_get_mass0_value )( src, src_id ) );

    NS( Particles_set_beta0_value )
    ( des, des_id, NS( Particles_get_beta0_value )( src, src_id ) );

    NS( Particles_set_gamma0_value )
    ( des, des_id, NS( Particles_get_gamma0_value )( src, src_id ) );

    NS( Particles_set_p0c_value )
    ( des, des_id, NS( Particles_get_p0c_value )( src, src_id ) );

    NS( Particles_set_particle_id_value )
    ( des, des_id, NS( Particles_get_particle_id_value )( src, src_id ) );

    NS( Particles_set_lost_at_element_id_value )
    ( des, des_id, NS( Particles_get_lost_at_element_id_value )( src, src_id ) );

    NS( Particles_set_lost_at_turn_value )
    ( des, des_id, NS( Particles_get_lost_at_turn_value )( src, src_id ) );

    NS( Particles_set_state_value )
    ( des, des_id, NS( Particles_get_state_value )( src, src_id ) );

    NS( Particles_set_s_value )
    ( des, des_id, NS( Particles_get_s_value )( src, src_id ) );

    NS( Particles_set_x_value )
    ( des, des_id, NS( Particles_get_x_value )( src, src_id ) );

    NS( Particles_set_y_value )
    ( des, des_id, NS( Particles_get_y_value )( src, src_id ) );

    NS( Particles_set_px_value )
    ( des, des_id, NS( Particles_get_px_value )( src, src_id ) );

    NS( Particles_set_py_value )
    ( des, des_id, NS( Particles_get_py_value )( src, src_id ) );

    NS( Particles_set_sigma_value )
    ( des, des_id, NS( Particles_get_sigma_value )( src, src_id ) );

    NS( Particles_set_psigma_value )
    ( des, des_id, NS( Particles_get_psigma_value )( src, src_id ) );

    NS( Particles_set_delta_value )
    ( des, des_id, NS( Particles_get_delta_value )( src, src_id ) );

    NS( Particles_set_rpp_value )
    ( des, des_id, NS( Particles_get_rpp_value )( src, src_id ) );

    NS( Particles_set_rvv_value )
    ( des, des_id, NS( Particles_get_rvv_value )( src, src_id ) );

    NS( Particles_set_chi_value )
    ( des, des_id, NS( Particles_get_chi_value )( src, src_id ) );
    
    return;
}

SIXTRL_INLINE void NS( Particles_copy_all_unchecked )(
    struct NS( Particles ) * SIXTRL_RESTRICT des,
    const struct NS( Particles ) *const SIXTRL_RESTRICT src )
{
    SIXTRL_ASSERT( 
        ( des != 0 ) && ( src != 0 ) &&
        ( NS(Particles_get_size)( des ) == NS(Particles_get_size( src ) ) ) );
    
    NS( Particles_set_q0     )( des, NS( Particles_get_q0     )( src ) );
    NS( Particles_set_mass0  )( des, NS( Particles_get_mass0  )( src ) );
    NS( Particles_set_beta0  )( des, NS( Particles_get_beta0  )( src ) );
    NS( Particles_set_gamma0 )( des, NS( Particles_get_gamma0 )( src ) );
    NS( Particles_set_p0c    )( des, NS( Particles_get_p0c    )( src ) );
    
    NS( Particles_set_s      )( des, NS( Particles_get_s      )( src ) );
    NS( Particles_set_x      )( des, NS( Particles_get_x      )( src ) );
    NS( Particles_set_y      )( des, NS( Particles_get_y      )( src ) );
    NS( Particles_set_px     )( des, NS( Particles_get_px     )( src ) );
    NS( Particles_set_py     )( des, NS( Particles_get_py     )( src ) );
    NS( Particles_set_sigma  )( des, NS( Particles_get_sigma  )( src ) );
    
    NS( Particles_set_particle_id )( des, 
        NS( Particles_get_particle_id )( src ) );
    
    NS( Particles_set_lost_at_element_id )( des, 
        NS( Particles_get_lost_at_element_id)( src ) );
    
    NS( Particles_set_lost_at_turn)( des, 
        NS(Particles_get_lost_at_turn)( src ) );
    
    NS( Particles_set_state  )( des, NS( Particles_get_state  )( src ) );        
    NS( Particles_set_psigma )( des, NS( Particles_get_psigma )( src ) );
    NS( Particles_set_delta  )( des, NS( Particles_get_delta  )( src ) );
    NS( Particles_set_rpp    )( des, NS( Particles_get_rpp    )( src ) );
    NS( Particles_set_rvv    )( des, NS( Particles_get_rvv    )( src ) );
    NS( Particles_set_chi    )( des, NS( Particles_get_chi    )( src ) );
    
    return;
}

/* ========================================================================= */

SIXTRL_INLINE SIXTRL_REAL_T NS( Particles_get_q0_value )( const NS( Particles ) *
                                                       const SIXTRL_RESTRICT p,
                                                   SIXTRL_UINT64_T const id )
{
    SIXTRL_ASSERT( ( p != 0 ) && ( id < p->npart ) && ( p->q0 != 0 ) );
    return p->q0[id];
}

SIXTRL_INLINE SIXTRL_REAL_T const* NS( Particles_get_q0 )( const NS( Particles ) *
                                                    const SIXTRL_RESTRICT p )
{
    return ( p != 0 ) ? p->q0 : 0;
}

SIXTRL_INLINE SIXTRL_REAL_T NS( Particles_get_mass0_value )(
    const NS( Particles ) * const SIXTRL_RESTRICT p, SIXTRL_UINT64_T const id )
{
    SIXTRL_ASSERT( ( p != 0 ) && ( id < p->npart ) && ( p->mass0 != 0 ) );
    return p->mass0[id];
}

SIXTRL_INLINE SIXTRL_REAL_T const* NS( Particles_get_mass0 )( const NS( Particles ) *
                                                       const SIXTRL_RESTRICT p )
{
    return ( p != 0 ) ? p->mass0 : 0;
}

SIXTRL_INLINE SIXTRL_REAL_T NS( Particles_get_beta0_value )(
    const NS( Particles ) * const SIXTRL_RESTRICT p, SIXTRL_UINT64_T const id )
{
    SIXTRL_ASSERT( ( p != 0 ) && ( id < p->npart ) && ( p->beta0 != 0 ) );
    return p->beta0[id];
}

SIXTRL_INLINE SIXTRL_REAL_T const* NS( Particles_get_beta0 )( const NS( Particles ) *
                                                       const SIXTRL_RESTRICT p )
{
    return ( p != 0 ) ? p->beta0 : 0;
}

SIXTRL_INLINE SIXTRL_REAL_T NS( Particles_get_gamma0_value )(
    const NS( Particles ) * const SIXTRL_RESTRICT p, SIXTRL_UINT64_T const id )
{
    SIXTRL_ASSERT( ( p != 0 ) && ( id < p->npart ) && ( p->gamma0 != 0 ) );
    return p->gamma0[id];
}

SIXTRL_INLINE SIXTRL_REAL_T const* NS( Particles_get_gamma0 )(
    const NS( Particles ) * const SIXTRL_RESTRICT p )
{
    return ( p != 0 ) ? p->gamma0 : 0;
}

SIXTRL_INLINE SIXTRL_REAL_T NS( Particles_get_p0c_value )( const NS( Particles ) *
                                                        const SIXTRL_RESTRICT p,
                                                    SIXTRL_UINT64_T const id )
{
    SIXTRL_ASSERT( ( p != 0 ) && ( id < p->npart ) && ( p->p0c != 0 ) );
    return p->p0c[id];
}

SIXTRL_INLINE SIXTRL_REAL_T const* NS( Particles_get_p0c )( const NS( Particles ) *
                                                     const SIXTRL_RESTRICT p )
{
    return ( p != 0 ) ? p->p0c : 0;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE int NS( Particles_is_particle_lost )( const NS( Particles ) *
                                                        const p,
                                                    SIXTRL_UINT64_T const id )
{
    SIXTRL_ASSERT( ( p != 0 ) && ( id < p->npart ) && ( p->state != 0 ) );
    return ( p->state[id] < 0 ) ? 1 : 0;
}

SIXTRL_INLINE SIXTRL_INT64_T NS( Particles_get_particle_id_value )(
    const NS( Particles ) * const SIXTRL_RESTRICT p, SIXTRL_UINT64_T const id )
{
    SIXTRL_ASSERT( ( p != 0 ) && ( id < p->npart ) && ( p->partid != 0 ) );
    return p->partid[id];
}

SIXTRL_INLINE SIXTRL_INT64_T const* NS( Particles_get_particle_id )(
    const NS( Particles ) * const SIXTRL_RESTRICT p )
{
    return ( p != 0 ) ? p->partid : 0;
}

SIXTRL_INLINE SIXTRL_INT64_T NS( Particles_get_lost_at_element_id_value )(
    const NS( Particles ) * const p, SIXTRL_UINT64_T const id )
{
    SIXTRL_ASSERT( ( p != 0 ) && ( id < p->npart ) && ( p->elemid != 0 ) );
    return p->elemid[id];
}

SIXTRL_INLINE SIXTRL_INT64_T const*
    NS( Particles_get_lost_at_element_id )( const NS( Particles ) * const p )
{
    return ( p != 0 ) ? p->elemid : 0;
}

SIXTRL_INLINE SIXTRL_INT64_T NS( Particles_get_lost_at_turn_value )(
    const NS( Particles ) * const p, SIXTRL_UINT64_T const id )
{
    SIXTRL_ASSERT( ( p != 0 ) && ( id < p->npart ) && ( p->turn != 0 ) );
    return p->turn[id];
}

SIXTRL_INLINE SIXTRL_INT64_T const*
    NS( Particles_get_lost_at_turn )( const NS( Particles ) * const p )
{
    return ( p != 0 ) ? p->turn : 0;
}

SIXTRL_INLINE SIXTRL_INT64_T NS( Particles_get_state_value )( const NS( Particles ) *
                                                           const p,
                                                       SIXTRL_UINT64_T const id )
{
    SIXTRL_ASSERT( ( p != 0 ) && ( id < p->npart ) && ( p->state != 0 ) );
    return p->state[id];
}

SIXTRL_INLINE SIXTRL_INT64_T const* NS( Particles_get_state )( const NS( Particles ) *
                                                        const p )
{
    return ( p != 0 ) ? p->state : 0;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE SIXTRL_REAL_T NS( Particles_get_s_value )( const NS( Particles ) *
                                                      const SIXTRL_RESTRICT p,
                                                  SIXTRL_UINT64_T const id )
{
    SIXTRL_ASSERT( ( p != 0 ) && ( id < p->npart ) && ( p->s != 0 ) );
    return p->s[id];
}

SIXTRL_INLINE SIXTRL_REAL_T const* NS( Particles_get_s )( const NS( Particles ) *
                                                   const SIXTRL_RESTRICT p )
{
    return ( p != 0 ) ? p->s : 0;
}

SIXTRL_INLINE SIXTRL_REAL_T NS( Particles_get_x_value )( const NS( Particles ) *
                                                      const SIXTRL_RESTRICT p,
                                                  SIXTRL_UINT64_T const id )
{
    SIXTRL_ASSERT( ( p != 0 ) && ( id < p->npart ) && ( p->x != 0 ) );
    return p->x[id];
}

SIXTRL_INLINE SIXTRL_REAL_T const* NS( Particles_get_x )( const NS( Particles ) *
                                                   const SIXTRL_RESTRICT p )
{
    return ( p != 0 ) ? p->x : 0;
}

SIXTRL_INLINE SIXTRL_REAL_T NS( Particles_get_y_value )( const NS( Particles ) *
                                                      const SIXTRL_RESTRICT p,
                                                  SIXTRL_UINT64_T const id )
{
    SIXTRL_ASSERT( ( p != 0 ) && ( id < p->npart ) && ( p->y != 0 ) );
    return p->y[id];
}

SIXTRL_INLINE SIXTRL_REAL_T const* NS( Particles_get_y )( const NS( Particles ) *
                                                   const SIXTRL_RESTRICT p )
{
    return ( p != 0 ) ? p->y : 0;
}

SIXTRL_INLINE SIXTRL_REAL_T NS( Particles_get_px_value )( const NS( Particles ) *
                                                       const SIXTRL_RESTRICT p,
                                                   SIXTRL_UINT64_T const id )
{
    SIXTRL_ASSERT( ( p != 0 ) && ( id < p->npart ) && ( p->px != 0 ) );
    return p->px[id];
}

SIXTRL_INLINE SIXTRL_REAL_T const* NS( Particles_get_px )( const NS( Particles ) *
                                                    const SIXTRL_RESTRICT p )
{
    return ( p != 0 ) ? p->px : 0;
}

SIXTRL_INLINE SIXTRL_REAL_T NS( Particles_get_py_value )( const NS( Particles ) *
                                                       const SIXTRL_RESTRICT p,
                                                   SIXTRL_UINT64_T const id )
{
    SIXTRL_ASSERT( ( p != 0 ) && ( id < p->npart ) && ( p->py != 0 ) );
    return p->py[id];
}

SIXTRL_INLINE SIXTRL_REAL_T const* NS( Particles_get_py )( const NS( Particles ) *
                                                    const SIXTRL_RESTRICT p )
{
    return ( p != 0 ) ? p->py : 0;
}

SIXTRL_INLINE SIXTRL_REAL_T NS( Particles_get_sigma_value )(
    const NS( Particles ) * const SIXTRL_RESTRICT p, SIXTRL_UINT64_T const id )
{
    SIXTRL_ASSERT( ( p != 0 ) && ( id < p->npart ) && ( p->sigma != 0 ) );
    return p->sigma[id];
}

SIXTRL_INLINE SIXTRL_REAL_T const* NS( Particles_get_sigma )( const NS( Particles ) *
                                                       const SIXTRL_RESTRICT p )
{
    return ( p != 0 ) ? p->sigma : 0;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE SIXTRL_REAL_T NS( Particles_get_psigma_value )(
    const NS( Particles ) * const SIXTRL_RESTRICT p, SIXTRL_UINT64_T const id )
{
    SIXTRL_ASSERT( ( p != 0 ) && ( id < p->npart ) && ( p->psigma != 0 ) );
    return p->psigma[id];
}

SIXTRL_INLINE SIXTRL_REAL_T const* NS( Particles_get_psigma )(
    const NS( Particles ) * const SIXTRL_RESTRICT p )
{
    return ( p != 0 ) ? p->psigma : 0;
}

SIXTRL_INLINE SIXTRL_REAL_T NS( Particles_get_delta_value )(
    const NS( Particles ) * const SIXTRL_RESTRICT p, SIXTRL_UINT64_T const id )
{
    SIXTRL_ASSERT( ( p != 0 ) && ( id < p->npart ) && ( p->delta != 0 ) );
    return p->delta[id];
}

SIXTRL_INLINE SIXTRL_REAL_T const* NS( Particles_get_delta )( const NS( Particles ) *
                                                       const SIXTRL_RESTRICT p )
{
    return ( p != 0 ) ? p->delta : 0;
}

SIXTRL_INLINE SIXTRL_REAL_T NS( Particles_get_rpp_value )( const NS( Particles ) *
                                                        const SIXTRL_RESTRICT p,
                                                    SIXTRL_UINT64_T const id )
{
    SIXTRL_ASSERT( ( p != 0 ) && ( id < p->npart ) && ( p->rpp != 0 ) );
    return p->rpp[id];
}

SIXTRL_INLINE SIXTRL_REAL_T const* NS( Particles_get_rpp )( const NS( Particles ) *
                                                     const SIXTRL_RESTRICT p )
{
    return ( p != 0 ) ? p->rpp : 0;
}

SIXTRL_INLINE SIXTRL_REAL_T NS( Particles_get_rvv_value )( const NS( Particles ) *
                                                        const SIXTRL_RESTRICT p,
                                                        SIXTRL_UINT64_T const id )
{
    SIXTRL_ASSERT( ( p != 0 ) && ( id < p->npart ) && ( p->rvv != 0 ) );
    return p->rvv[id];
}

SIXTRL_INLINE SIXTRL_REAL_T const* NS( Particles_get_rvv )( const NS( Particles ) *
                                                     const SIXTRL_RESTRICT p )
{
    return ( p != 0 ) ? p->rvv : 0;
}

SIXTRL_INLINE SIXTRL_REAL_T NS( Particles_get_chi_value )( const NS( Particles ) *
                                                        const SIXTRL_RESTRICT p,
                                                    SIXTRL_UINT64_T const id )
{
    SIXTRL_ASSERT( ( p != 0 ) && ( id < p->npart ) && ( p->chi != 0 ) );
    return p->chi[id];
}

SIXTRL_INLINE SIXTRL_REAL_T const* NS( Particles_get_chi )( const NS( Particles ) *
                                                     const SIXTRL_RESTRICT p )
{
    return ( p != 0 ) ? p->chi : 0;
}

/* ========================================================================= */

SIXTRL_INLINE void NS( Particles_set_q0_value )(
    NS( Particles ) * SIXTRL_RESTRICT p, SIXTRL_UINT64_T const id, SIXTRL_REAL_T q0 )
{
    SIXTRL_ASSERT( ( p != 0 ) && ( id < p->npart ) && ( p->q0 != 0 ) );
    p->q0[id] = q0;
    return;
}

SIXTRL_INLINE void
    NS( Particles_set_q0 )( NS( Particles ) * SIXTRL_RESTRICT p,
                            SIXTRL_REAL_T const* SIXTRL_RESTRICT ptr_q0 )
{
    SIXTRACKLIB_COPY_VALUES( SIXTRL_REAL_T, p->q0, ptr_q0, p->npart );
    return;
}

SIXTRL_INLINE void NS( Particles_assign_ptr_to_q0 )( NS( Particles ) *
                                                         SIXTRL_RESTRICT p,
                                                     SIXTRL_REAL_T* ptr_q0 )
{
    SIXTRL_ASSERT( p != 0 );
    p->q0 = ptr_q0;
    return;
}

SIXTRL_INLINE void NS( Particles_set_mass0_value )(
    NS( Particles ) * SIXTRL_RESTRICT p, SIXTRL_UINT64_T const id, SIXTRL_REAL_T mass0 )
{
    SIXTRL_ASSERT( ( p != 0 ) && ( id < p->npart ) && ( p->mass0 != 0 ) );
    p->mass0[id] = mass0;
    return;
}

SIXTRL_INLINE void
    NS( Particles_set_mass0 )( NS( Particles ) * SIXTRL_RESTRICT p,
                               SIXTRL_REAL_T const* SIXTRL_RESTRICT ptr_mass0 )
{
    SIXTRACKLIB_COPY_VALUES( SIXTRL_REAL_T, p->mass0, ptr_mass0, p->npart );
    return;
}

SIXTRL_INLINE void NS( Particles_assign_ptr_to_mass0 )( NS( Particles ) *
                                                            SIXTRL_RESTRICT p,
                                                        SIXTRL_REAL_T* ptr_mass0 )
{
    SIXTRL_ASSERT( p != 0 );
    p->mass0 = ptr_mass0;
    return;
}

SIXTRL_INLINE void NS( Particles_set_beta0_value )(
    NS( Particles ) * SIXTRL_RESTRICT p, SIXTRL_UINT64_T const id, SIXTRL_REAL_T beta0 )
{
    SIXTRL_ASSERT( ( p != 0 ) && ( id < p->npart ) && ( p->beta0 != 0 ) );
    p->beta0[id] = beta0;
    return;
}

SIXTRL_INLINE void
    NS( Particles_set_beta0 )( NS( Particles ) * SIXTRL_RESTRICT p,
                               SIXTRL_REAL_T const* SIXTRL_RESTRICT ptr_beta0 )
{
    SIXTRACKLIB_COPY_VALUES( SIXTRL_REAL_T, p->beta0, ptr_beta0, p->npart );
    return;
}

SIXTRL_INLINE void NS( Particles_assign_ptr_to_beta0 )( NS( Particles ) *
                                                            SIXTRL_RESTRICT p,
                                                        SIXTRL_REAL_T* ptr_beta0 )
{
    SIXTRL_ASSERT( p != 0 );
    p->beta0 = ptr_beta0;
    return;
}

SIXTRL_INLINE void NS( Particles_set_gamma0_value )(
    NS( Particles ) * SIXTRL_RESTRICT p, SIXTRL_UINT64_T const id, SIXTRL_REAL_T gamma0 )
{
    SIXTRL_ASSERT( ( p != 0 ) && ( id < p->npart ) && ( p->gamma0 != 0 ) );
    p->gamma0[id] = gamma0;
    return;
}

SIXTRL_INLINE void
    NS( Particles_set_gamma0 )( NS( Particles ) * SIXTRL_RESTRICT p,
                                SIXTRL_REAL_T const* SIXTRL_RESTRICT ptr_gamma0 )
{
    SIXTRACKLIB_COPY_VALUES( SIXTRL_REAL_T, p->gamma0, ptr_gamma0, p->npart );
    return;
}

SIXTRL_INLINE void NS( Particles_assign_ptr_to_gamma0 )( NS( Particles ) *
                                                             SIXTRL_RESTRICT p,
                                                         SIXTRL_REAL_T* ptr_gamma0 )
{
    SIXTRL_ASSERT( p != 0 );
    p->gamma0 = ptr_gamma0;
    return;
}

SIXTRL_INLINE void NS( Particles_set_p0c_value )(
    NS( Particles ) * SIXTRL_RESTRICT p, SIXTRL_UINT64_T const id, SIXTRL_REAL_T p0c )
{
    SIXTRL_ASSERT( ( p != 0 ) && ( id < p->npart ) && ( p->p0c != 0 ) );
    p->p0c[id] = p0c;
    return;
}

SIXTRL_INLINE void
    NS( Particles_set_p0c )( NS( Particles ) * SIXTRL_RESTRICT p,
                             SIXTRL_REAL_T const* SIXTRL_RESTRICT ptr_p0c )
{
    SIXTRACKLIB_COPY_VALUES( SIXTRL_REAL_T, p->p0c, ptr_p0c, p->npart );
    return;
}

SIXTRL_INLINE void NS( Particles_assign_ptr_to_p0c )( NS( Particles ) *
                                                          SIXTRL_RESTRICT p,
                                                      SIXTRL_REAL_T* ptr_p0c )
{
    SIXTRL_ASSERT( p != 0 );
    p->p0c = ptr_p0c;
    return;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE void NS( Particles_set_particle_id_value )(
    NS( Particles ) * SIXTRL_RESTRICT p, SIXTRL_UINT64_T const id, SIXTRL_INT64_T const partid )
{
    SIXTRL_ASSERT( ( p != 0 ) && ( id < p->npart ) && ( p->partid != 0 ) );
    p->partid[id] = partid;
    return;
}

SIXTRL_INLINE void
    NS( Particles_set_particle_id )( NS( Particles ) * SIXTRL_RESTRICT p,
                                     SIXTRL_INT64_T const* SIXTRL_RESTRICT ptr_partid )
{
    SIXTRACKLIB_COPY_VALUES( SIXTRL_INT64_T, p->partid, ptr_partid, p->npart );
    return;
}

SIXTRL_INLINE void NS( Particles_assign_ptr_to_particle_id )(
    NS( Particles ) * SIXTRL_RESTRICT p, SIXTRL_INT64_T* ptr_partid )
{
    SIXTRL_ASSERT( p != 0 );
    p->partid = ptr_partid;
    return;
}

SIXTRL_INLINE void NS( Particles_set_lost_at_element_id_value )(
    NS( Particles ) * SIXTRL_RESTRICT p, SIXTRL_UINT64_T const id, SIXTRL_INT64_T const elemid )
{
    SIXTRL_ASSERT( ( p != 0 ) && ( id < p->npart ) && ( p->elemid != 0 ) );
    p->elemid[id] = elemid;
    return;
}

SIXTRL_INLINE void NS( Particles_set_lost_at_element_id )(
    NS( Particles ) * SIXTRL_RESTRICT p,
    SIXTRL_INT64_T const* SIXTRL_RESTRICT ptr_elemid )
{
    SIXTRACKLIB_COPY_VALUES( SIXTRL_INT64_T, p->elemid, ptr_elemid, p->npart );
    return;
}

SIXTRL_INLINE void NS( Particles_assign_ptr_to_lost_at_element_id )(
    NS( Particles ) * SIXTRL_RESTRICT p, SIXTRL_INT64_T* ptr_elemid )
{
    SIXTRL_ASSERT( p != 0 );
    p->elemid = ptr_elemid;
    return;
}

SIXTRL_INLINE void NS( Particles_set_lost_at_turn_value )(
    NS( Particles ) * SIXTRL_RESTRICT p, SIXTRL_UINT64_T const id, SIXTRL_INT64_T const turn )
{
    SIXTRL_ASSERT( ( p != 0 ) && ( id < p->npart ) && ( p->p0c != 0 ) );
    p->turn[id] = turn;
    return;
}

SIXTRL_INLINE void
    NS( Particles_set_lost_at_turn )( NS( Particles ) * SIXTRL_RESTRICT p,
                                      SIXTRL_INT64_T const* SIXTRL_RESTRICT ptr_turn )
{
    SIXTRACKLIB_COPY_VALUES( SIXTRL_INT64_T, p->turn, ptr_turn, p->npart );
    return;
}

SIXTRL_INLINE void NS( Particles_assign_ptr_to_lost_at_turn )(
    NS( Particles ) * SIXTRL_RESTRICT p, SIXTRL_INT64_T* ptr_turn )
{
    SIXTRL_ASSERT( p != 0 );
    p->turn = ptr_turn;
    return;
}

SIXTRL_INLINE void NS( Particles_set_state_value )(
    NS( Particles ) * SIXTRL_RESTRICT p, SIXTRL_UINT64_T const id, SIXTRL_INT64_T const state )
{
    SIXTRL_ASSERT( ( p != 0 ) && ( id < p->npart ) && ( p->state != 0 ) );
    p->state[id] = state;
    return;
}

SIXTRL_INLINE void
    NS( Particles_set_state )( NS( Particles ) * SIXTRL_RESTRICT p,
                               SIXTRL_INT64_T const* SIXTRL_RESTRICT ptr_state )
{
    SIXTRACKLIB_COPY_VALUES( SIXTRL_INT64_T, p->state, ptr_state, p->npart );
    return;
}

SIXTRL_INLINE void NS( Particles_assign_ptr_to_state )( NS( Particles ) *
                                                            SIXTRL_RESTRICT p,
                                                        SIXTRL_INT64_T* ptr_state )
{
    SIXTRL_ASSERT( p != 0 );
    p->state = ptr_state;
    return;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE void NS( Particles_set_s_value )(
    NS( Particles ) * SIXTRL_RESTRICT p, SIXTRL_UINT64_T const id, SIXTRL_REAL_T s )
{
    SIXTRL_ASSERT( ( p != 0 ) && ( id < p->npart ) && ( p->s != 0 ) );
    p->s[id] = s;
    return;
}

SIXTRL_INLINE void NS( Particles_set_s )( NS( Particles ) * SIXTRL_RESTRICT p,
                                          SIXTRL_REAL_T const* SIXTRL_RESTRICT ptr_s )
{
    SIXTRACKLIB_COPY_VALUES( SIXTRL_REAL_T, p->s, ptr_s, p->npart );
    return;
}

SIXTRL_INLINE void NS( Particles_assign_ptr_to_s )( NS( Particles ) *
                                                        SIXTRL_RESTRICT p,
                                                    SIXTRL_REAL_T* ptr_s )
{
    SIXTRL_ASSERT( p != 0 );
    p->s = ptr_s;
    return;
}

SIXTRL_INLINE void NS( Particles_set_x_value )(
    NS( Particles ) * SIXTRL_RESTRICT p, SIXTRL_UINT64_T const id, SIXTRL_REAL_T x )
{
    SIXTRL_ASSERT( ( p != 0 ) && ( id < p->npart ) && ( p->x != 0 ) );
    p->x[id] = x;
    return;
}

SIXTRL_INLINE void NS( Particles_set_x )( NS( Particles ) * SIXTRL_RESTRICT p,
                                          SIXTRL_REAL_T const* SIXTRL_RESTRICT ptr_x )
{
    SIXTRACKLIB_COPY_VALUES( SIXTRL_REAL_T, p->x, ptr_x, p->npart );
    return;
}

SIXTRL_INLINE void NS( Particles_assign_ptr_to_x )( NS( Particles ) *
                                                        SIXTRL_RESTRICT p,
                                                    SIXTRL_REAL_T* ptr_x )
{
    SIXTRL_ASSERT( p != 0 );
    p->x = ptr_x;
    return;
}

SIXTRL_INLINE void NS( Particles_set_y_value )(
    NS( Particles ) * SIXTRL_RESTRICT p, SIXTRL_UINT64_T const id, SIXTRL_REAL_T y )
{
    SIXTRL_ASSERT( ( p != 0 ) && ( id < p->npart ) && ( p->y != 0 ) );
    p->y[id] = y;
    return;
}

SIXTRL_INLINE void NS( Particles_set_y )( NS( Particles ) * SIXTRL_RESTRICT p,
                                          SIXTRL_REAL_T const* SIXTRL_RESTRICT ptr_y )
{
    SIXTRACKLIB_COPY_VALUES( SIXTRL_REAL_T, p->y, ptr_y, p->npart );
    return;
}

SIXTRL_INLINE void NS( Particles_assign_ptr_to_y )( NS( Particles ) *
                                                        SIXTRL_RESTRICT p,
                                                    SIXTRL_REAL_T* ptr_y )
{
    SIXTRL_ASSERT( p != 0 );
    p->y = ptr_y;
    return;
}

SIXTRL_INLINE void NS( Particles_set_px_value )(
    NS( Particles ) * SIXTRL_RESTRICT p, SIXTRL_UINT64_T const id, SIXTRL_REAL_T px )
{
    SIXTRL_ASSERT( ( p != 0 ) && ( id < p->npart ) && ( p->px != 0 ) );
    p->px[id] = px;
    return;
}

SIXTRL_INLINE void
    NS( Particles_set_px )( NS( Particles ) * SIXTRL_RESTRICT p,
                            SIXTRL_REAL_T const* SIXTRL_RESTRICT ptr_px )
{
    SIXTRACKLIB_COPY_VALUES( SIXTRL_REAL_T, p->px, ptr_px, p->npart );
    return;
}

SIXTRL_INLINE void NS( Particles_assign_ptr_to_px )( NS( Particles ) *
                                                         SIXTRL_RESTRICT p,
                                                     SIXTRL_REAL_T* ptr_px )
{
    SIXTRL_ASSERT( p != 0 );
    p->px = ptr_px;
    return;
}

SIXTRL_INLINE void NS( Particles_set_py_value )(
    NS( Particles ) * SIXTRL_RESTRICT p, SIXTRL_UINT64_T const id, SIXTRL_REAL_T py )
{
    SIXTRL_ASSERT( ( p != 0 ) && ( id < p->npart ) && ( p->py != 0 ) );
    p->py[id] = py;
    return;
}

SIXTRL_INLINE void
    NS( Particles_set_py )( NS( Particles ) * SIXTRL_RESTRICT p,
                            SIXTRL_REAL_T const* SIXTRL_RESTRICT ptr_py )
{
    SIXTRACKLIB_COPY_VALUES( SIXTRL_REAL_T, p->py, ptr_py, p->npart );
    return;
}

SIXTRL_INLINE void NS( Particles_assign_ptr_to_py )( NS( Particles ) *
                                                         SIXTRL_RESTRICT p,
                                                     SIXTRL_REAL_T* ptr_py )
{
    SIXTRL_ASSERT( p != 0 );
    p->py = ptr_py;
    return;
}

SIXTRL_INLINE void NS( Particles_set_sigma_value )(
    NS( Particles ) * SIXTRL_RESTRICT p, SIXTRL_UINT64_T const id, SIXTRL_REAL_T sigma )
{
    SIXTRL_ASSERT( ( p != 0 ) && ( id < p->npart ) && ( p->sigma != 0 ) );
    p->sigma[id] = sigma;
    return;
}

SIXTRL_INLINE void
    NS( Particles_set_sigma )( NS( Particles ) * SIXTRL_RESTRICT p,
                               SIXTRL_REAL_T const* SIXTRL_RESTRICT ptr_sigma )
{
    SIXTRACKLIB_COPY_VALUES( SIXTRL_REAL_T, p->sigma, ptr_sigma, p->npart );
    return;
}

SIXTRL_INLINE void NS( Particles_assign_ptr_to_sigma )( NS( Particles ) *
                                                            SIXTRL_RESTRICT p,
                                                        SIXTRL_REAL_T* ptr_sigma )
{
    SIXTRL_ASSERT( p != 0 );
    p->sigma = ptr_sigma;
    return;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE void NS( Particles_set_psigma_value )(
    NS( Particles ) * SIXTRL_RESTRICT p, SIXTRL_UINT64_T const id, SIXTRL_REAL_T psigma )
{
    SIXTRL_ASSERT( ( p != 0 ) && ( id < p->npart ) && ( p->psigma != 0 ) );
    p->psigma[id] = psigma;
    return;
}

SIXTRL_INLINE void
    NS( Particles_set_psigma )( NS( Particles ) * SIXTRL_RESTRICT p,
                                SIXTRL_REAL_T const* SIXTRL_RESTRICT ptr_psigma )
{
    SIXTRACKLIB_COPY_VALUES( SIXTRL_REAL_T, p->psigma, ptr_psigma, p->npart );
    return;
}

SIXTRL_INLINE void NS( Particles_assign_ptr_to_psigma )( NS( Particles ) *
                                                             SIXTRL_RESTRICT p,
                                                         SIXTRL_REAL_T* ptr_psigma )
{
    SIXTRL_ASSERT( p != 0 );
    p->psigma = ptr_psigma;
    return;
}

SIXTRL_INLINE void NS( Particles_set_delta_value )(
    NS( Particles ) * SIXTRL_RESTRICT p, SIXTRL_UINT64_T const id, SIXTRL_REAL_T delta )
{
    SIXTRL_ASSERT( ( p != 0 ) && ( id < p->npart ) && ( p->p0c != 0 ) );
    p->delta[id] = delta;
    return;
}

SIXTRL_INLINE void
    NS( Particles_set_delta )( NS( Particles ) * SIXTRL_RESTRICT p,
                               SIXTRL_REAL_T const* SIXTRL_RESTRICT ptr_delta )
{
    SIXTRACKLIB_COPY_VALUES( SIXTRL_REAL_T, p->delta, ptr_delta, p->npart );
    return;
}

SIXTRL_INLINE void NS( Particles_assign_ptr_to_delta )( NS( Particles ) *
                                                            SIXTRL_RESTRICT p,
                                                        SIXTRL_REAL_T* ptr_delta )
{
    SIXTRL_ASSERT( p != 0 );
    p->delta = ptr_delta;
    return;
}

SIXTRL_INLINE void NS( Particles_set_rpp_value )(
    NS( Particles ) * SIXTRL_RESTRICT p, SIXTRL_UINT64_T const id, SIXTRL_REAL_T rpp )
{
    SIXTRL_ASSERT( ( p != 0 ) && ( id < p->npart ) && ( p->rpp != 0 ) );
    p->rpp[id] = rpp;
    return;
}

SIXTRL_INLINE void
    NS( Particles_set_rpp )( NS( Particles ) * SIXTRL_RESTRICT p,
                             SIXTRL_REAL_T const* SIXTRL_RESTRICT ptr_rpp )
{
    SIXTRACKLIB_COPY_VALUES( SIXTRL_REAL_T, p->rpp, ptr_rpp, p->npart );
    return;
}

SIXTRL_INLINE void NS( Particles_assign_ptr_to_rpp )( NS( Particles ) *
                                                          SIXTRL_RESTRICT p,
                                                      SIXTRL_REAL_T* ptr_rpp )
{
    SIXTRL_ASSERT( p != 0 );
    p->rpp = ptr_rpp;
    return;
}

SIXTRL_INLINE void NS( Particles_set_rvv_value )(
    NS( Particles ) * SIXTRL_RESTRICT p, SIXTRL_UINT64_T const id, SIXTRL_REAL_T rvv )
{
    SIXTRL_ASSERT( ( p != 0 ) && ( id < p->npart ) && ( p->rvv != 0 ) );
    p->rvv[id] = rvv;
    return;
}

SIXTRL_INLINE void
    NS( Particles_set_rvv )( NS( Particles ) * SIXTRL_RESTRICT p,
                             SIXTRL_REAL_T const* SIXTRL_RESTRICT ptr_rvv )
{
    SIXTRACKLIB_COPY_VALUES( SIXTRL_REAL_T, p->rvv, ptr_rvv, p->npart );
    return;
}

SIXTRL_INLINE void NS( Particles_assign_ptr_to_rvv )( NS( Particles ) *
                                                          SIXTRL_RESTRICT p,
                                                      SIXTRL_REAL_T* ptr_rvv )
{
    SIXTRL_ASSERT( p != 0 );
    p->rvv = ptr_rvv;
    return;
}

SIXTRL_INLINE void NS( Particles_set_chi_value )(
    NS( Particles ) * SIXTRL_RESTRICT p, SIXTRL_UINT64_T const id, SIXTRL_REAL_T chi )
{
    SIXTRL_ASSERT( ( p != 0 ) && ( id < p->npart ) && ( p->chi != 0 ) );
    p->chi[id] = chi;
    return;
}

SIXTRL_INLINE void
    NS( Particles_set_chi )( NS( Particles ) * SIXTRL_RESTRICT p,
                             SIXTRL_REAL_T const* SIXTRL_RESTRICT ptr_chi )
{
    SIXTRACKLIB_COPY_VALUES( SIXTRL_REAL_T, p->chi, ptr_chi, p->npart );
    return;
}

SIXTRL_INLINE void NS( Particles_assign_ptr_to_chi )( NS( Particles ) *
                                                          SIXTRL_RESTRICT p,
                                                      SIXTRL_REAL_T* ptr_chi )
{
    SIXTRL_ASSERT( p != 0 );
    p->chi = ptr_chi;
    return;
}

/* ========================================================================= */

#if !defined( _GPUCODE )

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* !defined( _GPUCODE ) */

#endif /* SIXTRACKLIB_COMMON_IMPL_PARTICLES_TYPE_H__ */

/* end: sixtracklib/sixtracklib/common/impl/particles_type.h */
