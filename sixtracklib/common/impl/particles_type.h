#ifndef SIXTRACKLIB_COMMON_IMPL_PARTICLES_TYPE_H__
#define SIXTRACKLIB_COMMON_IMPL_PARTICLES_TYPE_H__

#if !defined( _GPUCODE )

#include "sixtracklib/_impl/namespace_begin.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include "sixtracklib/common/restrict.h"
#include "sixtracklib/_impl/inline.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */
    
#endif /* !defined( _GPUCODE ) */
    
typedef struct NS(Particles)
{
    uint64_t    npart;
        
    double* q0;     /* C */
    double* mass0;  /* eV */
    double* beta0;  /* nounit */
    double* gamma0; /* nounit */
    double* p0c;    /* eV */

    /* coordinate arrays */
    int64_t* partid;
    int64_t* elemid; /* element at which the particle was lost */
    int64_t* turn;   /* turn at which the particle was lost */
    int64_t* state;  /* negative means particle lost */
    
    double* s;       /* [m] */
    double* x;       /* [m] */
    double* px;      /* Px/P0 */
    double* y;       /* [m] */
    double* py;      /* Py/P0 */
    double* sigma;   /* s-beta0*c*t  where t is the time
                        since the beginning of the simulation */
                        
    double* psigma;  /* (E-E0) / (beta0 P0c) conjugate of sigma */
    double* delta;   /* P/P0-1 = 1/rpp-1 */
    double* rpp;     /* ratio P0 /P */
    double* rvv;     /* ratio beta / beta0 */
    double* chi;     /* q/q0 * m/m0  */
    
    uint64_t flags;           /* particle flags */
    void*    ptr_mem_context; /* memory_context -> can contain */
}
NS(Particles);

static int64_t const NS(PARTICLE_VALID_STATE) = 
    INT64_C( 0 );

static uint64_t const NS(PARTICLES_FLAGS_NONE) = 
    UINT64_C( 0x0000 );
    
static uint64_t const NS(PARTICLES_FLAGS_PACKED) = 
    UINT64_C( 0x0001 );
    
static uint64_t const NS(PARTICLES_FLAGS_OWNS_MEMORY) = 
    UINT64_C( 0x0002 );

static uint64_t const NS(PARTICLES_FLAGS_MEM_CTX_MEMPOOL) = 
    UINT64_C( 0x0010 );

static uint64_t const NS(PARTICLES_FLAGS_MEM_CTX_SINGLEPARTICLE) = 
    UINT64_C( 0x0020 );
    
static uint64_t const NS(PARTICLES_FLAGS_ALIGN_MASK) = 
    UINT64_C( 0xFFFF00 );
    
static uint64_t const NS(PARTICLES_MAX_ALIGNMENT) = 
    UINT64_C( 0xFFFF );
    
static uint64_t const NS(PARTICLES_FLAGS_ALIGN_MASK_OFFSET_BITS) = 
    UINT64_C( 8 );

/* ========================================================================= */

static size_t const NS(PARTICLES_DEFAULT_CHUNK_SIZE)     = ( size_t )8u;
static size_t const NS(PARTICLES_DEFAULT_ALIGNMENT)      = ( size_t )16u;
static size_t const NS(PARTICLES_NUM_OF_DOUBLE_ELEMENTS) = ( size_t )16u;
static size_t const NS(PARTICLES_NUM_OF_INT64_ELEMENTS)  = ( size_t )4u;

static uint64_t NS(Particles_get_capacity_for_size)( 
    size_t num_particles, 
    size_t* SIXTRL_RESTRICT chunk_size, 
    size_t* SIXTRL_RESTRICT alignment );

/* ========================================================================= */

static uint64_t NS(Particles_get_size)(
    const struct NS(Particles) *const SIXTRL_RESTRICT p );

static void NS(Particles_set_size)( 
    struct NS(Particles)* SIXTRL_RESTRICT p, uint64_t npart );

static uint64_t NS(Particles_get_flags)(
    const struct NS(Particles) *const SIXTRL_RESTRICT p );

static void NS(Particles_set_flags)(
    struct NS(Particles)* SIXTRL_RESTRICT p, uint64_t flags );

static void const* NS(Particles_get_const_ptr_mem_context)(
    const struct NS(Particles) *const SIXTRL_RESTRICT p );

static void* NS(Particles_get_ptr_mem_context)(
    struct NS(Particles)* SIXTRL_RESTRICT p );
    
static void NS(Particles_set_ptr_mem_context)(
    struct NS(Particles)* SIXTRL_RESTRICT p, void* ptr_mem_context );

/* ========================================================================= */

static double NS(Particles_get_q0_value)( 
    const NS(Particles) *const SIXTRL_RESTRICT p, uint64_t id );

static double const* NS(Particles_get_q0)( 
    const NS(Particles) *const SIXTRL_RESTRICT p );


static double NS(Particles_get_mass0_value)( 
    const NS(Particles) *const SIXTRL_RESTRICT p, uint64_t id );

static double const* NS(Particles_get_mass0)( 
    const NS(Particles) *const SIXTRL_RESTRICT p );


static double NS(Particles_get_beta0_value)( 
    const NS(Particles) *const SIXTRL_RESTRICT p, uint64_t id );

static double const* NS(Particles_get_beta0)( 
    const NS(Particles) *const SIXTRL_RESTRICT p );


static double NS(Particles_get_gamma0_value)( 
    const NS(Particles) *const SIXTRL_RESTRICT p, uint64_t id );

static double const* NS(Particles_get_gamma0)( 
    const NS(Particles) *const SIXTRL_RESTRICT p );


static double NS(Particles_get_p0c_value)( 
    const NS(Particles) *const SIXTRL_RESTRICT p, uint64_t id );

static double const* NS(Particles_get_p0c)( 
    const NS(Particles) *const SIXTRL_RESTRICT p );


/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

static int NS(Particles_is_particle_lost)( 
    const NS(Particles) *const SIXTRL_RESTRICT p, uint64_t id );


static int64_t NS(Particles_get_particle_id_value )(
    const NS(Particles) *const SIXTRL_RESTRICT p, uint64_t id );

static int64_t const* NS(Particles_get_particle_id )(
    const NS(Particles) *const SIXTRL_RESTRICT p );


static int64_t NS(Particles_get_lost_at_element_id_value)(
    const NS(Particles) *const SIXTRL_RESTRICT p, uint64_t id );

static int64_t const* NS(Particles_get_lost_at_element_id)(
    const NS(Particles) *const SIXTRL_RESTRICT p );


static int64_t NS(Particles_get_lost_at_turn_value)( 
    const NS(Particles) *const SIXTRL_RESTRICT p, uint64_t id );

static int64_t const* NS(Particles_get_lost_at_turn)( 
    const NS(Particles) *const SIXTRL_RESTRICT p );


static int64_t NS(Particles_get_state_value)( 
    const NS(Particles) *const SIXTRL_RESTRICT p, uint64_t id );

static int64_t const* NS(Particles_get_state)( 
    const NS(Particles) *const SIXTRL_RESTRICT p );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

static double NS( Particles_get_s_value)( 
    const NS(Particles) *const SIXTRL_RESTRICT p, uint64_t id );

static double const* NS( Particles_get_s)( 
    const NS(Particles) *const SIXTRL_RESTRICT p );


static double NS(Particles_get_x_value)( 
    const NS(Particles) *const SIXTRL_RESTRICT p, uint64_t id );

static double const* NS(Particles_get_x)( 
    const NS(Particles) *const SIXTRL_RESTRICT p );


static double NS(Particles_get_y_value)( 
    const NS(Particles) *const SIXTRL_RESTRICT p, uint64_t id );

static double const* NS(Particles_get_y)( 
    const NS(Particles) *const SIXTRL_RESTRICT p );


static double NS( Particles_get_px_value)( 
    const NS(Particles) *const SIXTRL_RESTRICT p, uint64_t id );

static double const* NS( Particles_get_px)( 
    const NS(Particles) *const SIXTRL_RESTRICT p );


static double NS( Particles_get_py_value)( 
    const NS(Particles) *const SIXTRL_RESTRICT p, uint64_t id );

static double const* NS( Particles_get_py)( 
    const NS(Particles) *const SIXTRL_RESTRICT p );


static double NS( Particles_get_sigma_value)( 
    const NS(Particles) *const SIXTRL_RESTRICT p, uint64_t id );

static double const* NS( Particles_get_sigma)( 
    const NS(Particles) *const SIXTRL_RESTRICT p );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

static double NS(Particles_get_psigma_value)( 
    const NS(Particles) *const SIXTRL_RESTRICT p, uint64_t id );

static double const* NS(Particles_get_psigma)( 
    const NS(Particles) *const SIXTRL_RESTRICT p );


static double NS(Particles_get_delta_value)( 
    const NS(Particles) *const SIXTRL_RESTRICT p, uint64_t id );

static double const* NS(Particles_get_delta)( 
    const NS(Particles) *const SIXTRL_RESTRICT p );


static double NS(Particles_get_rpp_value)( 
    const NS(Particles) *const SIXTRL_RESTRICT p, uint64_t id );

static double const* NS(Particles_get_rpp)( 
    const NS(Particles) *const SIXTRL_RESTRICT p );


static double NS(Particles_get_rvv_value)( 
    const NS(Particles) *const SIXTRL_RESTRICT p, uint64_t id );

static double const* NS(Particles_get_rvv)( 
    const NS(Particles) *const SIXTRL_RESTRICT p );


static double NS(Particles_get_chi_value)( 
    const NS(Particles) *const SIXTRL_RESTRICT p, uint64_t id );

static double const* NS(Particles_get_chi)( 
    const NS(Particles) *const SIXTRL_RESTRICT p );

/* ========================================================================= */

static void NS(Particles_set_q0_value)( 
    NS(Particles)* SIXTRL_RESTRICT p, uint64_t id, double q0 );

static void NS(Particles_set_q0)( 
    NS(Particles)* SIXTRL_RESTRICT p, 
    double  const* SIXTRL_RESTRICT ptr_q0 );

static void NS(Particles_assign_ptr_to_q0)( 
    NS(Particles)* SIXTRL_RESTRICT p,  double* ptr_q0 );



static void NS(Particles_set_mass0_value)( 
    NS(Particles)* SIXTRL_RESTRICT p, uint64_t id, double mass0 );

static void NS(Particles_set_mass0)( 
    NS(Particles)* SIXTRL_RESTRICT p, 
    double  const* SIXTRL_RESTRICT ptr_mass0 );

static void NS(Particles_assign_ptr_to_mass0)( 
    NS(Particles)* SIXTRL_RESTRICT p, double* ptr_mass0 );



static void NS(Particles_set_beta0_value)( 
    NS(Particles)* SIXTRL_RESTRICT p, uint64_t id, double beta0 );

static void NS(Particles_set_beta0)( 
    NS(Particles)* SIXTRL_RESTRICT p, 
    double  const* SIXTRL_RESTRICT ptr_beta0 );

static void NS(Particles_assign_ptr_to_beta0)( 
    NS(Particles)* SIXTRL_RESTRICT p, double* ptr_beta0 );



static void NS(Particles_set_gamma0_value)( 
    NS(Particles)* SIXTRL_RESTRICT p, uint64_t id, double gamma0 );

static void NS(Particles_set_gamma0)( 
    NS(Particles)* SIXTRL_RESTRICT p, 
    double  const* SIXTRL_RESTRICT ptr_gamma0 );

static void NS(Particles_assign_ptr_to_gamma0)( 
    NS(Particles)* SIXTRL_RESTRICT p, double* ptr_gamma0 );



static void NS(Particles_set_p0c_value)( 
    NS(Particles)* SIXTRL_RESTRICT p, uint64_t id, double p0c );

static void NS(Particles_set_p0c)( 
    NS(Particles)* SIXTRL_RESTRICT p, 
    double  const* SIXTRL_RESTRICT ptr_p0c );

static void NS(Particles_assign_ptr_to_p0c)( 
    NS(Particles)* SIXTRL_RESTRICT p, double* ptr_p0c );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

static void NS(Particles_set_particle_id_value)(
    NS(Particles)* SIXTRL_RESTRICT p, uint64_t id, int64_t partid );

static void NS(Particles_set_particle_id)(
    NS(Particles)* SIXTRL_RESTRICT p, 
    int64_t const* SIXTRL_RESTRICT ptr_partid );

static void NS(Particles_assign_ptr_to_particle_id)(
    NS(Particles)* SIXTRL_RESTRICT p, int64_t* ptr_partid );



static void NS(Particles_set_lost_at_element_id_value)(
    NS(Particles)* SIXTRL_RESTRICT p, uint64_t id, int64_t elemid );

static void NS(Particles_set_lost_at_element_id)(
    NS(Particles)* SIXTRL_RESTRICT p, 
    int64_t const* SIXTRL_RESTRICT ptr_elemid );

static void NS(Particles_assign_ptr_to_lost_at_element_id)(
    NS(Particles)* SIXTRL_RESTRICT p, int64_t* ptr_elemid );



static void NS(Particles_set_lost_at_turn_value)( 
    NS(Particles)* SIXTRL_RESTRICT p, uint64_t id, int64_t turn );

static void NS(Particles_set_lost_at_turn)( 
    NS(Particles)* SIXTRL_RESTRICT p, 
    int64_t const* SIXTRL_RESTRICT ptr_turn );

static void NS(Particles_assign_ptr_to_lost_at_turn)( 
    NS(Particles)* SIXTRL_RESTRICT p, int64_t* ptr_turn );



static void NS(Particles_set_state_value)( 
    NS(Particles)* SIXTRL_RESTRICT p, uint64_t id, int64_t state );

static void NS(Particles_set_state)( 
    NS(Particles)* SIXTRL_RESTRICT p, 
    int64_t const* SIXTRL_RESTRICT ptr_state );

static void NS(Particles_assign_ptr_to_state)( 
    NS(Particles)* SIXTRL_RESTRICT p, int64_t* ptr_state );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

static void NS( Particles_set_s_value )( 
    NS(Particles)* SIXTRL_RESTRICT p, uint64_t id, double s );

static void NS( Particles_set_s )( 
    NS(Particles)* SIXTRL_RESTRICT p, double const* SIXTRL_RESTRICT ptr_s );

static void NS( Particles_assign_ptr_to_s )( 
    NS(Particles)* SIXTRL_RESTRICT p, double* ptr_s );



static void NS(Particles_set_x_value )( 
    NS(Particles)* SIXTRL_RESTRICT p, uint64_t id, double x );

static void NS(Particles_set_x )( 
    NS(Particles)* SIXTRL_RESTRICT p, 
    double  const* SIXTRL_RESTRICT ptr_x );

static void NS(Particles_assign_ptr_to_x )( 
    NS(Particles)* SIXTRL_RESTRICT p, double* ptr_x );



static void NS(Particles_set_y_value )( 
    NS(Particles)* SIXTRL_RESTRICT p, uint64_t id, double y );

static void NS(Particles_set_y )( 
    NS(Particles)* SIXTRL_RESTRICT p,
    double  const* SIXTRL_RESTRICT ptr_y );

static void NS(Particles_assign_ptr_to_y )( 
    NS(Particles)* SIXTRL_RESTRICT p, double* ptr_y );



static void NS( Particles_set_px_value )( 
    NS(Particles)* SIXTRL_RESTRICT p, uint64_t id, double px );

static void NS( Particles_set_px )( 
    NS(Particles)* SIXTRL_RESTRICT p, 
    double  const* SIXTRL_RESTRICT ptr_px );

static void NS( Particles_assign_ptr_to_px )( 
    NS(Particles)* SIXTRL_RESTRICT p, double* ptr_px );



static void NS( Particles_set_py_value )( 
    NS(Particles)* SIXTRL_RESTRICT p, uint64_t id, double py );

static void NS( Particles_set_py )( 
    NS(Particles)* SIXTRL_RESTRICT p, 
    double  const* SIXTRL_RESTRICT ptr_py );

static void NS( Particles_assign_ptr_to_py )( 
    NS(Particles)* SIXTRL_RESTRICT p, double* ptr_py );


static void NS( Particles_set_sigma_value )( 
    NS(Particles)* SIXTRL_RESTRICT p, uint64_t id, double sigma );

static void NS( Particles_set_sigma )( 
    NS(Particles)* SIXTRL_RESTRICT p, 
    double  const* SIXTRL_RESTRICT ptr_sigma );

static void NS( Particles_assign_ptr_to_sigma )( 
    NS(Particles)* SIXTRL_RESTRICT p, double* ptr_sigma );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

static void NS(Particles_set_psigma_value)( 
    NS(Particles)* SIXTRL_RESTRICT p, uint64_t id, double psigma );

static void NS(Particles_set_psigma)( 
    NS(Particles)* SIXTRL_RESTRICT p, 
    double  const* SIXTRL_RESTRICT ptr_psigma );

static void NS(Particles_assign_ptr_to_psigma)( 
    NS(Particles)* SIXTRL_RESTRICT p, double* ptr_psigma );



static void NS(Particles_set_delta_value)( 
    NS(Particles)* SIXTRL_RESTRICT p, uint64_t id, double delta );

static void NS(Particles_set_delta)( 
    NS(Particles)* SIXTRL_RESTRICT p, 
    double  const* SIXTRL_RESTRICT ptr_delta );

static void NS(Particles_assign_ptr_to_delta)( 
    NS(Particles)* SIXTRL_RESTRICT p, double* ptr_delta );



static void NS(Particles_set_rpp_value)( 
    NS(Particles)* SIXTRL_RESTRICT p, uint64_t id, double rpp );

static void NS(Particles_set_rpp)( 
    NS(Particles)* SIXTRL_RESTRICT p, 
    double  const* SIXTRL_RESTRICT ptr_rpp );

static void NS(Particles_assign_ptr_to_rpp)( 
    NS(Particles)* SIXTRL_RESTRICT p, double* ptr_rpp );



static void NS(Particles_set_rvv_value)( 
    NS(Particles)* SIXTRL_RESTRICT p, uint64_t id, double rvv );

static void NS(Particles_set_rvv)( 
    NS(Particles)* SIXTRL_RESTRICT p, 
    double  const* SIXTRL_RESTRICT ptr_rvv );

static void NS(Particles_assign_ptr_to_rvv)( 
    NS(Particles)* SIXTRL_RESTRICT p, double* ptr_rvv );



static void NS(Particles_set_chi_value)( 
    NS(Particles)* SIXTRL_RESTRICT p, uint64_t id, double chi );

static void NS(Particles_set_chi)( 
    NS(Particles)* SIXTRL_RESTRICT p, 
    double  const* SIXTRL_RESTRICT ptr_chi );

static void NS(Particles_assign_ptr_to_chi)( 
    NS(Particles)* SIXTRL_RESTRICT p, double* ptr_chi );

/* ========================================================================= *
 * ==== IMPLEMENTATION OF INLINE FUNCTIONS 
 * ========================================================================= */

#if !defined( SIXTRACKLIB_COPY_VALUES )    
    #if defined( _GPUCODE )
        #define SIXTRACKLIB_COPY_VALUES( T, dest, source, n ) \
            /* ----------------------------------------------------------- */ \
            /* ---- Inside SIXTRACKLIB_COPY_VALUES (ELEMENTWISE)       --- */ \
            /* ----------------------------------------------------------- */ \
            \
            for( uint64_t __ii = 0 ; __ii < ( n ) ; ++__ii ) \
            { \
                *( ( dest ) + __ii ) = *( ( source ) + __ii ); \
            } \
            \
            /* ----------------------------------------------------------- */ \
            /* ---- End Of SIXTRACKLIB_COPY_VALUES (ELEMENTWISE)       --- */ \
            /* ----------------------------------------------------------- */ \
            
            
    #elif !defined( _GPUCODE )
        #define SIXTRACKLIB_COPY_VALUES( T, dest, source, n ) \
            /* ----------------------------------------------------------- */ \
            /* ----  Inside SIXTRACKLIB_COPY_VALUES (MEMCPY BASED)    ---- */ \
            /* ----------------------------------------------------------- */ \
            \
            assert( ( ( dest ) != 0 ) && ( ( source ) != 0 ) && \
                    ( ( n ) > UINT64_C( 0 ) ) ); \
            \
            memcpy( ( dest ), ( source ), sizeof( T ) * ( n ) ); \
            \
            /* ----------------------------------------------------------- */ \
            /* ----  End Of SIXTRACKLIB_COPY_VALUES (MEMCPY BASED)    ---- */ \
            /* ----------------------------------------------------------- */ \
            
            
    #endif /* defined( _GPUCODE ) */
#endif /* defined( SIXTRACKLIB_COPY_VALUES ) */


#if !defined( SIXTRACKLIB_ASSIGN_PTR_TO_MEMBER )

    #if defined( _GPUCODE )
    
        #define SIXTRACKLIB_ASSIGN_PTR_TO_MEMBER( base, dest, source ) \
            /* ----------------------------------------------------------- */ \
            /* ----  Inside SIXTRACKLIB_ASSIGN_PTR_TO_MEMBER (GPU)    ---- */ \
            /* ----------------------------------------------------------- */ \
            \
            ( base )->dest = ( source ); \
            \
            /* ----------------------------------------------------------- */ \
            /* ----  End Of SIXTRACKLIB_ASSIGN_PTR_TO_MEMBER (GPU)    ---- */ \
            /* ----------------------------------------------------------- */ \
            
    #else 
        #define SIXTRACKLIB_ASSIGN_PTR_TO_MEMBER( base, dest, source ) \
            /* ----------------------------------------------------------- */ \
            /* ----  Inside SIXTRACKLIB_ASSIGN_PTR_TO_MEMBER (GPU)    ---- */ \
            /* ----------------------------------------------------------- */ \
            \
            assert( ( base ) != 0 ); \
            ( base )->dest = ( source ); \
            \
            /* ----------------------------------------------------------- */ \
            /* ----  End Of SIXTRACKLIB_ASSIGN_PTR_TO_MEMBER (GPU)    ---- */ \
            /* ----------------------------------------------------------- */ \
        
    #endif /* !defined( _GPUCODE ) */

#endif /* !defined( SIXTRACKLIB_ASSIGN_PTR_TO_MEMBER ) */

/* ========================================================================= */

SIXTRL_INLINE size_t NS(Particles_get_capacity_for_size)( 
    size_t num_particles, 
    size_t* SIXTRL_RESTRICT ptr_chunk_size, 
    size_t* SIXTRL_RESTRICT ptr_alignment )
{
    static size_t const ZERO_SIZE = ( size_t )0u;
    
    size_t predicted_capacity = ZERO_SIZE;           
    
    if( ( num_particles > ZERO_SIZE ) && 
        ( ptr_chunk_size != 0 ) && ( ptr_alignment != 0 ) )
    {
        size_t double_elem_length = sizeof( double  ) * num_particles;
        size_t int64_elem_length  = sizeof( int64_t ) * num_particles;
        
        size_t chunk_size = *ptr_chunk_size;
        size_t alignment  = *ptr_alignment;
        
        assert( ptr_chunk_size != ptr_alignment );
        
        if( chunk_size == ZERO_SIZE )
        {
            chunk_size = NS(PARTICLES_DEFAULT_CHUNK_SIZE);
        }
        
        assert( chunk_size <= NS(PARTICLES_MAX_ALIGNMENT) );
        
        if( alignment == ZERO_SIZE )
        {
            alignment = NS(PARTICLES_DEFAULT_ALIGNMENT);
        }
        
        if( alignment < chunk_size )
        {
            alignment = chunk_size;
        }
        
        if( ( alignment % chunk_size ) != ZERO_SIZE )
        {
            alignment = chunk_size + ( ( alignment / chunk_size ) * chunk_size );
        }
        
        assert( (   alignment <= NS(PARTICLES_MAX_ALIGNMENT) ) &&
                (   alignment >= chunk_size ) &&
                ( ( alignment %  chunk_size ) == ZERO_SIZE ) );
        
        /* ----------------------------------------------------------------- */
        
        size_t temp = ( double_elem_length / ( alignment ) ) * ( alignment );
        
        if( temp < double_elem_length )
        {
            temp += alignment;
        }
        
        double_elem_length = temp;
        
        /* ----------------------------------------------------------------- */
        
        temp = ( int64_elem_length / ( alignment ) ) * ( alignment );
        
        if( temp < int64_elem_length )
        {
            temp += alignment;
        }
        
        int64_elem_length = temp;
        
        /* ----------------------------------------------------------------- */
        
        assert( (   double_elem_length > ZERO_SIZE ) &&
                ( ( double_elem_length % alignment ) == ZERO_SIZE ) &&
                (   int64_elem_length  > ZERO_SIZE ) &&
                ( ( int64_elem_length  % alignment ) == ZERO_SIZE ) );
        
        predicted_capacity = 
            NS(PARTICLES_NUM_OF_DOUBLE_ELEMENTS) * double_elem_length +
            NS(PARTICLES_NUM_OF_INT64_ELEMENTS)  * int64_elem_length;
        
        /* By aligning every individual member of the Particles struct to the 
         * required alignment, we can ensure that the whole block used for 
         * packing the data will be aligned internall. We have, however, to 
         * account for the possibility that the initial address of the whole
         * memory region will be not properly aligned -> increase the capacity 
         * by the alignment to allow for some wiggle room here */
            
        predicted_capacity += alignment;        
        
        *ptr_chunk_size = chunk_size;
        *ptr_alignment  = alignment;
    }
    
    return predicted_capacity;
}

/* ========================================================================= */

SIXTRL_INLINE uint64_t NS(Particles_get_size)(
    const struct NS(Particles) *const SIXTRL_RESTRICT p )
{
    return ( p != 0 ) ? p->npart : UINT64_C( 0 );
}

SIXTRL_INLINE void NS(Particles_set_size)(
    struct NS(Particles)* SIXTRL_RESTRICT p, uint64_t npart )
{
    #if !defined( _GPUCODE )
    assert( p != 0 );
    #endif /* !defiend( _GPUCODE ) */
    
    p->npart = npart;
    return;
}

SIXTRL_INLINE uint64_t NS(Particles_get_flags)(
    const struct NS(Particles) *const SIXTRL_RESTRICT p )
{
    return ( p != 0 ) ? p->flags : NS(PARTICLES_FLAGS_NONE);
}

SIXTRL_INLINE void NS(Particles_set_flags)(
    struct NS(Particles)* SIXTRL_RESTRICT p, uint64_t flags )
{
    #if !defined( _GPUCODE )
    assert( p != 0 );
    #endif /* !defiend( _GPUCODE ) */
    
    p->flags = flags;
    return;
}

SIXTRL_INLINE void const* NS(Particles_get_const_ptr_mem_context)(
    const struct NS(Particles) *const SIXTRL_RESTRICT p )
{
    return ( p != 0 ) ? p->ptr_mem_context : 0;
}

SIXTRL_INLINE void* NS(Particles_get_ptr_mem_context)(
    struct NS(Particles)* SIXTRL_RESTRICT p )
{
    /* casting away const-ness of a pointer is ok even in C */
    return ( void* )NS(Particles_get_const_ptr_mem_context)( p );
}

SIXTRL_INLINE void NS(Particles_set_ptr_mem_context)(
    struct NS(Particles)* SIXTRL_RESTRICT p, void* ptr_mem_context )
{
    #if !defined( _GPUCODE )
    assert( p != 0 );
    #endif /* !defiend( _GPUCODE ) */
    
    p->ptr_mem_context = ptr_mem_context;
    return;
}

/* ========================================================================= */

SIXTRL_INLINE double NS(Particles_get_q0_value)( 
    const NS(Particles) *const SIXTRL_RESTRICT p, uint64_t id )
{
    #if !defined( _GPUCODE )
    assert( ( p != 0 ) && ( id < p->npart ) && ( p->q0 != 0 ) );
    #endif /* !defined( _GPUCODE ) */
    
    return p->q0[ id ];
}

SIXTRL_INLINE double const* NS(Particles_get_q0)( 
    const NS(Particles) *const SIXTRL_RESTRICT p )
{
    return ( p != 0 ) ? p->q0 : 0;
}



SIXTRL_INLINE double NS(Particles_get_mass0_value)( 
    const NS(Particles) *const SIXTRL_RESTRICT p, uint64_t id )
{
    #if !defined( _GPUCODE )
    assert( ( p != 0 ) && ( id < p->npart ) && ( p->mass0 != 0 ) );
    #endif /* !defined( _GPUCODE ) */
    
    return p->mass0[ id ];
}

SIXTRL_INLINE double const* NS(Particles_get_mass0)( 
    const NS(Particles) *const SIXTRL_RESTRICT p )
{
    return ( p != 0 ) ? p->mass0 : 0;
}



SIXTRL_INLINE double NS(Particles_get_beta0_value)( 
    const NS(Particles) *const SIXTRL_RESTRICT p, uint64_t id )
{
    #if !defined( _GPUCODE )
    assert( ( p != 0 ) && ( id < p->npart ) && ( p->beta0 != 0 ) );
    #endif /* !defined( _GPUCODE ) */
    
    return p->beta0[ id ];
}

SIXTRL_INLINE double const* NS(Particles_get_beta0)( 
    const NS(Particles) *const SIXTRL_RESTRICT p )
{
    return ( p != 0 ) ? p->beta0 : 0;
}



SIXTRL_INLINE double NS(Particles_get_gamma0_value)( 
    const NS(Particles) *const SIXTRL_RESTRICT p, uint64_t id )
{
    #if !defined( _GPUCODE )
    assert( ( p != 0 ) && ( id < p->npart ) && ( p->gamma0 != 0 ) );
    #endif /* !defined( _GPUCODE ) */
    
    return p->gamma0[ id ];
}

SIXTRL_INLINE double const* NS(Particles_get_gamma0)( 
    const NS(Particles) *const SIXTRL_RESTRICT p )
{
    return ( p != 0 ) ? p->gamma0 : 0;
}



SIXTRL_INLINE double NS(Particles_get_p0c_value)( 
    const NS(Particles) *const SIXTRL_RESTRICT p, uint64_t id )
{
    #if !defined( _GPUCODE )
    assert( ( p != 0 ) && ( id < p->npart ) && ( p->p0c != 0 ) );
    #endif /* !defined( _GPUCODE ) */
    
    return p->p0c[ id ];
}

SIXTRL_INLINE double const* NS(Particles_get_p0c)( 
    const NS(Particles) *const SIXTRL_RESTRICT p )
{
    return ( p != 0 ) ? p->p0c : 0;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE int NS(Particles_is_particle_lost)( 
    const NS(Particles) *const p, uint64_t id )
{
    #if !defined( _GPUCODE )
    assert( ( p != 0 ) && ( id < p->npart ) && ( p->state != 0 ) );
    #endif /* !defined( _GPUCODE ) */
    
    return ( p->state[ id ] < 0 ) ? 1 : 0;
}


SIXTRL_INLINE int64_t NS(Particles_get_particle_id_value)(
    const NS(Particles) *const SIXTRL_RESTRICT p, uint64_t id )
{
    #if !defined( _GPUCODE )
    assert( ( p != 0 ) && ( id < p->npart ) && ( p->partid != 0 ) );
    #endif /* !defined( _GPUCODE ) */
    
    return p->partid[ id ];
}

SIXTRL_INLINE int64_t const* NS(Particles_get_particle_id)(
    const NS(Particles) *const SIXTRL_RESTRICT p )
{
    return ( p != 0 ) ? p->partid : 0;
}



SIXTRL_INLINE int64_t NS(Particles_get_lost_at_element_id_value)(
    const NS(Particles) *const p, uint64_t id )
{
    #if !defined( _GPUCODE )
    assert( ( p != 0 ) && ( id < p->npart ) && ( p->elemid != 0 ) );
    #endif /* !defined( _GPUCODE ) */
    
    return p->elemid[ id ];
}

SIXTRL_INLINE int64_t const* NS(Particles_get_lost_at_element_id)(
    const NS(Particles) *const p )
{
    return ( p != 0 ) ? p->elemid : 0;
}



SIXTRL_INLINE int64_t NS(Particles_get_lost_at_turn_value)( 
    const NS(Particles) *const p, uint64_t id )
{
    #if !defined( _GPUCODE )
    assert( ( p != 0 ) && ( id < p->npart ) && ( p->turn != 0 ) );
    #endif /* !defined( _GPUCODE ) */
    
    return p->turn[ id ];
}

SIXTRL_INLINE int64_t const* NS(Particles_get_lost_at_turn)( 
    const NS(Particles) *const p )
{
    return ( p != 0 ) ? p->turn : 0;
}



SIXTRL_INLINE int64_t NS(Particles_get_state_value)( 
    const NS(Particles) *const p, uint64_t id )
{
    #if !defined( _GPUCODE )
    assert( ( p != 0 ) && ( id < p->npart ) && ( p->state != 0 ) );
    #endif /* !defined( _GPUCODE ) */
    
    return p->state[ id ];
}

SIXTRL_INLINE int64_t const* NS(Particles_get_state)( 
    const NS(Particles) *const p )
{
    return ( p != 0 ) ? p->state : 0;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE double NS(Particles_get_s_value)( 
    const NS(Particles) *const SIXTRL_RESTRICT p, uint64_t id )
{
    #if !defined( _GPUCODE )
    assert( ( p != 0 ) && ( id < p->npart ) && ( p->s != 0 ) );
    #endif /* !defined( _GPUCODE ) */
    
    return p->s[ id ];
}

SIXTRL_INLINE double const* NS(Particles_get_s )( 
    const NS(Particles) *const SIXTRL_RESTRICT p )
{
    return ( p != 0 ) ? p->s : 0;
}



SIXTRL_INLINE double NS(Particles_get_x_value )( 
    const NS(Particles) *const SIXTRL_RESTRICT p, uint64_t id )
{
    #if !defined( _GPUCODE )
    assert( ( p != 0 ) && ( id < p->npart ) && ( p->x != 0 ) );
    #endif /* !defined( _GPUCODE ) */
    
    return p->x[ id ];
}

SIXTRL_INLINE double const* NS(Particles_get_x )( 
    const NS(Particles) *const SIXTRL_RESTRICT p )
{
    return ( p != 0 ) ? p->x : 0;
}



SIXTRL_INLINE double NS(Particles_get_y_value )( 
    const NS(Particles) *const SIXTRL_RESTRICT p, uint64_t id )
{
    #if !defined( _GPUCODE )
    assert( ( p != 0 ) && ( id < p->npart ) && ( p->y != 0 ) );
    #endif /* !defined( _GPUCODE ) */
    
    return p->y[ id ];
}

SIXTRL_INLINE double const* NS(Particles_get_y )( 
    const NS(Particles) *const SIXTRL_RESTRICT p )
{
    return ( p != 0 ) ? p->y : 0;
}



SIXTRL_INLINE double NS( Particles_get_px_value )( 
    const NS(Particles) *const SIXTRL_RESTRICT p, uint64_t id )
{
    #if !defined( _GPUCODE )
    assert( ( p != 0 ) && ( id < p->npart ) && ( p->px != 0 ) );
    #endif /* !defined( _GPUCODE ) */
    
    return p->px[ id ];
}

SIXTRL_INLINE double const* NS( Particles_get_px )( 
    const NS(Particles) *const SIXTRL_RESTRICT p )
{
    return ( p != 0 ) ? p->px : 0;
}



SIXTRL_INLINE double NS( Particles_get_py_value )( 
    const NS(Particles) *const SIXTRL_RESTRICT p, uint64_t id )
{
    #if !defined( _GPUCODE )
    assert( ( p != 0 ) && ( id < p->npart ) && ( p->py != 0 ) );
    #endif /* !defined( _GPUCODE ) */
    
    return p->py[ id ];
}

SIXTRL_INLINE double const* NS( Particles_get_py )( 
    const NS(Particles) *const SIXTRL_RESTRICT p )
{
    return ( p != 0 ) ? p->py : 0;
}



SIXTRL_INLINE double NS( Particles_get_sigma_value )( 
    const NS(Particles) *const SIXTRL_RESTRICT p, uint64_t id )
{
    #if !defined( _GPUCODE )
    assert( ( p != 0 ) && ( id < p->npart ) && ( p->sigma != 0 ) );
    #endif /* !defined( _GPUCODE ) */
    
    return p->sigma[ id ];
}

SIXTRL_INLINE double const* NS( Particles_get_sigma )( 
    const NS(Particles) *const SIXTRL_RESTRICT p )
{
    return ( p != 0 ) ? p->sigma : 0;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE double NS(Particles_get_psigma_value)( 
    const NS(Particles) *const SIXTRL_RESTRICT p, uint64_t id )
{
    #if !defined( _GPUCODE )
    assert( ( p != 0 ) && ( id < p->npart ) && ( p->psigma != 0 ) );
    #endif /* !defined( _GPUCODE ) */
    
    return p->psigma[ id ];
}

SIXTRL_INLINE double const* NS(Particles_get_psigma)( 
    const NS(Particles) *const SIXTRL_RESTRICT p )
{
    return ( p != 0 ) ? p->psigma : 0;
}



SIXTRL_INLINE double NS(Particles_get_delta_value)( 
    const NS(Particles) *const SIXTRL_RESTRICT p, uint64_t id )
{
    #if !defined( _GPUCODE )
    assert( ( p != 0 ) && ( id < p->npart ) && ( p->delta != 0 ) );
    #endif /* !defined( _GPUCODE ) */
    
    return p->delta[ id ];
}

SIXTRL_INLINE double const* NS(Particles_get_delta)( 
    const NS(Particles) *const SIXTRL_RESTRICT p )
{
    return ( p != 0 ) ? p->delta : 0;
}



SIXTRL_INLINE double NS(Particles_get_rpp_value)( 
    const NS(Particles) *const SIXTRL_RESTRICT p, uint64_t id )
{
    #if !defined( _GPUCODE )
    assert( ( p != 0 ) && ( id < p->npart ) && ( p->rpp != 0 ) );
    #endif /* !defined( _GPUCODE ) */
    
    return p->rpp[ id ];
}

SIXTRL_INLINE double const* NS(Particles_get_rpp)( 
    const NS(Particles) *const SIXTRL_RESTRICT p )
{
    return ( p != 0 ) ? p->rpp : 0;
}



SIXTRL_INLINE double NS(Particles_get_rvv_value)( 
    const NS(Particles) *const SIXTRL_RESTRICT p, uint64_t id )
{
    #if !defined( _GPUCODE )
    assert( ( p != 0 ) && ( id < p->npart ) && ( p->rvv != 0 ) );
    #endif /* !defined( _GPUCODE ) */
    
    return p->rvv[ id ];
}

SIXTRL_INLINE double const* NS(Particles_get_rvv)( 
    const NS(Particles) *const SIXTRL_RESTRICT p )
{
    return ( p != 0 ) ? p->rvv : 0;
}



SIXTRL_INLINE double NS(Particles_get_chi_value)( 
    const NS(Particles) *const SIXTRL_RESTRICT p, uint64_t id )
{
    #if !defined( _GPUCODE )
    assert( ( p != 0 ) && ( id < p->npart ) && ( p->chi != 0 ) );
    #endif /* !defined( _GPUCODE ) */
    
    return p->chi[ id ];
}

SIXTRL_INLINE double const* NS(Particles_get_chi)( 
    const NS(Particles) *const SIXTRL_RESTRICT p )
{
    return ( p != 0 ) ? p->chi : 0;
}

/* ========================================================================= */

SIXTRL_INLINE void NS(Particles_set_q0_value)( 
    NS(Particles)* SIXTRL_RESTRICT p, uint64_t id, double q0 )
{
    #if !defined( _GPUCODE )
    assert( ( p != 0 ) && ( id < p->npart ) && ( p->q0 != 0 ) );
    #endif /* !defiend( _GPUCODE ) */
    
    p->q0[ id ] = q0;
    return;
}

SIXTRL_INLINE void NS(Particles_set_q0)( 
    NS(Particles)* SIXTRL_RESTRICT p, double const* SIXTRL_RESTRICT ptr_q0 )
{
    SIXTRACKLIB_COPY_VALUES( double, p->q0, ptr_q0, p->npart )        
    return;
}

SIXTRL_INLINE void NS(Particles_assign_ptr_to_q0)( 
    NS(Particles)* SIXTRL_RESTRICT p, double* ptr_q0 )
{
    SIXTRACKLIB_ASSIGN_PTR_TO_MEMBER( p, q0, ptr_q0 )
    return;
}



SIXTRL_INLINE void NS(Particles_set_mass0_value)( 
    NS(Particles)* SIXTRL_RESTRICT p, uint64_t id, double mass0 )
{
    #if !defined( _GPUCODE )
    assert( ( p != 0 ) && ( id < p->npart ) && ( p->mass0 != 0 ) );
    #endif /* !defiend( _GPUCODE ) */
    
    p->mass0[ id ] = mass0;
    return;
}

SIXTRL_INLINE void NS(Particles_set_mass0)( 
    NS(Particles)* SIXTRL_RESTRICT p, double const* SIXTRL_RESTRICT ptr_mass0 )
{
    SIXTRACKLIB_COPY_VALUES( double, p->mass0, ptr_mass0, p->npart )
    return;
}

SIXTRL_INLINE void NS(Particles_assign_ptr_to_mass0)( 
    NS(Particles)* SIXTRL_RESTRICT p, double* ptr_mass0 )
{
    SIXTRACKLIB_ASSIGN_PTR_TO_MEMBER( p, mass0, ptr_mass0 )
    return;
}



SIXTRL_INLINE void NS(Particles_set_beta0_value)( 
    NS(Particles)* SIXTRL_RESTRICT p, uint64_t id, double beta0 )
{
    #if !defined( _GPUCODE )
    assert( ( p != 0 ) && ( id < p->npart ) && ( p->beta0 != 0 ) );
    #endif /* !defiend( _GPUCODE ) */
    
    p->beta0[ id ] = beta0;
    return;
}

SIXTRL_INLINE void NS(Particles_set_beta0)( 
    NS(Particles)* SIXTRL_RESTRICT p, double const* SIXTRL_RESTRICT ptr_beta0 )
{
    SIXTRACKLIB_COPY_VALUES( double, p->beta0, ptr_beta0, p->npart )    
    return;
}

SIXTRL_INLINE void NS(Particles_assign_ptr_to_beta0)( 
    NS(Particles)* SIXTRL_RESTRICT p, double* ptr_beta0 )
{
    SIXTRACKLIB_ASSIGN_PTR_TO_MEMBER( p, beta0, ptr_beta0 )
    return;
}



SIXTRL_INLINE void NS(Particles_set_gamma0_value)( 
    NS(Particles)* SIXTRL_RESTRICT p, uint64_t id, double gamma0 )
{
    #if !defined( _GPUCODE )
    assert( ( p != 0 ) && ( id < p->npart ) && ( p->gamma0 != 0 ) );
    #endif /* !defiend( _GPUCODE ) */
    
    p->gamma0[ id ] = gamma0;
    return;
}

SIXTRL_INLINE void NS(Particles_set_gamma0)( 
    NS(Particles)* SIXTRL_RESTRICT p, double const* SIXTRL_RESTRICT ptr_gamma0 )
{
    SIXTRACKLIB_COPY_VALUES( double, p->gamma0, ptr_gamma0, p->npart )    
    return;
}

SIXTRL_INLINE void NS(Particles_assign_ptr_to_gamma0)( 
    NS(Particles)* SIXTRL_RESTRICT p, double* ptr_gamma0 )
{
    SIXTRACKLIB_ASSIGN_PTR_TO_MEMBER( p, gamma0, ptr_gamma0 )
    return;
}



SIXTRL_INLINE void NS(Particles_set_p0c_value)( 
    NS(Particles)* SIXTRL_RESTRICT p, uint64_t id, double p0c )
{
    #if !defined( _GPUCODE )
    assert( ( p != 0 ) && ( id < p->npart ) && ( p->p0c != 0 ) );
    #endif /* !defiend( _GPUCODE ) */
    
    p->p0c[ id ] = p0c;
    return;
}

SIXTRL_INLINE void NS(Particles_set_p0c)( 
    NS(Particles)* SIXTRL_RESTRICT p, 
    double  const* SIXTRL_RESTRICT ptr_p0c )
{
    SIXTRACKLIB_COPY_VALUES( double, p->p0c, ptr_p0c, p->npart )
    return;
}

SIXTRL_INLINE void NS(Particles_assign_ptr_to_p0c)( 
    NS(Particles)* SIXTRL_RESTRICT p, double* ptr_p0c )
{
    SIXTRACKLIB_ASSIGN_PTR_TO_MEMBER( p, p0c, ptr_p0c )
    return;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE void NS(Particles_set_particle_id_value)(
    NS(Particles)* SIXTRL_RESTRICT p, uint64_t id, int64_t partid )
{
    #if !defined( _GPUCODE )
    assert( ( p != 0 ) && ( id < p->npart ) && ( p->partid != 0 ) );
    #endif /* !defiend( _GPUCODE ) */
    
    p->partid[ id ] = partid;
    return;
}

SIXTRL_INLINE void NS(Particles_set_particle_id)(
    NS(Particles)* SIXTRL_RESTRICT p, 
    int64_t const* SIXTRL_RESTRICT ptr_partid )
{
    SIXTRACKLIB_COPY_VALUES( int64_t, p->partid, ptr_partid, p->npart )
    return;
}

SIXTRL_INLINE void NS(Particles_assign_ptr_to_particle_id)( 
    NS(Particles)* SIXTRL_RESTRICT p, int64_t* ptr_partid )
{
    SIXTRACKLIB_ASSIGN_PTR_TO_MEMBER( p, partid, ptr_partid )
    return;
}



SIXTRL_INLINE void NS(Particles_set_lost_at_element_id_value)(
    NS(Particles)* SIXTRL_RESTRICT p, uint64_t id, int64_t elemid )
{
    #if !defined( _GPUCODE )
    assert( ( p != 0 ) && ( id < p->npart ) && ( p->elemid != 0 ) );
    #endif /* !defiend( _GPUCODE ) */
    
    p->elemid[ id ] = elemid;
    return;
}

SIXTRL_INLINE void NS(Particles_set_lost_at_element_id)(
    NS(Particles)* SIXTRL_RESTRICT p, 
    int64_t const* SIXTRL_RESTRICT ptr_elemid )
{
    SIXTRACKLIB_COPY_VALUES( int64_t, p->elemid, ptr_elemid, p->npart )
    return;
}

SIXTRL_INLINE void NS(Particles_assign_ptr_to_lost_at_element_id)( 
    NS(Particles)* SIXTRL_RESTRICT p, int64_t* ptr_elemid )
{
    SIXTRACKLIB_ASSIGN_PTR_TO_MEMBER( p, elemid, ptr_elemid )
    return;
}



SIXTRL_INLINE void NS(Particles_set_lost_at_turn_value)( 
    NS(Particles)* SIXTRL_RESTRICT p, uint64_t id, int64_t turn )
{
    #if !defined( _GPUCODE )
    assert( ( p != 0 ) && ( id < p->npart ) && ( p->p0c != 0 ) );
    #endif /* !defiend( _GPUCODE ) */
    
    p->turn[ id ] = turn;
    return;
}

SIXTRL_INLINE void NS(Particles_set_lost_at_turn)( 
    NS(Particles)* SIXTRL_RESTRICT p, 
    int64_t const* SIXTRL_RESTRICT ptr_turn )
{
    SIXTRACKLIB_COPY_VALUES( int64_t, p->turn, ptr_turn, p->npart )
    return;
}

SIXTRL_INLINE void NS(Particles_assign_ptr_to_lost_at_turn)( 
    NS(Particles)* SIXTRL_RESTRICT p, int64_t* ptr_turn )
{
    SIXTRACKLIB_ASSIGN_PTR_TO_MEMBER( p, turn, ptr_turn )
    return;
}




SIXTRL_INLINE void NS(Particles_set_state_value)( 
    NS(Particles)* SIXTRL_RESTRICT p, uint64_t id, int64_t state )
{
    #if !defined( _GPUCODE )
    assert( ( p != 0 ) && ( id < p->npart ) && ( p->state != 0 ) );
    #endif /* !defiend( _GPUCODE ) */
    
    p->state[ id ] = state;
    return;
}

SIXTRL_INLINE void NS(Particles_set_state)( 
    NS(Particles)* SIXTRL_RESTRICT p, 
    int64_t const* SIXTRL_RESTRICT ptr_state )
{
    SIXTRACKLIB_COPY_VALUES( int64_t, p->state, ptr_state, p->npart )
    return;
}

SIXTRL_INLINE void NS(Particles_assign_ptr_to_state)( 
    NS(Particles)* SIXTRL_RESTRICT p, int64_t* ptr_state )
{
    SIXTRACKLIB_ASSIGN_PTR_TO_MEMBER( p, state, ptr_state )
    return;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE void NS( Particles_set_s_value)( 
    NS(Particles)* SIXTRL_RESTRICT p, uint64_t id, double s )
{
    #if !defined( _GPUCODE )
    assert( ( p != 0 ) && ( id < p->npart ) && ( p->s != 0 ) );
    #endif /* !defiend( _GPUCODE ) */
    
    p->s[ id ] = s;
    return;
}

SIXTRL_INLINE void NS( Particles_set_s )( 
    NS(Particles)* SIXTRL_RESTRICT p, double const* SIXTRL_RESTRICT ptr_s )
{
    SIXTRACKLIB_COPY_VALUES( double, p->s, ptr_s, p->npart )
    return;
}

SIXTRL_INLINE void NS(Particles_assign_ptr_to_s)( 
    NS(Particles)* SIXTRL_RESTRICT p, double* ptr_s )
{
    SIXTRACKLIB_ASSIGN_PTR_TO_MEMBER( p, s, ptr_s )
    return;
}



SIXTRL_INLINE void NS(Particles_set_x_value)( 
    NS(Particles)* SIXTRL_RESTRICT p, uint64_t id, double x )
{
    #if !defined( _GPUCODE )
    assert( ( p != 0 ) && ( id < p->npart ) && ( p->x != 0 ) );
    #endif /* !defiend( _GPUCODE ) */
    
    p->x[ id ] = x;
    return;
}

SIXTRL_INLINE void NS(Particles_set_x )( 
    NS(Particles)* SIXTRL_RESTRICT p, double const* SIXTRL_RESTRICT ptr_x )
{
    SIXTRACKLIB_COPY_VALUES( double, p->x, ptr_x, p->npart )
    return;
}

SIXTRL_INLINE void NS(Particles_assign_ptr_to_x)( 
    NS(Particles)* SIXTRL_RESTRICT p, double* ptr_x )
{
    SIXTRACKLIB_ASSIGN_PTR_TO_MEMBER( p, x, ptr_x )
    return;
}



SIXTRL_INLINE void NS(Particles_set_y_value)( 
    NS(Particles)* SIXTRL_RESTRICT p, uint64_t id, double y )
{
    #if !defined( _GPUCODE )
    assert( ( p != 0 ) && ( id < p->npart ) && ( p->y != 0 ) );
    #endif /* !defiend( _GPUCODE ) */
    
    p->y[ id ] = y;
    return;
}

SIXTRL_INLINE void NS(Particles_set_y )( 
    NS(Particles)* SIXTRL_RESTRICT p, double const* SIXTRL_RESTRICT ptr_y )
{
    SIXTRACKLIB_COPY_VALUES( double, p->y, ptr_y, p->npart )
    return;
}

SIXTRL_INLINE void NS(Particles_assign_ptr_to_y)( 
    NS(Particles)* SIXTRL_RESTRICT p, double* ptr_y )
{
    SIXTRACKLIB_ASSIGN_PTR_TO_MEMBER( p, y, ptr_y )
    return;
}



SIXTRL_INLINE void NS( Particles_set_px_value)( 
    NS(Particles)* SIXTRL_RESTRICT p, uint64_t id, double px )
{
    #if !defined( _GPUCODE )
    assert( ( p != 0 ) && ( id < p->npart ) && ( p->px != 0 ) );
    #endif /* !defiend( _GPUCODE ) */
    
    p->px[ id ] = px;
    return;
}

SIXTRL_INLINE void NS( Particles_set_px )( 
    NS(Particles)* SIXTRL_RESTRICT p, double const* SIXTRL_RESTRICT ptr_px )
{
    SIXTRACKLIB_COPY_VALUES( double, p->px, ptr_px, p->npart )
    return;
}

SIXTRL_INLINE void NS(Particles_assign_ptr_to_px)( 
    NS(Particles)* SIXTRL_RESTRICT p, double* ptr_px )
{
    SIXTRACKLIB_ASSIGN_PTR_TO_MEMBER( p, px, ptr_px )
    return;
}



SIXTRL_INLINE void NS( Particles_set_py_value)( 
    NS(Particles)* SIXTRL_RESTRICT p, uint64_t id, double py )
{
    #if !defined( _GPUCODE )
    assert( ( p != 0 ) && ( id < p->npart ) && ( p->py != 0 ) );
    #endif /* !defiend( _GPUCODE ) */
    
    p->py[ id ] = py;
    return;
}

SIXTRL_INLINE void NS( Particles_set_py )( 
    NS(Particles)* SIXTRL_RESTRICT p, double const* SIXTRL_RESTRICT ptr_py )
{
    SIXTRACKLIB_COPY_VALUES( double, p->py, ptr_py, p->npart )
    return;
}

SIXTRL_INLINE void NS(Particles_assign_ptr_to_py)( 
    NS(Particles)* SIXTRL_RESTRICT p, double* ptr_py )
{
    SIXTRACKLIB_ASSIGN_PTR_TO_MEMBER( p, py, ptr_py )
    return;
}



SIXTRL_INLINE void NS( Particles_set_sigma_value)( 
    NS(Particles)* SIXTRL_RESTRICT p, uint64_t id, double sigma )
{
    #if !defined( _GPUCODE )
    assert( ( p != 0 ) && ( id < p->npart ) && ( p->sigma != 0 ) );
    #endif /* !defiend( _GPUCODE ) */
    
    p->sigma[ id ] = sigma;
    return;
}

SIXTRL_INLINE void NS( Particles_set_sigma )( 
    NS(Particles)* SIXTRL_RESTRICT p, double const* SIXTRL_RESTRICT ptr_sigma )
{
    SIXTRACKLIB_COPY_VALUES( double, p->sigma, ptr_sigma, p->npart )
    return;
}

SIXTRL_INLINE void NS(Particles_assign_ptr_to_sigma)( 
    NS(Particles)* SIXTRL_RESTRICT p, double* ptr_sigma )
{
    SIXTRACKLIB_ASSIGN_PTR_TO_MEMBER( p, sigma, ptr_sigma )
    return;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE void NS(Particles_set_psigma_value)( 
    NS(Particles)* SIXTRL_RESTRICT p, uint64_t id, double psigma )
{
    #if !defined( _GPUCODE )
    assert( ( p != 0 ) && ( id < p->npart ) && ( p->psigma != 0 ) );
    #endif /* !defiend( _GPUCODE ) */
    
    p->psigma[ id ] = psigma;
    return;
}

SIXTRL_INLINE void NS(Particles_set_psigma)( 
    NS(Particles)* SIXTRL_RESTRICT p, 
    double  const* SIXTRL_RESTRICT ptr_psigma )
{
    SIXTRACKLIB_COPY_VALUES( double, p->psigma, ptr_psigma, p->npart )
    return;
}

SIXTRL_INLINE void NS(Particles_assign_ptr_to_psigma)( 
    NS(Particles)* SIXTRL_RESTRICT p, double* ptr_psigma )
{
    SIXTRACKLIB_ASSIGN_PTR_TO_MEMBER( p, psigma, ptr_psigma )
    return;
}



SIXTRL_INLINE void NS(Particles_set_delta_value)( 
    NS(Particles)* SIXTRL_RESTRICT p, uint64_t id, double delta )
{
    #if !defined( _GPUCODE )
    assert( ( p != 0 ) && ( id < p->npart ) && ( p->p0c != 0 ) );
    #endif /* !defiend( _GPUCODE ) */
    
    p->delta[ id ] = delta;
    return;
}

SIXTRL_INLINE void NS(Particles_set_delta)( 
    NS(Particles)* SIXTRL_RESTRICT p, double const* SIXTRL_RESTRICT ptr_delta )
{
    SIXTRACKLIB_COPY_VALUES( double, p->delta, ptr_delta, p->npart )
    return;
}

SIXTRL_INLINE void NS(Particles_assign_ptr_to_delta)( 
    NS(Particles)* SIXTRL_RESTRICT p, double* ptr_delta )
{
    SIXTRACKLIB_ASSIGN_PTR_TO_MEMBER( p, delta, ptr_delta )
    return;
}



SIXTRL_INLINE void NS(Particles_set_rpp_value)( 
    NS(Particles)* SIXTRL_RESTRICT p, uint64_t id, double rpp )
{
    #if !defined( _GPUCODE )
    assert( ( p != 0 ) && ( id < p->npart ) && ( p->rpp != 0 ) );
    #endif /* !defiend( _GPUCODE ) */
    
    p->rpp[ id ] = rpp;
    return;
}

SIXTRL_INLINE void NS(Particles_set_rpp)( 
    NS(Particles)* SIXTRL_RESTRICT p, double const* SIXTRL_RESTRICT ptr_rpp )
{
    SIXTRACKLIB_COPY_VALUES( double, p->rpp, ptr_rpp, p->npart )
    return;
}

SIXTRL_INLINE void NS(Particles_assign_ptr_to_rpp)( 
    NS(Particles)* SIXTRL_RESTRICT p, double* ptr_rpp )
{
    SIXTRACKLIB_ASSIGN_PTR_TO_MEMBER( p, rpp, ptr_rpp )
    return;
}



SIXTRL_INLINE void NS(Particles_set_rvv_value)( 
    NS(Particles)* SIXTRL_RESTRICT p, uint64_t id, double rvv )
{
    #if !defined( _GPUCODE )
    assert( ( p != 0 ) && ( id < p->npart ) && ( p->rvv != 0 ) );
    #endif /* !defiend( _GPUCODE ) */
    
    p->rvv[ id ] = rvv;
    return;
}

SIXTRL_INLINE void NS(Particles_set_rvv)( 
    NS(Particles)* SIXTRL_RESTRICT p, double const* SIXTRL_RESTRICT ptr_rvv )
{
    SIXTRACKLIB_COPY_VALUES( double, p->rvv, ptr_rvv, p->npart )
    return;
}

SIXTRL_INLINE void NS(Particles_assign_ptr_to_rvv)( 
    NS(Particles)* SIXTRL_RESTRICT p, double* ptr_rvv )
{
    SIXTRACKLIB_ASSIGN_PTR_TO_MEMBER( p, rvv, ptr_rvv )
    return;
}



SIXTRL_INLINE void NS(Particles_set_chi_value)( 
    NS(Particles)* SIXTRL_RESTRICT p, uint64_t id, double chi )
{
    #if !defined( _GPUCODE )
    assert( ( p != 0 ) && ( id < p->npart ) && ( p->chi != 0 ) );
    #endif /* !defiend( _GPUCODE ) */
    
    p->chi[ id ] = chi;
    return;
}

SIXTRL_INLINE void NS(Particles_set_chi)( 
    NS(Particles)* SIXTRL_RESTRICT p, 
    double const* SIXTRL_RESTRICT ptr_chi )
{
    SIXTRACKLIB_COPY_VALUES( double, p->chi, ptr_chi, p->npart )
    return;
}

SIXTRL_INLINE void NS(Particles_assign_ptr_to_chi)( 
    NS(Particles)* SIXTRL_RESTRICT p, double* ptr_chi )
{
    SIXTRACKLIB_ASSIGN_PTR_TO_MEMBER( p, p0c, ptr_chi )
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
