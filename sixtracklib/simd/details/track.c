#include "sixtracklib/simd/track.h"

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <x86intrin.h>

#include "sixtracklib/_impl/namespace_begin.h"
#include "sixtracklib/common/impl/particles_impl.h"
#include "sixtracklib/common/particles.h"
#include "sixtracklib/common/impl/be_drift_impl.h"
#include "sixtracklib/common/be_drift.h"

extern int NS(Track_simd_drift_sse2)(
    NS(Particles)* SIXTRL_RESTRICT particles, 
    const NS(Drift) *const SIXTRL_RESTRICT drift );

extern int NS(Track_simd_drift_avx)(
    NS(Particles)* SIXTRL_RESTRICT particles, 
    const NS(Drift) *const SIXTRL_RESTRICT drift );

/* -------------------------------------------------------------------------- */

int NS(Track_simd_drift_sse2)(
    NS(Particles)* SIXTRL_RESTRICT particles, 
    const struct NS(Drift) *const SIXTRL_RESTRICT drift )
{
    #if defined __SSE2__
    
    size_t const num = NS(Particles_get_num_particles)( particles );
    
    static uintptr_t const REQ_ALIGN = ( uintptr_t )16u;
    static size_t    const STRIDE    = ( size_t )2u;
    
    double const* length_in = NS(Drift_get_const_length)( drift );
    double const  length = length_in[ 0 ];
    
    double const* px_in  = NS(Particles_get_px)(  particles );
    double const* py_in  = NS(Particles_get_py)(  particles );
    double const* rpp_in = NS(Particles_get_rpp)( particles );
    double const* rvv_in = NS(Particles_get_rvv)( particles );
    
    double* s     = ( double* )NS(Particles_get_s)( particles );
    double* x     = ( double* )NS(Particles_get_x)( particles );
    double* y     = ( double* )NS(Particles_get_y)( particles );
    double* sigma = ( double* )NS(Particles_get_sigma)( particles );
    
    uintptr_t const addr_offset = ( ( ( uintptr_t )px_in ) % REQ_ALIGN );
    size_t ii = 0;
    
    /* verify that every needed member of the particles structure has the same
     * (mis-)alignment relative to the REQ_ALIGN block-size: */
    
    assert( ( num > 1 ) && 
            ( NS(Drift_get_num_elements)( drift) == 
              ( ( NS(block_num_elements_t) )1u ) ) &&
            ( ( addr_offset == ( size_t )0u ) || 
              ( addr_offset == ( size_t )8u ) ) &&
            ( ( ( ( uintptr_t )py_in  ) % REQ_ALIGN ) == addr_offset ) &&
            ( ( ( ( uintptr_t )rpp_in ) % REQ_ALIGN ) == addr_offset ) &&
            ( ( ( ( uintptr_t )rvv_in ) % REQ_ALIGN ) == addr_offset ) &&
            ( ( ( ( uintptr_t )s      ) % REQ_ALIGN ) == addr_offset ) &&
            ( ( ( ( uintptr_t )x      ) % REQ_ALIGN ) == addr_offset ) &&
            ( ( ( ( uintptr_t )y      ) % REQ_ALIGN ) == addr_offset ) &&
            ( ( ( ( uintptr_t )sigma  ) % REQ_ALIGN ) == addr_offset ) );
    
    if( addr_offset != ( uintptr_t )0u )
    {
        NS(Drift_track_particle_over_single_elem)( 
            particles, ii++, drift, 0 );       
    }
    
    if( ii < num )
    {
        size_t const steps     = ( num - ii ) / STRIDE;
        size_t const end_index = ii + steps * STRIDE;
        
        __m128d const one      = _mm_set1_pd( ( double )1.0L );
        __m128d const one_half = _mm_set1_pd( ( double )0.5L );
        __m128d const len      = _mm_set1_pd( length );
        
        assert( ( ( uintptr_t )( px_in + ii ) % REQ_ALIGN ) == 0u );
        
        for( ; ii < end_index ; ii += STRIDE )
        {
            __m128d const rpp    = _mm_load_pd( rpp_in + ii );
            __m128d const px     = _mm_mul_pd( _mm_load_pd( px_in  + ii ), rpp );
            __m128d const px_squ = _mm_mul_pd( px, px );
            
            __m128d const py     = _mm_mul_pd( _mm_load_pd( py_in + ii ), rpp );
            __m128d const py_squ = _mm_mul_pd(  py, py );
            
            __m128d temp = 
                _mm_mul_pd( one_half, _mm_add_pd( px_squ, py_squ ) );
                        
            __m128d const dsigma = _mm_sub_pd( one, _mm_mul_pd( 
                _mm_load_pd( rvv_in + ii ), _mm_add_pd( one, temp ) ) );
            
            temp = _mm_load_pd( sigma + ii );
            _mm_store_pd( sigma + ii, 
                _mm_add_pd( temp, _mm_mul_pd( len, dsigma ) ) );
            
            temp = _mm_load_pd( s + ii );
            _mm_store_pd( s + ii, _mm_add_pd( temp, len ) );
            
            temp = _mm_load_pd( x + ii );
            _mm_store_pd( x + ii, _mm_add_pd( temp, _mm_mul_pd( len, px ) ) );
            
            temp = _mm_load_pd( y + ii );
            _mm_store_pd( y + ii, _mm_add_pd( temp, _mm_mul_pd( len, py ) ) );
        }
        
        if( ii < num )
        {
            assert( num > 0 );
            NS(Drift_track_particle_over_single_elem)( 
                particles, num - 1, drift, 0 );
        }
    }
    
    return 1;
    
    #else 
    
    #if !defined( NDEBUG )
    int const SSE2 = 0;
    int const SUPPORTED = 1;        
    assert( SSE2 == SUPPORTED );    
    #endif /* !defined( NDEBUG ) */
    
    ( void )particles;
    ( void )length;
    
    return 0;
    
    #endif /* __SSE2__ */
}

int NS(Track_simd_drift_avx)(
    NS(Particles)* SIXTRL_RESTRICT particles, 
    const struct NS(Drift) *const SIXTRL_RESTRICT drift )
{
    #if defined( __AVX__ )
    
    size_t const num = NS(Particles_get_num_particles)( particles );
    
    static uintptr_t const REQ_ALIGN = ( uintptr_t )32u;
    static size_t    const STRIDE    = ( size_t )4u;
    
    double const* length_in = NS(Drift_get_const_length)( drift );
    double const  length    = length_in[ 0 ];
    
    double const* px_in  = NS(Particles_get_px)(  particles );
    double const* py_in  = NS(Particles_get_py)(  particles );
    double const* rpp_in = NS(Particles_get_rpp)( particles );
    double const* rvv_in = NS(Particles_get_rvv)( particles );
    
    double* s   = ( double* )NS(Particles_get_s)( particles );
    double* x   = ( double* )NS(Particles_get_x)( particles );
    double* y   = ( double* )NS(Particles_get_y)( particles );
    double* sig = ( double* )NS(Particles_get_sigma)( particles );
    
    uintptr_t const addr_offset = ( ( ( uintptr_t )px_in ) % REQ_ALIGN );
    size_t ii = 0;
    
    /* verify that every needed member of the particles structure has the same
     * (mis-)alignment relative to the REQ_ALIGN block-size: */
    
    assert( ( num > 1 ) && 
            ( NS(Drift_get_num_elements)( drift) == 
              ( ( NS(block_num_elements_t) )1u ) ) &&
            ( ( addr_offset == ( size_t )0u  ) || 
              ( addr_offset == ( size_t )8u  ) || 
              ( addr_offset == ( size_t )16u ) || 
              ( addr_offset == ( size_t )24u ) ) &&
            ( ( ( ( uintptr_t )py_in  ) % REQ_ALIGN ) == addr_offset ) &&
            ( ( ( ( uintptr_t )rpp_in ) % REQ_ALIGN ) == addr_offset ) &&
            ( ( ( ( uintptr_t )rvv_in ) % REQ_ALIGN ) == addr_offset ) &&
            ( ( ( ( uintptr_t )s      ) % REQ_ALIGN ) == addr_offset ) &&
            ( ( ( ( uintptr_t )x      ) % REQ_ALIGN ) == addr_offset ) &&
            ( ( ( ( uintptr_t )y      ) % REQ_ALIGN ) == addr_offset ) &&
            ( ( ( ( uintptr_t )sig    ) % REQ_ALIGN ) == addr_offset ) );
    
    if( addr_offset != ( uintptr_t )0u )
    {
        switch( addr_offset )
        {
            case 8:
            {
                NS(Drift_track_particle_over_single_elem)( 
                    particles, ii++, drift, 0 );
                
                break;
            }
            
            case 16:
            {
                NS(Drift_track_particle_over_single_elem)( 
                    particles, ii++, drift, 0 );
                
                NS(Drift_track_particle_over_single_elem)( 
                    particles, ii++, drift, 0 );
                
                break;
            }
            
            case 24:
            {
                NS(Drift_track_particle_over_single_elem)( 
                    particles, ii++, drift, 0 );
                
                NS(Drift_track_particle_over_single_elem)( 
                    particles, ii++, drift, 0 );
                
                NS(Drift_track_particle_over_single_elem)( 
                    particles, ii++, drift, 0 );
                
                break;
            }
            
            default:
            {
                /* should never happen! */
                assert( ( addr_offset != ( uintptr_t )0u ) &&
                        ( addr_offset < REQ_ALIGN ) );
            }
        };
        
        assert( ( ( addr_offset + ii ) % REQ_ALIGN ) == 0u );
    }
    
    if( ii < num )
    {
        size_t const steps     = ( num - ii ) / STRIDE;
        size_t const end_index = ii + steps * STRIDE;
        
        __m256d const one      = _mm256_set1_pd( ( double )1.0L );
        __m256d const one_half = _mm256_set1_pd( ( double )0.5L );
        __m256d const len      = _mm256_set1_pd( length );
        
        assert( ( ( uintptr_t )( px_in + ii ) % REQ_ALIGN ) == 0u );
        
        for( ; ii < end_index ; ii += STRIDE )
        {
            __m256d const rpp    = _mm256_load_pd( rpp_in + ii );
            __m256d const px     = _mm256_mul_pd( _mm256_load_pd( px_in  + ii ), rpp );
            __m256d const px_squ = _mm256_mul_pd( px, px );
            
            __m256d const py     = _mm256_mul_pd( _mm256_load_pd( py_in + ii ), rpp );
            __m256d const py_squ = _mm256_mul_pd(  py, py );
                        
            __m256d temp = 
                _mm256_mul_pd( _mm256_add_pd( px_squ, py_squ ), one_half );
            
            __m256d const dsig = _mm256_mul_pd( len, _mm256_sub_pd( one, 
                _mm256_mul_pd( _mm256_load_pd( rvv_in + ii ), 
                               _mm256_add_pd( one, temp ) ) ) );
            
            temp = _mm256_load_pd( sig + ii );
            _mm256_store_pd( sig + ii, _mm256_add_pd( temp, dsig ) );
            
            temp = _mm256_load_pd( s + ii );
            _mm256_store_pd( s + ii, _mm256_add_pd( temp, len ) );
            
            temp = _mm256_load_pd( x + ii );
            _mm256_store_pd( x + ii, _mm256_add_pd( temp, _mm256_mul_pd( len, px ) ) );
            
            temp = _mm256_load_pd( y + ii );
            _mm256_store_pd( y + ii, _mm256_add_pd( temp, _mm256_mul_pd( len, py ) ) );
        }
        
        for( ; ii < num ; ++ii )
        {
            NS(Drift_track_particle_over_single_elem)( 
                particles, ii, drift, 0 );
        }
    }
    
    return 1;
    
     #else 
    
    #if !defined( NDEBUG )
    int const AVX = 0;
    int const SUPPORTED = 1;        
    assert( AVX == SUPPORTED );
    #endif /* !defined( NDEBUG ) */
    
    ( void )particles;
    ( void )length;
    
    return 0;
    
    #endif /* __AVX__ */
}

/* -------------------------------------------------------------------------- */
