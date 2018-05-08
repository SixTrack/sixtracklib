#include "sixtracklib/simd/track.h"

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <x86intrin.h>

#include "sixtracklib/_impl/namespace_begin.h"
#include "sixtracklib/_impl/definitions.h"
#include "sixtracklib/common/impl/block_info_impl.h"
#include "sixtracklib/common/impl/particles_impl.h"
#include "sixtracklib/common/impl/be_drift_impl.h"
#include "sixtracklib/common/beam_elements.h"
#include "sixtracklib/common/particles.h"
#include "sixtracklib/common/track.h"

extern int NS(Track_beam_elements_simd_sse2)(
    NS(Particles)* SIXTRL_RESTRICT particles,
    NS(block_num_elements_t) const start_particle_index,
    NS(block_num_elements_t) const end_particle_index,
    const NS(BeamElements) *const SIXTRL_RESTRICT beam_elements,
    NS(block_num_elements_t) const elem_by_elem_start_index,
    NS(ParticlesContainer)* SIXTRL_RESTRICT elem_by_elem_buffer );

extern int NS(Track_beam_elements_simd_avx)(
    NS(Particles)* SIXTRL_RESTRICT particles,
    NS(block_num_elements_t) const start_particle_index,
    NS(block_num_elements_t) const end_particle_index,
    const NS(BeamElements) *const SIXTRL_RESTRICT beam_elements,
    NS(block_num_elements_t) const elem_by_elem_start_index,
    NS(ParticlesContainer)* SIXTRL_RESTRICT elem_by_elem_buffer );

extern int NS(Track_drift_simd_sse2)(
    struct NS(Particles)* SIXTRL_RESTRICT particles, 
    NS(block_num_elements_t) const start_particle_index,
    NS(block_num_elements_t) const end_particle_index,
    SIXTRL_REAL_T const length );

extern int NS(Track_drift_simd_avx)(
    struct NS(Particles)* SIXTRL_RESTRICT particles, 
    NS(block_num_elements_t) const start_particle_index,
    NS(block_num_elements_t) const end_particle_index,
    SIXTRL_REAL_T const length );

/* ========================================================================= */
/* =====                  Implementation of functions                   ==== */
/* ========================================================================= */

int NS(Track_beam_elements_simd_sse2)(
    NS(Particles)* SIXTRL_RESTRICT particles,
    NS(block_num_elements_t) const start_particle_index,
    NS(block_num_elements_t) const end_particle_index,
    const NS(BeamElements) *const SIXTRL_RESTRICT beam_elements,
    NS(block_num_elements_t) const elem_by_elem_start_index,
    NS(ParticlesContainer)* SIXTRL_RESTRICT elem_by_elem_buffer )
{
    int status = 0;
    
    int const use_elem_by_elem_store = (
        ( elem_by_elem_buffer != 0 ) && 
        ( NS(ParticlesContainer_get_num_of_blocks)( elem_by_elem_buffer ) >=
          ( NS(BeamElements_get_num_of_blocks)( beam_elements ) + 
                elem_by_elem_start_index ) ) );
    
    NS(BlockInfo) const* be_block_info_it  = 
        NS(BeamElements_get_const_block_infos_begin)( beam_elements );
        
    NS(BlockInfo) const* be_block_info_end =
        NS(BeamElements_get_const_block_infos_end)( beam_elements );
    
    NS(block_num_elements_t) elem_by_elem_idx = elem_by_elem_start_index;
        
    SIXTRL_ASSERT( ( beam_elements != 0 ) && ( particles != 0 ) && 
                   ( be_block_info_it != 0 ) );
        
    for( ; be_block_info_it != be_block_info_end ; ++be_block_info_it )
    {
        typedef SIXTRL_GLOBAL_DEC unsigned char* g_ptr_uchar_t;
        
        g_ptr_uchar_t be_mem_begin = ( g_ptr_uchar_t 
            )NS(BeamElements_get_const_ptr_data_begin)( beam_elements );
            
        NS(block_size_t) const be_max_num_bytes = 
            NS(BeamElements_get_data_capacity)( beam_elements );
        
        NS(BlockType) const type_id = 
            NS(BlockInfo_get_type_id)( be_block_info_it );
        
        int status = 0;    
            
        switch( type_id )
        {
            case NS(BLOCK_TYPE_DRIFT):
            {
                NS(Drift) drift;
                NS(Drift_preset)( &drift ); /* only to get rid of warnings! */
                
                status |= NS(Drift_remap_from_memory)( ( void* )&drift, 
                    be_block_info_it, be_mem_begin, be_max_num_bytes );
                
                status |= NS(Track_drift_simd_sse2)( 
                    particles, start_particle_index, end_particle_index, 
                        NS(Drift_get_length_value)( &drift ) );
                
                break;
            }
            
            
            case NS(BLOCK_TYPE_DRIFT_EXACT):
            {
                NS(Drift) drift;
                NS(Drift_preset)( &drift );
                
                status |= NS(Drift_remap_from_memory)( ( void* )&drift, 
                    be_block_info_it, be_mem_begin, be_max_num_bytes );
                
                status |= NS(Track_drift_simd_sse2)( 
                    particles, start_particle_index, end_particle_index, 
                        NS(Drift_get_length_value)( &drift ) );
                
                break;
            }
            
            default:
            {
                status = -1;
            }
        };
        
        if( use_elem_by_elem_store )
        {
            NS(Particles) particles_after_elem;
             
            status |= NS(ParticlesContainer_get_particles)( 
                &particles_after_elem, elem_by_elem_buffer, elem_by_elem_idx );
            
            NS(Particles_copy_all_unchecked)( &particles_after_elem, particles );
            
            ++elem_by_elem_idx;
        }
    }
    
    return status;
}

/* ------------------------------------------------------------------------- */

int NS(Track_beam_elements_simd_avx)(
    NS(Particles)* SIXTRL_RESTRICT particles,
    NS(block_num_elements_t) const start_particle_index,
    NS(block_num_elements_t) const end_particle_index,
    const NS(BeamElements) *const SIXTRL_RESTRICT beam_elements,
    NS(block_num_elements_t) const elem_by_elem_start_index,
    NS(ParticlesContainer)* SIXTRL_RESTRICT elem_by_elem_buffer )
{
    int status = 0;
    
    int const use_elem_by_elem_store = (
        ( elem_by_elem_buffer != 0 ) && 
        ( NS(ParticlesContainer_get_num_of_blocks)( elem_by_elem_buffer ) >=
          ( NS(BeamElements_get_num_of_blocks)( beam_elements ) + 
                elem_by_elem_start_index ) ) );
    
    NS(BlockInfo) const* be_block_info_it  = 
        NS(BeamElements_get_const_block_infos_begin)( beam_elements );
        
    NS(BlockInfo) const* be_block_info_end =
        NS(BeamElements_get_const_block_infos_end)( beam_elements );
    
    NS(block_num_elements_t) elem_by_elem_idx = elem_by_elem_start_index;
        
    SIXTRL_ASSERT( ( beam_elements != 0 ) && ( particles != 0 ) && 
                   ( be_block_info_it != 0 ) );
        
    for( ; be_block_info_it != be_block_info_end ; ++be_block_info_it )
    {
        typedef SIXTRL_GLOBAL_DEC unsigned char* g_ptr_uchar_t;
        
        g_ptr_uchar_t be_mem_begin = ( g_ptr_uchar_t 
            )NS(BeamElements_get_const_ptr_data_begin)( beam_elements );
            
        NS(block_size_t) const be_max_num_bytes = 
            NS(BeamElements_get_data_capacity)( beam_elements );
        
        NS(BlockType) const type_id = 
            NS(BlockInfo_get_type_id)( be_block_info_it );
        
        int status = 0;    
            
        switch( type_id )
        {
            case NS(BLOCK_TYPE_DRIFT):
            {
                NS(Drift) drift;
                NS(Drift_preset)( &drift ); /* only to get rid of warnings! */
                
                status |= NS(Drift_remap_from_memory)( 
                    &drift, be_block_info_it, be_mem_begin, be_max_num_bytes );
                
                status |= NS(Track_drift_simd_avx)( particles, 
                    start_particle_index, end_particle_index,
                        NS(Drift_get_length_value)( &drift ) );
                
                break;
            }
            
            
            case NS(BLOCK_TYPE_DRIFT_EXACT):
            {
                NS(Drift) drift;
                NS(Drift_preset)( &drift );
                
                status |= NS(Drift_remap_from_memory)( 
                    &drift, be_block_info_it, be_mem_begin, be_max_num_bytes );
                
                status |= NS(Track_drift_simd_avx)( particles, 
                    start_particle_index, end_particle_index,
                        NS(Drift_get_length_value)( &drift ) );
                
                break;
            }
            
            default:
            {
                status = -1;
            }
        };
        
        if( use_elem_by_elem_store )
        {
            NS(Particles) particles_after_elem;
             
            status |= NS(ParticlesContainer_get_particles)( 
                &particles_after_elem, elem_by_elem_buffer, elem_by_elem_idx );
            
            NS(Particles_copy_all_unchecked)( &particles_after_elem, particles );            
            ++elem_by_elem_idx;
        }
    }
    
    return status;
}

/* ------------------------------------------------------------------------- */

int NS(Track_drift_simd_sse2)(
    NS(Particles)* SIXTRL_RESTRICT particles, 
    NS(block_num_elements_t) const start_particle_index,
    NS(block_num_elements_t) const end_particle_index,
    SIXTRL_REAL_T const length )
{
    #if defined __SSE2__
    
    NS(block_num_elements_t) const num = end_particle_index - 
        start_particle_index;
    
    static uintptr_t const REQ_ALIGN = ( uintptr_t )16u;
    static size_t    const STRIDE    = ( size_t )2u;
    
    double const* px_in  = NS(Particles_get_px)(  particles );
    double const* py_in  = NS(Particles_get_py)(  particles );
    double const* rpp_in = NS(Particles_get_rpp)( particles );
    double const* rvv_in = NS(Particles_get_rvv)( particles );
    
    double* s     = ( double* )NS(Particles_get_s)( particles );
    double* x     = ( double* )NS(Particles_get_x)( particles );
    double* y     = ( double* )NS(Particles_get_y)( particles );
    double* sigma = ( double* )NS(Particles_get_sigma)( particles );
    
    uintptr_t const addr_offset = ( ( ( uintptr_t )px_in ) % REQ_ALIGN );
    size_t ii = start_particle_index;
    
    /* verify that every needed member of the particles structure has the same
     * (mis-)alignment relative to the REQ_ALIGN block-size: */
    
    SIXTRL_ASSERT( 
            ( num > ( NS(block_num_elements_t) )0u ) &&             
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
        NS(Track_drift_particle)( particles, ii++, length );
    }
    
    if( ii < num )
    {
        size_t const steps     = ( end_particle_index - ii ) / STRIDE;
        NS(block_num_elements_t) const end_index = ii + steps * STRIDE;
        
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
            SIXTRL_ASSERT( num > 0 );
            NS(Track_drift_particle)( particles, ii++, length );
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

/* ------------------------------------------------------------------------- */

int NS(Track_drift_simd_avx)(
    NS(Particles)* SIXTRL_RESTRICT particles, 
    NS(block_num_elements_t) const start_particle_index,
    NS(block_num_elements_t) const end_particle_index,
    SIXTRL_REAL_T const length )
{
    #if defined( __AVX__ )
    
    NS(block_num_elements_t) const num = 
        end_particle_index - start_particle_index;
        
    static uintptr_t const REQ_ALIGN = ( uintptr_t )32u;
    static size_t    const STRIDE    = ( size_t )4u;
    
    double const* px_in  = NS(Particles_get_px)(  particles );
    double const* py_in  = NS(Particles_get_py)(  particles );
    double const* rpp_in = NS(Particles_get_rpp)( particles );
    double const* rvv_in = NS(Particles_get_rvv)( particles );
    
    double* s   = ( double* )NS(Particles_get_s)( particles );
    double* x   = ( double* )NS(Particles_get_x)( particles );
    double* y   = ( double* )NS(Particles_get_y)( particles );
    double* sig = ( double* )NS(Particles_get_sigma)( particles );
    
    uintptr_t const addr_offset = ( ( ( uintptr_t )px_in ) % REQ_ALIGN );
    size_t ii = start_particle_index;
    
    /* verify that every needed member of the particles structure has the same
     * (mis-)alignment relative to the REQ_ALIGN block-size: */
    
    assert( ( num > ( NS(block_num_elements_t) )0u ) &&
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
                NS(Track_drift_particle)( particles, ii++, length );                
                break;
            }
            
            case 16:
            {
                NS(Track_drift_particle)( particles, ii++, length );
                NS(Track_drift_particle)( particles, ii++, length );                
                break;
            }
            
            case 24:
            {
                NS(Track_drift_particle)( particles, ii++, length );
                NS(Track_drift_particle)( particles, ii++, length );
                NS(Track_drift_particle)( particles, ii++, length );                
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
        size_t const steps     = ( end_particle_index - ii ) / STRIDE;
        NS(block_num_elements_t) const end_index = ii + steps * STRIDE;
        
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
        
        for( ; ii < end_particle_index ; ++ii )
        {
            NS(Track_drift_particle)( particles, ii++, length );
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

/* end: sixtracklib/simd/details/track.c */
