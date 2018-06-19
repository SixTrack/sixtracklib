#include "sixtracklib/mpfr4/track.h"

#include <stdint.h>
#include <stdio.h>
#include <stdarg.h>

#include <mpfr.h>

#include "sixtracklib/_impl/definitions.h"
#include "sixtracklib/_impl/namespace_begin.h"

#include "sixtracklib/mpfr4/impl/particles_impl.h"

#include "sixtracklib/common/impl/block_info_impl.h"
#include "sixtracklib/common/impl/particles_impl.h"

/* ------------------------------------------------------------------------- */

extern int NS(Track_beam_elements_mpfr4)(
    NS(Particles)* SIXTRL_RESTRICT particles,
    const NS(BeamElements)* const SIXTRL_RESTRICT beam_elements,
    mpfr_prec_t const prec, mpfr_rnd_t const rnd,
    NS(block_num_elements_t) const elem_by_elem_start_index,
    NS(ParticlesContainer)* SIXTRL_RESTRICT elem_by_elem_buffer );

extern int NS(Track_drift_mpfr4)(
    NS(Particles)* SIXTRL_RESTRICT particles, 
    SIXTRL_REAL_T const drift_length, 
    mpfr_prec_t const prec, mpfr_rnd_t const rnd );

extern int NS(Track_drift_exact_mpfr4)(
    NS(Particles)* SIXTRL_RESTRICT particles, 
    SIXTRL_REAL_T const drift_length, 
    mpfr_prec_t const prec, mpfr_rnd_t const rnd );

/* ------------------------------------------------------------------------- */

int NS(Track_beam_elements_mpfr4)(
    NS(Particles)* SIXTRL_RESTRICT particles,
    const NS(BeamElements) *const SIXTRL_RESTRICT beam_elements,
    mpfr_prec_t const prec, mpfr_rnd_t const rnd,
    NS(block_num_elements_t) const elem_by_elem_start_index,
    NS(ParticlesContainer)* SIXTRL_RESTRICT elem_by_elem_buffer )
{
    typedef SIXTRL_GLOBAL_DEC unsigned char* g_ptr_uchar_t;
    
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
        
    g_ptr_uchar_t be_mem_begin = ( g_ptr_uchar_t
        )NS(BeamElements_get_const_ptr_data_begin)( beam_elements );
        
    NS(block_size_t) const be_max_num_bytes =
        NS(BeamElements_get_data_capacity)( beam_elements );
        
    SIXTRL_ASSERT( ( beam_elements != 0 ) && ( particles != 0 ) && 
                   ( be_block_info_it != 0 ) );
        
    
    for( ; be_block_info_it != be_block_info_end ; ++be_block_info_it )
    {
        
        
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
                
                status |= NS(Track_drift_mpfr4)( particles, 
                    NS(Drift_get_length_value)( &drift ), prec, rnd );
                
                break;
            }
            
            
            case NS(BLOCK_TYPE_DRIFT_EXACT):
            {
                NS(Drift) drift;
                NS(Drift_preset)( &drift );
                
                status |= NS(Drift_remap_from_memory)( 
                    &drift, be_block_info_it, be_mem_begin, be_max_num_bytes );
                
                status |= NS(Track_drift_exact_mpfr4)( particles, 
                    NS(Drift_get_length_value)( &drift ), prec, rnd );
                
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
            
            NS(Particles_copy_all_unchecked_mpfr4)(
                &particles_after_elem, particles, rnd );
            
            ++elem_by_elem_idx;
        }
    }
    
    return status;
}


int NS(Track_drift_mpfr4)(
    struct NS(Particles)* SIXTRL_RESTRICT particles, SIXTRL_REAL_T const len,
    mpfr_prec_t const prec, mpfr_rnd_t const rnd )
{
    NS(block_num_elements_t ) ii = 0;
    
    NS(block_num_elements_t) const NUM_PARTICLES = 
        NS(Particles_get_num_particles)( particles );
    
    mpfr_t temp;
    mpfr_t one_half;
    mpfr_t one;
    mpfr_t px;
    mpfr_t px_squ;
    mpfr_t py;
    mpfr_t py_squ;
    mpfr_t delta_sigma;
    
    SIXTRL_REAL_T const* SIXTRL_RESTRICT px_in = NS(Particles_get_px)( particles );
    SIXTRL_REAL_T const* SIXTRL_RESTRICT py_in = NS(Particles_get_py)( particles );
    SIXTRL_REAL_T const* SIXTRL_RESTRICT rpp   = NS(Particles_get_delta)( particles );
    SIXTRL_REAL_T const* SIXTRL_RESTRICT rvv   = NS(Particles_get_beta0)( particles );
    
    SIXTRL_REAL_T* SIXTRL_RESTRICT s = NS(Particles_get_s)( particles );
    SIXTRL_REAL_T* SIXTRL_RESTRICT x = NS(Particles_get_x)( particles );
    SIXTRL_REAL_T* SIXTRL_RESTRICT y = NS(Particles_get_y)( particles );
    SIXTRL_REAL_T* SIXTRL_RESTRICT sigma = NS(Particles_get_sigma)( particles );
    
    SIXTRL_ASSERT( ( px_in != 0 ) && ( py_in != 0 ) && 
                   ( rpp != 0 ) && ( rvv != 0 ) );
    
    SIXTRL_ASSERT( ( s != 0 ) && ( x != 0 ) && ( y != 0 ) && ( sigma != 0 ) );
    
    mpfr_init2( temp, prec );
    mpfr_init2( one_half, prec );
    mpfr_init2( one, prec ); 
    mpfr_init2( px, prec );
    mpfr_init2( px_squ, prec );
    mpfr_init2( py, prec );
    mpfr_init2( py_squ, prec ); 
    mpfr_init2( delta_sigma, prec );
    
    mpfr_set_str( one_half, "0.5", 10, rnd );
    mpfr_set_str( one, "1.0", 10, rnd );
    
    for( ii = 0 ; ii < NUM_PARTICLES ; ++ii )
    {
        mpfr_set( temp, rpp[ ii ].value, rnd );
        mpfr_mul( px, temp, px_in[ ii ].value, rnd );        
        mpfr_mul( px_squ, px, px, rnd );
        
        mpfr_mul( py, temp, py_in[ ii ].value, rnd );
        mpfr_mul( py_squ, py, py, rnd );
        
        mpfr_add( temp, px_squ, py_squ, rnd );
        mpfr_mul( temp, one_half, temp, rnd );
        mpfr_add( temp, one, temp, rnd );
        mpfr_mul( temp, rvv[ ii ].value, temp, rnd );
        mpfr_sub( delta_sigma, one, temp, rnd );
        
        mpfr_add( s[ ii ].value, s[ ii ].value, len.value, rnd );
        
        mpfr_mul( temp, px, len.value, rnd );
        mpfr_add( x[ ii ].value, x[ ii ].value, temp, rnd );
        
        mpfr_mul( temp, py, len.value, rnd );
        mpfr_add( y[ ii ].value, y[ ii ].value, temp, rnd );
        
        mpfr_mul( temp, delta_sigma, len.value, rnd );
        mpfr_add( sigma[ ii ].value, sigma[ ii ].value, temp, rnd );
    }
    
    mpfr_clear( temp );
    mpfr_clear( one );
    mpfr_clear( one_half );
    mpfr_clear( px );
    mpfr_clear( px_squ );
    mpfr_clear( py );
    mpfr_clear( py_squ );
    mpfr_clear( delta_sigma );
    
    return 0;
}

int NS(Track_drift_exact_mpfr4)(
    struct NS(Particles)* SIXTRL_RESTRICT particles, SIXTRL_REAL_T const len,
    mpfr_prec_t const prec, mpfr_rnd_t const rnd )
{
    NS(block_num_elements_t ) ii = 0;
    
    NS(block_num_elements_t) const NUM_PARTICLES = 
        NS(Particles_get_num_particles)( particles );
    
    mpfr_t temp;
    mpfr_t one;
    mpfr_t opd;
    mpfr_t opd_squ;
    mpfr_t px;
    mpfr_t px_squ;
    mpfr_t py;
    mpfr_t py_squ;
    mpfr_t lpzi;
    mpfr_t lbzi;
    mpfr_t beta0;
        
    SIXTRL_REAL_T const* SIXTRL_RESTRICT px_in    = NS(Particles_get_px)( particles );
    SIXTRL_REAL_T const* SIXTRL_RESTRICT py_in    = NS(Particles_get_py)( particles );
    SIXTRL_REAL_T const* SIXTRL_RESTRICT delta    = NS(Particles_get_delta)( particles );
    SIXTRL_REAL_T const* SIXTRL_RESTRICT beta0_in = NS(Particles_get_beta0)( particles );
    
    SIXTRL_REAL_T* SIXTRL_RESTRICT s     = NS(Particles_get_s)( particles );
    SIXTRL_REAL_T* SIXTRL_RESTRICT x     = NS(Particles_get_x)( particles );
    SIXTRL_REAL_T* SIXTRL_RESTRICT y     = NS(Particles_get_y)( particles );
    SIXTRL_REAL_T* SIXTRL_RESTRICT sigma = NS(Particles_get_sigma)( particles );
    
    
    SIXTRL_ASSERT( ( px_in != 0 ) && ( py_in != 0 ) && 
                   ( delta != 0 ) && ( beta0_in != 0 ) );
    
    SIXTRL_ASSERT( ( sigma != 0 ) && ( s != 0 ) && ( x != 0 ) && ( y != 0 ) );
    
    
    mpfr_init2( temp, prec );
    mpfr_init2( one, prec );
    mpfr_init2( opd, prec );
    mpfr_init2( opd_squ, prec );
    mpfr_init2( px, prec );
    mpfr_init2( px_squ, prec );
    mpfr_init2( py, prec );
    mpfr_init2( py_squ, prec );
    mpfr_init2( lpzi, prec );
    mpfr_init2( lbzi, prec ); 
    mpfr_init2( beta0, prec );
    
    mpfr_set_str( one, "1.0", 10, rnd );
    
    for( ; ii < NUM_PARTICLES ; ++ii )
    {
        mpfr_set( px, px_in[ ii ].value, rnd );
        mpfr_mul( px_squ, px, px, rnd );
        
        mpfr_set( py, py_in[ ii ].value, rnd );
        mpfr_mul( py_squ, py, py, rnd );
        
        mpfr_add( temp, px_squ, py_squ, rnd );
        
        mpfr_set( opd, delta[ ii ].value, rnd );
        mpfr_add( opd, one, opd, rnd );
        mpfr_mul( opd_squ, opd, opd, rnd );
        mpfr_sub( temp, opd_squ, temp, rnd );
        mpfr_sqrt( temp, temp, rnd );
        mpfr_div( lpzi, len.value, temp, rnd );
        
        mpfr_set( beta0, beta0_in[ ii ].value, rnd );
        mpfr_mul( beta0, beta0, beta0, rnd );
        mpfr_mul( beta0, beta0, sigma[ ii ].value, rnd );
        mpfr_add( temp,  beta0, one, rnd );
        mpfr_mul( lbzi,  lpzi, temp, rnd );
        
        mpfr_add( s[ ii ].value, len.value, s[ ii ].value, rnd);
        
        mpfr_mul( temp, px, lpzi, rnd );
        mpfr_add( x[ ii ].value, temp, x[ ii ].value, rnd );
        
        mpfr_mul( temp, py, lpzi, rnd );
        mpfr_add( y[ ii ].value, temp, y[ ii ].value, rnd );
        
        mpfr_sub( temp, len.value, lbzi, rnd );
        mpfr_add( sigma[ ii ].value, temp, sigma[ ii ].value, rnd );
    }
        
    mpfr_clear( temp );
    mpfr_clear( one );
    mpfr_clear( opd );
    mpfr_clear( opd_squ );
    mpfr_clear( px );
    mpfr_clear( px_squ );
    mpfr_clear( py );
    mpfr_clear( py_squ );
    mpfr_clear( lpzi );
    mpfr_clear( lbzi );
    mpfr_clear( beta0 );
    
    return 0;    
}

/* end: sixtracklib/mpfr4/details/track.c */
