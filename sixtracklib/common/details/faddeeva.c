#include "sixtracklib/common/impl/faddeeva.h"

#include <float.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "sixtracklib/_impl/definitions.h"

extern SIXTRL_FN int NS(Faddeeva_calculate_w_mit)( 
    SIXTRL_REAL_T* SIXTRL_RESTRICT out_re, 
    SIXTRL_REAL_T* SIXTRL_RESTRICT out_im,
    SIXTRL_REAL_T const re, SIXTRL_REAL_T const im,
    SIXTRL_REAL_T rel_error, 
    int const use_continued_fractions );




SIXTRL_FN int NS(Faddeeva_calculate_w_mit)( 
    SIXTRL_REAL_T* SIXTRL_RESTRICT out_re, 
    SIXTRL_REAL_T* SIXTRL_RESTRICT out_im,
    SIXTRL_REAL_T const re, SIXTRL_REAL_T const im, 
    SIXTRL_REAL_T rel_error, 
    int const use_continued_fractions  )
{
    int ret = -1;
    
    SIXTRL_STATIC_VAR SIXTRL_REAL_T const ZERO     = ( SIXTRL_REAL_T )0.0L;
    SIXTRL_STATIC_VAR SIXTRL_REAL_T const ONE      = ( SIXTRL_REAL_T )1.0L;
    SIXTRL_STATIC_VAR SIXTRL_REAL_T const TWO      = ( SIXTRL_REAL_T )2.0L;
    SIXTRL_STATIC_VAR SIXTRL_REAL_T const ONE_HALF = ( SIXTRL_REAL_T )0.5L;
    
    SIXTRL_STATIC_VAR SIXTRL_REAL_T const INV_SQURT_PI = 
            ( SIXTRL_REAL_T )0.56418958354775628694807945156L;
    
    SIXTRL_REAL_T a    = ZERO;
    SIXTRL_REAL_T a2   = ZERO;
    SIXTRL_REAL_T c    = ZERO;
    
    SIXTRL_REAL_T sum1 = ZERO;
    SIXTRL_REAL_T sum2 = ZERO;
    SIXTRL_REAL_T sum3 = ZERO;
    SIXTRL_REAL_T sum4 = ZERO;
    SIXTRL_REAL_T sum5 = ZERO;
    
    SIXTRL_REAL_T const x  = ( re >= ZERO ) ? re : -re;
    SIXTRL_REAL_T const y = im;
    
    SIXTRL_REAL_T y_abs  = ( y >= ZERO ) ? y : -y;
    SIXTRL_REAL_T result_re = ZERO;
    SIXTRL_REAL_T result_im = ZERO;
    
    /*
    if( re == ZERO )
    {
        
    }
    else if( im == ZERO )
    {
        
    }
    */
    
    if( rel_error <= DBL_EPSILON )
    {
        rel_error = DBL_EPSILON;
        a         = ( SIXTRL_REAL_T )0.518321480430085929872L;
        c         = ( SIXTRL_REAL_T )0.329973702884629072537L;
        a2        = ( SIXTRL_REAL_T )0.268657157075235951582L;        
    }
    else
    {
        SIXTRL_STATIC_VAR SIXTRL_REAL_T const MAX_REL_ERROR = 
            ( SIXTRL_REAL_T )0.1L;
            
        SIXTRL_STATIC_VAR SIXTRL_REAL_T const PI = 
            ( SIXTRL_REAL_T )3.14159265358979323846264338327950288419716939937510582L;
            
        if( rel_error > MAX_REL_ERROR ) rel_error = MAX_REL_ERROR;
        
        a  = PI / sqrt( -log( rel_error * ONE_HALF ) );
        c  = ( TWO / PI ) * a;
        a2 = a*a;
    }
    
    if( use_continued_fractions )
    {
        if( ( y_abs > 7.0 ) || 
            ( ( x > 6.0 ) && ( ( y_abs > 0.1 ) || 
                               ( ( x > 8.0 ) && ( y_abs > 1e-10 ) ) ||
                               ( x > 28.0 ) ) ) )
        {
            SIXTRL_REAL_T const xs = ( y < ZERO ) ? -re : +re;
            SIXTRL_REAL_T const d  = x + y_abs;
            
            if( d > 4000.0 )
            {
                if( d > 1e7 )
                {
                    if( x > y_abs )
                    {
                        SIXTRL_REAL_T const y_abs_x = y_abs / xs;
                        SIXTRL_REAL_T const denom = 
                            INV_SQURT_PI / ( xs + y_abs_x * y_abs );
                            
                        result_re = denom * y_abs_x;
                        result_im = denom;
                        ret = 0;
                    }
                    else if( isinf( y_abs ) )
                    {
                        if( ( isnan( x ) ) || ( y < ZERO ) )
                        {
                            result_re = NAN;
                            result_im = NAN;
                        }
                        
                        ret = 0;
                    }
                    else
                    {
                        SIXTRL_REAL_T const x_y_abs = xs / y_abs; 
                        SIXTRL_REAL_T const denom = 
                            INV_SQURT_PI / ( x_y_abs * xs + y_abs );
                            
                        result_re = denom;
                        result_im = denom * x_y_abs;
                        ret = 0;
                    }
                }
                else
                {
                    SIXTRL_REAL_T const dr = xs  * xs - y_abs * y_abs - ONE_HALF;
                    SIXTRL_REAL_T const di = TWO * xs * y_abs;
                    SIXTRL_REAL_T const denom = 
                        INV_SQURT_PI / ( dr * dr + di * di );
                        
                    result_re = denom * ( xs * di - y_abs * dr );
                    result_im = denom * ( xs * dr + y_abs * di );
                    ret = 0;
                }
            }
            else
            {
                SIXTRL_STATIC_VAR SIXTRL_REAL_T const C0 = 
                    ( SIXTRL_REAL_T )3.9L;
                    
                SIXTRL_STATIC_VAR SIXTRL_REAL_T const C1 =
                    ( SIXTRL_REAL_T )11.398L;
                    
                SIXTRL_STATIC_VAR SIXTRL_REAL_T const C2 = 
                    ( SIXTRL_REAL_T )0.08254L;
                    
                SIXTRL_STATIC_VAR SIXTRL_REAL_T const C3 = 
                    ( SIXTRL_REAL_T )0.1421L;
                    
                SIXTRL_STATIC_VAR SIXTRL_REAL_T const C4 = 
                    ( SIXTRL_REAL_T )0.2023L; 
                    
                SIXTRL_REAL_T nu = 
                    floor( C0 + C1 / ( C2 * x + C3 * y_abs + C4 ) );

                SIXTRL_REAL_T denom = ZERO;
                    
                SIXTRL_REAL_T wr = xs;
                SIXTRL_REAL_T wi = y_abs;
                
                for( nu  = ONE_HALF * ( nu - ONE ) ; 
                     nu  > ( SIXTRL_REAL_T )0.4L ; nu -= ONE_HALF )
                {
                    denom = nu / ( wr * wr + wi * wi );
                    wr    = xs    - wr * denom;
                    wi    = y_abs + wi * denom;                                        
                }
                    
                denom = INV_SQURT_PI / ( wr * wr + wi * wi );
                
                result_re = denom * wi;
                result_im = denom * wr;
                ret = 0;
            }
            
            if( y < ZERO )
            {
                SIXTRL_REAL_T const exp_re = exp( ( y_abs - xs ) * ( xs + y_abs ) );
                SIXTRL_REAL_T const arg_im = TWO * xs * y;
                
                result_re = TWO * exp_re * cos( arg_im ) - result_re;
                result_im = TWO * exp_re * sin( arg_im ) - result_im;
                ret = 0;
            }
            else
            {
                ret = 0;
            }
        }
    }
    else /* !use continued fractions */
    {
        SIXTRL_REAL_T const xs = ( y < ZERO ) ? -re : +re;
        SIXTRL_REAL_T const d  = x + y_abs;
        
        if( d > 1e7 ) 
        { 
            if( x > y_abs )
            {
                SIXTRL_REAL_T const y_abs_x = y_abs / xs;
                SIXTRL_REAL_T const denom = 
                    INV_SQURT_PI / ( xs + y_abs_x * y_abs );
                    
                result_re = denom * y_abs_x;
                result_im = denom;
                ret = 0;
            }
            else
            {
                SIXTRL_REAL_T const x_y_abs = xs / y_abs;
                SIXTRL_REAL_T const denom = 
                    INV_SQURT_PI / ( x_y_abs * xs + y_abs );
                    
                result_re = denom;
                result_im = denom * x_y_abs;
                ret = 0;
            }
            
            if( y < ZERO )
            {
                SIXTRL_REAL_T const exp_re = exp( ( y_abs - xs ) * ( xs + y_abs ) );
                SIXTRL_REAL_T const arg_im = TWO * xs * y;
                
                result_re = TWO * exp_re * cos( arg_im ) - result_re;
                result_im = TWO * exp_re * sin( arg_im ) - result_im;
                ret = 0;
            }
        }
    }
  
  /* Note: The test that seems to be suggested in the paper is x <
     sqrt(-log(DBL_MIN)), about 26.6, since otherwise exp(-x^2)
     underflows to zero and sum1,sum2,sum4 are zero.  However, long
     before this occurs, the sum1,sum2,sum4 contributions are
     negligible in double precision; I find that this happens for x >
     about 6, for all y.  On the other hand, I find that the case
     where we compute all of the sums is faster (at least with the
     precomputed expa2n2 table) until about x=10.  Furthermore, if we
     try to compute all of the sums for x > 20, I find that we
     sometimes run into numerical problems because underflow/overflow
     problems start to appear in the various coefficients of the sums,
     below.  Therefore, we use x < 10 here. */
  /*
  else if (x < 10) {
    double prod2ax = 1, prodm2ax = 1;
    double expx2;

    if (isnan(y))
      return C(y,y); */
    
    /* Somewhat ugly copy-and-paste duplication here, but I see significant
       speedups from using the special-case code with the precomputed
       exponential, and the x < 5e-4 special case is needed for accuracy. */
/*
    if (relerr == DBL_EPSILON) { // use precomputed exp(-a2*(n*n)) table
      if (x < 5e-4) { // compute sum4 and sum5 together as sum5-sum4
        const double x2 = x*x;
        expx2 = 1 - x2 * (1 - 0.5*x2); // exp(-x*x) via Taylor
        // compute exp(2*a*x) and exp(-2*a*x) via Taylor, to double precision
        const double ax2 = 1.036642960860171859744*x; // 2*a*x
        const double exp2ax =
          1 + ax2 * (1 + ax2 * (0.5 + 0.166666666666666666667*ax2));
        const double expm2ax =
          1 - ax2 * (1 - ax2 * (0.5 - 0.166666666666666666667*ax2));
        for (int n = 1; 1; ++n) {
          const double coef = expa2n2[n-1] * expx2 / (a2*(n*n) + y*y);
          prod2ax *= exp2ax;
          prodm2ax *= expm2ax;
          sum1 += coef;
          sum2 += coef * prodm2ax;
          sum3 += coef * prod2ax;
          
          // really = sum5 - sum4
          sum5 += coef * (2*a) * n * sinh_taylor((2*a)*n*x);
          
          // test convergence via sum3
          if (coef * prod2ax < relerr * sum3) break;
        }
      }
      else { // x > 5e-4, compute sum4 and sum5 separately
        expx2 = exp(-x*x);
        const double exp2ax = exp((2*a)*x), expm2ax = 1 / exp2ax;
        for (int n = 1; 1; ++n) {
          const double coef = expa2n2[n-1] * expx2 / (a2*(n*n) + y*y);
          prod2ax *= exp2ax;
          prodm2ax *= expm2ax;
          sum1 += coef;
          sum2 += coef * prodm2ax;
          sum4 += (coef * prodm2ax) * (a*n);
          sum3 += coef * prod2ax;
          sum5 += (coef * prod2ax) * (a*n);
          // test convergence via sum5, since this sum has the slowest decay
          if ((coef * prod2ax) * (a*n) < relerr * sum5) break;
        }
      }
    }
    else { // relerr != DBL_EPSILON, compute exp(-a2*(n*n)) on the fly
      const double exp2ax = exp((2*a)*x), expm2ax = 1 / exp2ax;
      if (x < 5e-4) { // compute sum4 and sum5 together as sum5-sum4
        const double x2 = x*x;
        expx2 = 1 - x2 * (1 - 0.5*x2); // exp(-x*x) via Taylor
        for (int n = 1; 1; ++n) {
          const double coef = exp(-a2*(n*n)) * expx2 / (a2*(n*n) + y*y);
          prod2ax *= exp2ax;
          prodm2ax *= expm2ax;
          sum1 += coef;
          sum2 += coef * prodm2ax;
          sum3 += coef * prod2ax;
          
          // really = sum5 - sum4
          sum5 += coef * (2*a) * n * sinh_taylor((2*a)*n*x);
          
          // test convergence via sum3
          if (coef * prod2ax < relerr * sum3) break;
        }
      }
      else { // x > 5e-4, compute sum4 and sum5 separately
        expx2 = exp(-x*x);
        for (int n = 1; 1; ++n) {
          const double coef = exp(-a2*(n*n)) * expx2 / (a2*(n*n) + y*y);
          prod2ax *= exp2ax;
          prodm2ax *= expm2ax;
          sum1 += coef;
          sum2 += coef * prodm2ax;
          sum4 += (coef * prodm2ax) * (a*n);
          sum3 += coef * prod2ax;
          sum5 += (coef * prod2ax) * (a*n);
          // test convergence via sum5, since this sum has the slowest decay
          if ((coef * prod2ax) * (a*n) < relerr * sum5) break;
        }
      }
    }
    const double expx2erfcxy = // avoid spurious overflow for large negative y
      y > -6 // for y < -6, erfcx(y) = 2*exp(y*y) to double precision
      ? expx2*FADDEEVA_RE(erfcx)(y) : 2*exp(y*y-x*x);
    if (y > 5) { // imaginary terms cancel
      const double sinxy = sin(x*y);
      ret = (expx2erfcxy - c*y*sum1) * cos(2*x*y)
        + (c*x*expx2) * sinxy * sinc(x*y, sinxy);
    }
    else {
      double xs = creal(z);
      const double sinxy = sin(xs*y);
      const double sin2xy = sin(2*xs*y), cos2xy = cos(2*xs*y);
      const double coef1 = expx2erfcxy - c*y*sum1;
      const double coef2 = c*xs*expx2;
      ret = C(coef1 * cos2xy + coef2 * sinxy * sinc(xs*y, sinxy),
              coef2 * sinc(2*xs*y, sin2xy) - coef1 * sin2xy);
    }
  }
  else { // x large: only sum3 & sum5 contribute (see above note)    
    if (isnan(x))
      return C(x,x);
    if (isnan(y))
      return C(y,y);

#if USE_CONTINUED_FRACTION
    ret = exp(-x*x); // |y| < 1e-10, so we only need exp(-x*x) term
#else
    if (y < 0) { */
      /* erfcx(y) ~ 2*exp(y*y) + (< 1) if y < 0, so
         erfcx(y)*exp(-x*x) ~ 2*exp(y*y-x*x) term may not be negligible
         if y*y - x*x > -36 or so.  So, compute this term just in case.
         We also need the -exp(-x*x) term to compute Re[w] accurately
         in the case where y is very small. */
      /*
      ret = cpolar(2*exp(y*y-x*x) - exp(-x*x), -2*creal(z)*y);
    }
    else
      ret = exp(-x*x); // not negligible in real part if y very small
#endif
    // (round instead of ceil as in original paper; note that x/a > 1 here)
    double n0 = floor(x/a + 0.5); // sum in both directions, starting at n0
    double dx = a*n0 - x;
    sum3 = exp(-dx*dx) / (a2*(n0*n0) + y*y);
    sum5 = a*n0 * sum3;
    double exp1 = exp(4*a*dx), exp1dn = 1;
    int dn;
    for (dn = 1; n0 - dn > 0; ++dn) { // loop over n0-dn and n0+dn terms
      double np = n0 + dn, nm = n0 - dn;
      double tp = exp(-sqr(a*dn+dx));
      double tm = tp * (exp1dn *= exp1); // trick to get tm from tp
      tp /= (a2*(np*np) + y*y);
      tm /= (a2*(nm*nm) + y*y);
      sum3 += tp + tm;
      sum5 += a * (np * tp + nm * tm);
      if (a * (np * tp + nm * tm) < relerr * sum5) goto finish;
    }
    while (1) { // loop over n0+dn terms only (since n0-dn <= 0)
      double np = n0 + dn++;
      double tp = exp(-sqr(a*dn+dx)) / (a2*(np*np) + y*y);
      sum3 += tp;
      sum5 += a * np * tp;
      if (a * np * tp < relerr * sum5) goto finish;
    }
  }
 finish:
  return ret + C((0.5*c)*y*(sum2+sum3), 
                 (0.5*c)*copysign(sum5-sum4, creal(z)));
                 */
    ( void )a2;
    ( void )c;
    
    ( void )sum1;
    ( void )sum2;
    ( void )sum3;
    ( void )sum4;
    ( void )sum5;
    
    if( ret == 0 )
    {
        if( ( out_re != 0 ) && ( out_im != 0 ) )
        {
            *out_re = result_re;
            *out_im = result_im;
        }
        else 
        {
            ret = -1;
        }
    }

    return ret;
}

/* end: sixtracklib/common/details/faddeeva.c */
