#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/be_beamfields/definitions.h"
    #include "sixtracklib/common/be_beamfields/abq2011_coeff.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( _GPUCODE ) && \
     defined( SIXTRL_CERRF_ABQ2011_N_FOURIER ) && \
            ( SIXTRL_CERRF_ABQ2011_N_FOURIER == 24 ) && \
     defined( SIXTRL_CERRF_ABQ2011_TM ) && ( SIXTRL_CERRF_ABQ2011_TM == 12 )

typedef SIXTRL_REAL_T real_type;

real_type const NS(CERRF_ABQ2011_FOURIER_COEFF)[ SIXTRL_CERRF_ABQ2011_N_FOURIER ] = {
    ( real_type )0.295408975150919337883027913890,      /* a_n[  0 ] */
    ( real_type )0.275840233292177084395258287749,      /* a_n[  1 ] */
    ( real_type )0.224573955224615866231619198223,      /* a_n[  2 ] */
    ( real_type )0.159414938273911722757388079389,      /* a_n[  3 ] */
    ( real_type )0.0986657664154541891084237249422,     /* a_n[  4 ] */
    ( real_type )0.0532441407876394120414705837561,     /* a_n[  5 ] */
    ( real_type )0.0250521500053936483557475638078,     /* a_n[  6 ] */
    ( real_type )0.0102774656705395362477551802420,     /* a_n[  7 ] */
    ( real_type )0.00367616433284484706364335443079,    /* a_n[  8 ] */
    ( real_type )0.00114649364124223317199757239908,    /* a_n[  9 ] */
    ( real_type )0.000311757015046197600406683642851,   /* a_n[ 10 ] */
    ( real_type )0.0000739143342960301487751427184143,  /* a_n[ 11 ] */
    ( real_type )0.0000152794934280083634658979605774,  /* a_n[ 12 ] */
    ( real_type )0.00000275395660822107093261423133381, /* a_n[ 13 ] */
    ( real_type )4.32785878190124505246159684324E-7,    /* a_n[ 14 ] */
    ( real_type )5.93003040874588104132914772669E-8,    /* a_n[ 15 ] */
    ( real_type )7.08449030774820424708618991843E-9,    /* a_n[ 16 ] */
    ( real_type )7.37952063581678039121116498488E-10,   /* a_n[ 17 ] */
    ( real_type )6.70217160600200763046136003593E-11,   /* a_n[ 18 ] */
    ( real_type )5.30726516347079017807414252726E-12,   /* a_n[ 19 ] */
    ( real_type )3.66432411346763916925386157070E-13,   /* a_n[ 20 ] */
    ( real_type )2.20589494494103134281934595834E-14,   /* a_n[ 21 ] */
    ( real_type )1.15782686262855878932236226031E-15,   /* a_n[ 22 ] */
    ( real_type )5.29871142946730483659082681849E-17,   /* a_n[ 23 ] */
};

#elif !defined( _GPUCODE ) && \
      ( !defined( SIXTRL_CERRF_ABQ2011_N_FOURIER ) || \
                ( SIXTRL_CERRF_ABQ2011_N_FOURIER != 24 ) || \
        !defined( SIXTRL_CERRF_ABQ2011_TM ) || \
                ( SIXTRL_CERRF_ABQ2011_TM != 12 ) )

    #error "precomputed fourier coefficients only provided for " \
           "SIXTRL_CERRF_ABQ2011_N_FOURIER == 24 and SIXTRL_CERRF_ABQ2011_TM == 12" \
           "-> provide your own tabulated data for other configurations"

#endif /* !defined( _GPUCODE ) */
