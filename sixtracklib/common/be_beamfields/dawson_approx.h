#ifndef SIXTRACKLIB_COMMON_BE_BEAMFIELDS_DAWSON_APPROX_H__
#define SIXTRACKLIB_COMMON_BE_BEAMFIELDS_DAWSON_APPROX_H__

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/be_beamfields/definitions.h"
    #include "sixtracklib/common/internal/math_constants.h"
    #include "sixtracklib/common/internal/math_factorial.h"
    #include "sixtracklib/common/internal/math_functions.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined( _GPUCODE ) && defined( __cplusplus )  */

SIXTRL_STATIC SIXTRL_FN int NS(dawson_n_interval)(
    SIXTRL_REAL_T x ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_REAL_T NS(dawson_xi)(
    int n_interval ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN SIXTRL_REAL_T NS(dawson_fz_xi)(
    int n_interval ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN int NS(dawson_nt_xi_abs_d10)(
    int n_interval ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN int NS(dawson_nt_xi_rel_d14)(
    int n_interval ) SIXTRL_NOEXCEPT;

SIXTRL_STATIC SIXTRL_FN void NS(dawson_cerrf)(
    SIXTRL_REAL_T const x, SIXTRL_REAL_T const y,
    SIXTRL_CERRF_RESULT_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT out_real,
    SIXTRL_CERRF_RESULT_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT out_imag
) SIXTRL_NOEXCEPT;

#if defined( SIXTRL_CERRF_USE_DAWSON_COEFF ) && \
           ( SIXTRL_CERRF_USE_DAWSON_COEFF >= 1 )

SIXTRL_STATIC SIXTRL_FN void NS(dawson_cerrf_coeff)(
    SIXTRL_REAL_T const x, SIXTRL_REAL_T const y,
    SIXTRL_CERRF_RESULT_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT out_real,
    SIXTRL_CERRF_RESULT_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT out_imag,
    SIXTRL_CERRF_DAWSON_COEFF_XI_DEC SIXTRL_REAL_T  const* SIXTRL_RESTRICT xi,
    SIXTRL_CERRF_DAWSON_COEFF_FZ_DEC SIXTRL_REAL_T  const* SIXTRL_RESTRICT Fz_xi,
    SIXTRL_CERRF_DAWSON_COEFF_NT_DEC SIXTRL_INT32_TYPE const* SIXTRL_RESTRICT Fz_nt
    #if ( SIXTRL_CERRF_USE_DAWSON_COEFF > 1 )
    , SIXTRL_CERRF_DAWSON_COEFF_TAYLOR_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT Fz_kk_xi
    #endif /* ( SIXTRL_CERRF_USE_DAWSON_COEFF > 0 ) */
) SIXTRL_NOEXCEPT;

#endif /* ( SIXTRL_CERRF_USE_DAWSON_COEFF >= 0 ) */

#if !defined( _GPUCODE )
SIXTRL_EXTERN SIXTRL_HOST_FN int NS(dawson_n_interval_ext)(
    SIXTRL_REAL_T x ) SIXTRL_NOEXCEPT;

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_REAL_T NS(dawson_xi_ext)(
    int n_interval ) SIXTRL_NOEXCEPT;

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_REAL_T NS(dawson_fz_xi_ext)(
    int n_interval ) SIXTRL_NOEXCEPT;

SIXTRL_EXTERN SIXTRL_HOST_FN int NS(dawson_nt_xi_abs_d10_ext)(
    int n_interval ) SIXTRL_NOEXCEPT;

SIXTRL_EXTERN SIXTRL_HOST_FN int NS(dawson_nt_xi_rel_d14_ext)(
    int n_interval ) SIXTRL_NOEXCEPT;

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(dawson_cerrf_nocoeff_ext)(
    SIXTRL_REAL_T const x, SIXTRL_REAL_T const y,
    SIXTRL_CERRF_RESULT_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT out_real,
    SIXTRL_CERRF_RESULT_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT out_imag
) SIXTRL_NOEXCEPT;

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(dawson_cerrf_ext)(
    SIXTRL_REAL_T const x, SIXTRL_REAL_T const y,
    SIXTRL_CERRF_RESULT_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT out_real,
    SIXTRL_CERRF_RESULT_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT out_imag
) SIXTRL_NOEXCEPT;

#endif /* !defined( _GPUCODE ) */

/* ************************************************************************ */

SIXTRL_INLINE int NS(dawson_n_interval)( SIXTRL_REAL_T x ) SIXTRL_NOEXCEPT
{
    typedef SIXTRL_REAL_T real_type;

    int const n_interval = ( int )NS(round)(
        ( x - ( real_type )SIXTRL_CERRF_DAWSON_XN_MIN ) /
              ( real_type )SIXTRL_CERRF_DAWSON_DX );

    SIXTRL_ASSERT( x >= ( real_type )SIXTRL_CERRF_DAWSON_X_MIN );
    SIXTRL_ASSERT( x <= ( real_type )SIXTRL_CERRF_DAWSON_X_MAX );

    return n_interval;
}

#if defined( SIXTRL_CERRF_DAWSON_N_XN   ) && ( SIXTRL_CERRF_DAWSON_N_XN   == 96 ) && \
    defined( SIXTRL_CERRF_DAWSON_XN_MIN ) && ( SIXTRL_CERRF_DAWSON_XN_MIN ==  0 ) && \
    defined( SIXTRL_CERRF_DAWSON_XN_MAX ) && ( SIXTRL_CERRF_DAWSON_XN_MAX == 20 )

SIXTRL_INLINE SIXTRL_REAL_T NS(dawson_xi)( int n_interval ) SIXTRL_NOEXCEPT
{
    typedef SIXTRL_REAL_T real_type;
    real_type xi;

    SIXTRL_ASSERT( n_interval >= 0 );
    SIXTRL_ASSERT( n_interval < ( int )SIXTRL_CERRF_DAWSON_N_XN );

    switch( n_interval )
    {
        case  1: { xi = ( real_type )+0.210526315789473684210526315789473684210526; break; }
        case  2: { xi = ( real_type )+0.421052631578947368421052631578947368421053; break; }
        case  3: { xi = ( real_type )+0.631578947368421052631578947368421052631579; break; }
        case  4: { xi = ( real_type )+0.842105263157894736842105263157894736842105; break; }
        case  5: { xi = ( real_type )+1.05263157894736842105263157894736842105263;  break; }
        case  6: { xi = ( real_type )+1.26315789473684210526315789473684210526316;  break; }
        case  7: { xi = ( real_type )+1.47368421052631578947368421052631578947368;  break; }
        case  8: { xi = ( real_type )+1.68421052631578947368421052631578947368421;  break; }
        case  9: { xi = ( real_type )+1.89473684210526315789473684210526315789474;  break; }
        case 10: { xi = ( real_type )+2.10526315789473684210526315789473684210526;  break; }
        case 11: { xi = ( real_type )+2.31578947368421052631578947368421052631579;  break; }
        case 12: { xi = ( real_type )+2.52631578947368421052631578947368421052632;  break; }
        case 13: { xi = ( real_type )+2.73684210526315789473684210526315789473684;  break; }
        case 14: { xi = ( real_type )+2.94736842105263157894736842105263157894737;  break; }
        case 15: { xi = ( real_type )+3.15789473684210526315789473684210526315789;  break; }
        case 16: { xi = ( real_type )+3.36842105263157894736842105263157894736842;  break; }
        case 17: { xi = ( real_type )+3.57894736842105263157894736842105263157895;  break; }
        case 18: { xi = ( real_type )+3.78947368421052631578947368421052631578947;  break; }
        case 19: { xi = ( real_type )+4.0;                                          break; }
        case 20: { xi = ( real_type )+4.21052631578947368421052631578947368421053;  break; }
        case 21: { xi = ( real_type )+4.42105263157894736842105263157894736842105;  break; }
        case 22: { xi = ( real_type )+4.63157894736842105263157894736842105263158;  break; }
        case 23: { xi = ( real_type )+4.84210526315789473684210526315789473684211;  break; }
        case 24: { xi = ( real_type )+5.05263157894736842105263157894736842105263;  break; }
        case 25: { xi = ( real_type )+5.26315789473684210526315789473684210526316;  break; }
        case 26: { xi = ( real_type )+5.47368421052631578947368421052631578947368;  break; }
        case 27: { xi = ( real_type )+5.68421052631578947368421052631578947368421;  break; }
        case 28: { xi = ( real_type )+5.89473684210526315789473684210526315789474;  break; }
        case 29: { xi = ( real_type )+6.10526315789473684210526315789473684210526;  break; }
        case 30: { xi = ( real_type )+6.31578947368421052631578947368421052631579;  break; }
        case 31: { xi = ( real_type )+6.52631578947368421052631578947368421052632;  break; }
        case 32: { xi = ( real_type )+6.73684210526315789473684210526315789473684;  break; }
        case 33: { xi = ( real_type )+6.94736842105263157894736842105263157894737;  break; }
        case 34: { xi = ( real_type )+7.15789473684210526315789473684210526315789;  break; }
        case 35: { xi = ( real_type )+7.36842105263157894736842105263157894736842;  break; }
        case 36: { xi = ( real_type )+7.57894736842105263157894736842105263157895;  break; }
        case 37: { xi = ( real_type )+7.78947368421052631578947368421052631578947;  break; }
        case 38: { xi = ( real_type )+8.0;                                          break; }
        case 39: { xi = ( real_type )+8.21052631578947368421052631578947368421053;  break; }
        case 40: { xi = ( real_type )+8.42105263157894736842105263157894736842105;  break; }
        case 41: { xi = ( real_type )+8.63157894736842105263157894736842105263158;  break; }
        case 42: { xi = ( real_type )+8.84210526315789473684210526315789473684211;  break; }
        case 43: { xi = ( real_type )+9.05263157894736842105263157894736842105263;  break; }
        case 44: { xi = ( real_type )+9.26315789473684210526315789473684210526316;  break; }
        case 45: { xi = ( real_type )+9.47368421052631578947368421052631578947368;  break; }
        case 46: { xi = ( real_type )+9.68421052631578947368421052631578947368421;  break; }
        case 47: { xi = ( real_type )+9.89473684210526315789473684210526315789474;  break; }
        case 48: { xi = ( real_type )+10.1052631578947368421052631578947368421053;  break; }
        case 49: { xi = ( real_type )+10.3157894736842105263157894736842105263158;  break; }
        case 50: { xi = ( real_type )+10.5263157894736842105263157894736842105263;  break; }
        case 51: { xi = ( real_type )+10.7368421052631578947368421052631578947368;  break; }
        case 52: { xi = ( real_type )+10.9473684210526315789473684210526315789474;  break; }
        case 53: { xi = ( real_type )+11.1578947368421052631578947368421052631579;  break; }
        case 54: { xi = ( real_type )+11.3684210526315789473684210526315789473684;  break; }
        case 55: { xi = ( real_type )+11.5789473684210526315789473684210526315789;  break; }
        case 56: { xi = ( real_type )+11.7894736842105263157894736842105263157895;  break; }
        case 57: { xi = ( real_type )+12.0;                                         break; }
        case 58: { xi = ( real_type )+12.2105263157894736842105263157894736842105;  break; }
        case 59: { xi = ( real_type )+12.4210526315789473684210526315789473684211;  break; }
        case 60: { xi = ( real_type )+12.6315789473684210526315789473684210526316;  break; }
        case 61: { xi = ( real_type )+12.8421052631578947368421052631578947368421;  break; }
        case 62: { xi = ( real_type )+13.0526315789473684210526315789473684210526;  break; }
        case 63: { xi = ( real_type )+13.2631578947368421052631578947368421052632;  break; }
        case 64: { xi = ( real_type )+13.4736842105263157894736842105263157894737;  break; }
        case 65: { xi = ( real_type )+13.6842105263157894736842105263157894736842;  break; }
        case 66: { xi = ( real_type )+13.8947368421052631578947368421052631578947;  break; }
        case 67: { xi = ( real_type )+14.1052631578947368421052631578947368421053;  break; }
        case 68: { xi = ( real_type )+14.3157894736842105263157894736842105263158;  break; }
        case 69: { xi = ( real_type )+14.5263157894736842105263157894736842105263;  break; }
        case 70: { xi = ( real_type )+14.7368421052631578947368421052631578947368;  break; }
        case 71: { xi = ( real_type )+14.9473684210526315789473684210526315789474;  break; }
        case 72: { xi = ( real_type )+15.1578947368421052631578947368421052631579;  break; }
        case 73: { xi = ( real_type )+15.3684210526315789473684210526315789473684;  break; }
        case 74: { xi = ( real_type )+15.5789473684210526315789473684210526315789;  break; }
        case 75: { xi = ( real_type )+15.7894736842105263157894736842105263157895;  break; }
        case 76: { xi = ( real_type )+16.0;                                         break; }
        case 77: { xi = ( real_type )+16.2105263157894736842105263157894736842105;  break; }
        case 78: { xi = ( real_type )+16.4210526315789473684210526315789473684211;  break; }
        case 79: { xi = ( real_type )+16.6315789473684210526315789473684210526316;  break; }
        case 80: { xi = ( real_type )+16.8421052631578947368421052631578947368421;  break; }
        case 81: { xi = ( real_type )+17.0526315789473684210526315789473684210526;  break; }
        case 82: { xi = ( real_type )+17.2631578947368421052631578947368421052632;  break; }
        case 83: { xi = ( real_type )+17.4736842105263157894736842105263157894737;  break; }
        case 84: { xi = ( real_type )+17.6842105263157894736842105263157894736842;  break; }
        case 85: { xi = ( real_type )+17.8947368421052631578947368421052631578947;  break; }
        case 86: { xi = ( real_type )+18.1052631578947368421052631578947368421053;  break; }
        case 87: { xi = ( real_type )+18.3157894736842105263157894736842105263158;  break; }
        case 88: { xi = ( real_type )+18.5263157894736842105263157894736842105263;  break; }
        case 89: { xi = ( real_type )+18.7368421052631578947368421052631578947368;  break; }
        case 90: { xi = ( real_type )+18.9473684210526315789473684210526315789474;  break; }
        case 91: { xi = ( real_type )+19.1578947368421052631578947368421052631579;  break; }
        case 92: { xi = ( real_type )+19.3684210526315789473684210526315789473684;  break; }
        case 93: { xi = ( real_type )+19.5789473684210526315789473684210526315789;  break; }
        case 94: { xi = ( real_type )+19.7894736842105263157894736842105263157895;  break; }
        case 95: { xi = ( real_type )+20.0;                                         break; }
        default: { xi = ( real_type )+0.0; }
    };

    return xi;
}

SIXTRL_INLINE SIXTRL_REAL_T NS(dawson_fz_xi)( int n_interval ) SIXTRL_NOEXCEPT
{
    typedef SIXTRL_REAL_T real_type;
    real_type Fz;
    SIXTRL_ASSERT( n_interval >= 0 );
    SIXTRL_ASSERT( n_interval < ( int )SIXTRL_CERRF_DAWSON_N_XN );

    switch( n_interval )
    {
        case   1: { Fz = ( real_type )+0.204414676423299307852205738133635368467483;     break; }
        case   2: { Fz = ( real_type )+0.374645391062146565792903138151666837723415;     break; }
        case   3: { Fz = ( real_type )+0.487620779526159032600085065955994007602486;     break; }
        case   4: { Fz = ( real_type )+0.537225179665383656999471044984894541890407;     break; }
        case   5: { Fz = ( real_type )+0.532844209232800442512093957303667626022578;     break; }
        case   6: { Fz = ( real_type )+0.492641603905036547612208594020732218930945;     break; }
        case   7: { Fz = ( real_type )+0.435738088853804929355870047140050578499873;     break; }
        case   8: { Fz = ( real_type )+0.376790235869403627825298481421444312322929;     break; }
        case   9: { Fz = ( real_type )+0.324173662408523334351848683232449479096986;     break; }
        case  10: { Fz = ( real_type )+0.280919842933869131514286981833391921653268;     break; }
        case  11: { Fz = ( real_type )+0.246774606738902005981196594245619278443541;     break; }
        case  12: { Fz = ( real_type )+0.220091142230898461206668437270154827354414;     break; }
        case  13: { Fz = ( real_type )+0.199018995238297948215403032191131604653406;     break; }
        case  14: { Fz = ( real_type )+0.182022563204593041262776480636221471420828;     break; }
        case  15: { Fz = ( real_type )+0.167986296744659197005015872159861318018257;     break; }
        case  16: { Fz = ( real_type )+0.156145368002923681693730536418605551404876;     break; }
        case  17: { Fz = ( real_type )+0.145980869408444698072701944127637946509207;     break; }
        case  18: { Fz = ( real_type )+0.137133914188937040815955863740722859588214;     break; }
        case  19: { Fz = ( real_type )+0.129348001236005115591470526257665945278322;     break; }
        case  20: { Fz = ( real_type )+0.122433401676389459794733090423992918624035;     break; }
        case  21: { Fz = ( real_type )+0.116245536536035770230943819646142346156529;     break; }
        case  22: { Fz = ( real_type )+0.110671559758965998392672056514515221809768;     break; }
        case  23: { Fz = ( real_type )+0.105621693364230228188040070800433983455213;     break; }
        case  24: { Fz = ( real_type )+0.101023390864058696333111557061547844543024;     break; }
        case  25: { Fz = ( real_type )+0.0968172599708731096134786108738627105980966;    break; }
        case  26: { Fz = ( real_type )+0.0929541296762774182586342664150765738247869;    break; }
        case  27: { Fz = ( real_type )+0.0893928904886096481956461370618202971912525;    break; }
        case  28: { Fz = ( real_type )+0.0860988731325164934603718599503360383247846;    break; }
        case  29: { Fz = ( real_type )+0.0830426115241511768702601167929215319557302;    break; }
        case  30: { Fz = ( real_type )+0.0801988855781815476048148063389218190822483;    break; }
        case  31: { Fz = ( real_type )+0.0775459713212218568287642349004942070222095;    break; }
        case  32: { Fz = ( real_type )+0.0750650468966655593891682810790162389606471;    break; }
        case  33: { Fz = ( real_type )+0.0727397173579474671216481741375405058188894;    break; }
        case  34: { Fz = ( real_type )+0.0705556310552950551974987896970666311765803;    break; }
        case  35: { Fz = ( real_type )+0.0685001674055154771379396766403437184862358;    break; }
        case  36: { Fz = ( real_type )+0.0665621808371330882020689517139535800601475;    break; }
        case  37: { Fz = ( real_type )+0.0647317893379955483199347836217465191040756;    break; }
        case  38: { Fz = ( real_type )+0.0630001987075533879192457295169672688411135;    break; }
        case  39: { Fz = ( real_type )+0.0613595556079285561738754088706845246229021;    break; }
        case  40: { Fz = ( real_type )+0.0598028240070970025997168069552382204455551;    break; }
        case  41: { Fz = ( real_type )+0.0583236807470069792151640171108127990614153;    break; }
        case  42: { Fz = ( real_type )+0.056916426843506876837940629331046129974024;     break; }
        case  43: { Fz = ( real_type )+0.0555759118010602678845137769255961249291392;    break; }
        case  44: { Fz = ( real_type )+0.0542974687523859076954105481536284958849097;    break; }
        case  45: { Fz = ( real_type )+0.0530768586471334283581234719746748841532657;    break; }
        case  46: { Fz = ( real_type )+0.0519102220411792443100980154384748583981615;    break; }
        case  47: { Fz = ( real_type )+0.0507940372987049051842427501977029926302941;    break; }
        case  48: { Fz = ( real_type )+0.0497250842279749621621125383083789598731151;    break; }
        case  49: { Fz = ( real_type )+0.0487004123398296771663508633922339236261419;    break; }
        case  50: { Fz = ( real_type )+0.047717313054061251201007710934289736649871;     break; }
        case  51: { Fz = ( real_type )+0.0467732952896975256047898303754890525957012;    break; }
        case  52: { Fz = ( real_type )+0.0458660639658681409761153082730452877258985;    break; }
        case  53: { Fz = ( real_type )+0.0449935010144609170081544834090713795542632;    break; }
        case  54: { Fz = ( real_type )+0.0441536485672899944916871372517739024921971;    break; }
        case  55: { Fz = ( real_type )+0.043344694031490533589896575933680459971779;     break; }
        case  56: { Fz = ( real_type )+0.0425649568093194227216046680664783565680491;    break; }
        case  57: { Fz = ( real_type )+0.0418128764539882603179291175888089559911153;    break; }
        case  58: { Fz = ( real_type )+0.0410870020829319836299372632542900813461959;    break; }
        case  59: { Fz = ( real_type )+0.040385982894919183194249326844771251876668;     break; }
        case  60: { Fz = ( real_type )+0.0397085596585740386496577961382040023237114;    break; }
        case  61: { Fz = ( real_type )+0.0390535570577986355292801429237067764008847;    break; }
        case  62: { Fz = ( real_type )+0.0384198767947746146529233779408053168285224;    break; }
        case  63: { Fz = ( real_type )+0.0378064913642431632452704510295471834595252;    break; }
        case  64: { Fz = ( real_type )+0.0372124384238227623469476082943136111737884;    break; }
        case  65: { Fz = ( real_type )+0.03663681569465044069256321751749903523137;      break; }
        case  66: { Fz = ( real_type )+0.0360787763348041155456745796410421736938601;    break; }
        case  67: { Fz = ( real_type )+0.0355375247350140356986075299224587181901788;    break; }
        case  68: { Fz = ( real_type )+0.0350123126922469469101154386321616300405276;    break; }
        case  69: { Fz = ( real_type )+0.03450243592202935649461976396376297559226;      break; }
        case  70: { Fz = ( real_type )+0.0340072308749504430827663193995038626461181;    break; }
        case  71: { Fz = ( real_type )+0.0335260718267648448243884603329401316619297;    break; }
        case  72: { Fz = ( real_type )+0.0330583682149807959933751016989241274063893;    break; }
        case  73: { Fz = ( real_type )+0.0326035621978688818527138577945325741429771;    break; }
        case  74: { Fz = ( real_type )+0.032161126414470307789361137738387026075656;     break; }
        case  75: { Fz = ( real_type )+0.0317305619265100815660500263462697296874182;    break; }
        case  76: { Fz = ( real_type )+0.03131139632518461178355335195230901508146;      break; }
        case  77: { Fz = ( real_type )+0.0309031819875807833185597950940297846685743;    break; }
        case  78: { Fz = ( real_type )+0.0305054944690889621935730957812503043992676;    break; }
        case  79: { Fz = ( real_type )+0.0301179310195570542608243395223315653783517;    break; }
        case  80: { Fz = ( real_type )+0.0297401092122126450370115395178546539000625;    break; }
        case  81: { Fz = ( real_type )+0.0293716656754489088365959105310074286753141;    break; }
        case  82: { Fz = ( real_type )+0.0290122549185626204232460038847629393025906;    break; }
        case  83: { Fz = ( real_type )+0.0286615482434347706561739985173263709460918;    break; }
        case  84: { Fz = ( real_type )+0.028319232734848067795097432772015190860891;     break; }
        case  85: { Fz = ( real_type )+0.0279850103229300360218746106322339780429843;    break; }
        case  86: { Fz = ( real_type )+0.0276585969117343482506150690312596643118447;    break; }
        case  87: { Fz = ( real_type )+0.0273397215685935480035125872749087135987964;    break; }
        case  88: { Fz = ( real_type )+0.0270281257693319151623337823932836698764564;    break; }
        case  89: { Fz = ( real_type )+0.0267235626949087363281474632612449247565235;    break; }
        case  90: { Fz = ( real_type )+0.026425796575435644735625484197407954040757;     break; }
        case  91: { Fz = ( real_type )+0.0261346020778743874048668010927461155846186;    break; }
        case  92: { Fz = ( real_type )+0.0258497637340652429455729256209668852097065;    break; }
        case  93: { Fz = ( real_type )+0.0255710754060192979083778000275314785319582;    break; }
        case  94: { Fz = ( real_type )+0.0252983397856513709121225761435040371463168;    break; }
        case  95: { Fz = ( real_type )+0.0250313679264036719469949523478235318685783;    break; }
        default:  { Fz = ( real_type )0.0; }
    };

    return Fz;
}

SIXTRL_INLINE int NS(dawson_nt_xi_abs_d10)( int n_interval ) SIXTRL_NOEXCEPT
{
    int N_Taylor = ( int )SIXTRL_CERRF_DAWSON_MAX_N_TAYLOR_ABS_D10;

    #if !defined( SIXTRL_CERRF_DAWSON_USE_MAX_N_TAYLOR ) || \
                ( SIXTRL_CERRF_DAWSON_USE_MAX_N_TAYLOR != 1 )

    switch( n_interval )
    {
        case    0: { N_Taylor = ( int )25;    break; }
        case    1: { N_Taylor = ( int )25;    break; }
        case    2: { N_Taylor = ( int )25;    break; }
        case    3: { N_Taylor = ( int )24;    break; }
        case    4: { N_Taylor = ( int )25;    break; }
        case    5: { N_Taylor = ( int )24;    break; }
        case    6: { N_Taylor = ( int )24;    break; }
        case    7: { N_Taylor = ( int )24;    break; }
        case    8: { N_Taylor = ( int )24;    break; }
        case    9: { N_Taylor = ( int )24;    break; }
        case   10: { N_Taylor = ( int )24;    break; }
        case   11: { N_Taylor = ( int )23;    break; }
        case   12: { N_Taylor = ( int )24;    break; }
        case   13: { N_Taylor = ( int )23;    break; }
        case   14: { N_Taylor = ( int )24;    break; }
        case   15: { N_Taylor = ( int )23;    break; }
        case   16: { N_Taylor = ( int )23;    break; }
        case   17: { N_Taylor = ( int )22;    break; }
        case   18: { N_Taylor = ( int )22;    break; }
        case   19: { N_Taylor = ( int )21;    break; }
        case   20: { N_Taylor = ( int )21;    break; }
        case   21: { N_Taylor = ( int )21;    break; }
        case   22: { N_Taylor = ( int )21;    break; }
        case   23: { N_Taylor = ( int )20;    break; }
        case   24: { N_Taylor = ( int )19;    break; }
        case   25: { N_Taylor = ( int )19;    break; }
        case   26: { N_Taylor = ( int )19;    break; }
        case   27: { N_Taylor = ( int )17;    break; }
        case   28: { N_Taylor = ( int )17;    break; }
        case   29: { N_Taylor = ( int )17;    break; }
        case   30: { N_Taylor = ( int )16;    break; }
        case   31: { N_Taylor = ( int )16;    break; }
        case   32: { N_Taylor = ( int )15;    break; }
        case   33: { N_Taylor = ( int )15;    break; }
        case   34: { N_Taylor = ( int )15;    break; }
        case   35: { N_Taylor = ( int )15;    break; }
        case   36: { N_Taylor = ( int )15;    break; }
        case   37: { N_Taylor = ( int )14;    break; }
        case   38: { N_Taylor = ( int )14;    break; }
        case   39: { N_Taylor = ( int )14;    break; }
        case   40: { N_Taylor = ( int )14;    break; }
        case   41: { N_Taylor = ( int )14;    break; }
        case   42: { N_Taylor = ( int )14;    break; }
        case   43: { N_Taylor = ( int )13;    break; }
        case   44: { N_Taylor = ( int )13;    break; }
        case   45: { N_Taylor = ( int )13;    break; }
        case   46: { N_Taylor = ( int )13;    break; }
        case   47: { N_Taylor = ( int )13;    break; }
        case   48: { N_Taylor = ( int )13;    break; }
        case   49: { N_Taylor = ( int )13;    break; }
        case   50: { N_Taylor = ( int )13;    break; }
        case   51: { N_Taylor = ( int )13;    break; }
        case   52: { N_Taylor = ( int )12;    break; }
        case   53: { N_Taylor = ( int )12;    break; }
        case   54: { N_Taylor = ( int )12;    break; }
        case   55: { N_Taylor = ( int )12;    break; }
        case   56: { N_Taylor = ( int )12;    break; }
        case   57: { N_Taylor = ( int )12;    break; }
        case   58: { N_Taylor = ( int )12;    break; }
        case   59: { N_Taylor = ( int )12;    break; }
        case   60: { N_Taylor = ( int )12;    break; }
        case   61: { N_Taylor = ( int )12;    break; }
        case   62: { N_Taylor = ( int )12;    break; }
        case   63: { N_Taylor = ( int )12;    break; }
        case   64: { N_Taylor = ( int )12;    break; }
        case   65: { N_Taylor = ( int )12;    break; }
        case   66: { N_Taylor = ( int )12;    break; }
        case   67: { N_Taylor = ( int )11;    break; }
        case   68: { N_Taylor = ( int )11;    break; }
        case   69: { N_Taylor = ( int )11;    break; }
        case   70: { N_Taylor = ( int )11;    break; }
        case   71: { N_Taylor = ( int )11;    break; }
        case   72: { N_Taylor = ( int )11;    break; }
        case   73: { N_Taylor = ( int )11;    break; }
        case   74: { N_Taylor = ( int )11;    break; }
        case   75: { N_Taylor = ( int )11;    break; }
        case   76: { N_Taylor = ( int )11;    break; }
        case   77: { N_Taylor = ( int )11;    break; }
        case   78: { N_Taylor = ( int )11;    break; }
        case   79: { N_Taylor = ( int )11;    break; }
        case   80: { N_Taylor = ( int )11;    break; }
        case   81: { N_Taylor = ( int )11;    break; }
        case   82: { N_Taylor = ( int )11;    break; }
        case   83: { N_Taylor = ( int )11;    break; }
        case   84: { N_Taylor = ( int )11;    break; }
        case   85: { N_Taylor = ( int )11;    break; }
        case   86: { N_Taylor = ( int )11;    break; }
        case   87: { N_Taylor = ( int )11;    break; }
        case   88: { N_Taylor = ( int )11;    break; }
        case   89: { N_Taylor = ( int )11;    break; }
        case   90: { N_Taylor = ( int )10;    break; }
        case   91: { N_Taylor = ( int )10;    break; }
        case   92: { N_Taylor = ( int )10;    break; }
        case   93: { N_Taylor = ( int )10;    break; }
        case   94: { N_Taylor = ( int )10;    break; }
        case   95: { N_Taylor = ( int )10;    break; }
        default:   { N_Taylor = ( int )SIXTRL_CERRF_DAWSON_MAX_N_TAYLOR_ABS_D10; }
    };

    #endif /* ( SIXTRL_CERRF_DAWSON_USE_MAX_N_TAYLOR ) */

    return N_Taylor;
}

SIXTRL_INLINE int NS(dawson_nt_xi_rel_d14)( int n_interval ) SIXTRL_NOEXCEPT
{
    int N_Taylor = ( int )SIXTRL_CERRF_DAWSON_MAX_N_TAYLOR_REL_D14;

    #if !defined( SIXTRL_CERRF_DAWSON_USE_MAX_N_TAYLOR ) || \
                ( SIXTRL_CERRF_DAWSON_USE_MAX_N_TAYLOR != 1 )

    switch( n_interval )
    {
        case    0: { N_Taylor = ( int )25;    break; }
        case    1: { N_Taylor = ( int )25;    break; }
        case    2: { N_Taylor = ( int )25;    break; }
        case    3: { N_Taylor = ( int )24;    break; }
        case    4: { N_Taylor = ( int )25;    break; }
        case    5: { N_Taylor = ( int )24;    break; }
        case    6: { N_Taylor = ( int )24;    break; }
        case    7: { N_Taylor = ( int )24;    break; }
        case    8: { N_Taylor = ( int )24;    break; }
        case    9: { N_Taylor = ( int )24;    break; }
        case   10: { N_Taylor = ( int )24;    break; }
        case   11: { N_Taylor = ( int )23;    break; }
        case   12: { N_Taylor = ( int )24;    break; }
        case   13: { N_Taylor = ( int )23;    break; }
        case   14: { N_Taylor = ( int )24;    break; }
        case   15: { N_Taylor = ( int )23;    break; }
        case   16: { N_Taylor = ( int )23;    break; }
        case   17: { N_Taylor = ( int )22;    break; }
        case   18: { N_Taylor = ( int )22;    break; }
        case   19: { N_Taylor = ( int )21;    break; }
        case   20: { N_Taylor = ( int )21;    break; }
        case   21: { N_Taylor = ( int )21;    break; }
        case   22: { N_Taylor = ( int )21;    break; }
        case   23: { N_Taylor = ( int )20;    break; }
        case   24: { N_Taylor = ( int )19;    break; }
        case   25: { N_Taylor = ( int )19;    break; }
        case   26: { N_Taylor = ( int )19;    break; }
        case   27: { N_Taylor = ( int )17;    break; }
        case   28: { N_Taylor = ( int )17;    break; }
        case   29: { N_Taylor = ( int )17;    break; }
        case   30: { N_Taylor = ( int )16;    break; }
        case   31: { N_Taylor = ( int )16;    break; }
        case   32: { N_Taylor = ( int )15;    break; }
        case   33: { N_Taylor = ( int )15;    break; }
        case   34: { N_Taylor = ( int )15;    break; }
        case   35: { N_Taylor = ( int )15;    break; }
        case   36: { N_Taylor = ( int )15;    break; }
        case   37: { N_Taylor = ( int )14;    break; }
        case   38: { N_Taylor = ( int )14;    break; }
        case   39: { N_Taylor = ( int )14;    break; }
        case   40: { N_Taylor = ( int )14;    break; }
        case   41: { N_Taylor = ( int )14;    break; }
        case   42: { N_Taylor = ( int )14;    break; }
        case   43: { N_Taylor = ( int )13;    break; }
        case   44: { N_Taylor = ( int )13;    break; }
        case   45: { N_Taylor = ( int )13;    break; }
        case   46: { N_Taylor = ( int )13;    break; }
        case   47: { N_Taylor = ( int )13;    break; }
        case   48: { N_Taylor = ( int )13;    break; }
        case   49: { N_Taylor = ( int )13;    break; }
        case   50: { N_Taylor = ( int )13;    break; }
        case   51: { N_Taylor = ( int )13;    break; }
        case   52: { N_Taylor = ( int )12;    break; }
        case   53: { N_Taylor = ( int )12;    break; }
        case   54: { N_Taylor = ( int )12;    break; }
        case   55: { N_Taylor = ( int )12;    break; }
        case   56: { N_Taylor = ( int )12;    break; }
        case   57: { N_Taylor = ( int )12;    break; }
        case   58: { N_Taylor = ( int )12;    break; }
        case   59: { N_Taylor = ( int )12;    break; }
        case   60: { N_Taylor = ( int )12;    break; }
        case   61: { N_Taylor = ( int )12;    break; }
        case   62: { N_Taylor = ( int )12;    break; }
        case   63: { N_Taylor = ( int )12;    break; }
        case   64: { N_Taylor = ( int )12;    break; }
        case   65: { N_Taylor = ( int )12;    break; }
        case   66: { N_Taylor = ( int )12;    break; }
        case   67: { N_Taylor = ( int )11;    break; }
        case   68: { N_Taylor = ( int )11;    break; }
        case   69: { N_Taylor = ( int )11;    break; }
        case   70: { N_Taylor = ( int )11;    break; }
        case   71: { N_Taylor = ( int )11;    break; }
        case   72: { N_Taylor = ( int )11;    break; }
        case   73: { N_Taylor = ( int )11;    break; }
        case   74: { N_Taylor = ( int )11;    break; }
        case   75: { N_Taylor = ( int )11;    break; }
        case   76: { N_Taylor = ( int )11;    break; }
        case   77: { N_Taylor = ( int )11;    break; }
        case   78: { N_Taylor = ( int )11;    break; }
        case   79: { N_Taylor = ( int )11;    break; }
        case   80: { N_Taylor = ( int )11;    break; }
        case   81: { N_Taylor = ( int )11;    break; }
        case   82: { N_Taylor = ( int )11;    break; }
        case   83: { N_Taylor = ( int )11;    break; }
        case   84: { N_Taylor = ( int )11;    break; }
        case   85: { N_Taylor = ( int )11;    break; }
        case   86: { N_Taylor = ( int )11;    break; }
        case   87: { N_Taylor = ( int )11;    break; }
        case   88: { N_Taylor = ( int )11;    break; }
        case   89: { N_Taylor = ( int )11;    break; }
        case   90: { N_Taylor = ( int )10;    break; }
        case   91: { N_Taylor = ( int )10;    break; }
        case   92: { N_Taylor = ( int )10;    break; }
        case   93: { N_Taylor = ( int )10;    break; }
        case   94: { N_Taylor = ( int )10;    break; }
        case   95: { N_Taylor = ( int )10;    break; }
        default:   { N_Taylor = ( int )SIXTRL_CERRF_DAWSON_MAX_N_TAYLOR_REL_D14; }
    };

    #endif /* ( SIXTRL_CERRF_DAWSON_USE_MAX_N_TAYLOR ) */

    return N_Taylor;
}

#endif /* N = 96, X0 = 0.0, x95 = 20.0 */

SIXTRL_INLINE void NS(dawson_cerrf)(
    SIXTRL_REAL_T const x, SIXTRL_REAL_T const y,
    SIXTRL_CERRF_RESULT_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT out_re,
    SIXTRL_CERRF_RESULT_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT out_im
) SIXTRL_NOEXCEPT
{
    typedef SIXTRL_REAL_T real_type;

    int const n_interval     =  NS(dawson_n_interval)( x );
    real_type dn_minus_1     =  NS(dawson_fz_xi)( n_interval );
    real_type const minus_xn = -NS(dawson_xi)( n_interval );
    real_type const dz_re    = x + minus_xn;

    real_type dn = ( real_type )1. + ( real_type )2. * minus_xn * dn_minus_1;

    real_type dn_plus_1      =  ( real_type )0.0;
    real_type dz_pow_n_re    =  ( real_type )1.0;
    real_type dz_pow_n_im    =  ( real_type )0.0;

    real_type nn             =  ( real_type )1.0;
    real_type temp           = -( real_type )2. * x * y;
    real_type sum_daw_z_re   = dn_minus_1;
    real_type sum_daw_z_im   = ( real_type )0.0;

    real_type sum_re, sum_im, inv_factorial;
    SIXTRL_RESULT_PTR_DEC real_type exp_sin_two_x_y;
    SIXTRL_RESULT_PTR_DEC real_type exp_cos_two_x_y;

    #if ( SIXTRL_CERRF_METHOD == 0 ) || ( SIXTRL_CERRF_METHOD == 1 )
    int const N_Taylor = NS(dawson_nt_xi_abs_d10)( n_interval );
    #else /* ( SIXTRL_CERRF_METHOD > 1 ) */
    int const N_Taylor = NS(dawson_nt_xi_rel_d14)( n_interval );
    #endif /* SIXTRL_CERRF_METHOD */
    int n = 1;

    NS(sincos)( temp, &exp_sin_two_x_y, &exp_cos_two_x_y );

    temp = ( y - x ) * ( y + x );
    temp = NS(exp)( temp );
    exp_sin_two_x_y *= temp;
    exp_cos_two_x_y *= temp;

    SIXTRL_ASSERT( out_re != NULL );
    SIXTRL_ASSERT( out_im != NULL );

    SIXTRL_ASSERT( x >= ( real_type )SIXTRL_CERRF_DAWSON_X_MIN );
    SIXTRL_ASSERT( x <= ( real_type )SIXTRL_CERRF_DAWSON_X_MAX );
    SIXTRL_ASSERT( y >= ( real_type )SIXTRL_CERRF_DAWSON_Y_MIN );
    SIXTRL_ASSERT( y <= ( real_type )SIXTRL_CERRF_DAWSON_Y_MAX );

    for( ; n < N_Taylor ; ++n, nn += ( real_type )1.0 )
    {
        temp           = ( minus_xn * dn ) - ( nn * dn_minus_1 );
        dn_plus_1      = ( real_type )2. * temp;
        temp           = dz_pow_n_re * dz_re - dz_pow_n_im * y;
        dz_pow_n_im   *= dz_re;
        dz_pow_n_im   += dz_pow_n_re * y;
        dz_pow_n_re    = temp;

        sum_re         = dz_pow_n_re * dn;
        sum_im         = dz_pow_n_im * dn;
        dn_minus_1     = dn;
        dn             = dn_plus_1;
        inv_factorial  = NS(Math_inv_factorial)( n );

        sum_daw_z_re  += sum_re * inv_factorial;
        sum_daw_z_im  += sum_im * inv_factorial;
    }

    *out_re = exp_cos_two_x_y - NS(MathConst_two_over_sqrt_pi)() * sum_daw_z_im;
    *out_im = exp_sin_two_x_y + NS(MathConst_two_over_sqrt_pi)() * sum_daw_z_re;
}

#if defined( SIXTRL_CERRF_USE_DAWSON_COEFF ) && \
           ( SIXTRL_CERRF_USE_DAWSON_COEFF >= 1 )

SIXTRL_INLINE void NS(dawson_cerrf_coeff)(
    SIXTRL_REAL_T in_real, SIXTRL_REAL_T in_imag,
    SIXTRL_CERRF_RESULT_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT out_re,
    SIXTRL_CERRF_RESULT_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT out_im,
    SIXTRL_CERRF_DAWSON_COEFF_XI_DEC SIXTRL_REAL_T  const* SIXTRL_RESTRICT xi,
    SIXTRL_CERRF_DAWSON_COEFF_FZ_DEC SIXTRL_REAL_T  const* SIXTRL_RESTRICT Fz_xi,
    SIXTRL_CERRF_DAWSON_COEFF_NT_DEC SIXTRL_INT32_TYPE const* SIXTRL_RESTRICT Fz_nt
    #if ( SIXTRL_CERRF_USE_DAWSON_COEFF > 1 )
    , SIXTRL_CERRF_DAWSON_COEFF_TAYLOR_DEC SIXTRL_REAL_T const* SIXTRL_RESTRICT Fz_kk_xi
    #endif /* ( SIXTRL_CERRF_USE_DAWSON_COEFF > 0 ) */
) SIXTRL_NOEXCEPT
{
    int const n_interval     =  NS(dawson_n_interval)( x );
    real_type const minus_xn = -NS(CERRF_DAWSON_XI)[ n_interval ];
    real_type const dz_re    = x + minus_xn;
    real_type temp           = -( real_type )2. * x * y;
    real_type sum_daw_z_im;

    SIXTRL_RESULT_PTR_DEC real_type exp_sin_two_x_y;
    SIXTRL_RESULT_PTR_DEC real_type exp_cos_two_x_y;s

    #if ( SIXTRL_CERRF_METHOD == 0 ) || ( SIXTRL_CERRF_METHOD == 1 )
        #if !defined( SIXTRL_CERRF_DAWSON_USE_MAX_N_TAYLOR ) || \
                    ( SIXTRL_CERRF_DAWSON_USE_MAX_N_TAYLOR != 1 )
        int const N_Taylor = NS(CERRF_DAWSON_NT_XI_ABS_D10)[ n_interval ];

        #else /* ( SIXTRL_CERRF_DAWSON_USE_MAX_N_TAYLOR == 1 ) */
        int const N_Taylor = ( int )SIXTRL_CERRF_DAWSON_MAX_N_TAYLOR_ABS_D10;

        #endif /* ( SIXTRL_CERRF_DAWSON_USE_MAX_N_TAYLOR ) */
    #else
        #if !defined( SIXTRL_CERRF_DAWSON_USE_MAX_N_TAYLOR ) || \
                    ( SIXTRL_CERRF_DAWSON_USE_MAX_N_TAYLOR != 1 )
        int const N_Taylor = NS(CERRF_DAWSON_NT_XI_REL_D14)[ n_interval ];

        #else /* ( SIXTRL_CERRF_DAWSON_USE_MAX_N_TAYLOR == 1 ) */
        int const N_Taylor = ( int )SIXTRL_CERRF_DAWSON_MAX_N_TAYLOR_REL_D14;

        #endif /* ( SIXTRL_CERRF_DAWSON_USE_MAX_N_TAYLOR ) */
    #endif /* SIXTRL_CERRF_METHOD */

    #if ( SIXTRL_CERRF_USE_DAWSON_COEFF > 0 )
    int jj     = n_interval * ( int )SIXTRL_CERRF_DAWSON_N_TAYLOR_COEFF_PER_XN;
    int jj_end = jj + N_Taylor;

    real_type dz_pow_n_re    = dz_re;
    real_type dz_pow_n_im    = y;
    real_type dn             = Fz_kk_xi[ jj ];
    real_type sum_daw_z_re   = dn;

    #else  /* ( SIXTRL_CERRF_USE_DAWSON_COEFF == 0  -> recursion ) */
    real_type dz_pow_n_re   =  ( real_type )1.0;
    real_type dz_pow_n_im   =  ( real_type )0.0;

    real_type dn_minus_1    =  NS(CERRF_DAWSON_FZ_XI)[ n_interval ];
    real_type dn_plus_1     =  ( real_type )0.0;
    real_type sum_daw_z_re  =  dn_minus_1;
    real_type inv_factorial, sum_re, sum_im;

    real_type dn = ( real_type )1. + ( real_type )2. * minus_xn * dn_minus_1;
    real_type nn = ( real_type )1.0;
    int n = 1;

    sum_daw_z_im = ( real_type )0.0;

    #endif /* ( SIXTRL_CERRF_USE_DAWSON_COEFF > 0 ) */

    NS(sincos)( temp, &exp_sin_two_x_y, &exp_cos_two_x_y );

    temp = ( y - x ) * ( y + x );
    temp = NS(exp)( temp );
    exp_sin_two_x_y *= temp;
    exp_cos_two_x_y *= temp;

    SIXTRL_ASSERT( out_re != NULL );
    SIXTRL_ASSERT( out_im != NULL );

    SIXTRL_ASSERT( x >= ( real_type )SIXTRL_CERRF_DAWSON_X_MIN );
    SIXTRL_ASSERT( x <= ( real_type )SIXTRL_CERRF_DAWSON_X_MAX );
    SIXTRL_ASSERT( y >= ( real_type )SIXTRL_CERRF_DAWSON_Y_MIN );
    SIXTRL_ASSERT( y <= ( real_type )SIXTRL_CERRF_DAWSON_Y_MAX );
    SIXTRL_ASSERT( N_Taylor <= ( int )SIXTRL_CERRF_DAWSON_N_TAYLOR_COEFF_PER_XN );

    #if ( SIXTRL_CERRF_USE_DAWSON_COEFF > 0 )

    SIXTRL_ASSERT( jj_end >= jj + ( int )2 );
    SIXTRL_ASSERT( jj_end <= jj + ( int )SIXTRL_CERRF_DAWSON_N_TAYLOR_COEFF_PER_XN );

    ++jj;
    dn = Fz_kk_xi[ jj++ ];
    sum_daw_z_re += dn * dz_pow_n_re;
    sum_daw_z_im  = dn * dz_pow_n_im;

    for( ; jj < jj_end ; ++jj )
    {
        temp           = dz_pow_n_re * dz_re - dz_pow_n_im * y;
        dz_pow_n_im   *= dz_re;
        dz_pow_n_im   += dz_pow_n_re * y;
        dz_pow_n_re    = temp;

        dn             = Fz_kk_xi[ jj ];
        sum_daw_z_re  += dz_pow_n_re * dn;
        sum_daw_z_im  += dz_pow_n_im * dn;
    }

    #else /* ( SIXTRL_CERRF_USE_DAWSON_COEFF == 0 ) */
    sum_daw_z_im = ( real_type )0.0;

    for( ; n < N_Taylor ; ++n, nn += ( real_type )1.0 )
    {
        temp           = ( minus_xn * dn ) - ( nn * dn_minus_1 );
        dn_plus_1      = ( real_type )2. * temp;

        temp           = dz_pow_n_re * dz_re - dz_pow_n_im * y;
        dz_pow_n_im   *= dz_re;
        dz_pow_n_im   += dz_pow_n_re * y;
        dz_pow_n_re    = temp;

        sum_re         = dz_pow_n_re * dn;
        sum_im         = dz_pow_n_im * dn;
        dn_minus_1     = dn;
        dn             = dn_plus_1;
        inv_factorial  = NS(Math_inv_factorial)( n );

        sum_daw_z_re  += sum_re * inv_factorial;
        sum_daw_z_im  += sum_im * inv_factorial;
    }

    #endif /* ( SIXTRL_CERRF_USE_DAWSON_COEFF ) */

    *out_re = exp_cos_two_x_y - NS(MathConst_two_over_sqrt_pi)() * sum_daw_z_im;
    *out_im = exp_sin_two_x_y + NS(MathConst_two_over_sqrt_pi)() * sum_daw_z_re;
}

#endif /* ( SIXTRL_CERRF_USE_DAWSON_COEFF >= 0 ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined( _GPUCODE ) && defined( __cplusplus )  */
#endif /* SIXTRACKLIB_COMMON_BE_BEAMFIELDS_DAWSON_APPROX_H__ */
