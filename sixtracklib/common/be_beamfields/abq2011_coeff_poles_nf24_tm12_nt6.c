#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/be_beamfields/definitions.h"
    #include "sixtracklib/common/be_beamfields/abq2011_coeff.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( _GPUCODE ) && \
     defined( SIXTRL_CERRF_ABQ2011_N_FOURIER ) && \
            ( SIXTRL_CERRF_ABQ2011_N_FOURIER == 24 ) && \
           ( SIXTRL_CERRF_ABQ2011_N_FOURIER == 24 ) && \
     defined( SIXTRL_CERRF_ABQ2011_TM ) && ( SIXTRL_CERRF_ABQ2011_TM == 12 ) && \
     defined( SIXTRL_CERRF_ABQ2011_NUM_TAYLOR_COEFF ) && \
            ( SIXTRL_CERRF_ABQ2011_NUM_TAYLOR_COEFF == 288 )

typedef SIXTRL_REAL_T real_type;

real_type const NS(CERRF_ABQ2011_ROOT_TAYLOR_COEFF)[ SIXTRL_CERRF_ABQ2011_NUM_TAYLOR_COEFF ] = {
    /* =======  pole nn =  0 ====== */
    ( real_type )+1.0,                                            /* real part, component ii =  0 */
    ( real_type )+0.0,                                            /* imag part, component ii =  0 */
    ( real_type )+0.0,                                            /* real part, component ii =  1 */
    ( real_type )+1.1283791670955125738961589031215,              /* imag part, component ii =  1 */
    ( real_type )-1.0,                                            /* real part, component ii =  2 */
    ( real_type )+0.0,                                            /* imag part, component ii =  2 */
    ( real_type )+0.0,                                            /* real part, component ii =  3 */
    ( real_type )-0.75225277806367504926410593541436,             /* imag part, component ii =  3 */
    ( real_type )+0.5,                                            /* real part, component ii =  4 */
    ( real_type )+0.0,                                            /* imag part, component ii =  4 */
    ( real_type )+0.0,                                            /* real part, component ii =  5 */
    ( real_type )+0.30090111122547001970564237416575,             /* imag part, component ii =  5 */
    /* =======  pole nn =  1 ====== */
    ( real_type )+0.93375711808097597031678906145732,             /* real part, component ii =  0 */
    ( real_type )+0.28227388511512776948036687916347,             /* imag part, component ii =  0 */
    ( real_type )-0.48891408373339520033788238567145,             /* real part, component ii =  1 */
    ( real_type )+0.98058090646585679246427864137302,             /* imag part, component ii =  1 */
    ( real_type )-0.80575971027319102090828326394266,             /* real part, component ii =  2 */
    ( real_type )-0.53898936611542409305995484807186,             /* imag part, component ii =  2 */
    ( real_type )+0.46657432173075775373057596178821,             /* real part, component ii =  3 */
    ( real_type )-0.55964921359105809694982544087676,             /* imag part, component ii =  3 */
    ( real_type )+0.34180541924063762781081431174069,             /* real part, component ii =  4 */
    ( real_type )+0.3427525938079192626116722191677,              /* imag part, component ii =  4 */
    ( real_type )-0.22242350849375531934902438134807,             /* real part, component ii =  5 */
    ( real_type )+0.18796671774622971841554848610581,             /* imag part, component ii =  5 */
    /* =======  pole nn =  2 ====== */
    ( real_type )+0.76021371764309095658088511096778,             /* real part, component ii =  0 */
    ( real_type )+0.49380190394896794454680308387448,             /* imag part, component ii =  0 */
    ( real_type )-0.79609394350190664528589828493657,             /* real part, component ii =  1 */
    ( real_type )+0.61127102250393577202621591738052,             /* imag part, component ii =  1 */
    ( real_type )-0.34337990356427131768412822573628,             /* real part, component ii =  2 */
    ( real_type )-0.81386266289074891136720483222675,             /* imag part, component ii =  2 */
    ( real_type )+0.65059149371548070022184863113389,             /* real part, component ii =  3 */
    ( real_type )-0.12342235247277904610313013351651,             /* imag part, component ii =  3 */
    ( real_type )+0.0013654970200886334941510301728008,           /* real part, component ii =  4 */
    ( real_type )+0.43924322776347884757473075802491,             /* imag part, component ii =  4 */
    ( real_type )-0.26052258651331289377564770306822,             /* real part, component ii =  5 */
    ( real_type )-0.042625945509609278561220458493055,            /* imag part, component ii =  5 */
    /* =======  pole nn =  3 ====== */
    ( real_type )+0.53964148581629717588566532326912,             /* real part, component ii =  0 */
    ( real_type )+0.59780598866963161507104375663249,             /* imag part, component ii =  0 */
    ( real_type )-0.84766686370637990747772074744551,             /* real part, component ii =  1 */
    ( real_type )+0.18934771595726364581938916985108,             /* imag part, component ii =  1 */
    ( real_type )+0.12611451211156873692475598022312,             /* real part, component ii =  2 */
    ( real_type )-0.74651933702596819864107536767402,             /* imag part, component ii =  2 */
    ( real_type )+0.49907783834412571386136023355349,             /* real part, component ii =  3 */
    ( real_type )+0.26464480018907500619073693090441,             /* imag part, component ii =  3 */
    ( real_type )-0.2590446648697068388245213767399,              /* real part, component ii =  4 */
    ( real_type )+0.26933389850239200381296738575726,             /* imag part, component ii =  4 */
    ( real_type )-0.11824985372702018565304948760496,             /* real part, component ii =  5 */
    ( real_type )-0.19047165976541137590932087255505,             /* imag part, component ii =  5 */
    /* =======  pole nn =  4 ====== */
    ( real_type )+0.33399718598613178541580257101556,             /* real part, component ii =  0 */
    ( real_type )+0.60198345081269674225182633547039,             /* imag part, component ii =  0 */
    ( real_type )-0.69952207054246363927481892276438,             /* real part, component ii =  1 */
    ( real_type )-0.13241202400835458179124782676693,             /* imag part, component ii =  1 */
    ( real_type )+0.39854061329390984159311298877162,             /* real part, component ii =  2 */
    ( real_type )-0.46332190352216271517664217317236,             /* imag part, component ii =  2 */
    ( real_type )+0.18811421083246068241280801744611,             /* real part, component ii =  3 */
    ( real_type )+0.41173439119500646221605042095309,             /* imag part, component ii =  3 */
    ( real_type )-0.29776667711147158469441740055891,             /* real part, component ii =  4 */
    ( real_type )+0.016077328659664965595852112639495,            /* imag part, component ii =  4 */
    ( real_type )+0.049482529706648149074087999890876,            /* real part, component ii =  5 */
    ( real_type )-0.17142821215887619729887867959749,             /* imag part, component ii =  5 */
    /* =======  pole nn =  5 ====== */
    ( real_type )+0.18023873770401830576172373319967,             /* real part, component ii =  0 */
    ( real_type )+0.54283491474428325274547121825489,             /* imag part, component ii =  0 */
    ( real_type )-0.47186391188603465563784229200975,             /* real part, component ii =  1 */
    ( real_type )-0.2927593164650557618486229314101,              /* imag part, component ii =  1 */
    ( real_type )+0.43742967857736002430500383201776,             /* real part, component ii =  2 */
    ( real_type )-0.159613865629038011909226796683,               /* imag part, component ii =  2 */
    ( real_type )-0.067153465598415454913748421569707,            /* real part, component ii =  3 */
    ( real_type )+0.33446225199649668022814999373229,             /* imag part, component ii =  3 */
    ( real_type )-0.17476299883303899099667609200056,             /* real part, component ii =  4 */
    ( real_type )-0.13909809922200018718958845032981,             /* imag part, component ii =  4 */
    ( real_type )+0.11836707844823233243597803379156,             /* real part, component ii =  5 */
    ( real_type )-0.060953306357908685017330434139453,            /* imag part, component ii =  5 */
    /* =======  pole nn =  6 ====== */
    ( real_type )+0.084804972471113777302191522641103,            /* real part, component ii =  0 */
    ( real_type )+0.46054632922146286382179698685087,             /* imag part, component ii =  0 */
    ( real_type )-0.26642267850313569685111656389261,             /* real part, component ii =  1 */
    ( real_type )-0.31846979742438147997234732548487,             /* imag part, component ii =  1 */
    ( real_type )+0.33369079229646944121968803592685,             /* real part, component ii =  2 */
    ( real_type )+0.039704858767870393041133492489307,            /* imag part, component ii =  2 */
    ( real_type )-0.17182506154762485790208472321895,             /* real part, component ii =  3 */
    ( real_type )+0.17073436741060034782501439483468,             /* imag part, component ii =  3 */
    ( real_type )-0.031894308383076639938700257011888,            /* real part, component ii =  4 */
    ( real_type )-0.15394688797704586215013637936476,             /* imag part, component ii =  4 */
    ( real_type )+0.088769809600570128960259510168993,            /* real part, component ii =  5 */
    ( real_type )+0.028433935498099490227223620100417,            /* imag part, component ii =  5 */
    /* =======  pole nn =  7 ====== */
    ( real_type )+0.034790634459528376653392819006388,            /* real part, component ii =  0 */
    ( real_type )+0.38227751724449322418096873789343,             /* imag part, component ii =  0 */
    ( real_type )-0.12751433523707929738959936672643,             /* real part, component ii =  1 */
    ( real_type )-0.27274111268030707423319341816842,             /* imag part, component ii =  1 */
    ( real_type )+0.19889158984525170552262295294651,             /* real part, component ii =  2 */
    ( real_type )+0.11754667704704935449668489824441,             /* imag part, component ii =  2 */
    ( real_type )-0.15798235998808377725048076453576,             /* real part, component ii =  3 */
    ( real_type )+0.038217050706076073978298029293292,            /* imag part, component ii =  3 */
    ( real_type )+0.045313103025182256785838266674491,            /* real part, component ii =  4 */
    ( real_type )-0.093791540197713864831181307830144,            /* imag part, component ii =  4 */
    ( real_type )+0.029976704627670507707641046693031,            /* real part, component ii =  5 */
    ( real_type )+0.053465969570171824731493520484189,            /* imag part, component ii =  5 */
    /* =======  pole nn =  8 ====== */
    ( real_type )+0.012444321744005097562234654220266,            /* real part, component ii =  0 */
    ( real_type )+0.31923996806575228561577188610564,             /* imag part, component ii =  0 */
    ( real_type )-0.052126653026498850823909942205847,            /* real part, component ii =  1 */
    ( real_type )-0.20885008411463086079807767642065,             /* imag part, component ii =  1 */
    ( real_type )+0.096729485058843536811135865322049,            /* real part, component ii =  2 */
    ( real_type )+0.11817462523833750741987488245719,             /* imag part, component ii =  2 */
    ( real_type )-0.10030873782517255463742834493933,             /* real part, component ii =  3 */
    ( real_type )-0.025769514807796351542632760423039,            /* imag part, component ii =  3 */
    ( real_type )+0.056678322084720468739163647346852,            /* real part, component ii =  4 */
    ( real_type )-0.032101539816919950137056924642196,            /* imag part, component ii =  4 */
    ( real_type )-0.0073592249443720339537349911282626,           /* real part, component ii =  5 */
    ( real_type )+0.037201129031853461035176594053539,            /* imag part, component ii =  5 */
    /* =======  pole nn =  9 ====== */
    ( real_type )+0.003881038619955637411407040680584,            /* real part, component ii =  0 */
    ( real_type )+0.27209031085455034661059909971213,             /* imag part, component ii =  0 */
    ( real_type )-0.018288963625126350020805781556396,            /* real part, component ii =  1 */
    ( real_type )-0.15381621544491524528935233915374,             /* imag part, component ii =  1 */
    ( real_type )+0.039211316704895283468586906787111,            /* real part, component ii =  2 */
    ( real_type )+0.090330608478997621872435696751784,            /* imag part, component ii =  2 */
    ( real_type )-0.049400349832089980557644238245125,            /* real part, component ii =  3 */
    ( real_type )-0.039346844366013911033454378322106,            /* imag part, component ii =  3 */

    ( real_type )+0.03859275769152473033950548389918,             /* real part, component ii =  4 */
    ( real_type )+0.0011891047113300322739369782820103,           /* imag part, component ii =  4 */
    ( real_type )-0.016612677280803531999687829818964,            /* real part, component ii =  5 */
    ( real_type )+0.014618032958766532062795805087643,            /* imag part, component ii =  5 */
    /* =======  pole nn = 10 ====== */
    ( real_type )+0.0010553403629220348894046152644339,           /* real part, component ii =  0 */
    ( real_type )+0.2373265348988182880038096308366,              /* imag part, component ii =  0 */
    ( real_type )-0.0055257492186544183844855586567031,           /* real part, component ii =  1 */
    ( real_type )-0.11425966380456945538033770312825,             /* imag part, component ii =  1 */
    ( real_type )+0.013411037262831515835004193899036,            /* real part, component ii =  2 */
    ( real_type )+0.061804565442910883656215489601897,            /* imag part, component ii =  2 */
    ( real_type )-0.019722842821969531756171249897052,            /* real part, component ii =  3 */
    ( real_type )-0.031696206771263939653361057577715,            /* imag part, component ii =  3 */
    ( real_type )+0.019111622250836603455288276587677,            /* real part, component ii =  4 */
    ( real_type )+0.010587954919905330162492167657816,            /* imag part, component ii =  4 */
    ( real_type )-0.01212450689168268799693273266244,             /* real part, component ii =  5 */
    ( real_type )+0.0015908022442007448897003787977559,           /* imag part, component ii =  5 */
    /* =======  pole nn = 11 ====== */
    ( real_type )+0.00025021018490812133699464974855978,          /* real part, component ii =  0 */
    ( real_type )+0.2111310662193366471757139672037,              /* imag part, component ii =  0 */
    ( real_type )-0.0014411072110612792039117484966232,           /* real part, component ii =  1 */
    ( real_type )-0.087648478299775742512331974364018,            /* imag part, component ii =  1 */
    ( real_type )+0.0038998806567884865010486766884342,           /* real part, component ii =  2 */
    ( real_type )+0.041278431345154913237469604499635,            /* imag part, component ii =  2 */
    ( real_type )-0.0065264952278302648139234075330014,           /* real part, component ii =  3 */
    ( real_type )-0.020816580206935201937858758598365,            /* imag part, component ii =  3 */
    ( real_type )+0.0074475381747659418367127897836248,           /* real part, component ii =  4 */
    ( real_type )+0.0093345080757839438562300002140794,           /* imag part, component ii =  4 */
    ( real_type )-0.0059683500218317749345182218354011,           /* real part, component ii =  5 */
    ( real_type )-0.002425949315670312051431784454621,            /* imag part, component ii =  5 */
    /* =======  pole nn = 12 ====== */
    ( real_type )+0.000051723186203812306145465090382394,         /* real part, component ii =  0 */
    ( real_type )+0.19068111719759752027915870526796,             /* imag part, component ii =  0 */
    ( real_type )-0.00032498636359630737414466531011684,          /* real part, component ii =  1 */
    ( real_type )-0.06970562683702093127517986084811,             /* imag part, component ii =  1 */
    ( real_type )+0.00096925158618720835823880184883858,          /* real part, component ii =  2 */
    ( real_type )+0.028305567987458973213439387190091,            /* imag part, component ii =  2 */
    ( real_type )-0.0018133381993664538123873027483908,           /* real part, component ii =  3 */
    ( real_type )-0.012812625072044405147288715834784,            /* imag part, component ii =  3 */
    ( real_type )+0.0023637591897080934012266740590815,           /* real part, component ii =  4 */
    ( real_type )+0.0059732404060380626607434772749493,           /* imag part, component ii =  4 */
    ( real_type )-0.0022450521223503419275951871575783,           /* real part, component ii =  5 */
    ( real_type )-0.0023811452422761944596998911073548,           /* imag part, component ii =  5 */
    /* =======  pole nn = 13 ====== */
    ( real_type )+0.0000093225217914050245645220689669343,        /* real part, component ii =  0 */
    ( real_type )+0.17416766378502582902663119173603,             /* imag part, component ii =  0 */
    ( real_type )-0.000063456392941085698747219843597152,         /* real part, component ii =  1 */
    ( real_type )-0.05714251449101158318368194552633,             /* imag part, component ii =  1 */
    ( real_type )+0.00020664446091953552366712256389438,          /* real part, component ii =  2 */
    ( real_type )+0.020310715258635321661950090870278,            /* imag part, component ii =  2 */
    ( real_type )-0.00042655714716637992888163778879052,          /* real part, component ii =  3 */
    ( real_type )-0.0079885414500965540039700913561707,           /* imag part, component ii =  3 */
    ( real_type )+0.00062254836947204695346528029270674,          /* real part, component ii =  4 */
    ( real_type )+0.0034387115674644867988234477242484,           /* imag part, component ii =  4 */
    ( real_type )-0.00067688760754977906906419439941818,          /* real part, component ii =  5 */
    ( real_type )-0.0014858968524976706371609581554089,           /* imag part, component ii =  5 */
    /* =======  pole nn = 14 ====== */
    ( real_type )+0.0000014650397062886179500833912613532,        /* real part, component ii =  0 */
    ( real_type )+0.16047101168347768453054076590807,             /* imag part, component ii =  0 */
    ( real_type )-0.000010739301949818564645493195385236,         /* real part, component ii =  1 */
    ( real_type )-0.04793478621533662946897952188554,             /* imag part, component ii =  1 */
    ( real_type )+0.000037896557755649351273865026653075,         /* real part, component ii =  2 */
    ( real_type )+0.015219155912937633304360996579126,            /* imag part, component ii =  2 */
    ( real_type )-0.000085439224487945971694156887315906,         /* real part, component ii =  3 */
    ( real_type )-0.0052308890641597716674843143357707,           /* imag part, component ii =  3 */
    ( real_type )+0.0001376272777770237910684459659253,           /* real part, component ii =  4 */
    ( real_type )+0.001976526926027240933603775896867,            /* imag part, component ii =  4 */
    ( real_type )-0.00016759643777715616229661848766471,          /* real part, component ii =  5 */
    ( real_type )-0.00080538419386990317836393575029478,          /* imag part, component ii =  5 */
    /* =======  pole nn = 15 ====== */
    ( real_type )+0.00000020073968320415217695153659897956,       /* real part, component ii =  0 */
    ( real_type )+0.1488793485856626698339770988152,              /* imag part, component ii =  0 */
    ( real_type )-0.0000015766057850952672151246133252495,        /* real part, component ii =  1 */
    ( real_type )-0.040916502374366970677175935266772,            /* imag part, component ii =  1 */
    ( real_type )+0.0000059905767568739225978953780583554,        /* real part, component ii =  2 */
    ( real_type )+0.011799380501713088975053758620893,            /* imag part, component ii =  2 */
    ( real_type )-0.000014632222751737225312029010479964,         /* real part, component ii =  3 */
    ( real_type )-0.0036130376679990937784848106329236,           /* imag part, component ii =  3 */
    ( real_type )+0.000025735013810654973744508461675444,         /* real part, component ii =  4 */
    ( real_type )+0.0011944926209741751449997413319328,           /* imag part, component ii =  4 */
    ( real_type )-0.000034571576063097877760649761817811,         /* real part, component ii =  5 */
    ( real_type )-0.00043108955421020549336131906478999,          /* imag part, component ii =  5 */
    /* =======  pole nn = 16 ====== */
    ( real_type )+0.000000023981973818259450774325197133491,      /* real part, component ii =  0 */
    ( real_type )+0.13891644940561371062569280602016,             /* imag part, component ii =  0 */
    ( real_type )-0.00000020091091404273774347582071159638,       /* real part, component ii =  1 */
    ( real_type )-0.035404558012365380312943535121619,            /* imag part, component ii =  1 */
    ( real_type )+0.00000081759169495864097795259893595499,       /* real part, component ii =  2 */
    ( real_type )+0.0093858164013739305331682032307173,           /* imag part, component ii =  2 */
    ( real_type )-0.0000021492061128764803387587313707712,        /* real part, component ii =  3 */
    ( real_type )-0.0026071051957554622977849535416305,           /* imag part, component ii =  3 */
    ( real_type )+0.0000040924909093626972191440408216767,        /* real part, component ii =  4 */
    ( real_type )+0.00076740015272712817058730248694807,          /* imag part, component ii =  4 */
    ( real_type )-0.0000059973518885757342446236557950261,        /* real part, component ii =  5 */
    ( real_type )-0.00024294921885580505196871738081088,          /* imag part, component ii =  5 */
    /* =======  pole nn = 17 ====== */
    ( real_type )+0.0000000024980692045821258097510404711114,     /* real part, component ii =  0 */
    ( real_type )+0.13024740171229320629969008874562,             /* imag part, component ii =  0 */
    ( real_type )-0.000000022235761606943296664477868977993,      /* real part, component ii =  1 */
    ( real_type )-0.030976293948567907829288627886776,            /* imag part, component ii =  1 */
    ( real_type )+0.000000096464179986492842475383385936404,      /* real part, component ii =  2 */
    ( real_type )+0.0076153697520735797950354674456489,           /* imag part, component ii =  2 */
    ( real_type )-0.00000027139114259882675014310758466137,       /* real part, component ii =  3 */
    ( real_type )-0.0019443942758009957577073747085063,           /* imag part, component ii =  3 */
    ( real_type )+0.00000055569320739187190352894054194245,       /* real part, component ii =  4 */
    ( real_type )+0.00051916558784461541529890726719493,          /* imag part, component ii =  4 */
    ( real_type )-0.00000088070850515596665751578453637642,       /* real part, component ii =  5 */
    ( real_type )-0.0001464794745155215039282529674027,           /* imag part, component ii =  5 */
    /* =======  pole nn = 18 ====== */
    ( real_type )+2.2687772443535217693511914029833e-10,          /* real part, component ii =  0 */
    ( real_type )+0.12262715881089526716168626576724,             /* imag part, component ii =  0 */
    ( real_type )-0.000000002138272177047815761070063726908,      /* real part, component ii =  1 */
    ( real_type )-0.027354576657179796462370682551688,            /* imag part, component ii =  1 */
    ( real_type )+0.0000000098494925197479553681012513131119,     /* real part, component ii =  2 */
    ( real_type )+0.0062782467914870717677614781281006,           /* imag part, component ii =  2 */
    ( real_type )-0.000000029517578556929254245921153950795,      /* real part, component ii =  3 */
    ( real_type )-0.0014873095594396108088549204722592,           /* imag part, component ii =  3 */
    ( real_type )+0.000000064624409699782439037265677151585,      /* real part, component ii =  4 */
    ( real_type )+0.00036526719341847904307151664809437,          /* imag part, component ii =  4 */
    ( real_type )-0.00000011000711103047638993671089404554,       /* real part, component ii =  5 */
    ( real_type )-0.000093588615088669178693067460636049,         /* imag part, component ii =  5 */
    /* =======  pole nn = 19 ====== */
    ( real_type )+1.7965822334136598818238788950604e-11,          /* real part, component ii =  0 */
    ( real_type )+0.11587097251890988822145747203609,             /* imag part, component ii =  0 */
    ( real_type )-1.7873076895863940652596084829635e-10,          /* real part, component ii =  1 */
    ( real_type )-0.024348920331909153786372952219037,            /* imag part, component ii =  1 */
    ( real_type )+8.7107468965648074865261058066019e-10,          /* real part, component ii =  2 */
    ( real_type )+0.005245143773907612104584027369068,            /* imag part, component ii =  2 */
    ( real_type )-0.0000000027694392134333165369386960294087,     /* real part, component ii =  3 */
    ( real_type )-0.0011609418784759838536321207929917,           /* imag part, component ii =  3 */
    ( real_type )+0.0000000064523188160978617317336680723085,     /* real part, component ii =  4 */
    ( real_type )+0.00026479990707256154154634089159106,          /* imag part, component ii =  4 */
    ( real_type )-0.000000011730243995765755270415153450135,      /* real part, component ii =  5 */
    ( real_type )-0.000062489095672205330528357220419356,         /* imag part, component ii =  5 */
    /* =======  pole nn = 20 ====== */
    ( real_type )+1.2404240973367851584778897534523e-12,          /* real part, component ii =  0 */
    ( real_type )+0.1098362769681518483909285764697,              /* imag part, component ii =  0 */
    ( real_type )-1.298969077176331621982980936204e-11,           /* real part, component ii =  1 */
    ( real_type )-0.021823635640486277393965150233485,            /* imag part, component ii =  1 */
    ( real_type )+6.6773437737621157971588213132363e-11,          /* real part, component ii =  2 */
    ( real_type )+0.0044320120364682701884582044944389,           /* imag part, component ii =  2 */
    ( real_type )-2.2442347443154233770616810425146e-10,          /* real part, component ii =  3 */
    ( real_type )-0.00092155007788721109688240744356282,          /* imag part, component ii =  3 */
    ( real_type )+5.5415256327054792715433938450672e-10,          /* real part, component ii =  4 */
    ( real_type )+0.00019660644393716836366126600875097,          /* imag part, component ii =  4 */

    ( real_type )-0.0000000010708450247198540343349748623994,     /* real part, component ii =  5 */
    ( real_type )-0.000043151542126063345363431783857041,         /* imag part, component ii =  5 */
    /* =======  pole nn = 21 ====== */
    ( real_type )+7.4672577020182875440016859651146e-14,          /* real part, component ii =  0 */
    ( real_type )+0.10441096552127306445543857671136,             /* imag part, component ii =  0 */
    ( real_type )-8.2106786786928587328349576061717e-13,          /* real part, component ii =  1 */
    ( real_type )-0.019679360729957722054156214767237,            /* imag part, component ii =  1 */
    ( real_type )+4.4393837911241883207016186702722e-12,          /* real part, component ii =  2 */
    ( real_type )+0.0037819708977395739226231654500022,           /* imag part, component ii =  2 */
    ( real_type )-1.5723812843525390463303037280301e-11,          /* real part, component ii =  3 */
    ( real_type )-0.00074207349986206600447724806024643,          /* imag part, component ii =  3 */
    ( real_type )+4.1003396155623084161527569873175e-11,          /* real part, component ii =  4 */
    ( real_type )+0.00014889562477175441870019280993402,          /* imag part, component ii =  4 */
    ( real_type )-8.3881652556906059966631908674231e-11,          /* real part, component ii =  5 */
    ( real_type )-0.000030609180709397879714204992464072,         /* imag part, component ii =  5 */
    /* =======  pole nn = 22 ====== */
    ( real_type )+3.9194031326808708626455909986852e-15,          /* real part, component ii =  0 */
    ( real_type )+0.099505481546473999566411208266413,            /* imag part, component ii =  0 */
    ( real_type )-4.5148282989652500394710017721569e-14,          /* real part, component ii =  1 */
    ( real_type )-0.017841695571651428868463002112912,            /* imag part, component ii =  1 */
    ( real_type )+2.5611603949854222001811303619656e-13,          /* real part, component ii =  2 */
    ( real_type )+0.0032553079685830720387268042850504,           /* imag part, component ii =  2 */
    ( real_type )-9.5331613908539489537576380947164e-13,          /* real part, component ii =  3 */
    ( real_type )-0.00060502157356591611282409939338385,          /* imag part, component ii =  3 */
    ( real_type )+2.6172953777583858662015187181188e-12,          /* real part, component ii =  4 */
    ( real_type )+0.00011468306892164779307267070527788,          /* imag part, component ii =  4 */
    ( real_type )-5.6484892271287050682397438174311e-12,          /* real part, component ii =  5 */
    ( real_type )-0.00002220219423824602272742007102591,          /* imag part, component ii =  5 */
    /* =======  pole nn = 23 ====== */
    ( real_type )+1.7936866768385370756626538036594e-16,          /* real part, component ii =  0 */
    ( real_type )+0.095047308459488420379731640070237,            /* imag part, component ii =  0 */
    ( real_type )-2.160095939939171086934421782295e-15,           /* real part, component ii =  1 */
    ( real_type )-0.01625388257043277239408391728447,             /* imag part, component ii =  1 */
    ( real_type )+1.2827402609576721298174443554251e-14,          /* real part, component ii =  2 */
    ( real_type )+0.0028235911853785800300261290465192,           /* imag part, component ii =  2 */
    ( real_type )-5.0052430343726648115144665585684e-14,          /* real part, component ii =  3 */
    ( real_type )-0.00049869975686168450450500529427224,          /* imag part, component ii =  3 */
    ( real_type )+1.4427879834645452305577800957287e-13,          /* real part, component ii =  4 */
    ( real_type )+0.000089636254293407267781422284128181,         /* imag part, component ii =  4 */
    ( real_type )-3.2748235779389763984810293617617e-13,          /* real part, component ii =  5 */
    ( real_type )-0.00001641388904265690541641987229264,          /* imag part, component ii =  5 */
};

#else /* !defined( _GPUCODE ) &&
         ( ( N_FOURIER != 24 ) || || ( TM != 12 ) || ( N_TAYLOR != 6 ) ) */

    #error "precomputed fourier coefficients only provided for " \
           "SIXTRL_CERRF_ABQ2011_N_FOURIER == 24, "\
           "SIXTRL_CERRF_ABQ2011_TM == 12, and " \
           "SIXTRL_CERRF_ABQ2011_NUM_TAYLOR_COEFF == 6 " \
           "-> provide your own tabulated data for other configurations"

#endif /* !defined( _GPUCODE ) */
