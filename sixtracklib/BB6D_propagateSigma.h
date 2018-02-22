#ifndef _BB6D_PROPSIGMA_
#define _BB6D_PROPSIGMA_

#ifndef _GPUCODE
  #include <math.h>
  #define CLGLOBAL
#endif

//#include <stdio.h>

#define mysign(a) (((a) >= 0) - ((a) < 0))


typedef struct{
    double Sig_11_0;
    double Sig_12_0;
    double Sig_13_0;
    double Sig_14_0;
    double Sig_22_0;
    double Sig_23_0;
    double Sig_24_0;
    double Sig_33_0;
    double Sig_34_0;
    double Sig_44_0;
}BB6D_Sigmas;

void BB6D_propagate_Sigma_matrix(CLGLOBAL BB6D_Sigmas* data,
        double S, double threshold_singular, long int handle_singularities,
        double* Sig_11_hat_ptr, double* Sig_33_hat_ptr, 
        double* costheta_ptr, double* sintheta_ptr,
        double* dS_Sig_11_hat_ptr, double* dS_Sig_33_hat_ptr, 
        double* dS_costheta_ptr, double* dS_sintheta_ptr){
            
    double Sig_11_0 = data->Sig_11_0;
    double Sig_12_0 = data->Sig_12_0;
    double Sig_13_0 = data->Sig_13_0;
    double Sig_14_0 = data->Sig_14_0;
    double Sig_22_0 = data->Sig_22_0;
    double Sig_23_0 = data->Sig_23_0;
    double Sig_24_0 = data->Sig_24_0;
    double Sig_33_0 = data->Sig_33_0;
    double Sig_34_0 = data->Sig_34_0;
    double Sig_44_0 = data->Sig_44_0;
    
    // Propagate sigma matrix
    double Sig_11 = Sig_11_0 + 2.*Sig_12_0*S+Sig_22_0*S*S;
    double Sig_33 = Sig_33_0 + 2.*Sig_34_0*S+Sig_44_0*S*S;
    double Sig_13 = Sig_13_0 + (Sig_14_0+Sig_23_0)*S+Sig_24_0*S*S;
    
    double Sig_12 = Sig_12_0 + Sig_22_0*S;
    double Sig_14 = Sig_14_0 + Sig_24_0*S;
    double Sig_22 = Sig_22_0 + 0.*S;
    double Sig_23 = Sig_23_0 + Sig_24_0*S;
    double Sig_24 = Sig_24_0 + 0.*S;
    double Sig_34 = Sig_34_0 + Sig_44_0*S;
    double Sig_44 = Sig_44_0 + 0.*S;
    
    double R = Sig_11-Sig_33;
    double W = Sig_11+Sig_33;
    double T = R*R+4*Sig_13*Sig_13;
    
    //evaluate derivatives
    double dS_R = 2.*(Sig_12_0-Sig_34_0)+2*S*(Sig_22_0-Sig_44_0);
    double dS_W = 2.*(Sig_12_0+Sig_34_0)+2*S*(Sig_22_0+Sig_44_0);
    double dS_Sig_13 = Sig_14_0 + Sig_23_0 + 2*Sig_24_0*S;
    double dS_T = 2*R*dS_R+8.*Sig_13*dS_Sig_13;
    
    double Sig_11_hat, Sig_33_hat, costheta, sintheta, dS_Sig_11_hat, 
           dS_Sig_33_hat, dS_costheta, dS_sintheta, cos2theta, dS_cos2theta;
           
    double signR = mysign(R);
    
    //~ printf("handle: %ld\n",handle_singularities);
    
    if (T<threshold_singular && handle_singularities){
        
        double a = Sig_12-Sig_34;
        double b = Sig_22-Sig_44;
        double c = Sig_14+Sig_23;
        double d = Sig_24;
        
        double sqrt_a2_c2 = sqrt(a*a+c*c);
        
        if (sqrt_a2_c2*sqrt_a2_c2*sqrt_a2_c2 < threshold_singular){
        //equivalent to: if np.abs(c)<threshold_singular and np.abs(a)<threshold_singular:
            
            if (fabs(d)> threshold_singular){
                cos2theta = fabs(b)/sqrt(b*b+4*d*d);
                }
            else{
                cos2theta = 1.;
                } // Decoupled beam
            
            costheta = sqrt(0.5*(1.+cos2theta));
            sintheta = mysign(b)*mysign(d)*sqrt(0.5*(1.-cos2theta));
            
            dS_costheta = 0.;
            dS_sintheta = 0.;
            
            Sig_11_hat = 0.5*W;
            Sig_33_hat = 0.5*W;
            
            dS_Sig_11_hat = 0.5*dS_W;
            dS_Sig_33_hat = 0.5*dS_W;
        }
        else{
            //~ printf("I am here\n");
            //~ printf("a=%.2e c=%.2e\n", a, c);
            sqrt_a2_c2 = sqrt(a*a+c*c); //repeated?
            cos2theta = fabs(2.*a)/(2*sqrt_a2_c2);
            costheta = sqrt(0.5*(1.+cos2theta));
            sintheta = mysign(a)*mysign(c)*sqrt(0.5*(1.-cos2theta));
            
            dS_cos2theta = mysign(a)*(0.5*b/sqrt_a2_c2-a*(a*b+2.*c*d)/(2.*sqrt_a2_c2*sqrt_a2_c2*sqrt_a2_c2));
            
            dS_costheta = 1./(4.*costheta)*dS_cos2theta;
            if (fabs(sintheta)>threshold_singular){ 
            //equivalent to: if np.abs(c)>threshold_singular:
                dS_sintheta = -1./(4.*sintheta)*dS_cos2theta;
            }
            else{
                dS_sintheta = d/(2.*a);
            }
                
            Sig_11_hat = 0.5*W;
            Sig_33_hat = 0.5*W;
            
            dS_Sig_11_hat = 0.5*dS_W + mysign(a)*sqrt_a2_c2;
            dS_Sig_33_hat = 0.5*dS_W - mysign(a)*sqrt_a2_c2;
        }
    }
    else{
        
        double sqrtT = sqrt(T);
        cos2theta = signR*R/sqrtT;
        costheta = sqrt(0.5*(1.+cos2theta));
        sintheta = signR*mysign(Sig_13)*sqrt(0.5*(1.-cos2theta));

        //in sixtrack this line seems to be different different
        // sintheta = -mysign((Sig_11-Sig_33))*np.sqrt(0.5*(1.-cos2theta))

        Sig_11_hat = 0.5*(W+signR*sqrtT);
        Sig_33_hat = 0.5*(W-signR*sqrtT);

        dS_cos2theta = signR*(dS_R/sqrtT - R/(2*sqrtT*sqrtT*sqrtT)*dS_T);
        dS_costheta = 1./(4.*costheta)*dS_cos2theta;
        
        if (fabs(sintheta)<threshold_singular && handle_singularities){
        //equivalent to to np.abs(Sig_13)<threshold_singular
            dS_sintheta = (Sig_14+Sig_23)/R;
        }
        else{
            dS_sintheta = -1./(4.*sintheta)*dS_cos2theta;
        }
            
        dS_Sig_11_hat = 0.5*(dS_W + signR*0.5/sqrtT*dS_T);
        dS_Sig_33_hat = 0.5*(dS_W - signR*0.5/sqrtT*dS_T);
    }
    
    *Sig_11_hat_ptr = Sig_11_hat;
    *Sig_33_hat_ptr = Sig_33_hat;
    *costheta_ptr = costheta;
    *sintheta_ptr = sintheta;
    *dS_Sig_11_hat_ptr = dS_Sig_11_hat;
    *dS_Sig_33_hat_ptr = dS_Sig_33_hat;
    *dS_costheta_ptr = dS_costheta;
    *dS_sintheta_ptr = dS_sintheta;
    
}


#endif
