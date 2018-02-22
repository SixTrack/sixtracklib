#ifndef _BB6D_BOOST_
#define _BB6D_BOOST_

typedef struct{
    double sphi;
    double cphi;
    double tphi;
    double salpha;
    double calpha;
}BB6D_boost_data;


void BB6D_boost(CLGLOBAL BB6D_boost_data* data,
                double* x_star, double* px_star, 
                double* y_star, double* py_star,
                double* sigma_star, double*  delta_star){
    
    double sphi = data->sphi;
    double cphi = data->cphi;
    double tphi = data->tphi;
    double salpha = data->salpha;
    double calpha = data->calpha;
    

    double x = *x_star;
    double px = *px_star;
    double y = *y_star;
    double py = *py_star ;              
    double sigma = *sigma_star;
    double delta = *delta_star ; 
    
    double h = delta + 1. - sqrt((1.+delta)*(1.+delta)-px*px-py*py);

    
    double px_st = px/cphi-h*calpha*tphi/cphi;
    double py_st = py/cphi-h*salpha*tphi/cphi;
    double delta_st = delta -px*calpha*tphi-py*salpha*tphi+h*tphi*tphi;

    double pz_st = sqrt((1.+delta_st)*(1.+delta_st)-px_st*px_st-py_st*py_st);
    double hx_st = px_st/pz_st;
    double hy_st = py_st/pz_st;
    double hsigma_st = 1.-(delta_st+1)/pz_st;

    double L11 = 1.+hx_st*calpha*sphi;
    double L12 = hx_st*salpha*sphi;
    double L13 = calpha*tphi;

    double L21 = hy_st*calpha*sphi;
    double L22 = 1.+hy_st*salpha*sphi;
    double L23 = salpha*tphi;

    double L31 = hsigma_st*calpha*sphi;
    double L32 = hsigma_st*salpha*sphi;
    double L33 = 1./cphi;

    double x_st = L11*x + L12*y + L13*sigma;
    double y_st = L21*x + L22*y + L23*sigma;
    double sigma_st = L31*x + L32*y + L33*sigma;  
    
    *x_star = x_st;
    *px_star = px_st;
    *y_star = y_st;
    *py_star = py_st;
    *sigma_star = sigma_st;
    *delta_star = delta_st;
    
}


void BB6D_inv_boost(CLGLOBAL BB6D_boost_data* data,
                double* x, double* px, 
                double* y, double* py,
                double* sigma, double*  delta){
    
    double sphi = data->sphi;
    double cphi = data->cphi;
    double tphi = data->tphi;
    double salpha = data->salpha;
    double calpha = data->calpha;
    
    double x_st = *x;
    double px_st = *px;
    double y_st = *y;
    double py_st = *py ;              
    double sigma_st = *sigma;
    double delta_st = *delta ; 
    
    double pz_st = sqrt((1.+delta_st)*(1.+delta_st)-px_st*px_st-py_st*py_st);
    double hx_st = px_st/pz_st;
    double hy_st = py_st/pz_st;
    double hsigma_st = 1.-(delta_st+1)/pz_st;

    double Det_L = 1./cphi + (hx_st*calpha + hy_st*salpha-hsigma_st*sphi)*tphi;

    double Linv_11 = (1./cphi + salpha*tphi*(hy_st-hsigma_st*salpha*sphi))/Det_L;
    double Linv_12 = (salpha*tphi*(hsigma_st*calpha*sphi-hx_st))/Det_L;
    double Linv_13 = -tphi*(calpha - hx_st*salpha*salpha*sphi + hy_st*calpha*salpha*sphi)/Det_L;

    double Linv_21 = (calpha*tphi*(-hy_st + hsigma_st*salpha*sphi))/Det_L;
    double Linv_22 = (1./cphi + calpha*tphi*(hx_st-hsigma_st*calpha*sphi))/Det_L;
    double Linv_23 = -tphi*(salpha - hy_st*calpha*calpha*sphi + hx_st*calpha*salpha*sphi)/Det_L;

    double Linv_31 = -hsigma_st*calpha*sphi/Det_L;
    double Linv_32 = -hsigma_st*salpha*sphi/Det_L;
    double Linv_33 = (1. + hx_st*calpha*sphi + hy_st*salpha*sphi)/Det_L;

    double x_i = Linv_11*x_st + Linv_12*y_st + Linv_13*sigma_st;
    double y_i = Linv_21*x_st + Linv_22*y_st + Linv_23*sigma_st;
    double sigma_i = Linv_31*x_st + Linv_32*y_st + Linv_33*sigma_st;

    double h = (delta_st+1.-pz_st)*cphi*cphi;

    double px_i = px_st*cphi+h*calpha*tphi;
    double py_i = py_st*cphi+h*salpha*tphi;

    double delta_i = delta_st + px_i*calpha*tphi + py_i*salpha*tphi - h*tphi*tphi;

    
    *x = x_i;
    *px = px_i;
    *y = y_i;
    *py = py_i;
    *sigma = sigma_i;
    *delta = delta_i;
    
}

#endif
