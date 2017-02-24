#ifndef _GPUCODE

#include <math.h>

#endif


#define CLIGHT 299792458

#include "beam.h"

#include "track.h"

int Drift_track(CLGLOBAL Particle* p, CLGLOBAL Drift *el){
  double xp, yp;
  double length=el->length;
  xp = p->px * p->rpp;
  yp = p->py * p->rpp;
  p->x += xp * length;
  p->y += yp * length;
  p->sigma += length * (1 - p->rvv*( 1 + (xp*xp+yp*yp)/2 ) );
  p->s+=length;
//  _DP("Drift_track: length=%g\n",length);
  return 1;
};


int DriftExact_track(CLGLOBAL Particle* p, double length){
  double lpzi, lbzi, px, py, opd;
  opd=1+p->delta;
  px=p->px; py=p->py;
  lpzi= length/sqrt(opd*opd-px*px-py*py);
  lbzi=(p->beta0*p->beta0*p->psigma+1)*lpzi;
  p->x += px*lpzi ;
  p->y += py*lpzi ;
  p->sigma += length - lbzi;
  p->s += length ;
  return 1;
}

//int Multipole_track(CLGLOBAL Particle* p, long int order,
//                     double l, double hxl, double hyl, CLGLOBAL double *bal){
int Multipole_track(CLGLOBAL Particle* p, CLGLOBAL Multipole *el){
  double x,y,chi,dpx,dpy,zre,zim,b1l,a1l,hxx,hyy;
  long int order=el->order;
  double hxl=el->hxl;
  double hyl=el->hyl;
  double l=el->l;
  CLGLOBAL double *bal = el->bal;
  dpx=bal[order*2];
  dpy=bal[order*2+1];
  x=p->x; y=p->y; chi=p->chi;
//  _DP("Multipole_track: dpx,y=%g %G\n",dpx,dpy);
  for (int ii=order-1;ii>=0;ii--){
    zre=(dpx*x-dpy*y);
    zim=(dpx*y+dpy*x);
//    _DP("Multipole_track: y,x=%g %G\n",x,y);
    dpx=bal[ii*2]+zre;
    dpy=bal[ii*2+1]+zim;
//    _DP("Multipole_track: dpx,y=%g %G\n",dpx,dpy);
  }
  dpx=-chi*dpx ;
  dpy=chi*dpy ;
//  _DP("Multipole_track: dpx,y=%g %G\n",dpx,dpy);
  if (l>0){
     b1l=chi*bal[0]; a1l=chi*bal[1];
     hxx=hxl/l*x; hyy=hyl/l*y;
     dpx+=hxl + hxl*p->delta - b1l*hxx;
     dpy-=hyl + hyl*p->delta - a1l*hyy;
     p->sigma-=chi*(hxx-hyy)*l*p->rvv;
  }
  p->px+=dpx ;  p->py+=dpy ;
  return 1 ;
}

int Cavity_track(CLGLOBAL Particle* p, double volt, double freq, double lag ){
  double phase;
  phase=lag-freq/CLIGHT*p->sigma/p->beta0;
  p->psigma+=p->chi*volt*sin(phase)/(p->p0c*p->beta0);
  double pt=p->sigma*p->beta0;
  p->delta=sqrt( pt*pt+ 2*p->psigma + 1 ) - 1;
  p->beta=(1+p->delta)/(1+p->psigma)*p->beta0;
  p->gamma=sqrt(1-1/(p->beta*p->beta));
  p->rpp=1/(1+p->delta);
  p->rvv=p->beta/p->beta0;
  return 1;
}

int Align_track(CLGLOBAL Particle* p, double cz, double sz,
                                      double dx, double dy){
  double xn,yn;
  xn= cz*p->x-sz*p->y + dx;
  yn= sz*p->x+cz*p->y + dy;
  p->x=xn;
  p->y=yn;
  xn= cz*p->px-sz*p->py;
  yn= sz*p->px+cz*p->py;
  p->px=xn;
  p->py=yn;
  return 1;
};

