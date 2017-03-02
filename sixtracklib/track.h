//SixTrackLib
//
//Authors: R. De Maria, G. Iadarola, D. Pellegrini, H. Jasim
//
//Copyright 2017 CERN. This software is distributed under the terms of the GNU
//Lesser General Public License version 2.1, copied verbatim in the file
//`COPYING''.
//
//In applying this licence, CERN does not waive the privileges and immunities
//granted to it by virtue of its status as an Intergovernmental Organization or
//submit itself to any jurisdiction.


typedef struct {
    double length;
} Drift ;

typedef struct {
      double length;
} DriftExact ;

typedef struct {
      long int order;
      double l ;
      double hxl;
      double hyl;
      double bal[1];
} Multipole;

typedef struct {
      double volt;
      double freq;
      double lag;
} Cavity;

typedef struct {
      double cz;
      double sz;
      double dx;
      double dy;
} Align;
