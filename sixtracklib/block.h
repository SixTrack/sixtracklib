
#ifndef _BLOC_
#define _BLOC_

//#define DEBUG

#ifdef DEBUG
#include <stdio.h>
#define _D
#define _DP(...) printf(__VA_ARGS__)
#else
// int printf(const char *format, ...);
#define _D for (; 0;)
// #define _DP(...)
#endif

#ifdef __OPENCL_VERSION__
#define CLGLOBAL __global
#define CLKERNEL __kernel
#if __OPENCL_VERSION__ <= CL_VERSION_1_1
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif
#define _GPUCODE
#else
#define CLGLOBAL
#define CLKERNEL
#endif

#include "myint.h"

typedef union {
  double f64;
  int64_t i64;
  uint64_t u64;
  float f32[2];
  int8_t i8[8];
  uint8_t u8[8];
} value_t;

typedef enum type_t {
  DriftID = 2,
  DriftExactID = 3,
  MultipoleID = 4,
  CavityID = 5,
  AlignID = 6,
  BeamBeamID=10,
  RotationID=11
} type_t;


#endif
