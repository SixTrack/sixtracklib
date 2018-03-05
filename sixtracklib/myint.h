
#ifndef _MYINT_
#define _MYINT_

#ifndef _GPUCODE
#include <stdint.h>
#else
typedef long int64_t;
typedef unsigned long uint64_t;
typedef char int8_t;
typedef unsigned char uint8_t;
#endif

#endif
