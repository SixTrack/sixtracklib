#if defined( SIXTRL_NO_INCLUDES )
    #include <stddef.h>
    #include <stdint.h>
    #include <stdlib.h>
    #include <string.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#include <cuda_runtime_api.h>
#include <cuda.h>

#if !defined( SIXTRL_NO_INCLUDES )
#include "sixtracklib/_impl/definitions.h"
#include "sixtracklib/common/buffer.h"

#include "sixtracklib/testlib.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */