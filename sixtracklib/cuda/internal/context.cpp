#include "sixtracklib/cuda/context.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/cuda/internal/context_base.h"

namespace SIXTRL_CXX_NAMESPACE
{
    CudaContext::CudaContext( char const* config_str ) :
        SIXTRL_CXX_NAMESPACE::CudaContextBase( config_str )
    {

    }
}

/* end: sixtracklib/cuda/internal/context.cu */
