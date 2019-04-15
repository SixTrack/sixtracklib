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
        this->doSetReadyForSendFlag( true );
        this->doSetReadyForReceiveFlag( true );
        this->doSetReadyForRemapFlag( true );
    }
}

NS(CudaContext)* NS(CudaContext_create)( void )
{
    return new SIXTRL_CXX_NAMESPACE::CudaContext( "" );
}

void NS(CudaContext_delete)( NS(CudaContext)* SIXTRL_RESTRICT context )
{
    delete context;
    return;
}

/* end: sixtracklib/cuda/internal/context.cu */
