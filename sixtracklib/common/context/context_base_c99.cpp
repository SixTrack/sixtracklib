#include "sixtracklib/common/context/context_base.h"

#if defined( __cplusplus ) && !defined( _GPUCODE ) && !defined( __CUDA_ARCH__ )

#include <cstddef>
#include <cstdlib>
#include <cstdint>

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/context/definitions.h"
#include "sixtracklib/common/context/context_base.hpp"
#include "sixtracklib/common/context/argument_base.hpp"

void NS(Context_delete)( ::NS(ContextBase)* SIXTRL_RESTRICT ctx )
{
    if( ctx != nullptr ) delete ctx;
    return;
}

void NS(Context_clear)( ::NS(ContextBase)* SIXTRL_RESTRICT ctx )
{
    if( ctx != nullptr ) ctx->clear();
    return;
}

::NS(arch_id_t) NS(Context_get_arch_id)(
    const ::NS(ContextBase) *const SIXTRL_RESTRICT ctx )
{
    return ( ctx != nullptr )
        ? ctx->archId() : SIXTRL_CXX_NAMESPACE::CONTEXT_TYPE_INVALID;
}

bool NS(Context_has_arch_string)(
    const ::NS(ContextBase) *const SIXTRL_RESTRICT ctx )
{
    return ( ( ctx != nullptr ) && ( ctx->hasArchStr() ) );
}

char const* NS(Context_get_arch_string)(
    const ::NS(ContextBase) *const SIXTRL_RESTRICT ctx )
{
    return ( ctx != nullptr ) ? ctx->ptrArchStr() : nullptr;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

bool NS(Context_has_config_string)(
    const ::NS(ContextBase) *const SIXTRL_RESTRICT ctx )
{
    return ( ctx != nullptr ) ? ctx->hasConfigStr() : false;
}

char const* NS(Context_get_config_string)(
    const ::NS(ContextBase) *const SIXTRL_RESTRICT ctx )
{
    return ( ctx != nullptr ) ? ctx->ptrConfigStr() : nullptr;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

bool NS(Context_uses_nodes)(
    const ::NS(ContextBase) *const SIXTRL_RESTRICT ctx )
{
    return ( ( ctx != nullptr ) && ( ctx->usesNodes() ) );
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

::NS(context_status_t) NS(Context_send_detailed)(
    ::NS(ContextBase)* SIXTRL_RESTRICT ctx,
    ::NS(ArgumentBase)* SIXTRL_RESTRICT destination,
    void const* SIXTRL_RESTRICT source, NS(context_size_t) const src_len )
{
    return ( ctx != nullptr )
        ? ctx->send( destination, source, src_len )
        : ::NS(CONTEXT_STATUS_GENERAL_FAILURE);
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

::NS(context_status_t) NS(Context_send_buffer)(
    ::NS(ContextBase)* SIXTRL_RESTRICT ctx,
    ::NS(ArgumentBase)* SIXTRL_RESTRICT destination,
    ::NS(Buffer) const* SIXTRL_RESTRICT source )
{
    return ( ctx != nullptr )
        ? ctx->send( destination, source )
        : ::NS(CONTEXT_STATUS_GENERAL_FAILURE);
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

::NS(context_status_t) NS(Context_receive_detailed)(
    ::NS(ContextBase)* SIXTRL_RESTRICT ctx,
    void* SIXTRL_RESTRICT destination,
    ::NS(context_size_t) const destination_capacity,
    ::NS(ArgumentBase)* SIXTRL_RESTRICT source )
{
    return ( ctx != nullptr )
        ? ctx->receive( destination, destination_capacity, source )
        : ::NS(CONTEXT_STATUS_GENERAL_FAILURE);
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

::NS(context_status_t) NS(Context_receive_buffer)(
    ::NS(ContextBase)* SIXTRL_RESTRICT ctx,
    ::NS(Buffer)* SIXTRL_RESTRICT destination,
    ::NS(ArgumentBase)* SIXTRL_RESTRICT source )
{
    return ( ctx != nullptr )
        ? ctx->receive( destination, source )
        : ::NS(CONTEXT_STATUS_GENERAL_FAILURE);
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

::NS(context_status_t) NS(Context_remap_sent_cobjects_buffer)(
    ::NS(ContextBase)* SIXTRL_RESTRICT ctx,
    ::NS(ArgumentBase)* SIXTRL_RESTRICT arg )
{
    return ( ctx != nullptr )
        ? ctx->remapSentCObjectsBuffer( arg )
        : ::NS(CONTEXT_STATUS_GENERAL_FAILURE);
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

bool NS(Context_is_ready_to_remap)(
    const ::NS(ContextBase) *const SIXTRL_RESTRICT ctx )
{
    return ( ( ctx != nullptr ) && ( ctx->readyForRemap() ) );
}

bool NS(Context_is_ready_to_send)(
    const ::NS(ContextBase) *const SIXTRL_RESTRICT ctx )
{
    return ( ( ctx != nullptr ) && ( ctx->readyForSend() ) );
}

bool NS(Context_is_ready_to_receive)(
    const ::NS(ContextBase) *const SIXTRL_RESTRICT ctx )
{
    return ( ( ctx != nullptr ) && ( ctx->readyForReceive() ) );
}

bool NS(Context_is_in_debug_mode)(
    const ::NS(ContextBase) *const SIXTRL_RESTRICT ctx )
{
    return ( ( ctx != nullptr ) && ( ctx->isInDebugMode() ) );
}

#endif /* C++, Host */

/* end: sixtracklib/common/context/context_base_c99.cpp */
