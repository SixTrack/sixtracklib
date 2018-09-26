#ifndef SIXTRACKLIB_OPENCL_CONTEXT_H__
#define SIXTRACKLIB_OPENCL_CONTEXT_H__

#if !defined( __CUDACC__ )

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stdbool.h>
    #include <stddef.h>
    #include <stdint.h>
    #include <stdlib.h>
    #include <limits.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/_impl/definitions.h"
    #include "sixtracklib/opencl/private/base_context.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#ifndef __cplusplus

namespace SIXTRL_NAMESPACE
{
    class ClContext : public ClContextBase
    {
        private:

        using _base_context_t   = ClContextBase;

        public:

        CLContext();

        explicit CLContext( size_type const node_index );
        explicit CLContext( node_id_t const node_id );
        explicit CLContext( char const* node_id_str );

        CLContext( platform_id_t const platform_idx,
                   device_id_t const device_idx );

        CLContext( CLContext const& other ) = delete;
        CLContext( CLContext&& other ) = delete;

        CLContext& operator=( CLContext const& other ) = delete;
        CLContext& operator=( CLContext&& other ) = delete;

        virtual ~CLContext() SIXTRL_NOEXCEPT;
    };
}

#endif /* __cplusplus  */

#if !defined( _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined( _GPUCODE ) && defined( __cplusplus ) */

SIXTRL_HOST_FN NS(CLBaseContext)* NS(CLBaseContext_create)();

SIXTRL_HOST_FN NS(context_size_t) NS(CLBaseContext_get_num_available_nodes)(
    const NS(CLBaseContext) *const SIXTRL_RESTRICT ctx );

SIXTRL_HOST_FN NS(context_node_info_t) const*
NS(CLBaseContext_get_available_nodes_info_begin)(
    const NS(CLBaseContext) *const SIXTRL_RESTRICT ctx );

SIXTRL_HOST_FN NS(context_node_info_t) const*
NS(CLBaseContext_get_available_nodes_info_end)(
    const NS(CLBaseContext) *const SIXTRL_RESTRICT ctx );

SIXTRL_HOST_FN NS(context_node_info_t) const*
NS(CLBaseContext_get_available_node_info_by_index)(
    const NS(CLBaseContext) *const SIXTRL_RESTRICT context,
    NS(context_size_t) const node_index );

SIXTRL_HOST_FN NS(context_node_info_t) const*
NS(CLBaseContext_get_available_node_info_by_node_id)(
    const NS(CLBaseContext) *const SIXTRL_RESTRICT context,
    const NS(context_node_id_t) *const SIXTRL_RESTRICT node_id );

SIXTRL_HOST_FN bool NS(CLBaseContext_has_node)(
    const NS(CLBaseContext) *const SIXTRL_RESTRICT ctx );

SIXTRL_HOST_FN NS(context_node_info_t) const* NS(CLBaseContext_get_node_info)(
    const NS(CLBaseContext) *const SIXTRL_RESTRICT ctx );

SIXTRL_HOST_FN NS(context_node_id_t) const* NS(CLBaseContext_get_node_id)(
    const NS(CLBaseContext) *const SIXTRL_RESTRICT ctx );

SIXTRL_HOST_FN void NS(CLBaseContext_clear)(
    NS(CLBaseContext)* SIXTRL_RESTRICT ctx );

SIXTRL_HOST_FN bool NS(CLBaseContext_select_node)(
    NS(CLBaseContext)* SIXTRL_RESTRICT ctx );

SIXTRL_HOST_FN NS(CLBaseContext)*
NS(CLBaseContext_new)( char const* SIXTRL_RESTRICT node_id_str );

SIXTRL_HOST_FN void NS(CLBaseContext_free)(
    NS(CLBaseContext)* SIXTRL_RESTRICT ctx );

SIXTRL_HOST_FN void NS(CLBaseContext_delete)(
    NS(CLBaseContext)* SIXTRL_RESTRICT ctx );


#if !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined( _GPUCODE ) && defined( __cplusplus ) */

#endif /* !defined( __CUDACC__ ) */

#endif /* SIXTRACKLIB_OPENCL_CONTEXT_H__ */

/* end: sixtracklib/opencl/context.h */
