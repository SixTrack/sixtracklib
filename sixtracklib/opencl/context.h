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
    #include "sixtracklib/opencl/argument.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#ifndef __cplusplus

namespace SIXTRL_NAMESPACE
{
    class ClContext : public ClContextBase
    {
        private:

        using _base_context_t   = ClContextBase;
        using  num_turns_t      = SIXTRL_INT64_T;

        public:

        ClContext();

        explicit ClContext( size_type const node_index );
        explicit ClContext( node_id_t const node_id );
        explicit ClContext( char const* node_id_str );

        ClContext( platform_id_t const platform_idx,
                   device_id_t const device_idx );

        ClContext( ClContext const& other ) = delete;
        ClContext( ClContext&& other ) = delete;

        ClContext& operator=( ClContext const& other ) = delete;
        ClContext& operator=( ClContext&& other ) = delete;

        virtual ~ClContext() SIXTRL_NOEXCEPT;

        bool hasTrackingKernel() const SIXTRL_NOEXCEPT;
        kernel_id_t trackingKernelId() const SIXTRL_NOEXCEPT;
        bool setTrackingKernelId( kernel_id_t const kernel_id );

        int track( ClArgument& particles_arg,
                   ClArgument& beam_elements_arg,
                   num_turns_t num_turns = num_turns_t{ 1 } );

        int track( kernel_id_t const tracking_kernel_id,
                   ClArgument& particles_arg,
                   ClArgument& beam_elements_arg,
                   num_turns_t const num_turns = num_turns_t{ 1 } );

        protected:

        virtual bool doInitDefaultPrograms();

        private:

        bool doInitDefaultProgramsPrivImpl();

        program_id_t    m_tracking_program_id;
        kernel_id_t     m_tracking_kernel_id;
    };
}

typedef SIXTRL_NAMESPACE::ClContext::num_turns_t NS(context_num_turns_t);

#else /* __cplusplus  */

#if !defined( _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined( _GPUCODE ) && defined( __cplusplus ) */

typedef void NS(ClContext);


#if !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined( _GPUCODE ) && defined( __cplusplus ) */

#endif /* __cplusplus  */

#if !defined( _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined( _GPUCODE ) && defined( __cplusplus ) */

SIXTRL_HOST_FN NS(ClContext)* NS(ClContext_create)();
SIXTRL_HOST_FN NS(ClContext)* NS(ClContext_new)( const char* node_id_str );
SIXTRL_HOST_FN void NS(ClContext_delete)( NS(ClContext)* SIXTRL_RESTRICT ctx );

SIXTRL_HOST_FN bool NS(ClContext_has_tracking_kernel)(
    const NS(ClContext) *const SIXTRL_RESTRICT ctx );

SIXTRL_HOST_FN bool NS(ClContext_has_tracking_kernel)(
    const NS(ClContext) *const SIXTRL_RESTRICT ctx );

SIXTRL_HOST_FN int NS(ClContext_get_tracking_kernel_id)(
    const NS(ClContext) *const SIXTRL_RESTRICT ctx );

SIXTRL_HOST_FN bool NS(ClContext_set_tracking_kernel_id)(
    NS(ClContext)* SIXTRL_RESTRICT ctx, int const tracking_kernel_id );

/* ------------------------------------------------------------------------- */

SIXTRL_HOST_FN int NS(ClContext_track)(
    NS(ClArgument)* SIXTRL_RESTRICT particles_arg,
    NS(ClArgument)* SIXTRL_RESTRICT beam_elements_arg );

SIXTRL_HOST_FN int NS(ClContext_track_num_turns)(
    NS(ClArgument)* SIXTRL_RESTRICT particles_arg,
    NS(ClArgument)* SIXTRL_RESTRICT beam_elements_arg,
    NS(context_num_turns_t) const num_turns );

SIXTRL_HOST_FN int NS(ClContext_execute_tracking_kernel)(
    int const tracking_kernel_id,
    NS(ClArgument)* SIXTRL_RESTRICT particles_arg,
    NS(ClArgument)* SIXTRL_RESTRICT beam_elements_arg );

SIXTRL_HOST_FN int NS(ClContext_execute_tracking_kernel_num_turns)(
    int const tracking_kernel_id,
    NS(ClArgument)* SIXTRL_RESTRICT particles_arg,
    NS(ClArgument)* SIXTRL_RESTRICT beam_elements_arg,
    NS(context_num_turns_t) const num_turns );

#if !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined( _GPUCODE ) && defined( __cplusplus ) */

#endif /* !defined( __CUDACC__ ) */

#endif /* SIXTRACKLIB_OPENCL_CONTEXT_H__ */

/* end: sixtracklib/opencl/context.h */
