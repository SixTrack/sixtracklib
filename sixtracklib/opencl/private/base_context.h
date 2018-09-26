#ifndef SIXTRACKLIB_OPENCL_PRIVATE_BASE_CONTEXT_H__
#define SIXTRACKLIB_OPENCL_PRIVATE_BASE_CONTEXT_H__

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
    #include "sixtracklib/common/compute_arch.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus )

    #if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
        #include <cstddef>
        #include <cstdint>
        #include <cstdlib>
        #include <iterator>
        #include <string>
        #include <map>
        #include <vector>

        #include <CL/cl.hpp>
    #endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

using NS(context_size_t) = std::size_t;

namespace SIXTRL_NAMESPACE
{
    using node_id_t     = NS(ComputeNodeId);
    using node_info_t   = NS(ComputeNodeInfo);

    class CLContextBase
    {
        public:

        using node_id_t         = SIXTRL_NAMESPACE::node_id_t;
        using node_info_t       = SIXTRL_NAMESPACE::node_info_t;
        using size_type         = std::size_t;

        using platform_id_t     = NS(comp_node_id_num_t);
        using device_id_t       = NS(comp_node_id_num_t);

        using kernel_id_t       = int64_t;
        using program_id_t      = int64_t;
        using kernel_arg_id_t   = int64_t;

        CLContextBase();

        CLContextBase( CLContextBase const& other ) = delete;
        CLContextBase( CLContextBase&& other ) = delete;

        CLContextBase& operator=( CLContextBase const& other ) = delete;
        CLContextBase& operator=( CLContextBase&& other ) = delete;

        virtual ~CLContextBase() SIXTRL_NOEXCEPT;

        size_type numAvailableNodes() const SIXTRL_NOEXCEPT;

        node_info_t const*  availableNodesInfoBegin() const SIXTRL_NOEXCEPT;
        node_info_t const*  availableNodesInfoEnd()   const SIXTRL_NOEXCEPT;

        bool isNodeIdAvailable(
            node_id_t const node_id ) const SIXTRL_NOEXCEPT;

        bool isNodeIdAvailable(
            platform_id_t const platform_index,
            device_id_t const device_index ) const SIXTRL_NOEXCEPT;

        bool isNodeIdAvailable(
            char const* node_id_str ) const SIXTRL_NOEXCEPT;

        node_id_t const*   ptrAvailableNodesId(
            size_type const index ) const SIXTRL_NOEXCEPT;

        node_id_t const*   ptrAvailableNodesId(
            platform_id_t const platform_index,
            device_id_t   const device_index ) const SIXTRL_NOEXCEPT;

        node_id_t const*   ptrAvailableNodesId(
            char const* SIXTRL_RESTRICT node_id_str ) const SIXTRL_NOEXCEPT;


        node_info_t const*  ptrAvailableNodesInfo(
            size_type const index ) const SIXTRL_NOEXCEPT;

        node_info_t const*  ptrAvailableNodesInfo(
            node_id_t const node_id ) const SIXTRL_NOEXCEPT;

        node_info_t const*  ptrAvailableNodesInfo(
            char const* SIXTRL_RESTRICT node_id_str ) const SIXTRL_NOEXCEPT;

        bool hasSelectedNode() const SIXTRL_NOEXCEPT;

        node_id_t const*    ptrSelectedNodeId()     const SIXTRL_NOEXCEPT;
        node_info_t const*  ptrSelectedNodeInfo()   const SIXTRL_NOEXCEPT;

        bool selectNode( node_id_t const node_id );
        bool selectNode( char const* node_id_str );
        bool selectNode( size_type const index );

        void clear();

        void setDefaultCompileOptions( std::string const& compile_options_str );
        void setDefaultCompileOptions( char const* compile_options_str );

        char const* defaultCompileOptions() const SIXTRL_NOEXCEPT;

        program_id_t addProgramCode( std::string const& source_code );
        program_id_t addProgramCode( char const* source_code );

        program_id_t addProgramCode( std::string const& source_code,
                                     std::string const& compile_options );

        program_id_t addProgramCode( char const* source_code,
                                     char const* compile_options );

        program_id_t addProgramFile( std::string const& path_to_program );
        program_id_t addProgramFile( char const* path_to_program );

        program_id_t addProgramFile( std::string const& path_to_program,
                                     std::string const& compile_options );

        program_id_t addProgramFile( char const* path_to_program,
                                     char const* compile_options );

        size_type    numAvailablePrograms() const SIXTRL_NOEXCEPT;

        kernel_id_t  enableKernel( std::string const& kernel_name,
                                   program_id_t const program_id );

        kernel_id_t  enableKernel( char const* kernel_name,
                                   program_id_t const program_id );

        size_type   numAvailableKernels() const SIXTRL_NOEXCEPT;

        cl::Program* program( program_id_t const program_id ) SIXTRL_NOEXCEPT;
        cl::Kernel* kernel( kernel_id_t const kernel_id ) SIXTRL_NOEXCEPT;
        cl::CommandQueue* queue() SIXTRL_NOEXCEPT;

        protected:

        using program_data_t = struct ProgramData
        {
            ProgramData() :
                m_file_path(),
                m_source_code(),
                m_compile_options(),
                m_kernels(),
                m_compiled( false )
            {

            }

            ProgramData( ProgramData const& orig ) = default;
            ProgramData( ProgramData&& orig ) = default;

            ProgramData& operator=( ProgramData const& rhs ) = default;
            ProgramData& operator=( ProgramData&& rhs ) = default;

            ~ProgramData() = default;

            std::string                          m_file_path;
            std::string                          m_source_code;
            std::string                          m_compile_options;
            std::map< kernel_id_t, std::string > m_kernels;
            bool                                 m_compiled;
        };

        bool compileProgram( program_id_t const program_id );

        size_type findAvailableNodesIndex( platform_id_t const platform_index,
            device_id_t const device_index ) const SIXTRL_NOEXCEPT;

        size_type findAvailableNodesIndex( char const* node_id_str
            ) const SIXTRL_NOEXCEPT;


        private:

        static void UpdateAvailableNodes(
            std::vector< node_id_t>& available_nodes_id,
            std::vector< node_info_t >&  available_nodes_info,
            std::vector< cl::Device  >&  available_devices,
            const char *const filter_str = nullptr );

        std::vector< cl::Program >      m_cl_programs;
        std::vector< cl::Kernel  >      m_cl_kernels;
        std::vector< cl::Buffer  >      m_cl_buffers;
        std::vector< node_id_t >        m_available_nodes_id;
        std::vector< node_info_t>       m_available_nodes_info;
        std::vector< cl::Device >       m_available_devices;

        std::vector< program_data_t >   m_program_data;

        std::string                     m_default_compile_options;

        cl::Context                     m_cl_context;
        cl::CommandQueue                m_cl_queue;

        int64_t                         m_selected_node_index;
    };
}

typedef SIXTRL_NAMESPACE::CLContextBase     NS(CLContextBase);

#else /* defined( __cplusplus ) */

typedef void                                NS(CLContextBase);

#endif /* defined( __cplusplus ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined( _GPUCODE ) && defined( __cplusplus ) */

typedef size_t              NS(context_size_t);
typedef NS(ComputeNodeId)   NS(context_node_id_t);
typedef NS(ComputeNodeInfo) NS(context_node_info_t);
typedef SIXTRL_INT64_T      NS(context_kernel_id_t);
typedef SIXTRL_INT64_T      NS(context_buffer_id_t);

SIXTRL_HOST_FN NS(CLContextBase)* NS(CLContextBase_create)();

SIXTRL_HOST_FN NS(context_size_t) NS(CLContextBase_get_num_available_nodes)(
    const NS(CLContextBase) *const SIXTRL_RESTRICT ctx );

SIXTRL_HOST_FN NS(context_node_info_t) const*
NS(CLContextBase_get_available_nodes_info_begin)(
    const NS(CLContextBase) *const SIXTRL_RESTRICT ctx );

SIXTRL_HOST_FN NS(context_node_info_t) const*
NS(CLContextBase_get_available_nodes_info_end)(
    const NS(CLContextBase) *const SIXTRL_RESTRICT ctx );

SIXTRL_HOST_FN NS(context_node_info_t) const*
NS(CLContextBase_get_available_node_info_by_index)(
    const NS(CLContextBase) *const SIXTRL_RESTRICT context,
    NS(context_size_t) const node_index );

SIXTRL_HOST_FN NS(context_node_info_t) const*
NS(CLContextBase_get_available_node_info_by_node_id)(
    const NS(CLContextBase) *const SIXTRL_RESTRICT context,
    const NS(context_node_id_t) *const SIXTRL_RESTRICT node_id );

SIXTRL_HOST_FN bool NS(CLContextBase_is_node_id_str_available)(
    const NS(CLContextBase) *const SIXTRL_RESTRICT ctx,
    char const* SIXTRL_RESTRICT node_id_str );

SIXTRL_HOST_FN bool NS(CLContextBase_is_node_id_available)(
    const NS(CLContextBase) *const SIXTRL_RESTRICT ctx,
    const NS(context_node_id_t) *const SIXTRL_RESTRICT node_id );

SIXTRL_HOST_FN bool NS(CLContextBase_has_selected_node)(
    const NS(CLContextBase) *const SIXTRL_RESTRICT ctx );

SIXTRL_HOST_FN NS(context_node_info_t) const* NS(CLContextBase_get_node_info)(
    const NS(CLContextBase) *const SIXTRL_RESTRICT ctx );

SIXTRL_HOST_FN NS(context_node_id_t) const* NS(CLContextBase_get_node_id)(
    const NS(CLContextBase) *const SIXTRL_RESTRICT ctx );

SIXTRL_HOST_FN void NS(CLContextBase_clear)(
    NS(CLContextBase)* SIXTRL_RESTRICT ctx );

SIXTRL_HOST_FN bool NS(CLContextBase_select_node)(
    NS(CLContextBase)* SIXTRL_RESTRICT ctx,
    char const* node_id_str );

SIXTRL_HOST_FN bool NS(CLContextBase_select_node_by_node_id)(
    NS(CLContextBase)* SIXTRL_RESTRICT ctx,
    NS(context_node_id_t) const node_id );

SIXTRL_HOST_FN bool NS(CLContextBase_select_node_by_index)(
    NS(CLContextBase)* SIXTRL_RESTRICT ctx,
    NS(context_size_t) const index );

SIXTRL_HOST_FN NS(CLContextBase)*
NS(CLContextBase_new)( char const* SIXTRL_RESTRICT node_id_str );

SIXTRL_HOST_FN void NS(CLContextBase_free)(
    NS(CLContextBase)* SIXTRL_RESTRICT ctx );

SIXTRL_HOST_FN void NS(CLContextBase_delete)(
    NS(CLContextBase)* SIXTRL_RESTRICT ctx );


#if !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined( _GPUCODE ) && defined( __cplusplus ) */

#endif /* !defined( __CUDACC__ ) */

#endif /* SIXTRACKLIB_OPENCL_PRIVATE_BASE_CONTEXT_H__ */

/* end: sixtracklib/opencl/private/base_context.h */
