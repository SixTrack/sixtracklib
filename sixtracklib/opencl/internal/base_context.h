#ifndef SIXTRACKLIB_OPENCL_INTERNAL_BASE_CONTEXT_H__
#define SIXTRACKLIB_OPENCL_INTERNAL_BASE_CONTEXT_H__

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

namespace sixtrack
{
    using node_id_t     = NS(ComputeNodeId);
    using node_info_t   = NS(ComputeNodeInfo);

    class ClContextBase
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

        ClContextBase();

        explicit ClContextBase( size_type const node_index );
        explicit ClContextBase( node_id_t const node_id );
        explicit ClContextBase( char const* node_id_str );

        ClContextBase( platform_id_t const platform_idx,
                       device_id_t const device_idx );

        ClContextBase( ClContextBase const& other ) = delete;
        ClContextBase( ClContextBase&& other ) = delete;

        ClContextBase& operator=( ClContextBase const& other ) = delete;
        ClContextBase& operator=( ClContextBase&& other ) = delete;

        virtual ~ClContextBase() SIXTRL_NOEXCEPT;

        size_type numAvailableNodes() const SIXTRL_NOEXCEPT;

        node_info_t const*  availableNodesInfoBegin() const SIXTRL_NOEXCEPT;
        node_info_t const*  availableNodesInfoEnd()   const SIXTRL_NOEXCEPT;
        node_info_t const*  defaultNodeInfo()         const SIXTRL_NOEXCEPT;
        node_id_t defaultNodeId() const SIXTRL_NOEXCEPT;

        bool isNodeIndexAvailable(
            size_type const node_index ) const SIXTRL_RESTRICT;

        bool isNodeIdAvailable(
            node_id_t const node_id ) const SIXTRL_NOEXCEPT;

        bool isNodeIdAvailable(
            platform_id_t const platform_index,
            device_id_t const device_index ) const SIXTRL_NOEXCEPT;

        bool isNodeIdAvailable(
            char const* node_id_str ) const SIXTRL_NOEXCEPT;

        bool isDefaultNode( char const* node_id_str    ) const SIXTRL_NOEXCEPT;
        bool isDefaultNode( node_id_t const node_id    ) const SIXTRL_NOEXCEPT;
        bool isDefaultNode( size_type const node_index ) const SIXTRL_NOEXCEPT;
        bool isDefaultNode( platform_id_t const platform_index,
                            device_id_t const device_index
                          ) const SIXTRL_NOEXCEPT;

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

        cl::Device const* selectedNodeDevice() const SIXTRL_NOEXCEPT;
        cl::Device* selectedNodeDevice() SIXTRL_NOEXCEPT;

        node_id_t const*    ptrSelectedNodeId()     const SIXTRL_NOEXCEPT;
        node_info_t const*  ptrSelectedNodeInfo()   const SIXTRL_NOEXCEPT;

        bool selectNode( node_id_t const node_id );
        bool selectNode( platform_id_t const platform_idx,
                         device_id_t const device_idx );

        bool selectNode( char const* node_id_str );
        bool selectNode( size_type const index );

        void printNodesInfo() const SIXTRL_NOEXCEPT;

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

        bool compileProgram( program_id_t const program_id );

        char const* programSourceCode(
            program_id_t const program_id ) const SIXTRL_NOEXCEPT;

        bool programHasFilePath(
            program_id_t const program_id ) const SIXTRL_NOEXCEPT;

        char const* programPathToFile(
            program_id_t const program_id ) const SIXTRL_NOEXCEPT;

        char const* programCompileOptions(
            program_id_t const program_id ) const SIXTRL_NOEXCEPT;

        char const* programCompileReport(
            program_id_t const program_id ) const SIXTRL_NOEXCEPT;

        bool isProgramCompiled(
            program_id_t const program_id ) const SIXTRL_NOEXCEPT;

        size_type    numAvailablePrograms() const SIXTRL_NOEXCEPT;

        kernel_id_t  enableKernel( std::string const& kernel_name,
                                   program_id_t const program_id );

        kernel_id_t  enableKernel( char const* kernel_name,
                                   program_id_t const program_id );

        kernel_id_t findKernelByName(
            const char* SIXTRL_RESTRICT kernel_name ) const SIXTRL_NOEXCEPT;

        char const* kernelFunctionName(
            kernel_id_t const kernel_id ) const SIXTRL_NOEXCEPT;

        size_type kernelLocalMemSize(
            kernel_id_t const kernel_id ) const SIXTRL_NOEXCEPT;

        size_type kernelNumArgs(
            kernel_id_t const kernel_id ) const SIXTRL_NOEXCEPT;

        size_type kernelWorkGroupSize(
            kernel_id_t const kernel_id ) const SIXTRL_NOEXCEPT;

        size_type kernelPreferredWorkGroupSizeMultiple(
            kernel_id_t const kernel_id ) const SIXTRL_NOEXCEPT;

        program_id_t programIdByKernelId(
            kernel_id_t const kernel_id ) const SIXTRL_NOEXCEPT;

        bool hasRemappingProgram() const SIXTRL_NOEXCEPT;
        program_id_t remappingProgramId() const SIXTRL_NOEXCEPT;

        bool hasRemappingKernel()  const SIXTRL_NOEXCEPT;
        kernel_id_t remappingKernelId() const SIXTRL_NOEXCEPT;
        bool setRemappingKernelId( kernel_id_t const kernel_id ) SIXTRL_NOEXCEPT;

        size_type   numAvailableKernels() const SIXTRL_NOEXCEPT;

        cl::Program* openClProgram( program_id_t const program_id ) SIXTRL_NOEXCEPT;
        cl::Kernel* openClKernel( kernel_id_t const kernel_id ) SIXTRL_NOEXCEPT;
        cl::CommandQueue* openClQueue() SIXTRL_NOEXCEPT;
        cl::Context* openClContext() SIXTRL_NOEXCEPT;

        protected:

        using program_data_t = struct ProgramData
        {
            ProgramData() :
                m_file_path(),
                m_source_code(),
                m_compile_options(),
                m_compile_report(),
                m_kernels(),
                m_compiled( false )
            {

            }

            ProgramData( ProgramData const& orig ) = default;
            ProgramData( ProgramData&& orig ) = default;

            ProgramData& operator=( ProgramData const& rhs ) = default;
            ProgramData& operator=( ProgramData&& rhs ) = default;

            ~ProgramData() = default;

            std::string                m_file_path;
            std::string                m_source_code;
            std::string                m_compile_options;
            std::string                m_compile_report;
            std::vector< kernel_id_t > m_kernels;
            bool                       m_compiled;
        };

        using kernel_data_t = struct KernelData
        {
            KernelData() :
                m_kernel_name(),
                m_program_id( -1 ),
                m_num_args( -1 ),
                m_work_group_size( size_type{ 0 } ),
                m_preferred_work_group_multiple( size_type{ 0 } ),
                m_local_mem_size( size_type{ 0 } )
            {

            }

            KernelData( KernelData const& orig ) = default;
            KernelData( KernelData&& orig ) = default;

            KernelData& operator=( KernelData const& rhs ) = default;
            KernelData& operator=( KernelData&& rhs ) = default;

            ~KernelData() = default;

            std::string   m_kernel_name;
            program_id_t  m_program_id;
            size_type     m_num_args;
            size_type     m_work_group_size;
            size_type     m_preferred_work_group_multiple;
            size_type     m_local_mem_size;
        };

        using program_data_list_t = std::vector< program_data_t >;
        using kernel_data_list_t  = std::vector< kernel_data_t >;

        virtual void doClear();

        virtual bool doInitDefaultPrograms();
        virtual bool doInitDefaultKernels();

        virtual bool doCompileProgram(
            cl::Program& cl_program, program_data_t& program_data );

        virtual bool doSelectNode( size_type node_index );

        kernel_data_list_t  const& kernelData()  const SIXTRL_NOEXCEPT;
        program_data_list_t const& programData() const SIXTRL_NOEXCEPT;

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

        bool doInitDefaultProgramsBaseImpl();
        bool doInitDefaultKernelsBaseImpl();

        bool doCompileProgramBaseImpl(
            cl::Program& cl_program, program_data_t& program_data );

        bool doSelectNodeBaseImpl( size_type const node_index );

        void doClearBaseImpl() SIXTRL_NOEXCEPT;

        std::vector< cl::Program >      m_cl_programs;
        std::vector< cl::Kernel  >      m_cl_kernels;
        std::vector< cl::Buffer  >      m_cl_buffers;
        std::vector< node_id_t >        m_available_nodes_id;
        std::vector< node_info_t>       m_available_nodes_info;
        std::vector< cl::Device >       m_available_devices;

        std::vector< program_data_t >   m_program_data;
        std::vector< kernel_data_t  >   m_kernel_data;

        std::string                     m_default_compile_options;

        cl::Context                     m_cl_context;
        cl::CommandQueue                m_cl_queue;

        program_id_t                    m_remap_program_id;
        kernel_id_t                     m_remap_kernel_id;
        int64_t                         m_selected_node_index;
    };
}

typedef SIXTRL_NAMESPACE::ClContextBase     NS(ClContextBase);

#else /* defined( __cplusplus ) */

    #if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
        #include <CL/cl.h>
    #endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

typedef void                                NS(ClContextBase);

#endif /* defined( __cplusplus ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined( _GPUCODE ) && defined( __cplusplus ) */

typedef size_t              NS(context_size_t);
typedef NS(ComputeNodeId)   NS(context_node_id_t);
typedef NS(ComputeNodeInfo) NS(context_node_info_t);
typedef SIXTRL_INT64_T      NS(context_kernel_id_t);
typedef SIXTRL_INT64_T      NS(context_buffer_id_t);

SIXTRL_HOST_FN NS(ClContextBase)* NS(ClContextBase_create)();

SIXTRL_HOST_FN NS(context_size_t) NS(ClContextBase_get_num_available_nodes)(
    const NS(ClContextBase) *const SIXTRL_RESTRICT ctx );

SIXTRL_HOST_FN NS(context_node_info_t) const*
NS(ClContextBase_get_available_nodes_info_begin)(
    const NS(ClContextBase) *const SIXTRL_RESTRICT ctx );

SIXTRL_HOST_FN NS(context_node_info_t) const*
NS(ClContextBase_get_available_nodes_info_end)(
    const NS(ClContextBase) *const SIXTRL_RESTRICT ctx );

SIXTRL_HOST_FN NS(context_node_info_t) const*
NS(ClContextBase_get_available_node_info_by_index)(
    const NS(ClContextBase) *const SIXTRL_RESTRICT context,
    NS(context_size_t) const node_index );

SIXTRL_HOST_FN NS(context_node_info_t) const*
NS(ClContextBase_get_default_node_info)(
    const NS(ClContextBase) *const SIXTRL_RESTRICT context );

SIXTRL_HOST_FN NS(context_node_id_t)
NS(ClContextBase_get_default_node_id)(
    const NS(ClContextBase) *const SIXTRL_RESTRICT context );

SIXTRL_HOST_FN NS(context_node_info_t) const*
NS(ClContextBase_get_available_node_info_by_node_id)(
    const NS(ClContextBase) *const SIXTRL_RESTRICT context,
    const NS(context_node_id_t) *const SIXTRL_RESTRICT node_id );

SIXTRL_HOST_FN bool NS(ClContextBase_is_node_id_str_available)(
    const NS(ClContextBase) *const SIXTRL_RESTRICT ctx,
    char const* SIXTRL_RESTRICT node_id_str );

SIXTRL_HOST_FN bool NS(ClContextBase_is_node_id_available)(
    const NS(ClContextBase) *const SIXTRL_RESTRICT ctx,
    const NS(context_node_id_t) *const SIXTRL_RESTRICT node_id );

SIXTRL_HOST_FN bool NS(ClContextBase_is_node_index_available)(
    const NS(ClContextBase) *const SIXTRL_RESTRICT ctx,
    NS(context_size_t) const node_index );

SIXTRL_HOST_FN bool NS(ClContextBase_is_platform_device_tuple_available)(
    const NS(ClContextBase) *const SIXTRL_RESTRICT ctx,
    NS(comp_node_id_num_t) const platform_idx,
    NS(comp_node_id_num_t) const device_idx );

SIXTRL_HOST_FN bool NS(ClContextBase_is_node_id_default_node)(
    const NS(ClContextBase) *const SIXTRL_RESTRICT ctx,
    const NS(context_node_id_t) *const SIXTRL_RESTRICT node_id );

SIXTRL_HOST_FN bool NS(ClContextBase_is_node_id_str_default_node)(
    const NS(ClContextBase) *const SIXTRL_RESTRICT ctx,
    char const* SIXTRL_RESTRICT node_id_str );

SIXTRL_HOST_FN bool NS(ClContextBase_is_platform_device_tuple_default_node)(
    const NS(ClContextBase) *const SIXTRL_RESTRICT ctx,
    NS(comp_node_id_num_t) const platform_idx,
    NS(comp_node_id_num_t) const device_idx );

SIXTRL_HOST_FN bool NS(ClContextBase_is_node_index_default_node)(
    const NS(ClContextBase) *const SIXTRL_RESTRICT ctx,
    NS(context_size_t) const node_index );

SIXTRL_HOST_FN bool NS(ClContextBase_has_selected_node)(
    const NS(ClContextBase) *const SIXTRL_RESTRICT ctx );

SIXTRL_HOST_FN cl_device_id NS(ClContextBase_get_selected_node_device)(
    const NS(ClContextBase) *const SIXTRL_RESTRICT ctx );

SIXTRL_HOST_FN NS(context_node_info_t) const*
NS(ClContextBase_get_selected_node_info)(
    const NS(ClContextBase) *const SIXTRL_RESTRICT ctx );

SIXTRL_HOST_FN NS(context_node_id_t) const*
NS(ClContextBase_get_selected_node_id)(
    const NS(ClContextBase) *const SIXTRL_RESTRICT ctx );

SIXTRL_HOST_FN void NS(ClContextBase_print_nodes_info)(
    const NS(ClContextBase) *const SIXTRL_RESTRICT ctx );

SIXTRL_HOST_FN void NS(ClContextBase_clear)(
    NS(ClContextBase)* SIXTRL_RESTRICT ctx );

SIXTRL_HOST_FN bool NS(ClContextBase_select_node)(
    NS(ClContextBase)* SIXTRL_RESTRICT ctx,
    char const* node_id_str );

SIXTRL_HOST_FN bool NS(ClContextBase_select_node_by_node_id)(
    NS(ClContextBase)* SIXTRL_RESTRICT ctx,
    const NS(context_node_id_t) *const SIXTRL_RESTRICT node_id );

SIXTRL_HOST_FN bool NS(ClContextBase_select_node_by_index)(
    NS(ClContextBase)* SIXTRL_RESTRICT ctx,
    NS(context_size_t) const index );

SIXTRL_HOST_FN NS(context_size_t) NS(ClContextBase_get_num_available_programs)(
    const NS(ClContextBase) *const SIXTRL_RESTRICT ctx );

SIXTRL_HOST_FN int NS(ClContextBase_add_program_file)(
    NS(ClContextBase)* SIXTRL_RESTRICT ctx,
    char const* SIXTRL_RESTRICT path_to_program_file,
    char const* SIXTRL_RESTRICT compile_options );

SIXTRL_HOST_FN bool NS(ClContextBase_compile_program)(
    NS(ClContextBase)* SIXTRL_RESTRICT ctx, int const program_id );

SIXTRL_HOST_FN char const* NS(ClContextBase_get_program_source_code)(
    const NS(ClContextBase) *const SIXTRL_RESTRICT ctx, int const program_id );

SIXTRL_HOST_FN bool NS(ClContextBase_has_program_file_path)(
    const NS(ClContextBase) *const SIXTRL_RESTRICT ctx, int const program_id );

SIXTRL_HOST_FN char const* NS(ClContextBase_get_program_path_to_file)(
    const NS(ClContextBase) *const SIXTRL_RESTRICT ctx, int const program_id );

SIXTRL_HOST_FN char const* NS(ClContextBase_get_program_compile_options)(
    const NS(ClContextBase) *const SIXTRL_RESTRICT ctx, int const program_id );

SIXTRL_HOST_FN char const* NS(ClContextBase_get_program_compile_report)(
    const NS(ClContextBase) *const SIXTRL_RESTRICT ctx, int const program_id );

SIXTRL_HOST_FN bool NS(ClContextBase_is_program_compiled)(
    const NS(ClContextBase) *const SIXTRL_RESTRICT ctx, int const program_id );

SIXTRL_HOST_FN int NS(ClContextBase_enable_kernel)(
    NS(ClContextBase)* SIXTRL_RESTRICT ctx,
    char const* kernel_name, int const program_id );

SIXTRL_HOST_FN NS(context_size_t) NS(ClContextBase_get_num_available_kernels)(
    const NS(ClContextBase) *const SIXTRL_RESTRICT ctx );

SIXTRL_HOST_FN int NS(ClContextBase_find_kernel_id_by_name)(
    NS(ClContextBase)* SIXTRL_RESTRICT ctx,
    char const* SIXTRL_RESTRICT kernel_name );

SIXTRL_HOST_FN char const* NS(ClContextBase_get_kernel_function_name)(
    const NS(ClContextBase) *const SIXTRL_RESTRICT ctx, int const kernel_id );

SIXTRL_HOST_FN NS(context_size_t) NS(ClContextBase_get_kernel_local_mem_size)(
    const NS(ClContextBase) *const SIXTRL_RESTRICT ctx, int const kernel_id );

SIXTRL_HOST_FN NS(context_size_t) NS(ClContextBase_get_kernel_num_args)(
    const NS(ClContextBase) *const SIXTRL_RESTRICT ctx, int const kernel_id );

SIXTRL_HOST_FN NS(context_size_t) NS(ClContextBase_get_kernel_work_group_size)(
    const NS(ClContextBase) *const SIXTRL_RESTRICT ctx, int const kernel_id );

SIXTRL_HOST_FN NS(context_size_t)
NS(ClContextBase_get_kernel_preferred_work_group_size_multiple)(
    const NS(ClContextBase) *const SIXTRL_RESTRICT ctx, int const kernel_id );

SIXTRL_HOST_FN int NS(ClContextBase_get_program_id_by_kernel_id)(
    const NS(ClContextBase) *const SIXTRL_RESTRICT ctx, int const kernel_id );

SIXTRL_HOST_FN bool NS(ClContextBase_has_remapping_program)(
    const NS(ClContextBase) *const SIXTRL_RESTRICT ctx );

SIXTRL_HOST_FN int NS(ClContextBase_get_remapping_program_id)(
    const NS(ClContextBase) *const SIXTRL_RESTRICT ctx );

SIXTRL_HOST_FN bool NS(ClContextBase_has_remapping_kernel)(
    const NS(ClContextBase) *const SIXTRL_RESTRICT ctx );

SIXTRL_HOST_FN int NS(ClContextBase_get_remapping_kernel_id)(
    const NS(ClContextBase) *const SIXTRL_RESTRICT ctx );

SIXTRL_HOST_FN bool NS(ClContextBase_set_remapping_kernel_id)(
    NS(ClContextBase)* SIXTRL_RESTRICT ctx, int const kernel_id );

SIXTRL_HOST_FN cl_program NS(ClContextBase_get_program)(
    NS(ClContextBase)* SIXTRL_RESTRICT ctx,
    int const program_id );

SIXTRL_HOST_FN cl_kernel NS(ClContextBase_get_kernel)(
    NS(ClContextBase)* SIXTRL_RESTRICT ctx,
    int const kernel_id );

SIXTRL_HOST_FN cl_command_queue NS(ClContextBase_get_queue)(
    NS(ClContextBase)* SIXTRL_RESTRICT ctx );

SIXTRL_HOST_FN cl_context NS(ClContextBase_get_opencl_context)(
    NS(ClContextBase)* SIXTRL_RESTRICT ctx );

SIXTRL_HOST_FN NS(ClContextBase)*
NS(ClContextBase_new_on_selected_node_id_str)(
    char const* SIXTRL_RESTRICT node_id_str );

SIXTRL_HOST_FN NS(ClContextBase)*
NS(ClContextBase_new_on_selected_node_id)(
    const NS(context_node_id_t) *const node_id );

SIXTRL_HOST_FN NS(ClContextBase)* NS(ClContextBase_new)( void );

SIXTRL_HOST_FN void NS(ClContextBase_free)(
    NS(ClContextBase)* SIXTRL_RESTRICT ctx );

SIXTRL_HOST_FN void NS(ClContextBase_delete)(
    NS(ClContextBase)* SIXTRL_RESTRICT ctx );


#if !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined( _GPUCODE ) && defined( __cplusplus ) */

#endif /* !defined( __CUDACC__ ) */

#endif /* SIXTRACKLIB_OPENCL_INTERNAL_BASE_CONTEXT_H__ */

/* end: sixtracklib/opencl/internal/base_context.h */
