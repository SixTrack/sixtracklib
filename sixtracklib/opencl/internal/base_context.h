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
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/context/compute_arch.h"
    #include "sixtracklib/common/context/context_abs_base.h"
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

namespace SIXTRL_CXX_NAMESPACE
{
    using node_id_t     = NS(ComputeNodeId);
    using node_info_t   = NS(ComputeNodeInfo);

    class ClArgument;

    class ClContextBase
    {
        public:

        using node_id_t         = SIXTRL_CXX_NAMESPACE::node_id_t;
        using node_info_t       = SIXTRL_CXX_NAMESPACE::node_info_t;
        using size_type         = std::size_t;

        using platform_id_t     = NS(comp_node_id_num_t);
        using device_id_t       = NS(comp_node_id_num_t);

        using kernel_id_t       = int64_t;
        using program_id_t      = int64_t;
        using kernel_arg_id_t   = int64_t;
        using kernel_arg_type_t = uint32_t;

        static kernel_arg_type_t const ARG_TYPE_NONE;
        static kernel_arg_type_t const ARG_TYPE_VALUE;
        static kernel_arg_type_t const ARG_TYPE_RAW_PTR;
        static kernel_arg_type_t const ARG_TYPE_INVALID;

        explicit ClContextBase(
            const char *const SIXTRL_RESTRICT config_str = nullptr );

        explicit ClContextBase(
            size_type const node_index,
            const char *const SIXTRL_RESTRICT config_str = nullptr  );

        explicit ClContextBase( node_id_t const node_id,
            const char *const SIXTRL_RESTRICT config_str = nullptr   );

        ClContextBase( char const* node_id_str,
            const char *const SIXTRL_RESTRICT config_str  );

        ClContextBase(
            platform_id_t const platform_idx,
            device_id_t const device_idx,
            const char *const SIXTRL_RESTRICT config_str = nullptr   );

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
        size_type           selectedNodeIndex()     const SIXTRL_NOEXCEPT;

        std::string selectedNodeIdStr() const SIXTRL_NOEXCEPT;
        bool selectedNodeIdStr( char* SIXTRL_RESTRICT node_id_str,
                size_type const max_str_length ) const SIXTRL_NOEXCEPT;

        bool selectNode( node_id_t const node_id );
        bool selectNode( platform_id_t const platform_idx,
                         device_id_t const device_idx );

        bool selectNode( char const* node_id_str );
        bool selectNode( size_type const index );

        void printNodesInfo() const SIXTRL_NOEXCEPT;

        void clear();

        char const* configStr() const SIXTRL_NOEXCEPT;

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

        size_type kernelMaxWorkGroupSize(
            kernel_id_t const kernel_id ) const SIXTRL_NOEXCEPT;

        size_type kernelPreferredWorkGroupSizeMultiple(
            kernel_id_t const kernel_id ) const SIXTRL_NOEXCEPT;

        bool setKernelWorkGroupSize(
            kernel_id_t const kernel_id, size_type work_group_size ) SIXTRL_NOEXCEPT;

        size_type kernelExecCounter(
            kernel_id_t const kernel_id ) const SIXTRL_NOEXCEPT;

        ClArgument* ptrKernelArgument( kernel_id_t const kernel_id,
            size_type const arg_index ) SIXTRL_NOEXCEPT;

        kernel_arg_type_t kernelArgumentType( kernel_id_t const kernel_id,
            size_type const arg_index) const SIXTRL_NOEXCEPT;

        ClArgument const* ptrKernelArgument( kernel_id_t const kernel_id,
            size_type const arg_index ) const SIXTRL_NOEXCEPT;

        void resetSingleKernelArgument( kernel_id_t const kernel_id,
            size_type const arg_index ) SIXTRL_NOEXCEPT;

        void resetKernelArguments( kernel_id_t const kernel_id ) SIXTRL_NOEXCEPT;

        void assignKernelArgument( kernel_id_t const kernel_id,
            size_type const arg_index, ClArgument& SIXTRL_RESTRICT_REF arg );

        template< typename T >
        void assignKernelArgumentPtr( kernel_id_t const kernel_id,
            size_type const arg_index, T* SIXTRL_RESTRICT ptr ) SIXTRL_NOEXCEPT
        {
            using _this_t = ClContextBase;

            SIXTRL_ASSERT( kernel_id >= kernel_id_t{ 0 } );
            SIXTRL_ASSERT( static_cast< _this_t::size_type >( kernel_id ) <
                           this->numAvailableKernels() );

            this->m_kernel_data[ kernel_id ].setKernelArg(
                    _this_t::ARG_TYPE_RAW_PTR, arg_index, nullptr );

            cl::Kernel* kernel = this->openClKernel( kernel_id );
            if( kernel != nullptr ) kernel->setArg( arg_index, ptr );
        }

        template< typename T >
        void assignKernelArgumentValue( kernel_id_t const kernel_id,
            size_type const arg_index, T& SIXTRL_RESTRICT_REF ref ) SIXTRL_NOEXCEPT
        {
            using _this_t = ClContextBase;

            SIXTRL_ASSERT( kernel_id >= kernel_id_t{ 0 } );
            SIXTRL_ASSERT( static_cast< _this_t::size_type >( kernel_id ) <
                           this->numAvailableKernels() );

            this->m_kernel_data[ kernel_id ].setKernelArg(
                _this_t::ARG_TYPE_VALUE, arg_index, nullptr );

            cl::Kernel* kernel = this->openClKernel( kernel_id );
            if( kernel != nullptr ) kernel->setArg( arg_index, ref );
        }

        void assignKernelArgumentClBuffer(
            kernel_id_t const kernel_id, size_type const arg_index,
            cl::Buffer& SIXTRL_RESTRICT_REF cl_buffer_arg );

        size_type calculateKernelNumWorkItems(
            kernel_id_t const kernel_id,
            size_type const min_num_work_items ) const SIXTRL_NOEXCEPT;

        bool runKernel( kernel_id_t const kernel_id, size_type min_num_work_items );
        bool runKernel( kernel_id_t const kernel_id,
                        size_type min_num_work_items, size_type work_group_size );

        double lastExecTime( kernel_id_t const kernel_id ) const SIXTRL_NOEXCEPT;
        double minExecTime(  kernel_id_t const kernel_id ) const SIXTRL_NOEXCEPT;
        double maxExecTime(  kernel_id_t const kernel_id ) const SIXTRL_NOEXCEPT;
        double avgExecTime(  kernel_id_t const kernel_id ) const SIXTRL_NOEXCEPT;

        size_type lastExecWorkGroupSize(
            kernel_id_t const kernel_id ) const SIXTRL_NOEXCEPT;

        size_type lastExecNumWorkItems(
            kernel_id_t const kernel_id ) const SIXTRL_NOEXCEPT;

        void resetKernelExecTiming( kernel_id_t const kernel_id ) SIXTRL_NOEXCEPT;

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

        cl::Buffer const& internalSuccessFlagBuffer() const SIXTRL_NOEXCEPT;
        cl::Buffer& internalSuccessFlagBuffer() SIXTRL_NOEXCEPT;

        bool debugMode() const  SIXTRL_NOEXCEPT;
        void enableDebugMode()  SIXTRL_NOEXCEPT;
        void disableDebugMode() SIXTRL_NOEXCEPT;

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
            using arg_type_t = kernel_arg_type_t;

            static const arg_type_t
                ARG_TYPE_CL_ARGUMENT = arg_type_t{ 0x00000010 };

            static const arg_type_t
                ARG_TYPE_CL_BUFFER   = arg_type_t{ 0x00000020 };


            KernelData() :
                m_kernel_name(),
                m_program_id( -1 ),
                m_num_args( -1 ),
                m_work_group_size( size_type{ 0 } ),
                m_max_work_group_size( size_type{ 0 } ),
                m_preferred_work_group_multiple( size_type{ 0 } ),
                m_local_mem_size( size_type{ 0 } ),
                m_exec_count( size_type{ 0 } ),
                m_last_work_group_size( size_type{ 0 } ),
                m_last_num_of_threads( size_type{ 0 } ),
                m_min_exec_time(  std::numeric_limits< double >::max() ),
                m_max_exec_time(  std::numeric_limits< double >::min() ),
                m_last_exec_time( double{ 0 } ),
                m_sum_exec_time( double{ 0 } ),
                m_arguments(),
                m_arg_types()
            {

            }

            KernelData( KernelData const& orig ) = default;
            KernelData( KernelData&& orig ) = default;

            KernelData& operator=( KernelData const& rhs ) = default;
            KernelData& operator=( KernelData&& rhs ) = default;

            ~KernelData() = default;

            void resetArguments( size_type const nargs )
            {
                using _this_t = ClContextBase;

                this->m_arguments.clear();
                this->m_arg_types.clear();

                if( nargs > size_type{ 0 }  )
                {
                    this->m_arguments.resize( nargs, nullptr );
                    this->m_arg_types.resize( nargs, _this_t::ARG_TYPE_NONE );
                }

                this->m_num_args = nargs;
            }

            void setKernelArg( arg_type_t const type, size_type const index,
                               void* SIXTRL_RESTRICT ptr = nullptr )
            {
                SIXTRL_ASSERT( this->m_arguments.size() ==
                               this->m_arg_types.size() );

                if( index < this->m_arguments.size() )
                {
                    this->m_arg_types[ index ] = type;
                    this->m_arguments[ index ] =
                        reinterpret_cast< ClArgument* >( ptr );
                }
            }

            void resetTiming() SIXTRL_NOEXCEPT
            {
                this->m_min_exec_time  = std::numeric_limits< double >::max();
                this->m_max_exec_time  = std::numeric_limits< double >::min();
                this->m_last_exec_time = double{ 0 };
                this->m_sum_exec_time  = double{ 0 };
                this->m_exec_count     = size_type{ 0 };

                return;
            }

            void addExecTime( double const exec_time )
            {
                if( exec_time >= double{ 0 } )
                {
                    this->m_last_exec_time = exec_time;
                    this->m_sum_exec_time += exec_time;
                    ++this->m_exec_count;

                    if( this->m_min_exec_time > exec_time )
                        this->m_min_exec_time = exec_time;

                    if( this->m_max_exec_time < exec_time )
                        this->m_max_exec_time = exec_time;
                }

                return;
            }

            double avgExecTime() const
            {
                return ( this->m_exec_count > size_type{ 0 } )
                    ? this->m_sum_exec_time /
                      static_cast< double >( this->m_exec_count ) : double{ 0 };
            }

            ClArgument const* argument(
                size_type const arg_index ) const SIXTRL_NOEXCEPT
            {
                SIXTRL_ASSERT( arg_index < this->m_num_args );
                SIXTRL_ASSERT( this->m_arguments.size() > arg_index );
                return this->m_arguments[ arg_index ];
            }

            ClArgument* argument( size_type const arg_index ) SIXTRL_NOEXCEPT
            {
                SIXTRL_ASSERT( arg_index < this->m_num_args );
                SIXTRL_ASSERT( this->m_arguments.size() > arg_index );
                return this->m_arguments[ arg_index ];
            }

            void assignArgument( size_type const arg_index,
                                 ClArgument* ptr_to_arg )
            {
                SIXTRL_ASSERT( arg_index < this->m_num_args );
                SIXTRL_ASSERT( this->m_arguments.size() > arg_index );
                SIXTRL_ASSERT( this->m_arg_types.size() ==
                               this->m_arguments.size() );

                this->m_arguments[ arg_index ] = ptr_to_arg;
            }

            void setArgumentType( size_type const arg_index,
                                  kernel_arg_type_t const type )
            {
                SIXTRL_ASSERT( arg_index < this->m_num_args );
                SIXTRL_ASSERT( this->m_arg_types.size() > arg_index );
                SIXTRL_ASSERT( this->m_arg_types.size() ==
                               this->m_arguments.size() );

                this->m_arg_types[ arg_index ] = type;
            }

            std::string   m_kernel_name;
            program_id_t  m_program_id;
            size_type     m_num_args;
            size_type     m_work_group_size;
            size_type     m_max_work_group_size;
            size_type     m_preferred_work_group_multiple;
            size_type     m_local_mem_size;
            size_type     m_exec_count;

            size_type     m_last_work_group_size;
            size_type     m_last_num_of_threads;

            double        m_min_exec_time;
            double        m_max_exec_time;
            double        m_last_exec_time;
            double        m_sum_exec_time;

            std::vector< ClArgument* >       m_arguments;
            std::vector< kernel_arg_type_t > m_arg_types;
        };

        using program_data_list_t = std::vector< program_data_t >;
        using kernel_data_list_t  = std::vector< kernel_data_t >;

        virtual void doParseConfigString(
            const char *const SIXTRL_RESTRICT config_str );

        virtual void doClear();

        virtual bool doInitDefaultPrograms();
        virtual bool doInitDefaultKernels();

        virtual bool doCompileProgram(
            cl::Program& cl_program, program_data_t& program_data );

        virtual bool doSelectNode( size_type node_index );

        void doSetConfigStr( const char *const SIXTRL_RESTRICT config_str );

        void addKernelExecTime(
            double const time, kernel_id_t const kernel_id ) SIXTRL_NOEXCEPT;

        void setLastWorkGroupSize( size_type work_group_size,
            kernel_id_t const kernel_id ) SIXTRL_NOEXCEPT;

        void setLastNumWorkItems( size_type num_work_items,
            kernel_id_t const kernel_id ) SIXTRL_NOEXCEPT;

        kernel_data_list_t  const& kernelData()  const SIXTRL_NOEXCEPT;
        program_data_list_t const& programData() const SIXTRL_NOEXCEPT;

        size_type findAvailableNodesIndex( platform_id_t const platform_index,
            device_id_t const device_index ) const SIXTRL_NOEXCEPT;

        size_type findAvailableNodesIndex( char const* node_id_str
            ) const SIXTRL_NOEXCEPT;

//         int32_t handleKernelSuccessFlag();

        private:

        static void UpdateAvailableNodes(
            std::vector< node_id_t>& available_nodes_id,
            std::vector< node_info_t >&  available_nodes_info,
            std::vector< cl::Device  >&  available_devices,
            const char *const filter_str = nullptr );

        void doParseConfigStringBaseImpl(
            const char *const SIXTRL_RESTRICT config_str );

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
        std::string                     m_config_str;

        cl::Context                     m_cl_context;
        cl::CommandQueue                m_cl_queue;
        cl::Buffer                      m_cl_success_flag;

        program_id_t                    m_remap_program_id;
        kernel_id_t                     m_remap_kernel_id;
        int64_t                         m_selected_node_index;
        uint64_t                        m_default_kernel_arg;

        bool                            m_debug_mode;
    };
}

using NS(ClContextBase) = SIXTRL_CXX_NAMESPACE::ClContextBase;

using NS(kernel_arg_type_t) =
    SIXTRL_CXX_NAMESPACE::ClContextBase::kernel_arg_type_t;

#else /* defined( __cplusplus ) */

    #if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
        #include <CL/cl.h>
    #endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

typedef void NS(ClContextBase);
typedef uint32_t NS(kernel_arg_type_t);

#endif /* defined( __cplusplus ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/opencl/argument.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined( _GPUCODE ) && defined( __cplusplus ) */

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

SIXTRL_HOST_FN NS(context_size_t) NS(ClContextBase_get_selected_node_index)(
    const NS(ClContextBase) *const SIXTRL_RESTRICT ctx );

SIXTRL_HOST_FN bool NS(ClContextBase_get_selected_node_id_str)(
    const NS(ClContextBase) *const SIXTRL_RESTRICT ctx,
    char* SIXTRL_RESTRICT node_id_str,
    NS(context_size_t) const max_str_length );

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

SIXTRL_HOST_FN NS(context_size_t) NS(ClContextBase_get_kernel_max_work_group_size)(
    const NS(ClContextBase) *const SIXTRL_RESTRICT ctx, int const kernel_id );

SIXTRL_HOST_FN bool NS(ClContextBase_set_kernel_work_group_size)(
    NS(ClContextBase)* SIXTRL_RESTRICT context,
    int const kernel_id, NS(context_size_t) const work_group_size );

SIXTRL_HOST_FN NS(context_size_t)
NS(ClContextBase_get_kernel_preferred_work_group_size_multiple)(
    const NS(ClContextBase) *const SIXTRL_RESTRICT ctx, int const kernel_id );

SIXTRL_HOST_FN NS(context_size_t) NS(ClContextBase_get_kernel_exec_counter)(
    const NS(ClContextBase) *const SIXTRL_RESTRICT ctx, int const kernel_id );

SIXTRL_HOST_FN NS(ClArgument)* NS(ClContextBase_get_ptr_kernel_argument)(
    NS(ClContextBase)* SIXTRL_RESTRICT ctx, int const kernel_id,
    NS(context_size_t) const arg_index ) SIXTRL_NOEXCEPT;

SIXTRL_HOST_FN NS(kernel_arg_type_t)
NS(ClContextBase_get_kernel_argument_type)(
    NS(ClContextBase)* SIXTRL_RESTRICT ctx, int const kernel_id,
    NS(context_size_t) const arg_index ) SIXTRL_NOEXCEPT;

SIXTRL_HOST_FN NS(ClArgument) const*
NS(ClContextBase_get_const_ptr_kernel_argument)(
    const NS(ClContextBase) *const SIXTRL_RESTRICT ctx, int const kernel_id,
    NS(context_size_t) const arg_index ) SIXTRL_NOEXCEPT;

SIXTRL_HOST_FN void NS(ClContextBase_assign_kernel_argument)(
    NS(ClContextBase)* SIXTRL_RESTRICT ctx, int const kernel_id,
    NS(context_size_t) const arg_index, NS(ClArgument)* SIXTRL_RESTRICT arg );

SIXTRL_HOST_FN void NS(ClContextBase_reset_single_kernel_argument)(
    NS(ClContextBase)* SIXTRL_RESTRICT ctx, int const kernel_id,
    NS(context_size_t) const arg_index ) SIXTRL_NOEXCEPT;

SIXTRL_HOST_FN void NS(ClContextBase_assign_kernel_argument_ptr)(
    NS(ClContextBase)* SIXTRL_RESTRICT ctx, int const kernel_id,
    NS(context_size_t) const arg_index,
    void* SIXTRL_RESTRICT ptr ) SIXTRL_NOEXCEPT;

SIXTRL_HOST_FN void NS(ClContextBase_assign_kernel_argument_value)(
    NS(ClContextBase)* SIXTRL_RESTRICT ctx, int const kernel_id,
    NS(context_size_t) const arg_index, void* SIXTRL_RESTRICT arg_data,
    NS(context_size_t) const arg_data_size ) SIXTRL_NOEXCEPT;

SIXTRL_HOST_FN void NS(ClContextBase_reset_kernel_arguments)(
    NS(ClContextBase)* SIXTRL_RESTRICT ctx,
    int const kernel_id ) SIXTRL_NOEXCEPT;

SIXTRL_HOST_FN NS(context_size_t)
NS(ClContextBase_calculate_kernel_num_work_items)(
    const NS(ClContextBase) *const SIXTRL_RESTRICT ctx,
    int const kernel_id, NS(context_size_t) const min_num_work_items );

SIXTRL_HOST_FN bool NS(ClContextBase_run_kernel)(
    NS(ClContextBase)* SIXTRL_RESTRICT ctx,
    int const kernel_id, NS(context_size_t) num_work_items );

SIXTRL_HOST_FN bool NS(ClContextBase_run_kernel_wgsize)(
    NS(ClContextBase)* SIXTRL_RESTRICT ctx, int const kernel_id,
    NS(context_size_t) const num_work_items,
    NS(context_size_t) const work_group_size );

SIXTRL_HOST_FN double NS(ClContextBase_get_last_exec_time)(
    const NS(ClContextBase) *const SIXTRL_RESTRICT ctx, int const kernel_id );

SIXTRL_HOST_FN double NS(ClContextBase_get_min_exec_time)(
    const NS(ClContextBase) *const SIXTRL_RESTRICT ctx, int const kernel_id );

SIXTRL_HOST_FN double NS(ClContextBase_get_max_exec_time)(
    const NS(ClContextBase) *const SIXTRL_RESTRICT ctx, int const kernel_id );

SIXTRL_HOST_FN double NS(ClContextBase_get_avg_exec_time)(
    const NS(ClContextBase) *const SIXTRL_RESTRICT ctx, int const kernel_id );

SIXTRL_HOST_FN NS(context_size_t)
NS(ClContextBase_get_last_exec_work_group_size)(
    const NS(ClContextBase) *const SIXTRL_RESTRICT ctx, int const kernel_id );

SIXTRL_HOST_FN NS(context_size_t)
NS(ClContextBase_get_last_exec_num_work_items)(
    const NS(ClContextBase) *const SIXTRL_RESTRICT ctx, int const kernel_id );

SIXTRL_HOST_FN void NS(ClContextBase_reset_kernel_exec_timing)(
    NS(ClContextBase)* SIXTRL_RESTRICT ctx, int const kernel_id );

SIXTRL_HOST_FN int  NS(ClContextBase_get_program_id_by_kernel_id)(
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

SIXTRL_HOST_FN cl_mem NS(ClContextBase_get_internal_opencl_success_flag_buffer)(
    NS(ClContextBase)* SIXTRL_RESTRICT ctx );

SIXTRL_HOST_FN bool NS(ClContextBase_is_debug_mode_enabled)(
    const NS(ClContextBase) *const SIXTRL_RESTRICT ctx );

SIXTRL_HOST_FN void NS(ClContextBase_enable_debug_mode)(
    NS(ClContextBase)* SIXTRL_RESTRICT ctx );

SIXTRL_HOST_FN void NS(ClContextBase_disable_debug_mode)(
    NS(ClContextBase)* SIXTRL_RESTRICT ctx );

#if !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined( _GPUCODE ) && defined( __cplusplus ) */

#endif /* !defined( __CUDACC__ ) */

#endif /* SIXTRACKLIB_OPENCL_INTERNAL_BASE_CONTEXT_H__ */

/* end: sixtracklib/opencl/internal/base_context.h */
