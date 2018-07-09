#ifndef SIXTRACKLIB_OPENCL_CL_ENVIRONMENT_H__
#define SIXTRACKLIB_OPENCL_CL_ENVIRONMENT_H__

#include "sixtracklib/_impl/definitions.h"
#include "sixtracklib/common/compute_arch.h"

#if defined( __cplusplus )

#include <map>
#include <string>
#include <vector>
#include <utility>

#include <CL/cl.hpp>

struct NS(Blocks);

class NS(OclEnvironment)
{
    public:

    typedef NS(ComputeNodeId)   node_id_t;
    typedef NS(ComputeNodeInfo) node_info_t;

    NS(OclEnvironment)();

    NS(OclEnvironment)( NS(OclEnvironment) const& other ) = delete;
    NS(OclEnvironment)( NS(OclEnvironment)&& other ) = delete;

    NS(OclEnvironment)& operator=( NS(OclEnvironment) const& rhs ) = delete;
    NS(OclEnvironment)& operator=( NS(OclEnvironment)&& rhs ) = delete;

    virtual ~NS(OclEnvironment)() noexcept;

    node_id_t const* constAvailableNodesBegin() const noexcept;
    node_id_t const* constAvailableNodesEnd()   const noexcept;
    std::size_t numAvailableNodes() const noexcept;

    node_info_t const* getPtrNodeInfo( node_id_t const id ) const noexcept;

    bool prepareParticlesTracking(
        struct NS(Blocks) const& SIXTRL_RESTRICT_REF particles_buffer,
        struct NS(Blocks) const& SIXTRL_RESTRICT_REF beam_elements,
        struct NS(Blocks) const* SIXTRL_RESTRICT elem_by_elem_buffer,
        NS(ComputeNodeId) const* SIXTRL_RESTRICT selected_nodes_begin,
        std::size_t const num_of_selected_nodes );

    bool runParticlesTracking(
        uint64_t const num_of_turns,
        struct NS(Blocks)& SIXTRL_RESTRICT_REF particles_buffer,
        struct NS(Blocks)& SIXTRL_RESTRICT_REF beam_elements,
        struct NS(Blocks)* SIXTRL_RESTRICT elem_by_elem_buffer );


    private:

    using ocl_platform_dev_pair_t =
        std::pair< cl::Platform, cl::Device >;

    using ocl_platform_dev_map_t  =
        std::map< node_id_t, ocl_platform_dev_pair_t >;

    using ocl_node_id_to_node_info_map_t = std::map< node_id_t, node_info_t >;

    ocl_platform_dev_map_t                    m_ocl_platform_devices;
    ocl_node_id_to_node_info_map_t            m_node_id_to_info_map;

    std::vector< node_id_t >                  m_available_nodes;
    std::vector< node_id_t >                  m_selected_nodes;

    std::vector< cl::Context >                m_contexts;
    std::vector< cl::Program >                m_programs;
    std::vector< cl::CommandQueue >           m_queues;
    std::vector< std::vector< cl::Kernel > >  m_kernels;
    std::vector< std::vector< cl::Buffer > >  m_buffers;
    std::vector< int64_t >                    m_success_flags;

    std::vector< unsigned int >               m_preferred_work_group_multi;
    std::vector< unsigned int >               m_local_workgroup_size;

    std::size_t                               m_particles_data_buffer_size;
    std::size_t                               m_beam_elements_data_buffer_size;
    std::size_t                               m_elem_by_elem_data_buffer_size;
    uint64_t                                  m_num_of_particles;

    std::size_t                               m_num_of_platforms;
};

#else

#include <CL/cl.h>

typedef void NS(OclEnvironment);

#endif /* defined( __cplusplus ) */

#if defined( __cplusplus )

extern "C"
{

#endif /* !defined( __cplusplus ) */

NS(OclEnvironment)* NS(OclEnvironment_new)();
void NS(OclEnvironment_free)( NS(OclEnvironment)* SIXTRL_RESTRICT ocl_env );

SIXTRL_HOST_FN NS(ComputeNodeId) const*
NS(OclEnvironment_get_available_nodes_begin)(
    const NS(OclEnvironment) *const SIXTRL_RESTRICT ocl_env );

SIXTRL_HOST_FN NS(ComputeNodeId) const*
NS(OclEnvironment_get_available_nodes_end)(
    const NS(OclEnvironment) *const SIXTRL_RESTRICT ocl_env );

SIXTRL_HOST_FN SIXTRL_SIZE_T NS(OclEnvironment_get_num_available_nodes)(
    const NS(OclEnvironment) *const SIXTRL_RESTRICT ocl_env );

SIXTRL_HOST_FN NS(ComputeNodeInfo) const*
NS(OclEnvironment_get_ptr_node_info)(
    const NS(OclEnvironment) *const SIXTRL_RESTRICT ocl_env,
    NS(ComputeNodeId) const* SIXTRL_RESTRICT  node_id );

SIXTRL_HOST_FN int NS(OclEnvironment_prepare_particles_tracking)(
    NS(OclEnvironment)* SIXTRL_RESTRICT ocl_env,
    const struct NS(Blocks) *const SIXTRL_RESTRICT particles_buffer,
    const struct NS(Blocks) *const SIXTRL_RESTRICT beam_elements,
    const struct NS(Blocks) *const SIXTRL_RESTRICT elem_by_elem_buffer,
    NS(ComputeNodeId) const* selected_nodes_begin,
    size_t const num_of_selected_nodes );

SIXTRL_HOST_FN int NS(OclEnvironment_run_particle_tracking)(
    NS(OclEnvironment)* SIXTRL_RESTRICT ocl_env,
    uint64_t const num_of_turns,
    struct NS(Blocks)* SIXTRL_RESTRICT particles_buffer,
    struct NS(Blocks)* SIXTRL_RESTRICT beam_elements,
    struct NS(Blocks)* SIXTRL_RESTRICT elem_by_elem_buffer );

#if defined( __cplusplus )
}
#endif /* defined( __cplusplus ) */

#endif /* SIXTRACKLIB_OPENCL_CL_ENVIRONMENT_H__ */

/* end: sixtracklib/opencl/ocl_environment.h */
