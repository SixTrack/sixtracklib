#ifndef SIXTRACKLIB_OPENCL_CONTEXT_H__
#define SIXTRACKLIB_OPENCL_CONTEXT_H__

#if !defined( __CUDACC__ )

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stdbool.h>
    #include <stddef.h>
    #include <stdint.h>
    #include <stdlib.h>
    #include <limits.h>

    #if defined( __cplusplus )
        #include <algorithm>
    #endif /* !defined( __cplusplus ) */

#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/control/definitions.h"
    #include "sixtracklib/common/control/debug_register.h"
    #include "sixtracklib/common/output/elem_by_elem_config.h"
    #include "sixtracklib/common/output/output_buffer.h"
    #include "sixtracklib/common/track/definitions.h"
    #include "sixtracklib/common/buffer.h"
    #include "sixtracklib/common/particles.h"

    #include "sixtracklib/opencl/cl.h"
    #include "sixtracklib/opencl/internal/base_context.h"
    #include "sixtracklib/opencl/argument.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus )

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/buffer.hpp"
    #include "sixtracklib/common/particles.hpp"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

namespace SIXTRL_CXX_NAMESPACE
{
    class ClContext : public ClContextBase
    {
        private:

        using _base_context_t   = SIXTRL_CXX_NAMESPACE::ClContextBase;

        public:

        using  buffer_t              = SIXTRL_CXX_NAMESPACE::Buffer;
        using  c_buffer_t            = buffer_t::c_api_t;
        using  particles_t           = SIXTRL_CXX_NAMESPACE::Particles;
        using  c_particles_t         = particles_t::c_api_t;
        using  particle_index_t      = particles_t::index_t;
        using  track_status_t        = SIXTRL_CXX_NAMESPACE::track_status_t;
        using  elem_by_elem_config_t = ::NS(ElemByElemConfig);
        using  num_turns_t           = ::NS(particle_index_t);

        static constexpr size_type MIN_NUM_TRACK_UNTIL_ARGS   = size_type{ 5 };
        static constexpr size_type MIN_NUM_TRACK_LINE_ARGS    = size_type{ 7 };
        static constexpr size_type MIN_NUM_TRACK_ELEM_ARGS    = size_type{ 6 };
        static constexpr size_type MIN_NUM_ASSIGN_BE_MON_ARGS = size_type{ 5 };
        static constexpr size_type MIN_NUM_CLEAR_BE_MON_ARGS  = size_type{ 2 };
        static constexpr size_type MIN_NUM_ASSIGN_ELEM_ARGS   = size_type{ 4 };

        explicit ClContext(
            const char *const SIXTRL_RESTRICT config_str = nullptr );

        explicit ClContext( size_type const node_index,
            const char *const SIXTRL_RESTRICT config_str = nullptr );

        explicit ClContext( node_id_t const node_id,
            const char *const SIXTRL_RESTRICT config_str = nullptr );

        explicit ClContext( char const* node_id_str,
            const char *const SIXTRL_RESTRICT config_str = nullptr );

        ClContext( platform_id_t const platform_idx,
                   device_id_t const device_idx,
                   const char *const SIXTRL_RESTRICT config_str = nullptr );

        ClContext( ClContext const& other ) = delete;
        ClContext( ClContext&& other ) = delete;

        ClContext& operator=( ClContext const& other ) = delete;
        ClContext& operator=( ClContext&& other ) = delete;

        virtual ~ClContext() SIXTRL_NOEXCEPT;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        status_t assign_particles_arg(
            ClArgument& SIXTRL_RESTRICT_REF particles_arg );

        status_t assign_particle_set_arg(
            size_type const particle_set_index,
            size_type const num_particles_in_set );

        status_t assign_beam_elements_arg(
            ClArgument& SIXTRL_RESTRICT_REF beam_elements_arg );

        status_t assign_output_buffer_arg(
            ClArgument& SIXTRL_RESTRICT_REF output_buffer_arg );

        status_t assign_elem_by_elem_config_arg(
            cl_buffer_t& SIXTRL_RESTRICT_REF elem_by_elem_config_arg );

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        bool has_track_until_kernel() const SIXTRL_NOEXCEPT;
        kernel_id_t track_until_kernel_id() const SIXTRL_NOEXCEPT;
        status_t set_track_until_kernel_id( kernel_id_t const kernel_id );

        track_status_t track_until( num_turns_t const until_turn );

        track_status_t track_until( num_turns_t const until_turn,
            size_type const pset_index, size_type const nparticles_in_set,
            bool const restore_pset_index = true );

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        bool has_track_line_kernel() const SIXTRL_NOEXCEPT;
        kernel_id_t track_line_kernel_id() const SIXTRL_NOEXCEPT;
        status_t set_track_line_kernel_id( kernel_id_t const kernel_id );

        track_status_t track_line( size_type const line_begin_idx,
                       size_type const line_end_idx, bool const finish_turn );

        track_status_t track_line( size_type const line_begin_idx,
            size_type const line_end_idx, bool const finish_turn,
            size_type const pset_index, size_type const nparticles_in_set,
            bool const restore_pset_index = true );

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        bool has_track_elem_by_elem_kernel() const SIXTRL_NOEXCEPT;
        kernel_id_t track_elem_by_elem_kernel_id() const SIXTRL_NOEXCEPT;
        status_t set_track_elem_by_elem_kernel_id( kernel_id_t const kernel_id );

        track_status_t track_elem_by_elem( size_type until_turn );

        track_status_t track_elem_by_elem( size_type until_turn,
            size_type const pset_index, size_type const nparticles_in_set,
            bool const restore_pset_index = true );

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        bool has_assign_beam_monitor_output_kernel() const SIXTRL_NOEXCEPT;

        kernel_id_t assign_beam_monitor_output_kernel_id() const SIXTRL_NOEXCEPT;
        status_t set_assign_beam_monitor_output_kernel_id(
            kernel_id_t const kernel_id );

        status_t assign_beam_monitor_output(
            particle_index_t const min_turn_id,
            size_type const out_buffer_index_offset );

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        bool has_assign_elem_by_elem_output_kernel() const SIXTRL_NOEXCEPT;
        kernel_id_t assign_elem_by_elem_output_kernel_id() const SIXTRL_NOEXCEPT;
        status_t set_assign_elem_by_elem_output_kernel_id(
            kernel_id_t const kernel_id );

        status_t assign_elem_by_elem_output(
            size_type const out_buffer_index_offset );

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        bool has_clear_beam_monitor_output_kernel() const SIXTRL_NOEXCEPT;
        kernel_id_t clear_beam_monitor_output_kernel_id() const SIXTRL_NOEXCEPT;
        status_t set_clear_beam_monitor_output_kernel_id(
            kernel_id_t const kernel_id );

        status_t clear_beam_monitor_output();

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        size_type selected_particle_set() const SIXTRL_NOEXCEPT;
        size_type num_particles_in_selected_set() const SIXTRL_NOEXCEPT;

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        bool use_optimized_tracking() const SIXTRL_NOEXCEPT;
        void enable_optimized_tracking();
        void disable_optimized_tracking();

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        bool is_beam_beam_tracking_enabled() const SIXTRL_NOEXCEPT;
        void enable_beam_beam_tracking();
        void disable_beam_beam_tracking();

        /* ----------------------------------------------------------------- */

        std::unique_ptr< SIXTRL_CXX_NAMESPACE::ClContext::cl_buffer_t >
        create_elem_by_elem_config_arg();

        void delete_elem_by_elem_config_arg(
            std::unique_ptr< SIXTRL_CXX_NAMESPACE::ClContext::cl_buffer_t >&& );

        status_t init_elem_by_elem_config_arg(
            cl_buffer_t& SIXTRL_RESTRICT_REF elem_by_elem_config_arg,
            elem_by_elem_config_t& SIXTRL_RESTRICT_REF elem_by_elem_config,
            const ::NS(Buffer) *const SIXTRL_RESTRICT particles_buffer,
            size_type const num_particle_sets,
            size_type const* SIXTRL_RESTRICT pset_indices_begin,
            const ::NS(Buffer) *const SIXTRL_RESTRICT beam_elements_buffer,
            size_type const until_turn_elem_by_elem,
            particle_index_t const start_elem_id = particle_index_t{ 0 } );

        status_t collect_elem_by_elem_config_arg(
            cl_buffer_t& SIXTRL_RESTRICT_REF elem_by_elem_config_arg,
            elem_by_elem_config_t& SIXTRL_RESTRICT_REF elem_by_elem_config );

        status_t push_elem_by_elem_config_arg(
            cl_buffer_t& SIXTRL_RESTRICT_REF elem_by_elem_config_arg,
            elem_by_elem_config_t const& SIXTRL_RESTRICT_REF
                elem_by_elem_config );

        protected:

        bool doSelectNode( size_type node_index ) override;
        bool doInitDefaultPrograms() override;
        bool doInitDefaultKernels()  override;

        status_t doAssignStatusFlagsArg(
            cl::Buffer& SIXTRL_RESTRICT_REF status_flags_arg ) override;

        status_t doAssignSlotSizeArg( size_type const slot_size ) override;

        private:

        bool doInitDefaultProgramsPrivImpl();
        bool doInitDefaultKernelsPrivImpl();

        status_t doAssignStatusFlagsArgPrivImpl(
            cl::Buffer& SIXTRL_RESTRICT_REF status_flags_arg );

        status_t doAssignSlotSizeArgPrivImpl(
            size_type const slot_size );

        size_type    m_num_particles_in_pset;
        size_type    m_pset_index;

        program_id_t m_track_until_turn_program_id;
        program_id_t m_track_elem_by_elem_program_id;
        program_id_t m_track_line_program_id;
        program_id_t m_assign_elem_by_elem_out_buffer_program_id;
        program_id_t m_assign_be_mon_out_buffer_program_id;
        program_id_t m_clear_be_mon_program_id;

        kernel_id_t  m_track_until_turn_kernel_id;
        kernel_id_t  m_track_elem_by_elem_kernel_id;
        kernel_id_t  m_track_line_kernel_id;
        kernel_id_t  m_assign_elem_by_elem_out_buffer_kernel_id;
        kernel_id_t  m_assign_be_mon_out_buffer_kernel_id;
        kernel_id_t  m_clear_be_mon_kernel_id;

        bool         m_use_optimized_tracking;
        bool         m_enable_beam_beam;
    };
}

#if !defined( _GPUCODE )
extern "C" {
#endif /* !defined( _GPUCODE ) */

typedef SIXTRL_CXX_NAMESPACE::ClContext  NS(ClContext);
typedef int64_t                          NS(context_num_turns_t);

#if !defined( _GPUCODE )
}
#endif /* !defined( _GPUCODE ) */

#else /* !defined( __cplusplus ) */

typedef void NS(ClContext);
typedef SIXTRL_INT64_T NS(context_num_turns_t);

#endif /* defined( __cplusplus ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined( _GPUCODE ) && defined( __cplusplus ) */

SIXTRL_EXTERN SIXTRL_HOST_FN NS(ClContext)* NS(ClContext_create)();

SIXTRL_EXTERN SIXTRL_HOST_FN NS(ClContext)* NS(ClContext_new)(
    const char* node_id_str );

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(ClContext_delete)(
    NS(ClContext)* SIXTRL_RESTRICT ctx );

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(ClContext_clear)(
    NS(ClContext)* SIXTRL_RESTRICT ctx );

/* ========================================================================= */

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t)
NS(ClContext_assign_particles_arg)( NS(ClContext)* SIXTRL_RESTRICT ctx,
    NS(ClArgument)* SIXTRL_RESTRICT particles_arg );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t)
NS(ClContext_assign_particle_set_arg)(  NS(ClContext)* SIXTRL_RESTRICT ctx,
    NS(buffer_size_t) const particle_set_index,
    NS(buffer_size_t) const num_particles_in_selected_set );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t)
NS(ClContext_assign_beam_elements_arg)( NS(ClContext)* SIXTRL_RESTRICT ctx,
    NS(ClArgument)* SIXTRL_RESTRICT beam_elem_arg );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t)
NS(ClContext_assign_output_buffer_arg)( NS(ClContext)* SIXTRL_RESTRICT ctx,
    NS(ClArgument)* SIXTRL_RESTRICT_REF out_buffer_arg );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t)
NS(ClContext_assign_elem_by_elem_config_arg)(
    NS(ClContext)* SIXTRL_RESTRICT ctx, cl_mem elem_by_elem_config_arg  );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t)
NS(ClContext_assign_slot_size_arg)(
    NS(ClContext)* SIXTRL_RESTRICT ctx, NS(buffer_size_t) const slot_size );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t)
NS(ClContext_assign_status_flags_arg)( NS(ClContext)* SIXTRL_RESTRICT ctx,
    cl_mem status_flags_arg );

/* ========================================================================= */

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(ClContext_has_track_until_kernel)(
    const NS(ClContext) *const SIXTRL_RESTRICT ctx );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_kernel_id_t)
NS(ClContext_track_until_kernel_id)(
    const NS(ClContext) *const SIXTRL_RESTRICT ctx );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t)
NS(ClContext_set_track_until_kernel_id)(
    NS(ClContext)* SIXTRL_RESTRICT ctx, NS(arch_kernel_id_t) const kernel_id );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(track_status_t) NS(ClContext_track_until)(
    NS(ClContext)* SIXTRL_RESTRICT ctx,
    NS(context_num_turns_t) const until_turn );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(track_status_t)
NS(ClContext_track_until_for_particle_set)(
    NS(ClContext)* SIXTRL_RESTRICT ctx,
    NS(context_num_turns_t) const until_turn,
    NS(buffer_size_t) const particle_set_index,
    NS(buffer_size_t) const num_particles_in_set,
    bool const restore_particle_set_index );

/* ------------------------------------------------------------------------- */

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(ClContext_has_track_line_kernel)(
    const NS(ClContext) *const SIXTRL_RESTRICT ctx );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_kernel_id_t)
NS(ClContext_track_line_kernel_id)(
    const NS(ClContext) *const SIXTRL_RESTRICT ctx );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t)
NS(ClContext_set_track_line_kernel_id)(
    NS(ClContext)* SIXTRL_RESTRICT ctx, NS(arch_kernel_id_t) const kernel_id );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(track_status_t) NS(ClContext_track_line)(
    NS(ClContext)* SIXTRL_RESTRICT ctx,
    NS(buffer_size_t) const line_begin_idx,
    NS(buffer_size_t) const line_end_idx, bool const finish_turn );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(track_status_t)
NS(ClContext_track_line_for_particle_set)(
    NS(ClContext)* SIXTRL_RESTRICT ctx,
    NS(buffer_size_t) const line_begin_idx,
    NS(buffer_size_t) const line_end_idx, bool const finish_turn,
    NS(buffer_size_t) const particle_set_index,
    NS(buffer_size_t) const num_particles_in_set,
    bool const restore_particle_set_index );

/* ------------------------------------------------------------------------- */

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(ClContext_has_track_elem_by_elem_kernel)(
    const NS(ClContext) *const SIXTRL_RESTRICT ctx );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_kernel_id_t)
NS(ClContext_track_elem_by_elem_kernel_id)(
    const NS(ClContext) *const SIXTRL_RESTRICT ctx );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t)
NS(ClContext_set_track_elem_by_elem_kernel_id)(
    NS(ClContext)* SIXTRL_RESTRICT ctx,
    NS(arch_kernel_id_t) const tracking_kernel_id );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(track_status_t)
NS(ClContext_track_elem_by_elem)(
    NS(ClContext)* SIXTRL_RESTRICT ctx, NS(buffer_size_t) const until_turn );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(track_status_t)
NS(ClContext_track_elem_by_elem_for_particle_set)(
    NS(ClContext)* SIXTRL_RESTRICT ctx, NS(buffer_size_t) const until_turn,
    NS(buffer_size_t) const particle_set_index,
    NS(buffer_size_t) const num_particles_in_set,
    bool const restore_particle_set_index );

/* ------------------------------------------------------------------------- */

SIXTRL_EXTERN SIXTRL_HOST_FN bool
NS(ClContext_has_assign_beam_monitor_output_kernel)(
    const NS(ClContext) *const SIXTRL_RESTRICT ctx );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_kernel_id_t)
NS(ClContext_assign_beam_monitor_output_kernel_id)(
    const NS(ClContext) *const SIXTRL_RESTRICT ctx );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t)
NS(ClContext_set_assign_beam_monitor_output_kernel_id)(
    NS(ClContext)* SIXTRL_RESTRICT ctx,
    NS(arch_kernel_id_t) const kernel_id );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t)
NS(ClContext_assign_beam_monitor_output)(
    NS(ClContext)*  SIXTRL_RESTRICT ctx,
    NS(particle_index_t) const min_turn_id,
    NS(buffer_size_t) const out_buffer_index_offset );

/* ------------------------------------------------------------------------- */

SIXTRL_EXTERN SIXTRL_HOST_FN bool
NS(ClContext_has_assign_elem_by_elem_output_kernel)(
    const NS(ClContext) *const SIXTRL_RESTRICT ctx );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_kernel_id_t)
NS(ClContext_assign_elem_by_elem_output_kernel_id)(
    const NS(ClContext) *const SIXTRL_RESTRICT ctx );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t)
NS(ClContext_set_assign_elem_by_elem_output_kernel_id)(
    NS(ClContext)* SIXTRL_RESTRICT ctx,
    NS(arch_kernel_id_t) const kernel_id );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t)
NS(ClContext_assign_elem_by_elem_output)(
    NS(ClContext)*  SIXTRL_RESTRICT ctx,
    NS(buffer_size_t) const out_buffer_index_offset );

/* ------------------------------------------------------------------------- */

SIXTRL_EXTERN SIXTRL_HOST_FN bool
NS(ClContext_has_clear_beam_monitor_output_kernel)(
    const NS(ClContext) *const SIXTRL_RESTRICT ctx );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_kernel_id_t)
NS(ClContext_clear_beam_monitor_output_kernel_id)(
    const NS(ClContext) *const SIXTRL_RESTRICT ctx );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t)
NS(ClContext_set_clear_beam_monitor_output_kernel_id)(
    NS(ClContext)* SIXTRL_RESTRICT ctx, NS(arch_kernel_id_t) const kernel_id );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t)
NS(ClContext_clear_beam_monitor_output)( NS(ClContext)*  SIXTRL_RESTRICT ctx );

/* ========================================================================= */

SIXTRL_EXTERN SIXTRL_HOST_FN NS(buffer_size_t)
NS(ClContext_selected_particle_set)(
    const NS(ClContext) *const SIXTRL_RESTRICT ctx );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(buffer_size_t)
NS(ClContext_num_particles_in_selected_set)(
    const NS(ClContext) *const SIXTRL_RESTRICT ctx );

/* ------------------------------------------------------------------------- */

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(ClContext_uses_optimized_tracking)(
    const NS(ClContext) *const SIXTRL_RESTRICT ctx );

SIXTRL_EXTERN SIXTRL_HOST_FN void
NS(ClContext_enable_optimized_tracking)( NS(ClContext)* SIXTRL_RESTRICT ctx );

SIXTRL_EXTERN SIXTRL_HOST_FN void
NS(ClContext_disable_optimized_tracking)( NS(ClContext)* SIXTRL_RESTRICT ctx );

/* ------------------------------------------------------------------------- */

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(ClContext_is_beam_beam_tracking_enabled)(
    const NS(ClContext) *const SIXTRL_RESTRICT ctx );

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(ClContext_enable_beam_beam_tracking)(
    NS(ClContext)* SIXTRL_RESTRICT ctx );

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(ClContext_disable_beam_beam_tracking)(
    NS(ClContext)* SIXTRL_RESTRICT ctx );

/* ------------------------------------------------------------------------- */

cl_mem NS(ClContext_create_elem_by_elem_config_arg)(
    NS(ClContext)* SIXTRL_RESTRICT ctx );

void NS(ClContext_delete_elem_by_elem_config_arg)(
    NS(ClContext)* SIXTRL_RESTRICT ctx, cl_mem elem_by_elem_config_arg );

NS(arch_status_t) NS(ClContext_init_elem_by_elem_config_arg)(
    NS(ClContext)* SIXTRL_RESTRICT ctx,
    cl_mem elem_by_elem_config_arg,
    NS(ElemByElemConfig)* SIXTRL_RESTRICT elem_by_elem_config,
    const NS(Buffer) *const SIXTRL_RESTRICT particles_buffer,
    NS(buffer_size_t) const num_particle_sets,
    NS(buffer_size_t) const* SIXTRL_RESTRICT pset_indices_begin,
    const NS(Buffer) *const SIXTRL_RESTRICT beam_elements_buffer,
    NS(buffer_size_t) const until_turn_elem_by_elem,
    NS(particle_index_t) const start_elem_id );

NS(arch_status_t) NS(ClContext_collect_elem_by_elem_config_arg)(
    NS(ClContext)* SIXTRL_RESTRICT ctx,
    cl_mem elem_by_elem_config_arg,
    NS(ElemByElemConfig)* SIXTRL_RESTRICT elem_by_elem_config );

NS(arch_status_t) NS(ClContext_push_elem_by_elem_config_arg)(
    NS(ClContext)* SIXTRL_RESTRICT ctx,
    cl_mem elem_by_elem_config_arg,
    const NS(ElemByElemConfig) *const SIXTRL_RESTRICT elem_by_elem_config );


#if !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined( _GPUCODE ) && defined( __cplusplus ) */

#endif /* !defined( __CUDACC__ ) */

#endif /* SIXTRACKLIB_OPENCL_CONTEXT_H__ */

/* end: sixtracklib/opencl/context.h */
