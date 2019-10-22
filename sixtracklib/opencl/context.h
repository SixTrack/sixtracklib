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
    #include "sixtracklib/common/buffer.h"
    #include "sixtracklib/common/particles.h"

    #include "sixtracklib/opencl/cl.h"
    #include "sixtracklib/opencl/internal/base_context.h"
    #include "sixtracklib/opencl/argument.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus )

namespace SIXTRL_CXX_NAMESPACE
{
    class ClContext : public ClContextBase
    {
        private:

        using _base_context_t   = SIXTRL_CXX_NAMESPACE::ClContextBase;

        public:

        using  num_turns_t      = SIXTRL_INT64_T;
        using  program_id_t     = _base_context_t::program_id_t;
        using  kernel_id_t      = _base_context_t::kernel_id_t;
        using  cl_buffer_t      = cl::Buffer;
        using  status_t         = SIXTRL_CXX_NAMESPACE::ctrl_status_t;
        using  track_status_t   = SIXTRL_CXX_NAMESPACE::track_status_t;
        using  status_flags_t   = SIXTRL_CXX_NAMESPACE::arch_debugging_t;

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

        bool assignParticleArg( ClArgument& SIXTRL_RESTRICT_REF arg );
        bool assignParticleSetIndex( size_type const particle_set_index );

        bool assignBeamElementsArg( ClArgument& SIXTRL_RESTRICT_REF arg );
        bool assignOutputBufferArg( ClArgument& SIXTRL_RESTRICT_REF arg );

        bool assignElemByElemConfigBuffer(
            cl_buffer_t& SIXTRL_RESTRICT_REF cl_elem_by_elem_config_buffer );

        bool assignSlotSize( size_type const slot_size );

        bool assignSuccessFlagBuffer(
            cl_buffer_t& SIXTRL_RESTRICT_REF cl_success_flag_buffer );

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        bool hasTrackUntilKernel() const SIXTRL_NOEXCEPT;
        kernel_id_t trackUntilKernelId() const SIXTRL_NOEXCEPT;
        bool setTrackUntilKernelId( kernel_id_t const kernel_id );

        track_status_t trackUntil( num_turns_t const turn );
        track_status_t trackUntil( num_turns_t const turn, kernel_id_t const track_kernel_id );

        track_status_t trackUntil( ClArgument& SIXTRL_RESTRICT_REF particles_arg,
                        ClArgument& SIXTRL_RESTRICT_REF beam_elements_arg,
                        num_turns_t const turn );

        track_status_t trackUntil( ClArgument& particles_arg, ClArgument& beam_elements_arg,
                        num_turns_t const turn, kernel_id_t const track_kernel_id );

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        bool hasTrackLineKernel() const SIXTRL_NOEXCEPT;
        kernel_id_t trackLineKernelId() const SIXTRL_NOEXCEPT;
        bool setTrackLineKernelId( kernel_id_t const kernel_id );

        track_status_t trackLine( size_type const line_begin_idx,
                       size_type const line_end_idx, bool const finish_turn );

        track_status_t trackLine( size_type const line_begin_idx, size_type line_end_idx,
                       bool const finish_turn, kernel_id_t const kernel_id );

        track_status_t trackLine( ClArgument& SIXTRL_RESTRICT_REF particles_arg,
                       size_type const particle_set_index,
                       ClArgument& SIXTRL_RESTRICT_REF beam_elements_arg,
                       size_type const line_begin_idx,
                       size_type const line_end_idx, bool const finish_turn );

        track_status_t trackLine( ClArgument& SIXTRL_RESTRICT_REF particles_arg,
                       size_type const particle_set_index,
                       ClArgument& SIXTRL_RESTRICT_REF beam_elements_arg,
                       size_type const line_begin_idx,
                       size_type const line_end_idx, bool const finish_turn,
                       kernel_id_t const track_kernel_id );

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        bool hasElementByElementTrackingKernel() const SIXTRL_NOEXCEPT;
        kernel_id_t elementByElementTrackingKernelId() const SIXTRL_NOEXCEPT;

        bool setElementByElementTrackingKernelId(
            kernel_id_t const track_kernel_id );

        track_status_t trackElementByElement(
            size_type until_turn, size_type out_particle_block_offset );

        track_status_t trackElementByElement(
            size_type until_turn, size_type out_particle_block_offset,
            kernel_id_t const track_kernel_id );

        track_status_t trackElementByElement(
            ClArgument& SIXTRL_RESTRICT_REF particles_arg,
            ClArgument& SIXTRL_RESTRICT_REF beam_elements_arg,
            ClArgument& SIXTRL_RESTRICT_REF elem_by_elem_buffer,
            size_type until_turn,
            size_type out_particle_block_offset );

        track_status_t trackElementByElement(
            ClArgument& SIXTRL_RESTRICT_REF particles_arg,
            ClArgument& SIXTRL_RESTRICT_REF beam_elements_arg,
            ClArgument& SIXTRL_RESTRICT_REF elem_by_elem_buffer,
            size_type until_turn, size_type out_particle_block_offset,
            kernel_id_t const track_kernel_id );

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        bool hasAssignBeamMonitorIoBufferKernel() const SIXTRL_NOEXCEPT;
        kernel_id_t const assignBeamMonitorIoBufferKernelId() const SIXTRL_NOEXCEPT;
        bool setAssignBeamMonitorIoBufferKernelId(
            kernel_id_t const track_kernel_id );

        status_t assignBeamMonitorIoBuffer(
            ClArgument& SIXTRL_RESTRICT_REF beam_elements_arg,
            ClArgument& SIXTRL_RESTRICT_REF out_buffer_arg,
            size_type const min_turn_id,
            size_type const out_particle_block_offset = size_type{ 0 } );

        status_t assignBeamMonitorIoBuffer(
            ClArgument& SIXTRL_RESTRICT_REF beam_elements_arg,
            ClArgument& SIXTRL_RESTRICT_REF out_buffer_arg,
            size_type const min_turn_id,
            size_type const out_buffer_index_offset,
            kernel_id_t const assign_kernel_id );

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        bool hasAssignElemByElemOutputBufferKernel() const SIXTRL_NOEXCEPT;
        kernel_id_t assignElemByElemOutputBufferKernelId() const SIXTRL_NOEXCEPT;
        bool setAssignElemByElemOutputBufferKernelId(
            kernel_id_t const track_kernel_id );

        status_t assignElemByElemOutputBuffer(
            ClArgument& SIXTRL_RESTRICT_REF out_buffer_arg,
            size_type const out_buffer_index_offset );

        status_t assignElemByElemOutputBuffer(
            ClArgument& SIXTRL_RESTRICT_REF out_buffer_arg,
            size_type const out_buffer_index_offset,
            kernel_id_t const assign_kernel_id );

        status_t assignElemByElemOutputBuffer(
            cl_buffer_t& SIXTRL_RESTRICT_REF cl_elem_by_elem_config_buffer,
            ClArgument& SIXTRL_RESTRICT_REF out_buffer_arg,
            size_type const out_buffer_index_offset,
            kernel_id_t const assign_kernel_id );

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        bool hasClearBeamMonitorIoBufferAssignmentKernel() const SIXTRL_NOEXCEPT;
        kernel_id_t clearBeamMonitorIoBufferAssignmentKernelId() const SIXTRL_NOEXCEPT;
        bool setClearBeamMonitorIoBufferAssignmentKernelId(
            kernel_id_t const clear_assign_kernel_id );

        status_t clearBeamMonitorIoBufferAssignment(
            ClArgument& SIXTRL_RESTRICT_REF beam_elements_arg );

        status_t clearBeamMonitorIoBufferAssignment(
            ClArgument& SIXTRL_RESTRICT_REF beam_elements_arg,
            kernel_id_t const clear_assign_kernel_id );

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        bool useOptimizedTrackingByDefault() const SIXTRL_NOEXCEPT;
        void enableOptimizedtrackingByDefault();
        void disableOptimizedTrackingByDefault();

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        bool isBeamBeamTrackingEnabled() const SIXTRL_NOEXCEPT;
        void enableBeamBeamTracking();
        void disableBeamBeamTracking();

        protected:

        virtual void doClear() override;
        virtual bool doInitDefaultPrograms() override;
        virtual bool doInitDefaultKernels()  override;
        virtual bool doSelectNode( size_type const node_index ) override;

        template< typename Iter >
        bool doUpdateParticleSetIndices(
            Iter pset_indices_begin, Iter pset_indices_end,
            const ::NS(Buffer) *const SIXTRL_RESTRICT particles_buffer );

        void doResetParticleSetIndices(
            size_type const num_particles = size_type{ 0 } );

        private:

        bool doSelectNodePrivImpl( size_type const node_index );
        void doClearPrivImpl();

        bool doInitDefaultProgramsPrivImpl();
        bool doInitDefaultKernelsPrivImpl();

        cl::Buffer   m_elem_by_elem_config_buffer;

        std::vector< size_type > m_particle_set_indices;
        std::vector< size_type > m_particle_set_num_particles;
        std::vector< size_type > m_particle_set_index_begins;
        size_type m_total_num_particles;
        size_type m_selected_particle_set_index;

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

    template< typename Iter >
        bool ClContext::doUpdateParticleSetIndices(
        Iter indices_begin, Iter indices_end,
        const ::NS(Buffer) *const SIXTRL_RESTRICT pbuffer )
    {
        bool success = false;

        using diff_t = std::ptrdiff_t;
        using size_t = ClContext::size_type;

        diff_t const temp_len = std::distance( indices_begin, indices_end );

        if( ( temp_len > diff_t{ 0 } ) &&
            ( !::NS(Buffer_needs_remapping)( pbuffer ) ) &&
            ( std::is_sorted( indices_begin, indices_end ) ) )
        {
            size_t const num_psets = static_cast< size_t >( temp_len );
            size_t const nobjects  = ::NS(Buffer_get_num_of_objects)( pbuffer );

            SIXTRL_ASSERT( num_psets > size_t{ 0 } );
            SIXTRL_ASSERT( nobjects >= num_psets );

            Iter it = indices_begin;
            std::advance( it, num_psets - size_t{ 1 } );

            SIXTRL_ASSERT( ( ( num_psets > size_t{ 1 } ) &&
                ( *it > *indices_begin ) ) || ( num_psets == size_t{ 1 } ) );

            if( *it >= nobjects )
            {
                return success;
            }

            std::vector< size_t > temp_num_particles( nobjects, size_t{ 0 } );
            temp_num_particles.clear();

            std::vector< size_t > temp_pset_offsets( num_psets, size_t{ 0 } );
            temp_pset_offsets.clear();

            ::NS(Particles) const* p = nullptr;

            success = true;
            it = indices_begin;

            size_t total_num_particles = size_t{ 0 };

            for( size_t ii = size_t{ 0 } ; ii < nobjects ; ++ii )
            {
                p = ::NS(Particles_buffer_get_const_particles)( pbuffer, ii );
                size_t const num_particles = ( p != nullptr )
                    ? ::NS(Particles_get_num_of_particles)( p ) : size_t{ 0 };

                temp_num_particles.push_back( num_particles );

                if( it != indices_end )
                {
                    SIXTRL_ASSERT( *it >= ii );

                    if( ii == *it )
                    {
                        if( num_particles == size_t{ 0 } )
                        {
                            success = false;
                            break;
                        }

                        temp_pset_offsets.push_back( total_num_particles );
                        total_num_particles += num_particles;

                        ++it;
                        SIXTRL_ASSERT( ( it == indices_end ) || ( *it > ii ) );
                    }
                }
            }

            if( success )
            {
                SIXTRL_ASSERT( temp_pset_offsets.size()  == num_psets );
                SIXTRL_ASSERT( temp_num_particles.size() == nobjects );
                SIXTRL_ASSERT( total_num_particles > size_t{ 0 } );

                this->m_particle_set_indices.clear();
                this->m_particle_set_indices.reserve( num_psets );

                this->m_particle_set_num_particles.clear();
                this->m_particle_set_num_particles.reserve( num_psets );

                this->m_particle_set_index_begins.clear();
                this->m_particle_set_index_begins.reserve( num_psets );

                size_t ii = size_t{ 0 };
                it = indices_begin;

                for( ; it != indices_end ; ++it, ++ii )
                {
                    this->m_particle_set_index_begins.push_back(
                        temp_pset_offsets[ ii ] );

                    this->m_particle_set_num_particles.push_back(
                        temp_num_particles[ *it ] );

                    this->m_particle_set_indices.push_back( *it );
                }

                this->m_total_num_particles = total_num_particles;
            }
        }

        return success;
    }

    template< typename Iter >
    bool ClContext::assignParticleArg(
        ClArgument& SIXTRL_RESTRICT_REF particles_arg,
        Iter indices_begin, Iter indices_end )
    {
        bool success = false;

        using size_t = ClContext::size_type;
        using kernel_id_t = ClContext::kernel_id_t;

        if( ( particles_arg.usesCObjectBuffer() ) &&
            ( particles_arg.ptrCObjectBuffer() != nullptr ) &&
            ( this->doUpdateParticleSetIndices( indices_begin, indices_end,
                    particles_arg.ptrCObjectBuffer() ) ) )
        {
            success = true;

            uint64_t const pset_index = *indices_begin;

            if( this->hasTrackUntilKernel() )
            {
                kernel_id_t const kernel_id = this->trackUntilKernelId();
                success &= ( this->kernelNumArgs( kernel_id ) >= size_t{ 5u } );
                this->assignKernelArgument( kernel_id, 0u, particles_arg );
                this->assignKernelArgumentValue( kernel_id, 1u, pset_index );
            }

            if( this->hasElementByElementTrackingKernel() )
            {
                kernel_id_t const kernel_id =
                    this->elementByElementTrackingKernelId();

                success &= ( this->kernelNumArgs( kernel_id ) >= size_t{ 6u } );
                this->assignKernelArgument( kernel_id, 0u, particles_arg );
                this->assignKernelArgumentValue( kernel_id, 1u, pset_index );
            }

            if( this->hasTrackLineKernel() )
            {
                kernel_id_t const kernel_id = this->trackLineKernelId();
                success &= ( this->kernelNumArgs( kernel_id ) >= size_t{ 7u } );
                this->assignKernelArgument( kernel_id, 0u, particles_arg );
                this->assignKernelArgumentValue( kernel_id, 1u, pset_index );
            }
        }

        return success;
    }
}

#if !defined( _GPUCODE )
extern "C" {
#endif /* !defined( _GPUCODE ) */

typedef SIXTRL_CXX_NAMESPACE::ClContext              NS(ClContext);
typedef SIXTRL_CXX_NAMESPACE::ClContext::num_turns_t NS(context_num_turns_t);

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

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(ClContext_has_track_until_kernel)(
    const NS(ClContext) *const SIXTRL_RESTRICT ctx );

SIXTRL_EXTERN SIXTRL_HOST_FN int NS(ClContext_get_tracking_kernel_id)(
    const NS(ClContext) *const SIXTRL_RESTRICT ctx );

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(ClContext_set_tracking_kernel_id)(
    NS(ClContext)* SIXTRL_RESTRICT ctx, int const kernel_id );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_EXTERN SIXTRL_HOST_FN NS(track_status_t)
NS(ClContext_track_until)( NS(ClContext)* SIXTRL_RESTRICT ctx,
    NS(context_num_turns_t) const until_turn );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(track_status_t)
NS(ClContext_track_until_with_kernel_id)( NS(ClContext)* SIXTRL_RESTRICT ctx,
    NS(context_num_turns_t) const until_turn, int const kernel_id );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(track_status_t)
NS(ClContext_track_until_args)( NS(ClContext)* SIXTRL_RESTRICT ctx,
    NS(ClArgument)* SIXTRL_RESTRICT ptr_particles_arg,
    NS(ClArgument)* SIXTRL_RESTRICT ptr_beam_elements_arg,
    NS(context_num_turns_t) const until_turn );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(track_status_t)
NS(ClContext_track_until_args_with_kernel_id)(
    NS(ClContext)* SIXTRL_RESTRICT ctx,
    NS(ClArgument)* SIXTRL_RESTRICT ptr_particles_arg,
    NS(ClArgument)* SIXTRL_RESTRICT ptr_beam_elements_arg,
    NS(context_num_turns_t) const until_turn, int const kernel_id );

/* ------------------------------------------------------------------------- */


SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(ClContext_has_track_line_kernel)(
    const NS(ClContext) *const SIXTRL_RESTRICT ctx );

SIXTRL_EXTERN SIXTRL_HOST_FN int NS(ClContext_get_line_tracking_kernel_id)(
    const NS(ClContext) *const SIXTRL_RESTRICT ctx );

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(ClContext_set_line_tracking_kernel_id)(
    NS(ClContext)* SIXTRL_RESTRICT ctx, int const kernel_id );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_EXTERN SIXTRL_HOST_FN NS(track_status_t) NS(ClContext_track_line)(
    NS(ClContext)* SIXTRL_RESTRICT ctx, NS(buffer_size_t) const line_begin_idx,
    NS(buffer_size_t) const line_end_idx, bool const finish_turn );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(track_status_t)
NS(ClContext_track_line_with_kernel_id)( NS(ClContext)* SIXTRL_RESTRICT ctx,
    NS(buffer_size_t) const line_begin_idx,
    NS(buffer_size_t) const line_end_idx,
    bool const finish_turn, int const line_tracking_kernel_id );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(track_status_t) NS(ClContext_track_line_args)(
    NS(ClContext)* SIXTRL_RESTRICT ctx,
    NS(ClArgument)* SIXTRL_RESTRICT particles_arg,
    NS(ClArgument)* SIXTRL_RESTRICT beam_elem_arg,
    NS(buffer_size_t) const line_begin_idx,
    NS(buffer_size_t) const line_end_idx, bool const finish_turn );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(track_status_t)
NS(ClContext_track_line_args_with_kernel_id)(
    NS(ClContext)* SIXTRL_RESTRICT ctx,
    NS(ClArgument)* SIXTRL_RESTRICT particles_arg,
    NS(ClArgument)* SIXTRL_RESTRICT beam_elem_arg,
    NS(buffer_size_t) const line_begin_idx,
    NS(buffer_size_t) const line_end_idx, bool const finish_turn,
    int const line_tracking_kernel_id );

/* ------------------------------------------------------------------------- */

SIXTRL_EXTERN SIXTRL_HOST_FN bool
NS(ClContext_has_element_by_element_tracking_kernel)(
    const NS(ClContext) *const SIXTRL_RESTRICT ctx );

SIXTRL_EXTERN SIXTRL_HOST_FN int
NS(ClContext_get_element_by_element_tracking_kernel_id)(
    const NS(ClContext) *const SIXTRL_RESTRICT ctx );

SIXTRL_EXTERN SIXTRL_HOST_FN bool
NS(ClContext_set_element_by_element_tracking_kernel_id)(
    NS(ClContext)* SIXTRL_RESTRICT ctx, int const tracking_kernel_id );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_EXTERN SIXTRL_HOST_FN NS(track_status_t)
NS(ClContext_continue_tracking_element_by_element)(
    NS(ClContext)* SIXTRL_RESTRICT ctx, NS(buffer_size_t) const until_turn,
    NS(buffer_size_t) const out_particle_block_offset );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(track_status_t)
NS(ClContext_continue_tracking_element_by_element_with_kernel_id)(
    NS(ClContext)* SIXTRL_RESTRICT ctx, NS(buffer_size_t) const until_turn,
    NS(buffer_size_t) const out_particle_block_offset,
    int const kernel_id );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(track_status_t)
NS(ClContext_track_element_by_element)(
    NS(ClContext)* SIXTRL_RESTRICT ctx,
    NS(ClArgument)* SIXTRL_RESTRICT ptr_particles_arg,
    NS(ClArgument)* SIXTRL_RESTRICT ptr_beam_elements_arg,
    NS(ClArgument)* SIXTRL_RESTRICT ptr_elem_by_elem_buffer_arg,
    NS(buffer_size_t) const until_turn,
    NS(buffer_size_t) const out_particle_block_offset );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(track_status_t)
NS(ClContext_track_element_by_element_with_kernel_id)(
    NS(ClContext)* SIXTRL_RESTRICT ctx,
    NS(ClArgument)* SIXTRL_RESTRICT ptr_particles_arg,
    NS(ClArgument)* SIXTRL_RESTRICT ptr_beam_elements_arg,
    NS(ClArgument)* SIXTRL_RESTRICT ptr_elem_by_elem_buffer_arg,
    NS(buffer_size_t) const until_turn,
    NS(buffer_size_t) const out_particle_block_offset,
    int const tracking_kernel_id );

/* ------------------------------------------------------------------------- */

SIXTRL_EXTERN SIXTRL_HOST_FN bool
NS(ClContext_has_assign_beam_monitor_out_buffer_kernel)(
    const NS(ClContext) *const SIXTRL_RESTRICT ctx );

SIXTRL_EXTERN SIXTRL_HOST_FN int
NS(ClContext_get_assign_beam_monitor_out_buffer_kernel_id)(
    const NS(ClContext) *const SIXTRL_RESTRICT ctx );

SIXTRL_EXTERN SIXTRL_HOST_FN bool
NS(ClContext_set_assign_beam_monitor_out_buffer_kernel_id)(
    NS(ClContext)* SIXTRL_RESTRICT ctx, int const assign_kernel_id );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_EXTERN SIXTRL_HOST_FN NS(ctrl_status_t)
NS(ClContext_assign_beam_monitor_out_buffer)(
    NS(ClContext)*  SIXTRL_RESTRICT ctx,
    NS(ClArgument)* SIXTRL_RESTRICT beam_elements_arg,
    NS(ClArgument)* SIXTRL_RESTRICT out_buffer,
    NS(buffer_size_t) const min_turn_id,
    NS(buffer_size_t) const out_buffer_index_offset );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(ctrl_status_t)
NS(ClContext_assign_beam_monitor_out_buffer_with_kernel_id)(
    NS(ClContext)*  SIXTRL_RESTRICT ctx,
    NS(ClArgument)* SIXTRL_RESTRICT beam_elements_arg,
    NS(ClArgument)* SIXTRL_RESTRICT out_buffer,
    NS(buffer_size_t) const min_turn_id,
    NS(buffer_size_t) const out_buffer_index_offset,
    int const assign_kernel_id );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_EXTERN SIXTRL_HOST_FN bool
NS(ClContext_has_clear_beam_monitor_out_assignment_kernel)(
    const NS(ClContext) *const SIXTRL_RESTRICT ctx );

SIXTRL_EXTERN SIXTRL_HOST_FN int
NS(ClContext_get_clear_beam_monitor_out_assignment_kernel_id)(
    const NS(ClContext) *const SIXTRL_RESTRICT ctx );

SIXTRL_EXTERN SIXTRL_HOST_FN bool
NS(ClContext_set_clear_beam_monitor_out_assignment_kernel_id)(
    NS(ClContext)* SIXTRL_RESTRICT ctx, int const kernel_id );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_EXTERN SIXTRL_HOST_FN NS(ctrl_status_t) t
NS(ClContext_clear_beam_monitor_out_assignment)(
    NS(ClContext)*  SIXTRL_RESTRICT context,
    NS(ClArgument)* SIXTRL_RESTRICT beam_elements_arg );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(ctrl_status_t)
NS(ClContext_clear_beam_monitor_out_assignment_with_kernel)(
    NS(ClContext)*  SIXTRL_RESTRICT context,
    NS(ClArgument)* SIXTRL_RESTRICT beam_elements_arg, int const kernel_id );

/* ------------------------------------------------------------------------- */

SIXTRL_EXTERN SIXTRL_HOST_FN bool
NS(ClContext_has_assign_elem_by_elem_out_buffer_kernel)
    const NS(ClContext) *const SIXTRL_RESTRICT ctx );

SIXTRL_EXTERN SIXTRL_HOST_FN int
NS(ClContext_get_assign_elem_by_elem_out_buffer_kernel_id)(
    const NS(ClContext) *const SIXTRL_RESTRICT ctx );

SIXTRL_EXTERN SIXTRL_HOST_FN bool
NS(ClContext_set_assign_elem_by_elem_out_buffer_kernel_id)(
    NS(ClContext)* SIXTRL_RESTRICT ctx, int const assign_kernel_id );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_EXTERN SIXTRL_HOST_FN NS(ctrl_status_t)
NS(ClContext_assign_elem_by_elem_out_buffer)(
    NS(ClContext)*  SIXTRL_RESTRICT ctx,
    NS(ClArgument)* SIXTRL_RESTRICT out_buffer,
    NS(buffer_size_t) const out_buffer_index_offset );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(ctrl_status_t)
NS(ClContext_assign_beam_monitor_out_buffer_with_kernel_id)(
    NS(ClContext)*  SIXTRL_RESTRICT ctx,
    cl_mem elem_by_elem_buffer,
    NS(ClArgument)* SIXTRL_RESTRICT out_buffer,
    NS(buffer_size_t) const out_buffer_index_offset,
    int const assign_kernel_id );

/* ------------------------------------------------------------------------- */

SIXTRL_EXTERN SIXTRL_HOST_FN bool
NS(ClContext_uses_optimized_tracking_by_default)(
    const NS(ClContext) *const SIXTRL_RESTRICT ctx );

SIXTRL_EXTERN SIXTRL_HOST_FN void
NS(ClContext_enable_optimized_tracking_by_default)(
    NS(ClContext)* SIXTRL_RESTRICT ctx );

SIXTRL_EXTERN SIXTRL_HOST_FN void
NS(ClContext_disable_optimized_tracking_by_default)(
    NS(ClContext)* SIXTRL_RESTRICT ctx );

/* ------------------------------------------------------------------------- */

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(ClContext_is_beam_beam_tracking_enabled)(
    const NS(ClContext) *const SIXTRL_RESTRICT ctx );

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(ClContext_enable_beam_beam_tracking)(
    NS(ClContext)* SIXTRL_RESTRICT ctx );

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(ClContext_disable_beam_beam_tracking)(
    NS(ClContext)* SIXTRL_RESTRICT ctx );


#if !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined( _GPUCODE ) && defined( __cplusplus ) */

#endif /* !defined( __CUDACC__ ) */

#endif /* SIXTRACKLIB_OPENCL_CONTEXT_H__ */

/* end: sixtracklib/opencl/context.h */
