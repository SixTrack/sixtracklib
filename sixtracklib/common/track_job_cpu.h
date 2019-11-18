#ifndef SIXTRACKLIB_SIXTRACKLIB_COMMON_INTERNAL_TRACK_JOB_CPU_H__
#define SIXTRACKLIB_SIXTRACKLIB_COMMON_INTERNAL_TRACK_JOB_CPU_H__

#if !defined( SIXTRL_NO_INCLUDES )
    #if defined( __cplusplus )
        #include "sixtracklib/common/buffer.hpp"
    #endif /* defined( __cplusplus ) */

    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/buffer.h"
    #include "sixtracklib/common/particles.h"
    #include "sixtracklib/common/internal/track_job_base.h"
    #include "sixtracklib/common/track_job.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( _GPUCODE )

#if !defined( SIXTRL_TRACK_JOB_CPU_ID )
    #define   SIXTRL_TRACK_JOB_CPU_ID 1
#endif /* !defined( SIXTRL_TRACK_JOB_CPU_ID ) */

#if !defined( SIXTRL_TRACK_JOB_CPU_STR )
    #define   SIXTRL_TRACK_JOB_CPU_STR "cpu"
#endif /* !defined( SIXTRL_TRACK_JOB_CPU_ID ) */

#if defined( __cplusplus )

namespace SIXTRL_CXX_NAMESPACE
{
    SIXTRL_STATIC_VAR track_job_type_t const
        TRACK_JOB_CPU_ID = track_job_type_t{ SIXTRL_TRACK_JOB_CPU_ID };

    SIXTRL_STATIC_VAR char const
        TRACK_JOB_CPU_STR[] = SIXTRL_TRACK_JOB_CPU_STR;

    class TrackJobCpu : public TrackJobBase
    {
        private:

        using _base_t  = TrackJobBase;

        public:

        using c_buffer_t            = _base_t::c_buffer_t;
        using buffer_t              = _base_t::buffer_t;
        using size_type             = _base_t::size_type;
        using elem_by_elem_config_t = _base_t::elem_by_elem_config_t;
        using track_status_t        = _base_t::track_status_t;
        using type_t                = _base_t::type_t;
        using output_buffer_flag_t  = _base_t::output_buffer_flag_t;
        using collect_flag_t        = _base_t::collect_flag_t;

        SIXTRL_HOST_FN explicit TrackJobCpu(
            std::string const& config_str = std::string{} );

        SIXTRL_HOST_FN explicit TrackJobCpu(
            const char *const SIXTRL_RESTRICT config_str );

        SIXTRL_HOST_FN TrackJobCpu(
            c_buffer_t* SIXTRL_RESTRICT particles_buffer,
            c_buffer_t* SIXTRL_RESTRICT beam_elements_buffer,
            c_buffer_t* SIXTRL_RESTRICT ptr_output_buffer = nullptr,
            size_type const until_turn_elem_by_elem       = size_type{ 0 },
            const char *const SIXTRL_RESTRICT config_str  = nullptr );

        SIXTRL_HOST_FN TrackJobCpu(
            c_buffer_t* SIXTRL_RESTRICT particles_buffer,
            size_type const particle_set_index,
            c_buffer_t* SIXTRL_RESTRICT beam_elements_buffer,
            c_buffer_t* SIXTRL_RESTRICT ptr_output_buffer = nullptr,
            size_type const until_turn_elem_by_elem       = size_type{ 0 },
            const char *const SIXTRL_RESTRICT config_str  = nullptr );

        SIXTRL_HOST_FN TrackJobCpu(
            c_buffer_t* SIXTRL_RESTRICT particles_buffer,
            size_type const num_particle_sets,
            size_type const* SIXTRL_RESTRICT particle_set_indices_begin,
            c_buffer_t* SIXTRL_RESTRICT beam_elements_buffer,
            c_buffer_t* SIXTRL_RESTRICT ptr_output_buffer = nullptr,
            size_type const until_turn_elem_by_elem       = size_type{ 0 },
            const char *const SIXTRL_RESTRICT config_str  = nullptr );

        template< typename PartSetIndexIter >
        SIXTRL_HOST_FN TrackJobCpu(
            c_buffer_t* SIXTRL_RESTRICT particles_buffer,
            PartSetIndexIter particle_set_indices_begin,
            PartSetIndexIter particle_set_indices_end,
            c_buffer_t* SIXTRL_RESTRICT beam_elements_buffer,
            c_buffer_t* SIXTRL_RESTRICT ptr_output_buffer = nullptr,
            size_type const until_turn_elem_by_elem = size_type{ 0 },
            const char *const SIXTRL_RESTRICT config_str  = nullptr );

        SIXTRL_HOST_FN TrackJobCpu(
            buffer_t& SIXTRL_RESTRICT_REF particles_buffer,
            buffer_t& SIXTRL_RESTRICT_REF beam_elements_buffer,
            buffer_t* SIXTRL_RESTRICT ptr_output_buffer = nullptr,
            size_type const until_turn_elem_by_elem = size_type{ 0 },
            std::string const& config_str = std::string{} );

        SIXTRL_HOST_FN TrackJobCpu(
            buffer_t& SIXTRL_RESTRICT_REF particles_buffer,
            size_type const particle_set_index,
            buffer_t& SIXTRL_RESTRICT_REF beam_elements_buffer,
            buffer_t* SIXTRL_RESTRICT ptr_output_buffer = nullptr,
            size_type const until_turn_elem_by_elem = size_type{ 0 },
            std::string const& config_str = std::string{} );

        SIXTRL_HOST_FN TrackJobCpu(
            buffer_t& SIXTRL_RESTRICT_REF particles_buffer,
            size_type const num_particle_sets,
            size_type const* SIXTRL_RESTRICT particle_set_indices_begin,
            buffer_t& SIXTRL_RESTRICT_REF beam_elements_buffer,
            buffer_t* SIXTRL_RESTRICT ptr_output_buffer = nullptr,
            size_type const until_turn_elem_by_elem = size_type{ 0 },
            std::string const& SIXTRL_RESTRICT_REF config_str = std::string{} );

        template< typename PartSetIndexIter >
        SIXTRL_HOST_FN TrackJobCpu(
            buffer_t& SIXTRL_RESTRICT_REF particles_buffer,
            PartSetIndexIter particle_set_indices_begin,
            PartSetIndexIter particle_set_indices_end,
            buffer_t& SIXTRL_RESTRICT_REF beam_elements_buffer,
            buffer_t* SIXTRL_RESTRICT ptr_output_buffer = nullptr,
            size_type const until_turn_elem_by_elem = size_type{ 0 },
            std::string const& SIXTRL_RESTRICT_REF config_str = std::string{} );

        SIXTRL_HOST_FN TrackJobCpu( TrackJobCpu const& other ) = default;
        SIXTRL_HOST_FN TrackJobCpu( TrackJobCpu&& other ) = default;

        SIXTRL_HOST_FN TrackJobCpu&
        operator=( TrackJobCpu const& rhs ) = default;

        SIXTRL_HOST_FN TrackJobCpu&
        operator=( TrackJobCpu&& rhs ) = default;

        SIXTRL_HOST_FN virtual ~TrackJobCpu() SIXTRL_NOEXCEPT;

        protected:

        virtual track_status_t doTrackUntilTurn(
            size_type const until_turn ) override;

        virtual track_status_t doTrackElemByElem(
            size_type const until_turn_elem_by_elem ) override;

        virtual track_status_t doTrackLine(
            size_type const beam_elements_begin_index,
            size_type const beam_elements_end_index,
            bool const finish_turn ) override;

        virtual void doCollect( collect_flag_t const flags ) override;
    };

    SIXTRL_HOST_FN void collect( TrackJobCpu& job ) SIXTRL_NOEXCEPT;

    SIXTRL_HOST_FN void collect( TrackJobCpu& job,
        track_job_collect_flag_t const flags ) SIXTRL_NOEXCEPT;

    SIXTRL_HOST_FN TrackJobCpu::track_status_t trackUntil(
        TrackJobCpu& SIXTRL_RESTRICT_REF job,
        TrackJobCpu::size_type const until_turn ) SIXTRL_NOEXCEPT;

    SIXTRL_HOST_FN TrackJobCpu::track_status_t trackElemByElem(
        TrackJobCpu& SIXTRL_RESTRICT_REF job,
        TrackJobCpu::size_type const until_turn_elem_by_elem ) SIXTRL_NOEXCEPT;

    SIXTRL_HOST_FN TrackJobCpu::track_status_t trackLine(
        TrackJobCpu& SIXTRL_RESTRICT_REF job,
        TrackJobCpu::size_type const beam_elements_begin_index,
        TrackJobCpu::size_type const beam_elements_end_index,
        bool const finish_turn = false ) SIXTRL_NOEXCEPT;
}

typedef SIXTRL_CXX_NAMESPACE::TrackJobCpu NS(TrackJobCpu);

SIXTRL_STATIC_VAR NS(track_job_type_t) const
    NS(TRACK_JOB_CPU_ID) = SIXTRL_CXX_NAMESPACE::TRACK_JOB_CPU_ID;

SIXTRL_STATIC_VAR char const
    NS(TRACK_JOB_CPU_STR)[] = SIXTRL_TRACK_JOB_CPU_STR;

#else /* defined( __cplusplus ) */

typedef void NS(TrackJobCpu);

SIXTRL_STATIC_VAR NS(track_job_type_t) const
    NS(TRACK_JOB_CPU_ID) = SIXTRL_TRACK_JOB_CPU_ID;

SIXTRL_STATIC_VAR char const
    NS(TRACK_JOB_CPU_STR)[] = SIXTRL_TRACK_JOB_CPU_STR;

#endif /* defined( __cplusplus ) */

#if defined( __cplusplus ) && !defined( _GPUCODE )
extern "C" {
#endif /* defined( __cplusplus ) && !defined( _GPUCODE ) */

SIXTRL_EXTERN SIXTRL_HOST_FN NS(TrackJobCpu)* NS(TrackJobCpu_create)( void );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(TrackJobCpu)*
NS(TrackJobCpu_create_from_config_str)(
    const char *const SIXTRL_RESTRICT config_str );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(TrackJobCpu)* NS(TrackJobCpu_new)(
    NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
    NS(Buffer)* SIXTRL_RESTRICT beam_elements_buffer );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(TrackJobCpu)* NS(TrackJobCpu_new_with_output)(
    NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
    NS(Buffer)* SIXTRL_RESTRICT beam_elements_buffer,
    NS(Buffer)* SIXTRL_RESTRICT output_buffer,
    NS(buffer_size_t) const until_turn_elem_by_elem );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(TrackJobCpu)* NS(TrackJobCpu_new_detailed)(
    NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
    NS(buffer_size_t) const num_particle_sets,
    NS(buffer_size_t) const* SIXTRL_RESTRICT particle_set_indices_begin,
    NS(Buffer)* SIXTRL_RESTRICT beam_elements_buffer,
    NS(Buffer)* SIXTRL_RESTRICT output_buffer,
    NS(buffer_size_t) const until_turn_elem_by_elem,
    const char *const SIXTRL_RESTRICT config_str );

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(TrackJobCpu_delete)(
    NS(TrackJobCpu)* SIXTRL_RESTRICT job );

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(TrackJobCpu_clear)(
    NS(TrackJobCpu)* SIXTRL_RESTRICT job );

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(TrackJobCpu_reset)(
    NS(TrackJobCpu)* SIXTRL_RESTRICT job,
    NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
    NS(Buffer)* SIXTRL_RESTRICT beam_elements_buffer,
    NS(Buffer)* SIXTRL_RESTRICT output_buffer );

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(TrackJobCpu_reset_with_output)(
    NS(TrackJobCpu)* SIXTRL_RESTRICT job,
    NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
    NS(Buffer)* SIXTRL_RESTRICT beam_elements_buffer,
    NS(Buffer)* SIXTRL_RESTRICT output_buffer,
    NS(buffer_size_t) const until_turn_elem_by_elem );

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(TrackJobCpu_reset_detailed)(
    NS(TrackJobCpu)* SIXTRL_RESTRICT job,
    NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
    NS(buffer_size_t) const num_particle_sets,
    NS(buffer_size_t) const* SIXTRL_RESTRICT particle_set_indices_begin,
    NS(Buffer)* SIXTRL_RESTRICT beam_elements_buffer,
    NS(Buffer)* SIXTRL_RESTRICT output_buffer,
    NS(buffer_size_t) const until_turn_elem_by_elem );

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(TrackJobCpu_assign_output_buffer)(
    NS(TrackJobCpu)* SIXTRL_RESTRICT job,
    NS(Buffer)* SIXTRL_RESTRICT output_buffer );

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(TrackJobCpu_collect)(
    NS(TrackJobCpu)* SIXTRL_RESTRICT job );

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(TrackJobCpu_collect_detailed)(
    NS(TrackJobCpu)* SIXTRL_RESTRICT job,
    NS(track_job_collect_flag_t) const flags );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(track_status_t)
NS(TrackJobCpu_track_until_turn)( NS(TrackJobCpu)* SIXTRL_RESTRICT job,
    NS(buffer_size_t) const until_turn );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(track_status_t)
NS(TrackJobCpu_track_elem_by_elem)( NS(TrackJobCpu)* SIXTRL_RESTRICT job,
    NS(buffer_size_t) const until_turn_elem_by_elem );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(track_status_t)
NS(TrackJobCpu_track_line)( NS(TrackJobCpu)* SIXTRL_RESTRICT job,
    NS(buffer_size_t) const beam_elements_begin_index,
    NS(buffer_size_t) const beam_elements_end_index, bool const finish_turn );

#if defined( __cplusplus ) && !defined( _GPUCODE )
}
#endif /* defined( __cplusplus ) && !defined( _GPUCODE ) */

/* ************************************************************************* */
/* *****         Implementation of inline methods and functions        ***** */
/* ************************************************************************* */

#if defined( __cplusplus )

namespace SIXTRL_CXX_NAMESPACE
{
    template< typename PartSetIndexIter >
    TrackJobCpu::TrackJobCpu(
        TrackJobCpu::c_buffer_t* SIXTRL_RESTRICT particles_buffer,
        PartSetIndexIter particle_set_indices_begin,
        PartSetIndexIter particle_set_indices_end,
        TrackJobCpu::c_buffer_t* SIXTRL_RESTRICT beam_elements_buffer,
        TrackJobCpu::c_buffer_t* SIXTRL_RESTRICT ptr_output_buffer,
        TrackJobCpu::size_type const until_turn_elem_by_elem,
        const char *const SIXTRL_RESTRICT config_str ) :
        TrackJobBase( SIXTRL_CXX_NAMESPACE::TRACK_JOB_CPU_STR,
                      SIXTRL_CXX_NAMESPACE::TRACK_JOB_CPU_ID )
    {
        if( ( particle_set_indices_begin != particle_set_indices_end ) &&
            ( std::distance( particle_set_indices_begin,
                particle_set_indices_end ) >= std::ptrdiff_t{ 0 } ) )
        {
            this->doSetParticleSetIndices(
                particle_set_indices_begin, particle_set_indices_end,
                particles_buffer );
        }
        else
        {
            TrackJobBase::doInitDefaultParticleSetIndices();
        }

        if( TrackJobBase::doReset( particles_buffer, beam_elements_buffer,
                ptr_output_buffer, until_turn_elem_by_elem ) )
        {
            this->doSetPtrCParticleBuffer( particles_buffer );
            this->doSetPtrCBeamElementsBuffer( beam_elements_buffer );
        }

        if( ( config_str != nullptr ) &&
            ( std::strlen( config_str ) > TrackJobCpu::size_type{ 0 } ) )
        {
            TrackJobBase::doSetConfigStr( config_str );
            TrackJobBase::doParseConfigStr( config_str );
        }
    }

    template< typename PartSetIndexIter >
    TrackJobCpu::TrackJobCpu(
        TrackJobCpu::buffer_t& SIXTRL_RESTRICT_REF particles_buffer,
        PartSetIndexIter particle_set_indices_begin,
        PartSetIndexIter particle_set_indices_end,
        TrackJobCpu::buffer_t& SIXTRL_RESTRICT_REF beam_elements_buffer,
        TrackJobCpu::buffer_t* SIXTRL_RESTRICT ptr_output_buffer,
        TrackJobCpu::size_type const until_turn_elem_by_elem,
        std::string const& SIXTRL_RESTRICT_REF config_str ) :
        TrackJobBase( SIXTRL_CXX_NAMESPACE::TRACK_JOB_CPU_STR,
                      SIXTRL_CXX_NAMESPACE::TRACK_JOB_CPU_ID )
    {
        using c_buffer_t = TrackJobCpu::c_buffer_t;

        c_buffer_t* ptr_part_buffer  = particles_buffer.getCApiPtr();
        c_buffer_t* ptr_belem_buffer = beam_elements_buffer.getCApiPtr();
        c_buffer_t* ptr_out_buffer   = ( ptr_output_buffer != nullptr )
            ? ptr_output_buffer->getCApiPtr() : nullptr;

        if( ( particle_set_indices_begin != particle_set_indices_end ) &&
            ( std::distance( particle_set_indices_begin,
                particle_set_indices_end ) >= std::ptrdiff_t{ 0 } ) )
        {
            this->doSetParticleSetIndices(
                particle_set_indices_begin, particle_set_indices_end,
                particles_buffer.getCApiPtr() );
        }
        else
        {
            TrackJobBase::doInitDefaultParticleSetIndices();
        }

        if( TrackJobBase::doReset( ptr_part_buffer, ptr_belem_buffer,
                ptr_out_buffer, until_turn_elem_by_elem ) )
        {
            this->doSetPtrParticleBuffer( &particles_buffer );
            this->doSetPtrBeamElementsBuffer( &beam_elements_buffer );

            if( ( ptr_out_buffer != nullptr ) && ( this->hasOutputBuffer() ) )
            {
                this->doSetPtrOutputBuffer( ptr_output_buffer );
            }
        }

        if( !config_str.empty() )
        {
            TrackJobBase::doSetConfigStr( config_str.c_str() );
            TrackJobBase::doParseConfigStr( this->ptrConfigStr() );
        }
    }

    SIXTRL_INLINE void collect( TrackJobCpu& ) SIXTRL_NOEXCEPT
    {
        return;
    }

    SIXTRL_INLINE void collect( TrackJobCpu&,
        track_job_collect_flag_t const ) SIXTRL_NOEXCEPT
    {
        return;
    }
}

#endif /* defined( __cplusplus ) */

#endif /* !defined( _GPUCODE ) */

#endif /* SIXTRACKLIB_SIXTRACKLIB_COMMON_INTERNAL_TRACK_JOB_CPU_H__ */

/* end: sixtracklib/common/internal/track_job_cpu.h */
