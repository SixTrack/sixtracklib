#ifndef SIXTRACKLIB_COMMON_PARTICLES_PARTICLES_ADDR_C99_H__
#define SIXTRACKLIB_COMMON_PARTICLES_PARTICLES_ADDR_C99_H__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stdbool.h>
    #include <stdlib.h>
    #include <stdint.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/control/definitions.h"
    #include "sixtracklib/common/control/debug_register.h"
    #include "sixtracklib/common/internal/buffer_main_defines.h"
    #include "sixtracklib/common/internal/buffer_object_defines.h"
    #include "sixtracklib/common/internal/particles_defines.h"
    #include "sixtracklib/common/buffer/buffer_type.h"
    #include "sixtracklib/common/buffer/buffer_object.h"
    #include "sixtracklib/common/particles/definitions.h"
    #include "sixtracklib/common/particles.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus ) && !defined( _GPUCODE )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

typedef struct NS(ParticlesAddr)
{
    NS(particle_num_elements_t) num_particles   SIXTRL_ALIGN( 8 );
    NS(buffer_addr_t) q0_addr                   SIXTRL_ALIGN( 8 );
    NS(buffer_addr_t) mass0_addr                SIXTRL_ALIGN( 8 );
    NS(buffer_addr_t) beta0_addr                SIXTRL_ALIGN( 8 );
    NS(buffer_addr_t) gamma0_addr               SIXTRL_ALIGN( 8 );
    NS(buffer_addr_t) p0c_addr                  SIXTRL_ALIGN( 8 );

    NS(buffer_addr_t) s_addr                    SIXTRL_ALIGN( 8 );
    NS(buffer_addr_t) x_addr                    SIXTRL_ALIGN( 8 );
    NS(buffer_addr_t) y_addr                    SIXTRL_ALIGN( 8 );
    NS(buffer_addr_t) px_addr                   SIXTRL_ALIGN( 8 );
    NS(buffer_addr_t) py_addr                   SIXTRL_ALIGN( 8 );
    NS(buffer_addr_t) zeta_addr                 SIXTRL_ALIGN( 8 );

    NS(buffer_addr_t) psigma_addr               SIXTRL_ALIGN( 8 );
    NS(buffer_addr_t) delta_addr                SIXTRL_ALIGN( 8 );
    NS(buffer_addr_t) rpp_addr                  SIXTRL_ALIGN( 8 );
    NS(buffer_addr_t) rvv_addr                  SIXTRL_ALIGN( 8 );
    NS(buffer_addr_t) chi_addr                  SIXTRL_ALIGN( 8 );
    NS(buffer_addr_t) charge_ratio_addr         SIXTRL_ALIGN( 8 );

    NS(buffer_addr_t) particle_id_addr          SIXTRL_ALIGN( 8 );
    NS(buffer_addr_t) at_element_id_addr        SIXTRL_ALIGN( 8 );
    NS(buffer_addr_t) at_turn_addr              SIXTRL_ALIGN( 8 );
    NS(buffer_addr_t) state_addr                SIXTRL_ALIGN( 8 );
}
NS(ParticlesAddr);

/* ------------------------------------------------------------------------- */
/* Helper functions: */

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t)
NS(ParticlesAddr_get_required_num_dataptrs_on_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT buffer,
    NS(particle_num_elements_t) const num_particles,
    NS(buffer_size_t) const slot_size );

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t)
NS(ParticlesAddr_get_required_num_slots_on_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT buffer,
    NS(particle_num_elements_t) const num_particles,
    NS(buffer_size_t) const slot_size );


SIXTRL_STATIC SIXTRL_FN SIXTRL_PARTICLE_ARGPTR_DEC NS(ParticlesAddr)*
NS(ParticlesAddr_preset)( SIXTRL_PARTICLE_ARGPTR_DEC NS(ParticlesAddr)*
    SIXTRL_RESTRICT addr );

SIXTRL_STATIC SIXTRL_FN void NS(ParticlesAddr_clear)(
    SIXTRL_PARTICLE_ARGPTR_DEC  NS(ParticlesAddr)* SIXTRL_RESTRICT addr );

SIXTRL_STATIC SIXTRL_FN void NS(ParticlesAddr_copy)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(ParticlesAddr)* SIXTRL_RESTRICT destination,
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(ParticlesAddr)
        *const SIXTRL_RESTRICT source );

SIXTRL_STATIC SIXTRL_FN int NS(ParticlesAddr_compare_values)(
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(ParticlesAddr) *const
        SIXTRL_RESTRICT lhs,
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(ParticlesAddr) *const
        SIXTRL_RESTRICT rhs );

/* ------------------------------------------------------------------------- */
/* Assignment from and to NS(Particles) */

SIXTRL_STATIC SIXTRL_FN void NS(ParticlesAddr_assign_from_particles)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(ParticlesAddr)* SIXTRL_RESTRICT part_addr,
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT p );

SIXTRL_STATIC SIXTRL_FN void NS(ParticlesAddr_assign_to_particles)(
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(ParticlesAddr) *const
        SIXTRL_RESTRICT part_addr,
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)*  SIXTRL_RESTRICT p );

SIXTRL_STATIC SIXTRL_FN void NS(ParticlesAddr_remap_addresses)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(ParticlesAddr)* SIXTRL_RESTRICT part_addr,
    NS(buffer_addr_diff_t) const addr_offset );

/* ------------------------------------------------------------------------- */

SIXTRL_STATIC SIXTRL_FN void NS(ParticlesAddr_managed_buffer_remap_addresses)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT buffer_begin,
    NS(buffer_size_t) const buffer_index,
    NS(buffer_addr_diff_t) const addr_offset,
    NS(buffer_size_t) const slot_size );

SIXTRL_STATIC SIXTRL_FN void
NS(ParticlesAddr_managed_buffer_all_remap_addresses)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT buffer_begin,
    NS(buffer_addr_diff_t) const addr_offset,
    NS(buffer_size_t) const slot_size );

/* ------------------------------------------------------------------------- */

SIXTRL_STATIC SIXTRL_FN int NS(ParticlesAddr_managed_buffer_compare)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT lhs_begin,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT rhs_begin,
    NS(buffer_size_t) const buffer_index, NS(buffer_size_t) const slot_size );

SIXTRL_STATIC SIXTRL_FN int NS(ParticlesAddr_managed_buffer_all_compare)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT lhs_begin,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT rhs_begin,
    NS(buffer_size_t) const slot_size );

/* ------------------------------------------------------------------------- */

SIXTRL_STATIC SIXTRL_FN SIXTRL_BUFFER_DATAPTR_DEC NS(ParticlesAddr) const*
NS(BufferIndex_get_const_particles_addr)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC const NS(Object) *const buffer_obj );

SIXTRL_STATIC SIXTRL_FN SIXTRL_BUFFER_DATAPTR_DEC NS(ParticlesAddr)*
NS(BufferIndex_get_particles_addr)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC const NS(Object) *const buffer_obj );


SIXTRL_STATIC SIXTRL_FN SIXTRL_BUFFER_DATAPTR_DEC NS(ParticlesAddr) const*
NS(ParticlesAddr_managed_buffer_get_const_particle_addr)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const index, NS(buffer_size_t) const slot_size );

SIXTRL_STATIC SIXTRL_FN SIXTRL_BUFFER_DATAPTR_DEC NS(ParticlesAddr)*
NS(ParticlesAddr_managed_buffer_get_particle_addr)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const index, NS(buffer_size_t) const slot_size );

/* ------------------------------------------------------------------------- */

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t)
NS(Particles_managed_buffer_store_addresses)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT paddr_buffer,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT pbuffer,
    NS(buffer_size_t) const index, NS(buffer_size_t) const slot_size );

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t)
NS(Particles_managed_buffer_store_addresses_debug)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT paddr_buffer,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT pbuffer,
    NS(buffer_size_t) const index, NS(buffer_size_t) const slot_size,
    SIXTRL_ARGPTR_DEC NS(arch_debugging_t)* SIXTRL_RESTRICT ptr_debug_flag );

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t)
NS(Particles_managed_buffer_store_all_addresses)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT paddr_buffer,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT pbuffer,
    NS(buffer_size_t) const slot_size );

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t)
NS(Particles_managed_buffer_store_all_addresses_debug)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT paddr_buffer,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT pbuffer,
    NS(buffer_size_t) const slot_size,
    SIXTRL_DATAPTR_DEC NS(arch_debugging_t)* SIXTRL_RESTRICT ptr_error_flag );


/* ------------------------------------------------------------------------- */

#if !defined( _GPUCODE )

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_PARTICLE_ARGPTR_DEC NS(ParticlesAddr)*
NS(ParticlesAddr_preset_ext)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(ParticlesAddr)* SIXTRL_RESTRICT paddr );

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(ParticlesAddr_assign_from_particles_ext)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(ParticlesAddr)* SIXTRL_RESTRICT part_addr,
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT p );

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(ParticlesAddr_assign_to_particles_ext)(
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(ParticlesAddr) *const
        SIXTRL_RESTRICT part_addr,
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)*  SIXTRL_RESTRICT p );

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(ParticlesAddr_remap_addresses_ext)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(ParticlesAddr)* SIXTRL_RESTRICT part_addr,
    NS(buffer_addr_diff_t) const addr_offset );

/* ------------------------------------------------------------------------- */

SIXTRL_EXTERN SIXTRL_HOST_FN void 
NS(ParticlesAddr_managed_buffer_remap_addresses_ext)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT buffer_begin,
    NS(buffer_size_t) const buffer_index,
    NS(buffer_addr_diff_t) const addr_offset,
    NS(buffer_size_t) const slot_size );

SIXTRL_EXTERN SIXTRL_HOST_FN void
NS(ParticlesAddr_managed_buffer_all_remap_addresses_ext)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT buffer_begin,
    NS(buffer_addr_diff_t) const addr_offset,
    NS(buffer_size_t) const slot_size );

/* ------------------------------------------------------------------------- */

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t)
NS(ParticlesAddr_get_required_num_dataptrs)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(particle_num_elements_t) const num_particles );

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t)
NS(ParticlesAddr_get_required_num_slots)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(particle_num_elements_t) const num_particles );

SIXTRL_STATIC SIXTRL_FN SIXTRL_BUFFER_DATAPTR_DEC NS(ParticlesAddr) const*
NS(ParticlesAddr_buffer_get_const_particle_addr)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const index );

SIXTRL_STATIC SIXTRL_FN SIXTRL_BUFFER_DATAPTR_DEC NS(ParticlesAddr)*
NS(ParticlesAddr_buffer_get_particle_addr)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const index );

/* ------------------------------------------------------------------------ */

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) 
NS(ParticlesAddr_buffer_clear_single)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT paddr_buffer, 
    NS(buffer_size_t) const index );

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) 
NS(ParticlesAddr_buffer_clear_all)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT paddr_buffer );

/* ------------------------------------------------------------------------ */

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t)
NS(ParticlesAddr_prepare_buffer_based_on_particles_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT paddr_buffer,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer)
        *const SIXTRL_RESTRICT particles_buffer );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t)
NS(ParticlesAddr_buffer_store_addresses)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT paddr_buffer,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const
        SIXTRL_RESTRICT particles_buffer,
    NS(buffer_size_t) const index );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t)
NS(ParticlesAddr_buffer_store_all_addresses)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT paddr_buffer,
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const
        SIXTRL_RESTRICT particles_buffer );

/* ------------------------------------------------------------------------ */

SIXTRL_EXTERN SIXTRL_HOST_FN void
NS(ParticlesAddr_buffer_remap_adresses)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const buffer_index,
    NS(buffer_addr_diff_t) const addr_offset );

SIXTRL_EXTERN SIXTRL_HOST_FN void
NS(ParticlesAddr_buffer_all_remap_adresses)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_addr_diff_t) const addr_offset );

/* ------------------------------------------------------------------------ */

SIXTRL_STATIC SIXTRL_FN bool NS(ParticlesAddr_can_be_added)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(particle_num_elements_t) const num_particles,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_objects,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_slots,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_dataptrs );

SIXTRL_STATIC SIXTRL_FN SIXTRL_BUFFER_DATAPTR_DEC NS(ParticlesAddr)*
NS(ParticlesAddr_new)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(particle_num_elements_t) const num_particles );

SIXTRL_STATIC SIXTRL_FN SIXTRL_BUFFER_DATAPTR_DEC NS(ParticlesAddr)*
NS(ParticlesAddr_add)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(particle_num_elements_t) const num_particles,
    NS(buffer_addr_t) const q0_addr,    NS(buffer_addr_t)  const mass0_addr,
    NS(buffer_addr_t) const beta0_addr, NS(buffer_addr_t)  const gamma0_addr,
    NS(buffer_addr_t) const p0c_addr,   NS(buffer_addr_t)  const s_addr,
    NS(buffer_addr_t) const x_addr,     NS(buffer_addr_t)  const y_addr,
    NS(buffer_addr_t) const px_addr,    NS(buffer_addr_t)  const py_addr,
    NS(buffer_addr_t) const zeta_addr,  NS(buffer_addr_t)  const psigma_addr,
    NS(buffer_addr_t) const delta_addr, NS(buffer_addr_t)  const rpp_addr,
    NS(buffer_addr_t) const rvv_addr,   NS(buffer_addr_t)  const chi_addr,
    NS(buffer_addr_t) const charge_ratio_addr,
    NS(buffer_addr_t) const particle_id_addr,
    NS(buffer_addr_t) const at_element_id_addr,
    NS(buffer_addr_t) const at_turn_addr, NS(buffer_addr_t) const state_addr );

SIXTRL_STATIC SIXTRL_FN SIXTRL_BUFFER_DATAPTR_DEC NS(ParticlesAddr)*
NS(ParticlesAddr_add_copy)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(ParticlesAddr) *const
        SIXTRL_RESTRICT addr );

#endif /* !defined( _GPUCODE ) */

#if defined( __cplusplus ) && !defined( _GPUCODE )
}
#endif /* C++, Host */

/* ************************************************************************* */
/* *****          Implementation of C inline functions                   *** */
/* ************************************************************************* */

#if !defined( SIXTRL_NO_INCLUDES )
    #if !defined( _GPUCODE )
        #include "sixtracklib/common/buffer.h"
    #endif /* !defined( _GPUCODE ) */
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus ) && !defined( _GPUCODE )
extern "C" {
#endif /* C++, Host */

SIXTRL_INLINE NS(buffer_size_t)
NS(ParticlesAddr_get_required_num_dataptrs_on_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT buffer,
    NS(particle_num_elements_t) const num_particles,
    NS(buffer_size_t) const slot_size )
{
    ( void )buffer;
    ( void )num_particles;
    ( void )slot_size;

    return ( NS(buffer_size_t) )0u;
}

SIXTRL_INLINE NS(buffer_size_t)
NS(ParticlesAddr_get_required_num_slots_on_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT buffer,
    NS(particle_num_elements_t) const num_particles,
    NS(buffer_size_t) const slot_size )
{
    ( void )buffer;
    ( void )num_particles;

    return NS(ManagedBuffer_get_slot_based_length)(
        sizeof( NS(ParticlesAddr) ), slot_size );
}

SIXTRL_INLINE SIXTRL_PARTICLE_ARGPTR_DEC NS(ParticlesAddr)*
NS(ParticlesAddr_preset)( SIXTRL_PARTICLE_ARGPTR_DEC NS(ParticlesAddr)*
    SIXTRL_RESTRICT particles_addr )
{
    if( particles_addr != SIXTRL_NULLPTR )
    {
        particles_addr->num_particles = ( NS(particle_num_elements_t) )0u;
        NS(ParticlesAddr_clear)( particles_addr );
    }

    return particles_addr;
}

SIXTRL_INLINE void NS(ParticlesAddr_clear)( SIXTRL_PARTICLE_ARGPTR_DEC
    NS(ParticlesAddr)* SIXTRL_RESTRICT particles_addr )
{
    if( particles_addr != SIXTRL_NULLPTR )
    {
        typedef NS(buffer_addr_t) addr_t;
        SIXTRL_STATIC_VAR addr_t const ZERO_ADDR = ( addr_t )0u;

        particles_addr->q0_addr            = ZERO_ADDR;
        particles_addr->mass0_addr         = ZERO_ADDR;
        particles_addr->beta0_addr         = ZERO_ADDR;
        particles_addr->gamma0_addr        = ZERO_ADDR;
        particles_addr->p0c_addr           = ZERO_ADDR;

        particles_addr->s_addr             = ZERO_ADDR;
        particles_addr->x_addr             = ZERO_ADDR;
        particles_addr->y_addr             = ZERO_ADDR;
        particles_addr->px_addr            = ZERO_ADDR;
        particles_addr->py_addr            = ZERO_ADDR;
        particles_addr->zeta_addr          = ZERO_ADDR;

        particles_addr->psigma_addr        = ZERO_ADDR;
        particles_addr->delta_addr         = ZERO_ADDR;
        particles_addr->rpp_addr           = ZERO_ADDR;
        particles_addr->rvv_addr           = ZERO_ADDR;
        particles_addr->chi_addr           = ZERO_ADDR;
        particles_addr->charge_ratio_addr  = ZERO_ADDR;

        particles_addr->particle_id_addr   = ZERO_ADDR;
        particles_addr->at_element_id_addr = ZERO_ADDR;
        particles_addr->at_turn_addr       = ZERO_ADDR;
        particles_addr->state_addr         = ZERO_ADDR;
    }
}

SIXTRL_INLINE void NS(ParticlesAddr_copy)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(ParticlesAddr)* SIXTRL_RESTRICT destination,
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(ParticlesAddr)
        *const SIXTRL_RESTRICT source )
{
    if( ( destination != SIXTRL_NULLPTR ) && ( source != SIXTRL_NULLPTR ) )
    {
        destination->num_particles      = source->num_particles;
        destination->q0_addr            = source->q0_addr;
        destination->mass0_addr         = source->mass0_addr;
        destination->beta0_addr         = source->beta0_addr;
        destination->gamma0_addr        = source->gamma0_addr;
        destination->p0c_addr           = source->p0c_addr;

        destination->s_addr             = source->s_addr;
        destination->x_addr             = source->x_addr;
        destination->y_addr             = source->y_addr;
        destination->px_addr            = source->px_addr;
        destination->py_addr            = source->py_addr;
        destination->zeta_addr          = source->zeta_addr;

        destination->psigma_addr        = source->psigma_addr;
        destination->delta_addr         = source->delta_addr;
        destination->rpp_addr           = source->rpp_addr;
        destination->rvv_addr           = source->rvv_addr;
        destination->chi_addr           = source->chi_addr;
        destination->charge_ratio_addr  = source->charge_ratio_addr;

        destination->particle_id_addr   = source->particle_id_addr;
        destination->at_element_id_addr = source->at_element_id_addr;
        destination->at_turn_addr       = source->at_turn_addr;
        destination->state_addr         = source->state_addr;
    }
}

SIXTRL_INLINE int NS(ParticlesAddr_compare_values)(
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(ParticlesAddr) *const
        SIXTRL_RESTRICT lhs,
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(ParticlesAddr) *const
        SIXTRL_RESTRICT rhs )
{
    int cmp_result = -1;

    if( ( lhs != SIXTRL_NULLPTR ) && ( rhs != SIXTRL_NULLPTR ) )
    {
        cmp_result = ( lhs->num_particles == rhs->num_particles )
            ? 0 : ( ( lhs->num_particles > rhs->num_particles ) ? +1 : -1 );

        if( ( cmp_result == 0 ) && ( lhs->q0_addr != rhs->q0_addr ) )
        {
            cmp_result = ( lhs->q0_addr > rhs->q0_addr ) ? +1 : -1;
        }

        if( ( cmp_result == 0 ) && ( lhs->mass0_addr != rhs->mass0_addr ) )
        {
            cmp_result = ( lhs->mass0_addr > rhs->mass0_addr ) ? +1 : -1;
        }

        if( ( cmp_result == 0 ) && ( lhs->beta0_addr != rhs->beta0_addr ) )
        {
            cmp_result = ( lhs->beta0_addr > rhs->beta0_addr ) ? +1 : -1;
        }

        if( ( cmp_result == 0 ) && ( lhs->gamma0_addr != rhs->gamma0_addr ) )
        {
            cmp_result = ( lhs->gamma0_addr > rhs->gamma0_addr ) ? +1 : -1;
        }

        if( ( cmp_result == 0 ) && ( lhs->p0c_addr != rhs->p0c_addr ) )
        {
            cmp_result = ( lhs->p0c_addr > rhs->p0c_addr ) ? +1 : -1;
        }

        if( ( cmp_result == 0 ) && ( lhs->s_addr != rhs->s_addr ) )
        {
            cmp_result = ( lhs->s_addr > rhs->s_addr ) ? +1 : -1;
        }

        if( ( cmp_result == 0 ) && ( lhs->x_addr != rhs->x_addr ) )
        {
            cmp_result = ( lhs->x_addr > rhs->x_addr ) ? +1 : -1;
        }

        if( ( cmp_result == 0 ) && ( lhs->y_addr != rhs->y_addr ) )
        {
            cmp_result = ( lhs->y_addr > rhs->y_addr ) ? +1 : -1;
        }

        if( ( cmp_result == 0 ) && ( lhs->px_addr != rhs->px_addr ) )
        {
            cmp_result = ( lhs->px_addr > rhs->px_addr ) ? +1 : -1;
        }

        if( ( cmp_result == 0 ) && ( lhs->py_addr != rhs->py_addr ) )
        {
            cmp_result = ( lhs->py_addr > rhs->py_addr ) ? +1 : -1;
        }

        if( ( cmp_result == 0 ) && ( lhs->zeta_addr != rhs->zeta_addr ) )
        {
            cmp_result = ( lhs->zeta_addr > rhs->zeta_addr ) ? +1 : -1;
        }

        if( ( cmp_result == 0 ) && ( lhs->psigma_addr != rhs->psigma_addr ) )
        {
            cmp_result = ( lhs->psigma_addr > rhs->psigma_addr ) ? +1 : -1;
        }

        if( ( cmp_result == 0 ) && ( lhs->delta_addr != rhs->delta_addr ) )
        {
            cmp_result = ( lhs->delta_addr > rhs->delta_addr ) ? +1 : -1;
        }

        if( ( cmp_result == 0 ) && ( lhs->rpp_addr != rhs->rpp_addr ) )
        {
            cmp_result = ( lhs->rpp_addr > rhs->rpp_addr ) ? +1 : -1;
        }

        if( ( cmp_result == 0 ) && ( lhs->rvv_addr != rhs->rvv_addr ) )
        {
            cmp_result = ( lhs->rvv_addr > rhs->rvv_addr ) ? +1 : -1;
        }

        if( ( cmp_result == 0 ) && ( lhs->chi_addr != rhs->chi_addr ) )
        {
            cmp_result = ( lhs->chi_addr > rhs->chi_addr ) ? +1 : -1;
        }

        if( ( cmp_result == 0 ) &&
            ( lhs->charge_ratio_addr != rhs->charge_ratio_addr ) )
        {
            cmp_result = ( lhs->charge_ratio_addr > rhs->charge_ratio_addr )
                ? +1 : -1;
        }

        if( ( cmp_result == 0 ) &&
            ( lhs->particle_id_addr != rhs->particle_id_addr ) )
        {
            cmp_result = ( lhs->particle_id_addr > rhs->particle_id_addr )
                ? +1 : -1;
        }

        if( ( cmp_result == 0 ) &&
            ( lhs->at_element_id_addr != rhs->at_element_id_addr ) )
        {
            cmp_result = ( lhs->at_element_id_addr > rhs->at_element_id_addr )
                ? +1 : -1;
        }

        if( ( cmp_result == 0 ) && ( lhs->at_turn_addr != rhs->at_turn_addr ) )
        {
            cmp_result = ( lhs->at_turn_addr > rhs->at_turn_addr ) ? +1 : -1;
        }


        if( ( cmp_result == 0 ) && ( lhs->state_addr != rhs->state_addr ) )
        {
            cmp_result = ( lhs->state_addr > rhs->state_addr ) ? +1 : -1;
        }
    }
    else if( lhs != SIXTRL_NULLPTR )
    {
        return 1;
    }

    return -1;
}

/* ------------------------------------------------------------------------- */
/* Assignment from and to NS(Particles) */

SIXTRL_INLINE void NS(ParticlesAddr_assign_from_particles)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(ParticlesAddr)* SIXTRL_RESTRICT paddr,
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT p )
{
    typedef NS(buffer_addr_t) addr_t;

    if( ( paddr != SIXTRL_NULLPTR ) && ( p != SIXTRL_NULLPTR ) )
    {
        paddr->num_particles = p->num_particles;

        paddr->q0_addr            = ( addr_t )( uintptr_t )p->q0;
        paddr->mass0_addr         = ( addr_t )( uintptr_t )p->mass0;
        paddr->beta0_addr         = ( addr_t )( uintptr_t )p->beta0;
        paddr->gamma0_addr        = ( addr_t )( uintptr_t )p->gamma0;
        paddr->p0c_addr           = ( addr_t )( uintptr_t )p->p0c;

        paddr->s_addr             = ( addr_t )( uintptr_t )p->s;
        paddr->x_addr             = ( addr_t )( uintptr_t )p->x;
        paddr->y_addr             = ( addr_t )( uintptr_t )p->y;
        paddr->px_addr            = ( addr_t )( uintptr_t )p->px;
        paddr->py_addr            = ( addr_t )( uintptr_t )p->py;
        paddr->zeta_addr          = ( addr_t )( uintptr_t )p->zeta;

        paddr->psigma_addr        = ( addr_t )( uintptr_t )p->psigma;
        paddr->delta_addr         = ( addr_t )( uintptr_t )p->delta;
        paddr->rpp_addr           = ( addr_t )( uintptr_t )p->rpp;
        paddr->rvv_addr           = ( addr_t )( uintptr_t )p->rvv;
        paddr->chi_addr           = ( addr_t )( uintptr_t )p->chi;
        paddr->charge_ratio_addr  = ( addr_t )( uintptr_t )p->charge_ratio;

        paddr->particle_id_addr   = ( addr_t )( uintptr_t )p->particle_id;
        paddr->at_element_id_addr = ( addr_t )( uintptr_t )p->at_element_id;
        paddr->at_turn_addr       = ( addr_t )( uintptr_t )p->at_turn;
        paddr->state_addr         = ( addr_t )( uintptr_t )p->state;
    }

    return;
}

SIXTRL_INLINE void NS(ParticlesAddr_assign_to_particles)(
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(ParticlesAddr) *const
        SIXTRL_RESTRICT paddr,
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)*  SIXTRL_RESTRICT p )
{
    typedef NS(particle_real_ptr_t)     real_ptr_t;
    typedef NS(particle_index_ptr_t)    index_ptr_t;
    typedef uintptr_t uaddr_t;

    if( ( paddr != SIXTRL_NULLPTR ) && ( p != SIXTRL_NULLPTR ) )
    {
        SIXTRL_ASSERT(
            ( p->num_particles == ( NS(particle_num_elements_t) )0u ) ||
            ( p->num_particles == paddr->num_particles ) );

        SIXTRL_ASSERT( SIXTRL_NULLPTR == p->q0 );
        SIXTRL_ASSERT( SIXTRL_NULLPTR == p->mass0 );
        SIXTRL_ASSERT( SIXTRL_NULLPTR == p->beta0 );
        SIXTRL_ASSERT( SIXTRL_NULLPTR == p->gamma0 );
        SIXTRL_ASSERT( SIXTRL_NULLPTR == p->p0c );
        SIXTRL_ASSERT( SIXTRL_NULLPTR == p->s );
        SIXTRL_ASSERT( SIXTRL_NULLPTR == p->x );
        SIXTRL_ASSERT( SIXTRL_NULLPTR == p->y );
        SIXTRL_ASSERT( SIXTRL_NULLPTR == p->px );
        SIXTRL_ASSERT( SIXTRL_NULLPTR == p->py );
        SIXTRL_ASSERT( SIXTRL_NULLPTR == p->zeta );
        SIXTRL_ASSERT( SIXTRL_NULLPTR == p->psigma );
        SIXTRL_ASSERT( SIXTRL_NULLPTR == p->delta );
        SIXTRL_ASSERT( SIXTRL_NULLPTR == p->rpp );
        SIXTRL_ASSERT( SIXTRL_NULLPTR == p->rvv );
        SIXTRL_ASSERT( SIXTRL_NULLPTR == p->chi );
        SIXTRL_ASSERT( SIXTRL_NULLPTR == p->charge_ratio );
        SIXTRL_ASSERT( SIXTRL_NULLPTR == p->particle_id );
        SIXTRL_ASSERT( SIXTRL_NULLPTR == p->at_element_id );
        SIXTRL_ASSERT( SIXTRL_NULLPTR == p->at_turn );
        SIXTRL_ASSERT( SIXTRL_NULLPTR == p->state );

        p->num_particles = paddr->num_particles;

        p->q0            = ( real_ptr_t )( uaddr_t )paddr->q0_addr;
        p->mass0         = ( real_ptr_t )( uaddr_t )paddr->mass0_addr;
        p->beta0         = ( real_ptr_t )( uaddr_t )paddr->beta0_addr;
        p->gamma0        = ( real_ptr_t )( uaddr_t )paddr->gamma0_addr;
        p->p0c           = ( real_ptr_t )( uaddr_t )paddr->p0c_addr;

        p->s             = ( real_ptr_t )( uaddr_t )paddr->s_addr;
        p->x             = ( real_ptr_t )( uaddr_t )paddr->x_addr;
        p->y             = ( real_ptr_t )( uaddr_t )paddr->y_addr;
        p->px            = ( real_ptr_t )( uaddr_t )paddr->px_addr;
        p->py            = ( real_ptr_t )( uaddr_t )paddr->py_addr;
        p->zeta          = ( real_ptr_t )( uaddr_t )paddr->zeta_addr;

        p->psigma        = ( real_ptr_t )( uaddr_t )paddr->psigma_addr;
        p->delta         = ( real_ptr_t )( uaddr_t )paddr->delta_addr;
        p->rpp           = ( real_ptr_t )( uaddr_t )paddr->rpp_addr;
        p->rvv           = ( real_ptr_t )( uaddr_t )paddr->rvv_addr;
        p->chi           = ( real_ptr_t )( uaddr_t )paddr->chi_addr;
        p->charge_ratio  = ( real_ptr_t )( uaddr_t )paddr->charge_ratio_addr;

        p->particle_id   = ( index_ptr_t )( uaddr_t )paddr->particle_id_addr;
        p->at_element_id = ( index_ptr_t )( uaddr_t )paddr->at_element_id_addr;
        p->at_turn       = ( index_ptr_t )( uaddr_t )paddr->at_turn_addr;
        p->state         = ( index_ptr_t )( uaddr_t )paddr->state_addr;
    }
}

SIXTRL_INLINE void NS(ParticlesAddr_remap_addresses)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(ParticlesAddr)* SIXTRL_RESTRICT p,
    NS(buffer_addr_diff_t) const addr_offset )
{
    typedef NS(buffer_addr_t)       address_t;
    typedef NS(buffer_addr_diff_t)  diff_t;

    SIXTRL_STATIC_VAR address_t const ADDR0 = ( address_t )0u;

    if( p != SIXTRL_NULLPTR )
    {
        if( ( addr_offset == ( NS(buffer_addr_diff_t) )0u ) ||
            ( p->q0_addr    == ADDR0 ) || ( p->mass0_addr  == ADDR0 ) ||
            ( p->beta0_addr == ADDR0 ) || ( p->gamma0_addr == ADDR0 ) ||
            ( p->p0c_addr   == ADDR0 ) || ( p->s_addr      == ADDR0 ) ||
            ( p->x_addr     == ADDR0 ) || ( p->y_addr      == ADDR0 ) ||
            ( p->px_addr    == ADDR0 ) || ( p->py_addr     == ADDR0 ) ||
            ( p->zeta_addr  == ADDR0 ) || ( p->psigma_addr == ADDR0 ) ||
            ( p->delta_addr == ADDR0 ) || ( p->rpp_addr    == ADDR0 ) ||
            ( p->rvv_addr   == ADDR0 ) || ( p->chi_addr    == ADDR0 ) ||
            ( p->charge_ratio_addr == ADDR0 ) ||
            ( p->particle_id_addr  == ADDR0 ) ||
            ( p->at_element_id_addr == ADDR0 ) ||
            ( p->at_turn_addr == ADDR0 ) ||
            ( p->state_addr == ADDR0 ) )
        {
            return;
        }
    }

    if( ( p != SIXTRL_NULLPTR ) && ( addr_offset > ( diff_t )0u ) )
    {
        /* TODO: Prevent overflows by checking value + addr_offset <= MAX */

        p->q0_addr             += addr_offset;
        p->mass0_addr          += addr_offset;
        p->beta0_addr          += addr_offset;
        p->gamma0_addr         += addr_offset;
        p->p0c_addr            += addr_offset;
        p->s_addr              += addr_offset;
        p->x_addr              += addr_offset;
        p->y_addr              += addr_offset;
        p->px_addr             += addr_offset;
        p->py_addr             += addr_offset;
        p->zeta_addr           += addr_offset;
        p->psigma_addr         += addr_offset;
        p->delta_addr          += addr_offset;
        p->rpp_addr            += addr_offset;
        p->rvv_addr            += addr_offset;
        p->chi_addr            += addr_offset;
        p->charge_ratio_addr   += addr_offset;
        p->particle_id_addr    += addr_offset;
        p->at_element_id_addr  += addr_offset;
        p->at_turn_addr        += addr_offset;
        p->state_addr          += addr_offset;
    }
    else if( ( p != SIXTRL_NULLPTR ) && ( addr_offset < ( diff_t )0u ) )
    {
        address_t const _abs = ( address_t )( -addr_offset );

        p->q0_addr = ( p->q0_addr >= _abs ) ? p->q0_addr - _abs : ADDR0;

        p->mass0_addr = ( p->mass0_addr >= _abs )
            ? p->mass0_addr - _abs  : ADDR0;

        p->beta0_addr = ( p->beta0_addr  >= _abs )
            ? p->beta0_addr - _abs  : ADDR0;

        p->gamma0_addr = ( p->gamma0_addr >= _abs )
            ? p->gamma0_addr - _abs : ADDR0;

        p->p0c_addr = ( p->p0c_addr >= _abs ) ? p->p0c_addr - _abs : ADDR0;
        p->s_addr = ( p->s_addr >= _abs ) ? p->s_addr - _abs : ADDR0;
        p->x_addr = ( p->x_addr >= _abs ) ? p->x_addr - _abs : ADDR0;
        p->y_addr = ( p->y_addr >= _abs ) ? p->y_addr - _abs : ADDR0;
        p->px_addr = ( p->px_addr >= _abs ) ? p->px_addr - _abs : ADDR0;
        p->py_addr = ( p->py_addr >= _abs ) ? p->py_addr - _abs : ADDR0;
        p->zeta_addr = ( p->zeta_addr >= _abs ) ? p->zeta_addr - _abs : ADDR0;

        p->psigma_addr = ( p->psigma_addr >= _abs )
            ? p->psigma_addr - _abs : ADDR0;

        p->delta_addr = ( p->delta_addr  >= _abs ) ? p->delta_addr - _abs : ADDR0;
        p->rpp_addr = ( p->rpp_addr >= _abs ) ? p->rpp_addr - _abs : ADDR0;
        p->rvv_addr = ( p->rvv_addr >= _abs ) ? p->rvv_addr - _abs : ADDR0;
        p->chi_addr = ( p->chi_addr >= _abs ) ? p->chi_addr - _abs : ADDR0;

        p->charge_ratio_addr = ( p->charge_ratio_addr  >= _abs )
            ? p->charge_ratio_addr - _abs : ADDR0;

        p->particle_id_addr = ( p->particle_id_addr >= _abs )
            ? p->particle_id_addr - _abs : ADDR0;

        p->at_element_id_addr = ( p->at_element_id_addr >= _abs )
            ? p->at_element_id_addr - _abs : ADDR0;

        p->at_turn_addr = ( p->at_turn_addr >= _abs )
            ? p->at_turn_addr - _abs : ADDR0;

        p->state_addr = ( p->state_addr >= _abs )
            ? p->state_addr - _abs   : ADDR0;
    }

    return;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE void NS(ParticlesAddr_managed_buffer_remap_addresses)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT buffer_begin,
    NS(buffer_size_t) const buffer_index,
    NS(buffer_addr_diff_t) const addr_offset,
    NS(buffer_size_t) const slot_size )
{
    SIXTRL_PARTICLE_ARGPTR_DEC NS(ParticlesAddr)* paddr =
        NS(ParticlesAddr_managed_buffer_get_particle_addr)(
            buffer_begin, buffer_index, slot_size );

    NS(ParticlesAddr_remap_addresses)( paddr, addr_offset );
}

SIXTRL_INLINE void NS(ParticlesAddr_managed_buffer_all_remap_addresses)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT buffer_begin,
    NS(buffer_addr_diff_t) const addr_offset,
    NS(buffer_size_t) const slot_size )
{
    NS(buffer_size_t) const num_paddr_elements =
        NS(ManagedBuffer_get_num_objects)( buffer_begin, slot_size );

    NS(buffer_size_t) ii = ( NS(buffer_size_t) )0u;

    for( ; ii < num_paddr_elements ; ++ii )
    {
        NS(ParticlesAddr_managed_buffer_remap_addresses)(
            buffer_begin, ii, addr_offset, slot_size );
    }

    return;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE int NS(ParticlesAddr_managed_buffer_compare)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT lhs_begin,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT rhs_begin,
    NS(buffer_size_t) const buffer_index, NS(buffer_size_t) const slot_size )
{
    SIXTRL_PARTICLE_ARGPTR_DEC NS(ParticlesAddr) const* lhs =
        NS(ParticlesAddr_managed_buffer_get_const_particle_addr)(
            lhs_begin, buffer_index, slot_size );

    SIXTRL_PARTICLE_ARGPTR_DEC NS(ParticlesAddr) const* rhs =
        NS(ParticlesAddr_managed_buffer_get_const_particle_addr)(
            rhs_begin, buffer_index, slot_size );

    return NS(ParticlesAddr_compare_values)( lhs, rhs );
}

SIXTRL_INLINE int NS(ParticlesAddr_managed_buffer_all_compare)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT lhs_begin,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT rhs_begin,
    NS(buffer_size_t) const slot_size )
{
    int cmp_result = -1;

    NS(buffer_size_t) const lhs_num_elem = NS(ManagedBuffer_get_num_objects)(
        lhs_begin, slot_size );

    NS(buffer_size_t) const rhs_num_elem = NS(ManagedBuffer_get_num_objects)(
        rhs_begin, slot_size );

    if( lhs_num_elem == rhs_num_elem )
    {
        NS(buffer_size_t) ii = ( NS(buffer_size_t) )0u;
        cmp_result = 0;

        for( ; ii < lhs_num_elem ; ++ii )
        {
            SIXTRL_PARTICLE_ARGPTR_DEC NS(ParticlesAddr) const* lhs =
                NS(ParticlesAddr_managed_buffer_get_const_particle_addr)(
                    lhs_begin, ii, slot_size );

            SIXTRL_PARTICLE_ARGPTR_DEC NS(ParticlesAddr) const* rhs =
                NS(ParticlesAddr_managed_buffer_get_const_particle_addr)(
                    rhs_begin, ii, slot_size );

            cmp_result = NS(ParticlesAddr_compare_values)( lhs, rhs );

            if( cmp_result != 0 ) break;
        }
    }
    else if( lhs_num_elem > rhs_num_elem )
    {
        cmp_result = +1;
    }

    return cmp_result;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(ParticlesAddr) const*
NS(BufferIndex_get_const_particles_addr)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC const NS(Object) *const obj )
{
    typedef NS(ParticlesAddr) paddr_t;
    typedef SIXTRL_BUFFER_OBJ_DATAPTR_DEC paddr_t const* ptr_paddr_t;

    ptr_paddr_t ptr_to_paddr = SIXTRL_NULLPTR;

    if( ( obj != SIXTRL_NULLPTR ) &&
        ( NS(Object_get_type_id)( obj ) == NS(OBJECT_TYPE_PARTICLES_ADDR) ) &&
        ( NS(Object_get_size)( obj ) >= sizeof( paddr_t ) ) )
    {
        ptr_to_paddr = ( ptr_paddr_t )( uintptr_t
            )NS(Object_get_begin_addr)( obj );
    }

    return ptr_to_paddr;
}

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(ParticlesAddr)*
NS(BufferIndex_get_particles_addr)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC const NS(Object) *const obj )
{
    typedef SIXTRL_BUFFER_OBJ_DATAPTR_DEC NS(ParticlesAddr)* ptr_paddr_t;
    return ( ptr_paddr_t )NS(BufferIndex_get_const_particles_addr)( obj );
}


SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(ParticlesAddr) const*
NS(ParticlesAddr_managed_buffer_get_const_particle_addr)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const index, NS(buffer_size_t) const slot_size )
{
    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* ptr_obj_t;

    ptr_obj_t ptr_obj = NS(ManagedBuffer_get_const_objects_index_begin)(
        buffer, slot_size );

    SIXTRL_ASSERT( ptr_obj != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( index < NS(ManagedBuffer_get_num_objects)(
        buffer, slot_size ) );

    ptr_obj = ptr_obj + index;

    return NS(BufferIndex_get_const_particles_addr)( ptr_obj );
}

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(ParticlesAddr)*
NS(ParticlesAddr_managed_buffer_get_particle_addr)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const index, NS(buffer_size_t) const slot_size )
{
    return ( SIXTRL_BUFFER_DATAPTR_DEC NS(ParticlesAddr)*
        )NS(ParticlesAddr_managed_buffer_get_const_particle_addr)(
            buffer, index, slot_size );
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE NS(arch_status_t) NS(Particles_managed_buffer_store_addresses)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT paddr_buffer,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT pbuffer,
    NS(buffer_size_t) const index, NS(buffer_size_t) const slot_size )
{
    typedef SIXTRL_BUFFER_DATAPTR_DEC NS(Particles) const* ptr_particles_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC NS(ParticlesAddr)* ptr_paddr_t;

    int success = -1;

    ptr_paddr_t ptr_paddr = NS(ParticlesAddr_managed_buffer_get_particle_addr)(
        paddr_buffer, index, slot_size );

    if( ptr_paddr != SIXTRL_NULLPTR )
    {
        ptr_particles_t p = NS(Particles_managed_buffer_get_const_particles)(
            pbuffer, index, slot_size );

        if( p != SIXTRL_NULLPTR )
        {
            NS(ParticlesAddr_assign_from_particles)( ptr_paddr, p );
        }
        else
        {
            NS(ParticlesAddr_preset)( ptr_paddr );
        }

        success = 0;
    }

    return success;
}

SIXTRL_INLINE NS(arch_status_t)
NS(Particles_managed_buffer_store_addresses_debug)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT paddr_buffer,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT pbuffer,
    NS(buffer_size_t) const index, NS(buffer_size_t) const slot_size,
    SIXTRL_ARGPTR_DEC NS(arch_debugging_t)* SIXTRL_RESTRICT ptr_dbg_register )
{
    typedef SIXTRL_BUFFER_DATAPTR_DEC NS(Particles) const* ptr_particles_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC NS(ParticlesAddr)* ptr_paddr_t;

    NS(arch_status_t) const status =
    NS(Particles_managed_buffer_store_addresses)(
            paddr_buffer, pbuffer, index, slot_size );

    if( ptr_dbg_register != SIXTRL_NULLPTR )
    {
        NS(arch_debugging_t) dbg = SIXTRL_ARCH_DEBUGGING_REGISTER_EMPTY;

        if( status != SIXTRL_ARCH_STATUS_SUCCESS )
        {
            ptr_paddr_t ptr_paddr = SIXTRL_NULLPTR;

            if( slot_size == ( NS(buffer_size_t) )0u )
                dbg = NS(DebugReg_raise_next_error_flag)( dbg );

            if( pbuffer == SIXTRL_NULLPTR )
                    dbg = NS(DebugReg_raise_next_error_flag)( dbg );

            if( paddr_buffer == SIXTRL_NULLPTR )
                    dbg = NS(DebugReg_raise_next_error_flag)( dbg );

            if( NS(ManagedBuffer_needs_remapping)( pbuffer, slot_size ) )
                    dbg = NS(DebugReg_raise_next_error_flag)( dbg );

            if( NS(ManagedBuffer_needs_remapping)( paddr_buffer, slot_size ) )
                    dbg = NS(DebugReg_raise_next_error_flag)( dbg );

            if( NS(ManagedBuffer_get_num_objects)( pbuffer, slot_size ) == 0u )
                dbg = NS(DebugReg_raise_next_error_flag)( dbg );

            if( NS(ManagedBuffer_get_num_objects)( pbuffer, slot_size ) !=
                NS(ManagedBuffer_get_num_objects)( paddr_buffer, slot_size ) )
                dbg = NS(DebugReg_raise_next_error_flag)( dbg );

            if( NS(ManagedBuffer_get_num_objects)(
                    paddr_buffer, slot_size ) <= index )
                dbg = NS(DebugReg_raise_next_error_flag)( dbg );

            ptr_paddr = NS(ParticlesAddr_managed_buffer_get_particle_addr)(
                paddr_buffer, index, slot_size );

            if( ptr_paddr != SIXTRL_NULLPTR )
            {
                ptr_particles_t p =
                    NS(Particles_managed_buffer_get_const_particles)(
                        pbuffer, index, slot_size );

                if( p != SIXTRL_NULLPTR )
                {
                    NS(ParticlesAddr_assign_from_particles)( ptr_paddr, p );
                }
                else
                {
                    NS(ParticlesAddr_preset)( ptr_paddr );
                }
            }
            else
            {
                dbg = NS(DebugReg_raise_next_error_flag)( dbg );
            }
        }

        *ptr_dbg_register = NS(DebugReg_store_arch_status)( dbg, status );
    }

    return status;
}

SIXTRL_INLINE NS(arch_status_t)
NS(Particles_managed_buffer_store_all_addresses)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT paddr_buffer,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT pbuffer,
    NS(buffer_size_t) const slot_size )
{
    NS(arch_status_t) status = SIXTRL_ARCH_STATUS_GENERAL_FAILURE;

    typedef NS(buffer_size_t) buf_size_t;

    buf_size_t const num_objects =
        NS(ManagedBuffer_get_num_objects)( pbuffer, slot_size );

    if( ( num_objects > ( buf_size_t )0u ) && ( num_objects ==
            NS(ManagedBuffer_get_num_objects)( paddr_buffer, slot_size ) ) )
    {
        buf_size_t ii = ( buf_size_t )0u;

        for( ; ii < num_objects ; ++ii )
        {
            status = NS(Particles_managed_buffer_store_addresses)(
                paddr_buffer, pbuffer, ii, slot_size );

            if( status != SIXTRL_ARCH_STATUS_SUCCESS ) break;
        }
    }

    return status;
}

SIXTRL_INLINE NS(arch_status_t)
NS(Particles_managed_buffer_store_all_addresses_debug)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT paddr_buffer,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT pbuffer,
    NS(buffer_size_t) const slot_size,
    SIXTRL_ARGPTR_DEC NS(arch_debugging_t)* SIXTRL_RESTRICT ptr_debug_flag )
{
    NS(arch_status_t) status = SIXTRL_ARCH_STATUS_GENERAL_FAILURE;

    typedef NS(buffer_size_t) buf_size_t;

    buf_size_t const num_objects =
        NS(ManagedBuffer_get_num_objects)( pbuffer, slot_size );

    if( ( num_objects > ( buf_size_t )0u ) && ( num_objects ==
            NS(ManagedBuffer_get_num_objects)( paddr_buffer, slot_size ) ) )
    {
        buf_size_t ii = ( buf_size_t )0u;

        for( ; ii < num_objects ; ++ii )
        {
            status = NS(Particles_managed_buffer_store_addresses_debug)(
                paddr_buffer, pbuffer, ii, slot_size, ptr_debug_flag );

            if( status != SIXTRL_ARCH_STATUS_SUCCESS ) break;
        }
    }

    return status;
}

#if !defined( _GPUCODE )

SIXTRL_INLINE NS(buffer_size_t) NS(ParticlesAddr_get_required_num_dataptrs)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(particle_num_elements_t) const num_particles )
{
    return NS(ParticlesAddr_get_required_num_dataptrs_on_managed_buffer)(
        NS(Buffer_get_const_data_begin)( buffer ), num_particles,
            NS(Buffer_get_slot_size)( buffer ) );
}

SIXTRL_INLINE NS(buffer_size_t) NS(ParticlesAddr_get_required_num_slots)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(particle_num_elements_t) const num_particles )
{
    return NS(ParticlesAddr_get_required_num_slots_on_managed_buffer)(
        NS(Buffer_get_const_data_begin)( buffer ), num_particles,
            NS(Buffer_get_slot_size)( buffer ) );
}

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(ParticlesAddr) const*
NS(ParticlesAddr_buffer_get_const_particle_addr)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const index )
{
    return NS(ParticlesAddr_managed_buffer_get_const_particle_addr)(
        NS(Buffer_get_const_data_begin)( buffer ), index,
        NS(Buffer_get_slot_size)( buffer ) );
}

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(ParticlesAddr)*
NS(ParticlesAddr_buffer_get_particle_addr)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const index )
{
    return ( SIXTRL_BUFFER_DATAPTR_DEC NS(ParticlesAddr)*
        )NS(ParticlesAddr_buffer_get_const_particle_addr)( buffer, index );
}

/* ------------------------------------------------------------------------ */

SIXTRL_INLINE NS(arch_status_t) NS(ParticlesAddr_buffer_clear_single)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT paddr_buffer, 
    NS(buffer_size_t) const index )
{
    NS(arch_status_t) status = SIXTRL_ARCH_STATUS_GENERAL_FAILURE;
    
    SIXTRL_PARTICLE_ARGPTR_DEC NS(ParticlesAddr)* particles_addr = 
        NS(ParticlesAddr_buffer_get_particle_addr)( paddr_buffer, index );
        
    if( particles_addr != SIXTRL_NULLPTR )
    {
        NS(ParticlesAddr_clear)( particles_addr );
        status = SIXTRL_ARCH_STATUS_SUCCESS;
    }
    
    return status;
}

SIXTRL_INLINE NS(arch_status_t) NS(ParticlesAddr_buffer_clear_all)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT paddr_buffer )
{
    NS(arch_status_t) status = SIXTRL_ARCH_STATUS_GENERAL_FAILURE;
    NS(buffer_size_t) const num_objects = 
        NS(Buffer_get_num_of_objects)( paddr_buffer );
        
    if( num_objects > ( NS(buffer_size_t) )0u )
    {
        NS(buffer_size_t) ii = ( NS(buffer_size_t) )0u;
        NS(buffer_size_t) num_cleared_objs = ( NS(buffer_size_t) )0u;
    
        for( ; ii < num_objects ; ++ii )
        {
            SIXTRL_PARTICLE_ARGPTR_DEC NS(ParticlesAddr)* particles_addr = 
                NS(ParticlesAddr_buffer_get_particle_addr)( 
                    paddr_buffer, ii );
                
            if( particles_addr != SIXTRL_NULLPTR )
            {
                NS(ParticlesAddr_clear)( particles_addr );
                ++num_cleared_objs;
            }
        }
        
        if( num_cleared_objs > ( NS(buffer_size_t) )0u )
        {
            status = SIXTRL_ARCH_STATUS_SUCCESS;
        }
    }
    
    return status;
}

/* ----------------------------------------------------------------------- */

SIXTRL_INLINE bool NS(ParticlesAddr_can_be_added)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(particle_num_elements_t) const num_particles,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_objects,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_slots,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_dataptrs )
{
    NS(buffer_size_t) const num_dataptrs =
        NS(ParticlesAddr_get_required_num_dataptrs)( buffer, num_particles );

    SIXTRL_ASSERT( num_dataptrs == ( NS(buffer_size_t) )0u );

    return NS(Buffer_can_add_object)( buffer, sizeof( NS(ParticlesAddr) ),
        num_dataptrs, SIXTRL_NULLPTR, SIXTRL_NULLPTR, requ_objects,
            requ_slots, requ_dataptrs );
}

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(ParticlesAddr)*
NS(ParticlesAddr_new)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(particle_num_elements_t) const num_particles )
{
    typedef NS(ParticlesAddr) paddr_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC paddr_t* ptr_to_paddr_t;

    SIXTRL_ASSERT( buffer != SIXTRL_NULLPTR );

    paddr_t paddr;
    NS(ParticlesAddr_preset)( &paddr );
    paddr.num_particles = num_particles;

    return ( ptr_to_paddr_t )( uintptr_t )NS(Object_get_begin_addr)(
        NS(Buffer_add_object)( buffer, &paddr, sizeof( NS(ParticlesAddr) ),
            NS(OBJECT_TYPE_PARTICLES_ADDR), ( NS(buffer_size_t) )0u,
                SIXTRL_NULLPTR, SIXTRL_NULLPTR, SIXTRL_NULLPTR ) );
}

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(ParticlesAddr)*
NS(ParticlesAddr_add)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(particle_num_elements_t) const num_particles,
    NS(buffer_addr_t) const q0_addr,    NS(buffer_addr_t)  const mass0_addr,
    NS(buffer_addr_t) const beta0_addr, NS(buffer_addr_t)  const gamma0_addr,
    NS(buffer_addr_t) const p0c_addr,   NS(buffer_addr_t)  const s_addr,
    NS(buffer_addr_t) const x_addr,     NS(buffer_addr_t)  const y_addr,
    NS(buffer_addr_t) const px_addr,    NS(buffer_addr_t)  const py_addr,
    NS(buffer_addr_t) const zeta_addr,  NS(buffer_addr_t)  const psigma_addr,
    NS(buffer_addr_t) const delta_addr, NS(buffer_addr_t)  const rpp_addr,
    NS(buffer_addr_t) const rvv_addr,   NS(buffer_addr_t)  const chi_addr,
    NS(buffer_addr_t) const charge_ratio_addr,
    NS(buffer_addr_t) const particle_id_addr,
    NS(buffer_addr_t) const at_element_id_addr,
    NS(buffer_addr_t) const at_turn_addr, NS(buffer_addr_t) const state_addr )
{
    typedef NS(ParticlesAddr) paddr_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC paddr_t* ptr_to_paddr_t;

    SIXTRL_ASSERT( buffer != SIXTRL_NULLPTR );

    paddr_t paddr;
    NS(ParticlesAddr_preset)( &paddr );

    paddr.num_particles      = num_particles;
    paddr.q0_addr            = q0_addr;
    paddr.mass0_addr         = mass0_addr;
    paddr.beta0_addr         = beta0_addr;
    paddr.gamma0_addr        = gamma0_addr;
    paddr.p0c_addr           = p0c_addr;

    paddr.s_addr             = s_addr;
    paddr.x_addr             = x_addr;
    paddr.y_addr             = y_addr;
    paddr.px_addr            = px_addr;
    paddr.py_addr            = py_addr;
    paddr.zeta_addr          = zeta_addr;

    paddr.psigma_addr        = psigma_addr;
    paddr.delta_addr         = delta_addr;
    paddr.rpp_addr           = rpp_addr;
    paddr.rvv_addr           = rvv_addr;
    paddr.chi_addr           = chi_addr;
    paddr.charge_ratio_addr  = charge_ratio_addr;

    paddr.particle_id_addr   = particle_id_addr;
    paddr.at_element_id_addr = at_element_id_addr;
    paddr.at_turn_addr       = at_turn_addr;
    paddr.state_addr         = state_addr;

    return ( ptr_to_paddr_t )( uintptr_t )NS(Object_get_begin_addr)(
        NS(Buffer_add_object)( buffer, &paddr, sizeof( NS(ParticlesAddr) ),
            NS(OBJECT_TYPE_PARTICLES_ADDR), ( NS(buffer_size_t) )0u,
                SIXTRL_NULLPTR, SIXTRL_NULLPTR, SIXTRL_NULLPTR ) );
}

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(ParticlesAddr)*
NS(ParticlesAddr_add_copy)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(ParticlesAddr) *const SIXTRL_RESTRICT p )
{
    if( p != SIXTRL_NULLPTR )
    {
        return NS(ParticlesAddr_add)( buffer, p->num_particles,
            p->q0_addr, p->mass0_addr, p->beta0_addr,
            p->gamma0_addr, p->p0c_addr,

            p->s_addr, p->x_addr, p->y_addr, p->px_addr,
            p->py_addr, p->zeta_addr,

            p->psigma_addr, p->delta_addr, p->rpp_addr, p->rvv_addr,
            p->chi_addr, p->charge_ratio_addr,

            p->particle_id_addr, p->at_element_id_addr, p->at_turn_addr,
            p->state_addr );
    }

    return SIXTRL_NULLPTR;
}

#endif /* !defined( _GPUCODE ) */

#if defined( __cplusplus ) && !defined( _GPUCODE )
}
#endif /* C++, Host */

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
/* Helper functions: */

#endif /* SIXTRACKLIB_COMMON_PARTICLES_PARTICLES_ADDR_H__ */

/* end: sixtracklib/common/particles/particles_addr.h */
