#include "sixtracklib/testlib/testdata/track_testdata.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/particles.h"
#include "sixtracklib/common/beam_elements.h"
#include "sixtracklib/common/buffer.h"

/* ************************************************************************ */
/* ******              Implementation of functions                   ****** */
/* ************************************************************************ */

SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)*
NS(TrackTestdata_extract_initial_particles_buffer)( const char path_to_file[] )
{
    typedef NS(buffer_size_t) buf_size_t;
    typedef NS(Particles)     particles_t;

    NS(Buffer)* init_particles_buffer = SIXTRL_NULLPTR;
    NS(Buffer)* orig_buffer = NS(Buffer_new_from_file)( path_to_file );

    if( orig_buffer != SIXTRL_NULLPTR )
    {
        NS(Object) const* obj_begin =
            NS(Buffer_get_const_objects_begin)( orig_buffer );

        NS(Object) const* obj_it = obj_begin;

        NS(Object) const* obj_end   =
            NS(Buffer_get_const_objects_end)( orig_buffer );

        buf_size_t const slot_size = NS(Buffer_get_slot_size)( orig_buffer );

        buf_size_t num_objects  = ( buf_size_t )0u;
        buf_size_t num_dataptrs = ( buf_size_t )0u;
        buf_size_t num_slots    = ( buf_size_t )0u;
        buf_size_t num_garbage  = ( buf_size_t )0u;

        for( ; obj_it != obj_end ; ++obj_it )
        {
            if( NS(Object_get_type_id)( obj_it ) == NS(OBJECT_TYPE_PARTICLE) )
            {
                buf_size_t const obj_size = NS(Object_get_size)( obj_it );
                buf_size_t num_additional_slots = obj_size / slot_size;

                particles_t const* p = ( particles_t const* )( uintptr_t
                    )NS(Object_get_begin_addr)( obj_it );

                if( ( num_additional_slots * slot_size ) < obj_size )
                {
                    ++num_additional_slots;
                }

                ++num_objects;

                num_slots    += num_additional_slots;
                num_dataptrs += NS(Particles_get_num_dataptrs)( p );

                SIXTRL_ASSERT( p != SIXTRL_NULLPTR );
            }
            else
            {
                break;
            }
        }

        if( num_objects > ( buf_size_t )0u )
        {
            buf_size_t const requ_buffer_size =
                NS(ManagedBuffer_calculate_buffer_length)( SIXTRL_NULLPTR,
                   num_objects, num_slots, num_dataptrs, num_garbage,
                       slot_size );

            int success = 0;

            init_particles_buffer = NS(Buffer_new)( requ_buffer_size );

            success = NS(Buffer_reserve)( init_particles_buffer, num_objects,
                num_slots, num_dataptrs, num_garbage );

            if( success == 0 )
            {
                buf_size_t ii = ( buf_size_t )0u;
                obj_it = obj_begin;

                for( ; ii < num_objects ; ++ii, ++obj_it )
                {
                    particles_t const* p = ( particles_t const* )( uintptr_t
                        )NS(Object_get_begin_addr)( obj_it );

                    particles_t* copied_particle = NS(Particles_add_copy)(
                        init_particles_buffer, p );

                    if( copied_particle == SIXTRL_NULLPTR )
                    {
                        success = -1;
                        break;
                    }
                }
            }

            if( success != 0 )
            {
                NS(Buffer_delete)( init_particles_buffer );
                init_particles_buffer = SIXTRL_NULLPTR;
            }
        }
    }

    NS(Buffer_delete)( orig_buffer );
    orig_buffer = SIXTRL_NULLPTR;

    return init_particles_buffer;
}

SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)*
NS(TrackTestdata_extract_result_particles_buffer)( const char path_to_file[] )
{
    typedef NS(buffer_size_t) buf_size_t;
    typedef NS(Particles)     particles_t;

    NS(Buffer)* result_particles_buffer = SIXTRL_NULLPTR;
    NS(Buffer)* orig_buffer = NS(Buffer_new_from_file)( path_to_file );

    if( orig_buffer != SIXTRL_NULLPTR )
    {
        NS(Object) const* orig_begin =
            NS(Buffer_get_const_objects_begin)( orig_buffer );

        buf_size_t const orig_num_objects =
            NS(Buffer_get_num_of_objects)( orig_buffer );

        NS(Object) const* rev_orig_it =
            ( ( orig_num_objects > ( buf_size_t )0u ) &&
              ( orig_begin != SIXTRL_NULLPTR ) )
                ? ( orig_begin + ( orig_num_objects - ( buf_size_t )1u ) )
                : ( orig_begin );

        NS(Object) const* rev_orig_end =
            ( ( orig_num_objects > ( buf_size_t )0u ) &&
              ( orig_begin != SIXTRL_NULLPTR ) &&
              ( ( ( uintptr_t )orig_begin ) >
                  ( uintptr_t )sizeof( NS(Object) ) ) )
                ? ( orig_begin - ( buf_size_t )1u ) : ( orig_begin );

        buf_size_t const slot_size = NS(Buffer_get_slot_size)( orig_buffer );

        buf_size_t num_objects  = ( buf_size_t )0u;
        buf_size_t num_dataptrs = ( buf_size_t )0u;
        buf_size_t num_slots    = ( buf_size_t )0u;
        buf_size_t num_garbage  = ( buf_size_t )0u;

        for( ; rev_orig_it != rev_orig_end ; --rev_orig_it )
        {
            if( NS(Object_get_type_id)( rev_orig_it ) == NS(OBJECT_TYPE_PARTICLE) )
            {
                buf_size_t const obj_size = NS(Object_get_size)( rev_orig_it );
                buf_size_t num_additional_slots = obj_size / slot_size;

                particles_t const* p = ( particles_t const* )( uintptr_t
                    )NS(Object_get_begin_addr)( rev_orig_it );

                if( ( num_additional_slots * slot_size ) < obj_size )
                {
                    ++num_additional_slots;
                }

                ++num_objects;

                num_slots    += num_additional_slots;
                num_dataptrs += NS(Particles_get_num_dataptrs)( p );

                SIXTRL_ASSERT( p != SIXTRL_NULLPTR );
            }
            else
            {
                break;
            }
        }

        if( ( num_objects >  ( buf_size_t )0u ) &&
            ( num_objects <= orig_num_objects ) )
        {
            buf_size_t const requ_buffer_size =
                NS(ManagedBuffer_calculate_buffer_length)( SIXTRL_NULLPTR,
                   num_objects, num_slots, num_dataptrs, num_garbage,
                       slot_size );

            int success = 0;

            result_particles_buffer = NS(Buffer_new)( requ_buffer_size );

            success = NS(Buffer_reserve)( result_particles_buffer, num_objects,
                num_slots, num_dataptrs, num_garbage );

            if( success == 0 )
            {
                buf_size_t ii = ( buf_size_t )0u;
                NS(Object) const* obj_it =
                    orig_begin + ( orig_num_objects - num_objects );

                for( ; ii < num_objects ; ++ii, ++obj_it )
                {
                    particles_t const* p = ( particles_t const* )( uintptr_t
                        )NS(Object_get_begin_addr)( obj_it );

                    particles_t* copied_particle = NS(Particles_add_copy)(
                        result_particles_buffer, p );

                    if( copied_particle == SIXTRL_NULLPTR )
                    {
                        success = -1;
                        break;
                    }
                }
            }

            if( success != 0 )
            {
                NS(Buffer_delete)( result_particles_buffer );
                result_particles_buffer = SIXTRL_NULLPTR;
            }
        }
    }

    NS(Buffer_delete)( orig_buffer );
    orig_buffer = SIXTRL_NULLPTR;

    return result_particles_buffer;
}

SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)*
NS(TrackTestdata_extract_beam_elements_buffer)( const char path_to_file[] )
{
    typedef NS(buffer_size_t) buf_size_t;
    typedef NS(buffer_addr_t) address_t;

    NS(Buffer)* beam_elements_buffer = SIXTRL_NULLPTR;
    NS(Buffer)* orig_buffer = NS(Buffer_new_from_file)( path_to_file );

    if( orig_buffer != SIXTRL_NULLPTR )
    {
        buf_size_t const slot_size = NS(Buffer_get_slot_size)( orig_buffer );

        NS(Object) const* obj_end   =
            NS(Buffer_get_const_objects_end)( orig_buffer );

        NS(Object) const* obj_it = NS(Buffer_get_const_objects_begin)(
            orig_buffer );

        NS(Object) const* orig_be_begin = obj_it;
        NS(Object) const* orig_be_end   = obj_end;

        buf_size_t num_objects  = ( buf_size_t )0u;
        buf_size_t num_dataptrs = ( buf_size_t )0u;
        buf_size_t num_slots    = ( buf_size_t )0u;
        buf_size_t num_garbage  = ( buf_size_t )0u;

        bool in_beam_elements = false;

        for( ; obj_it != obj_end ; ++obj_it )
        {
            NS(object_type_id_t) const type_id =
                NS(Object_get_type_id)( obj_it );

            if( ( type_id != NS(OBJECT_TYPE_PARTICLE) ) && ( !in_beam_elements ) )
            {
                in_beam_elements = true;
                orig_be_begin = obj_it;
            }
            else if( ( type_id == NS(OBJECT_TYPE_PARTICLE) ) &&
                     ( in_beam_elements ) )
            {
                in_beam_elements = false;
                orig_be_end = obj_it;
                break;
            }
            else if( ( type_id == NS(OBJECT_TYPE_PARTICLE) ) &&
                     ( !in_beam_elements ) )
            {
                continue;
            }

            if( type_id != NS(OBJECT_TYPE_PARTICLE ) )
            {
                address_t const beginaddr = NS(Object_get_begin_addr)( obj_it );
                buf_size_t const obj_size = NS(Object_get_size)( obj_it );

                buf_size_t num_additional_slots = obj_size / slot_size;

                if( ( num_additional_slots * slot_size ) < obj_size )
                {
                    ++num_additional_slots;
                }

                switch( type_id )
                {
                    case NS(OBJECT_TYPE_DRIFT):
                    {
                        typedef NS(Drift) belem_t;
                        num_dataptrs += NS(Drift_num_dataptrs)(
                            ( belem_t const* )( uintptr_t )beginaddr );

                        break;
                    }

                    case NS(OBJECT_TYPE_DRIFT_EXACT):
                    {
                        typedef NS(DriftExact) belem_t;
                        num_dataptrs += NS(DriftExact_num_dataptrs)(
                            ( belem_t const* )( uintptr_t )beginaddr );

                        break;
                    }

                    case NS(OBJECT_TYPE_MULTIPOLE):
                    {
                        typedef NS(Multipole) belem_t;
                        num_dataptrs += NS(Multipole_num_dataptrs)(
                            ( belem_t const* )( uintptr_t )beginaddr );

                        break;
                    }

                    case NS(OBJECT_TYPE_XYSHIFT):
                    {
                        typedef SIXTRL_BE_ARGPTR_DEC NS(XYShift) const* ptr_t;
                        num_dataptrs += NS(XYShift_num_dataptrs)(
                            ( ptr_t )( uintptr_t )beginaddr );

                        break;
                    }

                    case NS(OBJECT_TYPE_SROTATION):
                    {
                        typedef SIXTRL_BE_ARGPTR_DEC NS(SRotation) const* ptr_t;
                        num_dataptrs += NS(SRotation_num_dataptrs)(
                            ( ptr_t )( uintptr_t )beginaddr );

                        break;
                    }

                    case NS(OBJECT_TYPE_CAVITY):
                    {
                        typedef SIXTRL_BE_ARGPTR_DEC NS(Cavity) const* ptr_t;
                        num_dataptrs += NS(Cavity_num_dataptrs)(
                            ( ptr_t )( uintptr_t )beginaddr );
                        break;
                    }

                    default:
                    {
                        num_dataptrs += ( buf_size_t )0u;
                    }
                };

                ++num_objects;
                num_slots += num_additional_slots;
            }
            else
            {
                break;
            }
        }

        if( num_objects > ( buf_size_t )0u )
        {
            buf_size_t const requ_buffer_size =
                NS(ManagedBuffer_calculate_buffer_length)( SIXTRL_NULLPTR,
                   num_objects, num_slots, num_dataptrs, num_garbage,
                       slot_size );

            int success = 0;

            beam_elements_buffer = NS(Buffer_new)( requ_buffer_size );

            success = NS(Buffer_reserve)( beam_elements_buffer, num_objects,
                num_slots, num_dataptrs, num_garbage );

            if( success == 0 )
            {
                obj_it = orig_be_begin;

                for( ; obj_it < orig_be_end ; ++obj_it )
                {
                    address_t const beginaddr =
                        NS(Object_get_begin_addr)( obj_it );

                    NS(object_type_id_t) const type_id =
                        NS(Object_get_type_id)( obj_it );

                    switch( type_id )
                    {
                        case NS(OBJECT_TYPE_DRIFT):
                        {
                            SIXTRL_BE_ARGPTR_DEC NS(Drift) const* orig = (
                                SIXTRL_BE_ARGPTR_DEC NS(Drift) const* )(
                                    uintptr_t )beginaddr;

                            SIXTRL_BE_ARGPTR_DEC NS(Drift)* copied_elem =
                                NS(Drift_add_copy)( beam_elements_buffer, orig );

                            if( copied_elem == SIXTRL_NULLPTR ) success = -1;
                            break;
                        }

                        case NS(OBJECT_TYPE_DRIFT_EXACT):
                        {
                            SIXTRL_BE_ARGPTR_DEC NS(DriftExact) const* orig = (
                                SIXTRL_BE_ARGPTR_DEC NS(DriftExact) const* )(
                                    uintptr_t )beginaddr;

                            SIXTRL_BE_ARGPTR_DEC NS(DriftExact)* copied_elem =
                                NS(DriftExact_add_copy)(
                                    beam_elements_buffer, orig );

                            if( copied_elem == SIXTRL_NULLPTR ) success = -1;
                            break;
                        }

                        case NS(OBJECT_TYPE_MULTIPOLE):
                        {
                            typedef SIXTRL_BE_ARGPTR_DEC NS(Multipole) const*
                                    ptr_elem_t;

                            SIXTRL_BE_ARGPTR_DEC NS(Multipole)* copied_belem =
                            NS(Multipole_add_copy)( beam_elements_buffer,
                                ( ptr_elem_t )( uintptr_t )beginaddr );
                            if( copied_belem == SIXTRL_NULLPTR ) success = -1;
                            break;
                        }

                        case NS(OBJECT_TYPE_XYSHIFT):
                        {
                            typedef NS(XYShift) elem_t;
                            typedef SIXTRL_BE_ARGPTR_DEC elem_t* ptr_dest_t;
                            typedef SIXTRL_BE_ARGPTR_DEC elem_t const* ptr_t;
                            ptr_t orig = ( ptr_t )( uintptr_t )beginaddr;
                            ptr_dest_t copied_belem = NS(XYShift_add_copy)(
                                beam_elements_buffer, orig );
                            if( copied_belem == SIXTRL_NULLPTR ) success = -1;
                            break;
                        }

                        case NS(OBJECT_TYPE_SROTATION):
                        {
                            typedef NS(SRotation) elem_t;
                            typedef SIXTRL_BE_ARGPTR_DEC elem_t* ptr_dest_t;
                            typedef SIXTRL_BE_ARGPTR_DEC elem_t const* ptr_t;
                            ptr_t orig = ( ptr_t )( uintptr_t )beginaddr;
                            ptr_dest_t copied_belem = NS(SRotation_add_copy)(
                                beam_elements_buffer, orig );
                            if( copied_belem == SIXTRL_NULLPTR ) success = -1;
                            break;
                        }

                        case NS(OBJECT_TYPE_CAVITY):
                        {
                            typedef SIXTRL_BE_ARGPTR_DEC NS(Cavity) const* ptr_t;
                            success = ( SIXTRL_NULLPTR != NS(Cavity_add_copy)(
                                beam_elements_buffer, ( ptr_t )( uintptr_t
                                    )beginaddr ) )
                                ? ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS
                                : ( NS(arch_status_t)
                                    )SIXTRL_ARCH_STATUS_GENERAL_FAILURE;
                            break;
                        }

                        default:
                        {
                            success = -1;
                        }
                    };

                    if( success != 0 ) break;
                }
            }

            if( success != 0 )
            {
                NS(Buffer_delete)( beam_elements_buffer );
                beam_elements_buffer = SIXTRL_NULLPTR;
            }
        }
    }

    NS(Buffer_delete)( orig_buffer );
    orig_buffer = SIXTRL_NULLPTR;

    return beam_elements_buffer;
}

/* -------------------------------------------------------------------------- */

void NS(TrackTestdata_generate_fodo_lattice)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    unsigned int const num_turns )
{
    NS(Multipole)* dipole0 = SIXTRL_NULLPTR;
    NS(Multipole)* dipole1 = SIXTRL_NULLPTR;
    NS(Multipole)* q0 = SIXTRL_NULLPTR;
    NS(Multipole)* q1 = SIXTRL_NULLPTR;
    NS(Cavity)* cavity = SIXTRL_NULLPTR;
    NS(BeamMonitor)* monitor = SIXTRL_NULLPTR;

    unsigned int const monitor_num_stores = ( 100u > num_turns )
        ? num_turns : 100u;

    dipole0 = NS(Multipole_add)( buffer, 0, 1.0, 0.1570796327, 0.0, 0u );
    NS(Multipole_set_knl_value)( dipole0, 0, 0.1570796327 );
    NS(Drift_add)( buffer, 5.0 );


    q0 = NS(Multipole_add)( buffer, 1, 0.0, 0.0, 0.0, 0u );
    NS(Multipole_set_knl_value)( q0, 1, 0.1657145946 );

    NS(Drift_add)( buffer, 5.0 );

    dipole1 = NS(Multipole_add)( buffer, 0, 1.0, 0.1570796327, 0.0, 0u );
    NS(Multipole_set_knl_value)( dipole1, 0, 0.1570796327 );

    NS(Drift_add)( buffer, 5.0 );

    q1 = NS(Multipole_add)( buffer, 1, 0.0, 0.0, 0.0, 0u );
    NS(Multipole_set_knl_value)( q1, 1, -0.1685973315 );

    NS(Drift_add)( buffer, 5.0 );

    cavity = NS(Cavity_new)( buffer );
    NS(Cavity_set_voltage)( cavity, 5000000.0 );
    NS(Cavity_set_frequency)( cavity, 239833966.4 );
    NS(Cavity_set_lag)( cavity, 180.0 );

    monitor = NS(BeamMonitor_new)( buffer );
    NS(BeamMonitor_set_num_stores)( monitor, monitor_num_stores);
    NS(BeamMonitor_set_start)( monitor, 0 );
    NS(BeamMonitor_set_skip)( monitor, 0 );
    NS(BeamMonitor_set_is_rolling)( monitor, true );
}


void NS(TrackTestdata_generate_particle_distr_x)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    unsigned int const num_particles, double const p0c,
    double const min_x, double const max_x, double const mass0,
    double const q0, double const chi, double const charge_ratio ) {

    NS(Particles)* pset = NS(Particles_new)( buffer, num_particles );

    double const energy0 = sqrt( p0c * p0c + mass0 * mass0 );
    double const beta0   = p0c / energy0;
    double const gamma0  = ( double )1.0 / sqrt( 1.0 - beta0 * beta0 );
    double const dx      = ( max_x - min_x ) / ( double )( num_particles - 1 );

    for( unsigned ii = 0 ; ii < num_particles ; ++ii )
    {
        NS(Particles_set_q0_value)( pset, ii, q0 );
        NS(Particles_set_mass0_value)( pset, ii, mass0 );
        NS(Particles_set_beta0_value)( pset, ii, beta0 );
        NS(Particles_set_gamma0_value)( pset, ii, gamma0 );
        NS(Particles_set_p0c_value)( pset, ii, p0c );

        NS(Particles_set_s_value)( pset, ii, 0.0 );
        NS(Particles_set_x_value)( pset, ii, dx * ii );
        NS(Particles_set_px_value)( pset, ii, 0.0 );
        NS(Particles_set_y_value)( pset, ii, 0.0 );
        NS(Particles_set_py_value)( pset, ii, 0.0 );
        NS(Particles_set_zeta_value)( pset, ii, 0.0 );
        NS(Particles_set_delta_value)( pset, ii, 0.0 );

        NS(Particles_set_psigma_value)( pset, ii, 0.0 );
        NS(Particles_set_rvv_value)( pset, ii, 1.0 );
        NS(Particles_set_rpp_value)( pset, ii, 1.0 );
        NS(Particles_set_chi_value)( pset, ii, 1.0 );
        NS(Particles_set_charge_ratio_value)( pset, ii, 1.0 );

        NS(Particles_set_particle_id_value)( pset, ii, ii );
        NS(Particles_set_at_element_id_value)( pset, ii, 0 );
        NS(Particles_set_at_turn_value)( pset, ii, 0 );
        NS(Particles_set_state_value)( pset, ii, 1 );
    }
}
