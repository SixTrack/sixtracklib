#include "sixtracklib/testlib/track_testdata_tools.h"

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "sixtracklib/_impl/definitions.h"
#include "sixtracklib/common/particles.h"
#include "sixtracklib/common/beam_elements.h"
#include "sixtracklib/common/buffer.h"

extern SIXTRL_HOST_FN NS(Buffer)*
    NS(TrackTestdata_extract_initial_particles_buffer)( const char path_to_file[] );

extern SIXTRL_HOST_FN NS(Buffer)*
    NS(TrackTestdata_extract_result_particles_buffer)( const char path_to_file[] );

extern SIXTRL_HOST_FN NS(Buffer)*
    NS(TrackTestdata_extract_beam_elements_buffer)( const char path_to_file[] );


/* ************************************************************************ */
/* ******              Implementation of functions                   ****** */
/* ************************************************************************ */

SIXTRL_HOST_FN NS(Buffer)* NS(TrackTestdata_extract_initial_particles_buffer)(
    const char path_to_file[] )
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

SIXTRL_HOST_FN NS(Buffer)*
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

SIXTRL_HOST_FN NS(Buffer)*
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
                        num_dataptrs += NS(Drift_get_num_dataptrs)(
                            ( belem_t const* )( uintptr_t )beginaddr );

                        break;
                    }

                    case NS(OBJECT_TYPE_DRIFT_EXACT):
                    {
                        typedef NS(DriftExact) belem_t;
                        num_dataptrs += NS(DriftExact_get_num_dataptrs)(
                            ( belem_t const* )( uintptr_t )beginaddr );

                        break;
                    }

                    case NS(OBJECT_TYPE_MULTIPOLE):
                    {
                        typedef NS(MultiPole) belem_t;
                        num_dataptrs += NS(MultiPole_get_num_dataptrs)(
                            ( belem_t const* )( uintptr_t )beginaddr );

                        break;
                    }

                    case NS(OBJECT_TYPE_XYSHIFT):
                    {
                        typedef NS(XYShift) belem_t;
                        num_dataptrs += NS(XYShift_get_num_dataptrs)(
                            ( belem_t const* )( uintptr_t )beginaddr );

                        break;
                    }

                    case NS(OBJECT_TYPE_SROTATION):
                    {
                        typedef NS(SRotation) belem_t;
                        num_dataptrs += NS(SRotation_get_num_dataptrs)(
                            ( belem_t const* )( uintptr_t )beginaddr );

                        break;
                    }

                    case NS(OBJECT_TYPE_CAVITY):
                    {
                        typedef NS(Cavity) belem_t;
                        num_dataptrs += NS(Cavity_get_num_dataptrs)(
                            ( belem_t const* )( uintptr_t )beginaddr );

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
                            typedef NS(Drift) belem_t;

                            belem_t const* orig_belem =
                                ( belem_t const* )( uintptr_t )beginaddr;

                            belem_t* copied_belem = NS(Drift_add)(
                                beam_elements_buffer,
                                NS(Drift_get_length)( orig_belem ) );

                            if( copied_belem == SIXTRL_NULLPTR ) success = -1;
                            break;
                        }

                        case NS(OBJECT_TYPE_DRIFT_EXACT):
                        {
                            typedef NS(DriftExact) belem_t;

                            belem_t const* orig_belem =
                                ( belem_t const* )( uintptr_t )beginaddr;

                            belem_t* copied_belem = NS(DriftExact_add)(
                                beam_elements_buffer,
                                NS(DriftExact_get_length)( orig_belem ) );

                            if( copied_belem == SIXTRL_NULLPTR ) success = -1;
                            break;
                        }

                        case NS(OBJECT_TYPE_MULTIPOLE):
                        {
                            typedef NS(MultiPole) belem_t;

                            belem_t const* orig_belem =
                                ( belem_t const* )( uintptr_t )beginaddr;

                            belem_t* copied_belem = NS(MultiPole_add)(
                                beam_elements_buffer,
                                NS(MultiPole_get_order)( orig_belem ),
                                NS(MultiPole_get_const_bal)( orig_belem ),
                                NS(MultiPole_get_length)( orig_belem ),
                                NS(MultiPole_get_hxl)( orig_belem ),
                                NS(MultiPole_get_hyl)( orig_belem ) );

                            if( copied_belem == SIXTRL_NULLPTR ) success = -1;
                            break;
                        }

                        case NS(OBJECT_TYPE_XYSHIFT):
                        {
                            typedef NS(XYShift) belem_t;

                            belem_t const* orig_belem =
                                ( belem_t const* )( uintptr_t )beginaddr;

                            belem_t* copied_belem = NS(XYShift_add)(
                                beam_elements_buffer,
                                NS(XYShift_get_dx)( orig_belem ),
                                NS(XYShift_get_dy)( orig_belem ) );

                            if( copied_belem == SIXTRL_NULLPTR ) success = -1;
                            break;
                        }

                        case NS(OBJECT_TYPE_SROTATION):
                        {
                            typedef NS(SRotation) belem_t;

                            belem_t const* orig_belem =
                                ( belem_t const* )( uintptr_t )beginaddr;

                            belem_t* copied_belem = NS(SRotation_add_detailed)(
                                beam_elements_buffer,
                                NS(SRotation_get_cos_angle)( orig_belem ),
                                NS(SRotation_get_sin_angle)( orig_belem ) );

                            if( copied_belem == SIXTRL_NULLPTR ) success = -1;
                            break;
                        }

                        case NS(OBJECT_TYPE_CAVITY):
                        {
                            typedef NS(Cavity) belem_t;

                            belem_t const* orig_belem =
                                ( belem_t const* )( uintptr_t )beginaddr;

                            belem_t* copied_belem = NS(Cavity_add)(
                                beam_elements_buffer,
                                NS(Cavity_get_voltage)( orig_belem ),
                                NS(Cavity_get_frequency)( orig_belem ),
                                NS(Cavity_get_lag)( orig_belem ) );

                            if( copied_belem == SIXTRL_NULLPTR ) success = -1;
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

/* end: tests/sixtracklib/testlib/details/track_testdata_tools.c */
