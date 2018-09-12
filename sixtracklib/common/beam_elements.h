#ifndef SIXTRACKLIB_COMMON_BEAM_ELEMENTS_H__
#define SIXTRACKLIB_COMMON_BEAM_ELEMENTS_H__

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/_impl/definitions.h"
    #include "sixtracklib/common/impl/buffer_defines.h"
    #include "sixtracklib/common/impl/beam_elements_defines.h"
    #include "sixtracklib/common/impl/be_drift.h"
    #include "sixtracklib/common/impl/be_multipole.h"
    #include "sixtracklib/common/impl/be_xyshift.h"
    #include "sixtracklib/common/impl/be_srotation.h"
    #include "sixtracklib/common/impl/be_cavity.h"

    #include "sixtracklib/common/impl/buffer_object.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined( _GPUCODE ) && defined( __cplusplus ) */

/* ========================================================================= */

SIXTRL_FN SIXTRL_STATIC int NS(BeamElements_calc_buffer_parameters_for_object)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC const NS(Object) *const SIXTRL_RESTRICT obj,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT num_objects,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT num_slots,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT num_dataptrs,
    NS(buffer_size_t) const slot_size );

SIXTRL_FN SIXTRL_STATIC int NS(BeamElements_copy_object)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)* SIXTRL_RESTRICT destination,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC const NS(Object) *const SIXTRL_RESTRICT src );

SIXTRL_FN SIXTRL_STATIC int NS(BeamElements_compare_objects)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC const NS(Object) *const SIXTRL_RESTRICT lhs,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC const NS(Object) *const SIXTRL_RESTRICT rhs );

SIXTRL_FN SIXTRL_STATIC int NS(BeamElements_compare_objects_with_treshold)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC const NS(Object) *const SIXTRL_RESTRICT lhs,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC const NS(Object) *const SIXTRL_RESTRICT rhs,
    SIXTRL_REAL_T const treshold );

/* ------------------------------------------------------------------------ */

SIXTRL_FN SIXTRL_STATIC int NS(BeamElements_calc_buffer_parameters_for_line)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT begin,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT end,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT num_objects,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT num_slots,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT num_dataptrs,
    NS(buffer_size_t) const slot_size );

SIXTRL_FN SIXTRL_STATIC int NS(BeamElements_copy_line)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT begin,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT end,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)* SIXTRL_RESTRICT destination_begin );

SIXTRL_FN SIXTRL_STATIC int NS(BeamElements_compare_lines)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT begin,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT end,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)* SIXTRL_RESTRICT rhs_begin );

SIXTRL_FN SIXTRL_STATIC int NS(BeamElements_compare_lines_with_treshold)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT begin,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT end,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)* SIXTRL_RESTRICT rhs_begin,
    SIXTRL_REAL_T const treshold );

/* ========================================================================= */

#if !defined( _GPUCODE )

SIXTRL_FN SIXTRL_STATIC int NS(BeamElements_add_single_new_to_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC const NS(Object) *const SIXTRL_RESTRICT obj );

SIXTRL_FN SIXTRL_STATIC int NS(BeamElements_copy_single_to_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC const NS(Object) *const SIXTRL_RESTRICT obj );

SIXTRL_FN SIXTRL_STATIC int NS(BeamElements_add_new_to_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT begin,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT end );

SIXTRL_FN SIXTRL_STATIC int NS(BeamElements_copy_to_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT begin,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT end );

#endif /* !defined( _GPUCODE ) */


/* ========================================================================= */

#if !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined( _GPUCODE ) && defined( __cplusplus ) */

/* ========================================================================= */
/* =====            Implementation of inline functions                ====== */
/* ========================================================================= */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/impl/managed_buffer_minimal.h"

    #if !defined( _GPUCODE )
        #include "sixtracklib/common/buffer.h"
    #endif /* !defined( _GPUCODE ) */

#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined( _GPUCODE ) && defined( __cplusplus ) */

SIXTRL_INLINE int NS(BeamElements_calc_buffer_parameters_for_object)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC const NS(Object) *const SIXTRL_RESTRICT obj,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_num_objects,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_num_slots,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_num_dataptrs,
    NS(buffer_size_t) const slot_size )
{
    int success = -1;

    typedef NS(buffer_size_t)       buf_size_t;
    typedef NS(object_type_id_t)    type_id_t;
    typedef NS(buffer_addr_t)       address_t;

    if( ( obj!= SIXTRL_NULLPTR ) && ( slot_size > ( buf_size_t )0u ) )
    {
        address_t const begin_addr = NS(Object_get_begin_addr)( obj );
        type_id_t const type_id    = NS(Object_get_type_id)( obj );

        buf_size_t requ_num_objects  = ( buf_size_t )0u;
        buf_size_t requ_num_slots    = ( buf_size_t )0u;
        buf_size_t requ_num_dataptrs = ( buf_size_t )0u;

        success = 0;

        switch( type_id )
        {
            case NS(OBJECT_TYPE_DRIFT):
            {
                typedef NS(Drift) beam_element_t;
                typedef SIXTRL_BE_ARGPTR_DEC beam_element_t const* ptr_belem_t;

                ptr_belem_t ptr_begin = ( ptr_belem_t )( uintptr_t )begin_addr;

                ++requ_num_objects;
                requ_num_dataptrs =
                    NS(Drift_get_num_dataptrs)( ptr_begin );

                requ_num_slots    =
                    NS(Drift_get_num_slots)( ptr_begin, slot_size );
                break;
            }

            case NS(OBJECT_TYPE_DRIFT_EXACT):
            {
                typedef NS(DriftExact) beam_element_t;
                typedef SIXTRL_BE_ARGPTR_DEC beam_element_t const* ptr_belem_t;

                ptr_belem_t ptr_begin = ( ptr_belem_t )( uintptr_t )begin_addr;

                ++requ_num_objects;

                requ_num_dataptrs =
                    NS(DriftExact_get_num_dataptrs)( ptr_begin );

                requ_num_slots    =
                    NS(DriftExact_get_num_slots)( ptr_begin, slot_size );
                break;
            }

            case NS(OBJECT_TYPE_MULTIPOLE):
            {
                typedef NS(MultiPole) beam_element_t;
                typedef SIXTRL_BE_ARGPTR_DEC beam_element_t const* ptr_belem_t;

                ptr_belem_t ptr_begin = ( ptr_belem_t )( uintptr_t )begin_addr;

                ++requ_num_objects;
                requ_num_slots =
                    NS(MultiPole_get_num_slots)( ptr_begin, slot_size );

                requ_num_dataptrs =
                    NS(MultiPole_get_num_dataptrs)( ptr_begin );
                break;
            }

            case NS(OBJECT_TYPE_XYSHIFT):
            {
                typedef NS(XYShift) beam_element_t;
                typedef SIXTRL_BE_ARGPTR_DEC beam_element_t const* ptr_belem_t;

                ptr_belem_t ptr_begin = ( ptr_belem_t )( uintptr_t )begin_addr;

                ++requ_num_objects;
                requ_num_slots =
                    NS(XYShift_get_num_slots)( ptr_begin, slot_size );

                requ_num_dataptrs =
                    NS(XYShift_get_num_dataptrs)( ptr_begin );

                break;
            }

            case NS(OBJECT_TYPE_SROTATION):
            {
                typedef NS(SRotation) beam_element_t;
                typedef SIXTRL_BE_ARGPTR_DEC beam_element_t const* ptr_belem_t;

                ptr_belem_t ptr_begin = ( ptr_belem_t )( uintptr_t )begin_addr;

                ++requ_num_objects;
                requ_num_slots =
                    NS(SRotation_get_num_slots)( ptr_begin, slot_size );

                requ_num_dataptrs =
                    NS(SRotation_get_num_dataptrs)( ptr_begin );
                break;
            }

            case NS(OBJECT_TYPE_CAVITY):
            {
                typedef NS(Cavity) beam_element_t;
                typedef SIXTRL_BE_ARGPTR_DEC beam_element_t const* ptr_belem_t;

                ptr_belem_t ptr_begin = ( ptr_belem_t )( uintptr_t )begin_addr;

                ++requ_num_objects;
                requ_num_slots =
                    NS(Cavity_get_num_slots)( ptr_begin, slot_size );

                requ_num_dataptrs =
                    NS(Cavity_get_num_dataptrs)( ptr_begin );
                break;
            }

            default:
            {
                success = -1;
            }
        };

        if( success == 0 )
        {
            if(  ptr_num_objects != SIXTRL_NULLPTR )
            {
                *ptr_num_objects += requ_num_objects;
            }

            if(  ptr_num_slots != SIXTRL_NULLPTR )
            {
                *ptr_num_slots += requ_num_slots;
            }

            if(  ptr_num_dataptrs != SIXTRL_NULLPTR )
            {
                *ptr_num_dataptrs += requ_num_dataptrs;
            }
        }
    }

    return success;
}

SIXTRL_INLINE int NS(BeamElements_copy_object)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)* SIXTRL_RESTRICT destination,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC const NS(Object) *const SIXTRL_RESTRICT src )
{
    int success = -1;

    if( ( destination != SIXTRL_NULLPTR ) && ( src != SIXTRL_NULLPTR ) )
    {
        typedef NS(buffer_size_t)       buf_size_t;
        typedef NS(buffer_addr_t)       address_t;
        typedef NS(object_type_id_t)    type_id_t;

        type_id_t  const    type_id = NS(Object_get_type_id)( src );
        address_t  const   src_addr = NS(Object_get_begin_addr)( src );
        buf_size_t const   src_size = NS(Object_get_size)( src );

        address_t  const  dest_addr = NS(Object_get_begin_addr)( destination );
        buf_size_t const  dest_size = NS(Object_get_size)( destination );

        if( ( type_id   == NS(Object_get_type_id)( destination ) ) &&
            ( src_addr  != ( address_t )0u ) && ( src_addr  != dest_addr ) &&
            ( dest_addr != ( address_t )0u ) && ( src_size  <= dest_size ) )
        {
            switch( type_id )
            {
                case NS(OBJECT_TYPE_DRIFT):
                {
                    typedef NS(Drift)                               belem_t;
                    typedef SIXTRL_BE_ARGPTR_DEC belem_t*           ptr_dest_t;
                    typedef SIXTRL_BE_ARGPTR_DEC belem_t const*     ptr_src_t;

                    success = NS(Drift_copy)(
                        ( ptr_dest_t )( uintptr_t )dest_addr,
                        ( ptr_src_t  )( uintptr_t )src_addr );

                    break;
                }

                case NS(OBJECT_TYPE_DRIFT_EXACT):
                {
                    typedef NS(DriftExact)                          belem_t;
                    typedef SIXTRL_BE_ARGPTR_DEC belem_t*           ptr_dest_t;
                    typedef SIXTRL_BE_ARGPTR_DEC belem_t const*     ptr_src_t;

                    success = NS(DriftExact_copy)(
                        ( ptr_dest_t )( uintptr_t )dest_addr,
                        ( ptr_src_t  )( uintptr_t )src_addr );

                    break;
                }

                case NS(OBJECT_TYPE_MULTIPOLE):
                {
                    typedef NS(MultiPole)                           belem_t;
                    typedef SIXTRL_BE_ARGPTR_DEC belem_t*           ptr_dest_t;
                    typedef SIXTRL_BE_ARGPTR_DEC belem_t const*     ptr_src_t;

                    success = NS(MultiPole_copy)(
                        ( ptr_dest_t )( uintptr_t )dest_addr,
                        ( ptr_src_t  )( uintptr_t )src_addr );

                    break;
                }

                case NS(OBJECT_TYPE_XYSHIFT):
                {
                    typedef NS(XYShift)                             belem_t;
                    typedef SIXTRL_BE_ARGPTR_DEC belem_t*           ptr_dest_t;
                    typedef SIXTRL_BE_ARGPTR_DEC belem_t const*     ptr_src_t;

                    success = NS(XYShift_copy)(
                        ( ptr_dest_t )( uintptr_t )dest_addr,
                        ( ptr_src_t  )( uintptr_t )src_addr );

                    break;
                }

                case NS(OBJECT_TYPE_SROTATION):
                {
                    typedef NS(SRotation)                           belem_t;
                    typedef SIXTRL_BE_ARGPTR_DEC belem_t*           ptr_dest_t;
                    typedef SIXTRL_BE_ARGPTR_DEC belem_t const*     ptr_src_t;

                    success = NS(SRotation_copy)(
                        ( ptr_dest_t )( uintptr_t )dest_addr,
                        ( ptr_src_t  )( uintptr_t )src_addr );

                    break;
                }

                case NS(OBJECT_TYPE_CAVITY):
                {
                    typedef NS(Cavity)                              belem_t;
                    typedef SIXTRL_BE_ARGPTR_DEC belem_t*           ptr_dest_t;
                    typedef SIXTRL_BE_ARGPTR_DEC belem_t const*     ptr_src_t;

                    success = NS(Cavity_copy)(
                        ( ptr_dest_t )( uintptr_t )dest_addr,
                        ( ptr_src_t  )( uintptr_t )src_addr );

                    break;
                }

                default:
                {
                    success = -1;
                }
            };
        }
    }

    return success;
}

SIXTRL_INLINE int NS(BeamElements_compare_objects)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC const NS(Object) *const SIXTRL_RESTRICT lhs,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC const NS(Object) *const SIXTRL_RESTRICT rhs )
{
    int compare_value = -1;

    if( ( lhs != SIXTRL_NULLPTR ) && ( rhs != SIXTRL_NULLPTR ) )
    {
        typedef NS(buffer_addr_t)       address_t;
        typedef NS(object_type_id_t)    type_id_t;

        type_id_t const lhs_type_id = NS(Object_get_type_id)( lhs );

        type_id_t const rhs_type_id = NS(Object_get_type_id)( rhs );

        if( lhs_type_id == rhs_type_id )
        {
            address_t const lhs_addr = NS(Object_get_begin_addr)( lhs );
            address_t const rhs_addr = NS(Object_get_begin_addr)( rhs );

            if( ( lhs_addr != ( address_t)0u ) &&
                ( rhs_addr != ( address_t)0u ) && ( lhs_addr != rhs_addr ) )
            {
                switch( lhs_type_id )
                {
                    case NS(OBJECT_TYPE_DRIFT):
                    {
                        typedef NS(Drift)                           belem_t;
                        typedef SIXTRL_BE_ARGPTR_DEC belem_t const* ptr_belem_t;

                        compare_value = NS(Drift_compare_values)(
                            ( ptr_belem_t )( uintptr_t )lhs_addr,
                            ( ptr_belem_t )( uintptr_t )rhs_addr );

                        break;
                    }

                    case NS(OBJECT_TYPE_DRIFT_EXACT):
                    {
                        typedef NS(DriftExact)                       belem_t;
                        typedef SIXTRL_BE_ARGPTR_DEC belem_t const*  ptr_belem_t;

                        compare_value = NS(DriftExact_compare_values)(
                            ( ptr_belem_t )( uintptr_t )lhs_addr,
                            ( ptr_belem_t )( uintptr_t )rhs_addr );

                        break;
                    }

                    case NS(OBJECT_TYPE_MULTIPOLE):
                    {
                        typedef NS(MultiPole)                       belem_t;
                        typedef SIXTRL_BE_ARGPTR_DEC belem_t const* ptr_belem_t;

                        compare_value = NS(MultiPole_compare_values)(
                            ( ptr_belem_t )( uintptr_t )lhs_addr,
                            ( ptr_belem_t )( uintptr_t )rhs_addr );

                        break;
                    }

                    case NS(OBJECT_TYPE_XYSHIFT):
                    {
                        typedef NS(XYShift)                          belem_t;
                        typedef SIXTRL_BE_ARGPTR_DEC belem_t const*  ptr_belem_t;

                        compare_value = NS(XYShift_compare_values)(
                            ( ptr_belem_t )( uintptr_t )lhs_addr,
                            ( ptr_belem_t )( uintptr_t )rhs_addr );

                        break;
                    }

                    case NS(OBJECT_TYPE_SROTATION):
                    {
                        typedef NS(SRotation)                        belem_t;
                        typedef SIXTRL_BE_ARGPTR_DEC belem_t const*  ptr_belem_t;

                        compare_value = NS(SRotation_compare_values)(
                            ( ptr_belem_t )( uintptr_t )lhs_addr,
                            ( ptr_belem_t )( uintptr_t )rhs_addr );

                        break;
                    }

                    case NS(OBJECT_TYPE_CAVITY):
                    {
                        typedef NS(Cavity)                           belem_t;
                        typedef SIXTRL_BE_ARGPTR_DEC belem_t const*  ptr_belem_t;

                        compare_value = NS(Cavity_compare_values)(
                            ( ptr_belem_t )( uintptr_t )lhs_addr,
                            ( ptr_belem_t )( uintptr_t )rhs_addr );

                        break;
                    }

                    default:
                    {
                        compare_value = -1;
                    }
                };
            }
            else if( lhs_addr != ( address_t )0u )
            {
                compare_value = ( rhs_addr != lhs_addr ) ? +1 : 0;
            }
            else if( rhs_addr != ( address_t )0u )
            {
                compare_value = ( rhs_addr != lhs_addr ) ? -1 : 0;
            }
        }
        else if( lhs_type_id > rhs_type_id )
        {
            compare_value = +1;
        }
    }

    return compare_value;
}

SIXTRL_INLINE int NS(BeamElements_compare_objects_with_treshold)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC const NS(Object) *const SIXTRL_RESTRICT lhs,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC const NS(Object) *const SIXTRL_RESTRICT rhs,
    SIXTRL_REAL_T const treshold )
{
    int compare_value = -1;

    if( ( lhs != SIXTRL_NULLPTR ) && ( rhs != SIXTRL_NULLPTR ) )
    {
        typedef NS(buffer_addr_t)       address_t;
        typedef NS(object_type_id_t)    type_id_t;

        type_id_t const lhs_type_id = NS(Object_get_type_id)( lhs );

        type_id_t const rhs_type_id = NS(Object_get_type_id)( rhs );

        if( lhs_type_id == rhs_type_id )
        {
            address_t const lhs_addr = NS(Object_get_begin_addr)( lhs );
            address_t const rhs_addr = NS(Object_get_begin_addr)( rhs );

            if( ( lhs_addr != ( address_t)0u ) &&
                ( rhs_addr != ( address_t)0u ) && ( lhs_addr != rhs_addr ) )
            {
                switch( lhs_type_id )
                {
                    case NS(OBJECT_TYPE_DRIFT):
                    {
                        typedef NS(Drift)                            belem_t;
                        typedef SIXTRL_BE_ARGPTR_DEC belem_t const*  ptr_belem_t;

                        compare_value = NS(Drift_compare_values_with_treshold)(
                            ( ptr_belem_t )( uintptr_t )lhs_addr,
                            ( ptr_belem_t )( uintptr_t )rhs_addr,
                            treshold );

                        break;
                    }

                    case NS(OBJECT_TYPE_DRIFT_EXACT):
                    {
                        typedef NS(DriftExact)                      belem_t;
                        typedef SIXTRL_BE_ARGPTR_DEC belem_t const* ptr_belem_t;

                        compare_value = NS(DriftExact_compare_values_with_treshold)(
                            ( ptr_belem_t )( uintptr_t )rhs_addr,
                            ( ptr_belem_t )( uintptr_t )lhs_addr,
                            treshold  );

                        break;
                    }

                    case NS(OBJECT_TYPE_MULTIPOLE):
                    {
                        typedef NS(MultiPole)                       belem_t;
                        typedef SIXTRL_BE_ARGPTR_DEC belem_t const* ptr_belem_t;

                        compare_value = NS(MultiPole_compare_values_with_treshold)(
                            ( ptr_belem_t )( uintptr_t )lhs_addr,
                            ( ptr_belem_t )( uintptr_t )rhs_addr,
                            treshold  );

                        break;
                    }

                    case NS(OBJECT_TYPE_XYSHIFT):
                    {
                        typedef NS(XYShift)                         belem_t;
                        typedef SIXTRL_BE_ARGPTR_DEC belem_t const* ptr_belem_t;

                        compare_value = NS(XYShift_compare_values_with_treshold)(
                            ( ptr_belem_t )( uintptr_t )lhs_addr,
                            ( ptr_belem_t )( uintptr_t )rhs_addr,
                            treshold  );

                        break;
                    }

                    case NS(OBJECT_TYPE_SROTATION):
                    {
                        typedef NS(SRotation)                       belem_t;
                        typedef SIXTRL_BE_ARGPTR_DEC belem_t const* ptr_belem_t;

                        compare_value = NS(SRotation_compare_values_with_treshold)(
                            ( ptr_belem_t )( uintptr_t )lhs_addr,
                            ( ptr_belem_t )( uintptr_t )rhs_addr,
                            treshold  );

                        break;
                    }

                    case NS(OBJECT_TYPE_CAVITY):
                    {
                        typedef NS(Cavity)                          belem_t;
                        typedef SIXTRL_BE_ARGPTR_DEC belem_t const* ptr_belem_t;

                        compare_value = NS(Cavity_compare_values_with_treshold)(
                            ( ptr_belem_t )( uintptr_t )lhs_addr,
                            ( ptr_belem_t )( uintptr_t )rhs_addr,
                            treshold  );

                        break;
                    }

                    default:
                    {
                        compare_value = -1;
                    }
                };
            }
            else if( lhs_addr != ( address_t )0u )
            {
                compare_value = ( rhs_addr != lhs_addr ) ? +1 : 0;
            }
            else if( rhs_addr != ( address_t )0u )
            {
                compare_value = ( rhs_addr != lhs_addr ) ? -1 : 0;
            }
        }
        else if( lhs_type_id > rhs_type_id )
        {
            compare_value = +1;
        }
    }

    return compare_value;
}

/* ------------------------------------------------------------------------ */

SIXTRL_INLINE int NS(BeamElements_calc_buffer_parameters_for_line)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT src_it,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT src_end,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT num_objects,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT num_slots,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT num_dataptrs,
    NS(buffer_size_t) const slot_size )
{
    int success = -1;

    if( ( src_it != SIXTRL_NULLPTR ) && ( src_end != SIXTRL_NULLPTR ) )
    {
        success = 0;

        for( ; src_it != src_end ; ++src_it )
        {
            int const ret = NS(BeamElements_calc_buffer_parameters_for_object)(
                    src_it, num_objects, num_slots, num_dataptrs, slot_size );

            if( 0 != ret )
            {
                success |= ret;
                break;
            }
        }
    }

    return success;
}

SIXTRL_INLINE int NS(BeamElements_copy_line)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT src_it,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT src_end,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)* SIXTRL_RESTRICT dest_it )
{
    int success = -1;

    if( ( src_it  != SIXTRL_NULLPTR ) && ( src_end != SIXTRL_NULLPTR ) &&
        ( dest_it != SIXTRL_NULLPTR ) )
    {
        SIXTRL_ASSERT( ( ptrdiff_t )( src_end - src_it) > 0 );

        success = 0;

        for( ; src_it != src_end ; ++src_it, ++dest_it )
        {
            int const ret = NS(BeamElements_copy_object)( dest_it, src_it );

            if( 0 != ret )
            {
                success |= ret;
                break;
            }
        }
    }

    return success;
}

SIXTRL_INLINE int NS(BeamElements_compare_lines)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT lhs_it,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT lhs_end,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)* SIXTRL_RESTRICT rhs_it )
{
    int compare_value = -1;

    if( ( lhs_it != SIXTRL_NULLPTR ) && ( lhs_end != SIXTRL_NULLPTR ) &&
        ( rhs_it != SIXTRL_NULLPTR ) )
    {
        SIXTRL_ASSERT( ( ptrdiff_t )( lhs_end - lhs_it) > 0 );

        for( ; lhs_it != lhs_end ; ++lhs_it, ++rhs_it )
        {
            compare_value = NS(BeamElements_compare_objects)( lhs_it, rhs_it );

            if( 0 != compare_value )
            {
                break;
            }
        }
    }

    return compare_value;
}

SIXTRL_INLINE int NS(BeamElements_compare_lines_with_treshold)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT lhs_it,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT lhs_end,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)* SIXTRL_RESTRICT rhs_it,
    SIXTRL_REAL_T const treshold )
{
    int compare_value = -1;

    if( ( lhs_it != SIXTRL_NULLPTR ) && ( lhs_end != SIXTRL_NULLPTR ) &&
        ( rhs_it != SIXTRL_NULLPTR ) )
    {
        SIXTRL_ASSERT( ( ptrdiff_t )( lhs_end - lhs_it) > 0 );

        for( ; lhs_it != lhs_end ; ++lhs_it, ++rhs_it )
        {
            compare_value = NS(BeamElements_compare_objects_with_treshold)(
                lhs_it, rhs_it, treshold );

            if( 0 != compare_value )
            {
                break;
            }
        }
    }

    return compare_value;
}

/* ========================================================================= */

#if !defined( _GPUCODE )

SIXTRL_INLINE int NS(BeamElements_add_single_new_to_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC const NS(Object) *const SIXTRL_RESTRICT obj )
{
    int success = -1;

    if( obj != SIXTRL_NULLPTR )
    {
        typedef NS(buffer_addr_t)       address_t;
        typedef NS(object_type_id_t)    type_id_t;

        type_id_t  const    type_id = NS(Object_get_type_id)( obj );
        address_t  const begin_addr = NS(Object_get_begin_addr)( obj );

        switch( type_id )
        {
            case NS(OBJECT_TYPE_DRIFT):
            {
                success = ( SIXTRL_NULLPTR !=
                    NS(Drift_new)( buffer ) ) ? 0 : -1;

                break;
            }

            case NS(OBJECT_TYPE_DRIFT_EXACT):
            {
                success = ( SIXTRL_NULLPTR !=
                    NS(DriftExact_new)( buffer ) ) ? 0 : -1;

                break;
            }

            case NS(OBJECT_TYPE_MULTIPOLE):
            {
                typedef NS(MultiPole)                    mp_t;
                typedef NS(multipole_order_t)            mp_order_t;
                typedef SIXTRL_BE_ARGPTR_DEC mp_t const* ptr_mp_t;

                ptr_mp_t ptr_mp = ( ptr_mp_t )( uintptr_t )begin_addr;
                mp_order_t const order = NS(MultiPole_get_order)( ptr_mp );

                success = ( SIXTRL_NULLPTR !=
                    NS(MultiPole_new)( buffer, order ) ) ? 0 : -1;

                break;
            }

            case NS(OBJECT_TYPE_XYSHIFT):
            {
                success = ( SIXTRL_NULLPTR !=
                    NS(XYShift_new)( buffer ) ) ? 0 : -1;

                break;
            }

            case NS(OBJECT_TYPE_SROTATION):
            {
                success = ( SIXTRL_NULLPTR !=
                    NS(SRotation_new)( buffer ) ) ? 0 : -1;

                break;
            }

            case NS(OBJECT_TYPE_CAVITY):
            {
                success = ( SIXTRL_NULLPTR !=
                    NS(Cavity_new)( buffer ) ) ? 0 : -1;

                break;
            }

            default:
            {
                success = -1;
            }
        };
    }

    return success;
}

SIXTRL_INLINE int NS(BeamElements_copy_single_to_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC const NS(Object) *const SIXTRL_RESTRICT obj )
{
    int success = -1;

    if( obj != SIXTRL_NULLPTR )
    {
        typedef NS(buffer_addr_t)       address_t;
        typedef NS(object_type_id_t)    type_id_t;

        type_id_t  const    type_id = NS(Object_get_type_id)( obj );
        address_t  const begin_addr = NS(Object_get_begin_addr)( obj );

        switch( type_id )
        {
            case NS(OBJECT_TYPE_DRIFT):
            {
                typedef NS(Drift)             beam_element_t;
                typedef SIXTRL_BE_ARGPTR_DEC  beam_element_t const* ptr_belem_t;

                ptr_belem_t orig = ( ptr_belem_t )( uintptr_t )begin_addr;
                success = ( SIXTRL_NULLPTR !=
                    NS(Drift_add_copy)( buffer, orig ) ) ? 0 : -1;

                break;
            }

            case NS(OBJECT_TYPE_DRIFT_EXACT):
            {
                typedef NS(DriftExact)        beam_element_t;
                typedef SIXTRL_BE_ARGPTR_DEC  beam_element_t const* ptr_belem_t;

                ptr_belem_t orig = ( ptr_belem_t )( uintptr_t )begin_addr;
                success = ( SIXTRL_NULLPTR !=
                    NS(DriftExact_add_copy)( buffer, orig ) ) ? 0 : -1;

                break;
            }

            case NS(OBJECT_TYPE_MULTIPOLE):
            {
                typedef NS(MultiPole)                    mp_t;
                typedef SIXTRL_BE_ARGPTR_DEC mp_t const* ptr_mp_t;

                ptr_mp_t ptr_mp = ( ptr_mp_t )( uintptr_t )begin_addr;

                success = ( SIXTRL_NULLPTR !=
                    NS(MultiPole_add_copy)( buffer, ptr_mp ) ) ? 0 : -1;

                break;
            }

            case NS(OBJECT_TYPE_XYSHIFT):
            {
                typedef  NS(XYShift)           beam_element_t;
                typedef  SIXTRL_BE_ARGPTR_DEC  beam_element_t const* ptr_belem_t;

                ptr_belem_t orig = ( ptr_belem_t )( uintptr_t )begin_addr;

                success = ( SIXTRL_NULLPTR !=
                    NS(XYShift_add_copy)( buffer, orig ) ) ? 0 : -1;

                break;
            }

            case NS(OBJECT_TYPE_SROTATION):
            {
                typedef  NS(SRotation)           beam_element_t;
                typedef  SIXTRL_BE_ARGPTR_DEC  beam_element_t const* ptr_belem_t;

                ptr_belem_t orig = ( ptr_belem_t )( uintptr_t )begin_addr;

                success = ( SIXTRL_NULLPTR !=
                    NS(SRotation_add_copy)( buffer, orig ) ) ? 0 : -1;

                break;
            }

            case NS(OBJECT_TYPE_CAVITY):
            {
                typedef  NS(Cavity)            beam_element_t;
                typedef  SIXTRL_BE_ARGPTR_DEC  beam_element_t const* ptr_belem_t;

                ptr_belem_t orig = ( ptr_belem_t )( uintptr_t )begin_addr;

                success = ( SIXTRL_NULLPTR !=
                    NS(Cavity_add_copy)( buffer, orig ) ) ? 0 : -1;

                break;
            }

            default:
            {
                success = -1;
            }
        };
    }

    return success;
}

SIXTRL_INLINE int NS(BeamElements_add_new_to_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT it,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT end )
{
    int success = -1;

    if( ( it != SIXTRL_NULLPTR ) && ( end != SIXTRL_NULLPTR ) )
    {
        success = ( ( ( ptrdiff_t )( end - it ) ) > 0 ) ? 0 : -1;

        for( ; it != end ; ++it )
        {
            int const ret = NS(BeamElements_add_single_new_to_buffer)( buffer, it );

            if( 0 != ret )
            {
                success |= ret;
                break;
            }
        }
    }

    return success;
}

SIXTRL_INLINE int NS(BeamElements_copy_to_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const*  SIXTRL_RESTRICT it,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const*  SIXTRL_RESTRICT end )
{
    int success = -1;

    if( ( it != SIXTRL_NULLPTR ) && ( end != SIXTRL_NULLPTR ) )
    {
        success = ( ( ( ptrdiff_t )( end - it ) ) > 0 ) ? 0 : -1;

        for( ; it != end ; ++it )
        {
            int const ret = NS(BeamElements_copy_single_to_buffer)( buffer, it );

            if( 0 != ret )
            {
                success |= ret;
                break;
            }
        }
    }

    return success;
}

#endif /* !defined( _GPUCODE ) */


#if !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined( _GPUCODE ) && defined( __cplusplus ) */

#endif /* SIXTRACKLIB_COMMON_BEAM_ELEMENTS_H__ */

/* end: sixtracklib/sixtracklib/common/beam_elements.h */
